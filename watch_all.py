#!/usr/bin/env python3
"""
watch_all.py — Security-grid viewer: watch ALL trained models play side-by-side.

Discovers every best_model.zip under models/best/, loads each with its
VecNormalize stats, and renders them in a tiled pygame grid window.

Place at REPO ROOT (next to menu.py, watch_agent.py).

Usage (from repo root):
    python watch_all.py [--fps 20] [--episodes 5] [--game platformer]

Controls:
    ESC / close window  — quit
    SPACE               — pause / resume
    R                   — reset all episodes
"""

import argparse
import importlib
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pygame

from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

try:
    from sb3_contrib import RecurrentPPO
    _HAS_RPPO = True
except ImportError:
    _HAS_RPPO = False

# Algo registry
ALGO_MAP = {"ppo": PPO, "a2c": A2C, "dqn": DQN}
if _HAS_RPPO:
    ALGO_MAP["rppo"] = RecurrentPPO
    ALGO_MAP["recurrent_ppo"] = RecurrentPPO

# Game render size (must match platformer_core constants)
GAME_W, GAME_H = 800, 600

# Grid visual settings
HEADER_H       = 42
FOOTER_H       = 30
LABEL_H        = 34
CELL_PAD       = 6
BG_COLOR       = (8, 12, 20)
HEADER_BG      = (14, 20, 32)
CARD_BG        = (12, 18, 28)
LABEL_BG       = (18, 26, 40)
LABEL_FG       = (228, 236, 248)
BORDER_ACTIVE  = (72, 174, 255)
BORDER_DONE    = (132, 64, 90)
SCORE_COLOR    = (255, 220, 120)
PAUSED_COLOR   = (255, 110, 110)

_ARCH_TAGS = {
    "lightmobile",
    "spatialattention",
    "channelattention",
    "deepchannelattention",
    "mlp",
}


# ---------------------------------------------------------------------------
# Model metadata parsing (mirrors watch_agent.py)
# ---------------------------------------------------------------------------
def parse_model_info(model_path: str):
    path = Path(model_path)
    folder = path.parent.name
    if folder not in {".", "models", "best"}:
        parts = folder.split("_")
    else:
        parts = path.stem.replace("_model", "").replace("best", "").split("_")

    if len(parts) >= 5 and parts[-1].lower() in _ARCH_TAGS:
        parts = parts[:-1]

    if len(parts) >= 4:
        game  = parts[0]
        algo  = parts[1]
        skill = parts[-1]
        raw   = "_".join(parts[2:-1])
        persona = (
            raw[len(game)+1:] if raw.startswith(f"{game}_") else
            raw[len(game):]   if raw.startswith(game) else raw
        )
        return game, algo, persona, skill
    elif parts:
        return parts[0], "ppo", "default", "unknown"
    raise ValueError(f"Cannot parse: {model_path}")


def find_vecnorm(model_path: str) -> Optional[Path]:
    p = Path(model_path)
    # Stem match
    c = p.parent / (p.stem + "_vecnorm.pkl")
    if c.exists():
        return c
    # Glob same dir
    for pkl in sorted(p.parent.glob("*vecnorm*.pkl")):
        return pkl
    # Glob parent
    for pkl in sorted(p.parent.parent.glob("*vecnorm*.pkl")):
        return pkl
    return None


# ---------------------------------------------------------------------------
# ModelSlot — one model + env in the grid
# ---------------------------------------------------------------------------
class ModelSlot:
    """Holds a loaded model, its environment, and episode state."""

    def __init__(self, model_path: Path, cell_size: tuple):
        self.path = model_path
        self.cell_w, self.cell_h = cell_size
        self.ok = False
        self.error = ""

        # Parse metadata
        try:
            self.game, self.algo, self.persona, self.skill = parse_model_info(str(model_path))
        except Exception as e:
            self.error = f"Parse error: {e}"
            return

        self.label = f"{self.persona}  [{self.skill}]"
        self.sub_label = f"{self.algo.upper()}"

        # Load model
        algo_cls = ALGO_MAP.get(self.algo.lower())
        if algo_cls is None:
            self.error = f"Unknown algo: {self.algo}"
            return

        try:
            self.model = algo_cls.load(str(model_path), device="cpu")
        except Exception as e:
            self.error = f"Load failed: {e}"
            return

        self.is_recurrent = _HAS_RPPO and isinstance(self.model, RecurrentPPO)

        # Load reward fn
        reward_fn = None
        try:
            reward_mod = importlib.import_module(f"code.rewards.train_{self.game}")
            for name in [self.persona, f"delta_{self.persona}", f"{self.game}_{self.persona}"]:
                if hasattr(reward_mod, name):
                    reward_fn = getattr(reward_mod, name)
                    break
        except Exception:
            pass

        # Build env (headless — we render manually)
        try:
            game_module = importlib.import_module(f"code.games.{self.game}_core")
            GameCls = getattr(game_module, f"{self.game.capitalize()}Core")
        except Exception as e:
            self.error = f"Game import: {e}"
            return

        try:
            from code.wrappers.generic_env import GameEnv
            self.raw_env = GameEnv(
                GameCls,
                render_mode="none",
                fps=None,
                persona=self.persona,
                **({"reward_fn": reward_fn} if reward_fn else {}),
            )
        except Exception as e:
            self.error = f"Env create: {e}"
            return

        # Wrap with VecNormalize if available
        vn_path = find_vecnorm(str(model_path))
        self.is_vec = False
        try:
            if vn_path and vn_path.exists():
                # IMPORTANT: DummyVecEnv expects a factory callable.
                # Capture raw_env via default arg to avoid closure issues.
                _env = self.raw_env
                vec_raw = DummyVecEnv([lambda e=_env: e])
                self.env = VecNormalize.load(str(vn_path), vec_raw)
                self.env.training = False
                self.env.norm_reward = False
                self.is_vec = True
                print(f"    ✓ VecNorm loaded: {vn_path.name}")
            else:
                self.env = self.raw_env
                print(f"    ⚠ No vecnorm found — obs will NOT be normalised")
        except Exception as e:
            self.env = self.raw_env
            print(f"    ⚠ VecNorm load failed ({e}) — using raw obs")

        # Create a surface to render the game onto
        self.game_surf = pygame.Surface((GAME_W, GAME_H))

        # Episode state
        self.obs = None
        self.lstm_states = None
        self.ep_start = np.ones((1,), dtype=bool)
        self.done = False
        self.score = 0
        self.ep_count = 0
        self.steps = 0
        self.ok = True

        # Lifetime stats for radar plot
        self.total_wins = 0
        self.total_deaths = 0
        self.total_stalls = 0
        self.total_coins = 0
        self.total_kills = 0
        self.scores_history = []

    def reset(self):
        """Reset the environment and episode counters."""
        if not self.ok:
            return
        try:
            if self.is_vec:
                self.obs = self.env.reset()
            else:
                self.obs, _ = self.env.reset()
        except Exception as e:
            self.error = f"Reset failed: {e}"
            self.ok = False
            return
        self.lstm_states = None
        self.ep_start = np.ones((1,), dtype=bool)
        self.done = False
        self.score = 0
        self.steps = 0

        # Reset camera so it snaps to player start position
        game = self.raw_env.game if hasattr(self.raw_env, 'game') else None
        if game:
            game.camera_x = 0.0
            game.camera_y = 0.0
            self._force_camera_update()
            # Snap immediately (skip smoothing on first frame)
            if hasattr(game, 'player') and game.player is not None:
                level_w = game.level_data.width
                game.camera_x = max(0, min(game.player.gObj.x - GAME_W // 3,
                                           level_w - GAME_W))

    def step(self):
        """Run one agent step. Returns True if episode just ended."""
        if not self.ok or self.done:
            return False
        try:
            game = self.raw_env.game if hasattr(self.raw_env, 'game') else None
            dm = getattr(game, 'debug_manager', None) if game else None
            if dm:
                dm.update_input()

            if self.is_recurrent:
                action, self.lstm_states = self.model.predict(
                    self.obs, state=self.lstm_states,
                    episode_start=self.ep_start, deterministic=True)
                self.ep_start = np.zeros((1,), dtype=bool)
            else:
                action, _ = self.model.predict(self.obs, deterministic=True)

            if dm and dm.free_cam_active:
                action = self._neutral_action(action)

            if self.is_vec:
                self.obs, reward, dones, infos = self.env.step(action)
                info = infos[0] if isinstance(infos, (list, tuple)) else infos
                ep_done = bool(dones[0])
            else:
                self.obs, reward, term, trunc, info = self.env.step(action)
                ep_done = term or trunc

            self.score = info.get("score", self.score) if isinstance(info, dict) else self.score
            self.steps += 1

            # Track stats from info
            if isinstance(info, dict):
                self.total_coins = info.get("coins_collected", self.total_coins)
                self.total_kills += int(info.get("enemies_killed_step", 0))
                if info.get("event", "") == "WIN":
                    self.total_wins += 1
                if info.get("cause", "") == "Stall":
                    self.total_stalls += 1

            # Force camera update (game core skips this in non-human mode)
            self._force_camera_update()

            if ep_done:
                self.ep_count += 1
                self.done = True
                self.scores_history.append(self.score)
                if isinstance(info, dict) and info.get("event", "") == "DIED":
                    self.total_deaths += 1
                return True
        except Exception as e:
            self.error = f"Step error: {e}"
            self.done = True
        return False

    def _neutral_action(self, action):
        if isinstance(action, np.ndarray):
            return np.zeros_like(action)
        if isinstance(action, (list, tuple)):
            return type(action)(0 for _ in action)
        return 0

    def _force_camera_update(self):
        """Manually update camera position on the game core.

        PlatformerCore._update_camera() early-returns when render_mode != "human",
        so headless envs always have camera_x = camera_y = 0. We replicate the
        smooth-follow logic here so the grid viewer shows a properly scrolled view.
        """
        game = self.raw_env.game if hasattr(self.raw_env, 'game') else None
        if game is None or not hasattr(game, 'player') or game.player is None:
            return

        p = game.player
        level_w = getattr(game, 'level_data', None)
        if level_w is None:
            return
        level_w = game.level_data.width
        level_h = game.level_data.height

        dm = getattr(game, 'debug_manager', None)
        if dm and getattr(dm, 'free_cam_active', False):
            dx, dy = getattr(dm, 'current_cam_move', [0.0, 0.0])
            dt = getattr(game, 'dt', 1 / 60.0)
            game.camera_x = max(0, min(game.camera_x + dx * dt, max(0, level_w - GAME_W)))
            game.camera_y = max(0, min(game.camera_y + dy * dt, max(0, level_h - GAME_H)))
            return

        smoothing = getattr(game, 'camera_smoothing', 0.15)

        # Horizontal: player at 1/3 from left
        target_x = max(0, min(p.gObj.x - GAME_W // 3, level_w - GAME_W))
        game.camera_x += (target_x - game.camera_x) * smoothing
        game.camera_x = max(0, min(game.camera_x, max(0, level_w - GAME_W)))

        # Vertical: center on player if level is taller than screen
        if level_h > GAME_H:
            target_y = max(0, min(p.gObj.y - GAME_H // 2, level_h - GAME_H))
            game.camera_y += (target_y - game.camera_y) * smoothing
            game.camera_y = max(0, min(game.camera_y, max(0, level_h - GAME_H)))

    def render_to_surface(self):
        """Draw the current game state onto self.game_surf, with optional overlays."""
        if not self.ok:
            return
        try:
            game = self.raw_env.game if hasattr(self.raw_env, 'game') else None
            if game and hasattr(game, 'render'):
                game.render(self.game_surf, blit_only=True)
                dm = getattr(game, 'debug_manager', None)
                if dm:
                    # Agent vision is useful in single-agent debugging, but it
                    # overwhelms the grid viewer and blocks the actual gameplay.
                    dm.agent_view_overlay.max_view = False
                    if getattr(dm, 'show_grid', False) and hasattr(dm, 'grid_overlay'):
                        dm.grid_overlay.render(self.game_surf, game)
                    if getattr(dm, 'show_hitboxes', False) and hasattr(dm, 'hitbox_overlay'):
                        dm.hitbox_overlay.render(self.game_surf, game)
                    if getattr(dm, 'show_sensors', False) and hasattr(dm, 'jump_arc_overlay'):
                        dm.jump_arc_overlay.render(self.game_surf, game)
                    if getattr(dm, 'show_sensors', False) and hasattr(game, 'last_rays'):
                        from code.games.modules.System.debugging_mods.overlays import (
                            RAY_EMPTY, RAY_SOLID, RAY_HAZARD, RAY_COIN, RAY_GOAL)
                        ray_surf = pygame.Surface((GAME_W, GAME_H), pygame.SRCALPHA)
                        for start, end, found, rtype in game.last_rays:
                            if   rtype == 0.0: color = RAY_EMPTY
                            elif rtype == 1.0: color = RAY_SOLID
                            elif rtype == 2.0: color = RAY_HAZARD
                            elif rtype == 3.0: color = RAY_COIN
                            elif rtype == 4.0: color = RAY_GOAL
                            else:              color = RAY_EMPTY
                            s_cam = (start[0] - game.camera_x, start[1] - game.camera_y)
                            e_cam = (end[0]   - game.camera_x, end[1]   - game.camera_y)
                            pygame.draw.line(ray_surf, color, s_cam, e_cam, 1)
                        self.game_surf.blit(ray_surf, (0, 0))
        except Exception:
            self.game_surf.fill((40, 10, 30))

    def close(self):
        try:
            if self.is_vec:
                self.env.close()
            else:
                self.raw_env.close()
        except Exception:
            pass

    def get_radar_metrics(self):
        """Return normalised metrics dict for the radar plot."""
        ep = max(1, self.ep_count)
        return {
            "Win Rate":   self.total_wins / ep,
            "Survival":   1.0 - (self.total_deaths / ep),
            "Coins":      min(1.0, self.total_coins / max(1, ep * 10)),
            "Kills":      min(1.0, self.total_kills / max(1, ep * 3)),
            "Avg Score":  min(1.0, (sum(self.scores_history) / max(1, len(self.scores_history))) / 500),
            "Anti-Stall": 1.0 - min(1.0, self.total_stalls / max(1, ep * 3)),
        }


# ---------------------------------------------------------------------------
# Radar (Spider) Chart Drawing
# ---------------------------------------------------------------------------
def draw_radar_chart(surface, center, radius, metrics: dict, label: str,
                     font, color=(80, 200, 120), bg_alpha=200):
    """
    Draw a spider/radar chart onto a pygame surface.

    metrics: dict of {axis_name: value} where value is 0.0–1.0
    """
    import math as _math

    n = len(metrics)
    if n < 3:
        return

    names = list(metrics.keys())
    values = [max(0.0, min(1.0, metrics[k])) for k in names]
    cx, cy = center
    angle_step = 2 * _math.pi / n

    # Background circle
    bg_surf = pygame.Surface((radius * 2 + 40, radius * 2 + 40), pygame.SRCALPHA)
    bx, by = radius + 20, radius + 20
    pygame.draw.circle(bg_surf, (10, 6, 18, bg_alpha), (bx, by), radius + 10)
    surface.blit(bg_surf, (cx - bx, cy - by))

    # Grid rings (25%, 50%, 75%, 100%)
    for ring in (0.25, 0.5, 0.75, 1.0):
        r = int(radius * ring)
        pygame.draw.circle(surface, (50, 40, 65), (cx, cy), r, 1)

    # Axis lines + labels
    axis_points = []
    for i in range(n):
        angle = -_math.pi / 2 + i * angle_step  # start from top
        ex = cx + int(radius * _math.cos(angle))
        ey = cy + int(radius * _math.sin(angle))
        axis_points.append((angle, ex, ey))
        pygame.draw.line(surface, (50, 40, 65), (cx, cy), (ex, ey), 1)

        # Label
        lbl = font.render(names[i], True, (160, 170, 200))
        lx = cx + int((radius + 14) * _math.cos(angle)) - lbl.get_width() // 2
        ly = cy + int((radius + 14) * _math.sin(angle)) - lbl.get_height() // 2
        surface.blit(lbl, (lx, ly))

    # Data polygon
    data_pts = []
    for i in range(n):
        angle = -_math.pi / 2 + i * angle_step
        v = values[i]
        dx = cx + int(radius * v * _math.cos(angle))
        dy = cy + int(radius * v * _math.sin(angle))
        data_pts.append((dx, dy))

    # Filled polygon (semi-transparent)
    if len(data_pts) >= 3:
        poly_surf = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        pygame.draw.polygon(poly_surf, (*color, 60), data_pts)
        pygame.draw.polygon(poly_surf, (*color, 200), data_pts, 2)
        surface.blit(poly_surf, (0, 0))

    # Data points
    for pt in data_pts:
        pygame.draw.circle(surface, color, pt, 3)

    # Title label
    title = font.render(label, True, (220, 230, 255))
    surface.blit(title, (cx - title.get_width() // 2, cy + radius + 22))


def draw_multi_radar_overlay(surface, rect, slots, font_title, font_body, mouse_pos, focus_key=None):
    radar_colors = [
        (80, 200, 120), (200, 110, 120), (90, 150, 255),
        (240, 190, 90), (180, 120, 240), (90, 210, 210),
        (255, 130, 90), (180, 220, 110),
    ]

    panel = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    panel.fill((8, 14, 22, 236))
    pygame.draw.rect(panel, (72, 174, 255, 220), panel.get_rect(), 2, border_radius=14)
    surface.blit(panel, rect.topleft)

    title = font_title.render("MODEL COMPARISON RADAR  (N to close)", True, LABEL_FG)
    surface.blit(title, (rect.x + 18, rect.y + 14))
    subtitle = font_body.render("Hover a row to spotlight it. Click to lock or clear focus.", True, (166, 182, 206))
    surface.blit(subtitle, (rect.x + 20, rect.y + 52))

    ok_slots = [s for s in slots if s.ok]
    if not ok_slots:
        return None, {}

    metrics_names = list(ok_slots[0].get_radar_metrics().keys())
    n = len(metrics_names)
    chart_area_w = int(rect.width * 0.6)
    chart_cx = rect.x + chart_area_w // 2
    chart_cy = rect.y + rect.height // 2 + 10
    radius = min(chart_area_w // 3, rect.height // 3)
    angle_step = 2 * math.pi / n

    for ring in (0.25, 0.5, 0.75, 1.0):
        r = int(radius * ring)
        pygame.draw.circle(surface, (58, 72, 96), (chart_cx, chart_cy), r, 1)

    axis_points = []
    for i, name in enumerate(metrics_names):
        angle = -math.pi / 2 + i * angle_step
        ex = chart_cx + int(radius * math.cos(angle))
        ey = chart_cy + int(radius * math.sin(angle))
        axis_points.append((angle, ex, ey))
        pygame.draw.line(surface, (58, 72, 96), (chart_cx, chart_cy), (ex, ey), 1)

        lbl = font_body.render(name, True, (168, 182, 208))
        lx = chart_cx + int((radius + 18) * math.cos(angle)) - lbl.get_width() // 2
        ly = chart_cy + int((radius + 18) * math.sin(angle)) - lbl.get_height() // 2
        surface.blit(lbl, (lx, ly))

    legend_x = rect.x + chart_area_w + 12
    legend_y = rect.y + 92
    row_h = 96
    small_font = pygame.font.SysFont("Consolas", 15)
    row_map = {}
    hovered_key = None
    slot_colors = {}

    for idx, slot in enumerate(ok_slots):
        slot_key = slot.path.parent.name
        slot_colors[slot_key] = radar_colors[idx % len(radar_colors)]
        row_y = legend_y + idx * row_h
        row_rect = pygame.Rect(legend_x, row_y, rect.right - legend_x - 16, row_h - 10)
        row_map[slot_key] = row_rect
        if row_rect.collidepoint(mouse_pos):
            hovered_key = slot_key

    active_key = focus_key or hovered_key

    for idx, slot in enumerate(ok_slots):
        slot_key = slot.path.parent.name
        color = slot_colors[slot_key]
        metrics = slot.get_radar_metrics()
        points = []
        for i, name in enumerate(metrics_names):
            angle = -math.pi / 2 + i * angle_step
            value = max(0.0, min(1.0, metrics[name]))
            dx = chart_cx + int(radius * value * math.cos(angle))
            dy = chart_cy + int(radius * value * math.sin(angle))
            points.append((dx, dy))

        is_active = active_key is None or active_key == slot_key
        poly_alpha = 62 if is_active else 16
        line_alpha = 236 if is_active else 70
        point_alpha = 255 if is_active else 110
        poly = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        pygame.draw.polygon(poly, (*color, poly_alpha), points)
        pygame.draw.polygon(poly, (*color, line_alpha), points, 3 if is_active else 1)
        surface.blit(poly, (0, 0))
        for pt in points:
            pygame.draw.circle(surface, (*color[:3],), pt, 4 if is_active else 2)

        row_rect = row_map[slot_key]
        row_bg = (18, 28, 42) if is_active else (14, 22, 34)
        pygame.draw.rect(surface, row_bg, row_rect, border_radius=10)
        pygame.draw.rect(surface, (*color,), row_rect, 3 if is_active else 2, border_radius=10)
        pygame.draw.rect(surface, color, (row_rect.x + 10, row_rect.y + 12, 12, 12), border_radius=3)

        label = f"{slot.persona} [{slot.skill}]"
        line1 = font_body.render(label, True, LABEL_FG)
        avg_score = int(sum(slot.scores_history) / max(1, len(slot.scores_history))) if slot.scores_history else slot.score
        stat_text = f"Wins {slot.total_wins}   Deaths {slot.total_deaths}   Kills {slot.total_kills}   Coins {slot.total_coins}   Avg {avg_score}"
        metric_line_a = f"Win {metrics['Win Rate']:.2f}   Survival {metrics['Survival']:.2f}   Coins {metrics['Coins']:.2f}"
        metric_line_b = f"Kills {metrics['Kills']:.2f}   AvgScore {metrics['Avg Score']:.2f}   AntiStall {metrics['Anti-Stall']:.2f}"
        line2 = small_font.render(stat_text, True, (188, 200, 220))
        line3 = small_font.render(metric_line_a, True, color)
        line4 = small_font.render(metric_line_b, True, color)
        surface.blit(line1, (row_rect.x + 32, row_rect.y + 8))
        surface.blit(line2, (row_rect.x + 32, row_rect.y + 31))
        surface.blit(line3, (row_rect.x + 32, row_rect.y + 52))
        surface.blit(line4, (row_rect.x + 32, row_rect.y + 70))

        if focus_key == slot_key:
            lock_text = small_font.render("LOCKED", True, color)
            surface.blit(lock_text, (row_rect.right - lock_text.get_width() - 14, row_rect.y + 10))

    return hovered_key, row_map


# ---------------------------------------------------------------------------
# Grid layout computation
# ---------------------------------------------------------------------------
def compute_grid(n_models: int, screen_w: int, screen_h: int):
    """Return (cols, rows, cell_w, cell_h) for a balanced grid."""
    if n_models == 0:
        return 1, 1, screen_w, screen_h
    cols = math.ceil(math.sqrt(n_models))
    rows = math.ceil(n_models / cols)
    cell_w = (screen_w - CELL_PAD * (cols + 1)) // cols
    cell_h = (screen_h - CELL_PAD * (rows + 1)) // rows
    return cols, rows, cell_w, cell_h


# ---------------------------------------------------------------------------
# Discover all trained models
# ---------------------------------------------------------------------------
def discover_models(game_filter: str | None = None) -> List[Path]:
    best_dir = Path("models/best")
    if not best_dir.exists():
        return []
    models = []
    for folder in sorted(best_dir.iterdir()):
        if folder.is_dir():
            zip_path = folder / "best_model.zip"
            if zip_path.exists():
                if game_filter:
                    try:
                        game, _, _, _ = parse_model_info(str(zip_path))
                    except Exception:
                        continue
                    if game.lower() != game_filter.lower():
                        continue
                models.append(zip_path)
    return models


# ---------------------------------------------------------------------------
# Main grid viewer
# ---------------------------------------------------------------------------
def run_grid(fps: int = 20, max_episodes: int = 5, game_filter: str | None = None):
    # Remove headless driver
    os.environ.pop("SDL_VIDEODRIVER", None)

    model_paths = discover_models(game_filter=game_filter)
    if not model_paths:
        scope = f" for game '{game_filter}'" if game_filter else ""
        print(f"[GridViewer] No models found in models/best/{scope}. Train first.")
        return

    print(f"[GridViewer] Found {len(model_paths)} trained model(s):")
    for p in model_paths:
        print(f"  • {p.parent.name}")

    # Initialize pygame with real display
    pygame.init()
    pygame.font.init()

    # Screen sizing: aim for ~80% of monitor, maintain readability
    disp_info = pygame.display.Info()
    screen_w = min(int(disp_info.current_w * 0.85), 1920)
    screen_h = min(int(disp_info.current_h * 0.85), 1080)

    cols, rows, cell_w, cell_h = compute_grid(len(model_paths), screen_w, screen_h - HEADER_H - FOOTER_H - CELL_PAD)
    # cell_h includes the label bar
    game_cell_h = cell_h - LABEL_H

    print(f"[GridViewer] Grid: {cols}×{rows}  |  Cell: {cell_w}×{cell_h}px")

    screen = pygame.display.set_mode((screen_w, screen_h))
    title_scope = game_filter if game_filter else "all games"
    pygame.display.set_caption(f"PEAK Model Grid - {title_scope} ({len(model_paths)} agents)")
    clock = pygame.time.Clock()

    # Fonts
    try:
        font_label = pygame.font.SysFont("Consolas", 16, bold=True)
        font_sub   = pygame.font.SysFont("Consolas", 13)
        font_score = pygame.font.SysFont("Consolas", 14, bold=True)
        font_big   = pygame.font.SysFont("Consolas", 28, bold=True)
    except Exception:
        font_label = pygame.font.Font(None, 18)
        font_sub   = pygame.font.Font(None, 15)
        font_score = pygame.font.Font(None, 16)
        font_big   = pygame.font.Font(None, 30)

    # Load models
    print("[GridViewer] Loading models...")
    slots: List[ModelSlot] = []
    for mp in model_paths:
        print(f"  Loading {mp.parent.name}...", end=" ")
        slot = ModelSlot(mp, (cell_w, game_cell_h))
        if slot.ok:
            slot.reset()
            print("OK")
        else:
            print(f"FAIL: {slot.error}")
        slots.append(slot)

    ok_count = sum(1 for s in slots if s.ok)
    print(f"[GridViewer] {ok_count}/{len(slots)} models loaded successfully.")
    if ok_count == 0:
        print("[GridViewer] No models could be loaded. Exiting.")
        pygame.quit()
        return

    print(f"\n[GridViewer] Running — {max_episodes} episodes per agent @ {fps} FPS")
    print("  ESC = quit  |  SPACE = pause  |  R = reset all")
    print("  F1-F4 = debug toggles  |  D = all debug  |  N = radar plots\n")

    paused = False
    running = True
    frame_count = 0

    # ── Debug overlay toggles ──
    show_radar = False
    radar_focus_key = None
    radar_hover_key = None
    radar_row_map = {}

    while running:
        # ── Events ──
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    for s in slots:
                        if s.ok:
                            s.reset()
                elif event.key == pygame.K_d:
                    for s in slots:
                        if not s.ok:
                            continue
                        dm = getattr(getattr(s.raw_env, "game", None), "debug_manager", None)
                        if not dm:
                            continue
                        all_on = (
                            dm.show_sensors
                            and dm.show_hitboxes
                            and dm.show_grid
                        )
                        new_state = not all_on
                        dm.show_sensors = new_state
                        dm.show_hitboxes = new_state
                        dm.show_grid = new_state
                elif event.key == pygame.K_n:
                    show_radar = not show_radar
                    if not show_radar:
                        radar_focus_key = None
                        radar_hover_key = None
                        radar_row_map = {}
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and show_radar:
                    hit_key = next((key for key, row_rect in radar_row_map.items() if row_rect.collidepoint(event.pos)), None)
                    if hit_key is not None:
                        radar_focus_key = None if radar_focus_key == hit_key else hit_key
                    else:
                        radar_focus_key = None

        if not running:
            break

        # ── Step all agents ──
        if not paused:
            all_done = True
            for slot in slots:
                if not slot.ok:
                    continue
                if slot.done:
                    if slot.ep_count < max_episodes:
                        slot.reset()
                        all_done = False
                    # else: this slot is finished
                else:
                    slot.step()
                    all_done = False

            if all_done and max_episodes > 0:
                # All agents finished all episodes — hold for 3 seconds then exit
                time.sleep(3)
                running = False
                continue

        # ── Render grid ──
        screen.fill(BG_COLOR)
        frame_count += 1

        for idx, slot in enumerate(slots):
            col = idx % cols
            row = idx // cols

            x = CELL_PAD + col * (cell_w + CELL_PAD)
            y = HEADER_H + CELL_PAD + row * (cell_h + CELL_PAD)

            pygame.draw.rect(
                screen,
                CARD_BG,
                pygame.Rect(x - 2, y - 2, cell_w + 4, cell_h + 4),
                border_radius=8,
            )

            # Label bar
            label_rect = pygame.Rect(x, y, cell_w, LABEL_H)
            pygame.draw.rect(screen, LABEL_BG, label_rect, border_top_left_radius=6, border_top_right_radius=6)

            if slot.ok:
                # Model name (left)
                lbl = font_label.render(f"{slot.game.upper()}  {slot.label}", True, LABEL_FG)
                screen.blit(lbl, (x + 6, y + 3))

                # Algo badge (right of name)
                sub = font_sub.render(slot.sub_label, True, (140, 150, 170))
                screen.blit(sub, (x + 6, y + LABEL_H - 14))

                # Score + episode (right side)
                info_txt = f"Ep {slot.ep_count+1}/{max_episodes}  Score: {slot.score}"
                info_surf = font_score.render(info_txt, True, SCORE_COLOR)
                screen.blit(info_surf, (x + cell_w - info_surf.get_width() - 6, y + 7))

                # Game cell
                game_rect = pygame.Rect(x, y + LABEL_H, cell_w, game_cell_h)

                # Render game state with overlays
                slot.render_to_surface()

                # Scale the 800×600 game surface to fit the cell
                scaled = pygame.transform.smoothscale(slot.game_surf, (cell_w, game_cell_h))
                screen.blit(scaled, (x, y + LABEL_H))

                # Border color: green if running, red if done
                border_col = BORDER_DONE if (slot.done and slot.ep_count >= max_episodes) else BORDER_ACTIVE
                pygame.draw.rect(screen, border_col, pygame.Rect(x-1, y-1, cell_w+2, cell_h+2), 2, border_radius=8)

                # Stats bar at bottom of cell
                stats = []
                if slot.total_wins > 0:   stats.append(f"W:{slot.total_wins}")
                if slot.total_deaths > 0: stats.append(f"D:{slot.total_deaths}")
                if slot.total_coins > 0:  stats.append(f"C:{slot.total_coins}")
                if stats:
                    stat_txt = font_sub.render("  ".join(stats), True, (200, 220, 255))
                    sbg = pygame.Surface((stat_txt.get_width() + 8, stat_txt.get_height() + 4), pygame.SRCALPHA)
                    sbg.fill((0, 0, 0, 170))
                    screen.blit(sbg, (x + 2, y + LABEL_H + game_cell_h - stat_txt.get_height() - 6))
                    screen.blit(stat_txt, (x + 6, y + LABEL_H + game_cell_h - stat_txt.get_height() - 4))

            else:
                # Error slot — show error
                err_surf = font_sub.render(f"ERROR: {slot.error[:50]}", True, (255, 80, 80))
                screen.blit(err_surf, (x + 6, y + 3))
                pygame.draw.rect(screen, BORDER_DONE, pygame.Rect(x, y + LABEL_H, cell_w, game_cell_h))

        # ── Radar Plot Overlay (N key) ──
        if show_radar:
            ok_slots = [s for s in slots if s.ok and s.ep_count > 0]
            if ok_slots:
                radar_rect = pygame.Rect(60, 80, screen_w - 120, screen_h - HEADER_H - FOOTER_H - 120)
                radar_hover_key, radar_row_map = draw_multi_radar_overlay(
                    screen,
                    radar_rect,
                    ok_slots,
                    font_big,
                    font_sub,
                    pygame.mouse.get_pos(),
                    radar_focus_key,
                )
            else:
                radar_hover_key = None
                radar_row_map = {}
        else:
            radar_hover_key = None
            radar_row_map = {}

        header = pygame.Rect(0, 0, screen_w, HEADER_H)
        pygame.draw.rect(screen, HEADER_BG, header)
        title = font_big.render(f"PEAK WATCH ALL  [{title_scope.upper()}]", True, LABEL_FG)
        screen.blit(title, (12, 6))
        summary = font_sub.render(
            f"{len([s for s in slots if s.ok])} loaded  |  {cols}x{rows} grid  |  {fps} FPS",
            True,
            (140, 156, 184),
        )
        screen.blit(summary, (screen_w - summary.get_width() - 14, 14))

        # Pause overlay
        if paused:
            pause_surf = font_big.render("▐▐  PAUSED  (SPACE to resume)", True, PAUSED_COLOR)
            px = (screen_w - pause_surf.get_width()) // 2
            py = screen_h - FOOTER_H - 90
            bg_rect = pygame.Rect(px - 10, py - 5, pause_surf.get_width() + 20, pause_surf.get_height() + 10)
            pygame.draw.rect(screen, (0, 0, 0), bg_rect)
            screen.blit(pause_surf, (px, py))

        # ── Controls HUD bar (bottom of screen) ──
        bar_h = FOOTER_H
        bar_y = screen_h - bar_h
        bar_surf = pygame.Surface((screen_w, bar_h))
        bar_surf.fill((12, 18, 28))
        bar_surf.set_alpha(220)
        screen.blit(bar_surf, (0, bar_y))
        pygame.draw.line(screen, (60, 40, 80), (0, bar_y), (screen_w, bar_y))

        first_dm = None
        for slot in slots:
            if slot.ok:
                first_dm = getattr(getattr(slot.raw_env, "game", None), "debug_manager", None)
                if first_dm:
                    break

        hud_items = [
            ("ESC", "quit",     None),
            ("SPACE", "pause",  None),
            ("R", "reset",      None),
            ("F1", "shot+arc",  getattr(first_dm, "show_sensors", False) if first_dm else False),
            ("F2", "cam",       getattr(first_dm, "free_cam_active", False) if first_dm else False),
            ("F3", "slow",      getattr(first_dm, "slow_motion", False) if first_dm else False),
            ("F4", "hitboxes",  getattr(first_dm, "show_hitboxes", False) if first_dm else False),
            ("N", "radar",      show_radar),
            ("D", "all debug",  bool(first_dm and first_dm.show_sensors and first_dm.show_hitboxes and first_dm.show_grid)),
        ]
        hx = 12
        for key, label, active in hud_items:
            key_surf = font_sub.render(key, True, (140, 160, 220))
            screen.blit(key_surf, (hx, bar_y + 5))
            hx += key_surf.get_width() + 3

            if active is None:
                lbl_col = (100, 105, 130)
            elif active:
                lbl_col = (85, 220, 120)
            else:
                lbl_col = (100, 105, 130)
            lbl_surf = font_sub.render(label, True, lbl_col)
            screen.blit(lbl_surf, (hx, bar_y + 5))
            hx += lbl_surf.get_width() + 16

        pygame.display.flip()
        clock.tick(fps)

    # ── Cleanup ──
    print("\n[GridViewer] Shutting down...")
    for slot in slots:
        slot.close()
    pygame.quit()

    print("[GridViewer] Done.")
    for slot in slots:
        if slot.ok:
            print(f"  {slot.label:30s}  {slot.ep_count} ep(s)  last_score={slot.score}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="PEAK Grid Viewer — watch all trained models.")
    ap.add_argument("--fps",      type=int, default=20,  help="Render FPS (default 20)")
    ap.add_argument("--episodes", type=int, default=5,   help="Episodes per agent (default 5, 0=infinite)")
    ap.add_argument("--game",     type=str, default=None, help="Optional game filter (platformer or megaman)")
    args = ap.parse_args()

    run_grid(fps=args.fps, max_episodes=args.episodes, game_filter=args.game)


if __name__ == "__main__":
    main()
