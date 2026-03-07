#!/usr/bin/env python3
"""
watch_all.py — Security-grid viewer: watch ALL trained models play side-by-side.

Discovers every best_model.zip under models/best/, loads each with its
VecNormalize stats, and renders them in a tiled pygame grid window.

Place at REPO ROOT (next to menu.py, watch_agent.py).

Usage (from repo root):
    python watch_all.py [--fps 20] [--episodes 5]

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
LABEL_H        = 28          # height of label bar above each cell
CELL_PAD       = 4           # padding between cells
BG_COLOR       = (18, 8, 24) # dark purple background
LABEL_BG       = (30, 14, 42)
LABEL_FG       = (200, 210, 230)
BORDER_ACTIVE  = (80, 200, 120)
BORDER_DONE    = (120, 40, 40)
SCORE_COLOR    = (255, 210, 80)
PAUSED_COLOR   = (255, 80, 80)

# Radar chart settings
RADAR_COLORS   = [
    (255, 100, 120), (80, 200, 255), (140, 255, 120),
    (255, 200, 60),  (200, 120, 255), (255, 160, 60),
    (60, 230, 200),  (255, 80, 200),
]


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
            # CRITICAL: disable free_cam so handle_input() is never suppressed.
            # When render_mode="none", DebugManager(default_active=False) leaves
            # free_cam_active=True, which triggers the guard in platformer_core.step():
            #   if not debug_manager.free_cam_active: player.handle_input(a=action)
            # With free_cam_active=True the player velocity is zeroed every frame.
            _game = getattr(self.raw_env, "game", None)
            if _game and hasattr(_game, "debug_manager"):
                _game.debug_manager.free_cam_active = False
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
        # Per-episode performance tracking (for radar chart)
        self.ep_scores:   list = []
        self.ep_steps:    list = []
        self.ep_goals:    list = []   # 1.0 = won, 0.0 = died
        self.ep_coins:    list = []
        self.ep_progress: list = []   # max_x_seen normalised [0,1]
        self.ok = True

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
        self._last_info: dict = {}

        # Reset camera so it snaps to player start position
        game = self.raw_env.game if hasattr(self.raw_env, 'game') else None
        if game:
            # Re-apply free_cam=False after every game reset (game.reset() may
            # re-initialise DebugManager state and restore free_cam_active=True).
            if hasattr(game, "debug_manager"):
                game.debug_manager.free_cam_active = False
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
            if self.is_recurrent:
                action, self.lstm_states = self.model.predict(
                    self.obs, state=self.lstm_states,
                    episode_start=self.ep_start, deterministic=True)
                self.ep_start = np.zeros((1,), dtype=bool)
            else:
                action, _ = self.model.predict(self.obs, deterministic=True)

            if self.is_vec:
                self.obs, reward, dones, infos = self.env.step(action)
                info = infos[0] if isinstance(infos, (list, tuple)) else infos
                ep_done = bool(dones[0])
            else:
                self.obs, reward, term, trunc, info = self.env.step(action)
                ep_done = term or trunc

            self.score = info.get("score", self.score) if isinstance(info, dict) else self.score
            self.steps += 1
            if isinstance(info, dict):
                self._last_info = info

            # Force camera update (game core skips this in non-human mode)
            self._force_camera_update()

            if ep_done:
                self.ep_count += 1
                self.done = True
                # Record episode metrics for radar
                game = self.raw_env.game if hasattr(self.raw_env, 'game') else None
                info_d = self._last_info
                self.ep_scores.append(float(self.score))
                self.ep_steps.append(float(self.steps))
                self.ep_goals.append(1.0 if info_d.get("won", False) else 0.0)
                self.ep_coins.append(float(info_d.get("coins_collected", 0)))
                raw_max_x = info_d.get("max_x_seen", 0.0)
                # Normalise progress by level width (fallback to 1)
                level_w = (game.level_data.width if game and hasattr(game, 'level_data') else 1) or 1
                self.ep_progress.append(float(raw_max_x) / float(level_w))
                return True
        except Exception as e:
            self.error = f"Step error: {e}"
            self.done = True
        return False

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

    def render_to_surface(self, show_hitboxes=False, show_grid=False):
        """Draw the current game state onto self.game_surf, with optional overlays."""
        if not self.ok:
            return
        try:
            game = self.raw_env.game if hasattr(self.raw_env, 'game') else None
            if game and hasattr(game, 'render'):
                game.render(self.game_surf, blit_only=True)

                # Render debug overlays directly onto the game surface,
                # bypassing DebugManager.render_overlays() which guards
                # on render_mode == "human".
                dm = getattr(game, 'debug_manager', None)
                if dm:
                    if show_grid and hasattr(dm, 'grid_overlay'):
                        dm.grid_overlay.render(self.game_surf, game)
                    if show_hitboxes and hasattr(dm, 'hitbox_overlay'):
                        dm.hitbox_overlay.render(self.game_surf, game)
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


# ---------------------------------------------------------------------------
# Radar chart rendering
# ---------------------------------------------------------------------------

# Fixed absolute scales per axis — chart only fills fully at peak performance.
# (score, survival_steps, goal_rate_0to1, coins, progress_0to1)
RADAR_AXIS_SCALES = [
    # (label, unit_str, max_value, ring_labels)
    ("Score",     "",    2000.0, ["500", "1000", "1500", "2000"]),
    ("Survival",  "steps", 1800.0, ["450", "900",  "1350", "1800"]),
    ("Goal Rate", "%",   1.0,   ["25%", "50%",  "75%",  "100%"]),
    ("Coins",     "",    20.0,  ["5",   "10",   "15",   "20"  ]),
    ("Progress",  "%",   1.0,   ["25%", "50%",  "75%",  "100%"]),
]
RADAR_METRICS = [s[0] for s in RADAR_AXIS_SCALES]


def _slot_radar_values(slot) -> list:
    """Return raw averaged values per axis (un-normalised)."""
    def _safe_avg(lst): return float(sum(lst) / len(lst)) if lst else 0.0
    return [
        _safe_avg(slot.ep_scores),
        _safe_avg(slot.ep_steps),
        _safe_avg(slot.ep_goals),       # already 0-1
        _safe_avg(slot.ep_coins),
        _safe_avg(slot.ep_progress),    # already 0-1
    ]


def _render_text(fonts, key, text, color, fallback_size=16):
    try:
        return fonts[key].render(text, True, color)
    except Exception:
        return pygame.font.Font(None, fallback_size).render(text, True, color)


def draw_radar_overlay(screen, slots, fonts):
    """Draw a full-screen radar chart with fixed absolute scales."""
    import math as _math

    sw, sh = screen.get_size()
    ok_slots = [s for s in slots if s.ok and s.ep_count > 0]
    if not ok_slots:
        return

    raw   = [_slot_radar_values(s) for s in ok_slots]
    n_axes = len(RADAR_AXIS_SCALES)

    # Clamp & normalise against FIXED scales (not peer-relative)
    normed = []
    for rvals in raw:
        row = []
        for ax, (_, _, ax_max, _) in enumerate(RADAR_AXIS_SCALES):
            row.append(min(1.0, max(0.0, rvals[ax] / ax_max)))
        normed.append(row)

    # ── Background dim ───────────────────────────────────────────────────────
    dim = pygame.Surface((sw, sh), pygame.SRCALPHA)
    dim.fill((0, 0, 0, 190))
    screen.blit(dim, (0, 0))

    # ── Layout ───────────────────────────────────────────────────────────────
    legend_w    = 300
    radar_area_w = sw - legend_w - 40
    cx = radar_area_w // 2 + 20
    cy = sh // 2 + 10
    radius = int(min(radar_area_w * 0.44, sh * 0.40))

    N_RINGS = 4

    # ── Grid rings with labels ────────────────────────────────────────────────
    ring_surf = pygame.Surface((sw, sh), pygame.SRCALPHA)
    for ring in range(1, N_RINGS + 1):
        r   = int(radius * ring / N_RINGS)
        frac = ring / N_RINGS
        # Darken inner rings, brighten outer
        alpha_fill  = 30 + ring * 8
        alpha_line  = 70 + ring * 15
        pts = []
        for ax in range(n_axes):
            ang = _math.pi / 2 + 2 * _math.pi * ax / n_axes
            pts.append((cx + r * _math.cos(ang), cy - r * _math.sin(ang)))
        if len(pts) >= 3:
            pygame.draw.polygon(ring_surf, (60, 40, 90, alpha_fill), pts, 0)
            pygame.draw.polygon(ring_surf, (140, 100, 180, alpha_line), pts, 1)
    screen.blit(ring_surf, (0, 0))

    # ── Axes & labels ────────────────────────────────────────────────────────
    for ax, (lbl, unit, ax_max, ring_lbls) in enumerate(RADAR_AXIS_SCALES):
        ang = _math.pi / 2 + 2 * _math.pi * ax / n_axes
        ex = cx + radius * _math.cos(ang)
        ey = cy - radius * _math.sin(ang)
        pygame.draw.line(screen, (100, 80, 140), (cx, cy), (int(ex), int(ey)), 1)

        # Axis tip label (metric name + max)
        lbl_str = f"{lbl}"
        lbl_surf = _render_text(fonts, 'label', lbl_str, (220, 210, 255), 18)
        tip_x = cx + (radius + 36) * _math.cos(ang) - lbl_surf.get_width() // 2
        tip_y = cy - (radius + 36) * _math.sin(ang) - lbl_surf.get_height() // 2
        screen.blit(lbl_surf, (int(tip_x), int(tip_y)))

        # Max value label just inside tip
        max_str = f"max {int(ax_max) if ax_max >= 10 else ax_max}{unit}"
        max_surf = _render_text(fonts, 'sub', max_str, (130, 110, 160), 14)
        mx = cx + (radius + 14) * _math.cos(ang) - max_surf.get_width() // 2
        my = cy - (radius + 14) * _math.sin(ang) - max_surf.get_height() // 2
        screen.blit(max_surf, (int(mx), int(my)))

        # Ring scale labels on first axis (top = Score)
        if ax == 0:
            for ring_i, rlbl in enumerate(ring_lbls, 1):
                r_pos = radius * ring_i / N_RINGS
                r_ang  = ang
                rx = cx + r_pos * _math.cos(r_ang) + 4
                ry = cy - r_pos * _math.sin(r_ang) - 4
                rs = _render_text(fonts, 'sub', rlbl, (120, 100, 150), 13)
                screen.blit(rs, (int(rx), int(ry)))

    # ── Agent polygons ───────────────────────────────────────────────────────
    agent_surf = pygame.Surface((sw, sh), pygame.SRCALPHA)
    for i, (slot, nvals) in enumerate(zip(ok_slots, normed)):
        col_rgb  = RADAR_COLORS[i % len(RADAR_COLORS)]
        col_fill = col_rgb + (40,)
        col_line = col_rgb + (210,)

        pts = []
        for ax in range(n_axes):
            ang = _math.pi / 2 + 2 * _math.pi * ax / n_axes
            r   = radius * max(0.01, nvals[ax])
            pts.append((cx + r * _math.cos(ang), cy - r * _math.sin(ang)))

        if len(pts) >= 3:
            pygame.draw.polygon(agent_surf, col_fill, pts, 0)
            pygame.draw.polygon(agent_surf, col_line, pts, 2)
        for px, py in pts:
            pygame.draw.circle(agent_surf, col_line, (int(px), int(py)), 5)

        # Value labels directly on the polygon vertices
        for ax, (px, py) in enumerate(pts):
            rvals_ax  = raw[i][ax]
            _, unit, ax_max, _ = RADAR_AXIS_SCALES[ax]
            if ax in (2, 4):  # goal rate, progress → show as %
                v_str = f"{rvals_ax * 100:.0f}%"
            elif ax_max < 5:
                v_str = f"{rvals_ax:.2f}"
            else:
                v_str = f"{rvals_ax:.0f}"
            v_surf = _render_text(fonts, 'sub', v_str, col_rgb, 13)
            # Offset label outward from centre
            ang = _math.pi / 2 + 2 * _math.pi * ax / n_axes
            ox = int(px + 10 * _math.cos(ang) - v_surf.get_width() // 2)
            oy = int(py - 10 * _math.sin(ang) - v_surf.get_height() // 2)
            agent_surf.blit(v_surf, (ox, oy))

    screen.blit(agent_surf, (0, 0))

    # ── Title ────────────────────────────────────────────────────────────────
    title_surf = _render_text(fonts, 'label', "Agent Performance  [P to hide]", (230, 220, 255), 20)
    screen.blit(title_surf, (cx - title_surf.get_width() // 2, 14))

    # ── Legend panel ─────────────────────────────────────────────────────────
    lx0 = sw - legend_w - 12
    ly0 = 50
    leg_h = min(sh - ly0 - 16, 30 + len(ok_slots) * 62)
    leg_bg = pygame.Surface((legend_w + 8, leg_h), pygame.SRCALPHA)
    leg_bg.fill((14, 6, 26, 215))
    screen.blit(leg_bg, (lx0 - 4, ly0 - 4))

    # Header
    hdr = _render_text(fonts, 'label', "PERFORMANCE SUMMARY", (180, 160, 230), 16)
    screen.blit(hdr, (lx0, ly0))
    ly0 += hdr.get_height() + 8

    for i, (slot, rvals) in enumerate(zip(ok_slots, raw)):
        col = RADAR_COLORS[i % len(RADAR_COLORS)]

        # Agent name row
        pygame.draw.rect(screen, col, pygame.Rect(lx0, ly0 + 2, 12, 12))
        name_surf = _render_text(fonts, 'score', slot.label[:22], (220, 210, 240), 15)
        screen.blit(name_surf, (lx0 + 18, ly0))
        ly0 += name_surf.get_height() + 3

        # Metric rows with mini bar graphs
        metric_data = [
            ("Score",    rvals[0], 2000.0, f"{rvals[0]:.0f}"),
            ("Survival", rvals[1], 1800.0, f"{rvals[1]:.0f} steps"),
            ("Goal Rate",rvals[2], 1.0,    f"{rvals[2]*100:.0f}%"),
            ("Coins",    rvals[3], 20.0,   f"{rvals[3]:.1f} avg"),
            ("Progress", rvals[4], 1.0,    f"{rvals[4]*100:.0f}%"),
        ]
        bar_w = 90
        for mname, mval, mmax, mstr in metric_data:
            # Label
            ml = _render_text(fonts, 'sub', f"  {mname}:", (140, 130, 160), 13)
            screen.blit(ml, (lx0, ly0))
            # Mini bar
            fill_frac = min(1.0, mval / mmax if mmax > 0 else 0)
            bar_x = lx0 + 76
            bar_y = ly0 + 3
            bar_h = 9
            pygame.draw.rect(screen, (40, 28, 55), pygame.Rect(bar_x, bar_y, bar_w, bar_h))
            if fill_frac > 0:
                # Colour: green > 60%, yellow > 30%, red otherwise
                if fill_frac > 0.6:
                    bc = (60, 200, 100)
                elif fill_frac > 0.3:
                    bc = (210, 180, 50)
                else:
                    bc = (200, 70, 70)
                pygame.draw.rect(screen, bc, pygame.Rect(bar_x, bar_y, int(bar_w * fill_frac), bar_h))
            pygame.draw.rect(screen, (80, 60, 100), pygame.Rect(bar_x, bar_y, bar_w, bar_h), 1)
            # Value text
            vl = _render_text(fonts, 'sub', mstr, (200, 190, 215), 13)
            screen.blit(vl, (bar_x + bar_w + 4, ly0))
            ly0 += ml.get_height() + 1

        ly0 += 8   # gap between agents


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
def discover_models() -> List[Path]:
    best_dir = Path("models/best")
    if not best_dir.exists():
        return []
    models = []
    for folder in sorted(best_dir.iterdir()):
        if folder.is_dir():
            zip_path = folder / "best_model.zip"
            if zip_path.exists():
                models.append(zip_path)
    return models


# ---------------------------------------------------------------------------
# Main grid viewer
# ---------------------------------------------------------------------------
def run_grid(fps: int = 20, max_episodes: int = 5):
    # Remove headless driver
    os.environ.pop("SDL_VIDEODRIVER", None)

    model_paths = discover_models()
    if not model_paths:
        print("[GridViewer] No models found in models/best/. Train first.")
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

    cols, rows, cell_w, cell_h = compute_grid(len(model_paths), screen_w, screen_h)
    # cell_h includes the label bar
    game_cell_h = cell_h - LABEL_H

    print(f"[GridViewer] Grid: {cols}×{rows}  |  Cell: {cell_w}×{cell_h}px")

    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption(f"PEAK — Model Grid  ({len(model_paths)} agents)")
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
    fonts_dict = {'label': font_label, 'sub': font_sub, 'score': font_score, 'big': font_big}

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
    print("  H = hitboxes  |  G = grid  |  D = all debug\n")

    paused = False
    running = True
    frame_count = 0
    show_radar = False

    # ── Debug overlay toggles ──
    show_hitboxes = False
    show_grid     = False

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
                elif event.key == pygame.K_h:
                    show_hitboxes = not show_hitboxes
                elif event.key == pygame.K_g:
                    show_grid = not show_grid
                elif event.key == pygame.K_d:
                    # Toggle all debug overlays at once
                    all_on = show_hitboxes and show_grid
                    show_hitboxes = not all_on
                    show_grid     = not all_on
                elif event.key == pygame.K_p:
                    show_radar = not show_radar

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
                # All agents finished — show radar, hold briefly, then exit
                show_radar = True
                if not getattr(run_grid, '_radar_timer_start', None):
                    import time as _t
                    run_grid._radar_timer_start = _t.time()
                elif _t.time() - run_grid._radar_timer_start > 10:
                    running = False

        # ── Render grid ──
        screen.fill(BG_COLOR)
        frame_count += 1

        for idx, slot in enumerate(slots):
            col = idx % cols
            row = idx // cols

            x = CELL_PAD + col * (cell_w + CELL_PAD)
            y = CELL_PAD + row * (cell_h + CELL_PAD)

            # Label bar
            label_rect = pygame.Rect(x, y, cell_w, LABEL_H)
            pygame.draw.rect(screen, LABEL_BG, label_rect)

            if slot.ok:
                # Model name (left)
                lbl = font_label.render(slot.label, True, LABEL_FG)
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
                slot.render_to_surface(
                    show_hitboxes=show_hitboxes,
                    show_grid=show_grid,
                )

                # Scale the 800×600 game surface to fit the cell
                scaled = pygame.transform.smoothscale(slot.game_surf, (cell_w, game_cell_h))
                screen.blit(scaled, (x, y + LABEL_H))

                # Border color: green if running, red if done
                border_col = BORDER_DONE if (slot.done and slot.ep_count >= max_episodes) else BORDER_ACTIVE
                pygame.draw.rect(screen, border_col, pygame.Rect(x-1, y-1, cell_w+2, cell_h+2), 2)

            else:
                # Error slot — show error
                err_surf = font_sub.render(f"ERROR: {slot.error[:50]}", True, (255, 80, 80))
                screen.blit(err_surf, (x + 6, y + 3))
                pygame.draw.rect(screen, BORDER_DONE, pygame.Rect(x, y + LABEL_H, cell_w, game_cell_h))

        # Pause overlay
        if paused:
            pause_surf = font_big.render("▐▐  PAUSED  (SPACE to resume)", True, PAUSED_COLOR)
            px = (screen_w - pause_surf.get_width()) // 2
            py = screen_h - 80
            bg_rect = pygame.Rect(px - 10, py - 5, pause_surf.get_width() + 20, pause_surf.get_height() + 10)
            pygame.draw.rect(screen, (0, 0, 0), bg_rect)
            screen.blit(pause_surf, (px, py))

        # ── Controls HUD bar (bottom of screen) ──
        bar_h = 24
        bar_y = screen_h - bar_h
        bar_surf = pygame.Surface((screen_w, bar_h))
        bar_surf.fill((16, 8, 22))
        bar_surf.set_alpha(220)
        screen.blit(bar_surf, (0, bar_y))
        pygame.draw.line(screen, (60, 40, 80), (0, bar_y), (screen_w, bar_y))

        # Build toggle items
        hud_items = [
            ("ESC", "quit",     None),
            ("SPACE", "pause",  None),
            ("R", "reset",      None),
            ("H", "hitboxes",   show_hitboxes),
            ("G", "grid",       show_grid),
            ("D", "all debug",  show_hitboxes and show_grid),
            ("P", "radar",      show_radar),
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

        if show_radar:
            draw_radar_overlay(screen, slots, fonts_dict)

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
    args = ap.parse_args()

    run_grid(fps=args.fps, max_episodes=args.episodes)


if __name__ == "__main__":
    main()