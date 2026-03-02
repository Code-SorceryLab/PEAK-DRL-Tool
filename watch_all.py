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

            # Force camera update (game core skips this in non-human mode)
            self._force_camera_update()

            if ep_done:
                self.ep_count += 1
                self.done = True
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
    args = ap.parse_args()

    run_grid(fps=args.fps, max_episodes=args.episodes)


if __name__ == "__main__":
    main()