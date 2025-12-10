# rl_agents_balance/scripts/evaluate.py
# ------------------------------------------------------------
"""
Generic evaluator / visualiser.

• Picks a trained model from the canonical folder layout:
      models/<game>/<algo>/<filename>.zip
• Runs N episodes, can render or head-less.
• Collects per-episode metrics via the game's MetricsCollector (optional).
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pygame
from tqdm import tqdm

# Gym is only used indirectly via your GameEnv wrapper
import gymnasium as gym  # noqa: F401

from stable_baselines3 import PPO, A2C, DQN, SAC  # extend if needed

# ---------- map short names to SB3 classes ----------
ALGO_REG = dict(ppo=PPO, a2c=A2C, dqn=DQN, sac=SAC)


# ---------- helpers ----------
def newest_zip(path: Path) -> Path | None:
    """Return newest .zip in a folder or None."""
    zips = sorted(path.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    return zips[0] if zips else None


def convert_np(obj):
    """Convert numpy scalars/arrays in nested structures to Python types for JSON."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: convert_np(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_np(v) for v in obj]
    return obj


# ---------- CLI ----------
parser = argparse.ArgumentParser()
parser.add_argument("--game", default="flappy", help="game key (e.g., flappy, tetris)")
parser.add_argument("--algo", default="ppo", help="algo key (ppo/a2c/dqn/sac)")
parser.add_argument("--model", help="explicit model .zip path")
parser.add_argument("--episodes", type=int, default=5)
parser.add_argument("--render", choices=["human", "none"], default="human")
parser.add_argument("--fps", type=int, default=30)
parser.add_argument(
    "--metrics",
    default="",
    help="import-path to MetricsCollector class (e.g., code.metrics.tetris_balance.TetrisBalanceStats)",
)
parser.add_argument("--out", help="Path to dump episode summaries as JSON list")
parser.add_argument(
    "--max_steps",
    type=int,
    default=5000,
    help="hard cap on steps per episode; episode will truncate at this limit",
)
parser.add_argument(
    "--ep_timeout_s",
    type=float,
    default=60.0,
    help="watchdog: truncate the episode if wall-clock time exceeds this many seconds",
)
parser.add_argument(
    "--device",
    default="cpu",
    help="device for SB3 model (cpu/cuda) — defaults to cpu for portability",
)
args = parser.parse_args()


# ---------- dynamic imports ----------
MetricsCls = None
if args.metrics:
    try:
        mod_name, cls_name = args.metrics.rsplit(".", 1)
        MetricsCls = getattr(importlib.import_module(mod_name), cls_name)
    except Exception as e:
        sys.exit(f"[!] Failed to import metrics class '{args.metrics}': {e}")

try:
    game_mod = importlib.import_module(f"code.games.{args.game}_core")
    GameCls = getattr(game_mod, next(attr for attr in dir(game_mod) if attr.endswith("Core")))
except (ModuleNotFoundError, StopIteration):
    sys.exit(f"[!] unknown game key '{args.game}' (expected code.games.{args.game}_core.<SomethingCore>)")

AlgoCls = ALGO_REG.get(args.algo.lower())
if AlgoCls is None:
    sys.exit(f"[!] unknown algo '{args.algo}'. Available: {list(ALGO_REG)}")


# ---------- locate model ----------
model_path = Path(args.model) if args.model else None
if model_path is None:
    # fallback to newest zip in canonical folder
    model_path = newest_zip(Path("models") / args.game / args.algo)
    if not model_path:
        sys.exit("[!] No checkpoint found — train first or pass --model")

print("✓ loading", model_path)
try:
    model = AlgoCls.load(model_path, device=args.device)
except Exception as e:
    sys.exit(f"[!] Failed to load model: {e}")


# ---------- build env ----------
from code.wrappers.generic_env import GameEnv  # import after args parsing for clarity

env = GameEnv(
    GameCls,
    render_mode=args.render,
    fps=args.fps if args.render == "human" else None,
    max_steps=args.max_steps,  # IMPORTANT: let wrapper truncate episodes too
    reward_fn=None,  # evaluation: use game's base reward or metrics only
)

# ---------- evaluation loop ----------
all_ep: list[dict] = []

try:
    for ep in tqdm(range(args.episodes), desc="Evaluating Episodes"):
        obs, _info = env.reset()
        done = False
        steps = 0
        t0 = time.monotonic()
        stats = MetricsCls() if MetricsCls else None

        while not done:
            # policy action
            act, _ = model.predict(obs, deterministic=True)

            # env step
            obs, rew, term, trunc, info = env.step(act)

            # metrics collector
            if stats:
                try:
                    stats.on_step(obs, act, rew, term or trunc, info)
                except Exception as e:
                    # Don't crash eval if a metrics bug appears
                    print(f"[metrics warn] on_step failed: {e}")

            # human render
            if args.render == "human":
                env.render()
                pygame.event.pump()
                # simple pacing for visual comfort
                if args.fps and args.fps > 0:
                    time.sleep(1 / args.fps)

            # watchdogs
            steps += 1
            if steps >= args.max_steps:
                trunc = True
            if (time.monotonic() - t0) > args.ep_timeout_s:
                trunc = True

            done = bool(term or trunc)

        # episode summary (either from metrics or final info)
        try:
            summary = stats.summary() if stats else info
        except Exception as e:
            print(f"[metrics warn] summary() failed: {e}")
            summary = info

        all_ep.append(convert_np(summary))
        print(f"Episode {ep + 1}: {summary}")

        # write rolling JSON (so long evals still produce partial results)
        if args.out:
            try:
                Path(args.out).write_text(json.dumps(all_ep, indent=2))
            except Exception as e:
                print(f"[warn] failed to write {args.out}: {e}")

except KeyboardInterrupt:
    print("\n[!] Evaluation interrupted by user.")

finally:
    try:
        env.close()
    except Exception:
        pass

print(f"Finished {len(all_ep)} episode(s).")
if args.out:
    print(f"✓ metrics saved → {args.out}")
