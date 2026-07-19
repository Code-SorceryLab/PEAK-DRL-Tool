#!/usr/bin/env python3
"""
eval_level.py — Headless per-level evaluation CLI for PEAK platformer agents.

Must be run as a module from the repo root:
    python -m code.scripts.eval_level --model <model.zip> --level Mario1-1 [options]

The eval environment is pinned to a single level with curriculum disabled and
terminate_on_goal=True, providing a trustworthy, drift-free measurement.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from code.scripts.watch_agent import (
    _HAS_RPPO,
    RecurrentPPO,
    find_vecnorm_path,
    load_model,
    parse_model_info,
)
from code.metrics.eval_summary import summarize_by_level, summarize_eval
from code.wrappers.generic_env import GameEnv


# ---------------------------------------------------------------------------
# Env builder (headless, eval-mode)
# ---------------------------------------------------------------------------

def build_eval_env(
    game: str,
    level: str,
    max_steps: int,
    vecnorm_path: Optional[Path],
    frame_skip: int = 1,
    stall_metric: str = "euclid",
):
    """
    Build a headless DummyVecEnv pinned to *level* with curriculum disabled
    and terminate_on_goal=True.  Mirrors watch_agent.build_env but headless
    and with eval-specific kwargs.

    frame_skip MUST match the value the model was trained with (recorded in
    model_info.json) — a skip-4 policy stepped per-frame acts 4x too often
    and its behaviour is meaningless.
    """
    game_module = importlib.import_module(f"code.games.{game}_core")
    GameCoreClass = getattr(game_module, f"{game.capitalize()}Core")

    game_kwargs = {
        "world": level,
        "curriculum_enabled": False,
        "terminate_on_goal": True,
        # euclid = legacy benchmark rules (comparable to historical numbers);
        # path = corrected rules (vertical play counts as anti-stall progress).
        "stall_metric": stall_metric,
    }

    def _make():
        return GameEnv(
            GameCoreClass,
            render_mode="none",
            max_steps=max_steps,
            frame_skip=frame_skip,
            **game_kwargs,
        )

    venv = DummyVecEnv([_make])

    if vecnorm_path is None:
        warnings.warn(
            "\n[WARN] No _vecnorm.pkl found — observations will NOT be normalised.\n"
            "       Place the _vecnorm.pkl next to the .zip for correct behaviour.\n",
            stacklevel=2,
        )
        return venv

    print(f"[INFO] Loading VecNormalize from: {vecnorm_path}")
    norm_env = VecNormalize.load(str(vecnorm_path), venv)
    norm_env.training = False
    norm_env.norm_reward = False
    print("[INFO] VecNormalize loaded — obs normalised correctly.")
    return norm_env


# ---------------------------------------------------------------------------
# Core episode-running function (importable, injectable for tests)
# ---------------------------------------------------------------------------

def run_eval(
    model,
    env,
    episodes: int,
    deterministic: bool,
    level: str,
) -> List[dict]:
    """
    Run *episodes* episodes and return a list of final per-episode info dicts.

    Parameters
    ----------
    model
        A loaded SB3 (or compatible) model with a ``predict(obs, ...)`` method.
    env
        A VecEnv (or duck-typed stub) with ``reset()`` / ``step(action)`` that
        returns ``(obs, reward, dones, info_list)``.
    episodes
        Number of full episodes to run.
    deterministic
        If True, use deterministic actions; otherwise stochastic.
    level
        Level name to inject into info dicts that don't carry the ``level`` key.

    Returns
    -------
    list[dict]
        One final-info dict per episode, each guaranteed to have ``won``,
        ``cause``, and ``level`` keys.
    """
    is_recurrent = _HAS_RPPO and isinstance(model, RecurrentPPO)
    collected: List[dict] = []

    for _ in range(episodes):
        obs = env.reset()
        done = False
        lstm_states = None
        ep_start = np.ones((1,), dtype=bool)
        final_info: dict = {}

        while not done:
            if is_recurrent:
                action, lstm_states = model.predict(
                    obs,
                    state=lstm_states,
                    episode_start=ep_start,
                    deterministic=deterministic,
                )
                ep_start = np.zeros((1,), dtype=bool)
            else:
                action, _ = model.predict(obs, deterministic=deterministic)

            obs, _reward, dones, info = env.step(action)
            done = bool(dones[0])
            raw_info = info[0] if isinstance(info, (list, tuple)) else info
            if done:
                final_info = dict(raw_info)

        # Ensure mandatory keys are present
        if "level" not in final_info or not final_info.get("level"):
            final_info["level"] = level
        if "won" not in final_info:
            final_info["won"] = False
        if "cause" not in final_info:
            final_info["cause"] = ""

        collected.append(final_info)

    return collected


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------

def _json_safe(o):
    """Recursively convert numpy / non-finite values so json.dump cannot raise.
    Crucially preserves bool semantics (np.bool_ -> bool) so 'won' stays correct.
    """
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, np.ndarray):
        return _json_safe(o.tolist())
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, (str, int, bool)) or o is None:
        return o
    return str(o)


def _print_summary(episodes: List[dict]) -> None:
    """Print a human-readable per-level breakdown table."""
    by_level = summarize_by_level(episodes)
    overall = summarize_eval(episodes)

    print("\n" + "=" * 64)
    print(f"{'LEVEL':<20} {'N':>4} {'WINS':>5} {'WIN%':>7}  BY CAUSE")
    print("-" * 64)

    for lvl, s in sorted(by_level.items()):
        cause_str = "  ".join(
            f"{k}={v}" for k, v in s["by_cause"].items() if v > 0
        ) or "—"
        print(
            f"{lvl:<20} {s['n']:>4} {s['wins']:>5} {s['win_rate']:>7.1%}  {cause_str}"
        )

    print("-" * 64)
    cause_str = "  ".join(
        f"{k}={v}" for k, v in overall["by_cause"].items() if v > 0
    ) or "—"
    print(
        f"{'TOTAL':<20} {overall['n']:>4} {overall['wins']:>5} "
        f"{overall['win_rate']:>7.1%}  {cause_str}"
    )
    print("=" * 64 + "\n")

    # Determinism-trap guard: a deterministic policy can fall into a fixed action
    # loop that the anti-stall watchdog kills every episode, yielding a misleading
    # 0% win rate even when the policy is capable stochastically. Warn loudly.
    n = overall["n"]
    stalls = overall["by_cause"].get("stall", 0)
    if n > 0 and overall["wins"] == 0 and stalls >= 0.8 * n:
        print(
            f"[WARN] {stalls}/{n} episodes ended in STALL with 0 wins. This often\n"
            f"       means a DETERMINISTIC action loop is tripping the anti-stall\n"
            f"       watchdog, NOT a true capability floor. Re-run with --stochastic\n"
            f"       to measure the policy's actual win rate before drawing conclusions.\n"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Headless per-level evaluation of a PEAK platformer agent."
    )
    ap.add_argument("--model",     required=True, help="Path to the .zip model file")
    ap.add_argument("--game",      default="platformer", help="Game name (default: platformer)")
    ap.add_argument("--algo",      default=None, help="Algorithm (auto-detected if omitted)")
    ap.add_argument("--level",     required=True, help="Level name to pin, e.g. Mario1-1")
    ap.add_argument("--episodes",  type=int, default=50, help="Number of episodes (default: 50)")
    ap.add_argument("--vecnorm",   default=None, help="Path to _vecnorm.pkl (auto-detected if omitted)")
    ap.add_argument("--max_steps", type=int, default=8000, help="Max steps per episode (default: 8000)")
    ap.add_argument("--out",       default=None, help="Optional JSON output path")
    ap.add_argument("--stochastic", action="store_true", help="Use stochastic actions (default: deterministic)")
    ap.add_argument("--frame-skip", type=int, default=None, dest="frame_skip",
                    help="Action repeat — MUST match training (model_info.json). "
                         "Auto-detected from model_info.json beside the model; defaults to 1.")
    ap.add_argument("--stall-metric", choices=["euclid", "path"], default="euclid",
                    dest="stall_metric",
                    help="Anti-stall progress test for the EVAL env. euclid = legacy "
                         "benchmark rules (comparable to historical numbers); path = "
                         "corrected rules (vertical play counts as progress).")
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    # Derive metadata from path when CLI args not provided
    try:
        p_game, p_algo, p_persona, p_skill = parse_model_info(str(model_path))
    except Exception as e:
        print(f"[WARN] Could not parse model info: {e}")
        p_game, p_algo, p_persona, p_skill = "platformer", "ppo", "unknown", "unknown"

    game = args.game or p_game
    algo = args.algo or p_algo

    # Resolve frame_skip: explicit CLI > model_info.json beside the model > 1.
    # A skip-4 policy stepped per-frame acts 4x too often — silently wrong.
    frame_skip = args.frame_skip
    if frame_skip is None:
        info_path = model_path.parent / "model_info.json"
        if info_path.exists():
            try:
                frame_skip = int(json.loads(info_path.read_text()).get("frame_skip", 1))
                print(f"[INFO] frame_skip={frame_skip} (from {info_path.name})")
            except Exception:
                frame_skip = 1
        else:
            frame_skip = 1

    print(f"[INFO] game={game}  algo={algo.upper()}")
    print(f"[INFO] level={args.level}  episodes={args.episodes}  "
          f"max_steps={args.max_steps}  deterministic={not args.stochastic}  "
          f"frame_skip={frame_skip}")

    # Load model
    model = load_model(str(model_path), algo)

    # Resolve vecnorm
    vn_path: Optional[Path]
    if args.vecnorm:
        vn_path = Path(args.vecnorm)
        if not vn_path.exists():
            print(f"[WARN] --vecnorm path not found: {vn_path}; will run unnormalised")
            vn_path = None
    else:
        vn_path = find_vecnorm_path(str(model_path))

    # Build env
    env = build_eval_env(
        game=game,
        level=args.level,
        max_steps=args.max_steps,
        vecnorm_path=vn_path,
        frame_skip=frame_skip,
        stall_metric=args.stall_metric,
    )

    # Run
    print(f"\n[INFO] Running {args.episodes} episode(s) on level '{args.level}' ...\n")
    try:
        episodes_data = run_eval(
            model=model,
            env=env,
            episodes=args.episodes,
            deterministic=not args.stochastic,
            level=args.level,
        )
    finally:
        try:
            env.close()
        except Exception:
            pass

    # Output
    _print_summary(episodes_data)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(_json_safe(episodes_data), f, indent=2)
        print(f"[INFO] Episodes saved to {out_path}")


if __name__ == "__main__":
    main()
