"""Quick device comparison for PPO training.

Runs a short PPO training job on each available device (cpu always; cuda
and mps if their backends are available) with profiling on, then prints
a comparison table of SPS and the env/fwd/train/eval percentage split.

Usage:
    python -m code.scripts.benchmark_device --steps 10000
"""
import argparse
import csv
import os
import sys
from pathlib import Path

# Headless rendering for environments that need pygame.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from code.callbacks.profiling_callback import ProfilingCallback
from code.scripts.train import _resolve_device
from code.wrappers.generic_env import GameEnv
from code.games.platformer_core import PlatformerCore


def _available_devices():
    devs = ["cpu"]
    if torch.cuda.is_available():
        devs.append("cuda")
    mps_ok = (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )
    if mps_ok:
        devs.append("mps")
    return devs


def _make_env():
    env = GameEnv(
        game_cls=PlatformerCore,
        render_mode="none",
        fps=None,
        max_steps=1000,
    )
    return Monitor(env)


def _summary_from_csv(csv_path: Path) -> dict:
    """Reduce a profiling_log.csv to overall percentages + mean SPS."""
    totals = {"env_step_s": 0.0, "policy_forward_s": 0.0, "train_s": 0.0, "eval_s": 0.0}
    sps_values = []
    with open(csv_path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            for k in totals:
                totals[k] += float(row[k])
            sps_values.append(float(row["sps"]))
    total = sum(totals.values()) or 1.0
    return {
        "sps": sum(sps_values) / max(1, len(sps_values)),
        "env_pct":   100 * totals["env_step_s"] / total,
        "fwd_pct":   100 * totals["policy_forward_s"] / total,
        "train_pct": 100 * totals["train_s"] / total,
        "eval_pct":  100 * totals["eval_s"] / total,
    }


def _run_device(device_spec: str, steps: int, log_root: Path) -> dict:
    device = _resolve_device(device_spec, verbose=True)
    log_dir = log_root / f"bench_{device}"
    log_dir.mkdir(parents=True, exist_ok=True)

    env = DummyVecEnv([_make_env])
    env = VecNormalize(env)

    model = PPO("MultiInputPolicy", env, device=device, n_steps=256, verbose=0)
    profiler = ProfilingCallback(log_dir=str(log_dir), device=device, verbose=0)
    model.learn(total_timesteps=steps, callback=profiler, progress_bar=False)
    env.close()
    return {"device": device, **_summary_from_csv(log_dir / "profiling_log.csv")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=10_000, help="total_timesteps per device")
    p.add_argument(
        "--log-root", type=str, default="mylogs/benchmark_device",
        help="Where to write profiling_log.csv files per device.",
    )
    args = p.parse_args()

    log_root = Path(args.log_root)
    log_root.mkdir(parents=True, exist_ok=True)

    results = []
    for spec in _available_devices():
        try:
            results.append(_run_device(spec, args.steps, log_root))
        except Exception as exc:
            print(f"[ERROR] device='{spec}' failed: {type(exc).__name__}: {exc}", file=sys.stderr)

    if not results:
        print("No devices ran successfully.")
        sys.exit(1)

    print()
    print(f"{'device':<8} {'sps':>8} {'env%':>6} {'fwd%':>6} {'train%':>7} {'eval%':>6}")
    print("-" * 48)
    for r in results:
        print(
            f"{r['device']:<8} {r['sps']:>8.0f} "
            f"{r['env_pct']:>6.0f} {r['fwd_pct']:>6.0f} "
            f"{r['train_pct']:>7.0f} {r['eval_pct']:>6.0f}"
        )


if __name__ == "__main__":
    main()
