"""Per-rollout PPO profiling: env step %, forward %, train %, eval %, SPS.

Emits TensorBoard scalars under ``profile/`` and a one-row-per-rollout CSV
at ``<log_dir>/profiling_log.csv``.  Optional ``EvalTimerCallback`` shim
is added in a later task to capture eval wall time from an EvalCallback.

Device sync (``torch.cuda.synchronize`` / ``torch.mps.synchronize``) is
opt-in via ``sync_device`` — defaults to True for cuda/mps, False for cpu.
"""
import csv
import time
from pathlib import Path
from typing import Optional

import torch
from stable_baselines3.common.callbacks import BaseCallback


_CSV_COLUMNS = [
    "step", "device", "n_envs",
    "rollout_wall_s", "env_step_s", "policy_forward_s",
    "train_s", "eval_s", "sps",
]


def _sync(device: str) -> None:
    """Block until pending kernels on the given device finish, if applicable."""
    if device == "cuda":
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
    elif device == "mps":
        try:
            if hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
        except Exception:
            pass


class ProfilingCallback(BaseCallback):
    def __init__(
        self,
        log_dir: str,
        device: str,
        sync_device: Optional[bool] = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.log_dir = Path(log_dir)
        self.device = device
        self.sync_device = (
            (device in ("cuda", "mps")) if sync_device is None else bool(sync_device)
        )

        self._env_step_s: float = 0.0
        self._rollout_start_t: Optional[float] = None
        self._train_start_t: Optional[float] = None
        self._last_step_t: Optional[float] = None
        self._eval_s_carry: float = 0.0

        self._totals = {
            "env_step_s": 0.0,
            "forward_s": 0.0,
            "train_s": 0.0,
            "eval_s": 0.0,
            "rollout_wall_s": 0.0,
        }
        self._last_train_s: float = 0.0

        self._csv_path: Optional[Path] = None
        self._csv_file = None
        self._csv_writer = None

    def _init_callback(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = self.log_dir / "profiling_log.csv"
        self._csv_file = open(self._csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(_CSV_COLUMNS)
        self._csv_file.flush()

    def _on_rollout_start(self) -> None:
        # Close out the previous train block, if any.
        if self._train_start_t is not None:
            if self.sync_device:
                _sync(self.device)
            self._last_train_s = time.perf_counter() - self._train_start_t
            self._totals["train_s"] += self._last_train_s
            self._train_start_t = None
        else:
            self._last_train_s = 0.0

        # Reset and start the new rollout.
        self._env_step_s = 0.0
        if self.sync_device:
            _sync(self.device)
        now = time.perf_counter()
        self._rollout_start_t = now
        self._last_step_t = now

    def _on_step(self) -> bool:
        now = time.perf_counter()
        if self._last_step_t is not None:
            self._env_step_s += now - self._last_step_t
        self._last_step_t = now
        return True

    def _on_rollout_end(self) -> None:
        if self.sync_device:
            _sync(self.device)
        t_now = time.perf_counter()
        rollout_wall_s = t_now - (self._rollout_start_t if self._rollout_start_t is not None else t_now)
        forward_s = max(0.0, rollout_wall_s - self._env_step_s)
        eval_s = self._eval_s_carry
        self._eval_s_carry = 0.0

        try:
            n_envs = int(self.training_env.num_envs)
        except Exception:
            n_envs = 1
        n_steps = int(getattr(self.model, "n_steps", 2048) or 2048)
        rollout_steps = max(1, n_steps * n_envs)
        sps = rollout_steps / max(1e-9, rollout_wall_s)

        # Accumulate totals (train_s is added at _on_rollout_start of next rollout).
        self._totals["env_step_s"] += self._env_step_s
        self._totals["forward_s"] += forward_s
        self._totals["eval_s"] += eval_s
        self._totals["rollout_wall_s"] += rollout_wall_s

        # CSV row. Diagnostic only — never block training on write failure.
        try:
            self._csv_writer.writerow([
                self.num_timesteps, self.device, n_envs,
                f"{rollout_wall_s:.6f}", f"{self._env_step_s:.6f}", f"{forward_s:.6f}",
                f"{self._last_train_s:.6f}", f"{eval_s:.6f}", f"{sps:.2f}",
            ])
            self._csv_file.flush()
        except Exception as exc:
            print(f"[ProfilingCallback] CSV write failed: {type(exc).__name__}: {exc}")

        # TB scalars.
        self.logger.record("profile/rollout_wall_s", rollout_wall_s)
        self.logger.record("profile/env_step_s", self._env_step_s)
        self.logger.record("profile/policy_forward_s", forward_s)
        self.logger.record("profile/train_s", self._last_train_s)
        self.logger.record("profile/eval_s", eval_s)
        self.logger.record("profile/sps", sps)

        # Mark train block start.
        if self.sync_device:
            _sync(self.device)
        self._train_start_t = time.perf_counter()

    def _on_training_end(self) -> None:
        # Close out the final train block if any.
        if self._train_start_t is not None:
            if self.sync_device:
                _sync(self.device)
            self._totals["train_s"] += time.perf_counter() - self._train_start_t
            self._train_start_t = None

        total = (
            self._totals["env_step_s"]
            + self._totals["forward_s"]
            + self._totals["train_s"]
            + self._totals["eval_s"]
        )
        if total > 0:
            env_pct = 100 * self._totals["env_step_s"] / total
            fwd_pct = 100 * self._totals["forward_s"] / total
            train_pct = 100 * self._totals["train_s"] / total
            eval_pct = 100 * self._totals["eval_s"] / total
            bottleneck = max(
                (("ENV", env_pct), ("FWD", fwd_pct), ("TRAIN", train_pct), ("EVAL", eval_pct)),
                key=lambda x: x[1],
            )[0]
            print(
                f"[Profile] env={env_pct:.0f}% fwd={fwd_pct:.0f}% "
                f"train={train_pct:.0f}% eval={eval_pct:.0f}% → bottleneck: {bottleneck}"
            )

        if self._csv_file is not None:
            try:
                self._csv_file.close()
            except Exception:
                pass


class EvalTimerCallback(BaseCallback):
    """Wraps an EvalCallback and reports its wall time to a ProfilingCallback.

    Composition mirrors the VecnormBestCallback pattern in train.py:36 —
    we call ``inner_cb.on_step()`` (public, increments n_calls), and on
    eval-frequency steps we measure perf_counter around the call.
    """

    def __init__(self, inner_cb, profiler: "ProfilingCallback", verbose: int = 0):
        super().__init__(verbose)
        self.inner_cb = inner_cb
        self.profiler = profiler

    def _init_callback(self) -> None:
        self.inner_cb.init_callback(self.model)

    def _on_step(self) -> bool:
        # Resolve eval_freq/n_calls from inner_cb; if inner_cb is a wrapper
        # (e.g. EvalPreviewCallback) it exposes the underlying EvalCallback
        # as inner_cb.eval_cb — look one level deeper in that case.
        eval_freq = getattr(self.inner_cb, "eval_freq", None)
        n_calls = getattr(self.inner_cb, "n_calls", 0)
        if eval_freq is None:
            wrapped = getattr(self.inner_cb, "eval_cb", None)
            if wrapped is not None:
                eval_freq = getattr(wrapped, "eval_freq", None)
                n_calls = getattr(wrapped, "n_calls", n_calls)

        # Predict whether THIS call will fire eval so we only pay sync cost on those.
        next_n_calls = n_calls + 1
        will_eval = bool(eval_freq) and (next_n_calls % int(eval_freq) == 0)

        if not will_eval:
            return self.inner_cb.on_step()

        if self.profiler.sync_device:
            _sync(self.profiler.device)
        t0 = time.perf_counter()
        result = self.inner_cb.on_step()
        if self.profiler.sync_device:
            _sync(self.profiler.device)
        self.profiler._eval_s_carry += time.perf_counter() - t0
        return result

    def _on_training_end(self) -> None:
        end_fn = getattr(self.inner_cb, "_on_training_end", None)
        if callable(end_fn):
            end_fn()
