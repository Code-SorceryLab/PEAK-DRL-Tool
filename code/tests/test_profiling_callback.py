"""Unit tests for code.callbacks.profiling_callback.ProfilingCallback.

Uses fake model/env/logger objects to avoid running real PPO training.
"""
from pathlib import Path

import pytest


# --- Test doubles --------------------------------------------------------

class _FakeLogger:
    def __init__(self):
        self.records = {}

    def record(self, key, value):
        self.records[key] = value


class _FakeEnv:
    def __init__(self, num_envs=2):
        self.num_envs = num_envs


class _FakeModel:
    def __init__(self, n_steps=4, num_timesteps=0, n_envs=2):
        self.n_steps = n_steps
        self.num_timesteps = num_timesteps
        self._env = _FakeEnv(num_envs=n_envs)
        self.logger = _FakeLogger()

    def get_env(self):
        return self._env


def make_callback(tmp_path, device="cpu", sync_device=None, n_envs=2, n_steps=4):
    # Wire _FakeEnv and _FakeLogger through _FakeModel so SB3 2.8.0's read-only properties resolve correctly.
    from code.callbacks.profiling_callback import ProfilingCallback
    cb = ProfilingCallback(log_dir=str(tmp_path), device=device, sync_device=sync_device)
    cb.model = _FakeModel(n_steps=n_steps, n_envs=n_envs)
    cb._init_callback()
    return cb


# --- Construction & sync_device default ---------------------------------

def test_constructor_sets_attributes(tmp_path):
    from code.callbacks.profiling_callback import ProfilingCallback
    cb = ProfilingCallback(log_dir=str(tmp_path), device="cpu")
    assert cb.device == "cpu"
    assert cb.sync_device is False  # auto default for cpu


def test_sync_device_auto_picks_true_for_cuda(tmp_path):
    from code.callbacks.profiling_callback import ProfilingCallback
    cb = ProfilingCallback(log_dir=str(tmp_path), device="cuda")
    assert cb.sync_device is True


def test_sync_device_auto_picks_true_for_mps(tmp_path):
    from code.callbacks.profiling_callback import ProfilingCallback
    cb = ProfilingCallback(log_dir=str(tmp_path), device="mps")
    assert cb.sync_device is True


def test_sync_device_explicit_override(tmp_path):
    from code.callbacks.profiling_callback import ProfilingCallback
    cb = ProfilingCallback(log_dir=str(tmp_path), device="cuda", sync_device=False)
    assert cb.sync_device is False


# --- init_callback creates CSV with header ------------------------------

def test_init_creates_log_dir(tmp_path):
    target = tmp_path / "fresh"
    cb = make_callback(target)
    assert target.is_dir()


def test_init_creates_csv_with_header(tmp_path):
    cb = make_callback(tmp_path)
    csv_path = tmp_path / "profiling_log.csv"
    assert csv_path.exists()
    header = csv_path.read_text().splitlines()[0]
    for col in [
        "step", "device", "n_envs",
        "rollout_wall_s", "env_step_s", "policy_forward_s",
        "train_s", "eval_s", "sps",
    ]:
        assert col in header, f"missing column '{col}' in header: {header}"


# --- Env-step timing -----------------------------------------------------

def test_on_step_accumulates_env_time(tmp_path, monkeypatch):
    """_on_step should accumulate wall-time deltas between consecutive calls."""
    cb = make_callback(tmp_path)

    # Drive perf_counter from a controlled fake clock.
    clock = [100.0]
    monkeypatch.setattr(
        "code.callbacks.profiling_callback.time.perf_counter",
        lambda: clock[0],
    )

    cb._on_rollout_start()        # baselines _last_step_t at 100.0
    clock[0] = 100.5
    cb._on_step()                 # +0.5
    clock[0] = 101.2
    cb._on_step()                 # +0.7
    assert cb._env_step_s == pytest.approx(1.2, abs=1e-6)


def test_rollout_start_resets_env_accumulator(tmp_path, monkeypatch):
    cb = make_callback(tmp_path)
    clock = [100.0]
    monkeypatch.setattr(
        "code.callbacks.profiling_callback.time.perf_counter",
        lambda: clock[0],
    )
    cb._on_rollout_start()
    clock[0] = 100.4
    cb._on_step()
    assert cb._env_step_s > 0

    clock[0] = 200.0
    cb._on_rollout_start()        # reset
    assert cb._env_step_s == 0.0


# --- Rollout end + CSV row ----------------------------------------------

def _run_one_rollout(cb, clock, monkeypatch, n_steps=4, env_step_dt=0.1, forward_dt=0.05):
    """Drive callback through one full rollout cycle on a fake clock."""
    monkeypatch.setattr(
        "code.callbacks.profiling_callback.time.perf_counter",
        lambda: clock[0],
    )
    cb._on_rollout_start()
    for _ in range(n_steps):
        clock[0] += env_step_dt
        cb._on_step()
    clock[0] += forward_dt  # forward time = residual between last step and rollout_end
    cb._on_rollout_end()


def test_rollout_end_writes_one_csv_row(tmp_path, monkeypatch):
    cb = make_callback(tmp_path, n_envs=2, n_steps=4)
    clock = [1000.0]
    _run_one_rollout(cb, clock, monkeypatch)

    rows = (tmp_path / "profiling_log.csv").read_text().splitlines()
    assert len(rows) == 2  # header + 1 data row
    data = rows[1].split(",")
    # Column order: step, device, n_envs, rollout_wall_s, env_step_s, policy_forward_s, train_s, eval_s, sps
    assert data[1] == "cpu"
    assert data[2] == "2"
    assert float(data[4]) == pytest.approx(0.4, abs=1e-6)   # env_step_s = 4 * 0.1
    # rollout_wall_s = 4*0.1 (env) + 0.05 (residual) = 0.45
    assert float(data[3]) == pytest.approx(0.45, abs=1e-6)
    # policy_forward_s = rollout_wall_s - env_step_s = 0.05
    assert float(data[5]) == pytest.approx(0.05, abs=1e-6)


def test_train_time_measured_between_rollouts(tmp_path, monkeypatch):
    """train_s of rollout N is wall time from rollout_end(N) to rollout_start(N+1)."""
    cb = make_callback(tmp_path)
    clock = [1000.0]
    _run_one_rollout(cb, clock, monkeypatch, env_step_dt=0.1, forward_dt=0.0)
    clock[0] += 0.3   # 0.3s of "training" between rollouts
    _run_one_rollout(cb, clock, monkeypatch, env_step_dt=0.1, forward_dt=0.0)

    rows = (tmp_path / "profiling_log.csv").read_text().splitlines()
    assert len(rows) == 3  # header + 2 data rows
    # First rollout: no prior train block, train_s should be 0.
    assert float(rows[1].split(",")[6]) == pytest.approx(0.0, abs=1e-6)
    # Second rollout: train_s = 0.3
    assert float(rows[2].split(",")[6]) == pytest.approx(0.3, abs=1e-6)


def test_sps_reported(tmp_path, monkeypatch):
    cb = make_callback(tmp_path, n_envs=2, n_steps=4)
    clock = [0.0]
    _run_one_rollout(cb, clock, monkeypatch, env_step_dt=0.1, forward_dt=0.05)
    # rollout_steps = n_steps * n_envs = 4 * 2 = 8
    # rollout_wall_s = 0.45 → sps ≈ 17.78
    rows = (tmp_path / "profiling_log.csv").read_text().splitlines()
    sps = float(rows[1].split(",")[8])
    assert sps == pytest.approx(8.0 / 0.45, rel=1e-3)


# --- TB scalars + summary -----------------------------------------------

def test_rollout_end_records_tb_scalars(tmp_path, monkeypatch):
    cb = make_callback(tmp_path, n_envs=2, n_steps=4)
    clock = [0.0]
    _run_one_rollout(cb, clock, monkeypatch, env_step_dt=0.1, forward_dt=0.05)
    for key in [
        "profile/rollout_wall_s", "profile/env_step_s",
        "profile/policy_forward_s", "profile/train_s",
        "profile/eval_s", "profile/sps",
    ]:
        assert key in cb.logger.records, f"missing TB scalar {key}"


def test_training_end_prints_summary(tmp_path, monkeypatch, capsys):
    cb = make_callback(tmp_path)
    clock = [0.0]
    _run_one_rollout(cb, clock, monkeypatch, env_step_dt=0.1, forward_dt=0.05)
    # No "next" rollout, so call _on_training_end directly.
    cb._on_training_end()
    out = capsys.readouterr().out
    assert "[Profile]" in out
    assert "bottleneck" in out
    # env was 0.4s and forward was 0.05s → ENV should win.
    assert "ENV" in out


def test_training_end_closes_csv(tmp_path, monkeypatch):
    cb = make_callback(tmp_path)
    clock = [0.0]
    _run_one_rollout(cb, clock, monkeypatch)
    cb._on_training_end()
    assert cb._csv_file is None or cb._csv_file.closed


# --- EvalTimerCallback shim ---------------------------------------------

class _FakeEvalCallback:
    """Mimics SB3 EvalCallback's on_step() / n_calls / eval_freq surface."""

    def __init__(self, eval_freq=3, eval_dt=0.5):
        self.eval_freq = eval_freq
        self.eval_dt = eval_dt
        self.n_calls = 0
        self._steps = 0
        self._clock_ref = None

    def attach_clock(self, clock_ref):
        self._clock_ref = clock_ref

    def init_callback(self, model):
        pass

    def on_step(self):
        self.n_calls += 1
        self._steps += 1
        # Simulate eval taking time on eval-freq boundaries.
        if self._clock_ref is not None and self.n_calls % self.eval_freq == 0:
            self._clock_ref[0] += self.eval_dt
        return True

    def _on_training_end(self):
        pass


def test_eval_timer_accumulates_only_on_eval_steps(tmp_path, monkeypatch):
    from code.callbacks.profiling_callback import EvalTimerCallback
    profiler = make_callback(tmp_path)
    inner = _FakeEvalCallback(eval_freq=2, eval_dt=0.5)
    shim = EvalTimerCallback(inner_cb=inner, profiler=profiler)
    shim.model = profiler.model

    clock = [0.0]
    inner.attach_clock(clock)
    monkeypatch.setattr(
        "code.callbacks.profiling_callback.time.perf_counter",
        lambda: clock[0],
    )

    shim._init_callback()
    # 4 calls: eval fires on 2 and 4 → 1.0s total eval.
    shim._on_step()  # n_calls=1, no eval
    shim._on_step()  # n_calls=2, eval (+0.5s)
    shim._on_step()  # n_calls=3, no eval
    shim._on_step()  # n_calls=4, eval (+0.5s)

    assert profiler._eval_s_carry == pytest.approx(1.0, abs=1e-6)


def test_eval_timer_passes_through_inner_result(tmp_path):
    from code.callbacks.profiling_callback import EvalTimerCallback

    class _Stop(_FakeEvalCallback):
        def on_step(self):
            super().on_step()
            return False

    profiler = make_callback(tmp_path)
    shim = EvalTimerCallback(inner_cb=_Stop(), profiler=profiler)
    shim.model = profiler.model
    shim._init_callback()
    assert shim._on_step() is False


# --- EvalTimerCallback: production wrapper case -------------------------

class _FakeEvalWrapper:
    """Mimics EvalPreviewCallback: no eval_freq of its own; exposes inner via .eval_cb."""

    def __init__(self, inner):
        self.eval_cb = inner

    def init_callback(self, model):
        pass

    def on_step(self):
        return self.eval_cb.on_step()

    def _on_training_end(self):
        pass


def test_eval_timer_unwraps_wrapper_for_eval_freq(tmp_path, monkeypatch):
    """EvalTimerCallback must find eval_freq on inner_cb.eval_cb when inner_cb is a wrapper."""
    from code.callbacks.profiling_callback import EvalTimerCallback
    profiler = make_callback(tmp_path)
    inner = _FakeEvalCallback(eval_freq=2, eval_dt=0.5)
    wrapper = _FakeEvalWrapper(inner)
    shim = EvalTimerCallback(inner_cb=wrapper, profiler=profiler)
    shim.model = profiler.model

    clock = [0.0]
    inner.attach_clock(clock)
    monkeypatch.setattr(
        "code.callbacks.profiling_callback.time.perf_counter",
        lambda: clock[0],
    )

    shim._init_callback()
    shim._on_step()  # n_calls=1, no eval
    shim._on_step()  # n_calls=2, eval (+0.5s)
    shim._on_step()  # n_calls=3, no eval
    shim._on_step()  # n_calls=4, eval (+0.5s)

    assert profiler._eval_s_carry == pytest.approx(1.0, abs=1e-6)


# --- CSV write resilience -----------------------------------------------

def test_csv_write_failure_does_not_block_training(tmp_path, monkeypatch, capsys):
    """If CSV write raises, _on_rollout_end must not propagate the exception."""
    cb = make_callback(tmp_path, n_envs=2, n_steps=4)
    clock = [0.0]

    # Replace the writer with one that raises.
    class _BoomWriter:
        def writerow(self, row):
            raise OSError("disk full")
    cb._csv_writer = _BoomWriter()

    monkeypatch.setattr(
        "code.callbacks.profiling_callback.time.perf_counter",
        lambda: clock[0],
    )
    cb._on_rollout_start()
    for _ in range(4):
        clock[0] += 0.1
        cb._on_step()
    clock[0] += 0.05
    cb._on_rollout_end()  # MUST NOT raise

    out = capsys.readouterr().out
    assert "CSV write failed" in out
    assert "disk full" in out
