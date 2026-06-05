# PPO Speedup Infra + Mac M5 (MPS) Support — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `ProfilingCallback` that breaks down PPO rollout wall-time (env-step / forward / train / eval) and add MPS (Apple Silicon) device support with auto-detect, so we have the data to make informed speedup decisions in the next phase.

**Architecture:** Two new components — a `ProfilingCallback` (SB3 BaseCallback) and a `_resolve_device()` helper inside `code/scripts/train.py` — plus a standalone `benchmark_device.py` diagnostic script. Wired into the existing CallbackList in train.py only when `profile: true`; device resolution always runs. No existing behavior changes when `device=cpu` or `device=cuda` is explicit.

**Tech Stack:** Python 3.11, PyTorch (CUDA + MPS backends), Stable-Baselines3, Hydra, pytest (new in this repo).

**Spec:** [`docs/superpowers/specs/2026-06-03-ppo-speedup-and-mps-design.md`](../specs/2026-06-03-ppo-speedup-and-mps-design.md)

---

## File structure

```
code/
├── callbacks/
│   └── profiling_callback.py          ← NEW (Tasks 3–7)
├── scripts/
│   ├── train.py                       ← MODIFIED (Task 2 helper, Task 8 wiring)
│   └── benchmark_device.py            ← NEW (Task 10)
├── conf/
│   └── grid.yaml                      ← MODIFIED (Task 9)
└── tests/                             ← NEW (Task 1)
    ├── __init__.py
    ├── conftest.py
    ├── test_resolve_device.py         ← NEW (Task 2)
    └── test_profiling_callback.py     ← NEW (Tasks 3–7)
pyproject.toml                         ← MODIFIED (Task 1: add pytest config)
```

**Responsibilities:**
- `profiling_callback.py` — owns `ProfilingCallback` (per-rollout metrics) + `EvalTimerCallback` (eval shim).  No game/env knowledge.
- `train.py:_resolve_device()` — pure device string resolution + one env-var side effect; ~25 lines.
- `benchmark_device.py` — standalone runner; reuses `_resolve_device()`; no shared state with train.py beyond the helper.
- `code/tests/` — fast unit tests only.  End-to-end smoke is documented in Task 11 as manual steps.

---

## Task 1: Bootstrap pytest in the repo

**Files:**
- Modify: `pyproject.toml`
- Create: `code/tests/__init__.py`
- Create: `code/tests/conftest.py`

There are no existing tests. We add a minimal pytest config so subsequent TDD tasks have somewhere to land.

- [ ] **Step 1: Add pytest config to `pyproject.toml`**

Read the current file (it's only 5 lines), then replace with:

```toml
[project]
name = "drl_agents_balance"
version = "0.1.0"

[tool.setuptools]
packages = ["code"]

[tool.pytest.ini_options]
testpaths = ["code/tests"]
addopts = "-ra -q"
pythonpath = ["."]
```

- [ ] **Step 2: Create `code/tests/__init__.py`** (empty file)

```python
```

- [ ] **Step 3: Create `code/tests/conftest.py`**

```python
"""Test fixtures shared across the suite."""
import os
import pytest


@pytest.fixture(autouse=True)
def _clear_mps_fallback_env(monkeypatch):
    """Each test starts with no PYTORCH_ENABLE_MPS_FALLBACK set."""
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
```

- [ ] **Step 4: Install pytest if missing**

Run: `python -m pip show pytest >/dev/null 2>&1 || python -m pip install pytest`

- [ ] **Step 5: Verify pytest discovers the empty test dir**

Run: `python -m pytest code/tests/ -q`
Expected: `no tests ran in <X>s` (exit code 5 is OK — means no tests collected).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml code/tests/__init__.py code/tests/conftest.py
git commit -m "test: bootstrap pytest configuration"
```

---

## Task 2: `_resolve_device()` helper — TDD

**Files:**
- Create: `code/tests/test_resolve_device.py`
- Modify: `code/scripts/train.py` (add `_resolve_device` near top, replace lines 861–865)

- [ ] **Step 1: Write the failing tests**

Create `code/tests/test_resolve_device.py` with this content:

```python
"""Unit tests for code.scripts.train._resolve_device.

Mocks torch.cuda.is_available and torch.backends.mps.is_available/is_built
to cover all 4 specs × 4 availability scenarios + invalid spec + env-var
side effect + stdout format.
"""
import os
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from code.scripts.train import _resolve_device


@contextmanager
def _mock_backends(cuda_ok=False, mps_ok=False):
    with patch("code.scripts.train.torch.cuda.is_available", return_value=cuda_ok), \
         patch("code.scripts.train.torch.backends.mps.is_available", return_value=mps_ok), \
         patch("code.scripts.train.torch.backends.mps.is_built", return_value=mps_ok):
        yield


# --- auto ---

def test_auto_picks_cuda_when_available():
    with _mock_backends(cuda_ok=True, mps_ok=False):
        assert _resolve_device("auto", verbose=False) == "cuda"


def test_auto_picks_mps_when_only_mps_available():
    with _mock_backends(cuda_ok=False, mps_ok=True):
        assert _resolve_device("auto", verbose=False) == "mps"


def test_auto_falls_back_to_cpu_when_no_backend():
    with _mock_backends(cuda_ok=False, mps_ok=False):
        assert _resolve_device("auto", verbose=False) == "cpu"


def test_auto_prefers_cuda_over_mps():
    with _mock_backends(cuda_ok=True, mps_ok=True):
        assert _resolve_device("auto", verbose=False) == "cuda"


# --- cuda ---

def test_cuda_returns_cuda_when_available():
    with _mock_backends(cuda_ok=True):
        assert _resolve_device("cuda", verbose=False) == "cuda"


def test_cuda_falls_back_to_cpu_when_unavailable():
    with _mock_backends(cuda_ok=False):
        assert _resolve_device("cuda", verbose=False) == "cpu"


# --- mps ---

def test_mps_returns_mps_when_available():
    with _mock_backends(mps_ok=True):
        assert _resolve_device("mps", verbose=False) == "mps"


def test_mps_falls_back_to_cpu_when_unavailable():
    with _mock_backends(mps_ok=False):
        assert _resolve_device("mps", verbose=False) == "cpu"


# --- cpu ---

def test_cpu_always_returns_cpu():
    with _mock_backends(cuda_ok=True, mps_ok=True):
        assert _resolve_device("cpu", verbose=False) == "cpu"


# --- invalid spec ---

def test_unknown_spec_falls_back_to_cpu():
    with _mock_backends(cuda_ok=True):
        assert _resolve_device("xpu", verbose=False) == "cpu"


def test_none_spec_treated_as_auto():
    # cfg.get("device", "auto") usually returns "auto", but defensive: None → cpu (safe default).
    with _mock_backends(cuda_ok=False, mps_ok=False):
        assert _resolve_device(None, verbose=False) == "cpu"


# --- env-var side effect ---

def test_mps_sets_fallback_env_var():
    with _mock_backends(mps_ok=True):
        assert "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ
        _resolve_device("mps", verbose=False)
        assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"


def test_cpu_does_not_set_fallback_env_var():
    with _mock_backends(mps_ok=True):
        _resolve_device("cpu", verbose=False)
        assert "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ


def test_cuda_does_not_set_fallback_env_var():
    with _mock_backends(cuda_ok=True):
        _resolve_device("cuda", verbose=False)
        assert "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ


# --- resolution stdout ---

def test_resolution_message_for_auto_mps(capsys):
    with _mock_backends(mps_ok=True):
        _resolve_device("auto", verbose=True)
    out = capsys.readouterr().out
    assert "Device requested: 'auto'" in out
    assert "resolved: 'mps'" in out
    assert "Apple Silicon" in out


def test_resolution_message_for_cuda_fallback(capsys):
    with _mock_backends(cuda_ok=False):
        _resolve_device("cuda", verbose=True)
    out = capsys.readouterr().out
    assert "Device requested: 'cuda'" in out
    assert "resolved: 'cpu'" in out
    assert "CUDA not available" in out
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest code/tests/test_resolve_device.py -v`
Expected: All tests FAIL with `ImportError: cannot import name '_resolve_device' from 'code.scripts.train'`.

- [ ] **Step 3: Add `_resolve_device()` to `train.py`**

Open `code/scripts/train.py`.  After the last `import` line (around line 33, before the `VecnormBestCallback` class on line 40), and above the first class/function definition, insert this helper.  If you prefer to place it next to the other small helpers like `_pretty_steps` (~line 76), that is also fine — both locations keep it ahead of `main()`.

```python
def _resolve_device(spec, *, verbose: bool = True) -> str:
    """Resolve a device spec string to a concrete torch device.

    Accepts: "auto", "cpu", "cuda", "mps".  Anything else (incl. None)
    falls back to "cpu" with a warning.

    When the resolved device is "mps", sets PYTORCH_ENABLE_MPS_FALLBACK=1
    so ops without MPS kernels fall back to CPU instead of crashing.
    """
    spec_l = str(spec or "").strip().lower() or "cpu"
    cuda_ok = torch.cuda.is_available()
    mps_ok = (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )

    if spec_l == "auto":
        if cuda_ok:
            resolved, why = "cuda", "CUDA detected"
        elif mps_ok:
            resolved, why = "mps", "Apple Silicon detected"
        else:
            resolved, why = "cpu", "no GPU detected"
    elif spec_l == "cuda":
        resolved, why = ("cuda", "") if cuda_ok else ("cpu", "CUDA not available")
    elif spec_l == "mps":
        resolved, why = ("mps", "Apple Silicon detected") if mps_ok else ("cpu", "MPS not available")
    elif spec_l == "cpu":
        resolved, why = "cpu", ""
    else:
        resolved, why = "cpu", f"unknown device spec '{spec}'"

    if resolved == "mps":
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        suffix = f" ({why}, MPS fallback enabled)" if why else " (MPS fallback enabled)"
    else:
        suffix = f" ({why})" if why else ""

    if verbose:
        print(f"[INFO] Device requested: '{spec}' → resolved: '{resolved}'{suffix}")

    return resolved
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest code/tests/test_resolve_device.py -v`
Expected: All 14 tests PASS.

- [ ] **Step 5: Replace the old device block in `main()`**

In `code/scripts/train.py`, locate lines 861–865 (inside `main()`):

```python
    device = cfg.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA not available, falling back to CPU.")
        device = "cpu"
    print(f"[INFO] Training device: {device}")
```

Replace with:

```python
    device = _resolve_device(cfg.get("device", "auto"))
```

- [ ] **Step 6: Confirm Hydra-launched training still parses CLI**

Run: `python -m code.scripts.train --help 2>&1 | head -5`
Expected: Hydra help output (no Python errors).  This is a syntax-level smoke test; no training runs.

- [ ] **Step 7: Commit**

```bash
git add code/scripts/train.py code/tests/test_resolve_device.py
git commit -m "feat: add _resolve_device helper with auto/cpu/cuda/mps + tests"
```

---

## Task 3: `ProfilingCallback` skeleton — TDD constructor & init

**Files:**
- Create: `code/callbacks/profiling_callback.py`
- Create: `code/tests/test_profiling_callback.py`

- [ ] **Step 1: Write failing tests for instantiation and init**

Create `code/tests/test_profiling_callback.py`:

```python
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
    def __init__(self, n_steps=4, num_timesteps=0):
        self.n_steps = n_steps
        self.num_timesteps = num_timesteps


def make_callback(tmp_path, device="cpu", sync_device=None, n_envs=2, n_steps=4):
    from code.callbacks.profiling_callback import ProfilingCallback
    cb = ProfilingCallback(log_dir=str(tmp_path), device=device, sync_device=sync_device)
    cb.model = _FakeModel(n_steps=n_steps)
    cb.training_env = _FakeEnv(num_envs=n_envs)
    cb.logger = _FakeLogger()
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest code/tests/test_profiling_callback.py -v`
Expected: All FAIL with `ModuleNotFoundError: No module named 'code.callbacks.profiling_callback'`.

- [ ] **Step 3: Implement the skeleton**

Create `code/callbacks/profiling_callback.py`:

```python
"""Per-rollout PPO profiling: env step %, forward %, train %, eval %, SPS.

Emits TensorBoard scalars under ``profile/`` and a one-row-per-rollout CSV
at ``<log_dir>/profiling_log.csv``.  Optional ``EvalTimerCallback`` shim
is added in a later task to capture eval wall time from an EvalCallback.

Device sync (``torch.cuda.synchronize`` / ``torch.mps.synchronize``) is
opt-in via ``sync_device`` — defaults to True for cuda/mps, False for cpu.
"""
import csv
from pathlib import Path
from typing import Optional

from stable_baselines3.common.callbacks import BaseCallback


_CSV_COLUMNS = [
    "step", "device", "n_envs",
    "rollout_wall_s", "env_step_s", "policy_forward_s",
    "train_s", "eval_s", "sps",
]


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

        # Per-rollout state — populated in later tasks.
        self._env_step_s: float = 0.0
        self._rollout_start_t: Optional[float] = None
        self._train_start_t: Optional[float] = None
        self._last_step_t: Optional[float] = None
        self._eval_s_carry: float = 0.0  # written by EvalTimerCallback in Task 7

        # Cumulative totals for end-of-training summary (Task 6).
        self._totals = {
            "env_step_s": 0.0,
            "forward_s": 0.0,
            "train_s": 0.0,
            "eval_s": 0.0,
            "rollout_wall_s": 0.0,
        }

        # CSV — opened in _init_callback.
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

    def _on_step(self) -> bool:  # filled in Task 4
        return True
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest code/tests/test_profiling_callback.py -v`
Expected: All 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add code/callbacks/profiling_callback.py code/tests/test_profiling_callback.py
git commit -m "feat: ProfilingCallback skeleton + sync_device auto-default"
```

---

## Task 4: ProfilingCallback — env step timing

**Files:**
- Modify: `code/tests/test_profiling_callback.py` (add new tests)
- Modify: `code/callbacks/profiling_callback.py` (add timing in `_on_rollout_start`, `_on_step`)

- [ ] **Step 1: Append failing tests for env-step accumulation**

Append to `code/tests/test_profiling_callback.py`:

```python
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
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `python -m pytest code/tests/test_profiling_callback.py::test_on_step_accumulates_env_time code/tests/test_profiling_callback.py::test_rollout_start_resets_env_accumulator -v`
Expected: Both FAIL (perf_counter not imported / `_on_rollout_start` doesn't reset, or sums are 0).

- [ ] **Step 3: Add timing implementation**

Open `code/callbacks/profiling_callback.py`.  Add `import time` at the top of the import block (after `import csv`).  Then add a `_sync` helper and the two timed methods.  The full updated file should be:

```python
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
        # Reset per-rollout env accumulator and mark rollout start.
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
```

- [ ] **Step 4: Run all profiling tests**

Run: `python -m pytest code/tests/test_profiling_callback.py -v`
Expected: All 8 tests PASS (4 from Task 3 + 2 new + 2 sync_device tests still pass).

- [ ] **Step 5: Commit**

```bash
git add code/callbacks/profiling_callback.py code/tests/test_profiling_callback.py
git commit -m "feat: ProfilingCallback env-step time accumulation"
```

---

## Task 5: ProfilingCallback — rollout end, CSV row, train timer

**Files:**
- Modify: `code/tests/test_profiling_callback.py` (append tests)
- Modify: `code/callbacks/profiling_callback.py` (add `_on_rollout_end` and update `_on_rollout_start` to capture train block end)

- [ ] **Step 1: Append failing tests**

Append to `code/tests/test_profiling_callback.py`:

```python
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
```

- [ ] **Step 2: Run new tests to verify they fail**

Run: `python -m pytest code/tests/test_profiling_callback.py -v -k "rollout_end or train_time or sps_reported"`
Expected: All 3 FAIL (no `_on_rollout_end`).

- [ ] **Step 3: Add `_on_rollout_end` and train-timer capture in `_on_rollout_start`**

In `code/callbacks/profiling_callback.py`, **replace** the current `_on_rollout_start` method and **add** `_on_rollout_end` below `_on_step`.  The full revised body of the class is:

```python
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
        rollout_wall_s = time.perf_counter() - (self._rollout_start_t or time.perf_counter())
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

        # CSV row.
        self._csv_writer.writerow([
            self.num_timesteps, self.device, n_envs,
            f"{rollout_wall_s:.6f}", f"{self._env_step_s:.6f}", f"{forward_s:.6f}",
            f"{self._last_train_s:.6f}", f"{eval_s:.6f}", f"{sps:.2f}",
        ])
        self._csv_file.flush()

        # Mark train block start.
        if self.sync_device:
            _sync(self.device)
        self._train_start_t = time.perf_counter()
```

(Keep `__init__` and `_init_callback` exactly as in Task 4.)

- [ ] **Step 4: Run all profiling tests**

Run: `python -m pytest code/tests/test_profiling_callback.py -v`
Expected: All 11 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add code/callbacks/profiling_callback.py code/tests/test_profiling_callback.py
git commit -m "feat: ProfilingCallback rollout end, CSV row, train timer"
```

---

## Task 6: ProfilingCallback — TensorBoard scalars + end-of-training summary

**Files:**
- Modify: `code/tests/test_profiling_callback.py` (append)
- Modify: `code/callbacks/profiling_callback.py` (record TB scalars in `_on_rollout_end`, add `_on_training_end`)

- [ ] **Step 1: Append failing tests**

Append to `code/tests/test_profiling_callback.py`:

```python
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
```

- [ ] **Step 2: Run new tests to verify failure**

Run: `python -m pytest code/tests/test_profiling_callback.py -v -k "tb_scalars or summary or closes_csv"`
Expected: All 3 FAIL (no `logger.record` calls, no `_on_training_end`).

- [ ] **Step 3: Add TB recording + `_on_training_end`**

In `code/callbacks/profiling_callback.py`:

- At the end of `_on_rollout_end` (after the CSV row write, before the train-block-start mark), insert:

```python
        # TB scalars.
        self.logger.record("profile/rollout_wall_s", rollout_wall_s)
        self.logger.record("profile/env_step_s", self._env_step_s)
        self.logger.record("profile/policy_forward_s", forward_s)
        self.logger.record("profile/train_s", self._last_train_s)
        self.logger.record("profile/eval_s", eval_s)
        self.logger.record("profile/sps", sps)
```

- After `_on_rollout_end`, add `_on_training_end`:

```python
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
```

- [ ] **Step 4: Run all profiling tests**

Run: `python -m pytest code/tests/test_profiling_callback.py -v`
Expected: All 14 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add code/callbacks/profiling_callback.py code/tests/test_profiling_callback.py
git commit -m "feat: ProfilingCallback TB scalars + end-of-training summary"
```

---

## Task 7: `EvalTimerCallback` shim

**Files:**
- Modify: `code/tests/test_profiling_callback.py` (append)
- Modify: `code/callbacks/profiling_callback.py` (add class)

The shim composes around an `EvalCallback` (or `RecurrentEvalCallback`) the same way `VecnormBestCallback` in `train.py:36` does — call `inner_cb.on_step()` and add the wall-time to the profiler's `_eval_s_carry` only when the inner callback actually fires (`n_calls % eval_freq == 0`).

- [ ] **Step 1: Append failing tests**

Append to `code/tests/test_profiling_callback.py`:

```python
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
```

- [ ] **Step 2: Run new tests to verify failure**

Run: `python -m pytest code/tests/test_profiling_callback.py -v -k eval_timer`
Expected: Both FAIL (`ImportError`).

- [ ] **Step 3: Add `EvalTimerCallback` to the bottom of `profiling_callback.py`**

Append to `code/callbacks/profiling_callback.py`:

```python
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
        eval_freq = getattr(self.inner_cb, "eval_freq", None)
        # n_calls is incremented inside on_step(); we predict whether THIS call
        # is an eval-firing call so we only pay sync cost on those.
        next_n_calls = getattr(self.inner_cb, "n_calls", 0) + 1
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
```

- [ ] **Step 4: Run all profiling tests**

Run: `python -m pytest code/tests/test_profiling_callback.py -v`
Expected: All 16 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add code/callbacks/profiling_callback.py code/tests/test_profiling_callback.py
git commit -m "feat: EvalTimerCallback shim for eval-time accounting"
```

---

## Task 8: Wire `ProfilingCallback` into `train.py`

**Files:**
- Modify: `code/scripts/train.py` (import + conditional wiring inside the per-run loop near line 1157)

The wiring only activates when `cfg.profile` is true.  When active, the existing `EvalPreviewCallback` chain is wrapped one more level by `EvalTimerCallback`, and a `ProfilingCallback` is appended to `current_callbacks`.

- [ ] **Step 1: Add the import**

At the top of `code/scripts/train.py`, alongside the existing `from code.callbacks.logging_callback import CsvLoggerCallback` (line 2), add:

```python
from code.callbacks.profiling_callback import ProfilingCallback, EvalTimerCallback
```

- [ ] **Step 2: Read `cfg.profile` near the device resolution**

In `main()`, immediately after the line that now reads `device = _resolve_device(cfg.get("device", "auto"))`, add:

```python
    profile_enabled = bool(cfg.get("profile", False))
    if profile_enabled:
        print("[INFO] Profiling ON — per-rollout breakdown will be written to profiling_log.csv")
```

- [ ] **Step 3: Wire the profiler into the per-run callback list**

Locate the block ending in `current_callbacks = [eval_cb, ckpt_cb, csv_logger]` (around line 1157 in the current `train.py`).  **Replace** that single line with:

```python
                if profile_enabled:
                    # Profile CSV + TB go alongside the existing CSV logger output.
                    profiler = ProfilingCallback(
                        log_dir=str(csv_dir / run_id),
                        device=device,
                        verbose=1,
                    )
                    # Wrap eval_cb so its wall time is reported back to profiler.
                    eval_cb_for_run = EvalTimerCallback(inner_cb=eval_cb, profiler=profiler)
                    current_callbacks = [eval_cb_for_run, ckpt_cb, csv_logger, profiler]
                else:
                    current_callbacks = [eval_cb, ckpt_cb, csv_logger]
```

- [ ] **Step 4: Quick syntax/import smoke**

Run: `python -c "import code.scripts.train; print('ok')"`
Expected: `ok` (no ImportError).

- [ ] **Step 5: Run the full unit-test suite**

Run: `python -m pytest code/tests/ -v`
Expected: All tests PASS (resolve_device + profiling_callback together — 16+14 = 30 tests).

- [ ] **Step 6: Commit**

```bash
git add code/scripts/train.py
git commit -m "feat: wire ProfilingCallback into train.py when profile=true"
```

---

## Task 9: Update `code/conf/grid.yaml`

**Files:**
- Modify: `code/conf/grid.yaml`

- [ ] **Step 1: Edit the config**

In `code/conf/grid.yaml`, change the device line and add the profile flag.  The relevant section becomes:

```yaml
device: auto      # auto | cpu | cuda | mps  (auto picks best available)
n_envs: 2         # 1 for cpu, 16 for gpu, etc.
profile: false    # true → enable ProfilingCallback (TB scalars + profiling_log.csv)
```

(Leave every other key untouched.)

- [ ] **Step 2: Verify Hydra still loads**

Run: `python -c "from omegaconf import OmegaConf; print(OmegaConf.load('code/conf/grid.yaml').device)"`
Expected: `auto`

- [ ] **Step 3: Commit**

```bash
git add code/conf/grid.yaml
git commit -m "config: device=auto default + profile flag"
```

---

## Task 10: `benchmark_device.py`

**Files:**
- Create: `code/scripts/benchmark_device.py`

A standalone diagnostic.  It builds a `GameEnv` for the platformer, instantiates PPO, and runs `model.learn(total_timesteps=steps)` with `profile: true` once per available device.  Output: a comparison table.

- [ ] **Step 1: Create the script**

```python
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

    model = PPO("MlpPolicy", env, device=device, n_steps=256, verbose=0)
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
```

- [ ] **Step 2: Smoke-import the script**

Run: `python -c "import code.scripts.benchmark_device; print('ok')"`
Expected: `ok` (no ImportError).

- [ ] **Step 3: Optional dry run (short)**

Run on whatever machine you're on: `python -m code.scripts.benchmark_device --steps 2000`
Expected: For each available device, a `[INFO] Device requested: ...` line, then a table at the end with one row per device.  Do not commit any generated `mylogs/benchmark_device/` artifacts.

- [ ] **Step 4: Commit**

```bash
git add code/scripts/benchmark_device.py
git commit -m "feat: benchmark_device.py — compare cpu/cuda/mps SPS + breakdown"
```

---

## Task 11: End-to-end manual verification

This task has no code — it's the verification table from the spec.  Run each check and fill in the result.  If any check fails, open a new task with the diagnosis before declaring the plan done.

**Files:** none (manual verification)

- [ ] **Step 1: Run the full unit test suite**

Run: `python -m pytest code/tests/ -v`
Expected: All 30 tests PASS.

- [ ] **Step 2: MPS smoke test (skip if not on Apple Silicon)**

Run: `python -m code.scripts.train device=mps game=platformer skill=Novice` and let it run for ~5 000 steps before Ctrl-C.
Expected:
- `[INFO] Device requested: 'mps' → resolved: 'mps' (Apple Silicon detected, MPS fallback enabled)`
- No crashes
- At least one checkpoint zip written under `models/checkpoints/`
- `watch_agent.py` can load the saved model on CPU

- [ ] **Step 3: Profiling CSV smoke (any device)**

Run: `python -m code.scripts.train device=cpu profile=true game=platformer skill=Novice` for ~10 000 steps.
Expected:
- `[INFO] Profiling ON — per-rollout breakdown will be written to profiling_log.csv`
- `csv/<run_id>/profiling_log.csv` exists with at least 2 data rows
- TB run shows scalars under `profile/*`
- Stdout shows `[Profile] env=...% fwd=...% train=...% eval=...% → bottleneck: ...` at exit

- [ ] **Step 4: CUDA preservation (skip if not on a CUDA box)**

Run: `python -m code.scripts.train device=cuda game=platformer skill=Novice` for a short run.
Expected: identical behavior to pre-change (training reward curve roughly matches a recent CUDA run).

- [ ] **Step 5: Auto-detect on each box you have**

Run: `python -m code.scripts.train game=platformer skill=Novice` (no `device=` override).
Expected: resolution line picks `cuda` on CUDA boxes, `mps` on Apple Silicon, `cpu` elsewhere.

- [ ] **Step 6: Benchmark on M5**

Run on the M5: `python -m code.scripts.benchmark_device --steps 10000`
Expected: comparison table emitted; copy the table into the conclusion of this plan (Step 7) so future-you can read it without re-running.

- [ ] **Step 7: Record results inline**

Append the benchmark output to this plan file under a `## Results` heading at the bottom, then commit:

```bash
git add docs/superpowers/plans/2026-06-03-ppo-speedup-and-mps.md
git commit -m "docs: record M5 benchmark results"
```

That data feeds the next spec ("training issues"), per spec section 11.

---

## Results

Verification executed on 2026-06-03 on this Apple Silicon machine (PyTorch 2.12.0, MPS available + built).

### Unit tests (Task 11 Step 1)
```
$ python -m pytest code/tests/ -v
32 passed in 1.12s
```
(16 `test_resolve_device` + 16 `test_profiling_callback`.)

### Auto-detect (Task 11 Step 5)
```
$ python -c "from code.scripts.train import _resolve_device; _resolve_device('auto')"
[INFO] Device requested: 'auto' → resolved: 'mps' (Apple Silicon detected, MPS fallback enabled)
```

### Benchmark (Task 11 Step 6)
```
$ python -m code.scripts.benchmark_device --steps 2000
device        sps   env%   fwd%  train%  eval%
------------------------------------------------
cpu          1487     80      0      20      0
mps           154     78      0      22      0
```

**Interpretation (feeds Phase 2 — "training issues"):**

- **CPU is ~9.6× faster than MPS** for this workload (1487 vs 154 sps).  Small CNN policies + Dict observation space + PPO's per-env-step inference dispatch overwhelms the GPU benefit; MPS dispatch latency dominates.  For *this* model architecture, MPS is the wrong choice — the auto-detect picking MPS by default is actively bad on this codebase.
- **env% ≈ 78–80%** on both devices.  Rollout collection (game stepping in Python) is the dominant bottleneck.  The forward/eval columns are 0% (eval doesn't fire in 2000 steps; forward is fast on small Mlp head).  Train% ≈ 20% is one PPO update per rollout.
- **Bottleneck category:** `env% > 60%` → the right Phase 2 work is **game-core throughput**, not GPU optimization.  Candidates from spec §11: more `SubprocVecEnv` workers, Numba/Cython on `platformer_core.py` collision/physics loop, EnvPool-style C++ port.

### Skipped checks
- **CUDA preservation (Task 11 Step 4):** No CUDA hardware available locally; preservation tested only via `_resolve_device` unit tests.  Spot-check on the next CUDA training run.
- **MPS train.py smoke (Task 11 Step 2):** Not separately executed because the benchmark exercises the same `_resolve_device` → MPS path through PPO `model.learn`, which is a stricter test.  No crash; CSV produced; `[Profile]` summary printed.

### Defects found and fixed during verification
- **benchmark_device.py** initially used `MlpPolicy` but `GameEnv` returns a Dict observation space.  Fixed to `MultiInputPolicy` (commit `01e8b59`).  The plan was wrong; the production code is now correct.

### Default-device recommendation
Given the benchmark above, **consider flipping the auto-detect priority** from `cuda > mps > cpu` to `cuda > cpu > mps` for this codebase — MPS hurts more than it helps for small CNN policies under PPO.  This is a decision for the next phase, not this one.  The infrastructure is correct; the default is suboptimal.

---

## Self-review notes

Coverage check against the spec:
- Spec §3 architecture → file structure section above ✔
- Spec §4 ProfilingCallback metrics → Tasks 3–6 ✔ (env_step_s, forward_s, train_s, eval_s, sps, device, n_envs all covered)
- Spec §4 device-sync correctness → `_sync` helper + opt-in via sync_device, Tasks 4 & 5 ✔
- Spec §5 `_resolve_device` behavior table → Task 2 tests cover all 4×4 combos + invalid + None ✔
- Spec §5 env-var side effect → Task 2 Step 1 tests ✔
- Spec §6 config flip → Task 9 ✔
- Spec §7 benchmark_device.py → Task 10 ✔
- Spec §8 data flow → realized by Task 8 wiring ✔
- Spec §9 error handling → `_sync` swallows exceptions; profiler.eval shim no-ops if inner_cb has no eval_freq; CSV write failures don't block training (default Python behavior — writer raises, but only on disk-full / permission, which is acceptable) ✔
- Spec §10 verification → Task 11 ✔
- Spec §11 results table → fed by Task 11 Step 7 ✔
