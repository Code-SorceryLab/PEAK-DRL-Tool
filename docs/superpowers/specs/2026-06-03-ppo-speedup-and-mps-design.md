# PPO Speedup Infrastructure + Mac M5 (MPS) Support

**Date:** 2026-06-03
**Branch:** `vLLm`
**Author:** Al (kevin.chua.says.hi@gmail.com)
**Status:** Approved design — ready for implementation planning

---

## 1. Problem statement

PPO training in PEAK-DRL-Tool feels slow, and there is no diagnostic data to say *where* the time is going. The repo currently supports only `device: cpu` and `device: cuda` (with cuda silently falling back to cpu); Apple Silicon — including the M5 — runs entirely on CPU even when an MPS-capable backend is available. Two related needs:

1. **Speed up PPO training**, but informed by measurement rather than guesswork.
2. **Add Mac silicon (MPS) support** so M5 (and earlier Apple Silicon) hardware can run training on the GPU.

The original framing ("add vLLM") is set aside: vLLM is an LLM-serving engine and cannot serve the small CNN policies (`lightmobile`, `spatialattention`, `channelattention`, `deepchannelattention`) this project uses. The branch name `vLLm` is retained for continuity, but the actual deliverables are profiling infrastructure and MPS support.

A subsequent phase ("training issues") is out of scope for this spec and will be addressed in its own design once profiling data is in hand.

## 2. Goals & non-goals

### Goals

- Permanent, opt-in profiling that reports the per-rollout breakdown of env-step time, policy forward time, gradient-update time, and eval time.
- A device resolver supporting `auto | cpu | cuda | mps`, defaulting to `auto`.
- Graceful, documented MPS fallback (`PYTORCH_ENABLE_MPS_FALLBACK=1`) so ops without MPS kernels do not crash training.
- A one-shot benchmark script that runs the same short training job on each available device and prints a comparison table.
- Backward-compatible defaults: existing CUDA users see no behavioral change; explicit `device=cpu` still works.

### Non-goals

- No introduction of `torch.compile`, bf16/mixed-precision, channels-last, EnvPool, Numba, Cython, or any other speedup technique in this phase. Those are *candidate fixes* selected after profiling data is available.
- No vLLM dependency, no LLM-as-policy work, no inference-server work.
- No replacement of Stable-Baselines3 with CleanRL / Sample Factory / RL Games.
- No changes to game cores, reward personas, level loading, or the dashboard.
- No new model artifacts; existing trained models remain valid.

## 3. Architecture

Two pieces of permanent infrastructure plus a one-time benchmark script:

```
code/
├── callbacks/
│   └── profiling_callback.py          ← NEW: ProfilingCallback (SB3 BaseCallback)
├── scripts/
│   ├── train.py                       ← MODIFIED: _resolve_device(), wire ProfilingCallback
│   └── benchmark_device.py            ← NEW: one-shot device comparison
├── conf/
│   └── grid.yaml                      ← MODIFIED: device=auto default, profile=false flag
└── tests/
    └── test_resolve_device.py         ← NEW: unit tests for the device resolver
```

Boundary rationale:

- `profiling_callback.py` is separate from the existing `logging_callback.py` because `CsvLoggerCallback` is already large and per-step; mixing rollout-level profiling into it would couple unrelated concerns. The two callbacks compose independently in train.py's CallbackList.
- `_resolve_device()` lives inline in train.py as a small helper (~20 lines, one caller). No standalone module — no premature abstraction.
- `benchmark_device.py` is a separate entry point because it is a diagnostic, not a training entry point. Run once per device, decide next steps, do not run on a schedule.

## 4. Component: `ProfilingCallback`

### Class shape

```python
# code/callbacks/profiling_callback.py
class ProfilingCallback(BaseCallback):
    def __init__(
        self,
        log_dir: str,
        device: str,
        sync_device: Optional[bool] = None,   # None → auto: True if device in {cuda, mps}, else False
        verbose: int = 0,
    ): ...
```

When `sync_device` is left at the default (`None`), the constructor picks `True` for `cuda`/`mps` and `False` for `cpu`. The override exists for advanced users who want to measure dispatch-time only.

### Metrics measured, per rollout cycle

| Metric            | Definition                                                                                  |
| ----------------- | ------------------------------------------------------------------------------------------- |
| `rollout_wall_s`  | wall time for one PPO rollout (n_steps × n_envs collection)                                 |
| `env_step_s`      | sum of env step durations, accumulated in `_on_step` via `time.perf_counter()` deltas       |
| `policy_forward_s`| `rollout_wall_s − env_step_s` (residual = forward + bookkeeping)                            |
| `train_s`         | gradient-update time: duration from `_on_rollout_end` to next `_on_rollout_start`           |
| `eval_s`          | eval pause time, accumulated via an `EvalTimerCallback` shim that wraps the user's EvalCallback (same composition pattern as the existing `VecnormBestCallback` in train.py:36) |
| `sps`             | steps/second for the rollout (mirrors SB3's `fps`, included for one consolidated CSV)       |
| `device`, `n_envs`| logged once for context                                                                     |

### Device-sync correctness

For CUDA/MPS, `time.perf_counter()` measures kernel *dispatch* time, not kernel *execution* time. When `sync_device=True`, the callback calls `torch.cuda.synchronize()` / `torch.mps.synchronize()` immediately before each timing boundary so the measurement reflects actual GPU work. The sync has cost (forces serialization), which is why profiling is opt-in via `profile: true`.

### Outputs

- TensorBoard scalars under `profile/*`, picked up by the existing TB pipeline rooted at `mylogs/`.
- CSV at `mylogs/<run>/profiling_log.csv`, one row per rollout. No per-step rows — that would balloon the file at PPO's typical 2048-step rollouts.
- End-of-training stdout summary: `[Profile] env=72% fwd=18% train=8% eval=2% → bottleneck: ENV`.

### Bounded scope

- Does not time individual gradient steps, only the train block as a whole. If profiling indicates train is the bottleneck, deeper instrumentation gets added in the follow-up phase.
- Does not call `torch.mps.empty_cache()` or otherwise manage memory. Added only if profiling reveals memory pressure.

## 5. Component: `_resolve_device()`

### Behavior

```
spec="auto"  → "cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built()
              else "cpu"
spec="cuda"  → "cuda" if available, else WARN + "cpu"   (preserves existing behavior at train.py:862)
spec="mps"   → "mps"  if MPS available+built, else WARN + "cpu"   (new)
spec="cpu"   → "cpu"   (preserves existing behavior)
anything else → WARN + "cpu"
```

### Side-effect on MPS resolution

When the resolved device is `mps`, set `os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"` so unsupported ops fall back to CPU instead of crashing training. This is documented behavior, not a bug.

### Logging

Prints exactly one resolution line, grep-friendly:

```
[INFO] Device requested: 'auto' → resolved: 'mps' (Apple Silicon, MPS fallback enabled)
[INFO] Device requested: 'cuda' → resolved: 'cpu' (CUDA not available)
[INFO] Device requested: 'cpu' → resolved: 'cpu'
```

### Replacement site

Replaces the 4-line block at `train.py:861–865`:

```python
# before
device = cfg.get("device", "cpu")
if device == "cuda" and not torch.cuda.is_available():
    print("[WARNING] CUDA not available, falling back to CPU.")
    device = "cpu"
print(f"[INFO] Training device: {device}")

# after
device = _resolve_device(cfg.get("device", "auto"))
```

## 6. Config surface

### `code/conf/grid.yaml` diff

```yaml
# before
device: cpu      # cuda or cpu
n_envs: 2 # 1 for cpu or 16 gpu , or whatever you want

# after
device: auto     # auto | cpu | cuda | mps  (auto picks best available)
n_envs: 2        # 1 for cpu, 16 for gpu, etc.
profile: false   # true → enable ProfilingCallback (TB scalars + profiling_log.csv)
```

### Default flip rationale

Switching the default from `cpu` to `auto`:

- CUDA users on existing boxes get cuda automatically (improvement, no manual config).
- Mac users get mps automatically (new capability).
- CPU is never chosen silently — the resolution line in stdout always says what was picked.
- Explicit `device=cpu` still works exactly as before for users who want it.

### CLI surface (Hydra existing pattern)

```bash
python -m code.scripts.train                                # auto
python -m code.scripts.train device=mps                     # force mps
python -m code.scripts.train device=cpu profile=true        # enable profiling
python -m code.scripts.benchmark_device --steps 10000       # compare devices
```

## 7. Component: `benchmark_device.py`

Standalone entry point. For each available device (cpu always; cuda if available; mps if available), runs a fixed-config PPO training for `--steps` (default 10 000) with `profile: true` and captures `sps` plus the env/fwd/train/eval percentage breakdown. Prints a comparison table:

```
device    sps     env%   fwd%   train%   eval%
cpu       820     78     16     5        1
mps       1140    62     22     12       4
```

No model artifacts are kept; output is purely diagnostic. Uses the same `_resolve_device()` helper so the comparison is honest about what each spec actually resolved to.

## 8. Data flow

```
grid.yaml (device, profile)
    │
    ▼
train.py: cfg = hydra.load()
    │
    ▼
_resolve_device(cfg.device) ──► sets PYTORCH_ENABLE_MPS_FALLBACK if mps
    │
    ▼
device str ──► passed to PPO(..., device=device)
              passed to ProfilingCallback(..., device=device)
    │
    ▼
CallbackList = [
    CsvLoggerCallback,           # existing per-step game logging
    ProfilingCallback,           # NEW — only added if cfg.profile is true
    EvalCallback (+ EvalTimerCallback shim if profile),
    CheckpointCallback,
    VecnormBestCallback,
    LiveVisualizationCallback?,  # existing, conditional on viz_enabled
    AnnealCallback,              # existing
]
    │
    ▼
model.learn() runs; ProfilingCallback writes rows to profiling_log.csv per rollout,
TB scalars per rollout, and prints end-of-training summary.
```

## 9. Error handling

| Failure                                        | Behavior                                                                                       |
| ---------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `device=mps` on non-Apple-Silicon machine      | `_resolve_device` warns, returns `cpu`. Training proceeds.                                     |
| MPS op not implemented mid-training            | `PYTORCH_ENABLE_MPS_FALLBACK=1` already set; op falls back to CPU silently (PyTorch behavior). |
| `profile=true` but writing CSV fails           | Print error, continue training. Profiling is diagnostic; never blocks training.                |
| `torch.mps.synchronize()` unavailable in build | Caught at callback init; `sync_device` forced to `False`, warning printed.                     |
| Hydra missing `profile` key (old configs)      | `cfg.get("profile", False)` — safe default.                                                    |

## 10. Verification plan

| Check                            | Command                                                                  | Pass criteria                                                                                                |
| -------------------------------- | ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------ |
| Device resolver — all combos     | `pytest code/tests/test_resolve_device.py`                               | All 4 specs × 4 availability combinations resolve correctly; warnings printed where expected; env var set on mps |
| MPS smoke test (no profile)      | `python -m code.scripts.train device=mps` for 5 000 steps                | No crash; model saves; loadable by `watch_agent.py`                                                          |
| Profiling CSV exists             | `python -m code.scripts.train device=cpu profile=true` for 10 000 steps  | `mylogs/<run>/profiling_log.csv` has ≥2 rows (PPO default `n_steps=2048` × `n_envs=2` = ~4 096 steps per rollout); TB shows `profile/*` scalars; end-of-training summary printed |
| CUDA path preserved              | `python -m code.scripts.train device=cuda` on existing CUDA box          | Runs identically to before; no regression in training metrics                                                |
| Auto-detect picks correctly      | `python -m code.scripts.train` (no device arg) on each box               | CUDA box → "cuda"; M5 → "mps"; CPU-only box → "cpu"                                                          |
| Benchmark script                 | `python -m code.scripts.benchmark_device --steps 10000` on M5            | Outputs comparison table; identifies bottleneck category in stdout                                           |

## 11. How the data feeds the next phase ("training issues")

Profiling outputs select the next move; the decisions are data-driven, not guesses:

| Profile observation | Bottleneck category | Candidate fixes for the next spec                                              |
| ------------------- | ------------------- | ------------------------------------------------------------------------------ |
| `env% > 60%`        | Game core           | More `n_envs` workers; Numba/Cython on `platformer_core.py` hot loop; EnvPool-style C++ port |
| `fwd% > 40%`        | Policy forward      | `torch.compile`; bf16; smaller architecture variant                            |
| `train% > 30%`      | Gradient updates    | Larger batch sizes; `torch.compile`; GPU upgrade                               |
| `eval% > 15%`       | Eval overhead       | Bump `eval_freq`; shorten eval episodes                                        |

This table is intentionally part of the spec so the rationale survives into the implementation phase.

## 12. Open questions

None at design time. All resolved during brainstorming:

- vLLM is not the right tool — confirmed and set aside.
- Mac scope is "MPS + auto-detect" — confirmed, no bf16 or torch.compile in this phase.
- Profiling is opt-in, not always-on — confirmed (sync cost).
- Default `device: auto` — confirmed backward-compatible.
