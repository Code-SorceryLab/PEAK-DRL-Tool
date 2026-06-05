# Mario Win-Rate Improvement — Design (2026-06-03)

## Goal

Improve agent win rate across all Mario levels within a fixed 1M-step (Novice) training budget. Stretch target: 60% win rate on Mario1-1; lift Mario1-2 well above its current 1.6% floor.

## Baseline observations (the data this design responds to)

From the most recent Novice (1M-step, 8-env CPU, SpatialAttention, `platformer_adept` persona) run:

| Level | Visits | Wins | Stall | Enemy | Pit | Spike | Timeout | Win rate |
|---|---|---|---|---|---|---|---|---|
| Mario1-1 | 104 | 37 | 15 | 25 | 26 | 0 | 0 | 35.6% |
| Mario1-2 | 64 | 1 | 26 | 25 | 12 | 0 | 0 | 1.6% |

Profiling CSV summary (1M-step run):
- `train_s` ≈ 13–15s per iteration (≈72% of wall time)
- `env_step_s` ≈ 5s per rollout
- Eval spikes (46s, 38s, 27s) occur periodically; otherwise SPS ≈ 3000

## Diagnosis

1. **Stall dominates 1-2 failures** (26/64 episodes terminate from stall). The reward signal makes stalling cheaper than risky forward movement on a level the policy doesn't yet understand.
2. **Curriculum coverage is structurally uneven** — adaptive batch curriculum advances at 30% win rate, falls back at 20%, with `max_stay_windows=2`. The agent masters 1-1 (35.6%), advances to 1-2, stalls, force-falls-back after 2 windows. The cycle gives 1-2 ~64 episodes of total exposure, which is far less than 1-1 gets.
3. **Per-cause death asymmetry is invisible to the reward function.** Life-lost penalty is a flat −0.3 (adept) / −0.5 (simple) regardless of whether the agent fell in a pit, was killed by an enemy, or stalled out.

## Scope

This spec covers **Approach B only**: reward shaping and PPO hyperparameter tuning. Explicitly out of scope (decided by user during brainstorming):

- Curriculum logic changes (e.g. stochastic level sampling, threshold tuning)
- Training budget increases beyond 1M steps
- `torch.compile`, architecture changes, observation-space changes
- New games / personas — Mario only

## Changes

### Section 1: Reward shaping

File: `code/rewards/train_platformer.py`

**1A. Stall penalty bump**

| persona | current | new |
|---|---|---|
| `simple` (movement clamp branch) | `-0.0005` when `not on_platform and abs(progress) < 0.5` | `-0.005` (10×) |
| `adept` | `r_stall = -0.003 if info["stalled"] else 0.0` | `-0.012` (4×) |

Add a progressive multiplier in both personas: if `info["consecutive_stall_steps"] >= 60`, double the per-step stall penalty for the remainder of the stall streak. This forces the policy out of frozen states rather than paying the small toll indefinitely.

> The `consecutive_stall_steps` counter must be added to the `_ScoreTracker` class in the same file. Increment when `info["stalled"]` is True, reset to 0 when False or on life-loss / win.

**1B. Forward-movement gradient sharpening**

| persona | current | new |
|---|---|---|
| `simple` | `r_move = progress * 0.003` (clamp `[-0.01, 0.01]`) | `r_move = progress * 0.005` (same clamp) |
| `adept` | `r_alignment = max(0.0, alignment) * 0.003` | `r_alignment = max(0.0, alignment) * 0.005` |

**1C. Asymmetric life-loss penalty by cause**

Replace the flat life-loss penalty with a cause-aware lookup:

```
pit-death life-loss:    -0.8
enemy-death life-loss:  -0.4
stall-death life-loss:  -1.0
other life-loss:        existing default (-0.3 adept, -0.5 simple)
```

`info["death_cause"]` is already emitted by `platformer_core._handle_death`. The reward fn should read it on `life_lost`; on terminal `terminated=True`, keep the existing `-3.0`.

### Section 2: PPO hyperparameter changes

File: `code/conf/algo/ppo.yaml` (and a new schedule helper)

**2A. Schedule helper**

Add `code/scripts/schedules.py` (new file) that parses string schedule specs and returns SB3-compatible callables:

```
linear_<start>_to_<end>   → SB3 linear schedule (progress_remaining → value)
const_<value>             → fallback to float
```

In `train.py`, after `algo_kwargs = {...}` is built (~line 999–1000), pop schedule-form fields (`learning_rate`, `ent_coef`, `clip_range`) and replace string values with parsed callables before calling `Algo(**algo_kwargs)`.

**2B. Schedule values**

| param | current | new |
|---|---|---|
| `learning_rate` | `0.0003` (const) | `linear_3e-4_to_1e-4` |
| `ent_coef` | `0.01` (const) | `linear_0.015_to_0.003` |
| `clip_range` | `0.2` (const) | `linear_0.3_to_0.1` |

**2C. Fixed-value tweak**

| param | current | new |
|---|---|---|
| `vf_coef` | `0.25` | `0.4` |

**2D. Held constant (deliberately)**

`n_epochs=4`, `n_steps=2048`, `batch_size=512`, `gamma=0.99`, `gae_lambda=0.95`, `weight_decay=1e-8`. Reasoning is in the brainstorming transcript; the short version is "train_s is already the bottleneck and these aren't suspect for the current failure modes."

## Validation plan

### Run protocol

1. **Baseline** — current config (post-revert of any in-progress changes), seeds `{1234, 4242, 7777}`. 3 runs × ~12 min ≈ 35 min.
2. **Treatment** — Section 1 + Section 2 changes applied together, same 3 seeds. Same wall-clock.
3. **Ablation** (only if treatment ≤ baseline) — single-seed runs reverting one cluster at a time, in this order:
   1. Revert PPO schedules (Section 2A/2B), keep reward changes
   2. Revert asymmetric life-loss (1C)
   3. Revert movement coef bump (1B)
   4. Revert stall penalty (1A) — if this is needed, the diagnosis is wrong

Total worst-case wall-clock: ~1h 45m of training.

### Metrics tracked per run

| metric | source | use |
|---|---|---|
| per-level win rate at 1M steps | `code/scripts/analyze_metrics.py` | primary success criterion |
| per-level death cause breakdown | same | confirms *mechanism* of improvement |
| episode reward curve | TensorBoard | regression detection |
| reward component breakdown | TensorBoard (`reward_components`) | catches one component dominating |
| value loss, policy loss, entropy | TensorBoard | divergence / over-regularization |
| visits per level | metrics CSV | curriculum behavior unchanged |

### Success criteria

| tier | Mario1-1 | Mario1-2 | any other Mario level reached by treatment runs |
|---|---|---|---|
| ship | ≥ 50% | ≥ 15% | no level regresses by > 5pp vs baseline (same level, mean across seeds) |
| stretch | ≥ 60% | ≥ 30% | all levels visited ≥ 30 times have ≥ 20% win rate |
| retry | < 40% | < 5% | OR any level regresses by ≥ 10pp |

"Reached" / "visited" means the level appears in the `analyze_metrics.py` per-level table at the end of training (visits ≥ 1).

### Diagnostic checks during runs

At each saved checkpoint (default: every 20k steps per `save_freq` in `grid.yaml`), open the TensorBoard scalars for the current run and check:

- **Stall penalty dominance**: in `reward_components/stall`, the per-episode magnitude should be ≤ ~20% of total `|reward|`. If larger, abort the run and dial 1A coefficients back by 50%.
- **Value loss after the `vf_coef` bump**: if `train/value_loss` is rising from ~500k onward, drop `vf_coef` to `0.3` and restart.
- **Entropy collapse**: end-of-training `train/entropy_loss` should correspond to roughly 0.5–1.0 nats. If < 0.2, raise the end-point of the entropy schedule (e.g. `0.003` → `0.005`) and restart.

These are manual interventions during the experiment, not automated callbacks.

## What we are *not* validating

- No hyperparameter sweeps. One treatment, paired against one baseline. Sweeps over LR endpoints / entropy endpoints / stall coefficients would balloon to dozens of runs.
- No curriculum probing — out of scope per Approach B. Metrics will tell us if curriculum starvation is a problem the reward fix can't reach.
- No architecture / `torch.compile` work.

## Risks

| risk | likelihood | mitigation |
|---|---|---|
| Stronger stall penalty makes the agent take more reckless jumps and *increases* pit deaths | medium | tracked in per-cause death breakdown; ablate 1A if observed |
| Higher early entropy (0.015 vs current 0.01) prevents the policy from converging on 1-1 | low–medium | spot-check 250k checkpoint; if 1-1 win rate is lower than baseline at 250k, drop entropy schedule start to 0.012 |
| `vf_coef` bump re-inflates value loss | low | drop to 0.3 if observed |
| LR schedule converges too early on a local optimum | low | end-point 1e-4 is conservative; if observed, raise end to 1.5e-4 |
| Asymmetric life-loss penalty interacts with the existing terminal death penalty in unexpected ways | low | logic only fires on `life_lost`, not terminal `terminated`; unit-test the reward fn change before running |

## Implementation order

1. Add `consecutive_stall_steps` counter to `_ScoreTracker` in `train_platformer.py`.
2. Apply Section 1 reward changes (1A, 1B, 1C).
3. Add the schedule helper (`schedules.py`) and wire into `train.py`.
4. Apply Section 2 YAML changes (2B, 2C).
5. Run baseline (3 seeds, current config — re-confirm we have the floor).
6. Run treatment (3 seeds, all changes).
7. Compare with `analyze_metrics.py`; ablate if needed.

## Files touched

- `code/rewards/train_platformer.py` (reward shaping + tracker)
- `code/conf/algo/ppo.yaml` (PPO config)
- `code/scripts/schedules.py` (new; schedule string → callable)
- `code/scripts/train.py` (wire schedule helper)
- `code/tests/test_train_platformer.py` (new file in existing `code/tests/` dir) — unit-test the new reward branches: progressive stall streak, asymmetric life-loss by cause, `consecutive_stall_steps` reset on win / life-loss

## Out of scope (explicit)

- Curriculum logic, level ordering, advance/fallback thresholds
- Training budget beyond 1M steps
- Architecture changes, `torch.compile`, MPS / GPU work
- New personas, new games, observation-space changes
