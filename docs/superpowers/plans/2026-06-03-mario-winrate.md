# Mario Win-Rate Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve agent win rate across all Mario levels within a fixed 1M-step training budget, via reward shaping (stronger stall penalty, sharper forward-movement gradient, cause-aware life-loss) and PPO hyperparameter schedules (LR / entropy / clip range) plus `vf_coef` bump.

**Architecture:** Modifications to two existing files (`code/rewards/train_platformer.py`, `code/conf/algo/ppo.yaml`), one new helper module (`code/scripts/schedules.py`), and a small wiring change in `code/scripts/train.py`. New tests cover the reward branches and schedule helpers.

**Tech Stack:** Stable-Baselines3 (PPO), Hydra (config), pytest (test framework), `LinearSchedule` from `stable_baselines3.common.utils`.

**Important — No auto-commits:** The user has explicit preference that the executing agent does NOT run `git add` or `git commit`. Commit steps below are written as **"Pause for user review"** checkpoints. The user runs commits manually after reviewing each task's diff.

**Spec:** `docs/superpowers/specs/2026-06-03-mario-winrate-design.md`

---

## File Structure

```
code/
  rewards/
    train_platformer.py     [MODIFY] _ScoreTracker (add stall streak counter), simple persona, adept persona
  scripts/
    schedules.py            [NEW]    LinearSchedule string-spec parser
    train.py                [MODIFY] wire schedule helper into algo_kwargs
  conf/
    algo/
      ppo.yaml              [MODIFY] LR/ent/clip as schedule strings, vf_coef 0.25 → 0.4
  tests/
    test_train_platformer.py [NEW]   reward fn tests (tracker, stall progressive, asymmetric life-loss, movement coef)
    test_schedules.py        [NEW]   schedule parser tests
```

Each file has one clear responsibility:
- `train_platformer.py` owns reward computation per persona.
- `schedules.py` owns string → callable parsing only (no SB3 imports beyond `LinearSchedule`).
- `train.py` orchestrates: it consumes parsed kwargs but doesn't define parsing logic.
- `ppo.yaml` is pure config — no Python.

---

## Task 1: Add `consecutive_stall_steps` counter to `_ScoreTracker`

**Why:** Reward functions need to know how long the agent has been stalling to apply the progressive multiplier (doubles after 60 steps). The counter must reset on win / life-loss / non-stall step.

**Files:**
- Modify: `code/rewards/train_platformer.py` (lines 16–27, 28–168) — `_ScoreTracker.__init__` and `_ScoreTracker.step`
- Create: `code/tests/test_train_platformer.py`

- [ ] **Step 1.1: Write failing tests for `consecutive_stall_steps`**

Create `code/tests/test_train_platformer.py`:

```python
"""Unit tests for reward functions and tracker in code.rewards.train_platformer."""
from code.rewards.train_platformer import _ScoreTracker


def _info(**overrides):
    """Build a minimal info dict with sensible defaults."""
    base = {
        "score": 0,
        "goal_dist": 0.0,
        "dijkstra_dist": -1.0,
        "x_position": 0.0,
        "max_x_seen": 0.0,
        "coins_collected": 0,
        "lives": 3,
        "stalled": False,
        "won": False,
        "terminated": False,
        "cause": "",
    }
    base.update(overrides)
    return base


# --- consecutive_stall_steps counter ---

def test_stall_streak_starts_at_zero():
    tracker = _ScoreTracker()
    info = _info()
    tracker.step(info)
    assert info["consecutive_stall_steps"] == 0


def test_stall_streak_increments_when_stalled():
    tracker = _ScoreTracker()
    info1 = _info(stalled=True)
    info2 = _info(stalled=True)
    tracker.step(info1)
    tracker.step(info2)
    assert info2["consecutive_stall_steps"] == 2


def test_stall_streak_resets_when_not_stalled():
    tracker = _ScoreTracker()
    tracker.step(_info(stalled=True))
    tracker.step(_info(stalled=True))
    info3 = _info(stalled=False)
    tracker.step(info3)
    assert info3["consecutive_stall_steps"] == 0


def test_stall_streak_resets_on_life_loss():
    tracker = _ScoreTracker()
    tracker.step(_info(stalled=True, lives=3))
    info2 = _info(stalled=True, lives=2)  # life lost
    tracker.step(info2)
    assert info2["consecutive_stall_steps"] == 0


def test_stall_streak_resets_on_win():
    tracker = _ScoreTracker()
    tracker.step(_info(stalled=True))
    info2 = _info(stalled=True, won=True)
    tracker.step(info2)
    assert info2["consecutive_stall_steps"] == 0
```

- [ ] **Step 1.2: Run tests, verify they fail**

Run: `pytest code/tests/test_train_platformer.py -v -k stall_streak`

Expected: 5 tests FAIL with `KeyError: 'consecutive_stall_steps'`.

- [ ] **Step 1.3: Implement counter in `_ScoreTracker`**

Modify `code/rewards/train_platformer.py`. In `_ScoreTracker.reset` (around line 19-26), add the new attribute:

```python
    def reset(self):
        self.prev_score    = 0
        self.last_dist     = None
        self.last_dijkstra = None
        self.last_x        = None
        self.max_x         = 0.0
        self.last_coins    = 0
        self.last_lives    = None
        self.consecutive_stall_steps = 0
```

In `_ScoreTracker.step`, near the end (just before the `return` statement at line 167), add the streak update and emit it into `info`:

```python
        # Stall streak — increments while stalled, resets on non-stall / life-loss / win.
        # Reward fns read info["consecutive_stall_steps"] to apply a progressive
        # multiplier (see adept / simple personas).
        is_stalled = bool(info.get("stalled", False))
        if life_lost or info.get("won", False) or not is_stalled:
            self.consecutive_stall_steps = 0
        else:
            self.consecutive_stall_steps += 1
        info["consecutive_stall_steps"] = self.consecutive_stall_steps

        return current_score, inc
```

- [ ] **Step 1.4: Run tests, verify they pass**

Run: `pytest code/tests/test_train_platformer.py -v -k stall_streak`

Expected: 5 tests PASS.

- [ ] **Step 1.5: Pause for user review**

Stop here. Show the diff to the user:
```bash
git diff code/rewards/train_platformer.py code/tests/test_train_platformer.py
```
Wait for the user's go-ahead before continuing to Task 2. The user will handle commits.

---

## Task 2: Apply stall penalty bump + progressive multiplier to `adept` persona

**Why:** Stall is 26/64 of Mario1-2 deaths. Current `r_stall = -0.003` is too small to bias against stalling. New value `-0.012` (4×) with progressive doubling after 60 streak steps makes stall progressively worse than risky movement.

**Files:**
- Modify: `code/rewards/train_platformer.py` (line 249) — `adept` persona `r_stall` computation

- [ ] **Step 2.1: Write failing tests for adept stall**

Append to `code/tests/test_train_platformer.py`:

```python
from code.rewards.train_platformer import adept


def _make_reward_fn(factory):
    """Helper: call the factory to get a fresh reward fn with its own tracker."""
    return factory()


def _step_adept(reward_fn, **info_overrides):
    """Call adept reward fn and return the components dict written to info."""
    info = _info(**info_overrides)
    reward_fn(obs=None, base=None, terminated=info["terminated"], info=info)
    return info["reward_components"]


# --- adept stall penalty ---

def test_adept_no_stall_when_not_stalled():
    fn = _make_reward_fn(adept)
    components = _step_adept(fn, stalled=False)
    assert components["stall"] == 0.0


def test_adept_base_stall_penalty():
    fn = _make_reward_fn(adept)
    components = _step_adept(fn, stalled=True)
    assert components["stall"] == -0.012


def test_adept_progressive_stall_after_60_steps():
    fn = _make_reward_fn(adept)
    # First 60 stall steps: base penalty
    for _ in range(60):
        info = _info(stalled=True)
        fn(obs=None, base=None, terminated=False, info=info)
        assert info["reward_components"]["stall"] == -0.012
    # 61st step: doubled
    info = _info(stalled=True)
    fn(obs=None, base=None, terminated=False, info=info)
    assert info["reward_components"]["stall"] == -0.024


def test_adept_stall_resets_after_non_stall():
    fn = _make_reward_fn(adept)
    for _ in range(70):
        fn(obs=None, base=None, terminated=False, info=_info(stalled=True))
    # One non-stall step
    fn(obs=None, base=None, terminated=False, info=_info(stalled=False))
    # Next stall step: back to base penalty
    info = _info(stalled=True)
    fn(obs=None, base=None, terminated=False, info=info)
    assert info["reward_components"]["stall"] == -0.012
```

- [ ] **Step 2.2: Run tests, verify they fail**

Run: `pytest code/tests/test_train_platformer.py -v -k adept_`

Expected: 4 tests FAIL — `test_adept_base_stall_penalty` fails because current value is `-0.003`; progressive tests fail because no progressive logic.

- [ ] **Step 2.3: Implement bumped stall + progressive multiplier in adept**

Modify `code/rewards/train_platformer.py`, in the `adept` function. Replace line 249:

```python
    r_stall = -0.003 if bool(info.get("stalled", False)) else 0.0
```

with:

```python
    # Stall penalty with progressive multiplier after a 60-step streak.
    # Why: at -0.003 the agent learned to stall indefinitely on Mario1-2
    # (stall = 26/64 deaths). Bumped to -0.012 base; doubles after 60
    # consecutive stall steps to force the policy out of frozen states.
    if bool(info.get("stalled", False)):
        streak = int(info.get("consecutive_stall_steps", 0))
        r_stall = -0.024 if streak > 60 else -0.012
    else:
        r_stall = 0.0
```

- [ ] **Step 2.4: Run tests, verify they pass**

Run: `pytest code/tests/test_train_platformer.py -v -k adept_`

Expected: 4 tests PASS. Also rerun the full test file to confirm no regression:

Run: `pytest code/tests/test_train_platformer.py -v`

Expected: All 9 tests (5 stall_streak + 4 adept) PASS.

- [ ] **Step 2.5: Pause for user review**

Show diff. Wait for user go-ahead before Task 3.

---

## Task 3: Apply forward-movement coefficient bump to `adept`

**Why:** At 0.003, alignment reward is small (~0.0015/step at max speed). Pit deaths on 1-1 (26/104) suggest the agent doesn't commit to running jumps. Bumping to 0.005 makes purposeful forward movement more attractive.

**Files:**
- Modify: `code/rewards/train_platformer.py` (line 239) — `adept` persona `r_alignment` coefficient

- [ ] **Step 3.1: Write failing test for adept alignment**

Append to `code/tests/test_train_platformer.py`:

```python
# --- adept alignment reward ---

def test_adept_alignment_uses_new_coefficient():
    """When velocity aligns with step_dx, alignment = 1.0 * 0.005 = 0.005."""
    fn = _make_reward_fn(adept)
    components = _step_adept(
        fn,
        dijkstra_dist=0.5,
        velocity_x=10.0,   # speed = 10, fully horizontal
        velocity_y=0.0,
        step_dx=1.0,       # goal directly right
        step_dy=0.0,
    )
    # Need a previous valid step to compute progress; do a second step.
    components = _step_adept(
        fn,
        dijkstra_dist=0.5,
        velocity_x=10.0,
        velocity_y=0.0,
        step_dx=1.0,
        step_dy=0.0,
    )
    assert components["alignment"] == pytest.approx(0.005, abs=1e-6)


def test_adept_alignment_zero_when_velocity_zero():
    fn = _make_reward_fn(adept)
    components = _step_adept(
        fn,
        dijkstra_dist=0.5,
        velocity_x=0.0,
        velocity_y=0.0,
    )
    assert components["alignment"] == 0.0
```

Also add `import pytest` near the top of the test file if not already present.

- [ ] **Step 3.2: Run tests, verify they fail**

Run: `pytest code/tests/test_train_platformer.py -v -k alignment`

Expected: `test_adept_alignment_uses_new_coefficient` FAILS (returns 0.003, expected 0.005).

- [ ] **Step 3.3: Implement the bump**

Modify `code/rewards/train_platformer.py`, line 239. Change:

```python
            r_alignment = max(0.0, alignment) * 0.003
```

to:

```python
            # Bumped 0.003 → 0.005 — agent under-commits to running jumps.
            r_alignment = max(0.0, alignment) * 0.005
```

- [ ] **Step 3.4: Run tests, verify they pass**

Run: `pytest code/tests/test_train_platformer.py -v -k alignment`

Expected: 2 PASS.

- [ ] **Step 3.5: Pause for user review**

---

## Task 4: Apply asymmetric life-loss penalty to `adept`

**Why:** Currently flat `-0.3` regardless of cause. Stall and pit are signal-poor failures that should be more painful than enemy hits. Cause-aware penalty: Pit=-0.8, Stall=-1.0, Enemy/Koopa/Spike=-0.4, other=-0.3.

`info["cause"]` is emitted by `platformer_core._info()` and is one of: `"Pit"`, `"Stall"`, `"Timeout"`, `"Koopa"`, `"Enemy"`, `"Spike"`, `"Goal"` (win), or `""` (no event).

**Files:**
- Modify: `code/rewards/train_platformer.py` (lines 251–257) — adept terminal/life-loss branch

- [ ] **Step 4.1: Write failing tests for adept asymmetric life-loss**

Append to `code/tests/test_train_platformer.py`:

```python
# --- adept asymmetric life-loss ---

def test_adept_life_loss_pit():
    fn = _make_reward_fn(adept)
    # Establish baseline (lives=3)
    fn(obs=None, base=None, terminated=False, info=_info(lives=3))
    # Life lost to pit
    info = _info(lives=2, cause="Pit")
    fn(obs=None, base=None, terminated=False, info=info)
    assert info["reward_components"]["death"] == -0.8


def test_adept_life_loss_stall():
    fn = _make_reward_fn(adept)
    fn(obs=None, base=None, terminated=False, info=_info(lives=3))
    info = _info(lives=2, cause="Stall")
    fn(obs=None, base=None, terminated=False, info=info)
    assert info["reward_components"]["death"] == -1.0


def test_adept_life_loss_enemy_koopa():
    fn = _make_reward_fn(adept)
    fn(obs=None, base=None, terminated=False, info=_info(lives=3))
    for cause in ("Enemy", "Koopa", "Spike"):
        fn_local = _make_reward_fn(adept)
        fn_local(obs=None, base=None, terminated=False, info=_info(lives=3))
        info = _info(lives=2, cause=cause)
        fn_local(obs=None, base=None, terminated=False, info=info)
        assert info["reward_components"]["death"] == -0.4, f"cause={cause}"


def test_adept_life_loss_unknown_cause_defaults():
    fn = _make_reward_fn(adept)
    fn(obs=None, base=None, terminated=False, info=_info(lives=3))
    info = _info(lives=2, cause="")
    fn(obs=None, base=None, terminated=False, info=info)
    assert info["reward_components"]["death"] == -0.3


def test_adept_terminal_death_still_uses_minus_three():
    fn = _make_reward_fn(adept)
    info = _info(lives=0, terminated=True, cause="Pit")
    fn(obs=None, base=None, terminated=True, info=info)
    assert info["reward_components"]["death"] == -3.0
```

- [ ] **Step 4.2: Run tests, verify they fail**

Run: `pytest code/tests/test_train_platformer.py -v -k life_loss`

Expected: 4 FAIL (pit / stall / enemy-koopa / unknown each expect new values; current returns -0.3 for all). `test_adept_terminal_death_still_uses_minus_three` likely PASSES (-3.0 already).

- [ ] **Step 4.3: Implement asymmetric penalty in adept**

Add a module-level helper near the top of `code/rewards/train_platformer.py` (after the existing imports, before `_ScoreTracker`):

```python
# --- Cause-aware life-loss penalty table ---
# Pit and Stall are signal-poor failures (the agent gets no information about
# what went wrong) so we penalise them more. Enemy/spike contact has clearer
# spatial signal so penalty is lighter.
_LIFE_LOSS_PENALTY = {
    "Pit":   -0.8,
    "Stall": -1.0,
    "Enemy": -0.4,
    "Koopa": -0.4,
    "Spike": -0.4,
}


def _life_loss_penalty(cause: str, default: float) -> float:
    return _LIFE_LOSS_PENALTY.get(cause, default)
```

Then in `adept`, replace lines 251–257:

```python
    r_win, r_death = 0.0, 0.0
    if won:
        r_win = 5.0
    elif terminated:
        r_death = -3.0
    elif life_lost:
        r_death = -0.3
```

with:

```python
    r_win, r_death = 0.0, 0.0
    if won:
        r_win = 5.0
    elif terminated:
        r_death = -3.0
    elif life_lost:
        r_death = _life_loss_penalty(str(info.get("cause", "")), default=-0.3)
```

- [ ] **Step 4.4: Run tests, verify they pass**

Run: `pytest code/tests/test_train_platformer.py -v -k life_loss`

Expected: 5 PASS.

Also run full file:

Run: `pytest code/tests/test_train_platformer.py -v`

Expected: All tests so far PASS.

- [ ] **Step 4.5: Pause for user review**

---

## Task 5: Apply the same three changes to `simple` persona

**Why:** `simple` is the other persona used for platformer training (see `grid.yaml` personas list). It needs the same reward improvements so we have consistent behavior across both.

**Files:**
- Modify: `code/rewards/train_platformer.py` (lines 273–335) — `simple` persona

- [ ] **Step 5.1: Write failing tests for simple persona**

Append to `code/tests/test_train_platformer.py`:

```python
from code.rewards.train_platformer import simple


def _step_simple(reward_fn, **info_overrides):
    info = _info(**info_overrides)
    reward_fn(obs=None, base=None, terminated=info["terminated"], info=info)
    return info["reward_components"]


# --- simple stall penalty ---

def test_simple_stall_penalty_when_off_platform_and_no_progress():
    fn = _make_reward_fn(simple)
    # First step to establish last_dist anchor.
    fn(obs=None, base=None, terminated=False, info=_info(goal_dist=100.0))
    # Second step: same goal_dist → progress ≈ 0; not on platform → stall.
    components = _step_simple(fn, goal_dist=100.0)
    # r_move = 0; stall sub-penalty applied: -0.005
    # The full r_move field will be the clamped r_move - stall_sub.
    # New design: stall sub-penalty bumped from -0.0005 to -0.005.
    assert components["movement"] == pytest.approx(-0.005, abs=1e-6)


def test_simple_progressive_stall_after_60_steps():
    """When info['stalled'] is True for 60+ steps, dedicated stall penalty doubles."""
    fn = _make_reward_fn(simple)
    for _ in range(60):
        info = _info(stalled=True, goal_dist=100.0)
        fn(obs=None, base=None, terminated=False, info=info)
    # 61st step: doubled penalty
    info = _info(stalled=True, goal_dist=100.0)
    fn(obs=None, base=None, terminated=False, info=info)
    # simple's stall component: -0.005 base, doubles to -0.010 after streak.
    assert info["reward_components"]["stall"] == -0.010


# --- simple movement coefficient ---

def test_simple_movement_uses_new_coefficient():
    fn = _make_reward_fn(simple)
    # Establish anchor: goal_dist=100.
    fn(obs=None, base=None, terminated=False, info=_info(goal_dist=100.0))
    # Progress = 100 - 99 = 1.0 → r_move = 1.0 * 0.005 = 0.005
    components = _step_simple(fn, goal_dist=99.0)
    assert components["movement"] == pytest.approx(0.005, abs=1e-6)


# --- simple asymmetric life-loss ---

def test_simple_life_loss_pit():
    fn = _make_reward_fn(simple)
    fn(obs=None, base=None, terminated=False, info=_info(lives=3))
    info = _info(lives=2, cause="Pit")
    fn(obs=None, base=None, terminated=False, info=info)
    assert info["reward_components"]["death"] == -0.8


def test_simple_life_loss_stall():
    fn = _make_reward_fn(simple)
    fn(obs=None, base=None, terminated=False, info=_info(lives=3))
    info = _info(lives=2, cause="Stall")
    fn(obs=None, base=None, terminated=False, info=info)
    assert info["reward_components"]["death"] == -1.0


def test_simple_life_loss_unknown_defaults_to_minus_half():
    fn = _make_reward_fn(simple)
    fn(obs=None, base=None, terminated=False, info=_info(lives=3))
    info = _info(lives=2, cause="")
    fn(obs=None, base=None, terminated=False, info=info)
    assert info["reward_components"]["death"] == -0.5
```

- [ ] **Step 5.2: Run tests, verify they fail**

Run: `pytest code/tests/test_train_platformer.py -v -k simple`

Expected: 6 FAIL.

- [ ] **Step 5.3: Implement simple persona changes**

Modify `code/rewards/train_platformer.py`. Replace the entire `simple` function body. Find the function definition (line 274) and replace its body (lines 289–335) with:

```python
    progress    = float(info.get("progress", 0.0))
    kills       = int(info.get("enemies_killed_step", 0))
    coins       = int(info.get("coins_delta", 0))
    won         = info.get("won", False)
    life_lost   = info.get("life_lost", False)
    on_platform = info.get("on_moving_platform", False)

    # ── Movement (clamped so it can't dominate) ───────────────────────────
    # Coefficient bumped 0.003 → 0.005 — agent under-commits to forward motion.
    r_move = progress * 0.005

    if progress < 0:
        r_move *= 1.5              # soft backtrack penalty
    r_move = max(-0.01, min(0.01, r_move))   # CLAMP — prevents runaway

    # ── Stall sub-penalty (not on platforms) ──────────────────────────────
    # Bumped -0.0005 → -0.005 (10×) — current value is too small relative
    # to the implicit cost of moving forward, so the policy prefers to freeze.
    if not on_platform and abs(progress) < 0.5:
        r_move -= 0.005

    # ── Small alive bonus (replaces frontier) ─────────────────────────────
    # 0.0005/step × 3000 steps = 1.5 per episode. Win is 5.0. Balanced.
    r_alive = 0.0005

    # ── Platform patience ─────────────────────────────────────────────────
    r_patience = 0.002 if on_platform else 0.0

    # ── Dedicated stall component with progressive multiplier ─────────────
    # When info["stalled"] is True for >60 consecutive steps, penalty doubles.
    # See _ScoreTracker for the streak counter logic.
    if bool(info.get("stalled", False)):
        streak = int(info.get("consecutive_stall_steps", 0))
        r_stall = -0.010 if streak > 60 else -0.005
    else:
        r_stall = 0.0

    r_coin = 0.08 * coins
    r_kill = 0.1 * kills

    r_win, r_death = 0.0, 0.0
    if won:
        r_win = 5.0
    elif terminated:
        r_death = -3.0
    elif life_lost:
        r_death = _life_loss_penalty(str(info.get("cause", "")), default=-0.5)

    return {
        "movement":  r_move,
        "alive":     r_alive,
        "patience":  r_patience,
        "stall":     r_stall,
        "coins":     r_coin,
        "kills":     r_kill,
        "win":       r_win,
        "time":      -0.00005,
        "death":     r_death,
    }
```

Note the additions vs original: new `stall` component (separate from `movement`), new `_life_loss_penalty()` call, bumped movement coefficient and stall sub-penalty inside `r_move`.

- [ ] **Step 5.4: Run tests, verify they pass**

Run: `pytest code/tests/test_train_platformer.py -v -k simple`

Expected: 6 PASS.

Full file:

Run: `pytest code/tests/test_train_platformer.py -v`

Expected: All tests PASS.

- [ ] **Step 5.5: Pause for user review**

---

## Task 6: Add schedule helper module

**Why:** SB3's PPO accepts `learning_rate`, `ent_coef`, and `clip_range` as floats or callables. Hydra YAML can only express scalars; we encode schedules as strings like `linear_3e-4_to_1e-4` and parse them to `LinearSchedule` callables at algo construction time.

**Files:**
- Create: `code/scripts/schedules.py`
- Create: `code/tests/test_schedules.py`

- [ ] **Step 6.1: Write failing tests for the schedule parser**

Create `code/tests/test_schedules.py`:

```python
"""Tests for code.scripts.schedules — string → callable parser for SB3 schedules."""
import pytest

from code.scripts.schedules import resolve_schedule


# --- linear schedules ---

def test_linear_schedule_start_value():
    sched = resolve_schedule("linear_3e-4_to_1e-4")
    # progress_remaining=1.0 at start → returns start value
    assert sched(1.0) == pytest.approx(3e-4, abs=1e-9)


def test_linear_schedule_end_value():
    sched = resolve_schedule("linear_3e-4_to_1e-4")
    # progress_remaining=0.0 at end → returns end value
    assert sched(0.0) == pytest.approx(1e-4, abs=1e-9)


def test_linear_schedule_midpoint():
    sched = resolve_schedule("linear_0.3_to_0.1")
    # progress_remaining=0.5 → midpoint between 0.3 and 0.1 = 0.2
    assert sched(0.5) == pytest.approx(0.2, abs=1e-9)


def test_linear_schedule_handles_scientific_notation():
    sched = resolve_schedule("linear_1.5e-2_to_3e-3")
    assert sched(1.0) == pytest.approx(1.5e-2, abs=1e-9)
    assert sched(0.0) == pytest.approx(3e-3, abs=1e-9)


def test_linear_schedule_handles_plain_decimals():
    sched = resolve_schedule("linear_0.015_to_0.003")
    assert sched(1.0) == pytest.approx(0.015, abs=1e-9)
    assert sched(0.0) == pytest.approx(0.003, abs=1e-9)


# --- float passthrough (already a number, not a string) ---

def test_float_passthrough_returns_value_unchanged():
    # If passed a float (not a string), return it as-is for SB3 to handle.
    assert resolve_schedule(0.0003) == 0.0003


def test_int_passthrough_returns_value_unchanged():
    assert resolve_schedule(0) == 0


# --- error cases ---

def test_invalid_string_raises():
    with pytest.raises(ValueError, match="schedule"):
        resolve_schedule("nonsense_string")


def test_linear_with_bad_numbers_raises():
    with pytest.raises(ValueError):
        resolve_schedule("linear_abc_to_def")
```

- [ ] **Step 6.2: Run tests, verify they fail**

Run: `pytest code/tests/test_schedules.py -v`

Expected: All FAIL with `ModuleNotFoundError: No module named 'code.scripts.schedules'`.

- [ ] **Step 6.3: Implement `code/scripts/schedules.py`**

Create the file:

```python
"""Parse schedule string specs into SB3-compatible callables.

YAML config can encode an SB3 schedule using strings like:
    "linear_3e-4_to_1e-4"   →  LinearSchedule(3e-4, 1e-4, 1.0)
    "linear_0.3_to_0.1"     →  LinearSchedule(0.3, 0.1, 1.0)

resolve_schedule() converts these strings into callables that SB3 PPO can
consume directly. Floats / ints pass through unchanged so YAML can mix
scalar and scheduled values without ceremony.
"""
from __future__ import annotations
import re
from typing import Callable, Union

from stable_baselines3.common.utils import LinearSchedule

ScheduleSpec = Union[str, float, int]
ScheduleValue = Union[float, Callable[[float], float]]

_LINEAR_RE = re.compile(
    r"^linear_(?P<start>[0-9.eE+\-]+)_to_(?P<end>[0-9.eE+\-]+)$"
)


def resolve_schedule(spec: ScheduleSpec) -> ScheduleValue:
    """Convert a schedule spec to either a float or an SB3 callable.

    Pass-through for non-string inputs (float / int returned unchanged).
    For strings matching ``linear_<start>_to_<end>``, returns a
    LinearSchedule that anneals from start at progress_remaining=1.0 to
    end at progress_remaining=0.0 (end_fraction=1.0).
    """
    if not isinstance(spec, str):
        return spec

    m = _LINEAR_RE.match(spec)
    if m:
        try:
            start = float(m.group("start"))
            end = float(m.group("end"))
        except ValueError as e:
            raise ValueError(f"linear schedule has non-numeric bounds: {spec!r}") from e
        return LinearSchedule(start, end, end_fraction=1.0)

    raise ValueError(
        f"unrecognised schedule spec {spec!r}; "
        "expected float or string of the form 'linear_<start>_to_<end>'"
    )
```

- [ ] **Step 6.4: Run tests, verify they pass**

Run: `pytest code/tests/test_schedules.py -v`

Expected: All 9 tests PASS.

- [ ] **Step 6.5: Pause for user review**

---

## Task 7: Wire schedule helper into `train.py`

**Why:** YAML strings are inert until train.py parses them. We pop the three schedulable fields out of `algo_kwargs` and replace them with the result of `resolve_schedule()` before instantiating the algo.

**Files:**
- Modify: `code/scripts/train.py` (around line 999–1000) — `algo_kwargs` construction

- [ ] **Step 7.1: Read current train.py wiring**

Run: `sed -n '993,1005p' code/scripts/train.py`

Confirm the structure:
- Line 994: `algo_conf = _load_yaml(conf_root, "algo", model_name)`
- Line 999: `algo_kwargs = {k: v for k, v in algo_conf.items() if k not in {"_target_", "name", "policy", "policy_kwargs"}}`

- [ ] **Step 7.2: Write failing integration test**

Append to `code/tests/test_schedules.py`:

```python
# --- integration: train.py reads schedule strings from YAML ---

def test_train_resolves_lr_schedule_string():
    """If algo_kwargs has a string LR, train.py should convert it before passing to SB3."""
    from code.scripts.train import _apply_schedule_kwargs

    raw = {
        "learning_rate": "linear_3e-4_to_1e-4",
        "ent_coef":      "linear_0.015_to_0.003",
        "clip_range":    "linear_0.3_to_0.1",
        "vf_coef":       0.4,  # scalar — passthrough
        "n_steps":       2048, # scalar — passthrough
    }
    resolved = _apply_schedule_kwargs(raw)
    assert callable(resolved["learning_rate"])
    assert resolved["learning_rate"](1.0) == pytest.approx(3e-4)
    assert resolved["learning_rate"](0.0) == pytest.approx(1e-4)
    assert callable(resolved["ent_coef"])
    assert callable(resolved["clip_range"])
    assert resolved["vf_coef"] == 0.4
    assert resolved["n_steps"] == 2048


def test_train_leaves_scalar_lr_alone():
    from code.scripts.train import _apply_schedule_kwargs

    raw = {"learning_rate": 3e-4, "ent_coef": 0.01, "clip_range": 0.2}
    resolved = _apply_schedule_kwargs(raw)
    assert resolved["learning_rate"] == 3e-4
    assert resolved["ent_coef"] == 0.01
    assert resolved["clip_range"] == 0.2
```

- [ ] **Step 7.3: Run tests, verify they fail**

Run: `pytest code/tests/test_schedules.py -v -k train_`

Expected: 2 FAIL with `ImportError` for `_apply_schedule_kwargs`.

- [ ] **Step 7.4: Implement `_apply_schedule_kwargs` in train.py**

Add the helper at module scope in `code/scripts/train.py`, near other utility functions (above `def main` or wherever fits). First add the import at the top of the file:

```python
from code.scripts.schedules import resolve_schedule
```

Then add the helper function:

```python
def _apply_schedule_kwargs(algo_kwargs: dict) -> dict:
    """Resolve schedule-spec strings in algo kwargs to SB3 callables.

    Hydra YAML can only express scalars, so schedules are encoded as strings
    like "linear_3e-4_to_1e-4". This helper rewrites the three SB3 PPO fields
    that accept callables (learning_rate, ent_coef, clip_range), leaving all
    other entries untouched.
    """
    schedulable = ("learning_rate", "ent_coef", "clip_range")
    resolved = dict(algo_kwargs)
    for key in schedulable:
        if key in resolved:
            resolved[key] = resolve_schedule(resolved[key])
    return resolved
```

Then call it from the main flow. Locate the line that builds `algo_kwargs` (around line 999):

```python
        algo_kwargs   = {k: v for k, v in algo_conf.items()
                         if k not in {"_target_", "name", "policy", "policy_kwargs"}}
```

Immediately after this line, add:

```python
        algo_kwargs = _apply_schedule_kwargs(algo_kwargs)
```

- [ ] **Step 7.5: Run tests, verify they pass**

Run: `pytest code/tests/test_schedules.py -v`

Expected: All 11 tests PASS.

Also confirm nothing else broke:

Run: `pytest code/tests/ -v`

Expected: All tests across the suite PASS.

- [ ] **Step 7.6: Pause for user review**

---

## Task 8: Update `ppo.yaml` with schedule strings + `vf_coef` bump

**Why:** With the parser wired, the YAML can now switch from scalar values to schedule strings. `vf_coef` 0.25 → 0.4 partially restores the SB3 default; the YAML's own note "reduce while value loss is high" implied this should come back once the reward signal sharpens (which Sections 1–4 do).

**Files:**
- Modify: `code/conf/algo/ppo.yaml` (lines 8, 12, 13)

- [ ] **Step 8.1: Apply YAML changes**

Modify `code/conf/algo/ppo.yaml`. Replace:

```yaml
learning_rate: 0.0003  # keep for now, add schedule after reward fix
```

with:

```yaml
learning_rate: linear_3e-4_to_1e-4   # 3e-4 → 1e-4 over training; parsed by code/scripts/schedules.py
```

Replace:

```yaml
clip_range: 0.2
```

with:

```yaml
clip_range: linear_0.3_to_0.1        # bigger early updates, tighter late
```

Replace:

```yaml
ent_coef: 0.01         # was 0.02 — reduce once reward signal is stronger
```

with:

```yaml
ent_coef: linear_0.015_to_0.003      # higher early exploration, decay below current 0.01 by end
```

Replace:

```yaml
vf_coef: 0.25          # was implicit 0.5 — reduce while value loss is high
```

with:

```yaml
vf_coef: 0.4           # partial restore — reward signal sharpened (see spec 2026-06-03)
```

- [ ] **Step 8.2: Smoke-test YAML loads and is parsed correctly**

Run a tiny Python smoke test to confirm the wiring works end-to-end:

```bash
python -c "
import yaml
with open('code/conf/algo/ppo.yaml') as f:
    cfg = yaml.safe_load(f)
from code.scripts.schedules import resolve_schedule
lr = resolve_schedule(cfg['learning_rate'])
ent = resolve_schedule(cfg['ent_coef'])
clip = resolve_schedule(cfg['clip_range'])
print(f'lr(1.0)={lr(1.0):.6f}  lr(0.0)={lr(0.0):.6f}')
print(f'ent(1.0)={ent(1.0):.6f}  ent(0.0)={ent(0.0):.6f}')
print(f'clip(1.0)={clip(1.0):.6f}  clip(0.0)={clip(0.0):.6f}')
print(f'vf_coef={cfg[\"vf_coef\"]}')
"
```

Expected output:
```
lr(1.0)=0.000300  lr(0.0)=0.000100
ent(1.0)=0.015000  ent(0.0)=0.003000
clip(1.0)=0.300000  clip(0.0)=0.100000
vf_coef=0.4
```

- [ ] **Step 8.3: Pause for user review**

---

## Task 9: Run baseline (3 seeds, **current main branch config**)

**Why:** The 35.6% / 1.6% numbers we're trying to beat are from a single seed. Without multi-seed variance estimates we can't tell if a treatment is real or noise. Baseline = current config (the state BEFORE Tasks 1–8 were merged).

This task is run by the **user**, not by an automated agent, because:
1. It takes 30–45 min wall-clock.
2. It requires a clean checkout of the pre-change config (or temporarily reverting Tasks 1–8 locally to run).
3. The output is a CSV of per-level metrics, which the user inspects.

- [ ] **Step 9.1: Stash or branch to get baseline config**

User-side step. Two clean options:

```bash
# Option A: run baseline from main first, before merging this work.
git checkout main
# … run training (see 9.2)
git checkout vLLm   # back to the branch with the changes

# Option B: stash the changes from Tasks 1–8 locally, run, restore.
git stash --include-untracked
# … run training
git stash pop
```

- [ ] **Step 9.2: Run 3 seeds**

```bash
cd "/Users/envy/Documents/Master's Projects/GitHub/PEAK-DRL-Tool"
for seed in 1234 4242 7777; do
  echo "=== baseline seed=$seed ==="
  python -m code.scripts.train \
    games=[platformer] \
    personas=[platformer_adept] \
    architectures=[spatialattention] \
    skill=Novice \
    seed=$seed \
    +extra=baseline_$seed
done
```

Each run writes outputs to `outputs/YYYY-MM-DD/HH-MM-SS/`. Note the three output dirs.

- [ ] **Step 9.3: Collect baseline metrics**

For each of the 3 runs, capture the per-level win-rate table:

```bash
python -m code.scripts.analyze_metrics --run outputs/<dir-for-seed-1234>
python -m code.scripts.analyze_metrics --run outputs/<dir-for-seed-4242>
python -m code.scripts.analyze_metrics --run outputs/<dir-for-seed-7777>
```

Record the mean ± stdev of Mario1-1 win rate, Mario1-2 win rate, and any other Mario level that appears. This is the comparison floor.

- [ ] **Step 9.4: Pause for user review**

User reviews baseline numbers. If baseline 1-1 is within ~10pp of the 35.6% reported in the spec, proceed. If baseline is wildly different (e.g. 1-1 < 20%), something else regressed — investigate before continuing.

---

## Task 10: Run treatment (3 seeds, changes applied)

**Why:** With Tasks 1–8 merged, run the same 3 seeds and compare per-level win rates to baseline.

- [ ] **Step 10.1: Confirm Tasks 1–8 are applied**

```bash
git status      # working tree clean
git log --oneline -10
# Tasks 1–8's changes should be on the branch (committed by user after each task).
```

- [ ] **Step 10.2: Run 3 seeds**

```bash
cd "/Users/envy/Documents/Master's Projects/GitHub/PEAK-DRL-Tool"
for seed in 1234 4242 7777; do
  echo "=== treatment seed=$seed ==="
  python -m code.scripts.train \
    games=[platformer] \
    personas=[platformer_adept] \
    architectures=[spatialattention] \
    skill=Novice \
    seed=$seed \
    +extra=treatment_$seed
done
```

- [ ] **Step 10.3: Collect treatment metrics**

```bash
python -m code.scripts.analyze_metrics --run outputs/<dir-for-seed-1234>
python -m code.scripts.analyze_metrics --run outputs/<dir-for-seed-4242>
python -m code.scripts.analyze_metrics --run outputs/<dir-for-seed-7777>
```

- [ ] **Step 10.4: Diagnostic checks**

For each treatment run, open TensorBoard and confirm the three diagnostic checks from the spec:

```bash
tensorboard --logdir outputs/<dir>/mylogs/
```

Check:
- **Stall dominance:** `reward_components/stall` magnitude ≤ ~20% of total `|reward|` per episode.
- **Value loss:** `train/value_loss` is not rising from ~500k onward.
- **Entropy:** end-of-training `train/entropy_loss` corresponds to 0.5–1.0 nats (rough rule of thumb: entropy_loss ≈ −entropy, so `train/entropy_loss` should be around -0.5 to -1.0 at end).

If any check fails, follow the spec's correction (abort + dial back stall coefs by 50% / drop `vf_coef` to 0.3 / raise entropy end-point to 0.005).

- [ ] **Step 10.5: Compare against baseline and apply success-criteria tiers**

Compute per-level mean ± stdev across the 3 treatment seeds. Apply the spec's success-criteria table:

| tier | Mario1-1 | Mario1-2 | any other Mario level reached |
|---|---|---|---|
| ship | ≥ 50% | ≥ 15% | no level regresses by > 5pp vs baseline |
| stretch | ≥ 60% | ≥ 30% | all levels visited ≥ 30 times have ≥ 20% win rate |
| retry | < 40% | < 5% | OR any level regresses by ≥ 10pp |

Record the tier reached.

- [ ] **Step 10.6: Pause for user decision**

If **ship** or **stretch** tier reached → done.

If **retry** tier reached → proceed to Task 11 (ablation).

---

## Task 11 (CONDITIONAL): Ablation

**Only run this task if Task 10 reached the "retry" tier.**

Single-seed runs reverting one cluster at a time in this order. After each, compare against baseline mean.

- [ ] **Step 11.1: Revert PPO schedules, keep reward changes**

Locally edit `code/conf/algo/ppo.yaml` to restore the original scalar values:

```yaml
learning_rate: 0.0003
ent_coef:      0.01
clip_range:    0.2
vf_coef:       0.25
```

Run one seed:

```bash
python -m code.scripts.train games=[platformer] personas=[platformer_adept] \
  architectures=[spatialattention] skill=Novice seed=1234 +extra=ablate_ppo
```

Record metrics. Restore the file when done.

- [ ] **Step 11.2: Revert asymmetric life-loss (Task 4) only**

In `train_platformer.py`, locally revert just the `_life_loss_penalty` calls in `adept` and `simple` back to the flat `-0.3` / `-0.5`. Run one seed, record, restore.

- [ ] **Step 11.3: Revert movement coef bump (Tasks 3 and 5B) only**

Locally revert `r_alignment` coefficient and `r_move` coefficient back to `0.003`. Run one seed, record, restore.

- [ ] **Step 11.4: (Last resort) revert stall penalty (Tasks 2 and 5A) only**

Locally revert the stall penalty bumps and the progressive multiplier. Run one seed, record, restore.

If even this doesn't help, the diagnosis is wrong — revisit the spec and consider Approach A (curriculum changes).

- [ ] **Step 11.5: Report findings to user**

Tabulate the 4 ablation results. The cluster whose removal most degrades win rate is the highest-impact change. The cluster whose removal IMPROVES win rate is the one to drop from the final design.

---

## Self-review

After writing the plan, I verified:

**Spec coverage:**
- Spec 1A (stall bump + progressive) → Tasks 2, 5
- Spec 1B (forward-movement coef) → Tasks 3, 5
- Spec 1C (asymmetric life-loss) → Tasks 4, 5
- Spec 2A (schedule helper) → Task 6
- Spec 2B (schedule values) → Tasks 6, 7, 8
- Spec 2C (vf_coef bump) → Task 8
- Validation plan (baseline / treatment / ablation) → Tasks 9, 10, 11
- Diagnostic checks → Step 10.4
- Success criteria → Step 10.5

**Placeholder scan:** No `TBD`, no `TODO`, no "implement later", no "similar to". Every code block contains the actual code.

**Type / signature consistency:**
- `consecutive_stall_steps` named identically in tracker (Task 1) and reward fns (Tasks 2, 5).
- `_life_loss_penalty(cause, default)` defined in Task 4, used in Tasks 4 and 5.
- `resolve_schedule` defined in Task 6, used in Tasks 6 and 7.
- `_apply_schedule_kwargs` defined in Task 7, tested in Task 7.
- YAML keys `learning_rate`, `ent_coef`, `clip_range`, `vf_coef` match SB3 PPO constructor.
