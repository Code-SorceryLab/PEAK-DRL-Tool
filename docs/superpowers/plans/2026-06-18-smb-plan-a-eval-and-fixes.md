# SMB Agent — Plan A: Trustworthy Eval + High-Confidence Fixes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the diagnosed root causes of the platformer (Super Mario Bros) agent's underperformance — first establish a trustworthy per-level evaluation, then rebalance reward so winning dominates, normalize returns, extend the credit-assignment horizon, register the real difficulty curriculum, and fix the curriculum/`rppo` bugs — then retrain to a measurable per-level win-rate improvement over the ~41.6% baseline.

**Architecture:** Phase 0 (Tasks 1–2) makes evaluation stationary and per-level so any improvement is measurable. Phase 1 (Tasks 3–9) applies the high-confidence learning fixes one lever at a time, each re-measured on that eval. All changes are config/code edits to the existing Stable-Baselines3 trainer — no new ML framework, no `imitation` library.

**Tech Stack:** Python 3.11, Stable-Baselines3 2.8, sb3-contrib, gymnasium 1.2, Hydra/OmegaConf, pytest. Interpreter: `.venv/bin/python` (run tests as `.venv/bin/python -m pytest code/tests/<file> -v`).

---

## ⚠️ Project rule: NEVER commit

The user's standing instruction is **never `git add` / `git commit`**. Every task's final step is a **Checkpoint** (run `.venv/bin/python -m pytest code/tests/ -q` and pause for review), NOT a commit. Do not run `git add`/`git commit` in any step.

## Task order & dependencies

Execute in number order. Phase 0 first (eval integrity) so every later change is measured on a trustworthy yardstick.

- **Phase 0 — measure first:** Task 1 (trustworthy eval env), Task 2 (per-level win-rate metric + baseline).
- **Phase 1 — high-confidence fixes:** Task 3 (reward rebalance), Task 4 (`norm_reward=True`), Task 5 (PPO horizon retune + `POTENTIAL_GAMMA` sync — after Task 4), Task 6 (register stage curriculum), Task 7 (curriculum double-pop fix), Task 8 (`rppo` import fix), Task 9 (scale envs + the real training run — depends on 1,3,4,5,6,7).

Task 2 depends on Task 1's eval env. Task 9 is the long training run that depends on the fixes landing. Re-measure on the Task-1/Task-2 eval after Task 9.

---

### Task 1: Trustworthy eval env (per-level, curriculum OFF, terminate-on-goal, fresh per run)

**Files:**
- Modify `code/games/platformer_core.py`:
  - `__init__` (add two kwargs near the existing curriculum knobs, ~line 456 and ~line 502)
  - `reset()` (lines 781–822 — add a `curriculum_enabled=False` branch mirroring `sonic_core.py:763–788`)
  - `_check_termination()` (lines 1080–1102 — terminate when `reached_goal` and `terminate_on_goal`)
  - `step()` inline-transition guard (lines 763–770 — skip transition under `terminate_on_goal`)
- Modify `code/scripts/train.py`: eval-env construction (lines 1129–1148) + move it inside the per-skill loop (after line 1149)
- Create `code/tests/test_eval_env_platformer.py`

This task mirrors the EXISTING `curriculum_enabled` mechanism already present in `code/games/sonic_core.py` (`self.curriculum_enabled = bool(kwargs.pop("curriculum_enabled", True))` at line 411, branch in `reset()` at lines 763–788) and `code/games/megaman_core.py` (line 773). The platformer core does NOT have it yet — that is the gap. The `world=<level>` kwarg already exists and pins the level (`platformer_core.py:412` `self.world = str(kwargs.pop("world", _default_world))`), and `load_level()` loads `self.world` (`platformer_core.py:866`). The goal-reach path is: `PhysicsManager.py:667–674` sets `core.reached_goal = True` and calls `core.complete_level()`, which sets `self._needs_level_transition = True` (`platformer_core.py:985–986`); the inline transition then fires in `step()` at lines 763–770. We make goal-reach a terminal event in eval instead.

- [ ] **Step 1: Write the failing test** — `code/tests/test_eval_env_platformer.py`. It builds the eval-style core directly (no training), drives the goal-reach code path the same way `PhysicsManager` does (`reached_goal=True; complete_level()`), and asserts the four properties. It also captures the shared/global curriculum state and asserts it is not mutated across resets.

```python
"""Task 1: trustworthy eval env for the platformer.

Asserts an eval-configured PlatformerCore:
  (a) stays pinned to ONE level across several resets,
  (b) does NOT mutate curriculum state on reset (curriculum OFF),
  (c) terminates the episode on reaching the goal (terminate_on_goal),
      instead of doing the inline next-level transition.
"""
import pytest
from code.games.platformer_core import PlatformerCore


def _make_eval_core(level):
    # Mirror how the eval env will be built in train.py: pin the level via the
    # existing `world=` kwarg, curriculum OFF, terminate-on-goal ON.
    return PlatformerCore(
        render_mode="none",
        world=level,
        curriculum_enabled=False,
        terminate_on_goal=True,
    )


def test_eval_env_pins_level_across_resets():
    core = _make_eval_core("Mario1-2")
    assert core.world == "Mario1-2"
    for _ in range(5):
        core.reset()
        assert core.world == "Mario1-2", "eval env drifted off the pinned level"
        assert core.current_index_world == core.level_order.index("Mario1-2")


def test_eval_env_does_not_mutate_curriculum_state():
    core = _make_eval_core("Mario1-1")
    # Snapshot the shared/global curriculum progression state.
    pos_before = core._curriculum_position
    batch_before = list(core._batch_results)
    for _ in range(5):
        core.reset()
    assert core._curriculum_position == pos_before, "curriculum position advanced during eval"
    assert core._batch_results == batch_before, "eval reset appended to curriculum batch"
    # _batch_results must stay empty: nothing should be recorded when curriculum is off.
    assert core._batch_results == []


def test_reaching_goal_terminates_without_level_transition():
    core = _make_eval_core("Mario1-1")
    core.reset()
    world_before = core.world
    # Reproduce the goal-reach effect that PhysicsManager.py:667-674 produces:
    #   core.reached_goal = True; core.complete_level()
    core.reached_goal = True
    core.complete_level()
    # Now take a normal step; with terminate_on_goal the episode must END here
    # and the world must NOT have transitioned to the next level.
    obs, reward, terminated, truncated, info = core.step([0, 0, 0])
    assert terminated is True, "reaching goal did not terminate the eval episode"
    assert core.world == world_before, "eval env transitioned levels instead of terminating"
    assert info.get("won") is True


def test_default_core_still_transitions_on_goal():
    # Regression guard: with terminate_on_goal default (False) + curriculum on,
    # goal-reach must still NOT terminate (inline transition behavior preserved).
    core = PlatformerCore(render_mode="none")  # defaults: curriculum on, terminate_on_goal off
    core.reset()
    core.reached_goal = True
    core.complete_level()
    obs, reward, terminated, truncated, info = core.step([0, 0, 0])
    assert terminated is False, "default training env must not terminate on goal"
```

- [ ] **Step 2: Run the test, confirm it fails** — `terminate_on_goal`/`curriculum_enabled` are unknown kwargs and the pin/terminate logic does not exist yet.

```
.venv/bin/python -m pytest code/tests/test_eval_env_platformer.py -v
```
Expected: `test_eval_env_pins_level_across_resets`, `test_eval_env_does_not_mutate_curriculum_state`, and `test_reaching_goal_terminates_without_level_transition` FAIL (level drifts back to curriculum pos 0 / `Mario1-1`; goal does not terminate). `test_default_core_still_transitions_on_goal` should already PASS.

- [ ] **Step 3: Add the two kwargs in `PlatformerCore.__init__`.** The existing code at `platformer_core.py:456–460` reads:
```python
        self._max_unlocked_index = int(kwargs.pop("start_unlocked", 0))
        self.speed_mult = float(kwargs.pop("speed_mult", 2.0))
        self.physics_manager.speed_mult = self.speed_mult

        self.max_steps = kwargs.pop("max_steps", None)
```
Insert the two flags right after `self._max_unlocked_index` (mirrors `sonic_core.py:411`):
```python
        self._max_unlocked_index = int(kwargs.pop("start_unlocked", 0))

        # ── Eval-trust flags (Task 1) ─────────────────────────────────────────
        # curriculum_enabled=False: reset() pins to self.world and never touches
        #   the shared curriculum/batch state. Mirrors sonic_core/megaman_core.
        # terminate_on_goal=True: reaching the goal ENDS the episode (eval only),
        #   instead of the inline next-level transition done during training.
        self.curriculum_enabled = bool(kwargs.pop("curriculum_enabled", True))
        self.terminate_on_goal  = bool(kwargs.pop("terminate_on_goal", False))

        self.speed_mult = float(kwargs.pop("speed_mult", 2.0))
        self.physics_manager.speed_mult = self.speed_mult

        self.max_steps = kwargs.pop("max_steps", None)
```

- [ ] **Step 4: Add the `curriculum_enabled=False` branch in `reset()`.** Current code at `platformer_core.py:784–818`:
```python
        # Record episode result into batch — only if it was a curriculum episode
        # (NOT a review episode, which played a different level)
        if self.level_order and not self.locked_level and not self._is_review_episode:
            self._batch_results.append(self._episode_won_current)

        # Evaluate batch when window is full
        if len(self._batch_results) >= self._batch_window and self.level_order and not self.locked_level:
            self._evaluate_curriculum_batch()

        self.reset_metrics()
        self._episode_won_current = False
        self._is_review_episode = False

        if self.locked_level:
            self.world = self.locked_level
            if self.locked_level in self.level_order:
                self.current_index_world = self.level_order.index(self.locked_level)
            else:
                self.current_index_world = 0
        elif self.level_order:
            self._curriculum_position = max(0, min(
                self._curriculum_position, len(self.level_order) - 1))

            # ── Review rotation: 25% chance to play a random earlier level ──
            ...
            else:
                self.current_index_world = self._curriculum_position

            self.world = self.level_order[self.current_index_world]
```
Gate the two batch lines on `self.curriculum_enabled` and add a pinned branch (mirrors `sonic_core.py:763–788`). Replace the block above with:
```python
        # Record episode result into batch — only if it was a curriculum episode
        # (NOT a review episode, which played a different level)
        if self.curriculum_enabled and self.level_order and not self.locked_level and not self._is_review_episode:
            self._batch_results.append(self._episode_won_current)

        # Evaluate batch when window is full
        if self.curriculum_enabled and len(self._batch_results) >= self._batch_window and self.level_order and not self.locked_level:
            self._evaluate_curriculum_batch()

        self.reset_metrics()
        self._episode_won_current = False
        self._is_review_episode = False

        if self.locked_level:
            self.world = self.locked_level
            if self.locked_level in self.level_order:
                self.current_index_world = self.level_order.index(self.locked_level)
            else:
                self.current_index_world = 0
        elif not self.curriculum_enabled and self.level_order:
            # Eval / playback: stay on the level we were constructed with.
            if self.world in self.level_order:
                self.current_index_world = self.level_order.index(self.world)
            else:
                self.current_index_world = 0
                self.world = self.level_order[self.current_index_world]
        elif self.level_order:
            self._curriculum_position = max(0, min(
                self._curriculum_position, len(self.level_order) - 1))

            # ── Review rotation: 25% chance to play a random earlier level ──
            # Keeps skills sharp on mastered levels, distributes visits,
            # prevents catastrophic forgetting. Results don't count for batch.
            if (self._curriculum_position > 0
                    and random.random() < self._review_prob):
                review_idx = random.randint(0, self._curriculum_position - 1)
                self.current_index_world = review_idx
                self._is_review_episode = True
            else:
                self.current_index_world = self._curriculum_position

            self.world = self.level_order[self.current_index_world]
```
(Leave the trailing `self._level_visits[...]`/`self.load_level()`/`return` at lines 820–822 unchanged.)

- [ ] **Step 5: Make goal-reach terminal under `terminate_on_goal` in `_check_termination()`.** Current code at `platformer_core.py:1080–1102` ends with `return False`. Add the goal check at the TOP of the method body (after the `player` guard), so it fires the same frame `reached_goal` is set:
```python
    def _check_termination(self) -> bool:
        """
        Returns True only when the episode should end (lives = 0, or no player).
        Goal completion is NOT a termination during training — it transitions
        inline in step(). In eval (terminate_on_goal=True) the goal ENDS the
        episode instead.
        """
        player = self.player
        if not player:
            return True

        # Eval-only: reaching the goal ends the episode.
        if self.terminate_on_goal and self.reached_goal:
            return True

        if self.use_timer and self.timer <= 0:
            return self._handle_death("Timeout")
        ...
```
(Keep the rest of the method body — Pit/Stall checks and final `return False` — unchanged.)

- [ ] **Step 6: Skip the inline level transition under `terminate_on_goal` in `step()`.** Current code at `platformer_core.py:763`:
```python
        if self._needs_level_transition:
```
Change to:
```python
        if self._needs_level_transition and not self.terminate_on_goal:
```
This leaves `self.world` pinned (no advance/`load_level`) when the eval env terminates on goal. Note `_needs_level_transition` stays True but is harmless: the env terminates and SB3 will `reset()` it, where `reset_metrics()`/`load_level()` run fresh.

- [ ] **Step 7: Run the unit test, confirm all green.**
```
.venv/bin/python -m pytest code/tests/test_eval_env_platformer.py -v
```
Expected: all 4 tests PASS.

- [ ] **Step 8: Wire the eval env in `train.py` — pass the flags AND construct it fresh per run.** Current code at `train.py:1129–1148`:
```python
            def make_monitored_env(render_mode=None):
                """Factory that wraps the env with Monitor for proper eval logging."""
                kw = env_kwargs.copy()
                if render_mode is not None:
                    kw['render_mode'] = render_mode
                def _init():
                    return Monitor(GameEnv(game_cls, reward_fn=active_reward_fn, **kw))
                return _init

            eval_raw_env = DummyVecEnv([make_monitored_env()])
            eval_env = VecNormalize(eval_raw_env, **vecnorm_kwargs)
            eval_env.obs_rms    = env.obs_rms
            eval_env.ret_rms    = env.ret_rms
            eval_env.training   = False
            eval_env.norm_reward = False
```
First update the factory to set the eval-trust kwargs (use the first level of `level_order` as the pinned eval level; if a config key `eval_level` exists, prefer it):
```python
            # Eval env is pinned to ONE level with the curriculum OFF and goal
            # made terminal, so best_model scores reflect single-level skill and
            # cannot drift via curriculum state. See PlatformerCore eval flags.
            _eval_level = cfg.get("eval_level") or (
                game_cls(render_mode="none").level_order[0]
                if hasattr(game_cls(render_mode="none"), "level_order") else None
            )
            def make_monitored_env(render_mode=None, _level=_eval_level):
                """Factory that wraps the env with Monitor for proper eval logging."""
                kw = env_kwargs.copy()
                if render_mode is not None:
                    kw['render_mode'] = render_mode
                kw["curriculum_enabled"] = False
                kw["terminate_on_goal"] = True
                if _level is not None:
                    kw["world"] = _level
                def _init():
                    return Monitor(GameEnv(game_cls, reward_fn=active_reward_fn, **kw))
                return _init
```
NOTE: instantiating `game_cls` twice just to read `level_order` is wasteful and pops up a pygame surface; if `game_name == "platformer"` you may instead hardcode the default by reading `cfg.get("eval_level")` and falling back to `None` (which makes the core use its own default world). Keep whichever the orchestrator prefers, but the load-bearing part is `curriculum_enabled=False`, `terminate_on_goal=True`, and a pinned `world`.

Then DELETE the two lines that build `eval_raw_env`/`eval_env` at module scope (1138–1147) and MOVE an equivalent fresh build INSIDE the per-skill loop, right after `run_count += 1` (currently line 1150), so each run gets a clean eval env with no leaked `obs_rms`/episode state:
```python
            for skill, total_timesteps in selected_skills.items():
                run_count += 1

                # Fresh eval env per run — no state leaks across runs.
                eval_raw_env = DummyVecEnv([make_monitored_env()])
                eval_env = VecNormalize(eval_raw_env, **vecnorm_kwargs)
                eval_env.obs_rms     = env.obs_rms   # share TRAIN obs normalisation
                eval_env.ret_rms     = env.ret_rms
                eval_env.training    = False
                eval_env.norm_reward = False         # keep eval rewards un-normalised
                ...
```
(`env` is the training `VecNormalize` already in scope; sharing `obs_rms` is the existing intent per the comment at `train.py:1140–1143`. Keep that.)

- [ ] **Step 9: Smoke-check the eval env builds and behaves inside the real wrapper stack.** No long training. Run a short ad-hoc check that the GameEnv path forwards the kwargs and terminates on goal:
```
.venv/bin/python -c "
import sys; sys.path.insert(0,'.')
from code.wrappers.generic_env import GameEnv
from code.games.platformer_core import PlatformerCore
e = GameEnv(PlatformerCore, world='Mario1-2', curriculum_enabled=False, terminate_on_goal=True)
e.reset()
assert e.game.world == 'Mario1-2'
e.game.reset(); assert e.game.world == 'Mario1-2'   # pinned across reset
e.game.reached_goal = True; e.game.complete_level()
obs, r, term, trunc, info = e.step([0,0,0])
assert term is True and e.game.world == 'Mario1-2', (term, e.game.world)
print('OK eval env: pinned + terminates on goal')
"
```
Expected: prints `OK eval env: pinned + terminates on goal`.

- [ ] **Step 10: Checkpoint (DO NOT COMMIT)** — run the full suite and pause for review:
```
.venv/bin/python -m pytest code/tests/ -q
```
Expected: all tests pass (existing `test_profiling_callback.py`, `test_resolve_device.py`, plus the new `test_eval_env_platformer.py`). Do NOT `git add`/`git commit`; pause for the user to review.

**What the tests prove / do NOT prove:** The unit tests prove the eval env (a) stays pinned across resets, (b) does not mutate `_curriculum_position`/`_batch_results`, and (c) returns `terminated=True` on goal without a level transition — by driving the exact goal-reach state PhysicsManager produces (`reached_goal=True; complete_level()`). They do NOT exercise real physics reaching the goal, and they do NOT prove the resulting best_model eval *scores* are better/more stable — that requires a training run (observe `models/eval_logs/<run_id>` mean reward stability across evals). The train.py wiring (Step 8) is verified structurally by Step 9's smoke check, not by a full run.

---

### Task 2: Per-level win-rate + death-cause metric, and a baseline-measurement procedure

**Phase:** 0  ·  **Depends on:** Task 1 (eval env)

**Files:**
- Create: `code/metrics/eval_summary.py`
- Create (test): `code/tests/test_eval_summary.py`
- Reference only (read, do not edit): `code/games/platformer_core.py` `_info()` lines 1705–1783; death-cause call sites lines 1090/1096/1100; `code/games/modules/System/PhysicsManager.py:263`; `code/callbacks/logging_callback.py:68-74` (existing `event`/`cause`/`level` consumer).

**Context the executor needs (already verified against the code):**
Each platformer episode's final `info` dict (from `platformer_core._info()`) carries these fields we rely on:
- `"won"` → `self.reached_goal` (bool). Source of truth for a win (`platformer_core.py:1760`).
- `"event"` → `"WIN"` on goal, `"DIED"` when `not self.alive`, else `""` (`platformer_core.py:1709-1716`).
- `"cause"` → death-cause string. The complete vocabulary actually emitted today: `"Goal"` (win), `"Pit"`, `"OOB"`, `"Stall"`, `"Timeout"`, `"Koopa"`, `"Enemy"`, `"Spike"`, `"Unknown"`, or `""` (truncation without death). Confirmed at `_handle_death("Timeout")` (1090), `_handle_death("Pit")` (1096), `_handle_death("Stall")` (1100), and `PhysicsManager.py:263` (`"Pit"`/`"OOB"`).
- `"level"` → `self.world`, the registered level name (e.g. `"Mario1-1"`, `"Mario1-2"`).

Note: there is **no** `"Wall"` death cause in the current platformer (wall collisions at `PhysicsManager.py:355` only stop motion). The requested `wall` bucket is kept for schema parity and will read `0`.

- [ ] **Step 1: Write the failing unit test** — create `code/tests/test_eval_summary.py`:
  ```python
  """Unit tests for code.metrics.eval_summary (pure metric over synthetic episode dicts)."""
  import math
  import pytest

  from code.metrics.eval_summary import (
      summarize_eval,
      summarize_by_level,
      CAUSE_BUCKETS,
  )

  # --- helpers: build synthetic FINAL per-episode info dicts ---
  def ep(level, won, cause, event=None):
      if event is None:
          event = "WIN" if won else ("DIED" if cause not in ("", "Timeout", "Time Over") else "")
      return {"level": level, "won": won, "cause": cause, "event": event}


  def test_empty_episodes_is_safe():
      out = summarize_eval([])
      assert out["n"] == 0
      assert out["wins"] == 0
      assert out["win_rate"] == 0.0
      # by_cause always has the full key set, all zero
      assert out["by_cause"] == {"pit": 0, "wall": 0, "stall": 0, "enemy": 0, "timeout": 0, "other": 0}


  def test_win_rate_basic():
      eps = [ep("Mario1-1", True, "Goal"),
             ep("Mario1-1", True, "Goal"),
             ep("Mario1-1", False, "Pit"),
             ep("Mario1-1", False, "Pit")]
      out = summarize_eval(eps)
      assert out["n"] == 4
      assert out["wins"] == 2
      assert math.isclose(out["win_rate"], 0.5)
      assert out["by_cause"]["pit"] == 2
      # wins are NOT counted in by_cause
      assert sum(out["by_cause"].values()) == 2


  def test_cause_bucket_mapping():
      eps = [
          ep("L", False, "Pit"),       # pit
          ep("L", False, "OOB"),       # pit (out-of-bounds folds into pit)
          ep("L", False, "Stall"),     # stall
          ep("L", False, "Koopa"),     # enemy
          ep("L", False, "Enemy"),     # enemy
          ep("L", False, "Spike"),     # enemy
          ep("L", False, "Timeout"),   # timeout
          ep("L", False, "Unknown"),   # other
      ]
      out = summarize_eval(eps)
      assert out["by_cause"] == {
          "pit": 2, "wall": 0, "stall": 1, "enemy": 3, "timeout": 1, "other": 1,
      }
      assert out["wins"] == 0
      assert out["win_rate"] == 0.0


  def test_truncation_without_death_counts_as_timeout():
      # An episode truncated by the eval harness step-cap: no WIN, no death cause.
      eps = [ep("L", False, "", event="")]
      out = summarize_eval(eps)
      assert out["by_cause"]["timeout"] == 1
      assert out["wins"] == 0


  def test_won_flag_wins_over_cause():
      # Defensive: if won=True but cause left stale, still a win, not a death bucket.
      eps = [ep("L", True, "Pit")]
      out = summarize_eval(eps)
      assert out["wins"] == 1
      assert sum(out["by_cause"].values()) == 0


  def test_summarize_by_level_groups():
      eps = [
          ep("Mario1-1", True, "Goal"),
          ep("Mario1-1", False, "Pit"),
          ep("Mario1-2", False, "Enemy"),
          ep("Mario1-2", False, "Enemy"),
      ]
      out = summarize_by_level(eps)
      assert set(out.keys()) == {"Mario1-1", "Mario1-2"}
      assert out["Mario1-1"]["win_rate"] == 0.5
      assert out["Mario1-1"]["by_cause"]["pit"] == 1
      assert out["Mario1-2"]["win_rate"] == 0.0
      assert out["Mario1-2"]["by_cause"]["enemy"] == 2


  def test_cause_buckets_table_is_complete():
      # Every raw cause string the platformer can emit is mapped.
      for raw in ["Pit", "OOB", "Stall", "Koopa", "Enemy", "Spike", "Timeout", "Time Over"]:
          assert raw in CAUSE_BUCKETS, f"{raw} missing from CAUSE_BUCKETS"
  ```

- [ ] **Step 2: Run the test, confirm it fails** with the module missing:
  ```
  .venv/bin/python -m pytest code/tests/test_eval_summary.py -v
  ```
  Expected: collection/import error `ModuleNotFoundError: No module named 'code.metrics.eval_summary'` (red).

- [ ] **Step 3: Implement the pure metric** — create `code/metrics/eval_summary.py`:
  ```python
  """Pure, dependency-free metrics over a list of FINAL per-episode `info` dicts.

  An episode dict is the last `info` returned by `platformer_core._info()` for an
  episode. We read three fields:
      - info["won"]   : bool  (truth for a win; == reached_goal)
      - info["cause"] : str   (death cause; "" when truncated without death)
      - info["level"] : str   (registered level name, e.g. "Mario1-1")

  Death-cause vocabulary emitted by the current platformer (verified at the
  _handle_death() call sites): "Goal","Pit","OOB","Stall","Timeout","Koopa",
  "Enemy","Spike","Unknown","".  There is NO "Wall" cause today (wall hits only
  stop motion), so the "wall" bucket below stays 0 — it is kept for schema parity.
  """
  from __future__ import annotations

  from collections import defaultdict
  from typing import Dict, List

  # Raw cause string -> coarse bucket. "Goal"/"" are NOT deaths and are absent here.
  CAUSE_BUCKETS: Dict[str, str] = {
      "Pit": "pit",
      "OOB": "pit",          # fell/ran out of bounds -> treat as a pit-style death
      "Stall": "stall",
      "Koopa": "enemy",
      "Enemy": "enemy",
      "Spike": "enemy",
      "Timeout": "timeout",
      "Time Over": "timeout",  # sonic_core variant; harmless to map here too
  }

  # The fixed set of buckets we always report (so callers can rely on the shape).
  _BUCKET_KEYS = ("pit", "wall", "stall", "enemy", "timeout", "other")

  # A truncated-without-death episode has these cause values.
  _NON_DEATH_CAUSES = ("", "Timeout", "Time Over")


  def _empty_by_cause() -> Dict[str, int]:
      return {k: 0 for k in _BUCKET_KEYS}


  def _is_win(info: dict) -> bool:
      # `won` is the source of truth; fall back to event=="WIN" if absent.
      if "won" in info:
          return bool(info["won"])
      return str(info.get("event", "")) == "WIN"


  def _death_bucket(info: dict) -> str:
      cause = str(info.get("cause", "") or "")
      event = str(info.get("event", "") or "")
      # Truncation without a death (no WIN, no death cause) -> timeout.
      if event != "WIN" and cause in _NON_DEATH_CAUSES:
          return "timeout"
      return CAUSE_BUCKETS.get(cause, "other")


  def summarize_eval(episodes: List[dict]) -> dict:
      """Aggregate a flat list of episode info dicts.

      Returns:
          {
            "n": int,                # number of episodes
            "wins": int,             # episodes with won==True
            "win_rate": float,       # wins / n (0.0 when n == 0)
            "by_cause": {            # death-cause histogram (wins excluded)
                "pit": int, "wall": int, "stall": int,
                "enemy": int, "timeout": int, "other": int,
            },
          }
      """
      n = len(episodes)
      wins = 0
      by_cause = _empty_by_cause()
      for info in episodes:
          if _is_win(info):
              wins += 1
              continue
          by_cause[_death_bucket(info)] += 1
      win_rate = (wins / n) if n else 0.0
      return {"n": n, "wins": wins, "win_rate": win_rate, "by_cause": by_cause}


  def summarize_by_level(episodes: List[dict]) -> Dict[str, dict]:
      """Group episodes by info['level'] and summarize each group."""
      groups: Dict[str, List[dict]] = defaultdict(list)
      for info in episodes:
          groups[str(info.get("level", "unknown"))].append(info)
      return {level: summarize_eval(eps) for level, eps in groups.items()}
  ```

- [ ] **Step 4: Run the unit test, confirm green:**
  ```
  .venv/bin/python -m pytest code/tests/test_eval_summary.py -v
  ```
  Expected: all 7 tests pass. This proves the classification/aggregation logic on synthetic dicts. It does **not** prove the live eval env emits these dicts — that is the baseline RUN below.

- [ ] **Step 5: Baseline-measurement PROCEDURE (uses Task 1's eval env; produces numbers to record).**
  Task 1 delivers an eval harness that runs N deterministic episodes per level on a given model and writes a JSON list of per-episode FINAL `info` dicts. Run it for **each** trained best model on **each** trained level, then feed the JSON into `summarize_by_level`.

  The two trained best models and their levels (verified on disk):
  - `models/best/platformer_ppo_platformer_simple_novice_spatialattention/best_model.zip`
  - `models/best/platformer_ppo_platformer_adept_novice_spatialattention/best_model.zip`
  - Registered platformer levels: `Mario1-1` (platformer/world1_1.txt), `Mario1-2` (platformer/world1_2.txt).
  - IMPORTANT: each best_model has a sibling `best_model_vecnorm.pkl` that MUST be loaded by Task 1's env for correct obs normalization.

  (a) Produce per-episode JSON with Task 1's harness (replace `<task1 eval entrypoint>`/flags with the exact names Task 1 introduced — e.g. `code.scripts.eval_baseline`; the contract is: deterministic, per-level, `--out` writes a JSON list of final info dicts):
  ```
  .venv/bin/python -m <task1 eval entrypoint> \
      --model models/best/platformer_ppo_platformer_simple_novice_spatialattention/best_model.zip \
      --vecnorm models/best/platformer_ppo_platformer_simple_novice_spatialattention/best_model_vecnorm.pkl \
      --level Mario1-1 --episodes 50 --deterministic \
      --out /tmp/baseline_simple_Mario1-1.json
  ```
  Repeat for `--level Mario1-2`, and for the `adept` model (4 runs total).

  (b) Reduce each JSON to the table with the pure metric (no harness dependency):
  ```
  .venv/bin/python - <<'PY'
  import glob, json
  from code.metrics.eval_summary import summarize_by_level
  for path in sorted(glob.glob("/tmp/baseline_*_*.json")):
      eps = json.load(open(path))
      print(path)
      for level, s in summarize_by_level(eps).items():
          print(f"  {level:10s} n={s['n']:3d} win_rate={s['win_rate']:.2f} by_cause={s['by_cause']}")
  PY
  ```
  **Record this baseline table** (one row per model × level): `n`, `win_rate`, and the `by_cause` breakdown `{pit, wall, stall, enemy, timeout, other}`. This is the Phase-0 reference the later phases must beat. Honesty note: these numbers depend on the trained weights and Task 1's env; the run cannot be unit-asserted to a fixed value, only recorded.

  Fallback if Task 1's harness emits a different per-episode shape: as long as each record contains the level name, a win flag, and a death cause, alias them before calling the metric (rename to `level`/`won`/`cause`), or pass `--key level=<f> won=<f> cause=<f>` if Task 1 exposed such flags.

- [ ] **Step 6: Checkpoint (DO NOT COMMIT).** Run the full suite and pause for review:
  ```
  .venv/bin/python -m pytest code/tests/ -q
  ```
  Expected: the existing suite plus the 7 new `test_eval_summary.py` tests all pass. Do not `git add`/`git commit` — pause for the user to review the diff and the recorded baseline table.

---

### Task 3: Reward rebalance — cap the non-PBRS distance term so goal + dijkstra-progress dominate

**Files:**
- Modify: `/Users/envy/Documents/Master's Projects/GitHub/PEAK-DRL-Tool/code/rewards/train_platformer.py` (the `simple` persona, lines 296–302)
- Create (test): `/Users/envy/Documents/Master's Projects/GitHub/PEAK-DRL-Tool/code/tests/test_reward_rebalance_platformer.py`

**Context the executor MUST read first (grounded in the actual code):**

The original diagnosis ("max_x is 98–99% of reward; cap the distance term") is based on a misread of `case-study/analysis.md:56–79`. That "REWARD BALANCE" table is produced by `_balance(df, rcols)` in `code/scripts/agent_analyzer.py:161–170`, which computes `share = abs_sum / total_abs_sum` over **all** non-standard CSV columns (column filter at `agent_analyzer.py:79`). It therefore mixes **logged telemetry** (`max_x_seen` mean ≈ 1386, `step_dx`, `dijkstra_dist`) with **actual reward components**. `max_x_seen` is a monotonic position counter that is logged, NOT an additive reward term.

Confirmed in `train_platformer.py`: `frontier_dx` is computed in the tracker at line 154 but **no platformer persona consumes it** (grep for `frontier` in the file returns only the tracker write + docstrings). The `adept` persona's win term already dominates a goal trajectory (verified at runtime: win ≈ 99.5% share, PBRS `potential` tiny and correct).

The REAL surviving "go-right-not-win" incentive lives in the **`simple`** persona (also exported as `default` at line 514). Current code, `train_platformer.py:296–302`:

```python
    # ── Movement (clamped so it can't dominate) ───────────────────────────
    r_move = progress * 0.003


    if progress < 0:
        r_move *= 1.5              # soft backtrack penalty
    r_move = max(-0.01, min(0.01, r_move))   # CLAMP — prevents runaway
```

Runtime measurement (script-verified): on a 500-step rightward, **non-winning** trajectory, `movement` reaches **94.2%** of `simple`'s reward share; the +0.01/step clamp over the analyzer's mean episode length of 6212 steps lets cumulative movement reach ≈ 62 while a single win pays only 5.0 — movement out-earns winning by ≈ 12x. That is the behavior to kill.

**The change:** sharply down-weight and tighten the clamp on the non-PBRS movement term so even a maximal-progress episode cannot out-earn the win bonus. PBRS terms in other personas are left untouched (they are already PBRS-only and win-dominated).

- [ ] **Step 1: Write the failing test.** Create `/Users/envy/Documents/Master's Projects/GitHub/PEAK-DRL-Tool/code/tests/test_reward_rebalance_platformer.py`:

```python
"""
Task 3 — verify the non-PBRS distance/movement term in the `simple` platformer
persona is capped so it cannot dominate cumulative reward, and that win + progress
dominate on a goal-reaching trajectory.

Personas are FACTORIES (train_platformer.py:_wrap_with_tracker). Call simple() to
get a fresh per-env reward closure with signature reward(obs, base, terminated, info).
Per-step components are exposed on info["reward_components"].
"""
import math
import pytest
from code.rewards.train_platformer import simple, adept


def _abs_share(reward_fn, trajectory):
    """Run a scripted [(terminated, info), ...] trajectory; return abs-share dict."""
    shares = {}
    for terminated, info in trajectory:
        reward_fn(None, None, terminated, info)
        for k, v in info["reward_components"].items():
            shares[k] = shares.get(k, 0.0) + abs(v)
    total = sum(shares.values()) or 1.0
    return {k: v / total for k, v in shares.items()}, shares


# Mean platformer episode length per case-study/analysis.md:24 ("Episode Length" 6212).
MEAN_EP_LEN = 6212
WIN_BONUS = 5.0  # train_platformer.py:320 (simple persona win)


def test_simple_movement_bounded_on_long_run_right_no_win():
    """
    WORST CASE: agent runs right with max progress for a full mean-length episode
    and never wins. The non-PBRS movement term must NOT dominate, and its
    cumulative magnitude must stay below the one-time win bonus so the agent is
    never paid more for 'go right' than for 'win'.
    """
    rfn = simple()
    gd = float(MEAN_EP_LEN * 5)  # start far; ensure positive progress every step
    traj = []
    for _ in range(MEAN_EP_LEN):
        gd = max(0.0, gd - 5.0)  # 5 units progress/step -> hits the movement clamp
        traj.append((False, {"goal_dist": gd, "x_position": 0.0, "lives": 3}))
    share, abs_sum = _abs_share(rfn, traj)

    # Movement must not dominate the (no-win) episode.
    assert share.get("movement", 0.0) < 0.50, share
    # Cumulative non-PBRS movement reward must stay below a single win.
    assert abs_sum.get("movement", 0.0) < WIN_BONUS, abs_sum


def test_simple_win_and_progress_dominate_on_goal_trajectory():
    """On a goal-reaching trajectory, win must dominate; movement stays minor."""
    rfn = simple()
    traj = []
    for gd in [2000, 1600, 1200, 800, 400, 100, 0]:
        won = gd == 0
        info = {"goal_dist": float(gd), "x_position": 0.0, "lives": 3}
        if won:
            info["won"] = True
        traj.append((won, info))
    share, _ = _abs_share(rfn, traj)

    assert share.get("win", 0.0) > 0.80, share          # win dominates
    assert share.get("movement", 0.0) < 0.05, share      # distance term minor


def test_adept_pbrs_potential_form_unchanged():
    """
    Guard: the dijkstra PBRS term keeps its potential form
    r_potential = (prev_d - GAMMA*curr_d) * SCALE  (train_platformer.py:227).
    With GAMMA=0.99, SCALE=0.3: stepping curr_d 0.8 -> 0.6 (prev=0.8) gives
    (0.8 - 0.99*0.6)*0.3 = (0.8 - 0.594)*0.3 = 0.0618.
    """
    rfn = adept()
    # step 1: anchor (prev=-1 sentinel -> potential 0)
    rfn(None, None, False, {"dijkstra_dist": 0.8, "x_position": 0.0, "lives": 3})
    info = {"dijkstra_dist": 0.6, "x_position": 0.0, "lives": 3}
    rfn(None, None, False, info)
    expected = (0.8 - 0.99 * 0.6) * 0.3
    assert info["reward_components"]["potential"] == pytest.approx(expected, abs=1e-9)
```

- [ ] **Step 2: Run the test, expect FAILURE on the first two cases.** Command:
  `.venv/bin/python -m pytest code/tests/test_reward_rebalance_platformer.py -v`
  Expected: `test_simple_movement_bounded_on_long_run_right_no_win` FAILS (current clamp +0.01/step gives movement share ≈ 94% and cumulative ≈ 62 ≫ 5.0); `test_adept_pbrs_potential_form_unchanged` PASSES (PBRS untouched). The goal-trajectory test may already pass because win is one-shot; that is fine — it locks in the desired end state.

- [ ] **Step 3: Apply the cap to the `simple` persona.** Edit `train_platformer.py` lines 296–302, replacing the movement block with named constants (down-weight 0.003→0.00005 and clamp ±0.01→±0.0005, sized so worst-case cumulative movement over 6212 steps ≈ 3.1 < win 5.0):

  Before (lines 296–302):
  ```python
      # ── Movement (clamped so it can't dominate) ───────────────────────────
      r_move = progress * 0.003


      if progress < 0:
          r_move *= 1.5              # soft backtrack penalty
      r_move = max(-0.01, min(0.01, r_move))   # CLAMP — prevents runaway
  ```

  After:
  ```python
      # ── Movement (sharply down-weighted + tightly clamped) ────────────────
      # TASK 3: the raw horizontal-progress bonus previously let an agent that
      # just runs right out-earn the win bonus (+0.01/step × ~6212 steps ≈ 62 ≫
      # win 5.0). Cap it so cumulative movement over a full episode stays well
      # below a single win, forcing win + dijkstra-progress (PBRS) to dominate.
      # NOTE: this is the NON-PBRS distance term. The dijkstra PBRS potential in
      # adept/speedrunner/completionist is intentionally left untouched.
      MOVE_SCALE = 0.00005   # was 0.003
      MOVE_CLAMP = 0.0005    # was 0.01 — caps cumulative to ~3.1 over 6212 steps
      r_move = progress * MOVE_SCALE
      if progress < 0:
          r_move *= 1.5              # soft backtrack penalty (kept)
      r_move = max(-MOVE_CLAMP, min(MOVE_CLAMP, r_move))   # CLAMP — prevents runaway
  ```

  Leave the stall penalty block (lines 304–306) and everything else in `simple` unchanged. Do NOT touch `adept`, `speedrunner`, `completionist`, or `enemy_hunter` — their distance influence is PBRS-only (verified win-dominated for `adept`).

- [ ] **Step 4: Re-run the focused test, expect PASS.** Command:
  `.venv/bin/python -m pytest code/tests/test_reward_rebalance_platformer.py -v`
  Expected: all 3 tests PASS. The bounded-movement test now sees movement share well under 50% and cumulative movement ≈ 3.1 < 5.0; the PBRS guard still passes (form unchanged).

- [ ] **Step 5: Training run to OBSERVE the learning effect (this is NOT covered by the unit test).** The unit test proves the per-step cap and share math; it does NOT prove the agent learns to win. Run a short training run on the `simple`/`default` persona and record win-rate trend:
  `.venv/bin/python code/scripts/train.py --persona simple` (use the team's standard short-run config; if unsure, ask the user for the canonical training invocation and step budget).
  Record: (a) win-rate learning curve from the analyzer (`.venv/bin/python code/scripts/agent_analyzer.py` over the run's CSV) — expect win-rate to stop declining vs the 28%→0% collapse in `case-study/analysis.md:44–49`; (b) confirm in the produced CSV that the `movement` reward component's cumulative magnitude per episode is now < the `win` component when the agent wins. Be explicit in the writeup that this step is observational, not a pass/fail unit assertion.

- [ ] **Step 6: Checkpoint (DO NOT COMMIT).** Run the full suite and pause for review:
  `.venv/bin/python -m pytest code/tests/ -q`
  Expected: full suite green (the new file plus existing tests). Do NOT `git add`/`git commit` — pause and hand back for the user to review the reward change before any training run is trusted.

---

### Task 4: VecNormalize norm_reward=True (train only) + clip_reward > win bonus

**Goal:** Turn ON reward normalization for the TRAIN `VecNormalize` wrapper with a `clip_reward` strictly greater than the win bonus (5.0) so the terminal win signal is not clipped away, while keeping the EVAL wrapper `norm_reward=False` for interpretable evaluation. A minimal refactor extracts the kwargs construction into a pure, testable helper.

**Files:**
- Modify: `code/scripts/train.py` (lines 986–990 — kwargs construction; line 1127 — train wrapper; line 1139 — eval wrapper)
- Create: `code/tests/test_vecnorm_kwargs.py`

**Context (actually read):**

Today the kwargs are built ONCE and reused for both train and eval. `code/scripts/train.py:986-990`:
```python
    vecnorm_kwargs = dict(norm_obs=True, norm_reward=False, clip_obs=10.0)
    if uses_dict_obs:
        norm_keys = [k for k, v in obs_space.spaces.items() if isinstance(v, spaces.Box)]
        if norm_keys:
            vecnorm_kwargs["norm_obs_keys"] = norm_keys
```
Train wrapper, `code/scripts/train.py:1127`:
```python
            env = VecNormalize(raw_env, **vecnorm_kwargs)
```
Eval wrapper, `code/scripts/train.py:1139` (note the existing post-construction force at 1147):
```python
            eval_env = VecNormalize(eval_raw_env, **vecnorm_kwargs)
            ...
            eval_env.norm_reward = False
```
Win bonus is 5.0 in the default personas (`code/rewards/train_platformer.py:253,320,487`, `code/rewards/train_meatboy.py:19`); `balanced_win` uses 8.0 (`train_platformer.py:428`), megaman 7.0 (`train_megaman.py:263`). We pick TRAIN `clip_reward = 10.0` (the SB3 default), which is `> 5.0` and clears every persona bonus.

- [ ] **Step 1: Write the failing unit test.** Create `code/tests/test_vecnorm_kwargs.py`:
  ```python
  """Unit tests for code.scripts.train._build_vecnorm_kwargs.

  Verifies the TRAIN VecNormalize wrapper normalizes reward with a clip_reward
  strictly above the 5.0 win bonus, while the EVAL wrapper leaves reward
  un-normalized for interpretable evaluation. Also checks that the Dict-obs
  norm_obs_keys handling is preserved for both train and eval.
  """
  import gymnasium.spaces as spaces
  import numpy as np
  import pytest

  from code.scripts.train import _build_vecnorm_kwargs

  WIN_BONUS = 5.0


  def _box_dict_space():
      return spaces.Dict({
          "image": spaces.Box(low=0, high=1, shape=(4, 8, 8), dtype=np.float32),
          "vec":   spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32),
          "flag":  spaces.Discrete(2),  # non-Box: must be excluded from norm_obs_keys
      })


  def test_train_normalizes_reward_and_clip_above_win_bonus():
      kw = _build_vecnorm_kwargs(uses_dict_obs=False, obs_space=None, training=True)
      assert kw["norm_obs"] is True
      assert kw["norm_reward"] is True, "TRAIN must normalize reward"
      assert kw["clip_obs"] == 10.0
      assert kw["clip_reward"] > WIN_BONUS, (
          f"clip_reward={kw['clip_reward']} must exceed win bonus {WIN_BONUS} "
          "so the win signal is not clipped away"
      )


  def test_eval_does_not_normalize_reward():
      kw = _build_vecnorm_kwargs(uses_dict_obs=False, obs_space=None, training=False)
      assert kw["norm_obs"] is True
      assert kw["norm_reward"] is False, "EVAL must leave reward un-normalized"
      assert kw["clip_obs"] == 10.0


  def test_dict_obs_keys_preserved_for_train_and_eval():
      space = _box_dict_space()
      for training in (True, False):
          kw = _build_vecnorm_kwargs(uses_dict_obs=True, obs_space=space, training=training)
          assert kw["norm_obs_keys"] == ["image", "vec"], (
              "only Box subspaces should be normalized; Discrete must be excluded"
          )


  def test_non_dict_obs_has_no_norm_obs_keys():
      kw = _build_vecnorm_kwargs(uses_dict_obs=False, obs_space=None, training=True)
      assert "norm_obs_keys" not in kw
  ```

- [ ] **Step 2: Run the test — expect a collection/attribute failure.** Command:
  ```
  .venv/bin/python -m pytest code/tests/test_vecnorm_kwargs.py -v
  ```
  Expected: ImportError / `cannot import name '_build_vecnorm_kwargs'` (the helper does not exist yet).

- [ ] **Step 3: Add the helper to `code/scripts/train.py`.** Place it at module scope, just ABOVE the existing block at line 986 (e.g. directly after the existing `_resolve_device` helper region, or immediately before line 980 where `probe_persona` is defined). Insert:
  ```python
  # TRAIN clip_reward must exceed the win bonus (5.0) so the terminal win
  # signal survives normalization-time clipping. 10.0 is the SB3 default and
  # clears every persona win bonus (5.0 default, 7.0 megaman, 8.0 balanced_win).
  _VECNORM_TRAIN_CLIP_REWARD = 10.0


  def _build_vecnorm_kwargs(uses_dict_obs, obs_space, *, training):
      """Construct VecNormalize kwargs.

      training=True  -> norm_reward=True  (+ clip_reward=10.0 > win bonus 5.0)
      training=False -> norm_reward=False (interpretable eval rewards)
      """
      kwargs = dict(
          norm_obs=True,
          clip_obs=10.0,
          norm_reward=bool(training),
          clip_reward=_VECNORM_TRAIN_CLIP_REWARD,
      )
      if uses_dict_obs:
          norm_keys = [k for k, v in obs_space.spaces.items()
                       if isinstance(v, spaces.Box)]
          if norm_keys:
              kwargs["norm_obs_keys"] = norm_keys
      return kwargs
  ```
  (`spaces` is already imported in train.py — it is used at line 984 `isinstance(obs_space, spaces.Dict)`.)

- [ ] **Step 4: Replace the single shared dict (lines 986–990) with two named dicts.** Change:
  ```python
      vecnorm_kwargs = dict(norm_obs=True, norm_reward=False, clip_obs=10.0)
      if uses_dict_obs:
          norm_keys = [k for k, v in obs_space.spaces.items() if isinstance(v, spaces.Box)]
          if norm_keys:
              vecnorm_kwargs["norm_obs_keys"] = norm_keys
  ```
  to:
  ```python
      train_vecnorm_kwargs = _build_vecnorm_kwargs(uses_dict_obs, obs_space, training=True)
      eval_vecnorm_kwargs  = _build_vecnorm_kwargs(uses_dict_obs, obs_space, training=False)
  ```

- [ ] **Step 5: Point the train wrapper at the train dict.** At `code/scripts/train.py:1127` change:
  ```python
              env = VecNormalize(raw_env, **vecnorm_kwargs)
  ```
  to:
  ```python
              env = VecNormalize(raw_env, **train_vecnorm_kwargs)
  ```

- [ ] **Step 6: Point the eval wrapper at the eval dict.** At `code/scripts/train.py:1139` change:
  ```python
              eval_env = VecNormalize(eval_raw_env, **vecnorm_kwargs)
  ```
  to:
  ```python
              eval_env = VecNormalize(eval_raw_env, **eval_vecnorm_kwargs)
  ```
  Leave the existing `eval_env.norm_reward = False` at line 1147 in place (now redundant but documents intent and is harmless).

- [ ] **Step 7: Confirm no other reference to the old name remains.** Run:
  ```
  grep -n "vecnorm_kwargs" code/scripts/train.py
  ```
  Expected: only `train_vecnorm_kwargs` (lines ~986, ~1127) and `eval_vecnorm_kwargs` (lines ~987, ~1139) appear; the bare name `vecnorm_kwargs` must NOT appear standalone anywhere.

- [ ] **Step 8: Run the new test — expect green.** Command:
  ```
  .venv/bin/python -m pytest code/tests/test_vecnorm_kwargs.py -v
  ```
  Expected: 4 passed.

- [ ] **Step 9: (Optional but recommended) Smoke-check the wiring in a real run.** This is the part a unit test cannot prove (the learning effect). Run a very short training session and confirm the train wrapper reports reward normalization on. Use the project's normal launch path; e.g.:
  ```
  .venv/bin/python code/scripts/train.py n_envs=1 <minimal model/persona/skill overrides for ~1-2k steps>
  ```
  Record: (a) training completes without error, (b) in a Python REPL after construction `env.norm_reward is True` and `env.clip_reward == 10.0`, while `eval_env.norm_reward is False`. Note honestly: this confirms wiring only, not that win-rate/return improves — that requires a full comparative run logged via the existing CSV/TensorBoard logging.

- [ ] **Step 10: Checkpoint (DO NOT COMMIT).** Run the full suite and pause for review:
  ```
  .venv/bin/python -m pytest code/tests/ -q
  ```
  Expected: all tests pass (the 4 new ones plus the pre-existing `test_profiling_callback.py` and `test_resolve_device.py`). Do not `git add`/`git commit`.

---

### Task 5: PPO horizon/value retune + POTENTIAL_GAMMA sync

> **PHASE 1 — apply AFTER Task 4 (reward normalization) is merged.** The gamma increase below lengthens the return horizon and will destabilize training if rewards are not yet normalized. Do not start this task until Task 4's normalization is in place.

**Files:**
- **Modify:** `code/conf/algo/ppo.yaml` (lines 7, 9, 10, 13)
- **Modify:** `code/rewards/train_platformer.py` (add module constant after line 5; replace local `POTENTIAL_GAMMA = 0.99` at lines 220, 412, 466)
- **Test (create):** `code/tests/test_gamma_sync.py`

---

**Context — exact current code (read, do not trust memory):**

`code/conf/algo/ppo.yaml` currently (lines 5-14):
```yaml
n_steps: 2048          # was 4096 — faster update cycle, better for noisy rewards
batch_size: 512        # was 256 — bigger batches use CPU vector units better
n_epochs: 4            # was 8 — fewer SGD passes; profile showed train_s ≈ 72% of wall time
learning_rate: 0.0003  # keep for now, add schedule after reward fix
gae_lambda: 0.95
gamma: 0.99
clip_range: 0.2
ent_coef: 0.01         # was 0.02 — reduce once reward signal is stronger
vf_coef: 0.25          # was implicit 0.5 — reduce while value loss is high
```

`code/rewards/train_platformer.py` defines `POTENTIAL_GAMMA = 0.99` as a LOCAL variable in three personas:
- line 220 (`adept`): `    POTENTIAL_GAMMA = 0.99`
- line 412 (`speedrunner`): `    POTENTIAL_GAMMA = 0.99`
- line 466 (`completionist`): `    POTENTIAL_GAMMA = 0.99`

Each is used identically, e.g. line 227: `        r_potential = (prev_d - POTENTIAL_GAMMA * curr_d) * POTENTIAL_SCALE`. The `simple` and `enemy_hunter` personas have **no** PBRS term and are left unchanged. The module currently has no top-level `POTENTIAL_GAMMA`.

---

- [ ] **Step 1: Write the failing drift-guard test.** Create `code/tests/test_gamma_sync.py`. It loads gamma from the YAML and the personas' `POTENTIAL_GAMMA`, and asserts equality. It is RED now because YAML gamma is 0.99 but we'll require the *new* value; more importantly it locks the two together forever.

```python
"""Drift-guard: PPO discount (ppo.yaml gamma) must equal the PBRS shaping
discount (POTENTIAL_GAMMA in train_platformer.py).

Potential-based reward shaping is policy-invariant ONLY when the shaping
gamma equals the RL gamma (Ng, Harada & Russell 1999). If the two ever
drift apart, the shaping silently biases the learned policy. This test
fails loudly if anyone edits one without the other.
"""
from pathlib import Path

import yaml

# Repo root = three parents up from this file (code/tests/<file> -> repo root).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PPO_YAML = _REPO_ROOT / "code" / "conf" / "algo" / "ppo.yaml"


def _load_ppo_gamma() -> float:
    """Read the raw `gamma` scalar from code/conf/algo/ppo.yaml."""
    with open(_PPO_YAML, "r") as f:
        cfg = yaml.safe_load(f)
    return float(cfg["gamma"])


def test_ppo_gamma_is_the_retuned_value():
    # Phase-1 retune target. Update this AND ppo.yaml together if it ever changes.
    assert _load_ppo_gamma() == 0.997


def test_ppo_yaml_other_hyperparams_retuned():
    with open(_PPO_YAML, "r") as f:
        cfg = yaml.safe_load(f)
    assert float(cfg["gae_lambda"]) == 0.97
    assert float(cfg["vf_coef"]) == 0.5
    assert int(cfg["n_epochs"]) == 8


def test_potential_gamma_constant_matches_ppo_gamma():
    """The module-level shaping discount must equal the PPO gamma."""
    from code.rewards.train_platformer import POTENTIAL_GAMMA
    assert POTENTIAL_GAMMA == _load_ppo_gamma()


def test_no_persona_redefines_potential_gamma_locally():
    """Guard against re-introducing a per-persona local POTENTIAL_GAMMA that
    could drift from the module constant. There must be exactly ONE assignment
    `POTENTIAL_GAMMA =` in the source (the module-level one)."""
    src = (_REPO_ROOT / "code" / "rewards" / "train_platformer.py").read_text()
    assignments = [
        ln for ln in src.splitlines()
        if ln.strip().startswith("POTENTIAL_GAMMA") and "=" in ln and "==" not in ln
    ]
    assert len(assignments) == 1, (
        f"Expected exactly one POTENTIAL_GAMMA assignment (module-level), "
        f"found {len(assignments)}: {assignments}"
    )
```

- [ ] **Step 2: Run the test — confirm it is RED.**
  ```
  .venv/bin/python -m pytest code/tests/test_gamma_sync.py -v
  ```
  Expected: `test_ppo_gamma_is_the_retuned_value` FAILS (gamma is 0.99, expected 0.997), `test_ppo_yaml_other_hyperparams_retuned` FAILS, `test_potential_gamma_constant_matches_ppo_gamma` FAILS with `ImportError` (no module-level `POTENTIAL_GAMMA` yet), `test_no_persona_redefines_potential_gamma_locally` FAILS (3 local assignments found).

- [ ] **Step 3: Apply the ppo.yaml diff.** Edit `code/conf/algo/ppo.yaml` lines 7, 9, 10, 13 (leave `ent_coef` on line 12 unchanged):

```diff
-n_epochs: 4            # was 8 — fewer SGD passes; profile showed train_s ≈ 72% of wall time
+n_epochs: 8            # raised 4->8: more value-fit passes to support longer gamma=0.997 horizon (Task 5)
 learning_rate: 0.0003  # keep for now, add schedule after reward fix
-gae_lambda: 0.95
-gamma: 0.99
+gae_lambda: 0.97       # raised 0.95->0.97: lower-bias GAE for the longer horizon (Task 5)
+gamma: 0.997           # raised 0.99->0.997: ~333-step effective horizon. MUST equal POTENTIAL_GAMMA in train_platformer.py
 clip_range: 0.2
 ent_coef: 0.01         # was 0.02 — reduce once reward signal is stronger
-vf_coef: 0.25          # was implicit 0.5 — reduce while value loss is high
+vf_coef: 0.5           # raised 0.25->0.5: stronger value fitting for longer-horizon returns (Task 5)
```

- [ ] **Step 4: Hoist `POTENTIAL_GAMMA` to a single module-level constant and point all three personas at it.** This makes drift structurally impossible (one source of truth).

  4a. Add the constant near the top of `code/rewards/train_platformer.py`, immediately after the `Info` alias (current line 5 is `Info = Dict[str, Any]`):
  ```python
  Info = Dict[str, Any]

  # PBRS shaping discount. MUST equal PPO `gamma` in code/conf/algo/ppo.yaml
  # (potential-based shaping is policy-invariant only when these are equal —
  # Ng, Harada & Russell 1999). Guarded by code/tests/test_gamma_sync.py.
  POTENTIAL_GAMMA = 0.997
  ```

  4b. Delete the three local re-definitions. In `adept` (line 220), `speedrunner` (line 412), and `completionist` (line 466), remove the line `    POTENTIAL_GAMMA = 0.99`. The personas already *reference* `POTENTIAL_GAMMA` (e.g. line 227); with the local removed, they resolve the module-level constant via Python's normal name lookup — no other edit to the usage lines is needed. Leave each persona's local `POTENTIAL_SCALE` line untouched.

  After edit, the `adept` block (was lines 219-227) reads:
  ```python
      # ── Potential-based shaping (THE FIX: scale 3.0 → 0.3) ───────────────
      POTENTIAL_SCALE = 0.3     # was 3.0 — dominated everything

      r_potential = 0.0
      curr_d = float(info.get("dijkstra_dist", -1.0))
      prev_d = float(info.get("dijkstra_dist_prev", -1.0))
      if dijkstra_valid and prev_d >= 0.0:
          r_potential = (prev_d - POTENTIAL_GAMMA * curr_d) * POTENTIAL_SCALE
  ```
  Apply the same removal (drop only the `POTENTIAL_GAMMA = 0.99` line) in `speedrunner` (~line 412, keeps `POTENTIAL_SCALE = 0.3`) and `completionist` (~line 466, keeps `POTENTIAL_SCALE = 0.3`).

- [ ] **Step 5: Run the test — confirm it is GREEN.**
  ```
  .venv/bin/python -m pytest code/tests/test_gamma_sync.py -v
  ```
  Expected: all 4 tests PASS. If `test_no_persona_redefines_potential_gamma_locally` still fails, you missed deleting one of the three local assignments.

  **What this proves / does not prove:** PASS proves the YAML gamma, the other retuned hyperparameters, and the module `POTENTIAL_GAMMA` are wired to the intended values and locked together. It does NOT prove the new hyperparameters improve learning. To observe the learning effect you must do a training run (run from repo root, persona must be one with shaping):
  ```
  .venv/bin/python -m code.scripts.train algo=ppo reward=platformer_adept
  ```
  Record from TensorBoard over the first ~200k steps: `train/value_loss` (should stay bounded, not diverge — this is the main risk of the higher gamma), `train/explained_variance` (should trend up, ideally > 0), and `rollout/ep_rew_mean`. If `value_loss` diverges, confirm Task 4's reward normalization is actually active — that is the precondition for this task.

- [ ] **Step 6: Checkpoint (DO NOT COMMIT).** Run the full suite and pause for review:
  ```
  .venv/bin/python -m pytest code/tests/ -q
  ```
  Expected: all tests pass (the existing `test_resolve_device.py` and `test_profiling_callback.py` are unaffected; the new `test_gamma_sync.py` is green). Do NOT `git add` or `git commit` — pause here for the user to review.

---

### Task 6: Register the stage_1..N difficulty curriculum in game_config.yaml (dedupe + monotonic order)

**Goal:** Add the existing `platformer/stage_*.txt` maps to the `levels:` block of `game_config.yaml` in a difficulty-monotonic order (easy stages first, then the original `Mario1-1`/`Mario1-2`), deduping the byte-identical pair and flagging the vertical-climb discontinuities. Order matters because `PlatformerCore.level_order = self.config_manager.get_level_order()` (code/games/platformer_core.py:409) drives the curriculum index directly.

**Files:**
- Modify: `code/games/game_config.yaml` (top-level `levels:` block, lines 56-70 — currently only `Mario1-1`/`Mario1-2`)
- Test (create): `code/tests/test_platformer_curriculum_order.py`
- Read-only context: `code/games/modules/System/config_manager.py:107-110` (`get_level_order` returns `list(self.yaml_data['levels'].keys())` — insertion order); `code/games/levels/platformer/stage_*.txt` (the 14 map files)

**Grounding — what the current `levels:` block looks like (game_config.yaml:56-70):**
```yaml
levels:
  Mario1-1:
    file: platformer/world1_1.txt
    time_limit: 300
    background_color:
    - 0
    - 0
    - 0
  Mario1-2:
    file: platformer/world1_2.txt
    time_limit: 300
    background_color:
    - 0
    - 0
    - 0
```

**Grounding — tile census I measured (counts of `E`=enemy, `O`=pit-marker, `^`=spike; `rows`=line count). This is the evidence for the ordering:**

| stage | rows | E | O | ^ | character |
|---|---|---|---|---|---|
| stage_1 | 18 | 0 | 0 | 0 | flat ground, coins only — easiest |
| stage_3 | 18 | 0 | 0 | 0 | flat + small stepping platforms, no hazards |
| stage_2 | 18 | 3 | 0 | 0 | introduces enemies, still no pits |
| stage_4 | 18 | 0 | 12 | 2 | introduces pits (`O`) + spikes |
| stage_5 | 19 | 0 | 26 | 0 | wide pit + floating `=` platforms |
| stage_6 | 19 | 7 | 0 | 0 | multi-tier `=` platforms + enemies |
| stage_11 | 18 | 3 | 0 | 0 | multi-tier maze with enemies |
| stage_13 | 14 | 0 | 2 | 2 | long horizontal gap-jump gauntlet |
| stage_7 | 34 | 11 | 0 | 1 | tall, dense enemies + coins |
| stage_8 | 34 | 12 | 17 | 0 | tall, enemies + many pits |
| stage_14 | 35 | 3 | 0 | 0 | tall vertical-climb (zig-zag platforms) |
| stage_9 | 40 | 0 | 0 | 0 | **vertical-climb discontinuity** (full border) |
| stage_10 | 41 | 0 | 0 | 0 | **vertical-climb discontinuity** (floor-only `#`, no side walls) |
| stage_12 | 18 | 3 | 0 | 0 | **DUPLICATE of stage_11** (`md5 5cbb0187…` identical) |

**Verified facts (run before drafting):**
- `md5 -r stage_11.txt stage_12.txt` → identical hash `5cbb0187162b4e1596fc6954be64dc92`; `diff stage_11.txt stage_12.txt` → `IDENTICAL`. So **stage_12 is dropped** (omitted from `level_order`).
- `diff stage_7.txt stage_8.txt` → DIFFERENT (they share a byte count of 2583 by coincidence; both are kept).
- stage_9 (40 rows) and stage_10 (41 rows) are vertical-climb maps — a structural discontinuity vs the ~17-19-row horizontal maps. stage_10's only `#` is the bottom floor row (no side border walls). Both still have exactly one `P` (spawn) and one `G` (goal). They are placed at the END of the stage block (immediately before the Marios) and flagged with a YAML comment, since difficulty here is "different skill," not strictly "more hazards."
- `file:` paths are resolved by `LevelLoader` relative to `code/games/levels/` (existing entries use `platformer/world1_1.txt`; the maps live at `code/games/levels/platformer/stage_N.txt`), so each new entry uses `file: platformer/stage_N.txt`.
- `yaml.safe_load` preserves insertion order on this interpreter (Py3.11) — confirmed.

---

- [ ] **Step 1: Write the failing test** — create `code/tests/test_platformer_curriculum_order.py`:
```python
"""Task 6: platformer difficulty curriculum is registered in game_config.yaml.

Asserts the level_order from ConfigManager (a) starts with the easy stages in
monotonic order, (b) ends with the original Mario1-1/Mario1-2, (c) has no
duplicate level id, and (d) every referenced .txt map file exists on disk.
"""
import os
import pytest

from code.games.modules.System.config_manager import ConfigManager

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LEVELS_DIR = os.path.join(REPO_ROOT, "code", "games", "levels")

# Canonical difficulty-monotonic order (easy -> hard), vertical-climb
# discontinuities last, then the original Marios. stage_12 is intentionally
# absent (byte-identical duplicate of stage_11).
EXPECTED_ORDER = [
    "stage_1", "stage_3", "stage_2", "stage_4", "stage_5", "stage_6",
    "stage_11", "stage_13", "stage_7", "stage_8", "stage_14",
    "stage_9", "stage_10",
    "Mario1-1", "Mario1-2",
]


@pytest.fixture(scope="module")
def cm():
    return ConfigManager("game_config.yaml")


def test_level_order_matches_expected(cm):
    assert cm.get_level_order() == EXPECTED_ORDER


def test_no_duplicate_levels(cm):
    order = cm.get_level_order()
    assert len(order) == len(set(order)), f"duplicate level ids in: {order}"


def test_stage_12_deduped(cm):
    # stage_12 is byte-identical to stage_11 -> must NOT be registered.
    assert "stage_12" not in cm.get_level_order()


def test_starts_with_easy_stages_and_ends_with_marios(cm):
    order = cm.get_level_order()
    assert order[0] == "stage_1"          # flat, no hazards
    assert order[1] == "stage_3"          # flat + steps, no hazards
    assert order[2] == "stage_2"          # first enemies
    assert order[-2:] == ["Mario1-1", "Mario1-2"]


def test_every_level_file_exists(cm):
    levels = cm.yaml_data.get("levels") or {}
    for level_id, cfg in levels.items():
        rel = cfg.get("file")
        assert rel, f"{level_id} missing 'file'"
        full = os.path.join(LEVELS_DIR, rel)
        assert os.path.isfile(full), f"{level_id} -> missing map file {full}"
```

- [ ] **Step 2: Confirm the test fails for the right reason** — run:
```
.venv/bin/python -m pytest code/tests/test_platformer_curriculum_order.py -v
```
Expected before the edit: `test_level_order_matches_expected`, `test_stage_12_deduped` (passes trivially, but) and the easy-stage assertions FAIL because `get_level_order()` currently returns only `['Mario1-1', 'Mario1-2']`.

- [ ] **Step 3: Make the change** — in `code/games/game_config.yaml`, replace the top-level `levels:` block (lines 56-70) so the new stages are inserted BEFORE the two Marios. Replace exactly:
```yaml
levels:
  Mario1-1:
    file: platformer/world1_1.txt
    time_limit: 300
    background_color:
    - 0
    - 0
    - 0
  Mario1-2:
    file: platformer/world1_2.txt
    time_limit: 300
    background_color:
    - 0
    - 0
    - 0
```
with:
```yaml
levels:
  # --- Difficulty curriculum (easy -> hard). Order drives PlatformerCore.level_order. ---
  # stage_12 is byte-identical to stage_11 (md5 5cbb0187…) and is intentionally omitted.
  stage_1:   # flat ground, coins only — easiest
    file: platformer/stage_1.txt
    time_limit: 300
    background_color:
    - 0
    - 0
    - 0
  stage_3:   # flat + small stepping platforms, no hazards
    file: platformer/stage_3.txt
    time_limit: 300
    background_color:
    - 0
    - 0
    - 0
  stage_2:   # introduces enemies (E), no pits
    file: platformer/stage_2.txt
    time_limit: 300
    background_color:
    - 0
    - 0
    - 0
  stage_4:   # introduces pits (O) + spikes (^)
    file: platformer/stage_4.txt
    time_limit: 300
    background_color:
    - 0
    - 0
    - 0
  stage_5:   # wide pit + floating platforms
    file: platformer/stage_5.txt
    time_limit: 300
    background_color:
    - 0
    - 0
    - 0
  stage_6:   # multi-tier platforms + enemies
    file: platformer/stage_6.txt
    time_limit: 300
    background_color:
    - 0
    - 0
    - 0
  stage_11:  # multi-tier maze with enemies (stage_12 duplicate dropped)
    file: platformer/stage_11.txt
    time_limit: 300
    background_color:
    - 0
    - 0
    - 0
  stage_13:  # long horizontal gap-jump gauntlet
    file: platformer/stage_13.txt
    time_limit: 300
    background_color:
    - 0
    - 0
    - 0
  stage_7:   # tall, dense enemies + coins
    file: platformer/stage_7.txt
    time_limit: 300
    background_color:
    - 0
    - 0
    - 0
  stage_8:   # tall, enemies + many pits
    file: platformer/stage_8.txt
    time_limit: 300
    background_color:
    - 0
    - 0
    - 0
  stage_14:  # tall vertical-climb (zig-zag platforms)
    file: platformer/stage_14.txt
    time_limit: 300
    background_color:
    - 0
    - 0
    - 0
  # DISCONTINUITY: stage_9 / stage_10 are vertical-climb maps (~40 rows) — a
  # different skill than the horizontal maps above; stage_10 has no side border
  # walls (floor-only). Kept last in the stage block, before the Marios.
  stage_9:
    file: platformer/stage_9.txt
    time_limit: 300
    background_color:
    - 0
    - 0
    - 0
  stage_10:
    file: platformer/stage_10.txt
    time_limit: 300
    background_color:
    - 0
    - 0
    - 0
  Mario1-1:
    file: platformer/world1_1.txt
    time_limit: 300
    background_color:
    - 0
    - 0
    - 0
  Mario1-2:
    file: platformer/world1_2.txt
    time_limit: 300
    background_color:
    - 0
    - 0
    - 0
```
Note: only the TOP-LEVEL `levels:` block is edited (the platformer). Do NOT touch the `megaman:` or `sonic:` sub-blocks, which have their own `levels:` keys further down the file.

- [ ] **Step 4: Verify the targeted test passes** — run:
```
.venv/bin/python -m pytest code/tests/test_platformer_curriculum_order.py -v
```
Expected: all 5 tests PASS. Then sanity-check the live order:
```
.venv/bin/python -c "from code.games.modules.System.config_manager import ConfigManager; print(ConfigManager('game_config.yaml').get_level_order())"
```
Expected stdout:
```
['stage_1', 'stage_3', 'stage_2', 'stage_4', 'stage_5', 'stage_6', 'stage_11', 'stage_13', 'stage_7', 'stage_8', 'stage_14', 'stage_9', 'stage_10', 'Mario1-1', 'Mario1-2']
```

- [ ] **Step 5: Smoke-test that PlatformerCore loads the new default world** — the first entry is now `stage_1` and `_default_world = self.level_order[0]` (platformer_core.py:411). Confirm the env still constructs (this loads `stage_1.txt` via LevelLoader):
```
.venv/bin/python -c "from code.games.platformer_core import PlatformerCore; e=PlatformerCore(); e.reset(); print('OK default world =', e.level_order[0])"
```
Expected: prints `OK default world = stage_1` with no traceback. (What this proves: the new map files parse and the curriculum head loads. It does NOT prove the difficulty ordering improves learning — that requires a training run, out of scope for this config task.)

- [ ] **Step 6: Checkpoint (DO NOT COMMIT)** — run the full suite and pause for review:
```
.venv/bin/python -m pytest code/tests/ -q
```
Expected: the new file's tests pass and no previously-green test regresses. Note: any existing test that hard-codes the platformer default world as `Mario1-1` (search with `grep -rn "Mario1-1" code/tests/`) will now fail because the default head is `stage_1`; if found, flag it for the reviewer rather than silently changing it. Do not `git add`/`git commit`.

---

### Task 7: Fix the curriculum double-pop bug + wire curriculum_win_rate
**Files:**
- Modify: `code/games/platformer_core.py` (constructor lines ~432-441 and ~480-483; `reset()` body lines ~781-822)
- Test (create): `code/tests/test_curriculum_thresholds.py`

**Background (verified by reading + running the code).** Two bugs:

1. **Double-pop.** The constructor pops `advance_threshold`/`fallback_threshold` TWICE. First into the DEAD "mastery-gated" fields (lines 432-435), then again into the ACTIVE batch curriculum (lines 482-483). Because `dict.pop` removes the key on the first call, the second pop never sees the YAML value and silently uses its `0.30`/`0.20` defaults. The active curriculum decision (`_evaluate_curriculum_batch`, lines 832 `effective_advance = self._batch_advance_threshold` and 843 `win_rate <= self._batch_fallback_threshold`) therefore IGNORES the YAML knobs. `self._advance_threshold` / `self._fallback_threshold` (the mastery fields) are read NOWHERE — confirmed by grep: only lines 433/434 reference them.

   Runtime proof:
   ```
   $ PlatformerCore(advance_threshold=0.55, fallback_threshold=0.15)
   batch_advance: 0.3      # WRONG — should be 0.55
   batch_fallback: 0.2     # WRONG — should be 0.15
   mastery advance: 0.55   # dead field swallowed the YAML value
   ```

2. **`_level_window` never written.** The per-level deque dict is built at line 438 but grep shows NO `.append` anywhere, so `_curriculum_win_rate()` (line 1700, reading `self._level_window.get(self.world, deque())`) always hits the empty-window branch and returns the stale `-1.0` that gets logged into `info["curriculum_win_rate"]` (lines 1732, 1780).

- [ ] **Step 1: Write the failing test.** Create `code/tests/test_curriculum_thresholds.py`. It proves (a) YAML thresholds reach the active batch curriculum, and (b) `curriculum_win_rate` reflects recorded outcomes.

  ```python
  import os
  os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

  import pytest
  from code.games.platformer_core import PlatformerCore


  @pytest.fixture
  def core():
      g = PlatformerCore(
          render_mode="none",
          advance_threshold=0.55,
          fallback_threshold=0.15,
          curriculum_window=4,
      )
      yield g


  def test_yaml_thresholds_reach_active_batch_curriculum(core):
      # The ACTIVE curriculum is the batch curriculum (_evaluate_curriculum_batch
      # reads _batch_advance_threshold / _batch_fallback_threshold). Before the
      # fix these were stuck at the 0.30 / 0.20 defaults because the mastery path
      # popped the kwargs first.
      assert core._batch_advance_threshold == pytest.approx(0.55)
      assert core._batch_fallback_threshold == pytest.approx(0.15)
      # Dead mastery fields must be gone (no longer swallow the kwargs).
      assert not hasattr(core, "_advance_threshold")
      assert not hasattr(core, "_fallback_threshold")


  def test_curriculum_win_rate_reflects_recorded_outcomes(core):
      lvl = core.level_order[0]
      core.world = lvl
      # No data yet -> sentinel.
      assert core._curriculum_win_rate() == -1.0

      # Drive the real reset() episode-boundary code path. _episode_won_current
      # is the per-episode win flag set by complete_level(); reset() consumes it.
      # window=4, feed W, L, W, W  -> win rate 3/4 = 0.75
      outcomes = [True, False, True, True]
      for won in outcomes:
          core._episode_won_current = won
          core.reset()  # records into _level_window, then loads a fresh level

      # After 4 episodes the deque (maxlen=4) holds exactly those outcomes.
      assert core._curriculum_win_rate() == pytest.approx(0.75)
      # And it's surfaced in info().
      assert core._info()["curriculum_win_rate"] == pytest.approx(0.75)
  ```

- [ ] **Step 2: Run the test, watch it FAIL.**
  `.venv/bin/python -m pytest code/tests/test_curriculum_thresholds.py -v`
  Expected before the fix: `test_yaml_thresholds_reach_active_batch_curriculum` FAILS (`_batch_advance_threshold` is `0.3`, and `_advance_threshold` still exists); `test_curriculum_win_rate_reflects_recorded_outcomes` FAILS (returns `-1.0`, never written).

- [ ] **Step 3: Remove the dead mastery pops.** In `code/games/platformer_core.py`, the current block at lines 432-441 reads:

  ```python
          self._curriculum_window_size  =   int(kwargs.pop("curriculum_window" ,   5))
          self._advance_threshold       = float(kwargs.pop("advance_threshold" , 0.6))
          self._fallback_threshold      = float(kwargs.pop("fallback_threshold", 0.2))
          self._explore_prob            = float(kwargs.pop("explore_prob",       0.10))

          # Per-level outcome windows — persist across episodes, never reset.
          self._level_window = {
              lvl: deque(maxlen=self._curriculum_window_size)
              for lvl in self.level_order
          }
  ```

  Replace with (drop the two dead pops; keep `curriculum_window` and `explore_prob`; keep building `_level_window`):

  ```python
          self._curriculum_window_size  =   int(kwargs.pop("curriculum_window" ,   5))
          # NOTE: advance_threshold / fallback_threshold are intentionally NOT
          # popped here. They are consumed by the ACTIVE batch curriculum below
          # (self._batch_advance_threshold / self._batch_fallback_threshold).
          # Popping them here was a dead "mastery-gated" path that silently
          # starved the batch curriculum of its YAML-configured thresholds.
          self._explore_prob            = float(kwargs.pop("explore_prob",       0.10))

          # Per-level outcome windows — diagnostic sliding window feeding
          # curriculum_win_rate() in _info(). Persist across episodes, never reset.
          self._level_window = {
              lvl: deque(maxlen=self._curriculum_window_size)
              for lvl in self.level_order
          }
  ```

- [ ] **Step 4: Confirm the batch block now receives the YAML values.** The block at lines 480-493 already pops the keys into the batch fields — now that Step 3 stopped consuming them first, no edit is needed there. For clarity update the default fallbacks to match documented intent and add a comment. Current lines 481-483:

  ```python
          self._batch_window            = int(kwargs.pop("batch_window", 10))
          self._batch_advance_threshold = float(kwargs.pop("advance_threshold", 0.30))
          self._batch_fallback_threshold= float(kwargs.pop("fallback_threshold", 0.20))
  ```

  Replace with (keep defaults; add comment noting these are now the SOLE consumer):

  ```python
          self._batch_window            = int(kwargs.pop("batch_window", 10))
          # SOLE consumer of advance_threshold / fallback_threshold kwargs (see
          # the curriculum block above). Defaults apply only when YAML omits them.
          self._batch_advance_threshold = float(kwargs.pop("advance_threshold", 0.30))
          self._batch_fallback_threshold= float(kwargs.pop("fallback_threshold", 0.20))
  ```

- [ ] **Step 5: Wire `_level_window` writes in `reset()`.** The episode-boundary block at lines 784-787 currently records ONLY into the batch list:

  ```python
          # Record episode result into batch — only if it was a curriculum episode
          # (NOT a review episode, which played a different level)
          if self.level_order and not self.locked_level and not self._is_review_episode:
              self._batch_results.append(self._episode_won_current)
  ```

  Replace with (also append the 1/0 outcome to the per-level diagnostic window, keyed by the level just played, i.e. `self.world` BEFORE reset reassigns it):

  ```python
          # Record episode result into batch — only if it was a curriculum episode
          # (NOT a review episode, which played a different level)
          if self.level_order and not self.locked_level and not self._is_review_episode:
              self._batch_results.append(self._episode_won_current)

          # Record into the per-level diagnostic window (feeds curriculum_win_rate
          # in _info()). self.world still names the level that was just played —
          # it is reassigned to the next level later in this method. Use setdefault
          # so levels not present in level_order (edge cases) still get a window.
          if self.level_order and not self.locked_level:
              win_dq = self._level_window.setdefault(
                  self.world, deque(maxlen=self._curriculum_window_size))
              win_dq.append(1 if self._episode_won_current else 0)
  ```

  Note: `_episode_won_current` is reset to `False` later at line 794, so this read must stay BEFORE that line (it does). The `deque` and `random` names are already imported at the top of the file (lines 6 and 14).

- [ ] **Step 6: Run the new test, watch it PASS.**
  `.venv/bin/python -m pytest code/tests/test_curriculum_thresholds.py -v`
  Expected: both tests PASS. `_batch_advance_threshold == 0.55`, `_batch_fallback_threshold == 0.15`, mastery fields absent, and `curriculum_win_rate == 0.75` after the synthetic W/L/W/W sequence.

- [ ] **Step 7: Manual runtime confirmation (optional sanity check).**
  ```
  .venv/bin/python -c "import os; os.environ['SDL_VIDEODRIVER']='dummy'; \
  from code.games.platformer_core import PlatformerCore; \
  g=PlatformerCore(render_mode='none', advance_threshold=0.55, fallback_threshold=0.15); \
  print(g._batch_advance_threshold, g._batch_fallback_threshold, hasattr(g,'_advance_threshold'))"
  ```
  Expected output: `0.55 0.15 False`. (Before the fix it printed `0.3 0.2 True`.)

- [ ] **Step 8: Checkpoint (DO NOT COMMIT).** Run the full suite and pause for review:
  `.venv/bin/python -m pytest code/tests/ -q`
  Expected: all tests pass (the new file plus the existing suite). Do NOT `git add` / `git commit` — pause for the user to review.

---

### Task 8: Fix the model=rppo NameError (add missing RecurrentEvalCallback import)

**Problem (verified):** `code/scripts/train.py` uses `RecurrentEvalCallback(...)` at line 1166 (the recurrent branch taken when `model_name.lower() in ['rppo', 'recurrent_ppo']`, set at line 1154) but never imports it. The name appears only inside a docstring (line 43). Confirmed at runtime: after `import code.scripts.train as m`, `hasattr(m, 'RecurrentEvalCallback')` is `False`. So selecting `model=rppo` crashes with `NameError: name 'RecurrentEvalCallback' is not defined`. The class exists and imports cleanly from `code/callbacks/RecurrentEvalCallback.py:8`.

**Files:**
- Modify: `code/scripts/train.py` (add one import near the existing callback imports at lines 2-3 / 26)
- Create: `code/tests/test_train_imports.py`

- [ ] **Step 1: Write the failing test first.** Create `code/tests/test_train_imports.py`. This imports the trainer module and asserts the symbol used at the `model=rppo` call site (line 1166) resolves in the module namespace. Run it BEFORE editing train.py to confirm it fails with the NameError-equivalent (the attribute is missing).

```python
# code/tests/test_train_imports.py
"""
Regression guard for the model=rppo NameError.

train.py references RecurrentEvalCallback at its recurrent-eval call site
(the branch taken when model_name in {'rppo','recurrent_ppo'}). If the import
is missing, that line raises NameError at runtime. Importing the trainer module
is safe: all training execution is gated behind @hydra.main + the
`if __name__ == "__main__"` guard, so import alone runs no training.
"""
import importlib


def test_train_module_imports_without_error():
    # Must not raise (e.g. ImportError) on import.
    importlib.import_module("code.scripts.train")


def test_recurrent_eval_callback_symbol_resolves_in_train_namespace():
    train = importlib.import_module("code.scripts.train")
    # This is the exact name referenced at the model=rppo call site (train.py:1166).
    # Before the fix this attribute does not exist -> the call site would NameError.
    assert hasattr(train, "RecurrentEvalCallback"), (
        "train.py references RecurrentEvalCallback (line ~1166) but does not "
        "import it; model=rppo would crash with NameError."
    )
    # And it must be the real callback class, not some unrelated rebinding.
    from code.callbacks.RecurrentEvalCallback import RecurrentEvalCallback
    assert train.RecurrentEvalCallback is RecurrentEvalCallback
```

- [ ] **Step 2: Run the test and confirm it FAILS.** Exact command:
  `.venv/bin/python -m pytest code/tests/test_train_imports.py -v`
  Expected: `test_train_module_imports_without_error` PASSES (the module currently imports fine — the bug is latent, not an import-time crash), and `test_recurrent_eval_callback_symbol_resolves_in_train_namespace` FAILS on the `assert hasattr(...)` line. (Benign `objc[...] Class SDL_* is implemented in both ...` stderr lines from cv2/pygame are expected noise, not failures.)

- [ ] **Step 3: Add the missing import.** In `code/scripts/train.py`, add the import next to the sibling callback imports at the top of the file (existing convention, lines 2-3: `from code.callbacks.logging_callback import CsvLoggerCallback` / `from code.callbacks.profiling_callback import ProfilingCallback, EvalTimerCallback`). Insert the new line immediately after line 3.

  Current (train.py:1-3):
```python
import os
from code.callbacks.logging_callback import CsvLoggerCallback
from code.callbacks.profiling_callback import ProfilingCallback, EvalTimerCallback
```

  After edit:
```python
import os
from code.callbacks.logging_callback import CsvLoggerCallback
from code.callbacks.profiling_callback import ProfilingCallback, EvalTimerCallback
from code.callbacks.RecurrentEvalCallback import RecurrentEvalCallback
```

  Exact import line added:
```python
from code.callbacks.RecurrentEvalCallback import RecurrentEvalCallback
```

- [ ] **Step 4: Run the test and confirm it PASSES.** Exact command:
  `.venv/bin/python -m pytest code/tests/test_train_imports.py -v`
  Expected: both tests PASS (2 passed). This proves `RecurrentEvalCallback` now resolves in `train.py`'s namespace, so the `model=rppo` call site at line 1166 no longer raises `NameError`.

- [ ] **Step 5 (optional, NOT a unit test — requires a real run): smoke-check the rppo path.** The unit test proves the symbol resolves but does not exercise an actual recurrent training run. To observe the end-to-end unblock, launch the shortest possible rppo run and confirm it gets past callback construction (no `NameError`). Example:
  `.venv/bin/python -m code.scripts.train model=rppo` (add whatever minimal hydra overrides the grid config needs to keep total_timesteps tiny, e.g. a small `total_timesteps`/grid override).
  Record: that training starts and the eval callback is constructed without `NameError: name 'RecurrentEvalCallback' is not defined`. This is a manual observation, not an automated assertion.

- [ ] **Step 6: Checkpoint (DO NOT COMMIT).** Run the full suite and pause for review:
  `.venv/bin/python -m pytest code/tests/ -q`
  Expected: all tests pass (the four existing test files plus the new `test_train_imports.py`). Do NOT `git add` / `git commit` — pause for the user to review.

---

### Task 9: Scale parallel envs + set the real training budget (PHASE 1 long run)

**Context (read before editing).** The vec-env is built in `code/scripts/train.py:1121-1125`:
```python
n_envs = int(cfg.get("n_envs", 1))
if n_envs > 1:
    raw_env = SubprocVecEnv([make_env() for _ in range(n_envs)])
else:
    raw_env = DummyVecEnv([make_env()])
```
`make_env()` (defined at `code/scripts/train.py:1105-1119`) returns a fresh `_init` callable each call, so the SubprocVecEnv path already gets `n_envs` independent env factories — the wiring exists; this task only raises the count and the budget. Current config (`code/conf/grid.yaml`): `n_envs: 6` (line 38), `eval_freq: 10000` (line 34), `save_freq: 20000` (line 35), and the budget `skills: Novice: 1_000_000 / Expert: 8_000_000` (lines 23-25). Device is `cpu` (line 37) — MPS is blocked (per project memory), so subproc envs and the learner share CPU cores. PPO uses `n_steps: 2048`, `batch_size: 512` (`code/conf/algo/ppo.yaml`).

**IMPORTANT cadence note (state this in the run log).** In SB3, `EvalCallback`/`CheckpointCallback` count **per-env** steps. `eval_freq`/`save_freq` are checked once per `n_envs` global steps, so the *effective* global cadence is `eval_freq * n_envs`. At `eval_freq: 10000`, `n_envs: 12` → eval every ~120k global steps; `save_freq: 20000` → checkpoint every ~240k global steps. That is acceptable for a multi-M run; do NOT also shrink eval_freq or evals will dominate wall time on CPU.

**Depends on:** Tasks 1, 3, 4, 5, 6, 7 (esp. Task 6 — the registered curriculum the long run trains on).

**Files:**
- Modify `code/conf/grid.yaml` (line 38 `n_envs`; lines 24-25 `skills` budget)
- Test (create) `code/tests/test_vec_env_wiring.py`

- [ ] **Step 1: Write the config-wiring test (failing first against current values).** Create `code/tests/test_vec_env_wiring.py`. It asserts (a) `grid.yaml` carries the scaled `n_envs` and budget, and (b) the train.py dispatch contract picks SubprocVecEnv for `n_envs>1` and DummyVecEnv otherwise. The dispatch is asserted via a local mirror of the exact branch (no env/pygame/SB3 import — fast, deterministic). Loading `grid.yaml` with `OmegaConf` resolves cleanly without a Hydra run (verified: `n_envs=6, Novice=1000000, Expert=8000000`).
  ```python
  """Config-wiring + vec-env dispatch assertions for the Phase-1 long run (Task 9).

  Proves throughput/wiring only:
    - grid.yaml carries the scaled n_envs and multi-M budget,
    - the train.py dispatch contract (n_envs>1 -> SubprocVecEnv, else DummyVecEnv)
      passes exactly n_envs env-init callables to the constructor.
  It does NOT prove the learning outcome — that is observed via the LAUNCH +
  MONITOR procedure in Step 4 (per-level win-rate rising over baseline).
  """
  from pathlib import Path

  from omegaconf import OmegaConf

  REPO_ROOT = Path(__file__).resolve().parents[2]
  GRID = REPO_ROOT / "code" / "conf" / "grid.yaml"


  # --- (a) config values are wired to the scaled Phase-1 budget ---------------

  def _cfg():
      return OmegaConf.load(GRID)

  def test_n_envs_scaled_for_subproc_path():
      n_envs = int(_cfg().get("n_envs"))
      assert n_envs >= 8, f"Phase-1 expects 8-16 parallel envs, got n_envs={n_envs}"
      assert n_envs <= 16, f"n_envs={n_envs} exceeds the 8-16 Phase-1 band"
      assert n_envs > 1, "n_envs>1 is required to take the SubprocVecEnv path"

  def test_training_budget_is_multi_million():
      skills = _cfg().get("skills")
      novice = int(skills.get("Novice"))
      expert = int(skills.get("Expert"))
      # Phase-1 targets ~5-10M on the registered curriculum.
      assert novice >= 5_000_000, f"Novice budget too small for Phase-1: {novice}"
      assert expert >= 10_000_000, f"Expert budget too small for Phase-1: {expert}"

  def test_eval_and_save_freq_are_per_env_documented_values():
      # These stay as-is; effective global cadence = freq * n_envs. Guard against
      # an accidental shrink that would make eval dominate CPU wall time.
      cfg = _cfg()
      assert int(cfg.get("eval_freq")) >= 10_000
      assert int(cfg.get("save_freq")) >= 20_000


  # --- (b) dispatch contract: mirrors train.py:1121-1125 exactly --------------

  class _FakeSubproc:
      def __init__(self, env_fns):
          self.env_fns = list(env_fns)
          self.kind = "subproc"

  class _FakeDummy:
      def __init__(self, env_fns):
          self.env_fns = list(env_fns)
          self.kind = "dummy"

  def _dispatch_vec_env(n_envs, make_env, SubprocVecEnv, DummyVecEnv):
      """Local mirror of train.py:1121-1125 — kept in lockstep with that branch."""
      n_envs = int(n_envs)
      if n_envs > 1:
          return SubprocVecEnv([make_env() for _ in range(n_envs)])
      return DummyVecEnv([make_env()])

  def test_dispatch_uses_subproc_when_n_envs_gt_1():
      calls = {"n": 0}
      def make_env():
          calls["n"] += 1
          return lambda: None
      vec = _dispatch_vec_env(12, make_env, _FakeSubproc, _FakeDummy)
      assert vec.kind == "subproc"
      assert len(vec.env_fns) == 12          # exactly n_envs init callables
      assert calls["n"] == 12                # make_env() called once per env

  def test_dispatch_uses_dummy_when_n_envs_eq_1():
      vec = _dispatch_vec_env(1, lambda: (lambda: None), _FakeSubproc, _FakeDummy)
      assert vec.kind == "dummy"
      assert len(vec.env_fns) == 1

  def test_config_n_envs_takes_subproc_branch():
      # The real config value, run through the real branch logic, must select subproc.
      n_envs = int(_cfg().get("n_envs"))
      vec = _dispatch_vec_env(n_envs, lambda: (lambda: None), _FakeSubproc, _FakeDummy)
      assert vec.kind == "subproc"
      assert len(vec.env_fns) == n_envs
  ```

- [ ] **Step 2: Run the test — confirm it FAILS on current config, passes after Step 3.** Run:
  ```
  .venv/bin/python -m pytest code/tests/test_vec_env_wiring.py -v
  ```
  Expected NOW (before Step 3): the dispatch tests pass, but `test_n_envs_scaled_for_subproc_path` FAILS (`n_envs=6 < 8`) and `test_training_budget_is_multi_million` FAILS (`Novice=1000000 < 5000000`, `Expert=8000000 < 10000000`). This confirms the test actually guards the config change.

- [ ] **Step 3: Apply the config edits to `code/conf/grid.yaml`.**
  Edit line 38 — raise `n_envs` into the 8-16 band (use 12 as the default for an 8-core+ Mac; an executor on a smaller machine may pick 8):
  ```yaml
  n_envs: 12        # 8-16 parallel SubprocVecEnv workers (Phase-1). CPU device: workers share cores with the learner.
  ```
  (was: `n_envs: 6         # 1 for cpu, 16 for gpu, etc.`)
  Edit lines 24-25 — set the Phase-1 budget toward ~5-10M on the registered curriculum:
  ```yaml
  skills:
    Novice: 6_000_000
    Expert: 10_000_000
  ```
  (was: `Novice: 1_000_000` / `Expert: 8_000_000`)
  Leave `eval_freq: 10000` and `save_freq: 20000` (lines 34-35) unchanged — see the cadence note above.

- [ ] **Step 4: Re-run the wiring test — expect all green.**
  ```
  .venv/bin/python -m pytest code/tests/test_vec_env_wiring.py -v
  ```
  Expected: all tests pass. This proves the config is wired (scaled `n_envs` flows into the `n_envs>1` SubprocVecEnv branch with exactly `n_envs` env factories) — it does NOT prove the learning outcome.

- [ ] **Step 5: (Optional, recommended) Throughput sanity check before committing to the long run.** Confirm the chosen `n_envs` actually raises steps/sec on CPU rather than thrashing cores. Use the existing profiler via a short real run (Novice budget temporarily overridden tiny). From repo root:
  ```
  .venv/bin/python -m code.scripts.train games=[platformer] models=[ppo] persona=platformer_simple skill=Novice skills.Novice=50000 n_envs=12 profile=true dashboard=false
  ```
  Then inspect the per-rollout SPS column written by `ProfilingCallback`:
  ```
  .venv/bin/python -c "import csv,glob; f=sorted(glob.glob('csv/**/profiling_log.csv',recursive=True))[-1]; rows=list(csv.DictReader(open(f))); print('file',f); print('mean sps', sum(float(r['sps']) for r in rows)/len(rows)); print('mean env%', 100*sum(float(r['env_step_s']) for r in rows)/sum(float(r['env_step_s'])+float(r['policy_forward_s'])+float(r['train_s'])+float(r['eval_s']) for r in rows))"
  ```
  Re-run with `n_envs=8` and compare mean SPS. RECORD: pick the `n_envs` with the highest SPS (if 8 beats 12 because workers starve the learner on this machine, set grid.yaml back to 8 and re-run Step 4). This step proves throughput, not learning. (Reference: `code/scripts/benchmark_device.py` compares devices but uses DummyVecEnv/`n_steps=256`, so it does NOT exercise the SubprocVecEnv scaling — the train.py command above is the right throughput probe.)

- [ ] **Step 6: LAUNCH the Phase-1 long run (procedure, not a unit test).** From repo root, train the Novice budget on the registered curriculum (Task 6). The curriculum-advance env kwargs are already set in `train.py:949-952`:
  ```
  .venv/bin/python -m code.scripts.train games=[platformer] models=[ppo] persona=platformer_simple skill=Novice n_envs=12 dashboard=false
  ```
  (Drop `dashboard=false` if you want the Streamlit Flight Recorder. To run Expert instead, use `skill=Expert`. To use a smaller box, append `n_envs=8`.) This is a multi-hour CPU run.

- [ ] **Step 7: MONITOR progress.** Watch two surfaces:
  - **TensorBoard** — logs go under `mylogs/` (`tb_root` in grid.yaml; run dir `mylogs/platformer_ppo_platformer_simple/`):
    ```
    tensorboard --logdir mylogs
    ```
    Track `rollout/ep_rew_mean` (rising), `eval/mean_reward` (rising at each eval), and the profiler SPS scalar (stable — no collapse mid-run).
  - **Eval logs** — `EvalCallback` writes `models/eval_logs/platformer_ppo_platformer_simple_novice_<extractor>/evaluations.npz`; new bests land in `models/best/<run_id>/best_model.zip` with `best_model_vecnorm.pkl` alongside (saved by `EvalPreviewCallback`). Eval fires every ~`eval_freq * n_envs` ≈ 120k global steps.

- [ ] **Step 8: Define + record SUCCESS (the learning outcome — observed, not unit-tested).** Success = **per-level win-rate on the Task-1 eval rising over baseline.** Run the Task-1 eval harness against `models/best/platformer_ppo_platformer_simple_novice_<extractor>/best_model.zip` (with its `best_model_vecnorm.pkl`) at a few checkpoints during/after the run, and compare per-level win-rate to the pre-Task-6/9 baseline checkpoint. RECORD in the run notes: chosen `n_envs`, mean SPS from Step 5, total_timesteps reached, eval mean_reward curve, and the per-level win-rate delta vs baseline. Be explicit in the notes that Steps 1-4 prove only throughput/wiring; the learning improvement is established solely by this observed eval delta.

- [ ] **Step 9: Checkpoint (DO NOT COMMIT).** Run the full suite and pause for review:
  ```
  .venv/bin/python -m pytest code/tests/ -q
  ```
  Expected: all tests pass (the new `test_vec_env_wiring.py` included). Do NOT `git add`/`git commit` — pause for the user to review the config change and the run notes.

---

## Measured baseline (2026-06-18) — reference for Task 9 re-measurement

Measured with `code/scripts/eval_level.py` (trustworthy per-level eval: level pinned, curriculum off,
`terminate_on_goal=True`) on the best shipped model
`models/best/platformer_ppo_platformer_adept_novice_spatialattention/best_model.zip`.

| Level | Deterministic | Stochastic (15 eps) | Dominant failure (stochastic) |
|---|---|---|---|
| Mario1-1 | 0% (stall-trap) | **60.0%** (9/15) | pit (5) |
| Mario1-2 | 0% (stall-trap) | **13.3%** (2/15) | pit (8), stall (3) |

- The previously-cited **~41.6% was a broken-eval artifact**; clean eval = 60% / 13% stochastic, 0% deterministic.
- **Deterministic eval traps the policy in a fixed action loop the anti-stall watchdog kills** (→ 0%).
  `eval_level.py` now warns when stalls dominate. **Compare future runs in STOCHASTIC mode** (watch deterministic separately as a robustness signal).
- Dominant death is **pit** (runs off ledges) — consistent with over-weighted forward movement; what Task 3 + curriculum target.
- **Bars to beat:** Mario1-1 > 60% and Mario1-2 > 13% (stochastic), ideally deterministic also rising above ~0%.

## Self-review notes (controller)

- **Spec coverage:** Task 1+2 = Phase 0 (eval integrity + baseline); Tasks 3–9 = Phase 1 (reward rebalance, `norm_reward`, horizon/`POTENTIAL_GAMMA`, curriculum registration, double-pop fix, `rppo` import, scale+train). Maps 1:1 to spec §5–§6 and diagnosis causes 1–7.
- **No-commit:** every task ends in a Checkpoint, not a commit.
- **Interfaces:** Task 1 introduces `PlatformerCore(curriculum_enabled=, terminate_on_goal=, world=)`; Tasks 2 and 9 consume that eval env. Task 5 depends on Task 4 (normalize before raising gamma) and keeps `POTENTIAL_GAMMA == gamma`.
- **Reward change caution:** Task 3 preserves potential-based-shaping invariance; only the non-PBRS distance bonus is capped.
