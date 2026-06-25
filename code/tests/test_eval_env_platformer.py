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
