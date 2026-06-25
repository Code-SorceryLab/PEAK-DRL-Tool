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

    # __init__ calls reset() once, which records _episode_won_current=False (a
    # loss) into the window. Clear it so we start from a known-empty state and
    # can verify the -1.0 sentinel.
    from collections import deque
    core._level_window[lvl] = deque(maxlen=core._curriculum_window_size)
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
