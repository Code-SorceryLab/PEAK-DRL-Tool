"""PBRS prev-anchor regression: dijkstra_dist_prev must lag dijkstra_dist by
one step.

The historical bug: _ScoreTracker.step overwrote self.last_dijkstra to the
CURRENT reading before publishing info["dijkstra_dist_prev"], so prev == curr
on every steady-state step and the PBRS term (prev - γ·curr)·scale collapsed
into a flat stand-still bonus (largest far from the goal) — the dense
directional gradient was dead for adept/speedrunner/completionist. The 1M-run
forensics confirmed it empirically: corr(potential, dijkstra_dist) ≈ 0.05.
"""
import pytest

from code.rewards.train_platformer import _ScoreTracker, adept

POTENTIAL_GAMMA = 0.99
POTENTIAL_SCALE = 0.3


def _info(dijkstra, lives=3, **extra):
    base = {
        "lives": lives,
        "score": 0,
        "goal_dist": 10.0,
        "dijkstra_dist": dijkstra,
        "x_position": 100.0,
    }
    base.update(extra)
    return base


def _step(tracker, dijkstra, lives=3, **extra):
    info = _info(dijkstra, lives=lives, **extra)
    tracker.step(info)
    return info


def test_first_step_publishes_sentinel():
    t = _ScoreTracker()
    info = _step(t, 0.5)
    # No previous reading exists on the first step → sentinel, persona skips.
    assert info["dijkstra_dist_prev"] == -1.0


def test_prev_lags_curr_by_one_step():
    t = _ScoreTracker()
    _step(t, 0.5)                    # anchor
    info2 = _step(t, 0.4)            # approached the goal
    assert info2["dijkstra_dist_prev"] == pytest.approx(0.5)   # NOT 0.4
    info3 = _step(t, 0.3)
    assert info3["dijkstra_dist_prev"] == pytest.approx(0.4)   # NOT 0.3


def test_standing_still_prev_equals_curr():
    # When genuinely stationary prev == curr is CORRECT (γ-baseline only).
    t = _ScoreTracker()
    _step(t, 0.5)
    info = _step(t, 0.5)
    assert info["dijkstra_dist_prev"] == pytest.approx(0.5)


def test_life_lost_publishes_sentinel():
    # Respawn teleports the player; pre-death prev with post-respawn curr
    # would produce a spurious potential spike → sentinel instead.
    t = _ScoreTracker()
    _step(t, 0.2, lives=3)
    info = _step(t, 0.9, lives=2)    # died, respawned far from goal
    assert info["life_lost"] is True
    assert info["dijkstra_dist_prev"] == -1.0


def test_invalid_reading_publishes_sentinel():
    t = _ScoreTracker()
    _step(t, 0.5)
    info = _step(t, -1.0)            # off-grid / unreachable this step
    assert info["dijkstra_dist_prev"] == -1.0


def _potential_after(readings):
    core = adept._core_fn
    t = _ScoreTracker()
    info = None
    for d in readings:
        info = _step(t, d)
    comps = core(False, False, info, 0)
    return comps["potential"]


def _expected(prev, curr, start):
    # (prev - γ·curr)·scale minus the Φ(start)=0 anchor constant (1-γ)·d_start·scale
    return ((prev - POTENTIAL_GAMMA * curr)
            - (1.0 - POTENTIAL_GAMMA) * start) * POTENTIAL_SCALE


def test_adept_potential_rewards_approach_over_standing():
    """End-to-end through the tracker: approaching the goal must pay strictly
    more potential than standing still, and retreating must pay less (the
    gradient is alive and correctly signed)."""
    approach = _potential_after([0.5, 0.4])
    standing = _potential_after([0.5, 0.5])
    retreat  = _potential_after([0.5, 0.6])

    assert approach == pytest.approx(_expected(0.5, 0.4, start=0.5))
    assert standing == pytest.approx(_expected(0.5, 0.5, start=0.5))
    assert retreat  == pytest.approx(_expected(0.5, 0.6, start=0.5))
    assert approach > standing > retreat
    assert retreat < 0.0


def test_phi_anchor_standing_at_start_pays_zero():
    """The Φ(start)=0 anchor: a stationary agent at spawn earns EXACTLY 0
    potential — the old (1-γ)·d standing income (which made farm-to-the-cap
    out-earn winning in eval) is gone."""
    assert _potential_after([0.5, 0.5]) == pytest.approx(0.0)
    assert _potential_after([0.9, 0.9]) == pytest.approx(0.0)


def test_phi_anchor_standing_income_never_positive():
    """Standing anywhere pays ≤ 0: exactly 0 at spawn, slightly negative when
    closer than spawn (mild urgency near the goal, not a reward for camping)."""
    near_goal_camp = _potential_after([0.5, 0.2, 0.2])  # advanced, then camped
    assert near_goal_camp == pytest.approx(_expected(0.2, 0.2, start=0.5))
    assert near_goal_camp < 0.0


def test_phi_anchor_reanchors_on_win():
    """On won the tracker must re-anchor: the next level's spawn distance
    becomes the new Φ(start)=0 reference."""
    t = _ScoreTracker()
    _step(t, 0.3)
    info = _step(t, 0.1, won=True)          # win → transition
    assert info["dijkstra_dist_start"] == pytest.approx(0.3)
    info = _step(t, 0.8)                     # first step of next level
    assert info["dijkstra_dist_start"] == pytest.approx(0.8)


def test_phi_anchor_survives_life_lost():
    """Respawn returns to the same spawn — the anchor must NOT re-anchor to
    the (pre-death) position where the agent happened to die."""
    t = _ScoreTracker()
    _step(t, 0.5)
    _step(t, 0.2)                            # advanced deep into the level
    info = _step(t, 0.5, lives=2)            # died, respawned
    assert info["life_lost"] is True
    assert info["dijkstra_dist_start"] == pytest.approx(0.5)


def test_first_step_pays_no_potential():
    """Before the fix, step 1 paid (curr - 0.99*curr)*0.3 = 0.003*curr — a
    bonus for merely existing far from the goal. The sentinel must gate it."""
    core = adept._core_fn
    t = _ScoreTracker()
    info = _step(t, 0.9)
    comps = core(False, False, info, 0)
    assert comps["potential"] == 0.0
