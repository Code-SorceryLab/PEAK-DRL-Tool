"""
Task 3b — PBRS tracker correctness tests.

Three properties that must hold after the ordering bug is fixed in
_ScoreTracker.step():

1. Gradient correctness: potential = (prev_d - GAMMA*curr_d)*SCALE using the
   ACTUAL previous step's distance, not the current one.
2. First-step anchor: the first step of an episode yields potential=0 (sentinel
   path prevents a spike when there is no meaningful previous distance).
3. No cross-episode leak: after episode termination, the first step of a new
   episode also anchors to 0 (last_dijkstra is reset so the prior episode's
   final distance does not pollute the new episode).
"""
import pytest
from code.rewards.train_platformer import adept, POTENTIAL_GAMMA

# POTENTIAL_GAMMA imported from module — stays in sync with ppo.yaml via test_gamma_sync.py
POTENTIAL_SCALE = 0.3


def _step(rfn, dijkstra_dist, terminated=False, extra=None):
    """Helper: run one step through rfn, return (info, potential_component)."""
    info = {"dijkstra_dist": dijkstra_dist, "x_position": 0.0, "lives": 3}
    if extra:
        info.update(extra)
    rfn(None, None, terminated, info)
    return info, info["reward_components"]["potential"]


# ---------------------------------------------------------------------------
# Test 1: Gradient correctness
# Feed two steps with decreasing dijkstra (0.8 → 0.6).
# After the fix: prev_d=0.8, curr_d=0.6 → potential=(0.8-POTENTIAL_GAMMA*0.6)*0.3
# Before the fix (bug): prev_d==curr_d==0.6 → potential=(0.6-POTENTIAL_GAMMA*0.6)*0.3
# ---------------------------------------------------------------------------
def test_gradient_uses_actual_previous_distance():
    """
    The PBRS potential must use the PRIOR step's distance as prev_d, not the
    current step's. Feeding dijkstra 0.8 then 0.6 must yield
    (0.8 - POTENTIAL_GAMMA*0.6)*0.3, NOT the buggy (0.6 - POTENTIAL_GAMMA*0.6)*0.3.
    """
    rfn = adept()
    # Step 1: anchor — last_dijkstra is None; after this step last_dijkstra=0.8.
    # The dijkstra_dist_prev should come out as -1.0 (no previous), so r_potential=0.
    _step(rfn, dijkstra_dist=0.8)

    # Step 2: the bug shows up here.
    # CORRECT (fixed): prev_d=0.8, curr_d=0.6 → (0.8 - POTENTIAL_GAMMA*0.6)*0.3
    # BUGGY         : prev_d=0.6, curr_d=0.6 → (0.6 - POTENTIAL_GAMMA*0.6)*0.3
    info, potential = _step(rfn, dijkstra_dist=0.6)

    expected = (0.8 - POTENTIAL_GAMMA * 0.6) * POTENTIAL_SCALE
    buggy    = (0.6 - POTENTIAL_GAMMA * 0.6) * POTENTIAL_SCALE

    # Must equal the gradient value, not the collapsed value.
    assert potential == pytest.approx(expected, abs=1e-9), (
        f"potential={potential:.6f}, expected gradient={expected:.6f}, "
        f"buggy collapsed value would be={buggy:.6f}"
    )


# ---------------------------------------------------------------------------
# Test 2: First-step anchor
# On the very first step of an episode, last_dijkstra is None.
# The sentinel path must prevent any potential from being emitted (potential=0).
# ---------------------------------------------------------------------------
def test_first_step_potential_is_zero():
    """
    On the first step of a fresh episode, there is no prior distance.
    The sentinel (-1.0) must cause the persona guard (prev_d >= 0) to skip
    the potential computation → potential == 0.0, no spike.
    """
    rfn = adept()
    # Very first step — last_dijkstra is None.
    _, potential = _step(rfn, dijkstra_dist=0.5)
    assert potential == pytest.approx(0.0, abs=1e-9), (
        f"First step should yield potential=0.0, got {potential}"
    )


# ---------------------------------------------------------------------------
# Test 3: No cross-episode leak
# After an episode terminates, tracker.reset() is called (by _wrap_with_tracker).
# The first step of the NEW episode must NOT use the prior episode's last_dijkstra
# as prev_d — it should again anchor to 0 (sentinel path).
# ---------------------------------------------------------------------------
def test_no_cross_episode_leak():
    """
    After a terminated episode, the tracker resets last_dijkstra to None.
    The first step of the next episode must yield potential=0 (sentinel path),
    not a bogus potential spike computed from the previous episode's final distance.
    """
    rfn = adept()

    # Simulate a short episode with dijkstra decreasing to near goal.
    _step(rfn, dijkstra_dist=0.8)
    _step(rfn, dijkstra_dist=0.5)
    # Terminate the episode (lives=0 signals game-over).
    _step(rfn, dijkstra_dist=0.1, terminated=True, extra={"lives": 0})

    # --- New episode starts ---
    # The tracker should have reset last_dijkstra to None on terminated=True.
    # The first step should therefore anchor cleanly (sentinel → potential=0).
    _, potential_new_ep_first = _step(rfn, dijkstra_dist=0.9)
    assert potential_new_ep_first == pytest.approx(0.0, abs=1e-9), (
        f"First step of new episode should yield potential=0.0 (no cross-episode "
        f"leak), got {potential_new_ep_first}"
    )
