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
from code.rewards.train_platformer import simple, adept, POTENTIAL_GAMMA


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
    Guard: the dijkstra PBRS term in adept is computed in train_platformer.py as
    r_potential = (prev_d - POTENTIAL_GAMMA*curr_d) * SCALE, where SCALE=0.3.
    POTENTIAL_GAMMA is imported from the module (tied to ppo.yaml via test_gamma_sync.py).

    Drive through TWO steps so the tracker has a real previous distance to use:
      step 1: dijkstra=0.8  → anchors last_dijkstra; prev_d=None (sentinel) → potential=0
      step 2: dijkstra=0.6  → prev_d=0.8, curr_d=0.6 → (0.8 - POTENTIAL_GAMMA*0.6)*0.3

    This guards both the formula/constants AND that the tracker correctly passes
    the PRIOR step's distance as prev_d (not the current step's — that was the bug
    fixed in Task 3b).
    """
    rfn = adept()
    # Step 1: anchor — last_dijkstra starts as None; after this step last_dijkstra=0.8.
    # No previous reading exists, so dijkstra_dist_prev=-1.0 → potential=0.
    rfn(None, None, False, {"dijkstra_dist": 0.8, "x_position": 0.0, "lives": 3})

    # Step 2: now prev_d=0.8 (the prior step's value), curr_d=0.6.
    info = {"dijkstra_dist": 0.6, "x_position": 0.0, "lives": 3}
    rfn(None, None, False, info)

    # Expected: (prev_d=0.8 - POTENTIAL_GAMMA*curr_d=0.6) * SCALE
    expected = (0.8 - POTENTIAL_GAMMA * 0.6) * 0.3
    assert info["reward_components"]["potential"] == pytest.approx(expected, abs=1e-9)
