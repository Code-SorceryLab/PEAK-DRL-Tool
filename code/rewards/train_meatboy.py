from __future__ import annotations
from typing import Dict, Any
from code.rewards.train_platformer import _wrap_with_tracker

Info = Dict[str, Any]


@_wrap_with_tracker
def meatboy_simple(score_inc: bool, terminated: bool, info: Info, score: int) -> Dict[str, float]:
    """Reach the goal alive. Dense progress toward goal + completion bonus +
    death penalty + a small step cost to discourage dawdling."""
    progress = float(info.get("progress", 0.0))   # tracker derives this from goal_dist
    won = bool(info.get("won", False))

    r_move = max(-0.02, min(0.02, progress * 0.01))
    r_alive = 0.0005
    r_win, r_death = 0.0, 0.0
    if won:
        r_win = 5.0
    elif terminated:
        r_death = -2.0

    return {
        "movement": r_move,
        "alive": r_alive,
        "time": -0.0005,
        "win": r_win,
        "death": r_death,
    }


@_wrap_with_tracker
def meatboy_bfs(score_inc: bool, terminated: bool, info: Info, score: int) -> Dict[str, float]:
    """Reach the goal via POTENTIAL-BASED shaping on the BFS goal-distance.

    Unlike meatboy_simple's euclidean progress, the BFS path distance is
    correct *through* walls — the right metric for Meat Boy's vertical /
    wall-jump mazes. Potential Phi = -bfs_dist; shaping is the pure potential
    DIFFERENCE. The core emits the normalised BFS distance as `dijkstra_dist`,
    so the shared _ScoreTracker fills in `dijkstra_dist_prev` /
    `dijkstra_valid` (including sentinel handling for airborne cells)."""
    won = bool(info.get("won", False))

    # Difference-based potential shaping:  F = (prev - curr) * SCALE
    #
    # BUG THIS FIXES — "stand still and farm the discount drift":
    # Strict PBRS is F = prev - gamma*curr with gamma = the RL discount (0.99).
    # But when the agent DOESN'T MOVE, prev == curr == d, so
    #     F = d - 0.99*d = 0.01*d  > 0
    # i.e. a free +0.005/step at mid-level (d~0.5) for doing NOTHING. Over a
    # 4000-step episode that is ~+16, dwarfing the win bonus (5.0), so the
    # optimal policy became "stand still". Observed directly: the agent moved
    # under meatboy_simple but froze under meatboy_bfs.
    #
    # gamma_shape = 1.0 makes standing still EXACTLY 0, so only real progress
    # pays. It telescopes to (d_start - d_end), bounded by 1.0, and back-and-
    # forth wiggling nets 0. (Dropping strict gamma-matching costs negligible
    # policy-invariance here and removes the pathological fixed point.)
    # SCALE only sets gradient strength: with gamma_shape=1 the total shaping
    # telescopes to SCALE*(d_start - d_end) <= SCALE, and backtracking is
    # exactly symmetric, so there is no farming exploit at any scale.
    # 3.0 => ~3.0 cumulative dense pull vs the 5.0 win bonus (win stays
    # dominant) — comparable pull to meatboy_simple, which does move the agent.
    SCALE = 3.0

    # Anchors come from the CORE (bfs_dist / bfs_dist_prev), not the shared
    # tracker: the tracker only resets on `terminated`, so on a TRUNCATED
    # (timed-out) episode its anchor leaked into the next episode and produced
    # a large spurious potential spike on step 1. The core resets its anchor in
    # reset(), so it is correct across both terminations and truncations.
    r_potential = 0.0
    curr = float(info.get("bfs_dist", -1.0))
    prev = float(info.get("bfs_dist_prev", -1.0))
    if curr >= 0.0 and prev >= 0.0:
        r_potential = (prev - curr) * SCALE

    r_win, r_death = 0.0, 0.0
    if won:
        r_win = 5.0
    elif terminated:
        r_death = -2.0

    # Time penalty must stay small relative to the terminal signals. Episodes
    # run up to grid.yaml max_steps (10_000), so |time| * max_steps must not
    # exceed |death| (2.0) — otherwise timing out costs more than dying and
    # SUICIDE becomes the optimal policy. 0.0002 * 10_000 = 2.0 (the cap).
    return {
        "potential": r_potential,
        "time": -0.0002,
        "win": r_win,
        "death": r_death,
    }


# train.py looks up reward functions by persona name; provide a default too.
default = meatboy_simple
