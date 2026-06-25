from __future__ import annotations
import math
from typing import Callable, Tuple, Dict, Any

Info = Dict[str, Any]

# PBRS shaping discount. MUST equal PPO `gamma` in code/conf/algo/ppo.yaml
# (potential-based shaping is policy-invariant only when these are equal —
# Ng, Harada & Russell 1999). Guarded by code/tests/test_gamma_sync.py.
POTENTIAL_GAMMA = 0.997

# =============================================================================
# Score Tracker
# =============================================================================

class _ScoreTracker:
    """
    Per-environment tracker. Handles soft-reset detection (life lost) to
    prevent poisoned progress deltas on respawn.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.prev_score    = 0
        self.last_dist     = None
        self.last_dijkstra = None
        self.last_x        = None
        self.max_x         = 0.0
        self.last_coins    = 0
        self.last_lives    = None

    def step(self, info: Info) -> Tuple[int, bool]:
        # 0. Life-loss detection
        current_lives = int(info.get("lives", 3))
        if self.last_lives is None:
            self.last_lives = current_lives
        life_lost = current_lives < self.last_lives
        self.last_lives = current_lives
        info["life_lost"] = life_lost

        # 1. Score delta
        current_score = int(info.get("score", 0))
        inc = current_score > self.prev_score
        self.prev_score = current_score

        # 2. Euclidean progress
        current_dist = float(info.get("goal_dist", 0.0))
        if math.isinf(current_dist):
            current_dist = 0.0
        if self.last_dist is None:
            self.last_dist = current_dist
        if life_lost:
            progress = 0.0
            self.last_dist = current_dist
        else:
            progress = self.last_dist - current_dist
            self.last_dist = current_dist
        info["progress"] = progress

        # 3. Dijkstra progress
        # BUG (was): platformer_core reported dijkstra_dist = 1.0 whenever the
        # player's current tile was unreachable by the solver (e.g. mid-air, or
        # on a platform edge that the flood-fill hadn't visited). When the player
        # was near the goal (e.g. last_dijkstra ≈ 0.3) and then jumped, the
        # sentinel value caused: dijkstra_progress = 0.3 - 1.0 = -0.7, which
        # translated to a reward of -70 per step while airborne. The agent learned
        # to never jump.
        #
        # FIX: Treat sentinel values (< 0 or > 1 indicate "unknown / unreachable")
        # as "no new information" → progress = 0 rather than a large negative.
        # platformer_core already clips valid values to [0, 1]; we rely on that
        # convention and check for the sentinel 1.0 that is set when d < 0.
        raw_dijkstra = float(info.get("dijkstra_dist", -1.0))

        # platformer_core emits:
        #   [0.0, 1.0]  → valid normalised distance (0 = at goal, 1 = furthest tile)
        #   -1.0        → sentinel: tile is unreachable or player is off-grid
        #
        # BUG (was): check was `raw_dijkstra < 1.0`, which passed -1.0 as valid
        # because -1 < 1 is True. The tracker then stored -1.0 as last_dijkstra,
        # and on the next valid step computed: progress = -1.0 - 0.4 = -1.4,
        # producing a reward of -140. The comment also referenced the old sentinel
        # value (1.0) which was changed when we fixed platformer_core.
        #
        # FIX: valid means the value is a real distance, i.e. >= 0.
        # This correctly handles all three cases:
        #   -1.0          → invalid (unreachable)     → False ✓
        #    0.0 to 1.0   → valid distance            → True  ✓
        #    1.0 exactly  → furthest reachable tile   → True  ✓ (was wrongly excluded before)
        dijkstra_valid = (raw_dijkstra >= 0.0)
        info["dijkstra_valid"] = dijkstra_valid

        # Capture the PREVIOUS step's anchor before ANY modification.
        # This is used for the PBRS potential below so that dijkstra_dist_prev
        # always reflects the distance from the step BEFORE this one, not the
        # current step.  (Bug was: last_dijkstra was overwritten first, so
        # dijkstra_dist_prev ended up equal to curr_d every step.)
        prev_dijkstra_anchor = self.last_dijkstra

        if self.last_dijkstra is None:
            self.last_dijkstra = raw_dijkstra if dijkstra_valid else None

        if life_lost:
            dijkstra_progress = 0.0
            # Reset anchor only if we have a valid reading after respawn
            self.last_dijkstra = raw_dijkstra if dijkstra_valid else None
        elif not dijkstra_valid or self.last_dijkstra is None:
            # No valid Dijkstra reading this step → neutral (no reward, no penalty)
            dijkstra_progress = 0.0
            # Don't update last_dijkstra so we resume from a clean anchor next
            # time we land on a valid tile
        else:
            dijkstra_progress = self.last_dijkstra - raw_dijkstra
            self.last_dijkstra = raw_dijkstra

        info["dijkstra_progress"] = dijkstra_progress

        # --- Potential-Based Reward Shaping ---
        # Pass the previous step's raw Dijkstra distance so the persona can
        # compute the properly gamma-discounted potential: prev - γ*curr.
        # The persona reads info["dijkstra_dist"] for curr and this for prev.
        # Set to -1.0 (sentinel) when no valid previous reading exists so the
        # persona can skip the computation on the first step of an episode/level.
        #
        # NOTE: we use prev_dijkstra_anchor (captured before any write above)
        # so that this always reflects the PRIOR step's distance, not the current.
        info["dijkstra_dist_prev"] = (
            prev_dijkstra_anchor
            if (prev_dijkstra_anchor is not None and dijkstra_valid)
            else -1.0
        )
        # --- End Potential-Based Reward Shaping ---

        if info.get("won", False) or info.get("terminated", False):
            info["progress"]          = 0.0
            info["dijkstra_progress"] = 0.0

        # BUG: Level transitions caused large negative gradient spikes.
        #
        # tracker.reset() only fires on terminated=True (lives=0). Level
        # completions continue the episode, so last_dijkstra kept its
        # near-goal value from level N (e.g. 0.05). On the first step of
        # level N+1 raw_dijkstra is ~0.9 (far from the new goal), giving:
        #     dijkstra_progress = 0.05 - 0.9 = -0.85  →  r_gradient = -42.5
        # This was the source of the large negative spikes in TensorBoard
        # at every level transition.
        #
        # FIX: On win, reset last_dijkstra and last_dist to None so both
        # anchors re-initialise cleanly on the first step of the new level,
        # exactly as they do at episode start.
        if info.get("won", False):
            self.last_dijkstra = None
            self.last_dist     = None  # also reset euclidean anchor for same reason

        # 4. Frontier
        current_x = float(info.get("x_position", 0.0))
        if self.last_x is None:
            self.last_x = current_x
            self.max_x  = current_x
        if life_lost:
            self.max_x = current_x
        env_max = float(info.get("max_x_seen", 0.0))
        if env_max > 0:
            frontier_delta = max(0.0, env_max - self.max_x)
            self.max_x = env_max
        else:
            frontier_delta = max(0.0, current_x - self.max_x)
            self.max_x = max(self.max_x, current_x)
        info["frontier_dx"] = frontier_delta

        # 5. Coin delta
        current_coins = int(info.get("coins_collected", 0))
        if current_coins < self.last_coins:
            self.last_coins = current_coins
        info["coins_delta"] = max(0, current_coins - self.last_coins)
        self.last_coins = current_coins

        # 6. Kill passthrough
        if "enemies_killed_step" not in info:
            info["enemies_killed_step"] = 0

        return current_score, inc


def _wrap_with_tracker(core_fn) -> Callable:
    """
    Returns a FACTORY callable instead of a single shared reward fn.
    Each GameEnv calls the factory to get its own tracker instance.
    This is critical for parallel training (SubprocVecEnv / DummyVecEnv).
    """
    def make_reward_fn():
        tracker = _ScoreTracker()

        def reward(obs, base, terminated: bool, info: Info) -> float:
            _score, inc = tracker.step(info or {})
            components  = core_fn(inc, terminated, info or {}, _score)
            info["reward_components"] = components
            total = float(sum(components.values()))
            if terminated or (info and info.get("terminated", False)):
                tracker.reset()
            return total

        return reward

    make_reward_fn._core_fn    = core_fn
    make_reward_fn._is_factory = True
    return make_reward_fn


# =============================================================================
# Reward Personas
# =============================================================================

@_wrap_with_tracker
def adept(score_inc: bool, terminated: bool, info: Info, score: int) -> Dict[str, float]:
    """
    DIJKSTRA — follow the pathfinding gradient.

    REWRITE: Previous version had POTENTIAL_SCALE=3.0 which gave ~0.02/step.
    Over a 10,000-step episode that's 200 cumulative — the win bonus of 3.0
    was invisible (1.5% of total reward). Agent learned to oscillate along
    the gradient instead of actually reaching the goal.

    Fix: POTENTIAL_SCALE=0.3 → ~0.002/step → ~6 cumulative per 3000 steps.
    Win bonus 5.0 is now 45% of total reward. Agent actually cares about winning.
    """
    kills       = int(info.get("enemies_killed_step", 0))
    coins       = int(info.get("coins_delta", 0))
    won         = info.get("won", False)
    life_lost   = info.get("life_lost", False)
    dijkstra_valid = info.get("dijkstra_valid", False)
    on_platform = info.get("on_moving_platform", False)

    # ── Potential-based shaping (THE FIX: scale 3.0 → 0.3) ───────────────
    POTENTIAL_SCALE = 0.3     # was 3.0 — dominated everything

    r_potential = 0.0
    curr_d = float(info.get("dijkstra_dist", -1.0))
    prev_d = float(info.get("dijkstra_dist_prev", -1.0))
    if dijkstra_valid and prev_d >= 0.0:
        r_potential = (prev_d - POTENTIAL_GAMMA * curr_d) * POTENTIAL_SCALE

    # ── Velocity alignment (skip on platforms) ────────────────────────────
    r_alignment = 0.0
    if dijkstra_valid and not on_platform:
        vx    = float(info.get("velocity_x", 0.0))
        vy    = float(info.get("velocity_y", 0.0))
        speed = math.sqrt(vx * vx + vy * vy)
        if speed > 0.5:
            step_dx = float(info.get("step_dx", 0.0))
            step_dy = float(info.get("step_dy", 0.0))
            alignment = (vx / speed) * step_dx + (vy / speed) * step_dy
            r_alignment = max(0.0, alignment) * 0.003

    # ── Small alive bonus (replaces frontier as exploration incentive) ────
    r_alive = 0.0005

    # ── Platform patience ─────────────────────────────────────────────────
    r_patience = 0.002 if on_platform else 0.0

    r_coin = 0.08 * coins
    r_kill = 0.1 * kills
    r_stall = -0.003 if bool(info.get("stalled", False)) else 0.0

    r_win, r_death = 0.0, 0.0
    if won:
        r_win = 5.0
    elif terminated:
        r_death = -3.0
    elif life_lost:
        r_death = -0.3

    return {
        "potential":  r_potential,
        "alignment":  r_alignment,
        "alive":      r_alive,
        "patience":   r_patience,
        "stall":      r_stall,
        "coins":      r_coin,
        "kills":      r_kill,
        "win":        r_win,
        "time":       -0.00003,
        "death":      r_death,
    }


@_wrap_with_tracker
def simple(score_inc: bool, terminated: bool, info: Info, score: int) -> Dict[str, float]:
    """
    SIMPLE — euclidean progress toward goal.

    REWRITE: Previous version had frontier at 93-97% of total reward.
    The agent learned "run right into new territory" instead of "reach the goal."
    Frontier gives 200+ reward per episode, win gives 3.0 — agent doesn't care
    about winning because exploring pays 60x more.

    Fix: Frontier REMOVED entirely. It's a trap — teaches the agent to explore
    not to navigate. Replaced with a small alive bonus (survive = good) and
    stronger terminal signals (win/death actually matter now).

    Movement reward clamped to prevent it from dominating like frontier did.
    """
    progress    = float(info.get("progress", 0.0))
    kills       = int(info.get("enemies_killed_step", 0))
    coins       = int(info.get("coins_delta", 0))
    won         = info.get("won", False)
    life_lost   = info.get("life_lost", False)
    on_platform = info.get("on_moving_platform", False)

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

    # ── Stall penalty (not on platforms) ──────────────────────────────────
    if not on_platform and abs(progress) < 0.5:
        r_move -= 0.0005

    # ── Small alive bonus (replaces frontier) ─────────────────────────────
    # 0.0005/step × 3000 steps = 1.5 per episode. Win is 5.0. Balanced.
    r_alive = 0.0005

    # ── Platform patience ─────────────────────────────────────────────────
    r_patience = 0.002 if on_platform else 0.0

    r_coin = 0.08 * coins
    r_kill = 0.1 * kills

    r_win, r_death = 0.0, 0.0
    if won:
        r_win = 5.0
    elif terminated:
        r_death = -3.0
    elif life_lost:
        r_death = -0.5

    return {
        "movement":  r_move,
        "alive":     r_alive,
        "patience":  r_patience,
        "coins":     r_coin,
        "kills":     r_kill,
        "win":       r_win,
        "time":      -0.00005,
        "death":     r_death,
    }



@_wrap_with_tracker
def enemy_hunter(score_inc: bool, terminated: bool, info: Info, score: int) -> Dict[str, float]:
    """
    ENEMY HUNTER: Maximise enemy kills.

    KEY DESIGN: NO dijkstra potential. The goal gradient is removed so the
    agent doesn't just skip past enemies to reach the goal. The ONLY strong
    positive signal comes from stomping enemies.

    Without a goal gradient, the agent must learn to approach enemies using
    the hazard channel in its observation grid. Frontier reward nudges it
    forward to encounter new enemies. Powered-up bonus encourages grabbing
    mushrooms/stars that make killing safer.
    """
    kills      = int(info.get("enemies_killed_step", 0))
    coins      = int(info.get("coins_delta", 0))
    won        = info.get("won", False)
    life_lost  = info.get("life_lost", False)
    powered_up = bool(info.get("powered_up", False))

    # ── Primary signal: kills (the ONLY strong positive) ───────────────
    r_kill = 0.5 * kills

    r_powerup = 0.002 if powered_up else 0.0

    r_alive = 0.0005

    r_coin = 0.02 * coins

    r_stall = -0.003 if bool(info.get("stalled", False)) else 0.0

    r_win, r_death = 0.0, 0.0
    if won:
        r_win  = 3.0
    elif terminated:
        r_death = -2.0
    elif life_lost:
        r_death = -0.1

    return {
        "kills":     r_kill,
        "powerup":   r_powerup,
        "alive":     r_alive,
        "coins":     r_coin,
        "stall":     r_stall,
        "win":       r_win,
        "time":      -0.00003,
        "death":     r_death,
    }


@_wrap_with_tracker
def speedrunner(score_inc: bool, terminated: bool, info: Info, score: int) -> Dict[str, float]:
    """
    SPEEDRUNNER: Finish as fast as possible.

    Dominant signal is rightward velocity. Dijkstra potential is kept but
    at moderate weight — it provides direction while velocity provides
    urgency. Steep time penalty makes every step without progress costly.
    Backtracking is severely punished. Coins and kills are irrelevant.
    """
    kills     = int(info.get("enemies_killed_step", 0))
    coins     = int(info.get("coins_delta", 0))
    won       = info.get("won", False)
    life_lost = info.get("life_lost", False)
    dijkstra_valid = info.get("dijkstra_valid", False)

    # ── Primary signal: rightward velocity (clamped) ─────────────────────
    vx = float(info.get("velocity_x", 0.0))
    r_velocity = max(0.0, vx) * 0.001   # was 0.005 — dominated everything
    r_velocity = min(0.005, r_velocity)  # clamp

    # ── Dijkstra potential for direction ──────────────────────────────────
    POTENTIAL_SCALE = 0.3              # was 0.8

    r_potential = 0.0
    curr_d = float(info.get("dijkstra_dist", -1.0))
    prev_d = float(info.get("dijkstra_dist_prev", -1.0))
    if dijkstra_valid and prev_d >= 0.0:
        r_potential = (prev_d - POTENTIAL_GAMMA * curr_d) * POTENTIAL_SCALE

    progress = float(info.get("progress", 0.0))
    r_backtrack = min(0.0, progress) * 0.01

    r_stall = -0.003 if bool(info.get("stalled", False)) else 0.0

    r_win, r_death = 0.0, 0.0
    if won:
        r_win  = 8.0               # strongest win bonus — whole point is finishing fast
    elif terminated:
        r_death = -3.0
    elif life_lost:
        r_death = -0.3

    return {
        "velocity":  r_velocity,
        "potential": r_potential,
        "backtrack": r_backtrack,
        "stall":     r_stall,
        "win":       r_win,
        "time":      -0.0002,      # time pressure — makes speed matter
        "death":     r_death,
    }


@_wrap_with_tracker
def completionist(score_inc: bool, terminated: bool, info: Info, score: int) -> Dict[str, float]:
    """
    COMPLETIONIST: Do everything — coins, kills, progress, survive.

    Uses dijkstra potential at moderate weight, plus meaningful rewards for
    coins and kills. Unlike coin_hunter/enemy_hunter (which remove the goal
    gradient), completionist keeps it but at parity with the other signals
    so no single component dominates.

    The agent should learn versatile play: grab coins when nearby, stomp
    enemies when possible, but keep moving toward the goal.
    """
    kills     = int(info.get("enemies_killed_step", 0))
    coins     = int(info.get("coins_delta", 0))
    total_coins = int(info.get("coins_collected", 0))
    won       = info.get("won", False)
    life_lost = info.get("life_lost", False)
    dijkstra_valid = info.get("dijkstra_valid", False)

    # ── Moderate dijkstra potential ──────────────────────────────────────
    POTENTIAL_SCALE = 0.3        # was 0.6 — same fix as dijkstra persona

    r_potential = 0.0
    curr_d = float(info.get("dijkstra_dist", -1.0))
    prev_d = float(info.get("dijkstra_dist_prev", -1.0))
    if dijkstra_valid and prev_d >= 0.0:
        r_potential = (prev_d - POTENTIAL_GAMMA * curr_d) * POTENTIAL_SCALE

    # ── All objectives at meaningful weight ────────────────────────────
    r_coin     = 0.15 * coins
    r_kill     = 0.15 * kills
    r_alive    = 0.0005                # replaces frontier

    on_platform = info.get("on_moving_platform", False)
    r_patience  = 0.002 if on_platform else 0.0

    r_stall = -0.003 if bool(info.get("stalled", False)) else 0.0

    r_win, r_death = 0.0, 0.0
    if won:
        r_win = 5.0 + min(total_coins, 30) * 0.1   # coin bonus on top
    elif terminated:
        r_death = -3.0
    elif life_lost:
        r_death = -0.5

    return {
        "potential": r_potential,
        "coins":     r_coin,
        "kills":     r_kill,
        "alive":     r_alive,
        "patience":  r_patience,
        "stall":     r_stall,
        "win":       r_win,
        "time":      -0.00005,
        "death":     r_death,
    }


# =============================================================================
# Aliases
# =============================================================================
platformer_simple     = simple
platformer_adept      = adept
platformer_enemy_hunter  = enemy_hunter
platformer_speedrunner   = speedrunner
platformer_completionist = completionist
default                  = simple