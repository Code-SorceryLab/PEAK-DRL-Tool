from __future__ import annotations
import math
from typing import Callable, Tuple, Dict, Any

Info = Dict[str, Any]

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
        info["dijkstra_dist_prev"] = (
            self.last_dijkstra if (self.last_dijkstra is not None and dijkstra_valid)
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
def delta_dijkstra(score_inc: bool, terminated: bool, info: Info, score: int) -> Dict[str, float]:
    """
    DIJKSTRA: Primary signal is Dijkstra distance improvement toward goal.

    The Dijkstra solver outputs a normalised distance in [0, 1] where 0 = at
    goal and 1 = furthest possible tile (or unreachable). The tracker computes
    the per-step delta and zeroes it on invalid readings (airborne / unreachable)
    so that jumping is never penalised.

    Also includes:
      - Potential-based reward shaping (Ng et al. 1999): γ·Φ(s') - Φ(s)
      - Velocity alignment: bonus when player velocity points toward goal
    """
    kills     = int(info.get("enemies_killed_step", 0))
    coins     = int(info.get("coins_delta", 0))
    won       = info.get("won", False)
    life_lost = info.get("life_lost", False)
    dijkstra_valid = info.get("dijkstra_valid", False)

    # -------------------------------------------------------------------------
    # Original gradient signal — commented out in favour of r_potential, which
    # is the theoretically correct (gamma-discounted) version of the same idea.
    # Remove the comment block below to re-enable if needed for comparison.
    # -------------------------------------------------------------------------
    # d_prog = float(info.get("dijkstra_progress", 0.0))
    # if dijkstra_valid:
    #     r_gradient = d_prog * 50.0
    # else:
    #     r_gradient = 0.0
    r_gradient = 0.0  # disabled — see r_potential below

    # =========================================================================
    # Potential-Based Reward Shaping
    # =========================================================================
    # Φ(s) = -dijkstra_dist  (lower cost = closer to goal = higher potential)
    # Shaped reward = γ·Φ(s') - Φ(s)
    #              = γ·(-curr) - (-prev)
    #              = prev - γ·curr
    #
    # With γ < 1 this differs from the plain delta (prev - curr) by adding a
    # small positive bonus that scales with how close the player already is to
    # the goal. This is theoretically grounded: it cannot alter the optimal
    # policy, it can only speed up convergence.
    #
    # GAMMA must match the PPO gamma in your algo config (default 0.99).
    # If you change PPO gamma, update this constant to match.
    POTENTIAL_GAMMA  = 0.99
    POTENTIAL_SCALE  = 10.0   # tune: scales the shaped bonus relative to r_gradient

    r_potential = 0.0
    curr_dijkstra = float(info.get("dijkstra_dist", -1.0))
    prev_dijkstra = float(info.get("dijkstra_dist_prev", -1.0))

    if dijkstra_valid and prev_dijkstra >= 0.0:
        r_potential = (prev_dijkstra - POTENTIAL_GAMMA * curr_dijkstra) * POTENTIAL_SCALE
    # =========================================================================
    # End Potential-Based Reward Shaping
    # =========================================================================

    # =========================================================================
    # Velocity Alignment
    # =========================================================================
    # Rewards the agent when its movement direction aligns with the direction
    # toward the cheapest reachable tile among all 8 neighbours (cardinal +
    # diagonal). The direction vector comes from platformer_core._tracking_obs,
    # where the 8-direction search already excludes wall tiles (inf cost).
    #
    # Guards:
    #   - on_ground:      avoids rewarding random mid-air drift during jumps
    #   - dijkstra_valid: no valid path signal → no alignment reward
    #   - speed > 0.1:    ignore tiny jitter movements
    #
    # ALIGNMENT_SCALE: 0.01 × 60fps = 0.6 max reward/sec at perfect alignment.
    # Intentionally weaker than r_potential.
    ALIGNMENT_SCALE = 0.05  # Raised from 0.01; on_ground gate removed (was killing 99% of signal)

    r_alignment = 0.0
    if dijkstra_valid:
        vx    = float(info.get("velocity_x", 0.0))
        vy    = float(info.get("velocity_y", 0.0))
        speed = math.sqrt(vx * vx + vy * vy)
        if speed > 0.5:
            step_dx = float(info.get("step_dx", 0.0))
            step_dy = float(info.get("step_dy", 0.0))
            alignment   = (vx / speed) * step_dx + (vy / speed) * step_dy
            r_alignment = max(0.0, alignment) * ALIGNMENT_SCALE
    # =========================================================================
    # End Velocity Alignment
    # =========================================================================

    r_coin = 0.2 * coins   # Small: coins shouldn't compete with forward progress
    r_kill = 0.5 * kills

    # Per-step stall penalty (61% of deaths are stalls — punish before kill fires)
    r_stall = -0.01 if bool(info.get("stalled", False)) else 0.0

    r_win, r_death = 0.0, 0.0
    if won:
        r_win  = 20.0
    elif terminated:
        r_death = -15.0
    elif life_lost:
        r_death = -1.0

    return {
        "gradient":   r_gradient,
        "potential":  r_potential,  # Potential-Based Reward Shaping
        "alignment":  r_alignment,  # Velocity Alignment
        "stall":      r_stall,
        "coins":      r_coin,
        "kills":      r_kill,
        "win":        r_win,
        "time":       -0.0005,
        "death":      r_death,
    }


@_wrap_with_tracker
def simple(score_inc: bool, terminated: bool, info: Info, score: int) -> Dict[str, float]:
    """
    SIMPLE: Euclidean progress + frontier exploration bonus.

    Deliberately lightweight — good baseline for debugging reward shaping.
    """
    progress  = float(info.get("progress", 0.0))
    frontier  = float(info.get("frontier_dx", 0.0))
    kills     = int(info.get("enemies_killed_step", 0))
    coins     = int(info.get("coins_delta", 0))
    won       = info.get("won", False)
    life_lost = info.get("life_lost", False)

    # Stronger forward signal; extra penalty for backtracking
    r_move = progress * 0.15
    if progress < 0:
        r_move *= 2.5       # Extra backtrack penalty
    if abs(progress) < 0.5:
        r_move -= 0.01      # Stall penalty

    r_frontier = frontier * 0.2   # Only reward genuinely new ground
    r_coin     = 0.05 * coins     # Weak coin reward — prevents coin-farming policy
    r_kill     = 0.5 * kills

    r_win, r_death = 0.0, 0.0
    if won:
        r_win  = 200.0
    elif terminated:
        r_death = -15.0
    elif life_lost:
        r_death = -3.0

    return {
        "movement": r_move,
        "frontier": r_frontier,
        "coins":    r_coin,
        "kills":    r_kill,
        "win":      r_win,
        "time":     -0.002,
        "death":    r_death,
    }


@_wrap_with_tracker
def coin_hunter(score_inc: bool, terminated: bool, info: Info, score: int) -> Dict[str, float]:
    """
    COIN HUNTER: Maximise coin collection.

    Heavy reward for each coin picked up, plus an exploration bonus to
    incentivise visiting new territory (where uncollected coins live).
    Moderate dijkstra potential keeps the agent moving toward the goal so
    it doesn't get stuck farming the same area.

    Design contrast vs dijkstra: 10× coin weight, 0.5× potential weight.
    The agent should learn to detour for coins even when it slows progress.
    """
    kills     = int(info.get("enemies_killed_step", 0))
    coins     = int(info.get("coins_delta", 0))
    won       = info.get("won", False)
    life_lost = info.get("life_lost", False)
    dijkstra_valid = info.get("dijkstra_valid", False)

    # ── Primary signal: coins ──────────────────────────────────────────
    r_coin = 2.0 * coins           # 10× vs dijkstra's 0.2

    # ── Exploration bonus (new ground = unseen coins) ──────────────────
    frontier = float(info.get("frontier_dx", 0.0))
    r_frontier = frontier * 0.4

    # ── Light dijkstra potential to keep moving toward goal ─────────────
    POTENTIAL_GAMMA = 0.99
    POTENTIAL_SCALE = 5.0          # 0.5× vs dijkstra's 10.0

    r_potential = 0.0
    curr_d = float(info.get("dijkstra_dist", -1.0))
    prev_d = float(info.get("dijkstra_dist_prev", -1.0))
    if dijkstra_valid and prev_d >= 0.0:
        r_potential = (prev_d - POTENTIAL_GAMMA * curr_d) * POTENTIAL_SCALE

    r_kill = 0.3 * kills           # Incidental kills still rewarded lightly

    r_stall = -0.02 if bool(info.get("stalled", False)) else 0.0

    r_win, r_death = 0.0, 0.0
    if won:
        r_win  = 15.0              # Moderate: finishing matters, but coins matter more
    elif terminated:
        r_death = -20.0            # Harsh: dying loses all collected coins
    elif life_lost:
        r_death = -3.0

    return {
        "coins":     r_coin,
        "frontier":  r_frontier,
        "potential": r_potential,
        "kills":     r_kill,
        "stall":     r_stall,
        "win":       r_win,
        "time":      -0.0003,      # Light time pressure
        "death":     r_death,
    }


@_wrap_with_tracker
def enemy_hunter(score_inc: bool, terminated: bool, info: Info, score: int) -> Dict[str, float]:
    """
    ENEMY HUNTER: Maximise enemy kills.

    Huge reward per stomp, plus a bonus for having a powerup (mushroom/star
    make killing safer). Moderate forward progress prevents camping at the
    start. Death penalty is lighter than coin_hunter because aggressive play
    means more deaths are expected.

    Design contrast vs dijkstra: 10× kill weight, small powerup incentive.
    The agent should learn to seek out and engage enemies rather than avoid them.
    """
    kills      = int(info.get("enemies_killed_step", 0))
    coins      = int(info.get("coins_delta", 0))
    won        = info.get("won", False)
    life_lost  = info.get("life_lost", False)
    powered_up = bool(info.get("powered_up", False))
    dijkstra_valid = info.get("dijkstra_valid", False)

    # ── Primary signal: kills ──────────────────────────────────────────
    r_kill = 5.0 * kills           # 10× vs dijkstra's 0.5

    # ── Powerup bonus: being powered up enables safer kills ────────────
    r_powerup = 0.005 if powered_up else 0.0

    # ── Moderate dijkstra potential to keep advancing ───────────────────
    POTENTIAL_GAMMA = 0.99
    POTENTIAL_SCALE = 6.0

    r_potential = 0.0
    curr_d = float(info.get("dijkstra_dist", -1.0))
    prev_d = float(info.get("dijkstra_dist_prev", -1.0))
    if dijkstra_valid and prev_d >= 0.0:
        r_potential = (prev_d - POTENTIAL_GAMMA * curr_d) * POTENTIAL_SCALE

    r_coin = 0.1 * coins           # Incidental, not a priority

    # Stall penalty slightly harsher — hunter shouldn't camp
    r_stall = -0.015 if bool(info.get("stalled", False)) else 0.0

    r_win, r_death = 0.0, 0.0
    if won:
        r_win  = 15.0
    elif terminated:
        r_death = -10.0            # Lighter: aggressive play = more deaths expected
    elif life_lost:
        r_death = -1.0

    return {
        "kills":     r_kill,
        "powerup":   r_powerup,
        "potential": r_potential,
        "coins":     r_coin,
        "stall":     r_stall,
        "win":       r_win,
        "time":      -0.0005,
        "death":     r_death,
    }


@_wrap_with_tracker
def speedrunner(score_inc: bool, terminated: bool, info: Info, score: int) -> Dict[str, float]:
    """
    SPEEDRUNNER: Finish as fast as possible.

    Heavy velocity reward, steep time penalty, massive win bonus. Coins
    and kills are almost irrelevant — only speed and completion matter.
    Backtracking is severely punished.

    Design contrast vs dijkstra: velocity-driven rather than distance-driven.
    The agent should learn to sprint right and take risky shortcuts.
    """
    kills     = int(info.get("enemies_killed_step", 0))
    coins     = int(info.get("coins_delta", 0))
    won       = info.get("won", False)
    life_lost = info.get("life_lost", False)
    dijkstra_valid = info.get("dijkstra_valid", False)

    # ── Primary signal: horizontal velocity ────────────────────────────
    vx = float(info.get("velocity_x", 0.0))
    # Reward rightward speed, penalise leftward
    r_velocity = vx * 0.03         # ~0.03 per tile/sec of rightward speed

    # ── Strong dijkstra potential — finishing fast ──────────────────────
    POTENTIAL_GAMMA = 0.99
    POTENTIAL_SCALE = 12.0         # 1.2× vs dijkstra's 10.0

    r_potential = 0.0
    curr_d = float(info.get("dijkstra_dist", -1.0))
    prev_d = float(info.get("dijkstra_dist_prev", -1.0))
    if dijkstra_valid and prev_d >= 0.0:
        r_potential = (prev_d - POTENTIAL_GAMMA * curr_d) * POTENTIAL_SCALE

    # ── Backtrack penalty ──────────────────────────────────────────────
    progress = float(info.get("progress", 0.0))
    r_backtrack = min(0.0, progress) * 0.3  # Only applies when progress < 0

    r_coin = 0.05 * coins          # Negligible
    r_kill = 0.2  * kills          # Only if they're in the way

    r_stall = -0.03 if bool(info.get("stalled", False)) else 0.0

    r_win, r_death = 0.0, 0.0
    if won:
        r_win  = 30.0              # Huge: the whole point is to reach the goal
    elif terminated:
        r_death = -12.0
    elif life_lost:
        r_death = -2.0             # Moderate: dying wastes time, but risk is expected

    return {
        "velocity":  r_velocity,
        "potential": r_potential,
        "backtrack": r_backtrack,
        "coins":     r_coin,
        "kills":     r_kill,
        "stall":     r_stall,
        "win":       r_win,
        "time":      -0.003,       # 6× vs dijkstra — heavy time pressure
        "death":     r_death,
    }


@_wrap_with_tracker
def completionist(score_inc: bool, terminated: bool, info: Info, score: int) -> Dict[str, float]:
    """
    COMPLETIONIST: Do everything — coins, kills, progress, survive.

    Balanced multi-objective with no single dominant signal. Every action
    type is meaningful: collecting coins, stomping enemies, making progress,
    and staying alive all contribute roughly equally to the total reward.

    Design contrast vs dijkstra: wider reward surface, no single dominant
    gradient. The agent should learn versatile play that adapts to whatever
    opportunities each level presents.
    """
    kills     = int(info.get("enemies_killed_step", 0))
    coins     = int(info.get("coins_delta", 0))
    won       = info.get("won", False)
    life_lost = info.get("life_lost", False)
    dijkstra_valid = info.get("dijkstra_valid", False)

    # ── Balanced dijkstra potential ────────────────────────────────────
    POTENTIAL_GAMMA = 0.99
    POTENTIAL_SCALE = 8.0          # Between coin_hunter (5) and speedrunner (12)

    r_potential = 0.0
    curr_d = float(info.get("dijkstra_dist", -1.0))
    prev_d = float(info.get("dijkstra_dist_prev", -1.0))
    if dijkstra_valid and prev_d >= 0.0:
        r_potential = (prev_d - POTENTIAL_GAMMA * curr_d) * POTENTIAL_SCALE

    # ── All objectives matter equally ──────────────────────────────────
    r_coin     = 1.0 * coins       # Middle ground (0.2 dijkstra / 2.0 coin_hunter)
    r_kill     = 2.0 * kills       # Middle ground (0.5 dijkstra / 5.0 enemy_hunter)
    r_frontier = float(info.get("frontier_dx", 0.0)) * 0.2

    # ── Velocity alignment (from dijkstra, same params) ────────────────
    r_alignment = 0.0
    if dijkstra_valid:
        vx    = float(info.get("velocity_x", 0.0))
        vy    = float(info.get("velocity_y", 0.0))
        speed = math.sqrt(vx * vx + vy * vy)
        if speed > 0.5:
            step_dx = float(info.get("step_dx", 0.0))
            step_dy = float(info.get("step_dy", 0.0))
            alignment   = (vx / speed) * step_dx + (vy / speed) * step_dy
            r_alignment = max(0.0, alignment) * 0.03

    r_stall = -0.015 if bool(info.get("stalled", False)) else 0.0

    r_win, r_death = 0.0, 0.0
    if won:
        r_win  = 25.0              # Strong but not dominant
    elif terminated:
        r_death = -20.0            # Harsh: completionists don't die
    elif life_lost:
        r_death = -4.0             # Strictest life penalty

    return {
        "potential": r_potential,
        "alignment": r_alignment,
        "coins":     r_coin,
        "kills":     r_kill,
        "frontier":  r_frontier,
        "stall":     r_stall,
        "win":       r_win,
        "time":      -0.001,       # Moderate time pressure
        "death":     r_death,
    }


# =============================================================================
# Aliases
# =============================================================================
platformer_simple       = simple
platformer_dijkstra    = delta_dijkstra
platformer_coin_hunter = coin_hunter
platformer_enemy_hunter = enemy_hunter
platformer_speedrunner = speedrunner
platformer_completionist = completionist
default                = simple