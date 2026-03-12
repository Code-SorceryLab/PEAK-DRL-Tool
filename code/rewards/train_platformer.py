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
    POTENTIAL_GAMMA = 0.99
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

    # ── Movement (clamped so it can't dominate) ───────────────────────────
    r_move = progress * 0.003
    if progress < 0:
        r_move *= 1.5              # soft backtrack penalty
    r_move = max(-0.01, min(0.01, r_move))   # CLAMP — prevents runaway

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
def coin_hunter(score_inc: bool, terminated: bool, info: Info, score: int) -> Dict[str, float]:
    """
    COIN HUNTER (legacy): Maximise coin collection.
    Preserved for backwards compatibility — new runs should use coin_collector.
    """
    kills     = int(info.get("enemies_killed_step", 0))
    coins     = int(info.get("coins_delta", 0))
    total_coins = int(info.get("coins_collected", 0))
    won       = info.get("won", False)
    life_lost = info.get("life_lost", False)

    r_coin = 0.3 * coins
    r_alive = 0.0005
    r_kill = 0.02 * kills
    r_stall = -0.003 if bool(info.get("stalled", False)) else 0.0
    r_win, r_death = 0.0, 0.0
    if won:
        r_win = 3.0 + min(total_coins, 30) * 0.15
    elif terminated:
        r_death = -3.0
    elif life_lost:
        r_death = -0.3

    return {
        "coins":     r_coin,
        "alive":     r_alive,
        "kills":     r_kill,
        "stall":     r_stall,
        "win":       r_win,
        "time":      -0.00003,
        "death":     r_death,
    }


# =============================================================================
# COIN COLLECTOR — modular reward shaping (Features 2 & 3)
# =============================================================================

class _CoinCollectorWeights:
    """
    All tunable scalars in one place.  Edit here to tune without touching logic.
    """
    # ── Feature 2: Coin-collection signals ────────────────────────────────────
    COIN_PICKUP       = 2.0      # reward per coin collected this step
    COIN_PROXIMITY    = 0.12     # scale of 1/dist proximity gradient
    PROXIMITY_MIN_DIST= 0.5      # floor distance (tiles) to avoid div-by-zero
    TIME_PENALTY      = -0.01    # per-step time cost (pushes agent to act fast)

    # Combo multiplier: reward when ≥2 coins collected in COMBO_WINDOW steps
    COMBO_WINDOW      = 8        # steps within which coins count as a combo
    COMBO_BONUS       = 0.5      # flat bonus per combo trigger
    COMBO_MAX         = 3.0      # cap on total combo bonus per episode

    # ── Feature 3: Exploration / backtracking signals ─────────────────────────
    EXPLORATION_BONUS = 0.06     # reward for first visit to a new tile
    BACKTRACK_BONUS   = 0.25     # reward for revisiting a tile that had a coin
    REVISIT_PENALTY   = -0.004   # per-step penalty when visit_count ≥ STALE_THRESH
    STALE_THRESH      = 6        # visit_count above which tile is "stale"

    # ── Terminal signals ─────────────────────────────────────────────────────
    WIN_BASE          = 3.0
    WIN_COIN_SCALE    = 0.20     # extra reward per coin at win
    WIN_COIN_CAP      = 30       # cap on coins counted for win bonus
    DEATH_PENALTY     = -3.0
    LIFE_LOST_PENALTY = -0.3


W = _CoinCollectorWeights   # short alias for use inside shape_reward


def shape_reward(info: Info, state: dict) -> Dict[str, float]:
    """
    Modular reward shaping function for the coin_collector persona.

    Parameters
    ----------
    info  : the full info dict from platformer_core._info()
    state : mutable per-env state dict managed by the factory closure below.
            Keys: 'combo_steps', 'combo_count', 'combo_total',
                  'prev_visit_counts' (set of tiles seen this ep)

    Returns
    -------
    Dict mapping component name → float reward.
    Sum of values = total shaped reward.

    How to tune
    -----------
    Edit _CoinCollectorWeights above and re-run training. Each component is
    logged separately via info["reward_components"] so you can see which signal
    dominates in TensorBoard.
    """
    components: Dict[str, float] = {}

    coins       = int(info.get("coins_delta", 0))
    total_coins = int(info.get("coins_collected", 0))
    won         = info.get("won", False)
    terminated  = info.get("terminated", False)
    life_lost   = info.get("life_lost", False)
    near_dist   = float(info.get("nearest_coin_dist", math.inf))
    visit_count = int(info.get("visit_count", 0))
    had_coin    = bool(info.get("had_coin_here", False))

    # ── Feature 2a: coin pickup reward ───────────────────────────────────────
    components["coin_pickup"] = W.COIN_PICKUP * coins

    # ── Feature 2b: proximity gradient ───────────────────────────────────────
    # Gives a small dense signal that draws the agent toward coins.
    # Capped at PROXIMITY_MIN_DIST to prevent explosion near coins.
    if not math.isinf(near_dist) and near_dist > 0:
        clamped = max(near_dist, W.PROXIMITY_MIN_DIST)
        components["coin_proximity"] = W.COIN_PROXIMITY / clamped
    else:
        components["coin_proximity"] = 0.0

    # ── Feature 2c: time penalty ──────────────────────────────────────────────
    components["time_penalty"] = W.TIME_PENALTY

    # ── Feature 2d: combo multiplier ─────────────────────────────────────────
    # Count consecutive steps since last coin; reset on pickup.
    # Trigger a flat bonus whenever agent collects coins close together in time.
    r_combo = 0.0
    if coins > 0:
        steps_gap = state.get("combo_steps", W.COMBO_WINDOW + 1)
        if steps_gap <= W.COMBO_WINDOW:
            combo_so_far = state.get("combo_total", 0.0)
            if combo_so_far < W.COMBO_MAX:
                r_combo = min(W.COMBO_BONUS, W.COMBO_MAX - combo_so_far)
                state["combo_total"] = combo_so_far + r_combo
        state["combo_steps"] = 0
    else:
        state["combo_steps"] = state.get("combo_steps", 0) + 1
    components["combo"] = r_combo

    # ── Feature 3a: novelty / exploration bonus ───────────────────────────────
    # Reward first visit to any tile (count-based exploration).
    seen = state.setdefault("prev_visit_counts", set())
    tile_key = (info.get("x_position", 0.0), info.get("y_position", 0.0))
    r_explore = 0.0
    if visit_count == 1 and tile_key not in seen:
        r_explore = W.EXPLORATION_BONUS
        seen.add(tile_key)
    components["exploration"] = r_explore

    # ── Feature 3b: backtracking bonus ────────────────────────────────────────
    # Small positive when agent returns to a tile that originally had a coin
    # (meaning it may have missed it and came back — smart map sweeping).
    # Guard: visit_count > 1 so we only reward the RE-visit, not the first pass.
    r_backtrack = 0.0
    if had_coin and visit_count > 1 and visit_count <= 4:
        r_backtrack = W.BACKTRACK_BONUS
    components["backtrack"] = r_backtrack

    # ── Feature 3c: stale-tile penalty ────────────────────────────────────────
    # Discourages the agent from camping or pacing the same few tiles.
    r_stale = 0.0
    if visit_count >= W.STALE_THRESH:
        r_stale = W.REVISIT_PENALTY
    components["stale"] = r_stale

    # ── Terminal signals ──────────────────────────────────────────────────────
    r_win = r_death = 0.0
    if won:
        r_win = W.WIN_BASE + min(total_coins, W.WIN_COIN_CAP) * W.WIN_COIN_SCALE
    elif terminated:
        r_death = W.DEATH_PENALTY
    elif life_lost:
        r_death = W.LIFE_LOST_PENALTY
    components["win"]   = r_win
    components["death"] = r_death

    return components


def _make_coin_collector_fn():
    """
    Factory that builds a stateful coin_collector reward callable.
    Called once per env instance so each parallel env has its own state dict
    and _ScoreTracker — identical isolation guarantee as _wrap_with_tracker.
    """
    tracker = _ScoreTracker()
    state   = {
        "combo_steps":       0,
        "combo_total":       0.0,
        "prev_visit_counts": set(),
    }

    def _reset_state():
        state["combo_steps"]       = 0
        state["combo_total"]       = 0.0
        state["prev_visit_counts"] = set()

    def reward_fn(obs, base: float, terminated: bool, info: Info) -> float:
        _score, _inc = tracker.step(info or {})
        components   = shape_reward(info or {}, state)
        info["reward_components"] = components
        total = float(sum(components.values()))
        if terminated or (info and info.get("terminated", False)):
            tracker.reset()
            _reset_state()
        return total

    reward_fn._core_fn    = None   # no @_wrap_with_tracker core, factory is self-contained
    reward_fn._is_factory = False  # signal to GameEnv: already a callable, not a factory
    return reward_fn


def coin_collector():
    """
    COIN COLLECTOR persona factory.

    Implements Features 2 & 3:
      • Feature 2 — Aggressive coin incentives
          - +COIN_PICKUP per coin collected
          - +COIN_PROXIMITY / dist  proximity gradient to nearest coin
          - TIME_PENALTY per step   discourages idling
          - COMBO_BONUS             reward for collecting coins in rapid succession
      • Feature 3 — Exploration / backtracking bonuses
          - +EXPLORATION_BONUS on first visit to any tile (count-based novelty)
          - +BACKTRACK_BONUS when revisiting a tile that originally held a coin
          - REVISIT_PENALTY when visit_count ≥ STALE_THRESH (anti-camping)

    All weights live in _CoinCollectorWeights — edit there to tune.
    The shape_reward() function is importable and testable in isolation.

    Returns a fresh stateful callable (not a factory) suitable for GameEnv.
    """
    return _make_coin_collector_fn()


# Mark as a factory so GameEnv calls coin_collector() to get the actual fn.
coin_collector._is_factory = True


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
    POTENTIAL_GAMMA = 0.99
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
    POTENTIAL_GAMMA = 0.99
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
platformer_simple        = simple
platformer_dijkstra      = delta_dijkstra
platformer_coin_hunter   = coin_hunter
platformer_coin_collector= coin_collector   # Feature 2 & 3 persona
platformer_enemy_hunter  = enemy_hunter
platformer_speedrunner   = speedrunner
platformer_completionist = completionist
default                  = simple