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
        # BUG FIX: max_x was reset to current_x on life loss, meaning the agent
        # earned frontier reward again for ground it had already covered after
        # respawning behind its previous furthest point. max(0.0, ...) already
        # prevents negative deltas so the reset was redundant and harmful.
        # max_x now only ever increases — frontier reward requires genuinely new ground.
        current_x = float(info.get("x_position", 0.0))
        if self.last_x is None:
            self.last_x = current_x
            self.max_x  = current_x
        env_max = float(info.get("max_x_seen", 0.0))
        if env_max > 0:
            frontier_delta = max(0.0, env_max - self.max_x)
            self.max_x = max(self.max_x, env_max)
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
    DIJKSTRA: Primary signal is potential-based Dijkstra reward shaping.

    Designed to run after simple — the agent already knows how to move,
    this persona teaches it to move *optimally* along the Dijkstra gradient.

    Reward budget (approx, 3000-step winning episode):
      The potential-based reward telescopes approximately to:
        POTENTIAL_SCALE × [d_start + (1-γ) × avg_d × T]
        = 1.0 × [0.9 + 0.01 × 0.5 × 3000]
        = 1.0 × [0.9 + 15]
        ≈ 16.0  (step potential, varies with level layout)
      r_alignment ≈  0.5   (0.005 × 0.5 alignment × 60fps × moderate speed)
      r_win       = 15.0
      r_time      ≈ -1.5
      ─────────────────
      Total win   ≈ +30

    Losing episode (1500 steps, gets to d≈0.3):
      Step potential ≈ 1500 × 0.01 × 0.6 + 0.6 ≈ 9.6
      r_time        ≈ -0.75
      r_death       = -5.0
      ─────────────────
      Total loss    ≈ +3.85

    Winning is clearly better (+30 vs +4). The value function has a clean
    ~26-point gap to learn from, and episode variance will be much lower
    than simple because potential reward is bounded and continuous.

    Changes from original:
      - POTENTIAL_SCALE: 10.0 → 1.0
        Was producing 100-1500 in cumulative step potential, completely
        swamping the win bonus and inflating episode variance to 5000+.
        With scale=1.0 step potential contributes ~16 — meaningful but
        not dominant, win bonus remains the strongest terminal signal.
      - ALIGNMENT_SCALE: 0.05 → 0.005
        At 60fps the old value contributed up to 3.0/sec at full speed,
        comparable to potential. Now a supporting signal (~0.3/sec max)
        that nudges direction without competing with potential for dominance.
      - r_kill: 0.5 → 0.05   (incidental, don't let it distract navigation)
      - r_coin: 0.2 → 0.02   (incidental)
      - r_stall: -0.01 → -0.002
      - r_win:   20.0 → 15.0  (consistent with simple for comparable curves)
      - r_death: -15.0 → -5.0 (game over)
      - r_life:  -1.0  → -0.5 (life lost — lighter, jumping risk is fine)
      - POTENTIAL_GAMMA kept at 0.99 — must match PPO gamma in algo config.
        If you change PPO gamma, update this constant to match.
    """
    kills     = int(info.get("enemies_killed_step", 0))
    coins     = int(info.get("coins_delta", 0))
    won       = info.get("won", False)
    life_lost = info.get("life_lost", False)
    dijkstra_valid = info.get("dijkstra_valid", False)

    r_gradient = 0.0  # disabled — superseded by r_potential below

    # =========================================================================
    # Potential-Based Reward Shaping  (Ng et al. 1999)
    # =========================================================================
    # Φ(s) = -dijkstra_dist  (lower cost = closer to goal = higher potential)
    # F(s,s') = γ·Φ(s') - Φ(s) = prev_d - γ·curr_d
    #
    # Telescopes over a full episode to approximately:
    #   d_start + (1-γ)·sum(d_t)  ×  POTENTIAL_SCALE
    # With γ=0.99 the (1-γ) leak adds a small "living near goal" bonus that
    # is bounded and continuous — much lower variance than sparse win bonuses.
    #
    # POTENTIAL_GAMMA must match the PPO gamma in your algo config.
    POTENTIAL_GAMMA = 0.99
    POTENTIAL_SCALE = 1.0    # was 10.0 — see docstring for budget analysis

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
    # Bonus when movement direction aligns with cheapest reachable neighbour.
    # step_dx/step_dy come from _tracking_obs (8-direction Dijkstra search).
    # Guard: speed > 0.5 ignores jitter; dijkstra_valid ensures a valid path.
    # ALIGNMENT_SCALE: 0.005 × 60fps ≈ 0.3 max reward/sec — supporting signal,
    # intentionally weaker than r_potential.
    ALIGNMENT_SCALE = 0.005   # was 0.05 — see docstring for budget analysis

    r_alignment = 0.0
    if dijkstra_valid:
        vx    = float(info.get("velocity_x", 0.0))
        vy    = float(info.get("velocity_y", 0.0))
        speed = math.sqrt(vx * vx + vy * vy)
        if speed > 0.5:
            step_dx   = float(info.get("step_dx", 0.0))
            step_dy   = float(info.get("step_dy", 0.0))
            alignment = (vx / speed) * step_dx + (vy / speed) * step_dy
            r_alignment = max(0.0, alignment) * ALIGNMENT_SCALE
    # =========================================================================
    # End Velocity Alignment
    # =========================================================================

    r_coin  = 0.002 * coins   # incidental — navigation takes priority
    r_kill  = 0.005 * kills   # incidental — don't let kills distract the gradient

    r_stall = -0.05 if bool(info.get("stalled", False)) else 0.0

    r_win, r_death = 0.0, 0.0
    if won:
        r_win  = 1.50              # consistent with simple for comparable TB curves
    elif terminated:
        r_death = -0.50             # game over
    elif life_lost:
        r_death = -0.05             # life lost — kept light so jumping risk is acceptable

    return {
        "gradient":   r_gradient,
        "potential":  r_potential,
        "alignment":  r_alignment,
        "stall":      r_stall,
        "coins":      r_coin,
        "kills":      r_kill,
        "win":        r_win,
        "time":       -0.00005,     # -1.5 over 3000 steps — consistent with simple
        "death":      r_death,
    }


@_wrap_with_tracker
def simple(score_inc: bool, terminated: bool, info: Info, score: int) -> Dict[str, float]:
    """
    SIMPLE: Euclidean progress + frontier exploration bonus.

    Deliberately lightweight — good baseline and first training stage before
    delta_dijkstra. All signals are rescaled so total episode returns land
    in roughly [-5, +20], giving VecNormalize a stable distribution to work
    with and keeping the win bonus clearly the dominant terminal signal.

    Reward budget (approx, 3000-step winning episode):
      r_move     ≈  1.5   (0.02 × 0.05 tiles/step × 3000 × 50% moving)
      r_frontier ≈  0.6   (dries up after early exploration)
      r_win      = 15.0   (dominant terminal signal)
      r_time     ≈ -1.5   (-0.0005 × 3000)
      ─────────────────
      Total win  ≈ +15.6

    Losing episode (1500 steps):
      r_move     ≈  0.5
      r_time     ≈ -0.75
      r_death    = -5.0   (game over) or -1.5 (life lost × 3)
      ─────────────────
      Total loss ≈ -5.25

    The ~20-point gap between winning and losing is large enough to give
    the value function a clear signal while keeping variance manageable.

    Changes from original:
      - r_win:      200 → 15    (was dominating all step rewards by 100×)
      - r_move:     ×0.15 → ×0.02
      - r_frontier: ×0.2  → ×0.03
      - r_coin:     ×0.05 → ×0.01
      - r_kill:     ×0.5  → ×0.1
      - r_death:    -15   → -5  (game over), -3 → -1.5 (life lost)
      - r_time:     -0.002 → -0.0005  (was -6 over long episodes, > win bonus)
      - Removed inline stall penalty (abs(progress)<0.5 branch) — this fired
        on every jump frame and silently punished vertical movement. The env's
        own anti-stall system already handles genuine stalling.
      - Removed backtrack multiplier (progress<0 × 2.5) — the base negative
        r_move already penalises backward movement; doubling it over-punished
        legitimate repositioning (e.g. backing up to get a run-up).
    """
    progress  = float(info.get("progress", 0.0))
    frontier  = float(info.get("frontier_dx", 0.0))
    kills     = int(info.get("enemies_killed_step", 0))
    coins     = int(info.get("coins_delta", 0))
    won       = info.get("won", False)
    life_lost = info.get("life_lost", False)

    # Forward/backward movement signal — linear, no multipliers
    r_move = progress * 0.0015

    # Frontier: reward genuinely new ground only (bug-fixed in tracker)
    r_frontier = frontier * 0.003

    r_coin = 0.001 * coins
    r_kill = 0.010 * kills
    r_stall = -0.05 if bool(info.get("stalled", False)) else 0.0
    r_win, r_death = 0.0, 0.0
    if won:
        r_win  = 1.50              # Clearly dominant — winning must be the best outcome
    elif terminated:
        r_death = -0.50             # Game over — meaningfully bad
    elif life_lost:
        r_death = -0.15             # Life lost — hurts but not catastrophic

    return {
        "movement": r_move,
        "frontier": r_frontier,
        "coins":    r_coin,
        "kills":    r_kill,
        "win":      r_win,
        "stall":    r_stall,
        "time":     -0.0005,       # -1.5 over 3000 steps — modest pressure, won't dominate
        "death":    r_death,
    }


@_wrap_with_tracker
def coin_hunter(score_inc: bool, terminated: bool, info: Info, score: int) -> Dict[str, float]:
    """
    COIN HUNTER: Maximise coin collection.

    KEY DESIGN: NO dijkstra potential. The goal gradient is removed so the
    agent cannot "cheat" by just running to the goal. The ONLY consistent
    positive signal comes from collecting coins.

    Without a goal gradient, the agent must learn to navigate by following
    the coin channel in its observation grid. Frontier reward provides a
    weak nudge to explore new territory (where uncollected coins are).
    A small survival bonus keeps the agent alive long enough to find coins.
    Win bonus scales with total coins collected.
    """
    kills     = int(info.get("enemies_killed_step", 0))
    coins     = int(info.get("coins_delta", 0))
    total_coins = int(info.get("coins_collected", 0))
    won       = info.get("won", False)
    life_lost = info.get("life_lost", False)

    # ── Primary signal: coins (the ONLY strong positive) ───────────────
    r_coin = 5.0 * coins

    # ── Exploration: reward visiting new ground (where uncollected coins live)
    frontier = float(info.get("frontier_dx", 0.0))
    r_frontier = frontier * 0.3

    # ── Survival bonus: staying alive = more time to find coins ────────
    r_alive = 0.002

    # ── NO dijkstra potential — deliberately omitted ───────────────────
    # r_potential = 0.0  (not computed at all)

    r_kill = 0.2 * kills           # Incidental, not a focus

    r_stall = -0.015 if bool(info.get("stalled", False)) else 0.0

    r_win, r_death = 0.0, 0.0
    if won:
        # Win bonus scales with coins collected — completing with 0 coins
        # is worth almost nothing; completing with 20+ coins is huge.
        r_win = 5.0 + min(total_coins, 30) * 1.0   # 5 base + up to 30 bonus
    elif terminated:
        r_death = -15.0
    elif life_lost:
        r_death = -2.0

    return {
        "coins":     r_coin,
        "frontier":  r_frontier,
        "alive":     r_alive,
        "kills":     r_kill,
        "stall":     r_stall,
        "win":       r_win,
        "time":      -0.0001,      # Near-zero: don't pressure speed
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
    r_kill = 8.0 * kills

    # ── Powerup bonus: being big = safer stomps ────────────────────────
    r_powerup = 0.008 if powered_up else 0.0

    # ── Exploration: move forward to find new enemies ──────────────────
    frontier = float(info.get("frontier_dx", 0.0))
    r_frontier = frontier * 0.25

    # ── Survival bonus: alive = more chances to kill ───────────────────
    r_alive = 0.001

    # ── NO dijkstra potential — deliberately omitted ───────────────────

    r_coin = 0.1 * coins           # Incidental

    r_stall = -0.015 if bool(info.get("stalled", False)) else 0.0

    r_win, r_death = 0.0, 0.0
    if won:
        r_win  = 10.0              # Moderate: finishing is fine but not the goal
    elif terminated:
        r_death = -8.0             # Lighter: aggressive play = more deaths expected
    elif life_lost:
        r_death = -0.5             # Very light: dying to attempt a kill is acceptable

    return {
        "kills":     r_kill,
        "powerup":   r_powerup,
        "frontier":  r_frontier,
        "alive":     r_alive,
        "coins":     r_coin,
        "stall":     r_stall,
        "win":       r_win,
        "time":      -0.0001,      # Near-zero: no rush, find enemies
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

    # ── Primary signal: rightward velocity ─────────────────────────────
    vx = float(info.get("velocity_x", 0.0))
    r_velocity = max(0.0, vx) * 0.05    # Only reward rightward, ignore leftward
    # Penalise standing still or moving left
    if vx < 0.5:
        r_velocity -= 0.01

    # ── Moderate dijkstra potential for direction ──────────────────────
    POTENTIAL_GAMMA = 0.99
    POTENTIAL_SCALE = 8.0

    r_potential = 0.0
    curr_d = float(info.get("dijkstra_dist", -1.0))
    prev_d = float(info.get("dijkstra_dist_prev", -1.0))
    if dijkstra_valid and prev_d >= 0.0:
        r_potential = (prev_d - POTENTIAL_GAMMA * curr_d) * POTENTIAL_SCALE

    # ── Backtrack penalty ──────────────────────────────────────────────
    progress = float(info.get("progress", 0.0))
    r_backtrack = min(0.0, progress) * 0.5  # Only when progress < 0

    r_coin = 0.0                   # Zero: don't waste time on coins
    r_kill = 0.0                   # Zero: don't waste time on enemies

    r_stall = -0.04 if bool(info.get("stalled", False)) else 0.0

    r_win, r_death = 0.0, 0.0
    if won:
        r_win  = 40.0              # Massive: the whole point
    elif terminated:
        r_death = -10.0
    elif life_lost:
        r_death = -2.0

    return {
        "velocity":  r_velocity,
        "potential": r_potential,
        "backtrack": r_backtrack,
        "stall":     r_stall,
        "win":       r_win,
        "time":      -0.005,       # 10× vs dijkstra — brutal time pressure
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

    # ── Moderate dijkstra potential (same magnitude as coin/kill events) ─
    POTENTIAL_GAMMA = 0.99
    POTENTIAL_SCALE = 6.0          # Weaker than dijkstra(10) / speedrunner(8)

    r_potential = 0.0
    curr_d = float(info.get("dijkstra_dist", -1.0))
    prev_d = float(info.get("dijkstra_dist_prev", -1.0))
    if dijkstra_valid and prev_d >= 0.0:
        r_potential = (prev_d - POTENTIAL_GAMMA * curr_d) * POTENTIAL_SCALE

    # ── All objectives at meaningful weight ────────────────────────────
    r_coin     = 3.0 * coins       # Strong (between dijkstra 0.2 and coin_hunter 5.0)
    r_kill     = 4.0 * kills       # Strong (between dijkstra 0.5 and enemy_hunter 8.0)
    r_frontier = float(info.get("frontier_dx", 0.0)) * 0.2

    r_stall = -0.015 if bool(info.get("stalled", False)) else 0.0

    r_win, r_death = 0.0, 0.0
    if won:
        # Bonus scales with how "complete" the run was
        r_win = 15.0 + min(total_coins, 30) * 0.5
    elif terminated:
        r_death = -20.0            # Harsh: completionists don't die
    elif life_lost:
        r_death = -4.0             # Strictest life penalty

    return {
        "potential": r_potential,
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