from __future__ import annotations
import math
from typing import Callable, Tuple, Dict, Any

Info = Dict[str, Any]

# ---- Score Tracker -------------------------------------------------------

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
        if math.isinf(current_dist): current_dist = 0.0
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
        current_dijkstra = float(info.get("dijkstra_dist", 1.0))
        if self.last_dijkstra is None:
            self.last_dijkstra = current_dijkstra
        if life_lost:
            dijkstra_progress = 0.0
            self.last_dijkstra = current_dijkstra
        else:
            dijkstra_progress = self.last_dijkstra - current_dijkstra
            self.last_dijkstra = current_dijkstra
        info["dijkstra_progress"] = dijkstra_progress

        if info.get("won", False) or info.get("terminated", False):
            info["progress"] = 0.0
            info["dijkstra_progress"] = 0.0

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


# ---- Reward Personas -----------------------------------------------------

@_wrap_with_tracker
def delta_dijkstra(score_inc: bool, terminated: bool, info: Info, score: int) -> Dict[str, float]:
    """DIJKSTRA: Primary signal is Dijkstra distance improvement toward goal."""
    d_prog    = float(info.get("dijkstra_progress", 0.0))
    kills     = int(info.get("enemies_killed_step", 0))
    coins     = int(info.get("coins_delta", 0))
    won       = info.get("won", False)
    life_lost = info.get("life_lost", False)

    r_gradient = d_prog * 100.0
    if d_prog <= 0.0:
        r_gradient -= 0.002   # stall penalty per frame

    r_coin = 0.2 * coins      # small so coins don't compete with forward progress
    r_kill = 0.5 * kills

    r_win, r_death = 0.0, 0.0
    if won:
        r_win  = 200.0
    elif terminated:
        r_death = -15.0
    elif life_lost:
        r_death = -3.0

    return {"gradient": r_gradient, "coins": r_coin, "kills": r_kill,
            "win": r_win, "time": -0.0005, "death": r_death}


@_wrap_with_tracker
def complex_navigation(score_inc: bool, terminated: bool, info: Info, score: int) -> Dict[str, float]:
    """COMPLEX: Euclidean progress + proximity multiplier."""
    progress  = float(info.get("progress", 0.0))
    goal_dist = float(info.get("goal_dist", 0.0))
    kills     = int(info.get("enemies_killed_step", 0))
    coins     = int(info.get("coins_delta", 0))
    won       = info.get("won", False)
    life_lost = info.get("life_lost", False)

    proximity_mult = 1.0 + (400.0 / (goal_dist + 200.0))
    r_move = (progress * 0.03) * proximity_mult
    if abs(progress) < 0.5:
        r_move -= 0.003

    r_coin = 0.2 * coins
    r_kill = 0.5 * kills

    r_win, r_death = 0.0, 0.0
    if won:
        r_win  = 200.0
    elif terminated:
        r_death = -15.0
    elif life_lost:
        r_death = -3.0

    return {"movement": r_move, "coins": r_coin, "kills": r_kill,
            "win": r_win, "time": -0.001, "death": r_death}


@_wrap_with_tracker
def simple(score_inc, terminated, info, score):
    progress  = float(info.get("progress", 0.0))
    frontier  = float(info.get("frontier_dx", 0.0))   # NEW
    kills     = int(info.get("enemies_killed_step", 0))
    coins     = int(info.get("coins_delta", 0))
    won       = info.get("won", False)
    life_lost = info.get("life_lost", False)

    # Stronger forward signal, penalize going backwards
    r_move = progress * 0.15          # 3x stronger
    if progress < 0:
        r_move *= 2.5                 # extra backtrack penalty
    if abs(progress) < 0.5:
        r_move -= 0.01                # stronger stall penalty

    r_frontier = frontier * 0.2       # reward exploring new ground only
    r_coin = 0.05 * coins             # 4x weaker coins so they don't hijack policy
    r_kill = 0.5 * kills

    r_win, r_death = 0.0, 0.0
    if won:
        r_win  = 200.0
    elif terminated:
        r_death = -15.0
    elif life_lost:
        r_death = -3.0

    return {"movement": r_move, "frontier": r_frontier, "coins": r_coin,
            "kills": r_kill, "win": r_win, "time": -0.002, "death": r_death}


# ---- Aliases -------------------------------------------------------------
platformer_simple   = simple
platformer_complex  = complex_navigation
platformer_dijkstra = delta_dijkstra
default             = simple
