from __future__ import annotations
import random
import math
from typing import Callable, Tuple, Dict, Any, List

Info = Dict[str, Any]

# ---- Score Tracker Wrapper --------------------------------------------

class _ScoreTracker:
    """
    Tracks previous score, position, and goal distance for reward calculation.
    """
    def __init__(self):
        self.prev_score = 0
        self.last_dist = None
        self.last_x = None
        self.max_x = 0.0
        self.last_coins = 0
        self.last_kills = 0
    
    def step(self, info: Info) -> Tuple[int, bool]:
        # 1. Track Score & Increment
        current_score = int(info.get("score", 0))
        inc = current_score > self.prev_score
        self.prev_score = current_score

        # 2. Track Real Progress (Goal Distance Delta)
        current_dist = float(info.get("goal_dist", 0.0))
        if math.isinf(current_dist): current_dist = 0.0
        
        if self.last_dist is None:
            self.last_dist = current_dist
            
        progress = self.last_dist - current_dist
        self.last_dist = current_dist
        
        info["progress"] = progress
        
        if info.get("won", False) or info.get("terminated", False):
            info["progress"] = 0.0

        # 3. Track Frontier
        current_x = float(info.get("x_position", 0.0))
        if self.last_x is None:
            self.last_x = current_x
            self.max_x = current_x
        
        env_max = float(info.get("max_x_seen", 0.0))
        if env_max > 0:
            frontier_delta = max(0.0, env_max - self.max_x)
            self.max_x = env_max
        else:
            frontier_delta = max(0.0, current_x - self.max_x)
            self.max_x = max(self.max_x, current_x)
        info["frontier_dx"] = frontier_delta
        
        # 4. Track Coins
        current_coins = int(info.get("coins_collected", 0))
        info["coins_delta"] = max(0, current_coins - self.last_coins)
        self.last_coins = current_coins

        # 5. Track Kills
        if "enemies_killed_step" not in info:
            info["enemies_killed_step"] = 0
            
        return current_score, inc

    def reset(self):
        self.prev_score = 0
        self.last_dist = None
        self.last_x = None
        self.max_x = 0.0
        self.last_coins = 0
        self.last_kills = 0


def _wrap_with_tracker(core_fn) -> Callable:
    """
    Decorator that maintains the _ScoreTracker state.
    """
    tracker = _ScoreTracker()

    def reward(obs, base, terminated: bool, info: Info):
        # Step the tracker
        _score, inc = tracker.step(info or {})

        result = core_fn(inc, terminated, info or {}, _score)
        
        if terminated or (info and info.get("terminated", False)):
            tracker.reset()
        return result

    return reward


# ---- Reward Personas --------------------------------------------------

@_wrap_with_tracker
def simple(score_inc: bool, terminated: bool, info: Info, score: int) -> Dict[str, float]:
    """
    SIMPLE: Returns a dictionary of reward components.
    """
    # 1. Unpack Metrics
    progress = float(info.get("progress", 0.0))
    kills = int(info.get("enemies_killed_step", 0))
    coins = int(info.get("coins_delta", 0))
    won = info.get("won", False)
    current_x = float(info.get("x_position", 0.0))
    
    # 2. Calculate Components
    
    # A. Movement
    r_move = progress / 8.0 
    if progress < 0.001: 
        r_move -= 0.005 # Stall penalty

    # B. Actions
    r_coin = 5.0 * coins 
    r_kill = 2.5 * kills 

    # C. Win/Loss
    if won:
        if current_x > 200.0:
            r_win = 50.0
            r_death = 0.0
        else:
            r_win = 0.0
            r_death = -5.0 # Anti-cheese
    else:
        r_win = 0.0
        r_death = -5.0 if terminated else 0.0

    r_time = 0

    # 3. Return the Breakdown directly
    return {
        "movement": r_move,
        "coins":    r_coin,
        "kills":    r_kill,
        "win":      r_win,
        "time":     r_time,
        "death":    r_death
    }

# ---- ALIASES -----------------------------------------------------------
platformer_simple = simple
default = simple