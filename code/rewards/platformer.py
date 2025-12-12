# code/rewards/mario.py
from __future__ import annotations
import random
import math
from typing import Callable, Tuple, Dict, Any

Info = Dict[str, Any]

# ---- Score tracking utility -------------------------------------------

class _ScoreTracker:
    """
    Tracks previous score, position, and specific metrics for reward calculation.
    """
    def __init__(self):
        self.prev_score = 0
        self.last_x = None
        self.max_x = 0.0
        self.last_coins = 0
        self.last_kills = 0

    def step(self, info: Info) -> Tuple[int, bool]:
        # --- score increment flag
        score = int(info.get("score", info.get("total_reward", 0)) or 0)
        inc = score > self.prev_score

        # --- progress deltas ---------------------------------------
        current_x = float(info.get("x_position", 0.0))
        
        # Initialize last_x if this is the first step
        if self.last_x is None:
            self.last_x = current_x
            self.max_x = current_x

        # Standard forward progress (dx)
        dx = current_x - self.last_x
        info["dx"] = dx
        
        # "Frontier" progress (did we reach new ground?)
        env_max = float(info.get("max_x_seen", 0.0))
        if env_max > 0:
            frontier_delta = max(0.0, env_max - self.max_x)
            self.max_x = env_max
        else:
            frontier_delta = max(0.0, current_x - self.max_x)
            self.max_x = max(self.max_x, current_x)
            
        info["frontier_dx"] = frontier_delta

        # --- coins delta -------------------------------------------------
        coins = int(info.get("coins_collected", 0) or 0)
        info["coins_delta"] = max(0, coins - self.last_coins)

        # --- kills per step --------------------------------
        kills_now = 0
        if "enemies_killed" in info:
            kills_now = int(info.get("enemies_killed") or 0)
        elif "kills" in info:
            kills_now = int(info.get("kills") or 0)
        
        if "enemies_killed_step" in info:
            kills_step = int(info["enemies_killed_step"])
        else:
            kills_step = max(0, kills_now - self.last_kills)
            
        info["kills_step"] = kills_step

        # --- update trackers --------------------------------------------
        self.prev_score = score
        self.last_x = current_x
        self.last_coins = coins
        self.last_kills = kills_now

        return score, inc

    def reset(self):
        self.prev_score = 0
        self.last_x = None
        self.max_x = 0.0
        self.last_coins = 0
        self.last_kills = 0


def _wrap_with_tracker(core_fn) -> Callable:
    tracker = _ScoreTracker()

    def reward(obs, base, terminated: bool, info: Info) -> float:
        _score, inc = tracker.step(info or {})
        r = float(core_fn(inc, terminated, info or {}, _score))
        if terminated or (info and info.get("terminated", False)):
            tracker.reset()
        return r

    return reward


# ---- Personas ----------------------------------------------------------

@_wrap_with_tracker
def baseline(score_inc: bool, terminated: bool, info: Info, score: int) -> float:
    """Random baseline for sanity checks."""
    return random.random() - 0.5


@_wrap_with_tracker
def simple(score_inc: bool, terminated: bool, info: Info, score: int) -> float:
    """
    SIMPLE: Gentle forward shaping + coins/kills; tiny time tax.
    """
    if terminated and not info.get("won", False):
        return -0.5

    dx = float(info.get("dx", 0.0))
    coins = int(info.get("coins_delta", 0))
    kills = int(info.get("kills_step", 0))

    r = -0.005
    r += dx / 8.0
    r += 0.5 * coins
    r += 1.0 * kills
    if dx < 0.05:
        r -= 0.005
    return r


@_wrap_with_tracker
def speedrunner(score_inc: bool, terminated: bool, info: Info, score: int) -> float:
    """
    SPEEDRUNNER: Strong progress pressure, light time tax.
    """
    if terminated and not info.get("won", False):
        return -1.0

    dx = float(info.get("dx", 0.0))
    r = (dx / 5.5) - 0.01

    if info.get("won", False) or info.get("goal_reached", False):
        r += 25.0
    return r


@_wrap_with_tracker
def coin_collector(score_inc: bool, terminated: bool, info: Info, score: int) -> float:
    """
    COIN COLLECTOR: Prioritizes grabbing coins.
    """
    if terminated and not info.get("won", False):
        return -0.25

    coins = int(info.get("coins_delta", 0))
    r = -0.003
    r += 3.0 * coins
    return r


@_wrap_with_tracker
def master(score_inc: bool, terminated: bool, info: Info, score: int) -> float:
    """
    MASTER: Balanced—progress, coins, and kills.
    """
    if terminated and not info.get("won", False):
        return -1.5

    dx = float(info.get("dx", 0.0))
    coins = int(info.get("coins_delta", 0))
    kills = int(info.get("kills_step", 0))
    powered = 1.0 if info.get("powered_up") else 0.0

    r = 0.0
    r += 1.2 * coins
    r += 2.0 * kills
    r += 0.002 * powered
    if dx < 0.05:
        r -= 0.015
    if info.get("won", False):
        r += 10.0
    return r


@_wrap_with_tracker
def explorer(score_inc: bool, terminated: bool, info: Info, score: int) -> float:
    """
    EXPLORER: Solves "Stuck at Pit".
    Rewards only NEW territory (frontier_dx).
    """
    if terminated and not info.get("won", False):
        return -2.0 

    frontier_dx = float(info.get("frontier_dx", 0.0))
    
    r = 0.0
    if frontier_dx > 0:
        r += frontier_dx / 5.0
    
    r -= 0.005
    if info.get("won", False):
        r += 20.0
    return r


@_wrap_with_tracker
def platformer_momentum(score_inc: bool, terminated: bool, info: Info, score: int) -> float:
    """
    MOMENTUM: Rewards maintaining high velocity.
    Solves "stutter-stepping".
    """
    if terminated:
        return 10.0 if info.get("won") else -5.0

    vx = float(info.get("velocity_x", 0.0))
    dx = float(info.get("dx", 0.0))
    
    norm_v = vx / 240.0
    r = norm_v * 0.5
    
    if abs(vx) < 5.0:
        r -= 0.05
    if dx < 0:
        r -= 0.1
    return r


@_wrap_with_tracker
def platformer_dense(score_inc: bool, terminated: bool, info: Info, score: int) -> float:
    """
    DENSE: High performance 'PPO Standard'.
    """
    if terminated:
        return 15.0 if info.get("won", False) else -5.0

    dx = float(info.get("dx", 0.0))
    kills = int(info.get("kills_step", 0))
    coins = int(info.get("coins_delta", 0))
    
    r = 0.0
    r += dx / 5.0  
    r += kills * 0.5
    r += coins * 0.2
    r -= 0.01 
    
    return r