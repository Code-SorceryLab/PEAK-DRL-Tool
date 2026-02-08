from __future__ import annotations
import random
import math
from typing import Callable, Tuple, Dict, Any, List

Info = Dict[str, Any]

# ---- Score Tracker Wrapper --------------------------------------------

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
        # 1. Track Score & Increment
        current_score = int(info.get("score", 0))
        inc = current_score > self.prev_score
        self.prev_score = current_score

        # 2. Track Position (dx)
        current_x = float(info.get("x_position", 0.0))
        if self.last_x is None:
            self.last_x = current_x
            self.max_x = current_x

        dx = current_x - self.last_x
        info["dx"] = dx
        
        # Frontier (max distance seen)
        env_max = float(info.get("max_x_seen", 0.0))
        if env_max > 0:
            frontier_delta = max(0.0, env_max - self.max_x)
            self.max_x = env_max
        else:
            frontier_delta = max(0.0, current_x - self.max_x)
            self.max_x = max(self.max_x, current_x)
        info["frontier_dx"] = frontier_delta
        
        # 3. Track Coins (coins_delta)
        current_coins = int(info.get("coins_collected", 0))
        info["coins_delta"] = max(0, current_coins - self.last_coins)
        self.last_coins = current_coins

        # 4. Track Kills
        if "enemies_killed_step" not in info:
            info["enemies_killed_step"] = 0
            
        return current_score, inc

    def reset(self):
        self.prev_score = 0
        self.last_x = None
        self.max_x = 0.0
        self.last_coins = 0
        self.last_kills = 0


def _wrap_with_tracker(core_fn) -> Callable:
    """
    Decorator that maintains the _ScoreTracker state across steps.
    """
    tracker = _ScoreTracker()

    def reward(obs, base, terminated: bool, info: Info) -> float:
        # Step the tracker and get the required tuple
        _score, inc = tracker.step(info or {})
        
        # Call the persona function
        r = float(core_fn(inc, terminated, info or {}, _score))
        
        if terminated or (info and info.get("terminated", False)):
            tracker.reset()
        return r

    return reward


# ---- Reward Personas --------------------------------------------------

@_wrap_with_tracker
def simple(score_inc: bool, terminated: bool, info: Info, score: int) -> float:
    """
    SIMPLE: Balanced explorer + ANTI-CHEESE protection.
    """
    # 1. Unpack Metrics
    dx = float(info.get("dx", 0.0))
    kills = int(info.get("enemies_killed_step", 0))
    coins = int(info.get("coins_delta", 0))
    won = info.get("won", False)
    
    # 2. Calculate Components
    # A. Movement
    r_move = dx / 8.0 
    if dx < 0.001: 
        r_move -= 0.005 # Stall penalty

    # B. Actions
    r_coin = 5.0 * coins 
    r_kill = 2.5 * kills 

    # C. ANTI-CHEESE WIN CONDITION
    current_x = float(info.get("x_position", 0.0))
    if won:
        if current_x > 200.0:
            r_win = 50.0  # Real Win
            r_death = 0.0
        else:
            r_win = 0.0   # Fake Win (Left Wall Glitch)
            r_death = -5.0 # Punish it
    else:
        r_win = 0.0
        r_death = -5.0 if terminated else 0.0

    r_time = -0.01

    # 3. DICTIONARY INJECTION FOR DASHBOARD
    # Keys must match dashboard configuration
    info["reward_components"] = {
        "movement": r_move,
        "coins":    r_coin,
        "kills":    r_kill,
        "win":      r_win,
        "time":     r_time,
        "death":    r_death
    }

    # 4. Total
    return r_move + r_coin + r_kill + r_win + r_time + r_death


@_wrap_with_tracker
def speedrunner(score_inc: bool, terminated: bool, info: Info, score: int) -> float:
    dx = float(info.get("dx", 0.0))
    won = info.get("won", False)
    current_x = float(info.get("x_position", 0.0))
    
    r_move = (dx / 5.5) - 0.01
    
    r_win = 0.0
    if won:
        r_win = 25.0 if current_x > 200 else -5.0
    
    info["reward_components"] = {
        "movement": r_move,
        "win": r_win
    }
    return r_move + r_win


@_wrap_with_tracker
def coin_collector(score_inc: bool, terminated: bool, info: Info, score: int) -> float:
    coins = int(info.get("coins_delta", 0))
    won = info.get("won", False)
    dx = float(info.get("dx", 0.0))
    current_x = float(info.get("x_position", 0.0))
    
    r_move = dx / 20.0
    r_coin = 3.0 * coins
    r_time = -0.003
    
    r_win = 0.0
    if won:
        r_win = 20.0 if current_x > 200 else -5.0
    
    info["reward_components"] = {
        "movement": r_move,
        "coins": r_coin,
        "win": r_win,
        "time": r_time
    }
    return r_move + r_coin + r_win + r_time


@_wrap_with_tracker
def master(score_inc: bool, terminated: bool, info: Info, score: int) -> float:
    dx = float(info.get("dx", 0.0))
    coins = int(info.get("coins_delta", 0))
    kills = int(info.get("enemies_killed_step", 0))
    won = info.get("won", False)
    current_x = float(info.get("x_position", 0.0))

    r_move = dx / 10.0 if dx > 0 else -0.015
    r_coin = 1.2 * coins
    r_kill = 2.0 * kills
    
    r_win = 0.0
    if won:
        r_win = 10.0 if current_x > 200 else -5.0

    info["reward_components"] = {
        "movement": r_move,
        "coins": r_coin,
        "kills": r_kill,
        "win": r_win
    }
    return r_move + r_coin + r_kill + r_win


@_wrap_with_tracker
def explorer(score_inc: bool, terminated: bool, info: Info, score: int) -> float:
    frontier_dx = float(info.get("frontier_dx", 0.0))
    won = info.get("won", False)
    current_x = float(info.get("x_position", 0.0))
    
    r_move = 0.0
    if frontier_dx > 0:
        r_move += frontier_dx / 5.0
    r_move -= 0.005
    
    r_win = 0.0
    if won:
        r_win = 20.0 if current_x > 200 else -5.0
    
    info["reward_components"] = {
        "movement": r_move,
        "win": r_win
    }
    return r_move + r_win


@_wrap_with_tracker
def platformer_momentum(score_inc: bool, terminated: bool, info: Info, score: int) -> float:
    if terminated:
        return 10.0 if info.get("won") else -5.0

    vx = float(info.get("velocity_x", 0.0))
    dx = float(info.get("dx", 0.0))
    
    norm_v = vx / 240.0
    r_move = norm_v * 0.5
    
    if abs(vx) < 5.0:
        r_move -= 0.05
    if dx < 0:
        r_move -= 0.1
        
    info["reward_components"] = {"movement": r_move}
    return r_move


@_wrap_with_tracker
def platformer_dense(score_inc: bool, terminated: bool, info: Info, score: int) -> float:
    if terminated:
        return 15.0 if info.get("won", False) else -5.0

    dx = float(info.get("dx", 0.0))
    kills = int(info.get("enemies_killed_step", 0))
    coins = int(info.get("coins_delta", 0))
    
    r_move = dx / 5.0
    r_kill = kills * 0.5
    r_coin = coins * 0.2
    
    info["reward_components"] = {
        "movement": r_move,
        "coins": r_coin,
        "kills": r_kill
    }
    return r_move + r_kill + r_coin - 0.01


# ---- ALIASES -----------------------------------------------------------
platformer_simple = simple
platformer_speedrunner = speedrunner
platformer_coin_collector = coin_collector
platformer_master = master
platformer_explorer = explorer

default = simple