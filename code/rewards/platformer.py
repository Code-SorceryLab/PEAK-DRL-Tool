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
        
        # Update records
        self.last_x = current_x
        self.max_x = max(self.max_x, current_x)

        # 3. Track Coins (coins_delta)
        current_coins = int(info.get("coins_collected", 0))
        info["coins_delta"] = max(0, current_coins - self.last_coins)
        self.last_coins = current_coins

        # 4. Track Kills
        if "enemies_killed_step" not in info:
            info["enemies_killed_step"] = 0
            
        # FIX: Ensure this tuple is returned to avoid the "NoneType" crash
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
        
        # Call the persona function with the CORRECT signature
        # We pass (inc, terminated, info, score)
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
    # If the agent wins but is still near the start (x < 200), it's the "Left Wall Glitch".
    # We revoke the win points and treat it as a death to discourage this.
    current_x = float(info.get("x_position", 0.0))
    
    if won:
        if current_x > 200.0:
            r_win = 50.0  # Real Win
            r_death = 0.0
        else:
            r_win = 0.0   # Fake Win (Glitch)
            r_death = -5.0 # Punish it like a death
    else:
        r_win = 0.0
        r_death = -5.0 if terminated else 0.0

    r_time = -0.01

    # 3. INJECT FOR DASHBOARD
    # Must match dashboard labels: [R:Move, R:Coin, R:Kill, R:Win]
    info["reward_components"] = [r_move, r_coin, r_kill, r_win]

    # 4. Total
    return r_move + r_coin + r_kill + r_win + r_time + r_death


@_wrap_with_tracker
def baseline(score_inc: bool, terminated: bool, info: Info, score: int) -> float:
    return random.random() - 0.5


@_wrap_with_tracker
def speedrunner(score_inc: bool, terminated: bool, info: Info, score: int) -> float:
    if terminated and not info.get("won", False):
        return -1.0

    dx = float(info.get("dx", 0.0))
    won = info.get("won", False)
    current_x = float(info.get("x_position", 0.0))
    
    r_move = (dx / 5.5) - 0.01
    
    # Anti-Cheese Win
    r_win = 0.0
    if won:
        r_win = 25.0 if current_x > 200 else -5.0
    
    info["reward_components"] = [r_move, 0.0, 0.0, r_win]
    return r_move + r_win


@_wrap_with_tracker
def coin_collector(score_inc: bool, terminated: bool, info: Info, score: int) -> float:
    if terminated and not info.get("won", False):
        return -0.25

    coins = int(info.get("coins_delta", 0))
    won = info.get("won", False)
    dx = float(info.get("dx", 0.0))
    current_x = float(info.get("x_position", 0.0))
    
    r_move = dx / 20.0
    r_coin = 3.0 * coins
    
    # Anti-Cheese Win
    r_win = 0.0
    if won:
        r_win = 20.0 if current_x > 200 else -5.0
    
    info["reward_components"] = [r_move, r_coin, 0.0, r_win]
    return r_move + r_coin + r_win - 0.003


@_wrap_with_tracker
def master(score_inc: bool, terminated: bool, info: Info, score: int) -> float:
    if terminated and not info.get("won", False):
        return -1.5

    dx = float(info.get("dx", 0.0))
    coins = int(info.get("coins_delta", 0))
    kills = int(info.get("enemies_killed_step", 0))
    won = info.get("won", False)
    current_x = float(info.get("x_position", 0.0))

    r_move = dx / 10.0 if dx > 0 else -0.015
    r_coin = 1.2 * coins
    r_kill = 2.0 * kills
    
    # Anti-Cheese Win
    r_win = 0.0
    if won:
        r_win = 10.0 if current_x > 200 else -5.0

    info["reward_components"] = [r_move, r_coin, r_kill, r_win]
    return r_move + r_coin + r_kill + r_win


@_wrap_with_tracker
def explorer(score_inc: bool, terminated: bool, info: Info, score: int) -> float:
    if terminated and not info.get("won", False):
        return -2.0 

    frontier_dx = float(info.get("frontier_dx", 0.0))
    won = info.get("won", False)
    current_x = float(info.get("x_position", 0.0))
    
    r_move = 0.0
    if frontier_dx > 0:
        r_move += frontier_dx / 5.0
    r_move -= 0.005
    
    # Anti-Cheese Win
    r_win = 0.0
    if won:
        r_win = 20.0 if current_x > 200 else -5.0
    
    info["reward_components"] = [r_move, 0.0, 0.0, r_win]
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
        
    info["reward_components"] = [r_move, 0.0, 0.0, 0.0]
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
    
    info["reward_components"] = [r_move, r_coin, r_kill, 0.0]
    return r_move + r_kill + r_coin - 0.01


# ---- ALIASES -----------------------------------------------------------
platformer_simple = simple
platformer_speedrunner = speedrunner
platformer_coin_collector = coin_collector
platformer_master = master
platformer_explorer = explorer
platformer_baseline = baseline
default = simple