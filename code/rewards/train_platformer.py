from __future__ import annotations
import random
import math
from typing import Callable, Tuple, Dict, Any, List
import numpy as np
Info = Dict[str, Any]

# ---- Score Tracker Wrapper --------------------------------------------

class _ScoreTracker:
    """
    Tracks previous score, position, goal distance, and dijkstra cost.
    FIX: No longer zeros out progress on death - agent must feel the cost.
    """
    def __init__(self):
        self.prev_score = 0
        self.last_dist = None  # Euclidean distance
        self.last_dijkstra = None # Dijkstra distance
        self.last_x = None
        self.max_x = 0.0
        self.last_coins = 0
        self.last_lives = None # Track lives to detect soft resets
    
    def step(self, info: Info) -> Tuple[int, bool]:
        # 0. Check for Life Loss (Soft Reset)
        current_lives = int(info.get("lives", 3))
        
        # Initialize last_lives on first step
        if self.last_lives is None:
            self.last_lives = current_lives
            
        life_lost = current_lives < self.last_lives
        self.last_lives = current_lives
        
        # Inject life_lost into info for the persona to use
        info["life_lost"] = life_lost

        # 1. Track Score & Increment
        current_score = int(info.get("score", 0))
        inc = current_score > self.prev_score
        self.prev_score = current_score

        # 2. Track Real Progress (Euclidean)
        current_dist = float(info.get("goal_dist", 0.0))
        if math.isinf(current_dist): current_dist = 0.0
        
        if self.last_dist is None:
            self.last_dist = current_dist
            
        # FIX: Allow negative progress - agent MUST feel cost of death
        progress = self.last_dist - current_dist
        self.last_dist = current_dist
        
        info["progress"] = progress
        
        # 3. Track Dijkstra Progress
        current_dijkstra = float(info.get("dijkstra_dist", 1.0)) # Normalized 0-1
        
        if self.last_dijkstra is None:
            self.last_dijkstra = current_dijkstra
            
        # FIX: Allow negative dijkstra progress - agent must feel cost
        dijkstra_progress = self.last_dijkstra - current_dijkstra
        self.last_dijkstra = current_dijkstra
        
        info["dijkstra_progress"] = dijkstra_progress

        if info.get("won", False) or info.get("terminated", False):
            info["progress"] = 0.0
            info["dijkstra_progress"] = 0.0

        # 4. Track Frontier
        current_x = float(info.get("x_position", 0.0))
        if self.last_x is None:
            self.last_x = current_x
            self.max_x = current_x
        
        # Keep max_x on death to encourage returning to frontier
        env_max = float(info.get("max_x_seen", 0.0))
        if env_max > 0:
            frontier_delta = max(0.0, env_max - self.max_x)
            self.max_x = env_max
        else:
            frontier_delta = max(0.0, current_x - self.max_x)
            self.max_x = max(self.max_x, current_x)
        info["frontier_dx"] = frontier_delta
        
        # 5. Track Coins
        current_coins = int(info.get("coins_collected", 0))
        if current_coins < self.last_coins:
            self.last_coins = current_coins
        info["coins_delta"] = max(0, current_coins - self.last_coins)
        self.last_coins = current_coins

        # 6. Track Kills
        if "enemies_killed_step" not in info:
            info["enemies_killed_step"] = 0
            
        return current_score, inc

    def reset(self):
        self.prev_score = 0
        self.last_dist = None
        self.last_dijkstra = None
        self.last_x = None
        self.max_x = 0.0
        self.last_coins = 0
        self.last_lives = None


def _wrap_with_tracker(core_fn) -> Callable:
    """
    Decorator that maintains the _ScoreTracker state.
    FIX: Properly passes truncated parameter.
    """
    tracker = _ScoreTracker()

    def reward(obs, base, terminated: bool, truncated: bool, info: Info) -> float:
        # Step the tracker
        _score, inc = tracker.step(info or {})

        # CRITICAL FIX: Pass truncated to core function
        components = core_fn(inc, terminated, truncated, info or {}, _score)
        
        # Inject breakdown into info for dashboard
        info["reward_components"] = components
        
        # Sum components to get the actual float reward for PPO
        total_reward = sum(components.values())

        if terminated or truncated or (info and info.get("terminated", False)):
            tracker.reset()
            
        return np.clip(float(total_reward), -100.0, 100.0)

    return reward


# ---- Reward Personas --------------------------------------------------

@_wrap_with_tracker
def delta_dijkstra(score_inc: bool, terminated: bool, truncated: bool, info: Info, score: int) -> Dict[str, float]:
    """
    DELTA DIJKSTRA - REBALANCED
    Primary driver is improvement in Dijkstra Distance.
    """
    # 1. Unpack Metrics
    d_prog = float(info.get("dijkstra_progress", 0.0))
    kills = int(info.get("enemies_killed_step", 0))
    coins = int(info.get("coins_delta", 0))
    won = info.get("won", False)
    life_lost = info.get("life_lost", False)
    
    # 2. Gradient Reward - SCALED for visibility
    r_gradient = d_prog * 500.0  # Was 100, now 500
    
    # 3. Interaction Rewards
    r_coin = 2.0 * coins  # Was 1.0
    r_kill = 1.0 * kills
    
    # 4. Win/Loss/Life - PROPERLY BALANCED
    r_win = 0.0
    r_death = 0.0
    
    if won:
        r_win = 200.0  # Was 100
    elif terminated:
        r_death = -50.0  # Was -2.0 (CRITICAL FIX)
    elif truncated:
        r_death = 0.0  # No penalty for timeout
    elif life_lost:
        r_death = -10.0  # Was -1.0
        
    r_time = -0.001

    return {
        "gradient": r_gradient,
        "coins":    r_coin,
        "kills":    r_kill,
        "win":      r_win,
        "time":     r_time,
        "death":    r_death
    }

@_wrap_with_tracker
def complex_navigation(score_inc: bool, terminated: bool, truncated: bool, info: Info, score: int) -> Dict[str, float]:
    """
    COMPLEX NAVIGATION - REBALANCED
    Uses progress with simplified scaling.
    """
    # 1. Unpack Metrics
    progress = float(info.get("progress", 0.0))
    goal_dist = float(info.get("goal_dist", 0.0))
    current_x = float(info.get("x_position", 0.0))
    
    kills = int(info.get("enemies_killed_step", 0))
    coins = int(info.get("coins_delta", 0))
    won = info.get("won", False)
    life_lost = info.get("life_lost", False)

    # 2. Movement Logic - SIMPLIFIED (removed proximity multiplier)
    r_move = progress * 1.0
    
    if progress > 5.0:
        r_move *= 1.5

    # 3. Actions
    r_coin = 2.0 * coins 
    r_kill = 1.0 * kills 

    # 4. Win/Loss - REBALANCED
    if won:
        r_win = 200.0
        r_death = 0.0
    elif terminated:
        r_win = 0.0
        r_death = -50.0  # Was -2.0
    elif truncated:
        r_win = 0.0
        r_death = 0.0
    elif life_lost:
        r_win = 0.0
        r_death = -10.0  # Was -1.0
    else:
        r_win = 0.0
        r_death = 0.0

    r_time = -0.001

    return {
        "movement": r_move,
        "coins":    r_coin,
        "kills":    r_kill,
        "win":      r_win,
        "time":     r_time,
        "death":    r_death
    }

@_wrap_with_tracker
def simple(score_inc: bool, terminated: bool, truncated: bool, info: Info, score: int) -> Dict[str, float]:
    """
    SIMPLE - REBALANCED
    Basic progress reward with proper penalties.
    """
    # 1. Unpack Metrics
    progress = float(info.get("progress", 0.0))
    kills = int(info.get("enemies_killed_step", 0))
    coins = int(info.get("coins_delta", 0))
    won = info.get("won", False)
    current_x = float(info.get("x_position", 0.0))
    life_lost = info.get("life_lost", False)
    
    # 2. Calculate Components
    
    # A. Movement - SCALED UP
    r_move = progress * 2.0  # Was 0.5

    # B. Actions
    r_coin = 2.0 * coins  # Was 1.0
    r_kill = 1.0 * kills 
    
    # C. Win/Loss - REBALANCED
    if won:
        if current_x > 200.0:
            r_win = 200.0  # Was 50
            r_death = 0.0
        else:
            r_win = 20.0
            r_death = 0.0
    elif terminated:
        r_win = 0.0
        r_death = -50.0  # Was -2.0 (CRITICAL FIX)
    elif truncated:
        r_win = 0.0
        r_death = 0.0  # No penalty for timeout
    elif life_lost:
        r_win = 0.0
        r_death = -10.0  # Was -1.0
    else:
        r_win = 0.0
        r_death = 0.0

    r_time = -0.001

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
platformer_complex = complex_navigation 
platformer_dijkstra = delta_dijkstra 
default = simple