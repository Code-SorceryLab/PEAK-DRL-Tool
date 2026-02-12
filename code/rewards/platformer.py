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
        self.last_dist = None  # Track distance to goal
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
        # We prioritize Euclidean Distance to Goal provided by Core
        current_dist = float(info.get("goal_dist", 0.0))
        if math.isinf(current_dist): current_dist = 0.0
        
        if self.last_dist is None:
            self.last_dist = current_dist
            
        # Calculate Progress:
        # If Dist went from 100 -> 99, progress is +1.0 (Good)
        # If Dist went from 100 -> 101, progress is -1.0 (Bad)
        progress = self.last_dist - current_dist
        self.last_dist = current_dist
        
        # Store for Personas to use
        info["progress"] = progress
        
        if info.get("won", False) or info.get("terminated", False):
            info["progress"] = 0.0  # Bonus for winning (encourages finishing)

        # 3. Track Legacy X (for fallback/frontier logic)
        current_x = float(info.get("x_position", 0.0))
        if self.last_x is None:
            self.last_x = current_x
            self.max_x = current_x

        # dx = current_x - self.last_x
        # info["dx"] = dx # Still provided for compatibility
        
        # Frontier (max distance seen)
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
    Decorator that maintains the _ScoreTracker state across steps.
    """
    tracker = _ScoreTracker()

    def reward(obs, base, terminated: bool, info: Info) -> float:
        # Step the tracker
        _score, inc = tracker.step(info or {})
        
        # Call the persona
        r = float(core_fn(inc, terminated, info or {}, _score))
        
        if terminated or (info and info.get("terminated", False)):
            tracker.reset()
        return r

    return reward


# ---- Reward Personas --------------------------------------------------

@_wrap_with_tracker
def simple(score_inc: bool, terminated: bool, info: Info, score: int) -> float:
    """
    SIMPLE: Rewards reducing distance to goal (Euclidean), not just X movement.
    """
    # 1. Unpack Metrics
    # FIX: Use 'progress' (Goal Delta) instead of 'dx'
    progress = float(info.get("progress", 0.0))
    
    kills = int(info.get("enemies_killed_step", 0))
    coins = int(info.get("coins_delta", 0))
    won = info.get("won", False)
    
    # 2. Calculate Components
    # A. Movement (Goal Progress)
    r_move = progress / 8.0 
    
    # Stall penalty if not getting closer
    if progress < 0.001: 
        r_move -= 0.005 

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
            r_win = 0.0   # Fake Win
            r_death = -5.0
    else:
        r_win = 0.0
        r_death = -5.0 if terminated else 0.0

    r_time = -0.01

    # 3. INJECT FOR DASHBOARD
    info["reward_components"] = {
        "movement": r_move,
        "coins":    r_coin,
        "kills":    r_kill,
        "win":      r_win,
        "time":     r_time,
        "death":    r_death
    }

    return r_move + r_coin + r_kill + r_win + r_time + r_death


# @_wrap_with_tracker
# def speedrunner(score_inc: bool, terminated: bool, info: Info, score: int) -> float:
#     # Speedrunner uses RAW VELOCITY or Progress?
#     # Let's switch to Progress to handle pits/verticality better.
#     progress = float(info.get("progress", 0.0))
#     won = info.get("won", False)
#     current_x = float(info.get("x_position", 0.0))
    
#     r_move = (progress / 5.5) - 0.01
    
#     r_win = 0.0
#     if won:
#         r_win = 25.0 if current_x > 200 else -5.0
    
#     info["reward_components"] = {
#         "movement": r_move,
#         "win": r_win
#     }
#     return r_move + r_win


@_wrap_with_tracker
def coin_collector(score_inc: bool, terminated: bool, info: Info, score: int) -> float:
    coins = int(info.get("coins_delta", 0))
    won = info.get("won", False)
    progress = float(info.get("progress", 0.0))
    current_x = float(info.get("x_position", 0.0))
    
    r_move = progress / 20.0
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
    """
    MASTER: Hunts for coins and enemies, but STILL needs a breadcrumb trail
    to know which way to go.
    """
    coins = int(info.get("coins_delta", 0))
    kills = int(info.get("enemies_killed_step", 0))
    won = info.get("won", False)
    progress = float(info.get("progress", 0.0))
    current_x = float(info.get("x_position", 0.0))
    lives_left = int(info.get("lives", 3))

    # 1. Rewards
    r_move = progress / 10.0  # Gives it a reason to walk right!
    r_coin = coins * 3.0
    r_kill = kills * 2.0
    r_win = 100.0 if won else 0.0
    
    # 2. Penalties
    r_death = 0.0
    if terminated and not won:
        if lives_left > 0:
            r_death = -5.0   # Lost a single life
        else:
            r_death = -50.0  # Game over (no lives left)
    
    # FIXED: Actually inject r_move into the components dict!
    info["reward_components"] = {
        "r_move": r_move,
        "r_coins": r_coin, 
        "r_kills": r_kill, 
        "r_win": r_win, 
        "r_death": r_death
    }
    total_reward = sum(info["reward_components"].values())
    
    return total_reward


@_wrap_with_tracker
def explorer(score_inc: bool, terminated: bool, info: Info, score: int) -> float:
    # Explorer still rewards NEW territory (frontier_dx) because that is its purpose.
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
    dx = float(info.get("dx", 0.0)) # Momentum still relies on raw velocity/dx
    
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

    progress = float(info.get("progress", 0.0))
    kills = int(info.get("enemies_killed_step", 0))
    coins = int(info.get("coins_delta", 0))
    
    r_move = progress / 5.0
    r_kill = kills * 0.5
    r_coin = coins * 0.2
    
    info["reward_components"] = {
        "movement": r_move,
        "coins": r_coin,
        "kills": r_kill
    }
    return r_move + r_kill + r_coin - 0.01

# NEW OPTIMIZED REWARD PERSONA FOR platformer.py
# Add this to your platformer.py file

@_wrap_with_tracker
def adaptive(score_inc: bool, terminated: bool, info: Info, score: int) -> float:
    """
    ADAPTIVE CHAMPION: A curriculum-based reward function that adapts strategy
    based on progress toward the goal.
    
    DESIGN PRINCIPLES:
    1. Strong progress incentive (primary objective)
    2. Minimal collectible distraction (secondary objectives)
    3. Large win bonus (terminal reward)
    4. Adaptive time pressure (increases near goal)
    5. Risk-aware death penalty (context-dependent)
    6. Anti-cheese measures (validates real wins)
    
    CURRICULUM STAGES:
    - Far from goal (>500px): Exploration encouraged, lenient time
    - Mid-range (200-500px): Balanced play, moderate time pressure
    - Near goal (<200px): Aggressive push, high time pressure
    """
    
    # ========== 1. EXTRACT METRICS ==========
    progress = float(info.get("progress", 0.0))
    coins = int(info.get("coins_delta", 0))
    kills = int(info.get("enemies_killed_step", 0))
    won = info.get("won", False)
    current_x = float(info.get("x_position", 0.0))
    goal_dist = float(info.get("goal_dist", 9999.0))
    if math.isinf(goal_dist): 
        goal_dist = 9999.0
    
    # ========== 2. CURRICULUM: ADAPT BASED ON GOAL DISTANCE ==========
    # Calculate how close we are to goal (0.0 = at goal, 1.0 = very far)
    goal_proximity = min(1.0, goal_dist / 1000.0)  # Normalize to [0, 1]
    
    # ========== 3. PROGRESS REWARD (PRIMARY OBJECTIVE) ==========
    # Base progress reward: Aggressive coefficient for strong forward incentive
    r_move = progress / 5.0  # Best coefficient from analysis
    
    # BONUS: Extra reward when making consistent progress (momentum)
    if progress > 1.0:  # Moving more than 1 pixel toward goal
        r_move *= 1.2  # 20% bonus for strong forward movement
    
    # PENALTY: Stall prevention (stronger than other personas)
    if abs(progress) < 0.1:  # Nearly standing still
        r_move -= 0.01  # Small but meaningful penalty
    
    # PENALTY: Backtracking (moving away from goal)
    if progress < -0.5:
        r_move -= 0.05  # Penalize going backwards
    
    # ========== 4. COLLECTIBLES (SECONDARY OBJECTIVES) ==========
    # Keep rewards low to avoid distraction from main goal
    # But high enough to encourage strategic collection
    r_coin = coins * 0.3   # Reduced from 3-5 in other personas
    r_kill = kills * 0.4   # Reduced from 2-2.5 in other personas
    
    # DIMINISHING RETURNS: If farming too much, reduce rewards
    total_coins = int(info.get("coins_collected", 0))
    
    if total_coins > 20:  # Collected a lot of coins
        r_coin *= 0.5  # Halve coin rewards (stop farming)
    
    # ========== 5. TIME PRESSURE (EFFICIENCY INCENTIVE) ==========
    # Adaptive time penalty: Increases as you get closer to goal
    # Far from goal: Minimal pressure (-0.02)
    # Near goal: High pressure (-0.10)
    time_multiplier = 1.0 + (1.0 - goal_proximity) * 3.0  # Range: 1x to 4x
    r_time = -0.025 * time_multiplier
    
    # Additional penalty for being slow near the goal
    if goal_dist < 200 and progress < 0.5:
        r_time -= 0.05  # "Hurry up, you're almost there!"
    
    # ========== 6. WIN BONUS (TERMINAL REWARD) ==========
    r_win = 0.0
    r_death = 0.0
    
    if won:
        # ANTI-CHEESE: Validate real win (must be past x=200)
        if current_x > 200.0:
            # MASSIVE win bonus (highest of all personas)
            r_win = 200.0
            
            # SPEED BONUS: Extra reward for fast completion
            time_left = float(info.get("time_left", 0))
            if time_left > 300:  # Lots of time remaining
                r_win += 50.0  # "Speedrun bonus!"
            elif time_left > 150:
                r_win += 25.0  # "Efficient bonus"
            
            # PERFECTION BONUS: No deaths during level
            lives = int(info.get("lives", 3))
            max_lives = 3  # Assume default
            if lives == max_lives:
                r_win += 30.0  # "Flawless victory!"
            
            r_death = 0.0  # No death penalty on win
        else:
            # FAKE WIN: Probably glitched or cheesed
            r_win = 0.0
            r_death = -10.0  # Mild penalty for wasting time
    
    # ========== 7. DEATH PENALTY (RISK MANAGEMENT) ==========
    elif terminated:
        # Context-aware death penalty
        lives_left = int(info.get("lives", 3))
        
        # Base penalty
        r_death = -15.0
        
        # SCALE BY PROGRESS: Dying near goal hurts more
        if goal_dist < 100:  # Very close to goal
            r_death = -30.0  # "You almost had it!"
        elif goal_dist < 300:  # Close to goal
            r_death = -20.0  # "Come on, push through!"
        
        # GAME OVER: Severe penalty for using all lives
        if lives_left == 0:
            r_death = -50.0  # "You had 3 chances!"
        
        # LENIENCY: If died very far from goal, be gentler
        if goal_dist > 800:
            r_death *= 0.5  # "Okay, you were exploring"
    
    # ========== 8. EXPLORATION BONUS (CURRICULUM EARLY GAME) ==========
    # Early game: Encourage exploration
    # Late game: Discourage wandering
    r_explore = 0.0
    frontier_dx = float(info.get("frontier_dx", 0.0))
    
    if goal_proximity > 0.7:  # Far from goal (early game)
        # Reward discovering new areas
        r_explore = frontier_dx / 10.0
    elif goal_proximity < 0.3:  # Near goal (late game)
        # Penalize wandering when close to goal
        if frontier_dx > progress:  # Exploring more than progressing
            r_explore = -0.02  # "Focus on the goal!"
    
    # ========== 9. RISK/REWARD: SHORTCUT BONUS ==========
    # Reward aggressive play if it leads to big progress
    r_risk = 0.0
    velocity_x = float(info.get("velocity_x", 0.0))
    
    if progress > 5.0 and abs(velocity_x) > 150:  # Big jump/dash
        r_risk = 0.5  # "Nice shortcut!"
    
    # ========== 10. ASSEMBLE FINAL REWARD ==========
    total_reward = (
        r_move +      # Primary: Progress toward goal
        r_coin +      # Secondary: Coin collection
        r_kill +      # Secondary: Enemy elimination
        r_time +      # Pressure: Efficiency incentive
        r_win +       # Terminal: Victory bonus
        r_death +     # Terminal: Failure penalty
        r_explore +   # Curriculum: Exploration
        r_risk        # Bonus: Risk-taking
    )
    
    # ========== 11. COMPONENT TRACKING (FOR DEBUGGING) ==========
    info["reward_components"] = {
        "movement": round(r_move, 3),
        "coins": round(r_coin, 3),
        "kills": round(r_kill, 3),
        "time": round(r_time, 3),
        "win": round(r_win, 3),
        "death": round(r_death, 3),
        "explore": round(r_explore, 3),
        "risk": round(r_risk, 3),
        "total": round(total_reward, 3)
    }
    
    # ========== 12. SANITY CHECKS ==========
    # Prevent extreme rewards that could destabilize learning
    total_reward = max(-100.0, min(300.0, total_reward))
    
    return total_reward


# ============================================================================
# VARIANT: SPEEDRUN CHAMPION (For advanced agents)
# ============================================================================

@_wrap_with_tracker  
def speedrunner(score_inc: bool, terminated: bool, info: Info, score: int) -> float:
    """
    SPEEDRUN CHAMPION: Optimized for agents that already know how to complete levels.
    Focus: Complete levels as fast as possible, ignore distractions.
    
    Use this AFTER pre-training with adaptive_champion.
    """
    
    progress = float(info.get("progress", 0.0))
    won = info.get("won", False)
    current_x = float(info.get("x_position", 0.0))
    time_left = float(info.get("time_left", 0))
    
    # AGGRESSIVE progress reward
    r_move = progress / 3.0  # Even more aggressive than adaptive
    
    # STRONG time pressure (every step counts)
    r_time = -0.1
    
    # MASSIVE win bonus weighted by speed
    r_win = 0.0
    r_death = 0.0
    
    if won and current_x > 200:
        # Base win
        r_win = 150.0
        
        # HUGE time bonus (encourages speedrunning)
        time_bonus = (time_left / 400.0) * 150.0  # Up to 150 bonus
        r_win += time_bonus
        
    elif terminated:
        r_death = -40.0  # Harsh penalty (no time to waste)
    
    # NO collectible rewards (pure speed focus)
    
    info["reward_components"] = {
        "movement": r_move,
        "time": r_time,
        "win": r_win,
        "death": r_death
    }
    
    return r_move + r_time + r_win + r_death


# ============================================================================
# VARIANT: PERFECTIONIST (For 100% completion)
# ============================================================================

@_wrap_with_tracker
def perfectionist(score_inc: bool, terminated: bool, info: Info, score: int) -> float:
    """
    PERFECTIONIST: For agents that should collect everything AND complete levels.
    Balance between completion and collection.
    
    Use for: Levels with important collectibles, achievement hunting
    """
    
    progress = float(info.get("progress", 0.0))
    coins = int(info.get("coins_delta", 0))
    kills = int(info.get("enemies_killed_step", 0))
    won = info.get("won", False)
    current_x = float(info.get("x_position", 0.0))
    
    total_coins = int(info.get("coins_collected", 0))
    
    # Balanced progress (not too aggressive)
    r_move = progress / 7.0
    
    # Higher collectible rewards than adaptive
    r_coin = coins * 0.8
    r_kill = kills * 0.6
    
    # Time pressure (but lenient to allow collection)
    r_time = -0.02
    
    # Win bonus scales with collection
    r_win = 0.0
    r_death = 0.0
    
    if won and current_x > 200:
        r_win = 150.0
        
        # COLLECTION BONUS
        coin_bonus = min(50.0, total_coins * 2.0)  # Up to 50 bonus
        r_win += coin_bonus
        
        # PERFECT BONUS: If collected >90% of coins
        if total_coins > 15:  # Assume ~20 coins per level
            r_win += 30.0  # "Perfect collection!"
    
    elif terminated:
        r_death = -20.0
    
    info["reward_components"] = {
        "movement": r_move,
        "coins": r_coin,
        "kills": r_kill,
        "time": r_time,
        "win": r_win,
        "death": r_death
    }
    
    return r_move + r_coin + r_kill + r_time + r_win + r_death

# ---- ALIASES -----------------------------------------------------------
platformer_simple = simple
platformer_speedrunner = speedrunner
platformer_coin_collector = coin_collector
platformer_master = master
platformer_explorer = explorer
platformer_perfectionist = perfectionist
platformer_adaptive = adaptive


default = simple