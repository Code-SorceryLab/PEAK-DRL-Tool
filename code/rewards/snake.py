# Add this to your snake.py file (or wherever your snake rewards are)

import numpy as np
from typing import Callable


# Use the same _ScoreTracker class from asteroids
class _ScoreTracker:
    """
    Tracks previous score and exposes score increase detection.
    Used by reward wrapper to detect when agent scores points.
    """
    def __init__(self):
        """Initialize with score of 0."""
        self.prev = 0

    def step(self, info: dict) -> tuple:
        """
        Update score tracking and detect if score increased.
        
        Args:
            info: Info dict containing current score
        
        Returns:
            tuple[int, bool]: (current_score, score_increased_this_step)
        """
        score = int(info.get("score", 0))
        increased = score > self.prev
        self.prev = score
        return score, increased

    def reset(self):
        """Reset score tracking (call at episode end)."""
        self.prev = 0


def _wrap_with_tracker(core_fn) -> Callable:
    """
    Decorator that wraps reward functions with score tracking.
    Adapts reward functions to the GameEnv signature and handles
    episode-local score tracking with automatic reset.
    """
    tracker = _ScoreTracker()

    def reward(obs, base, terminated: bool, info: dict) -> float:
        """
        Wrapped reward function that handles score tracking.
        
        Args:
            obs: Observation (unused - all info comes from info dict now)
            base: Base reward from environment (unused)
            terminated: Whether episode ended
            info: Info dict with all calculated values from core
        
        Returns:
            float: Calculated reward value
        """
        _score, inc = tracker.step(info)
        r = float(core_fn(inc, terminated, info, _score))
        
        # Reset tracker at episode end
        if terminated or info.get("terminated", False):
            tracker.reset()
        
        return r

    return reward


# Helper function
def _dist(hx, hy, fx, fy):
    return float(np.hypot(hx - fx, hy - fy))


@_wrap_with_tracker
def baseline(score_inc: bool, terminated: bool, info: dict, score: int) -> float:
    """
    AGGRESSIVE food-seeking reward.
    Heavily incentivizes moving toward and eating food.
    """
    # Get positions from info
    hx = info.get("head_x", 0.0)
    hy = info.get("head_y", 0.0)
    fx = info.get("food_x", 0.0)
    fy = info.get("food_y", 0.0)
    
    # Calculate distance
    d = _dist(hx, hy, fx, fy)
    
    r = 0.0
    
    # 1. MASSIVE FOOD REWARD (primary goal)
    if score_inc:
        r += 10.0  # HUGE reward for eating
    
    # 2. STRONG DISTANCE GRADIENT (gets much stronger as you approach)
    # This creates a clear "pull" toward food
    if d < 0.1:  # Very close (< 10% of board)
        r += (1.0 - d) * 2.0  # Massive gradient when very close
    elif d < 0.3:  # Close
        r += (1.0 - d) * 1.0  # Strong gradient
    elif d < 0.5:  # Medium distance
        r += (1.0 - d) * 0.5  # Medium gradient
    else:  # Far
        r += (1.0 - d) * 0.2  # Weak gradient
    
    # 3. IMMEDIATE MOVEMENT REWARD (instant feedback)
    moved_closer = bool(info.get("moved_closer", False))
    if moved_closer:
        # Scale with how close you are
        if d < 0.2:
            r += 0.5  # Big bonus when close
        elif d < 0.4:
            r += 0.3  # Medium bonus
        else:
            r += 0.1  # Small bonus when far
    else:
        # Penalty for moving away (but smaller than moving closer bonus)
        if d < 0.3:
            r -= 0.2  # Bigger penalty when close
        else:
            r -= 0.05  # Small penalty when far
    
    # 4. STAGNATION PENALTY (prevent getting stuck)
    no_progress_steps = int(info.get("no_progress_steps", 0))
    if no_progress_steps >= 5:  # Start earlier
        r -= min(0.5, 0.05 * (no_progress_steps - 4))  # Ramps up faster
    
    # 5. HARSH OSCILLATION PENALTY
    oscillating = bool(info.get("oscillating", False))
    if oscillating:
        r -= 50  # Much stronger penalty
    
    # 6. DEATH PENALTY
    if terminated:
        r -= 5.0  # Moderate penalty
    
    # 7. SMALL TIME PENALTY (encourages efficiency)
    r -= 0.005
    
    # Clip for stability
    r = max(-10.0, min(15.0, r))
    return float(r)

@_wrap_with_tracker
def aggressive(score_inc: bool, terminated: bool, info: dict, score: int) -> float:
    """
    Aggressive food-seeking variant.
    Higher rewards for progress, lower death penalty (take risks).
    """
    r = 0.0
    
    # 1. HUGE FOOD REWARD
    if score_inc:
        r += 20.0  # Even bigger reward
    
    # 2. AMPLIFIED MOVE DELTA
    move_delta = info.get('move_delta', 0.0)
    r += move_delta * 2.0  # Much stronger signal
    
    # 3. LIGHT STAGNATION PENALTY
    no_progress_steps = int(info.get("no_progress_steps", 0))
    if no_progress_steps >= 8:  # More lenient
        r -= 0.2
    
    # 4. LIGHT OSCILLATION PENALTY
    if info.get("oscillating", False):
        r -= 0.2
    
    # 5. MODERATE DEATH PENALTY (willing to take risks)
    if terminated:
        r -= 3.0
    
    # 6. HIGHER TIME PRESSURE
    r -= 0.01
    
    r = max(-10.0, min(25.0, r))
    return float(r)
