# code/games/asteroids.py
from __future__ import annotations
import random
import math
from typing import Callable, Tuple
import numpy as np

# ---- Score tracking utility -------------------------------------------

class _ScoreTracker:
    """
    Tracks previous score and exposes score increase detection.
    Used by reward wrapper to detect when agent scores points.
    """
    def __init__(self):
        """Initialize with score of 0."""
        self.prev = 0

    def step(self, info: dict) -> Tuple[int, bool]:
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
    
    All distance and targeting calculations are now done in asteroids_core.py
    and passed through the info dict for efficiency.
    
    Args:
        core_fn: Core reward function with signature (score_inc, terminated, info, score)
    
    Returns:
        Callable: Wrapped reward function with GameEnv signature
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

# ---- Distance utility functions ---------------------------------------

def distance_band_bonus_single(distance, ideal_min=100, ideal_max=300, penalty_scale=0.01):
    """
    Calculate reward for staying within an ideal distance band from nearest asteroid.
    Provides positive reward within the band, linear penalty outside.
    
    Args:
        distance: Distance to nearest asteroid
        ideal_min: Minimum ideal distance (closer = penalty)
        ideal_max: Maximum ideal distance (farther = penalty)
        penalty_scale: Scale factor for penalty (higher = steeper penalty)
    
    Returns:
        float: Reward value (+1.0 in band, negative outside)
    """
    if ideal_min <= distance <= ideal_max:
        return 1.0  # In sweet spot
    elif distance < ideal_min:
        return -penalty_scale * (ideal_min - distance)  # Too close penalty
    else:
        return -penalty_scale * (distance - ideal_max)  # Too far penalty

def distance_band_bonus_multi(closest_distances, ideal_min=100, ideal_max=300, penalty_scale=0.01):
    """
    Calculate reward for staying within ideal distance from multiple asteroids.
    Uses average distance to closest N asteroids for more stable positioning.
    
    Args:
        closest_distances: List of distances to closest N asteroids
        ideal_min: Minimum ideal distance
        ideal_max: Maximum ideal distance  
        penalty_scale: Scale factor for penalty
    
    Returns:
        float: Reward value based on average distance to multiple asteroids
    """
    return sum(
        distance_band_bonus_single(d, ideal_min, ideal_max, penalty_scale)
        for d in closest_distances
    ) / len(closest_distances)

def calculate_gradient_alignment(info: dict) -> float:
    """
    Calculate how well the ship is moving along the safety gradient (away from danger).
    
    Args:
        info: Info dict with ship velocity and danger gradient magnitude
    
    Returns:
        float: Alignment score from -1 (moving toward danger) to +1 (moving away from danger)
        0 means moving perpendicular or ship not moving
    """
    ship_vel = info.get("ship_velocity", np.array([0.0, 0.0]))
    
    # Get gradient magnitude - need to reconstruct direction from current_gradient
    # Since we only have magnitude in info, we'll use a simplified approach
    grad_magnitude = info.get("danger_gradient_magnitude", 0.0)
    
    # If no gradient or not moving, no alignment
    vel_magnitude = np.linalg.norm(ship_vel)
    if vel_magnitude < 1e-6 or grad_magnitude < 1e-6:
        return 0.0
    
    # Note: Without gradient direction in info, we can't calculate true alignment
    # You may want to add gradient_x/y back to info if this is critical
    # For now, return 0 as we can't calculate it
    return 0.0

@_wrap_with_tracker  
def hunter(score_inc: bool, terminated: bool, info: dict, score: int) -> float:
    """
    HUNTER PERSONA: Aggressive destruction of asteroids.
    Rewards targeting, shooting accuracy, and destruction while maintaining safe distance.
    """
    if terminated:
        return -100.0
    if info.get("collision", False):
        return -20.0 
    
    r = 0.01
    fired = info.get("bullets_fired", 0)
    targeting = info.get("targeting_bonus", 0.0)
    
    # Use first element of distances_to_closest_3 for nearest distance
    closest_distances = info.get("distances_to_closest_3", [800.0, 800.0, 800.0])
    distance = closest_distances[0]
    d_bonus = distance_band_bonus_single(distance, 100, 300, 0.01)
    
    # Shooting rewards
    if fired > 0:
        if targeting > 3.0:
            r += (targeting ** 2) * fired * d_bonus
        elif targeting > 1.0:
            r += targeting * fired * d_bonus
    
    # Destruction and progress rewards
    r += info.get("asteroids_destroyed", 0) * 5.0
    
    if info.get("level_completed", False):
        r += 25.0
    
    r += info.get("score_delta", 0) * 0.2
    
    # MUCH SOFTER danger field penalty with cap
    danger = info.get("danger_field", 0.0)
    
    # Soft, capped penalty
    if danger > 10.0:
        danger_penalty = -2.0  # Hard cap at -2
    elif danger > 5.0:
        danger_penalty = -1.0 - (danger - 5.0) * 0.2  # -1 to -2 range
    elif danger > 3.0:
        danger_penalty = -(danger - 3.0) * 0.5  # -0 to -1 range
    else:
        danger_penalty = 0.0
    
    r += danger_penalty
    
    # SOFTER alignment bonus/penalty
    alignment = calculate_gradient_alignment(info)
    if danger > 3.0:
        if alignment > 0:  # Moving toward safety
            r += alignment * 0.5  # Small bonus, capped at +0.5
        else:  # Moving toward danger
            r += alignment * 0.3  # Small penalty, capped at -0.3
    
    return r

@_wrap_with_tracker
def survivor(score_inc: bool, terminated: bool, info: dict, score: int) -> float:
    """
    SURVIVOR PERSONA: Prioritizes staying alive and avoiding danger.
    Rewards safe positioning and escaping dangerous areas.
    """
    if terminated:
        return -200.0  # Harsh penalty for dying (survival prioritized)
    if info.get("collision", False):
        return -25.0   # High penalty for touching an asteroid
    
    danger = info.get("danger_field", 0.0)
    alignment = calculate_gradient_alignment(info)
    survived_bonus = 0.25   # Per time step out of danger
    
    # Survival/zone reward:
    if danger < 2.0:
        r = survived_bonus    # Good reward for staying safe
        r += max(alignment, 0) * 1.0  # Bonus for actively moving away from threat
    elif danger < 5.0:
        r = survived_bonus * 0.5  # Reduced time step reward in caution zone
        r += max(alignment, 0) * 0.5
        r -= (danger - 2.0) * 0.2  # Soft penalty for being in caution
        r -= max(-alignment, 0) * (danger - 2.0) * 0.5  # Penalize moving INTO greater danger
    else:
        r = -1.0  # Strong penalty for staying in severe threat
        r += max(alignment, 0) * 0.3   # Still slight reward for escaping
        r -= max(-alignment, 0) * 1.0  # Large penalty for moving deeper into danger
    
    # Small bonus for asteroid destruction and level completion (but much less than hunter/speedrunner)
    r += info.get("asteroids_destroyed", 0) * 0.25
    if info.get("level_completed", False):
        r += 5.0   # Nice bonus if you beat the level purely by surviving
    
    return r

@_wrap_with_tracker
def speedrunner(score_inc: bool, terminated: bool, info: dict, score: int) -> float:
    """
    SPEEDRUNNER PERSONA: Fast level completion.
    Rewards quick destruction, movement, and level completion with time pressure.
    """
    # Death penalty (moderate - speed is prioritized over safety)
    if terminated:
        return -100.0
    if info.get("collision", False):
        return -5.0  # Light penalty - speedrunner takes calculated risks
    
    r = -0.001  # Time pressure - every frame costs a small amount
    fired = info.get("bullets_fired", 0)
    targeting = info.get("targeting_bonus", 0.0)
    ship_speed = info.get("ship_speed", 0.0) 
    
    # Level completion (main goal for speedrunner)
    if info.get("level_completed", False):
        r += 50.0
    
    # Asteroid destruction (progress toward level completion)
    r += info.get("asteroids_destroyed", 0) * 10.0
    
    # Movement bonus (encourage active play)
    r += ship_speed * 0.05
    
    # Score momentum (maintain forward progress)
    r += info.get("score_delta", 0) * 0.5
    
    # Targeting bonus (efficient aiming saves time)
    r += info.get("targeting_bonus_delta", 0.0)
    if fired > 0:
        r += targeting
    
    # Danger field penalty (lighter for speedrunner - takes risks for speed)
    danger = info.get("danger_field", 0.0)
    if danger > 8.0:  # Only penalize extreme danger
        r -= (danger - 8.0) * 0.2
    elif danger > 5.0:  # Very light penalty for high danger
        r -= (danger - 5.0) * 0.05
    
    # Small bonus for escaping when in extreme danger
    if danger > 7.0:
        alignment = calculate_gradient_alignment(info)
        if alignment > 0:
            r += alignment * 0.2
    
    return r

# ---- Reference reward functions -----------------------------------

@_wrap_with_tracker
def baseline(score_inc: bool, terminated: bool, info: dict, score: int) -> float:
    """
    BASELINE PERSONA: Random rewards for benchmarking.
    Used to test that agents are actually learning vs random performance.
    
    Returns:
        float: Random reward between -0.5 and 0.5
    """
    return random.random() - 0.5

@_wrap_with_tracker
def simple(score_inc: bool, terminated: bool, info: dict, score: int) -> float:
    """
    SIMPLE PERSONA: Basic reward structure.
    Simple destruction bonuses and collision penalties.
    """
    r = -0.01
    r += info.get("asteroids_destroyed", 0) * 1.0  # Destruction bonus
    
    if info.get("collision", False):
        r -= 10.0
    
    # Use first element of distances_to_closest_3
    closest_distances = info.get("distances_to_closest_3", [800.0, 800.0, 800.0])
    distance = closest_distances[0]
    
    threat = 80
    if distance <= threat:
        r -= 0.5
    
    return r
