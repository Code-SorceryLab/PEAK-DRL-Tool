# code/rewards/flappy.py
from __future__ import annotations
import random
from typing import Callable, Tuple
import numpy as np

# ---- Small utilities -----------------------------------------------------

def _vertical_from_obs(obs) -> float:
    """
    Extract vertical distance between bird (y) and gap center from the obs vector.
    Expected (y, ?, dx, gap_top, gap_bottom). Falls back gracefully if shapes differ.
    """
    try:
        y, _, _dx, gap_top, gap_bottom = obs
    except Exception:
        # try to be permissive: y first, last two are gap bounds
        y = float(obs[0])
        gap_top = float(obs[-2])
        gap_bottom = float(obs[-1])
    gap_mid = (gap_top + gap_bottom) / 2.0
    return abs(y - gap_mid)

def _gap_centering_shaping(obs, gap_reward_weight=2.0, min_penalty=-0.05):
    y, _, _dx, gap_top, gap_bottom = obs
    pipe_gap = gap_bottom - gap_top
    gap_center_y = gap_top + pipe_gap * 0.5
    reward = 1 - (abs(y - gap_center_y) / (pipe_gap / 3))
    reward = max(reward, min_penalty)
    if reward > 0:
        reward *= gap_reward_weight
    return reward

class _ScoreTracker:
    """Tracks previous score and exposes helpers."""
    def __init__(self):
        self.prev = 0

    def step(self, info: dict) -> Tuple[int, bool]:
        score = int(info.get("score", 0))
        increased = score > self.prev
        self.prev = score
        return score, increased

    def reset(self):
        self.prev = 0


def _wrap_with_tracker(core_fn) -> Callable:
    """
    Adapts a core reward function that takes:
      core_fn(vertical: float, score_inc: bool, terminated: bool) -> float
    into the GameEnv signature:
      reward(obs, base, terminated, info) -> float
    and handles episode-local score tracking/reset.
    """
    tracker = _ScoreTracker()

    def reward(obs, base, terminated: bool, info: dict) -> float:
        vertical = _vertical_from_obs(obs)
        shaped_vertical = _gap_centering_shaping(obs)
        _score, inc = tracker.step(info)
        r = float(core_fn(vertical, inc, terminated, shaped_vertical, _score))
        # reset tracker at episode end (both env APIs)
        if terminated or info.get("episode_end", False):
            tracker.reset()
        return r

    return reward

# ---- Final three rewards --------------------------------------------------

@_wrap_with_tracker
def simple(vertical: float, inc: bool, terminated: bool, shaped_vertical: float, score: int) -> float:
    """+1 per pipe, nothing else."""
    return 1.0 if inc else 0.1


@_wrap_with_tracker
def speedrunner(vertical: float, inc: bool, terminated: bool, shaped_vertical: float, score: int) -> float:
    """
    SPEEDRUNNER:
    +1.0 per pipe
    -0.002 per frame   (gentle time pressure)
    +0.30 * shaped_vertical  (policy-safe centering nudge)
    + score/100.0      (tiny momentum bonus: later pipes slightly sweeter)
    -1.0 on crash      (make "cash 1 then die" clearly worse)
    """
    r = -0.002
    if inc:
        r += 1.0
    r += 0.30 * float(shaped_vertical)   
    r += float(score) / 100.0
    if terminated:
        r -= 1.0
    return r


@_wrap_with_tracker
def master(vertical: float, inc: bool, terminated: bool, shaped_vertical: float, score: int) -> float:
    """
    Dense shaping:
      +0.2 per frame
      +5 per pipe
      -20 on crash
      -0.05 * vertical distance from gap centre
    """
    if terminated:
        return -20.0
    r = 0.2
    if inc:
        r += 5.0
    r -= 0.05 * float(vertical)
    return r


@_wrap_with_tracker
def baseline(vertical: float, inc: bool, terminated: bool, shaped_vertical: float, score: int) -> float:
    """
    Good for benchmarking pure randomness.
    """
    return random.random() - 0.5


@_wrap_with_tracker
def shaped(vertical: float, inc: bool, terminated: bool, shaped_vertical: float, score: int) -> float:
    """
    +1 per pipe
    + Gap centering shaping bonus/penalty every frame.
    """
    r = 1.0 if inc else 0.0
    r += shaped_vertical
    r += float(score) / 50.0
    return r