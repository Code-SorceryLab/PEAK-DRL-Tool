# code/rewards/mario.py
from __future__ import annotations
import random
from typing import Callable, Tuple, Dict, Any

Info = Dict[str, Any]

# ---- Score tracking utility -------------------------------------------

class _ScoreTracker:
    """
    Tracks previous score and episode-local deltas we care about for Mario.
    Keeps the API identical to your asteroids version (returns score, inc),
    but also injects useful per-step deltas into `info`:
      info["dx"]           : forward progress this step (>=0)
      info["coins_delta"]  : coins gained this step (>=0)
      info["kills_step"]   : enemies killed this step (>=0) if provided/calcable
    """
    def __init__(self):
        self.prev_score = 0
        self.last_x = None
        self.last_coins = 0
        self.last_kills = 0

    def step(self, info: Info) -> Tuple[int, bool]:
        # --- score increment flag (keeps parity with your asteroids API)
        score = int(info.get("score", info.get("total_reward", 0)) or 0)
        inc = score > self.prev_score

        # --- progress delta (>=0) ---------------------------------------
        x = float(info.get("x_position", self.last_x if self.last_x is not None else 0.0))
        if self.last_x is None:
            dx = 0.0
        else:
            dx = max(0.0, x - self.last_x)
        info["dx"] = dx

        # --- coins delta -------------------------------------------------
        coins = int(info.get("coins_collected", 0) or 0)
        info["coins_delta"] = max(0, coins - self.last_coins)

        # --- kills per step (best-effort) --------------------------------
        if "enemies_killed" in info:
            kills_now = int(info.get("enemies_killed") or 0)
            kills_step = max(0, kills_now - self.last_kills)
        elif "kills" in info:
            kills_now = int(info.get("kills") or 0)
            kills_step = max(0, kills_now - self.last_kills)
        else:
            kills_now = self.last_kills
            kills_step = int(info.get("enemies_killed_step", 0) or 0)
        info["kills_step"] = kills_step

        # --- update trackers --------------------------------------------
        self.prev_score = score
        self.last_x = x
        self.last_coins = coins
        self.last_kills = kills_now

        return score, inc

    def reset(self):
        self.prev_score = 0
        self.last_x = None
        self.last_coins = 0
        self.last_kills = 0


def _wrap_with_tracker(core_fn) -> Callable:
    """
    Adapter that matches your GameEnv signature and asteroids style.

    Exposed callable: reward(obs, base, terminated, info) -> float
    Internally we compute deltas and pass the same asteroids core signature:
        core_fn(score_inc, terminated, info, score)
    """
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
    SIMPLE: gentle forward shaping + coins/kills; tiny time tax.
    Meant to be easy-to-optimize and stable.
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
        r -= 0.005  # discourage idling/backpedal
    return r


@_wrap_with_tracker
def speedrunner(score_inc: bool, terminated: bool, info: Info, score: int) -> float:
    """
    SPEEDRUNNER: strong progress pressure, light time tax.
    Rewards reaching the goal quickly; tolerates some risk.
    """
    if terminated and not info.get("won", False):
        return -1.0

    dx = float(info.get("dx", 0.0))
    r = (dx / 5.5) - 0.01

    # bonus if env signals success
    if info.get("won", False) or info.get("goal_reached", False):
        r += 25.0
    return r


@_wrap_with_tracker
def coin_collector(score_inc: bool, terminated: bool, info: Info, score: int) -> float:
    """
    COIN COLLECTOR: prioritizes grabbing coins while still moving forward.
    """
    if terminated and not info.get("won", False):
        return -0.25

    dx = float(info.get("dx", 0.0))
    coins = int(info.get("coins_delta", 0))

    r = -0.003
    r += 3.0 * coins
    return r


@_wrap_with_tracker
def master(score_inc: bool, terminated: bool, info: Info, score: int) -> float:
    """
    MASTER: balanced—progress, coins, and kills; small idle penalty.
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
