from __future__ import annotations
import numpy as np

# Tracks all stateful stuff between steps score, lives, combo count, etc.
class _Tracker:
    def __init__(self):
        self.raw_return = 0.0
        self.no_score = 0
        self.prev_lives = None
        self.steps = 0
        self.power_left = 0
        self.last_score_step = -10**9
        self.combo = 0

    def reset(self, info: dict):
        self.raw_return = 0.0
        self.no_score = 0
        self.prev_lives = info.get("lives", None)
        self.steps = 0
        self.power_left = 0
        self.last_score_step = -10**9
        self.combo = 0


def _wrap_with_tracker(core_fn):
    tracker = _Tracker()
    def reward(obs, base: float, terminated: bool, info: dict) -> float:
        tracker.steps += 1
        raw_r = float(base)
        tracker.raw_return += raw_r

        r = float(core_fn(obs, raw_r, terminated, info or {}, tracker))

        if terminated or (info and info.get("episode_end", False)):
            tracker.reset(info or {})
        return r
    return reward


# pac man reward logic  
@_wrap_with_tracker
def baseline(
    obs, raw_r: float, terminated: bool, info: dict, t: _Tracker,
    *,
    base_scale: float = 0.05,
    survive_bonus: float = 0.0,
    no_score_patience: int = 999,
    no_score_penalty: float = 0.0,
    death_penalty: float = 3.0,
    pos_boost: float = 0.0,
    combo_window: int = 0,
    combo_step_bonus: float = 0.0,
    power_trigger_min: float = float("inf"),
    power_steps: int = 0,
    power_step_bonus: float = 0.0,
    ghost_threshold: float = float("inf"),
    ghost_mult: float = 0.0,
    max_steps: int | None = None,
) -> float:
    # Base reward is scaled raw ALE reward, plus small survival incentive
    r = base_scale * raw_r + survive_bonus

    # Handle positive reward events and combo stacking
    if raw_r > 0.0:
        r += pos_boost
        if combo_window > 0 and combo_step_bonus > 0.0:
            if (t.steps - t.last_score_step) <= combo_window:
                t.combo += 1
            else:
                t.combo = 1
            r += t.combo * combo_step_bonus
        t.last_score_step = t.steps
        t.no_score = 0
    else:
        t.no_score += 1

    # Power mode bonus (big pellets, ghost kills, etc.)
    if power_steps > 0 and np.isfinite(power_trigger_min):
        if raw_r >= power_trigger_min:
            t.power_left = max(t.power_left, power_steps)
        if t.power_left > 0:
            r += power_step_bonus
            if raw_r >= ghost_threshold:
                r += ghost_mult
            t.power_left -= 1
    else:
        t.power_left = 0

    # Penalize long periods of inactivity (no scoring)
    if no_score_penalty > 0.0 and t.no_score > int(no_score_patience):
        r -= no_score_penalty

    # Life loss detection and penalty
    lives = info.get("lives", t.prev_lives)
    if (t.prev_lives is not None) and (lives is not None) and (lives < t.prev_lives):
        r -= death_penalty
        t.no_score = 0
        t.combo = 0
        t.power_left = 0
    t.prev_lives = lives

    # Optional truncation cap if needed
    if (max_steps is not None) and (t.steps >= max_steps):
        pass

    return float(r)
