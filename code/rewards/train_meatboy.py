from __future__ import annotations
from typing import Dict, Any
from code.rewards.train_platformer import _wrap_with_tracker

Info = Dict[str, Any]


@_wrap_with_tracker
def meatboy_simple(score_inc: bool, terminated: bool, info: Info, score: int) -> Dict[str, float]:
    """Reach the goal alive. Dense progress toward goal + completion bonus +
    death penalty + a small step cost to discourage dawdling."""
    progress = float(info.get("progress", 0.0))   # tracker derives this from goal_dist
    won = bool(info.get("won", False))

    r_move = max(-0.02, min(0.02, progress * 0.01))
    r_alive = 0.0005
    r_win, r_death = 0.0, 0.0
    if won:
        r_win = 5.0
    elif terminated:
        r_death = -2.0

    return {
        "movement": r_move,
        "alive": r_alive,
        "time": -0.0005,
        "win": r_win,
        "death": r_death,
    }


# train.py looks up reward functions by persona name; provide a default too.
default = meatboy_simple
