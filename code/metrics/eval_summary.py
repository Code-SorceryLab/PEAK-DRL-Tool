"""Pure, dependency-free metrics over a list of FINAL per-episode `info` dicts.

An episode dict is the last `info` returned by `platformer_core._info()` for an
episode. We read three fields:
    - info["won"]   : bool  (truth for a win; == reached_goal)
    - info["cause"] : str   (death cause; "" when truncated without death)
    - info["level"] : str   (registered level name, e.g. "Mario1-1")

Death-cause vocabulary emitted by the current platformer (verified at the
_handle_death() call sites): "Goal","Pit","OOB","Stall","Timeout","Koopa",
"Enemy","Spike","Unknown","".  There is NO "Wall" cause today (wall hits only
stop motion), so the "wall" bucket below stays 0 — it is kept for schema parity.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

# Raw cause string -> coarse bucket. "Goal"/"" are NOT deaths and are absent here.
CAUSE_BUCKETS: Dict[str, str] = {
    "Pit": "pit",
    "OOB": "pit",          # fell/ran out of bounds -> treat as a pit-style death
    "Stall": "stall",
    "Koopa": "enemy",
    "Enemy": "enemy",
    "Spike": "enemy",
    "Timeout": "timeout",
    "Time Over": "timeout",  # sonic_core variant; harmless to map here too
}

# The fixed set of buckets we always report (so callers can rely on the shape).
_BUCKET_KEYS = ("pit", "wall", "stall", "enemy", "timeout", "other")

# A truncated-without-death episode has these cause values.
_NON_DEATH_CAUSES = ("", "Timeout", "Time Over")


def _empty_by_cause() -> Dict[str, int]:
    return {k: 0 for k in _BUCKET_KEYS}


def _is_win(info: dict) -> bool:
    # `won` is the source of truth; fall back to event=="WIN" if absent.
    if "won" in info:
        return bool(info["won"])
    return str(info.get("event", "")) == "WIN"


def _death_bucket(info: dict) -> str:
    cause = str(info.get("cause", "") or "")
    event = str(info.get("event", "") or "")
    # Truncation without a death (no WIN, no death cause) -> timeout.
    if event != "WIN" and cause in _NON_DEATH_CAUSES:
        return "timeout"
    return CAUSE_BUCKETS.get(cause, "other")


def summarize_eval(episodes: List[dict]) -> dict:
    """Aggregate a flat list of episode info dicts.

    Returns:
        {
          "n": int,                # number of episodes
          "wins": int,             # episodes with won==True
          "win_rate": float,       # wins / n (0.0 when n == 0)
          "by_cause": {            # death-cause histogram (wins excluded)
              "pit": int, "wall": int, "stall": int,
              "enemy": int, "timeout": int, "other": int,
          },
        }
    """
    n = len(episodes)
    wins = 0
    by_cause = _empty_by_cause()
    for info in episodes:
        if _is_win(info):
            wins += 1
            continue
        by_cause[_death_bucket(info)] += 1
    win_rate = (wins / n) if n else 0.0
    return {"n": n, "wins": wins, "win_rate": win_rate, "by_cause": by_cause}


def summarize_by_level(episodes: List[dict]) -> Dict[str, dict]:
    """Group episodes by info['level'] and summarize each group."""
    groups: Dict[str, List[dict]] = defaultdict(list)
    for info in episodes:
        groups[str(info.get("level", "unknown"))].append(info)
    return {level: summarize_eval(eps) for level, eps in groups.items()}
