"""Unit tests for code.metrics.eval_summary (pure metric over synthetic episode dicts)."""
import math
import pytest

from code.metrics.eval_summary import (
    summarize_eval,
    summarize_by_level,
    CAUSE_BUCKETS,
)

# --- helpers: build synthetic FINAL per-episode info dicts ---
def ep(level, won, cause, event=None):
    if event is None:
        event = "WIN" if won else ("DIED" if cause not in ("", "Timeout", "Time Over") else "")
    return {"level": level, "won": won, "cause": cause, "event": event}


def test_empty_episodes_is_safe():
    out = summarize_eval([])
    assert out["n"] == 0
    assert out["wins"] == 0
    assert out["win_rate"] == 0.0
    # by_cause always has the full key set, all zero
    assert out["by_cause"] == {"pit": 0, "wall": 0, "stall": 0, "enemy": 0, "timeout": 0, "other": 0}


def test_win_rate_basic():
    eps = [ep("Mario1-1", True, "Goal"),
           ep("Mario1-1", True, "Goal"),
           ep("Mario1-1", False, "Pit"),
           ep("Mario1-1", False, "Pit")]
    out = summarize_eval(eps)
    assert out["n"] == 4
    assert out["wins"] == 2
    assert math.isclose(out["win_rate"], 0.5)
    assert out["by_cause"]["pit"] == 2
    # wins are NOT counted in by_cause
    assert sum(out["by_cause"].values()) == 2


def test_cause_bucket_mapping():
    eps = [
        ep("L", False, "Pit"),       # pit
        ep("L", False, "OOB"),       # pit (out-of-bounds folds into pit)
        ep("L", False, "Stall"),     # stall
        ep("L", False, "Koopa"),     # enemy
        ep("L", False, "Enemy"),     # enemy
        ep("L", False, "Spike"),     # enemy
        ep("L", False, "Timeout"),   # timeout
        ep("L", False, "Unknown"),   # other
    ]
    out = summarize_eval(eps)
    assert out["by_cause"] == {
        "pit": 2, "wall": 0, "stall": 1, "enemy": 3, "timeout": 1, "other": 1,
    }
    assert out["wins"] == 0
    assert out["win_rate"] == 0.0


def test_truncation_without_death_counts_as_timeout():
    # An episode truncated by the eval harness step-cap: no WIN, no death cause.
    eps = [ep("L", False, "", event="")]
    out = summarize_eval(eps)
    assert out["by_cause"]["timeout"] == 1
    assert out["wins"] == 0


def test_won_flag_wins_over_cause():
    # Defensive: if won=True but cause left stale, still a win, not a death bucket.
    eps = [ep("L", True, "Pit")]
    out = summarize_eval(eps)
    assert out["wins"] == 1
    assert sum(out["by_cause"].values()) == 0


def test_summarize_by_level_groups():
    eps = [
        ep("Mario1-1", True, "Goal"),
        ep("Mario1-1", False, "Pit"),
        ep("Mario1-2", False, "Enemy"),
        ep("Mario1-2", False, "Enemy"),
    ]
    out = summarize_by_level(eps)
    assert set(out.keys()) == {"Mario1-1", "Mario1-2"}
    assert out["Mario1-1"]["win_rate"] == 0.5
    assert out["Mario1-1"]["by_cause"]["pit"] == 1
    assert out["Mario1-2"]["win_rate"] == 0.0
    assert out["Mario1-2"]["by_cause"]["enemy"] == 2


def test_cause_buckets_table_is_complete():
    # Every raw cause string the platformer can emit is mapped.
    for raw in ["Pit", "OOB", "Stall", "Koopa", "Enemy", "Spike", "Timeout", "Time Over"]:
        assert raw in CAUSE_BUCKETS, f"{raw} missing from CAUSE_BUCKETS"
