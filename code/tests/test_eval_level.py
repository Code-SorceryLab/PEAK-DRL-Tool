"""Unit tests for code.scripts.eval_level.run_eval using fake model + fake env."""
from __future__ import annotations

import numpy as np
import pytest

from code.metrics.eval_summary import summarize_eval


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeModel:
    """Always returns action=0 and no LSTM state."""

    def predict(self, obs, deterministic=True, state=None, episode_start=None):
        # obs may be a numpy array of shape (1, ...) from a VecEnv stub
        return np.array([0]), None


class FakeVecEnv:
    """
    A minimal VecEnv stub.

    Each episode lasts exactly `steps_per_episode` steps, then done=True.
    The final info contains won/cause/level.
    """

    def __init__(self, steps_per_episode: int = 3, won: bool = True,
                 cause: str = "Goal", level: str = "Mario1-1"):
        self.steps_per_episode = steps_per_episode
        self._won = won
        self._cause = cause
        self._level = level
        self._step_count = 0
        self.num_envs = 1

    def reset(self):
        self._step_count = 0
        obs = np.zeros((1, 4), dtype=np.float32)
        return obs

    def step(self, action):
        self._step_count += 1
        obs = np.zeros((1, 4), dtype=np.float32)
        done = self._step_count >= self.steps_per_episode
        reward = np.array([1.0])
        info = [{"won": self._won, "cause": self._cause, "level": self._level}]
        return obs, reward, np.array([done]), info

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Import the function under test
# ---------------------------------------------------------------------------

from code.scripts.eval_level import run_eval  # noqa: E402


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_run_eval_returns_one_dict_per_episode():
    """run_eval should return exactly `episodes` final-info dicts."""
    model = FakeModel()
    env = FakeVecEnv(steps_per_episode=3, won=True, cause="Goal", level="Mario1-1")

    result = run_eval(model, env, episodes=5, deterministic=True, level="Mario1-1")

    assert len(result) == 5, f"Expected 5 dicts, got {len(result)}"
    for d in result:
        assert isinstance(d, dict)
        assert "won" in d
        assert "cause" in d
        assert "level" in d


def test_run_eval_captures_correct_won_value():
    """run_eval captures won=True when FakeVecEnv always wins."""
    model = FakeModel()
    env = FakeVecEnv(steps_per_episode=2, won=True, cause="Goal", level="Mario1-1")

    result = run_eval(model, env, episodes=4, deterministic=True, level="Mario1-1")

    assert all(d["won"] is True for d in result)


def test_run_eval_captures_losing_episodes():
    """run_eval captures won=False with the correct cause."""
    model = FakeModel()
    env = FakeVecEnv(steps_per_episode=2, won=False, cause="Pit", level="Mario1-1")

    result = run_eval(model, env, episodes=3, deterministic=True, level="Mario1-1")

    assert all(d["won"] is False for d in result)
    assert all(d["cause"] == "Pit" for d in result)


def test_run_eval_injects_level_when_missing():
    """run_eval injects --level into the info dict if env doesn't provide it."""
    model = FakeModel()

    class NoLevelEnv(FakeVecEnv):
        def step(self, action):
            obs, reward, dones, info = super().step(action)
            # Remove 'level' key from info
            for d in info:
                d.pop("level", None)
            return obs, reward, dones, info

    env = NoLevelEnv(steps_per_episode=2, won=True, cause="Goal", level="Mario1-1")

    result = run_eval(model, env, episodes=2, deterministic=True, level="Injected1-1")

    assert all(d.get("level") == "Injected1-1" for d in result)


def test_run_eval_feeds_summarize_eval():
    """Dicts from run_eval should feed summarize_eval without errors."""
    model = FakeModel()
    # 2 wins, 3 losses (Pit)
    results = []
    for won, cause in [(True, "Goal"), (True, "Goal"), (False, "Pit"),
                       (False, "Pit"), (False, "Pit")]:
        env = FakeVecEnv(steps_per_episode=2, won=won, cause=cause, level="Mario1-1")
        ep_result = run_eval(model, env, episodes=1, deterministic=True, level="Mario1-1")
        results.extend(ep_result)

    summary = summarize_eval(results)
    assert summary["n"] == 5
    assert summary["wins"] == 2
    assert abs(summary["win_rate"] - 0.4) < 1e-9
    assert summary["by_cause"]["pit"] == 3


def test_run_eval_zero_episodes():
    """run_eval with episodes=0 returns an empty list."""
    model = FakeModel()
    env = FakeVecEnv()

    result = run_eval(model, env, episodes=0, deterministic=True, level="Mario1-1")

    assert result == []
