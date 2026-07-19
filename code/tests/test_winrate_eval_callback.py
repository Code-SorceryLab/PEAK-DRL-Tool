"""Best-model selection regression: rank checkpoints by WIN RATE.

The historical bug: the stock EvalCallback selects best_model.zip by mean
episode reward. Under these personas a long stalling episode (alive +
potential income over thousands of steps) out-earned fast wins — the shipped
best_model.zip was a 480k checkpoint selected on a 4,486-step STALL episode.
WinRateEvalCallback ranks by (win_rate, mean_reward) lexicographic instead.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from code.scripts.train import WinRateEvalCallback


class _TinyEnv(gym.Env):
    """Minimal env — exists only so EvalCallback.__init__ has an eval_env."""
    observation_space = spaces.Box(-1, 1, (2,), np.float32)
    action_space = spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(2, np.float32), {}

    def step(self, action):
        return np.zeros(2, np.float32), 0.0, True, False, {}


class _RecordingModel:
    def __init__(self):
        self.saves = 0

    def save(self, path):
        self.saves += 1


def _make_cb(tmp_path):
    cb = WinRateEvalCallback(
        DummyVecEnv([lambda: _TinyEnv()]),
        best_model_save_path=str(tmp_path),
        eval_freq=1, verbose=0,
    )
    cb.model = _RecordingModel()
    cb.n_calls = 0
    return cb


def _run_eval(cb, monkeypatch, mean_reward, successes):
    """Simulate one eval tick: stub the stock parent _on_step to publish
    last_mean_reward + the success buffer exactly the way SB3 does."""
    def fake_parent_on_step(self):
        self.last_mean_reward = mean_reward
        self._is_success_buffer = list(successes)
        return True

    monkeypatch.setattr(EvalCallback, "_on_step", fake_parent_on_step)
    cb.n_calls += 1
    return cb._on_step()


def test_higher_win_rate_beats_higher_reward(tmp_path, monkeypatch):
    """The exact regression: a high-reward 0%-win eval must NOT dethrone a
    lower-reward eval that actually wins."""
    cb = _make_cb(tmp_path)

    _run_eval(cb, monkeypatch, mean_reward=5.0, successes=[False] * 5)
    assert cb.model.saves == 1                       # first eval = first best

    _run_eval(cb, monkeypatch, mean_reward=1.0, successes=[True, True, False, False, False])
    assert cb.model.saves == 2                       # 40% wins > 0% wins, despite lower reward
    assert cb.best_win_rate == 0.4

    _run_eval(cb, monkeypatch, mean_reward=99.0, successes=[False] * 5)
    assert cb.model.saves == 2                       # stall artifact: huge reward, 0% wins → NO save
    assert cb.best_win_rate == 0.4


def test_reward_breaks_ties_within_equal_win_rate(tmp_path, monkeypatch):
    cb = _make_cb(tmp_path)
    _run_eval(cb, monkeypatch, mean_reward=1.0, successes=[True, False])
    assert cb.model.saves == 1
    _run_eval(cb, monkeypatch, mean_reward=2.0, successes=[True, False])
    assert cb.model.saves == 2                       # same 50%, faster/cleaner win → save
    _run_eval(cb, monkeypatch, mean_reward=1.5, successes=[True, False])
    assert cb.model.saves == 2                       # same 50%, worse reward → no save


def test_fallback_to_reward_when_no_success_info(tmp_path, monkeypatch):
    """Envs that never emit is_success must degrade to stock reward ranking,
    not silently stop saving best models."""
    cb = _make_cb(tmp_path)
    _run_eval(cb, monkeypatch, mean_reward=1.0, successes=[])
    assert cb.model.saves == 1
    _run_eval(cb, monkeypatch, mean_reward=0.5, successes=[])
    assert cb.model.saves == 1
    _run_eval(cb, monkeypatch, mean_reward=2.0, successes=[])
    assert cb.model.saves == 2


def test_best_marker_bumps_on_every_save(tmp_path, monkeypatch):
    """EvalPreviewCallback watches best_marker to keep the vecnorm snapshot
    in sync — it must increment exactly when a save happens."""
    cb = _make_cb(tmp_path)
    assert cb.best_marker == 0
    _run_eval(cb, monkeypatch, mean_reward=1.0, successes=[False])
    assert cb.best_marker == 1
    _run_eval(cb, monkeypatch, mean_reward=0.1, successes=[True])
    assert cb.best_marker == 2
    _run_eval(cb, monkeypatch, mean_reward=50.0, successes=[False])
    assert cb.best_marker == 2                       # no save, no bump


def test_gameenv_emits_is_success_on_terminal_only():
    """GameEnv must emit is_success exactly on terminal steps (SB3's
    success-buffer convention) — False here since truncation is not a win."""
    from code.wrappers.generic_env import GameEnv
    from code.games.platformer_core import PlatformerCore

    env = GameEnv(PlatformerCore, render_mode="none", max_steps=3, world="Mario1-1")
    env.reset()
    done, infos = False, []
    while not done:
        _, _, term, trunc, info = env.step([0, 0, 0])
        done = term or trunc
        infos.append((done, info))
        assert len(infos) <= 10, "env never terminated"

    *body, (last_done, last_info) = infos
    assert all("is_success" not in i for d, i in body)
    assert last_done and last_info["is_success"] is False
    env.close()
