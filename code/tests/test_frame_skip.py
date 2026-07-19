"""Frame-skip (action repeat) regression tests.

Classic Mario/Atari recipe: each agent decision spans k physics frames,
rewards summed over the skip, terminal frames break out immediately, the
agent sees the last frame's obs. frame_skip=1 (default) must be
byte-identical to the legacy per-frame behaviour — every reference clone
(uvipen, CleanRL, SB3-zoo) uses k=4; PEAK's lack of it was the single
biggest effective-experience gap vs the 70% baselines.
"""
import numpy as np
import gymnasium.spaces as spaces

from code.wrappers.generic_env import GameEnv


class _StubGame:
    """Minimal game core: 1D walk, +1 base reward per frame, terminates at
    frame `die_at`. Lets tests count frames and check mid-skip handling."""
    WIDTH, HEIGHT = 84, 84
    die_at = None          # class attr so GameEnv's ctor kwargs stay simple

    def __init__(self, render_mode="none", max_steps=None, **kwargs):
        self.frame = 0

    def get_action_space(self):
        return spaces.Discrete(2)

    def get_observation_space(self):
        return spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)

    def reset(self):
        self.frame = 0
        return np.array([0.0], dtype=np.float32), {}

    def step(self, action):
        self.frame += 1
        terminated = self.die_at is not None and self.frame >= self.die_at
        info = {"frame": self.frame, "won": False}
        return (np.array([float(self.frame)], dtype=np.float32),
                1.0, terminated, False, info)


class _StubGameDiesAt2(_StubGame):
    die_at = 2


def _mk(game_cls=_StubGame, **kwargs):
    env = GameEnv(game_cls, render_mode="none", **kwargs)
    env.reset()
    return env


def test_default_skip_is_one_frame_per_step():
    env = _mk()
    obs, r, term, trunc, info = env.step(0)
    assert info["frame"] == 1          # exactly one physics frame
    assert r == 1.0
    env.close()


def test_skip4_advances_four_frames_and_sums_reward():
    env = _mk(frame_skip=4)
    obs, r, term, trunc, info = env.step(0)
    assert info["frame"] == 4          # four physics frames per decision
    assert r == 4.0                    # +1 base reward per frame, summed
    assert obs[0] == 4.0               # obs is the LAST frame's
    env.close()


def test_mid_skip_termination_breaks_early():
    env = _mk(_StubGameDiesAt2, frame_skip=4)
    obs, r, term, trunc, info = env.step(0)
    assert term is True
    assert info["frame"] == 2          # stopped AT the terminal frame, not 4
    assert r == 2.0                    # reward up to and including terminal frame
    assert info["is_success"] is False # terminal info flows through the skip
    env.close()


def test_skip_sums_reward_breakdown_components():
    calls = {"n": 0}

    def per_frame_reward(obs, base, terminated, info):
        calls["n"] += 1
        return {"move": 0.25, "alive": 0.5}

    env = GameEnv(_StubGame, render_mode="none", frame_skip=4,
                  reward_fn=per_frame_reward)
    env.reset()
    obs, r, term, trunc, info = env.step(0)
    assert calls["n"] == 4                                # persona runs per frame
    assert r == 4 * 0.75                                  # scalar summed
    assert info["reward_breakdown"] == {"move": 1.0, "alive": 2.0}  # per-key sums
    env.close()


def test_skip_invalid_values_clamped():
    env = _mk(frame_skip=0)
    assert env.frame_skip == 1
    obs, r, term, trunc, info = env.step(0)
    assert info["frame"] == 1
    env.close()


def test_platformer_core_respects_skip():
    """End-to-end with the real game: frame_count must advance by k per step."""
    from code.games.platformer_core import PlatformerCore
    env = GameEnv(PlatformerCore, render_mode="none", frame_skip=4,
                  max_steps=100, world="Mario1-1")
    env.reset()
    _, _, _, _, i1 = env.step([3, 0, 0])
    _, _, _, _, i2 = env.step([3, 0, 0])
    assert i2["frame_count"] - i1["frame_count"] == 4
    env.close()
