import numpy as np
import gymnasium as gym
from gymnasium import spaces

class Controls(gym.ActionWrapper):
    """Discrete(4): 0=UP,1=DOWN,2=LEFT,3=RIGHT -> ALE indices."""
    def __init__(self, env):
        super().__init__(env)
        meanings = self.unwrapped.get_action_meanings()
        want = {"UP": None, "DOWN": None, "LEFT": None, "RIGHT": None}
        for i, m in enumerate(meanings):
            mm = m.upper()
            if mm in want and want[mm] is None:
                want[mm] = i
        missing = [k for k, v in want.items() if v is None]
        if missing:
            raise RuntimeError(f"Missing actions {missing} in {meanings}")
        self._map = np.array([want["UP"], want["DOWN"], want["LEFT"], want["RIGHT"]], dtype=np.int64)
        self.action_space = spaces.Discrete(4)
    def action(self, a): return int(self._map[int(a)])

class RawScoreTracker(gym.Wrapper):
    """Expose raw ALE score via info; episode total at done."""
    def __init__(self, env): super().__init__(env); self._score = 0.0
    def reset(self, **kwargs):
        self._score = 0.0
        return self.env.reset(**kwargs)
    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        self._score += float(r)
        info = dict(info); info["ale_score"] = self._score
        if term or trunc: info["episode_ale_score"] = self._score
        return obs, r, term, trunc, info

class Scale(gym.ObservationWrapper):
    """(N,) uint8 RAM -> float32 [0,1]."""
    def __init__(self, env, scale=True):
        super().__init__(env)
        self.scale = bool(scale)
        n = int(env.observation_space.shape[0])
        low, high = (0.0, 1.0) if self.scale else (0, 255)
        self.observation_space = spaces.Box(low=low, high=high, shape=(n,), dtype=np.float32 if self.scale else np.uint8)
    def observation(self, obs):
        arr = np.asarray(obs);  return (arr.astype(np.float32)/255.0) if self.scale else arr

class FrameStack(gym.ObservationWrapper):
    """Stack k RAM frames: (N,) -> (N*k,)."""
    def __init__(self, env, num_stack=4):
        super().__init__(env)
        self.k = int(num_stack)
        n = int(env.observation_space.shape[0])
        self._buf = np.zeros((self.k, n), dtype=env.observation_space.dtype)
        low = np.repeat(env.observation_space.low, self.k)
        high = np.repeat(env.observation_space.high, self.k)
        self.observation_space = spaces.Box(low=low, high=high, dtype=env.observation_space.dtype)
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs); self._buf[...] = obs
        return self._buf.reshape(-1), info
    def observation(self, obs):
        self._buf[:-1] = self._buf[1:]; self._buf[-1] = obs
        return self._buf.reshape(-1)
