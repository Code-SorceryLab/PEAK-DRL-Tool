# code/games/pacman_core.py
from __future__ import annotations
import os
os.environ["ALE_ACCEPT_ROM_LICENSE"] = "YES"  # auto-accept ALE ROM license

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from code.wrappers.pacman_wrapper import Controls, RawScoreTracker, FrameStack, Scale

class PacmanCore:
    def __init__(
        self,
        render_mode=None,
        fps=None,
        max_steps=None,
        stack: int = 4,
        scale: bool = True,
        frameskip: int = 4,
        sticky: float = 0.0,
    ):
        self.render_mode = render_mode
        self.stack = int(stack)
        self.scale = bool(scale)
        self.frameskip = int(frameskip)
        self.sticky = float(sticky)

        env = gym.make(
            "ALE/MsPacman-v5",
            obs_type="ram",
            render_mode=self.render_mode,
            frameskip=self.frameskip,
            repeat_action_probability=self.sticky,
        )

        # Wrapper order: actions -> tracking -> RAM stack -> scale -> monitor
        env = Controls(env)
        env = RawScoreTracker(env)
        env = FrameStack(env, num_stack=self.stack)
        env = Scale(env, scale=self.scale)
        env = Monitor(env)

        self.env = env

    def reset(self, *, seed=None):
        return self.env.reset(seed=seed)

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        if "lives" not in info:
            try:
                info = dict(info)
                info["lives"] = int(self.env.unwrapped.ale.lives())
            except Exception:
                pass
        return obs, rew, term, trunc, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()
