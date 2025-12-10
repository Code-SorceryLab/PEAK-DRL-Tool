# callbacks.py  ── SB3 helpers for preview / no-op
from stable_baselines3.common.callbacks import BaseCallback
import pygame, time

class NoOp(BaseCallback):
    """Does nothing – keeps head-less run fast."""
    def _on_step(self) -> bool:
        return True

class RenderCallback(BaseCallback):
    """
    Call env.render() every `every` env-steps so we can watch training.
    fps-throttling is handled by the env (fps argument).
    """
    def __init__(self, every: int = 1, verbose: int = 0):
        super().__init__(verbose)
        self.every = every

    def _on_step(self) -> bool:
        if self.n_calls % self.every == 0:
            env = self.training_env.envs[0]        # unwrap VecEnv
            env.render()
            pygame.event.pump()                    # keep window responsive
        return True
