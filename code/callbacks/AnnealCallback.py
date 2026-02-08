import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EventCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv, DummyVecEnv
from sb3_contrib import RecurrentPPO
from typing import Any, Dict, Optional, Union
import os

class AnnealCallback(BaseCallback):
    """
    Callback to anneal entropy coefficient and gradient clipping during training.
    Only works with algorithms that have these attributes (PPO, A2C).
    """
    def __init__(self, total_timesteps, start_ent=0.1, end_ent=0.01, start_grad_clip=1.0, end_grad_clip=0.3, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.start_ent = start_ent
        self.end_ent = end_ent
        self.start_grad_clip = start_grad_clip
        self.end_grad_clip = end_grad_clip

    def _on_step(self) -> bool:
        # Calculate current fraction of progress
        frac = self.num_timesteps / self.total_timesteps
        # Linearly interpolate
        ent_coef = self.start_ent * (1 - frac) + self.end_ent * frac
        max_grad_norm = self.start_grad_clip * (1 - frac) + self.end_grad_clip * frac
        
        # Update parameters if they exist (PPO, A2C, RecurrentPPO)
        if hasattr(self.model, 'ent_coef'):
            self.model.ent_coef = ent_coef
        if hasattr(self.model, 'max_grad_norm'):
            self.model.max_grad_norm = max_grad_norm
        
        return True
