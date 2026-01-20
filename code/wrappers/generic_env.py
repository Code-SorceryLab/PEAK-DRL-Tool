"""Universal Gymnasium wrapper with pluggable reward & HUD."""
from __future__ import annotations
import os, pygame, gymnasium as gym
import numpy as np
from typing import Callable, Any
import importlib
from .RewardHub import RewardHub

Obs = Any
Info = dict

class GameEnv(gym.Env):
    metadata = {"render_modes": ["none", "human", "random", "rgb_array"]}

    def __init__(
        self,
        game_cls: type,               # subclass of games.*.FlappyCore-like
        *,
        render_mode: str = "none",
        fps: int | None = 30,
        max_steps: int | None = None,
        reward_fn: Callable[[Obs, float | None, bool, Info], float] | None = None,
        hud_fn: Callable[[pygame.Surface, "GameEnv"], None] | None = None,
        **game_kwargs,
    ):
        assert render_mode in self.metadata["render_modes"]
        self.game = game_cls(render_mode = render_mode, **game_kwargs)
        # self.game = game_cls  # not game_cls(**game_kwargs)
        self.render_mode = render_mode
        self.fps = fps
        self.max_steps = max_steps
        self.reward_fn = reward_fn or self._default_reward
        self.hud_fn = hud_fn

        # spaces from game
        self.action_space = self.game.get_action_space()
        self.observation_space = self.game.get_observation_space()

        # episode counters
        self._step_count = 0

        # GUI lazy data
        self.screen = None
        self.clock = None
        self.font = None

    # -------------------------------- default reward
    @staticmethod
    def _default_reward(obs, base, terminated, info):
        return 0.0 if base is None else base

    # -------------------------------- Gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        obs = self.game.reset()
        return obs, {}

    def step(self, action):
        self._step_count += 1

        # --- NEW: normalize action once, do NOT cast to bool unconditionally
        if hasattr(self.action_space, "n"):
            if self.render_mode == "random":
                action = self.action_space.sample()
            # Discrete
            if isinstance(action, (np.generic, np.ndarray)):  # e.g., numpy scalar from SB3
                action = int(action)
        # For Binary(1)/MultiBinary just pass through
        # For Box actions (not used here), pass as-is

        # OLD (remove): obs, base, terminated, info = self.game.step(bool(action))
        obs, base, terminated, info = self.game.step(action)

        truncated = bool(self.max_steps and self._step_count >= self.max_steps)
        reward = self.reward_fn(obs, base, terminated, info)
        hub = RewardHub.get_instance()
        hub.update_reward(reward=reward, action_name=action, is_episode_end=terminated)
        return obs, reward, terminated, truncated, info


    # -------------------------------- rendering
    # -------------------------------- rendering
    def render(self, mode=None):
        """
        Modes:
        - "human": normal on-screen rendering
        - "random": same as human but with random actions (handled in step)
        - "rgb_array": return (H, W, 3) numpy array for video recording
        - "none": no rendering
        """
        # Resolve effective mode
        if mode is not None:
            render_mode = mode
        else:
            render_mode = self.render_mode

        if render_mode == "none":
            return

        # Lazy-init screen for any mode that needs a surface
        if self.screen is None:
            if os.environ.get("DISPLAY", "") == "":
                os.environ["SDL_VIDEODRIVER"] = "dummy"
            pygame.init()
            self.screen = pygame.display.set_mode(
                (self.game.WIDTH, self.game.HEIGHT)
            )
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(None, 20)

        # Ask game to draw onto our surface
        # (game.render must support blit_only=True in your cores)
        self.game.render(self.screen, blit_only=True)

        # Optional HUD
        if self.hud_fn:
            self.hud_fn(self.screen, self)

        if render_mode in ("human", "random"):
            # On-screen display
            if self.fps and self.clock:
                self.clock.tick(self.fps)
            pygame.display.flip()
            return

        if render_mode == "rgb_array":
            # Return pixels for VecVideoRecorder (H, W, 3)
            import numpy as np
            arr = pygame.surfarray.array3d(self.screen)  # (W, H, 3)
            arr = np.transpose(arr, (1, 0, 2))          # (H, W, 3)
            return arr


    def close(self):
        if self.screen:
            pygame.quit()
            self.screen = None