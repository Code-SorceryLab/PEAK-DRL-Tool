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
        frame_skip: int = 1,
        reward_fn: Callable[[Obs, float | None, bool, Info], float] | None = None,
        hud_fn: Callable[[pygame.Surface, "GameEnv"], None] | None = None,
        **game_kwargs,
    ):
        assert render_mode in self.metadata["render_modes"]
        self.game = game_cls(render_mode = render_mode, max_steps=max_steps, **game_kwargs)
        self.render_mode = render_mode
        self.fps = fps
        self.max_steps = max_steps
        # Action repeat (classic Mario/Atari "frame-skip"): each agent decision
        # is applied for k physics frames, rewards summed over the skip.
        # k=1 (default) is byte-identical to the pre-frame-skip behaviour.
        # Episode budgets: cores with a frame counter (platformer/megaman/
        # sonic) truncate on max_steps FRAMES, so their in-game episode budget
        # is invariant to the skip — episodes just take ~k× fewer decisions.
        # The wrapper's own _step_count bound below counts DECISIONS and acts
        # as a secondary guard (and the only bound for cores without a frame
        # counter, e.g. meatboy).
        self.frame_skip = max(1, int(frame_skip))
        self.hud_fn = hud_fn

        # If the reward_fn is a factory (produced by _wrap_with_tracker),
        # call it to get a fresh instance with its own _ScoreTracker.
        # This is critical for parallel training — each env MUST have its own tracker.
        if reward_fn is not None and getattr(reward_fn, "_is_factory", False):
            self.reward_fn = reward_fn()
        else:
            self.reward_fn = reward_fn or self._default_reward

        # spaces from game
        self.action_space = self.game.get_action_space()
        self.observation_space = self.game.get_observation_space()

        # episode counters
        self._step_count = 0

        # --- FIX: Create a dedicated RewardHub for THIS environment instance ---
        self.hub = RewardHub()

        # --- FIX: Inject Hub into Game's Debug Manager ---
        # This connects the wrapper's reward tracking to the inner game's visualizer
        if hasattr(self.game, "debug_manager"):
            self.game.debug_manager.hub = self.hub

        # GUI lazy data
        self.screen = None
        self.clock = None
        self.font = None

    # -------------------------------- Default reward
    @staticmethod
    def _default_reward(obs, base, terminated, info):
        return 0.0 if base is None else base

    # -------------------------------- Gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        obs, info = self.game.reset()
        return obs, info

    def step(self, action):
        self._step_count += 1

        # --- Normalizing Action (Legacy) ---
        if hasattr(self.action_space, "n"):
            if self.render_mode == "random":
                action = self.action_space.sample()
            if isinstance(action, (np.generic, np.ndarray)):
                action = int(action)

        # --- Frame-skip loop (action repeat) ---
        # The chosen action is applied for frame_skip physics frames; the
        # persona reward is computed PER FRAME (tracker cadence unchanged)
        # and summed, so discrete events inside the skip (coin, kill, death,
        # win) are never lost. Terminal frames break out immediately — their
        # reward (win bonus / death penalty) is captured before the break.
        # The agent sees the LAST frame's obs/info, per Mario/Atari convention.
        final_scalar_reward = 0.0
        summed_breakdown: dict = {}

        for _ in range(self.frame_skip):
            obs, base, terminated, truncated, info = self.game.step(action)
            # FIX: OR with game's own truncated rather than overwriting it entirely.
            # If the game sets truncated=True (e.g. internal timer), the wrapper was
            # previously silently discarding it.
            truncated = truncated or bool(self.max_steps and self._step_count >= self.max_steps)

            # --- REWARD PROCESSING (THE FIX) ---
            # 1. Get raw reward from Persona (Could be Float OR Dict)
            if self.reward_fn:
                raw_reward = self.reward_fn(obs, base, terminated, info)
            else:
                raw_reward = self.hub.compute_default_reward(info)

            # 2. Handle both formats safely
            if isinstance(raw_reward, dict):
                frame_reward = float(sum(raw_reward.values()))
                frame_breakdown = raw_reward
            else:
                frame_reward = float(raw_reward)
                # Prefer the breakdown injected by the persona wrapper over a generic fallback
                frame_breakdown = info.get("reward_components", {"reward": frame_reward})

            final_scalar_reward += frame_reward
            for k, v in frame_breakdown.items():
                summed_breakdown[k] = summed_breakdown.get(k, 0.0) + float(v)

            if terminated or truncated:
                break

        # 3. Inject Breakdown into Info (for CSV Logger) — summed over the skip
        info["reward_breakdown"] = summed_breakdown

        # SB3 success-rate convention: emit is_success on the terminal step so
        # evaluate_policy fills EvalCallback's success buffer. This is what
        # lets best-model selection rank checkpoints by WIN RATE instead of
        # mean reward (a long stalling episode can out-earn a fast win — the
        # historically shipped best_model.zip was exactly such an artifact).
        # Info-only: does not touch obs, reward, or termination.
        if terminated or truncated:
            info["is_success"] = bool(info.get("won", False))

        # 4. Update Hub (Visuals) - PASS THE FLOAT SUM HERE!
        act_name = info.get("action_name")
        if act_name is None:
            if hasattr(self.game, "action_to_str"):
                act_name = self.game.action_to_str(action)
            elif hasattr(self.game, "ACTION_NAMES"):
                try:
                    act_name = self.game.ACTION_NAMES.get(int(action), str(action))
                except (TypeError, ValueError):
                    act_name = str(action)
            else:
                act_name = str(action)


        info["action_name"] = str(act_name)


        # <--- THIS WAS CRASHING BEFORE (reward=raw_reward would fail)
        self.hub.update_reward(reward=final_scalar_reward, action_name=act_name, is_episode_end=terminated)

        # 5. Return the Float Sum to the Agent
        return obs, final_scalar_reward, terminated, truncated, info

    # -------------------------------- Rendering
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
            # Only force headless on Linux when no X display is available AND
            # the caller didn't explicitly request a visible window.
            # DISPLAY is a Linux/X11 env-var — it's never set on Windows,
            # so the old check always forced dummy driver on Windows too.
            import sys as _sys
            if (_sys.platform.startswith("linux")
                    and os.environ.get("DISPLAY", "") == ""
                    and render_mode not in ("human", "random")):
                os.environ["SDL_VIDEODRIVER"] = "dummy"
            pygame.init()
            # Use the game's total width (includes debug panel in human mode)
            total_w = getattr(self.game, 'TOTAL_WIDTH', self.game.WIDTH)
            self.screen = pygame.display.set_mode(
                (total_w, self.game.HEIGHT)
            )
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(None, 20)

        # Ask game nicely to draw onto our surface
        # NOTE: (game.render must support blit_only=True in your cores)
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
            arr = pygame.surfarray.array3d(self.screen)  # (W, H, 3)
            arr = np.transpose(arr, (1, 0, 2))          # (H, W, 3)
            return arr


    def close(self):
        if self.screen:
            pygame.quit()
            self.screen = None
