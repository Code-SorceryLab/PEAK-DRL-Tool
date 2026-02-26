"""
LiveVisualizationCallback
=========================
Opens a real pygame window showing the agent playing the game at a set
interval during training.

How it works
------------
- On training START: builds one GameEnv in render_mode="human" and keeps it
  alive for the entire training run. This avoids repeatedly calling
  pygame.display.set_mode() which exhausts Windows WM object handles.
- Every viz_freq steps: copies live VecNorm stats from the training env,
  resets the preview env, plays n_preview_episodes, then hides the window.
- On training END: closes the env cleanly.

Training is paused for the duration of the preview (a few seconds per episode).

Setup (conf/grid.yaml)
----------------------
    viz_enabled: true
    viz_freq: 50000
    viz_preview_episodes: 1
    viz_fps: 30
"""

import os
import traceback

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


class LiveVisualizationCallback(BaseCallback):
    """
    Inline live preview — one persistent pygame window, opened at training
    start and reused at every viz trigger.

    Parameters
    ----------
    game_cls : type
        The game core class (e.g. PlatformerCore).
    reward_fn : callable | None
        The active reward function — same one used for training.
    env_kwargs : dict
        Extra kwargs forwarded to GameEnv (persona, world, etc.).
    viz_freq : int
        Fire every N training timesteps.
    n_preview_episodes : int
        Number of episodes to play per preview.
    fps : int
        Playback FPS.
    verbose : int
        SB3 verbosity level.
    """

    def __init__(
        self,
        game_cls,
        reward_fn,
        env_kwargs: dict,
        viz_freq: int = 50_000,
        n_preview_episodes: int = 1,
        fps: int = 30,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.game_cls           = game_cls
        self.reward_fn          = reward_fn
        self.env_kwargs         = env_kwargs
        self.viz_freq           = viz_freq
        self.n_preview_episodes = n_preview_episodes
        self.fps                = fps
        self._last_viz_step     = 0
        self._preview_env       = None   # built once in _on_training_start

    # ------------------------------------------------------------------
    def _build_preview_env(self):
        """
        Create the persistent preview env. Called ONCE at training start.
        Building it here means only one pygame.display.set_mode() call ever
        occurs, avoiding Windows WM handle exhaustion.
        """
        from code.wrappers.generic_env import GameEnv

        # Remove headless SDL flag so pygame can open a real window
        os.environ.pop("SDL_VIDEODRIVER", None)

        kw = {k: v for k, v in self.env_kwargs.items() if k != "render_mode"}
        kw["render_mode"] = "human"
        kw["fps"]         = self.fps

        raw_env = GameEnv(self.game_cls, reward_fn=self.reward_fn, **kw)
        vec_env = DummyVecEnv([lambda: raw_env])

        # Wrap in VecNormalize — stats will be synced from training env at
        # each preview trigger.
        train_norm = self._get_training_vecnorm()
        if train_norm is not None:
            preview_env = VecNormalize(
                vec_env,
                norm_obs    = train_norm.norm_obs,
                norm_reward = False,
                clip_obs    = train_norm.clip_obs,
            )
            preview_env.obs_rms  = train_norm.obs_rms
            preview_env.ret_rms  = train_norm.ret_rms
            preview_env.training = False
        else:
            preview_env = vec_env
            print("[LiveViz] WARNING: VecNormalize not found — obs will NOT be normalised.")

        return preview_env

    # ------------------------------------------------------------------
    def _get_training_vecnorm(self):
        """Walk the wrapper stack to find the VecNormalize used in training."""
        env = self.training_env
        if isinstance(env, VecNormalize):
            return env
        while hasattr(env, "venv"):
            env = env.venv
            if isinstance(env, VecNormalize):
                return env
        return None

    # ------------------------------------------------------------------
    def _sync_vecnorm(self):
        """Copy the latest running stats from the training env into the preview env."""
        if not isinstance(self._preview_env, VecNormalize):
            return
        train_norm = self._get_training_vecnorm()
        if train_norm is not None:
            self._preview_env.obs_rms  = train_norm.obs_rms
            self._preview_env.ret_rms  = train_norm.ret_rms
            self._preview_env.training = False

    # ------------------------------------------------------------------
    def _on_training_start(self) -> None:
        if self.verbose:
            print(
                f"[LiveViz] Enabled — preview every {self.viz_freq:,} steps | "
                f"{self.n_preview_episodes} episode(s) at {self.fps} FPS"
            )
        try:
            self._preview_env = self._build_preview_env()
            if self.verbose:
                print("[LiveViz] Preview env created — window will appear at first trigger.")
        except Exception as e:
            print(f"[LiveViz] WARNING: could not create preview env — {e}")
            traceback.print_exc()
            self._preview_env = None

    # ------------------------------------------------------------------
    def _on_step(self) -> bool:
        if self._preview_env is None:
            return True
        if self.num_timesteps - self._last_viz_step < self.viz_freq:
            return True

        self._last_viz_step = self.num_timesteps

        if self.verbose:
            print(f"\n[LiveViz] Step {self.num_timesteps:,} — running preview...")

        try:
            self._sync_vecnorm()
            self._run_preview()
        except Exception as e:
            print(f"[LiveViz] WARNING: preview failed — {e}")
            traceback.print_exc()

        return True

    # ------------------------------------------------------------------
    def _run_preview(self):
        """Play n_preview_episodes using the persistent preview env."""
        import pygame

        # Detect RecurrentPPO
        try:
            from sb3_contrib import RecurrentPPO
            is_recurrent = isinstance(self.model, RecurrentPPO)
        except ImportError:
            is_recurrent = False

        pygame.display.set_caption(f"PEAK Live Preview — step {self.num_timesteps:,}")

        for ep in range(self.n_preview_episodes):
            obs         = self._preview_env.reset()
            done        = False
            lstm_states = None
            ep_start    = np.ones((1,), dtype=bool)
            ep_reward   = 0.0
            steps       = 0

            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        if self.verbose:
                            print("[LiveViz] Preview skipped (ESC).")
                        return
                pygame.event.pump()

                if is_recurrent:
                    action, lstm_states = self.model.predict(
                        obs, state=lstm_states,
                        episode_start=ep_start, deterministic=True,
                    )
                    ep_start = np.zeros((1,), dtype=bool)
                else:
                    action, _ = self.model.predict(obs, deterministic=True)

                obs, reward, dones, _ = self._preview_env.step(action)
                done       = bool(dones[0])
                ep_reward += float(reward[0])
                steps     += 1

            if self.verbose:
                print(
                    f"[LiveViz] Ep {ep + 1}/{self.n_preview_episodes} "
                    f"reward={ep_reward:.1f}  steps={steps}"
                )

        if self.verbose:
            print("[LiveViz] Preview done — training resumed.\n")

    # ------------------------------------------------------------------
    def _on_training_end(self) -> None:
        if self._preview_env is not None:
            try:
                self._preview_env.close()
            except Exception:
                pass
            self._preview_env = None
        try:
            import pygame
            pygame.quit()
        except Exception:
            pass
        if self.verbose:
            print("[LiveViz] Preview env closed.")