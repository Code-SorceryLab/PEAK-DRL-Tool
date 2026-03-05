"""
LiveVisualizationCallback
=========================
Shows the agent playing in a live pygame window at a set interval during
training — WITHOUT spawning a subprocess (which causes flash-and-close on
Windows and path/import issues on all platforms).

How it works
------------
Every `viz_freq` training steps:
  1. The current model is saved to a temporary `_live_preview.zip` checkpoint.
  2. A daemon Thread is spawned IN THE SAME PROCESS that calls
     watch_agent_play() directly.  No subprocess, no new console, no SDL
     driver conflicts.
  3. If a previous preview thread is still alive it is signalled to stop first.

Training continues at full speed — the thread is non-blocking.

Why threads instead of subprocess?
-----------------------------------
On Windows, subprocess.Popen of a pygame script either:
  - Opens a console that immediately closes (stdout/stderr redirected)
  - Has SDL_VIDEODRIVER conflicts in the child environment
  - Fails to resolve the correct module import paths
A thread in the same process avoids all of these: imports are already resolved,
pygame can be initialised inside the thread (it does not need to run in the
main thread as long as training never calls pygame.init() itself), and no new
OS process is needed.
"""

import sys
import threading
import importlib.util
from pathlib import Path
from stable_baselines3.common.callbacks import BaseCallback


class LiveVisualizationCallback(BaseCallback):
    """
    Opens a live pygame window showing the agent play during training.
    Uses a background thread instead of a subprocess.

    Parameters
    ----------
    watch_agent_script : str | Path
        Absolute path to watch_agent.py.
    vecnorm_path : str | Path | None
        Path to the _vecnorm.pkl companion file produced by train.py.
    game : str
        Game name (e.g. "platformer").
    algo : str
        Algorithm name (e.g. "ppo").
    persona : str
        Persona string (e.g. "dijkstra").
    viz_freq : int
        Trigger every N training timesteps.
    n_preview_episodes : int
        Number of episodes per preview window.
    fps : int
        Playback FPS inside the preview window.
    preview_save_dir : str | Path
        Where the temporary _live_preview.zip is written.
    verbose : int
        SB3 verbosity level.
    """

    def __init__(
        self,
        watch_agent_script: str,
        vecnorm_path,
        game: str,
        algo: str,
        persona: str,
        viz_freq: int       = 50_000,
        n_preview_episodes: int = 1,
        fps: int            = 30,
        preview_save_dir: str = "models",
        verbose: int        = 1,
    ):
        super().__init__(verbose)
        self.watch_agent_script = Path(watch_agent_script)
        self.vecnorm_path       = str(vecnorm_path) if vecnorm_path else None
        self.game               = game
        self.algo               = algo
        self.persona            = persona
        self.viz_freq           = viz_freq
        self.n_preview_episodes = n_preview_episodes
        self.fps                = fps
        self.preview_save_dir   = Path(preview_save_dir)
        self.preview_save_dir.mkdir(parents=True, exist_ok=True)

        self._last_viz_step: int             = 0
        self._stop_event: threading.Event    = threading.Event()
        self._preview_thread: threading.Thread | None = None
        self._preview_zip = self.preview_save_dir / "_live_preview.zip"

        # Resolve watch_agent_play once at init time
        self._watch_fn = self._resolve_watch_fn()

    # ------------------------------------------------------------------
    # Resolve the callable
    # ------------------------------------------------------------------
    def _resolve_watch_fn(self):
        """
        Try to import watch_agent_play from the watch_agent module.
        Returns the function or None if it cannot be found.
        """
        # 1. Package import (works when running from the repo root)
        try:
            from code.scripts.watch_agent import watch_agent_play
            return watch_agent_play
        except ImportError:
            pass

        # 2. Direct file import using the provided script path
        try:
            spec = importlib.util.spec_from_file_location(
                "watch_agent", str(self.watch_agent_script)
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "watch_agent_play"):
                return mod.watch_agent_play
        except Exception as e:
            print(f"[LiveViz] WARNING: could not load watch_agent from "
                  f"{self.watch_agent_script}: {e}")

        return None

    # ------------------------------------------------------------------
    # SB3 callbacks
    # ------------------------------------------------------------------
    def _on_training_start(self) -> None:
        if self._watch_fn is None:
            print("[LiveViz] WARNING: watch_agent_play not found — "
                  "live preview disabled.")
        elif self.verbose:
            print(
                f"[LiveViz] Thread-mode ON — preview every {self.viz_freq:,} "
                f"steps | {self.n_preview_episodes} ep(s) at {self.fps} FPS"
            )

    def _on_step(self) -> bool:
        if self._watch_fn is None:
            return True
        if self.num_timesteps - self._last_viz_step < self.viz_freq:
            return True

        self._last_viz_step = self.num_timesteps
        self._stop_previous()
        self._save_checkpoint()
        self._launch_thread()
        return True

    def _on_training_end(self) -> None:
        self._stop_previous()
        try:
            if self._preview_zip.exists():
                self._preview_zip.unlink()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _stop_previous(self):
        """Signal the previous thread to quit and wait briefly."""
        if self._preview_thread and self._preview_thread.is_alive():
            if self.verbose:
                print("[LiveViz] Stopping previous preview …")
            self._stop_event.set()
            self._preview_thread.join(timeout=5)
        self._stop_event.clear()
        self._preview_thread = None

    def _save_checkpoint(self):
        try:
            self.model.save(str(self._preview_zip))
            if self.verbose:
                print(f"[LiveViz] Checkpoint saved → {self._preview_zip} "
                      f"(step {self.num_timesteps:,})")
        except Exception as e:
            print(f"[LiveViz] Could not save checkpoint: {e}")

    def _launch_thread(self):
        # Capture everything we need — avoid closing over `self` in the thread
        watch_fn     = self._watch_fn
        model_path   = str(self._preview_zip)
        vecnorm      = self.vecnorm_path
        game         = self.game
        algo         = self.algo
        persona      = self.persona
        episodes     = self.n_preview_episodes
        fps          = self.fps
        stop_event   = self._stop_event
        verbose      = self.verbose

        def _target():
            try:
                if verbose:
                    print("[LiveViz] Preview thread started.")

                import inspect
                sig = inspect.signature(watch_fn)

                kwargs: dict = dict(
                    model_path = model_path,
                    episodes   = episodes,
                    fps        = fps,
                    game       = game,
                    algo       = algo,
                    persona    = persona,
                    vecnorm    = vecnorm,
                )
                # Pass stop_event only if the function accepts it
                if "stop_event" in sig.parameters:
                    kwargs["stop_event"] = stop_event

                watch_fn(**kwargs)

                if verbose:
                    print("[LiveViz] Preview thread finished.")

            except KeyboardInterrupt:
                pass   # normal ESC / window-close path
            except Exception as e:
                print(f"[LiveViz] Preview thread raised: {type(e).__name__}: {e}")

        self._preview_thread = threading.Thread(
            target  = _target,
            name    = "LiveVizPreview",
            daemon  = True,   # killed automatically when training exits
        )
        self._preview_thread.start()

        if self.verbose:
            print("[LiveViz] Preview window open — training continues.")