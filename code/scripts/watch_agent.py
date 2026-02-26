#!/usr/bin/env python3
"""
watch_agent.py — Watch a trained AI agent play visually.

CHANGES FROM ORIGINAL:
  - CRITICAL FIX: Loads the companion _vecnorm.pkl and wraps the environment
    in VecNormalize so observations are normalised exactly as during training.
    Without this fix the model receives raw (unnormalised) observations and
    produces essentially random actions even if it was perfectly trained.
  - Added RecurrentPPO (rppo / sb3_contrib) to the algorithm registry.
  - Improved persona/world forwarding to the environment.
  - Graceful fallback when no vecnorm sidecar is found (warns, still runs).
"""

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pygame
from stable_baselines3 import PPO, A2C, DQN

try:
    from sb3_contrib import RecurrentPPO
    _HAS_RPPO = True
except ImportError:
    _HAS_RPPO = False

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from code.wrappers.generic_env import GameEnv

# Remove headless SDL so pygame can create a visible window
if "SDL_VIDEODRIVER" in os.environ:
    del os.environ["SDL_VIDEODRIVER"]


# ---------------------------------------------------------------------------
# Model-path parsing
# ---------------------------------------------------------------------------

def parse_model_info(model_path: str):
    """
    Parse (game, algo, persona, skill) from the model path / directory name.
    Returns: (game, algo, persona, skill)
    """
    path = Path(model_path)

    if path.parent.name not in {".", "models", "best"}:
        parts = path.parent.name.split("_")
    else:
        stem = path.stem.replace("_model", "").replace("best", "")
        parts = stem.split("_")

    if len(parts) >= 4:
        game  = parts[0]
        algo  = parts[1]
        skill = parts[-1]
        raw_persona = "_".join(parts[2:-1])
        persona = (raw_persona[len(game)+1:]
                   if raw_persona.startswith(f"{game}_") else
                   raw_persona[len(game):]
                   if raw_persona.startswith(game) else
                   raw_persona)
        return game, algo, persona, skill
    elif len(parts) >= 1:
        return parts[0], "ppo", "default", "unknown"
    else:
        raise ValueError(f"Cannot parse model info from path: {model_path}")


# ---------------------------------------------------------------------------
# VecNormalize sidecar discovery
# ---------------------------------------------------------------------------

def find_vecnorm_path(model_path: str) -> Optional[Path]:
    """
    Look for a _vecnorm.pkl file next to the .zip.
    Training saves it as: <same_stem>_vecnorm.pkl
    """
    p = Path(model_path)
    # Direct sibling: replace .zip with _vecnorm.pkl
    candidate = p.parent / (p.stem + "_vecnorm.pkl")
    if candidate.exists():
        return candidate

    # Also try inside models/ root (one level up from checkpoints/)
    for parent in [p.parent, p.parent.parent]:
        for pkl in parent.glob(f"*vecnorm*.pkl"):
            return pkl   # return first match — print warning if ambiguous

    return None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path: str, algo: str = "ppo"):
    """Load a SB3/sb3_contrib model by algorithm name."""
    algo_classes = {
        "ppo":  PPO,
        "a2c":  A2C,
        "dqn":  DQN,
    }
    if _HAS_RPPO:
        algo_classes["rppo"] = RecurrentPPO
        algo_classes["recurrent_ppo"] = RecurrentPPO

    ModelClass = algo_classes.get(algo.lower(), PPO)
    try:
        return ModelClass.load(model_path)
    except Exception as e:
        print(f"Failed to load {algo.upper()} model: {e}")
        print("Trying PPO as fallback…")
        return PPO.load(model_path)


# ---------------------------------------------------------------------------
# Environment creation
# ---------------------------------------------------------------------------

def create_raw_env(game: str, persona: str = "default", fps: int = 30, **kwargs):
    """Create a single (non-vectorised) GameEnv in human render mode."""
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    game_module = importlib.import_module(f"code.games.{game}_core")
    GameCoreClass = getattr(game_module, f"{game.capitalize()}Core")

    return GameEnv(
        GameCoreClass,
        render_mode="human",
        fps=fps,
        persona=persona,
        **kwargs,
    )


def build_watch_env(
    game: str,
    persona: str,
    fps: int,
    vecnorm_path: Optional[Path],
    **kwargs,
) -> tuple:
    """
    Build the observation-normalised environment for watching.

    Returns (env, is_vecnorm_wrapped):
      env                — the environment to use for model.predict()
      is_vecnorm_wrapped — True if wrapped in VecNormalize
    """
    raw_env = create_raw_env(game, persona=persona, fps=fps, **kwargs)

    if vecnorm_path is None:
        print("\n[WARNING] No _vecnorm.pkl sidecar found next to the model.")
        print("          Observations will NOT be normalised.")
        print("          The agent may behave incorrectly (distribution mismatch).")
        print("          To fix: ensure the _vecnorm.pkl produced by train.py")
        print("          is in the same folder as the .zip file.\n")
        return raw_env, False

    print(f"[INFO] Loading VecNormalize stats from: {vecnorm_path}")
    vec_env = DummyVecEnv([lambda: raw_env])
    # Load the saved stats and freeze them (training=False)
    norm_env = VecNormalize.load(str(vecnorm_path), vec_env)
    norm_env.training  = False   # do NOT update running stats during eval
    norm_env.norm_reward = False  # reward normalisation not needed for watch
    print("[INFO] VecNormalize loaded — observations will be normalised correctly.\n")
    return norm_env, True


# ---------------------------------------------------------------------------
# Core game instance accessor (for debug overlays)
# ---------------------------------------------------------------------------

def find_core_game(env_instance):
    curr = env_instance
    while hasattr(curr, "env"):
        if hasattr(curr, "game"):
            return curr.game
        curr = curr.env
    return getattr(curr, "game", None)


# ---------------------------------------------------------------------------
# Main watch loop
# ---------------------------------------------------------------------------

def watch_agent_play(
    model_path: str,
    episodes: int = 5,
    fps: int = 30,
    deterministic: bool = True,
    game: Optional[str] = None,
    algo: Optional[str] = None,
):
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    # --- Parse model info ---
    persona = "default"
    skill   = "unknown"
    try:
        parsed_game, parsed_algo, parsed_persona, parsed_skill = parse_model_info(model_path)
        if game is None: game  = parsed_game
        if algo is None: algo  = parsed_algo
        persona = parsed_persona
        skill   = parsed_skill
    except Exception as e:
        print(f"Warning: Could not parse model info from path: {e}")

    print(f"Detected: {game} | {algo.upper()} | {persona} | {skill}")

    # --- Load model ---
    print(f"Loading model from: {model_path}")
    model = load_model(model_path, algo)

    # --- CRITICAL FIX: find and load VecNormalize sidecar ---
    vecnorm_path = find_vecnorm_path(model_path)

    # --- Init pygame ---
    pygame.init()
    pygame.display.set_mode((800, 600))
    pygame.display.set_caption(f"AI Agent Viewer — {game} ({persona})")

    # --- Build environment (with or without VecNormalize) ---
    env, is_vec = build_watch_env(
        game=game, persona=persona, fps=fps,
        vecnorm_path=vecnorm_path,
    )

    # Fetch core game for debug overlay access
    core_game = find_core_game(env)

    print(f"Watching {episodes} episode(s) at {fps} FPS…")
    print("Press ESC or close the window to stop early.\n")

    total_score       = 0
    completed_episodes = 0

    # For RecurrentPPO: maintain LSTM state across steps
    is_recurrent = _HAS_RPPO and isinstance(model, RecurrentPPO)
    lstm_states  = None

    try:
        for episode in range(episodes):
            print(f"--- Episode {episode + 1}/{episodes} ---")

            if is_vec:
                obs = env.reset()
            else:
                obs, _info = env.reset()
                core_game = find_core_game(env)

            done = truncated = False
            episode_score = 0
            step_count    = 0
            lstm_states   = None          # reset LSTM state each episode
            ep_start      = np.ones((1,), dtype=bool)

            while not (done or truncated):
                # Event handling (exit, ESC)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt("Window closed")
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        raise KeyboardInterrupt("ESC pressed")

                pygame.event.pump()

                # Debug overlay updates (F1, F5 key handling)
                if core_game and hasattr(core_game, "debug_manager"):
                    core_game.debug_manager.update_input()

                # --- ACTION (normalised obs → policy → action) ---
                if is_recurrent:
                    action, lstm_states = model.predict(
                        obs, state=lstm_states,
                        episode_start=ep_start, deterministic=deterministic,
                    )
                    ep_start = np.zeros((1,), dtype=bool)
                else:
                    action, _states = model.predict(obs, deterministic=deterministic)

                # Free-cam override
                if core_game and hasattr(core_game, "debug_manager"):
                    if core_game.debug_manager.free_cam_active:
                        action = 0

                # --- STEP ---
                if is_vec:
                    obs, reward, dones, info = env.step(action)
                    done = bool(dones[0])
                    info_dict = info[0] if isinstance(info, (list, tuple)) else info
                else:
                    obs, reward, done, truncated, info_dict = env.step(action)

                # --- RENDER ---
                if is_vec:
                    # VecEnv doesn't expose render() directly; call inner env
                    try:
                        env.env_method("render", indices=[0])
                    except Exception:
                        pass
                else:
                    env.render()

                pygame.time.wait(max(1, int(1000 / fps)))

                step_count   += 1
                episode_score = info_dict.get("score", episode_score)

            completed_episodes += 1
            total_score        += episode_score
            print(f"Episode {episode + 1} done — score={episode_score}  steps={step_count}")

    except KeyboardInterrupt as e:
        print(f"\nStopped early: {e}")

    finally:
        try:
            env.close()
        except Exception:
            pass
        try:
            pygame.quit()
        except Exception:
            pass

        if completed_episodes > 0:
            print(f"\nFinal stats:")
            print(f"  Episodes: {completed_episodes}")
            print(f"  Avg score: {total_score / completed_episodes:.2f}")
            print(f"  Total score: {total_score}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Watch a trained PEAK AI agent play.")
    parser.add_argument("model_path", help="Path to the .zip model file")
    parser.add_argument("--episodes",   type=int,  default=5)
    parser.add_argument("--fps",        type=int,  default=30)
    parser.add_argument("--game",       type=str,  default=None, help="Override game name")
    parser.add_argument("--algo",       type=str,  default=None, help="Override algorithm (ppo/a2c/rppo)")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic policy")
    parser.add_argument("--vecnorm",    type=str,  default=None,
                        help="Explicit path to _vecnorm.pkl (auto-detected if omitted)")
    args = parser.parse_args()

    if not Path(args.model_path).exists():
        print(f"Error: model file not found: {args.model_path}")
        sys.exit(1)

    # Allow explicit vecnorm path override
    if args.vecnorm:
        _orig_find = find_vecnorm_path
        globals()["find_vecnorm_path"] = lambda _: Path(args.vecnorm)

    try:
        watch_agent_play(
            model_path=args.model_path,
            episodes=args.episodes,
            fps=args.fps,
            deterministic=not args.stochastic,
            game=args.game,
            algo=args.algo,
        )
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()