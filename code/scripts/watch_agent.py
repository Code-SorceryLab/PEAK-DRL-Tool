#!/usr/bin/env python3
"""
watch_agent.py — Watch a trained AI agent play visually.

Must be run as a module from the repo root:
    python -m code.scripts.watch_agent <model.zip> [options]

Key fixes vs original:
  - Loads _vecnorm.pkl so observations are normalised identically to training.
  - Loads the correct reward persona so the env matches training exactly.
  - --persona CLI arg (required when called from menu.py).
  - RecurrentPPO / LSTM state support.
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
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

try:
    from sb3_contrib import RecurrentPPO
    _HAS_RPPO = True
except ImportError:
    _HAS_RPPO = False

from code.wrappers.generic_env import GameEnv

# Remove headless SDL so pygame can open a real window
os.environ.pop("SDL_VIDEODRIVER", None)


# ---------------------------------------------------------------------------
# Parse (game, algo, persona, skill) from the model folder/filename
# ---------------------------------------------------------------------------
def parse_model_info(model_path: str):
    path = Path(model_path)
    folder = path.parent.name
    if folder not in {".", "models", "best"}:
        parts = folder.split("_")
    else:
        parts = path.stem.replace("_model", "").replace("best", "").split("_")

    if len(parts) >= 4:
        game        = parts[0]
        algo        = parts[1]
        skill       = parts[-1]
        raw_persona = "_".join(parts[2:-1])
        persona = (
            raw_persona[len(game) + 1:] if raw_persona.startswith(f"{game}_") else
            raw_persona[len(game):]     if raw_persona.startswith(game) else
            raw_persona
        )
        return game, algo, persona, skill
    elif parts:
        return parts[0], "ppo", "default", "unknown"
    raise ValueError(f"Cannot parse model info from: {model_path}")


# ---------------------------------------------------------------------------
# Find the _vecnorm.pkl sidecar
# ---------------------------------------------------------------------------
def find_vecnorm_path(model_path: str) -> Optional[Path]:
    p = Path(model_path)
    # Same folder, same stem + _vecnorm.pkl
    candidate = p.parent / (p.stem + "_vecnorm.pkl")
    if candidate.exists():
        return candidate
    # Search one level up
    for parent in [p.parent, p.parent.parent]:
        for pkl in parent.glob("*vecnorm*.pkl"):
            return pkl
    return None


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
def load_model(model_path: str, algo: str = "ppo"):
    classes = {"ppo": PPO, "a2c": A2C, "dqn": DQN}
    if _HAS_RPPO:
        classes["rppo"] = RecurrentPPO
        classes["recurrent_ppo"] = RecurrentPPO
    cls = classes.get(algo.lower(), PPO)
    try:
        return cls.load(model_path)
    except Exception as e:
        print(f"[WARN] Failed to load as {algo.upper()}: {e} — trying PPO")
        return PPO.load(model_path)


# ---------------------------------------------------------------------------
# Build env (matches training: same game class, reward fn, VecNormalize)
# ---------------------------------------------------------------------------
def build_env(game: str, persona: str, fps: int, vecnorm_path: Optional[Path], random_start_world: bool = False):
    game_module   = importlib.import_module(f"code.games.{game}_core")
    GameCoreClass = getattr(game_module, f"{game.capitalize()}Core")

    # Load reward fn to match training exactly
    reward_fn = None
    try:
        reward_module = importlib.import_module(f"code.rewards.{game}_rewards")
        if hasattr(reward_module, persona):
            reward_fn = getattr(reward_module, persona)
            print(f"[INFO] Loaded reward persona: {persona}")
        else:
            print(f"[WARN] Persona '{persona}' not found in rewards — using default")
    except Exception as e:
        print(f"[WARN] Could not load reward module: {e}")

    raw_env = GameEnv(
        GameCoreClass,
        render_mode="human",
        fps=fps,
        persona=persona,
        random_start_world=random_start_world,
        **({"reward_fn": reward_fn} if reward_fn else {}),
    )

    if vecnorm_path is None:
        print("\n[WARN] No _vecnorm.pkl found — obs will NOT be normalised.")
        print("       Place the _vecnorm.pkl next to the .zip for correct behaviour.\n")
        return raw_env, False

    print(f"[INFO] Loading VecNormalize from: {vecnorm_path}")
    vec_env  = DummyVecEnv([lambda: raw_env])
    norm_env = VecNormalize.load(str(vecnorm_path), vec_env)
    norm_env.training    = False
    norm_env.norm_reward = False
    print("[INFO] VecNormalize loaded — obs normalised correctly.\n")
    return norm_env, True


# ---------------------------------------------------------------------------
# Main play loop
# ---------------------------------------------------------------------------
def watch_agent_play(
    model_path:         str,
    episodes:           int  = 5,
    fps:                int  = 30,
    deterministic:      bool = True,
    game:    Optional[str] = None,
    algo:    Optional[str] = None,
    persona: Optional[str] = None,
    vecnorm: Optional[str] = None,
    random_start_world: bool = False,
):
    os.environ.pop("SDL_VIDEODRIVER", None)

    # Parse metadata from path, but CLI args always win
    try:
        p_game, p_algo, p_persona, skill = parse_model_info(model_path)
    except Exception as e:
        print(f"[WARN] Could not parse model info: {e}")
        p_game, p_algo, p_persona, skill = "unknown", "ppo", "default", "unknown"

    if game    is None: game    = p_game
    if algo    is None: algo    = p_algo
    if persona is None: persona = p_persona

    print(f"[INFO] game={game}  algo={algo.upper()}  persona={persona}  skill={skill}")

    model = load_model(model_path, algo)

    vn_path = Path(vecnorm) if vecnorm else find_vecnorm_path(model_path)

    pygame.init()
    pygame.display.set_caption(f"PEAK — {game} / {persona}")

    env, is_vec = build_env(game, persona, fps, vn_path, random_start_world=random_start_world)

    is_recurrent = _HAS_RPPO and isinstance(model, RecurrentPPO)
    print(f"[INFO] Watching {episodes} episode(s) at {fps} FPS — ESC or close to stop.\n")

    total_score, completed = 0, 0

    try:
        for ep in range(episodes):
            print(f"--- Episode {ep + 1}/{episodes} ---")

            obs       = env.reset() if is_vec else env.reset()[0]
            done      = truncated = False
            lstm_states = None
            ep_start  = np.ones((1,), dtype=bool)
            score, steps = 0, 0

            while not (done or truncated):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt("window closed")
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        raise KeyboardInterrupt("ESC")
                pygame.event.pump()

                # Debug overlay input (F1 / free-cam)
                core = getattr(getattr(env, "env", env), "game", None)
                if core and hasattr(core, "debug_manager"):
                    core.debug_manager.update_input()

                if is_recurrent:
                    action, lstm_states = model.predict(
                        obs, state=lstm_states,
                        episode_start=ep_start, deterministic=deterministic,
                    )
                    ep_start = np.zeros((1,), dtype=bool)
                else:
                    action, _ = model.predict(obs, deterministic=deterministic)

                if is_vec:
                    obs, reward, dones, info = env.step(action)
                    done = bool(dones[0])
                    info_dict = info[0] if isinstance(info, (list, tuple)) else info
                    try:
                        env.env_method("render", indices=[0])
                    except Exception:
                        pass
                else:
                    obs, reward, done, truncated, info_dict = env.step(action)
                    env.render()

                pygame.time.wait(max(1, 1000 // fps))
                score = info_dict.get("score", score)
                steps += 1

            completed += 1
            total_score += score
            print(f"Episode {ep + 1} done — score={score}  steps={steps}")

    except KeyboardInterrupt as e:
        print(f"\nStopped: {e}")
    finally:
        try: env.close()
        except Exception: pass
        try: pygame.quit()
        except Exception: pass

    if completed:
        print(f"\nDone — {completed} ep(s)  avg score={total_score / completed:.2f}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Watch a trained PEAK agent play.")
    ap.add_argument("model_path",             help="Path to the .zip model file")
    ap.add_argument("--episodes",  type=int,  default=5)
    ap.add_argument("--fps",       type=int,  default=30)
    ap.add_argument("--game",      type=str,  default=None)
    ap.add_argument("--algo",      type=str,  default=None)
    ap.add_argument("--persona",   type=str,  default=None)
    ap.add_argument("--vecnorm",   type=str,  default=None,
                    help="Explicit path to _vecnorm.pkl (auto-detected if omitted)")
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--random-level", action="store_true",
                    help="Start each episode on a random level (mirrors curriculum training)")
    args = ap.parse_args()

    if not Path(args.model_path).exists():
        print(f"Error: model not found: {args.model_path}")
        sys.exit(1)

    try:
        watch_agent_play(
            model_path         = args.model_path,
            episodes           = args.episodes,
            fps                = args.fps,
            deterministic      = not args.stochastic,
            game               = args.game,
            algo               = args.algo,
            persona            = args.persona,
            vecnorm            = args.vecnorm,
            random_start_world = args.random_level,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()