#!/usr/bin/env python3
import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import Optional

import pygame
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

try:
    from sb3_contrib import RecurrentPPO
    HAS_RPPO = True
except ImportError:
    HAS_RPPO = False

from code.wrappers.generic_env import GameEnv
import code.rewards.train_platformer as reward_module

# FORCE VISUAL MODE
if "SDL_VIDEODRIVER" in os.environ:
    del os.environ["SDL_VIDEODRIVER"]


def parse_model_info(model_path: str) -> tuple[str, str, str, str]:
    path = Path(model_path)

    if path.parent.name != "." and path.parent.name not in ["models", "best"]:
        folder_name = path.parent.name
        parts = folder_name.split("_")
    else:
        stem = path.stem.replace("_model", "").replace("best", "")
        parts = stem.split("_")

    if len(parts) >= 4:
        game        = parts[0]
        algo        = parts[1]
        skill       = parts[-1]
        raw_persona = "_".join(parts[2:-1])

        if raw_persona.startswith(f"{game}_"):
            persona = raw_persona[len(game)+1:]
        elif raw_persona.startswith(game):
            persona = raw_persona[len(game):]
        else:
            persona = raw_persona

        return game, algo, persona, skill
    elif len(parts) >= 1:
        return parts[0], "ppo", "default", "unknown"
    else:
        raise ValueError(f"Cannot parse model info from path: {model_path}")


def load_model(model_path: str, algo: str = "ppo", env=None):
    algo_classes = {
        "ppo": PPO,
        "a2c": A2C,
        "dqn": DQN,
    }
    if HAS_RPPO:
        algo_classes["rppo"]          = RecurrentPPO
        algo_classes["recurrent_ppo"] = RecurrentPPO

    ModelClass = algo_classes.get(algo.lower(), PPO)

    try:
        return ModelClass.load(model_path, env=env)
    except Exception as e:
        print(f"Failed to load {algo.upper()} model: {e}")
        print("Trying PPO as fallback...")
        return PPO.load(model_path, env=env)


def find_core_game(env_instance):
    curr = env_instance
    while hasattr(curr, 'env'):
        if hasattr(curr, 'game'):
            return curr.game
        curr = curr.env
    if hasattr(curr, 'game'):
        return curr.game
    return None


def find_vecnorm(model_path: str) -> Optional[Path]:
    """
    Look for the vecnorm sidecar in a few likely locations:
      1. Next to the model zip:   path/to/best_model_vecnorm.pkl
      2. Parent models/ dir:      models/<stem>_vecnorm.pkl
      3. Parent models/ dir stem: models/<folder_name>_vecnorm.pkl
    """
    p = Path(model_path)

    candidates = [
        p.with_name(p.stem + "_vecnorm.pkl"),
        p.parent.parent / (p.stem + "_vecnorm.pkl"),
        p.parent.parent / (p.parent.name + "_vecnorm.pkl"),
    ]

    for c in candidates:
        if c.exists():
            return c
    return None


def watch_agent_play(
    model_path:    str,
    episodes:      int  = 5,
    fps:           int  = 60,
    deterministic: bool = True,
    game:          Optional[str] = None,
    algo:          Optional[str] = None,
):
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    persona = "default"
    skill   = "unknown"

    try:
        parsed_game, parsed_algo, parsed_persona, parsed_skill = parse_model_info(model_path)
        if game is None: game = parsed_game
        if algo is None: algo = parsed_algo
        persona = parsed_persona
        skill   = parsed_skill
    except Exception as e:
        print(f"Warning: Could not parse model info from path: {e}")

    print(f"Detected: {game} | {algo.upper()} | {persona} | {skill}")

    reward_fn = getattr(reward_module, persona, reward_module.default)

    # --- 1. Init pygame ---
    pygame.init()
    pygame.display.set_caption(f"AI Agent - {game} ({persona})")
    clock = pygame.time.Clock()
    print("Pygame initialized")

    # --- 2. Build env ---
    game_module   = importlib.import_module(f"code.games.{game}_core")
    class_name    = f"{game.capitalize()}Core"
    GameCoreClass = getattr(game_module, class_name)

    def make_env():
        return GameEnv(
            GameCoreClass,
            reward_fn=reward_fn,
            render_mode="human",
            persona=persona,
            fps=fps,
        )

    raw_env = DummyVecEnv([make_env])

    # --- 3. Wrap with VecNormalize if sidecar exists ---
    vecnorm_path = find_vecnorm(model_path)
    if vecnorm_path:
        env = VecNormalize.load(str(vecnorm_path), raw_env)
        env.training    = False
        env.norm_reward = False
        print(f"[INFO] Loaded VecNormalize stats: {vecnorm_path}")
    else:
        env = raw_env
        print(f"[WARNING] No VecNorm sidecar found for {model_path}")
        print(f"          Observations are NOT normalized — agent may behave erratically")

    # --- 4. Load model with env (order matters) ---
    print(f"Loading model from: {model_path}")
    model = load_model(model_path, algo, env=env)
    print(f"Model loaded. Watching {episodes} episode(s) at {fps} FPS — ESC or close to stop")

    total_score        = 0
    completed_episodes = 0

    try:
        for episode in range(episodes):
            print(f"\n--- Episode {episode + 1}/{episodes} ---")

            obs       = env.reset()
            core_game = find_core_game(env.venv if hasattr(env, 'venv') else env)
            done      = False
            episode_score = 0
            step_count    = 0

            while not done:
                clock.tick(fps)

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt("Window closed")
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        raise KeyboardInterrupt("ESC pressed")

                pygame.event.pump()

                if core_game and hasattr(core_game, 'debug_manager'):
                    core_game.debug_manager.update_input()

                action, _states = model.predict(obs, deterministic=deterministic)

                if core_game and hasattr(core_game, 'debug_manager'):
                    if core_game.debug_manager.free_cam_active:
                        action = 0

                obs, reward, done, info = env.step(action)
                env.render()

                # VecEnv returns arrays — grab first element
                if hasattr(done, '__len__'):
                    done = done[0]

                step_count    += 1
                episode_score  = info[0].get("score", episode_score) \
                    if isinstance(info, (list, tuple)) \
                    else info.get("score", episode_score)

            completed_episodes += 1
            total_score        += episode_score
            print(f"Episode {episode + 1}: Score={episode_score}, Steps={step_count}")

    except KeyboardInterrupt as e:
        print(f"\nStopped: {e}")

    finally:
        env.close()
        pygame.quit()

        if completed_episodes > 0:
            print(f"\nFinal Stats:")
            print(f"  Episodes:  {completed_episodes}")
            print(f"  Avg score: {total_score / completed_episodes:.2f}")
            print(f"  Total:     {total_score}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path",   help="Path to .zip model file")
    parser.add_argument("--episodes",   type=int,  default=5)
    parser.add_argument("--fps",        type=int,  default=60)
    parser.add_argument("--game",       type=str,  default=None)
    parser.add_argument("--algo",       type=str,  default=None)
    parser.add_argument("--stochastic", action="store_true")

    args = parser.parse_args()

    if not Path(args.model_path).exists():
        print(f"Error: Model not found: {args.model_path}")
        sys.exit(1)

    watch_agent_play(
        model_path    = args.model_path,
        episodes      = args.episodes,
        fps           = args.fps,
        deterministic = not args.stochastic,
        game          = args.game,
        algo          = args.algo,
    )


if __name__ == "__main__":
    main()