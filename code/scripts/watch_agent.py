#!/usr/bin/env python3
"""
Universal script to watch any trained AI agent play its game visually.
Automatically detects game type, loads appropriate environment, and displays the AI playing.
"""
import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import Optional

import pygame
from stable_baselines3 import PPO, A2C, DQN

from code.wrappers.generic_env import GameEnv

# FORCE VISUAL MODE - Remove headless settings
if "SDL_VIDEODRIVER" in os.environ:
    del os.environ["SDL_VIDEODRIVER"]

def parse_model_info(model_path: str) -> tuple[str, str, str, str]:
    """
    Parse model path to extract game, algorithm, persona, skill.
    Expected formats:
    - models/best/flappy_ppo_flappy_speedrunner_expert/best_model.zip
    - models/flappy_ppo_flappy_speedrunner_expert.zip  
    - models/best_model.zip (manual specification required)
    
    Returns: (game, algo, persona, skill)
    """
    path = Path(model_path)
    
    # Try to parse from parent directory name first (best/ structure)
    if path.parent.name != "." and path.parent.name not in ["models", "best"]:
        folder_name = path.parent.name
        parts = folder_name.split("_")
    else:
        # Parse from filename
        stem = path.stem.replace("_model", "").replace("best", "")
        parts = stem.split("_")
    
    if len(parts) >= 4:
        game = parts[0]
        algo = parts[1] 
        persona = "_".join(parts[2:-1])  # Handle multi-word personas
        skill = parts[-1]
        return game, algo, persona, skill
    elif len(parts) >= 1:
        # Minimal parsing - game only
        game = parts[0]
        return game, "ppo", "unknown", "unknown"
    else:
        raise ValueError(f"Cannot parse model info from path: {model_path}")

def load_model(model_path: str, algo: str = "ppo"):
    """Load model based on algorithm type"""
    algo_classes = {
        "ppo": PPO,
        "a2c": A2C, 
        "dqn": DQN,
    }
    
    ModelClass = algo_classes.get(algo.lower(), PPO)
    
    try:
        return ModelClass.load(model_path)
    except Exception as e:
        print(f"Failed to load {algo.upper()} model: {e}")
        print("Trying with PPO as fallback...")
        return PPO.load(model_path)

def create_env(game: str, **kwargs):
    """Dynamically create environment for the given game with FORCED visual rendering"""
    try:
        # Import the game core class
        game_module = importlib.import_module(f"code.games.{game}_core")
        GameCoreClass = getattr(game_module, f"{game.capitalize()}Core")
        
        # Force visual rendering by removing headless settings
        if "SDL_VIDEODRIVER" in os.environ:
            del os.environ["SDL_VIDEODRIVER"]
        
        # Create environment with FORCED human rendering
        env = GameEnv(
            GameCoreClass,
            render_mode="human",  # FORCE human mode
            **kwargs
        )
        return env
        
    except ImportError as e:
        raise ImportError(f"Could not import game '{game}': {e}")
    except AttributeError as e:
        raise AttributeError(f"Game core class not found for '{game}': {e}")

def watch_agent_play(
    model_path: str, 
    episodes: int = 5,
    fps: int = 30,
    deterministic: bool = True,
    game: Optional[str] = None,
    algo: Optional[str] = None
):
    """Main function to watch an agent play"""
    
    # FORCE remove headless mode
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        print("Removed headless SDL driver - forcing visual mode")
    
    # Parse model information
    if game is None or algo is None:
        parsed_game, parsed_algo, persona, skill = parse_model_info(model_path)
        game = game or parsed_game
        algo = algo or parsed_algo
        print(f"Detected: {game} | {algo.upper()} | {persona} | {skill}")
    
    # Load the trained model
    print(f"Loading model from: {model_path}")
    model = load_model(model_path, algo)
    
    # Initialize pygame FIRST to ensure display works
    pygame.init()
    pygame.display.set_mode((800, 600))  # Create a dummy window to test
    pygame.display.set_caption("AI Agent Viewer - Loading...")
    print("Pygame initialized successfully")
    
    # Create environment
    print(f"Creating {game} environment...")
    env = create_env(game, fps=fps)
    
    print(f"Watching agent play {episodes} episode(s) at {fps} FPS...")
    print("Press ESC or close window to stop early")
    print("Window should appear now...")
    
    total_score = 0
    completed_episodes = 0
    
    try:
        for episode in range(episodes):
            print(f"\n--- Episode {episode + 1}/{episodes} ---")
            
            obs, info = env.reset()
            done = truncated = False
            episode_score = 0
            step_count = 0
            
            while not (done or truncated):
                # Check for early exit
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt("Window closed")
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        raise KeyboardInterrupt("ESC pressed")
                
                # Get AI action
                action, _states = model.predict(obs, deterministic=deterministic)
                
                # Step environment
                obs, reward, done, truncated, info = env.step(action)
                
                # FORCE render every step
                env.render()
                
                # Track stats
                step_count += 1
                episode_score = info.get("score", episode_score)
                
                # Add small delay to make it watchable
                pygame.time.wait(max(1, int(1000 / fps)))
            
            # Episode finished
            completed_episodes += 1
            total_score += episode_score
            print(f"Episode {episode + 1} finished: Score = {episode_score}, Steps = {step_count}")
            
            if done:
                print("Episode ended: Agent died/failed")
            if truncated:
                print("Episode ended: Max steps reached")
    
    except KeyboardInterrupt as e:
        print(f"\nStopped early: {e}")
    
    finally:
        env.close()
        pygame.quit()
        
        # Final stats
        if completed_episodes > 0:
            avg_score = total_score / completed_episodes
            print(f"\n Final Stats:")
            print(f"   Episodes completed: {completed_episodes}")
            print(f"   Average score: {avg_score:.2f}")
            print(f"   Total score: {total_score}")

def main():
    parser = argparse.ArgumentParser(
        description="Watch a trained AI agent play its game",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect everything from path
  python watch_agent.py models/best/flappy_ppo_flappy_speedrunner_expert/best_model.zip
  
  # Manual specification
  python watch_agent.py --game asteroids --algo ppo models/my_model.zip
  
  # Watch 10 episodes at 60 FPS
  python watch_agent.py --episodes 10 --fps 60 models/best/tetris_ppo_tetris_master_expert/best_model.zip
        """)
    
    parser.add_argument("model_path", 
                       help="Path to the trained model (.zip file)")
    parser.add_argument("--episodes", type=int, default=5,
                       help="Number of episodes to watch (default: 5)")
    parser.add_argument("--fps", type=int, default=30,
                       help="Rendering FPS (default: 30)")
    parser.add_argument("--game", type=str, default=None,
                       help="Game name (auto-detected if not provided)")
    parser.add_argument("--algo", type=str, default=None,
                       help="Algorithm (ppo/a2c/dqn, auto-detected if not provided)")
    parser.add_argument("--stochastic", action="store_true",
                       help="Use stochastic actions instead of deterministic")
    
    args = parser.parse_args()
    
    # Validate model path exists
    if not Path(args.model_path).exists():
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)
    
    try:
        watch_agent_play(
            model_path=args.model_path,
            episodes=args.episodes,
            fps=args.fps,
            deterministic=not args.stochastic,
            game=args.game,
            algo=args.algo
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
