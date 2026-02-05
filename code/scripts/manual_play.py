# code/scripts/manual_play.py
import argparse
import importlib
import os
from pathlib import Path

import pygame

from code.wrappers.generic_env import GameEnv

os.environ["SDL_VIDEO_WINDOW_POS"] = "100,100"

parser = argparse.ArgumentParser()
parser.add_argument("--game", default="platformer", help="game key (e.g., platformer, mario)")
parser.add_argument("--fps", type=int, default=30, help="target FPS for manual play")
args = parser.parse_args()

# Normalize game name (mario -> platformer) if needed
if args.game == "mario":
    args.game = "platformer"

# Enable manual play mode for mario/platformer
if args.game == "platformer":
    os.environ["MARIO_MANUAL_PLAY"] = "1"

# --- Load game core class dynamically
game_mod = importlib.import_module(f"code.games.{args.game}_core")
GameCls = getattr(game_mod, next(attr for attr in dir(game_mod) if attr.endswith("Core")))

# try:
#     game_mod = importlib.import_module(f"code.games.{args.game}_core")
#     GameCls = getattr(game_mod, next(attr for attr in dir(game_mod) if attr.endswith("Core")))
# except ImportError:
#     print(f"Error: Could not load game module 'code.games.{args.game}_core'.")
#     exit(1)

# --- Init pygame BEFORE key handling
pygame.init()
clock = pygame.time.Clock()

def _platformer_action(keys: pygame.key.ScancodeWrapper) -> int:
        k = pygame.key.get_pressed()
        
        # STRICT CONTROL SCHEME:
        # Player Movement: WASD + Space
        # Debug Camera: Arrow Keys (handled internally by Core/DebugManager)
        
        # ARROW KEYS REMOVED from movement logic
        left  = k[pygame.K_a]
        right = k[pygame.K_d]
        jump  = k[pygame.K_SPACE] or k[pygame.K_w]
        run   = k[pygame.K_LSHIFT] or k[pygame.K_RSHIFT] or k[pygame.K_j]

        # IMPORTANT: run is a modifier; by itself it does nothing
        if right and run and jump: 
            return 7  # Run+Right+Jump
        if right and run:          
            return 5  # Run+Right
        if right and jump:         
            return 4  # Right+Jump
        if left  and jump:         
            return 6  # Left+Jump
        if jump:                   
            return 3  # Jump only
        if right:                  
            return 2  # Right
        if left:                   
            return 1 # left
        if left and run and jump:
            return 7
        if left and run:          
            return 5  # Run+left
        if left and jump:         
            return 4  # left+Jump
        return 0                   # Noop

ACTION_MAPPING = {
    "platformer": _platformer_action,
    "mario": _platformer_action
}

CONTROL_DESCRIPTIONS = {
    "platformer": "\n[PLAYER] WASD to Move, SPACE to Jump, SHIFT to Run\n[DEBUG]  ARROWS to Pan Camera (F5 to toggle Free Cam), ESC to Quit",
}

controls = CONTROL_DESCRIPTIONS.get(args.game, "Use game-specific keys. ESC = quit")
print(f"\n=== MANUAL PLAY: {args.game} ===")
print(f"{controls}")
print("=================================")

# --- Build env
env = GameEnv(GameCls, render_mode="human", fps=args.fps)

# Helper to find the actual game core instance through wrappers
def find_core_game(env_instance):
    curr = env_instance
    while hasattr(curr, 'env'):
        if hasattr(curr, 'game'):
            return curr.game
        curr = curr.env
    if hasattr(curr, 'game'):
        return curr.game
    return None

core_game = find_core_game(env)

obs, _ = env.reset()
running = True

while running:
    clock.tick(args.fps)
    action = 0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False

    # Get keys for this frame
    keys = pygame.key.get_pressed()
    
    # Calculate Action
    action_fn = ACTION_MAPPING.get(args.game, lambda k: 0)
    action = action_fn(keys)

    # --- CHECK FOR DEBUG OVERRIDES (Free Cam) ---
    if core_game and hasattr(core_game, 'debug_manager'):
        # If the user has enabled Free Cam (F5), force the player action to IDLE (0).
        if core_game.debug_manager.free_cam_active:
            action = 0 

    # Step the environment
    step_result = env.step(action)

    if len(step_result) == 5:
        obs, reward, terminated, truncated, info = step_result
        done = terminated or truncated
    else:
        obs, reward, done, info = step_result
    
    env.render()
    
    if (info.get("episode_end", False)) or (done if 'done' in locals() else False):
        obs, _ = env.reset()
        # Re-fetch core game in case reset created a new instance
        core_game = find_core_game(env)

env.close()
pygame.quit()
print("Game session ended.\n")