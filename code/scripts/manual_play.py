# code/scripts/manual_play.py
import argparse
import importlib
import os
from pathlib import Path

import pygame

from code.wrappers.generic_env import GameEnv

os.environ["SDL_VIDEO_WINDOW_POS"] = "100,100"

parser = argparse.ArgumentParser()
parser.add_argument("--game", default="flappy", help="game key (e.g., flappy, tetris, asteroids, mario)")
parser.add_argument("--fps", type=int, default=30, help="target FPS for manual play")
args = parser.parse_args()

# ✅ Enable manual play mode for mario
if args.game == "mario":
    os.environ["MARIO_MANUAL_PLAY"] = "1"

# --- Load game core class dynamically
game_mod = importlib.import_module(f"code.games.{args.game}_core")
GameCls = getattr(game_mod, next(attr for attr in dir(game_mod) if attr.endswith("Core")))

# --- Init pygame BEFORE key handling
pygame.init()
clock = pygame.time.Clock()

# --- Global action state for event-based games
# --- Global action state for event-based games
current_action = 0
_last_snake_direction = 1  # Start with RIGHT

def _mario_action(keys: pygame.key.ScancodeWrapper) -> int:
        k = pygame.key.get_pressed()
        left  = k[pygame.K_LEFT]  or k[pygame.K_a]
        right = k[pygame.K_RIGHT] or k[pygame.K_d]
        jump  = k[pygame.K_SPACE] or k[pygame.K_k]
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

def _snake_action(keys: pygame.key.ScancodeWrapper) -> int:
    """
    Discrete action mapping for SnakeCore:
      0: UP, 1: RIGHT, 2: DOWN, 3: LEFT
    Remembers last direction pressed
    """
    global _last_snake_direction
    
    if keys[pygame.K_UP] or keys[pygame.K_w]:
        _last_snake_direction = 0
    elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        _last_snake_direction = 1
    elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
        _last_snake_direction = 2
    elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
        _last_snake_direction = 3
    
    return _last_snake_direction

# --- Helpful control text per game
def _handle_tetris_events(event) -> int:
    """Event-driven tetris controls"""
    if event.type != pygame.KEYDOWN:
        return -1
    
    if event.key == pygame.K_SPACE:
        return 5
    elif event.key in [pygame.K_UP, pygame.K_x, pygame.K_w]:
        return 3
    elif event.key == pygame.K_LEFT:
        return 1
    elif event.key == pygame.K_RIGHT:
        return 2
    elif event.key in [pygame.K_DOWN, pygame.K_s]:
        return 4
    return -1

def _asteroids_action(keys: pygame.key.ScancodeWrapper) -> int:
    """Asteroids action mapping"""
    if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
        return 5
    if keys[pygame.K_SPACE]:
        return 4
    if keys[pygame.K_UP] or keys[pygame.K_w]:
        return 3
    if keys[pygame.K_LEFT] or keys[pygame.K_a]:
        return 1
    if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        return 2
    return 0

ACTION_MAPPING = {
    "flappy": lambda keys: 1 if keys[pygame.K_SPACE] else 0,
    "asteroids": _asteroids_action,
    "tetris": _handle_tetris_events, 
    "mario": _mario_action,
    "snake": _snake_action,  # NEW
}


CONTROL_DESCRIPTIONS = {
    "flappy": "SPACE = flap, ESC = quit",
    "tetris": "\n←/→ move, ↑/X/W rotate, ↓/S soft drop, SPACE hard drop, ESC quit",
    "asteroids": "\n←/→/A/D turn, ↑/W thrust, SPACE shoot, SHIFT hyperspace, ESC quit",
    "mario": "\n←/→ move, ↑/SPACE jump, ESC quit",
    "snake": "\n↑/W up, ↓/S down, ←/A left, →/D right, ESC quit",  # NEW
}


controls = CONTROL_DESCRIPTIONS.get(args.game, "Use game-specific keys. ESC = quit")
print(f"\nUse controls: {controls}")

# --- Build env
env = GameEnv(GameCls, render_mode="human", fps=args.fps)

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
        
        if args.game == "tetris":
            tetris_action = _handle_tetris_events(event)
            if tetris_action != -1:
                action = tetris_action
        
        

    if args.game != "tetris":
        keys = pygame.key.get_pressed()
        action_fn = ACTION_MAPPING.get(args.game, lambda k: 0)
        action = action_fn(keys)

        if args.game == "snake":
            action = _snake_action(keys)  # Uses global state
        else:
            action = action_fn(keys)
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

env.close()
pygame.quit()
print("Game session ended.\n")
