# code/scripts/manual_play.py
import argparse
import importlib
import os
from pathlib import Path

import pygame

from code.wrappers.generic_env import GameEnv

os.environ["SDL_VIDEO_WINDOW_POS"] = "100,100"

parser = argparse.ArgumentParser()
parser.add_argument("--game", default="platformer", help="game key")
parser.add_argument("--fps", type=int, default=30, help="target FPS")
parser.add_argument("--level", default=None, help="Level ID from game_config (e.g. 1-3)")
parser.add_argument("--file", default=None, help="Absolute path to a .txt level file (unlisted)")
args = parser.parse_args()

# Level ID priority: CLI arg > env var > default
level_id  = args.level or os.environ.get('PEAK_PLAY_LEVEL', None)
level_file = args.file  or os.environ.get('PEAK_PLAY_FILE',  None)

# Normalize game name (mario -> platformer) if needed
if args.game == "mario":
    args.game = "platformer"

# Enable manual play mode for mario/platformer
if args.game == "platformer":
    os.environ["MARIO_MANUAL_PLAY"] = "1"

# --- Load game core class dynamically
game_mod = importlib.import_module(f"code.games.{args.game}_core")
GameCls = getattr(game_mod, next(attr for attr in dir(game_mod) if attr.endswith("Core")))

# --- Init pygame BEFORE key handling
pygame.init()
clock = pygame.time.Clock()

def _platformer_action(keys) -> list:
    """
    Return a MultiDiscrete action [move, jump, fire].
      move : 0=idle  1=left  2=sprint_left  3=right  4=sprint_right
      jump : 0=idle  1=jump
      fire : 0=idle  1=fire   (Z key)
    """
    k = pygame.key.get_pressed()

    left  = k[pygame.K_a]
    right = k[pygame.K_d]
    jump  = k[pygame.K_SPACE] or k[pygame.K_w]
    run   = k[pygame.K_LSHIFT] or k[pygame.K_RSHIFT] or k[pygame.K_j]
    fire  = k[pygame.K_z]

    # Move axis
    if left and right:
        move = 0                        # cancel out
    elif left:
        move = 2 if run else 1          # sprint_left / left
    elif right:
        move = 4 if run else 3          # sprint_right / right
    else:
        move = 0

    return [move, int(bool(jump)), int(bool(fire))]

ACTION_MAPPING = {
    "platformer": _platformer_action,
    "mario": _platformer_action
}

CONTROL_DESCRIPTIONS = {
    "platformer": "\n[PLAYER] WASD to Move, SPACE to Jump, SHIFT to Run\n[DEBUG]  ARROWS to Pan Camera (F5 to toggle Free Cam), ESC to Quit",
}

controls = CONTROL_DESCRIPTIONS.get(args.game, "Use game-specific keys. ESC = quit")


# --- Validate level_file early
TEMP_ID = '__editor_test__'
if level_file:
    _fp = Path(level_file).resolve()
    if not _fp.exists():
        print(f"[Play] ERROR: level file not found: {level_file}")
        level_file = None
    else:
        level_file = str(_fp)  # normalise to absolute string

# --- Build env (let it load the default first level on __init__)
env_kwargs = {}
if level_id:
    env_kwargs['world'] = level_id
    env_kwargs['lock_level'] = True
    print(f"[Play] Loading level: {level_id}")
# NOTE: do NOT pass world='__editor_test__' here — it doesn't exist in config
# yet so platformer_core.__init__ -> reset() would load an empty level.
env = GameEnv(GameCls, render_mode="human", persona="simple", fps=args.fps, **env_kwargs)

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

# ── Handle level_id override ──────────────────────────────────────
if level_id and core_game and hasattr(core_game, 'world'):
    if core_game.world.lower() != level_id.lower():
        core_game.world = level_id.lower()
        core_game.locked_level = level_id.lower()
        if hasattr(core_game, 'load_level'):
            core_game.load_level()
        print(f"[Play] Forced level override: {level_id}")

# ── Handle raw file path (unlisted level) ────────────────────────
# Strategy: inject the entry directly into the live config_manager instance
# (post-construction), then call load_level() to switch to the editor file.
# LevelLoader supports absolute paths in the 'file' field, so no file copying.
if level_file and core_game:
    fp = Path(level_file)
    cm = core_game.config_manager
    if 'levels' not in cm.yaml_data:
        cm.yaml_data['levels'] = {}
    cm.yaml_data['levels'][TEMP_ID] = {
        'file': str(fp),   # absolute path — LevelLoader handles this directly
        'time_limit': 300,
    }
    if hasattr(core_game, 'level_order') and TEMP_ID not in core_game.level_order:
        core_game.level_order.append(TEMP_ID)
    core_game.world = TEMP_ID
    core_game.locked_level = TEMP_ID
    core_game.load_level()
    print(f"[Play] Loaded editor file: {fp.name} as '{TEMP_ID}'")

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
            action = [0, 0, 0]

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
        core_game = find_core_game(env)
        # Keep locked level after reset
        active_id = level_id or ('__editor_test__' if level_file else None)
        if active_id and core_game and hasattr(core_game,'world'):
            if core_game.world.lower() != active_id.lower():
                core_game.world = active_id.lower()
                if hasattr(core_game,'locked_level'): core_game.locked_level=active_id.lower()
                if hasattr(core_game,'load_level'): core_game.load_level()

env.close()
pygame.quit()
