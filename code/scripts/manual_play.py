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
        
        left  = k[pygame.K_a]
        right = k[pygame.K_d]
        jump  = k[pygame.K_SPACE] or k[pygame.K_w]
        run   = k[pygame.K_LSHIFT] or k[pygame.K_RSHIFT] or k[pygame.K_j]

        # FIX: Check compound combos BEFORE simple ones — most specific first.
        # The old code checked bare `left` before `left+run`, making the
        # left+run branches unreachable dead code.
        if right and run and jump: return 7   # RUN+RIGHT+JUMP
        if left  and run and jump: return 9   # RUN+LEFT+JUMP
        if right and run:          return 5   # RUN+RIGHT
        if left  and run:          return 8   # RUN+LEFT
        if right and jump:         return 4   # RIGHT+JUMP
        if left  and jump:         return 6   # LEFT+JUMP
        if jump:                   return 3   # JUMP
        if right:                  return 2   # RIGHT
        if left:                   return 1   # LEFT
        return 0                               # IDLE

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
env_kwargs = {}
if level_id:
    env_kwargs['world'] = level_id
    env_kwargs['lock_level'] = True   # stay on this level on every reset
    print(f"[Play] Loading level: {level_id}")
if level_file:
    env_kwargs['level_file'] = level_file
    env_kwargs['lock_level'] = True   # always replay the same file
    print(f"[Play] Loading level file: {level_file}")
env = GameEnv(GameCls, render_mode="human", persona="simple", fps=args.fps, **env_kwargs)

# FIX: Force lock_level for manual play so complete_level() doesn't
# auto-advance to the next level.  In training the episode continues
# through multiple levels, but in human play that's confusing.
# We override complete_level to soft-reset the same level instead.

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

# FIX: Override complete_level() for human play.
# In training, completing a level silently loads the next one so the episode
# continues uninterrupted. In manual play that's confusing — the player
# expects to see "LEVEL COMPLETE" and then replay or stay on the same level.
# We monkey-patch complete_level to just reload the current level.
if core_game:
    _original_complete = core_game.complete_level

    def _manual_complete_level():
        """Manual play: record the win but stay on the same level."""
        core_game._level_wins[core_game.world] = core_game._level_wins.get(core_game.world, 0) + 1
        core_game.reached_goal = True
        # Don't advance world — just signal a reload of the same level
        core_game._needs_level_transition = True

    core_game.complete_level = _manual_complete_level
    # Also force locked_level so reset() stays here too
    if not core_game.locked_level:
        core_game.locked_level = core_game.world


if level_id and core_game and hasattr(core_game, 'world'):
    if core_game.world.lower() != level_id.lower():
        core_game.world = level_id.lower()
        core_game.locked_level = level_id.lower()
        if hasattr(core_game, 'load_level'):
            core_game.load_level()
        print(f"[Play] Forced level override: {level_id}")

# ── Handle raw file path (unlisted level) ────────────────────────
if level_file and core_game:
    fp = Path(level_file)
    if not fp.exists():
        print(f"[Play] ERROR: file not found: {level_file}")
    else:
        # Inject a synthetic config entry so the existing load pipeline works.
        # The LevelLoader builds path as: levels_dir / basename, so we must
        # ensure the file is accessible there — either it's already in levels/,
        # or we inject the FULL path into yaml_data so loader can find it.
        import shutil as _shutil
        levels_dir = Path(core_game.loader.level_path)
        dest = levels_dir / fp.name
        if not dest.exists() or dest.resolve() != fp.resolve():
            try:
                _shutil.copy2(str(fp), str(dest))
                print(f"[Play] Copied {fp.name} → {dest}")
            except Exception as _e:
                print(f"[Play] Could not copy file: {_e}")
        # Register as '__editor_test__' in the config manager's in-memory dict
        TEMP_ID = '__editor_test__'
        if 'levels' not in core_game.config_manager.yaml_data:
            core_game.config_manager.yaml_data['levels'] = {}
        core_game.config_manager.yaml_data['levels'][TEMP_ID] = {
            'file': fp.name,
            'time_limit': 300,
        }
        if TEMP_ID not in core_game.config_manager.get_level_order():
            core_game.config_manager.yaml_data['levels'][TEMP_ID] = {
                'file': fp.name, 'time_limit': 300
            }
        # Update level_order in-place if it is cached
        if hasattr(core_game, 'level_order') and TEMP_ID not in core_game.level_order:
            core_game.level_order.append(TEMP_ID)
        core_game.world = TEMP_ID
        core_game.locked_level = TEMP_ID
        if hasattr(core_game, 'load_level'):
            core_game.load_level()
        print(f"[Play] Loaded file: {fp.name} as '{TEMP_ID}'")

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
print("Game session ended.\n")