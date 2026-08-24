# code/games/tools/manual_play.py — classic PEAK manual play, driving the game cores directly.
import argparse
import importlib
import os
import random
import sys
from pathlib import Path

import pygame

os.environ["SDL_VIDEO_WINDOW_POS"] = "100,100"

# The debug help banner prints box glyphs; legacy consoles default to cp1252.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

parser = argparse.ArgumentParser()
parser.add_argument("--game", default="platformer", help="game key")
parser.add_argument("--fps", type=int, default=30, help="target FPS")
parser.add_argument("--level", default=None, help="Level ID from game_config (e.g. Mario1-2; meatboy: level index)")
parser.add_argument("--file", default=None, help="Absolute path to a .txt level file (unlisted)")
parser.add_argument("--random", action="store_true", help="random actions instead of keyboard")
args = parser.parse_args()

# Level ID priority: CLI arg > env var > default
level_id  = args.level or os.environ.get('PEAK_PLAY_LEVEL', None)
level_file = args.file  or os.environ.get('PEAK_PLAY_FILE',  None)

# Normalize game name (mario -> platformer) if needed
if args.game == "mario":
    args.game = "platformer"

# Fail fast on a level the core can't load (otherwise it falls back to a blank world).
INDEXED_GAMES = {"meatboy", "bomberman"}   # levels are list indices, not named ids
if level_id and not level_file and args.game not in INDEXED_GAMES:
    from code.neuro.adapters import validate_level
    validate_level("mario" if args.game == "platformer" else args.game, level_id)

# Enable manual play mode for platformer-like games
if args.game in {"platformer", "megaman"}:
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

    if left and right:
        move = 0                        # cancel out
    elif left:
        move = 2 if run else 1          # sprint_left / left
    elif right:
        move = 4 if run else 3          # sprint_right / right
    else:
        move = 0

    return [move, int(bool(jump)), int(bool(fire))]


def _megaman_action(keys) -> list:
    """
    Return a MultiDiscrete action [move, climb, jump, fire].
    """
    k = pygame.key.get_pressed()

    left  = k[pygame.K_a]
    right = k[pygame.K_d]
    up    = k[pygame.K_w] or k[pygame.K_UP]
    down  = k[pygame.K_s] or k[pygame.K_DOWN]
    jump  = k[pygame.K_SPACE]
    run   = k[pygame.K_LSHIFT] or k[pygame.K_RSHIFT] or k[pygame.K_j]
    fire  = k[pygame.K_z]

    if left and right:
        move = 0
    elif left:
        move = 2 if run else 1
    elif right:
        move = 4 if run else 3
    else:
        move = 0

    if up and not down:
        climb = 1
    elif down and not up:
        climb = 2
    else:
        climb = 0

    return [move, climb, int(bool(jump)), int(bool(fire))]


def _sonic_action(keys) -> list:
    """
    Return a Sonic action [move, jump, down].
    """
    k = pygame.key.get_pressed()

    left = k[pygame.K_a]
    right = k[pygame.K_d]
    jump = k[pygame.K_SPACE] or k[pygame.K_w] or k[pygame.K_UP]
    run = k[pygame.K_LSHIFT] or k[pygame.K_RSHIFT] or k[pygame.K_j]
    down = k[pygame.K_s] or k[pygame.K_DOWN]

    if left and right:
        move = 0
    elif left:
        move = 2 if run else 1
    elif right:
        move = 4 if run else 3
    else:
        move = 0

    return [move, int(bool(jump)), int(bool(down))]


def _meatboy_action(keys) -> list:
    """Return a Meat Boy action [move, run, jump].
      move : 0=idle  1=left  2=right      (3-valued, unlike the platformer's 5)
      run  : 0/1 held  ·  jump : 0/1 held
    MeatboyPlayer.handle_input also reads the keyboard directly in human mode, so
    this mapping and that read agree; both are ORed together."""
    k = pygame.key.get_pressed()

    left  = k[pygame.K_a] or k[pygame.K_LEFT]
    right = k[pygame.K_d] or k[pygame.K_RIGHT]
    jump  = k[pygame.K_SPACE] or k[pygame.K_w] or k[pygame.K_UP]
    run   = k[pygame.K_LSHIFT] or k[pygame.K_RSHIFT] or k[pygame.K_j]

    if left and right:
        move = 0
    elif left:
        move = 1
    elif right:
        move = 2
    else:
        move = 0

    return [move, int(bool(run)), int(bool(jump))]


def _bomberman_action(keys) -> list:
    """[dx, dy, bomb] — arrows / WASD move, SPACE (or Z) drops a bomb."""
    k = pygame.key.get_pressed()
    dx = int(k[pygame.K_d] or k[pygame.K_RIGHT]) - int(k[pygame.K_a] or k[pygame.K_LEFT])
    dy = int(k[pygame.K_s] or k[pygame.K_DOWN]) - int(k[pygame.K_w] or k[pygame.K_UP])
    return [dx, dy, int(bool(k[pygame.K_SPACE] or k[pygame.K_z]))]


ACTION_MAPPING = {
    "platformer": _platformer_action,
    "bomberman": _bomberman_action,
    "mario": _platformer_action,
    "megaman": _megaman_action,
    "sonic": _sonic_action,
    "meatboy": _meatboy_action,
    "bomberman": _bomberman_action,
}

_IDLE = {"megaman": [0, 0, 0, 0], "meatboy": [0, 0, 0]}


def _random_action() -> list:
    """Uniform random action in the game's MultiDiscrete space."""
    if args.game == "megaman":
        return [random.randrange(5), random.randrange(3), random.randrange(2), random.randrange(2)]
    if args.game == "meatboy":
        return [random.randrange(3), random.randrange(2), random.randrange(2)]
    if args.game == "meatboy":
        return [random.randrange(3), random.randrange(2), random.randrange(2)]
    if args.game == "bomberman":
        return [random.randrange(-1, 2), random.randrange(-1, 2), int(random.random() < 0.05)]
    return [random.randrange(5), random.randrange(2), random.randrange(2)]


# --- Validate level_file early
TEMP_ID = '__editor_test__'
if level_file:
    _fp = Path(level_file).resolve()
    if not _fp.exists():
        print(f"[Play] ERROR: level file not found: {level_file}")
        level_file = None
    else:
        level_file = str(_fp)  # normalise to absolute string

# --- Build the core directly (let it load the default first level on __init__)
env_kwargs = {}
if args.game == "meatboy":
    pass          # meatboy takes no world/curriculum kwargs; level is picked below
elif args.game == "bomberman":
    env_kwargs = {"level_idx": int(level_id)} if level_id else {}
elif level_id:
    env_kwargs['world'] = level_id
    env_kwargs['lock_level'] = True
    print(f"[Play] Loading level: {level_id}")
elif args.game == "megaman":
    env_kwargs['curriculum_enabled'] = False
elif args.game == "sonic":
    env_kwargs['curriculum_enabled'] = False
if args.game == "platformer":
    env_kwargs['skip_obs'] = True   # obs dict is unused in manual play

core_game = GameCls(render_mode="human", **env_kwargs)
if args.game == "meatboy" and level_id:
    core_game._level_idx = int(level_id)   # meatboy levels are indexed, not named
substeps = max(1, round(getattr(core_game, "fps", args.fps) / args.fps))  # fixed-dt cores keep real-time pace
# Meatboy draws onto whatever surface it is given; the other cores own a window.
screen = core_game._surf if hasattr(core_game, "_surf") else \
    pygame.display.set_mode((core_game.WIDTH, core_game.HEIGHT))

# ── Handle raw file path (unlisted level) ────────────────────────
# Inject the entry directly into the live config_manager instance, then
# load_level() to switch to the editor file (LevelLoader handles abs paths).
if level_file:
    fp = Path(level_file)
    cm = core_game.config_manager
    if 'levels' not in cm.yaml_data:
        cm.yaml_data['levels'] = {}
    cm.yaml_data['levels'][TEMP_ID] = {
        'file': str(fp),
        'time_limit': 300,
    }
    if hasattr(core_game, 'level_order') and TEMP_ID not in core_game.level_order:
        core_game.level_order.append(TEMP_ID)
    core_game.world = TEMP_ID
    core_game.locked_level = TEMP_ID
    core_game.load_level()
    print(f"[Play] Loaded editor file: {fp.name} as '{TEMP_ID}'")

# Meat Boy is not a platformer_core subclass: it has no _surf, no config_manager
# and no `world` id. Open a window for it and select the level by list index.
if args.game == "meatboy":
    if level_file:
        core_game.levels = [str(Path(level_file).resolve())]
        core_game._level_idx = 0
        print(f"[Play] Loaded file: {Path(level_file).name}")
    elif level_id is not None:
        try:                                   # an index into meatboy_config levels
            core_game._level_idx = int(level_id) % len(core_game.levels)
        except ValueError:                     # or a level file name
            match = [i for i, p in enumerate(core_game.levels) if level_id in p]
            core_game._level_idx = match[0] if match else 0
        print(f"[Play] Level: {core_game.levels[core_game._level_idx]}")
    core_game._surf = pygame.display.set_mode((core_game.WIDTH, core_game.HEIGHT))
    pygame.display.set_caption("PEAK — Super Meat Boy")

core_game.reset()
running = True
random_action = _random_action()
frame = 0

while running:
    clock.tick(args.fps)
    frame += 1

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False

    keys = pygame.key.get_pressed()

    if args.random:
        frame = getattr(core_game, "frame", getattr(core_game, "_steps", 0))
        if frame % 8 == 0:   # re-roll every 8 frames so movement is visible
            random_action = _random_action()
        action = random_action
    else:
        action_fn = ACTION_MAPPING.get(args.game, lambda k: 0)
        action = action_fn(keys)

    # --- CHECK FOR DEBUG OVERRIDES (Free Cam) ---
    if hasattr(core_game, 'debug_manager') and core_game.debug_manager.free_cam_active:
        action = _IDLE.get(args.game, [0, 0, 0])

    for _ in range(substeps):
        _, _, terminated, truncated, info = core_game.step(action)
        done = terminated or truncated
        if done:
            break

    core_game.render(screen, blit_only=True)
    pygame.display.flip()

    if info.get("episode_end", False) or done:
        if args.game in INDEXED_GAMES and level_id:
            core_game.won = False          # reset() would otherwise advance to the next level
            core_game._level_idx = int(level_id)
        core_game.reset()
        if args.game == "meatboy":
            continue          # meatboy.reset() already re-selects its own level
        # Keep locked level after reset
        active_id = level_id or (TEMP_ID if level_file else None)
        if active_id and hasattr(core_game, 'world'):
            if core_game.world.lower() != active_id.lower():
                core_game.world = active_id.lower()
                if hasattr(core_game, 'locked_level'):
                    core_game.locked_level = active_id.lower()
                if hasattr(core_game, 'load_level'):
                    core_game.load_level()

pygame.quit()
