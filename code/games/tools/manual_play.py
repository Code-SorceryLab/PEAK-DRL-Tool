# code/games/tools/manual_play.py — classic PEAK manual play, driving the game cores directly.
import argparse
import importlib
import json
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
parser.add_argument("--record", nargs="?", const="runs/manual", default=None, metavar="DIR",
                    help="append one row per attempt to DIR/<game>/<player>/episodes.csv "
                         "(same schema as agent runs); one life per attempt")
parser.add_argument("--player", default=None, help="player label for --record (default: human / random)")
parser.add_argument("--episodes", type=int, default=0, help="with --record: quit after N recorded attempts")
parser.add_argument("--warmup", type=float, default=0.0, metavar="MINUTES",
                    help="with --record / --demo: play this long first without anything being recorded; "
                         "then the level restarts and recording begins")
parser.add_argument("--demo", nargs="?", const="runs/demos", default=None, metavar="DIR",
                    help="megaman: record demonstrations for the agents to learn from — every frame's "
                         "sensor reading and buttons, one file per attempt under DIR/megaman/<player>/<level>/. "
                         "You play through the agent's own input limits, so what you show it can do")
parser.add_argument("--invincible", action="store_true",
                    help="megaman debug: take no damage from enemies, shots or spikes (toggle in game with I)")
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

    if left and right:
        move = 0                        # cancel out
    elif left:
        move = 2 if run else 1          # sprint_left / left
    elif right:
        move = 4 if run else 3          # sprint_right / right
    else:
        move = 0

    return [move, int(bool(jump)), int(bool(fire))]


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


def _bomberman_action(keys) -> list:
    """[dx, dy, bomb] — arrows / WASD move, SPACE (or Z) drops a bomb."""
    k = pygame.key.get_pressed()
    dx = int(k[pygame.K_d] or k[pygame.K_RIGHT]) - int(k[pygame.K_a] or k[pygame.K_LEFT])
    dy = int(k[pygame.K_s] or k[pygame.K_DOWN]) - int(k[pygame.K_w] or k[pygame.K_UP])
    return [dx, dy, int(bool(k[pygame.K_SPACE] or k[pygame.K_z]))]


_pad = None                      # the first connected game controller, if any
PAD_JUMP = (0,)                  # DualSense on Linux: 0 Cross, 1 Circle, 2 Triangle, 3 Square
PAD_FIRE = (1, 3)
PAD_DEADZONE = 0.5


def _open_pad() -> None:
    """(Re)open the first controller; called at start and whenever one is plugged or unplugged."""
    global _pad
    pygame.joystick.init()
    _pad = pygame.joystick.Joystick(0) if pygame.joystick.get_count() else None
    if _pad:
        print(f"[Play] Controller: {_pad.get_name()} (stick / d-pad move, Cross jump, Square or Circle fire)")


def _pad_state() -> tuple[int, int, bool, bool]:
    """(dx, dy, jump, fire) from the controller: left stick or d-pad, plus the face buttons."""
    if _pad is None:
        return 0, 0, False, False
    x, y = _pad.get_axis(0), _pad.get_axis(1)
    dx = (x > PAD_DEADZONE) - (x < -PAD_DEADZONE)
    dy = (y > PAD_DEADZONE) - (y < -PAD_DEADZONE)
    if _pad.get_numhats():
        hx, hy = _pad.get_hat(0)
        dx, dy = hx or dx, -hy or dy             # the hat's y points up
    held = lambda ids: any(b < _pad.get_numbuttons() and _pad.get_button(b) for b in ids)
    return dx, dy, held(PAD_JUMP), held(PAD_FIRE)


STAGE_GAMES = {"megaman": "CutMan", "castlevania": "Entrance"}   # stage games: [dx, dy, jump, fire], own level list
STAGED = args.game in STAGE_GAMES


def _megaman_action(keys) -> list:
    """[dx, dy, jump, fire] — arrows / WASD move and climb, SPACE jumps, Z (or J) fires;
    a controller works alongside the keyboard."""
    k = pygame.key.get_pressed()
    px, py, pjump, pfire = _pad_state()
    dx = int(k[pygame.K_d] or k[pygame.K_RIGHT]) - int(k[pygame.K_a] or k[pygame.K_LEFT]) or px
    dy = int(k[pygame.K_s] or k[pygame.K_DOWN]) - int(k[pygame.K_w] or k[pygame.K_UP]) or py
    return [dx, dy, int(bool(k[pygame.K_SPACE]) or pjump), int(bool(k[pygame.K_z] or k[pygame.K_j]) or pfire)]


ACTION_MAPPING = {
    "platformer": _platformer_action,
    "bomberman": _bomberman_action,
    "megaman": _megaman_action,
    "castlevania": _megaman_action,      # the same pad: up / down take the stairs, fire is the whip
    "mario": _platformer_action,
    "sonic": _sonic_action,
    "meatboy": lambda k: [0, 0, 0],   # MeatboyPlayer reads the keyboard itself in human mode
}

_IDLE = {"megaman": [0, 0, 0, 0], "castlevania": [0, 0, 0, 0]}


def _random_action() -> list:
    """Uniform random action in the game's MultiDiscrete space."""
    if args.game == "meatboy":
        return [random.randrange(3), random.randrange(2), random.randrange(2)]
    if args.game == "bomberman":
        return [random.randrange(-1, 2), random.randrange(-1, 2), int(random.random() < 0.05)]
    if STAGED:
        return [random.randrange(-1, 2), random.randrange(-1, 2), random.randrange(2), random.randrange(2)]
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
if args.game == "bomberman":
    env_kwargs = {"level_idx": int(level_id)} if level_id else {}
elif STAGED:
    env_kwargs = {"level": level_id} if level_id else {"level": STAGE_GAMES[args.game]}   # default: the whole stage
    if True:
        env_kwargs["invincible"] = args.invincible
        _open_pad()
    if level_file:
        env_kwargs["level_file"] = level_file   # the core loads an unlisted file itself
        print(f"[Play] Loaded editor file: {Path(level_file).name}")
elif level_id:
    env_kwargs['world'] = level_id
    env_kwargs['lock_level'] = True
    print(f"[Play] Loading level: {level_id}")
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
if level_file and not STAGED:
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

# ── Recording: the same per-episode row an agent run writes ──────
JUMP_IDX = {"bomberman": 2, "megaman": 2, "castlevania": 2}   # where "jump" (bomberman: bomb) sits in the action


class _Recorder:
    """Tracks one attempt on the live core and appends it to episodes.csv when it ends."""

    def __init__(self) -> None:
        self.player = args.player or ("random" if args.random else "human")
        self.game = "mario" if args.game == "platformer" else args.game
        self.run_dir = os.path.join(args.record, self.game, self.player)
        self.recorded = 0
        self.begin()

    def _xy(self) -> tuple[float, float]:
        p = core_game.player
        g = getattr(p, "gObj", p)
        return g.x + g.width / 2.0, g.y + g.height / 2.0

    def begin(self) -> None:
        if hasattr(core_game, "lives"):   # agents get one life per episode; so does a recorded attempt
            core_game.lives = 1
        self.frames = 0
        self.jumps = 0
        self.prev_jump = False
        self.vx_sum = 0.0
        self.route = [tuple(round(v, 1) for v in self._xy())]
        if args.game == "bomberman":
            self.start_cost = max(core_game.start_cost(), 1.0)
            self.best_cost = self.start_cost

    def step(self, action) -> None:
        if args.game == "meatboy" and not args.random:   # MeatboyPlayer reads the keyboard itself
            k = pygame.key.get_pressed()
            jump = bool(k[pygame.K_SPACE] or k[pygame.K_w] or k[pygame.K_UP])
        else:
            jump = bool(action[JUMP_IDX.get(args.game, 1)])
        self.jumps += int(jump and not self.prev_jump)
        self.prev_jump = jump
        self.frames += 1
        self.vx_sum += float(core_game.player.vx)
        if self.frames % 8 == 0:
            self.route.append(tuple(round(v, 1) for v in self._xy()))
        if args.game == "bomberman":
            self.best_cost = min(self.best_cost, core_game.goal_cost())

    def end(self) -> None:
        won = bool(getattr(core_game, "reached_goal", False) or getattr(core_game, "won", False))
        cause = "Success" if won else str(getattr(core_game, "death_cause", "")
                                          or getattr(core_game, "last_cause", "") or "Dead")
        if args.game == "bomberman":
            progress = 1.0 - self.best_cost / self.start_cost
        else:
            progress = self._xy()[0] / (core_game.level_data.width or 1.0)
        progress = min(max(progress, 0.0), 1.0)
        world = level_id or (Path(level_file).stem if level_file else None) \
            or getattr(core_game, "world", None) or getattr(core_game, "_level_idx", "auto")
        path = append_rows(self.run_dir, [[
            self.player, self.game, world, cause, self.jumps,
            int(getattr(core_game, "coins_total", 0) or 0),
            round(self.vx_sum / max(self.frames, 1), 2), round(progress, 3),
            repr(self.route), int(getattr(core_game, "kills_total", 0) or 0),
            int(getattr(core_game, "bricks_destroyed", 0) or 0), round(self.frames / 60.0, 2)]])
        self.recorded += 1
        print(f"[Record] attempt {self.recorded}: {cause}, {progress:.0%} -> {path}")


class _StatsRecorder:
    """megaman: record through the shared telemetry writer (code/stats), the same rows an agent
    test run writes, so the telemetry dashboard cannot tell who played."""

    def __init__(self) -> None:
        from code.stats import episode_log
        from code.stats.episode_stats import make_stats
        self._log, self._make = episode_log, make_stats
        self.player = args.player or ("random" if args.random else "human")
        self.source = "random" if args.random else "human"
        self.level = level_id or (Path(level_file).stem if level_file else "CutMan")
        self.path = os.path.join(args.record, args.game, self.player, "episodes.csv")
        self.recorded = 0
        self.begin()

    def begin(self) -> None:
        self.stats = self._make(args.game, core_game, level=self.level, persona=self.player,
                                source=self.source, agent=self.recorded, session=self.player)

    def step(self, action) -> None:
        self.stats.on_step(bool(action[2]))

    def end(self) -> None:
        self.stats.finish("WON" if core_game.won else "DEAD")
        row = self.stats.to_dict()
        self._log.append(self.path, [row])
        self.recorded += 1
        print(f"[Record] attempt {self.recorded}: {row['status']} {row['cause']}, "
              f"{(row['progress'] or 0):.0%} -> {self.path}")


class _DemoRecorder:
    """megaman: a demonstration is what the agent would have sensed on each frame and what the
    player held. The live core is driven through the training adapter, so the player's input
    passes the same limits an agent's does (megaman_config.yaml, agent_input) and the sensors
    are exactly the network's inputs."""

    def __init__(self) -> None:
        import numpy as np
        from code.neuro.adapters import _ADAPTERS
        from code.neuro.sensors import read_sensors
        self._np, self._read = np, read_sensors
        self.player = args.player or ("random" if args.random else "human")
        self.level = level_id or STAGE_GAMES[args.game]
        self.dir = os.path.join(args.demo, args.game, self.player, self.level)
        os.makedirs(self.dir, exist_ok=True)
        self.saved = len([f for f in os.listdir(self.dir) if f.endswith(".npz")])
        self.adapter = _ADAPTERS[args.game](self.level, 10 ** 9, 5000.0)
        self.adapter.core = core_game          # sense and step the game on screen, not a hidden copy
        self.adapter.attempts = 1
        self.attempts_done = 0
        self.begin()

    def begin(self) -> None:
        self.adapter.reset()
        self.sensors, self.held, self.sent, self.where = [], [], [], []

    def step(self, action):
        a, core = self.adapter, core_game
        vec = self._read(a, "rays")[0]
        dx, dy, jump, fire = int(action[0]), int(action[1]), bool(action[2]), bool(action[3])
        a.step(dx, jump, dy, fire)
        self.sensors.append(vec)
        self.held.append((dx, int(jump), dy, int(fire)))                       # what the player held
        self.sent.append((a._held[0], int(core._prev_jump), a._held[1], int(core._prev_fire)))   # what got through
        self.where.append((core.screen[0], core.screen[1], round(a.x), round(a.y), core.player.hp))
        return not a.alive

    def end(self) -> None:
        np, core, a = self._np, core_game, self.adapter
        self.attempts_done += 1
        if len(self.sensors) < 30:
            return                                                             # a mis-start, not a demonstration
        status = "WON" if core.won else "DEAD"
        span = a._start_cost - a._end_cost                                    # how far along the route, 0..1
        route = 1.0 if core.won else max(0.0, min(1.0, (a._start_cost - a._best_cost) / span))
        path = os.path.join(self.dir, f"{self.saved:03d}_{status.lower()}.npz")
        np.savez_compressed(
            path, sensors=np.asarray(self.sensors, dtype=np.float32), held=np.asarray(self.held, dtype=np.int8),
            sent=np.asarray(self.sent, dtype=np.int8), where=np.asarray(self.where, dtype=np.int32),
            meta=json.dumps({"game": args.game, "level": self.level, "player": self.player, "status": status,
                             "cause": core.death_cause, "progress": round(route, 4),
                             "frames": len(self.sensors), "invincible": bool(core.invincible), "compass": "screen",
                             "columns": {"held": ["move_x", "jump", "move_y", "fire"],
                                         "where": ["screen_x", "screen_y", "x", "y", "hp"]},
                             "agent_input": {"fire_period": a.fire_period, "jump_period": a.jump_period,
                                             "turn_period": a.turn_period}}))
        self.saved += 1
        print(f"[Demo] {status} {core.death_cause}, {route:.0%} of the way, "
              f"{len(self.sensors)} frames -> {path}")


core_game.reset()
demo = recorder = None


def _start_recording() -> None:
    global demo, recorder, append_rows
    demo = _DemoRecorder() if args.demo and STAGED else None
    if args.record and STAGED:
        recorder = _StatsRecorder()
    elif args.record:
        from code.neuro.episodes import append_rows
        recorder = _Recorder()


def _restart_level() -> None:
    """Back to the start of the level being played (a core's reset() may move on to the next)."""
    if STAGED:
        core_game.won = False          # stay on the level being played
    if args.game in INDEXED_GAMES and level_id:
        core_game.won = False          # reset() would otherwise advance to the next level
        core_game._level_idx = int(level_id)
    core_game.reset()
    # Keep locked level after reset
    active_id = level_id or (TEMP_ID if level_file else None)
    if active_id and hasattr(core_game, 'world'):
        if core_game.world.lower() != active_id.lower():
            core_game.world = active_id.lower()
            if hasattr(core_game, 'locked_level'):
                core_game.locked_level = active_id.lower()
            if hasattr(core_game, 'load_level'):
                core_game.load_level()


def _banner(text: str, colour) -> None:
    """A line of text across the top of the window."""
    font = pygame.font.Font(None, 28)
    label = font.render(text, True, colour)
    bar = pygame.Surface((screen.get_width(), label.get_height() + 10), pygame.SRCALPHA)
    bar.fill((0, 0, 0, 170))
    screen.blit(bar, (0, 0))
    screen.blit(label, ((screen.get_width() - label.get_width()) // 2, 5))


# Warm-up: time to get used to the controls before anything counts. Wall-clock, so it runs out
# whatever the player does; the level then restarts and the first recorded attempt begins.
warm_until = pygame.time.get_ticks() + int(args.warmup * 60000) if args.warmup > 0 and (args.record or args.demo) else 0
recording_since = 0
if warm_until:
    print(f"[Play] Warm-up: {args.warmup:g} min, not recorded", flush=True)
else:
    _start_recording()
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
        elif STAGED and event.type in (pygame.JOYDEVICEADDED, pygame.JOYDEVICEREMOVED):
            _open_pad()
        elif STAGED and event.type == pygame.KEYDOWN and event.key == pygame.K_i:
            core_game.invincible = not core_game.invincible
            print(f"[Play] Invincible: {'on' if core_game.invincible else 'off'}")

    if warm_until and pygame.time.get_ticks() >= warm_until:
        warm_until, recording_since = 0, pygame.time.get_ticks()
        _restart_level()
        _start_recording()
        print("[Play] Warm-up over: recording starts now", flush=True)

    keys = pygame.key.get_pressed()

    if args.random:
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
        if demo:                                   # the adapter steps the core, and remembers the frame
            terminated, truncated, info = demo.step(action), False, {}
        else:
            _, _, terminated, truncated, info = core_game.step(action)
        done = terminated or truncated
        if recorder:
            recorder.step(action)
        if done:
            break

    core_game.render(screen, blit_only=True)
    if warm_until:
        left = max(0, warm_until - pygame.time.get_ticks()) // 1000
        _banner(f"Warm-up, not recorded: {left // 60}:{left % 60:02d} left", (255, 210, 90))
    elif recording_since and pygame.time.get_ticks() - recording_since < 4000:
        _banner("Recording starts now", (120, 230, 150))
    pygame.display.flip()

    if info.get("episode_end", False) or done:
        if recorder:
            recorder.end()   # before reset() clears the outcome
            if args.episodes and recorder.recorded >= args.episodes:
                running = False
        if demo:
            demo.end()
            if args.episodes and demo.attempts_done >= args.episodes:
                running = False
        _restart_level()
        if recorder:
            recorder.begin()
        if demo:
            demo.begin()

pygame.quit()
