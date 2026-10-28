# code/games/tools/manual_play.py — classic PEAK manual play, driving the game cores directly.
import argparse
import importlib
import json
import os
import random
import sys
from pathlib import Path

import pygame

from code.games.modules.System.debugging_mods.overlays import PAD_W, draw_nes_pad
from code.games.tools import gamepad
from code.games.modules.System import music
from code.games.modules.System.nes_font import nes_text

os.environ["SDL_VIDEO_CENTERED"] = "1"
# A small arena would lose its bottom rows under the controls card: these games get a band below it
OVERLAY_BAND = {"bomberman": 72}

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
parser.add_argument("--lives", type=int, default=3,
                    help="lives per run (0 = unlimited, for practice); a run ends on a win or when they "
                         "are gone, and --episodes counts runs. Each life is its own recorded row")
parser.add_argument("--quit-after", type=float, default=0.0, metavar="MINUTES",
                    help="count down this much play time on screen (pauses excluded), then quit")
parser.add_argument("--demo", nargs="?", const="runs/demos", default=None, metavar="DIR",
                    help="megaman: record demonstrations for the agents to learn from — every frame's "
                         "sensor reading and buttons, one file per attempt under DIR/megaman/<player>/<level>/. "
                         "You play through the agent's own input limits, so what you show it can do")
parser.add_argument("--invincible", action="store_true",
                    help="megaman debug: take no damage from enemies, shots or spikes (toggle in game with I)")
args = parser.parse_known_args()[0]   # menu.NO_RESTORE rides along on macOS

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
    px, _, pjump, pfire, prun = gamepad.pad_state()

    left  = k[pygame.K_a] or px < 0
    right = k[pygame.K_d] or px > 0
    jump  = k[pygame.K_SPACE] or k[pygame.K_w] or pjump
    run   = k[pygame.K_LSHIFT] or k[pygame.K_RSHIFT] or k[pygame.K_j] or prun
    fire  = k[pygame.K_z] or pfire

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
    px, py, pjump, _, prun = gamepad.pad_state()

    left = k[pygame.K_a] or px < 0
    right = k[pygame.K_d] or px > 0
    jump = k[pygame.K_SPACE] or k[pygame.K_w] or k[pygame.K_UP] or pjump
    run = k[pygame.K_LSHIFT] or k[pygame.K_RSHIFT] or k[pygame.K_j] or prun
    down = k[pygame.K_s] or k[pygame.K_DOWN] or py > 0

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
    """[dx, dy, bomb, detonate] — arrows / WASD move, A (SPACE or Z) drops a bomb, B (X or SHIFT)
    sets off a remote bomb, as on Bomberman II's pad."""
    k = pygame.key.get_pressed()
    px, py, pjump, pfire, _ = gamepad.pad_state()
    dx = int(k[pygame.K_d] or k[pygame.K_RIGHT]) - int(k[pygame.K_a] or k[pygame.K_LEFT]) or px
    dy = int(k[pygame.K_s] or k[pygame.K_DOWN]) - int(k[pygame.K_w] or k[pygame.K_UP]) or py
    return [dx, dy, int(bool(k[pygame.K_SPACE] or k[pygame.K_z] or pjump)),
            int(bool(k[pygame.K_x] or k[pygame.K_LSHIFT] or k[pygame.K_RSHIFT] or pfire))]


def _meatboy_action(keys) -> list:
    """[move, run, jump] from the controller only — MeatboyPlayer adds the keyboard itself in human mode."""
    px, _, pjump, _, prun = gamepad.pad_state()
    return [{-1: 1, 1: 2}.get(px, 0), int(prun), int(pjump)]


def _megaman_action(keys) -> list:
    """[dx, dy, jump, fire] — arrows / WASD move and climb, SPACE jumps, Z (or J) fires;
    a controller works alongside the keyboard."""
    k = pygame.key.get_pressed()
    px, py, pjump, pfire, prun = gamepad.pad_state()
    pfire = pfire or prun                        # no run in Mega Man: Square / X fires as well
    dx = int(k[pygame.K_d] or k[pygame.K_RIGHT]) - int(k[pygame.K_a] or k[pygame.K_LEFT]) or px
    dy = int(k[pygame.K_s] or k[pygame.K_DOWN]) - int(k[pygame.K_w] or k[pygame.K_UP]) or py
    return [dx, dy, int(bool(k[pygame.K_SPACE]) or pjump), int(bool(k[pygame.K_z] or k[pygame.K_j]) or pfire)]


ACTION_MAPPING = {
    "platformer": _platformer_action,
    "bomberman": _bomberman_action,
    "megaman": _megaman_action,
    "mario": _platformer_action,
    "sonic": _sonic_action,
    "meatboy": _meatboy_action,
}

_IDLE = {"megaman": [0, 0, 0, 0]}
# Keys besides SPACE that jump: the pad overlay and inputs.csv show them as A, not as up
JUMP_KEYS = {"platformer": (pygame.K_w,), "meatboy": (pygame.K_w, pygame.K_UP), "sonic": (pygame.K_w, pygame.K_UP)}


def _random_action() -> list:
    """Uniform random action in the game's MultiDiscrete space."""
    if args.game == "meatboy":
        return [random.randrange(3), random.randrange(2), random.randrange(2)]
    if args.game == "bomberman":
        return [random.randrange(-1, 2), random.randrange(-1, 2), int(random.random() < 0.05)]
    if args.game == "megaman":
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
elif args.game == "megaman":
    env_kwargs = {"level": level_id} if level_id else {"level": "CutMan"}   # default: the whole stage
    env_kwargs["invincible"] = args.invincible
    env_kwargs["time_limit"] = 0                # no clock, as on the console
    if args.game == "megaman" and level_file:
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
    env_kwargs['curriculum_enabled'] = False  # levels in campaign order, not the training curriculum's
    env_kwargs['anti_stall'] = False          # standing still is allowed: the stall death is for agents
    # One chosen level (the study, or a level picked): clearing it ends the run. Otherwise the
    # regular game: the core moves straight on to the next level.
    env_kwargs['terminate_on_goal'] = bool(level_id or level_file)

gamepad.open_pad()
core_game = GameCls(render_mode="human", **env_kwargs)
if args.game == "meatboy" and level_id:
    core_game._level_idx = int(level_id)   # meatboy levels are indexed, not named
if args.record and hasattr(core_game, "debug_manager"):
    core_game.debug_manager.update_input = lambda: None   # F3 slow motion and F2 free cam would leave no trace in a row
substeps = max(1, round(getattr(core_game, "fps", args.fps) / args.fps))  # fixed-dt cores keep real-time pace
# Meatboy draws onto whatever surface it is given; the other cores own a window.
screen = core_game._surf if hasattr(core_game, "_surf") else \
    pygame.display.set_mode((core_game.WIDTH, core_game.HEIGHT + OVERLAY_BAND.get(args.game, 0)))

# ── Handle raw file path (unlisted level) ────────────────────────
# Inject the entry directly into the live config_manager instance, then
# load_level() to switch to the editor file (LevelLoader handles abs paths).
if level_file and args.game == "meatboy":       # the level editor's play-test: that file is the only level
    core_game.levels, core_game._level_idx = [level_file], 0
    core_game.reset()
    print(f"[Play] Loaded editor file: {Path(level_file).name}")
elif level_file and args.game != "megaman":
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
JUMP_IDX = {"bomberman": 2, "megaman": 2, "meatboy": 2}   # where "jump" (bomberman: bomb) sits in the action


class _Recorder:
    """Tracks one life on the live core and appends it to episodes.csv when it ends: a row per life,
    as an agent's episode is one life, though a run gives the player the game's usual lives."""

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

    def _clears(self) -> int:
        return sum(getattr(core_game, "_level_wins", {}).values())

    def begin(self) -> None:
        self.lives_seen = getattr(core_game, "lives", None)
        self.clears = self._clears()
        self.last_xy = self._xy()
        self.base = [int(getattr(core_game, k, 0) or 0) for k in ("coins_total", "kills_total", "bricks_destroyed")]
        self.frames = 0
        self.jumps = 0
        self.prev_jump = False
        self.vx_sum = 0.0
        self.route = [tuple(round(v, 1) for v in self._xy())]
        if args.game == "bomberman":
            self.start_cost = max(core_game.start_cost(), 1.0)
            self.best_cost = self.start_cost

    def step(self, action) -> None:
        lives = getattr(core_game, "lives", None)
        if lives is not None and self.lives_seen is not None and lives < self.lives_seen and core_game.alive:
            self.end(self.last_xy)             # Mario lost a life and the core already restarted the level
            self.begin()
            return
        if self._clears() > self.clears and not getattr(core_game, "terminate_on_goal", False):
            self.end(self.last_xy, won=True)   # Mario cleared the level and the core moved on to the next
            self.begin()
            return
        jump = bool(action[JUMP_IDX.get(args.game, 1)])
        if args.game == "meatboy" and not args.random:   # MeatboyPlayer reads the keyboard itself
            k = pygame.key.get_pressed()
            jump = jump or bool(k[pygame.K_SPACE] or k[pygame.K_w] or k[pygame.K_UP])
        self.jumps += int(jump and not self.prev_jump)
        self.prev_jump = jump
        self.frames += 1
        self.vx_sum += float(core_game.player.vx)
        if self.frames % 8 == 0:
            self.route.append(tuple(round(v, 1) for v in self._xy()))
        if args.game == "bomberman":
            self.best_cost = min(self.best_cost, core_game.goal_cost())
        self.last_xy = self._xy()

    def end(self, xy=None, won=None) -> None:
        xy = xy or self._xy()
        if won is None:
            won = bool(getattr(core_game, "reached_goal", False) or getattr(core_game, "won", False))
        cause = "Success" if won else str(getattr(core_game, "death_cause", "")
                                          or getattr(core_game, "last_cause", "")
                                          or ("Timeout" if core_game.alive else "Dead"))   # Meat Boy names no time-out
        if args.game == "bomberman":
            progress = 1.0 - self.best_cost / self.start_cost
        else:
            progress = xy[0] / (core_game.level_data.width or 1.0)
        progress = min(max(progress, 0.0), 1.0)
        world = level_id or (Path(level_file).stem if level_file else None) \
            or getattr(core_game, "world", None) or getattr(core_game, "_level_idx", "auto")
        coins, kills, bricks = (int(getattr(core_game, k, 0) or 0) - b for k, b in
                                zip(("coins_total", "kills_total", "bricks_destroyed"), self.base))
        path = append_rows(self.run_dir, [[
            self.player, self.game, world, cause, self.jumps, coins,
            round(self.vx_sum / max(self.frames, 1), 2), round(progress, 3),
            repr(self.route + [tuple(round(v, 1) for v in xy)]), kills, bricks, round(self.frames / 60.0, 2)]])
        self.recorded += 1
        print(f"[Record] life {self.recorded}: {cause}, {progress:.0%} -> {path}")


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
        from code.neuro.adapters import MegamanAdapter
        from code.neuro.sensors import read_sensors
        self._np, self._read = np, read_sensors
        self.player = args.player or ("random" if args.random else "human")
        self.level = level_id or "CutMan"
        self.dir = os.path.join(args.demo, args.game, self.player, self.level)
        os.makedirs(self.dir, exist_ok=True)
        self.saved = len([f for f in os.listdir(self.dir) if f.endswith(".npz")])
        self.adapter = MegamanAdapter(self.level, 10 ** 9, 5000.0)
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


if args.game == "bomberman" and not level_id and core_game.cfg.get("campaign"):
    core_game._level_idx = int(core_game.cfg["campaign"][0])   # free play is Bomberman II from area 1-1
core_game.reset()
demo = recorder = None
MEGAMAN_NES = args.game == "megaman" and not args.demo   # READY, the death sequence and restart points
NES_SHOW = args.game in ("megaman", "bomberman") and not args.demo   # the console's card / READY and death
UNLIMITED_LIVES = 99                                      # what Mario's HUD shows when lives never run out
lives_left = args.lives
runs_done = 0
if args.game == "platformer":                             # Mario's core counts its own lives
    core_game.max_lives = core_game.lives = args.lives or UNLIMITED_LIVES
if NES_SHOW:
    core_game.start_intro()


class _InputLog:
    """inputs.csv beside episodes.csv: one row per played frame with the buttons held, e.g.
    "[up, left, b]". `life` is the matching row of episodes.csv (1 = its first row)."""
    BUTTONS = ("up", "down", "left", "right", "a", "b", "select", "start")

    def __init__(self, folder: str) -> None:
        import csv
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, "inputs.csv")
        new = not os.path.exists(path)
        self.f = open(path, "a", newline="", encoding="utf-8")
        self.w = csv.writer(self.f)
        if new:
            self.w.writerow(["run", "life", "frame", "buttons"])
        self.frame = 0

    def log(self, run: int, life: int, held) -> None:
        dx, dy, a, b, sel, start = held
        on = (dy < 0, dy > 0, dx < 0, dx > 0, a, b, sel, start)
        self.frame += 1
        self.w.writerow([run, life, self.frame, "[" + ", ".join(n for n, h in zip(self.BUTTONS, on) if h) + "]"])

    def close(self) -> None:
        self.f.close()


input_log = None


def _start_recording() -> None:
    global demo, recorder, append_rows, input_log
    demo = _DemoRecorder() if args.demo and args.game == "megaman" else None
    if args.record and args.game == "megaman":
        recorder = _StatsRecorder()
    elif args.record:
        from code.neuro.episodes import append_rows
        recorder = _Recorder()
    if recorder and not args.random:
        input_log = _InputLog(os.path.dirname(recorder.path) if args.game == "megaman" else recorder.run_dir)


def _restart_level(carry: dict | None = None) -> None:
    """Back to the start of the level being played (a core's reset() may move on to the next).
    carry: what the player brings along (Bomberman's bombs and fire)."""
    if args.game == "megaman":
        core_game.won = False          # stay on the level being played
    if args.game in INDEXED_GAMES and level_id:
        core_game.won = False          # reset() would otherwise advance to the next level
        core_game._level_idx = int(level_id)
    core_game.reset(options={"carry": carry} if carry else None)
    # Keep locked level after reset
    active_id = level_id or (TEMP_ID if level_file else None)
    if active_id and hasattr(core_game, 'world'):
        if core_game.world.lower() != active_id.lower():
            core_game.world = active_id.lower()
            if hasattr(core_game, 'locked_level'):
                core_game.locked_level = active_id.lower()
            if hasattr(core_game, 'load_level'):
                core_game.load_level()
    if NES_SHOW:
        core_game.start_intro()


def _hold(frames: int) -> None:
    """The last frame stays up (Bomberman II freezes 3.5 s on a cleared area)."""
    global running
    for _ in range(frames):
        if not running:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
        pygame.display.flip()
        clock.tick(args.fps)


def _death_sequence() -> None:
    """Play the console's death (Mega Man's halt, puffs and orbs; Bomberman's collapse) before anything restarts."""
    global running
    while running and core_game.death_tick():
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
        core_game.render(screen, blit_only=True)
        pygame.display.flip()
        clock.tick(args.fps)


def _next_life() -> None:
    """A life lost mid-run: Mega Man restarts from the last restart point he reached with his score;
    the other games restart the level."""
    if MEGAMAN_NES:
        start, score = core_game.checkpoint(), core_game.score
        core_game.won = False
        core_game.reset(options={"start": start} if start else None)
        core_game.score = score
        core_game.start_intro()
    elif args.game == "bomberman":
        _restart_level(core_game.carry(died=True))     # bombs and fire survive a death
    else:
        _restart_level()


def _game_over() -> None:
    """GAME OVER over the last frame for three seconds."""
    frozen = screen.copy()
    for _ in range(180):
        screen.blit(frozen, (0, 0))
        img = nes_text("GAME OVER", 4)
        screen.blit(img, ((_view_w() - img.get_width()) // 2, (screen.get_height() - img.get_height()) // 2))
        pygame.event.pump()
        pygame.display.flip()
        clock.tick(args.fps)


def _banner(text: str, colour) -> None:
    """A line of text across the top of the window."""
    font = pygame.font.Font(None, 28)
    label = font.render(text, True, colour)
    bar = pygame.Surface((screen.get_width(), label.get_height() + 10), pygame.SRCALPHA)
    bar.fill((0, 0, 0, 170))
    screen.blit(bar, (0, 0))
    screen.blit(label, ((screen.get_width() - label.get_width()) // 2, 5))


_fonts = {}


def _font(size: int, bold: bool = False) -> pygame.font.Font:
    if (size, bold) not in _fonts:
        _fonts[size, bold] = pygame.font.SysFont("segoeui", size, bold=bold)
    return _fonts[size, bold]


def _view_w() -> int:
    """Width of the game view (the window minus any debug panel)."""
    return min(getattr(core_game, "WIDTH", screen.get_width()), screen.get_width())


def _toggle_panel() -> None:
    """F7: close or reopen the debug panel, resizing the window to fit and re-centring it."""
    global screen
    dm = core_game.debug_manager
    dm.show_panel = not dm.show_panel
    size = (core_game.TOTAL_WIDTH if dm.show_panel else core_game.WIDTH, core_game.HEIGHT)
    screen = core_game._surf = pygame.display.set_mode(size)   # SDL_VIDEO_CENTERED re-centres it


# What A and B do in each game, for the controls overlay
PAD_LABELS = {"platformer": ("Jump", "Run · Fire"), "megaman": ("Jump", "Fire"), "bomberman": ("Bomb", "Remote")}


def _draw_controls(state) -> None:
    """Always-on controls card, bottom-right of the game view: the pad lit by what is held."""
    a_lbl, b_lbl = PAD_LABELS.get(args.game, ("Jump", "Run"))
    w, h = PAD_W + 128, 54
    x, y = _view_w() - w - 10, screen.get_height() - h - 10
    card = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.rect(card, (14, 15, 22, 205), card.get_rect(), border_radius=8)
    screen.blit(card, (x, y))
    draw_nes_pad(screen, x + 7, y + 7, h - 14, state, _font(11, True))
    ty = y + 6
    for key, what in (("A", a_lbl), ("B", b_lbl), ("SELECT", "Pause")):
        screen.blit(_font(11, True).render(key, True, (150, 170, 235)), (x + PAD_W + 16, ty))
        screen.blit(_font(11).render(what, True, (205, 210, 228)), (x + PAD_W + 66, ty))
        ty += 14


TIMER_Y = {"platformer": 74}   # clear of each game's own HUD


def _draw_timer(ms: int, label: str) -> None:
    """A m:ss pill, top-right of the game view."""
    s = max(0, ms) // 1000
    lbl = _font(10, True).render(label, True, (130, 140, 180))
    val = _font(18, True).render(f"{s // 60}:{s % 60:02d}", True, (235, 238, 250))
    w, h = lbl.get_width() + val.get_width() + 30, 32
    x, y = _view_w() - w - 10, TIMER_Y.get(args.game, 10)
    if args.game in OVERLAY_BAND:                       # in the band under the arena, left of the pad
        x, y = 10, screen.get_height() - OVERLAY_BAND[args.game] // 2 - h // 2
    pill = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.rect(pill, (14, 15, 22, 205), pill.get_rect(), border_radius=16)
    screen.blit(pill, (x, y))
    screen.blit(lbl, (x + 12, y + (h - lbl.get_height()) // 2))
    screen.blit(val, (x + w - 12 - val.get_width(), y + (h - val.get_height()) // 2))


PAUSE_ITEMS = ("CONTINUE", "QUIT")


def _draw_pause(frozen: pygame.Surface, choice: int) -> None:
    """The paused frame, dimmed, under an NES-style box: black, white border, a blinking > on the choice."""
    screen.blit(frozen, (0, 0))
    veil = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    veil.fill((0, 0, 0, 150))
    screen.blit(veil, (0, 0))

    cw, ch = min(400, _view_w() - 32), 236
    cx, cy = (_view_w() - cw) // 2, (screen.get_height() - ch) // 2
    pygame.draw.rect(screen, (0, 0, 0), (cx, cy, cw, ch))
    pygame.draw.rect(screen, (252, 252, 252), (cx + 6, cy + 6, cw - 12, ch - 12), 4)
    centre = lambda img, y: screen.blit(img, (cx + (cw - img.get_width()) // 2, y))

    centre(nes_text("PAUSE", 5), cy + 26)
    level = level_id or (Path(level_file).stem if level_file else getattr(core_game, "world", ""))
    centre(nes_text(str(level or ("mario" if args.game == "platformer" else args.game)), 2, (188, 188, 188)), cy + 74)

    blink = (pygame.time.get_ticks() // 400) % 2 == 0
    items = [nes_text(item, 3) for item in PAUSE_ITEMS]
    x = cx + (cw - max(i.get_width() for i in items)) // 2
    for i, img in enumerate(items):
        y = cy + 110 + i * 36
        screen.blit(img, (x, y))
        if i == choice and blink:
            screen.blit(nes_text(">", 3, (252, 188, 60)), (x - 30, y))
    centre(nes_text("SELECT  RESUME", 2, (188, 188, 188)), cy + ch - 36)


# Warm-up: time to get used to the controls before anything counts. Wall-clock, so it runs out
# whatever the player does; the level then restarts and the first recorded attempt begins.
warm_until = pygame.time.get_ticks() + int(args.warmup * 60000) if args.warmup > 0 and (args.record or args.demo) else 0
recording_since = 0
if warm_until:
    print(f"[Play] Warm-up: {args.warmup:g} min, not recorded", flush=True)
else:
    _start_recording()
music.play(args.game)
running = True
random_action = _random_action()
frame = 0
paused_frame = None     # the frozen screen while paused (Select on the pad, or P)
paused_at = 0
pause_choice, pause_prev, pause_armed = 0, (0, False), False   # the pause menu cursor and its input edges
select_was_held = False
play_ms = attempt_ms = 0   # play time (pauses excluded): this session / this attempt

while running:
    tick_ms = clock.tick(args.fps)
    if paused_frame is None:
        play_ms += tick_ms
        attempt_ms += tick_ms
    if args.quit_after and play_ms >= args.quit_after * 60000:
        print("[Play] Time is up", flush=True)
        break

    toggle_pause = False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
            toggle_pause = True
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_F7 and hasattr(core_game, "debug_manager") \
                and hasattr(core_game, "TOTAL_WIDTH") and paused_frame is None:
            _toggle_panel()
        elif event.type in (pygame.JOYDEVICEADDED, pygame.JOYDEVICEREMOVED):
            gamepad.open_pad()
        elif args.game == "megaman" and event.type == pygame.KEYDOWN and event.key == pygame.K_i:
            core_game.invincible = not core_game.invincible
            print(f"[Play] Invincible: {'on' if core_game.invincible else 'off'}")

    select_held = gamepad.button(pygame.CONTROLLER_BUTTON_BACK)
    toggle_pause |= select_held and not select_was_held
    select_was_held = select_held
    resume = False
    if toggle_pause and paused_frame is None:
        paused_frame, paused_at = screen.copy(), pygame.time.get_ticks()
        pause_choice, pause_prev, pause_armed = 0, gamepad.menu_input(), False
    elif toggle_pause:
        resume = True
    elif paused_frame is not None:            # d-pad / arrows move the cursor, A / Start / Enter choose
        dy, confirm = gamepad.menu_input()
        if dy and dy != pause_prev[0]:
            pause_choice = (pause_choice + dy) % len(PAUSE_ITEMS)
        pause_armed |= confirm and not pause_prev[1]
        if pause_armed and not confirm:       # act on release, so the press never reaches the game
            if PAUSE_ITEMS[pause_choice] == "QUIT":
                running = False
            resume = True
        pause_prev = (dy, confirm)
    if resume:
        paused_frame = None
        if warm_until:   # the warm-up clock stops while paused
            warm_until += pygame.time.get_ticks() - paused_at
    if paused_frame is not None:
        _draw_pause(paused_frame, pause_choice)
        pygame.display.flip()
        continue
    frame += 1

    if warm_until and pygame.time.get_ticks() >= warm_until:
        warm_until, recording_since = 0, pygame.time.get_ticks()
        _restart_level()
        attempt_ms = 0
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
        if recorder and not getattr(core_game, "intro", 0):   # READY is not part of the attempt
            recorder.step(action)
        if done:
            break

    if getattr(core_game, "extra_lives", 0):         # a 1-UP from Bomberman's "?" item
        lives_left += core_game.extra_lives
        core_game.extra_lives = 0
    if hasattr(type(core_game), "hud_lives"):          # LEFT on Bomberman's score band: the lives after this one
        core_game.hud_lives = None if not args.lives or warm_until else lives_left - 1
    held = None if args.random else gamepad.held(JUMP_KEYS.get(args.game, ()))
    if input_log and not getattr(core_game, "intro", 0):
        input_log.log(runs_done + 1, recorder.recorded + 1, held)
    if hasattr(core_game, "debug_manager"):
        core_game.debug_manager.pad_input = held
    core_game.render(screen, blit_only=True)
    dm = getattr(core_game, "debug_manager", None)
    if not (dm and dm.show_panel and getattr(core_game, "render_mode", None) == "human"):
        _draw_controls(held)   # the open debug panel already shows the pad in its Input card
    if warm_until:
        left = max(0, warm_until - pygame.time.get_ticks()) // 1000
        _banner(f"Warm-up, not recorded: {left // 60}:{left % 60:02d} left", (255, 210, 90))
    else:
        if recording_since and pygame.time.get_ticks() - recording_since < 4000:
            _banner("Recording starts now", (120, 230, 150))
        if args.quit_after:
            _draw_timer(int(args.quit_after * 60000) - play_ms, "TIME LEFT")
        elif args.game != "meatboy":                   # Meat Boy's own sign shows the attempt's time
            _draw_timer(attempt_ms, "TIME")
    pygame.display.flip()

    if args.game == "platformer" and (not args.lives or warm_until):
        core_game.lives = core_game.max_lives          # practice: lives never run out

    if info.get("episode_end", False) or done:
        won = bool(getattr(core_game, "reached_goal", False) or getattr(core_game, "won", False))
        if NES_SHOW and not core_game.alive:
            _death_sequence()
        if args.game == "meatboy" and not core_game.alive:
            _hold(18)                                  # the splat, then straight back: the game's respawn is near instant
        if won and args.game == "bomberman":
            _hold(210)
        if recorder:
            recorder.end()   # this life's row, before reset() clears the outcome
        if demo:
            demo.end()
            if args.episodes and demo.attempts_done >= args.episodes:
                running = False
        if won and not (level_id or level_file) and not demo:
            print("[Play] Level cleared: on to the next", flush=True)
            # the regular game: next level, lives kept (Bomberman keeps every item while alive)
            _restart_level(core_game.carry(died=False) if args.game == "bomberman" else None)
            if recorder:
                recorder.begin()
            continue
        if not won and args.game != "platformer" and args.lives and not warm_until:
            lives_left -= 1                            # (Mario's core counted its own)
        run_over = won or bool(demo) or args.game == "platformer" or (args.lives and not warm_until and lives_left <= 0)
        if not run_over:
            if running:
                _next_life()                           # same run, next life
                if recorder:
                    recorder.begin()
            continue
        runs_done += 1
        if input_log:
            input_log.frame = 0                        # frames count from 1 in each run
        print("[Play] Level cleared" if won else "[Play] Game over", flush=True)
        if recorder and args.episodes and runs_done >= args.episodes:
            running = False
        if not won and not demo and running:
            _game_over()
        lives_left = args.lives
        _restart_level()
        attempt_ms = 0
        if recorder:
            recorder.begin()
        if demo:
            demo.begin()

if input_log:
    input_log.close()
pygame.quit()
