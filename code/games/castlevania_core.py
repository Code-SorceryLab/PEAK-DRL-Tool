"""Castlevania (NES, 1986) — stage 1, with the layout and the numbers taken from a C++ remake of
block 1 (github.com/NearHuscarl/Castlevania; see tools/cv_extract_level.py and
castlevania_config.yaml).

Built like MegamanCore, and with the same interface, so the training adapter, the relay, the
telemetry and the mutants work on both: the caller drives it with step((dx, dy, jump, fire)) and
reads player / level_data / camera / enemies. A level is a stage file or a *section* of one. A
stage is one grid stitched from 16 x 12-tile screens; the camera follows sideways inside a room
(a run of screens cut at the doors), stays locked to one screen row, and slides to the next
screen when the player takes the stairs through the floor.

What differs from Mega Man is the game itself: a jump is committed (no steering in the air), the
weapon is a whip with a wind-up, stairs are flights walked diagonally and entered only at their
foot or head, a hit throws Simon back until he lands, and most enemies arrive in waves from
spawners instead of standing at fixed points. Every random choice goes through `self.rng`.

Not modelled yet: sub-weapons and their pickups, the breakable walls, the hidden money bags.
One knowing departure: down ducks only with no direction held (on the console it always ducks),
so that "down and along" walks on to the next stairs instead of stopping.
"""
from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass

import pygame
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(_HERE, "castlevania_config.yaml")
LEVELS_PATH = os.path.join(_HERE, "castlevania_levels.yaml")

AIR, VOID, SOLID, WATER, GOAL, START = ".", " ", "#", "~", "G", "P"
STAIR_R, STAIR_L = "/", "\\"        # a stair cell rising to the right / to the left
STAIRS = STAIR_R + STAIR_L
DOOR, DOOR_SHUT = "D", "d"          # a door lets the player through once, then shuts behind them
SPIKE = WATER                       # the lethal tile, under the name the shared tools know it by
LADDER = STAIR_R
CANDLES = {"c": "heart", "C": "big_heart", "w": "whip", "m": "money"}   # glyph -> what it drops
PANTHER, ZOMBIES, BATS, FISHMEN = "p", "Z", "B", "F"                    # one enemy; three spawners
SPAWNERS = {ZOMBIES: "zombie", BATS: "bat", FISHMEN: "fishman"}
ENEMY_GLYPHS = PANTHER + ZOMBIES + BATS + FISHMEN
ITEM_GLYPHS = "".join(CANDLES)
BOSS_GLYPH = "X"
NAMES = {"zombie": "Zombie", "panther": "Panther", "bat": "Vampire Bat", "fishman": "Fishman",
         "boss": "Phantom Bat", "fireball": "Fireball"}
KILL_BUDGET = {PANTHER: 1, ZOMBIES: 6, BATS: 3, FISHMEN: 4}   # kills a level expects from each glyph
EPS = 1e-6
MUTANT_ENV = "PEAK_CV_MUTANT"       # a mutant's folder or mutant.json: play that copy of the game


def _config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


BOSS_HP = int(_config()["enemies"]["boss"]["hp"])


def active_mutant() -> dict | None:
    """The game mutant this process plays, if PEAK_CV_MUTANT names one (see megaman_core.py)."""
    path = os.environ.get(MUTANT_ENV)
    if not path:
        return None
    if os.path.isdir(path):
        path = os.path.join(path, "mutant.json")
    import json
    with open(path, encoding="utf-8") as f:
        record = json.load(f)
    record["dir"] = os.path.dirname(os.path.abspath(path))
    return record


@dataclass
class Flight:
    """One flight of stairs: its foot (where the feet stand), which way it rises, how far."""
    bx: float
    by: float
    way: int                # +1 rises to the right, -1 to the left
    length: float           # world px, the same on each axis

    @property
    def top(self) -> tuple[float, float]:
        return self.bx + self.way * self.length, self.by - self.length


@dataclass
class LevelData:
    grid: list[list[str]]
    rows: int
    cols: int
    tile: int
    start: tuple[int, int]
    goal: tuple[int, int]
    screens: set            # (sx, sy) of every screen block that holds anything
    doors: list             # (row, col) of every door cell
    spawns: list            # (glyph, col, row) of every enemy, spawner and boss
    candles: list           # (glyph, col, row)
    flights: list           # every Flight

    @property
    def width(self) -> float:
        return float(self.cols * self.tile)

    @property
    def height(self) -> float:
        return float(self.rows * self.tile)


@dataclass
class Player:
    x: float
    y: float
    width: int
    height: int
    vx: float = 0.0
    vy: float = 0.0
    facing: int = 1
    grounded: bool = False
    climbing: bool = False      # on a flight of stairs
    stair: Flight | None = None
    stair_t: float = 0.0        # world px along the flight, from its foot
    ducking: bool = False
    hp: int = 16
    hp_max: int = 16
    blink: int = 0              # frames of invulnerability left after a hit
    knock: int = 0              # thrown back by a hit: no control until landing
    knock_dir: int = 0
    attack: int = -1            # frames into the whip swing; -1 when the whip is away
    whip_level: int = 1
    hearts: int = 0


@dataclass
class Enemy:
    """NES pixels, centre-anchored, y growing downward (as megaman_enemies.Enemy)."""
    kind: str
    x: float
    y: float
    hw: int
    hh: int
    hp: int
    facing: int = -1
    vx: float = 0.0
    vy: float = 0.0
    state: str = ""
    t: int = 0
    spawn: int = -1             # index of the spawn point that owns it
    visible: bool = True
    hurts: bool = True
    hittable: bool = True
    blink: int = 0
    y0: float = 0.0
    target: tuple = (0.0, 0.0)

    @property
    def name(self) -> str:
        return NAMES[self.kind]


@dataclass
class Bullet:
    x: float
    y: float
    vx: float
    vy: float
    src: str = "Fireball"
    facing: int = 1


@dataclass
class Item:
    kind: str                   # heart, big_heart, whip, money, clear
    x: float
    y: float
    vy: float = 0.0
    ttl: int = 0                # frames left; 0 = stays


ITEM_HALF = (6, 6)
CANDLE_HALF = (4, 12)        # generous: a brazier is taller than a candle


def parse_level(rows: list[str], sc: int, sr: int, tile: int, where: str = "level") -> LevelData:
    """A stage file's lines as the game sees them: sc x sr-tile screens, air, start, flights, spawns."""
    rows = list(rows)
    while rows and not rows[-1].strip():
        rows.pop()
    cols = max(len(r) for r in rows)
    grid = [list(r.ljust(cols, VOID)) for r in rows]
    screens = {(x // sc, y // sr) for y, row in enumerate(grid) for x, ch in enumerate(row) if ch != VOID}
    start = goal = None
    doors, spawns, candles, flights = [], [], [], []
    for y, row in enumerate(grid):
        for x, ch in enumerate(row):
            if ch == VOID and (x // sc, y // sr) in screens:
                row[x] = AIR
            elif ch == START:
                start = (x, y); row[x] = AIR
            elif ch == GOAL:
                goal = (x, y)
            elif ch == DOOR:
                doors.append((y, x))
            elif ch in ENEMY_GLYPHS + BOSS_GLYPH:
                spawns.append((ch, x, y)); row[x] = AIR
            elif ch in CANDLES:
                candles.append((ch, x, y)); row[x] = AIR
    n_rows = len(grid)
    for y, row in enumerate(grid):                      # a flight is a diagonal run of stair cells
        for x, ch in enumerate(row):
            if ch not in STAIRS:
                continue
            way = 1 if ch == STAIR_R else -1
            if y + 1 < n_rows and 0 <= x - way < cols and grid[y + 1][x - way] == ch:
                continue                                # not the foot of its flight
            n = 0
            while y - n >= 0 and 0 <= x + way * n < cols and grid[y - n][x + way * n] == ch:
                n += 1
            flights.append(Flight(float((x if way > 0 else x + 1) * tile), float((y + 1) * tile), way, float(n * tile)))
    if start is None or (goal is None and not any(g == BOSS_GLYPH for g, _, _ in spawns)):
        raise ValueError(f"{where}: needs one P, and a G or a boss (X) to end on")
    return LevelData(grid, len(grid), cols, tile, start, goal, screens, doors, spawns, candles, flights)


def _overlap(ax0, ay0, ax1, ay1, x, y, hw, hh) -> bool:
    return ax0 < x + hw and ax1 > x - hw and ay0 < y + hh and ay1 > y - hh


class CastlevaniaCore:
    WIDTH, HEIGHT = 16 * 32, 12 * 32

    def __init__(self, render_mode: str = "none", max_steps=None, level_idx: int = 0, seed: int = 0,
                 level_file: str | None = None, invincible: bool = False, level: str | None = None) -> None:
        self.cfg = _config()
        self.mutant = active_mutant()
        for key, value in ((self.mutant or {}).get("config") or {}).items():   # "player.gravity": 0.3
            node = self.cfg
            *parents, leaf = key.split(".")
            for k in parents:
                node = node[k]
            node[leaf] = value
        with open(LEVELS_PATH, encoding="utf-8") as f:   # {id: {file: ...}} in campaign order
            self.levels = list(((yaml.safe_load(f) or {}).get("levels") or {}).items())
        self._level_file = level_file
        self.invincible = invincible
        self.render_mode = render_mode
        self.tile_size = int(self.cfg["tile_size"])
        self.scale = self.tile_size / 16.0          # world px per NES px
        self.fps = int(self.cfg.get("fps", 60))
        self.dt = 1.0 / self.fps
        self.max_steps = int(max_steps) if max_steps else int(self.cfg.get("time_limit", 300) * self.fps)
        self.screen_cols = int(self.cfg["screen"]["cols"])
        self.screen_rows = int(self.cfg["screen"]["rows"])
        self.WIDTH = self.screen_cols * self.tile_size
        self.HEIGHT = self.screen_rows * self.tile_size
        self._level_idx = self.level_index(level) if level is not None else level_idx
        self._seed = seed
        self.rng = random.Random(seed)
        self._font = None
        self._backdrop = None
        self._sprites = None
        self._sounds = None
        self.sfx_events: list[str] = []
        self.won = False
        self.reset()

    # ── level ────────────────────────────────────────────────────────────────
    def level_entry(self, idx: int | None = None) -> dict:
        idx = self._level_idx if idx is None else idx
        if not 0 <= idx < len(self.levels):
            raise IndexError(f"castlevania level {idx} out of range (0..{len(self.levels) - 1})")
        name, entry = self.levels[idx]
        return {"id": name, **(entry if isinstance(entry, dict) else {"file": entry})}

    def level_index(self, level) -> int:
        names = [name for name, _ in self.levels]
        if str(level) in names:
            return names.index(str(level))
        if str(level).isdigit() and int(level) < len(names):
            return int(level)
        raise ValueError(f"unknown castlevania level '{level}' (levels: {', '.join(names)})")

    def level_file(self, idx: int | None = None) -> str:
        if self._level_file and idx is None:
            return self._level_file
        file = self.level_entry(idx)["file"]
        m = self.mutant or {}
        if m.get("level_file") and m.get("base_file") == file:      # the mutant's copy of this stage
            return os.path.join(m["dir"], m["level_file"])
        return os.path.join(_HERE, "levels", file)

    def _load_level(self) -> LevelData:
        with open(self.level_file(), encoding="utf-8") as f:
            rows = [r.rstrip("\n") for r in f]
        return parse_level(rows, self.screen_cols, self.screen_rows, self.tile_size, self.level_file())

    def reset(self, *, seed=None, options=None):
        if self.won and not self._level_file:  # beating a level advances to the next (wrapping)
            self._level_idx = (self._level_idx + 1) % len(self.levels)
        if seed is not None:
            self._seed = seed
        self.rng = random.Random(self._seed)
        self.level_data = ld = self._load_level()
        ts, s = self.tile_size, self.scale
        pc = self.cfg["player"]
        self.k = {name: float(pc[name]) * s for name in (
            "walk_speed", "stair_speed", "jump_velocity", "gravity", "max_fall_speed", "stair_reach",
            "knockback_speed", "knockback_hop")}
        self.hurt_blink = int(pc.get("hurt_invulnerable_frames", 120))
        wc = self.cfg["whip"]
        self.whip_windup, self.whip_active = int(wc["windup_frames"]), int(wc["active_frames"])
        w, h = int(pc["width"] * s), int(pc["height"] * s)
        sec = (self.level_entry().get("section") or {}) if not self._level_file else {}
        tx, ty = sec.get("start") or ld.start
        self.player = Player(tx * ts + (ts - w) / 2, (ty + 1) * ts - h, w, h,
                             hp=int(pc.get("hp", 16)), hp_max=int(pc.get("hp", 16)))
        if sec.get("climbing"):
            # A section entered on the stairs starts where the full stage hands it over: on the
            # flight through its start cell, the body's centre just across the edge of the screen
            # (the top edge if the start cell is in the screen's top row, else the bottom edge).
            fl = next(f for f in ld.flights if self._flight_has(f, tx, ty))
            top_row = ty % self.screen_rows == 0
            edge = (ty if top_row else ty + 1) * ts
            centre = edge + self.k["stair_speed"] if top_row else edge - self.k["stair_speed"]
            p = self.player
            p.climbing, p.stair, p.stair_t = True, fl, fl.by - (centre + h / 2.0)
            p.facing = -fl.way if top_row else fl.way
            self._on_flight()
        self.goal_screens = {tuple(g) for g in sec.get("goal_screens") or []}
        for (r, c) in ld.doors:                              # doors already behind the player are shut
            if r // self.screen_rows == ty // self.screen_rows and c < tx:
                ld.grid[r][c] = DOOR_SHUT
        self.candles = [[g, (c + 0.5) * 16.0, (r + 0.5) * 16.0] for g, c, r in ld.candles]   # NES px
        n = len(ld.spawns)
        self.enemies: list[Enemy] = []
        self.enemy_bullets: list[Bullet] = []
        self.items: list[Item] = []
        self.shots: list = []           # the shared tools count projectiles; the whip has none
        self._live = [None] * n         # the enemy a fixed spawn point currently owns
        self._in_view = [False] * n     # a point re-arms only by leaving the screen and coming back
        self._spawn_t = [0] * n         # frames each spawner has been working
        self._spawn_side = [1] * n
        self._room = None
        self.boss: Enemy | None = None
        self.boss_cutter = None         # (Mega Man's boss weapon: the shared sensors look for it)
        self.boss_intro = 0
        self.boss_lock: tuple[float, float] | None = None    # the room is this screen once the boss wakes
        self.boss_dead = False
        self.boss_damage = 0
        self._whip_hit: set = set()
        self.attacks_total = 0
        self.fx: list[list] = []
        self.screen = (tx // self.screen_cols, ty // self.screen_rows)
        self.player.grounded = not self.player.climbing and \
            self._floor_under(self.player.y + h + 1.0, self.player.y + h) is not None
        self.camera_x, self.camera_y = self._camera_target()
        self.scroll_step = float(self.cfg.get("scroll_speed", 4.0)) * s
        self._prev_jump = self._prev_fire = False
        self.score = 0
        self.coins_total = 0
        self.kills_total = 0
        self.timer = float(self.cfg.get("time_limit", 300))
        self.won = False
        self.alive = True
        self.death_cause = ""
        self._steps = 0
        return self._obs(), self._info()

    # ── geometry ─────────────────────────────────────────────────────────────
    def tile(self, tx: int, ty: int) -> str:
        ld = self.level_data
        if 0 <= ty < ld.rows and 0 <= tx < ld.cols:
            return ld.grid[ty][tx]
        return VOID

    def tile_at(self, wx: float, wy: float) -> str:
        return self.tile(int(wx // self.tile_size), int(wy // self.tile_size))

    def solid(self, tx: int, ty: int) -> bool:
        if not 0 <= tx < self.level_data.cols:
            return True
        return self.tile(tx, ty) in (SOLID, DOOR_SHUT)

    def stair_floor(self, tx: int, ty: int) -> bool:
        """A stair cell set into a floor (a flight that leaves through it) holds a walker up."""
        return self.tile(tx, ty) in STAIRS and self.tile(tx, ty - 1) not in STAIRS \
            and self.solid(tx - 1, ty) and self.solid(tx + 1, ty)

    def _centre(self) -> tuple[float, float]:
        p = self.player
        return p.x + p.width / 2.0, p.y + p.height / 2.0

    def _feet(self) -> tuple[float, float]:
        p = self.player
        return p.x + p.width / 2.0, p.y + p.height

    def route_point(self) -> tuple[float, float]:
        """The point progress along the route is read at: just inside the cell the feet are in.
        On a flight that is the stair cell he is on, however close to its corner he stands."""
        p, ts = self.player, self.tile_size
        fl = p.stair
        if fl is None:
            cx, feet = self._feet()
            if p.grounded:                              # half over a ledge: the cell is the one that holds him up
                col, row = int(cx // ts), int(feet // ts)
                held = [tx for tx in self._columns() if self.solid(tx, row) or self.stair_floor(tx, row)]
                if held and col not in held:
                    col = min(held, key=lambda tx: abs(tx - col))
                    cx = min(max(cx, col * ts), (col + 1) * ts - 0.01)
            return cx, feet - 1.0
        k = min(int(p.stair_t // ts), int(fl.length // ts) - 1)        # cells up from the foot
        f = min(p.stair_t / ts - k, 0.999)                             # how far across that cell
        col = fl.bx / ts + k if fl.way > 0 else fl.bx / ts - 1 - k
        x = (col + f) * ts if fl.way > 0 else (col + 1 - f) * ts - 0.01
        return x, (fl.by / ts - k - f) * ts - 0.01

    def _columns(self) -> range:
        p, ts = self.player, self.tile_size
        return range(int(p.x // ts), int((p.x + p.width - EPS) // ts) + 1)

    def _row_bounds(self) -> tuple[float, float]:
        """World-x span of the unbroken run of screens the player's screen sits in."""
        sx, sy = self.screen
        screens = self.level_data.screens
        lo = hi = sx
        while (lo - 1, sy) in screens:
            lo -= 1
        while (hi + 1, sy) in screens:
            hi += 1
        return lo * float(self.WIDTH), (hi + 1) * float(self.WIDTH)

    def _segment(self) -> tuple[float, float]:
        """World-x span of the room the player is in: the run of screens, cut at the doors."""
        if self.boss_lock:
            return self.boss_lock
        lo, hi = self._row_bounds()
        ts, sc, sr = self.tile_size, self.screen_cols, self.screen_rows
        p = self.player
        for (r, c) in self.level_data.doors:
            if r // sr != self.screen[1]:
                continue
            edge = round((c + 0.5) / sc) * sc * ts      # the screen boundary the door stands on
            if edge <= p.x + p.width / 2.0:
                lo = max(lo, edge)
            else:
                hi = min(hi, edge)
        return lo, hi

    def _camera_target(self) -> tuple[float, float]:
        lo, hi = self._segment()
        cx = self.player.x + self.player.width / 2.0
        x = min(max(cx - self.WIDTH / 2.0, lo), max(lo, hi - self.WIDTH))
        return x, float(self.screen[1] * self.HEIGHT)

    @property
    def camera(self) -> tuple[float, float]:
        return self.camera_x, self.camera_y

    @property
    def frame(self) -> int:
        return self._steps

    def nes_view(self) -> tuple[float, float]:
        return self.camera_x / self.scale, self.camera_y / self.scale

    def nes_solid(self, x: float, y: float) -> bool:
        return self.solid(int(x // 16), int(y // 16))

    def nes_player(self) -> tuple[float, float]:
        cx, cy = self._centre()
        return cx / self.scale, cy / self.scale

    def sfx(self, name: str) -> None:
        if self.render_mode == "human":
            self.sfx_events.append(name)

    # ── movement ─────────────────────────────────────────────────────────────
    def _move_x(self) -> None:
        p, ts = self.player, self.tile_size
        if not p.vx:
            return
        lo, hi = self.boss_lock or self._row_bounds()
        nx = min(max(p.x + p.vx, lo), hi - p.width)
        rows = range(int(p.y // ts), int((p.y + p.height - EPS) // ts) + 1)
        if p.vx > 0:
            tx = int((nx + p.width - EPS) // ts)
            if any(self.solid(tx, ty) for ty in rows):
                nx = tx * ts - p.width
        else:
            tx = int(nx // ts)
            if any(self.solid(tx, ty) for ty in rows):
                nx = (tx + 1) * ts
        p.x = nx

    def _floor_under(self, bottom: float, old_bottom: float) -> int | None:
        """Tile row that stops feet arriving at `bottom`, if any."""
        ts = self.tile_size
        ty = int((bottom - EPS) // ts)
        for tx in self._columns():
            if self.solid(tx, ty) or (self.stair_floor(tx, ty) and old_bottom <= ty * ts + EPS):
                return ty
        return None

    def _move_y(self) -> None:
        p, ts = self.player, self.tile_size
        bottom = p.y + p.height
        if p.grounded:
            if self._floor_under(bottom + 1.0, bottom) is None:
                p.grounded = False                 # walked off a ledge: straight down
                p.vx = 0.0
            return
        ny = p.y + p.vy
        if p.vy >= 0:
            ty = self._floor_under(ny + p.height + 2 * EPS, bottom)
            if ty is not None:
                if p.vy > self.k["gravity"] * 4:
                    self.sfx("land")
                p.y, p.vy, p.grounded = ty * ts - p.height, 0.0, True
                p.vx = 0.0
                return
        else:
            ty = int(ny // ts)
            if any(self.solid(tx, ty) for tx in self._columns()):
                ny, p.vy = (ty + 1) * ts, 0.0      # bumped a ceiling
        p.y = ny
        p.vy = min(p.vy + self.k["gravity"], self.k["max_fall_speed"])

    def flight_at(self, dy: int) -> Flight | None:
        """The flight the player would step onto by pressing up (dy < 0, at a foot) or down (at a head)."""
        cx, feet = self._feet()
        reach = self.k["stair_reach"]
        for fl in self.level_data.flights:
            x, y = (fl.bx, fl.by) if dy < 0 else fl.top
            if abs(cx - x) <= reach and abs(feet - y) <= 2.0:
                return fl
        return None

    def _flight_has(self, fl: Flight, col: int, row: int) -> bool:
        """Whether this stair cell belongs to this flight."""
        ts = self.tile_size
        k = fl.by / ts - 1 - row                        # cells up from the foot
        return 0 <= k < fl.length / ts and col == (fl.bx / ts + k if fl.way > 0 else fl.bx / ts - 1 - k)

    def _on_flight(self) -> None:
        p, fl = self.player, self.player.stair
        p.x = fl.bx + fl.way * p.stair_t - p.width / 2.0
        p.y = fl.by - p.stair_t - p.height

    def _walk(self, dx: int, dy: int, jump_edge: bool, attacking: bool) -> None:
        p, k = self.player, self.k
        p.ducking = False
        if p.grounded:
            if dy and not attacking:
                fl = self.flight_at(dy)
                if fl:                                  # onto the stairs: he walks to the step first, as on the console
                    x = fl.bx if dy < 0 else fl.top[0]
                    cx = p.x + p.width / 2.0
                    if abs(cx - x) > k["walk_speed"]:
                        p.facing = 1 if x > cx else -1
                        p.vx = p.facing * k["walk_speed"]
                        self._move_x()
                        return self._move_y()
                    p.climbing, p.grounded, p.stair = True, False, fl
                    p.stair_t = 0.0 if dy < 0 else fl.length
                    p.vx = p.vy = 0.0
                    return self._on_flight()
            p.ducking = dy > 0 and not dx               # down with a direction held walks (the console ducks)
            if attacking or p.ducking:
                p.vx = 0.0                              # the whip roots him to the spot
            else:
                p.vx = dx * k["walk_speed"]
                if dx:
                    p.facing = dx
                if jump_edge:                           # a jump keeps the speed it left the ground with
                    p.vy, p.grounded = -k["jump_velocity"], False
        self._move_x()
        self._move_y()

    def _stairs(self, dx: int, dy: int, attacking: bool) -> None:
        p, fl = self.player, self.player.stair
        move = 0 if attacking else (-dy if dy else dx * fl.way)      # +1: up the flight
        if not move:
            return
        p.facing = fl.way * move
        p.stair_t += move * self.k["stair_speed"]
        if p.stair_t >= fl.length or p.stair_t <= 0.0:              # off the end, onto the floor there
            p.stair_t = min(max(p.stair_t, 0.0), fl.length)
            p.climbing, p.grounded = False, True
        self._on_flight()
        if not p.climbing:
            p.stair = None

    def whip_box(self) -> tuple[float, float, float, float] | None:
        """The lash's box while it can hit (NES px: x0, y0, x1, y1), else None."""
        p, wc, s = self.player, self.cfg["whip"], self.scale
        if not self.whip_windup <= p.attack < self.whip_windup + self.whip_active:
            return None
        cx, feet = self._feet()
        cx, feet = cx / s, feet / s
        tip = cx + p.facing * (p.width / s / 2.0 + wc["reach"][p.whip_level - 1])
        yc = feet - (wc["shoulder"] - (8 if p.ducking else 0))
        return min(cx, tip), yc - wc["height"] / 2.0, max(cx, tip), yc + wc["height"] / 2.0

    def hurt_box(self) -> tuple[float, float, float, float]:
        """The box enemies can touch (NES px)."""
        p, s = self.player, self.scale
        h = self.cfg["player"]["duck_height"] if p.ducking else p.height / s
        return p.x / s, (p.y + p.height) / s - h, (p.x + p.width) / s, (p.y + p.height) / s

    def _track_screen(self) -> None:
        """Which screen the player is in. Going up or down a row needs a screen there: up by the
        stairs, down by the stairs or a fall, and a fall with nothing below is a pit."""
        cx, cy = self._centre()
        sx, sy = int(cx // self.WIDTH), self.screen[1]
        screens = self.level_data.screens
        if cy >= (sy + 1) * self.HEIGHT:
            if (sx, sy + 1) not in screens:
                return self._die("Pit")
            sy += 1
        elif cy < sy * self.HEIGHT and self.player.climbing and (sx, sy - 1) in screens:
            sy -= 1
        self.screen = (sx, sy)

    def _update_camera(self) -> None:
        tx, ty = self._camera_target()
        step = self.scroll_step
        self.camera_x += min(max(tx - self.camera_x, -step), step)
        self.camera_y += min(max(ty - self.camera_y, -step), step)

    def step(self, action):
        """action = (dx, dy, jump, fire) with dx, dy in {-1, 0, 1}; returns
        (obs, reward, terminated, truncated, info)."""
        if not self.alive or self.won:
            return self._obs(), 0.0, True, False, self._info()
        dx = (int(action[0]) > 0) - (int(action[0]) < 0)
        dy = (int(action[1]) > 0) - (int(action[1]) < 0)
        jump = bool(action[2])
        fire = bool(action[3]) if len(action) > 3 else False
        jump_edge, fire_edge = jump and not self._prev_jump, fire and not self._prev_fire
        self._prev_jump, self._prev_fire = jump, fire
        self._steps += 1
        self.timer = max(0.0, self.timer - self.dt)
        p, ts = self.player, self.tile_size

        if self.camera_y != self._camera_target()[1]:   # sliding to the next screen: the world holds still
            self._update_camera()
            return self._finish()

        if p.attack >= 0:
            p.attack += 1
            if p.attack >= self.whip_windup + self.whip_active:
                p.attack = -1
        if fire_edge and p.attack < 0 and not p.knock:
            p.attack = 0
            self._whip_hit = set()
            self.attacks_total += 1
            self.sfx("whip")
        attacking = p.attack >= 0

        if p.knock:                                     # thrown back by a hit: no control until landing
            p.vx = p.knock_dir * self.k["knockback_speed"]
            self._move_x()
            self._move_y()
            if p.grounded:
                p.knock = 0
        elif p.climbing:
            self._stairs(dx, dy, attacking)
        else:
            self._walk(dx, dy, jump_edge, attacking)
        self._track_screen()

        if self.alive and not p.climbing:
            for ty in range(int(p.y // ts), int((p.y + p.height - EPS) // ts) + 1):
                for tx in self._columns():
                    if self.tile(tx, ty) == WATER and not self.invincible:
                        self._die("Water")
        if self.alive:
            grid = self.level_data.grid
            for (r, c) in self.level_data.doors:        # a door shuts once the player is through
                if grid[r][c] == DOOR and p.x >= (c + 1) * ts and r // self.screen_rows == self.screen[1]:
                    grid[r][c] = DOOR_SHUT
                    self.sfx("door")
            if self.tile_at(*self._centre()) == GOAL or self.screen in self.goal_screens:
                self.won = True
                self.score += 1000
        if self.alive and not self.won:
            self._update_world()
        self._update_camera()
        return self._finish()

    def _finish(self):
        if self.alive and not self.won and (self.timer <= 0 or self._steps >= self.max_steps):
            self._die("Timeout")
        done = (not self.alive) or self.won
        return self._obs(), 0.0, done, False, self._info()

    # ── enemies, candles, items ──────────────────────────────────────────────
    def _update_world(self) -> None:
        p, s = self.player, self.scale
        room = (self.screen[1], *self._segment())
        if room != self._room:                          # a new room forgets everything in the old one
            self._room = room
            self.enemies = [e for e in self.enemies if e is self.boss]
            self.enemy_bullets.clear()
            self.items = [it for it in self.items if it.kind == "clear"]
            self._live = [None] * len(self._live)
            self._in_view = [False] * len(self._in_view)
            self._spawn_t = [0] * len(self._spawn_t)
        if p.blink > 0:
            p.blink -= 1
        self.fx = [[x, y, t - 1] for x, y, t in self.fx if t > 1]
        self._spawn(room)
        cam_x, cam_y = self.nes_view()

        for e in list(self.enemies):
            e.t += 1
            if e.blink > 0:
                e.blink -= 1
            keep = getattr(self, "_ai_" + e.kind)(e)
            off = not (cam_x - 40 <= e.x <= cam_x + 256 + 40) or e.y > cam_y + self.HEIGHT / s + 24
            if not keep or (off and e is not self.boss):
                self._remove(e)
        for b in self.enemy_bullets:
            b.x += b.vx
            b.y += b.vy
        self.enemy_bullets = [b for b in self.enemy_bullets if cam_x - 8 <= b.x <= cam_x + 264]

        box = self.whip_box()
        if box:
            for e in list(self.enemies):
                if e.hittable and id(e) not in self._whip_hit and _overlap(*box, e.x, e.y, e.hw, e.hh):
                    self._whip_hit.add(id(e))
                    self._hit_enemy(e)
            for cdl in list(self.candles):
                if _overlap(*box, cdl[1], cdl[2], *CANDLE_HALF):
                    self.candles.remove(cdl)
                    self.sfx("hit")
                    self.items.append(Item(CANDLES[cdl[0]], cdl[1], cdl[2], ttl=int(self.cfg["items"]["ttl"])))
            self.enemy_bullets = [b for b in self.enemy_bullets if not _overlap(*box, b.x, b.y, 4, 4)]

        hb = self.hurt_box()
        for e in self.enemies:
            if e.hurts and _overlap(*hb, e.x, e.y, e.hw, e.hh):
                self._hurt(int(self.cfg["enemies"][e.kind]["damage"]), e.name, e.x)
        for b in self.enemy_bullets:
            if _overlap(*hb, b.x, b.y, 3, 3):
                self._hurt(int(self.cfg["enemies"]["fishman"]["damage"]), b.src, b.x)
        g = self.cfg["player"]["gravity"]
        for it in list(self.items):
            it.vy = min(it.vy + g, 4.0)
            if self.nes_solid(it.x, it.y + it.vy + ITEM_HALF[1]):
                it.y, it.vy = int((it.y + it.vy + ITEM_HALF[1]) // 16) * 16 - ITEM_HALF[1], 0.0
            else:
                it.y += it.vy
            if _overlap(*hb, it.x, it.y, *ITEM_HALF):
                self._collect(it)
            elif it.ttl > 0:
                it.ttl -= 1
                if it.ttl == 0:
                    self.items.remove(it)

    def _spawn(self, room) -> None:
        """Fixed points (panthers, the boss) fill when the screen edge uncovers them; spawners send
        their waves while the player is in their room and near enough."""
        s, sr, sp = self.scale, self.screen_rows, self.cfg["spawners"]
        lo, hi = room[1] / s, room[2] / s
        cam_x, _ = self.nes_view()
        px, py = self.nes_player()
        for i, (glyph, col, row) in enumerate(self.level_data.spawns):
            x, feet = (col + 0.5) * 16.0, (row + 1) * 16.0
            here = row // sr == self.screen[1] and lo <= x < hi
            if glyph in SPAWNERS:
                kind = SPAWNERS[glyph]
                c = sp[kind]
                if not (here and abs(px - x) <= c["reach"] * 16.0):
                    self._spawn_t[i] = 0
                    continue
                t = self._spawn_t[i]
                self._spawn_t[i] += 1
                phase, gap = (t + c["group_every"] - 60) % c["group_every"], max(1, int(c["member_every"]))
                if phase == 0:
                    self._spawn_side[i] = 1 if self.rng.random() < c.get("from_right", 0.5) else -1
                if phase % gap == 0 and phase // gap < c["group"] and \
                        sum(1 for e in self.enemies if e.spawn == i) < 2 * c["group"]:
                    e = getattr(self, "_make_" + kind)(i, feet, self._spawn_side[i], lo, hi)
                    if e:
                        self.enemies.append(e)
                continue
            seen = here and cam_x <= x <= cam_x + 255
            if seen and not self._in_view[i] and self._live[i] is None:
                if glyph == BOSS_GLYPH:
                    if self.boss is None and not self.boss_dead:
                        hw, hh, hp = self._stats("boss")
                        self.boss = self._live[i] = Enemy("boss", x, feet - 8.0, hw, hh, hp, state="sleep", spawn=i,
                                                          hurts=False, hittable=False)
                        self.enemies.append(self.boss)
                else:
                    hw, hh, hp = self._stats("panther")
                    self._live[i] = Enemy("panther", x, feet - hh, hw, hh, hp, state="sit", spawn=i)
                    self.enemies.append(self._live[i])
            self._in_view[i] = seen
        b = self.boss                                   # the boss wakes when the player is well into his screen
        if b and b.state == "sleep":
            left = int(b.x * s // self.WIDTH) * float(self.WIDTH)
            if self.player.x >= left + 3 * self.tile_size:
                self.boss_lock = (left, left + self.WIDTH)
                b.state, b.t, b.hurts, b.hittable = "rest", 0, True, True

    def _stats(self, kind: str) -> tuple[int, int, int]:
        c = self.cfg["enemies"][kind]
        return int(c["box"][0]), int(c["box"][1]), int(c["hp"])

    def _edge(self, side: int, lo: float, hi: float) -> float | None:
        """NES x just outside the screen on that side, if the room goes on there."""
        cam_x, _ = self.nes_view()
        x = cam_x + 264 if side > 0 else cam_x - 8
        return x if lo <= x < hi else None

    def _make_zombie(self, i, feet, side, lo, hi):
        hw, hh, hp = self._stats("zombie")
        x = self._edge(side, lo, hi)
        if x is None:
            x, side = self._edge(-side, lo, hi), -side
        if x is None or self.nes_solid(x, feet - 1) or not self.nes_solid(x, feet + 1):
            return None
        return Enemy("zombie", x, feet - hh, hw, hh, hp, facing=-side, spawn=i)

    def _make_bat(self, i, feet, side, lo, hi):
        hw, hh, hp = self._stats("bat")
        x = self._edge(side, lo, hi)
        if x is None:
            x, side = self._edge(-side, lo, hi), -side
        if x is None:
            return None
        y = self.nes_player()[1] - 8.0
        return Enemy("bat", x, y, hw, hh, hp, facing=-side, spawn=i, y0=y)

    def _make_fishman(self, i, feet, side, lo, hi):
        hw, hh, hp = self._stats("fishman")
        px, _ = self.nes_player()
        x = px + self.rng.choice((-1, 1)) * self.rng.uniform(40.0, 72.0)
        x = min(max(x, lo + 24), hi - 24)
        col, row = int(x // 16), int(feet // 16)
        while row < self.level_data.rows and self.tile(col, row) != WATER:
            row += 1
        if row >= self.level_data.rows:
            return None                                 # no water under there to leap from
        e = Enemy("fishman", x, row * 16.0 + hh, hw, hh, hp, facing=1 if px > x else -1, spawn=i, state="leap")
        e.vy = -float(self.cfg["enemies"]["fishman"]["launch_velocity"])
        e.y0 = row * 16.0                               # the water's surface
        self.sfx("fishman")
        return e

    def _ai_zombie(self, e: Enemy) -> bool:
        e.x += e.facing * self.cfg["enemies"]["zombie"]["speed"]
        return not self.nes_solid(e.x + e.facing * e.hw, e.y)

    def _ai_bat(self, e: Enemy) -> bool:
        c = self.cfg["enemies"]["bat"]
        e.x += e.facing * c["speed"]
        e.y = e.y0 + c["wave"] * math.sin(2 * math.pi * e.t / c["wave_frames"])
        return True

    def _ai_panther(self, e: Enemy) -> bool:
        c = self.cfg["enemies"]["panther"]
        px, py = self.nes_player()
        if e.state == "sit":
            if abs(px - e.x) <= c["wake_x"] and abs(py - e.y) <= c["wake_y"]:
                e.state, e.facing = "run", 1 if px > e.x else -1
            return True
        e.x += e.facing * c["speed"]
        feet = e.y + e.hh
        if e.state == "run":
            if not self.nes_solid(e.x + e.facing * e.hw, feet + 1):      # the ledge: leap off it
                e.state, e.vy = "jump", -c["jump_velocity"]
        else:
            e.vy = min(e.vy + self.cfg["player"]["gravity"], self.cfg["player"]["max_fall_speed"])
            if e.vy > 0 and self.nes_solid(e.x, feet + e.vy):
                e.y, e.vy, e.state = int((feet + e.vy) // 16) * 16.0 - e.hh, 0.0, "run"
                if not e.y0:                                             # it turns on the player once
                    e.y0, e.facing = 1.0, 1 if px > e.x else -1
            else:
                e.y += e.vy
        return True

    def _ai_fishman(self, e: Enemy) -> bool:
        c = self.cfg["enemies"]["fishman"]
        px, _ = self.nes_player()
        feet = e.y + e.hh
        if e.state == "leap" or e.state == "fall":
            e.vy = min(e.vy + self.cfg["player"]["gravity"], self.cfg["player"]["max_fall_speed"])
            if e.vy > 0 and self.nes_solid(e.x, feet + e.vy) and not self.nes_solid(e.x, feet - 1):
                e.y, e.vy, e.state, e.t = int((feet + e.vy) // 16) * 16.0 - e.hh, 0.0, "walk", 0
                e.facing = 1 if px > e.x else -1
            else:
                e.y += e.vy
            return not (e.vy > 0 and e.y - e.hh > e.y0)                  # back under the water
        if e.t % c["fire_every"] >= c["fire_every"] - 30:               # stops, spits, turns to the player
            if e.t % c["fire_every"] == c["fire_every"] - 15:
                self.enemy_bullets.append(Bullet(e.x + e.facing * e.hw, e.y - 6, e.facing * c["fireball_speed"], 0.0,
                                                 facing=e.facing))
            if e.t % c["fire_every"] == c["fire_every"] - 1:
                e.facing = 1 if px > e.x else -1
            return True
        e.x += e.facing * c["speed"]
        if not self.nes_solid(e.x, feet + 1):
            e.state = "fall"
        return True

    def _ai_boss(self, e: Enemy) -> bool:
        c = self.cfg["enemies"]["boss"]
        if e.state == "sleep":
            return True
        s = self.scale
        left, top = self.boss_lock[0] / s, self.screen[1] * self.HEIGHT / s
        px, py = self.nes_player()
        e.hurts = e.blink == 0                          # a bat that has just been whipped does not bite
        if e.state == "rest":
            if e.t >= c["rest_frames"]:                 # down to the player's height, off to one side ...
                side = (1 if e.x > px else -1) if e.x != px else self.rng.choice((-1, 1))
                e.state, e.facing = "dive", -side
                e.target = (min(max(px + side * c["swoop_from"], left + 12.0), left + 244.0), py - 6.0)
            return True
        if e.blink == c["blink_frames"] and e.state in ("dive", "sweep"):
            e.state = "sweep"                           # whipped on the way in: it breaks off
            e.target = (e.x, e.y)
        speed = c["speed"] if e.state == "rise" else c["dive_speed"]
        dx, dy = e.target[0] - e.x, e.target[1] - e.y
        dist = math.hypot(dx, dy)
        if dist <= speed:
            e.x, e.y = e.target
            if e.state == "dive":                       # ... then straight across where the player stood
                e.state = "sweep"
                e.target = (min(max(px + e.facing * c["swoop_past"], left + 12.0), left + 244.0), e.y)
            elif e.state == "sweep":                    # swoop done: back up to somewhere new
                e.state = "rise"
                e.target = (left + self.rng.uniform(40.0, 216.0), top + self.rng.uniform(28.0, 60.0))
            else:
                e.state, e.t = "rest", 0
        else:
            e.x += speed * dx / dist
            e.y += speed * dy / dist
            e.facing = 1 if dx > 0 else -1
        return True

    def _remove(self, e) -> None:
        if e in self.enemies:
            self.enemies.remove(e)
        if e.spawn >= 0 and self._live[e.spawn] is e:
            self._live[e.spawn] = None

    def _collect(self, it) -> None:
        p, ic = self.player, self.cfg["items"]
        self.items.remove(it)
        self.sfx({"clear": "clear", "heart": "heart", "big_heart": "heart", "whip": "powerup", "money": "money"}[it.kind])
        if it.kind == "clear":                          # the orb the boss leaves behind ends the stage
            self.won = True
            self.score += 1000
        elif it.kind in ("heart", "big_heart"):
            p.hearts += int(ic[it.kind])
            self.coins_total += 1
        elif it.kind == "whip":
            p.whip_level = min(3, p.whip_level + 1)
        elif it.kind == "money":
            self.score += int(ic["money"])

    def _hurt(self, amount: int, cause: str, from_x: float) -> None:
        """Damage from an enemy or its shot: thrown back, away from it, until landing; then a
        blink during which nothing but water and pits can hurt. On the stairs he keeps his footing."""
        p = self.player
        if p.blink > 0 or not self.alive or self.invincible:
            return
        p.hp -= amount
        if p.hp <= 0:
            p.hp = 0
            return self._die(cause)
        p.blink = self.hurt_blink
        self.sfx("hurt")
        if p.climbing:
            return
        cx = (p.x + p.width / 2.0) / self.scale
        p.knock, p.knock_dir = 1, (1 if cx > from_x else -1) if cx != from_x else -p.facing
        p.vy, p.grounded, p.ducking, p.attack = -self.k["knockback_hop"], False, False, -1

    def _hit_enemy(self, e) -> bool:
        """The whip touched this enemy."""
        dmg = int(self.cfg["whip"]["damage"][self.player.whip_level - 1])
        self.sfx("hit")
        if e is self.boss:
            if e.blink:
                return False
            hp = e.hp
            e.hp -= dmg
            e.blink = int(self.cfg["enemies"]["boss"]["blink_frames"])
            self.boss_damage += hp - max(e.hp, 0)
            if e.hp <= 0:
                self._boss_defeated()
            return True
        e.hp -= dmg
        if e.hp <= 0:
            self.fx.append([e.x, e.y, 12])
            self.score += int(self.cfg["enemies"][e.kind]["score"])
            self.kills_total += 1
            self._remove(e)
        return True

    def _boss_defeated(self) -> None:
        s = self.scale
        self.fx.append([self.boss.x, self.boss.y, 24])
        self.score += int(self.cfg["enemies"]["boss"]["score"])
        self._remove(self.boss)
        self.boss, self.boss_dead = None, True
        self.enemy_bullets.clear()
        self.kills_total += 1
        col = next(c for g, c, _ in self.level_data.spawns if g == BOSS_GLYPH)
        self.items.append(Item("clear", (col + 0.5) * 16.0,      # the orb falls from where the bat hung: the route's end
                               self.screen[1] * self.HEIGHT / s + 48.0))

    def _die(self, cause: str) -> None:
        self.alive = False
        self.death_cause = cause
        self.sfx("splash" if cause == "Water" else "death")

    window = 21         # cells per side of the grid observation (the grid sensor mode resizes it)

    def _obs(self):
        """A window of cells centred on the player, for the grid sensor mode: channel 0 solid,
        1 worth reaching (goal, items, stairs), 2 dangerous (water, enemies, fireballs)."""
        import numpy as np
        W, half, ts, s = self.window, self.window // 2, self.tile_size, self.scale
        cx, cy = self._centre()
        px, py = int(cx // ts), int(cy // ts)
        out = np.zeros((3, W, W), dtype=np.float32)
        for j in range(W):
            for i in range(W):
                t = self.tile(px - half + i, py - half + j)
                if t in (SOLID, DOOR_SHUT):
                    out[0, j, i] = 1.0
                elif t == GOAL:
                    out[1, j, i] = 1.0
                elif t in STAIRS:
                    out[1, j, i] = 0.5
                elif t == WATER:
                    out[2, j, i] = -1.0

        def mark(ch, x, y, v):                      # an NES-pixel point
            i, j = int(x * s // ts) - px + half, int(y * s // ts) - py + half
            if 0 <= i < W and 0 <= j < W:
                out[ch, j, i] = v
        for it in self.items:
            mark(1, it.x, it.y, 1.0)
        for e in self.enemies:
            if e.hurts:
                mark(2, e.x, e.y, -1.0)
        for b in self.enemy_bullets:
            mark(2, b.x, b.y, -1.0)
        p = self.player
        return {"grids": out, "scalars": np.asarray([p.vx / (3.0 * s), p.vy / (8.0 * s)], dtype=np.float32)}

    def _info(self):
        return {"score": self.score, "won": self.won, "alive": self.alive, "steps": self._steps,
                "hp": self.player.hp, "screen": self.screen, "kills": self.kills_total,
                "boss_hp": self.boss.hp if self.boss else None}

    # ── rendering (the reference maps as a backdrop when they are there; flat shapes otherwise) ──
    PAL = {"void": (0, 0, 0), "sky": (16, 16, 40), "solid": (156, 74, 0), "solid_hi": (228, 148, 60),
           "solid_lo": (80, 32, 0), "stair": (188, 188, 188), "water": (32, 56, 236), "door": (120, 60, 20),
           "door_lo": (60, 24, 8), "goal": (88, 216, 84), "body": (200, 120, 40), "body_hi": (252, 216, 168),
           "eye": (16, 16, 24), "whip": (252, 228, 160), "hp": (228, 60, 40), "hp_bg": (0, 0, 0),
           "boss_hp": (252, 152, 56), "candle": (252, 228, 160), "flame": (252, 116, 96),
           "heart": (228, 60, 40), "money": (252, 252, 252), "whip_item": (188, 188, 188), "clear": (255, 255, 255)}
    ENEMY_COL = {"zombie": (88, 148, 120), "panther": (60, 60, 70), "bat": (60, 90, 200), "fishman": (60, 160, 90),
                 "boss": (150, 60, 170)}

    def _backdrop_art(self):
        if self._backdrop is None:
            from .castlevania_assets import Backdrop
            self._backdrop = Backdrop.get(self.cfg.get("assets"), self.tile_size) or False
        return self._backdrop if not self.mutant else None      # a mutant's layout is not the reference's

    def _sprite_art(self):
        if self._sprites is None:
            from .castlevania_assets import Sprites
            self._sprites = Sprites.get(self.cfg.get("assets"), self.tile_size) or False
        return self._sprites

    def _player_sprite(self) -> tuple[str, bool]:
        """(frame name, flipped) for Simon; the sheet faces right."""
        p, tick = self.player, self._steps
        left = p.facing < 0
        swing = min(3, 1 + p.attack * 3 // (self.whip_windup + 1)) if p.attack >= 0 else 0
        if p.knock:
            return "simon/take_damage_01", left
        if p.climbing:
            way = "up" if p.facing == p.stair.way else "down"
            if swing:
                return f"simon/go_{way}stairs_attack_0{swing}", left
            return f"simon/go_{way}stairs_0{1 + int(p.stair_t // 8) % 2}", left
        if not p.grounded or p.ducking:
            return (f"simon/jump_attack_0{swing}" if swing else "simon/jump_01"), left
        if swing:
            return f"simon/attack_0{swing}", left
        return (f"simon/walk_0{1 + tick // 8 % 4}" if p.vx else "simon/walk_01"), left

    def _enemy_sprite(self, e) -> tuple[str, bool, bool]:
        """(frame name, flipped, hangs from its feet) for an enemy; the sheets face right."""
        tick, right = self._steps, e.facing < 0
        if e.kind == "zombie":
            return f"zombie/walk_0{1 + tick // 12 % 2}", right, True
        if e.kind == "panther":
            return ("panther/idle_01" if e.state == "sit" else f"panther/run_0{1 + tick // 5 % 3}"), right, True
        if e.kind == "bat":
            return f"bat/fly_0{1 + tick // 6 % 3}", right, False
        if e.kind == "fishman":
            every = self.cfg["enemies"]["fishman"]["fire_every"]
            firing = e.state == "walk" and e.t % every >= every - 30
            return ("fishman/shoot_01" if firing else f"fishman/walk_0{1 + tick // 10 % 2}"), right, True
        return ("boss/idle_01" if e.state == "sleep" else f"boss/fly_0{1 + tick // 8 % 2}"), False, False

    def render(self, surface=None, blit_only: bool = False):
        ld, ts, pal, sc = self.level_data, self.tile_size, self.PAL, self.scale
        if surface is None:
            surface = pygame.Surface((self.WIDTH, self.HEIGHT))
        vw, vh = surface.get_size()
        camx, camy = int(self.camera_x), int(self.camera_y)
        surface.fill(pal["void"])
        art = self._backdrop_art()
        if art:
            art.draw(surface, camx, camy)
        for ty in range(max(0, camy // ts), min(ld.rows, (camy + vh) // ts + 1)):
            for tx in range(max(0, camx // ts), min(ld.cols, (camx + vw) // ts + 1)):
                t = ld.grid[ty][tx]
                r = pygame.Rect(tx * ts - camx, ty * ts - camy, ts, ts)
                if t in (DOOR, DOOR_SHUT):              # doors are drawn over the art too: they change
                    pygame.draw.rect(surface, pal["door"], r.inflate(-16 if t == DOOR else 0, 0))
                    pygame.draw.line(surface, pal["door_lo"], r.midtop, r.midbottom, 2)
                if t == VOID or (art and art.covers(tx, ty)):
                    continue
                if t == SOLID:
                    pygame.draw.rect(surface, pal["solid"], r)
                    pygame.draw.line(surface, pal["solid_hi"], r.topleft, (r.right - 1, r.top), 2)
                    pygame.draw.line(surface, pal["solid_lo"], (r.left, r.bottom - 1), (r.right - 1, r.bottom - 1), 2)
                    pygame.draw.line(surface, pal["solid_lo"], r.midtop, r.midbottom, 1)
                    continue
                if t != DOOR_SHUT and t != DOOR:
                    pygame.draw.rect(surface, pal["sky"], r)
                if t in STAIRS:                         # two steps per cell
                    for i in range(2):
                        x = r.left + i * ts // 2 if t == STAIR_R else r.right - (i + 1) * ts // 2
                        pygame.draw.rect(surface, pal["stair"], (x, r.bottom - (i + 1) * ts // 2, ts // 2, 4))
                elif t == WATER:
                    pygame.draw.rect(surface, pal["water"], r)
                elif t == GOAL:
                    pygame.draw.rect(surface, pal["goal"], r.inflate(-8, -8), border_radius=6)

        def box(x, y, hw, hh):                      # an NES centre-anchored box, on screen
            return pygame.Rect(int((x - hw) * sc) - camx, int((y - hh) * sc) - camy, int(2 * hw * sc), int(2 * hh * sc))

        tick = self._steps
        A = self._sprite_art()

        def put(name, x, y, anchor="cc", flip=False) -> bool:
            """Draw a frame with its centre (cc) or bottom centre (cb) at NES point (x, y)."""
            img = A.frame(name, flip) if A else None
            if img is None:
                return False
            w, h = img.get_size()
            surface.blit(img, (int(x * sc - w / 2) - camx, int(y * sc - (h if anchor == "cb" else h / 2)) - camy))
            return True

        for _, x, y in self.candles:
            on_floor = self.nes_solid(x, y + 16)                        # a brazier stands on the ground; a candle hangs
            if put(f"brazier/brazier_0{1 + tick // 8 % 2}", x, y + 8, "cb") if on_floor else \
                    put(f"candle/candle_0{1 + tick // 8 % 2}", x, y):
                continue
            r = box(x, y, *CANDLE_HALF)
            pygame.draw.rect(surface, pal["candle"], (r.left, r.centery, r.width, r.height // 2))
            pygame.draw.circle(surface, pal["flame"], (r.centerx, r.top + r.height // 4 + (tick // 8) % 2), r.width // 2)
        item_art = {"heart": "heart", "big_heart": "big_heart", "whip": "whip_item", "money": "money/money_bag_white",
                    "clear": f"orb/flash_0{1 + tick // 6 % 2}"}
        for it in self.items:
            if put(item_art[it.kind], it.x, it.y + ITEM_HALF[1], "cb"):
                continue
            r = box(it.x, it.y, *ITEM_HALF)
            col = pal.get({"heart": "heart", "big_heart": "heart", "money": "money", "whip": "whip_item"}.get(it.kind, "clear"))
            pygame.draw.ellipse(surface, col, r if it.kind != "heart" else r.inflate(-8, -8))
        for e in self.enemies:
            if e.blink and (tick // 2) % 2:
                continue
            name, flip, feet = self._enemy_sprite(e)
            if put(name, e.x, e.y + e.hh if feet else e.y, "cb" if feet else "cc", flip):
                continue
            r = box(e.x, e.y, e.hw, e.hh)
            pygame.draw.rect(surface, self.ENEMY_COL[e.kind], r, border_radius=4)
            pygame.draw.rect(surface, pal["eye"], r, 1, border_radius=4)
            pygame.draw.rect(surface, pal["flame"], (r.centerx + e.facing * r.width // 4 - 2, r.top + r.height // 4, 4, 4))
        for b in self.enemy_bullets:
            if not put("fireball", b.x, b.y, "cc", b.vx < 0):
                pygame.draw.circle(surface, pal["flame"], (int(b.x * sc) - camx, int(b.y * sc) - camy), int(3 * sc))
        for x, y, t in self.fx:
            if not put(f"flame/flame_0{1 + (12 - min(t, 12)) // 4 % 3}", x, y):
                pygame.draw.circle(surface, pal["clear"], (int(x * sc) - camx, int(y * sc) - camy), int(8 * sc), 3)
        p = self.player
        if (self.alive or self.won) and not (p.blink and (tick // 2) % 2):
            x0, y0, x1, y1 = self.hurt_box()
            body = pygame.Rect(int(x0 * sc) - camx, int(y0 * sc) - camy, int((x1 - x0) * sc), int((y1 - y0) * sc))
            cx, feet = (v / sc for v in self._feet())
            name, flip = self._player_sprite()
            drawn = put(name, cx, feet + 1, "cb", flip)
            if not drawn:
                pygame.draw.rect(surface, pal["body"], body, border_radius=4)
                head = pygame.Rect(0, 0, body.width - 4, body.height // 4)
                head.midtop = (body.centerx, body.top)
                pygame.draw.rect(surface, pal["body_hi"], head, border_radius=3)
                pygame.draw.rect(surface, pal["eye"], (head.centerx + p.facing * 5 - 1, head.top + 4, 3, 4))
            if p.attack >= 0:
                wb = self.whip_box()
                yc = int((feet - (self.cfg["whip"]["shoulder"] - (8 if p.ducking else 0))) * sc) - camy
                level = {1: "1", 2: "2", 3: "3_magenta"}[p.whip_level]
                if wb and put(f"whip/whip_level_{level}_03", (wb[0] + wb[2]) / 2.0 + p.facing * 6, (wb[1] + wb[3]) / 2.0,
                              "cc", p.facing < 0):
                    pass                                # the lash, out (its sheet points right)
                elif wb:
                    pygame.draw.line(surface, pal["whip"], (int(wb[0] * sc) - camx, yc), (int(wb[2] * sc) - camx, yc), 4)
                elif not drawn:                         # the wind-up, behind the shoulder
                    pygame.draw.line(surface, pal["whip"], (body.centerx, yc),
                                     (body.centerx - p.facing * 20, yc - 16), 4)
        if self.render_mode != "none":
            self._hud(surface)
        if self.render_mode == "human":
            if self._sounds is None:
                from .castlevania_assets import Sounds
                self._sounds = Sounds.get(self.cfg.get("assets")) or False
            if self._sounds:
                self._sounds.play(self.sfx_events)
                self._sounds.track(None if not self.alive or self.won else "boss" if self.boss_lock and not self.boss_dead
                                   else "stage")
            self.sfx_events.clear()
        return surface

    def _hud(self, surface) -> None:
        """The console's status bar, reduced: PLAYER and ENEMY energy as rows of pips, and the hearts."""
        p, s = self.player, int(self.scale)
        colour = self.PAL["goal"] if self.invincible else self.PAL["hp"]
        boss_hp = self.boss.hp if self.boss else (0 if self.boss_dead else BOSS_HP)
        for row, (n, total, col) in enumerate(((p.hp, p.hp_max, colour), (boss_hp, BOSS_HP, self.PAL["boss_hp"]))):
            x, y = 8 * s, (6 + 8 * row) * s
            pygame.draw.rect(surface, self.PAL["hp_bg"], (x - s, y - s, total * 5 * s + s, 7 * s))
            for i in range(total):
                pygame.draw.rect(surface, col if i < n else (60, 60, 60), (x + i * 5 * s, y, 4 * s, 5 * s))
        if self._font is None:
            pygame.font.init()
            self._font = pygame.font.Font(None, 10 * s)
        text = self._font.render(f"hearts {p.hearts}   whip {p.whip_level}   time {int(self.timer)}", True, (255, 255, 255))
        surface.blit(text, (8 * s + 17 * 5 * s, 5 * s))
