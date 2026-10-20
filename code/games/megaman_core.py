"""Mega Man (NES, 1987) — one stage, with the layout read from the ROM and the movement numbers
read from the disassembly (github.com/TheRealQuantam/mm1dasm).

Plain class like BombermanCore (no gym, no shared modules): the caller drives it with
step((dx, dy, jump, fire)) and reads player / level_data / camera. Deterministic: no randomness
yet, and every later random choice must go through `self.rng`, seeded at reset().

A level is a stage file, or a *section* of one: the same file, world and enemies, but with its
own starting point and a win on reaching the screens where the next section begins. Sections
are how the stage is trained piece by piece without the pieces differing from the whole.

A stage is one grid stitched from its 16 x 15-tile screens (tools/mm1_extract_level.py). A screen
block left entirely blank is void: there is nothing there, so walking off a row of screens is
blocked and falling out of one with nothing below is a pit. Inside a screen a blank reads as air.
The camera behaves like the console's: it follows sideways within a run of screens, stays locked
to one screen row, and slides to the next screen when the player climbs or falls into it.
"""
from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass

import pygame
import yaml

from . import megaman_enemies as en
from .megaman_assets import Assets

_HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(_HERE, "megaman_config.yaml")
LEVELS_PATH = os.path.join(_HERE, "megaman_levels.yaml")   # the level editor owns this one

AIR, VOID, SOLID, LADDER, SPIKE, GOAL, START = ".", " ", "#", "H", "^", "G", "P"
DOOR, DOOR_SHUT = "D", "d"      # a shutter lets the player through once, then seals behind them
EPS = 1e-6
MUTANT_ENV = "PEAK_MM1_MUTANT"  # a mutant's folder or mutant.json (megaman_mutants.py): play that copy of the game


def active_mutant() -> dict | None:
    """The game mutant this process plays, if PEAK_MM1_MUTANT names one: its record, with `dir`.
    A mutant swaps the stage file for its own copy, or overrides values of megaman_config.yaml;
    level names, sections and every command stay the same."""
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
class LevelData:
    grid: list[list[str]]
    rows: int
    cols: int
    tile: int
    start: tuple[int, int]
    goal: tuple[int, int]
    screens: set            # (sx, sy) of every screen block that holds anything
    doors: list             # (row, col) of every shutter cell
    spawns: list            # (glyph, col, row) of every enemy, item and boss placed in the level

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
    facing: int = 1             # +1 right, -1 left
    grounded: bool = False
    climbing: bool = False
    hp: int = 28
    hp_max: int = 28
    moving_frames: int = 0      # frames the pad has been held on the ground (the first-step inch)
    slide_frames: int = 0       # frames of slide left after letting go
    blink: int = 0              # frames of invulnerability left after a hit
    knock: int = 0              # frames of knockback left (no control)
    knock_dir: int = 0
    shoot_timer: int = 0        # frames the arm stays out after a shot (animation only)


@dataclass
class Shot:
    x: float
    y: float
    vx: float


def parse_level(rows: list[str], sc: int, sr: int, tile: int, where: str = "level") -> LevelData:
    """A stage file's lines as the game sees them: sc x sr-tile screens, air, start, exits, spawns."""
    rows = list(rows)
    while rows and not rows[-1].strip():
        rows.pop()
    cols = max(len(r) for r in rows)
    grid = [list(r.ljust(cols, VOID)) for r in rows]
    screens = {(x // sc, y // sr) for y, row in enumerate(grid) for x, ch in enumerate(row) if ch != VOID}
    start = goal = None
    doors, spawns = [], []
    placed = en.ENEMY_GLYPHS + en.ITEM_GLYPHS + en.BOSS_GLYPH
    for y, row in enumerate(grid):
        for x, ch in enumerate(row):
            if ch == VOID and (x // sc, y // sr) in screens:
                row[x] = AIR                       # the editor writes air as a blank
            elif ch == START:
                start = (x, y); row[x] = AIR
            elif ch == GOAL:
                goal = (x, y)
            elif ch == DOOR:
                doors.append((y, x))
            elif ch in placed:
                spawns.append((ch, x, y)); row[x] = AIR
    if start is None or (goal is None and not any(g == en.BOSS_GLYPH for g, _, _ in spawns)):
        raise ValueError(f"{where}: needs one P, and a G or a boss (X) to end on")
    return LevelData(grid, len(grid), cols, tile, start, goal, screens, doors, spawns)


class MegamanCore:
    WIDTH, HEIGHT = 16 * 32, 15 * 32   # one NES screen at 2x

    def __init__(self, render_mode: str = "none", max_steps=None, level_idx: int = 0, seed: int = 0,
                 level_file: str | None = None, invincible: bool = False, level: str | None = None) -> None:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        self.mutant = active_mutant()
        for key, value in ((self.mutant or {}).get("config") or {}).items():   # "player.gravity": 0.3
            node = self.cfg
            *parents, leaf = key.split(".")
            for k in parents:
                node = node[k]
            node[leaf] = value
        with open(LEVELS_PATH, encoding="utf-8") as f:   # {id: {file: ...}} in campaign order
            data = yaml.safe_load(f) or {}
        self.levels = list((data.get("levels") or {}).items())
        self._n_campaign = len(self.levels)              # disabled ones (practice sections) follow, playable by name
        self.levels += list((data.get("disabled_levels") or {}).items())
        self._level_file = level_file                    # an unlisted file (the editor's Play button)
        self.invincible = invincible                     # debug: enemies, shots and spikes do no damage
        ac = self.cfg.get("assets") or {}
        self.assets = Assets.get(ac, int(self.cfg["tile_size"]) / 16.0) if ac.get("enabled", True) else None
        self.sfx_events: list[str] = []                  # this frame's sounds, played by render()
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
        self.won = False
        self.reset()

    # ── level ────────────────────────────────────────────────────────────────
    def level_entry(self, idx: int | None = None) -> dict:
        """Levels are addressed by position in megaman_levels.yaml (menu / --level N)."""
        idx = self._level_idx if idx is None else idx
        if not 0 <= idx < len(self.levels):
            raise IndexError(f"megaman level {idx} out of range (0..{len(self.levels) - 1})")
        name, entry = self.levels[idx]
        return {"id": name, **(entry if isinstance(entry, dict) else {"file": entry})}

    def level_index(self, level) -> int:
        """A level id (its name in megaman_levels.yaml) or its position in that list."""
        names = [name for name, _ in self.levels]
        if str(level) in names:
            return names.index(str(level))
        if str(level).isdigit() and int(level) < len(names):
            return int(level)
        raise ValueError(f"unknown megaman level '{level}' (levels: {', '.join(names)})")

    def art_entry(self) -> dict:
        """The level entry that says which stage's art to draw (an unlisted file borrows the
        entry of the listed level with the same file name)."""
        if not self._level_file:
            return self.level_entry()
        base = os.path.basename(self._level_file)
        return next((e for _, e in self.levels if isinstance(e, dict) and os.path.basename(e.get("file", "")) == base), {})

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
        if self.won and not self._level_file and self._level_idx < self._n_campaign:
            self._level_idx = (self._level_idx + 1) % self._n_campaign  # beating a level advances to the next (wrapping)
        if seed is not None:
            self._seed = seed
        self.rng = random.Random(self._seed)
        self.level_data = self._load_level()
        self._art = self.art_entry()
        ts, s = self.tile_size, self.scale
        pc = self.cfg["player"]
        self.k = {name: float(pc[name]) * s for name in (
            "run_speed", "air_speed", "step_speed", "slide_speed", "jump_velocity", "jump_cut_velocity",
            "gravity", "max_fall_speed", "ladder_speed", "ladder_mount_drop",
            "knockback_speed", "knockback_hop")}
        self.hurt_blink = int(pc.get("hurt_invulnerable_frames", 111))
        self.knock_frames = int(pc.get("knockback_frames", 29))
        self.step_frames = int(pc.get("step_frames", 7))
        self.slide_frames = int(pc.get("slide_frames", 8))
        w, h = int(pc["width"] * s), int(pc["height"] * s)
        sec = (self.level_entry().get("section") or {}) if not self._level_file else {}
        tx, ty = sec.get("start") or self.level_data.start
        self.player = Player(tx * ts + (ts - w) / 2, (ty + 1) * ts - h, w, h,
                             hp=int(pc.get("hp", 28)), hp_max=int(pc.get("hp", 28)))
        self.player.climbing = bool(sec.get("climbing"))     # a section entered up a ladder starts on it
        if self.player.climbing:
            # ... and where the full stage hands it over: the screen flips as the body's centre
            # crosses the edge between the two screens, not once its feet are in the start tile
            edge = (ty // self.screen_rows + 1) * self.screen_rows * ts
            self.player.y = edge - self.k["ladder_speed"] - h / 2.0
        self.goal_screens = {tuple(s) for s in sec.get("goal_screens") or []}
        for (r, c) in self.level_data.doors:                 # shutters already behind the player are shut
            if r // self.screen_rows == ty // self.screen_rows and c < tx:
                self.level_data.grid[r][c] = DOOR_SHUT
        bc = self.cfg["buster"]
        self.shot_speed = float(bc["speed"]) * s
        self.max_shots = int(bc["max_shots"])
        self.shot_size = (int(bc["size"][0] * s), int(bc["size"][1] * s))
        self.shots: list[Shot] = []
        n = len(self.level_data.spawns)
        self.enemies: list[en.Enemy] = []
        self.enemy_bullets: list[en.Bullet] = []
        self.items: list[en.Item] = []
        self._live = [None] * n         # the enemy each spawn point currently owns
        self._in_view = [False] * n     # a point re-arms only by leaving the screen and coming back
        self._taken = set()             # placed items already collected
        self._room = None
        self.boss: en.Enemy | None = None
        self.boss_cutter: en.RollingCutter | None = None
        self.boss_intro = 0             # frames of the boss-room intro left (no control)
        self.boss_dead = False
        self.fx: list[list] = []        # [x, y, frames left] of each explosion puff
        self.sfx("start")
        self.screen = (tx // self.screen_cols, ty // self.screen_rows)
        self.player.grounded = not self.player.climbing and \
            self._floor_under(self.player.y + h + 1.0, self.player.y + h) is not None
        self.camera_x, self.camera_y = self._camera_target()
        self.scroll_step = float(self.cfg.get("scroll_speed", 4.0)) * s
        self._prev_jump = self._prev_fire = False
        self.boss_damage = 0            # hit points taken off the boss this attempt
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

    def ladder_top(self, tx: int, ty: int) -> bool:
        """The top rung is a floor you can stand on and drop through."""
        return self.tile(tx, ty) == LADDER and self.tile(tx, ty - 1) != LADDER

    def _centre(self) -> tuple[float, float]:
        p = self.player
        return p.x + p.width / 2.0, p.y + p.height / 2.0

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
        """World-x span of the room the player is in: the run of screens, cut at shutters."""
        lo, hi = self._row_bounds()
        ts, sc, sr = self.tile_size, self.screen_cols, self.screen_rows
        p = self.player
        for (r, c) in self.level_data.doors:
            if r // sr != self.screen[1]:
                continue
            edge = round((c + 0.5) / sc) * sc * ts      # the screen boundary the shutter stands on
            if edge <= p.x:
                lo = max(lo, edge)
            else:
                hi = min(hi, edge)
        return lo, hi

    def _camera_target(self) -> tuple[float, float]:
        """Follow the player inside the current room."""
        lo, hi = self._segment()
        p = self.player
        cx = p.x + p.width / 2.0
        x = min(max(cx - self.WIDTH / 2.0, lo), max(lo, hi - self.WIDTH))
        return x, float(self.screen[1] * self.HEIGHT)

    @property
    def camera(self) -> tuple[float, float]:
        return self.camera_x, self.camera_y

    # What the enemy module sees: NES pixels, centre-anchored.
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

    def nes_player_feet(self) -> float:
        return (self.player.y + self.player.height) / self.scale

    def player_airborne(self) -> bool:
        return not self.player.grounded and not self.player.climbing

    def sfx(self, name: str) -> None:
        if self.render_mode == "human":
            self.sfx_events.append(name)

    # ── movement ─────────────────────────────────────────────────────────────
    def _move_x(self) -> None:
        p, ts = self.player, self.tile_size
        if not p.vx:
            return
        lo, hi = self._row_bounds()
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
        """Tile row that stops feet arriving at `bottom`, if any. A ladder top only counts
        when the feet came from above it."""
        ts = self.tile_size
        ty = int((bottom - EPS) // ts)
        for tx in self._columns():
            if self.solid(tx, ty) or (self.ladder_top(tx, ty) and old_bottom <= ty * ts + EPS):
                return ty
        return None

    def _move_y(self) -> None:
        p, ts = self.player, self.tile_size
        bottom = p.y + p.height
        if p.grounded:
            if self._floor_under(bottom + 1.0, bottom) is None:
                p.grounded = False                 # walked off a ledge
            return
        ny = p.y + p.vy
        if p.vy >= 0:
            ty = self._floor_under(ny + p.height + 2 * EPS, bottom)   # feet resting exactly on a tile count
            if ty is not None:
                p.y, p.vy, p.grounded = ty * ts - p.height, 0.0, True
                self.sfx("land")
                return
        else:
            ty = int(ny // ts)
            if any(self.solid(tx, ty) for tx in self._columns()):
                ny, p.vy = (ty + 1) * ts, 0.0      # bumped a ceiling
        p.y = ny
        p.vy = min(p.vy + self.k["gravity"], self.k["max_fall_speed"])

    def _mount(self, tx: int, drop: float = 0.0) -> None:
        p, ts = self.player, self.tile_size
        p.x = tx * ts + (ts - p.width) / 2.0
        p.y += drop
        p.vx = p.vy = 0.0
        p.climbing, p.grounded = True, False
        p.moving_frames = p.slide_frames = 0

    def _walk(self, dx: int, dy: int, jump: bool, jump_edge: bool) -> None:
        p, k, ts = self.player, self.k, self.tile_size
        cx, cy = self._centre()
        if dy < 0 and self.tile_at(cx, cy) == LADDER:
            return self._mount(int(cx // ts))
        if dy > 0 and p.grounded and self.ladder_top(int(cx // ts), int((p.y + p.height + 1) // ts)):
            return self._mount(int(cx // ts), k["ladder_mount_drop"])

        if dx:
            p.facing = dx
            if p.grounded:
                p.moving_frames += 1
                speed = k["step_speed"] if p.moving_frames <= self.step_frames else k["run_speed"]
            else:
                p.moving_frames = self.step_frames + 1     # landing with the pad held keeps running
                speed = k["air_speed"]
            p.vx = dx * speed
            p.slide_frames = self.slide_frames if p.moving_frames > self.step_frames else 0
        else:
            p.moving_frames = 0
            if p.grounded and p.slide_frames > 0:
                p.slide_frames -= 1
                p.vx = p.facing * k["slide_speed"]
            else:
                p.slide_frames = 0
                p.vx = 0.0

        if jump_edge and p.grounded:
            p.vy, p.grounded = -k["jump_velocity"], False
        elif not jump and p.vy < -k["jump_cut_velocity"]:
            p.vy = -k["jump_cut_velocity"]          # letting go early cuts the jump short
        self._move_x()
        self._move_y()

    def _climb(self, dx: int, dy: int, jump_edge: bool) -> None:
        p, k, ts = self.player, self.k, self.tile_size
        if dx:
            p.facing = dx
        tx = int((p.x + p.width / 2.0) // ts)
        if jump_edge and not dy:                    # jump lets go of the ladder
            p.climbing, p.vy = False, k["ladder_speed"]
            return
        if dy < 0:
            ny = p.y - k["ladder_speed"]
            if self.solid(tx, int(ny // ts)):
                return
            p.y = ny
            cy = p.y + p.height / 2.0
            if self.tile_at(p.x + p.width / 2.0, cy) != LADDER:      # past the top rung: step off onto it
                p.y = (int(cy // ts) + 1) * ts - p.height
                p.climbing, p.grounded = False, True
        elif dy > 0:
            ny = p.y + k["ladder_speed"]
            feet = int((ny + p.height - EPS) // ts)
            if self.solid(tx, feet):
                p.y = feet * ts - p.height
                p.climbing, p.grounded = False, True
                return
            p.y = ny
            if self.tile(tx, int((p.y + p.height / 2.0) // ts)) != LADDER and self.tile(tx, feet) != LADDER:
                p.climbing = False                  # ran out of ladder: fall

    def _update_shots(self, fire_edge: bool) -> None:
        p = self.player
        w, h = self.shot_size
        p.shoot_timer = max(0, p.shoot_timer - 1)
        if fire_edge and len(self.shots) < self.max_shots:
            x = p.x + p.width if p.facing > 0 else p.x - w
            self.shots.append(Shot(x, p.y + p.height / 2.0 - h, p.facing * self.shot_speed))
            p.shoot_timer = 16
            self.sfx("shoot")
        kept = []
        for s in self.shots:
            s.x += s.vx                             # buster shots ignore walls, as on the console
            sx, sy = (s.x + w / 2.0) / self.scale, (s.y + h / 2.0) / self.scale
            target = next((e for e in self.enemies if e.hittable and en.shot_hits(e, sx, sy)), None)
            if target is None or not self._hit_enemy(target):
                kept.append(s)
        self.shots = [s for s in kept if self.camera_x - w <= s.x <= self.camera_x + self.WIDTH]

    def _track_screen(self) -> None:
        """Which screen the player is in. Going up needs a ladder; going down only needs a screen
        to land in, and without one it is a pit."""
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

        if self.camera_y != self._camera_target()[1]:   # flipping to the next screen: the world holds still
            self._update_camera()
            return self._finish()
        if self.boss_intro > 0:                         # the boss-room intro takes the pad away
            self.boss_intro -= 1
            dx = dy = 0
            jump = jump_edge = fire_edge = False
            if self.boss_intro == en.BOSS_INTRO_FRAMES - en.BOSS_APPEARS_AT:
                self._spawn_boss()

        if p.knock > 0:                                 # thrown back by a hit: no control, no firing
            p.knock -= 1
            p.vx = p.knock_dir * self.k["knockback_speed"]
            fire_edge = False
            self._move_x()
            self._move_y()
        elif p.climbing:
            self._climb(dx, dy, jump_edge)
        else:
            self._walk(dx, dy, jump, jump_edge)
        self._track_screen()

        if self.alive:
            inset = 2 * self.scale
            for ty in range(int((p.y + inset) // ts), int((p.y + p.height - inset) // ts) + 1):
                for tx in range(int((p.x + inset) // ts), int((p.x + p.width - inset) // ts) + 1):
                    if self.tile(tx, ty) == SPIKE and not self.invincible:
                        self._die("Spike")              # spikes ignore the invulnerability blink
        if self.alive:
            grid = self.level_data.grid
            for (r, c) in self.level_data.doors:      # a shutter seals once the player is through
                if grid[r][c] == DOOR and p.x >= (c + 1) * ts and r // self.screen_rows == self.screen[1]:
                    grid[r][c] = DOOR_SHUT
                    self.sfx("gate")
            if self.tile_at(*self._centre()) == GOAL or self.screen in self.goal_screens:
                self.won = True                         # the goal tile, or the next section's first screen
                self.score += 1000
        if self.alive and not self.won:
            self._update_world(fire_edge)
        self._update_camera()
        return self._finish()

    def _finish(self):
        if self.alive and not self.won and (self.timer <= 0 or self._steps >= self.max_steps):
            self._die("Timeout")
        done = (not self.alive) or self.won
        return self._obs(), 0.0, done, False, self._info()

    # ── enemies, bullets, items (the console's order: contact, shots, enemy AI, spawning) ──
    def _update_world(self, fire_edge: bool) -> None:
        p, s = self.player, self.scale
        room = (self.screen[1], *self._segment())
        if room != self._room:                          # a new room forgets everything in the old one
            self._room = room
            self.enemies.clear(); self.enemy_bullets.clear(); self.shots.clear()
            self.items = [it for it in self.items if it.kind == "clear"]
            self._live = [None] * len(self._live)
            self._in_view = [False] * len(self._in_view)
            self.boss_cutter = None
        if p.blink > 0:
            p.blink -= 1
        self.fx = [[x, y, t - 1] for x, y, t in self.fx if t > 1]
        px, py = self.nes_player()

        for e in self.enemies:
            if e.hurts and en.touches_player(e.x, e.y, e.hw, e.hh, px, py):
                self._hurt(en.STATS[e.kind][4], e.name, e.facing)
        for b in self.enemy_bullets:
            if en.touches_player(b.x, b.y, en.BULLET_HALF, en.BULLET_HALF, px, py):
                self._hurt(en.BULLET_DAMAGE, b.src, b.facing)
        c = self.boss_cutter
        if c and en.touches_player(c.x, c.y, en.ROLLING_CUTTER_HALF, en.ROLLING_CUTTER_HALF, px, py):
            self._hurt(en.STATS[en.BOSS_GLYPH][4], "Cut Man", 1 if c.vx >= 0 else -1)
        for it in list(self.items):
            if en.touches_player(it.x, it.y, *en.ITEM_HALF[it.kind], px, py):
                self._collect(it)
        if not self.alive or self.won:
            return

        self._update_shots(fire_edge)
        for e in list(self.enemies):
            if e.kind != en.BOSS_GLYPH and not en.ROUTINES[e.kind](e, self):
                self._remove(e)
        if self.boss and self.boss_intro == 0:
            en.cut_man(self.boss, self)
            if self.boss_cutter and not en.rolling_cutter(self.boss_cutter, self.boss, self):
                self.boss_cutter = None
        cam_x, cam_y = self.nes_view()
        for b in self.enemy_bullets:                    # enemy shots ignore walls too
            b.x += b.vx
            b.y += b.vy
        self.enemy_bullets = [b for b in self.enemy_bullets
                              if cam_x <= b.x <= cam_x + 255 and en.CULL_TOP <= b.y - cam_y < en.CULL_BOTTOM]
        for it in list(self.items):
            hh = en.ITEM_HALF[it.kind][1]
            it.vy = min(it.vy + en.GRAVITY, en.MAX_FALL)
            if it.vy > 0 and self.nes_solid(it.x, it.y + it.vy + hh):
                it.y, it.vy = int((it.y + it.vy + hh) // 16) * 16 - hh, 0.0
            else:
                it.y += it.vy
            if it.ttl > 0:
                it.ttl -= 1
                if it.ttl == 0:
                    self.items.remove(it)

        lo, hi = room[1] / s, room[2] / s               # spawn what the screen edge just uncovered
        sr = self.screen_rows
        for i, (glyph, col, row) in enumerate(self.level_data.spawns):
            x, y = en.spawn_position(glyph, col, row)
            seen = row // sr == self.screen[1] and lo <= x < hi and cam_x <= x <= cam_x + 255
            if seen and not self._in_view[i] and self._live[i] is None:
                if glyph == en.BOSS_GLYPH:
                    if self.boss is None and not self.boss_dead and self.boss_intro == 0:
                        self.boss_intro = en.BOSS_INTRO_FRAMES
                elif glyph in en.ITEM_GLYPHS:
                    if i not in self._taken:
                        self._live[i] = en.Item("life_large", x, y, spawn=i)
                        self.items.append(self._live[i])
                else:
                    self._live[i] = en.make(glyph, col, row, i)
                    self.enemies.append(self._live[i])
            self._in_view[i] = seen

    def _spawn_boss(self) -> None:
        for i, (glyph, col, row) in enumerate(self.level_data.spawns):
            if glyph == en.BOSS_GLYPH and row // self.screen_rows == self.screen[1]:
                self.boss = en.make(glyph, col, row, i)
                self.enemies.append(self.boss)
                return

    def _remove(self, e) -> None:
        if e in self.enemies:
            self.enemies.remove(e)
        if e.spawn >= 0 and self._live[e.spawn] is e:
            self._live[e.spawn] = None

    def _collect(self, it) -> None:
        p = self.player
        self.items.remove(it)
        if it.spawn >= 0:
            self._taken.add(it.spawn)
            self._live[it.spawn] = None
        if it.kind == "clear":                          # the orb the boss leaves behind ends the stage
            self.won = True
            self.score += 1000
        elif it.kind == "pearl":
            self.score += en.PEARL_SCORE
            self.coins_total += 1
            self.sfx("pearl")
        else:
            p.hp = min(p.hp_max, p.hp + en.ITEM_HEAL[it.kind])
            self.sfx("heal")

    def _hurt(self, amount: int, cause: str, facing: int) -> None:
        """Damage from an enemy or its shot: thrown back the way the attacker faces, then a
        long blink during which nothing but spikes and pits can hurt."""
        p = self.player
        if p.blink > 0 or not self.alive or self.invincible:
            return
        p.hp -= amount
        if p.hp <= 0:
            p.hp = 0
            return self._die(cause)
        self.sfx("hurt")
        p.blink, p.knock, p.knock_dir = self.hurt_blink, self.knock_frames, facing
        p.facing = -facing
        p.vy, p.grounded, p.climbing = -self.k["knockback_hop"], False, False
        p.moving_frames = p.slide_frames = 0

    def _hit_enemy(self, e) -> bool:
        """A buster shot touched this enemy. Returns False if the shot passes through untouched."""
        if e is self.boss:
            if e.blink or self.boss_intro:
                return False                            # blinking: shots fly straight through him
            hp = e.hp
            en.hit_boss(e, self)
            self.boss_damage += hp - max(e.hp, 0)
            self.sfx("enemy_hit")
            if e.hp <= 0:
                self._boss_defeated()
            return True
        if e.shielded:                                  # bounces off armour
            self.sfx("deflect")
            return True
        if self._steps - e.last_hit < 4:                # absorbed by the console's hit latch
            return True
        e.last_hit = self._steps
        e.hp -= en.STATS[e.kind][3]
        if e.hp > 0:
            self.sfx("enemy_hit")
        else:
            self.fx.append([e.x, e.y, 12])
            self.sfx("explosion")
            if en.kill(e, self):
                self._remove(e)
        return True

    def _boss_defeated(self) -> None:
        cam_x, cam_y = self.nes_view()
        self.fx.append([self.boss.x, self.boss.y, 24])
        self.sfx("explosion")
        self._remove(self.boss)
        self.boss, self.boss_cutter, self.boss_dead = None, None, True
        self.enemy_bullets.clear()
        self.kills_total += 1
        self.items.append(en.Item("clear", cam_x + 0x80, cam_y + 0x28))

    def _die(self, cause: str) -> None:
        self.alive = False
        self.death_cause = cause
        self.sfx("death")

    window = 21         # cells per side of the grid observation (the grid sensor mode resizes it)

    def _obs(self):
        """A window of cells centred on the player, for the grid sensor mode: channel 0 solid,
        1 worth reaching (goal, items), 2 dangerous (spikes, enemies, shots). The ray sensors do
        not use it: the adapter reads tile() / player / enemies directly."""
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
                elif t == SPIKE:
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
        if self.boss_cutter:
            mark(2, self.boss_cutter.x, self.boss_cutter.y, -1.0)
        p = self.player
        return {"grids": out, "scalars": np.asarray([p.vx / (3.0 * s), p.vy / (12.0 * s)], dtype=np.float32)}

    def _info(self):
        return {"score": self.score, "won": self.won, "alive": self.alive, "steps": self._steps,
                "hp": self.player.hp, "screen": self.screen, "kills": self.kills_total,
                "boss_hp": self.boss.hp if self.boss else None}

    # ── rendering (placeholder art: flat tiles, no sprites) ──────────────────
    PAL = {"void": (0, 0, 0), "sky": (20, 28, 60), "solid": (176, 184, 196), "solid_hi": (228, 232, 240),
           "solid_lo": (96, 104, 124), "ladder": (236, 180, 84), "ladder_lo": (148, 92, 28),
           "spike": (228, 232, 240), "spike_lo": (140, 148, 164), "door": (232, 92, 60), "door_lo": (120, 36, 24),
           "goal": (88, 216, 84), "body": (0, 112, 236), "body_hi": (60, 188, 252), "skin": (252, 216, 168),
           "eye": (16, 16, 24), "shot": (252, 228, 160), "hp": (252, 228, 160), "hp_bg": (0, 0, 0),
           "bullet": (255, 255, 255), "armour": (120, 124, 136), "boss_hp": (232, 92, 60),
           "life": (252, 228, 160), "pearl": (120, 220, 252), "clear": (255, 255, 255)}
    ENEMY_COL = {"b": (60, 120, 220), "f": (70, 170, 90), "v": (220, 120, 40), "h": (220, 120, 40),
                 "w": (200, 160, 60), "W": (200, 160, 60), "e": (210, 60, 90), "s": (120, 90, 190),
                 "k": (200, 70, 60), "j": (200, 70, 60), "K": (200, 70, 60), "J": (200, 70, 60),
                 "cutter": (200, 204, 212), "X": (236, 236, 240)}

    def render(self, surface=None, blit_only: bool = False):
        ld, ts, pal, A, sc = self.level_data, self.tile_size, self.PAL, self.assets, self.scale
        if surface is None:
            surface = pygame.Surface((self.WIDTH, self.HEIGHT))
        vw, vh = surface.get_size()
        camx, camy = int(self.camera_x), int(self.camera_y)
        surface.fill(pal["void"])

        painted = set()                             # screens drawn from the stage map
        if A:
            for sy in range(camy // self.HEIGHT, (camy + vh - 1) // self.HEIGHT + 1):
                for sx in range(camx // self.WIDTH, (camx + vw - 1) // self.WIDTH + 1):
                    if (sx, sy) in ld.screens:
                        art = A.backdrop(self.level_file(), self._art, ld.grid, sx, sy,
                                         self.screen_cols, self.screen_rows)
                        if art:
                            surface.blit(art, (sx * self.WIDTH - camx, sy * self.HEIGHT - camy))
                            painted.add((sx, sy))
        for ty in range(max(0, camy // ts), min(ld.rows, (camy + vh) // ts + 1)):
            for tx in range(max(0, camx // ts), min(ld.cols, (camx + vw) // ts + 1)):
                t = ld.grid[ty][tx]
                if t == VOID:
                    continue
                r = pygame.Rect(tx * ts - camx, ty * ts - camy, ts, ts)
                if (tx // self.screen_cols, ty // self.screen_rows) in painted:
                    if t != GOAL:
                        continue
                else:
                    pygame.draw.rect(surface, pal["sky"], r)
                if t == SOLID:
                    pygame.draw.rect(surface, pal["solid"], r)
                    pygame.draw.line(surface, pal["solid_hi"], r.topleft, (r.right - 1, r.top), 2)
                    pygame.draw.line(surface, pal["solid_hi"], r.topleft, (r.left, r.bottom - 1), 2)
                    pygame.draw.line(surface, pal["solid_lo"], (r.left, r.bottom - 1), (r.right - 1, r.bottom - 1), 2)
                    pygame.draw.line(surface, pal["solid_lo"], (r.right - 1, r.top), (r.right - 1, r.bottom - 1), 2)
                elif t == LADDER:
                    for x in (r.left + 6, r.right - 8):
                        pygame.draw.rect(surface, pal["ladder"], (x, r.top, 3, ts))
                    for y in range(r.top + 4, r.bottom, 8):
                        pygame.draw.rect(surface, pal["ladder_lo"], (r.left + 6, y, ts - 12, 3))
                elif t == SPIKE:
                    for i in range(2):
                        x0 = r.left + i * ts // 2
                        pts = [(x0, r.bottom), (x0 + ts // 4, r.top + 4), (x0 + ts // 2, r.bottom)]
                        pygame.draw.polygon(surface, pal["spike"], pts)
                        pygame.draw.polygon(surface, pal["spike_lo"], pts, 2)
                elif t in (DOOR, DOOR_SHUT):
                    pygame.draw.rect(surface, pal["door"], r.inflate(-12 if t == DOOR else 0, 0))
                    for y in range(r.top, r.bottom, 8):
                        pygame.draw.line(surface, pal["door_lo"], (r.left, y), (r.right - 1, y), 2)
                elif t == GOAL:
                    glow = 40 * math.sin(self._steps / 8)
                    col = tuple(int(min(255, max(0, c + glow))) for c in pal["goal"])
                    pygame.draw.rect(surface, col, r.inflate(-8, -8), border_radius=6)

        def box(x, y, hw, hh):                      # an NES centre-anchored box, on screen
            return pygame.Rect(int((x - hw) * sc) - camx, int((y - hh) * sc) - camy, int(2 * hw * sc), int(2 * hh * sc))

        def put(name, x, y, anchor="cc", flip_x=False, flip_y=False) -> bool:
            """Draw a sprite with its left/centre/right and top/centre/bottom at NES point (x, y)."""
            img = A.sprite(name, flip_x, flip_y) if A else None
            if img is None:
                return False
            w, h = img.get_size()
            px = x * sc - camx - {"l": 0, "c": w / 2, "r": w}[anchor[0]]
            py = y * sc - camy - {"t": 0, "c": h / 2, "b": h}[anchor[1]]
            surface.blit(img, (int(px), int(py)))
            return True

        tick = self._steps
        for it in self.items:
            if not put(it.kind, it.x, it.y + en.ITEM_HALF[it.kind][1], "cb"):
                r = box(it.x, it.y, *en.ITEM_HALF[it.kind])
                if it.kind == "clear":
                    pygame.draw.ellipse(surface, pal["clear"], r.inflate(int(4 * math.sin(tick / 6)), 0))
                else:
                    pygame.draw.rect(surface, pal["pearl" if it.kind == "pearl" else "life"], r, border_radius=3)
                    pygame.draw.rect(surface, pal["eye"], r, 2, border_radius=3)
        for e in self.enemies:
            if not e.visible or (e.blink and (tick // 2) % 2):
                continue
            if put(*self._enemy_sprite(e)):
                continue
            r = box(e.x, e.y, max(e.hw, 6) if e.kind in "kjKJ" else e.hw, e.hh)
            pygame.draw.rect(surface, self.ENEMY_COL[e.kind], r, border_radius=4)
            pygame.draw.rect(surface, pal["armour"] if e.shielded else pal["eye"], r, 3 if e.shielded else 1,
                             border_radius=4)
            look = e.look if e is self.boss else e.facing
            pygame.draw.rect(surface, pal["eye"], (r.centerx + look * r.width // 4 - 2, r.top + r.height // 4, 4, 5))
        for b in self.enemy_bullets:
            if not put("bullet", b.x, b.y):
                pygame.draw.circle(surface, pal["bullet"], (int(b.x * sc) - camx, int(b.y * sc) - camy), int(2 * sc))
        if self.boss_cutter:
            c = self.boss_cutter
            if not put(f"rolling{(tick // 4) % 4}", c.x, c.y):
                pygame.draw.circle(surface, pal["armour"], (int(c.x * sc) - camx, int(c.y * sc) - camy), int(6 * sc))
                pygame.draw.circle(surface, pal["clear"], (int(c.x * sc) - camx, int(c.y * sc) - camy), int(6 * sc), 2)
        for s in self.shots:
            w, h = self.shot_size
            if not put("shot", (s.x + w / 2) / sc, (s.y + h / 2) / sc):
                pygame.draw.ellipse(surface, pal["shot"], (int(s.x) - camx, int(s.y) - camy, w, h))
        for x, y, _ in self.fx:
            if not put("boom", x, y):
                pygame.draw.circle(surface, pal["clear"], (int(x * sc) - camx, int(y * sc) - camy), int(10 * sc), 3)
        p = self.player
        if (self.alive or self.won) and not (p.blink and (tick // 2) % 2):
            if not put(*self._player_sprite()):
                body = pygame.Rect(int(p.x) - camx, int(p.y) - camy, p.width, p.height)
                pygame.draw.rect(surface, pal["body"], body, border_radius=4)
                pygame.draw.rect(surface, pal["body_hi"], (body.left, body.top, body.width, body.height // 3),
                                 border_radius=4)
                face = pygame.Rect(0, 0, body.width // 2, body.height // 5)
                face.midtop = (body.centerx + p.facing * body.width // 5, body.top + body.height // 4)
                pygame.draw.rect(surface, pal["skin"], face)
                pygame.draw.rect(surface, pal["eye"], (face.centerx + p.facing * 3 - 1, face.top + 2, 3, face.height - 4))
                arm = pygame.Rect(0, 0, 10, 8)      # the buster arm points where a shot would leave
                arm.midleft = body.midright if p.facing > 0 else (body.left - 10, body.centery)
                pygame.draw.rect(surface, pal["body_hi"], arm, border_radius=3)
        if self.render_mode != "none":
            self._hud(surface)
        if self.sfx_events:
            if A:
                A.play(self.sfx_events)
            self.sfx_events.clear()
        return surface

    # Which frame, and where it hangs: (name, x, y, anchor, flip_x, flip_y) in NES pixels. The
    # anchor names the sprite's left/centre/right and top/centre/bottom that sits on (x, y).
    def _player_sprite(self):
        p, tick = self.player, self._steps
        cx, feet = self.nes_player()[0], self.nes_player_feet()
        right = p.facing > 0                         # the sheet faces left
        shoot = p.shoot_timer > 0

        def aiming(name, half):                      # arm-out frames are wider on the facing side
            return (name, cx - half if right else cx + half, feet, "lb" if right else "rb", right, False)

        if p.knock > 0:
            return ("mm_hurt", cx, feet + 4, "cb", right, False)
        if p.climbing:
            if shoot:
                return aiming("mm_climb_shoot", 8)[:2] + (feet + 4,) + aiming("mm_climb_shoot", 8)[3:]
            return (f"mm_climb{int(p.y / self.scale) // 8 % 2}", cx, feet + 4, "cb", False, False)
        if not p.grounded:
            return ("mm_jump_shoot" if shoot else "mm_jump", cx, feet + 4, "cb", right, False)
        if p.moving_frames > self.step_frames:
            i = (0, 1, 2, 1)[tick // 6 % 4]
            return aiming(f"mm_walk{i}_shoot", 12) if shoot else (f"mm_walk{i}", cx, feet, "cb", right, False)
        if shoot:
            return aiming("mm_idle_shoot", 12)
        name = "mm_inch" if p.moving_frames else "mm_blink" if tick % 128 < 8 else "mm_idle"
        return (name, cx, feet, "cb", right, False)

    def _enemy_sprite(self, e):
        k, tick = e.kind, self._steps
        right = e.facing > 0
        if k == "b":
            return (f"blader{tick // 4 % 2}", e.x, e.y + e.hh, "cb", right, False)
        if k in "kjKJ":                              # hangs off the wall behind it
            i = 0 if e.t < 82 else 1 if e.t < 89 or e.t >= 199 else 2 if e.t < 96 or e.t >= 192 else 3
            return (f"beak{i}", e.x - 8 if right else e.x + 8, e.y, "lc" if right else "rc", right, False)
        if k == "f":
            return ("flea_jump" if e.state == "air" else "flea_sit", e.x, e.y + e.hh, "cb", right, False)
        if k in "vh":
            i = 2 if e.fd == 0 else 1 if e.fd < 8 else 0
            return (f"octopus{i}", e.x, e.y, "cc", False, False)
        if k in "wW":                                # grows out of the floor below / ceiling above
            name = f"screw{tick // 4 % 4}" if e.state == "up" else "screw_shut"
            return (name, e.x, e.home[1], "cb" if k == "w" else "ct", False, k == "W")
        if k == "e":
            return ("bigeye_high" if e.state == "air" else "bigeye_low", e.x, e.y + e.hh, "cb", right, False)
        if k == "s":
            return ("shell_shut" if e.shielded else "shell_open", e.x, e.y + 8, "cb", False, False)
        if k == "cutter":
            return (f"cutter{tick // 4 % 2}", e.x, e.y, "cc", not right, False)
        if k == en.BOSS_GLYPH:                       # his sheet faces right
            left = e.look < 0
            bare = "bare_" if self.boss_cutter else ""
            if e.state in (15, 17, 19, 21, 23):
                name = "cut_throw" if e.anim < 16 and not bare else "cut_thrown"
            elif e.vy != en.REST_SPEED:
                name = f"cut_{bare}jump"
            elif e.state == 6:
                name = f"cut_{bare}walk{tick // 6 % 3}"
                left = not left                      # the walk frames are drawn facing the other way
            else:
                name = f"cut_{bare}idle{1 if tick % 64 < 8 else 0}"
            return (name, e.x, e.y + e.hh, "cb", left, False)
        return ("bullet", e.x, e.y, "cc", False, False)

    def _hud(self, surface) -> None:
        """The console's vertical energy meter, top left."""
        p, s = self.player, int(self.scale)
        x, y, w = 24 * s, 16 * s, 6 * s
        pygame.draw.rect(surface, self.PAL["hp_bg"], (x - s, y - s, w + 2 * s, p.hp_max * 2 * s + s))
        colour = self.PAL["goal"] if self.invincible else self.PAL["hp"]   # green meter: debug invincibility
        for i in range(p.hp):
            pygame.draw.rect(surface, colour, (x, y + (p.hp_max - 1 - i) * 2 * s, w, s))
        if self.boss:                                # the boss's meter stands beside it
            x += 16 * s
            pygame.draw.rect(surface, self.PAL["hp_bg"], (x - s, y - s, w + 2 * s, en.BOSS_HP * 2 * s + s))
            for i in range(max(0, self.boss.hp)):
                pygame.draw.rect(surface, self.PAL["boss_hp"], (x, y + (en.BOSS_HP - 1 - i) * 2 * s, w, s))


if __name__ == "__main__":  # smoke: movement numbers, ladders, screen flips, both deaths, a win
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    from code.games.megaman_core import MegamanCore, en, SPIKE, DOOR_SHUT   # one copy of the classes
    IDLE, RIGHT, LEFT, UP, DOWN, JUMP = (0, 0, 0, 0), (1, 0, 0, 0), (-1, 0, 0, 0), (0, -1, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0)

    def place(core, col, row):
        """Stand the player in a cell, feet on its bottom edge."""
        p, ts = core.player, core.tile_size
        p.x, p.y = col * ts + (ts - p.width) / 2.0, (row + 1) * ts - p.height
        p.vx = p.vy = 0.0
        p.grounded = p.climbing = False
        core.screen = (col // core.screen_cols, row // core.screen_rows)
        core.camera_x, core.camera_y = core._camera_target()

    def find(core, glyph):
        return [(c, r) for r, row in enumerate(core.level_data.grid) for c, ch in enumerate(row) if ch == glyph]

    core = MegamanCore(level="CutMan")
    p, ts, s = core.player, core.tile_size, core.scale
    y0 = p.y
    for _ in range(10):
        core.step(IDLE)
    assert p.grounded and p.y == y0, (p.grounded, p.y, y0)

    x0 = p.x                                         # 7 frames of inching, then 1.375 px/frame
    for _ in range(27):
        core.step(RIGHT)
    assert abs((p.x - x0) / s - (7 * 0.125 + 20 * 1.375)) < 1e-6, (p.x - x0) / s

    core = MegamanCore(level="CutMan")
    p = core.player
    core.step(IDLE)
    top = y0 = p.y
    for _ in range(60):
        core.step(JUMP)
        top = min(top, p.y)
    assert 48.0 < (y0 - top) / s < 51.0 and p.grounded, (y0 - top) / s       # full jump: about 3.1 tiles
    core.step(IDLE)
    top = p.y
    for i in range(60):
        core.step(JUMP if i < 3 else IDLE)
        top = min(top, p.y)
    assert (y0 - top) / s < 20.0, (y0 - top) / s                              # tap: a short hop

    core = MegamanCore(level="CutMan")                            # first ladder: down from the walkway, back up
    p = core.player
    place(core, 5, 125)
    core.step(IDLE)
    assert p.grounded, "the ladder top should hold the player up"
    for _ in range(400):
        core.step(DOWN)
    assert not p.climbing and p.grounded and p.y + p.height == 132 * ts, (p.climbing, p.y)
    for _ in range(400):
        core.step(UP)
    assert not p.climbing and p.grounded and p.y + p.height == 126 * ts, (p.climbing, p.y)

    core = MegamanCore(level="CutMan")                            # the ladder out of the first run of screens
    p = core.player
    place(core, 61, 122)
    core.step(UP)
    assert p.climbing and core.screen == (3, 8)
    for _ in range(200):
        core.step(UP)
    assert core.screen == (3, 7) and core.camera_y == 7 * core.HEIGHT, (core.screen, core.camera_y)
    assert p.grounded and not p.climbing             # stepped off the top rung, one screen up
    core.step(DOWN)                                  # back onto the ladder ...
    core.step(IDLE)
    core.step(JUMP)                                  # ... and let go: fall back into the screen below
    for _ in range(200):
        core.step(IDLE)
    assert core.alive and core.screen == (3, 8), (core.alive, core.screen)

    core = MegamanCore(level="CutMan")
    place(core, 42, 131)                             # over the first real pit
    for _ in range(120):
        core.step(IDLE)
    assert not core.alive and core.death_cause == "Pit", (core.alive, core.death_cause)

    core = MegamanCore(level="CutMan")
    c, r = find(core, SPIKE)[0]
    place(core, c, r - 2)
    for _ in range(120):
        core.step(IDLE)
    assert not core.alive and core.death_cause == "Spike", (core.alive, core.death_cause)

    core = MegamanCore(level="CutMan")                            # through the first shutter; it seals behind
    p = core.player
    doors = core.level_data.doors
    r, c = max(d for d in doors if d[1] == min(d[1] for d in doors))
    place(core, c - 2, r)
    for _ in range(120):
        core.step(RIGHT)
    assert p.x > (c + 1) * ts and core.tile(c, r) == DOOR_SHUT, (p.x, core.tile(c, r))
    for _ in range(240):
        core.step(LEFT)
    assert p.x == (c + 1) * ts, p.x
    assert core.camera_x == (c + 1) * ts, core.camera_x

    def spawn_cell(glyph):
        return next((c, r) for g, c, r in MegamanCore(level="CutMan").level_data.spawns if g == glyph)

    core = MegamanCore(level="CutMan")                            # a Beak: armoured shut, one shot kills it open
    p = core.player
    c, r = spawn_cell("k")                           # fires left; stand five tiles to its left
    place(core, c - 5, r + 3)
    core.step(IDLE)
    beak = next(e for e in core.enemies if e.kind == "k" and e.spawn >= 0 and (e.x // 16, e.y // 16) == (c, r))
    assert beak.shielded and core._hit_enemy(beak) and beak.hp == en.ENEMY_HP
    while beak.t != 100:
        core.step(IDLE)
    assert not beak.shielded and len(core.enemy_bullets) >= 1, (beak.t, core.enemy_bullets)
    core._hit_enemy(beak)
    assert beak not in core.enemies and core.kills_total == 1 and core.score >= 200

    core = MegamanCore(level="CutMan")                            # contact damage, knockback, then the blink
    p = core.player
    c, r = spawn_cell("v")
    place(core, c, r)
    hp = p.hp
    core.step(IDLE)
    core.step(IDLE)
    assert p.hp == hp - 4 and p.blink > 100 and p.knock > 0, (p.hp, p.blink, p.knock)
    for _ in range(60):
        core.step(RIGHT)
    assert p.hp == hp - 4 and p.knock == 0, (p.hp, p.knock)

    core = MegamanCore(level="CutMan", seed=7)                      # the boss room: intro, fight, the orb he leaves
    p = core.player
    bx, by = spawn_cell(en.BOSS_GLYPH)
    place(core, bx - 6, by)
    for _ in range(en.BOSS_INTRO_FRAMES + 2):
        core.step(RIGHT)
    assert core.boss and core.boss.hp == en.BOSS_HP and p.x < (bx - 5) * ts, (core.boss, p.x)
    states = set()
    for i in range(1500):
        states.add(core.boss.state)
        core.step(IDLE)
        if not core.alive:
            break
    assert len(states) >= 6, states                  # he walks, jumps and throws
    assert not core.alive and core.death_cause == "Cut Man", (core.alive, core.death_cause, p.hp)

    core = MegamanCore(level="CutMan", seed=7)
    p = core.player
    place(core, bx - 6, by)
    for _ in range(en.BOSS_INTRO_FRAMES + 2):
        core.step(IDLE)
    core.boss.hp = 3                                 # one buster shot from defeat
    for i in range(3000):
        if core.boss is None:
            break
        toward = 1 if core.boss.x * s > p.x else -1
        core.step((toward if i % 20 == 0 else 0, 0, 0, i % 4 == 0))
    assert core.boss_dead and core.alive, (core.boss_dead, core.alive, core.death_cause)
    orb = next(it for it in core.items if it.kind == "clear")
    for _ in range(600):
        _, _, done, _, _ = core.step((1 if orb.x * s > p.x + p.width / 2 else -1, 0, 0, 0))
        if done:
            break
    assert core.won and core.alive, (core.won, core.alive, core.death_cause)
    assert core.render().get_size() == (core.WIDTH, core.HEIGHT)
    print("megaman core ok")
