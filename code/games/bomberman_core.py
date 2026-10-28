"""Bomberman clone — single-screen arena, tile-aligned movement, bombs with chain blasts,
five NES-style enemy types, exit that opens when the arena is clear.

Plain class like MeatboyCore (no gym): the adapter drives it with step((dx, dy, bomb)),
reads player / level_data / bombs / enemies, and pulls `_obs()["grids"]` for grid sensing.
Deterministic: every random choice goes through `self.rng`, seeded at reset().
"""
from __future__ import annotations

import math
import os
import random
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pygame
import yaml

from . import mutants
from .modules.System.nes_font import nes_text

_HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(_HERE, "bomberman_config.yaml")

WALL, FLOOR, BRICK = "#", ".", "?"
EXIT, EXIT_HIDDEN = "G", "@"
# Bomberman II's items, each under a brick until bombed (upper case), then lying on the floor (lower)
POWERUPS = {"C": "bombs", "F": "range", "S": "speed", "R": "remote", "W": "wallpass", "X": "bombpass",
            "V": "vest", "D": "dollar", "Q": "mystery"}
REVEALED = {**{k: k.lower() for k in POWERUPS}, "@": "G"}    # what a bombed brick leaves behind
DESTRUCTIBLE = (BRICK, *POWERUPS, EXIT_HIDDEN)               # every tile a blast opens
ITEM_OF = {k.lower(): v for k, v in POWERUPS.items()}
ENEMY_GLYPHS = "EkKMBbtnpo"
DIRS = {0: (0, 0), 1: (0, -1), 2: (1, 0), 3: (0, 1), 4: (-1, 0)}  # idle, up, right, down, left


@dataclass
class LevelData:
    grid: list[list[str]]
    rows: int
    cols: int
    tile: int
    start: tuple[int, int]
    exit: tuple[int, int]
    exit_hidden: bool
    spawns: list[tuple[str, int, int]]

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
    speed: float = 112.0
    bombs_max: int = 1
    blast_range: int = 1
    facing: int = 2              # DIRS key, never 0
    on_ground: bool = True       # platformer adapters read this; always grounded top-down
    items: int = 0
    remote: bool = False         # bombs wait for the detonate button (B)
    wallpass: bool = False       # walks through bricks
    bombpass: bool = False       # walks through bombs
    vest: int = 0                # frames of immunity to blasts left


@dataclass
class Bomb:
    tx: int
    ty: int
    fuse: int
    blast_range: int
    passable: bool = True        # the player walks off the bomb they just dropped
    remote: bool = False         # waits for the detonate button instead of its fuse


@dataclass
class Enemy:
    kind: str
    x: float
    y: float
    speed: float
    passes_bricks: bool
    chase: float
    direction: int = 1
    alive: bool = True
    width: int = 24
    height: int = 24
    turn: float = 0.15           # chance of a new direction at a crossing (0: only when blocked)
    reverse: float = 0.0         # chance of turning back at a crossing (an erratic walker)
    chase_range: float = 99.0    # tiles: chases only when the player is this close
    teleport: int = 0            # frames between jumps to a random open tile (0: never)
    guard: int = 0               # frames it cannot be hurt (just spawned out of a door or an item)
    clock: int = 0


@dataclass
class Blast:
    cells: set
    frames: int
    pieces: dict = field(default_factory=dict)   # (x, y) -> flame piece to draw: c, h, v, or an end u/d/l/r
    opened: set = field(default_factory=set)     # bricks it set burning (drawn until the flames die)
    kills: int = 0                               # enemies it has killed: each further one is worth double


class BombermanCore:
    WIDTH, HEIGHT = 15 * 32, 13 * 32 + 32   # manual_play window: a 15 x 13 arena under the 32 px score band

    def __init__(self, render_mode: str = "none", max_steps=None, level_idx: int = 0, seed: int = 0) -> None:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        self.mutant = mutants.active("bomberman")  # PEAK_MUTANT: play that copy of the game
        mutants.apply_config(self.cfg, self.mutant)
        self.render_mode = render_mode
        self.tile_size = int(self.cfg["tile_size"])
        self.fps = int(self.cfg.get("fps", 60))
        self.dt = 1.0 / self.fps
        self.max_steps = int(max_steps) if max_steps else int(self.cfg.get("time_limit", 200) * self.fps)
        self.window = 11
        self._level_idx = level_idx
        self._seed = seed
        self.rng = random.Random(seed)
        self.won = False
        self.reset()

    # ── level ────────────────────────────────────────────────────────────────
    def level_entry(self, idx: int | None = None) -> dict:
        """A level is either a bare path or a dict of per-level overrides around one.
        Overrides: range (blast arms), enemy_speed (x multiplier), requires_clear (exit rule)."""
        levels = self.cfg["levels"]
        idx = self._level_idx if idx is None else idx
        if not 0 <= idx < len(levels):
            raise IndexError(f"bomberman level {idx} out of range (0..{len(levels) - 1})")
        entry = levels[idx]
        return dict(entry) if isinstance(entry, dict) else {"file": entry}

    def level_file(self, idx: int | None = None) -> str:
        file = self.level_entry(idx)["file"]
        return mutants.level_path(self.mutant, file) or os.path.join(_HERE, "levels", file)

    def _load_level(self) -> LevelData:
        with open(self.level_file(), encoding="utf-8") as f:
            rows = [r.rstrip("\n") for r in f if r.strip()]
        if not rows:
            raise ValueError(f"{self.level_file()}: empty level file")
        cols = max(len(r) for r in rows)
        grid = [list(r.ljust(cols, WALL)) for r in rows]
        start = exit_ = None
        hidden = False
        spawns = []
        for y, row in enumerate(grid):
            for x, ch in enumerate(row):
                if ch == "P":
                    start = (x, y); grid[y][x] = FLOOR
                elif ch == EXIT:
                    exit_ = (x, y)
                elif ch == EXIT_HIDDEN:
                    exit_ = (x, y); hidden = True
                elif ch in ENEMY_GLYPHS:
                    spawns.append((ch, x, y)); grid[y][x] = FLOOR
                elif ch == " ":
                    grid[y][x] = FLOOR
        if start is None or exit_ is None:
            raise ValueError(f"{self.level_file()}: needs one P and one G/@")
        return LevelData(grid, len(grid), cols, self.tile_size, start, exit_, hidden, spawns)

    def _make_enemy(self, k: str, x: int, y: int, guard: int = 0) -> Enemy:
        c, ts = self.cfg["enemies"][k], self.tile_size
        return Enemy(k, x * ts + 4, y * ts + 4, float(c["speed"]) * self._enemy_speed, bool(c["passes_bricks"]),
                     float(c["chase"]), direction=self.rng.choice([1, 2, 3, 4]), turn=float(c.get("turn", 0.15)),
                     reverse=float(c.get("reverse", 0.0)), chase_range=float(c.get("chase_range", 99)),
                     teleport=int(c.get("teleport", 0)), guard=guard)

    def next_level(self) -> int:
        """Where beating this level leads: the next area of the campaign it belongs to (Bomberman II's
        world 1), or the next level of the training ladder; both wrap."""
        camp = [int(i) for i in self.cfg.get("campaign") or []]
        if self._level_idx in camp:
            return camp[(camp.index(self._level_idx) + 1) % len(camp)]
        return (self._level_idx + 1) % len(self.cfg["levels"])

    def reset(self, *, seed=None, options=None):
        if self.won:  # beating a level advances to the next (wrapping); dying/timeout replays
            self._level_idx = self.next_level()
        if seed is not None:
            self._seed = seed
        self.rng = random.Random(self._seed)
        self.level_data = self._load_level()
        opts = self.level_entry()
        ts = self.tile_size
        pc = self.cfg["player"]
        size = int(pc.get("size", 24))
        sx, sy = self.level_data.start
        self.player = Player(sx * ts + (ts - size) / 2, sy * ts + (ts - size) / 2, size, size,
                             speed=float(pc.get("walk_speed", 112)), bombs_max=int(pc.get("bombs", 1)),
                             blast_range=int(opts.get("range", pc.get("range", 1))))
        self._enemy_speed = float(opts.get("enemy_speed", 1.0))
        self.enemies = [self._make_enemy(k, x, y) for k, x, y in self.level_data.spawns]
        # Some levels gate the exit on a cleared arena (killing is the lesson); others only ask
        # the player to reach it (the lesson is the maze and staying alive).
        self.requires_clear = bool(opts.get("requires_clear", True))
        # A bigger blast needs a longer walk to clear it: levels that raise `range` can buy the
        # retreat time back with `fuse`, or every bomb the agent drops is a suicide.
        self.fuse_frames = int(opts.get("fuse", self.cfg["bomb"]["fuse_frames"]))
        for k, v in ((options or {}).get("carry") or {}).items():   # what the player brings in (manual play)
            setattr(self.player, k, v)
        self.bombs: list[Bomb] = []
        self.blasts: list[Blast] = []
        self.intro = 0             # manual play: frames of the AREA card still to show (start_intro)
        self._detonator = False    # the caller has a detonate button (a 4th action value)
        self.score = 0
        self.coins_total = 0       # power-ups collected (the adapter's "coins")
        self.kills_total = 0
        self.bricks_destroyed = 0
        self.safe_detonations = 0    # bombs that went off while the player stood clear (learned retreat)
        self.useful_detonations = 0  # ...of those, the ones that opened a brick or landed on an enemy
        self.best_aim = 0.0        # closest a blast came to a living enemy (1 = hit); the aiming gradient
        self.timer = float(self.cfg.get("time_limit", 200))
        self.won = False
        self.alive = True
        self.death_cause = ""
        self._steps = 0
        self.dying: list[list] = []   # [x, y, kind, frames since, points] of enemies just killed (drawing only)
        self.extra_lives = 0          # lives won from items this attempt (manual play adds them)
        self._prev_detonate = False
        self.walked = 0.0             # px walked (picks the walk frame)
        self.death_t = 0              # frames of the death animation shown (manual play's death_tick)
        self._dist = self._distance_field()
        return self._obs(), self._info()

    # ── geometry ─────────────────────────────────────────────────────────────
    def tile(self, x: int, y: int) -> str:
        ld = self.level_data
        if 0 <= y < ld.rows and 0 <= x < ld.cols:
            return ld.grid[y][x]
        return WALL

    def bomb_at(self, x: int, y: int) -> Bomb | None:
        for b in self.bombs:
            if b.tx == x and b.ty == y:
                return b
        return None

    def solid(self, x: int, y: int, *, for_enemy: Enemy | None = None) -> bool:
        t = self.tile(x, y)
        if t == WALL:
            return True
        if t in DESTRUCTIBLE:
            return not (for_enemy is not None and for_enemy.passes_bricks)
        return False

    def _center_tile(self, x: float, y: float, w: int, h: int) -> tuple[int, int]:
        ts = self.tile_size
        return int((x + w / 2) // ts), int((y + h / 2) // ts)

    def _blocked(self, x: float, y: float, w: int, h: int, enemy: Enemy | None = None) -> bool:
        """Any solid tile (or bomb) under the hitbox at (x, y)?"""
        ts, p = self.tile_size, self.player
        x0, y0 = int(x // ts), int(y // ts)
        x1, y1 = int((x + w - 1) // ts), int((y + h - 1) // ts)
        for ty in range(y0, y1 + 1):
            for tx in range(x0, x1 + 1):
                if self.solid(tx, ty, for_enemy=enemy) and not (
                        enemy is None and p.wallpass and self.tile(tx, ty) in DESTRUCTIBLE):
                    return True
                b = self.bomb_at(tx, ty)
                if b is not None and not (enemy is None and (b.passable or p.bombpass)):
                    return True
        return False

    # ── player ───────────────────────────────────────────────────────────────
    def _move(self, dx: int, dy: int) -> None:
        p = self.player
        step = p.speed * self.dt
        ts = self.tile_size
        p.vx = p.vy = 0.0
        if dx:
            nx = p.x + dx * step
            if not self._blocked(nx, p.y, p.width, p.height):
                p.x = nx; p.vx = dx * p.speed
            else:  # corner cut: slide toward the open lane the hitbox already leans into
                cy = int((p.y + p.height / 2) // ts)
                ahead = int((p.x + (p.width if dx > 0 else -1) + dx * step) // ts)
                for lane, sgn in ((cy - 1, -1), (cy + 1, 1)):
                    if dy and sgn != dy:
                        continue  # never nudge against a direction the player is actively pushing
                    overlaps = (p.y < (lane + 1) * ts) if sgn < 0 else (p.y + p.height > lane * ts)
                    if overlaps and not self.solid(ahead, lane) and self.bomb_at(ahead, lane) is None:
                        target = lane * ts + (ts - p.height) / 2
                        ny = p.y + max(-step, min(step, target - p.y))
                        if not self._blocked(p.x, ny, p.width, p.height):
                            p.y = ny; p.vy = sgn * p.speed
                        break
        if dy and not (p.vx or p.vy):  # one axis per frame, as on the NES: vertical only when horizontal could not move
            ny = p.y + dy * step
            if not self._blocked(p.x, ny, p.width, p.height):
                p.y = ny; p.vy = dy * p.speed
            else:
                cx = int((p.x + p.width / 2) // ts)
                ahead = int((p.y + (p.height if dy > 0 else -1) + dy * step) // ts)
                for lane, sgn in ((cx - 1, -1), (cx + 1, 1)):
                    if dx and sgn != dx:
                        continue
                    overlaps = (p.x < (lane + 1) * ts) if sgn < 0 else (p.x + p.width > lane * ts)
                    if overlaps and not self.solid(lane, ahead) and self.bomb_at(lane, ahead) is None:
                        target = lane * ts + (ts - p.width) / 2
                        nx = p.x + max(-step, min(step, target - p.x))
                        if not self._blocked(nx, p.y, p.width, p.height):
                            p.x = nx; p.vx = sgn * p.speed
                        break
        if dx or dy:
            p.facing = {(0, -1): 1, (1, 0): 2, (0, 1): 3, (-1, 0): 4}.get((dx, dy), p.facing)

    def _drop_bomb(self) -> bool:
        p = self.player
        tx, ty = self._center_tile(p.x, p.y, p.width, p.height)
        if self.bomb_at(tx, ty) is not None or len(self.bombs) >= p.bombs_max:
            return False
        self.bombs.append(Bomb(tx, ty, self.fuse_frames, p.blast_range, remote=p.remote and self._detonator))
        return True

    # ── bombs & blasts ───────────────────────────────────────────────────────
    def blast_cells(self, bomb: Bomb) -> set:
        """Cells a bomb will cover: centre + four arms, stopped by walls, ending on the first brick."""
        cells = {(bomb.tx, bomb.ty)}
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            for r in range(1, bomb.blast_range + 1):
                x, y = bomb.tx + dx * r, bomb.ty + dy * r
                t = self.tile(x, y)
                if t == WALL:
                    break
                cells.add((x, y))
                if t in DESTRUCTIBLE:
                    break
        return cells

    def _flame_pieces(self, bomb: Bomb, pieces: dict, opened: set) -> None:
        """How the console draws one bomb's flames: a centre, arms, a cap where an arm ends, and a
        burning brick (no flame) where one stops it."""
        def put(cell, piece):
            have = pieces.get(cell)
            pieces[cell] = piece if have is None or have == piece else "c"
        put((bomb.tx, bomb.ty), "c")
        for (dx, dy), mid, cap in (((0, -1), "v", "u"), ((1, 0), "h", "r"), ((0, 1), "v", "d"), ((-1, 0), "h", "l")):
            arm = []
            for r in range(1, bomb.blast_range + 1):
                x, y = bomb.tx + dx * r, bomb.ty + dy * r
                t = self.tile(x, y)
                if t == WALL:
                    break
                if t in DESTRUCTIBLE:
                    opened.add((x, y))
                    break
                arm.append((x, y))
            for i, cell in enumerate(arm):
                put(cell, cap if i == len(arm) - 1 else mid)

    def _explode(self, first: Bomb) -> bool:
        """Detonate a bomb (chaining neighbours). True if the blast opened a brick or
        landed next to an enemy — an empty blast is just litter and earns nothing."""
        queue, done, cells, pieces, burnt = [first], set(), set(), {}, set()
        while queue:
            b = queue.pop()
            if id(b) in done:
                continue
            done.add(id(b))
            if b in self.bombs:
                self.bombs.remove(b)
            bc = self.blast_cells(b)
            cells |= bc
            self._flame_pieces(b, pieces, burnt)
            for other in list(self.bombs):  # chain reaction
                if (other.tx, other.ty) in bc:
                    queue.append(other)
        ld = self.level_data
        opened = 0
        for x, y in cells:                    # a blast over a lying item or the open door calls up enemies
            t = self.tile(x, y)
            if t in ITEM_OF:
                ld.grid[y][x] = FLOOR         # the item is lost, and a Pass comes out of it
                self._spawn("K", x, y, 1)
            elif t == EXIT and (x, y) == ld.exit:
                self._spawn("o", x, y, 4)     # four Pontans out of the door
        for x, y in cells:
            t = self.tile(x, y)
            if t in DESTRUCTIBLE:
                opened += 1
                ld.grid[y][x] = REVEALED.get(t, FLOOR)
                self.bricks_destroyed += 1
                self.score += int(self.cfg["scoring"]["brick"])
        alive = [self._center_tile(e.x, e.y, e.width, e.height) for e in self.enemies if e.alive]
        gap = None
        if alive:  # how near the blast came: a smooth gradient from "bombed nowhere" to "killed it"
            gap = min(abs(ex - cx) + abs(ey - cy) for cx, cy in cells for ex, ey in alive)
            self.best_aim = max(self.best_aim, 1.0 - min(gap, self.AIM_REACH) / self.AIM_REACH)
        self.blasts.append(Blast(cells, int(self.cfg["bomb"]["blast_frames"]), pieces, burnt))
        self._dist = self._distance_field()
        return opened > 0 or (gap is not None and gap <= 1)

    AIM_REACH = 4    # tiles: how far from an enemy a blast still counts as "aimed at it"

    def _spawn(self, kind: str, x: int, y: int, n: int) -> None:
        """Enemies called up by bombing the door or an item, unhurt by the flames that called them."""
        if kind not in self.cfg["enemies"]:
            return
        guard = int(self.cfg["bomb"]["blast_frames"]) + 30
        self.enemies += [self._make_enemy(kind, x, y, guard) for _ in range(n)]

    def danger_cells(self, lookahead: int | None = None) -> set:
        """Cells that are burning now or will be within `lookahead` frames (for sensing)."""
        la = int(self.cfg["bomb"].get("danger_lookahead", 30)) if lookahead is None else lookahead
        cells = set()
        for bl in self.blasts:
            cells |= bl.cells
        for b in self.bombs:
            if b.fuse <= la:
                cells |= self.blast_cells(b)
        return cells

    def time_to_boom(self, tx: int, ty: int) -> int | None:
        """Frames until (tx, ty) burns, None if no bomb reaches it. Burning now = 0."""
        best = None
        for bl in self.blasts:
            if (tx, ty) in bl.cells:
                return 0
        for b in self.bombs:
            if (tx, ty) in self.blast_cells(b):
                best = b.fuse if best is None else min(best, b.fuse)
        return best

    # ── enemies ──────────────────────────────────────────────────────────────
    def _enemy_step(self, e: Enemy) -> None:
        ts = self.tile_size
        step = e.speed * self.dt
        e.clock += 1
        if e.guard > 0:
            e.guard -= 1
        if e.teleport and e.clock % e.teleport == 0:      # Nagacham: vanishes and turns up elsewhere
            p = self.player
            ptx, pty = self._center_tile(p.x, p.y, p.width, p.height)
            spots = [(x, y) for y in range(self.level_data.rows) for x in range(self.level_data.cols)
                     if self.tile(x, y) == FLOOR and abs(x - ptx) + abs(y - pty) >= 4 and self.bomb_at(x, y) is None]
            if spots:
                x, y = self.rng.choice(spots)
                e.x, e.y = x * ts + (ts - e.width) / 2, y * ts + (ts - e.height) / 2
                return
        tx, ty = self._center_tile(e.x, e.y, e.width, e.height)
        cx, cy = tx * ts + (ts - e.width) / 2, ty * ts + (ts - e.height) / 2
        at_center = abs(e.x - cx) < step and abs(e.y - cy) < step
        if at_center:
            e.x, e.y = cx, cy
            free = [d for d, (dx, dy) in DIRS.items() if d and not self.solid(tx + dx, ty + dy, for_enemy=e)
                    and self.bomb_at(tx + dx, ty + dy) is None]
            p = self.player
            px, py = self._center_tile(p.x, p.y, p.width, p.height)
            near = abs(px - tx) + abs(py - ty) <= e.chase_range
            back = {1: 3, 2: 4, 3: 1, 4: 2}.get(e.direction)
            if free:
                if e.reverse and back in free and self.rng.random() < e.reverse:
                    e.direction = back                    # Boyon's back-and-forth
                elif near and self.rng.random() < e.chase:
                    prefer = ([2 if px > tx else 4] if abs(px - tx) >= abs(py - ty) else [3 if py > ty else 1])
                    prefer += [3 if py > ty else 1] if abs(px - tx) >= abs(py - ty) else [2 if px > tx else 4]
                    pick = next((d for d in prefer if d in free), None)
                    e.direction = pick if pick is not None else self.rng.choice(free)
                elif e.direction not in free or self.rng.random() < e.turn:
                    e.direction = self.rng.choice(free)
            else:
                e.direction = 0
        dx, dy = DIRS[e.direction]
        nx, ny = e.x + dx * step, e.y + dy * step
        if not self._blocked(nx, ny, e.width, e.height, enemy=e):
            e.x, e.y = nx, ny
        else:
            e.x, e.y = cx, cy
            e.direction = 0

    # ── step ─────────────────────────────────────────────────────────────────
    def step(self, action):
        """action = (dx, dy, bomb) with dx, dy in {-1, 0, 1}; returns (obs, reward, terminated, truncated, info)."""
        if not self.alive or self.won:
            return self._obs(), 0.0, True, False, self._info()
        if self.intro > 0:                       # the AREA card: nothing moves yet
            self.intro -= 1
            return self._obs(), 0.0, False, False, self._info()
        dx, dy, bomb = int(action[0]), int(action[1]), bool(action[2])
        self._detonator = len(action) > 3
        detonate = self._detonator and bool(action[3]) and not self._prev_detonate
        self._prev_detonate = self._detonator and bool(action[3])
        self._steps += 1
        self.timer = max(0.0, self.timer - self.dt)
        if bomb:
            self._drop_bomb()
        self._move(dx, dy)
        p = self.player
        self.walked += abs(p.vx or p.vy) * self.dt
        for d in self.dying:
            d[3] += 1
        self.dying = [d for d in self.dying if d[3] < 90]
        ptx, pty = self._center_tile(p.x, p.y, p.width, p.height)
        ts = self.tile_size
        for b in self.bombs:  # a bomb becomes solid once the player's hitbox has fully left its tile
            if b.passable and not (p.x < (b.tx + 1) * ts and p.x + p.width > b.tx * ts
                                   and p.y < (b.ty + 1) * ts and p.y + p.height > b.ty * ts):
                b.passable = False
        exploded = useful = 0
        if detonate:                             # the remote: B sets off the oldest waiting bomb
            waiting = [b for b in self.bombs if b.remote]
            if waiting:
                exploded += 1
                useful += int(self._explode(waiting[0]))
        if p.vest > 0:
            p.vest -= 1
        for b in list(self.bombs):
            if b.remote:
                continue
            b.fuse -= 1
            if b.fuse <= 0 and b in self.bombs:
                exploded += 1
                useful += int(self._explode(b))
        for bl in list(self.blasts):
            bl.frames -= 1
            if bl.frames <= 0:
                self.blasts.remove(bl)
        burning = set()
        for bl in self.blasts:
            burning |= bl.cells
        # enemies move, burn, and bite
        for e in self.enemies:
            if not e.alive:
                continue
            self._enemy_step(e)
            etx, ety = self._center_tile(e.x, e.y, e.width, e.height)
            if (etx, ety) in burning and e.guard <= 0:
                e.alive = False
                self.kills_total += 1
                bl = next(b for b in self.blasts if (etx, ety) in b.cells)
                points = int(self.cfg["scoring"]["enemy"].get(e.kind, 100)) * 2 ** bl.kills   # each further kill
                bl.kills += 1                                                                    # of one blast: x2
                self.score += points
                self.dying.append([e.x, e.y, e.kind, 0, points])   # drawn: shocked, then the points
                continue
            if (abs((e.x + e.width / 2) - (p.x + p.width / 2)) < (e.width + p.width) / 2 - 6
                    and abs((e.y + e.height / 2) - (p.y + p.height / 2)) < (e.height + p.height) / 2 - 6):
                self._die("Enemy")
        # pick-ups
        t = self.tile(ptx, pty)
        if t in ITEM_OF:
            self._pickup(ITEM_OF[t])
            self.level_data.grid[pty][ptx] = FLOOR
        if (ptx, pty) in burning and self.alive and p.vest <= 0:
            self._die("Bomb")
        elif exploded and self.alive:
            self.safe_detonations += exploded
            self.useful_detonations += useful
        if self.alive and self.timer <= 0:
            self._die("Timeout")
        if self.alive and (ptx, pty) == self.level_data.exit and self.tile(ptx, pty) == EXIT \
                and self.arena_clear:
            self.won = True
            self.score += int(self.cfg["scoring"].get("clear", 0))
        if self._steps >= self.max_steps and self.alive and not self.won:
            self._die("Timeout")
        done = (not self.alive) or self.won
        return self._obs(), 0.0, done, False, self._info()

    def _die(self, cause: str) -> None:
        self.alive = False
        self.death_cause = cause

    def _pickup(self, item: str) -> None:
        p = self.player
        self.coins_total += 1
        p.items += 1
        if item == "mystery":                      # "?": a life, a fire suit, or points
            item = self.rng.choice(["life", "vest", "dollar"])
        if item == "bombs":
            p.bombs_max += 1
        elif item == "range":
            p.blast_range += 1
        elif item == "speed":
            p.speed = float(self.cfg["player"].get("sprint_speed", 168))
        elif item in ("remote", "wallpass", "bombpass"):
            setattr(p, item, True)
        elif item == "vest":
            p.vest = int(self.cfg["player"].get("vest_frames", 30 * self.fps))
        elif item == "dollar":
            self.score += self.rng.choice([100, 200, 400, 800, 1600, 3200, 6400])
        elif item == "life":
            self.extra_lives += 1                  # manual play reads it and adds a life

    # ── the console's area card and death, for manual play (agents never call these) ──
    def start_intro(self, frames: int = 190) -> None:
        """The black "AREA x-x / LEFT n" card before play starts (3.15 s on the console)."""
        self.intro = frames

    def area_name(self) -> str:
        return str(self.level_entry().get("name", self._level_idx + 1))

    def death_tick(self) -> bool:
        """One frame of the death animation; False once it is over."""
        self.death_t += 1
        return self.death_t < 75

    def carry(self, died: bool) -> dict:
        """What the player keeps into the next life or area: bombs and fire always, the rest only
        while alive (Bomberman II's manual)."""
        p = self.player
        keep = {"bombs_max": p.bombs_max, "blast_range": p.blast_range}
        if not died:
            keep.update(speed=p.speed, remote=p.remote, wallpass=p.wallpass, bombpass=p.bombpass)
        return keep

    # ── progress: Dijkstra to the exit, bricks cost extra ─────────────────────
    BRICK_COST = 6

    def _distance_field(self) -> np.ndarray:
        """Cost-to-exit per tile (floor 1, brick BRICK_COST, wall ∞) — bombing a brick on the
        way shortens it, so destroying the right brick is progress."""
        ld = self.level_data
        dist = np.full((ld.rows, ld.cols), np.inf)
        ex, ey = ld.exit
        dist[ey, ex] = 0.0
        frontier = deque([(ex, ey)])
        while frontier:
            x, y = frontier.popleft()
            for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                nx, ny = x + dx, y + dy
                t = self.tile(nx, ny)
                if t == WALL:
                    continue
                cost = self.BRICK_COST if t in DESTRUCTIBLE else 1.0
                nd = dist[y, x] + cost
                if nd < dist[ny, nx]:
                    dist[ny, nx] = nd
                    frontier.append((nx, ny))
        return dist

    @property
    def arena_clear(self) -> bool:
        """Is the exit live? Every enemy dead, or this level does not gate on that."""
        return not self.requires_clear or all(not e.alive for e in self.enemies)

    def goal_cost(self) -> float:
        p = self.player
        tx, ty = self._center_tile(p.x, p.y, p.width, p.height)
        d = self._dist[ty, tx]
        return float(d) if np.isfinite(d) else float(self.level_data.rows * self.level_data.cols)

    def start_cost(self) -> float:
        sx, sy = self.level_data.start
        d = self._distance_field()[sy, sx]  # pristine field; self._dist changes as bricks go
        return float(d) if np.isfinite(d) else 1.0

    # ── observation: 3 × window × window around the player ───────────────────
    def _obs(self):
        W = self.window
        half = W // 2
        p = self.player
        px, py = self._center_tile(p.x, p.y, p.width, p.height)
        out = np.zeros((3, W, W), dtype=np.float32)
        danger = self.danger_cells()
        enemy_cells = {self._center_tile(e.x, e.y, e.width, e.height) for e in self.enemies if e.alive}
        for j in range(W):
            for i in range(W):
                gx, gy = px - half + i, py - half + j
                t = self.tile(gx, gy)
                if t == WALL or t in DESTRUCTIBLE or self.bomb_at(gx, gy) is not None:
                    out[0, j, i] = 1.0
                if t in ITEM_OF or t == EXIT:
                    out[1, j, i] = 1.0
                if (gx, gy) in danger or (gx, gy) in enemy_cells:
                    out[2, j, i] = -1.0
        return {"grids": out, "scalars": np.asarray([p.vx / p.speed, p.vy / p.speed], dtype=np.float32)}

    def _info(self):
        return {"score": self.score, "won": self.won, "alive": self.alive, "steps": self._steps,
                "bricks": self.bricks_destroyed, "kills": self.kills_total}

    # ── rendering: Bomberman II's look (bomberman_art) ──────────────────────
    HUD_H = 32           # the black score band above the arena (human mode only)
    hud_lives = None     # lives left to show as LEFT (manual play sets it; None hides it)

    def _frame_cells(self) -> set:
        """Wall tiles that belong to the metal frame: on the border, or in a row or column that is
        wall end to end (the old ladder levels have a double border). Any other wall is a pillar."""
        ld = self.level_data
        full_rows = {y for y in range(ld.rows) if all(t == WALL for t in ld.grid[y])}
        full_cols = {x for x in range(ld.cols) if all(ld.grid[y][x] == WALL for y in range(ld.rows))}
        return {(x, y) for y in range(ld.rows) for x in range(ld.cols) if ld.grid[y][x] == WALL and
                (x in (0, ld.cols - 1) or y in (0, ld.rows - 1) or y in full_rows or x in full_cols)}

    def render(self, surface=None, blit_only: bool = False):
        from . import bomberman_art as A
        ld, ts = self.level_data, self.tile_size
        oy = self.HUD_H if self.render_mode == "human" else 0
        if surface is None:
            surface = pygame.Surface((int(ld.width), int(ld.height) + oy))
        surface.fill((0, 0, 0))
        if self.intro > 0:                                    # the console's card between areas
            w, h = surface.get_size()
            lines = [(f"SC {self.score:08d}", 2, (248, 248, 248)) if self.score else None,
                     ("AREA", 4, (248, 144, 0)), (self.area_name(), 4, (248, 248, 248)),
                     (f"LEFT {'--' if self.hud_lives is None else max(0, self.hud_lives)}", 2, (248, 248, 248))]
            y = h // 2 - 110
            for line in lines:
                if line:
                    img = nes_text(*line)
                    surface.blit(img, ((w - img.get_width()) // 2, y))
                    y += img.get_height() + 28
                else:
                    y += 42
            return surface
        if getattr(self, "_frames_of", None) is not ld:          # the frame, worked out once per level
            self._frames_of, self._frame = ld, self._frame_cells()
        frame, tick = self._frame, self._steps

        def at(img, x, y):
            surface.blit(img, (x, y + oy))

        burning_bricks, pieces, life = set(), {}, {}
        for bl in self.blasts:
            burning_bricks |= bl.opened
            for cell, pc in bl.pieces.items():
                pieces[cell] = pc
                life[cell] = bl.frames
        for y in range(ld.rows):
            for x in range(ld.cols):
                t, px, py = ld.grid[y][x], x * ts, y * ts
                if (x, y) in frame:
                    open_ = [0 <= x + dx < ld.cols and 0 <= y + dy < ld.rows and (x + dx, y + dy) not in frame
                             for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1))]
                    at(A.get(A.frame_tile(open_[1], open_[0], open_[3], open_[2], x % 4 == 2)), px, py)
                    continue
                pygame.draw.rect(surface, A.FLOOR, (px, py + oy, ts, ts))
                if t == WALL:
                    at(A.get(A.PILLAR), px, py)
                elif t in DESTRUCTIBLE:
                    at(A.get(A.BRICK), px, py)
                elif t == EXIT:
                    at(A.get(A.DOOR), px, py)
                elif t in A.ITEM_ICONS:
                    at(A.get(A.ITEM_ICONS[t]), px, py)
                if (x, y) in burning_bricks:
                    at(A.get(A.BURNING_BRICK[(tick // 4) % 2]), px, py)
        for (x, y), pc in pieces.items():
            n = life[(x, y)]
            level = 2 if 6 <= n <= 22 else 1 if 3 <= n <= 25 else 0      # the flames grow, hold, and fade
            at(A.get(A._flame(pc, level)), x * ts, y * ts)
        for b in self.bombs:
            at(A.get(A.BOMB[(b.fuse // 10) % 2]), b.tx * ts, b.ty * ts)
        for e in self.enemies:
            if e.alive:
                frames = A.ENEMIES.get(e.kind, A.BALLOON)
                img = A.get(frames[(tick // 12) % len(frames)], e.direction == 4)
                at(img, int(e.x + e.width / 2 - img.get_width() / 2), int(e.y + e.height - img.get_height() + 2))
        for x, y, kind, t, points in self.dying:        # shocked, then the points it was worth
            if t < 40:
                img = A.get(A.ENEMY_DEATH.get(kind, A.DEATH_FACE))
                at(img, int(x + 12 - img.get_width() / 2), int(y + 24 - img.get_height() + 2))
            else:
                img = nes_text(str(points), 1, (255, 255, 255))
                at(img, int(x + 12 - img.get_width() / 2), int(y + 6))
        p = self.player
        cx, feet = p.x + p.width / 2, p.y + p.height
        if self.alive or self.won:
            moving = bool(p.vx or p.vy)
            step = (0, 1, 0, 2)[int(self.walked // 12) % 4] if moving else 0
            img = A.get(A.PLAYER[(p.facing, step)])
            at(img, int(cx - img.get_width() / 2), int(feet - img.get_height() + 4))
        elif self.death_t:
            img = A.get(A.PLAYER_DEATH[min(len(A.PLAYER_DEATH) - 1, self.death_t // 15)])
            at(img, int(cx - img.get_width() / 2), int(feet - img.get_height() + 4))
        if self.render_mode != "none":
            self._hud(surface)
        return surface

    def _hud(self, surface) -> None:
        """Bomberman II's band above the arena: the 8-digit score, T: the clock, LEFT the lives."""
        w = surface.get_width()
        pygame.draw.rect(surface, (0, 0, 0), (0, 0, w, self.HUD_H))
        left = "--" if self.hud_lives is None else str(max(0, self.hud_lives))
        cells = [f"{min(self.score, 99_999_999):08d}", f"T:{max(0, math.ceil(self.timer)):03d}", f"LEFT {left}"]
        imgs = [nes_text(c, 2, (248, 248, 248)) for c in cells]
        gap = (w - 32 - sum(i.get_width() for i in imgs)) // (len(imgs) - 1)
        x = 16
        for img in imgs:
            surface.blit(img, (x, (self.HUD_H - img.get_height()) // 2))
            x += img.get_width() + gap


if __name__ == "__main__":  # smoke: walk level 0 to the exit with a scripted path
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    core = BombermanCore(level_idx=0)
    for _ in range(60 * 6):
        core.step((1, 0, 0))
    for _ in range(60 * 6):
        core.step((0, 1, 0))
    assert core.won, (core.alive, core.death_cause, core.player.x, core.player.y)
    core = BombermanCore(level_idx=3)   # 04_one_wall: a brick wall at x = 7, player starts on row 1
    for _ in range(60 * 2):
        core.step((1, 0, 0))            # walk right until the brick stops us
    assert not core.won and core.alive
    core.step((0, 0, 1))                # drop a bomb against that brick
    for _ in range(60 * 2):
        core.step((-1, 0, 0))           # run away
    for _ in range(200):
        core.step((0, 0, 0))
    assert core.alive and core.bricks_destroyed >= 1, (core.alive, core.death_cause, core.bricks_destroyed)

    core = BombermanCore(level_idx=9)   # per-level overrides: 10_brick_maze is a maze, not a kill list
    assert not core.requires_clear, core.level_entry()
    assert core.enemies and all(e.speed < 48 for e in core.enemies), [e.speed for e in core.enemies]
    assert BombermanCore(level_idx=8).requires_clear                # 09_two_balloms is an arena: gated
    assert core.fuse_frames == 150, core.fuse_frames

    # Bomberman II, area 1-1: the item under its brick, an item bombed into a Pass, double points
    core = BombermanCore(level_idx=12)
    assert core.area_name() == "1-1" and [e.kind for e in core.enemies] == ["E"] * 3
    ix, iy = 9, 8                                      # Fire Up under the brick at (9, 8)
    assert core.tile(ix, iy) == "F"
    core.level_data.grid[iy][ix] = "f"                 # as if bombed open
    core.player.x, core.player.y = ix * 32 + 4, iy * 32 + 4
    core.enemies = []
    core.step((0, 0, 0))
    assert core.player.blast_range == 2 and core.tile(ix, iy) == ".", (core.player.blast_range, core.tile(ix, iy))
    core.level_data.grid[iy][ix] = "f"                 # a lying item caught by a blast becomes a Pass
    core.player.x, core.player.y = 1 * 32 + 4, 1 * 32 + 4
    core._explode(Bomb(ix - 1, iy, 0, 2))
    assert core.tile(ix, iy) == "." and [e.kind for e in core.enemies] == ["K"], [e.kind for e in core.enemies]
    core = BombermanCore(level_idx=12)                 # two Balloms in one blast: 100, then 200
    for e, (x, y) in zip(core.enemies, ((5, 1), (6, 1))):
        e.x, e.y, e.speed = x * 32 + 4, y * 32 + 4, 0.0
    core.enemies[2].alive = False
    core._explode(Bomb(4, 1, 0, 3))
    core.step((0, 0, 0))
    assert core.score == 300 and core.kills_total == 2, (core.score, core.kills_total)
    core = BombermanCore(level_idx=15)                 # 1-4's Remote: bombs wait for B
    core.player.remote = True
    core.step((0, 0, 1, 0))
    for _ in range(300):
        core.step((0, 0, 0, 0))
    assert len(core.bombs) == 1 and core.bombs[0].remote
    core.step((0, 0, 0, 1))
    assert not core.bombs and core.blasts
    print("bomberman core ok")
