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

_HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(_HERE, "bomberman_config.yaml")

WALL, FLOOR, BRICK = "#", ".", "?"
EXIT, EXIT_HIDDEN = "G", "@"
POWERUPS = {"C": "bombs", "F": "range", "S": "speed"}       # under a brick until bombed
REVEALED = {"C": "c", "F": "f", "S": "s", "@": "G"}          # what a bombed brick leaves behind
ITEM_OF = {"c": "bombs", "f": "range", "s": "speed"}
ENEMY_GLYPHS = "EkKMB"
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


@dataclass
class Bomb:
    tx: int
    ty: int
    fuse: int
    blast_range: int
    passable: bool = True        # the player walks off the bomb they just dropped


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


@dataclass
class Blast:
    cells: set
    frames: int


class BombermanCore:
    WIDTH, HEIGHT = 15 * 32, 13 * 32  # manual_play window (every authored level is 15 × 13)

    def __init__(self, render_mode: str = "none", max_steps=None, level_idx: int = 0, seed: int = 0) -> None:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        self.render_mode = render_mode
        self.tile_size = int(self.cfg["tile_size"])
        self.fps = int(self.cfg.get("fps", 60))
        self.dt = 1.0 / self.fps
        self.max_steps = int(max_steps) if max_steps else int(self.cfg.get("time_limit", 200) * self.fps)
        self.window = 11
        self._level_idx = level_idx
        self._seed = seed
        self.rng = random.Random(seed)
        self._font = None
        self.reset()

    # ── level ────────────────────────────────────────────────────────────────
    def level_file(self, idx: int | None = None) -> str:
        levels = self.cfg["levels"]
        idx = self._level_idx if idx is None else idx
        if not 0 <= idx < len(levels):
            raise IndexError(f"bomberman level {idx} out of range (0..{len(levels) - 1})")
        return os.path.join(_HERE, levels[idx]["file"])

    def _load_level(self) -> LevelData:
        with open(self.level_file(), encoding="utf-8") as f:
            rows = [r.rstrip("\n") for r in f if r.strip()]
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

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._seed = seed
        self.rng = random.Random(self._seed)
        self.level_data = self._load_level()
        ts = self.tile_size
        pc = self.cfg["player"]
        size = int(pc.get("size", 24))
        sx, sy = self.level_data.start
        self.player = Player(sx * ts + (ts - size) / 2, sy * ts + (ts - size) / 2, size, size,
                             speed=float(pc.get("walk_speed", 112)), bombs_max=int(pc.get("bombs", 1)),
                             blast_range=int(pc.get("range", 1)))
        ec = self.cfg["enemies"]
        self.enemies = [Enemy(k, x * ts + 4, y * ts + 4, float(ec[k]["speed"]), bool(ec[k]["passes_bricks"]),
                              float(ec[k]["chase"]), direction=self.rng.choice([1, 2, 3, 4]))
                        for k, x, y in self.level_data.spawns]
        self.bombs: list[Bomb] = []
        self.blasts: list[Blast] = []
        self.score = 0
        self.coins_total = 0       # power-ups collected (the adapter's "coins")
        self.kills_total = 0
        self.bricks_destroyed = 0
        self.timer = float(self.cfg.get("time_limit", 200))
        self.won = False
        self.alive = True
        self.death_cause = ""
        self._steps = 0
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
        if t in (BRICK, "C", "F", "S", EXIT_HIDDEN):
            return not (for_enemy is not None and for_enemy.passes_bricks)
        return False

    def _center_tile(self, x: float, y: float, w: int, h: int) -> tuple[int, int]:
        ts = self.tile_size
        return int((x + w / 2) // ts), int((y + h / 2) // ts)

    def _blocked(self, x: float, y: float, w: int, h: int, enemy: Enemy | None = None) -> bool:
        """Any solid tile (or bomb) under the hitbox at (x, y)?"""
        ts = self.tile_size
        x0, y0 = int(x // ts), int(y // ts)
        x1, y1 = int((x + w - 1) // ts), int((y + h - 1) // ts)
        for ty in range(y0, y1 + 1):
            for tx in range(x0, x1 + 1):
                if self.solid(tx, ty, for_enemy=enemy):
                    return True
                b = self.bomb_at(tx, ty)
                if b is not None and not (enemy is None and b.passable):
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
                    overlaps = (p.y < (lane + 1) * ts) if sgn < 0 else (p.y + p.height > lane * ts)
                    if overlaps and not self.solid(ahead, lane) and self.bomb_at(ahead, lane) is None:
                        target = lane * ts + (ts - p.height) / 2
                        ny = p.y + max(-step, min(step, target - p.y))
                        if not self._blocked(p.x, ny, p.width, p.height):
                            p.y = ny; p.vy = sgn * p.speed
                        break
        if dy:
            ny = p.y + dy * step
            if not self._blocked(p.x, ny, p.width, p.height):
                p.y = ny; p.vy = dy * p.speed
            else:
                cx = int((p.x + p.width / 2) // ts)
                ahead = int((p.y + (p.height if dy > 0 else -1) + dy * step) // ts)
                for lane, sgn in ((cx - 1, -1), (cx + 1, 1)):
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
        self.bombs.append(Bomb(tx, ty, int(self.cfg["bomb"]["fuse_frames"]), p.blast_range))
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
                if t in (BRICK, "C", "F", "S", EXIT_HIDDEN):
                    break
        return cells

    def _explode(self, first: Bomb) -> None:
        queue, done, cells = [first], set(), set()
        while queue:
            b = queue.pop()
            if id(b) in done:
                continue
            done.add(id(b))
            if b in self.bombs:
                self.bombs.remove(b)
            bc = self.blast_cells(b)
            cells |= bc
            for other in list(self.bombs):  # chain reaction
                if (other.tx, other.ty) in bc:
                    queue.append(other)
        ld = self.level_data
        for x, y in cells:
            t = self.tile(x, y)
            if t in (BRICK, "C", "F", "S", EXIT_HIDDEN):
                ld.grid[y][x] = REVEALED.get(t, FLOOR)
                self.bricks_destroyed += 1
                self.score += int(self.cfg["scoring"]["brick"])
        self.blasts.append(Blast(cells, int(self.cfg["bomb"]["blast_frames"])))
        self._dist = self._distance_field()

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
        tx, ty = self._center_tile(e.x, e.y, e.width, e.height)
        cx, cy = tx * ts + (ts - e.width) / 2, ty * ts + (ts - e.height) / 2
        at_center = abs(e.x - cx) < step and abs(e.y - cy) < step
        if at_center:
            e.x, e.y = cx, cy
            free = [d for d, (dx, dy) in DIRS.items() if d and not self.solid(tx + dx, ty + dy, for_enemy=e)
                    and self.bomb_at(tx + dx, ty + dy) is None]
            if free:
                if self.rng.random() < e.chase:
                    p = self.player
                    px, py = self._center_tile(p.x, p.y, p.width, p.height)
                    prefer = ([2 if px > tx else 4] if abs(px - tx) >= abs(py - ty) else [3 if py > ty else 1])
                    prefer += [3 if py > ty else 1] if abs(px - tx) >= abs(py - ty) else [2 if px > tx else 4]
                    pick = next((d for d in prefer if d in free), None)
                    e.direction = pick if pick is not None else self.rng.choice(free)
                elif e.direction not in free or self.rng.random() < 0.15:
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
        dx, dy, bomb = int(action[0]), int(action[1]), bool(action[2])
        self._steps += 1
        self.timer = max(0.0, self.timer - self.dt)
        if bomb:
            self._drop_bomb()
        self._move(dx, dy)
        p = self.player
        ptx, pty = self._center_tile(p.x, p.y, p.width, p.height)
        ts = self.tile_size
        for b in self.bombs:  # a bomb becomes solid once the player's hitbox has fully left its tile
            if b.passable and not (p.x < (b.tx + 1) * ts and p.x + p.width > b.tx * ts
                                   and p.y < (b.ty + 1) * ts and p.y + p.height > b.ty * ts):
                b.passable = False
        for b in list(self.bombs):
            b.fuse -= 1
            if b.fuse <= 0 and b in self.bombs:
                self._explode(b)
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
            if (etx, ety) in burning:
                e.alive = False
                self.kills_total += 1
                self.score += int(self.cfg["scoring"]["enemy"].get(e.kind, 100))
                continue
            if (abs((e.x + e.width / 2) - (p.x + p.width / 2)) < (e.width + p.width) / 2 - 6
                    and abs((e.y + e.height / 2) - (p.y + p.height / 2)) < (e.height + p.height) / 2 - 6):
                self._die("Enemy")
        # pick-ups
        t = self.tile(ptx, pty)
        if t in ITEM_OF:
            self._pickup(ITEM_OF[t])
            self.level_data.grid[pty][ptx] = FLOOR
        if (ptx, pty) in burning and self.alive:
            self._die("Bomb")
        if self.alive and self.timer <= 0:
            self._die("Timeout")
        if self.alive and (ptx, pty) == self.level_data.exit and self.tile(ptx, pty) == EXIT \
                and all(not e.alive for e in self.enemies):
            self.won = True
            self.score += 1000
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
        self.score += 50
        if item == "bombs":
            p.bombs_max += 1
        elif item == "range":
            p.blast_range += 1
        elif item == "speed":
            p.speed = float(self.cfg["player"].get("sprint_speed", 168))

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
                cost = self.BRICK_COST if t in (BRICK, "C", "F", "S", EXIT_HIDDEN) else 1.0
                nd = dist[y, x] + cost
                if nd < dist[ny, nx]:
                    dist[ny, nx] = nd
                    frontier.append((nx, ny))
        return dist

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
                if t == WALL or t in (BRICK, "C", "F", "S", EXIT_HIDDEN) or self.bomb_at(gx, gy) is not None:
                    out[0, j, i] = 1.0
                if t in ITEM_OF or t == EXIT:
                    out[1, j, i] = 1.0
                if (gx, gy) in danger or (gx, gy) in enemy_cells:
                    out[2, j, i] = -1.0
        return {"grids": out, "scalars": np.asarray([p.vx / p.speed, p.vy / p.speed], dtype=np.float32)}

    def _info(self):
        return {"score": self.score, "won": self.won, "alive": self.alive, "steps": self._steps,
                "bricks": self.bricks_destroyed, "kills": self.kills_total}

    # ── rendering ────────────────────────────────────────────────────────────
    PAL = {"floor": (64, 140, 64), "floor2": (58, 130, 58), "wall": (120, 124, 128), "wall_hi": (160, 164, 168),
           "wall_lo": (72, 76, 80), "brick": (190, 110, 50), "brick_lo": (120, 64, 28), "brick_hi": (224, 150, 90),
           "exit": (30, 30, 40), "exit_hi": (250, 230, 120), "bomb": (24, 24, 28), "fuse": (255, 200, 60),
           "fire": (255, 120, 30), "fire_core": (255, 240, 160), "player": (245, 245, 250), "player_acc": (60, 120, 255),
           "E": (245, 245, 255), "k": (80, 180, 255), "K": (255, 90, 90), "M": (255, 150, 220), "B": (170, 120, 255),
           "eye": (20, 20, 30)}

    def render(self, surface=None, blit_only: bool = False):
        ld, ts = self.level_data, self.tile_size
        if surface is None:
            surface = pygame.Surface((int(ld.width), int(ld.height)))
        vw, vh = surface.get_size()
        cols, rows = min(ld.cols, vw // ts + 1), min(ld.rows, vh // ts + 1)
        burning = set()
        for bl in self.blasts:
            burning |= bl.cells
        for y in range(rows):
            for x in range(cols):
                r = pygame.Rect(x * ts, y * ts, ts, ts)
                t = ld.grid[y][x]
                pygame.draw.rect(surface, self.PAL["floor" if (x + y) % 2 else "floor2"], r)
                if t == WALL:
                    pygame.draw.rect(surface, self.PAL["wall"], r)
                    pygame.draw.line(surface, self.PAL["wall_hi"], r.topleft, r.topright, 3)
                    pygame.draw.line(surface, self.PAL["wall_hi"], r.topleft, r.bottomleft, 3)
                    pygame.draw.line(surface, self.PAL["wall_lo"], r.bottomleft, r.bottomright, 3)
                    pygame.draw.line(surface, self.PAL["wall_lo"], r.topright, r.bottomright, 3)
                elif t in (BRICK, "C", "F", "S", EXIT_HIDDEN):
                    pygame.draw.rect(surface, self.PAL["brick"], r)
                    for k in range(4):  # mortar lines, offset every other course
                        yy = r.top + k * ts // 4
                        pygame.draw.line(surface, self.PAL["brick_lo"], (r.left, yy), (r.right, yy), 2)
                        xx = r.left + (ts // 2 if k % 2 else ts // 4)
                        pygame.draw.line(surface, self.PAL["brick_lo"], (xx, yy), (xx, yy + ts // 4), 2)
                    pygame.draw.line(surface, self.PAL["brick_hi"], r.topleft, r.topright, 1)
                elif t == EXIT:
                    pygame.draw.rect(surface, self.PAL["exit"], r.inflate(-6, -6), border_radius=4)
                    if all(not e.alive for e in self.enemies):
                        glow = 120 + int(60 * math.sin(self._steps / 8))
                        pygame.draw.rect(surface, (glow, glow - 30, 40), r.inflate(-14, -14), border_radius=3)
                elif t in ITEM_OF:
                    col = {"c": (60, 60, 70), "f": (255, 110, 40), "s": (80, 200, 255)}[t]
                    pygame.draw.rect(surface, (250, 250, 250), r.inflate(-8, -8), border_radius=4)
                    pygame.draw.rect(surface, col, r.inflate(-14, -14), border_radius=3)
        for (x, y) in burning:  # flames
            r = pygame.Rect(x * ts, y * ts, ts, ts)
            pygame.draw.rect(surface, self.PAL["fire"], r.inflate(-4, -4), border_radius=8)
            pygame.draw.rect(surface, self.PAL["fire_core"], r.inflate(-16, -16), border_radius=6)
        for b in self.bombs:
            cx, cy = b.tx * ts + ts // 2, b.ty * ts + ts // 2
            pulse = 1 + (0.12 if (b.fuse // 8) % 2 else 0.0)
            pygame.draw.circle(surface, self.PAL["bomb"], (cx, cy + 2), int(11 * pulse))
            pygame.draw.circle(surface, (70, 70, 80), (cx - 4, cy - 3), 3)
            pygame.draw.line(surface, (90, 90, 100), (cx + 6, cy - 8), (cx + 10, cy - 13), 3)
            if (b.fuse // 4) % 2:
                pygame.draw.circle(surface, self.PAL["fuse"], (cx + 10, cy - 14), 3)
        for e in self.enemies:
            if not e.alive:
                continue
            col = self.PAL.get(e.kind, (220, 220, 220))
            cx, cy = int(e.x + e.width / 2), int(e.y + e.height / 2)
            bob = int(2 * math.sin(self._steps / 6 + cx))
            pygame.draw.circle(surface, col, (cx, cy + bob), 12)
            pygame.draw.circle(surface, tuple(max(0, c - 60) for c in col), (cx, cy + bob), 12, 2)
            ex = cx + (3 if e.direction == 2 else -3 if e.direction == 4 else 0)
            pygame.draw.circle(surface, self.PAL["eye"], (ex - 4, cy + bob - 2), 2)
            pygame.draw.circle(surface, self.PAL["eye"], (ex + 4, cy + bob - 2), 2)
        p = self.player
        if self.alive or self.won:
            cx, cy = int(p.x + p.width / 2), int(p.y + p.height / 2)
            pygame.draw.ellipse(surface, (20, 20, 30, 90), pygame.Rect(cx - 10, cy + 6, 20, 8))
            pygame.draw.rect(surface, self.PAL["player_acc"], pygame.Rect(cx - 8, cy - 2, 16, 12), border_radius=4)
            pygame.draw.circle(surface, self.PAL["player"], (cx, cy - 6), 9)
            pygame.draw.rect(surface, (250, 200, 190), pygame.Rect(cx - 5, cy - 6, 10, 6), border_radius=2)
            fx = {1: (0, -1), 2: (1, 0), 3: (0, 1), 4: (-1, 0)}[p.facing]
            pygame.draw.circle(surface, self.PAL["eye"], (cx - 3 + fx[0] * 2, cy - 5), 1)
            pygame.draw.circle(surface, self.PAL["eye"], (cx + 3 + fx[0] * 2, cy - 5), 1)
        if self.render_mode != "none":
            self._hud(surface)
        return surface

    def _hud(self, surface) -> None:
        if self._font is None:
            pygame.font.init()
            self._font = pygame.font.SysFont("consolas,dejavusansmono,monospace", 14, bold=True)
        alive = sum(1 for e in self.enemies if e.alive)
        txt = f"Score:{self.score}  Bombs:{self.player.bombs_max - len(self.bombs)}/{self.player.bombs_max}  " \
              f"Fire:{self.player.blast_range}  Enemies:{alive}  Time:{int(self.timer)}"
        img = self._font.render(txt, True, (255, 255, 255))
        bg = pygame.Surface((img.get_width() + 12, img.get_height() + 6), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 150))
        surface.blit(bg, (6, 6))
        surface.blit(img, (12, 9))


if __name__ == "__main__":  # smoke: walk level 0 to the exit with a scripted path
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    core = BombermanCore(level_idx=0)
    for _ in range(60 * 6):
        core.step((1, 0, 0))
    for _ in range(60 * 6):
        core.step((0, 1, 0))
    assert core.won, (core.alive, core.death_cause, core.player.x, core.player.y)
    core = BombermanCore(level_idx=1)
    for _ in range(60 * 2):
        core.step((1, 0, 0))
    assert not core.won and core.alive
    core.step((0, 0, 1))            # drop a bomb against the wall
    for _ in range(60 * 2):
        core.step((-1, 0, 0))       # run away
    for _ in range(200):
        core.step((0, 0, 0))
    assert core.alive and core.bricks_destroyed == 1, (core.alive, core.death_cause, core.bricks_destroyed)
    print("bomberman core ok")
