"""The route through a Castlevania stage: how much travel from each place a body can stand to
the exit, and what the next move is.

megaman_route.py searches a grid with approximate rules, which is enough for a game whose jump
can be steered. Here a jump is committed: once in the air nothing changes its path. So this
search is over the places a body can be at rest (a cell it stands in, or a cell of a flight of
stairs), joined by the moves the game really allows:

    walk    to the next cell of the same floor
    drop    off the end of a floor, straight down to whatever is below
    jump    left or right; where it lands is found by playing the jump out with the game's own
            numbers (castlevania_config.yaml) against the tiles
    stairs  one cell along a flight; a flight is entered only at its foot or its head

A cell is the one the feet are in. The cost of a move is roughly its time in cells walked
(stairs are slower). The returned array holds the cost to the exit of every such cell and
infinity everywhere else, and carries the move the search took out of each cell (see `move_at`),
so "which way next" is always a move that works.
"""
from __future__ import annotations

import heapq
import os

import numpy as np
import yaml

JUMP_CELLS = 2           # how many cells a jump lifts the feet    } with the standard numbers; the mutant
DRIFT_CELLS = 2          # how many cells it carries sideways      } generator reports changes against these
BOSS_GLYPH = "X"
STAIRS = "/\\"
WALLS = ("#", "d")
T = 16.0                 # one tile, NES px
WALK, DROP, JUMP, STAIR = 1, 2, 3, 4
STAIR_COST = 2.0         # a cell of stairs takes about twice a cell of floor
_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "castlevania_config.yaml")


class Field(np.ndarray):
    """The cost array, plus for each cell the move out of it: `kind` (0 none, WALK, DROP, JUMP,
    STAIR), `to` (the cell it leads to), `way` (-1 left, +1 right), `take_off` (for a jump: the
    x in NES px it was found from) and `arc` (the function that plays a jump or a fall out)."""
    kind = to = way = take_off = arc = None


def _player() -> dict:
    with open(_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)["player"]


def route_field(ld, jump_cells=None, drift_cells=None, player: dict | None = None):
    """Cost-to-exit for every cell a body can rest in. `player` replaces the movement numbers of
    castlevania_config.yaml (the mutant generator's mechanics mutants); the two cell counts are
    accepted for megaman_route's signature and not used: the jump is played out instead."""
    grid, rows, cols = ld.grid, ld.rows, ld.cols
    pc = dict(_player(), **(player or {}))
    walk, jump_v, grav, max_fall = (float(pc[k]) for k in ("walk_speed", "jump_velocity", "gravity", "max_fall_speed"))
    half, height = float(pc["width"]) / 2.0, float(pc["height"])

    def at(c, r):
        return grid[r][c] if 0 <= r < rows and 0 <= c < cols else " "

    def wall(c, r):
        return not 0 <= c < cols or at(c, r) in WALLS

    def in_floor(c, r):                     # a stair cell set into a floor: it holds a walker up
        return at(c, r) in STAIRS and at(c, r - 1) not in STAIRS and wall(c - 1, r) and wall(c + 1, r)

    def floor(c, r):
        return wall(c, r) or in_floor(c, r)

    def open_(c, r):                        # a cell a standing body's feet may be in: the body is two cells tall
        ch = at(c, r)
        return ch not in ("#", " ", "~", "d") and not wall(c, r - 1) and at(c, r - 1) != "~"

    def stands(c, r):
        return open_(c, r) and floor(c, r + 1) and not (at(c, r) in STAIRS and wall(c - 1, r) and wall(c + 1, r))

    def arc(x: float, feet: float, way: int, jump: bool = True):
        """Play out a jump (or, with jump=False, a fall with no sideways speed) from this centre
        x and feet y, as castlevania_core moves the body. Returns the cell it lands in, or None
        if it ends in water or a pit."""
        vx, vy = (way * walk, -jump_v) if jump else (0.0, 0.0)
        for _ in range(900):
            nx = x + vx
            if vx:
                lead = nx + half if vx > 0 else nx - half
                tc = int((lead - 1e-6) // T) if vx > 0 else int(lead // T)
                for tr in range(int((feet - height) // T), int((feet - 1e-6) // T) + 1):
                    if wall(tc, tr):
                        nx = tc * T - half if vx > 0 else (tc + 1) * T + half
                        break
            x = nx
            ny = feet + vy
            c0, c1 = int((x - half) // T), int((x + half - 1e-6) // T)
            if vy >= 0:
                tr = int((ny + 1e-6) // T)
                if any(wall(c, tr) or (in_floor(c, tr) and feet <= tr * T + 1e-6) for c in range(c0, c1 + 1)):
                    col = int(x // T)                   # the cell the feet are in: the centre's, if that holds him up
                    if not floor(col, tr):
                        col = c0 if floor(c0, tr) else c1
                    return col, tr - 1
            else:
                tr = int((ny - height) // T)
                if any(wall(c, tr) for c in range(c0, c1 + 1)):
                    ny, vy = (tr + 1) * T + height, 0.0
            feet = ny
            vy = min(vy + grav, max_fall)
            if feet - height > rows * T or any(at(c, int((feet - 1e-6) // T)) == "~" for c in range(c0, c1 + 1)):
                return None                             # a pit, or water
        return None

    nodes = {(c, r) for r in range(rows) for c in range(cols) if stands(c, r) or at(c, r) in STAIRS}
    out: dict = {n: [] for n in nodes}      # node -> [(target, cost, kind, way, take-off x)]

    def add(a, b, cost, kind, way, x0=0.0):
        if a in nodes and b in nodes and a != b:
            out[a].append((b, cost, kind, way, x0))

    for (c, r) in nodes:
        ch = at(c, r)
        if ch in STAIRS:                    # along the flight, and on and off at its two ends
            w = 1 if ch == "/" else -1
            up = (c + w, r - 1)
            if up in nodes:
                add((c, r), up, STAIR_COST, STAIR, w)
                add(up, (c, r), STAIR_COST, STAIR, -w)
            foot = (c - w, r)
            if at(c - w, r + 1) != ch and foot in nodes:
                add((c, r), foot, 1.0, STAIR, -w)
                add(foot, (c, r), 1.0, STAIR, w)
        if not stands(c, r):
            continue
        feet = (r + 1) * T
        for w in (-1, 1):
            n = (c + w, r)
            if stands(*n):
                add((c, r), n, 1.0, WALK, w)
            elif open_(*n) or at(*n) in STAIRS:         # the end of the floor: off it and straight down
                edge = (c + 1) * T if w > 0 else c * T
                land = arc(edge + w * (half + 0.5), feet, w, jump=False)
                if land:
                    add((c, r), land, 1.0 + abs(land[1] - r), DROP, w)
            for frac in (0.5, 0.2, 0.8, 0.05, 0.95):    # a jump, from a few places in the cell
                x0 = (c + frac) * T
                land = arc(x0, feet, w)
                if land and land != (c, r) and not any(t == land and k == JUMP for t, _, k, _, _ in out[(c, r)]):
                    add((c, r), land, max(abs(land[0] - c), abs(land[1] - r)) + 1.0, JUMP, w, x0)

    back: dict = {n: [] for n in nodes}
    for a, edges in out.items():
        for b, cost_, kind, w, x0 in edges:
            back[b].append((a, cost_, kind, w, x0))
    goals = [(c, r) for r in range(rows) for c in range(cols) if grid[r][c] == "G" and (c, r) in nodes]
    for g, c, r in ld.spawns:                           # a boss in the air is fought from the floor below him
        if g == BOSS_GLYPH:
            while r + 1 < rows and (c, r) not in nodes:
                r += 1
            goals.append((c, r))
    cost = np.full((rows, cols), np.inf, dtype=np.float32).view(Field)
    cost.kind = np.zeros((rows, cols), dtype=np.int8)
    cost.to = np.zeros((rows, cols, 2), dtype=np.int32)
    cost.way = np.zeros((rows, cols), dtype=np.int8)
    cost.take_off = np.zeros((rows, cols), dtype=np.float32)
    cost.arc = arc
    heap = [(0.0, c, r) for c, r in goals if (c, r) in nodes]
    for _, c, r in heap:
        cost[r, c] = 0.0
    while heap:                             # backwards from the exit: who has a move INTO (c, r)?
        d, c, r = heapq.heappop(heap)
        if d > cost[r, c]:
            continue
        for (pc_, pr), cost_, kind, w, x0 in back[(c, r)]:
            nd = d + cost_
            if nd < cost[pr, pc_]:
                cost[pr, pc_] = nd
                cost.kind[pr, pc_], cost.to[pr, pc_], cost.way[pr, pc_], cost.take_off[pr, pc_] = kind, (c, r), w, x0
                heapq.heappush(heap, (nd, pc_, pr))
    return cost


def move_at(cost, c: int, r: int) -> tuple[int, tuple[int, int], int, float]:
    """(kind, cell it leads to, way, take-off x) of the route's next move out of this cell;
    kind 0 at the exit and anywhere off the route."""
    rows, cols = cost.shape
    if not (0 <= r < rows and 0 <= c < cols) or cost[r, c] == float("inf"):
        return 0, (c, r), 0, 0.0
    return int(cost.kind[r, c]), (int(cost.to[r, c, 0]), int(cost.to[r, c, 1])), int(cost.way[r, c]), float(cost.take_off[r, c])


def path(cost, start: tuple[int, int], stop: float = 0.0) -> list[tuple[int, int]]:
    """The cells of the route from `start` until its cost falls to `stop`: every cell walked or
    climbed, and for a jump or a drop the cell it leaves and the cell it lands in."""
    c, r = start
    cells = [(c, r)]
    while cost[r, c] > stop and len(cells) < 4000:
        kind, (c, r), _, _ = move_at(cost, c, r)
        if not kind:
            break
        cells.append((c, r))
    return cells


def cost_at(cost, x: float, y: float, tile: float) -> float:
    """Route cost at an exact world position: the cell's cost, eased toward the cell the route
    walks or climbs to next by how far across the cell the body already is, so every pixel
    counts. Infinity anywhere a body cannot rest (in the air)."""
    c, r = int(x // tile), int(y // tile)
    kind, (nc, nr), _, _ = move_at(cost, c, r)
    rows, cols = cost.shape
    if not (0 <= r < rows and 0 <= c < cols):
        return float("inf")
    here = float(cost[r, c])
    dc, dr = nc - c, nr - r
    if kind not in (WALK, STAIR) or not (dc or dr):
        return here
    along = ((x / tile - c - 0.5) * dc + (y / tile - r - 0.5) * dr) / (dc * dc + dr * dr)
    return here - (here - float(cost[nr, nc])) * max(0.0, min(1.0, 0.5 + along))


def section_costs(cost, entry: dict, start_xy: tuple[float, float], tile: float) -> tuple[float, float]:
    """(start cost, end cost) of a level: a section ends where the next one starts; a whole
    stage ends at cost 0."""
    end = (entry.get("section") or {}).get("end")
    end_cost = float(cost[end[1], end[0]]) if end else 0.0
    if end_cost == float("inf"):
        end_cost = 0.0
    return max(cost_at(cost, *start_xy, tile), end_cost + 1.0), end_cost
