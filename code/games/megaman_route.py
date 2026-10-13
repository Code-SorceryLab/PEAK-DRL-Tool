"""The route through a Mega Man 1 stage: how many cells of travel from each cell to the exit.

Shared by the training adapter (fitness and the route compass) and by telemetry (how far an
attempt got), so "progress" means one thing everywhere. A stage climbs, doubles back and drops,
so progress is not x: it is distance along this field.
"""
from __future__ import annotations

import heapq

import numpy as np

from . import megaman_enemies as en

JUMP_CELLS = 3           # how many cells a jump lifts the feet
DRIFT_CELLS = 3          # how many cells a jump carries sideways over nothing


def route_field(ld, jump_cells: int = JUMP_CELLS, drift_cells: int = DRIFT_CELLS):
    """Cost-to-exit for every cell, searched backwards from the exit. A move is allowed straight
    down anywhere, and diagonally down out of a jump's arc; up only by a jump from somewhere to
    stand or straight up a ladder; sideways only within a jump's height of somewhere to stand,
    or a jump's drift beyond that. Letting go of a ladder in mid-air goes down, never across.

    The field is deliberately cautious: a crossing that only a pixel-exact jump makes is not a
    route, so falling toward one earns nothing."""

    grid, rows, cols = ld.grid, ld.rows, ld.cols

    def free(c, r):                         # a cell a body may be in: not solid, not spikes, and
        if not (0 <= r < rows and 0 <= c < cols) or grid[r][c] in ("#", " ", "^"):
            return False                    # not resting on spikes (the cell right above them)
        return not (r + 1 < rows and grid[r + 1][c] == "^")

    def stands(c, r):                       # a body can stand in this cell: floor or a ladder's top rung
        if not free(c, r) or r + 1 >= rows:
            return False
        under = grid[r + 1][c]
        return under in ("#", "d") or (under == "H" and grid[r][c] != "H")

    jump = [[False] * cols for _ in range(rows)]    # within a jump's height of a place to stand
    for r in range(rows):
        for c in range(cols):
            for k in range(jump_cells + 1):
                if not free(c, r + k):
                    break
                if stands(c, r + k):
                    jump[r][c] = True
                    break
    # ...plus the cells a jump can drift across. A gap wider than a jump's drift is not crossed:
    # each side reaches less than halfway in, so a body there can only come down.
    air = [row[:] for row in jump]
    for r in range(rows):
        c = 0
        while c < cols:
            if jump[r][c] or not free(c, r):
                c += 1
                continue
            end = c
            while end < cols and free(end, r) and not jump[r][end]:
                end += 1
            n = end - c                     # a run of open cells with nowhere to stand in reach
            left = c > 0 and jump[r][c - 1]
            right = end < cols and jump[r][end]
            if left and right and n <= drift_cells:
                reach_l = reach_r = n
            else:
                both = left and right
                reach_l = min(drift_cells, (n - 1) // 2 if both else n) if left else 0
                reach_r = min(drift_cells, (n - 1) // 2 if both else n) if right else 0
            for k in range(reach_l):
                air[r][c + k] = True
            for k in range(reach_r):
                air[r][end - 1 - k] = True
            c = end

    goals = [(c, r) for r in range(rows) for c in range(cols) if grid[r][c] == "G"]
    goals += [(c, r) for g, c, r in ld.spawns if g == en.BOSS_GLYPH]
    cost = np.full((rows, cols), np.inf, dtype=np.float32)
    heap = [(0.0, c, r) for c, r in goals]
    for _, c, r in heap:
        cost[r, c] = 0.0
    while heap:                             # walk the moves backwards: who can step INTO (c, r)?
        d, c, r = heapq.heappop(heap)
        if d > cost[r, c]:
            continue
        for pc, pr in ((c, r - 1), (c - 1, r - 1), (c + 1, r - 1),      # from above: falling in
                       (c - 1, r), (c + 1, r),                           # from the side
                       (c, r + 1), (c - 1, r + 1), (c + 1, r + 1)):      # from below: climbing / jumping in
            if not free(pc, pr):
                continue
            if pr == r and not (air[r][c] and air[pr][pc]):
                continue                    # sideways only where a walk or a jump carries a body
            if pr > r and not ((jump[r][c] and air[pr][pc]) or (pc == c and grid[pr][pc] == "H")):
                continue                    # up only mid-jump (from a cell a jump passes through), or straight up a ladder
            if pc != c and pr != r and not (free(c, pr) or free(pc, r)):
                continue                    # a diagonal needs one of its corners open
            if pc != c and pr < r and not air[pr][pc]:
                continue                    # drifting while falling starts from a jump's arc; past it, a fall is straight down
            nd = d + (1.0 if pc == c or pr == r else 1.4)
            if nd < cost[pr, pc]:
                cost[pr, pc] = nd
                heapq.heappush(heap, (nd, pc, pr))
    return cost


def cost_at(cost, x: float, y: float, tile: float) -> float:
    """Route cost at an exact world position: the cell's cost, eased toward its cheapest
    neighbour by how far across the cell the body already is. Every pixel along the route
    counts, so a slow climb up a ladder pays from its first frame, not once per rung."""
    c, r = int(x // tile), int(y // tile)
    rows, cols = cost.shape
    if not (0 <= r < rows and 0 <= c < cols) or cost[r, c] == float("inf"):
        return float("inf")
    here = float(cost[r, c])
    best, step = here, None
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if (dr or dc) and 0 <= r + dr < rows and 0 <= c + dc < cols and cost[r + dr, c + dc] < best:
                best, step = float(cost[r + dr, c + dc]), (dc, dr)
    if step is None:
        return here
    along = ((x / tile - c - 0.5) * step[0] + (y / tile - r - 0.5) * step[1]) / (step[0] ** 2 + step[1] ** 2)
    return here - (here - best) * max(0.0, min(1.0, 0.5 + along))


def section_costs(cost, entry: dict, start_xy: tuple[float, float], tile: float) -> tuple[float, float]:
    """(start cost, end cost) of a level: a section ends where the next one starts; a whole
    stage ends at cost 0."""
    end = (entry.get("section") or {}).get("end")
    end_cost = float(cost[end[1], end[0]]) if end else 0.0
    if end_cost == float("inf"):
        end_cost = 0.0
    return max(cost_at(cost, *start_xy, tile), end_cost + 1.0), end_cost
