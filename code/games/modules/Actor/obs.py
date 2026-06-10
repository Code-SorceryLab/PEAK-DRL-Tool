from __future__ import annotations
from collections import deque
import numpy as np
from ..Parameters.Map_parameters import (
    TILE_GROUND, TILE_PLATFORM, TILE_QBLOCK, TILE_SPIKE, TILE_PIT, TILE_GOAL,
    TILE_CRUMBLE,
)

# Crumble tiles are solid to the player NOW but dissolve on touch, so they
# render as solid in ch0 yet stay passable for the BFS goal gradient.
_SOLID = {TILE_GROUND, TILE_PLATFORM, TILE_QBLOCK}
_HAZARD = {TILE_SPIKE, TILE_PIT}
_CORE_SCALARS = 11   # see build_scalars layout below


class ObsBuilder:
    """Builds the Dict{grids,scalars} observation. grids shape is fixed (4,W,W);
    scalars = core(11) + ability fields. Distance gradient (ch3) is a BFS over
    non-solid cells from the goal, computed once per level via prepare()."""

    def __init__(self, window: int = 21, n_ability_scalars: int = 0, tile_size: int = 32):
        self.window = window
        self.n_scalars = _CORE_SCALARS + n_ability_scalars
        self.tile_size = tile_size
        self._dist = None
        self._dist_max = 1.0

    def prepare(self, grid, goal_tiles):
        """Compute a BFS distance-to-goal map over non-solid cells (4-connected)."""
        rows = len(grid); cols = len(grid[0]) if rows else 0
        dist = np.full((rows, cols), -1.0, dtype=np.float32)
        q = deque()
        for (gx, gy) in goal_tiles:
            if 0 <= gy < rows and 0 <= gx < cols:
                dist[gy, gx] = 0.0
                q.append((gx, gy))
        while q:
            x, y = q.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < cols and 0 <= ny < rows and dist[ny, nx] < 0:
                    if grid[ny][nx] in _SOLID:
                        continue
                    dist[ny, nx] = dist[y, x] + 1.0
                    q.append((nx, ny))
        self._dist = dist
        self._dist_max = max(1.0, float(dist.max()))

    def build_grids(self, grid, player_tile, goal_tiles, hazard_cells=None):
        """hazard_cells: optional iterable of (col, row) world cells covered by
        dynamic hazards (e.g. saw blades) to overlay on the hazard channel."""
        if self._dist is None:
            self.prepare(grid, goal_tiles)
        W = self.window
        half = W // 2
        rows = len(grid); cols = len(grid[0]) if rows else 0
        px, py = player_tile
        out = np.zeros((4, W, W), dtype=np.float32)
        for j in range(W):
            gy = py - half + j
            if not (0 <= gy < rows):
                out[0, j, :] = 1.0  # out-of-bounds treated as solid wall
                continue
            for i in range(W):
                gx = px - half + i
                if not (0 <= gx < cols):
                    out[0, j, i] = 1.0
                    continue
                t = grid[gy][gx]
                if t in _SOLID or t == TILE_CRUMBLE:
                    out[0, j, i] = 1.0
                if t == TILE_GOAL:
                    out[1, j, i] = 1.0
                if t in _HAZARD:
                    out[2, j, i] = -1.0
                d = self._dist[gy, gx]
                out[3, j, i] = (1.0 - d / self._dist_max) if d >= 0 else 0.0
        if hazard_cells:
            for (cx, cy) in hazard_cells:
                i = cx - (px - half)
                j = cy - (py - half)
                if 0 <= i < W and 0 <= j < W:
                    out[2, j, i] = -1.0
        return out

    def build_scalars(self, state, controller, player_xy, goal_xy, level_wh) -> np.ndarray:
        ts = float(self.tile_size)
        lw, lh = level_wh
        gx, gy = goal_xy
        px, py = player_xy
        core = [
            px / ts, py / ts,
            float(np.clip(state.vx / 360.0, -1.0, 1.0)),
            float(np.clip(state.vy / 1200.0, -1.0, 1.0)),
            1.0 if state.on_ground else 0.0,
            1.0 if state.facing_right else 0.0,
            1.0 if state.contact_left else 0.0,
            1.0 if state.contact_right else 0.0,
            1.0 if state.contact_ceiling else 0.0,
            float(np.clip((gx - px) / max(lw, 1.0), -1.0, 1.0)),
            float(np.clip((gy - py) / max(lh, 1.0), -1.0, 1.0)),
        ]
        core.extend(controller.write_obs())
        return np.asarray(core, dtype=np.float32)
