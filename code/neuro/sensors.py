"""Game-agnostic sensors: ray marching + scalar state, computed via GameAdapter queries.

Two exteroception modes, selected by GAConfig.sensors (same body scalars in both):
  "rays" — 14 floats (below), the default
  "grid" — the cores' obs window sized to 11x11 around the agent, 3 channels
           (solid / collectible / hazard; the Dijkstra oracle channel is dropped) + 5 body
           scalars = 368 floats. The Mario-AI-competition-style input, for the sensor ablation.

Ray sensor vector (14 floats, all roughly [-1, 1]):
  0-5  solid-ray distances (fwd, fwd-up30, fwd-up60, fwd-down30, fwd-down60, back); 1.0 = clear
  6    forward enemy distance (corridor of +-1 tile around the forward ray); 1.0 = none
  7    pit ahead (no ground within 4 tiles below a point 1.5 tiles ahead)
  8    grounded
  9    vx / VX_NORM
  10   vy / VY_NORM
  11   can_jump
  12   qblock count within 5 tiles / 5
  13   bias (1.0)
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .adapters import GameAdapter

SENSOR_MODES = ("rays", "grid")
GRID_HALF = 5   # 11x11 window centred on the agent's tile
GRID_N = 2 * GRID_HALF + 1
GRID_CH = 3     # solid, collectible, hazard
N_BODY = 5      # grounded, vx, vy, can_jump, bias
RAY_MAX_DIST = 250.0
RAY_STEP = 8.0
VX_NORM = 300.0
VY_NORM = 600.0

# (dx, dy) unit vectors; screen-space y grows downward
_C30, _S30 = math.cos(math.radians(30)), math.sin(math.radians(30))
_C60, _S60 = math.cos(math.radians(60)), math.sin(math.radians(60))
RAY_DIRS: list[tuple[float, float]] = [
    (1.0, 0.0),       # forward
    (_C30, -_S30),    # forward-up 30
    (_C60, -_S60),    # forward-up 60
    (_C30, _S30),     # forward-down 30
    (_C60, _S60),     # forward-down 60
    (-1.0, 0.0),      # back
]

# Ray hit types for the debug overlay
HIT_NONE = 0
HIT_SOLID = 1
HIT_ENEMY = 2

# One ray for the overlay: (x1, y1, x2, y2, hit_type) in world coordinates
Ray = tuple[float, float, float, float, int]

# Tile highlight kinds for the overlay (the classic PEAK debug look:
# red outline = tile a ray hit, translucent blue = tile the pit probe scanned)
TILE_HIT = 1
TILE_PROBE = 2
# One tile box: (world_x, world_y, size, kind)
Tile = tuple[float, float, float, int]


def _march(adapter: "GameAdapter", ox: float, oy: float, dx: float, dy: float) -> float:
    d = RAY_STEP
    while d <= RAY_MAX_DIST:
        if adapter.solid_at(ox + dx * d, oy + dy * d):
            return d
        d += RAY_STEP
    return RAY_MAX_DIST


def sensor_dim(mode: str) -> int:
    if mode not in SENSOR_MODES:
        raise ValueError(f"unknown sensor mode {mode!r}, expected one of {SENSOR_MODES}")
    return 14 if mode == "rays" else GRID_CH * (2 * GRID_HALF + 1) ** 2 + N_BODY


def _body(adapter: "GameAdapter") -> list[float]:
    return [1.0 if adapter.grounded else 0.0,
            float(np.clip(adapter.vx / VX_NORM, -1.0, 1.0)),
            float(np.clip(adapter.vy / VY_NORM, -1.0, 1.0)),
            1.0 if adapter.can_jump else 0.0,
            1.0]


def _fit_window(core) -> None:
    """Size the core's obs window to GRID_N (default 21). Every core reads these attributes at
    call time, so _obs() builds 121 cells instead of 441 and the agent stays the centre cell."""
    if hasattr(core, "window"):  # meatboy
        core.window = GRID_N
    else:  # platformer / megaman / sonic
        core.obs_width = core.obs_height = GRID_N
        core.obs_pad_x = core.obs_pad_y = GRID_HALF
        for k in ("_hazard_window_cache", "_dijkstra_window_cache", "_solid_window_cache"):
            if hasattr(core, k):  # per-step caches were built at the old size
                setattr(core, k, None)
        if hasattr(core, "_update_debug_caches"):  # megaman fills hazards from that cache
            core._update_debug_caches()


def _read_grid(adapter: "GameAdapter") -> tuple[np.ndarray, list[Ray], list[Tile]]:
    core = adapter.core
    if getattr(core, "window", getattr(core, "obs_width", None)) != GRID_N:
        _fit_window(core)
    win = core._obs()["grids"][:GRID_CH]  # (3, 11, 11); the agent's tile is the centre cell
    assert win.shape[1:] == (GRID_N, GRID_N), win.shape
    vec = np.concatenate([win.ravel(), _body(adapter)]).astype(np.float32)
    # overlay: outline the solid cells the agent sees, so the crop is visibly centred
    tile = float(adapter.tile_size)
    ox = adapter.x // tile * tile - GRID_HALF * tile
    oy = adapter.y // tile * tile - GRID_HALF * tile
    tiles: list[Tile] = [(ox + lx * tile, oy + ly * tile, tile, TILE_HIT)
                         for ly, lx in zip(*np.nonzero(win[0] > 0.5))]
    return vec, [], tiles


def read_sensors(adapter: "GameAdapter", mode: str = "rays") -> tuple[np.ndarray, list[Ray], list[Tile]]:
    """Returns (sensor vector, rays for the debug overlay, tile highlights)."""
    if mode == "grid":
        return _read_grid(adapter)
    sense = getattr(adapter, "sense", None)  # top-down games own their 14-slot ray layout
    if sense is not None:
        return sense(_march, RAY_MAX_DIST, HIT_SOLID, HIT_ENEMY, HIT_NONE, TILE_HIT, TILE_PROBE)
    ox, oy = adapter.x, adapter.y
    facing = -1.0 if adapter.vx < -1.0 else 1.0  # rays flip when moving left
    tile = float(adapter.tile_size)
    vec = np.empty(14, dtype=np.float32)
    rays: list[Ray] = []
    tiles: list[Tile] = []

    for i, (dx, dy) in enumerate(RAY_DIRS):
        dx *= facing
        d = _march(adapter, ox, oy, dx, dy)
        vec[i] = d / RAY_MAX_DIST
        hit = d < RAY_MAX_DIST
        rays.append((ox, oy, ox + dx * d, oy + dy * d, HIT_SOLID if hit else HIT_NONE))
        if hit:  # outline the exact tile the ray stopped on
            tiles.append(((ox + dx * d) // tile * tile, (oy + dy * d) // tile * tile, tile, TILE_HIT))

    # Forward enemy distance: nearest enemy inside a +-1 tile corridor along the forward ray
    enemy_d = RAY_MAX_DIST
    for ex, ey in adapter.enemy_positions():
        along = (ex - ox) * facing
        if 0.0 < along < enemy_d and abs(ey - oy) <= tile:
            enemy_d = along
    vec[6] = enemy_d / RAY_MAX_DIST
    if enemy_d < RAY_MAX_DIST:
        rays.append((ox, oy, ox + facing * enemy_d, oy, HIT_ENEMY))

    # Pit ahead: probe 1.5 tiles ahead, look for solid ground within 4 tiles below.
    # The scanned column is highlighted as translucent probe tiles in the overlay.
    px = ox + facing * tile * 1.5
    ground = False
    for k in range(1, 5):
        py = oy + tile * k
        tiles.append((px // tile * tile, py // tile * tile, tile, TILE_PROBE))
        if adapter.solid_at(px, py):
            ground = True
            break
    vec[7] = 0.0 if ground else 1.0

    vec[8:12] = _body(adapter)[:4]
    vec[12] = min(adapter.qblock_count_near(5), 5) / 5.0
    vec[13] = 1.0
    return vec, rays, tiles
