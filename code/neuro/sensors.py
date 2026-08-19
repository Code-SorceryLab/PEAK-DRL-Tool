"""Game-agnostic sensors: ray marching + scalar state, computed via GameAdapter queries.

Sensor vector (14 floats, all roughly [-1, 1]):
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


def _march(adapter: "GameAdapter", ox: float, oy: float, dx: float, dy: float) -> float:
    d = RAY_STEP
    while d <= RAY_MAX_DIST:
        if adapter.solid_at(ox + dx * d, oy + dy * d):
            return d
        d += RAY_STEP
    return RAY_MAX_DIST


def read_sensors(adapter: "GameAdapter") -> tuple[np.ndarray, list[Ray]]:
    """Returns (14-float sensor vector, rays for the debug overlay)."""
    ox, oy = adapter.x, adapter.y
    facing = -1.0 if adapter.vx < -1.0 else 1.0  # rays flip when moving left
    vec = np.empty(14, dtype=np.float32)
    rays: list[Ray] = []

    for i, (dx, dy) in enumerate(RAY_DIRS):
        dx *= facing
        d = _march(adapter, ox, oy, dx, dy)
        vec[i] = d / RAY_MAX_DIST
        rays.append((ox, oy, ox + dx * d, oy + dy * d, HIT_SOLID if d < RAY_MAX_DIST else HIT_NONE))

    # Forward enemy distance: nearest enemy inside a +-1 tile corridor along the forward ray
    tile = float(adapter.tile_size)
    enemy_d = RAY_MAX_DIST
    for ex, ey in adapter.enemy_positions():
        along = (ex - ox) * facing
        if 0.0 < along < enemy_d and abs(ey - oy) <= tile:
            enemy_d = along
    vec[6] = enemy_d / RAY_MAX_DIST
    if enemy_d < RAY_MAX_DIST:
        rays.append((ox, oy, ox + facing * enemy_d, oy, HIT_ENEMY))

    # Pit ahead: probe 1.5 tiles ahead, look for solid ground within 4 tiles below
    px = ox + facing * tile * 1.5
    vec[7] = 0.0 if any(adapter.solid_at(px, oy + tile * k) for k in range(1, 5)) else 1.0

    vec[8] = 1.0 if adapter.grounded else 0.0
    vec[9] = float(np.clip(adapter.vx / VX_NORM, -1.0, 1.0))
    vec[10] = float(np.clip(adapter.vy / VY_NORM, -1.0, 1.0))
    vec[11] = 1.0 if adapter.can_jump else 0.0
    vec[12] = min(adapter.qblock_count_near(5), 5) / 5.0
    vec[13] = 1.0
    return vec, rays
