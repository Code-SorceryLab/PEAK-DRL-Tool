"""
SlopeTile.py
============
Height-map slope tiles for Sonic-style diagonal ramps.

How it works
────────────
Every slope tile stores a height_map[0..TILE_SIZE-1] array.  Each entry is
the FLOOR Y offset from the tile's top edge at that pixel column.

  height_map[col] = 0   → floor is at the very top of the tile
  height_map[col] = 32  → floor is at the very bottom (no surface here)

When Sonic's foot position (his gObj bottom edge) enters a slope tile, the
physics code reads the height_map at his local X within the tile and snaps
him to that Y.  This gives smooth diagonal movement without needing polygon
collisions.

ASCII map characters
────────────────────
  /   Slope rising left-to-right   (gentle 22.5°, bottom-left → mid-right)
  \\   Slope falling left-to-right  (gentle 22.5°, mid-left → bottom-right)
  (   Steep slope up  (45°, bottom-left → top-right)
  )   Steep slope down (45°, top-left → bottom-right)
  U   Concave (valley bottom)
  ^   Still used for spikes — NOT slopes
  n   Convex (hilltop curve)

Rendering
─────────
Slope tiles draw a filled polygon from the height-map profile down to the
tile bottom, with a green grass cap line on top.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Tuple
import math
import pygame
import numpy as np

try:
    from .GameObject import GameObject
    from ..Parameters.Sonic_Map_parameters import (
        TILE_SIZE, COLOR_GROUND, COLOR_GROUND_CHECK, COLOR_GRASS_TOP, COLOR_BLACK
    )
except ImportError:
    from GameObject import GameObject
    TILE_SIZE = 32
    COLOR_GROUND = (139, 90, 43)
    COLOR_GROUND_CHECK = (115, 66, 23)
    COLOR_GRASS_TOP = (0, 200, 50)
    COLOR_BLACK = (0, 0, 0)


class SlopeType(Enum):
    """Pre-defined slope profiles."""
    FLAT            = auto()   # No slope (regular ground tile)
    GENTLE_UP       = auto()   # /  — 22.5° ascending left→right (bottom half)
    GENTLE_UP_TOP   = auto()   # /  — 22.5° ascending left→right (top half)
    GENTLE_DOWN     = auto()   # \  — 22.5° descending left→right (bottom half)
    GENTLE_DOWN_TOP = auto()   # \  — 22.5° descending left→right (top half)
    STEEP_UP        = auto()   # (  — 45° ascending left→right
    STEEP_DOWN      = auto()   # )  — 45° descending left→right
    CONCAVE         = auto()   # U  — valley bottom (quarter pipe)
    CONVEX          = auto()   # n  — hilltop curve (quarter pipe)
    HALF_UP_LEFT    = auto()   # left half of a step
    HALF_UP_RIGHT   = auto()   # right half of a step


def _build_height_map(slope_type: SlopeType, size: int = TILE_SIZE) -> np.ndarray:
    """
    Build a height_map array of length `size`.

    height_map[col] = distance from tile TOP to the floor surface.
      0  = floor at top of tile (solid all the way down)
      size = floor at bottom of tile (air — no surface)

    Sonic stands ON the floor, so his foot Y = tile.y + height_map[col].
    """
    hm = np.full(size, size, dtype=np.float32)  # default: no surface

    if slope_type == SlopeType.FLAT:
        hm[:] = 0  # Fully solid from top

    elif slope_type == SlopeType.GENTLE_UP:
        # 22.5° ascending: left=bottom(size), right=half(size//2)
        # This is the BOTTOM tile of a gentle slope pair
        for col in range(size):
            t = col / max(size - 1, 1)
            hm[col] = size - int(t * (size // 2))  # size → size//2

    elif slope_type == SlopeType.GENTLE_UP_TOP:
        # 22.5° ascending TOP tile: left=half(size//2), right=top(0)
        for col in range(size):
            t = col / max(size - 1, 1)
            hm[col] = (size // 2) - int(t * (size // 2))  # size//2 → 0

    elif slope_type == SlopeType.GENTLE_DOWN:
        # 22.5° descending: left=half(size//2), right=bottom(size)
        for col in range(size):
            t = col / max(size - 1, 1)
            hm[col] = (size // 2) + int(t * (size // 2))

    elif slope_type == SlopeType.GENTLE_DOWN_TOP:
        # 22.5° descending TOP tile: left=top(0), right=half(size//2)
        for col in range(size):
            t = col / max(size - 1, 1)
            hm[col] = int(t * (size // 2))

    elif slope_type == SlopeType.STEEP_UP:
        # 45° ascending: left=bottom(size), right=top(0)
        for col in range(size):
            t = col / max(size - 1, 1)
            hm[col] = size - int(t * size)

    elif slope_type == SlopeType.STEEP_DOWN:
        # 45° descending: left=top(0), right=bottom(size)
        for col in range(size):
            t = col / max(size - 1, 1)
            hm[col] = int(t * size)

    elif slope_type == SlopeType.CONCAVE:
        # Quarter circle valley: parabolic floor curving up at edges
        for col in range(size):
            t = (col - size / 2) / (size / 2)  # -1 to +1
            hm[col] = int(size * (1.0 - t * t) * 0.5)

    elif slope_type == SlopeType.CONVEX:
        # Quarter circle hilltop: parabolic floor curving down at edges
        for col in range(size):
            t = (col - size / 2) / (size / 2)  # -1 to +1
            hm[col] = int(size * t * t * 0.5)

    elif slope_type == SlopeType.HALF_UP_LEFT:
        # Left half step: right half is solid, left half is air
        hm[:size // 2] = size
        hm[size // 2:] = 0

    elif slope_type == SlopeType.HALF_UP_RIGHT:
        # Right half step: left half is solid, right half is air
        hm[:size // 2] = 0
        hm[size // 2:] = size

    return hm


@dataclass
class SlopeTile:
    """
    A tile with a height-map defined floor surface.

    gObj.x, gObj.y define the tile's top-left corner in world pixels.
    height_map[col] gives the floor offset from tile top at each column.
    """
    gObj: GameObject
    slope_type: SlopeType = SlopeType.FLAT
    height_map: np.ndarray = field(default=None)
    # solid=False so PhysicsManager's AABB collision skips slope tiles.
    # SlopePhysics handles floor-snapping via the height map instead.
    # The tile is still inserted into static_hash for rendering and obs queries.
    solid: bool = False

    # Tile-type ID for the grid (used by Dijkstra, obs)
    type_id: int = 1  # TILE_GROUND

    def __post_init__(self):
        if self.height_map is None:
            self.height_map = _build_height_map(self.slope_type, self.gObj.width)

    # ── Convenience properties ───────────────────────────────────────────
    @property
    def x(self): return self.gObj.x
    @property
    def y(self): return self.gObj.y
    @property
    def width(self): return self.gObj.width
    @property
    def height(self): return self.gObj.height
    @property
    def active(self): return self.gObj.active

    # ── Surface query ────────────────────────────────────────────────────
    def get_surface_y(self, world_x: float) -> float:
        """
        Returns the world Y of the floor surface at the given world X.

        If world_x is outside this tile, returns tile bottom (no surface).
        """
        local_x = int(world_x - self.gObj.x)
        local_x = max(0, min(local_x, len(self.height_map) - 1))
        return self.gObj.y + self.height_map[local_x]

    def get_surface_angle(self, world_x: float) -> float:
        """
        Returns the slope angle in degrees at the given world X.
        0° = flat, positive = ascending right, negative = descending right.
        Used for physics (uphill decel, downhill accel).
        """
        local_x = int(world_x - self.gObj.x)
        local_x = max(1, min(local_x, len(self.height_map) - 2))

        dy = self.height_map[local_x + 1] - self.height_map[local_x - 1]
        # dy is in pixel space (positive = going down in screen coords)
        # For Sonic: negative dy = ascending (going up), positive = descending
        return math.degrees(math.atan2(-dy, 2.0))

    def is_inside(self, world_x: float, world_y: float) -> bool:
        """Check if a point is inside the solid part of this slope."""
        if not (self.gObj.x <= world_x < self.gObj.x + self.gObj.width):
            return False
        if not (self.gObj.y <= world_y < self.gObj.y + self.gObj.height):
            return False
        surface_y = self.get_surface_y(world_x)
        return world_y >= surface_y

    # ── Factory ──────────────────────────────────────────────────────────
    @classmethod
    def create(cls, col: int, row: int, slope_type: SlopeType,
               size: int = TILE_SIZE) -> "SlopeTile":
        gobj = GameObject(
            float(col * size), float(row * size),
            size, size, active=True
        )
        return cls(gObj=gobj, slope_type=slope_type)

    # ── Rendering ────────────────────────────────────────────────────────
    def render(self, surface: pygame.Surface, cam_x: float, cam_y: float):
        sx = self.gObj.x - cam_x
        sy = self.gObj.y - cam_y
        w = self.gObj.width
        h = self.gObj.height

        # Skip if off-screen
        if sx + w < 0 or sx > surface.get_width() or sy + h < 0 or sy > surface.get_height():
            return

        if self.slope_type == SlopeType.FLAT:
            # Regular ground tile (checkered)
            gx = int(self.gObj.x // TILE_SIZE)
            gy = int(self.gObj.y // TILE_SIZE)
            color = COLOR_GROUND if (gx + gy) % 2 == 0 else COLOR_GROUND_CHECK
            pygame.draw.rect(surface, color, (sx, sy, w, h))
            return

        # ── Build polygon from height map ────────────────────────────────
        # Top profile points (the slope surface)
        points = []
        step = max(1, w // 16)  # Sample every few pixels for performance
        for col in range(0, w, step):
            px = sx + col
            py = sy + self.height_map[min(col, len(self.height_map) - 1)]
            points.append((int(px), int(py)))

        # Last point at right edge
        points.append((int(sx + w), int(sy + self.height_map[-1])))

        # Close the polygon along the bottom
        points.append((int(sx + w), int(sy + h)))
        points.append((int(sx), int(sy + h)))

        if len(points) >= 3:
            # Fill with checkered ground color
            gx = int(self.gObj.x // TILE_SIZE)
            gy = int(self.gObj.y // TILE_SIZE)
            fill_color = COLOR_GROUND if (gx + gy) % 2 == 0 else COLOR_GROUND_CHECK
            pygame.draw.polygon(surface, fill_color, points)

            # Draw grass cap line on the surface
            grass_points = points[:-2]  # Just the top surface
            if len(grass_points) >= 2:
                pygame.draw.lines(surface, COLOR_GRASS_TOP, False, grass_points, 3)

                # Grass blades along the surface
                for i in range(0, len(grass_points) - 1, 3):
                    bx, by = grass_points[i]
                    blade_h = 3 + (i % 3)
                    pygame.draw.line(surface, (0, 220, 60),
                                   (bx, by), (bx + 1, by - blade_h), 1)

    def render_at(self, surface: pygame.Surface, sx: float, sy: float):
        """Render at explicit screen coordinates (for entity-style rendering)."""
        self.render(surface, self.gObj.x - sx, self.gObj.y - sy)

    def __repr__(self) -> str:
        return (f"<SlopeTile type={self.slope_type.name} "
                f"x={self.gObj.x:.0f} y={self.gObj.y:.0f}>")


# ═════════════════════════════════════════════════════════════════════════════
# ASCII MAP CHARACTER → SLOPE TYPE MAPPING
# ═════════════════════════════════════════════════════════════════════════════
SLOPE_CHAR_MAP = {
    '/':  SlopeType.STEEP_UP,        # 45° ascending left→right
    '\\': SlopeType.STEEP_DOWN,      # 45° descending left→right
    '(':  SlopeType.GENTLE_UP,       # 22.5° ascending (bottom tile)
    ')':  SlopeType.GENTLE_DOWN,     # 22.5° descending (bottom tile)
    '[':  SlopeType.GENTLE_UP_TOP,   # 22.5° ascending (top tile)
    ']':  SlopeType.GENTLE_DOWN_TOP, # 22.5° descending (top tile)
    'U':  SlopeType.CONCAVE,         # Valley bottom
    'n':  SlopeType.CONVEX,          # Hilltop curve
}