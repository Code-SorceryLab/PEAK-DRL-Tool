"""
Saw.py
------
A circular sawblade hazard (Super Meat Boy style): kills the player on
contact. Static, or ping-ponging between two points on a fixed period.

Collision is circle-vs-AABB on the saw's circle against the player's rect
(the same check the reference JS remake does in ``hits_saw``, with the
nearest-point clamp done properly so the corners are survivable).

LevelLoader integration
-----------------------
- ASCII char ``*`` spawns a static saw centred in that tile, two tiles in
  diameter (the most common saw size in the reference levels).
- The sidecar YAML ``dynamics: saws:`` section spawns sized/moving saws:

      dynamics:
        saws:
          - x: 416          # centre, world pixels
            y: 240
            diameter: 96    # optional, default 64
            end: [416, 80]  # optional -> moving saw (ping-pong)
            period: 4.0     # optional, seconds for a full A->B->A cycle
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import pygame

from ..System.EntityType import EntityType
from ..Parameters.Map_parameters import TILE_SIZE

_COL_BLADE = (110, 110, 120)
_COL_TEETH = (160, 160, 170)
_COL_HUB = (70, 70, 80)


@dataclass
class Saw:
    """Circular kill hazard. (cx, cy) is the blade centre in world pixels."""

    cx: float
    cy: float
    diameter: float = 64.0
    end: Optional[Tuple[float, float]] = None   # second waypoint -> moving saw
    period: float = 4.0                          # seconds per full cycle
    type_id: EntityType = field(default=EntityType.SAW, init=False, repr=False)

    def __post_init__(self):
        self._start = (float(self.cx), float(self.cy))
        if self.end is not None:
            self.end = (float(self.end[0]), float(self.end[1]))
        self._direction = 1
        self._spin = 0.0   # render-only blade rotation

    @property
    def radius(self) -> float:
        return self.diameter / 2.0

    # ── Motion ────────────────────────────────────────────────────────────
    def reset(self) -> None:
        """Back to the starting waypoint (called on level reset)."""
        self.cx, self.cy = self._start
        self._direction = 1
        self._spin = 0.0

    def update(self, dt: float, context=None) -> None:
        self._spin += dt * 8.0
        if self.end is None:
            return
        target = self.end if self._direction == 1 else self._start
        dx = target[0] - self.cx
        dy = target[1] - self.cy
        dist = math.hypot(dx, dy)
        leg = math.hypot(self.end[0] - self._start[0], self.end[1] - self._start[1])
        speed = 2.0 * leg / max(self.period, 1e-6)
        move = speed * dt
        if dist <= move:
            self.cx, self.cy = target
            self._direction *= -1
        else:
            self.cx += dx / dist * move
            self.cy += dy / dist * move

    # ── Collision ─────────────────────────────────────────────────────────
    def hits_rect(self, rect: pygame.Rect) -> bool:
        """Circle vs AABB: clamp the centre to the rect, compare to radius."""
        nx = min(max(self.cx, rect.left), rect.right)
        ny = min(max(self.cy, rect.top), rect.bottom)
        dx = self.cx - nx
        dy = self.cy - ny
        return dx * dx + dy * dy <= self.radius * self.radius

    def covered_cells(self, tile_size: int, rows: int, cols: int):
        """Grid cells the blade overlaps — feeds the obs hazard channel."""
        r = self.radius
        c0 = int((self.cx - r) // tile_size)
        c1 = int((self.cx + r) // tile_size)
        r0 = int((self.cy - r) // tile_size)
        r1 = int((self.cy + r) // tile_size)
        cells = []
        for row in range(max(0, r0), min(rows, r1 + 1)):
            for col in range(max(0, c0), min(cols, c1 + 1)):
                cell = pygame.Rect(col * tile_size, row * tile_size, tile_size, tile_size)
                if self.hits_rect(cell):
                    cells.append((col, row))
        return cells

    # ── Rendering ─────────────────────────────────────────────────────────
    def render(self, surface: pygame.Surface, sx: float, sy: float, debug: bool = False) -> None:
        """Draw at screen-space centre (sx, sy)."""
        r = int(self.radius)
        cx, cy = int(sx), int(sy)
        # teeth: alternating outer/inner vertices around the rim
        pts = []
        n = 16
        for i in range(2 * n):
            ang = self._spin + i * math.pi / n
            rad = r if i % 2 == 0 else r * 0.8
            pts.append((cx + rad * math.cos(ang), cy + rad * math.sin(ang)))
        pygame.draw.polygon(surface, _COL_TEETH, pts)
        pygame.draw.circle(surface, _COL_BLADE, (cx, cy), int(r * 0.78))
        pygame.draw.circle(surface, _COL_HUB, (cx, cy), max(2, int(r * 0.2)))
        if debug:
            pygame.draw.circle(surface, (255, 64, 64), (cx, cy), r, 1)

    @classmethod
    def from_tile(cls, col: int, row: int, tile_size: int = TILE_SIZE,
                  diameter: Optional[float] = None) -> "Saw":
        """Static saw centred in grid cell (col, row); default 2 tiles wide."""
        return cls(
            cx=col * tile_size + tile_size / 2.0,
            cy=row * tile_size + tile_size / 2.0,
            diameter=float(diameter if diameter is not None else 2 * tile_size),
        )

    def __repr__(self) -> str:
        kind = "moving" if self.end else "static"
        return f"<Saw {kind} c=({self.cx:.0f},{self.cy:.0f}) d={self.diameter:.0f}>"
