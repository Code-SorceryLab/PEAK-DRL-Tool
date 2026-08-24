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
            end: [416, 80]  # optional -> moving saw (ping-pong along a rail)
            period: 4.0     # optional, seconds for a full A->B->A cycle
            pivot: [416, 240]   # optional -> saw on a swinging ARM around this point
            arc: [-60, 60]      # optional, degrees swept; omit for a full circle

  A saw may use `end` (slides along a rail) or `pivot` (swings on an arm), not
  both. The rail and the arm are drawn, so the mount is visible like the real game.
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
_COL_ARM = (88, 90, 100)     # the bar a swinging saw hangs off
_COL_RAIL = (64, 66, 74)     # the track a sliding saw rides
_COL_MOUNT = (54, 56, 64)    # bolted anchor plate


@dataclass
class Saw:
    """Circular kill hazard. (cx, cy) is the blade centre in world pixels."""

    cx: float
    cy: float
    diameter: float = 64.0
    end: Optional[Tuple[float, float]] = None   # second waypoint -> saw on a rail
    period: float = 4.0                          # seconds per full cycle
    pivot: Optional[Tuple[float, float]] = None  # anchor -> saw on a swinging arm
    arc: Optional[Tuple[float, float]] = None    # degrees swept; None = full circle
    type_id: EntityType = field(default=EntityType.SAW, init=False, repr=False)

    def __post_init__(self):
        self._start = (float(self.cx), float(self.cy))
        if self.end is not None:
            self.end = (float(self.end[0]), float(self.end[1]))
        self._direction = 1
        self._spin = 0.0   # render-only blade rotation
        # Arm mode: remember how long the arm is and where it starts, then drive
        # the blade round the pivot instead of sliding it down a rail.
        self._arm = 0.0
        self._angle = 0.0
        if self.pivot is not None:
            self.pivot = (float(self.pivot[0]), float(self.pivot[1]))
            self._arm = math.hypot(self.cx - self.pivot[0], self.cy - self.pivot[1])
            self._angle = math.atan2(self.cy - self.pivot[1], self.cx - self.pivot[0])
            self._angle0 = self._angle
            if self.arc is not None:
                self.arc = (float(self.arc[0]), float(self.arc[1]))

    @property
    def radius(self) -> float:
        return self.diameter / 2.0

    # ── Motion ────────────────────────────────────────────────────────────
    def reset(self) -> None:
        """Back to the starting waypoint (called on level reset)."""
        self.cx, self.cy = self._start
        self._direction = 1
        self._spin = 0.0
        if self.pivot is not None:
            self._angle = self._angle0

    def update(self, dt: float, context=None) -> None:
        self._spin += dt * 8.0
        if self.pivot is not None:
            self._update_arm(dt)
            return
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

    def _update_arm(self, dt: float) -> None:
        """Swing the blade around `pivot`. With `arc` it ping-pongs between the two
        angles; without it the arm rotates continuously."""
        rate = 2.0 * math.pi / max(self.period, 1e-6)
        if self.arc is None:
            self._angle += rate * dt
        else:
            a0 = self._angle0 + math.radians(self.arc[0])
            a1 = self._angle0 + math.radians(self.arc[1])
            span = abs(a1 - a0)
            speed = 2.0 * span / max(self.period, 1e-6)
            self._angle += self._direction * speed * dt
            lo, hi = min(a0, a1), max(a0, a1)
            if self._angle >= hi:
                self._angle, self._direction = hi, -1
            elif self._angle <= lo:
                self._angle, self._direction = lo, 1
        self.cx = self.pivot[0] + self._arm * math.cos(self._angle)
        self.cy = self.pivot[1] + self._arm * math.sin(self._angle)

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
        # Mount first, so the blade is drawn on top of it.
        if self.pivot is not None:      # swinging arm back to the anchor
            px = int(sx + (self.pivot[0] - self.cx))
            py = int(sy + (self.pivot[1] - self.cy))
            pygame.draw.line(surface, _COL_ARM, (px, py), (cx, cy), max(3, r // 6))
            pygame.draw.circle(surface, _COL_MOUNT, (px, py), max(4, r // 4))
        elif self.end is not None:      # the rail it slides along
            ax = int(sx + (self._start[0] - self.cx)); ay = int(sy + (self._start[1] - self.cy))
            bx = int(sx + (self.end[0] - self.cx));    by = int(sy + (self.end[1] - self.cy))
            pygame.draw.line(surface, _COL_RAIL, (ax, ay), (bx, by), 3)
            pygame.draw.circle(surface, _COL_MOUNT, (ax, ay), 4)
            pygame.draw.circle(surface, _COL_MOUNT, (bx, by), 4)
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
