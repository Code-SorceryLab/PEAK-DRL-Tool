"""
Crusher.py
----------
A slamming metal press (Super Meat Boy style). Unlike a Saw it is **solid** —
it blocks movement and can shove the player into a wall — and it kills on
contact with its toothed leading face.

Motion is a ping-pong between two points with a pause (``dwell``) at each end,
which is what gives a crusher its slam-hold-retract rhythm.

Solidity
--------
Meat Boy's physics reads solidity off the tile grid, not off an object list, so
``MeatboyCore`` stamps each crusher's cells into the grid as ground for the
duration of a step and clears them again afterwards. Only cells that are
currently ``TILE_AIR`` are stamped, so a crusher can never erase real level
geometry or resurrect a dissolved crumble tile.

LevelLoader integration
-----------------------
    dynamics:
      crushers:
        - x: 320          # top-left, world pixels
          y: 96
          width: 64
          height: 64
          end: [320, 288] # slammed-out position
          period: 2.5     # seconds for a full out-and-back
          dwell: 0.4      # seconds held at each end
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import pygame

from ..System.EntityType import EntityType
from ..Parameters.Map_parameters import TILE_SIZE

_COL_BODY = (96, 99, 110)
_COL_EDGE = (140, 144, 158)
_COL_TEETH = (176, 180, 194)
_COL_RAIL = (64, 66, 74)


@dataclass
class Crusher:
    """Solid slamming block. (x, y) is the top-left corner in world pixels."""

    x: float
    y: float
    width: float = float(2 * TILE_SIZE)
    height: float = float(TILE_SIZE)
    end: Optional[Tuple[float, float]] = None
    period: float = 2.5
    dwell: float = 0.4
    type_id: EntityType = field(default=EntityType.TILE, init=False, repr=False)

    def __post_init__(self):
        self.x, self.y = float(self.x), float(self.y)
        self.width, self.height = float(self.width), float(self.height)
        self._start = (self.x, self.y)
        if self.end is not None:
            self.end = (float(self.end[0]), float(self.end[1]))
        self._direction = 1
        self._wait = 0.0

    # ── Motion ────────────────────────────────────────────────────────────
    def reset(self) -> None:
        self.x, self.y = self._start
        self._direction = 1
        self._wait = 0.0

    def update(self, dt: float, context=None) -> None:
        if self.end is None:
            return
        if self._wait > 0.0:
            self._wait -= dt
            return
        target = self.end if self._direction == 1 else self._start
        dx, dy = target[0] - self.x, target[1] - self.y
        dist = math.hypot(dx, dy)
        leg = math.hypot(self.end[0] - self._start[0], self.end[1] - self._start[1])
        # `period` covers both legs, so one leg gets half of the travel time
        travel = max(self.period - 2.0 * self.dwell, 1e-6)
        speed = 2.0 * leg / travel
        move = speed * dt
        if dist <= move:
            self.x, self.y = target
            self._direction *= -1
            self._wait = self.dwell
        else:
            self.x += dx / dist * move
            self.y += dy / dist * move

    @property
    def travel_axis(self) -> str:
        """'v' when it slams vertically, 'h' horizontally. Picks the toothed face."""
        if self.end is None:
            return "v"
        return "v" if abs(self.end[1] - self._start[1]) >= abs(self.end[0] - self._start[0]) else "h"

    # ── Collision ─────────────────────────────────────────────────────────
    def get_rect(self) -> pygame.Rect:
        return pygame.Rect(int(self.x), int(self.y), int(self.width), int(self.height))

    def hits_rect(self, rect: pygame.Rect) -> bool:
        return self.get_rect().colliderect(rect)

    def covered_cells(self, tile_size: int, rows: int, cols: int):
        """Grid cells the body overlaps — stamped solid, and fed to the obs channels."""
        c0 = max(0, int(self.x // tile_size))
        c1 = min(cols - 1, int((self.x + self.width - 1) // tile_size))
        r0 = max(0, int(self.y // tile_size))
        r1 = min(rows - 1, int((self.y + self.height - 1) // tile_size))
        return [(col, row) for row in range(r0, r1 + 1) for col in range(c0, c1 + 1)]

    # ── Rendering ─────────────────────────────────────────────────────────
    def render(self, surface: pygame.Surface, sx: float, sy: float, debug: bool = False) -> None:
        """Draw at screen-space top-left (sx, sy)."""
        body = pygame.Rect(int(sx), int(sy), int(self.width), int(self.height))
        if self.end is not None:      # the rail it slams along
            ex = sx + (self.end[0] - self.x)
            ey = sy + (self.end[1] - self.y)
            pygame.draw.line(surface, _COL_RAIL,
                             (int(sx + self.width / 2), int(sy + self.height / 2)),
                             (int(ex + self.width / 2), int(ey + self.height / 2)), 3)
        pygame.draw.rect(surface, _COL_BODY, body)
        pygame.draw.rect(surface, _COL_EDGE, body, 2)
        # teeth along the leading face
        n = max(2, int(self.width // 12) if self.travel_axis == "v" else int(self.height // 12))
        if self.travel_axis == "v":
            lead = body.bottom if (self.end and self.end[1] > self._start[1]) else body.top
            sign = 1 if lead == body.bottom else -1
            step = body.width / n
            for i in range(n):
                x0 = body.left + i * step
                pygame.draw.polygon(surface, _COL_TEETH, [
                    (x0, lead), (x0 + step, lead), (x0 + step / 2, lead + sign * step * 0.7)])
        else:
            lead = body.right if (self.end and self.end[0] > self._start[0]) else body.left
            sign = 1 if lead == body.right else -1
            step = body.height / n
            for i in range(n):
                y0 = body.top + i * step
                pygame.draw.polygon(surface, _COL_TEETH, [
                    (lead, y0), (lead, y0 + step), (lead + sign * step * 0.7, y0 + step / 2)])
        if debug:
            pygame.draw.rect(surface, (255, 64, 64), body, 1)

    @classmethod
    def from_tiles(cls, col: int, row: int, w_tiles: int = 2, h_tiles: int = 1,
                   travel_tiles: int = 4, axis: str = "v",
                   tile_size: int = TILE_SIZE, **kw) -> "Crusher":
        """Build from grid coordinates; `travel_tiles` is how far it slams."""
        x, y = col * tile_size, row * tile_size
        d = travel_tiles * tile_size
        end = (x, y + d) if axis == "v" else (x + d, y)
        return cls(x=x, y=y, width=w_tiles * tile_size, height=h_tiles * tile_size, end=end, **kw)

    def __repr__(self) -> str:
        return f"<Crusher {self.travel_axis} ({self.x:.0f},{self.y:.0f}) {self.width:.0f}x{self.height:.0f}>"
