"""
FireBar.py
----------
A rotating bar of fireballs (Super Mario Bros. castle hazard): a chain of
fireballs spins around a fixed pivot and hurts the player on contact. It is
not solid and cannot be stomped or killed.

LevelLoader integration
-----------------------
The sidecar YAML ``dynamics: firebars:`` section spawns them:

      dynamics:
        firebars:
          - x: 976            # pivot centre, world pixels
            y: 368
            length: 6         # optional, fireballs in the chain (default 6)
            period: 3.4       # optional, seconds per full turn
            clockwise: true   # optional
            angle: 30         # optional, start angle in degrees (0 = right, 90 = down)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import pygame

_COL_OUTER = (228, 92, 16)
_COL_INNER = (252, 216, 168)


@dataclass
class FireBar:
    """Rotating fireball chain. (cx, cy) is the pivot centre in world pixels."""

    cx: float
    cy: float
    length: int = 6            # fireballs, the first one sits on the pivot
    spacing: float = 16.0      # px between fireball centres
    period: float = 3.4        # seconds per full turn
    clockwise: bool = True
    angle: float = 0.0         # degrees; 0 = right, 90 = down (screen coordinates)
    ball_radius: float = 6.0

    def update(self, dt: float, context=None) -> None:
        step = 360.0 * dt / max(self.period, 1e-6)
        self.angle = (self.angle + (step if self.clockwise else -step)) % 360.0

    def balls(self) -> List[Tuple[float, float]]:
        """Centres of the fireballs, pivot first."""
        a = math.radians(self.angle)
        dx, dy = math.cos(a) * self.spacing, math.sin(a) * self.spacing
        return [(self.cx + dx * i, self.cy + dy * i) for i in range(self.length)]

    def hits_rect(self, rect: pygame.Rect) -> bool:
        """Any fireball (circle) overlapping the rect."""
        r2 = self.ball_radius * self.ball_radius
        for bx, by in self.balls():
            nx = min(max(bx, rect.left), rect.right)
            ny = min(max(by, rect.top), rect.bottom)
            if (bx - nx) ** 2 + (by - ny) ** 2 <= r2:
                return True
        return False

    def render(self, surface: pygame.Surface, camera_x: float, camera_y: float) -> None:
        from ... import mario_art                   # violet flame orbs
        d = 2 * int(self.ball_radius) + 2
        for bx, by in self.balls():
            mario_art.draw(surface, "flame_orb", bx - camera_x - d / 2, by - camera_y - d / 2, d, d)
