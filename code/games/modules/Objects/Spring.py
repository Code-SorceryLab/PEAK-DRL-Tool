"""
Spring.py
---------
Bouncing spring pads that launch Sonic into the air.

Types:
  - Red spring: Strong bounce (high launch)
  - Yellow spring: Weak bounce (lower launch)
  - Can be oriented: up, right, left, diagonal
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
import pygame

try:
    from .GameObject import GameObject
    from ..Parameters.Sonic_Map_parameters import (
        COLOR_SPRING_RED, COLOR_SPRING_YELLOW, COLOR_BLACK, TILE_SIZE
    )
    from ..Parameters.Sonic_Movement_parameters import (
        SPRING_BOUNCE_VEL, SPRING_BOUNCE_WEAK
    )
except ImportError:
    from GameObject import GameObject
    TILE_SIZE = 32
    COLOR_SPRING_RED = (255, 50, 50)
    COLOR_SPRING_YELLOW = (255, 220, 0)
    COLOR_BLACK = (0, 0, 0)
    SPRING_BOUNCE_VEL = -700.0
    SPRING_BOUNCE_WEAK = -450.0


class SpringType(Enum):
    RED    = auto()    # Strong
    YELLOW = auto()    # Weak


class SpringDir(Enum):
    UP    = auto()
    RIGHT = auto()
    LEFT  = auto()


@dataclass
class Spring:
    """A bouncy spring pad."""
    gObj: GameObject
    spring_type: SpringType = SpringType.RED
    direction: SpringDir = SpringDir.UP
    compressed: bool = False
    compress_timer: float = 0.0
    solid: bool = True

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

    @property
    def bounce_velocity(self) -> float:
        """Vertical launch speed (negative = upward)."""
        if self.spring_type == SpringType.RED:
            return SPRING_BOUNCE_VEL
        return SPRING_BOUNCE_WEAK

    @classmethod
    def from_tile(cls, pixel_x: float, pixel_y: float,
                  spring_type: SpringType = SpringType.RED,
                  direction: SpringDir = SpringDir.UP) -> "Spring":
        gobj = GameObject(float(pixel_x), float(pixel_y), TILE_SIZE, TILE_SIZE, active=True)
        return cls(gObj=gobj, spring_type=spring_type, direction=direction)

    def trigger(self):
        """Called when Sonic lands on the spring."""
        self.compressed = True
        self.compress_timer = 0.15

    def update(self, dt: float = 0.016, context=None):
        if self.compressed:
            self.compress_timer -= dt
            if self.compress_timer <= 0:
                self.compressed = False

    def render(self, surface: pygame.Surface, sx: float, sy: float, debug: bool = False):
        w = self.gObj.width
        h = self.gObj.height

        top_color = COLOR_SPRING_RED if self.spring_type == SpringType.RED else COLOR_SPRING_YELLOW

        if self.direction == SpringDir.UP:
            # Base
            base_h = h // 2
            base_y = sy + h - base_h
            pygame.draw.rect(surface, (180, 180, 180), (sx + 4, base_y, w - 8, base_h))

            # Coil lines
            coil_y = base_y
            if self.compressed:
                coil_y += 6
            for i in range(3):
                cy = int(coil_y - 2 - i * 4)
                if cy >= int(sy):
                    pygame.draw.line(surface, (140, 140, 140),
                                   (int(sx + 6), cy), (int(sx + w - 6), cy), 2)

            # Top plate
            plate_h = 6
            plate_y = coil_y - 14 if not self.compressed else coil_y - 6
            pygame.draw.rect(surface, top_color, (sx + 2, plate_y, w - 4, plate_h))
            pygame.draw.rect(surface, COLOR_BLACK, (sx + 2, plate_y, w - 4, plate_h), 1)

        elif self.direction == SpringDir.RIGHT:
            # Horizontal spring pointing right
            base_w = w // 2
            pygame.draw.rect(surface, (180, 180, 180), (sx, sy + 4, base_w, h - 8))
            plate_x = sx + w - 8 if not self.compressed else sx + w - 12
            pygame.draw.rect(surface, top_color, (plate_x, sy + 2, 6, h - 4))

        elif self.direction == SpringDir.LEFT:
            base_w = w // 2
            pygame.draw.rect(surface, (180, 180, 180), (sx + w - base_w, sy + 4, base_w, h - 8))
            plate_x = sx + 2 if not self.compressed else sx + 6
            pygame.draw.rect(surface, top_color, (plate_x, sy + 2, 6, h - 4))

        if debug:
            pygame.draw.rect(surface, (255, 64, 64), (sx, sy, w, h), 1)

    def __repr__(self) -> str:
        return (f"<Spring type={self.spring_type.name} dir={self.direction.name} "
                f"x={self.gObj.x:.0f} y={self.gObj.y:.0f}>")
