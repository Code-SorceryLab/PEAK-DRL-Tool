"""
Badnik.py
---------
Eggman's robot enemies for the Sonic clone.

Behaviour:
  - Patrols back and forth on platforms
  - Destroyed when Sonic hits while rolling/jumping (ball attack)
  - Damages Sonic if touched while NOT in ball form
  - Releases an animal friend sprite on destruction (cosmetic)
  - Different types: Moto Bug (ground), Buzz Bomber (flying), Crabmeat (ground)
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
import pygame

try:
    from .GameObject import GameObject
    from ..Parameters.Sonic_Map_parameters import (
        COLOR_BADNIK, COLOR_BADNIK_EYE, COLOR_WHITE, COLOR_BLACK, TILE_SIZE
    )
except ImportError:
    from GameObject import GameObject
    TILE_SIZE = 32
    COLOR_BADNIK = (100, 100, 200)
    COLOR_BADNIK_EYE = (255, 0, 0)
    COLOR_WHITE = (255, 255, 255)
    COLOR_BLACK = (0, 0, 0)


class BadnikType(Enum):
    MOTOBUG    = auto()   # Ground crawler (like Goomba)
    BUZZBOMBER = auto()   # Flying bee
    CRABMEAT   = auto()   # Sideways crab


@dataclass
class Badnik:
    """A robotic enemy (Badnik) in the Sonic universe."""
    gObj: GameObject
    vx: float = -80.0
    vy: float = 0.0
    badnik_type: BadnikType = BadnikType.MOTOBUG
    color: tuple = COLOR_BADNIK
    alive: bool = True
    destroy_timer: float = 0.0   # Brief explosion animation

    # Convenience properties
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

    def update(self, dt: float, context=None):
        if not self.gObj.active:
            return

        if not self.alive:
            # Destruction animation countdown
            self.destroy_timer -= dt
            if self.destroy_timer <= 0:
                self.gObj.active = False
            return

        # Gravity
        if context:
            grav = context.GRAVITY
            max_fall = context.MAX_FALL_SPEED
        else:
            grav = 1100.0
            max_fall = 600.0

        if self.badnik_type == BadnikType.BUZZBOMBER:
            # Flying enemy — no gravity, patrol horizontally
            self.gObj.x += self.vx * dt
        else:
            # Ground enemy — apply gravity
            self.vy += grav * dt
            self.vy = min(self.vy, max_fall)
            self.gObj.y += self.vy * dt
            self.gObj.x += self.vx * dt

    def destroy(self):
        """Called when Sonic defeats this badnik."""
        self.alive = False
        self.destroy_timer = 0.25  # Brief pop animation
        self.vx = 0
        self.vy = 0

    def render(self, surface: pygame.Surface, sx: float, sy: float, debug: bool = False):
        if not self.gObj.active:
            return

        w = self.gObj.width
        h = self.gObj.height

        if not self.alive:
            # Explosion effect: expanding circle
            t = 1.0 - (self.destroy_timer / 0.25)
            radius = int(w * 0.5 + w * t)
            alpha = max(0, int(255 * (1.0 - t)))
            cx, cy = int(sx + w // 2), int(sy + h // 2)
            pygame.draw.circle(surface, (255, 200, 50), (cx, cy), radius)
            pygame.draw.circle(surface, (255, 255, 200), (cx, cy), max(1, radius // 2))
            return

        if self.badnik_type == BadnikType.MOTOBUG:
            self._render_motobug(surface, sx, sy, w, h)
        elif self.badnik_type == BadnikType.BUZZBOMBER:
            self._render_buzzbomber(surface, sx, sy, w, h)
        elif self.badnik_type == BadnikType.CRABMEAT:
            self._render_crabmeat(surface, sx, sy, w, h)
        else:
            # Fallback rectangle
            pygame.draw.rect(surface, self.color, (sx, sy, w, h))

    def _render_motobug(self, surface, sx, sy, w, h):
        """Ladybug-like ground robot."""
        # Body (rounded rectangle approximation)
        body_rect = (int(sx + 2), int(sy + 4), w - 4, h - 8)
        pygame.draw.ellipse(surface, (180, 60, 60), body_rect)
        pygame.draw.ellipse(surface, COLOR_BLACK, body_rect, 1)

        # Wheel
        wheel_y = int(sy + h - 6)
        pygame.draw.circle(surface, (80, 80, 80), (int(sx + w // 2), wheel_y), 5)
        pygame.draw.circle(surface, (40, 40, 40), (int(sx + w // 2), wheel_y), 5, 1)

        # Eye
        eye_x = int(sx + (w * 0.7 if self.vx < 0 else w * 0.3))
        eye_y = int(sy + 8)
        pygame.draw.circle(surface, COLOR_WHITE, (eye_x, eye_y), 4)
        pygame.draw.circle(surface, COLOR_BADNIK_EYE, (eye_x, eye_y), 2)

    def _render_buzzbomber(self, surface, sx, sy, w, h):
        """Flying bee robot."""
        # Body
        cx, cy = int(sx + w // 2), int(sy + h // 2)
        pygame.draw.ellipse(surface, (220, 180, 40), (sx + 4, sy + 6, w - 8, h - 12))

        # Stripes
        for i in range(3):
            stripe_y = int(sy + 8 + i * 6)
            pygame.draw.line(surface, COLOR_BLACK,
                           (int(sx + 6), stripe_y), (int(sx + w - 6), stripe_y), 1)

        # Wings
        pygame.draw.ellipse(surface, (200, 220, 255, 128),
                          (sx - 2, sy - 4, 14, 10))
        pygame.draw.ellipse(surface, (200, 220, 255, 128),
                          (sx + w - 12, sy - 4, 14, 10))

        # Eye
        pygame.draw.circle(surface, COLOR_BADNIK_EYE, (cx, int(sy + 10)), 3)

    def _render_crabmeat(self, surface, sx, sy, w, h):
        """Crab robot with pincers."""
        # Body
        body_rect = (int(sx + 6), int(sy + 4), w - 12, h - 8)
        pygame.draw.ellipse(surface, (200, 80, 80), body_rect)

        # Pincers
        pygame.draw.circle(surface, (220, 100, 100), (int(sx + 2), int(sy + h // 2)), 5)
        pygame.draw.circle(surface, (220, 100, 100), (int(sx + w - 2), int(sy + h // 2)), 5)

        # Eyes
        pygame.draw.circle(surface, COLOR_WHITE, (int(sx + w * 0.35), int(sy + 8)), 3)
        pygame.draw.circle(surface, COLOR_WHITE, (int(sx + w * 0.65), int(sy + 8)), 3)
        pygame.draw.circle(surface, COLOR_BLACK, (int(sx + w * 0.35), int(sy + 8)), 1)
        pygame.draw.circle(surface, COLOR_BLACK, (int(sx + w * 0.65), int(sy + 8)), 1)

    def __repr__(self) -> str:
        t = self.badnik_type.name
        return f"<Badnik type={t} x={self.gObj.x:.0f} y={self.gObj.y:.0f} alive={self.alive}>"
