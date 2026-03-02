from __future__ import annotations
from dataclasses import dataclass
from .GameObject import GameObject
import pygame
from ..Parameters.Map_parameters import COLOR_POWERUP_MUSH


@dataclass
class Mushroom:
    """
    Mushroom powerup — walks off QBlocks, bounces off walls, falls with gravity.
    Collected by touching → player goes SMALL→BIG.
    Handled by PhysicsManager._solve_aabb_collision (bounce_x=True).
    """
    gObj:  GameObject
    vx:    float = 60.0
    vy:    float = 0.0
    kind:  str   = "mushroom"

    @property
    def x(self):      return self.gObj.x
    @property
    def y(self):      return self.gObj.y
    @property
    def width(self):  return self.gObj.width
    @property
    def height(self): return self.gObj.height
    @property
    def active(self): return self.gObj.active

    def update(self, dt: float, context) -> None:
        if not self.active:
            return
        self.vy = min(self.vy + context.GRAVITY * dt, context.MAX_FALL_SPEED)
        self.gObj.x += self.vx * dt
        self.gObj.y += self.vy * dt

    def render(self, surface: pygame.Surface, sx: float, sy: float,
               debug: bool = False) -> None:
        w = self.gObj.width
        h = self.gObj.height
        # Cap (top half) — red
        pygame.draw.ellipse(surface, COLOR_POWERUP_MUSH,
                            (int(sx), int(sy), w, h * 2 // 3))
        # Stem (bottom third) — cream
        pygame.draw.rect(surface, (255, 230, 180),
                         (int(sx + w // 4), int(sy + h // 2), w // 2, h // 2))
        # White spots on cap
        spot_r = max(2, w // 8)
        pygame.draw.circle(surface, (255, 255, 255),
                           (int(sx + w // 3), int(sy + h // 5)), spot_r)
        pygame.draw.circle(surface, (255, 255, 255),
                           (int(sx + w * 2 // 3), int(sy + h // 5)), spot_r)
        if debug:
            pygame.draw.rect(surface, (255, 0, 0),
                             (int(sx), int(sy), w, h), 1)