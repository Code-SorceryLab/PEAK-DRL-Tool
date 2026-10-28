from __future__ import annotations
from dataclasses import dataclass
from .GameObject import GameObject
import pygame


# 1-UP green
_COL_BODY = ( 50, 200,  80)
_COL_SPOT = (255, 255, 255)
_COL_STEM = (200, 240, 180)


@dataclass
class LifeUp:
    """
    1-UP powerup — walks and falls identically to Mushroom.
    Collected by touching → core.lives += 1  (handled in PhysicsManager).
    Handled by PhysicsManager._solve_aabb_collision (bounce_x=True).
    """
    gObj:  GameObject
    vx:    float = 120.0   # SMB: 1 px/frame
    vy:    float = 0.0
    kind:  str   = "life"

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
        from ... import mario_art                   # the 1-UP
        mario_art.draw(surface, "one_up", sx, sy, self.gObj.width, self.gObj.height)
        if debug:
            pygame.draw.rect(surface, (0, 255, 0), (int(sx), int(sy), self.gObj.width, self.gObj.height), 1)
