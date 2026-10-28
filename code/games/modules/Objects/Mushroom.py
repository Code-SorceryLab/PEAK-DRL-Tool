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
    vx:    float = 120.0   # SMB: 1 px/frame
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
        from ... import mario_art                   # the mushroom
        mario_art.draw(surface, "mushroom", sx, sy, self.gObj.width, self.gObj.height)
        if debug:
            pygame.draw.rect(surface, (255, 0, 0), (int(sx), int(sy), self.gObj.width, self.gObj.height), 1)
