from __future__ import annotations
from dataclasses import dataclass
from .GameObject import GameObject
import pygame

from ..Parameters import Map_parameters as MapP

@dataclass
class Powerup():
    gObj: GameObject
    color = (0,0,0)
    vx: float = 60.0 
    vy: float = 0.0
    kind: str = "mushroom"

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

    def update(self, dt: float, context):
        if not self.active: return
        
        grav = context.GRAVITY
        max_fall = context.MAX_FALL_SPEED
        
        self.vy += grav * dt
        self.vy = min(self.vy, max_fall)

        # Integration only - Collision logic moved to PhysicsManager
        self.gObj.x += self.vx * dt
        self.gObj.y += self.vy * dt

    def render(self, surface: pygame.Surface, sx: float, sy: float, debug:bool = False):
        col = MapP.COLOR_POWERUP_MUSH if self.kind == "mushroom" else MapP.COLOR_POWERUP_STAR
        pygame.draw.rect(surface, col, (sx, sy, self.gObj.width, self.gObj.height))