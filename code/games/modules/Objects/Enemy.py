from __future__ import annotations
from dataclasses import dataclass
from .GameObject import GameObject
import pygame

from ..Parameters import Map_parameters as MapP

@dataclass
class Enemy():
    gObj: GameObject
    vx: float = -60.0  
    vy: float = 0.0
    color = MapP.COLOR_ENEMY
    
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
        if not self.gObj.active: return
        
        # Use context parameters for gravity
        grav = context.GRAVITY
        max_fall = context.MAX_FALL_SPEED

        self.vy += grav * dt
        self.vy = min(self.vy, max_fall)

        # Integration only - Collision logic moved to PhysicsManager
        self.gObj.y += self.vy * dt
        self.gObj.x += self.vx * dt

    def render(self, surface: pygame.Surface, sx: float, sy: float, debug: bool = False):
        pygame.draw.rect(surface, self.color, (sx, sy, self.gObj.width, self.gObj.height))
        eye_offset = 4 if self.vx > 0 else 14
        pygame.draw.rect(surface, (255, 255, 255), (sx + eye_offset, sy + 4, 4, 8))