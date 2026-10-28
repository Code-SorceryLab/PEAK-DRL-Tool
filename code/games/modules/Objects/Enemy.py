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
    on_ground: bool = False
    facing_right: bool = False
    hp: int = 1
    hp_max: int = 1
    contact_damage: int = 1
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
        self.facing_right = self.vx >= 0.0
        
        # Use context parameters for gravity
        grav = context.GRAVITY
        max_fall = context.MAX_FALL_SPEED

        self.vy += grav * dt
        self.vy = min(self.vy, max_fall)

        # Integration only - Collision logic moved to PhysicsManager
        self.gObj.y += self.vy * dt
        self.gObj.x += self.vx * dt

    def render(self, surface: pygame.Surface, sx: float, sy: float, debug: bool = False):
        from ... import mario_art                   # the imp, its feet stepping as it walks
        mario_art.draw(surface, f"imp_{int(self.gObj.x // 8) % 2}", sx, sy, self.gObj.width, self.gObj.height,
                       flip=self.vx > 0)
