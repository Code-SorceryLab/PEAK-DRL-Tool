from __future__ import annotations
from dataclasses import dataclass
from .GameObject import GameObject
import pygame
from ..Parameters import Map_parameters as MapP

@dataclass
class Coin():
    gObj: GameObject
    color = MapP.COLOR_COIN
    radius = 8
    collected: bool = False
    animation: int = 0
    flyup: bool = False
    vy: float = -280.0 
    life: float = 0.3 
    auto_collect: bool = False

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

    def update(self, dt: float = 0.016):
        if self.flyup:
            self.gObj.y += self.vy * dt
            self.vy += 900.0 * dt 
            self.life -= dt
        self.animation = (self.animation + 1) % 60

    def render(self, surface:pygame.Surface, sx:float, sy:float, debug:bool = False):
        pygame.draw.circle(surface, self.color, (int(sx), int(sy)), self.radius)
        pygame.draw.circle(surface, MapP.COLOR_BLACK, (int(sx), int(sy)), self.radius, 2)