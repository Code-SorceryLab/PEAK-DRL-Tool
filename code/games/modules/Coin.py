from __future__ import annotations
from dataclasses import dataclass
from .GameObject import GameObject
import pygame
from .Map_parameters import COLOR_COIN, COLOR_BLACK

@dataclass
class Coin():
    gObj: GameObject
    color = COLOR_COIN
    radius = 8
    collected: bool = False
    animation: int = 0
    flyup: bool = False
    vy: float = -280.0 # Pixels per sec
    life: float = 0.3 # Seconds
    auto_collect: bool = False

    # --- Properties for Hash ---
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
            self.vy += 900.0 * dt # Gravity
            self.life -= dt
        self.animation = (self.animation + 1) % 60

    def render(self, surface:pygame.Surface, sx:float, sy:float, debug:bool = False):
        pygame.draw.circle(surface, self.color, (int(sx), int(sy)), self.radius)
        pygame.draw.circle(surface, COLOR_BLACK, (int(sx), int(sy)), self.radius, 2)