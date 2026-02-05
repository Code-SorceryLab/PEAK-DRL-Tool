from __future__ import annotations
from dataclasses import dataclass
from .GameObject import GameObject
import pygame

from ..Parameters import Movement_parameters as MP
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

    def update(self, dt: float, tiles: list):
        if not self.active: return
        
        self.vy += MP.GRAVITY * dt
        self.vy = min(self.vy, MP.MAX_FALL_SPEED)

        self.gObj.x += self.vx * dt
        self._resolve_x(tiles)

        self.gObj.y += self.vy * dt
        self._resolve_y(tiles)

    def _resolve_x(self, tiles: list):
        my_rect = self.gObj.get_rect()
        for tile in tiles:
            if not tile.solid: continue
            tile_rect = tile.gObj.get_rect()
            if my_rect.colliderect(tile_rect):
                if self.vx < 0:
                    self.gObj.x = tile_rect.right
                    self.vx *= -1.0
                elif self.vx > 0:
                    self.gObj.x = tile_rect.left - self.gObj.width
                    self.vx *= -1.0
                my_rect = self.gObj.get_rect()

    def _resolve_y(self, tiles: list):
        my_rect = self.gObj.get_rect()
        for tile in tiles:
            if not tile.solid: continue
            tile_rect = tile.gObj.get_rect()
            if my_rect.colliderect(tile_rect):
                if self.vy >= 0: 
                    self.gObj.y = tile_rect.top - self.gObj.height
                    self.vy = 0
                my_rect = self.gObj.get_rect()

    def render(self, surface: pygame.Surface, sx: float, sy: float, debug:bool = False):
        col = MapP.COLOR_POWERUP_MUSH if self.kind == "mushroom" else MapP.COLOR_POWERUP_STAR
        pygame.draw.rect(surface, col, (sx, sy, self.gObj.width, self.gObj.height))
        if debug:
            self._debug(surface, sx, sy)

    def _debug(self, surface: pygame.Surface, sx: float, sy: float):
        pygame.draw.rect(surface, (255, 255, 0), (sx, sy, self.gObj.width, self.gObj.height), 1)