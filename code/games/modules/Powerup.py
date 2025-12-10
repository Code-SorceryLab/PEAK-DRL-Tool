from __future__ import annotations
from dataclasses import dataclass
from .GameObject import GameObject
from .Movement_parameters import GRAVITY, MAX_FALL_SPEED
from .Map_parameters import COLOR_POWERUP_MUSH, COLOR_POWERUP_STAR
import pygame

@dataclass
class Powerup():
    gObj: GameObject
    color = (0,0,0)
    vx: float = 60.0 # Pixels per second
    vy: float = 0.0
    kind: str = "mushroom"

    # --- Properties for Hash/Core compatibility ---
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
        """
        Update physics with a list of nearby Tile objects.
        """
        if not self.active:
            return
        
        # 1. Gravity
        self.vy += GRAVITY * dt
        self.vy = min(self.vy, MAX_FALL_SPEED)

        # 2. Move X
        self.gObj.x += self.vx * dt
        self._resolve_x(tiles)

        # 3. Move Y
        self.gObj.y += self.vy * dt
        self._resolve_y(tiles)

    def _resolve_x(self, tiles: list):
        my_rect = self.gObj.get_rect()
        for tile in tiles:
            if not tile.solid: continue
            
            tile_rect = tile.gObj.get_rect()
            if my_rect.colliderect(tile_rect):
                if self.vx < 0: # Moving Left
                    self.gObj.x = tile_rect.right
                    self.vx *= -1.0
                elif self.vx > 0: # Moving Right
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
                    # Land on top
                    self.gObj.y = tile_rect.top - self.gObj.height
                    self.vy = 0
                my_rect = self.gObj.get_rect()

    def render(self, surface: pygame.Surface, sx: float, sy: float, debug:bool = False):
        col = COLOR_POWERUP_MUSH if self.kind == "mushroom" else COLOR_POWERUP_STAR
        pygame.draw.rect(surface, col, (sx, sy, self.gObj.width, self.gObj.height))
        if debug:
            self._debug(surface, sx, sy)

    def _debug(self, surface: pygame.Surface, sx: float, sy: float):
        pygame.draw.rect(surface, (255, 255, 0), (sx, sy, self.gObj.width, self.gObj.height), 1)