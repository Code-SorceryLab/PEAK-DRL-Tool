from __future__ import annotations
from dataclasses import dataclass
from .GameObject import GameObject
from .Movement_parameters import GRAVITY, MAX_FALL_SPEED
from .Map_parameters import COLOR_ENEMY
import pygame

@dataclass
class Enemy():
    gObj: GameObject
    vx: float = -60.0  
    vy: float = 0.0
    color = COLOR_ENEMY
    bounce_clock: int = 0 

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
        if not self.gObj.active:
            return
        
        # 1. Gravity
        self.vy += GRAVITY * dt
        self.vy = min(self.vy, MAX_FALL_SPEED)

        # 3. Move Y
        self.gObj.y += self.vy * dt
        self._resolve_y(tiles)

        # 4. Move X # 2. Move X
        self.gObj.x += self.vx * dt
        self._resolve_x(tiles)
        
        
    def _resolve_x(self, tiles: list):
        my_rect = self.gObj.get_rect()
        
        for tile in tiles:
            # Filter: Only collide with SOLID tiles
            if not tile.solid: 
                continue

            # FIX: Get the Rect from the Tile's GameObject
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
            if not tile.solid: 
                continue

            # FIX: Get the Rect from the Tile's GameObject
            tile_rect = tile.gObj.get_rect()

            if my_rect.colliderect(tile_rect):
                if self.vy >= 0: 
                    # Land on top
                    self.gObj.y = tile_rect.top - self.gObj.height
                    self.vy = 0
                elif self.vy < 0: 
                    # Head bonk
                    self.gObj.y = tile_rect.bottom
                    self.vy = 0
                
                my_rect = self.gObj.get_rect()

    def render(self, surface: pygame.Surface, sx: float, sy: float, debug: bool = False):
        pygame.draw.rect(surface, self.color, (sx, sy, self.gObj.width, self.gObj.height))
        
        # Eyes
        eye_offset = 4 if self.vx > 0 else 14
        pygame.draw.rect(surface, (255, 255, 255), (sx + eye_offset, sy + 4, 4, 8))

        if debug:
            self._debug(surface, sx, sy)

    def _debug(self, surface: pygame.Surface, sx: float, sy: float):
        pygame.draw.rect(surface, (255, 0, 0), (sx, sy, self.gObj.width, self.gObj.height), 1)