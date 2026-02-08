from __future__ import annotations
from dataclasses import dataclass
from .GameObject import GameObject
import pygame

# FIX: Import from sibling Parameters package
from ..Parameters.Map_parameters import TILE_SIZE, COLOR_GROUND, TILE_AIR

@dataclass
class Tile:
    gObj: GameObject
    type_id: int
    solid: bool = False
    color: tuple = COLOR_GROUND

    @property
    def x(self): return self.gObj.x
    @property
    def y(self): return self.gObj.y
    @property
    def width(self): return self.gObj.width
    @property
    def height(self): return self.gObj.height

    def render(self, surface: pygame.Surface, cam_x: float, cam_y: float):
        if self.type_id == TILE_AIR: return
        sx = self.x - cam_x
        sy = self.y - cam_y
        
        # Optimization: Don't draw if off screen
        if sx < -TILE_SIZE or sx > surface.get_width() or sy < -TILE_SIZE or sy > surface.get_height():
            return

        pygame.draw.rect(surface, self.color, (sx, sy, self.width, self.height))

def create_tile(type_id: int, x: int, y: int, solid: bool, color: tuple) -> Tile:
    return Tile(
        gObj=GameObject(float(x), float(y), TILE_SIZE, TILE_SIZE, True),
        type_id=type_id,
        solid=solid,
        color=color
    )