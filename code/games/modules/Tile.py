from __future__ import annotations
from dataclasses import dataclass
from .GameObject import GameObject
from .Map_parameters import COLOR_GROUND, COLOR_PLATFORM, COLOR_GOAL, COLOR_SPIKE, TILE_SIZE
import pygame

@dataclass
class Tile:
    gObj: GameObject
    type_id: int
    solid: bool = False
    color: tuple = (255, 255, 255)
    
    # --- ADD THESE PROPERTIES ---
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

    def render(self, surface: pygame.Surface, camera_x: float, camera_y: float):
        sx = self.gObj.x - camera_x
        sy = self.gObj.y - camera_y
        
        # Simple optimization: don't draw if off screen
        if sx + TILE_SIZE < 0 or sx > surface.get_width():
            return
            
        pygame.draw.rect(surface, self.color, (sx, sy, self.gObj.width, self.gObj.height))
        # Optional outline
        pygame.draw.rect(surface, (0,0,0), (sx, sy, self.gObj.width, self.gObj.height), 1)

# Factory / Helper to create specific tiles
def create_tile(type_id: int, x: int, y: int, solid: bool, color: tuple) -> Tile:
    return Tile(
        gObj=GameObject(float(x), float(y), TILE_SIZE, TILE_SIZE),
        type_id=type_id,
        solid=solid,
        color=color
    )