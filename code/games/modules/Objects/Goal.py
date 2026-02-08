from .Tile import Tile
from .GameObject import GameObject
from ..System.EntityType import EntityType
from ..Parameters.Map_parameters import COLOR_GOAL
import pygame

class Goal(Tile):
    def __init__(self, gObj: GameObject):
        # Initialize as a non-solid Tile with GOAL type and color
        super().__init__(gObj=gObj, type_id=EntityType.GOAL, solid=False, color=COLOR_GOAL)
        
        # Explicitly set the inner GameObject type for physics interactions
        self.gObj.type_id = EntityType.GOAL
        
    def update(self, dt, context=None):
        # Placeholder for potential future animations (e.g., flag waving)
        pass
        
    def render(self, surface: pygame.Surface, sx: float, sy: float, debug: bool = False):
        # Override Tile.render to match the Entity render signature 
        # (receives pre-calculated screen coordinates sx, sy)
        pygame.draw.rect(surface, self.color, (sx, sy, self.width, self.height))