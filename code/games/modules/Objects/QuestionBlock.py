from __future__ import annotations
from typing import Tuple
from dataclasses import dataclass
import pygame
from .GameObject import GameObject
from ..Parameters import Map_parameters as MapP

@dataclass
class QuestionBlock:
    gObj: GameObject
    contains: str = "coin"
    hit: bool = False
    solid: bool = True   # QBlocks are solid — required by _get_tile_rects_near filter
    hidden: bool = False # the invisible 1-UP block: no tile until bumped from below
    deadline: int = -1   # the 10-coin brick: frame its coins stop (set on the first bump)
    
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
    
    def tc(self) -> Tuple[int, int]:
        return (self.x // MapP.TILE_SIZE, self.y // MapP.TILE_SIZE)
    
    def render(self, surface: pygame.Surface, sx: float, sy: float):
        if self.hidden:
            return
        from ... import mario_art
        if self.hit:
            name = "question_block_used"
        elif self.contains in ("brick", "coins", "starbrick"):   # looks like any brick
            name = "brick"
        else:
            name = f"question_block_{mario_art.tick(2, 400)}"    # the ? glows
        mario_art.draw(surface, name, sx, sy, MapP.TILE_SIZE, MapP.TILE_SIZE)