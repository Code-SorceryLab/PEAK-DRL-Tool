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
        if not self.hit and self.contains in ("brick", "coins", "starbrick"):   # looks like any brick
            pygame.draw.rect(surface, MapP.COLOR_PLATFORM, (sx, sy, MapP.TILE_SIZE, MapP.TILE_SIZE))
            pygame.draw.rect(surface, MapP.COLOR_BLACK, (sx, sy, MapP.TILE_SIZE, MapP.TILE_SIZE), 1)
            return
        pygame.draw.rect(surface, MapP.COLOR_QBLOCK, (sx, sy, MapP.TILE_SIZE, MapP.TILE_SIZE))
        pygame.draw.rect(surface, MapP.COLOR_BLACK, (sx, sy, MapP.TILE_SIZE, MapP.TILE_SIZE), 1)

        if not self.hit:
            font = pygame.font.Font(None, 26)
            q = font.render("?", True, MapP.COLOR_WHITE)
            surface.blit(q, q.get_rect(center=(sx + MapP.TILE_SIZE // 2, sy + MapP.TILE_SIZE // 2)))
        return