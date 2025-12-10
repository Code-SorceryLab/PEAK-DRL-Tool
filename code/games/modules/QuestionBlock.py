from __future__ import annotations
from typing import Tuple
from dataclasses import dataclass
from .Map_parameters import TILE_SIZE, COLOR_QBLOCK, COLOR_BLACK, COLOR_WHITE
import pygame
from .GameObject import GameObject

@dataclass
class QuestionBlock:
    
    gObj: GameObject
    contains: str = "coin"
    hit: bool = False
    
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
        return (self.x // TILE_SIZE, self.y // TILE_SIZE)
    
    def render(self, surface: pygame.Surface, sx: float, sy: float):
        pygame.draw.rect(surface, COLOR_QBLOCK, (sx, sy, TILE_SIZE, TILE_SIZE))
        pygame.draw.rect(surface, COLOR_BLACK, (sx, sy, TILE_SIZE, TILE_SIZE), 1)
        # self.hit = False
        # for qb in self.qblocks:
        #     qc, qr = qb.tc()
        #     if qc == c and qr == r:
        #         hit = qb.hit
        #         break
        if not self.hit:
            font = pygame.font.Font(None, 26)
            q = font.render("?", True, COLOR_WHITE)
            surface.blit(q, q.get_rect(center=(sx + TILE_SIZE // 2, sy + TILE_SIZE // 2)))
        return