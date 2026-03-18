from __future__ import annotations

from dataclasses import dataclass

import pygame

from .GameObject import GameObject
from ..System.EntityType import EntityType


@dataclass
class Ladder:
    gObj: GameObject

    def __post_init__(self):
        self.gObj.type_id = EntityType.LADDER

    @property
    def x(self):
        return self.gObj.x

    @property
    def y(self):
        return self.gObj.y

    @classmethod
    def from_tile(cls, x: float, y: float, tile_size: int) -> "Ladder":
        return cls(GameObject(x, y, tile_size, tile_size, False))

    def render(self, surface: pygame.Surface, sx: float, sy: float):
        rail = (240, 220, 110)
        rung = (255, 246, 182)
        w = self.gObj.width
        h = self.gObj.height
        left_x = int(sx + w * 0.28)
        right_x = int(sx + w * 0.72)
        pygame.draw.line(surface, rail, (left_x, int(sy)), (left_x, int(sy + h)), 3)
        pygame.draw.line(surface, rail, (right_x, int(sy)), (right_x, int(sy + h)), 3)
        for i in range(1, 4):
            ry = int(sy + i * (h / 4))
            pygame.draw.line(surface, rung, (left_x - 1, ry), (right_x + 1, ry), 2)
