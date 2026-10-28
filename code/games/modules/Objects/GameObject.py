from __future__ import annotations
from dataclasses import dataclass
from ..System.EntityType import EntityType
import pygame

@dataclass
class GameObject:
    x: float 
    y: float
    width: int 
    height: int 
    active: bool = True
    type_id = EntityType.NONE
    def get_rect(self) -> pygame.Rect:
        return pygame.Rect(int(self.x), int(self.y), self.width, self.height)
    def collides_with(self, other: "GameObject") -> bool:
        return self.active and other.active and self.get_rect().colliderect(other.get_rect())
    def count_kill(self, core) -> None:
        """Credit this enemy's kill once: a Koopa's wings, its revived shell or a second hit is the same enemy."""
        if not getattr(self, "kill_counted", False) and hasattr(core, "kills_step"):
            self.kill_counted = True
            core.kills_step += 1