from __future__ import annotations

from dataclasses import dataclass, field

from .GameObject import GameObject
from ..System.EntityType import EntityType


@dataclass
class Projectile:
    gObj: GameObject
    vx: float = 0.0
    vy: float = 0.0
    on_ground: bool = False
    owner: str = "player"
    damage: int = 1
    lifetime: float = 0.0
    age: float = field(default=0.0, repr=False)
    prev_x: float = field(default=0.0, repr=False)
    prev_y: float = field(default=0.0, repr=False)

    def __post_init__(self):
        self.gObj.type_id = EntityType.PROJECTILE
        self.prev_x = float(self.gObj.x)
        self.prev_y = float(self.gObj.y)

    def begin_frame(self):
        self.prev_x = float(self.gObj.x)
        self.prev_y = float(self.gObj.y)

    def tick_lifetime(self, dt: float) -> bool:
        self.age += float(dt)
        if self.lifetime > 0.0 and self.age >= self.lifetime:
            self.gObj.active = False
            return False
        return True
