from enum import Enum, auto

class EntityType(Enum):
    NONE = auto()
    PLAYER = auto()
    ENEMY = auto()
    COIN = auto()
    POWERUP = auto()
    QBLOCK = auto()
    TILE = auto()
    GOAL = auto()
    SPIKE = auto()
    PROJECTILE = auto()
    PIT = auto()