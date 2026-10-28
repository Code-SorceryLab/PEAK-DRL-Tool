from __future__ import annotations
from dataclasses import dataclass
from .GameObject import GameObject
import math
import pygame


_COL_PETALS = (255,  60,  20)   # red-orange petals
_COL_CENTRE = (255, 240,  60)   # yellow centre
_COL_STEM   = ( 40, 180,  40)   # green stem
_COL_LEAF   = ( 60, 210,  60)   # lighter leaf


@dataclass
class FireFlower:
    """
    Fire flower powerup — completely stationary, no gravity, no movement.

    Stays exactly where it was spawned (on top of a QBlock) until the
    player walks into it.  PhysicsManager._resolve_powerup_world is still
    called but has no effect because vx=vy=0.

    Collected by touching → player.power_machine.collect_flower().
    """
    gObj:  GameObject
    vx:    float = 0.0    # stationary
    vy:    float = 0.0    # stationary
    kind:  str   = "flower"

    # Internal animation timer (bob + petal pulse)
    _t:    float = 0.0

    @property
    def x(self):      return self.gObj.x
    @property
    def y(self):      return self.gObj.y
    @property
    def width(self):  return self.gObj.width
    @property
    def height(self): return self.gObj.height
    @property
    def active(self): return self.gObj.active

    def update(self, dt: float, context) -> None:
        """No physics — just tick the animation timer."""
        if not self.active:
            return
        self._t += dt

    def render(self, surface: pygame.Surface, sx: float, sy: float,
               debug: bool = False) -> None:
        from ... import mario_art                   # the fire flower, bobbing gently
        bob = math.sin(self._t * 3.0) * 2.0
        mario_art.draw(surface, "fire_flower", sx, sy + bob, self.gObj.width, self.gObj.height)
        if debug:
            pygame.draw.rect(surface, (255, 100, 0), (int(sx), int(sy), self.gObj.width, self.gObj.height), 1)
