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
        w  = self.gObj.width
        h  = self.gObj.height

        # Gentle vertical bob
        bob = math.sin(self._t * 3.0) * 2.0
        sx_i = int(sx)
        sy_i = int(sy + bob)

        cx = sx_i + w // 2

        # ── Stem ──────────────────────────────────────────────────────────────
        stem_x  = cx - 1
        stem_y0 = sy_i + h * 2 // 3
        stem_y1 = sy_i + h
        pygame.draw.line(surface, _COL_STEM,
                         (stem_x, stem_y0), (stem_x, stem_y1), 3)

        # ── Leaf ──────────────────────────────────────────────────────────────
        leaf_y = sy_i + h * 3 // 4
        pygame.draw.ellipse(surface, _COL_LEAF,
                            (stem_x - 6, leaf_y - 3, 8, 5))

        # ── Petals (4 around centre) ───────────────────────────────────────────
        petal_r  = max(3, w // 4)
        orbit_r  = petal_r + 1
        head_cy  = sy_i + h // 3
        pulse    = 1.0 + 0.08 * math.sin(self._t * 5.0)
        pr       = int(petal_r * pulse)
        for i in range(4):
            angle = math.pi / 2 * i
            px    = cx      + int(orbit_r * math.cos(angle))
            py    = head_cy + int(orbit_r * math.sin(angle))
            pygame.draw.circle(surface, _COL_PETALS, (px, py), pr)

        # ── Centre ────────────────────────────────────────────────────────────
        centre_r = max(2, w // 5)
        pygame.draw.circle(surface, _COL_CENTRE, (cx, head_cy), centre_r)

        if debug:
            pygame.draw.rect(surface, (255, 100, 0),
                             (int(sx), int(sy), w, h), 1)