from __future__ import annotations
from dataclasses import dataclass
from .GameObject import GameObject
import pygame


# 1-UP green
_COL_BODY = ( 50, 200,  80)
_COL_SPOT = (255, 255, 255)
_COL_STEM = (200, 240, 180)


@dataclass
class LifeUp:
    """
    1-UP powerup — walks and falls identically to Mushroom.
    Collected by touching → core.lives += 1  (handled in PhysicsManager).
    Handled by PhysicsManager._solve_aabb_collision (bounce_x=True).
    """
    gObj:  GameObject
    vx:    float = 60.0
    vy:    float = 0.0
    kind:  str   = "life"

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
        if not self.active:
            return
        self.vy = min(self.vy + context.GRAVITY * dt, context.MAX_FALL_SPEED)
        self.gObj.x += self.vx * dt
        self.gObj.y += self.vy * dt

    def render(self, surface: pygame.Surface, sx: float, sy: float,
               debug: bool = False) -> None:
        w = self.gObj.width
        h = self.gObj.height
        # Green mushroom cap
        pygame.draw.ellipse(surface, _COL_BODY,
                            (int(sx), int(sy), w, h * 2 // 3))
        # Stem
        pygame.draw.rect(surface, _COL_STEM,
                         (int(sx + w // 4), int(sy + h // 2), w // 2, h // 2))
        # "1" text on cap
        try:
            font = pygame.font.SysFont(None, max(10, w // 2))
            txt  = font.render("1", True, (0, 80, 20))
            surface.blit(txt, (int(sx + w // 2 - txt.get_width() // 2),
                               int(sy + h // 6 - txt.get_height() // 2)))
        except Exception:
            # Fallback: white dot
            pygame.draw.circle(surface, _COL_SPOT,
                               (int(sx + w // 2), int(sy + h // 4)),
                               max(2, w // 6))
        if debug:
            pygame.draw.rect(surface, (0, 255, 0),
                             (int(sx), int(sy), w, h), 1)