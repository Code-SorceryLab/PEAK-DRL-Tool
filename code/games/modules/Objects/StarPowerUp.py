from __future__ import annotations
from dataclasses import dataclass, field
from .GameObject import GameObject
import pygame
from ..Parameters.Map_parameters import COLOR_POWERUP_STAR


# How often the star auto-jumps (seconds)
_JUMP_INTERVAL = 0.55
# Jump strength (negative = upward)
_JUMP_VEL      = -320.0
# Horizontal walk speed
_WALK_SPEED    = 80.0


@dataclass
class StarPowerUp:
    """
    Star powerup — bounces across the ground, jumping at regular intervals.

    Movement
    --------
    - Horizontal: constant vx, reversed on wall contact (handled by
      PhysicsManager._solve_aabb_collision bounce_x=True).
    - Vertical: gravity applied normally; star jumps every _JUMP_INTERVAL
      seconds when on or near the ground (vy >= 0).

    Collected by touching → player.power_machine.collect_star().
    """
    gObj:        GameObject
    vx:          float = _WALK_SPEED
    vy:          float = _JUMP_VEL     # start with an initial jump
    kind:        str   = "star"

    _jump_timer: float = field(default=0.0, init=False, repr=False)
    on_ground:   bool  = False   # set by PhysicsManager._solve_aabb_collision

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

        # Gravity
        self.vy = min(self.vy + context.GRAVITY * dt, context.MAX_FALL_SPEED)

        # Auto-jump: on_ground was set True by PhysicsManager after last frame's
        # collision resolution — same pattern as Player.update(). Check it here
        # (before clearing) so we always read the value PhysicsManager wrote.
        self._jump_timer += dt
        if self._jump_timer >= _JUMP_INTERVAL and self.on_ground:
            self.vy          = _JUMP_VEL
            self._jump_timer = 0.0

        self.gObj.x += self.vx * dt
        self.gObj.y += self.vy * dt
        self.on_ground = False   # PhysicsManager sets True again if still grounded

    def render(self, surface: pygame.Surface, sx: float, sy: float,
               debug: bool = False) -> None:
        w  = self.gObj.width
        h  = self.gObj.height
        cx = int(sx + w // 2)
        cy = int(sy + h // 2)
        r  = min(w, h) // 2

        # Outer glow
        glow = pygame.Surface((w + 8, h + 8), pygame.SRCALPHA)
        pygame.draw.circle(glow, (*COLOR_POWERUP_STAR, 60),
                           (w // 2 + 4, h // 2 + 4), r + 4)
        surface.blit(glow, (int(sx) - 4, int(sy) - 4))

        # 5-pointed star
        import math
        points = []
        for i in range(10):
            angle  = math.pi / 5 * i - math.pi / 2
            radius = r if i % 2 == 0 else r * 0.45
            points.append((
                cx + int(radius * math.cos(angle)),
                cy + int(radius * math.sin(angle))
            ))
        pygame.draw.polygon(surface, COLOR_POWERUP_STAR, points)
        pygame.draw.polygon(surface, (255, 255, 200), points, 1)

        if debug:
            pygame.draw.rect(surface, (255, 255, 0),
                             (int(sx), int(sy), w, h), 1)