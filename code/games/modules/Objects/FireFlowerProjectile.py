from __future__ import annotations
import pygame
from dataclasses import dataclass, field
from ..Objects.GameObject import GameObject
from ..System.EntityType import EntityType

PROJECTILE_SPEED  = 420.0   # px/s
PROJECTILE_WIDTH  = 12
PROJECTILE_HEIGHT = 12
PROJECTILE_LIFETIME = 3.0   # seconds before auto-expiry


@dataclass
class FireFlowerProjectile:
    """
    A fireball fired by the player when in Fire state.

    Spawned via FireFlowerProjectile.from_player(player).
    Travels horizontally at a fixed speed in the direction the player was facing.
    Deactivates on:
      - hitting an enemy  (handled by PhysicsManager)
      - hitting a solid wall (handled by PhysicsManager._resolve_projectile_world)
      - exceeding its lifetime
    """
    gObj: GameObject
    vx:   float = 0.0
    vy:   float = 0.0
    _age: float = field(default=0.0, repr=False)

    def __post_init__(self):
        self.gObj.type_id = EntityType.PROJECTILE

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_player(cls, player) -> "FireFlowerProjectile":
        """
        Spawn a projectile centred on the player, travelling in the direction
        they are currently facing.
        """
        cx = player.gObj.x + player.gObj.width  / 2 - PROJECTILE_WIDTH  / 2
        cy = player.gObj.y + player.gObj.height / 2 - PROJECTILE_HEIGHT / 2
        speed = PROJECTILE_SPEED if player.facing_right else -PROJECTILE_SPEED
        proj = cls(
            gObj=GameObject(cx, cy, PROJECTILE_WIDTH, PROJECTILE_HEIGHT, True),
            vx=speed,
            vy=0.0,
        )
        return proj

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, dt: float, _ctx=None):
        if not self.gObj.active:
            return
        self._age += dt
        if self._age >= PROJECTILE_LIFETIME:
            self.gObj.active = False
            return
        self.gObj.x += self.vx * dt
        self.gObj.y += self.vy * dt

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(self, surface: pygame.Surface, screen_x: float, screen_y: float):
        if not self.gObj.active:
            return
        # Outer glow (orange)
        pygame.draw.circle(
            surface,
            (255, 140, 0),
            (int(screen_x + PROJECTILE_WIDTH // 2), int(screen_y + PROJECTILE_HEIGHT // 2)),
            PROJECTILE_WIDTH // 2,
        )
        # Inner core (bright yellow/white)
        pygame.draw.circle(
            surface,
            (255, 255, 180),
            (int(screen_x + PROJECTILE_WIDTH // 2), int(screen_y + PROJECTILE_HEIGHT // 2)),
            PROJECTILE_WIDTH // 4,
        )