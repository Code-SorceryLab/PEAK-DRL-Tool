from __future__ import annotations
import pygame
from dataclasses import dataclass
from ..Objects.GameObject import GameObject
from ..Objects.Projectile import Projectile
from ..System.EntityType import EntityType

# ── Tunable parameters ────────────────────────────────────────────────────────
# Horizontal speed (px/s).  Positive = rightward; sign is flipped for left-facing shots.
PROJECTILE_SPEED    = 280.0

# Upward velocity applied on every ground bounce (negative = up in screen coords).
# Increase magnitude for higher bounces, decrease for flatter arcs.
PROJ_JUMP_VEL       = -260.0

# Gravity acceleration applied each frame (px/s²).
# Matches world gravity for a natural arc; lower values produce floatier fireballs.
PROJ_GRAVITY        = 900.0

# Maximum downward fall speed (px/s).
PROJ_MAX_FALL_SPEED = 480.0

# Seconds before the projectile auto-expires regardless of bounces.
PROJECTILE_LIFETIME = 3.0

PROJECTILE_WIDTH    = 12
PROJECTILE_HEIGHT   = 12
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class FireFlowerProjectile(Projectile):
    """
    A fireball fired by the player when in Fire state.

    Movement
    --------
    - Horizontal : constant vx (set at spawn, preserved across bounces).
    - Vertical   : gravity pulls it down each frame; the moment PhysicsManager
                   sets on_ground=True (floor contact), update() launches an
                   instant upward bounce (PROJ_JUMP_VEL). No jump timer —
                   every landing bounces immediately, giving the classic
                   Mario fireball arc.

    Deactivates on:
      - hitting a solid wall  (PhysicsManager._resolve_projectile_world,
                               bounce_x=False → vx zeroed → deactivates next frame
                               via the vx==0 guard below)
      - hitting an enemy      (PhysicsManager._resolve_dynamic_interactions)
      - exceeding its lifetime

    Spawn via FireFlowerProjectile.from_player(player).
    """
    gObj: GameObject
    vx: float = 0.0
    vy: float = 0.0
    on_ground: bool = False
    owner: str = "player"
    damage: int = 1
    lifetime: float = PROJECTILE_LIFETIME

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_player(
        cls,
        player,
        speed: float    = PROJECTILE_SPEED,
        jump_vel: float = PROJ_JUMP_VEL,
    ) -> "FireFlowerProjectile":
        """
        Spawn a projectile centred on the player, travelling in the direction
        they are currently facing.

        Parameters
        ----------
        speed    : horizontal speed in px/s (sign is applied automatically).
        jump_vel : initial upward vy in px/s (negative = up).
                   Pass a less-negative value for a flatter arc.
        """
        cx = player.gObj.x + player.gObj.width  / 2 - PROJECTILE_WIDTH  / 2
        cy = player.gObj.y + player.gObj.height / 2 - PROJECTILE_HEIGHT / 2
        vx = speed if player.facing_right else -speed
        return cls(
            gObj=GameObject(cx, cy, PROJECTILE_WIDTH, PROJECTILE_HEIGHT, True),
            vx=vx,
            vy=jump_vel,    # launch immediately upward on spawn
            owner="player",
            damage=1,
            lifetime=PROJECTILE_LIFETIME,
        )

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, dt: float, ctx=None):
        if not self.gObj.active:
            return

        # Lifetime expiry
        self.begin_frame()
        if not self.tick_lifetime(dt):
            return

        # Wall hit: PhysicsManager zeroed vx when bounce_x=False.
        # A stationary fireball is dead — deactivate.
        if self.vx == 0.0:
            self.gObj.active = False
            return

        # Gravity — use ctx values when available so level physics config applies.
        gravity        = ctx.GRAVITY        if ctx else PROJ_GRAVITY
        max_fall_speed = ctx.MAX_FALL_SPEED if ctx else PROJ_MAX_FALL_SPEED
        self.vy = min(self.vy + gravity * dt, max_fall_speed)

        # Bounce — PhysicsManager sets on_ground=True after snapping to floor.
        # We read it here (same pattern as StarPowerUp) and immediately re-launch.
        if self.on_ground:
            self.vy = PROJ_JUMP_VEL
        self.on_ground = False   # PhysicsManager sets True again next frame if still grounded

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
