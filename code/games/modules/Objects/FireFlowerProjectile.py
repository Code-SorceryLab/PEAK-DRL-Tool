from __future__ import annotations
import pygame
from dataclasses import dataclass
from ..Objects.GameObject import GameObject
from ..Objects.Projectile import Projectile
from ..System.EntityType import EntityType

# ── Tunable parameters ────────────────────────────────────────────────────────
# Horizontal speed (px/s).  Positive = rightward; sign is flipped for left-facing shots.
PROJECTILE_SPEED    = 480.0   # SMB: 4 px/frame

# Upward velocity applied on every ground bounce (negative = up in screen coords).
PROJ_JUMP_VEL       = -360.0  # SMB: 3 px/frame up off the floor

# Thrown downward: SMB's FireballObjCore sets Fireball_Y_Speed = $04 (4 px/frame down).
PROJ_LAUNCH_VEL     = 480.0

# The fireball's own gravity and fall cap, not Mario's: ImposeGravity with force $50 (/256 px per
# frame²) and maximum 3 px/frame. So it skips along in low hops, about half a tile high.
PROJ_GRAVITY        = 2250.0
PROJ_MAX_FALL_SPEED = 360.0

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
        launch_vel: float = PROJ_LAUNCH_VEL,
    ) -> "FireFlowerProjectile":
        """
        Spawn a projectile at the player's head height (SMB: Fireball_Y = Player_Y), travelling
        down and forward in the direction they are facing.

        Parameters
        ----------
        speed      : horizontal speed in px/s (sign is applied automatically).
        launch_vel : initial downward vy in px/s (positive = down).
        """
        cx = player.gObj.x + player.gObj.width  / 2 - PROJECTILE_WIDTH  / 2
        cy = player.gObj.y
        vx = speed if player.facing_right else -speed
        return cls(
            gObj=GameObject(cx, cy, PROJECTILE_WIDTH, PROJECTILE_HEIGHT, True),
            vx=vx,
            vy=launch_vel,  # thrown toward the ground
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

        # The fireball's own gravity (SMB gives it its own, not Mario's).
        self.vy = min(self.vy + PROJ_GRAVITY * dt, PROJ_MAX_FALL_SPEED)

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
        from ... import mario_art                   # a magic bolt
        mario_art.draw(surface, f"magic_bolt_{mario_art.tick(2, 60)}", screen_x, screen_y,
                       PROJECTILE_WIDTH, PROJECTILE_HEIGHT)
