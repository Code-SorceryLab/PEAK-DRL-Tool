"""
SonicPlayer.py
--------------
The Sonic player character for the Sonic NES clone.

Key mechanics unique to Sonic:
  - Spin Dash: Crouch + tap jump to charge, release crouch to launch
  - Rolling: Press down while moving fast → curl into ball
  - Ball Attack: Jumping or rolling = invulnerable to badniks from sides/above
  - Ring Loss: Getting hit while holding rings scatters them (not instant death)
  - Speed Tiers: Walk → Run → Top Speed with visual feedback
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
import math
import numpy as np
import pygame

try:
    from .GameObject import GameObject
    from ..Parameters.Sonic_Map_parameters import (
        COLOR_SONIC_BLUE, COLOR_SONIC_SKIN, COLOR_SONIC_SHOE,
        COLOR_SONIC_BALL, COLOR_WHITE, COLOR_BLACK, COLOR_STREAK, TILE_SIZE
    )
    from ..Parameters.Sonic_Movement_parameters import (
        RUN_ACCEL, WALK_ACCEL, MAX_WALK_SPEED, MAX_RUN_SPEED, TOP_SPEED,
        GROUND_FRICTION, AIR_FRICTION, ROLL_FRICTION, AIR_CONTROL,
        SKID_DECEL, GRAVITY, FAST_FALL_GRAV, MAX_FALL_SPEED,
        SPIN_DASH_MIN, SPIN_DASH_MAX, SPIN_DASH_CHARGE_RATE,
        SPIN_DASH_CHARGES_MAX, ROLL_SPEED_THRESHOLD
    )
    from ..Parameters.Sonic_Jump_parameters import (
        JUMP_VEL_MIN, JUMP_VEL_MAX, JUMP_HOLD_FRAMES,
        SPEED_JUMP_BONUS, COYOTE_FRAMES, JUMP_BUFFER_FRAMES, BOUNCE_VEL
    )
except ImportError:
    from GameObject import GameObject
    TILE_SIZE = 32
    COLOR_SONIC_BLUE = (30, 50, 220)
    COLOR_SONIC_SKIN = (255, 200, 150)
    COLOR_SONIC_SHOE = (255, 50, 50)
    COLOR_SONIC_BALL = (20, 40, 180)
    COLOR_WHITE = (255, 255, 255)
    COLOR_BLACK = (0, 0, 0)
    COLOR_STREAK = (200, 200, 255)
    RUN_ACCEL = 600.0; WALK_ACCEL = 480.0
    MAX_WALK_SPEED = 180.0; MAX_RUN_SPEED = 380.0; TOP_SPEED = 560.0
    GROUND_FRICTION = 600.0; AIR_FRICTION = 100.0; ROLL_FRICTION = 200.0
    AIR_CONTROL = 0.40; SKID_DECEL = 1800.0
    GRAVITY = 1100.0; FAST_FALL_GRAV = 2000.0; MAX_FALL_SPEED = 600.0
    SPIN_DASH_MIN = 300.0; SPIN_DASH_MAX = 560.0
    SPIN_DASH_CHARGE_RATE = 60.0; SPIN_DASH_CHARGES_MAX = 8
    ROLL_SPEED_THRESHOLD = 60.0
    JUMP_VEL_MIN = -520.0; JUMP_VEL_MAX = -700.0
    JUMP_HOLD_FRAMES = 20; SPEED_JUMP_BONUS = 0.12
    COYOTE_FRAMES = 6; JUMP_BUFFER_FRAMES = 8; BOUNCE_VEL = -400.0


class SonicState(Enum):
    IDLE       = auto()
    WALKING    = auto()
    RUNNING    = auto()
    SKIDDING   = auto()
    JUMPING    = auto()
    FALLING    = auto()
    ROLLING    = auto()    # Ball on ground
    SPIN_DASH  = auto()    # Charging spin dash
    SPRING     = auto()    # Launched by spring (no air control briefly)
    HURT       = auto()    # Hit, brief invulnerability
    LOOKING_UP = auto()
    CROUCHING  = auto()


@dataclass
class SonicPlayer:
    """The Sonic player character."""
    gObj: GameObject
    vx: float = 0.0
    vy: float = 0.0
    on_ground: bool = False
    facing_right: bool = True

    # ── State ────────────────────────────────────────────────────────────
    state: SonicState = SonicState.IDLE

    # ── Rings & Health ───────────────────────────────────────────────────
    rings: int = 0
    invincible_timer: float = 0.0   # Post-hit invincibility
    hurt_timer: float = 0.0         # Knockback duration
    shield: bool = False            # Basic shield (absorbs one hit)

    # ── Spin Dash ────────────────────────────────────────────────────────
    spin_dash_charge: int = 0       # Number of charge taps
    spin_dash_rev: float = 0.0      # Accumulated speed

    # ── Rolling ──────────────────────────────────────────────────────────
    is_ball: bool = False           # True when in ball form (jumping or rolling)

    # ── Jump ─────────────────────────────────────────────────────────────
    coyote: int = 0
    jump_buffer: int = 0
    jump_pressed: bool = False
    last_jump_pressed: bool = False

    # ── Input ────────────────────────────────────────────────────────────
    input_dir: int = 0
    run_held: bool = False
    down_held: bool = False
    up_held: bool = False
    dt: float = 1.0

    # ── Animation ────────────────────────────────────────────────────────
    anim_tick: int = 0
    leg_anim: int = 0              # Leg animation frame
    ball_rotation: float = 0.0     # Ball spin angle

    # ── Power-ups (simplified) ───────────────────────────────────────────
    powered_up: bool = False        # Super Sonic / shield active
    star_timer: float = 0.0         # Invincibility power-up timer

    # ── Platform tracking ────────────────────────────────────────────────
    _on_moving_platform: bool = False

    # ── Fire compat (unused in Sonic but needed for interface compat) ────
    fire_requested: bool = False
    _fire_cooldown: float = 0.0

    def __post_init__(self):
        self.color = COLOR_SONIC_BLUE

    # =====================================================================
    # INPUT
    # =====================================================================
    def handle_input(self, a=None):
        """
        Decode action [move, jump, down/spindash].
        """
        try:
            move_idx = int(a[0])
            jump_idx = int(a[1])
            down_idx = int(a[2])
        except (TypeError, IndexError):
            move_idx = jump_idx = down_idx = 0

        agent_left  = move_idx in (1, 2)
        agent_right = move_idx in (3, 4)
        agent_run   = move_idx in (2, 4)
        agent_jump  = jump_idx == 1
        agent_down  = down_idx == 1

        # Keyboard fallback
        kb_left = kb_right = kb_jump = kb_run = kb_down = kb_up = False
        if pygame.get_init():
            keys = pygame.key.get_pressed()
            kb_left  = keys[pygame.K_LEFT]  or keys[pygame.K_a]
            kb_right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
            kb_jump  = keys[pygame.K_SPACE] or keys[pygame.K_w] or keys[pygame.K_UP]
            kb_run   = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
            kb_down  = keys[pygame.K_DOWN] or keys[pygame.K_s]
            kb_up    = keys[pygame.K_UP]

        is_left  = agent_left  or kb_left
        is_right = agent_right or kb_right

        self.last_jump_pressed = self.jump_pressed
        self.jump_pressed = agent_jump or kb_jump
        self.run_held     = agent_run  or kb_run
        self.down_held    = agent_down or kb_down
        self.up_held      = kb_up and not self.jump_pressed

        if is_left and is_right:
            self.input_dir = 0
        elif is_left:
            self.input_dir = -1
        elif is_right:
            self.input_dir = 1
        else:
            self.input_dir = 0

        if self.jump_pressed and not self.last_jump_pressed:
            self.jump_buffer = JUMP_BUFFER_FRAMES

    # =====================================================================
    # UPDATE
    # =====================================================================
    def update(self, dt: float, context):
        self.dt = dt
        self.anim_tick += 1

        # Tick timers
        if self.invincible_timer > 0:
            self.invincible_timer = max(0, self.invincible_timer - dt)
        if self.star_timer > 0:
            self.star_timer = max(0, self.star_timer - dt)
        if self.hurt_timer > 0:
            self.hurt_timer = max(0, self.hurt_timer - dt)
            if self.hurt_timer <= 0:
                self.state = SonicState.IDLE

        # State machine
        self._update_state(dt, context)

        # Physics
        self._apply_physics(dt, context)

        # Gravity
        grav = context.FAST_FALL_GRAV if self.vy > 0 else context.GRAVITY
        self.vy = min(self.vy + grav * dt, context.MAX_FALL_SPEED)

        # Integration
        self.gObj.x += self.vx * dt
        self.gObj.y += self.vy * dt
        self.on_ground = False

        # Facing direction (not while hurt)
        if self.state != SonicState.HURT:
            if self.vx > 1.0:
                self.facing_right = True
            elif self.vx < -1.0:
                self.facing_right = False

        # Ball state
        self.is_ball = self.state in (
            SonicState.JUMPING, SonicState.ROLLING, SonicState.SPIN_DASH
        )

        # Ball rotation
        if self.is_ball and abs(self.vx) > 10:
            self.ball_rotation += self.vx * dt * 0.05

    def _update_state(self, dt, context):
        """State machine transitions."""
        if self.state == SonicState.HURT:
            return  # Can't act while hurt

        # ── Spin Dash ────────────────────────────────────────────────────
        if self.state == SonicState.SPIN_DASH:
            if not self.down_held:
                # Release spin dash!
                speed = SPIN_DASH_MIN + self.spin_dash_rev
                speed = min(speed, SPIN_DASH_MAX)
                self.vx = speed if self.facing_right else -speed
                self.state = SonicState.ROLLING
                self.spin_dash_charge = 0
                self.spin_dash_rev = 0.0
                return

            # Charge with jump taps
            if self.jump_pressed and not self.last_jump_pressed:
                self.spin_dash_charge = min(self.spin_dash_charge + 1, SPIN_DASH_CHARGES_MAX)
                self.spin_dash_rev = self.spin_dash_charge * SPIN_DASH_CHARGE_RATE
            return

        # ── Crouching / Roll initiation ──────────────────────────────────
        if self.on_ground and self.down_held:
            if abs(self.vx) > ROLL_SPEED_THRESHOLD:
                self.state = SonicState.ROLLING
            elif self.state not in (SonicState.ROLLING, SonicState.SPIN_DASH):
                self.state = SonicState.CROUCHING
                # Start spin dash on jump while crouching
                if self.jump_pressed and not self.last_jump_pressed:
                    self.state = SonicState.SPIN_DASH
                    self.spin_dash_charge = 0
                    self.spin_dash_rev = 0.0
                return

        # ── Jumping ──────────────────────────────────────────────────────
        if not self.on_ground:
            if self.vy < 0:
                self.state = SonicState.JUMPING
            else:
                self.state = SonicState.FALLING
            return

        # ── Rolling on ground ────────────────────────────────────────────
        if self.state == SonicState.ROLLING:
            if abs(self.vx) < 30.0:
                # Too slow, uncurl
                self.state = SonicState.IDLE
            return

        # ── Ground states ────────────────────────────────────────────────
        if self.up_held and abs(self.vx) < 10:
            self.state = SonicState.LOOKING_UP
            return

        speed = abs(self.vx)
        if self.input_dir != 0:
            skidding = (self.vx > 30 and self.input_dir < 0) or \
                       (self.vx < -30 and self.input_dir > 0)
            if skidding and speed > 100:
                self.state = SonicState.SKIDDING
            elif speed > MAX_WALK_SPEED:
                self.state = SonicState.RUNNING
            else:
                self.state = SonicState.WALKING
        else:
            if speed < 10:
                self.state = SonicState.IDLE
            elif speed > MAX_WALK_SPEED:
                self.state = SonicState.RUNNING
            else:
                self.state = SonicState.WALKING

    def _apply_physics(self, dt, ctx):
        """Sonic-specific momentum physics."""
        if self.state == SonicState.HURT:
            # Knockback deceleration
            friction = GROUND_FRICTION * 0.5 * dt
            if self.vx > 0:
                self.vx = max(0.0, self.vx - friction)
            elif self.vx < 0:
                self.vx = min(0.0, self.vx + friction)
            return

        if self.state == SonicState.SPIN_DASH:
            self.vx *= 0.95  # Slow to a stop
            return

        # ── Rolling physics ──────────────────────────────────────────────
        if self.state == SonicState.ROLLING:
            # High speed, low friction when rolling
            friction = ROLL_FRICTION * dt
            if self.vx > 0:
                self.vx = max(0.0, self.vx - friction)
            elif self.vx < 0:
                self.vx = min(0.0, self.vx + friction)
            self._handle_jump(dt, ctx)
            return

        # ── Normal movement ──────────────────────────────────────────────
        target_max = ctx.MAX_RUN_SPEED if hasattr(ctx, 'MAX_RUN_SPEED') and self.run_held else MAX_WALK_SPEED
        accel_rate = ctx.RUN_ACCEL if hasattr(ctx, 'RUN_ACCEL') and self.run_held else WALK_ACCEL

        # Air acceleration is doubled in classic Sonic
        if not self.on_ground:
            accel_rate *= 2.0

        if self.input_dir != 0:
            skidding = (self.vx > 0 and self.input_dir < 0) or \
                       (self.vx < 0 and self.input_dir > 0)

            if skidding:
                # Decelerate when pushing opposite direction
                decel = SKID_DECEL if self.on_ground else accel_rate
                self.vx += self.input_dir * decel * dt
            else:
                # Fix Discrepancy 1: Momentum Clamping
                # Accelerate towards max, but DO NOT clamp if we are already exceeding it!
                if self.input_dir > 0:
                    if self.vx < target_max:
                        self.vx = min(self.vx + accel_rate * dt, target_max)
                else:
                    if self.vx > -target_max:
                        self.vx = max(self.vx - accel_rate * dt, -target_max)
        else:
            # No directional input
            if self.on_ground:
                friction = GROUND_FRICTION * dt
                if self.vx > 0:
                    self.vx = max(0.0, self.vx - friction)
                elif self.vx < 0:
                    self.vx = min(0.0, self.vx + friction)
            else:
                # Fix Discrepancy 4: Zero Air Friction
                # Sonic maintains his momentum perfectly if you let go of the D-Pad in mid-air
                pass

        self._handle_jump(dt, ctx)

    def _handle_jump(self, dt, ctx):
        """Fix Discrepancy 3: The Velocity Chop jump."""
        if self.jump_pressed and not self.last_jump_pressed:
            self.jump_buffer = JUMP_BUFFER_FRAMES

        self.coyote = COYOTE_FRAMES if self.on_ground else max(0, self.coyote - 1)
        if self.jump_buffer > 0:
            self.jump_buffer -= 1

        # Instant Takeoff
        if self.coyote > 0 and self.jump_buffer > 0:
            base_jump = ctx.JUMP_VEL_MAX if hasattr(ctx, 'JUMP_VEL_MAX') else JUMP_VEL_MAX
            # Tiny bonus based on momentum
            bonus = min(80.0, abs(self.vx) * SPEED_JUMP_BONUS)
            self.vy = base_jump - bonus
            
            self.on_ground = False
            self.coyote = 0
            self.jump_buffer = 0
            self.state = SonicState.JUMPING

        # The "Velocity Chop" for variable jump height
        # If Sonic is moving upwards, and the player RELEASES jump, aggressively cut speed
        if not self.jump_pressed and not self.on_ground and self.vy < 0:
            chop_vel = ctx.JUMP_VEL_MIN if hasattr(ctx, 'JUMP_VEL_MIN') else JUMP_VEL_MIN
            if self.vy < chop_vel:
                self.vy = chop_vel

    # =====================================================================
    # DAMAGE
    # =====================================================================
    def take_hit(self) -> bool:
        if self.invincible_timer > 0 or self.star_timer > 0:
            return False

        if self.shield:
            self.shield = False
            self.invincible_timer = 1.0
            return False

        if self.rings > 0:
            self.rings = 0
            self.state = SonicState.HURT
            self.hurt_timer = 0.5
            self.invincible_timer = 2.0
            self.vy = -300.0
            self.vx = -120.0 if self.facing_right else 120.0
            return False

        return True

    def bounce_off_enemy(self):
        self.vy = BOUNCE_VEL
        self.state = SonicState.JUMPING

    def spring_launch(self, velocity: float):
        self.vy = velocity
        self.state = SonicState.SPRING
        self.on_ground = False

    # =====================================================================
    # OBSERVATION VECTOR
    # =====================================================================
    def obs_vector(self, max_run_speed: float, max_fall_speed: float) -> np.ndarray:
        charge_frac = self.spin_dash_charge / max(SPIN_DASH_CHARGES_MAX, 1)
        inv_frac = np.clip(self.invincible_timer / 2.0, 0.0, 1.0)

        return np.array([
            self.gObj.x / TILE_SIZE,
            self.gObj.y / TILE_SIZE,
            np.clip(self.vx / max(max_run_speed, 1.0), -1.0, 1.0),
            np.clip(self.vy / max(max_fall_speed, 1.0), -1.0, 1.0),
            1.0 if self.on_ground else 0.0,
            1.0 if self.is_ball else 0.0,
            1.0 if self.rings > 0 else 0.0,
            1.0 if (self.invincible_timer > 0 or self.star_timer > 0) else 0.0,
            1.0 if self.facing_right else 0.0,
            charge_frac,
            inv_frac,
            1.0 if self.coyote > 0 else 0.0,
            0.0, # Deprecated jump_extendable flag
        ], dtype=np.float32)

    # =====================================================================
    # RENDER
    # =====================================================================
    def render(self, surface: pygame.Surface, sx: float, sy: float, debug: bool = False):
        w = self.gObj.width
        h = self.gObj.height

        if self.invincible_timer > 0 and int(self.invincible_timer * 10) % 2 == 0:
            if self.star_timer <= 0:
                return

        if self.is_ball:
            self._render_ball(surface, sx, sy, w, h)
        elif self.state == SonicState.HURT:
            self._render_hurt(surface, sx, sy, w, h)
        elif self.state == SonicState.CROUCHING:
            self._render_crouch(surface, sx, sy, w, h)
        else:
            self._render_standing(surface, sx, sy, w, h)

        if self.star_timer > 0 and self.anim_tick % 4 < 2:
            import random
            for _ in range(3):
                spark_x = sx + random.randint(-4, int(w) + 4)
                spark_y = sy + random.randint(-4, int(h) + 4)
                pygame.draw.circle(surface, (255, 0, 0), (int(spark_x), int(spark_y)), 2)

        if abs(self.vx) > MAX_RUN_SPEED * 0.8:
            self._render_speed_lines(surface, sx, sy, w, h)

        if debug:
            self._render_debug(surface, sx, sy, w, h)

    def _render_ball(self, surface, sx, sy, w, h):
        cx = int(sx + w // 2)
        cy = int(sy + h // 2)
        radius = int(min(w, h) // 2)

        ball_col = (255, 0, 0) if self.star_timer > 0 else COLOR_SONIC_BALL
        pygame.draw.circle(surface, ball_col, (cx, cy), radius)
        pygame.draw.circle(surface, COLOR_BLACK, (cx, cy), radius, 1)

        angle = self.ball_rotation
        for i in range(4):
            a = angle + i * (math.pi / 2)
            lx = int(cx + math.cos(a) * radius * 0.6)
            ly = int(cy + math.sin(a) * radius * 0.6)
            pygame.draw.circle(surface, (60, 80, 255), (lx, ly), 2)

    def _render_standing(self, surface, sx, sy, w, h):
        body_rect = (int(sx + 2), int(sy + 2), int(w - 4), int(h * 0.6))
        pygame.draw.rect(surface, COLOR_SONIC_BLUE, body_rect)

        head_y = int(sy)
        head_x = int(sx + w // 2)
        pygame.draw.circle(surface, COLOR_SONIC_BLUE, (head_x, head_y + 8), 8)

        if self.facing_right:
            for i in range(3):
                spine_x = int(sx - 2 - i * 3)
                spine_y = int(sy + 2 + i * 3)
                pygame.draw.line(surface, (20, 40, 180),
                               (head_x, head_y + 5), (spine_x, spine_y), 2)
        else:
            for i in range(3):
                spine_x = int(sx + w + 2 + i * 3)
                spine_y = int(sy + 2 + i * 3)
                pygame.draw.line(surface, (20, 40, 180),
                               (head_x, head_y + 5), (spine_x, spine_y), 2)

        eye_x = int(sx + (w * 0.65 if self.facing_right else w * 0.35))
        eye_y = int(sy + 6)
        pygame.draw.circle(surface, COLOR_WHITE, (eye_x, eye_y), 4)
        pupil_off = 1 if self.facing_right else -1
        pygame.draw.circle(surface, COLOR_BLACK, (eye_x + pupil_off, eye_y), 2)

        belly_rect = (int(sx + w * 0.25), int(sy + h * 0.35), int(w * 0.5), int(h * 0.25))
        pygame.draw.ellipse(surface, COLOR_SONIC_SKIN, belly_rect)

        shoe_y = int(sy + h - 8)
        if abs(self.vx) > 10:
            self.leg_anim = (self.anim_tick // max(1, int(8 - abs(self.vx) / 60))) % 4
            offsets = [(-2, 0, 4, 0), (0, -2, 0, 4), (2, 0, -4, 0), (0, 2, 0, -4)]
            lo = offsets[self.leg_anim % len(offsets)]
            pygame.draw.rect(surface, COLOR_SONIC_SHOE,
                           (int(sx + 2 + lo[0]), int(shoe_y + lo[1]), 8, 6))
            pygame.draw.rect(surface, COLOR_SONIC_SHOE,
                           (int(sx + w - 10 + lo[2]), int(shoe_y + lo[3]), 8, 6))
        else:
            pygame.draw.rect(surface, COLOR_SONIC_SHOE, (int(sx + 2), shoe_y, 8, 6))
            pygame.draw.rect(surface, COLOR_SONIC_SHOE, (int(sx + w - 10), shoe_y, 8, 6))

    def _render_crouch(self, surface, sx, sy, w, h):
        cx = int(sx + w // 2)
        cy = int(sy + h - 10)
        pygame.draw.ellipse(surface, COLOR_SONIC_BLUE, (int(sx + 2), int(sy + h // 2), int(w - 4), int(h // 2)))
        
        pygame.draw.circle(surface, COLOR_WHITE, (int(cx + (3 if self.facing_right else -3)), int(cy - 4)), 3)
        pygame.draw.circle(surface, COLOR_BLACK, (int(cx + (4 if self.facing_right else -4)), int(cy - 4)), 1)

    def _render_hurt(self, surface, sx, sy, w, h):
        cx = int(sx + w // 2)
        cy = int(sy + h // 2)
        pygame.draw.circle(surface, (200, 80, 80), (cx, cy), int(min(w, h) // 2))
        pygame.draw.circle(surface, COLOR_WHITE, (cx, cy - 3), 3)
        pygame.draw.line(surface, COLOR_BLACK, (cx - 3, cy - 5), (cx - 1, cy - 3), 1)
        pygame.draw.line(surface, COLOR_BLACK, (cx - 1, cy - 5), (cx - 3, cy - 3), 1)

    def _render_speed_lines(self, surface, sx, sy, w, h):
        n = 4
        spacing = 5
        length = int(abs(self.vx) / 30)
        for i in range(n):
            offset = (i + 1) * spacing
            if self.facing_right:
                x1 = int(sx - offset)
                x2 = int(x1 - length)
            else:
                x1 = int(sx + w + offset)
                x2 = int(x1 + length)
            y = int(sy + 6 + i * 6)
            pygame.draw.line(surface, COLOR_STREAK, (x1, y), (x2, y), 1)

    def _render_debug(self, surface, sx, sy, w, h):
        v_end = (
            int(sx + self.vx * 5 * self.dt),
            int(sy + self.vy * 5 * self.dt)
        )
        pygame.draw.line(surface, (100, 255, 255),
                        (int(sx + w / 2), int(sy + h / 2)), v_end, 2)
        pygame.draw.rect(surface, (255, 64, 64), (int(sx), int(sy), int(w), int(h)), 1)