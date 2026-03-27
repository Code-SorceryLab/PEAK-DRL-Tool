from __future__ import annotations
from dataclasses import dataclass, field
from .GameObject import GameObject
import pygame
from typing import Dict, List, Any, Optional
from enum import Enum, auto
import numpy as np
# Imports from sibling packages
from ..Parameters import Movement_parameters as MP
from ..Parameters import Jump_parameters as JP
from ..Parameters.Map_parameters import COLOR_WHITE, COLOR_STREAK, COLOR_SENSOR, TILE_SIZE
from ..System.PhysicsManager import PhysicsContext
from ..System.AnimationHandler import AnimationHandler
from ..System.PlayerStateMachine import PlayerStateMachine
from ..System.PlayerStateMachine import PowerState

# --- PLAYER SPECIFIC ENUM ---
class PlayerAnim(Enum):
    IDLE = auto()
    RUN  = auto()
    JUMP = auto()
    FALL = auto()


# How long (seconds) the player must wait between shots
_FIRE_COOLDOWN = 0.35


@dataclass
class Player():
    gObj: GameObject
    vx: float = 0.0
    vy: float = 0.0
    color = (255, 0, 0)
    eye_radius = 3
    on_ground: bool = False
    facing_right: bool = True
    star_timer:   float = 0.0
    iframes_timer: float = 0.0
    # --- POWER STATE ---
    # powered_up and invincible_timer are kept as plain fields because the rest
    # of the codebase (platformer_core rendering, PhysicsManager) reads them
    # directly. PlayerStateMachine.apply_to_player() writes to them every frame —
    # do NOT mutate them manually anywhere else.
    powered_up: bool = False
    invincible_timer: float = 0.0

    # --- ANIMATION STATE ---
    anim_handler: AnimationHandler = field(init=False, default=None)

    # --- POWER STATE MACHINE ---
    # on_death is wired in after construction by platformer_core so the machine
    # can call core._handle_death() without Player holding a core reference.
    # PhysicsManager checks take_hit() → False and calls core._handle_death()
    # itself, so on_death=None is safe during normal gameplay.
    power_machine: PlayerStateMachine = field(init=False)

    coyote: int = 0
    jump_hold: int = 0
    jump_buffer: int = 0
    last_jump_pressed: bool = False
    run_pressed: bool = False
    dt: float = 1.0

    # Input State
    input_dir: int = 0
    run_held: bool = False
    jump_pressed: bool = False

    # --- FIRE STATE ---
    # fire_requested is set True by try_fire() and consumed by platformer_core
    # in step() to actually spawn the projectile. platformer_core is the only
    # place that should reset this flag to False after reading it.
    fire_requested: bool = False
    hp: int = 1
    hp_max: int = 1
    shot_cooldown: float = 0.0
    jump_cut: bool = False
    on_ladder: bool = False
    ladder_x: float = 0.0
    can_jump: bool = False
    grounded: bool = False

    def __post_init__(self):
        # 1. Initialise power state machine (no on_death yet — wired later if needed)
        self.power_machine = PlayerStateMachine()

        # 2. Fire cooldown timer — tracked here so try_fire() is self-contained.
        #    Decremented in update() so it ticks even when no fire input is given.
        self._fire_cooldown: float = 0.0

        # 3. Animation paths
        default_asset = 'code/games/assets/idle1.png'

        anim_paths = {
            PlayerAnim.IDLE: [
                'code/games/assets/idle1.png',
                'code/games/assets/idle2.png',
                'code/games/assets/idle3.png',
                'code/games/assets/idle4.png',
                'code/games/assets/idle5.png',
                'code/games/assets/idle6.png',
            ],
            PlayerAnim.RUN:  [
                'code/games/assets/run (1).png',
                'code/games/assets/run (2).png',
                'code/games/assets/run (3).png',
                'code/games/assets/run (4).png'
            ],
            PlayerAnim.JUMP: ['code/games/assets/jump.png'],
            PlayerAnim.FALL: ['code/games/assets/fall.png']
        }

        int_keyed_paths = {state.value: paths for state, paths in anim_paths.items()}
        base_size = (int(self.gObj.width), int(self.gObj.height))
        loaded_anims = AnimationHandler.load_animations(int_keyed_paths, base_size)

        self.anim_handler = AnimationHandler(
            loaded_anims,
            default_state=PlayerAnim.IDLE.value,
            duration=0.1
        )

    def update(self, dt: float, context: PhysicsContext):
        self.dt = dt

        # 1. Tick power machine — handles star timer, i-frame timer
        #    Must come before apply_to_player so timers are up to date
        self.power_machine.update(dt)

        # 2. Sync machine state → Player fields (powered_up, invincible_timer, height)
        #    Everything downstream reads these fields, so this is the only sync point.
        self.power_machine.apply_to_player(self)

        # 3. Facing direction
        if self.vx > 1.0:
            self.facing_right = True
        elif self.vx < -1.0:
            self.facing_right = False

        # 4. Animation
        self._update_animation_logic(dt)

        # 5. Physics & gravity
        self.apply_physics(dt, context)
        grav = context.FAST_FALL_GRAV if self.vy > 0 else context.GRAVITY
        self.vy = min(self.vy + (grav * dt), context.MAX_FALL_SPEED)

        self.gObj.x += self.vx * dt
        self.gObj.y += self.vy * dt
        self.on_ground = False

        # 6. Tick fire cooldown — keeps it draining even when fire key is not held
        if self._fire_cooldown > 0.0:
            self._fire_cooldown = max(0.0, self._fire_cooldown - dt)
        self.shot_cooldown = self._fire_cooldown

    def _update_animation_logic(self, dt: float):
        """Decides which animation state to use based on physics."""
        if not self.anim_handler:
            return

        self.anim_handler.update(dt)

        if not self.on_ground:
            target_state = PlayerAnim.JUMP if self.vy < 0 else PlayerAnim.FALL
        elif abs(self.vx) > 10:
            target_state = PlayerAnim.RUN
        else:
            target_state = PlayerAnim.IDLE

        self.anim_handler.set_state(target_state.value)

    def handle_input(self, a):
        """
        Decode a MultiDiscrete action [move, jump, fire] or fall back to
        keyboard input in human mode.

        move axis : 0=idle  1=left  2=sprint_left  3=right  4=sprint_right
        jump axis : 0=idle  1=jump
        fire axis : 0=idle  1=fire
        """
        try:
            move_idx = int(a[0])
            jump_idx = int(a[1])
            fire_idx = int(a[2])
        except (TypeError, IndexError):
            move_idx = jump_idx = fire_idx = 0

        # Decode movement axis
        agent_left  = move_idx in (1, 2)
        agent_right = move_idx in (3, 4)
        agent_run   = move_idx in (2, 4)   # sprint variants
        agent_jump  = jump_idx == 1
        agent_fire  = fire_idx == 1

        kb_left = kb_right = kb_jump = kb_run = kb_fire = False

        if pygame.get_init():
            keys = pygame.key.get_pressed()
            kb_left  = keys[pygame.K_LEFT]  or keys[pygame.K_a]
            kb_right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
            kb_jump  = keys[pygame.K_SPACE] or keys[pygame.K_w] or keys[pygame.K_UP]
            kb_run   = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
            kb_fire  = keys[pygame.K_z]   # Z fires a fireball (human mode)

        is_left  = agent_left  or kb_left
        is_right = agent_right or kb_right
        self.jump_pressed = agent_jump or kb_jump
        self.run_held     = agent_run  or kb_run

        if is_left and is_right: self.input_dir = 0
        elif is_left:            self.input_dir = -1
        elif is_right:           self.input_dir = 1
        else:                    self.input_dir = 0

        if self.jump_pressed:
            self.jump_buffer = 6

        # Fire input — try_fire() enforces cooldown and power state internally.
        # fire_requested is consumed by platformer_core.step() to spawn the projectile.
        if agent_fire or kb_fire:
            self.try_fire()

    def try_fire(self) -> bool:
        """
        Request a FireFlowerProjectile to be spawned on this frame.

        Called by handle_input() (human key), or directly by the game loop
        for any other trigger (e.g. an RL action).

        Guards:
          - Player must be in the FIRE power state (power_machine.is_fire).
          - Cooldown of _FIRE_COOLDOWN seconds between shots.

        When both guards pass, sets fire_requested = True and starts the
        cooldown. platformer_core.step() reads fire_requested, spawns the
        projectile, then resets the flag.

        Returns True if a shot was queued this frame, False otherwise.
        """
        # print ("try fire")
        if not self.power_machine.state == PowerState.FIRE:
            #print ("can't fire: not in fire state")
            return False
        if self._fire_cooldown > 0.0:
            #print (f"can't fire: on cooldown for {self._fire_cooldown:.2f} more seconds")
            return False
        #print ("fire!")
        self.fire_requested = True
        self._fire_cooldown = _FIRE_COOLDOWN
        return True

    def apply_physics(self, dt: float, ctx: PhysicsContext):
        if self.run_held:
            target_max = ctx.MAX_RUN_SPEED
            accel_rate = ctx.RUN_ACCEL
        else:
            target_max = ctx.MAX_WALK_SPEED
            accel_rate = ctx.WALK_ACCEL

        if not self.on_ground:
            accel_rate *= ctx.AIR_CONTROL

        if self.input_dir != 0:
            skidding = (self.vx > 0 and self.input_dir < 0) or \
                       (self.vx < 0 and self.input_dir > 0)

            if self.on_ground and skidding:
                self.set_horizontal_velocity(self.vx + self.input_dir * ctx.SKID_DECEL * dt)
            else:
                if self.input_dir > 0:
                    self.set_horizontal_velocity(min(self.vx + accel_rate * dt,  target_max))
                else:
                    self.set_horizontal_velocity(max(self.vx - accel_rate * dt, -target_max))
        else:
            friction = (ctx.GROUND_FRICTION if self.on_ground else ctx.AIR_FRICTION) * dt
            if self.vx > 0:   self.set_horizontal_velocity(max(0.0, self.vx - friction))
            elif self.vx < 0: self.set_horizontal_velocity(min(0.0, self.vx + friction))

        self.handle_jump(dt, ctx)

    def handle_jump(self, dt: float, ctx: PhysicsContext):
        if self.jump_pressed:
            self.jump_buffer = ctx.JUMP_BUFFER_FRAMES

        self.coyote = ctx.COYOTE_FRAMES if self.on_ground else max(0, self.coyote - 1)
        if self.jump_buffer > 0: self.jump_buffer -= 1

        if self.coyote > 0 and self.jump_hold == 0 and self.jump_buffer > 0:
            self.start_jump(ctx)

        if self.jump_hold > 0:
            if self.jump_pressed:
                # Variable jump height — hold for high, tap for short
                self.vy -= ctx.GRAVITY * 0.12 * dt
            self.jump_hold -= 1

    def start_jump(self, ctx: PhysicsContext):
        base = ctx.JUMP_VEL_MIN
        bonus = min(2.2, abs(self.vx) * ctx.SPEED_JUMP_BONUS)
        self.vy = base - bonus 
        self.on_ground = False
        self.coyote = 0
        self.jump_hold = ctx.JUMP_HOLD_FRAMES
        self.jump_buffer = 0

    def render(self, surface: pygame.Surface, sx: float, sy: float, debug: bool = True):
        sprite = None
        if self.anim_handler:
            sprite = self.anim_handler.get_sprite(self.facing_right)

        if sprite:
            # Scale sprite to match current gObj dimensions (handles power-up size changes)
            current_size = (int(self.gObj.width), int(self.gObj.height))
            if (sprite.get_width(), sprite.get_height()) != current_size:
                sprite = pygame.transform.scale(sprite, current_size)
            y_offset = sprite.get_height() - self.gObj.height
            surface.blit(sprite, (int(sx), int(sy - y_offset)))
            if debug:
                self._debug(surface, sx, sy)
        else:
            # Fallback: coloured rectangle + eye
            pygame.draw.rect(surface, self.color, (sx, sy, self.gObj.width, self.gObj.height))
            pygame.draw.circle(
                surface, COLOR_WHITE,
                (int(sx + (14 if self.facing_right else 6)), int(sy + 8)),
                self.eye_radius
            )
            if debug:
                self._debug(surface, sx, sy)
            if self.run_held and abs(self.vx) > 100:
                n = 3; spacing = 6; length = 10
                for i in range(n):
                    offset = (i + 1) * spacing
                    if self.facing_right:
                        x1 = sx - offset;              x2 = x1 - length
                    else:
                        x1 = sx + self.gObj.width + offset; x2 = x1 + length
                    y = sy + 10 + (i % 2) * 4
                    pygame.draw.line(surface, COLOR_STREAK,
                                     (int(x1), int(y)), (int(x2), int(y)), 2)

    def _debug(self, surface: pygame.Surface, sx: float, sy: float):
        v_end = (
            int(sx + self.vx * 5 * self.dt),
            int(sy + self.vy * 5 * self.dt)
        )
        pygame.draw.line(
            surface, (100, 255, 255),
            (int(sx + self.gObj.width / 2), int(sy + self.gObj.height / 2)),
            v_end, 2
        )
        
    def obs_vector(self, max_run_speed: float, max_fall_speed: float) -> np.ndarray:
        """
        Returns a flat float32 array of all player state relevant to the obs.

        Designed to be called by platformer_core._player_obs() so the obs
        construction always stays in sync with the Player fields.

        Layout (13 values):
        [0]  x position          in tiles, unbounded (consistent across all level sizes)
        [1]  y position          in tiles, unbounded (consistent across all level sizes)
        [2]  velocity x          normalised [-1, 1]
        [3]  velocity y          normalised [-1, 1]
        [4]  on_ground           binary {0, 1}
        [5]  powered_up          binary {0, 1}  (SUPER or FIRE state)
        [6]  can_fire            binary {0, 1}  (FIRE state specifically)
        [7]  invincible          binary {0, 1}  (star or i-frames active)
        [8]  facing_right        binary {0, 1}
        [9]  fire_cooldown       normalised [0, 1]  (0 = ready, 1 = just fired)
        [10] invincible_timer    normalised [0, 1]  (fraction of max star duration)
        [11] coyote_active       binary {0, 1}  (can still jump after walking off edge)
        [12] jump_extendable     binary {0, 1}  (variable jump still accepting hold)
        """

        state = self.power_machine.state
        is_fire       = (state == PowerState.FIRE)
        is_powered_up = self.powered_up                         # True for SUPER and FIRE
        is_invincible = self.invincible_timer > 0.0

        # Cooldown fraction: 1.0 = just fired (full cooldown), 0.0 = ready to fire
        cooldown_frac = np.clip(self._fire_cooldown / max(_FIRE_COOLDOWN, 1e-6), 0.0, 1.0)

        # Star timer normalised — 10 s is the typical max star duration
        _MAX_STAR = 10.0
        inv_frac = np.clip(self.invincible_timer / _MAX_STAR, 0.0, 1.0)

        return np.array([
            self.gObj.x / TILE_SIZE,               # [0]  x pos in tiles
            self.gObj.y / TILE_SIZE,               # [1]  y pos in tiles
            np.clip(self.vx / max_run_speed,  -1.0,  1.0),    # [2]  vx
            np.clip(self.vy / max_fall_speed, -1.0,  1.0),    # [3]  vy
            1.0 if self.on_ground    else 0.0,                 # [4]  on_ground
            1.0 if is_powered_up     else 0.0,                 # [5]  powered_up
            1.0 if is_fire           else 0.0,                 # [6]  can_fire
            1.0 if is_invincible     else 0.0,                 # [7]  invincible
            1.0 if self.facing_right else 0.0,                 # [8]  facing_right
            cooldown_frac,                                     # [9]  fire_cooldown
            inv_frac,                                          # [10] invincible_timer
            1.0 if self.coyote > 0   else 0.0,                 # [11] coyote_active
            1.0 if self.jump_hold > 0 else 0.0,               # [12] jump_extendable
        ], dtype=np.float32)

    def set_horizontal_velocity(self, vx):
        self.vx = vx