from __future__ import annotations
from dataclasses import dataclass
from .GameObject import GameObject
import pygame

# Imports from sibling packages
from ..Parameters import Movement_parameters as MP
from ..Parameters import Jump_parameters as JP
from ..Parameters.Map_parameters import COLOR_WHITE, COLOR_STREAK, COLOR_SENSOR

# FIX: HelperFunctions is in System, not Objects
from ..System.HelperFunctions import _world_to_screen

@dataclass
class Player():
    gObj: GameObject
    vx: float = 0.0
    vy: float = 0.0
    color = (255, 0, 0)
    eye_radius = 3
    on_ground: bool = False
    facing_right: bool = True
    powered_up: bool = False
    
    invincible_timer: int = 0
    coyote: int = 0
    jump_hold: int = 0
    jump_buffer: int = 0
    last_jump_pressed: bool = False
    run_pressed: bool = False
    dt = 1.0
    
    # Input State
    input_dir: int = 0
    run_held: bool = False
    jump_pressed: bool = False
    
    # --- DYNAMIC PHYSICS PARAMETERS ---
    gravity: float = MP.GRAVITY
    fast_fall_gravity: float = MP.FAST_FALL_GRAV
    max_fall_speed: float = MP.MAX_FALL_SPEED
    
    run_accel: float = MP.RUN_ACCEL
    walk_accel: float = MP.WALK_ACCEL
    max_run: float = MP.MAX_RUN_SPEED
    max_walk: float = MP.MAX_WALK_SPEED
    air_control: float = MP.AIR_CONTROL
    
    ground_friction: float = MP.GROUND_FRICTION
    air_friction: float = MP.AIR_FRICTION
    
    jump_vel_max: float = JP.JUMP_VEL_MAX
    jump_vel_min: float = JP.JUMP_VEL_MIN
    jump_hold_frames: int = JP.JUMP_HOLD_FRAMES
    coyote_frames: int = JP.COYOTE_FRAMES
    jump_buffer_frames: int = JP.JUMP_BUFFER_FRAMES
    
    def update(self, dt: float):
        self.dt = dt
        # 1. Handle Timers
        if self.invincible_timer > 0:
            self.invincible_timer -= dt
        
        # 2. Apply Input-based Physics
        self.apply_physics(dt)

        # 3. Apply Gravity
        grav = self.fast_fall_gravity if self.vy > 0 else self.gravity
        self.vy = min(self.vy + (grav * dt), self.max_fall_speed)

        # 4. Apply Velocity to Position
        self.gObj.x += self.vx * dt
        self.gObj.y += self.vy * dt
        self.on_ground = False

    def handle_input(self, a: int):    
        agent_left = (a in (1,6))
        agent_right = (a in (2,4,5,7))
        agent_jump = (a in (3,4,6,7))
        agent_run = (a in (5,7))
        
        kb_left = kb_right = kb_jump = kb_run = False
        
        if pygame.get_init():
            keys = pygame.key.get_pressed()
            kb_left = keys[pygame.K_LEFT] or keys[pygame.K_a]
            kb_right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
            kb_jump = keys[pygame.K_SPACE] or keys[pygame.K_w] or keys[pygame.K_UP]
            kb_run = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        
        is_left = agent_left or kb_left
        is_right = agent_right or kb_right
        self.jump_pressed = agent_jump or kb_jump
        self.run_held = agent_run or kb_run

        if is_left and is_right: self.input_dir = 0
        elif is_left: self.input_dir = -1
        elif is_right: self.input_dir = 1
        else: self.input_dir = 0
            
        if self.jump_pressed:
            self.jump_buffer = self.jump_buffer_frames

    def apply_physics(self, dt: float):
        if self.run_held:
            target_max = self.max_run
            accel_rate = self.run_accel
        else:
            target_max = self.max_walk
            accel_rate = self.walk_accel

        if not self.on_ground:
            accel_rate *= self.air_control

        if self.input_dir != 0:
            target_vx = self.input_dir * target_max
            skidding = (self.vx > 0 and self.input_dir < 0) or (self.vx < 0 and self.input_dir > 0)
            
            if self.on_ground and skidding:
                self.vx += (self.input_dir * MP.SKID_DECEL * dt)
            else:
                if self.input_dir > 0: self.vx = min(self.vx + (accel_rate * dt), target_max)
                else: self.vx = max(self.vx - (accel_rate * dt), -target_max)
            self.facing_right = (self.input_dir > 0)
        else:
            friction = (self.ground_friction if self.on_ground else self.air_friction) * dt
            if self.vx > 0: self.vx = max(0, self.vx - friction)
            elif self.vx < 0: self.vx = min(0, self.vx + friction)

        self.handle_jump(dt)

    def handle_jump(self, dt: float):
        self.coyote = self.coyote_frames if self.on_ground else max(0, self.coyote - 1)
        if self.jump_buffer > 0: self.jump_buffer -= 1

        if (self.coyote > 0) and (self.jump_hold == 0) and (self.jump_buffer > 0):
            base = self.jump_vel_min
            bonus = min(2.2, abs(self.vx) * JP.SPEED_JUMP_BONUS)
            self.vy = base - bonus 
            self.on_ground = False
            self.coyote = 0
            self.jump_hold = self.jump_hold_frames
            self.jump_buffer = 0

        if self.jump_hold > 0:
            if self.jump_pressed:
                self.vy -= 0.30 * (dt * 60) 
            self.jump_hold -= 1
                
    def render(self, surface: pygame.Surface, sx: float, sy: float, debug: bool = True):        
        pygame.draw.rect(surface, self.color, (sx, sy, self.gObj.width, self.gObj.height))
        pygame.draw.circle(surface, COLOR_WHITE, (int(sx + (14 if self.facing_right else 6)), int(sy + 8)), self.eye_radius)
        if debug:
            self._debug(surface, sx, sy)
        if self.run_pressed and abs(self.vx) > self.max_walk * 0.6:
            n = 3; spacing = 6; length = 10
            for i in range(n):
                offset = (i + 1) * spacing
                if self.facing_right:
                    x1 = sx - offset; x2 = x1 - length
                else:
                    x1 = sx + self.gObj.width + offset; x2 = x1 + length
                y = sy + 10 + (i % 2) * 4
                pygame.draw.line(surface, COLOR_STREAK, (int(x1), int(y)), (int(x2), int(y)), 2)

    def _debug(self, surface: pygame.Surface, sx: float, sy: float):
        rays = [((sx + self.gObj.width // 2, sy + self.gObj.height), (sx + self.gObj.width // 2, sy + self.gObj.height + 10)),
                ((sx + self.gObj.width // 2, sy), (sx + self.gObj.width // 2, sy - 10)),
                ((sx, sy + self.gObj.height // 2), (sx - 10, sy + self.gObj.height // 2)),
                ((sx + self.gObj.width, sy + self.gObj.height // 2), (sx + self.gObj.width + 10, sy + self.gObj.height // 2))]
        for a, b in rays:
            pygame.draw.line(surface, COLOR_SENSOR, a, b, 2)
        v_end = (int(sx + (self.vx * 5* self.dt)), int(sy + (self.vy * 5 *self.dt)))
        pygame.draw.line(surface, (100, 255, 255), (int(sx + self.gObj.width / 2), int(sy + self.gObj.height / 2)), v_end, 2)