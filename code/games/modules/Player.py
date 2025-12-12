from __future__ import annotations
from dataclasses import dataclass
from .GameObject import GameObject

from .Movement_parameters import (
    GRAVITY, FAST_FALL_GRAV, MAX_FALL_SPEED, MAX_WALK_SPEED, MAX_RUN_SPEED, 
    RUN_ACCEL, WALK_ACCEL, GROUND_FRICTION, AIR_FRICTION, AIR_CONTROL, SKID_DECEL
)
from .Jump_parameters import (
    JUMP_VEL_MIN, JUMP_VEL_MAX, JUMP_HOLD_FRAMES, SPEED_JUMP_BONUS, 
    COYOTE_FRAMES, JUMP_BUFFER_FRAMES
)
from .HelperFunctions import _world_to_screen
from .Map_parameters import COLOR_WHITE, COLOR_STREAK, COLOR_SENSOR
import pygame

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
    
    # Input handling constants
    max_run: float = MAX_RUN_SPEED
    max_walk: float = MAX_WALK_SPEED
    run_accel: float = RUN_ACCEL
    walk_accel: float = WALK_ACCEL
    
    
    def update(self, dt: float):
        self.dt = dt
        # 1. Handle Timers
        if self.invincible_timer > 0:
            self.invincible_timer -= dt
        
        # 2. Apply Input-based Physics (Movement)
        # We pass dt here so acceleration scales correctly
        self.apply_physics(dt)

        # 3. Apply Gravity
        grav = FAST_FALL_GRAV if self.vy > 0 else GRAVITY
        # Apply gravity scaled by time
        self.vy = min(self.vy + (grav * dt), MAX_FALL_SPEED)

        # 4. Apply Velocity to Position
        self.gObj.x += self.vx * dt
        self.gObj.y += self.vy * dt
        
        # Reset ground state (collision detection usually sets this back to True later)
        self.on_ground = False

    def handle_input(self, a: int):    
        # DECOUPLED: This function now only sets state flags.
        # It does NOT calculate physics.
        
        #Decode Agent Actions (0-7)
        agent_left = (a in (1,6))
        agent_right = (a in (2,4,5,7))
        agent_jump = (a in (3,4,6,7))
        agent_run = (a in (5,7))
        
        # Decode Keyboard Input (Manual Override)
        kb_left = False
        kb_right = False
        kb_jump = False
        kb_run = False
        
        if pygame.get_init():
            keys = pygame.key.get_pressed()
            kb_left = keys[pygame.K_LEFT] or keys[pygame.K_a]
            kb_right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
            kb_jump = keys[pygame.K_SPACE] or keys[pygame.K_w] or keys[pygame.K_UP]
            kb_run = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        
        # Combine Agent and Keyboard Inputs
        is_left = agent_left or kb_left
        is_right = agent_right or kb_right
        self.jump_pressed = agent_jump or kb_jump
        self.run_held = agent_run or kb_run

        # State Setting
        if is_left and is_right:
            self.input_dir = 0
        elif is_left:
            self.input_dir = -1
        elif is_right:
            self.input_dir = 1
        else:
            self.input_dir = 0
            
        # Jump Buffering logic
        if self.jump_pressed:
            self.jump_buffer = JUMP_BUFFER_FRAMES

    def apply_physics(self, dt: float):
        # 1. Determine Target Speed & Acceleration
        # If running, use run params, else walk params
        if self.run_held:
            target_max = self.max_run
            accel_rate = self.run_accel
        else:
            target_max = self.max_walk
            accel_rate = self.walk_accel

        # Air control modifier
        if not self.on_ground:
            accel_rate *= AIR_CONTROL

        # 2. Apply Acceleration / Deceleration
        if self.input_dir != 0:
            # We are trying to move
            target_vx = self.input_dir * target_max
            
            # SKID CHECK: Are we moving one way but holding the other?
            # (Moving right (vx > 0) but holding left (input < 0))
            skidding = (self.vx > 0 and self.input_dir < 0) or (self.vx < 0 and self.input_dir > 0)
            
            if self.on_ground and skidding:
                # Mario Skid: Stronger deceleration to turn around
                # We use SKID_DECEL as a force, not a multiplier
                self.vx += (self.input_dir * SKID_DECEL * dt)
            else:
                # Normal Acceleration
                # We approach the target speed
                if self.input_dir > 0:
                    self.vx = min(self.vx + (accel_rate * dt), target_max)
                else:
                    self.vx = max(self.vx - (accel_rate * dt), -target_max)
                    
            self.facing_right = (self.input_dir > 0)
            
        else:
            # 3. No Input - Apply Linear Friction (The "Snappy" Stop)
            friction = (GROUND_FRICTION if self.on_ground else AIR_FRICTION) * dt
            
            if self.vx > 0:
                self.vx = max(0, self.vx - friction)
            elif self.vx < 0:
                self.vx = min(0, self.vx + friction)

        # 4. Jump Logic
        self.handle_jump(dt)

    def handle_jump(self, dt: float):
        # Coyote Time
        self.coyote = COYOTE_FRAMES if self.on_ground else max(0, self.coyote - 1)
        if self.jump_buffer > 0:
            self.jump_buffer -= 1

        # Jump Start
        # Note: changed 'p' to 'self' and fixed logic
        if (self.coyote > 0) and (self.jump_hold == 0) and (self.jump_buffer > 0):
            base = JUMP_VEL_MIN
            long = JUMP_VEL_MAX # Ensure this is negative (e.g. -10) for upward movement
            
            # Momentum Bonus (Running jumps go higher)
            bonus = min(2.2, abs(self.vx) * SPEED_JUMP_BONUS)
            
            self.vy = base - bonus # Assuming negative is UP
            self.on_ground = False
            self.coyote = 0
            self.jump_hold = JUMP_HOLD_FRAMES
            self.jump_buffer = 0

        # Variable Jump Height (Holding the button)
        if self.jump_hold > 0:
            if self.jump_pressed:
                # While holding, we add upward force (negative Y)
                self.vy -= 0.30 * (dt * 60) # Normalize to 60fps feel
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
        return
    def _debug(self, surface: pygame.Surface, sx: float, sy: float):
        rays = [((sx + self.gObj.width // 2, sy + self.gObj.height), (sx + self.gObj.width // 2, sy + self.gObj.height + 10)),
                ((sx + self.gObj.width // 2, sy), (sx + self.gObj.width // 2, sy - 10)),
                ((sx, sy + self.gObj.height // 2), (sx - 10, sy + self.gObj.height // 2)),
                ((sx + self.gObj.width, sy + self.gObj.height // 2), (sx + self.gObj.width + 10, sy + self.gObj.height // 2))]
        for a, b in rays:
            pygame.draw.line(surface, COLOR_SENSOR, a, b, 2)
        v_end = (int(sx + (self.vx * 5* self.dt)), int(sy + (self.vy * 5 *self.dt)))
        pygame.draw.line(surface, (100, 255, 255), (int(sx + self.gObj.width / 2), int(sy + self.gObj.height / 2)), v_end, 2)