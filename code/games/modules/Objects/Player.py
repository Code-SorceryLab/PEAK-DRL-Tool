from __future__ import annotations
from dataclasses import dataclass, field
from .GameObject import GameObject
import pygame

# Imports from sibling packages
from ..Parameters import Movement_parameters as MP
from ..Parameters import Jump_parameters as JP
from ..Parameters.Map_parameters import COLOR_WHITE, COLOR_STREAK, COLOR_SENSOR
from ..System.PhysicsManager import PhysicsContext

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
    
    # NOTE: Physics parameters removed. 
    # They are now accessed directly from the PhysicsContext passed to update()
    
    def update(self, dt: float, context: PhysicsContext):
        self.dt = dt
        
        # 1. Handle Timers
        if self.invincible_timer > 0:
            self.invincible_timer -= dt
        
        # 2. Apply Input-based Physics
        self.apply_physics(dt, context)

        # 3. Apply Gravity
        # Retrieve gravity values directly from context
        grav = context.FAST_FALL_GRAV if self.vy > 0 else context.GRAVITY
        self.vy = min(self.vy + (grav * dt), context.MAX_FALL_SPEED)

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
            # We assume context values for buffering might be needed, 
            # but usually buffer length is static or context-derived in update.
            # For simplicity, we just set the counter here, logic handles it in physics.
            self.jump_buffer = 6 # Default fallback, overwritten in apply_physics logic if needed

    def apply_physics(self, dt: float, ctx: PhysicsContext):
        # 1. Movement params from Context
        if self.run_held:
            target_max = ctx.MAX_RUN_SPEED
            accel_rate = ctx.RUN_ACCEL
        else:
            target_max = ctx.MAX_WALK_SPEED
            accel_rate = ctx.WALK_ACCEL

        if not self.on_ground:
            accel_rate *= ctx.AIR_CONTROL

        # 2. Horizontal Movement Logic
        if self.input_dir != 0:
            target_vx = self.input_dir * target_max
            skidding = (self.vx > 0 and self.input_dir < 0) or (self.vx < 0 and self.input_dir > 0)
            
            if self.on_ground and skidding:
                self.vx += (self.input_dir * ctx.SKID_DECEL * dt)
            else:
                if self.input_dir > 0: self.vx = min(self.vx + (accel_rate * dt), target_max)
                else: self.vx = max(self.vx - (accel_rate * dt), -target_max)
            self.facing_right = (self.input_dir > 0)
        else:
            friction = (ctx.GROUND_FRICTION if self.on_ground else ctx.AIR_FRICTION) * dt
            if self.vx > 0: self.vx = max(0, self.vx - friction)
            elif self.vx < 0: self.vx = min(0, self.vx + friction)

        # 3. Jump Logic
        self.handle_jump(dt, ctx)

    def handle_jump(self, dt: float, ctx: PhysicsContext):
        # Update buffer using Context frames if needed, otherwise use local counter
        if self.jump_pressed:
            self.jump_buffer = ctx.JUMP_BUFFER_FRAMES

        self.coyote = ctx.COYOTE_FRAMES if self.on_ground else max(0, self.coyote - 1)
        if self.jump_buffer > 0: self.jump_buffer -= 1

        if (self.coyote > 0) and (self.jump_hold == 0) and (self.jump_buffer > 0):
            base = ctx.JUMP_VEL_MIN
            bonus = min(2.2, abs(self.vx) * ctx.SPEED_JUMP_BONUS)
            self.vy = base - bonus 
            self.on_ground = False
            self.coyote = 0
            self.jump_hold = ctx.JUMP_HOLD_FRAMES
            self.jump_buffer = 0

        if self.jump_hold > 0:
            if self.jump_pressed:
                # Variable jump height logic
                self.vy -= 0.30 * (dt * 60) 
            self.jump_hold -= 1
                
    def render(self, surface: pygame.Surface, sx: float, sy: float, debug: bool = True):        
        pygame.draw.rect(surface, self.color, (sx, sy, self.gObj.width, self.gObj.height))
        pygame.draw.circle(surface, COLOR_WHITE, (int(sx + (14 if self.facing_right else 6)), int(sy + 8)), self.eye_radius)
        if debug:
            self._debug(surface, sx, sy)
        if self.run_pressed and abs(self.vx) > 100: # Simple threshold
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
        v_end = (int(sx + (self.vx * 5* self.dt)), int(sy + (self.vy * 5 *self.dt)))
        pygame.draw.line(surface, (100, 255, 255), (int(sx + self.gObj.width / 2), int(sy + self.gObj.height / 2)), v_end, 2)