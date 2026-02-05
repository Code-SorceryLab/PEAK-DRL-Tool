from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from .GameObject import GameObject
import pygame

from ..Parameters.Map_parameters import COLOR_WHITE, COLOR_STREAK
from ..System.PhysicsManager import PhysicsContext
from ..System.SpriteManager import SpriteManager

@dataclass
class Player:
    gObj: GameObject
    sprite_manager: Optional[SpriteManager] = None
    
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
    
    # --- ANIMATION STATE ---
    anim_state: str = "idle"   # Dictionary Key
    frame_idx: float = 0.0
    anim_timer: float = 0.0
    anim_speed: float = 0.15 

    def update(self, dt: float, context: PhysicsContext):
        self.dt = dt
        if self.invincible_timer > 0: self.invincible_timer -= dt
        
        self.apply_physics(dt, context)

        grav = context.FAST_FALL_GRAV if self.vy > 0 else context.GRAVITY
        self.vy = min(self.vy + (grav * dt), context.MAX_FALL_SPEED)

        self.gObj.x += self.vx * dt
        self.gObj.y += self.vy * dt
        self.on_ground = False
        
        # UPDATE ANIMATION
        self._update_animation(dt)

    def _update_animation(self, dt: float):
        self.anim_timer += dt
        prev_state = self.anim_state

        # 1. STATE LOGIC (Select the Dictionary Key)
        if not self.on_ground:
            # Air States
            if self.vy < -50: self.anim_state = "jump"
            else: self.anim_state = "fall"
        elif abs(self.vx) > 5.0:
            # Moving
            self.anim_state = "run"
            # Dynamic speed adjustment
            speed_ratio = min(1.0, abs(self.vx) / 300.0)
            self.anim_speed = 0.15 - (speed_ratio * 0.10)
        else:
            # Idle
            self.anim_state = "idle"
            self.anim_speed = 0.15

        # 2. FACING
        if self.vx > 1.0: self.facing_right = True
        elif self.vx < -1.0: self.facing_right = False

        # 3. RESET FRAME IF STATE CHANGED
        # Important: Keeps animations snappy
        if self.anim_state != prev_state:
            self.frame_idx = 0
            self.anim_timer = 0
        
        # 4. INCREMENT FRAME
        if self.anim_timer > self.anim_speed:
            self.frame_idx += 1
            self.anim_timer = 0

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
            
        if self.jump_pressed: self.jump_buffer = 6 

    def apply_physics(self, dt: float, ctx: PhysicsContext):
        if self.run_held:
            target_max = ctx.MAX_RUN_SPEED
            accel_rate = ctx.RUN_ACCEL
        else:
            target_max = ctx.MAX_WALK_SPEED
            accel_rate = ctx.WALK_ACCEL

        if not self.on_ground: accel_rate *= ctx.AIR_CONTROL

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

        self.handle_jump(dt, ctx)

    def handle_jump(self, dt: float, ctx: PhysicsContext):
        if self.jump_pressed: self.jump_buffer = ctx.JUMP_BUFFER_FRAMES
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
            if self.jump_pressed: self.vy -= 0.30 * (dt * 60) 
            self.jump_hold -= 1
                
    def render(self, surface: pygame.Surface, sx: float, sy: float, debug: bool = True):
        drawn = False
        if self.sprite_manager:
            img = self.sprite_manager.get_frame(self.anim_state, self.frame_idx, self.facing_right)
            if img:
                off_x = (img.get_width() - self.gObj.width) / 2
                off_y = (img.get_height() - self.gObj.height) 
                surface.blit(img, (sx - off_x, sy - off_y))
                drawn = True
        
        if not drawn:
            pygame.draw.rect(surface, self.color, (sx, sy, self.gObj.width, self.gObj.height))
        
        if debug:
            self._debug(surface, sx, sy)
            
    def _debug(self, surface: pygame.Surface, sx: float, sy: float):
        pygame.draw.rect(surface, (255, 255, 255), (sx, sy, self.gObj.width, self.gObj.height), 1)
        v_end = (int(sx + (self.vx * 5* self.dt)), int(sy + (self.vy * 5 *self.dt)))
        pygame.draw.line(surface, (100, 255, 255), (int(sx + self.gObj.width / 2), int(sy + self.gObj.height / 2)), v_end, 2)