from __future__ import annotations
import numpy as np
import pygame
from .GameObject import GameObject


class MeatboyPlayer:
    """Monolithic Super Meat Boy character — the "old way", like Player/SonicPlayer.

    Owns its own kinematics and input, and inlines what used to be four
    composable abilities (WallSlide -> WallJump -> Move -> Jump) in that fixed
    order. The physics manager applies gravity, integrates position, resolves
    collisions, and writes back contact flags (read next frame).

    Field names (vx, vy, on_ground, gravity_scale, contact_*) match what
    MeatboyPhysicsManager reads, so the manager needs no player-specific code.
    """

    def __init__(self, gObj: GameObject, movement: dict, jump: dict, wall: dict,
                 human_mode: bool = False):
        self.gObj = gObj
        self.mv = movement
        self.jp = jump
        self.wl = wall
        self.human_mode = human_mode

        # kinematics (read/written by the physics manager)
        self.vx = 0.0
        self.vy = 0.0
        self.on_ground = False
        self.gravity_scale = 1.0
        self.facing_right = True
        self.contact_left = False
        self.contact_right = False
        self.contact_ceiling = False

        # jump/wall-jump state
        self.air_lockout = 0            # WallJump locks horizontal accel briefly
        self.coyote = 0
        self.buffer = 0
        self.jumps_left = int(jump.get("max_jumps", 1))
        self._wall_jumped = False       # this-frame flag: WallJump owns vy

        # wall-slide diagnostics (for obs)
        self.sliding = False
        self.wall_dir = 0

        # decoded input for the current frame
        self.move_x = 0                 # -1 / 0 / +1
        self.run_held = False
        self.jump_pressed = False       # rising edge
        self.jump_held = False
        self._prev_jump = False

    def reset_kinematics(self):
        self.vx = self.vy = 0.0
        self.on_ground = False
        self.gravity_scale = 1.0
        self.facing_right = True
        self.contact_left = self.contact_right = self.contact_ceiling = False
        self.air_lockout = 0
        self.coyote = self.buffer = 0
        self.jumps_left = int(self.jp.get("max_jumps", 1))
        self._wall_jumped = False
        self.sliding = False
        self.wall_dir = 0
        self.move_x = 0
        self.run_held = self.jump_pressed = self.jump_held = self._prev_jump = False

    # --- input ---------------------------------------------------------------
    def handle_input(self, action):
        """Decode MultiDiscrete([3,2,2]) = [move_x, run, jump] (or keyboard).
        move_x: 0=idle 1=left 2=right.  jump/run: 0/1 held. jump_pressed = edge."""
        move_i = run_i = jump_i = 0
        if self.human_mode and pygame.get_init():
            keys = pygame.key.get_pressed()
            left = keys[pygame.K_LEFT] or keys[pygame.K_a]
            right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
            move_i = 1 if (left and not right) else 2 if (right and not left) else 0
            run_i = 1 if (keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]) else 0
            jump_i = 1 if (keys[pygame.K_SPACE] or keys[pygame.K_w] or keys[pygame.K_UP]) else 0
        else:
            try:
                a = np.asarray(action).astype(int).reshape(-1)
                move_i, run_i, jump_i = int(a[0]), int(a[1]), int(a[2])
            except (TypeError, ValueError, IndexError):
                move_i = run_i = jump_i = 0

        self.move_x = {0: 0, 1: -1, 2: 1}.get(move_i, 0)
        self.run_held = bool(run_i)
        held = bool(jump_i)
        self.jump_pressed = held and not self._prev_jump
        self.jump_held = held
        self._prev_jump = held

    # --- per-frame control (velocity only; gravity/integration in manager) ----
    def control(self, dt: float, ctx):
        self.gravity_scale = 1.0
        self._wall_slide(dt)
        self._wall_jump(dt)
        self._move(dt)
        self._jump(dt)
        if self.air_lockout > 0:        # decayed once per frame (was ActorController)
            self.air_lockout -= 1

    def _wall_slide(self, dt):
        # +1 = right wall & pushing right, -1 = left wall & pushing left
        wall = 0
        if self.contact_right and self.move_x > 0:
            wall = 1
        elif self.contact_left and self.move_x < 0:
            wall = -1
        self.sliding = (not self.on_ground) and self.vy > 0 and wall != 0
        self.wall_dir = wall
        if self.sliding:
            cap = float(self.wl.get("slide_max_speed", 120.0))
            if self.vy > cap:
                self.vy = cap

    def _wall_jump(self, dt):
        if self.on_ground or not self.jump_pressed:
            return
        # contact-only: available on any wall touch (right wins ties)
        wall = 1 if self.contact_right else -1 if self.contact_left else 0
        if wall == 0:
            return
        push = float(self.wl["wall_jump_push"])
        if self.move_x == -wall:                       # holding away = long jump
            push = float(self.wl.get("wall_jump_push_away", push))
        self.vy = -float(self.wl["wall_jump_vy"])
        self.vx = -wall * push                          # away from the wall
        self.facing_right = (wall < 0)
        self.air_lockout = int(self.wl.get("control_lockout_frames", 6))
        self._wall_jumped = True
        self.jump_pressed = False                       # consume so _jump won't refire

    def _move(self, dt):
        if self.air_lockout > 0:                        # preserve wall-jump push
            return
        p = self.mv
        sprint = self.run_held
        target = p["max_run_speed"] if sprint else p["max_walk_speed"]
        accel = p["run_accel"] if sprint else p["walk_accel"]
        if not self.on_ground:
            accel *= p["air_control"]

        if self.move_x != 0:
            skidding = (self.vx > 0 and self.move_x < 0) or (self.vx < 0 and self.move_x > 0)
            if self.on_ground and skidding:
                self.vx += self.move_x * p.get("skid_decel", accel * 2.0) * dt
            elif self.move_x > 0:
                if self.vx <= target:
                    self.vx = min(self.vx + accel * dt, target)
                else:                                   # above cap: bleed with friction
                    fr = (p["ground_friction"] if self.on_ground else p["air_friction"]) * dt
                    self.vx = max(target, self.vx - fr)
            else:
                if self.vx >= -target:
                    self.vx = max(self.vx - accel * dt, -target)
                else:
                    fr = (p["ground_friction"] if self.on_ground else p["air_friction"]) * dt
                    self.vx = min(-target, self.vx + fr)
        else:
            fr = (p["ground_friction"] if self.on_ground else p["air_friction"]) * dt
            if self.vx > 0:
                self.vx = max(0.0, self.vx - fr)
            elif self.vx < 0:
                self.vx = min(0.0, self.vx + fr)

        if self.vx > 1.0:
            self.facing_right = True
        elif self.vx < -1.0:
            self.facing_right = False

    def _jump(self, dt):
        p = self.jp
        max_jumps = int(p.get("max_jumps", 1))
        if self.on_ground:
            self.coyote = p.get("coyote_frames", 6)
            self.jumps_left = max_jumps
        else:
            self.coyote = max(0, self.coyote - 1)

        if self.jump_pressed:
            self.buffer = p.get("buffer_frames", 6)
        elif self.buffer > 0:
            self.buffer -= 1

        can_ground = self.on_ground or self.coyote > 0
        can_air = self.jumps_left > 0 and max_jumps > 1
        if self.buffer > 0 and (can_ground or can_air):
            self.vy = -float(p["jump_vel"])
            self.on_ground = False
            self.coyote = 0
            self.buffer = 0
            if not can_ground:
                self.jumps_left -= 1
            else:
                self.jumps_left = max(0, max_jumps - 1)

        if self._wall_jumped:                # wall jump owns vy this frame
            self._wall_jumped = False
            return
        if (not self.jump_held) and self.vy < 0:     # variable height cut
            cut = -float(p.get("cut_vel", 200.0))
            if self.vy < cut:
                self.vy = cut

    # --- observation ---------------------------------------------------------
    N_EXTRA = 7

    def obs_extra(self) -> list:
        """Meat-Boy-specific scalar features (was the abilities' obs fields,
        minus the always-on _active flags)."""
        max_j = max(1, int(self.jp.get("max_jumps", 1)))
        return [
            1.0 if self.run_held else 0.0,                               # sprinting
            1.0 if self.coyote > 0 else 0.0,                             # coyote_active
            1.0 if (not self.on_ground and self.vy < 0) else 0.0,        # jump_extendable
            self.jumps_left / max_j,                                     # jumps_left frac
            1.0 if self.sliding else 0.0,                                # wall_slide_active
            float(self.wall_dir),                                        # touching_wall dir
            1.0 if (not self.on_ground and (self.contact_left or self.contact_right)) else 0.0,  # wall_jump_ready
        ]

    # --- render helpers ------------------------------------------------------
    @property
    def x(self):
        return self.gObj.x

    @property
    def y(self):
        return self.gObj.y

    @property
    def width(self):
        return self.gObj.width

    @property
    def height(self):
        return self.gObj.height

    def get_rect(self):
        return self.gObj.get_rect()


if __name__ == "__main__":
    # Self-check: the control pipeline in isolation (no collision).
    mv = dict(walk_accel=2600, run_accel=4800, max_walk_speed=300, max_run_speed=600,
              air_control=1.0, ground_friction=3600, air_friction=300, skid_decel=7000)
    jp = dict(jump_vel=480, cut_vel=150, coyote_frames=6, buffer_frames=6, max_jumps=1)
    wl = dict(slide_max_speed=280, wall_jump_vy=580, wall_jump_push=620,
              wall_jump_push_away=800, control_lockout_frames=6)
    p = MeatboyPlayer(GameObject(0.0, 0.0, 18, 26), mv, jp, wl)

    # grounded jump: move right + jump -> vx>0 and vy<0
    p.on_ground = True
    p.handle_input([2, 0, 1])
    p.control(1 / 60, None)
    assert p.vx > 0, p.vx
    assert p.vy < 0, "jump should set upward velocity"

    # wall jump off a right wall pushes left and up
    p2 = MeatboyPlayer(GameObject(0.0, 0.0, 18, 26), mv, jp, wl)
    p2.on_ground = False
    p2.contact_right = True
    p2.handle_input([0, 0, 1])          # jump edge, no move
    p2.control(1 / 60, None)
    assert p2.vy < 0 and p2.vx < 0, (p2.vx, p2.vy)
    assert p2.air_lockout >= 0
    print("MeatboyPlayer self-check OK")
