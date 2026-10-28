from __future__ import annotations
import numpy as np
import pygame
from .GameObject import GameObject


class MeatboyPlayer:
    """Monolithic Super Meat Boy character (the "old way", like Player/SonicPlayer).

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
        self.reset_kinematics()

    def reset_kinematics(self):
        # kinematics (read/written by the physics manager)
        self.vx = self.vy = 0.0
        self.on_ground = False
        self.gravity_scale = 1.0
        self.facing_right = True
        self.contact_left = self.contact_right = self.contact_ceiling = False
        # jump: rising edge only, no buffer and no coyote time (DiGRA 2016)
        self.jumping = False            # going up from a jump the button still controls
        self.jump_frames = 0
        # wall
        self.sliding = False
        self.wall_dir = 0               # +1 wall on the right, -1 on the left (in the air), 0 none
        self.stick = 0                  # frames left on the wall while pulling away
        self.stick_held = False
        self.fall_cap = None
        self.wall_jumps = 0             # this life's wall jumps (episode stats)
        # decoded input for the current frame
        self.move_x = 0                 # -1 / 0 / +1
        self.run_held = False
        self.jump_pressed = False       # rising edge
        self.jump_held = False
        self._prev_jump = False

    # --- input ---------------------------------------------------------------
    def handle_input(self, action):
        """Decode MultiDiscrete([3,2,2]) = [move_x, run, jump].
        move_x: 0=idle 1=left 2=right.  jump/run: 0/1 held. jump_pressed = edge.

        The AGENT ACTION always drives (so a rendered eval/watch window, which
        runs render_mode='human', still moves under the policy). Keyboard is
        additive on top, for a real human in manual play (mirrors Platformer's
        Player.handle_input). Reading keyboard instead of the action here was
        the bug that froze the agent during eval."""
        try:
            a = np.asarray(action).astype(int).reshape(-1)
            a_move, a_run, a_jump = int(a[0]), int(a[1]), int(a[2])
        except (TypeError, ValueError, IndexError):
            a_move = a_run = a_jump = 0

        a_left, a_right = (a_move == 1), (a_move == 2)
        run_held = bool(a_run)
        jump_held = bool(a_jump)

        # Keyboard is additive (only a real human presses keys; empty during
        # headless training and agent-driven eval).
        if self.human_mode and pygame.get_init():
            keys = pygame.key.get_pressed()
            a_left = a_left or keys[pygame.K_LEFT] or keys[pygame.K_a]
            a_right = a_right or keys[pygame.K_RIGHT] or keys[pygame.K_d]
            run_held = run_held or keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
            jump_held = jump_held or keys[pygame.K_SPACE] or keys[pygame.K_w] or keys[pygame.K_UP]

        if a_left and not a_right:
            self.move_x = -1
        elif a_right and not a_left:
            self.move_x = 1
        else:
            self.move_x = 0
        self.run_held = bool(run_held)
        held = bool(jump_held)
        self.jump_pressed = held and not self._prev_jump
        self.jump_held = held
        self._prev_jump = held

    # --- per-frame control (velocity only; gravity/integration in manager) ----
    # Super Meat Boy's rules, values and sources in docs/MEATBOY_SMB_RESEARCH.md
    def control(self, dt: float, ctx):
        self.gravity_scale = 1.0
        self._wall()
        self._move(dt)
        self._jump()

    def _wall(self):
        wl = self.wl
        self.fall_cap = None                            # the physics manager's fall cap this frame, if lower
        self.stick_held = False
        wall = 0 if self.on_ground else 1 if self.contact_right else -1 if self.contact_left else 0
        self.wall_dir = wall
        if not wall:
            self.stick = 0
            self.sliding = False
            return
        if self.jump_pressed:                           # wall jump: up and away from the wall
            self.vy = -float(wl["wall_jump_vy"])
            self.vx = -wall * float(wl["wall_jump_push"])
            self.facing_right = wall < 0
            self.jumping, self.jump_frames = True, 0    # a tap gives a short hop, as off the ground
            self.jump_pressed = False
            self.stick = 0
            self.sliding = False
            self.wall_jumps += 1
            return
        if self.move_x == -wall and self.stick > 0:     # pulling away: he stays on the wall 200 ms
            self.stick -= 1
            self.move_x = 0
            self.stick_held = True
        elif self.move_x != -wall:
            self.stick = int(wl["stick_frames"])
        self.sliding = self.vy >= 0 and (self.move_x == wall or self.stick_held)   # let go of the wall: he falls
        if self.sliding:
            self.fall_cap = float(wl["slide_max_speed"])

    def _move(self, dt):
        p = self.mv
        top = p["max_run_speed"] if self.run_held else p["max_walk_speed"]
        if self.move_x:
            accel = p["run_accel"] if self.run_held else p["walk_accel"]
            if self.vx * self.move_x < 0:               # against the motion: twice as hard (the skid)
                accel *= p["turn_factor"]
            self.vx += self.move_x * accel * dt
            self.facing_right = self.move_x > 0
        elif self.on_ground:                            # no direction on the ground: he stops dead
            self.vx = 0.0
        self.vx = max(-top, min(top, self.vx))          # run let go: down to walk speed at once, in the air too

    def _jump(self):
        p = self.jp
        if self.jump_pressed and self.on_ground:
            self.vy = -float(p["jump_vel"])
            self.on_ground = False
            self.jumping, self.jump_frames = True, 0
        if not self.jumping:
            return
        self.jump_frames += 1
        if self.vy >= 0:
            self.jumping = False
        elif not self.jump_held and self.jump_frames >= int(p["min_hold_frames"]):
            self.vy = 0.0                               # let go: the climb stops there
            self.jumping = False

    # --- observation ---------------------------------------------------------
    N_EXTRA = 6

    def obs_extra(self) -> list:
        """Meat-Boy-specific scalar features for the core's observation."""
        return [
            1.0 if self.run_held else 0.0,              # running
            1.0 if self.jumping else 0.0,               # the jump can still be cut
            1.0 if self.sliding else 0.0,               # wall slide
            float(self.wall_dir),                       # wall touched in the air: -1 left, +1 right
            1.0 if self.stick > 0 else 0.0,             # still held on the wall
            1.0 if self.wall_dir else 0.0,              # a wall jump is possible
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
    import yaml, os
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "..", "meatboy_config.yaml")))
    mk = lambda: MeatboyPlayer(GameObject(0.0, 0.0, 22, 22), cfg["movement"], cfg["jump"], cfg["wall"])

    p = mk()                                    # grounded jump: move right + jump -> vx>0 and vy<0
    p.on_ground = True
    p.handle_input([2, 0, 1])
    p.control(1 / 60, None)
    assert p.vx > 0 and p.vy < 0, (p.vx, p.vy)

    p.on_ground = True                          # no direction on the ground: stops dead
    p.handle_input([0, 0, 0])
    p.control(1 / 60, None)
    assert p.vx == 0.0

    p2 = mk()                                   # wall jump off a right wall pushes left and up
    p2.contact_right = True
    p2.handle_input([0, 0, 1])
    p2.control(1 / 60, None)
    assert p2.vy < 0 and p2.vx < 0 and p2.wall_jumps == 1, (p2.vx, p2.vy)
    print("MeatboyPlayer self-check OK")
