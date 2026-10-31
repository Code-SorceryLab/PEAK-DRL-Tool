"""Sonic: the 1991 Mega Drive hedgehog platformer's Green Hill, frame for frame, on this repo's engine.

The player routine keeps the console's order of operations and its fixed-point units: positions are
in 1/256 px, speeds in 1/256 px per frame, angles in 1/256 of a turn. Numbers are written in the
original's hex so they can be checked against the public physics guide. The one addition is the
Spin Dash, which the 1991 game did not have: it follows the guide's rule from the later games.

Headless by default. render_mode="human" keeps the animation, effects and sound state; the physics
do not depend on it, so a replay matches its training run.
"""
from __future__ import annotations

import math
import os
from bisect import bisect_left, bisect_right
from types import SimpleNamespace

import pygame
import yaml

from . import mutants, sonic_level as lv
from .modules.System.debugging_mods.manager import DebugManager
from .sonic_objects import ARENA_X, Boss, Effect, RingScatter, animate_player, build_objects, draw_world

HERE = os.path.dirname(os.path.abspath(__file__))
SIN = [int(round(256 * math.sin(i * math.pi / 128))) for i in range(256)]
COS = [SIN[(i + 64) & 255] for i in range(256)]

ACC, DEC, TOP = 0x0C, 0x80, 0x600
GRAVITY, JUMP_CUT = 0x38, -0x400
HURT_GRAVITY, INVULN = 0x30, 120
FALL_SPEED = 0x280
AIR_CAP = 0xFC0
SHOES_ACC, SHOES_DEC, SHOES_TOP = 0x18, 0x80, 0xC00
TIME_LIMIT = 10 * 60 * 60        # frames: the clock kills at 10:00
STAND_H, STAND_W, ROLL_H, ROLL_W = 19, 9, 14, 7
DEBUG_PANEL_WIDTH = 350          # the side debug panel in play, as Mario's and Meat Boy's
COLS = 32                        # px per grid cell of the sensor window, as Mario's tile


def angle_of(dx: int, dy: int) -> int:
    return 0x40 if dx == 0 and dy == 0 else int(math.atan2(dy, dx) * 128 / math.pi) & 255


def quadrant(a: int) -> int:
    """The console's rounding of an angle to a collision mode: a shallow slope rounds toward the
    floor on one side and away on the other, so the mode does not chatter at exactly 45 degrees."""
    if (a + 0x20) & 0x80:
        d = (a - 1 if a & 0x80 else a) + 0x20
    else:
        d = (a + 1 if a & 0x80 else a) + 0x1F
    return d & 0xC0


class Player:
    __slots__ = ("x", "y", "vx", "vy", "gsp", "angle", "air", "rolling", "jumping", "roll_jump", "pushing",
                 "facing", "h", "w", "lock", "routine", "invuln", "invincible", "shoes", "shield", "plane",
                 "on_obj", "spin", "balance", "spring_anim", "idle", "frame", "anim_t", "pose", "acc", "dec", "top")

    def __init__(self, x: int, y: int):
        self.x, self.y = x << 8, y << 8
        self.vx = self.vy = self.gsp = 0
        self.angle = 0
        self.air, self.rolling, self.jumping, self.roll_jump, self.pushing = True, False, False, False, False
        self.facing = 1
        self.h, self.w = STAND_H, STAND_W
        self.lock = 0
        self.routine = "control"            # control / hurt / dead
        self.invuln = self.invincible = self.shoes = 0
        self.shield = False
        self.plane = 0
        self.on_obj = None
        self.spin = None                    # the Spin Dash charge, None when not crouched and charging
        self.balance = self.spring_anim = False
        self.idle = self.frame = self.anim_t = 0
        self.pose = "idle_0"
        self.acc, self.dec, self.top = ACC, DEC, TOP

    @property
    def gObj(self) -> SimpleNamespace:
        """The body as the debug panel reads every game's player."""
        return SimpleNamespace(x=self.px - self.w, y=self.py - self.h, width=2 * self.w + 1, height=2 * self.h + 1)

    @property
    def px(self) -> int:
        return self.x >> 8

    @property
    def py(self) -> int:
        return self.y >> 8


class SonicCore:
    VIEW = (320, 224)                       # the console's picture
    WIDTH, HEIGHT = 960, 672                # the window: the picture at 3x
    DEBUG_PANEL_X = WIDTH
    fps = 60
    hud_lives = None                        # manual_play's count of lives after this one

    def __init__(self, render_mode: str = "none", level: str | None = None, max_steps: int = TIME_LIMIT,
                 stall_frames: int = 0):
        with open(os.path.join(HERE, "sonic_config.yaml"), encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f) or {}
        self.mutant = mutants.active("sonic")      # PEAK_MUTANT: play that copy of the game
        mutants.apply_config(self.cfg, self.mutant)
        self.jump = int(self.cfg["physics"]["jump"])
        self.render_mode = render_mode
        self._cam_offset = [0.0, 0.0]       # F2 free camera
        self._odd_frame = False             # F3 slow motion runs every other frame
        if render_mode == "human":          # the core owns the window and its F1-F7 panel, as Meat Boy's does
            pygame.init()
            pygame.display.set_caption("SONIC (PEAK)")
            self._surf = pygame.display.set_mode((self.TOTAL_WIDTH, self.HEIGHT))
            self.debug_manager = DebugManager(print_help=True)
        self.names = list(self.cfg["levels"])
        self.level_name = level or self.names[0]
        self.max_steps, self.stall_frames = max_steps, stall_frames
        self.extra_lives = 0                # manual_play moves these into its own count of lives
        self.score = 0
        self.won = False
        self.tile_size = COLS
        self._view = None
        self.checkpoint = None
        self.obs_width = self.obs_height = 21
        self.obs_pad_x = self.obs_pad_y = 10
        self.reset()

    # ── state ────────────────────────────────────────────────────────────────

    def reset(self, options: dict | None = None, fresh: bool = False) -> None:
        """A new attempt. Headless every reset is fresh. In play a win moves on to the next act, a death
        keeps the lamp post and the score, and the last life's death starts over."""
        if self.won and self.render_mode != "none":
            i = self.names.index(self.level_name) + 1
            self.level_name = self.names[i % len(self.names)]
            self.checkpoint = None
        if fresh or self.render_mode == "none" or self.hud_lives == 0:
            self.checkpoint = None
            self.score = 0
        self.level = self.level_data = lv.load(self.level_name, mutants.level_path(self.mutant, f"sonic/{self.level_name}.yaml"))
        L = self.level
        self.masks = list(L.masks)
        self.hit = 3                            # the mask bits the wall and ceiling sensors can see
        self.collapsed: list = []
        self.W, self.H, self.tw = L.px_w, L.px_h, L.tw
        self.size = self.W * self.H
        self.objs, self.rings, self.ring_grid = build_objects(self)
        self.ring_on = bytearray(b"\x01") * len(self.rings)
        self.xs = [o.x0 for o in self.objs]
        self.awake: list = []
        self.obj_window = -1
        self.extra: list = []                   # spawned objects: shots, scattered rings, effects
        sx, sy = self.checkpoint[:2] if self.checkpoint else L.start
        self.player = Player(sx, sy)
        self.player.air = bool(self.checkpoint)
        self.rings_held = 0
        self.coins_total = self.kills_total = 0
        self.frames = self.checkpoint[2] if self.checkpoint else 0      # the clock resumes from the lamp post
        self.chain = 0
        self.keys = self.prev_keys = (0, 0, 0)
        self.camera_x = max(0, min(sx - 160, L.camera_right))
        self.camera_y = max(0, sy - 96)
        self.cam_left, self.cam_right = 0, L.camera_right
        self.bottom = self.bottom_target = 768
        self._update_bottom()
        self.bottom = self.bottom_target
        self.camera_y = min(self.camera_y, self.bottom)
        self.look = 0x60
        self.locked = False                     # the boss arena holds the camera
        self.boss = None
        self.dead = self.won = self.reached_goal = False
        self.death_cause = ""
        self.max_x_seen = float(sx)
        self.stall = 0
        self.lives_awarded = 0
        self.events: list = []                  # sound names for the frame, read by the player's loop
        self.intro = 90 if self.render_mode != "none" else 0     # the title card; manual_play skips it in a row
        self.outro = 0                          # the act's tally on screen
        self.bonus_time = self.bonus_rings = 0
        self.frozen = False

    @property
    def TOTAL_WIDTH(self) -> int:
        return self.WIDTH + DEBUG_PANEL_WIDTH if self.render_mode == "human" else self.WIDTH

    @property
    def alive(self) -> bool:
        return not self.dead

    @property
    def timer(self) -> float:
        return max(TIME_LIMIT - self.frames, 0) / 60.0

    def play(self, name: str) -> None:
        if self.render_mode != "none":
            self.events.append(name)

    def add_score(self, n: int) -> None:
        before = self.score
        self.score += n
        if self.score // 50000 > before // 50000:
            self.extra_lives += 1
            self.play("life")

    def add_rings(self, n: int) -> None:
        before = self.rings_held
        self.rings_held = min(self.rings_held + n, 999)
        self.coins_total += n
        for mark in (100, 200):
            if before < mark <= self.rings_held and self.lives_awarded < mark:
                self.lives_awarded = mark
                self.extra_lives += 1
                self.play("life")
        self.play("ring")

    # ── the frame ────────────────────────────────────────────────────────────

    def step(self, action) -> tuple:
        """action: (move_x -1/0/1, jump 0/1, move_y -1/0/1). Gym's five values, the first two unused."""
        self.events = []
        dm = getattr(self, "debug_manager", None)
        if dm:
            dm.update_input()
            if dm.free_cam_active:              # F2: IJKL
                self._cam_offset = [o + m / 60.0 for o, m in zip(self._cam_offset, dm.current_cam_move)]
            else:
                self._cam_offset = [0.0, 0.0]
            self._odd_frame = not self._odd_frame
            if dm.slow_motion and self._odd_frame:
                return None, 0.0, False, False, {"won": False}
        if self.intro:
            self.intro -= 1
            return None, 0.0, False, False, {"won": False}
        if self.outro:
            self.outro -= 1
            if self.outro == 0:
                self.won = True
            self._sound()
            return None, 0.0, self.won, False, {"won": self.won}
        self.prev_keys = self.keys
        self.keys = (int(action[0]), int(bool(action[1])), int(action[2]) if len(action) > 2 else 0)
        if self.frozen:
            self.keys = (0, 0, 0)
        self.frames += 1
        p = self.player
        if p.routine == "control":
            self._control(p)
        elif p.routine == "hurt":
            self._hurt(p)
        else:
            self._dying(p)
        if p.invuln and p.routine == "control":
            p.invuln -= 1
        if p.invincible:
            p.invincible -= 1
        if p.shoes:
            p.shoes -= 1
            if not p.shoes:
                p.acc, p.dec, p.top = ACC, DEC, TOP
        self._run_objects(p)
        if self.level.act == 3 and self.boss is None and self.camera_x >= ARENA_X:
            self.locked = True
            self.cam_left = self.cam_right = ARENA_X
            self.extra.append(Boss(self))
        self._scroll(p)
        self._update_bottom()
        if self.render_mode != "none":
            animate_player(self, p)
            self._sound()
        return self._finish(p)

    def _sound(self) -> None:
        if self.events and self.render_mode == "human":
            from .sonic_sfx import play
            for name in self.events:
                play(name)

    def _finish(self, p: Player) -> tuple:
        x = p.x / 256.0
        if x > self.max_x_seen:
            self.max_x_seen = x
            self.stall = 0
        else:
            self.stall += 1
        if self.frames >= TIME_LIMIT and p.routine == "control":
            self.kill(p, "Time")
        truncated = self.frames >= self.max_steps
        if self.stall_frames and self.stall >= self.stall_frames and not self.reached_goal:
            self.death_cause = "Stall"
            truncated = True
        return None, 0.0, self.won or self.dead, truncated, {"won": self.won}

    # ── sensors: the distance from a sensor to the surface along one axis ────
    # d = 0 when the sensor sits on the pixel just outside the surface, negative when inside it.

    def _floor(self, x: int, y: int) -> tuple[int, int]:
        if not (0 <= x < self.W and 0 <= y < self.H):
            return 32, lv.NO_ANGLE
        m, W = self.masks[self.player.plane], self.W
        i = y * W + x
        if m[i]:
            n = 1
            while n < 32 and i - n * W >= 0 and m[i - n * W]:
                n += 1
            sy = y - n + 1
            return -n, self.level.angles[self.player.plane][lv.FLOOR][(sy >> 4) * self.tw + (x >> 4)]
        n, end = 1, self.size
        while n < 33:
            j = i + n * W
            if j >= end:
                return 32, lv.NO_ANGLE
            if m[j]:
                return n - 1, self.level.angles[self.player.plane][lv.FLOOR][((y + n) >> 4) * self.tw + (x >> 4)]
            n += 1
        return 32, lv.NO_ANGLE

    def _ceiling(self, x: int, y: int) -> tuple[int, int]:
        if not (0 <= x < self.W and 0 <= y < self.H):
            return 32, lv.NO_ANGLE
        m, W, hit = self.masks[self.player.plane], self.W, self.hit
        i = y * W + x
        if m[i] & hit:                   # hit is 1 in the air, where the inside of a hill does not count
            n = 1
            while n < 32 and i + n * W < self.size and m[i + n * W] & hit:
                n += 1
            sy = y + n - 1
            return -n, self.level.angles[self.player.plane][lv.CEILING][(sy >> 4) * self.tw + (x >> 4)]
        n = 1
        while n < 33:
            j = i - n * W
            if j < 0:
                return 32, lv.NO_ANGLE
            if m[j] & hit:
                return n - 1, self.level.angles[self.player.plane][lv.CEILING][((y - n) >> 4) * self.tw + (x >> 4)]
            n += 1
        return 32, lv.NO_ANGLE

    def _wall(self, x: int, y: int, right: bool) -> tuple[int, int]:
        """Distance to a wall on the right (right=True) or left of the sensor."""
        if not (0 <= y < self.H and 0 <= x < self.W):
            return 32, lv.NO_ANGLE
        m, W, hit = self.masks[self.player.plane], self.W, self.hit
        i, s, side = y * W + x, (1 if right else -1), (lv.RIGHT if right else lv.LEFT)
        lo, hi = y * W, (y + 1) * W
        if m[i] & hit:
            n = 1
            while n < 32 and lo <= i - n * s < hi and m[i - n * s] & hit:
                n += 1
            return -n, self.level.angles[self.player.plane][side][(y >> 4) * self.tw + ((x - s * (n - 1)) >> 4)]
        n = 1
        while n < 33:
            j = i + n * s
            if not lo <= j < hi:
                return 32, lv.NO_ANGLE
            if m[j] & hit:
                return n - 1, self.level.angles[self.player.plane][side][(y >> 4) * self.tw + ((x + s * n) >> 4)]
            n += 1
        return 32, lv.NO_ANGLE

    def solid_at(self, x: float, y: float) -> bool:
        x, y = int(x), int(y)
        return 0 <= x < self.W and 0 <= y < self.H and bool(self.masks[self.player.plane][y * self.W + x])

    def collapse(self, box: tuple) -> None:
        """A ledge fell: this attempt's own copy of the masks loses it (the level's stay whole for the next)."""
        x0, y0, x1, y1 = box
        self.masks = [bytearray(m) if isinstance(m, bytes) else m for m in self.masks]
        for m in self.masks:
            for y in range(y0, y1):
                m[y * self.W + x0:y * self.W + x1] = bytes(x1 - x0)
        self.collapsed.append(box)

    def ground_dist(self, x: int, y: int) -> int:
        """Floor distance on plane 0, for everything that is not the player."""
        if not (0 <= x < self.W and 0 <= y < self.H):
            return 32
        m, W = self.masks[0], self.W
        i = y * W + x
        if m[i]:
            n = 1
            while n < 32 and i - n * W >= 0 and m[i - n * W]:
                n += 1
            return -n
        n, end = 1, self.size
        while n < 33:
            j = i + n * W
            if j >= end:
                return 32
            if m[j]:
                return n - 1
            n += 1
        return 32

    # ── the player: ground and air, in the console's order ───────────────────

    def _control(self, p: Player) -> None:
        self.hit = 1 if p.air else 3
        if p.air:
            p.spin = None
            self._air(p)
        elif p.rolling:
            if not self._jump(p):
                self._roll_repel(p)
                self._roll_speed(p)
                self._ground_end(p)
        elif not self._spin_dash(p) and not self._jump(p):
            self._slope_resist(p)
            self._move(p)
            self._roll(p)
            self._ground_end(p)
        self._loop_plane(p)
        if p.routine == "control":
            self._react(p)

    def _ground_end(self, p: Player) -> None:
        self._bound(p)
        p.x += p.vx
        p.y += p.vy
        self._angle_pos(p)
        self._slope_repel(p)

    def _air(self, p: Player) -> None:
        # jump height control
        if p.jumping:
            if p.vy < JUMP_CUT and not self.keys[1]:
                p.vy = JUMP_CUT
        elif p.vy < -AIR_CAP:
            p.vy = -AIR_CAP
        # air control
        if not p.roll_jump:
            speed, top = p.acc * 2, p.top
            if self.keys[0] < 0:
                p.facing = -1
                p.vx = max(p.vx - speed, -top)
            elif self.keys[0] > 0:
                p.facing = 1
                p.vx = min(p.vx + speed, top)
        # air drag near the top of a jump
        if -0x400 <= p.vy < 0:
            d = p.vx >> 5
            if d:
                if p.vx > 0:
                    p.vx = max(p.vx - d, 0)
                else:
                    p.vx = min(p.vx - d, 0)
        self._bound(p)
        p.x += p.vx
        p.y += p.vy
        p.vy += GRAVITY
        # the body rights itself in the air
        if p.angle:
            if p.angle & 0x80:
                p.angle = p.angle + 2 if p.angle + 2 < 256 else 0
            else:
                p.angle = p.angle - 2 if p.angle >= 2 else 0
        self._floor_collide(p)

    def _spin_dash(self, p: Player) -> bool:
        """Down held at rest, jump starts and adds charge, letting go of down fires. True while it holds the body."""
        press = self.keys[1] and not self.prev_keys[1]
        if p.spin is None:
            if not (press and self.keys[2] > 0 and p.gsp == 0 and not (p.angle + 0x20) & 0xC0):
                return False
            p.spin, press = 0.0, False                  # the press that starts it adds nothing
        if self.keys[2] > 0:
            p.spin -= int(p.spin / 0.125) / 256         # the charge drains, as air drag does
            if press:
                p.spin = min(p.spin + 2, 8.0)
                self.play("spin")
            p.vx = p.vy = 0
        else:
            p.gsp = int((8 + int(p.spin) / 2) * 256) * p.facing
            p.spin = None
            self._start_roll(p)
        self._ground_end(p)
        return True

    def _jump(self, p: Player) -> bool:
        if not (self.keys[1] and not self.prev_keys[1]):
            return False
        d, _ = self._overhead(p)
        if d < 6:
            return False
        a = (p.angle - 0x40) & 255
        p.vx += (COS[a] * self.jump) >> 8
        p.vy += (SIN[a] * self.jump) >> 8
        p.air = True
        p.pushing = False
        p.jumping = True
        p.on_obj = None
        self.play("jump")
        p.h, p.w = STAND_H, STAND_W     # a jump out of a roll keeps the standing box, as the 1991 game does
        if p.rolling:
            p.roll_jump = True
        else:
            p.rolling = True
            p.h, p.w = ROLL_H, ROLL_W
            p.y += 5 << 8
        return True

    def _overhead(self, p: Player) -> tuple[int, int]:
        """Clearance above the head along the current floor normal."""
        a = quadrant((p.angle + 0x80) & 255)
        x, y = p.px, p.py
        if a == 0x80:
            return self._two(self._ceiling, x - p.w, y - p.h, x + p.w, y - p.h)
        if a == 0:
            return self._two(self._floor, x - p.w, y + p.h, x + p.w, y + p.h)
        if a == 0x40:
            return min(self._wall(x - p.h, y - p.w, False), self._wall(x - p.h, y + p.w, False), key=lambda t: t[0])
        return min(self._wall(x + p.h, y - p.w, True), self._wall(x + p.h, y + p.w, True), key=lambda t: t[0])

    @staticmethod
    def _two(fn, x0, y0, x1, y1) -> tuple[int, int]:
        a, b = fn(x0, y0), fn(x1, y1)
        return a if a[0] <= b[0] else b

    def _slope_resist(self, p: Player) -> None:
        if ((p.angle + 0x60) & 255) >= 0xC0 or p.gsp == 0:
            return
        p.gsp += (SIN[p.angle] * 0x20) >> 8

    def _roll_repel(self, p: Player) -> None:
        if ((p.angle + 0x60) & 255) >= 0xC0:
            return
        d = (SIN[p.angle] * 0x50) >> 8
        if (p.gsp >= 0 and d < 0) or (p.gsp < 0 and d >= 0):    # rolling uphill: gravity pulls a quarter as hard
            d >>= 2
        p.gsp += d

    def _move(self, p: Player) -> None:
        left, right = self.keys[0] < 0, self.keys[0] > 0
        rest = True
        if not p.lock:
            if left:
                self._walk(p, -1)
            if right:
                self._walk(p, 1)
            p.balance = False
            if not ((p.angle + 0x20) & 0xC0) and p.gsp == 0:
                p.pushing = False
                p.balance = p.on_obj is None and self._floor(p.px, p.py + p.h)[0] >= 12
                if self.keys[2]:
                    self.look = min(self.look + 2, 0xC8) if self.keys[2] < 0 else max(self.look - 2, 8)
                    rest = False
        if rest and self.look != 0x60:
            self.look += 2 if self.look < 0x60 else -2
        if not left and not right and p.gsp:
            p.gsp = max(p.gsp - p.acc, 0) if p.gsp > 0 else min(p.gsp + p.acc, 0)
        p.vx = (COS[p.angle] * p.gsp) >> 8
        p.vy = (SIN[p.angle] * p.gsp) >> 8
        self._wall_stop(p)

    def _walk(self, p: Player, d: int) -> None:
        g = p.gsp * d                                 # speed along the pressed direction
        if g >= 0:
            if p.facing != d:
                p.facing = d
                p.pushing = False
            g = min(g + p.acc, p.top)                 # the 1991 game clamps: pressing forward cannot exceed top
        else:
            g += p.dec
            if g > 0:
                g = 0x80                              # braking past zero lands at half a pixel the other way
            if not ((p.angle + 0x20) & 0xC0) and -g >= 0x400:
                self.play("skid")
        p.gsp = g * d

    def _roll_speed(self, p: Player) -> None:
        frc, dec = p.acc >> 1, p.dec >> 2
        if not p.lock:
            if self.keys[0] < 0:
                self._roll_brake(p, -1, dec)
            if self.keys[0] > 0:
                self._roll_brake(p, 1, dec)
        if p.gsp > 0:
            p.gsp = max(p.gsp - frc, 0)
        elif p.gsp < 0:
            p.gsp = min(p.gsp + frc, 0)
        if p.gsp == 0:
            p.rolling = False
            p.h, p.w = STAND_H, STAND_W
            p.y -= 5 << 8
        p.vy = (SIN[p.angle] * p.gsp) >> 8
        p.vx = max(min((COS[p.angle] * p.gsp) >> 8, 0x1000), -0x1000)
        self._wall_stop(p)

    @staticmethod
    def _roll_brake(p: Player, d: int, dec: int) -> None:
        g = p.gsp * d
        if g >= 0:
            p.facing = d
        else:
            g += dec
            p.gsp = (0x80 if g > 0 else g) * d

    def _roll(self, p: Player) -> None:
        if p.rolling or abs(p.gsp) < 0x80 or self.keys[0] or self.keys[2] <= 0:
            return
        self._start_roll(p)

    def _start_roll(self, p: Player) -> None:
        p.rolling = True
        p.h, p.w = ROLL_H, ROLL_W
        p.y += 5 << 8
        self.play("roll")
        if p.gsp == 0:
            p.gsp = 0x200

    def _wall_stop(self, p: Player) -> None:
        """Stop against a wall ahead, read at next frame's position, only on floor-like angles."""
        if ((p.angle + 0x40) & 0x80) or p.gsp == 0:
            return
        a = (p.angle + (-0x40 if p.gsp > 0 else 0x40)) & 255
        d = self._ahead(p, a)
        if d >= 0:
            return
        d <<= 8
        q = (a + 0x20) & 0xC0
        if q == 0:
            p.vy += d
        elif q == 0x40:
            p.vx -= d
            p.pushing = True
            p.gsp = 0
        elif q == 0x80:
            p.vy -= d
        else:
            p.vx += d
            p.pushing = True
            p.gsp = 0

    def _ahead(self, p: Player, a: int) -> int:
        x, y = (p.x + p.vx) >> 8, (p.y + p.vy) >> 8
        q = quadrant(a)
        if q == 0:
            return self._floor(x, y + 10)[0]
        if q == 0x80:
            return self._ceiling(x, y - 10)[0]
        if not a & 0x38:
            y += 8
        if q == 0x40:
            return self._wall(x - 10, y, False)[0]
        return self._wall(x + 10, y, True)[0]

    def _angle_pos(self, p: Player) -> None:
        if p.on_obj is not None:
            return
        x, y = p.px, p.py
        q = quadrant(p.angle)
        h, w = p.h, p.w
        if q == 0:
            a, b = self._floor(x + w, y + h), self._floor(x - w, y + h)
        elif q == 0x80:
            a, b = self._ceiling(x + w, y - h), self._ceiling(x - w, y - h)
        elif q == 0xC0:
            a, b = self._wall(x + h, y - w, True), self._wall(x + h, y + w, True)
        else:
            a, b = self._wall(x - h, y - w, False), self._wall(x - h, y + w, False)
        d, ang = b if b[0] <= a[0] else a
        if not ang & 1:             # no surface data here: keep the angle rather than jolt the body to flat
            p.angle = ang
        if d == 0 or d < -14:
            return
        if d > 14:
            p.air = True
            p.pushing = False
            return
        if q == 0:
            p.y += d << 8
        elif q == 0x80:
            p.y -= d << 8
        elif q == 0xC0:
            p.x += d << 8
        else:
            p.x -= d << 8

    def _slope_repel(self, p: Player) -> None:
        if p.lock:                  # the control lock only runs down here, on the ground
            p.lock -= 1
        elif ((p.angle + 0x20) & 0xC0) and abs(p.gsp) < FALL_SPEED:
            p.gsp = 0
            p.air = True
            p.lock = 30

    def _floor_collide(self, p: Player) -> None:
        """Landing and bumping in the air: which sides are checked depends on the way the body moves."""
        q = (angle_of(p.vx, p.vy) - 0x20) & 0xC0
        x, y = p.px, p.py
        h, w = p.h, p.w
        if q == 0:                                      # mostly down
            self._air_walls(p)
            x = p.px
            a, b = self._floor(x + w, y + h), self._floor(x - w, y + h)
            lo, hi = (a, b) if a[0] <= b[0] else (b, a)
            d = lo[0]
            if d >= 0:
                return
            lim = -((p.vy >> 8) + 8)
            if d < lim and hi[0] < lim:
                return
            ang = 0 if lo[1] & 1 else lo[1]
            p.y += d << 8
            p.angle = ang
            self._land(p)
            if not (ang + 0x20) & 0x40:
                if (ang + 0x10) & 0x20:
                    p.vy >>= 1
                else:
                    p.vy = 0
                    p.gsp = p.vx
                    return
            else:
                p.vx = 0
                p.vy = min(p.vy, AIR_CAP)
            p.gsp = -p.vy if ang & 0x80 else p.vy
        elif q == 0x80:                                 # mostly up
            self._air_walls(p)
            x = p.px
            d, ang = self._two(self._ceiling, x + w, y - h, x - w, y - h)
            if d >= 0:
                return
            p.y -= d << 8
            ang = 0x80 if ang & 1 else ang
            if (ang + 0x20) & 0x40:                     # a steep ceiling catches the body
                p.angle = ang
                self._land(p)
                p.gsp = -p.vy if ang & 0x80 else p.vy
            else:
                p.vy = 0
        else:                                           # mostly left or right
            right = q == 0xC0
            d, _ = self._wall(x + 10, y, True) if right else self._wall(x - 10, y, False)
            if d < 0:
                p.x += (d if right else -d) << 8
                p.vx = 0
                p.gsp = p.vy
                return
            d, ang = self._two(self._ceiling, x + w, y - h, x - w, y - h)
            if d < 0:
                p.y -= d << 8
                p.vy = max(p.vy, 0)
                return
            if p.vy >= 0:
                d, ang = self._two(self._floor, x + w, y + h, x - w, y + h)
                if d < 0:
                    p.y += d << 8
                    p.angle = 0 if ang & 1 else ang
                    self._land(p)
                    p.vy = 0
                    p.gsp = p.vx

    def _air_walls(self, p: Player) -> None:
        x, y = p.px, p.py
        d, _ = self._wall(x - 10, y, False)
        if d < 0:
            p.x -= d << 8
            p.vx = 0
        d, _ = self._wall(x + 10, y, True)
        if d < 0:
            p.x += d << 8
            p.vx = 0

    def _land(self, p: Player) -> None:
        p.pushing = False
        p.air = False
        p.roll_jump = False
        if p.rolling:
            p.rolling = False
            p.h, p.w = STAND_H, STAND_W
            p.y -= 5 << 8
        p.jumping = False
        self.chain = 0

    def _bound(self, p: Player) -> None:
        nx = (p.x + p.vx) >> 8
        left, right = self.cam_left + 16, self.cam_right + 0x128 + (0 if self.locked else 0x40)
        if nx < left or nx > right:
            p.x = (left if nx < left else right) << 8
            p.vx = p.gsp = 0
        if p.py > self.bottom + 0xE0:
            self.kill(p, "Fell")

    def _loop_plane(self, p: Player) -> None:
        """Which plane the body is on: the 1991 rule, by column inside a loop, then by angle."""
        x, y = p.px, p.py
        for cx, cy, r, floor in self.level.loops:
            if cx - 128 <= x < cx + 128 and floor - 224 <= y <= floor + 32:
                local = x - (cx - 128)
                if p.air and y < cy:
                    p.plane = 0
                elif local < 0x2C:
                    p.plane = 0
                elif local >= 0xE0:
                    p.plane = 1
                elif p.plane == 0:
                    if p.angle and p.angle <= 0x80:
                        p.plane = 1
                elif p.angle > 0x80:
                    p.plane = 0
                return
        p.plane = 0
        if self._in_tube(p) and not p.rolling and not p.air:
            self._start_roll(p)

    def _in_tube(self, p: Player) -> bool:
        x, y = p.px, p.py
        for x0, y0, x1, y1 in self.level.tubes:
            if x0 + 16 <= x < x1 - 16 and y0 <= y < y1:
                return True
        return False

    # ── hurt and death ───────────────────────────────────────────────────────

    def hurt(self, p: Player, src_x: int, spikes: bool = False) -> None:
        if p.routine != "control" or p.invincible or (p.invuln and not spikes):     # the 1991 spike bug: flashing does not save you
            return
        if p.shield:
            p.shield = False
        elif self.rings_held:
            self.extra.extend(RingScatter.burst(self, p, self.rings_held))
            self.rings_held = 0
        else:
            self.kill(p, "Spikes" if spikes else "Enemy")
            return
        self._land(p)
        p.air = True
        p.on_obj = None
        p.routine = "hurt"
        p.vy, p.vx, p.gsp = -0x400, -0x200, 0
        if p.px >= src_x:
            p.vx = 0x200
        p.invuln = INVULN
        self.play("spikes" if spikes else "hurt")

    def _hurt(self, p: Player) -> None:
        self.hit = 1
        p.x += p.vx
        p.y += p.vy
        p.vy += HURT_GRAVITY
        self._bound(p)
        if p.routine != "hurt":
            return
        self._floor_collide(p)
        if not p.air:
            p.vx = p.vy = p.gsp = 0
            p.routine = "control"
            p.invuln = INVULN

    def kill(self, p: Player, cause: str) -> None:
        if p.routine == "dead":
            return
        self.death_cause = cause
        p.routine = "dead"
        p.vx, p.gsp, p.vy = 0, 0, -0x700
        p.air = True
        p.shield = False
        p.invincible = 0
        p.on_obj = None
        self.play("hurt")
        if self.render_mode == "none":
            self.dead = True

    def _dying(self, p: Player) -> None:
        p.y += p.vy
        p.vy += GRAVITY
        if p.py > (self.camera_y + 256 + 32):
            self.dead = True

    # ── objects ──────────────────────────────────────────────────────────────

    def _run_objects(self, p: Player) -> None:
        cam = (int(self.camera_x) - 128) & ~0x7F
        if cam != self.obj_window:
            lo, hi = bisect_left(self.xs, cam), bisect_right(self.xs, cam + 0x280 + 127)
            self.obj_window = cam
            keep = set(range(lo, hi))
            for o in self.awake:
                if o.index not in keep:
                    o.sleep(self)
            self.awake = [self.objs[i] for i in range(lo, hi) if self.objs[i].alive]
            for o in self.awake:
                if not o.awake:
                    o.wake(self)
        for o in self.awake:
            if o.alive:
                o.update(self, p)
        if self.extra:
            for e in self.extra:
                e.update(self, p)
            self.extra = [e for e in self.extra if e.alive]

    def _react(self, p: Player) -> None:
        """Rings are too many for the object loop: they live in a grid keyed by 64 px columns."""
        if p.invuln >= 90:          # a body still reeling from a hit cannot pick its rings back up
            return
        px, py = p.px, p.py
        bx0, bx1 = (px - 16) >> 6, (px + 16) >> 6
        for b in range(bx0, bx1 + 1):
            for i in self.ring_grid.get(b, ()):
                if self.ring_on[i]:
                    rx, ry = self.rings[i]
                    if abs(rx - px) < 14 and ry - 6 - p.h + 3 < py < ry + 6 + p.h - 3:
                        self.ring_on[i] = 0
                        self.add_rings(1)
                        if self.render_mode != "none":
                            self.extra.append(Effect(rx, ry, "sparkle", 4, 6))

    def touch_box(self, p: Player) -> tuple[int, int, int, int]:
        """(left, top, width, height) of the body against objects: 16 wide, a little shorter than it stands."""
        h = p.h - 3
        top = p.py - h
        ducking = not p.air and not p.rolling and p.gsp == 0 and self.keys[2] > 0
        if ducking:
            top += 12
            h = 10
        return p.px - 8, top, 16, h * 2

    def finish_act(self) -> None:
        """Tally the act: the time bonus goes by 15 s steps on the clock, rings pay 100 each."""
        table = (50000, 50000, 10000, 5000, 4000, 4000, 3000, 3000, 2000, 2000, 2000, 2000, 1000, 1000, 1000, 1000,
                 500, 500, 500, 500, 0)
        self.bonus_time = table[min(self.frames // 60 // 15, 20)]
        self.bonus_rings = self.rings_held * 100
        self.add_score(self.bonus_time + self.bonus_rings)
        self.frozen = True
        self.outro = 240

    # ── camera and the moving bottom edge ────────────────────────────────────

    def _scroll(self, p: Player) -> None:
        x = p.px - int(self.camera_x)
        if x < 144:
            self.camera_x = max(self.camera_x + max(x - 144, -16), self.cam_left)
        elif x >= 160:
            self.camera_x = min(self.camera_x + min(x - 160, 16), self.cam_right)
        y = p.py - int(self.camera_y) - (5 if p.rolling else 0)
        speed = 2 if self.look != 0x60 else (6 if abs(p.gsp) < 0x800 else 16)
        if p.air:
            d = y + 32 - self.look
            if d < 0:
                move = max(d, -16)
            elif d >= 64:
                move = min(d - 64, 16)
            else:
                move = 0
        else:
            d = y - self.look
            move = max(min(d, speed), -speed)
        self.camera_y = max(min(self.camera_y + move, self.bottom), 0)

    def _update_bottom(self) -> None:
        """How low the camera may go at this point of the act (the original's dynamic level events)."""
        cx, cy, act = int(self.camera_x), int(self.camera_y), self.level.act
        if act == 1:
            b = 0x400 if cx >= 0x1780 else 0x300
        elif act == 2:
            b = 0x300
            if cx >= 0xED0:
                b = 0x200
            if cx >= 0x1600:
                b = 0x400
            if cx >= 0x1D60:
                b = 0x300
        else:
            b = 0x310 if cx >= 0x380 else 0x300
            if cx >= 0x960:
                if cy >= 0x280:
                    b = 0x4C0 if cx < 0x1380 else 0x400
                    if cx >= 0x1700:
                        b = 0x300
                else:
                    b = 0x300
        self.bottom_target = b
        if self.bottom < b:
            self.bottom = min(self.bottom + (8 if self.player and self.player.air else 2), b)
        elif self.bottom > b:
            self.bottom = max(self.bottom - 2, b)

    # ── what the agents read ─────────────────────────────────────────────────

    @property
    def enemies(self) -> list:
        return [o for o in self.awake if o.alive and o.enemy]

    def hazards(self) -> list:
        return [o for o in self.awake if o.alive and (o.enemy or o.hazard)]

    def _obs(self) -> dict:
        """The (solid, collectible, hazard) window around the body, for the sensor ablation."""
        import numpy as np
        n, half, s = self.obs_width, self.obs_pad_x, COLS
        grids = np.zeros((3, n, n), dtype=np.float32)
        p = self.player
        ox, oy = (p.px // s - half) * s, (p.py // s - half) * s
        for ly in range(n):
            for lx in range(n):
                if self.solid_at(ox + lx * s + s // 2, oy + ly * s + s // 2):
                    grids[0, ly, lx] = 1.0
        for b in range(ox >> 6, ((ox + n * s) >> 6) + 1):
            for i in self.ring_grid.get(b, ()):
                if self.ring_on[i]:
                    lx, ly = (self.rings[i][0] - ox) // s, (self.rings[i][1] - oy) // s
                    if 0 <= lx < n and 0 <= ly < n:
                        grids[1, ly, lx] = 1.0
        for o in self.hazards():
            lx, ly = (o.x - ox) // s, (o.y - oy) // s
            if 0 <= lx < n and 0 <= ly < n:
                grids[2, ly, lx] = 1.0
        return {"grids": grids}

    # ── drawing (off the physics path) ───────────────────────────────────────

    def render(self, surface, blit_only: bool = False) -> None:
        from . import sonic_art as art
        window = surface
        if surface.get_width() > self.WIDTH:            # the window with the debug panel
            surface = surface.subsurface((0, 0, self.WIDTH, min(self.HEIGHT, surface.get_height())))
        if self._view is None:
            self._view = pygame.Surface(self.VIEW)
        dm = getattr(self, "debug_manager", None)
        self.camera_x += self._cam_offset[0]            # F2 moves the picture, never the game's own camera
        self.camera_y += self._cam_offset[1]
        draw_world(self, self._view)
        if dm:
            self._debug_on_view(self._view, dm)
        self.camera_x -= self._cam_offset[0]
        self.camera_y -= self._cam_offset[1]
        if self.intro:
            art.title_card(self._view, self.level.act, 1.0 - self.intro / 90)
        elif self.outro:
            art.results(self._view, self.level.act, self.bonus_time, self.bonus_rings, self.score,
                        min((240 - self.outro) / 90, 1.0))
        pygame.transform.scale(self._view, surface.get_size(), surface)
        if dm:
            dm.render_overlays(window, self, game_overlays=False)

    def _debug_on_view(self, view, dm) -> None:
        """F1, F4 and F6 on the console's picture, so they line up with the world before the scale-up."""
        cx, cy = int(self.camera_x), int(self.camera_y)
        p = self.player
        if dm.show_grid:                                # F6: the 16 px tiles
            for x in range(-(cx % 16), 320, 16):
                pygame.draw.line(view, (70, 80, 110), (x, 0), (x, 224))
            for y in range(-(cy % 16), 224, 16):
                pygame.draw.line(view, (70, 80, 110), (0, y), (320, y))
        if dm.show_hitboxes:                            # F4: the body, what it touches with, every hitbox
            pygame.draw.rect(view, (60, 235, 60), (p.px - p.w - cx, p.py - p.h - cy, 2 * p.w + 1, 2 * p.h + 1), 1)
            bx, by, bw, bh = self.touch_box(p)
            pygame.draw.rect(view, (255, 235, 60), (bx - cx, by - cy, bw, bh), 1)
            for o in self.awake:
                if o.alive and o.enemy:
                    pygame.draw.rect(view, (255, 55, 55), (o.x - o.hw - cx, o.y - o.hh - cy, 2 * o.hw, 2 * o.hh), 1)
        if dm.show_sensors:                             # F1: what an agent senses from here, and the body's sensors
            from code.neuro.adapters import SonicAdapter
            from code.neuro.sensors import read_sensors
            probe = SonicAdapter.__new__(SonicAdapter)
            probe.core, probe.tile_size = self, self.tile_size
            for x1, y1, x2, y2, hit in read_sensors(probe)[1]:
                pygame.draw.line(view, (255, 120, 30) if hit == 1 else (255, 40, 40) if hit == 2 else (255, 230, 90),
                                 (x1 - cx, y1 - cy), (x2 - cx, y2 - cy))
            for sx, sy in ((p.px - p.w, p.py + p.h), (p.px + p.w, p.py + p.h), (p.px - 10, p.py), (p.px + 10, p.py),
                           (p.px - p.w, p.py - p.h), (p.px + p.w, p.py - p.h)):
                pygame.draw.circle(view, (255, 255, 255), (sx - cx, sy - cy), 1)

    # ---- debug panel hooks (DebugManager / overlays.py read these, as they read Mario's core) ----
    @property
    def frame(self) -> int:
        return self.frames

    @property
    def gravity(self) -> float:
        return GRAVITY / 256.0

    @property
    def jump_force(self) -> float:
        return self.jump / 256.0

    def _world_to_screen(self, obj) -> tuple:
        return obj.x - self.camera_x, obj.y - self.camera_y, True

    def _player_obs(self):
        import numpy as np
        p = self.player
        return np.array([p.px, p.py, p.vx / 256, p.vy / 256, not p.air, p.facing, p.angle, p.plane], dtype=np.float32)

    def _tracking_obs(self):
        import numpy as np
        p = self.player
        return np.array([p.gsp / 256, p.rolling, p.spin or 0.0, p.lock, p.invuln, self.rings_held], dtype=np.float32)

    def get_obs_value_labels(self) -> tuple:
        return ["Px", "Py", "Vx", "Vy", "Grnd", "Face", "Angle", "Plane"], ["Gsp", "Roll", "Spin", "Lock", "Inv", "Rings"]

    def get_debug_hud_strip_items(self) -> list:
        p = self.player
        state = "HURT" if p.routine == "hurt" else "DEAD" if p.routine == "dead" else "SPIN" if p.spin is not None \
            else "ROLL" if p.rolling else "AIR" if p.air else "RUN" if p.gsp else "STAND"
        return [("LEVEL", f"ACT {self.level.act}", (210, 215, 228)),
                ("TIME", f"{self.frames / 60:.2f}", (210, 215, 228)),
                ("STATE", state, (80, 225, 115)),
                ("ANGLE", f"{p.angle:02X}", (210, 215, 228)),
                ("SPEED", f"{abs(p.gsp) / 256:.2f}", (85, 190, 255)),
                ("PLANE", str(p.plane), (210, 215, 228))]
