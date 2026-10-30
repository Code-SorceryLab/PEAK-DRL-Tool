"""SPEEDY's objects: rings, monitors, springs, spikes, bridges, platforms, badniks, the boss and the end
of the act. Sizes, speeds and timers come from the public physics guide (SPG:Game_Objects,
SPG:Game_Enemies, SPG:Solid_Objects, SPG:Ring_Loss, SPG:Getting_Hit); where it is silent they are the
values of the 1991 game as best known, in the original's units (1/256 px per frame).

Objects past the camera's reach sleep and reset, as the console's object manager does.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

SIN256 = [math.sin(i * math.pi / 128) for i in range(256)]
PUSH = 11                   # the body's push radius + 1 in every solid-object check


def overlap(box: tuple, x: int, y: int, w: int, h: int) -> bool:
    bx, by, bw, bh = box
    return bx < x + w and x < bx + bw and by < y + h and y < by + bh


def land_on(core, p, o, top: int) -> None:
    core._land(p)
    p.y = (top - p.h) << 8
    p.vy = 0
    p.angle = 0
    p.gsp = p.vx
    p.on_obj = o


def leave(core, p) -> None:
    p.on_obj = None
    p.air = True
    p.pushing = False


class Obj:
    enemy = hazard = False
    alive = True
    awake = False
    index = 0

    def __init__(self, x: int, y: int):
        self.x0, self.y0 = x, y
        self.x, self.y = x, y

    @property
    def gObj(self) -> SimpleNamespace:
        """The object as the debug panel reads every game's entities."""
        return SimpleNamespace(x=self.x, y=self.y, width=2 * getattr(self, "hw", 8), height=2 * getattr(self, "hh", 8),
                               active=self.alive)

    def wake(self, core) -> None:
        self.awake = True

    def sleep(self, core) -> None:
        self.awake = False
        if core.player.on_obj is self:
            leave(core, core.player)

    def update(self, core, p) -> None:
        pass

    def solid(self, core, p, hw: int, hh: int) -> str | None:
        """Box solidity: 'top', 'side', 'bottom' or None. The body counts as push radius wide."""
        px, py = p.px, p.py
        cw = hw + PUSH
        if p.on_obj is self:
            left = px - self.x + cw
            if left < 0 or left >= 2 * cw:
                leave(core, p)
                return None
            p.y = ((self.y - hh - p.h) << 8) | (p.y & 255)
            return "top"
        if p.routine == "dead":
            return None
        ch = hh + p.h
        left = px - self.x + cw
        if left < 0 or left > 2 * cw:
            return None
        top = py - self.y + 4 + ch
        if top < 0 or top > 2 * ch + 4:
            return None
        xd = left if px <= self.x else left - 2 * cw
        yd = top if py <= self.y else top - 2 * ch - 4
        if abs(xd) > abs(yd):
            if yd < 0:                              # the underside
                if p.vy == 0 and not p.air:
                    core.kill(p, "Crushed")
                elif p.vy < 0:
                    p.y -= yd << 8
                    p.vy = 0
                return "bottom"
            if yd >= 16 or p.vy < 0:
                return None
            left = px - self.x + hw
            if left < 0 or left > 2 * hw:           # a landing reads the body as one pixel wide
                return None
            land_on(core, p, self, self.y - hh)
            return "top"
        if abs(yd) <= 4:
            return None
        if xd > 0 and p.vx > 0 or xd < 0 and p.vx < 0:
            p.vx = p.gsp = 0
            if not p.air:
                p.pushing = True
        p.x -= xd << 8
        return "side"


# ── loose rings and effects ──────────────────────────────────────────────────

class RingScatter:
    __slots__ = ("x", "y", "vx", "vy", "t", "alive", "last")

    def __init__(self, x, y, vx, vy):
        self.x, self.y, self.vx, self.vy = x << 8, y << 8, vx, vy
        self.t = 0
        self.alive = True

    @classmethod
    def burst(cls, core, p, n: int) -> list:
        out, angle, speed, flip = [], 101.25, 4.0, False
        for i in range(min(n, 32)):
            c, s = math.cos(math.radians(angle)) * speed, -math.sin(math.radians(angle)) * speed
            if flip:
                c = -c
                angle += 22.5
            out.append(cls(p.px, p.py, int(c * 256), int(s * 256)))
            flip = not flip
            if i == 15:
                speed, angle = 2.0, 101.25
        return out

    def update(self, core, p) -> None:
        self.t += 1
        if self.t >= 256:
            self.alive = False
            return
        self.x += self.vx
        self.y += self.vy
        self.vy += 24
        if (core.frames & 3) == 0 and self.vy > 0:
            d = core.ground_dist(self.x >> 8, (self.y >> 8) + 8)
            if d < 0:
                self.y += d << 8
                self.vy = -(self.vy * 3 >> 2)
        if self.t > 64 and p.routine == "control" and p.invuln < 90 and \
                overlap(core.touch_box(p), (self.x >> 8) - 6, (self.y >> 8) - 6, 12, 12):
            self.alive = False
            core.add_rings(1)

    def draw(self, surf, core, art) -> None:
        if self.t < 200 or core.frames & 2:
            s = art.sprite(f"ring_{(core.frames >> 2) & 3}")
            surf.blit(s, s.get_rect(center=((self.x >> 8) - int(core.camera_x), (self.y >> 8) - int(core.camera_y))))


class Effect:
    """A sprite sequence that plays once and goes: sparkle, explosion, points."""

    def __init__(self, x, y, name, frames, hold, vy=0):
        self.x, self.y, self.name, self.frames, self.hold, self.vy = x, y, name, frames, hold, vy
        self.t = 0
        self.alive = True

    def update(self, core, p) -> None:
        self.t += 1
        self.y += self.vy
        if self.t >= self.frames * self.hold:
            self.alive = False

    def draw(self, surf, core, art) -> None:
        name = self.name if self.frames == 1 else f"{self.name}_{min(self.t // self.hold, self.frames - 1)}"
        s = art.sprite(name)
        surf.blit(s, s.get_rect(center=(int(self.x) - int(core.camera_x), int(self.y) - int(core.camera_y))))


class Shot(Obj):
    """A badnik's projectile: hurts on touch, leaves with the screen."""
    hazard = True

    def __init__(self, x, y, vx, vy, gravity=0, name="crab_ball", r=6):
        super().__init__(x, y)
        self.fx, self.fy, self.vx, self.vy, self.g, self.name, self.r = x << 8, y << 8, vx, vy, gravity, name, r
        self.t = 0

    def update(self, core, p) -> None:
        self.t += 1
        self.fx += self.vx
        self.fy += self.vy
        self.vy += self.g
        self.x, self.y = self.fx >> 8, self.fy >> 8
        if self.y > core.camera_y + 256 or self.t > 400:
            self.alive = False
        elif overlap(core.touch_box(p), self.x - self.r, self.y - self.r, self.r * 2, self.r * 2):
            core.hurt(p, self.x)

    def draw(self, surf, core, art) -> None:
        s = art.sprite(self.name)
        surf.blit(s, s.get_rect(center=(self.x - int(core.camera_x), self.y - int(core.camera_y))))


# ── level furniture ──────────────────────────────────────────────────────────

class Monitor(Obj):
    hw = hh = 16

    @property
    def hazard(self) -> bool:           # not harmful, but it stops a runner: the agents' rays must see it
        return not self.broken

    def __init__(self, x, y, kind):
        super().__init__(x, y)
        self.kind = kind
        self.broken = False
        self.vy = 0
        self.falling = False

    def update(self, core, p) -> None:
        if self.broken:
            return
        if self.falling:
            self.vy += 56
            self.y += self.vy >> 8
            d = core.ground_dist(self.x, self.y + 15)
            if d < 0 and self.vy > 0:
                self.y += d
                self.vy = 0
                self.falling = False
        if p.routine != "control":
            return
        if p.rolling and overlap(core.touch_box(p), self.x - 16, self.y - 16, 32, 32):
            if p.vy >= 0:
                p.vy = -p.vy
                self.smash(core)
                return
            if p.py - 16 >= self.y:                 # bumped from below: the monitor hops
                p.vy = -p.vy
                if not self.falling:
                    self.vy, self.falling = -0x180, True
            return
        if self.falling or p.rolling:
            if p.on_obj is self:
                leave(core, p)
            return
        px, py = p.px, p.py
        if p.on_obj is self:
            if abs(px - self.x) >= 15 + PUSH:
                leave(core, p)
            else:
                p.y = (self.y - 15 - p.h) << 8
            return
        cw, ch = 15 + PUSH, 15 + p.h
        left, top = px - self.x + cw, py - self.y + ch
        if left < 0 or left > 2 * cw or top < 0 or top > 2 * ch:
            return
        if top < 16 and abs(px - self.x) <= 19 and p.vy >= 0:
            land_on(core, p, self, self.y - 15)
            return
        xd = left if px <= self.x else left - 2 * cw
        if xd > 0 and p.vx > 0 or xd < 0 and p.vx < 0:
            p.vx = p.gsp = 0
            if not p.air:
                p.pushing = True
        p.x -= xd << 8

    def smash(self, core) -> None:
        self.broken = True
        core.play("pop")
        core.add_score(100)
        core.extra.append(PowerUp(self.x, self.y - 4, self.kind))
        if core.render_mode != "none":
            core.extra.append(Effect(self.x, self.y, "explode", 5, 6))

    def draw(self, surf, core, art) -> None:
        sx, sy = self.x - int(core.camera_x), self.y - int(core.camera_y)
        if self.broken:
            s = art.sprite("monitor_broken")
            surf.blit(s, s.get_rect(center=(sx, sy + 8)))
            return
        s = art.sprite("monitor")
        surf.blit(s, s.get_rect(center=(sx, sy)))
        i = art.sprite(f"icon_{self.kind}")
        surf.blit(i, i.get_rect(center=(sx, sy - 3)))


class PowerUp:
    """The icon that rises out of a broken monitor; it pays when it stops rising."""

    def __init__(self, x, y, kind):
        self.x, self.y, self.kind, self.vy = x, y << 8, kind, -0x300
        self.alive = True
        self.t = 0

    def update(self, core, p) -> None:
        self.t += 1
        if self.vy < 0:
            self.y += self.vy
            self.vy += 0x18
            if self.vy >= 0:
                self.pay(core, p)
        elif self.t > 60:
            self.alive = False

    def pay(self, core, p) -> None:
        k = self.kind
        if k == "rings":
            core.add_rings(10)
        elif k == "life":
            core.extra_lives += 1
            core.play("life")
        elif k == "shield":
            p.shield = True
            core.play("shield")
        elif k == "invincible":
            p.invincible = 1200
            core.play("invincible")
        elif k == "shoes":
            from .speedy_core import SHOES_ACC, SHOES_DEC, SHOES_TOP
            p.shoes = 1200
            p.acc, p.dec, p.top = SHOES_ACC, SHOES_DEC, SHOES_TOP

    def draw(self, surf, core, art) -> None:
        s = art.sprite(f"icon_{self.kind}")
        surf.blit(s, s.get_rect(center=(self.x - int(core.camera_x), (self.y >> 8) - int(core.camera_y))))


class Spikes(Obj):
    hazard = True

    def __init__(self, x0, x1, y):
        super().__init__((x0 + x1) // 2, y)
        self.hw = (x1 - x0) // 2 + 8

    def update(self, core, p) -> None:
        if self.solid(core, p, self.hw, 16) == "top" and not p.invincible and p.routine == "control":
            p.y -= p.vy                             # the 1991 spike code steps the body back first
            core.hurt(p, self.x, spikes=True)

    def draw(self, surf, core, art) -> None:
        s = art.sprite("spike")
        x = self.x - self.hw
        while x < self.x + self.hw - 8:
            surf.blit(s, (x - int(core.camera_x), self.y - 16 - int(core.camera_y)))
            x += s.get_width()


class SpringUp(Obj):
    def __init__(self, x, y, red):
        super().__init__(x, y)
        self.power, self.name = (0x1000, "spring_r") if red else (0xA00, "spring_y")
        self.t = 0

    def update(self, core, p) -> None:
        if self.t:
            self.t -= 1
            return
        vx, hit = p.vx, self.solid(core, p, 16, 8)
        if hit == "top" or (hit == "side" and not p.air):   # walking into one launches too, and keeps the run
            if hit == "top":
                p.y += 8 << 8
            else:
                p.vx = vx
            p.vy = -self.power
            p.air = True
            p.jumping = False
            p.on_obj = None
            p.spring_anim = True
            self.t = 9
            core.play("spring")

    def draw(self, surf, core, art) -> None:
        s = art.sprite(f"{self.name}_{1 if self.t else 0}")
        surf.blit(s, s.get_rect(center=(self.x - int(core.camera_x), self.y - int(core.camera_y))))


class SpringSide(Obj):
    def __init__(self, x, y, level):
        super().__init__(x, y)
        self.dir = -1 if level.solid(x + 20, y, 0) else 1     # it faces away from the wall it is bolted to
        self.t = 0

    def update(self, core, p) -> None:
        if self.t:
            self.t -= 1
            return
        if self.solid(core, p, 8, 14) == "side" and (p.px - self.x) * self.dir > 0:
            p.x -= self.dir * 8 << 8
            p.vx = p.gsp = 0x1000 * self.dir
            p.facing = self.dir
            p.lock = 15
            p.pushing = False
            self.t = 9
            core.play("spring")

    def draw(self, surf, core, art) -> None:
        s = art.sprite(f"spring_side_r_{1 if self.t else 0}", flip_x=self.dir > 0)
        surf.blit(s, s.get_rect(center=(self.x - int(core.camera_x), self.y - int(core.camera_y))))


class Rock(Obj):
    hazard = True                       # in the way, as a monitor is

    def update(self, core, p) -> None:
        self.solid(core, p, 16, 16)

    def draw(self, surf, core, art) -> None:
        s = art.sprite("rock")
        surf.blit(s, s.get_rect(center=(self.x - int(core.camera_x), self.y - int(core.camera_y))))


def stand(level, x: int, y: int, half: int) -> int:
    """y of something half px tall standing on the first ground below (x, y): the map shows where its
    sprite sat, and not every sprite was centred on the object."""
    for g in range(y, min(y + 96, level.px_h)):
        if level.solid(x, g, 0):
            return g - half
    return y


class Lamp(Obj):
    def __init__(self, x, y, level):
        super().__init__(x, stand(level, x, y, 32))
        self.on = False

    def update(self, core, p) -> None:
        if not self.on and self.x - 18 <= p.px < self.x - 2 and self.y - 64 <= p.py < self.y + 40:
            self.on = True
            core.checkpoint = (self.x, self.y, core.frames)
            core.play("lamp")

    def draw(self, surf, core, art) -> None:
        s = art.sprite("lamp_1" if self.on else "lamp_0")
        surf.blit(s, s.get_rect(center=(self.x - int(core.camera_x), self.y - int(core.camera_y))))


class Sign(Obj):
    def __init__(self, x, y, level):
        super().__init__(x, stand(level, x, y, 32))
        self.t = 0

    def update(self, core, p) -> None:
        if not self.t and 0 <= p.px - self.x:
            self.t = 1
            core.reached_goal = True
            core.locked = True
            core.cam_left = core.cam_right = min(int(core.camera_x), core.level.camera_right)
            core.play("sign")
            if core.render_mode == "none":
                core.won = True
        elif self.t and core.render_mode != "none":
            self.t += 1
            if self.t == 120:
                core.finish_act()

    def draw(self, surf, core, art) -> None:
        if not self.t:
            name = "sign_robot"
        elif self.t < 100:
            name = f"sign_spin_{(self.t >> 2) & 1}"
        else:
            name = "sign_hero"
        s = art.sprite(name)
        surf.blit(s, s.get_rect(center=(self.x - int(core.camera_x), self.y - int(core.camera_y))))


class Bridge(Obj):
    """Only the log under the feet matters to the body; the others follow it for the eye."""

    def __init__(self, x0, x1, y, level):
        left, n = x0 + 24, (x1 - x0 - 47) // 16 + 1
        super().__init__((x0 + x1) // 2, y)
        self.left, self.n = left, n
        col = left - 8
        top = y - 24
        while top < y + 40 and not level.solid(col, top, 0):
            top += 1
        self.top = top
        half = (n + 1) // 2
        rise = [2 * (i + 1) for i in range(half)]
        self.dip_max = rise + rise[::-1][n % 2:]
        self.dip, self.stand = 0, 1

    def log_y(self, i: int) -> int:
        d = abs(i + 1 - self.stand)
        frac = 1 - d / self.stand if i < self.stand else 1 - d / (self.n - self.stand + 1)
        return self.top + int(self.dip_max[self.stand - 1] * math.sin(math.radians(self.dip)) *
                              math.sin(math.radians(frac * 90)))

    def update(self, core, p) -> None:
        px = p.px
        on = p.on_obj is self
        if on:
            if p.air or not self.left - 4 <= px < self.left + self.n * 16 + 4:
                leave(core, p)
                on = False
            else:
                self.stand = min(max((px - self.left) // 16 + 1, 1), self.n)
                self.dip = min(self.dip + 5.625, 90)
                p.y = (self.log_y(self.stand - 1) - p.h) << 8
        elif p.routine == "control" and p.vy >= 0 and self.left - 4 <= px < self.left + self.n * 16 + 4:
            self.stand = min(max((px - self.left) // 16 + 1, 1), self.n)
            d = self.top - (p.py + p.h + 4)
            if -16 <= d <= 0:
                land_on(core, p, self, self.top)
        if not on and self.dip:
            self.dip = max(self.dip - 5.625, 0)

    def draw(self, surf, core, art) -> None:
        s = art.sprite("log")
        cx, cy = int(core.camera_x), int(core.camera_y)
        for i in range(self.n):
            surf.blit(s, (self.left + i * 16 - cx, self.log_y(i) - cy))


class Platform(Obj):
    """Grass-topped floating platform: sags a little when stood on; 'h' ones sweep 128 px sideways.
    shortcut: a still map cannot show which of the original's platforms move, so only the one a route
    needs is marked; add 'h' in the level file when the real subtypes are known."""

    def __init__(self, x, y, moves=False):
        super().__init__(x, y)
        self.moves = moves
        self.sag = 0

    def update(self, core, p) -> None:
        if self.moves:
            ox = self.x
            self.x = self.x0 + int(64 * SIN256[core.frames & 255])
            if p.on_obj is self:
                p.x += (self.x - ox) << 8
        top = self.y0 + self.sag
        if p.on_obj is self:
            if p.air or abs(p.px - self.x) >= 32:
                leave(core, p)
            else:
                self.sag = min(self.sag + 1, 4)
                p.y = (self.y0 + self.sag - p.h) << 8
                return
        elif p.routine == "control" and p.vy >= 0 and abs(p.px - self.x) < 32:
            d = top - (p.py + p.h + 4)
            if -16 <= d <= 0:
                land_on(core, p, self, top)
                return
        if self.sag and p.on_obj is not self:
            self.sag -= 1

    def draw(self, surf, core, art) -> None:
        s = art.sprite("platform")
        surf.blit(s, s.get_rect(midtop=(self.x - int(core.camera_x), self.y0 + self.sag - int(core.camera_y))))


class Ledge(Obj):
    """A grassy lip that gives way under the feet: it holds for a moment, then its ground is gone."""
    DELAY = 24

    def __init__(self, x0, y0, x1, y1):
        super().__init__((x0 + x1) // 2, y0)
        self.box = (x0, y0, x1, y1)
        self.t = 0

    def update(self, core, p) -> None:
        x0, y0, x1, y1 = self.box
        if not self.t:
            if not p.air and x0 <= p.px < x1 and y0 - 24 <= p.py + p.h <= y1:
                self.t = 1
            return
        self.t += 1
        if self.t >= self.DELAY:
            self.alive = False
            core.collapse(self.box)
            core.play("spikes")
            if core.render_mode != "none":
                for i in range(6):
                    core.extra.append(Effect(x0 + 8 + i * 16, y0 + 8 + (i % 3) * 16, "ledge_piece", 1, 40 + i * 2, vy=2.0))

    def draw(self, surf, core, art) -> None:
        pass


class Swing(Obj):
    def __init__(self, x, y_end, anchor):
        super().__init__(x, anchor)
        self.len = y_end - anchor
        self.px_, self.py_ = x, y_end

    def update(self, core, p) -> None:
        a = 0x30 * SIN256[(core.frames * 2) & 255]
        rad = a * math.pi / 128
        ox, oy = self.px_, self.py_
        self.px_ = self.x + int(self.len * math.sin(rad))
        self.py_ = self.y + int(self.len * math.cos(rad))
        top = self.py_ - 8
        if p.on_obj is self:
            if p.air or abs(p.px - self.px_) >= 24:
                leave(core, p)
            else:
                p.x += (self.px_ - ox) << 8
                p.y = (top - p.h) << 8
        elif p.routine == "control" and p.vy >= 0 and abs(p.px - self.px_) < 24:
            d = top - (p.py + p.h + 4)
            if -16 <= d <= 0:
                land_on(core, p, self, top)

    def draw(self, surf, core, art) -> None:
        cx, cy = int(core.camera_x), int(core.camera_y)
        s = art.sprite("swing_anchor")
        surf.blit(s, s.get_rect(center=(self.x - cx, self.y - cy)))
        link = art.sprite("swing_link")
        for i in range(1, self.len // 16):
            f = i * 16 / self.len
            surf.blit(link, link.get_rect(center=(int(self.x + (self.px_ - self.x) * f) - cx,
                                                  int(self.y + (self.py_ - self.y) * f) - cy)))
        s = art.sprite("swing_plat")
        surf.blit(s, s.get_rect(center=(self.px_ - cx, self.py_ - cy)))


# ── badniks ──────────────────────────────────────────────────────────────────

POINTS = (100, 200, 500, 1000)


class Badnik(Obj):
    enemy = True
    hw = hh = 16                # hitbox radii

    def wake(self, core) -> None:
        super().wake(core)
        self.x, self.y = self.x0, self.y0
        self.fx, self.fy = self.x << 8, self.y << 8
        self.vx = self.vy = 0
        self.t = 0
        self.state = 0
        self.dir = -1
        self.setup(core)

    def setup(self, core) -> None:
        pass

    def touch(self, core, p) -> None:
        if p.routine != "control" or not overlap(core.touch_box(p), self.x - self.hw, self.y - self.hh,
                                                 self.hw * 2, self.hh * 2):
            return
        if p.invincible or p.rolling:
            self.destroy(core, p)
        else:
            core.hurt(p, self.x)

    def destroy(self, core, p) -> None:
        self.alive = False
        core.kills_total += 1
        n = core.chain
        core.chain += 1
        pts = 10000 if n >= 15 else POINTS[min(n, 3)]
        core.add_score(pts)
        core.play("pop")
        if p.vy < 0:
            p.vy += 0x100
        elif p.py < self.y:
            p.vy = -p.vy
        else:
            p.vy -= 0x100
        if core.render_mode != "none":
            core.extra.append(Effect(self.x, self.y, "explode", 5, 6))
            core.extra.append(Effect(self.x, self.y, f"points_{min(pts, 1000)}", 1, 40, vy=-0.5))
            core.extra.append(Effect(self.x, self.y, "freed", 2, 8, vy=-1.0))

    def blit(self, surf, core, art, name, flip=None) -> None:
        s = art.sprite(name, flip_x=(self.dir > 0) if flip is None else flip)
        surf.blit(s, s.get_rect(center=(self.x - int(core.camera_x), self.y - int(core.camera_y))))


class Moto(Badnik):
    hw, hh = 20, 16

    def setup(self, core) -> None:
        d = core.ground_dist(self.x, self.y - 14)
        for _ in range(4):
            if d == 32:
                self.y += 24
                d = core.ground_dist(self.x, self.y + 14)
        self.y += d if d != 32 else 0
        self.fy = self.y << 8
        self.state, self.vx = 1, -0x100

    def update(self, core, p) -> None:
        if self.state == 0:                 # stopped at a ledge
            self.t -= 1
            if self.t < 0:
                self.state, self.dir, self.vx = 1, -self.dir, 0x100 * -self.dir
        else:
            self.fx += self.vx
            self.x = self.fx >> 8
            d = core.ground_dist(self.x, self.y + 14)
            if d < -8 or d >= 12:
                self.state, self.t, self.vx = 0, 59, 0
            else:
                self.y += d
        self.touch(core, p)

    def draw(self, surf, core, art) -> None:
        self.blit(surf, core, art, f"moto_{(core.frames >> 3) % 3 if self.state else 0}")


class Crab(Badnik):
    hw, hh = 16, 16

    def setup(self, core) -> None:
        d = core.ground_dist(self.x, self.y + 16)
        self.y += d if d != 32 else 0
        self.state, self.t = 1, 128
        self.vx = -0x80

    def update(self, core, p) -> None:
        self.t -= 1
        if self.state == 1:                 # walking
            self.fx += self.vx
            self.x = self.fx >> 8
            probe = self.x + (16 * self.dir if core.frames & 1 else 0)
            d = core.ground_dist(probe, self.y + 16)
            if self.t <= 0 or d < -8 or d >= 12:
                self.state, self.t = 2, 90
            else:
                self.y += d
        elif self.state == 2:               # stopped, about to fire
            if self.t <= 0:
                core.extra.append(Shot(self.x - 16, self.y, -0x100, -0x400, 56))
                core.extra.append(Shot(self.x + 16, self.y, 0x100, -0x400, 56))
                self.state, self.t = 3, 60
        elif self.t <= 0:                   # after the shots: walk the other way
            self.dir = -self.dir
            self.vx = 0x80 * self.dir
            self.state, self.t = 1, 128
        self.touch(core, p)

    def draw(self, surf, core, art) -> None:
        name = f"crab_{(core.frames >> 3) % 3}" if self.state == 1 else ("crab_fire_0" if self.state == 3 else "crab_0")
        self.blit(surf, core, art, name)


class Buzz(Badnik):
    hw, hh = 24, 12

    def setup(self, core) -> None:
        self.state, self.t, self.vx = 1, 127, -0x400
        self.fired = False

    def update(self, core, p) -> None:
        self.t -= 1
        if self.state == 1:                 # flying
            self.fx += self.vx
            self.x = self.fx >> 8
            if not self.fired and abs(p.px - self.x) < 96 and self.on_screen(core):
                self.fired = True
                self.state, self.t, self.vx = 2, 30, 0
            elif self.t < 0:
                self.state, self.t, self.vx = 4, 59, 0
        elif self.state == 2:               # hovering, then the stinger
            if self.t < 0:
                self.state, self.t = 3, 31
        elif self.state == 3:
            if self.t == 0:
                core.extra.append(Shot(self.x + 24 * self.dir, self.y + 28, 0x200 * self.dir, 0x200, 0, "buzz_shot"))
            if self.t < -29:
                self.state, self.t, self.vx = 1, 127, 0x400 * self.dir
        elif self.t < 0:                    # a turn at the end of its run
            self.dir = -self.dir
            self.fired = False
            self.state, self.t, self.vx = 1, 127, 0x400 * self.dir
        self.touch(core, p)

    def on_screen(self, core) -> bool:
        return core.camera_x - 32 <= self.x <= core.camera_x + 352

    def draw(self, surf, core, art) -> None:
        name = "buzz_fire_0" if self.state == 3 else f"buzz_{(core.frames >> 1) & 1}"
        self.blit(surf, core, art, name)
        if self.state == 1:
            f = art.sprite(f"buzz_flame_{(core.frames >> 1) & 1}", flip_x=self.dir > 0)
            surf.blit(f, f.get_rect(center=(self.x - 24 * self.dir - int(core.camera_x), self.y + 4 - int(core.camera_y))))


class Chopper(Badnik):
    hw, hh = 12, 16

    def setup(self, core) -> None:
        self.vy = -0x700

    def update(self, core, p) -> None:
        self.fy += self.vy
        self.vy += 24
        self.y = self.fy >> 8
        if self.y >= self.y0:
            self.fy, self.y, self.vy = self.y0 << 8, self.y0, -0x700
        self.touch(core, p)

    def draw(self, surf, core, art) -> None:
        self.blit(surf, core, art, f"chopper_{(core.frames >> (3 if abs(self.vy) > 0x300 else 4)) & 1}", flip=False)


class Newtron(Badnik):
    """Hidden on a wall until the body is within 128 px. The blue one drops and runs along the ground;
    the green one fires once and fades."""
    hw, hh = 20, 16

    def __init__(self, x, y, blue):
        super().__init__(x, y)
        self.blue = blue

    def setup(self, core) -> None:
        self.state, self.t = 0, 0

    def update(self, core, p) -> None:
        if self.state == 0:
            d = p.px - self.x
            self.dir = 1 if d >= 0 else -1
            if abs(d) < 128:
                self.state, self.t = 1, 0
            return
        self.t += 1
        if self.state == 1:                 # materialising
            if self.t >= 24:
                self.state = 2 if self.blue else 5
                self.t = 0
        elif self.state == 2:               # dropping
            self.fy += self.vy
            self.vy += 0x38
            self.y = self.fy >> 8
            d = core.ground_dist(self.x, self.y + 16)
            if d < 0:
                self.y += d
                self.vy = 0
                self.vx, self.state = 0x200 * self.dir, 3
        elif self.state == 3:               # running on the floor
            self.fx += self.vx
            self.x = self.fx >> 8
            d = core.ground_dist(self.x, self.y + 16)
            if d < -8 or d >= 12:
                self.state = 4
            else:
                self.y += d
        elif self.state == 4:               # off the edge it keeps flying level
            self.fx += self.vx
            self.x = self.fx >> 8
        else:                               # the green one's shot, then gone
            if self.t == 16:
                core.extra.append(Shot(self.x + 20 * self.dir, self.y - 8, 0x200 * self.dir, 0, 0, "newtron_shot"))
            if self.t >= 48:
                self.alive = False
                return
        if self.t >= 8 or self.state in (2, 3, 4):
            self.touch(core, p)

    def draw(self, surf, core, art) -> None:
        if self.state == 0:
            return
        c = "b" if self.blue else "g"
        if self.state == 1:
            name = f"newtron_{c}_{min(self.t // 8, 2)}"
        elif self.state in (3, 4):
            name = f"newtron_fly_{(core.frames >> 2) & 1}"
        else:
            name = f"newtron_{c}_2"
        self.blit(surf, core, art, name)


# ── the boss and the end of Act 3 ────────────────────────────────────────────

ARENA_X = 0x2960
ARENA_END = 0x2AC0


class Boss:
    """The wrecking-ball machine: 8 hits, a ball on 128 px of chain swinging once every 256 frames."""
    SHIP_Y = 800

    def __init__(self, core):
        self.x, self.y = float(ARENA_X + 320 + 64), float(core.camera_y - 48)
        self.hx = float(ARENA_X + 160)
        self.state, self.t = "enter", 0
        self.hits, self.flash = 8, 0
        self.chain = 0
        self.swing_t = 0
        self.dir = -1
        self.alive = True
        self.ball_x, self.ball_y = self.x, self.y
        core.boss = self

    def update(self, core, p) -> None:
        self.t += 1
        bob = 4 * math.sin(self.t * math.pi / 64) if self.state not in ("enter", "dying", "leaving") else 0
        if self.state == "enter":
            self.x = max(self.x - 2, self.hx)
            self.y = min(self.y + 2, self.SHIP_Y)
            if self.x == self.hx and self.y == self.SHIP_Y:
                self.state, self.t = "lower", 0
        elif self.state == "lower":
            self.chain += 1
            if self.chain >= 128:
                self.state, self.t = "wait", 0
        elif self.state == "wait":
            if self.t >= 24:
                self.state, self.t = "first", 0
        elif self.state == "first":
            self.swing_t += 1
            self.x -= 0.25
            if self.t >= 128:
                self.state, self.t, self.dir = "pause", 0, 1
        elif self.state == "pause":
            self.swing_t += 1
            if self.t >= 64:
                self.state, self.t = "move", 0
        elif self.state == "move":
            self.swing_t += 1
            self.x += self.dir
            if self.t >= 64:
                self.state, self.t, self.dir = "pause", 0, -self.dir
        elif self.state == "dying":
            if self.t % 8 == 0 and core.render_mode != "none":
                core.extra.append(Effect(self.x + (self.t * 7) % 48 - 24, self.y + (self.t * 13) % 48 - 24, "explode", 5, 4))
            if self.t >= 180:
                self.state, self.t = "leaving", 0
                core.locked = False
                core.cam_right = ARENA_END
        else:
            self.x += 4
            self.y -= 0.5
            if self.t > 120:
                self.alive = False
            return
        angle = 0x36 * math.sin(self.swing_t * math.pi / 128)
        rad = angle * math.pi / 128
        self.ball_x = self.x + self.chain * math.sin(rad)
        self.ball_y = self.y + self.chain * math.cos(rad)
        if self.state == "dying" or p.routine != "control":
            return
        box = core.touch_box(p)
        if self.chain and overlap(box, self.ball_x - 20, self.ball_y - 20, 40, 40):
            core.hurt(p, int(self.ball_x))
            box = None
        if self.flash:
            self.flash -= 1
        elif box and overlap(box, self.x - 24, self.y + bob - 24, 48, 48):
            if p.rolling or p.invincible:
                p.vx, p.vy = -p.vx >> 1, -p.vy >> 1
                self.hits -= 1
                self.flash = 32
                core.play("boss_hit")
                if self.hits == 0:
                    self.state, self.t = "dying", 0
                    core.add_score(1000)
                    core.kills_total += 1
            else:
                core.hurt(p, int(self.x))

    def draw(self, surf, core, art) -> None:
        cx, cy = int(core.camera_x), int(core.camera_y)
        if self.chain:
            link = art.sprite("chain_link")
            for i in range(0, 4):
                f = (i + 1) / 5
                surf.blit(link, link.get_rect(center=(int(self.x + (self.ball_x - self.x) * f) - cx,
                                                      int(self.y + (self.ball_y - self.y) * f) - cy)))
            s = art.sprite("boss_ball")
            surf.blit(s, s.get_rect(center=(int(self.ball_x) - cx, int(self.ball_y) - cy)))
        name = "boss_hurt" if self.flash and core.frames & 2 else f"boss_ship_{(core.frames >> 3) & 1}"
        s = art.sprite(name, flip_x=self.dir > 0)
        surf.blit(s, s.get_rect(center=(int(self.x) - cx, int(self.y) - cy)))


class Capsule(Obj):
    def __init__(self, x, y, level):
        super().__init__(x, y)
        while y < level.px_h - 1 and not level.solid(x, y, 0):
            y += 1
        self.y = self.y0 = y - 30
        self.open = False
        self.t = 0

    def update(self, core, p) -> None:
        if self.open:
            self.t += 1
            if self.t == 90 and core.render_mode != "none":
                core.finish_act()
            return
        if self.solid(core, p, 32, 30) == "top":
            self.open = True
            core.reached_goal = True
            core.play("pop")
            if core.render_mode == "none":
                core.won = True

    def draw(self, surf, core, art) -> None:
        s = art.sprite("capsule_open" if self.open else "capsule")
        surf.blit(s, s.get_rect(center=(self.x - int(core.camera_x), self.y - int(core.camera_y))))


# ── building an act's object list ────────────────────────────────────────────

def build_objects(core) -> tuple[list, list, dict]:
    L = core.level
    o = L.objects
    objs: list[Obj] = []
    objs += [Monitor(x, y, k) for x, y, k in o["monitors"]]
    objs += [Spikes(x0, x1, y) for x0, x1, y in o["spikes"]]
    objs += [SpringUp(x, y, False) for x, y in o["spring_up"]]
    objs += [SpringUp(x, y, True) for x, y in o["spring_up_red"]]
    objs += [SpringSide(x, y, L) for x, y in o["spring_side"]]
    objs += [Rock(x, y) for x, y in o["rock"]]
    objs += [Lamp(x, y, L) for x, y in o["lamp"]]
    objs += [Bridge(x0, x1, y, L) for x0, x1, y in L.bridges]
    objs += [Platform(n[0], n[1], n[2:] == ["h"]) for n in o["platforms"]]
    objs += [Ledge(*n) for n in o["ledges"]]
    objs += [Swing(x, y, a) for x, y, a in o["swing"]]
    objs += [Moto(x, y) for x, y in o["moto"]]
    objs += [Crab(x, y) for x, y in o["crab"]]
    objs += [Buzz(x, y) for x, y in o["buzz"]]
    objs += [Chopper(x, y) for x, y in o["chopper"]]
    objs += [Newtron(n[0], n[1], n[2:] == ["b"]) for n in o["newtron"]]
    if L.sign:
        objs.append(Sign(*L.sign, L))
    if L.capsule:
        objs.append(Capsule(*L.capsule, L))
    objs.sort(key=lambda b: b.x0)
    for i, b in enumerate(objs):
        b.index = i
    rings = [(x, y) for x, y in o["rings"]]
    grid: dict = {}
    for i, (x, _) in enumerate(rings):
        grid.setdefault(x >> 6, []).append(i)
    return objs, rings, grid


# ── drawing and animation (never reached headless) ───────────────────────────

WALK_RUN = 0x600


def animate_player(core, p) -> None:
    """Pose and rotation for this frame: the walk and run cycles speed up with ground speed."""
    speed = abs(p.gsp)
    if p.spin is not None:
        p.pose = f"roll_{core.frames % 5}"
    elif p.routine == "dead":
        p.pose = "dead_0"
    elif p.routine == "hurt":
        p.pose = "hurt_0"
    elif p.rolling and (p.h == 14 or p.air):
        hold = max(0, 4 - (speed >> 8))
        p.anim_t += 1
        if p.anim_t > hold:
            p.anim_t = 0
            p.frame += 1
        p.pose = f"roll_{p.frame % 5}"              # the walk cycle shares this counter and runs well past 5
    elif p.air and p.spring_anim:
        p.pose = "spring_0"
        if p.vy > 0:
            p.spring_anim = False
    elif not p.air and p.pushing:
        p.pose = f"push_{(core.frames >> 4) % 4}"
    elif not p.air and speed == 0:
        p.idle += 1
        if core.keys[2] < 0:
            p.pose = "lookup_0"
        elif core.keys[2] > 0:
            p.pose = "duck_0"
        elif p.balance:
            p.pose = f"balance_{(core.frames >> 4) & 1}"
        elif p.idle > 180:
            p.pose = f"wait_{(core.frames >> 4) & 1}"
        else:
            p.pose = "idle_0"
        return
    elif not p.air and core.keys[0] and core.keys[0] * p.gsp < 0 and speed >= 0x400:
        p.pose = f"skid_{(core.frames >> 3) & 1}"
    else:
        hold = max(0, 8 - (speed >> 8))
        p.anim_t += 1
        if p.anim_t > hold:
            p.anim_t = 0
            p.frame += 1
        p.pose = f"run_{p.frame % 4}" if speed >= WALK_RUN else f"walk_{p.frame % 6}"
    p.idle = 0


def draw_world(core, surf) -> None:
    from . import speedy_art as art
    L = core.level
    cx, cy, frame = int(core.camera_x), int(core.camera_y), core.frames
    art.backdrop(surf, cx, cy, frame)
    art.decor_back(surf, L, cx, cy, frame)
    sky = surf.copy() if core.collapsed else None       # what shows through where a ledge fell
    for o in core.awake:
        if o.alive and isinstance(o, (Newtron, Swing)):
            o.draw(surf, core, art)
    for chunk_y in range(cy >> 8, ((cy + 224) >> 8) + 1):
        for chunk_x in range(cx >> 8, ((cx + 320) >> 8) + 1):
            surf.blit(art.terrain_chunk(L, chunk_x, chunk_y), ((chunk_x << 8) - cx, (chunk_y << 8) - cy))
    for x0, y0, x1, y1 in core.collapsed:
        hole = (x0 - cx, y0 - cy, x1 - x0, y1 - y0)
        surf.blit(sky, hole, hole)
    ring = art.sprite(f"ring_{(frame >> 3) & 3}")
    for b in range(cx >> 6, ((cx + 320) >> 6) + 1):
        for i in core.ring_grid.get(b, ()):
            if core.ring_on[i]:
                rx, ry = core.rings[i]
                surf.blit(ring, (rx - 8 - cx, ry - 8 - cy))
    for o in core.awake:
        if o.alive and not isinstance(o, (Newtron, Swing)) and o.awake:
            o.draw(surf, core, art)
    for e in core.extra:
        e.draw(surf, core, art)
    p = core.player
    if not (p.invuln and p.routine != "dead" and frame & 4):
        rot = 0 if p.rolling or p.routine != "control" or not p.pose.startswith(("walk", "run")) \
            else ((p.angle + 0x10) >> 5) & 7
        s = art.hero(p.pose, rot, flip_x=p.facing < 0)
        surf.blit(s, s.get_rect(center=(p.px - cx, p.py - cy + (5 if p.spin is not None else 0))))
    if p.shield:
        s = art.sprite(f"shield_{(frame >> 2) & 1}")
        surf.blit(s, s.get_rect(center=(p.px - cx, p.py - cy)))
    if p.invincible:
        s = art.sprite(f"star_{(frame >> 2) & 3}")
        surf.blit(s, s.get_rect(center=(p.px - cx + int(16 * math.cos(frame * 0.4)),
                                        p.py - cy + int(16 * math.sin(frame * 0.4)))))
    art.hud(surf, core.score, core.frames, core.rings_held, 3 if core.hud_lives is None else core.hud_lives + 1,
            bool(frame & 8))
