"""Enemies of Mega Man (NES, 1987), decoded from the disassembly (bank5.s enemy AI, bank7.s
movement and collision) and the ROM's own tables.

Everything here works in NES pixels and px/frame, centre-anchored, with y growing downward —
the core converts at its edge (world px = NES px x core.scale). Each enemy is the console's
state machine frame for frame, with two knowing departures:

  * the console's random byte is not reproducible without emulating the whole machine, so every
    random choice draws from `core.rng` instead (seeded, so runs stay deterministic);
  * the console tests each enemy for hits on alternate frames; here it is every frame.
"""
from __future__ import annotations

from dataclasses import dataclass, field

GRAVITY = 0.25          # $00.40 per frame
MAX_FALL = 12.0         # $0C.00
REST_SPEED = 0.75       # the downward speed a body is left with after a tile stops it ($FF.40)
SHOT_HALF = 4           # the buster shot's collision half-size
PLAYER_HALF = (7, 11)   # Mega Man's box for enemy contact
CULL_TOP, CULL_BOTTOM = 4, 0xE8     # a body leaving this band of the screen is removed

# glyph: (name, half w, half h, buster damage, contact damage, score); all enemies have 20 HP
STATS = {
    "b": ("Blader", 8, 8, 20, 3, 500),
    "f": ("Flea", 8, 8, 20, 2, 300),
    "v": ("Octopus Battery", 7, 7, 4, 4, 300),
    "h": ("Octopus Battery", 7, 7, 4, 4, 300),
    "w": ("Screw Bomber", 8, 8, 7, 1, 500),
    "W": ("Screw Bomber", 8, 8, 7, 2, 500),
    "e": ("Big Eye", 12, 20, 1, 10, 9000),
    "s": ("Flying Shell", 8, 9, 20, 1, 800),
    "k": ("Beak", 2, 8, 20, 1, 200),
    "j": ("Beak", 2, 8, 20, 1, 200),
    "K": ("Beak", 2, 8, 20, 1, 200),
    "J": ("Beak", 2, 8, 20, 1, 200),
    "c": ("Super Cutter", 2, 2, 0, 0, 0),         # the launcher itself cannot be touched or shot
    "cutter": ("Super Cutter", 8, 8, 4, 4, 1500),
    "X": ("Cut Man", 8, 12, 3, 4, 0),
}
ENEMY_GLYPHS = "bfvhwWeskjKJc"
BOSS_GLYPH = "X"
BOSS_HP = 28
BOSS_INTRO_FRAMES = 303     # room flash, shutter, pose and energy fill before control returns
BOSS_APPEARS_AT = 208       # frame of the intro on which he drops in
BOSS_BLINK = 63             # frames he cannot be hurt again after a hit
BOSS_RECOIL_UNTIL = 31      # ...and he is thrown back while the blink counter is at least this
ROLLING_CUTTER_HALF = 6
ROLLING_CUTTER_SPEED = 2.375                      # $02.60
ITEM_HALF = {"life_large": (4, 8), "life_small": (6, 6), "pearl": (4, 4), "clear": (8, 8)}
ITEM_HEAL = {"life_large": 10, "life_small": 2}
ITEM_TTL = 240              # a dropped item lasts four seconds
PEARL_SCORE = 1000
ITEM_GLYPHS = "+"
BULLET_DAMAGE = 2
BULLET_HALF = 1
ENEMY_HP = 20

D = 2.171875            # $02.2C: the diagonal component of a 3 px/frame bullet
BEAK_SHOTS = [(D, -D), (3.05859375, -0.40625), (3.05859375, 0.40625), (D, D)]   # (toward, down)
SCREW_SHOTS = [(-3.0, 0.0), (-D, -D), (0.0, -3.0), (D, -D), (3.0, 0.0)]        # floor; ceiling mirrors y
S = 2.8125              # $02.D0
SHELL_SHOTS = [(0, -4.0), (0, 4.0), (-4.0, 0), (4.0, 0), (-S, -S), (S, -S), (-S, S), (S, S)]


@dataclass
class Enemy:
    kind: str
    x: float
    y: float
    hw: int
    hh: int
    facing: int = -1            # +1 right, -1 left
    vx: float = 0.0             # speed along `facing`, never negative
    vy: float = 0.0             # +down
    hp: int = ENEMY_HP
    tiles: bool = False         # stopped by solid tiles
    gravity: bool = False
    hittable: bool = True       # buster shots register
    hurts: bool = True          # touching it damages the player
    shielded: bool = False      # shots register but bounce off
    visible: bool = True
    fd: int = 0                 # the console's per-object "fire delay" counter
    t: int = 0                  # frames in the current cycle
    state: str = ""
    spawn: int = -1             # index of the level spawn point that owns it
    last_hit: int = -99
    look: int = -1              # boss: which way he faces (he moves along `facing`, a latched direction)
    timer: int = 0
    anim: int = 0
    blink: int = 0
    hit_x: bool = False
    hit_y: bool = False
    home: tuple = (0.0, 0.0)

    @property
    def name(self) -> str:
        return STATS[self.kind][0]


@dataclass
class Bullet:
    x: float
    y: float
    vx: float
    vy: float
    src: str = ""               # who fired it (the cause of death it reports)
    facing: int = -1            # the shooter's facing: which way a hit throws Mega Man


@dataclass
class RollingCutter:
    x: float
    y: float
    vx: float
    vy: float
    speed: float = ROLLING_CUTTER_SPEED
    returning: bool = False


@dataclass
class Item:
    kind: str                   # "life_large" | "life_small" | "pearl"
    x: float
    y: float
    vy: float = 0.0
    ttl: int = -1               # frames left; -1 = placed in the level, stays
    spawn: int = -1


def spawn_position(glyph: str, col: int, row: int) -> tuple[float, float]:
    """Where the ROM puts an object whose glyph sits in this 16 px cell."""
    x, y = col * 16 + 8.0, row * 16 + 8.0
    if glyph == "w":
        y = (row + 1) * 16.0            # half sunk into the floor below
    elif glyph == "W":
        y = row * 16.0                  # half sunk into the ceiling above
    elif glyph == "e":
        y = row * 16 + 12.0             # 40 px tall, feet on the floor a cell and a half down
    elif glyph == "s":
        y = row * 16.0
    elif glyph == "c":
        x, y = col * 16.0, row * 16 + 4.0
    elif glyph == BOSS_GLYPH:
        y = row * 16 + 4.0              # 24 px tall, feet on the floor below this cell
    return x, y


def make(glyph: str, col: int, row: int, spawn: int) -> Enemy:
    _, hw, hh, _, _, _ = STATS[glyph]
    x, y = spawn_position(glyph, col, row)
    e = Enemy(glyph, x, y, hw, hh, spawn=spawn, home=(x, y))
    if glyph == "b":
        e.vx = 1.125
    elif glyph == "f":
        e.tiles = e.gravity = True
        e.fd = 1
    elif glyph == "v":
        e.tiles, e.vy = True, -2.0
    elif glyph == "h":
        e.tiles, e.vx = True, 2.0
    elif glyph == "e":
        e.tiles = e.gravity = True
        e.vy, e.fd = 0.25, 15
    elif glyph == "s":
        e.vx, e.shielded = 1.125, True
    elif glyph in "kjKJ":
        e.facing = 1 if glyph in "KJ" else -1      # mounted on the wall behind it
        e.t = 38 if glyph in "jJ" else 0           # the second variant is 38 frames into its cycle
        e.shielded = True
    elif glyph == "c":
        e.hittable = e.hurts = e.visible = False
    elif glyph == BOSS_GLYPH:
        e.tiles = e.gravity = True
        e.hp, e.vy, e.state = BOSS_HP, 0.25, 5
    return e


# ── shared movement (EnemyAI_MovementsAndDamageCheck) ────────────────────────
def move(e: Enemy, core) -> bool:
    """One frame of motion. Returns False when the body left the screen and must be removed."""
    cam_x, cam_y = core.nes_view()
    e.hit_x = e.hit_y = False
    nx = e.x + e.facing * e.vx
    if e.tiles and e.vx:
        edge = nx + e.facing * e.hw
        if core.nes_solid(edge, e.y - e.hh) or core.nes_solid(edge, e.y + e.hh - 1):
            col = int(edge // 16)
            nx = col * 16 - e.hw if e.facing > 0 else (col + 1) * 16 + e.hw
            e.hit_x = True
    e.x = nx
    if not cam_x <= e.x <= cam_x + 255:
        return False
    ny = e.y + e.vy
    if e.tiles or e.gravity:
        if e.tiles:
            probe = ny - e.hh if e.vy <= 0 else ny + e.hh
            if core.nes_solid(e.x - (e.hw - 1), probe) or core.nes_solid(e.x + (e.hw - 1), probe):
                row = int(probe // 16)
                ny = (row + 1) * 16 + e.hh if e.vy <= 0 else row * 16 - e.hh
                e.vy = REST_SPEED
                e.hit_y = True
        if e.gravity and not e.hit_y:
            e.vy = min(e.vy + GRAVITY, MAX_FALL)
    e.y = ny
    return CULL_TOP <= e.y - cam_y < CULL_BOTTOM


def _dx(e: Enemy, core) -> float:
    return core.nes_player()[0] - e.x


def _on_screen(e: Enemy, core) -> bool:
    """Enemies that never move are removed too once the screen leaves them, as every object is
    (EnemyAI_MovementsAndDamageCheck), so their spawn points re-arm."""
    cam_x = core.nes_view()[0]
    return cam_x <= e.x <= cam_x + 255


def _face(e: Enemy, core) -> float:
    """Turn toward Mega Man; returns the horizontal distance."""
    dx = _dx(e, core)
    e.facing = 1 if dx >= 0 else -1
    return abs(dx)


# ── one routine per enemy; each returns False when the enemy is gone ─────────
def blader(e: Enemy, core) -> bool:
    if e.fd == 0:
        if _face(e, core) >= 40:
            e.vx = 1.125
            return move(e, core)
        dy = core.nes_player()[1] - e.y            # close enough: swoop at where Mega Man is
        speed = int(abs(dy) * 256 / 24) / 256
        e.vy = speed if dy >= 0 else -speed
        e.vx, e.fd = 0.0, 51
    n = 52 - e.fd                                  # frame of the 51-frame swoop
    if n <= 25:
        e.vx += 0.09375
    elif n <= 50:
        if n == 26:
            e.vy = -e.vy                           # pull out
        e.vx += 0.015625
    else:
        e.vy, e.vx = 0.0, 1.125
    e.fd -= 1
    return move(e, core)


def _hopper(e: Enemy, core, jumps) -> bool:
    """Flea and Big Eye: rest, face Mega Man, hop. In the air they update 3 frames in 4."""
    if e.state == "air":
        if core.frame & 3 == 0:
            return True
        falling = e.vy > 0
        alive = move(e, core)
        if falling and e.hit_y:
            e.vx, e.fd, e.state = 0.0, 15, ""
            if e.kind == "e":
                core.sfx("big_eye")
        return alive
    e.fd -= 1
    if e.fd == 0:
        _face(e, core)
        up, along = jumps[core.rng.getrandbits(1)]
        e.vy, e.vx, e.state = -up, along, "air"
    return move(e, core)


def flea(e: Enemy, core) -> bool:
    return _hopper(e, core, ((4.875, 1.1875), (4.0, 3.0)))


def big_eye(e: Enemy, core) -> bool:
    return _hopper(e, core, ((3.31640625, 1.203125), (4.8984375, 0.81640625)))


def octopus(e: Enemy, core) -> bool:
    """Slides until a wall stops it, waits 63 or 126 frames against it, then goes back."""
    if e.fd > 0:
        e.fd -= 1
        if e.fd == 0:
            e.facing, e.vy = -e.facing, -e.vy
        return True
    vy = e.vy
    alive = move(e, core)
    e.vy = vy                                      # never picks up the resting fall speed
    if e.hit_x or e.hit_y:
        e.fd = (63, 126)[core.rng.getrandbits(1)]
    return alive


def beak(e: Enemy, core) -> bool:
    """206-frame cycle: shut for 82 (shots bounce off), opens, fires four shots 32 frames apart."""
    e.shielded = e.t < 82
    if e.t in (96, 128, 160, 192):
        along, down = BEAK_SHOTS[(e.t - 96) // 32]
        core.enemy_bullets.append(Bullet(e.x + 7 * e.facing, e.y, along * e.facing, down, e.name, e.facing))
        core.sfx("enemy_shoot")
    e.t = (e.t + 1) % 206
    return _on_screen(e, core)


def screw_bomber(e: Enemy, core) -> bool:
    out = -8 if e.kind == "w" else 8               # which way it rises out of its surface
    if e.state != "up":
        if e.fd > 0:
            e.fd -= 1
        elif abs(_dx(e, core)) < 80:
            e.state, e.t, e.y = "up", 0, e.home[1] + out
        return _on_screen(e, core)
    e.t += 1
    if e.t in (32, 64):
        sign = 1 if e.kind == "w" else -1
        for vx, vy in SCREW_SHOTS:
            core.enemy_bullets.append(Bullet(e.x, e.y + out / 2, vx, vy * sign, e.name, e.facing))
        core.sfx("enemy_shoot")
    elif e.t == 96:
        e.state, e.y = "", e.home[1]
        e.fd = (0x20, 0x3F, 0x7D)[core.rng.randrange(3)]
    return _on_screen(e, core)


def flying_shell(e: Enemy, core) -> bool:
    """Crosses the screen right to left, stopping every 88 frames to open and fire eight ways.
    One per spawn point, forever: shot down or off the edge, it comes back at the right."""
    cam_x, cam_y = core.nes_view()
    if e.state == "dead":
        e.fd -= 1
        if e.fd > 0:
            return True
        e.state, e.visible, e.hittable, e.hurts, e.hp = "", True, True, True, ENEMY_HP
        e.x, e.y, e.t = cam_x + 0xF8, cam_y + 0x80, 0
    e.t += 1
    e.shielded = e.t < 56
    if e.t < 56:
        e.x -= 1.125
        if e.x < cam_x:
            e.x, e.y, e.t = cam_x + 0xF8, cam_y + 0x80, 0
    elif e.t == 72:
        for vx, vy in SHELL_SHOTS:
            core.enemy_bullets.append(Bullet(e.x, e.y, vx, vy, e.name, e.facing))
        core.sfx("enemy_shoot")
    elif e.t == 88:
        e.t = 0
    return True


def cutter_launcher(e: Enemy, core) -> bool:
    """Throws a cutter every 32 frames while Mega Man is within 80 px sideways."""
    if e.fd > 0:
        e.fd -= 1
    if e.fd == 0:
        dist = _face(e, core)
        if dist < 80:
            _, hw, hh, _, _, _ = STATS["cutter"]
            core.enemies.append(Enemy("cutter", e.x, e.y, hw, hh, facing=e.facing, gravity=True,
                                      vx=int(dist * 256 / 51) / 256, vy=-4.8984375))
            e.fd = 32
    return _on_screen(e, core)


def cutter(e: Enemy, core) -> bool:
    return move(e, core)


ROUTINES = {"b": blader, "f": flea, "v": octopus, "h": octopus, "w": screw_bomber, "W": screw_bomber,
            "e": big_eye, "s": flying_shell, "k": beak, "j": beak, "K": beak, "J": beak,
            "c": cutter_launcher, "cutter": cutter, "X": lambda e, core: cut_man(e, core)}


def kill(e: Enemy, core) -> bool:
    """An enemy ran out of HP. Returns True if it should be removed from the list."""
    core.score += STATS[e.kind][5]
    core.kills_total += 1
    r = core.rng.randrange(128) % 100              # the console's seed is 7 bits wide, hence the skew
    drop = "pearl" if 12 <= r < 65 else "life_small" if 80 <= r < 95 else "life_large" if 97 <= r < 99 else None
    if drop:                                       # weapon energy and extra lives have no use here
        core.items.append(Item(drop, e.x, e.y, vy=-4.0, ttl=ITEM_TTL))
    if e.kind == "s":                              # the shell's slot re-arms after 72 frames
        e.state, e.fd = "dead", 72
        e.visible = e.hittable = e.hurts = False
        return False
    return True


def shot_hits(e: Enemy, sx: float, sy: float) -> bool:
    return abs(e.x - sx) <= e.hw + SHOT_HALF and -(e.hh + SHOT_HALF) <= e.y - sy <= e.hh + SHOT_HALF - 1


def touches_player(x: float, y: float, hw: int, hh: int, px: float, py: float) -> bool:
    pw, ph = PLAYER_HALF
    return abs(x - px) <= hw + pw and -(hh + ph) <= y - py <= hh + ph - 1


# ── Cut Man ──────────────────────────────────────────────────────────────────
def aim(sx: float, sy: float, tx: float, ty: float, speed: float) -> tuple[float, float]:
    """The console's aiming: full speed on the longer axis, about 2 * minor / major on the other."""
    dx, dy = tx - sx, ty - sy
    ax, ay = abs(dx), abs(dy)
    half = int(max(ax, ay)) // 2
    minor = min(ax, ay) / half if half else 0.0
    sx_, sy_ = (1 if dx >= 0 else -1), (1 if dy >= 0 else -1)
    return (sx_ * speed, sy_ * minor) if ax >= ay else (sx_ * minor, sy_ * speed)


def cut_man(e: Enemy, core) -> bool:
    """The boss's strategy table (states 5-24, numbered as in the ROM). He closes in on foot,
    then jumps short of, onto, or past Mega Man, throwing the Rolling Cutter from the ground or
    mid-jump; he has one cutter and waits for it to come back."""
    px, _ = core.nes_player()
    same_y = abs(core.nes_player_feet() - (e.y + e.hh)) < 1.0
    airborne = core.player_airborne()
    cutter_out = core.boss_cutter is not None

    def dist() -> float:
        e.look = 1 if px >= e.x else -1
        return abs(px - e.x)

    def r3() -> bool:
        return core.rng.randrange(3) == 1

    def step() -> bool:
        """Move one frame; True when this frame ended on the ground."""
        falling = e.vy > 0
        move(e, core)
        return falling and e.hit_y

    def throw() -> None:
        if core.boss_cutter is None:
            dist()
            x, y = e.x + 8 * e.look, e.y - 10
            core.boss_cutter = RollingCutter(x, y, *aim(x, y, *core.nes_player(), ROLLING_CUTTER_SPEED))
            core.sfx("rolling_cutter")

    def wait() -> None:                              # state 24: Mega Man is on another level
        e.vx, e.timer, e.state = 0.0, 63, 24

    def decide() -> None:                            # state 5
        if dist() >= 0x40:
            e.state = 6
        elif same_y or airborne:
            e.state = 7
        else:
            wait()

    if e.blink:
        e.blink -= 1
        if e.blink >= BOSS_RECOIL_UNTIL:             # thrown back by a hit
            step()
            return True
    s = e.state
    if s == 5:
        decide()
    elif s == 6:
        dist()
        e.facing, e.vx = e.look, 1.125
        step()
        if e.hit_x:
            e.state = 7
        else:
            decide()
    elif s == 7:                                     # take off: short of, onto, or past Mega Man
        d = dist()
        e.facing, e.vy = e.look, -6.25
        r = core.rng.randrange(6)
        pick = 2 if r & 1 else 0 if r == 2 else 1
        target = int(d + (-0x20, 0, 0x20)[pick]) & 0xFF
        if target >= 0x80:
            target = 0x100 - target
        e.vx = int(target * 256 / 50) / 256          # the jump lasts 50 frames
        e.state = (8, 9, 12)[pick]
    elif s == 8:
        step()
        if r3() or cutter_out:
            e.state = 14
        else:
            e.state, e.timer = 16, 8
    elif s == 9:
        step()
        e.state = 10
        if not r3() and not cutter_out:
            e.state, e.timer = 18, 18
    elif s == 12:
        step()
        if r3() or cutter_out:
            e.state = 13
        else:
            e.state, e.timer = 22, 21
    elif s == 10:
        if step():
            if dist() >= 0x0E:
                e.state = 5
            else:                                    # landed on top of Mega Man: hop again
                e.vy, e.vx, e.state = -4.875, 0.75, 11
                if r3() and not cutter_out:
                    e.state, e.timer = 20, 18
    elif s == 11:
        if step():
            e.state = 5
    elif s == 13:
        if step():
            e.state = 5 if core.rng.getrandbits(1) else 14
    elif s == 14:
        if step():
            if cutter_out:
                e.state = 5
            elif not same_y and not airborne:
                wait()
            else:
                e.vx, e.anim, e.state = 0.0, 0, 15
    elif s == 15:                                    # standing throw: 16 frames either side
        step()
        e.anim += 1
        if e.anim == 16:
            throw()
        elif e.anim >= 32:
            e.state = 5
    elif s in (16, 18, 20, 22):                      # in the air, counting down to a throw
        step()
        e.timer -= 1
        if e.timer <= 0:
            e.anim, e.state = 0, s + 1
    elif s in (17, 19, 21, 23):                      # throwing in mid-air
        e.anim += 1
        if e.anim == 16:
            throw()
        if step():
            if not same_y and not airborne:
                wait()
            else:
                e.state = {17: 14, 19: 10, 21: 5, 23: 5}[s]
    elif s == 24:
        step()
        e.timer -= 1
        if (e.timer <= 0 or same_y) and not cutter_out:
            e.vx, e.anim, e.state = 0.0, 0, 15
    return True


def hit_boss(e: Enemy, core) -> None:
    """A buster shot landed: he is knocked back, away from Mega Man, and blinks."""
    e.hp -= STATS[e.kind][3]
    e.blink = BOSS_BLINK
    e.facing = -1 if core.nes_player()[0] >= e.x else 1
    e.vy, e.vx, e.state = -1.25, 1.125, 14


def rolling_cutter(c: RollingCutter, boss: Enemy, core) -> bool:
    """Flies straight until it nears the edge of the screen, then homes back on Cut Man."""
    cam_x, cam_y = core.nes_view()
    c.x += c.vx
    c.y += c.vy
    rx, ry = c.x - cam_x, c.y - cam_y
    if not c.returning:
        if rx < 0x18 or rx >= 0xE8 or ry < 0x18 or ry >= 0xD8:
            c.vx, c.vy, c.returning, c.speed = -c.vx, -c.vy, True, ROLLING_CUTTER_SPEED
    else:
        if core.frame & 3 == 0:
            c.speed += 0.015625
            c.vx, c.vy = aim(c.x, c.y, boss.x, boss.y, c.speed)
        if abs(c.x - boss.x) < 8 and abs(c.y - boss.y) < 8:
            return False                             # caught
    return CULL_TOP <= ry < CULL_BOTTOM and 0 <= rx <= 255
