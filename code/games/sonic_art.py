"""Sonic's look: a Mega Drive style Green Hill, every picture our own drawing, made here in code.

No sprite sheet, ripped art or ROM data went into this file. Shapes are drawn at one pixel per game pixel
with no anti-aliasing, turned into rows of palette letters (the same text form as mario_art.py) and kept
in SPRITES. The palette keeps to the console's 3 bits per channel. PNGs in assets/art/sonic/ (art_png.py)
are drawn in place of the text when they are there: after changing a drawing here, delete the PNG.
Heroes and enemies are centred on the object's position; the hero's feet sit at centre + 19.
"""
from __future__ import annotations

import os
from functools import lru_cache
from math import cos, pi, sin

import numpy as np
import pygame
import yaml

from . import art_png, sonic_level
from .modules.System.nes_font import nes_text

PAL = {
    "K": (0x00, 0x00, 0x24),    # outline
    # the hero's blues, dark to light
    "a": (0x00, 0x00, 0x6D), "b": (0x00, 0x24, 0xB6), "c": (0x24, 0x49, 0xDB), "d": (0x49, 0x6D, 0xFF),
    "e": (0x92, 0xB6, 0xFF),
    # skin
    "f": (0xFF, 0xDB, 0xB6), "g": (0xFF, 0xB6, 0x92), "h": (0xDB, 0x92, 0x6D), "i": (0x92, 0x49, 0x24),
    # whites and greys
    "W": (0xFF, 0xFF, 0xFF), "w": (0xDB, 0xDB, 0xDB), "x": (0xB6, 0xB6, 0xB6), "y": (0x92, 0x92, 0x92),
    "z": (0x6D, 0x6D, 0x6D), "Z": (0x49, 0x49, 0x49), "X": (0x24, 0x24, 0x24),
    # reds
    "S": (0xFF, 0x6D, 0x49), "R": (0xFF, 0x00, 0x00), "r": (0xDB, 0x00, 0x00), "q": (0x92, 0x00, 0x00),
    "Q": (0x49, 0x00, 0x00),
    # oranges and browns of the dirt
    "O": (0xFF, 0x92, 0x00), "o": (0xDB, 0x6D, 0x00), "n": (0x92, 0x49, 0x00), "N": (0x6D, 0x24, 0x00),
    "m": (0x24, 0x00, 0x00),
    # gold
    "T": (0xFF, 0xFF, 0xB6), "U": (0xFF, 0xFF, 0x49), "Y": (0xFF, 0xDB, 0x00), "t": (0xDB, 0x92, 0x00),
    "u": (0x92, 0x6D, 0x00),
    # greens of the grass
    "E": (0xB6, 0xFF, 0x49), "F": (0x49, 0xDB, 0x24), "G": (0x00, 0xB6, 0x00), "H": (0x00, 0x6D, 0x00),
    "I": (0x00, 0x49, 0x00),
    # purples of the rock and the flowers
    "P": (0xDB, 0x92, 0xFF), "p": (0xB6, 0x6D, 0xDB), "l": (0x92, 0x49, 0xB6), "L": (0x6D, 0x24, 0x92),
    "M": (0x49, 0x00, 0x6D),
    # water and shield
    "C": (0xB6, 0xFF, 0xFF), "D": (0x49, 0xDB, 0xFF), "B": (0x00, 0x92, 0xFF), "A": (0x00, 0x6D, 0xB6),
    "v": (0x00, 0x49, 0x92), "V": (0x6D, 0xB6, 0xFF),
    # wood and pink
    "k": (0xDB, 0xB6, 0x6D), "J": (0xB6, 0x92, 0x49), "j": (0x6D, 0x49, 0x24), "s": (0xFF, 0x92, 0xB6),
}


class _Art:
    """One sprite being drawn: a grid of palette letters. Shapes are masks, painted back to front."""

    def __init__(self, w: int, h: int):
        self.g = np.full((h, w), ord("."), np.uint8)
        self.y, self.x = np.mgrid[0:h, 0:w] + 0.5

    def ell(self, cx, cy, rx, ry, rot=0.0):
        c, s = cos(rot), sin(rot)
        dx, dy = self.x - cx, self.y - cy
        u, v = (dx * c + dy * s) / rx, (dy * c - dx * s) / ry
        return u * u + v * v <= 1

    def poly(self, pts):
        out = np.zeros(self.g.shape, bool)
        for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]):
            if y0 != y1:
                out ^= ((y0 <= self.y) != (y1 <= self.y)) & (self.x < x0 + (self.y - y0) * (x1 - x0) / (y1 - y0))
        return out

    def line(self, pts, w):
        out = np.zeros(self.g.shape, bool)
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            dx, dy = x1 - x0, y1 - y0
            t = np.clip(((self.x - x0) * dx + (self.y - y0) * dy) / max(dx * dx + dy * dy, 1e-6), 0, 1)
            out |= (self.x - x0 - t * dx) ** 2 + (self.y - y0 - t * dy) ** 2 <= w * w / 4
        return out

    def paint(self, mask, ink, line="K"):
        """ink: one letter, or an array of letter codes from _ramp. line: the outline's letter, or None."""
        if line:
            ring = mask.copy()
            ring[1:] |= mask[:-1]
            ring[:-1] |= mask[1:]
            ring[:, 1:] |= mask[:, :-1]
            ring[:, :-1] |= mask[:, 1:]
            self.g[ring & ~mask] = ord(line)
        self.g[mask] = ord(ink) if isinstance(ink, str) else ink[mask]

    def rows(self):
        return ["".join(map(chr, r)) for r in self.g]


def _ramp(v, letters, cuts):
    """A smooth value turned into flat bands of colour: letters[0] below cuts[0], and so on."""
    return np.frombuffer(letters.encode(), np.uint8)[np.searchsorted(cuts, v)]


def _ball(a, cx, cy, rx, ry, light=(0.35, -0.6, 0.72)):
    """How much a ball at (cx, cy) faces the light, pixel by pixel."""
    nx, ny = (a.x - cx) / rx, (a.y - cy) / ry
    nz = np.sqrt(np.clip(1 - nx * nx - ny * ny, 0, 1))
    return nx * light[0] + ny * light[1] + nz * light[2]


def _bez(p0, p1, p2, n=7):
    return [((1 - t) ** 2 * p0[0] + 2 * t * (1 - t) * p1[0] + t * t * p2[0],
             (1 - t) ** 2 * p0[1] + 2 * t * (1 - t) * p1[1] + t * t * p2[1]) for t in [i / n for i in range(n + 1)]]


BLUES = "abcde"
BLUE_CUTS = (-0.25, 0.25, 0.8, 1.05)


def _head(a, cx, cy, tilt=0.0, look="right", mouth="smile"):
    """The big round head: three swept-back spines, ear, joined eyes, muzzle, nose. tilt in radians,
    positive leans the face down and forward."""
    c, s = cos(tilt), sin(tilt)

    def P(x, y):
        return cx + x * c - y * s, cy + x * s + y * c

    def Q(*pts):
        return [P(*p) for p in pts]

    tips = [(-15.5, -7.5), (-17, 1), (-13.5, 9.5)]
    outline = (_bez(P(2, -9.8), P(-6.5, -11.5), P(*tips[0]))
               + _bez(P(*tips[0]), P(-11, -5), P(-8.5, -3))
               + _bez(P(-8.5, -3), P(-12, -2), P(*tips[1]))
               + _bez(P(*tips[1]), P(-11.5, 2.5), P(-8.5, 3.5))
               + _bez(P(-8.5, 3.5), P(-10.5, 6), P(*tips[2]))
               + _bez(P(*tips[2]), P(-7.5, 10), P(-4, 7.5)))
    spines = a.poly(outline)
    skull = a.ell(cx, cy, 10.2, 9.8, tilt)
    light = (-0.1 + 0.3 * c, -0.72, 0.65)
    v = _ball(a, *P(-3, -0.5), 15.5, 13.5, light)
    a.paint(skull | spines, _ramp(v, BLUES, BLUE_CUTS))
    # the crease under each spine
    for p in (((-14, -6.5), (-11, -4.4), (-8, -2.8), (-5.5, -2.4)), ((-14.5, 1.2), (-11, 2.2), (-8, 3.4), (-5.5, 3.5))):
        a.paint(a.line(Q(*p), 1.1) & (skull | spines), "a", None)
    # ear
    a.paint(a.poly(Q((-1.5, -8), (1.5, -12), (4.5, -13.5), (5, -8.5))),
            _ramp(_ball(a, *P(2.5, -9.5), 6, 7, light), BLUES, BLUE_CUTS))
    a.paint(a.poly(Q((1.5, -8.8), (3.5, -11.5), (3.8, -8.8))), "g", None)
    # muzzle and cheek
    muzzle = a.ell(*P(7.4, 4.0), 6.0, 3.5, tilt) | a.ell(*P(3.2, 5.0), 3.8, 3.0, tilt)
    a.paint(muzzle, _ramp(_ball(a, *P(6.5, 2.5), 9, 6, (0.2, -0.7, 0.7)), "hgf", (0.0, 0.75)), "a")
    # eyes: near and far, joined, with the blue brow dipping between them
    near = a.ell(*P(3.5, -2.6), 3.9, 4.8, tilt)
    far = a.ell(*P(8.4, -2.4), 2.7, 4.4, tilt)
    brow = a.poly(Q((4.4, -9), (6.0, -4.2), (7.2, -9))) | a.poly(Q((-2, -6), (4.5, -8), (-2, -9)))
    a.paint((near | far) & ~brow, "W", "a")
    if look == "shut":
        for x0, x1, w in ((1, 6, near), (7, 11, far)):
            a.paint(a.line(Q((x0, -5), (x1, -1)), 1.2) & w, "K", None)
            a.paint(a.line(Q((x0, -1), (x1, -5)), 1.2) & w, "K", None)
    else:
        px = {"right": (5.4, 9.7), "front": (3.6, 8.2), "back": (1.6, 6.8), "up": (5, 9.2), "down": (5, 9.2)}[look]
        py = {"up": -4.8, "down": -1.0}.get(look, -2.6)
        a.paint(a.ell(*P(px[0], py), 1.7, 2.8, tilt) & near, "K", None)
        a.paint(a.ell(*P(px[1], py), 1.2, 2.6, tilt) & far, "K", None)
    # nose and mouth
    a.paint(a.ell(*P(13.2, 1.0), 1.7, 1.5, tilt), "K", None)
    if mouth == "smile":
        a.paint(a.line(_bez(P(3.5, 4.5), P(7, 7.4), P(11.5, 5.6)), 1), "i", None)
    elif mouth == "open":
        a.paint(a.ell(*P(8.5, 5.5), 3, 2.0, tilt), "Q", "K")


def _shoe(a, cx, cy, ang=0.0):
    """A red shoe with a white strap. (cx, cy) is the ankle; ang turns the toe, positive is down."""
    c, s = cos(ang), sin(ang)

    def P(x, y):
        return cx + x * c - y * s, cy + x * s + y * c

    body = a.poly([P(*p) for p in (
        (-3, -1.5), (-1, -2.5), (1.5, -2.2), (4.5, -1.6), (7, -0.3), (8, 1.3), (7.2, 2.9), (-3, 2.9), (-3.8, 0.8))])
    u = (a.x - cx) * c + (a.y - cy) * s
    v = (a.y - cy) * c - (a.x - cx) * s
    ink = _ramp(v - 0.25 * u, "SRr", (-2.4, 0.5)).copy()
    a.paint(body, ink)
    a.paint(body & (u > -0.2) & (u < 2.0), "W", None)
    a.paint(body & (u > 1.3) & (u < 2.0), "x", None)
    a.paint(body & (v > 1.9), "q", None)


def _limb(a, pts, width, ink):
    a.paint(a.line(pts, width), ink)


def _arm(a, pts, hand=None):
    _limb(a, pts, 1.9, "g")
    hx, hy = hand or pts[-1]
    glove = a.ell(hx, hy, 2.7, 2.5)
    a.paint(glove, _ramp(_ball(a, hx, hy, 3.0, 2.8, (0.3, -0.7, 0.65)), "xwW", (0.1, 0.5)))


def _leg(a, pts, ang=0.0):
    x, y = pts[-1]
    _limb(a, pts, 2.6, "c")
    _shoe(a, x, y, ang)


def _body(a, cx, cy, rx=5.6, ry=6.6, tilt=0.0, belly=(1.6, 0.8)):
    v = _ball(a, cx - 1, cy, rx + 3, ry + 3, (0.1, -0.75, 0.65))
    a.paint(a.ell(cx, cy, rx, ry, tilt), _ramp(v, BLUES, BLUE_CUTS))
    c, s = cos(tilt), sin(tilt)
    bx, by = cx + belly[0] * c - belly[1] * s, cy + belly[0] * s + belly[1] * c
    a.paint(a.ell(bx, by, rx * 0.72, ry * 0.78, tilt) & a.ell(cx, cy, rx, ry, tilt),
            _ramp(_ball(a, bx, by, rx, ry, (0.3, -0.7, 0.65)), "hgf", (0.0, 0.75)), None)


def _tail(a, x, y, ang=2.6):
    a.paint(a.poly([(x, y - 2), (x + 6 * cos(ang), y + 6 * sin(ang)), (x + 1, y + 2)]), "c")


HERO_W, HERO_H = 56, 40
AXIS = HERO_W // 2


def _crop(a, min_half=12):
    """Trim the blank columns evenly about the axis, so frames of different widths stay centred."""
    cols = np.nonzero((a.g != ord(".")).any(axis=0))[0]
    half = max(min_half, AXIS - int(cols[0]), int(cols[-1]) + 1 - AXIS)
    return [r[AXIS - half:AXIS + half] for r in a.rows()]


def _hero(head, body, arm_back, arm_front, leg_back, leg_front, tail=None, look="right", mouth="smile", under=None):
    """Back to front: far arm, far leg, tail, body, near leg, head, near arm. Coordinates have x from the
    axis, y from the top of the 40 px frame."""
    a = _Art(HERO_W, HERO_H)

    def X(pts):
        return [(x + AXIS, y) for x, y in pts]

    if arm_back:
        _arm(a, X(arm_back))
    if leg_back:
        _leg(a, X(leg_back[0]), leg_back[1])
    if under:
        under(a)
    if tail:
        _tail(a, tail[0] + AXIS, tail[1])
    _body(a, body[0] + AXIS, *body[1:])
    if leg_front:
        _leg(a, X(leg_front[0]), leg_front[1])
    _head(a, head[0] + AXIS + 1.5, head[1], head[2] if len(head) > 2 else 0.0, look, mouth)
    if arm_front:
        _arm(a, X(arm_front))
    return _crop(a)


def _figure8(a, phase):
    """Legs too fast to see: a red figure of eight with the white straps flashing round it."""
    cx, cy = AXIS - 1, 33.5
    pts = [(cx + 11.5 * sin(k * pi / 16), cy + 4.2 * sin(k * pi / 8)) for k in range(33)]
    a.paint(a.line(pts, 3.4), "r")
    for k in range(4):
        u = phase * pi / 8 + k * pi / 2
        a.paint(a.line([(cx + 11.5 * sin(u + d), cy + 4.2 * sin(2 * (u + d))) for d in (0, 0.3, 0.6)], 3.0), "RSRW"[k], None)
    for k in (0, 2):
        u = phase * pi / 8 + k * pi / 2 + 1.1
        a.paint(a.line([(cx + 11.5 * sin(u + d), cy + 4.2 * sin(2 * (u + d))) for d in (0, 0.2)], 2.0), "q", None)


def _stride(t):
    """A foot on its walking loop: (x from the hip, ankle y, toe angle)."""
    ang = 0.6 if cos(t) < -0.3 else -0.2 if cos(t) > 0.3 else 0.15
    return 8.0 * cos(t), 36 - 4.5 * max(0.0, sin(t)) - (2.4 if ang > 0.3 else 0), ang


def _walk(i):
    t = i * pi / 3
    fx, fy, fa = _stride(t)
    bx, by, ba = _stride(t + pi)
    bob = 0 if i % 3 == 0 else -1
    swing = cos(t)
    return _hero(
        head=(1.5, 12 + bob, 0.1), body=(-0.5, 25.5 + bob, 5.4, 6.4, 0.18),
        arm_back=[(3, 21.5), (3 + 3 * swing, 26), (3 + 7 * swing, 27 - 3 * abs(swing))],
        arm_front=[(-3, 22.5), (-4 - 3 * swing, 26.5), (-4 - 7 * swing, 28 - 3 * abs(swing))],
        leg_back=[[(1, 30.5), (1 + bx * 0.6 + 1.5, (30.5 + by) / 2), (1 + bx, by)], ba],
        leg_front=[[(-2, 30.5), (-2 + fx * 0.6 + 1.5, (30.5 + fy) / 2), (-2 + fx, fy)], fa],
        tail=(-5.5, 27),
    )


def _run(i):
    up = i % 2 == 0
    bob = 0 if up else 1
    return _hero(
        head=(2.5, 12.5 + bob, 0.22), body=(-0.5, 24.5 + bob, 5.2, 6.2, 0.55),
        arm_back=[(3, 21 + bob), (7, 19 + bob), (10, 16 + bob)] if up else [(2, 22 + bob), (-3, 26 + bob), (-8, 25 + bob)],
        arm_front=[(-3, 23 + bob), (-8, 25 + bob), (-11, 21 + bob)] if up else [(0, 23 + bob), (5, 27 + bob), (9, 24 + bob)],
        leg_back=None, leg_front=None, tail=(-6, 25 + bob),
        under=lambda a: (a.paint(a.line([(AXIS - 2, 28), (AXIS - 1, 32)], 2.6), "c"), _figure8(a, i * 2)),
    )


def _roll(i):
    """The curled ball, one spin frame every 72 degrees, with a glimpse of face and belly."""
    a = _Art(30, 30)
    cx, cy, th = 15.0, 15.0, i * 2 * pi / 5
    pts = []
    for k in range(8):                  # eight blades, swept round
        u = th + k * pi / 4
        for du, r in ((0.0, 12.2), (0.22, 12.7), (0.42, 13.4), (0.6, 14.8), (0.68, 11.8)):
            pts.append((cx + r * cos(u + du), cy + r * sin(u + du)))
    a.paint(a.poly(pts), _ramp(_ball(a, cx, cy - 1, 16, 16, (-0.15, -0.72, 0.68)), BLUES, (-0.2, 0.3, 0.75, 0.98)))
    for k in range(8):                  # creases spiralling to the middle
        u = th + k * pi / 4 + 0.68
        a.paint(a.line(_bez((cx + 12 * cos(u), cy + 12 * sin(u)), (cx + 10 * cos(u - 0.5), cy + 10 * sin(u - 0.5)),
                            (cx + 7 * cos(u - 1.1), cy + 7 * sin(u - 1.1)), 4), 1.1), "a", None)
    u = th + 0.3                        # belly, eye and shoe, spinning past
    bx, by = cx + 2.5 * cos(u), cy + 2.5 * sin(u)
    a.paint(a.ell(bx, by, 5.4, 3.9, u + pi / 2), _ramp(_ball(a, bx, by, 7, 6, (0.2, -0.7, 0.7)), "hgf", (0.0, 0.7)), "a")
    a.paint(a.ell(bx + 4.8 * cos(u), by + 4.8 * sin(u), 1.4, 2.6, u), "W", "a")
    a.paint(a.ell(cx + 8.5 * cos(u + 2.3), cy + 8.5 * sin(u + 2.3), 2.6, 1.7, u + 2.3 + pi / 2), "r", "K")
    return a.rows()


SPRITES: dict = {}
SPRITES["idle_0"] = _hero(
    head=(0, 11), body=(-1, 25.5),
    arm_back=[(3, 21), (6.5, 24), (7.5, 28)],
    arm_front=[(-3, 22), (-6, 25.5), (-5, 29)],
    leg_back=[[(2, 31), (4, 36)], 0.0],
    leg_front=[[(-3, 31), (-4, 36)], 0.0],
    tail=(-6, 27),
)
for _i, _ang in enumerate((-0.45, 0.0)):
    SPRITES[f"wait_{_i}"] = _hero(
        head=(0, 11), body=(-1, 25.5), look="front",
        arm_back=[(3, 22), (7, 25), (3.5, 27)],
        arm_front=[(-3, 22), (-7, 25), (-3.5, 27.5)],
        leg_back=[[(-3, 31), (-5, 36)], 0.0],
        leg_front=[[(2, 31), (3, 36 - (1.5 if _i == 0 else 0))], _ang],
        tail=(-6, 27),
    )
SPRITES["lookup_0"] = _hero(
    head=(-1, 11, -0.38), body=(-1, 25.5, 5.6, 6.6, -0.12), look="up",
    arm_back=[(3, 21), (6.5, 24), (7.5, 28)],
    arm_front=[(-3, 22), (-6, 25.5), (-5, 29)],
    leg_back=[[(2, 31), (4, 36)], 0.0],
    leg_front=[[(-3, 31), (-4, 36)], 0.0],
    tail=(-6, 27),
)
SPRITES["duck_0"] = _hero(
    head=(3, 19, 0.45), body=(-1, 29, 5.6, 5.2, 0.5),
    arm_back=[(3, 28), (7, 31), (9, 31)],
    arm_front=[(0, 29), (4, 32), (7, 32)],
    leg_back=[[(1, 33), (4, 36)], 0.0],
    leg_front=[[(-3, 33), (-4, 36)], 0.0],
    tail=(-6, 31),
)
for _i in range(6):
    SPRITES[f"walk_{_i}"] = _walk(_i)
for _i in range(4):
    SPRITES[f"run_{_i}"] = _run(_i)
for _i in range(5):
    SPRITES[f"roll_{_i}"] = _roll(_i)
for _i in range(2):
    SPRITES[f"skid_{_i}"] = _hero(
        head=(-3, 12.5 + _i, -0.22), body=(-2.5, 26 + _i, 5.4, 6.2, -0.38), look="right",
        arm_back=[(0, 23), (5, 25 - _i), (9, 26 - _i)],
        arm_front=[(-5, 22), (-9, 22 + _i), (-13, 19 + _i)],
        leg_back=[[(-3, 30), (0, 32), (1, 36)], -0.25],
        leg_front=[[(0, 31), (5, 32), (8 + _i, 36)], -0.25],
        tail=(-8, 27),
    )
for _i in range(4):
    _a, _b = (-14, -2) if _i % 2 == 0 else (-2, -14)       # the feet trade places, the rear one on its toes
    _bob = (0, 1, 1, 0)[_i]                                # the strain runs out of step with them: four frames
    SPRITES[f"push_{_i}"] = _hero(
        head=(2, 13.5 + _bob, 0.32), body=(-2, 25.5 + _bob, 5.2, 6.2, 0.75),
        arm_back=[(3, 23), (8, 22), (12, 22)],
        arm_front=[(1, 26), (6, 27), (11, 26)],
        leg_back=[[(-4, 29), (_b + 2, 32), (_b, 33.5 if _b == -14 else 36)], 0.55 if _b == -14 else 0.0],
        leg_front=[[(-4, 29), (_a + 2, 32), (_a, 33.5 if _a == -14 else 36)], 0.55 if _a == -14 else 0.0],
        tail=(-8, 23),
    )
for _i in range(2):
    SPRITES[f"balance_{_i}"] = _hero(
        head=(2, 12, 0.2), body=(0, 25.5, 5.4, 6.4, 0.25), look="down", mouth="open",
        arm_back=[(3, 22), (9, 24 + _i), (14, 24 - 4 * _i)],
        arm_front=[(-4, 22), (-9, 21 + 3 * _i), (-13, 16 + 7 * _i)],
        leg_back=[[(1, 31), (2, 36)], 0.0],
        leg_front=[[(-2, 31), (-3, 36)], 0.0],
        tail=(-6, 28),
    )
SPRITES["spring_0"] = _hero(
    head=(0, 11, -0.42), body=(-1, 23, 4.8, 6.6, -0.06), look="up",
    arm_back=[(3, 19), (12, 15), (15, 8)],
    arm_front=[(-3, 20), (-10, 17), (-14, 11)],
    leg_back=[[(1, 28), (2, 31)], 0.72],
    leg_front=[[(-3, 28), (-3, 30.5)], 0.72],
    tail=(-5, 26),
)
SPRITES["hurt_0"] = _hero(
    head=(-2, 13, -0.3), body=(-1, 26, 5.4, 6.2, -0.45), look="front", mouth="open",
    arm_back=[(2, 23), (9, 24), (14, 22)],
    arm_front=[(-4, 22), (-10, 19), (-14, 15)],
    leg_back=[[(1, 30), (6, 32), (10, 29)], -0.7],
    leg_front=[[(-3, 31), (-1, 35), (4, 35)], -0.5],
    tail=(-7, 29),
)
SPRITES["dead_0"] = _hero(
    head=(0, 13, 0.0), body=(-1, 26, 5.4, 6.2, 0.0), look="shut", mouth="open",
    arm_back=[(3, 23), (10, 24), (15, 22)],
    arm_front=[(-4, 22), (-10, 19), (-13, 15)],
    leg_back=[[(2, 31), (7, 33), (10, 35)], 0.4],
    leg_front=[[(-3, 31), (-6, 33), (-9, 35)], 0.4],
)


# ── helpers shared by everything that is not the hero ─────────────────────────
def _orb(a, cx, cy, rx, ry, letters, line="K", light=(-0.4, -0.6, 0.7), rot=0.0, cuts=None):
    """An ellipse shaded as a ball, one band per letter, dark to light. Returns its mask."""
    m = a.ell(cx, cy, rx, ry, rot)
    a.paint(m, _ramp(_ball(a, cx, cy, rx * 1.15, ry * 1.15, light), letters,
                     cuts or tuple(np.linspace(-0.25, 0.85, len(letters) - 1))), line)
    return m


def _across(a, x0, x1, letters):
    """Bands of colour from left to right: a cone or post lit from the left."""
    return _ramp((a.x - x0) / (x1 - x0), letters, tuple((i + 1) / len(letters) for i in range(len(letters) - 1)))


def _thin(rows, keep):
    """Rows with pixels dropped in a checker pattern (keep: 1 of `keep` left) so a sprite seems to fade."""
    return ["".join(c if (x + y) % keep == 0 or c == "." else "." for x, c in enumerate(r)) for y, r in enumerate(rows)]


def _side(rows):
    """A spring turned on its side: its plate, on top in `rows`, now faces left."""
    return ["".join(c) for c in zip(*rows)]


def _star(a, cx, cy, r, ink, line=None, waist=0.3):
    pts = []
    for k in range(8):
        u, rr = k * pi / 4 - pi / 2, r if k % 2 == 0 else r * waist
        pts.append((cx + rr * cos(u), cy + rr * sin(u)))
    m = a.poly(pts)
    if r < 4:                       # a tiny star falls between pixel centres: all spike, no body
        m |= a.line([(cx - r, cy), (cx + r, cy)], 1.1) | a.line([(cx, cy - r), (cx, cy + r)], 1.1)
    a.paint(m, ink, line)


# ── enemies, all facing left ──────────────────────────────────────────────────
def _moto(i):
    a = _Art(40, 28)
    wx, wy = 21.0, 20.5
    a.paint(a.ell(wx, wy, 6.8, 6.8), "Z")
    a.paint(a.ell(wx, wy, 4.2, 4.2), "z", None)
    a.paint(a.ell(wx, wy, 1.8, 1.8), "x", None)
    u = i * pi / 3
    a.paint(a.line([(wx - 6 * cos(u), wy - 6 * sin(u)), (wx + 6 * cos(u), wy + 6 * sin(u))], 1.1), "X", None)
    a.paint(a.line([(31, 17), (37, 19)], 3), "x")                      # exhaust
    a.paint(a.line([(36.5, 19), (37.5, 19.3)], 2.0), "z", None)
    a.paint(a.line([(10, 3), (6, 0.8)], 1), "x")                       # antennae
    a.paint(a.line([(14, 2), (12, 0.8)], 1), "x")
    shell = a.ell(20, 12, 15.5, 10.5) & (a.y < 18.5)
    a.paint(shell, _ramp(_ball(a, 20, 12, 18, 13, (-0.4, -0.6, 0.7)), "qrRS", (0.0, 0.55, 0.9)))
    for x, y, r in ((21, 6, 2.8), (29, 10, 2.4), (16, 13, 2.2), (25, 15, 1.8)):    # the ladybug's spots
        a.paint(a.ell(x, y, r, r) & shell, "X", None)
    a.paint(shell & (a.y > 16.5), "X", None)
    a.paint(a.ell(9.5, 9, 3.6, 4.0), "W")                              # eye
    a.paint(a.ell(8.2, 9.5, 1.6, 2.2), "K", None)
    a.paint(a.poly([(3, 13), (13, 13), (15, 19), (6, 19)]), _across(a, 3, 15, "wxy"))
    a.paint(a.line([(5, 15.5), (13, 15.5)], 1), "X", None)
    return a.rows()


def _crab_arm(a, sx, sy, ex, ey, side):
    a.paint(a.line([(sx, sy), ((sx + ex) / 2, (sy + ey) / 2 + 1), (ex, ey)], 3), "q")
    _orb(a, ex, ey, 5.2, 5.0, "tOYU", rot=0.0)
    d = -1 if ey < sy - 6 else 1                                       # the pincers open away from the arm
    a.paint(a.poly([(ex - 1.5, ey + d * 1), (ex + 1.5, ey + d * 1), (ex, ey + d * 6.5)]), "m", None)


def _crab(i, fire=False):
    a = _Art(42, 32)
    cx = 21.0
    for s in (-1, 1):                                                  # three legs a side, walking
        for k in range(3):
            lift = 2 if (k + i + (s > 0)) % 2 == 0 else 0
            fx = cx + s * (10 + k * 3.5 + (i - 1) * 1.2)
            a.paint(a.line([(cx + s * (4 + k * 2), 21), (cx + s * (8 + k * 3), 24), (fx, 31 - lift)], 2.2), "x")
    ax, ay = (6, 5) if fire else (5.5, 15)
    _crab_arm(a, 11, 15, ax, ay, -1)
    _crab_arm(a, 31, 15, 42 - ax, ay, 1)
    _orb(a, cx, 15, 12.5, 9.5, "qrRS")
    a.paint(a.ell(cx, 20, 9, 4) & a.ell(cx, 15, 12.5, 9.5), "O", None)
    a.paint(a.ell(cx, 21, 7, 2.5) & a.ell(cx, 15, 12.5, 9.5), "t", None)
    for s in (-1, 1):                                                  # eyes on stalks
        a.paint(a.line([(cx + s * 4, 8), (cx + s * 5, 4)], 2), "q")
        a.paint(a.ell(cx + s * 5, 4, 3, 3.3), "W")
        a.paint(a.ell(cx + s * 5 - 1, 4.2, 1.2, 1.8), "K", None)
    a.paint(a.line([(cx - 4, 13), (cx + 4, 13)], 1), "Q", None)
    return a.rows()


def _buzz(i, fire=False):
    a = _Art(48, 24)
    # wings
    w = (a.ell(26, 6, 10, 4.2, 0.25), a.ell(34, 6, 8, 3.2, 0.55)) if i == 0 else (a.ell(30, 5, 12, 3.2, 0.05), a.ell(36, 8, 9, 2.6, 0.2))
    for m in w:
        a.paint(m, "C", "A")
        a.paint(m & ((a.x + a.y).astype(int) % 3 == 0), "W", None)
    # abdomen, curling under when it fires
    if fire:
        tip = [(24, 14), (31, 15), (34, 19), (30, 22)]
        a.paint(a.line(tip[:3], 7.5), _across(a, 24, 36, "KY" * 3))
        a.paint(a.poly([(33, 19), (27, 19), (25, 23.5)]), "x")
    else:
        a.paint(a.line([(24, 14), (34, 14), (40, 15)], 7.5), _ramp(np.floor((a.x - 22) / 4) % 2, "KY", (0.5,)))
        a.paint(a.poly([(41, 12.5), (41, 17.5), (47, 15)]), "x")
    # thorax and head
    _orb(a, 19, 13, 7.5, 6.5, "abcd")
    _orb(a, 9, 12, 6.8, 6.2, "abcd")
    _orb(a, 6.5, 11, 3.6, 3.8, "qRSW", light=(-0.5, -0.6, 0.6))
    a.paint(a.poly([(3, 15), (0.5, 18), (6, 17)]), "x")
    a.paint(a.line([(15, 19), (13, 22)], 1.6), "z")
    a.paint(a.line([(21, 19), (20, 22)], 1.6), "z")
    return a.rows()


def _flame(i):
    a = _Art(12, 8)
    n = 11 if i == 0 else 8
    a.paint(a.poly([(0, 1), (n, 4), (0, 7)]), "O", None)
    a.paint(a.poly([(0, 2.5), (n * 0.6, 4), (0, 5.5)]), "T", None)
    return a.rows()


def _shot(letters):
    a = _Art(8, 8)
    _orb(a, 4, 4, 3.4, 3.4, letters)
    return a.rows()


def _chopper(i):
    a = _Art(28, 32)
    # tail and fins
    a.paint(a.poly([(16, 24), (27, 22), (24, 31), (15, 31)]), _across(a, 15, 27, "OYU"))
    a.paint(a.poly([(13, 3), (21, 0.5), (25, 11), (17, 8)]), _across(a, 13, 25, "UYO"))
    a.paint(a.poly([(15, 21), (21, 19), (20, 26)]), "O")
    body = a.ell(14, 15, 12, 12.5)
    a.paint(body, _ramp(_ball(a, 14, 15, 14, 14.5), "qrRS", (0.0, 0.55, 0.9)))
    a.paint(a.ell(11, 22, 8.5, 5.0) & body, "w", None)
    a.paint(a.ell(11, 23, 7.5, 3.5) & body, "x", None)
    gape = (2.0, 15.5, 17.5) if i == 0 else (5.0, 14, 21)
    mouth = a.poly([(0, gape[1] - gape[0]), (13, gape[1] + 1), (0, gape[1] + gape[0] + 1)]) & body
    a.paint(mouth, "Q", None)
    a.paint(a.line([(1, gape[1] - gape[0]), (13, gape[1] + 1), (1.5, gape[1] + gape[0] + 1)], 1.1), "K", None)
    for x in (3, 6, 9):                                                # teeth
        a.paint(a.poly([(x - 1.4, gape[1] - gape[0] * (13 - x) / 13), (x + 1.4, gape[1] - gape[0] * (13 - x - 1.4) / 13),
                        (x, gape[1] - gape[0] * (13 - x) / 13 + 2.6)]) & body, "W", None)
        a.paint(a.poly([(x - 1.4, gape[1] + 1 + gape[0] * (13 - x) / 13), (x + 1.4, gape[1] + 1 + gape[0] * (13 - x - 1.4) / 13),
                        (x, gape[1] + 1 + gape[0] * (13 - x) / 13 - 2.6)]) & body, "W", None)
    a.paint(a.ell(11, 8, 3.8, 4.0), "W")
    a.paint(a.ell(9.5, 8.5, 1.7, 2.3), "K", None)
    return a.rows()


def _newtron(c):
    """A chameleon robot on a wall, c its four shades dark to light."""
    a = _Art(32, 38)
    a.paint(a.line(_bez((21, 26), (30, 34), (20, 36), 8) + _bez((20, 36), (13, 36), (15, 31), 5), 3.6), _ramp(a.y, c[0:3], (29.0, 33.0)))
    a.paint(a.line([(14, 29), (10, 33), (6, 34)], 2.8), c[1])
    a.paint(a.line([(21, 29), (21, 33), (17, 35)], 2.8), c[1])
    _orb(a, 18, 21, 9.5, 11, c)
    a.paint(a.ell(14, 23, 5.5, 8.5) & a.ell(18, 21, 9.5, 11), _ramp(_ball(a, 14, 23, 7, 10), "hgf", (0.0, 0.7)), None)
    a.paint(a.line([(13, 18), (8, 22), (5, 21)], 2.6), c[2])           # the arm
    a.paint(a.ell(4.5, 21, 2.2, 2.0), "W")
    a.paint(a.poly([(19, 2), (29, 0.5), (25, 8), (27, 13), (19, 11)]), c[2])    # the crest
    _orb(a, 15, 9, 10, 8, c)
    a.paint(a.ell(9, 13, 8, 2.5) & a.ell(15, 9, 10, 8), "g", None)     # the jaw
    a.paint(a.line([(1.5, 12), (13, 13)], 1), "K", None)
    a.paint(a.ell(12, 6, 4.4, 4.4), "W")
    a.paint(a.ell(10.8, 6.5, 2.0, 2.6), "K", None)
    a.paint(a.ell(10.3, 5.4, 0.8, 0.8), "W", None)
    return a.rows()


def _newtron_fly(c, i):
    a = _Art(40, 18)
    n = 9 if i == 0 else 6                                             # the jet
    a.paint(a.poly([(31, 5), (31 + n, 8.5), (31, 12)]), "O", None)
    a.paint(a.poly([(31, 7), (31 + n * 0.6, 8.5), (31, 10)]), "T", None)
    _orb(a, 19, 9, 13, 6.5, c)
    a.paint(a.poly([(22, 3), (30, 0.5), (30, 6)]), c[2])
    a.paint(a.ell(13, 12, 8, 2.8) & a.ell(19, 9, 13, 6.5), "g", None)
    a.paint(a.ell(10, 6, 3.2, 3.2), "W")
    a.paint(a.ell(9.2, 6.4, 1.4, 1.9), "K", None)
    a.paint(a.line([(7, 11), (1, 11)], 1), "K", None)
    return a.rows()


# ── effects ───────────────────────────────────────────────────────────────────
def _explode(i):
    a = _Art(32, 32)
    r = (5, 9, 12.5, 14.5, 15)[i]
    m = a.ell(16, 16, r * 0.72, r * 0.72)
    for k in range(7):
        u = k * 2 * pi / 7 + i * 0.7
        m |= a.ell(16 + r * 0.56 * cos(u), 16 + r * 0.56 * sin(u), r * 0.44, r * 0.44)
    d = np.hypot(a.x - 16, a.y - 16) / r
    letters = ("WTY", "WTYO", "TYOt", "YOry", "xyz")[i]
    ink = _ramp(d, letters, (0.45, 0.7, 0.9)[:len(letters) - 1])
    a.paint(m, ink)
    if i >= 3:
        a.g[(a.ell(16, 16, r * (0.15 if i == 3 else 0.5), r * (0.15 if i == 3 else 0.5)))] = ord(".")
    return a.rows()


def _bird(i):
    a = _Art(16, 16)
    _orb(a, 8, 9, 5, 4.2, "abcd")
    _orb(a, 4.5, 6, 3.2, 3.0, "bcde")
    a.paint(a.ell(3.4, 5.6, 1, 1), "W", None)
    a.paint(a.poly([(1.5, 5), (-0.5, 6.5), (1.8, 7)]), "O", None)
    a.paint(a.poly([(13, 9), (15.5, 7.5 + 3 * i), (13, 11)]), "b")
    a.paint(a.poly([(6, 8), (11, 8), (9.5, 1.5 + 11 * i)]), "d")
    return a.rows()


def _rabbit(i):
    a = _Art(16, 16)
    a.paint(a.line([(5, 7), (5.5, 1)], 2.2), "W")
    a.paint(a.line([(8, 7), (9.5, 2)], 2.2), "w")
    _orb(a, 9, 11, 5, 3.8, "xwW")
    _orb(a, 5, 7, 3.4, 3.2, "xwW")
    a.paint(a.ell(4, 6.6, 0.9, 1.1), "K", None)
    a.paint(a.ell(1.9, 8, 0.9, 0.8), "s", None)
    a.paint(a.line([(10, 14), (13 + 2 * i, 15 - 2 * i)], 2), "W")
    a.paint(a.line([(6, 14), (4 - i, 15)], 2), "W")
    return a.rows()


_DIGIT = {"0": ("###", "#.#", "#.#", "#.#", "###"), "1": (".#.", "##.", ".#.", ".#.", "###"),
          "2": ("###", "..#", "###", "#..", "###"), "5": ("###", "#..", "###", "..#", "###")}


def _points(text):
    h, w = 7, len(text) * 4 + 2
    g = [["."] * w for _ in range(h)]
    for n, ch in enumerate(text):
        for y, row in enumerate(_DIGIT[ch]):
            for x, c in enumerate(row):
                if c == "#":
                    for dx, dy in ((0, 1), (2, 1), (1, 0), (1, 2), (1, 1)):     # outlined, shadow first
                        if g[y + dy][n * 4 + x + dx] == ".":
                            g[y + dy][n * 4 + x + dx] = "K"
        for y, row in enumerate(_DIGIT[ch]):
            for x, c in enumerate(row):
                if c == "#":
                    g[y + 1][n * 4 + x + 1] = "T" if y < 3 else "Y"
    return ["".join(r) for r in g]


def _sparkle(i):
    a = _Art(16, 16)
    _star(a, 8, 8, (7.5, 7.5, 5.0, 3.0)[i], ("T", "W", "Y", "t")[i], "t" if i < 2 else None)
    if i < 2:
        _star(a, 8, 8, (3.5, 4.5)[i], "W")
    return a.rows()


def _shield(i):
    a = _Art(46, 46)
    d = np.hypot(a.x - 23, a.y - 23)
    a.paint((d < 22.6) & (d > 20.5), "D", None)
    a.paint((d < 21.5) & (d > 20.5) & (a.x + a.y < 46), "C", None)
    dither = ((a.x.astype(int) + a.y.astype(int) + i) % 2 == 0)
    a.paint((d <= 20.5) & dither & ((d > 15) | (i == 1)), "B", None)
    a.paint(a.ell(16, 15, 6, 3.5, -0.75) & dither, "C", None)
    return a.rows()


def _star_fx(i):
    a = _Art(16, 16)
    r = (7.5, 5.5, 3.5, 2.0)[i]
    _star(a, 8, 8, r, ("W", "T", "Y", "t")[i], "t" if i < 2 else None, 0.25)
    return a.rows()


# ── objects ───────────────────────────────────────────────────────────────────
GOLD = "utYTW"


def _ring(a, cx, cy, R, r, squeeze=1.0, line="n"):
    """A gold ring of outer radius R, hole r, turned on its axis (squeeze < 1)."""
    outer = a.ell(cx, cy, max(R * squeeze, 0.8), R)
    hole = a.ell(cx, cy, r * squeeze - 0.4, r) if r * squeeze > 1.2 else np.zeros(a.g.shape, bool)
    v = (cx - a.x) / (R * max(squeeze, 0.35)) * 0.55 + (cy - a.y) / R * 0.7
    a.paint(outer & ~hole, _ramp(v, GOLD, (-0.5, -0.05, 0.42, 0.72)), line)
    if hole.any():                                                     # the inner rim, lit the other way round
        rim = a.ell(cx, cy, r * squeeze + 0.2, r + 0.8) & ~hole & outer
        a.paint(rim, _ramp(-v, "tY", (0.0,)), None)
        a.g[hole] = ord(".")
        ring = hole.copy()
        ring[1:] |= hole[:-1]
        ring[:-1] |= hole[1:]
        ring[:, 1:] |= hole[:, :-1]
        ring[:, :-1] |= hole[:, 1:]
        a.g[ring & ~hole] = ord("n")


def _ringframe(squeeze, size=16, R=7.6, r=5.0):
    a = _Art(size, size)
    _ring(a, size / 2, size / 2, R, r, squeeze)
    return a.rows()


def _monitor():
    a = _Art(30, 32)
    a.paint(a.poly([(2, 1), (28, 1), (29, 3), (29, 26), (28, 28), (2, 28), (1, 26), (1, 3)]), _across(a, 1, 29, "yxwwwxy"))
    a.paint(a.poly([(1, 2), (29, 2), (29, 4), (1, 4)]), "W", None)
    a.paint(a.poly([(5, 6), (25, 6), (26, 7), (26, 22), (25, 23), (5, 23), (4, 22), (4, 7)]), "X", "Z")
    a.paint(a.poly([(6, 7), (10, 7), (6, 11)]), "Z", None)
    a.paint(a.poly([(4, 28), (26, 28), (27, 31.5), (3, 31.5)]), _across(a, 3, 27, "Zzyz"))
    a.paint(a.line([(7, 25.5), (9, 25.5)], 1.4), "R", None)
    return a.rows()


def _icon(kind):
    a = _Art(16, 12)
    if kind == "rings":
        _ring(a, 8, 6, 5.6, 3.0, 1.0)
    elif kind == "shield":
        a.paint(a.ell(8, 6, 5.6, 5.6), "B", "D")
        a.paint(a.ell(6, 4.5, 2.2, 1.6), "C", None)
    elif kind == "invincible":
        _star(a, 5, 4, 4, "T")
        _star(a, 11, 7, 4.8, "U")
        _star(a, 4, 9, 2.5, "W")
    elif kind == "shoes":
        _shoe(a, 5, 5.5, 0.0)
    elif kind == "life":
        a.paint(a.poly([(8, 1), (1, 2), (4, 5), (0, 6), (4, 8), (2, 11), (8, 11)]), "c")
        _orb(a, 8.5, 6, 5, 4.8, "bcd")
        a.paint(a.ell(9.5, 4.8, 2.2, 2.6) | a.ell(12.3, 4.8, 1.5, 2.3), "W", None)
        a.paint(a.ell(10.4, 4.8, 0.8, 1.5), "K", None)
        a.paint(a.ell(12.8, 4.8, 0.6, 1.4), "K", None)
        a.paint(a.ell(11, 8.6, 3.4, 1.6), "g", None)
        a.paint(a.ell(14.2, 7.2, 1, 0.8), "K", None)
    return a.rows()


def _broken():
    a = _Art(30, 32)
    a.paint(a.poly([(4, 16), (10, 12), (14, 15), (18, 11), (22, 15), (27, 13), (28, 28), (2, 28)]), _across(a, 2, 28, "yxwxy"))
    a.paint(a.poly([(6, 21), (10, 16), (14, 19), (18, 15), (22, 19), (25, 17), (25, 25), (5, 25)]), "X", "Z")
    a.paint(a.poly([(4, 28), (26, 28), (27, 31.5), (3, 31.5)]), _across(a, 3, 27, "Zzyz"))
    return a.rows()


def _spring(top, low, up):
    """A plate on a coil on a base. top: the plate's letters dark to light; up: extended."""
    a = _Art(28, 16)
    a.paint(a.poly([(2, 12), (26, 12), (26, 15.5), (2, 15.5)]), _across(a, 2, 26, "xwxy"))
    y = 1.0 if up else 7.0
    n = 3 if up else 1
    for k in range(n):                                                 # the coil
        h = (12 - (y + 5)) / n
        yy = y + 5 + k * h
        a.paint(a.line([(7, yy + h), (21, yy + h / 2), (7, yy)], 2), "y")
    a.paint(a.poly([(2, y + 1), (3, y), (25, y), (26, y + 1), (26, y + 5), (2, y + 5)]), _across(a, 2, 26, top))
    a.paint(a.poly([(2, y + 3.6), (26, y + 3.6), (26, y + 5), (2, y + 5)]), low, None)
    return a.rows()


def _spike():
    a = _Art(40, 32)
    for k in range(3):
        x0 = 1 + k * 13
        a.paint(a.poly([(x0, 28), (x0 + 6, 0.5), (x0 + 12, 28)]), _across(a, x0 + 1, x0 + 11, "WwxyZ"))
    a.paint(a.poly([(0, 27), (40, 27), (40, 31.5), (0, 31.5)]), _across(a, 0, 40, "xy" * 6))
    return a.rows()


def _rock():
    a = _Art(48, 32)
    outline = [(1, 31), (3, 20), (9, 9), (17, 5), (22, 1), (33, 2), (41, 9), (45, 18), (47, 31)]
    m = a.poly(outline)
    a.paint(m, "l")
    for ink, pts in (("P", [(9, 9), (17, 5), (22, 1), (24, 10), (15, 15)]),
                     ("p", [(22, 1), (33, 2), (34, 11), (24, 10)]),
                     ("p", [(3, 20), (9, 9), (15, 15), (13, 26), (1, 31)]),
                     ("L", [(33, 2), (41, 9), (45, 18), (36, 16), (34, 11)]),
                     ("L", [(24, 10), (34, 11), (36, 16), (30, 26), (20, 22)]),
                     ("M", [(36, 16), (45, 18), (47, 31), (32, 31), (30, 26)]),
                     ("l", [(15, 15), (24, 10), (20, 22), (13, 26)]),
                     ("L", [(13, 26), (20, 22), (30, 26), (32, 31), (10, 31)])):
        a.paint(a.poly(pts) & m, ink, None)
    a.paint(a.poly([(10, 10), (16, 7), (15, 11)]) & m, "M" if False else "W", None)
    return a.rows()


def _lamp(on):
    a = _Art(16, 64)
    a.paint(a.poly([(6.5, 14), (9.5, 14), (9.5, 58), (6.5, 58)]), _across(a, 6, 10, "whyy"[:3]))
    for y in (22, 34, 46):
        a.paint(a.poly([(6.5, y), (9.5, y), (9.5, y + 2), (6.5, y + 2)]), "B", None)
    a.paint(a.poly([(3, 58), (13, 58), (14, 63.5), (2, 63.5)]), _across(a, 2, 14, "wxyz"))
    a.paint(a.poly([(4, 13), (12, 13), (12, 16), (4, 16)]), "y")
    _orb(a, 8, 7, 6.6, 6.6, "qrRSW" if on else "abcde")
    return a.rows()


def _post(a, cx, top):
    a.paint(a.poly([(cx - 2, top), (cx + 2, top), (cx + 2, 60), (cx - 2, 60)]), _across(a, cx - 2, cx + 2, "wxy"))
    a.paint(a.poly([(cx - 8, 60), (cx + 8, 60), (cx + 10, 63.5), (cx - 10, 63.5)]), _across(a, cx - 10, cx + 10, "xwxyz"))


def _sign_plate(a, half):
    """The sign's disc, half its half-width as it spins."""
    a.paint(a.ell(24, 22, max(half, 1), 21), _across(a, 24 - half, 24 + half, "txYTYt") if half > 2 else "Y")
    if half > 6:
        a.paint(a.ell(24, 22, half - 3, 18), "B", None)


def _pilot_face(a, cx, cy):
    """The robot-maker's round head: bald, red goggles, big moustache."""
    _orb(a, cx, cy, 11, 11, "hgf", light=(-0.3, -0.6, 0.7))
    a.paint(a.ell(cx, cy - 3.5, 8.5, 3.2), "K", None)
    a.paint(a.ell(cx - 4, cy - 3.5, 3.2, 2.5) | a.ell(cx + 4, cy - 3.5, 3.2, 2.5), "R", None)
    a.paint(a.ell(cx - 4.8, cy - 4.2, 1, 1) | a.ell(cx + 3.2, cy - 4.2, 1, 1), "W", None)
    a.paint(a.ell(cx, cy + 1, 2.4, 2), "h", None)
    a.paint(a.poly([(cx, cy + 2.5), (cx - 11, cy + 2), (cx - 8, cy + 7), (cx, cy + 5), (cx + 8, cy + 7), (cx + 11, cy + 2)]), "n", None)


def _sign(kind):
    a = _Art(48, 64)
    _post(a, 24, 40)
    if kind == "robot":
        _sign_plate(a, 23)
        _pilot_face(a, 24, 22)
    elif kind == "spin_0":
        _sign_plate(a, 12)
    elif kind == "spin_1":
        _sign_plate(a, 2.5)
    else:
        _sign_plate(a, 23)
        _head(a, 25, 22)
    return a.rows()


def _log():
    a = _Art(16, 16)
    a.paint(a.poly([(0, 2), (2, 0), (14, 0), (16, 2), (16, 13), (14, 15), (2, 15), (0, 13)]), _ramp(a.y, "ktnNN"[:4], (3.0, 7.0, 12.0)), None)
    a.paint(a.line([(0, 8), (5, 8), (9, 6), (16, 6)], 1), "N", None)
    a.paint(a.line([(0.5, 0), (0.5, 15)], 1), "m", None)
    a.paint(a.line([(15.5, 0), (15.5, 15)], 1), "m", None)
    a.paint(a.line([(1, 15.5), (15, 15.5)], 1), "m", None)
    return a.rows()


def _swing_plat():
    a = _Art(48, 24)
    a.paint(a.line([(24, 0), (24, 8)], 2), "x")
    for k in range(3):
        x = 1 + k * 16
        a.paint(a.poly([(x, 8), (x + 2, 6), (x + 13, 6), (x + 15, 8), (x + 15, 19), (x + 13, 21), (x + 2, 21), (x, 19)]),
                _ramp(a.y, "ktnN", (10.0, 14.0, 18.0)))
        a.paint(a.line([(x + 1, 13), (x + 7, 13), (x + 10, 11), (x + 14, 11)], 1), "N", None)
    return a.rows()


def _boss(i=0, hurt=False):
    a = _Art(64, 48)
    if i == 0 or hurt:
        a.paint(a.poly([(46, 40), (60, 36), (62, 42), (48, 46)]), "O", None)  # the jet
        a.paint(a.poly([(46, 41), (55, 39), (56, 42), (48, 44)]), "T", None)
    a.paint(a.ell(29, 16, 21, 15) & (a.y < 24), "C", "K")                # the glass dome
    a.paint(a.ell(21, 10, 8, 5, -0.5) & (a.y < 24), "W", None)
    # pilot
    _orb(a, 26, 24, 14, 13, "qrRS")
    _orb(a, 27, 11, 10, 10, "hgf", light=(-0.3, -0.6, 0.7))
    a.paint(a.ell(21, 10, 6, 3), "K", None)
    a.paint(a.ell(19, 10, 2.8, 2.3) | a.ell(25, 10, 2.8, 2.3), "S" if hurt else "R", None)
    a.paint(a.poly([(12, 14), (27, 14), (27, 20), (21, 18), (14, 19)]), "n", None)
    # the hull, an egg lying on its side
    hull = a.ell(30, 32, 28, 15.5) & (a.y > 20)
    a.paint(hull, _ramp(_ball(a, 30, 30, 32, 19), "zyxwW", (-0.2, 0.2, 0.55, 0.85)))
    a.paint(a.poly([(2, 30), (58, 30), (58, 33), (2, 33)]) & hull, "R", None)
    a.paint(a.poly([(2, 33), (58, 33), (58, 34), (2, 34)]) & hull, "q", None)
    a.paint(a.ell(10, 25, 2, 2) | a.ell(50, 25, 2, 2), "Y", "K")
    a.paint(a.poly([(22, 45), (38, 45), (40, 47.5), (20, 47.5)]), _across(a, 20, 40, "xyzZ"))
    a.paint(a.poly([(25, 46), (35, 46), (34, 47.5), (26, 47.5)]), "Y" if i else "O", None)
    if hurt:
        a.g[(a.ell(30, 32, 28, 15.5) & (a.y > 20)) & ((a.x.astype(int) + a.y.astype(int)) % 2 == 0)] = ord("W")
    return a.rows()


def _boss_ball():
    a = _Art(48, 48)
    m = a.ell(24, 24, 23.5, 23.5)
    nx, ny = (a.x - 24) / 24, (a.y - 24) / 24
    nz = np.sqrt(np.clip(1 - nx * nx - ny * ny, 0.02, 1))
    lon, lat = np.arctan2(nx, nz), np.arcsin(np.clip(ny, -1, 1))        # the squares wrap round the ball
    light = (np.floor(lon / (pi / 6)) + np.floor(lat / (pi / 6))) % 2 == 0
    v = _ball(a, 24, 24, 26, 26)
    ink = np.where(light, _ramp(v, "ntYT", (0.0, 0.5, 0.9)), _ramp(v, "KXzy", (0.2, 0.7, 0.98)))
    a.paint(m, ink)
    return a.rows()


def _capsule(opened=False):
    a = _Art(64, 60)
    a.paint(a.poly([(6, 48), (58, 48), (62, 59.5), (2, 59.5)]), _across(a, 2, 62, "wxyzZ"))
    a.paint(a.poly([(6, 48), (58, 48), (59, 50), (5, 50)]), "W", None)
    egg = a.ell(32, 26, 24, 23) & (a.y < 49)
    a.paint(egg, _ramp(_ball(a, 32, 24, 28, 26), "zyxwW", (-0.2, 0.2, 0.6, 0.9)))
    a.paint(egg & (a.y > 26) & (a.y < 29), "R", None)
    a.paint(egg & (a.y >= 29) & (a.y < 30), "q", None)
    a.paint(a.ell(32, 16, 12, 9), "Z", "K")
    if opened:
        a.paint(a.ell(32, 16, 11, 8), "X", None)
        a.g[a.ell(32, 12, 22, 8)] = ord(".")
    else:
        a.paint(a.poly([(27, 3), (37, 3), (38, 7), (26, 7)]), "R")
        a.paint(a.ell(29, 17, 2, 3) | a.ell(35, 17, 2, 3), "W", None)
    return a.rows()


def _palm_crown():
    a = _Art(60, 40)

    def frond(tip, ctrl, width, ink):
        pts = _bez((30, 16), ctrl, tip, 10)
        m = np.zeros(a.g.shape, bool)
        for k in range(len(pts) - 1):
            w = width * (1 - k / (len(pts) - 1)) + 0.8
            m |= a.line(pts[k:k + 2], w)
            if k % 2 == 0 and k > 1:                                   # leaflets hanging from the rib
                m |= a.line([pts[k], (pts[k][0], pts[k][1] + w + 2.5)], 1.6)
        a.paint(m, ink, "I")

    frond((3, 33), (11, 3), 5.5, _ramp(a.y, "HGFE", (16.0, 24.0, 32.0))[::-1])
    frond((57, 33), (49, 3), 5.5, _ramp(a.y, "HGFE", (16.0, 24.0, 32.0))[::-1])
    frond((10, 39), (20, 13), 5, _ramp(a.y, "HGF", (22.0, 30.0)))
    frond((50, 39), (40, 13), 5, _ramp(a.y, "HGF", (22.0, 30.0)))
    frond((4, 12), (16, -2), 5.5, _ramp(a.x, "EFG", (14.0, 22.0)))
    frond((56, 12), (44, -2), 5.5, _ramp(a.x, "GFE", (38.0, 46.0)))
    frond((30, 1), (30, 8), 4.5, "F")
    _orb(a, 27, 17, 3.2, 3.2, "jnk")
    _orb(a, 33, 17, 3.2, 3.2, "jnk")
    return a.rows()


def _palm_trunk():
    a = _Art(12, 16)
    ink = _across(a, 1, 11, "kJnj")
    a.paint(a.poly([(1.5, 0), (10.5, 0), (10.5, 16), (1.5, 16)]), ink, "m")
    a.paint(a.poly([(1.5, 0), (10.5, 0), (10.5, 3), (1.5, 3)]), _across(a, 1, 11, "Jnjj"), None)
    a.paint(a.line([(1.5, 8.5), (10.5, 8.5)], 1), "n", None)
    return a.rows()


def _sunflower(i):
    a = _Art(32, 64)
    sway = 1 if i else -1
    a.paint(a.line([(16, 62), (16, 44), (16 + sway, 26)], 3), _ramp(a.x, "EFG", (15.5, 17.0)), "I")
    for x, y, s in ((16, 50, -1), (16, 42, 1)):
        a.paint(a.poly([(x, y), (x + s * 5, y - 7), (x + s * 13, y - 6), (x + s * 8, y)]), _across(a, x - 13, x + 13, "FGH"), "I")
    cx, cy = 16 + 2 * sway, 15
    for k in range(10):                                                # petals round the head
        u = k * pi / 5 + (pi / 10 if i else 0)
        a.paint(a.ell(cx + 10 * cos(u), cy + 10 * sin(u), 4.8, 3.0, u), "U" if k % 2 == 0 else "Y", "t")
    _orb(a, cx, cy, 7.2, 7.2, "NnJk"[:3], line="m")
    d = (a.x.astype(int) + a.y.astype(int)) % 3 == 0
    a.paint(a.ell(cx, cy, 6, 6) & d, "N", None)
    return a.rows()


def _flower(i):
    a = _Art(32, 32)
    a.paint(a.line([(16, 31), (16, 14)], 3), "G", "I")
    for s in (-1, 1):
        a.paint(a.poly([(16, 30), (16 + s * 6, 21), (16 + s * 12, 24), (16 + s * 7, 30)]), _across(a, 3, 29, "EFG"), "I")
    cx, cy = 16 + (1 if i else -1), 10
    for k in range(6):
        u = k * pi / 3 + (0.5 if i else 0.0)
        a.paint(a.ell(cx + 6 * cos(u), cy + 6 * sin(u), 4.2, 3.4, u), "pP"[k % 2], "L")
    a.paint(a.ell(cx, cy, 3.2, 3.2), "Y", "t")
    return a.rows()


def _totem():
    a = _Art(16, 48)
    a.paint(a.poly([(4, 7), (12, 7), (12, 47.5), (4, 47.5)]), _across(a, 4, 12, "ktnN"))
    for y in (12, 22, 32, 42):
        a.paint(a.poly([(4, y), (12, y), (12, y + 2), (4, y + 2)]), "N", None)
        a.paint(a.ell(8, y - 4, 2, 1.6), "n", None)
    for s in (-1, 1):                                                  # the wings
        a.paint(a.poly([(8 + s * 3, 8), (8 + s * 7.5, 0.5), (8 + s * 7.5, 12), (8 + s * 3, 14)]), "W" if s < 0 else "w", "z")
    a.paint(a.ell(8, 6, 4.4, 4.4), _ramp(_ball(a, 8, 6, 5, 5), "ntYT", (0.0, 0.5, 0.85)))
    a.paint(a.line([(5, 6), (11, 6)], 1), "K", None)
    return a.rows()


def _waterfall(i):
    table = (3, 12, 6, 15, 1, 9, 4, 13)
    rows = []
    for y in range(16):
        row = ""
        for x in range(16):
            v = (table[x // 2] + y - i * 4) % 16
            row += "W" if v < 2 else "C" if v < 4 else "D" if v < 8 else "B" if v < 13 else "A"
        rows.append(row)
    return rows


# the dirt: 16 px checks, the two tones with a bevel, as a repeating 32 x 32 tile
def _check_tile():
    rows = [["."] * 32 for _ in range(32)]
    for y in range(32):
        for x in range(32):
            u, v = x % 16, y % 16
            dark = (x // 16 + y // 16) % 2
            hi, mid, lo = ("o", "n", "N") if dark else ("O", "o", "n")
            rows[y][x] = hi if u == 0 or v == 0 else lo if u == 15 or v == 15 else mid
            if u + v in (2, 3) and min(u, v) > 0:                      # the corner catches the light
                rows[y][x] = hi
    return ["".join(r) for r in rows]


CHECK = _check_tile()


def _grass_h(x):
    """How deep the grass hangs at world column x: round scallops every 16 px, a nick between them."""
    u = (x % 16 - 7.5) / 8
    return 9 + int(4.5 * (1 - u * u) ** 0.5)


def _grass_tile():
    """32 columns by 15 rows of grass as it hangs from the surface; '.' lets the dirt through."""
    rows = []
    for d in range(15):
        row = ""
        for x in range(32):
            h = _grass_h(x)
            ch = "."
            if d < h:
                ch = "E" if d == 0 and x % 4 else "F" if d == 0 else "F" if d < 3 else "G"
                if d >= h - 1:
                    ch = "H"
                elif d >= h - 3:
                    ch = "I" if (x + d) % 5 == 0 else "G"
                elif d in (3, 4) and (x * 3 + d) % 7 == 0:
                    ch = "E"
                elif d == 2 and x % 4 == 2:
                    ch = "G"
            elif d == h:
                ch = "m"
            row += ch
        rows.append(row)
    return rows


GRASS = _grass_tile()


def _platform():
    a = _Art(64, 30)
    m = a.poly([(0, 0), (64, 0), (64, 12), (60, 20), (52, 27), (40, 30), (24, 30), (12, 27), (4, 20), (0, 12)])
    g = np.array([[ord(CHECK[y % 32][x % 32]) for x in range(64)] for y in range(30)], np.uint8)
    a.paint(m, g, None)
    gr = np.array([[ord(GRASS[y][x % 32]) if y < 15 else 46 for x in range(64)] for y in range(30)], np.uint8)
    a.g[m & (gr != 46)] = gr[m & (gr != 46)]
    edge = m & ~(np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1))
    a.g[edge & (a.y > 14)] = ord("m")
    return a.rows()


def _ledge():
    gr = [GRASS[y][:16] if y < 15 else "." * 16 for y in range(16)]
    return ["".join(c if c != "." else CHECK[y][x] for x, c in enumerate(r)) for y, r in enumerate(gr)]


# ── the sprite table ──────────────────────────────────────────────────────────
for _i in range(3):
    SPRITES[f"moto_{_i}"] = _moto(_i)
    SPRITES[f"crab_{_i}"] = _crab(_i)
SPRITES["crab_fire_0"] = _crab(1, True)
for _i in range(2):
    _a = _Art(10, 10)
    _orb(_a, 5, 5, 3.5 + _i * 1.2, 3.5 + _i * 1.2, "yxwW", line="z" if _i else "y")
    SPRITES[f"moto_smoke_{_i}"] = _a.rows()
    SPRITES[f"buzz_{_i}"] = _buzz(_i)
    SPRITES[f"buzz_flame_{_i}"] = _flame(_i)
    SPRITES[f"chopper_{_i}"] = _chopper(_i)
_a = _Art(12, 12)
_orb(_a, 6, 6, 5.2, 5.2, "ntOYT")
SPRITES["crab_ball"] = _a.rows()
SPRITES["buzz_fire_0"] = _buzz(0, True)
SPRITES["buzz_shot"] = _shot("qRST")
_BLUE_N, _GREEN_N = "bcde", "HGFE"
for _t, _c in (("b", _BLUE_N), ("g", _GREEN_N)):
    _full = _newtron(_c)
    SPRITES[f"newtron_{_t}_0"] = _thin(_full, 3)
    SPRITES[f"newtron_{_t}_1"] = _thin(_full, 2)
    SPRITES[f"newtron_{_t}_2"] = _full
for _i in range(2):
    SPRITES[f"newtron_fly_{_i}"] = _newtron_fly(_GREEN_N, _i)
SPRITES["newtron_shot"] = _shot("qRSW")
for _i in range(5):
    SPRITES[f"explode_{_i}"] = _explode(_i)
for _i in range(2):
    SPRITES[f"freed_{_i}"] = _bird(_i)
    SPRITES[f"freed_{_i + 2}"] = _rabbit(_i)
    SPRITES[f"shield_{_i}"] = _shield(_i)
    SPRITES[f"bigring_{_i}"] = _ringframe((1.0, 0.45)[_i], 64, 31, 20)
for _p in ("100", "200", "500", "1000"):
    SPRITES[f"points_{_p}"] = _points(_p)
for _i in range(4):
    SPRITES[f"sparkle_{_i}"] = _sparkle(_i)
    SPRITES[f"star_{_i}"] = _star_fx(_i)
    SPRITES[f"ring_{_i}"] = _ringframe((1.0, 0.72, 0.42, 0.14)[_i])
    SPRITES[f"waterfall_{_i}"] = _waterfall(_i)
SPRITES["monitor"], SPRITES["monitor_broken"] = _monitor(), _broken()
for _k in ("rings", "shield", "invincible", "shoes", "life"):
    SPRITES[f"icon_{_k}"] = _icon(_k)
for _i in range(2):
    SPRITES[f"spring_y_{_i}"] = _spring("tYUT", "t", _i == 1)
    SPRITES[f"spring_r_{_i}"] = _spring("qrRS", "q", _i == 1)
    SPRITES[f"spring_side_y_{_i}"] = _side(SPRITES[f"spring_y_{_i}"])
    SPRITES[f"spring_side_r_{_i}"] = _side(SPRITES[f"spring_r_{_i}"])
    SPRITES[f"lamp_{_i}"] = _lamp(_i == 1)
    SPRITES[f"boss_ship_{_i}"] = _boss(_i)
    SPRITES[f"sunflower_{_i}"] = _sunflower(_i)
    SPRITES[f"flower_{_i}"] = _flower(_i)
SPRITES["spike"], SPRITES["rock"], SPRITES["log"] = _spike(), _rock(), _log()
SPRITES["sign_robot"], SPRITES["sign_hero"] = _sign("robot"), _sign("hero")
SPRITES["sign_spin_0"], SPRITES["sign_spin_1"] = _sign("spin_0"), _sign("spin_1")
SPRITES["platform"], SPRITES["ledge_piece"] = _platform(), _ledge()
SPRITES["swing_link"] = _ringframe(1.0, 16, 7.0, 3.6)
SPRITES["chain_link"] = _ringframe(0.8, 12, 5.2, 2.4)
_a = _Art(24, 16)
_a.paint(_a.poly([(2, 0), (22, 0), (22, 4), (15, 8), (9, 8), (2, 4)]), _across(_a, 2, 22, "wxyz"))
_a.paint(_a.ell(12, 10, 4.2, 4.2), "Y", "n")
_a.g[_a.ell(12, 10, 1.8, 1.8)] = ord(".")
SPRITES["swing_anchor"] = _a.rows()
SPRITES["swing_plat"] = _swing_plat()
SPRITES["boss_hurt"], SPRITES["boss_ball"] = _boss(0, True), _boss_ball()
SPRITES["capsule"], SPRITES["capsule_open"] = _capsule(), _capsule(True)
SPRITES["palm_crown"], SPRITES["palm_trunk"] = _palm_crown(), _palm_trunk()
SPRITES["totem"] = _totem()


# ── surfaces ──────────────────────────────────────────────────────────────────
@lru_cache(maxsize=None)
def _index() -> dict:
    return art_png.index(SPRITES)


_OWN_ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "sonic-data", "art")


def sprite(name: str, flip_x: bool = False, flip_y: bool = False) -> pygame.Surface:
    """A sprite at one art pixel per game pixel, per-pixel alpha: its PNG when there is one, else its text."""
    return _sprite(name, bool(flip_x), bool(flip_y))       # one cache entry however the flags were passed


@lru_cache(maxsize=None)
def _sprite(name: str, flip_x: bool, flip_y: bool) -> pygame.Surface:
    rows = SPRITES[name]
    own = os.path.join(_OWN_ART, f"{name}.png")     # art you own, kept out of the repo (sonic-data/ is ignored)
    png = pygame.image.load(own) if os.path.isfile(own) else art_png.load("sonic", _index()[tuple(rows)])
    img = art_png.to_surface(rows, PAL) if png is None else png
    out = pygame.Surface(img.get_size(), pygame.SRCALPHA, 32)
    out.blit(img, (0, 0))
    return pygame.transform.flip(out, flip_x, flip_y) if flip_x or flip_y else out


@lru_cache(maxsize=None)
def hero(pose: str, rot: int = 0, flip_x: bool = False) -> pygame.Surface:
    """A hero frame turned rot * 45 degrees clockwise. A diagonal is cut from an 8x blow-up (scale2x keeps
    the outline clean) so the pixels stay hard; the result is bigger than the frame but still centred."""
    img = sprite(pose, flip_x)
    rot %= 8
    if rot == 0:
        return img
    if rot % 2 == 0:
        return pygame.transform.rotate(img, -45 * rot)
    big = pygame.transform.scale2x(pygame.transform.scale2x(pygame.transform.scale2x(img)))
    big = pygame.transform.rotate(big, -45 * rot)
    w, h = big.get_width() // 8, big.get_height() // 8
    return pygame.transform.scale(big, (w + w % 2, h + h % 2))


# ── terrain ───────────────────────────────────────────────────────────────────
CHUNK = 256
_MARGIN = 80                # how far round a chunk its surfaces are looked for: more than any band below
_BAND = 56                  # the checks follow the surface down to here, then the plain grid takes over
_LUT = np.zeros((128, 4), np.uint8)
for _k, _v in PAL.items():
    _LUT[ord(_k)] = (*_v, 255)
_DARK = np.arange(128, dtype=np.uint8)      # one step down the dirt's ramp, for shadow
for _k, _v in zip("OonNmEFGHI", "onNmmFGHIm"):
    _DARK[ord(_k)] = ord(_v)
_CHECK = np.array([[ord(c) for c in r] for r in CHECK], np.uint8)
_GRASS = np.array([[ord(c) for c in r] for r in GRASS], np.uint8)
_RELIEF = [                 # dark blocks in deep dirt, one letter a 16 px cell, repeating every 192 px
    "............",
    ".######.....",
    ".######..###",
    "...###...###",
    "...###.#####",
    ".......#####",
    "####.....###",
    "####.....###",
    "####..#.....",
    ".###.###....",
    ".....###..##",
    ".........###",
]
_RELIEF_CELLS = np.array([[c == "#" for c in r] for r in _RELIEF], bool)


def _shift(m, dy, dx):
    """m moved by (dy, dx), with False coming in at the edges."""
    out = np.zeros_like(m)
    out[max(dy, 0):m.shape[0] + min(dy, 0), max(dx, 0):m.shape[1] + min(dx, 0)] = \
        m[max(-dy, 0):m.shape[0] + min(-dy, 0), max(-dx, 0):m.shape[1] + min(-dx, 0)]
    return out


@lru_cache(maxsize=1)
def _level_maps(level):
    """Whole-level masks, made once: what is solid in either plane, the tunnels' bores, and the wall left
    behind each carved walkway."""
    solid = level.masks_np[0] | level.masks_np[1]
    polys = level.polys
    size = (level.px_w, level.px_h)
    bore = np.zeros(solid.shape, bool)
    if level.tubes:
        with open(os.path.join(sonic_level.LEVEL_DIR, "tube.yaml"), encoding="utf-8") as f:
            tube = sonic_level.delta(yaml.safe_load(f)["tube"])
        pad = 16
        one = sonic_level._grow(sonic_level._fill((256 + 2 * pad, 288 + 2 * pad), [[(x + pad, y + pad) for x, y in tube]]),
                                 sonic_level.TUBE_GROW + 6)    # past the carved bore: tracing leaves slivers of air along the walls
        for x, y in level.data["tubes"]:
            bore[y - pad:y + 288 + pad, x - pad:x + 256 + pad] |= one
        bore &= ~solid
    wall = sonic_level._open(sonic_level._fill(size, polys) & ~solid & ~bore, 3)   # a room, not a shaved edge
    for x0, x1, y in level.bridges:
        wall[max(y - 32, 0):y + 24, x0 + 24:x1 - 23] = False
    for cx, cy, r, _ in level.loops:
        yy, xx = np.ogrid[cy - r - 2:cy + r + 2, cx - r - 2:cx + r + 2]
        wall[cy - r - 2:cy + r + 2, cx - r - 2:cx + r + 2] &= (xx - cx) ** 2 + (yy - cy) ** 2 > (r + 1) ** 2
    return solid, bore, wall


def _window(m, x0, y0, size):
    """size x size of m from (x0, y0), blank where that runs off the level."""
    H, W = m.shape
    out = np.zeros((size, size), bool)
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + size, W), min(y0 + size, H)
    if xa < xb and ya < yb:
        out[ya - y0:yb - y0, xa - x0:xb - x0] = m[ya:yb, xa:xb]
    return out


@lru_cache(maxsize=256)
def terrain_chunk(level, cx: int, cy: int) -> pygame.Surface:
    """One 256 x 256 px chunk: checks that follow the slope, scalloped grass on open tops, dark rims on
    cliffs and roofs, dark relief in deep dirt, walls behind walkways, tunnels as dark bores."""
    solid, bore, wall = _level_maps(level)
    x0, y0, m, size = cx * CHUNK, cy * CHUNK, _MARGIN, CHUNK + 2 * _MARGIN
    s = _window(solid, x0 - m, y0 - m, size)
    b = _window(bore, x0 - m, y0 - m, size)
    w = _window(wall, x0 - m, y0 - m, size)
    ys, xs = np.mgrid[y0 - m:y0 - m + size, x0 - m:x0 - m + size]
    # how far each pixel lies under the surface above it; a tunnel's floor and a wall's foot count as dirt
    row = np.arange(size)[:, None]
    open_air = ~(s | b)
    depth = row - np.maximum.accumulate(np.where(open_air, row, -1), axis=0) - 1
    near = depth < _BAND
    # checks: on the grid deep down, hanging from the surface near it
    ty = np.where(near, depth, ys)
    tile = _CHECK[ty % 32, xs % 32]
    tile[(depth >= _BAND - 2) & (depth < _BAND)] = ord("n")     # the two meet on a ledge line, not a broken row
    tile[(depth >= _BAND) & (depth < _BAND + 2)] = ord("m")
    tile[(depth >= _BAND + 2) & (depth < _BAND + 5)] = _DARK[tile[(depth >= _BAND + 2) & (depth < _BAND + 5)]]
    # dark relief in the deep dirt, in 16 px cells
    cells = s[::16, ::16] & s[15::16, ::16] & s[::16, 15::16] & s[15::16, 15::16]
    ok = cells & _shift(cells, 1, 0) & _shift(cells, -1, 0) & _shift(cells, 0, 1) & _shift(cells, 0, -1)
    ok = ok & _shift(ok, 2, 0) & _shift(ok, 3, 0) & _shift(ok, 0, 1) & _shift(ok, 0, -1)
    ci, cj = (y0 - m) // 16, (x0 - m) // 16
    pat = _RELIEF_CELLS[(ci + np.arange(ok.shape[0]))[:, None] % 12, (cj + np.arange(ok.shape[1]))[None, :] % 12]
    relief = np.repeat(np.repeat(ok & pat, 16, axis=0), 16, axis=1)[:size, :size] & ~near
    if relief.any():
        tile[relief] = _DARK[_CHECK[ys[relief] % 32, xs[relief] % 32]]
        tile[relief & ~_shift(relief, 1, 0)] = ord("m")
        tile[relief & ~_shift(relief, 0, 1)] = ord("m")
        tile[relief & ~_shift(relief, -1, 0)] = ord("o")
        tile[relief & ~_shift(relief, 0, -1)] = ord("n")
    # the wall behind a walkway, a step darker with a shadow under its roof
    if w.any():
        tile[w] = _DARK[_DARK[_CHECK[ys[w] % 32, xs[w] % 32]]]
        roof = w & _shift(s, 1, 0)
        for k in range(1, 6):
            roof |= w & _shift(s, k + 1, 0)
        tile[roof] = _DARK[tile[roof]]
    # rims: cliffs left and right, roofs below
    inner = s.copy()
    for k in (1, 2, 3):
        inner &= _shift(s, 0, k) & _shift(s, 0, -k) & _shift(s, -k, 0)
        if k == 1:
            edge = s & ~inner
    rim = s & ~inner
    tile[rim] = _DARK[tile[rim]]
    tile[edge] = ord("m")
    # grass on what stands open to the sky
    grass = _GRASS[np.clip(depth, 0, 14), xs % 32]
    show = s & (depth >= 0) & (depth < 15) & (grass != ord("."))
    tile[show] = grass[show]
    # tunnels: a dark bore with an orange rim
    if b.any():
        tile[b] = ord("m")
        tile[b & ((xs // 16 + ys // 16) % 2 == 0)] = ord("N")
        near_wall = _shift(s, 1, 0) | _shift(s, -1, 0) | _shift(s, 0, 1) | _shift(s, 0, -1)
        ring = b & near_wall
        tile[ring] = ord("O")
        ring2 = b & ~ring & (_shift(ring, 1, 0) | _shift(ring, -1, 0) | _shift(ring, 0, 1) | _shift(ring, 0, -1))
        tile[ring2] = ord("o")
    rgba = _LUT[tile[m:-m, m:-m]].copy()
    rgba[..., 3] = (s | b | w)[m:-m, m:-m] * 255
    return pygame.image.frombuffer(rgba.tobytes(), (CHUNK, CHUNK), "RGBA").copy()


# ── the backdrop ──────────────────────────────────────────────────────────────
_BACK_W = 1024              # every band tiles at this width
_SKY = (0x6D, 0xB6, 0xFF)
_LAKE_FRAMES = 4


def _wave(x, *parts):
    """A height from whole cosine waves, so it tiles: parts are (cycles per tile, height, phase)."""
    return sum(h * np.cos(2 * pi * (c * x / _BACK_W) + p) for c, h, p in parts)


def _band(h, paint):
    """A tiling strip: paint(y, x) gives letters; '.' stays clear."""
    y, x = np.mgrid[0:h, 0:_BACK_W]
    rgba = _LUT[paint(y, x)]
    return pygame.image.frombuffer(rgba.tobytes(), (_BACK_W, h), "RGBA").copy()


def _puffs(x, *parts):
    """Cloud tops: rows of round bumps. parts are (bumps per tile, height, phase)."""
    return sum(h * np.abs(np.cos(pi * c * x / _BACK_W + p)) for c, h, p in parts)


def _sky_far(y, x):
    """Pale sky, a long white bank of cloud, its foot in shadow."""
    out = np.full(y.shape, ord("V"), np.uint8)
    top = 34 - _puffs(x, (9, 11, 0.3), (23, 7, 1.4), (53, 3, 0.2))
    out[y > top] = ord("W")
    out[y > 40 - _puffs(x, (13, 6, 2.0), (31, 4, 0.7))] = ord("C")
    out[y > 50 - _puffs(x, (17, 3, 1.1))] = ord("V")
    return out


def _sky_near(y, x):
    """Lower, brighter puffs that drift past the far bank."""
    out = np.full(y.shape, ord("."), np.uint8)
    top = 26 - _puffs(x, (7, 10, 0.0), (19, 8, 2.0), (43, 3, 1.0))
    out[y > top] = ord("W")
    out[y > 30 - _puffs(x, (11, 5, 0.9), (29, 3, 0.2))] = ord("C")
    out[y > 34] = ord("D")
    return out


def _mountains(y, x):
    """Faceted far peaks: each ridge is a tent, its sunny slope lighter."""
    t = x / _BACK_W
    out = np.full(y.shape, ord("."), np.uint8)
    for cycles, base, height, lit, shade, off in ((7, 52, 44, "B", "A", 0.13), (11, 60, 34, "A", "v", 0.41)):
        u = (t * cycles + off) % 1
        ridge = base - height * (1 - np.abs(u * 2 - 1))
        ridge = ridge + 3 * np.cos(2 * pi * t * cycles * 5)
        face = u < 0.5
        out[(y > ridge) & face] = ord(lit)
        out[(y > ridge) & ~face] = ord(shade)
    return out


def _hills(y, x):
    edge = 24 + _wave(x, (6, 9, 0.5), (15, 5, 2.0), (37, 1.5, 1.0))
    out = np.full(y.shape, ord("."), np.uint8)
    out[y > edge] = ord("G")
    out[y > edge + 2] = ord("H")
    stripes = (((x // 12) + (y // 8)) % 2 == 0) & (y > edge + 10)
    out[stripes] = ord("I")
    bushes = 10 + _wave(x, (11, 6, 1.0), (29, 5, 0.3), (59, 3, 2.0))
    out[(y > bushes) & (y <= edge + 1) & (bushes < edge)] = ord("H")
    out[(y > bushes) & (y <= bushes + 2) & (bushes < edge)] = ord("G")
    return out


def _lake(frame):
    def paint(y, x):
        out = np.full(y.shape, ord("B"), np.uint8)
        out[y >= 18] = ord("A")
        out[y >= 44] = ord("v")
        # shimmer: dashes, longer and slower toward the viewer, drifting a little frame by frame
        for row in range(2, 64, 3):
            dash, gap = 10 + row // 3, 34 + row * 2
            drift = (row * row * 101 + frame * (2 if row % 2 else -2)) % (dash + gap)
            mask = ((x + drift) % (dash + gap)) < dash
            out[(y == row) & mask] = ord("W" if row < 26 else "C")
        return out
    return paint


@lru_cache(maxsize=1)
def _bands():
    """Far to near: (surface, top y, share of the camera's x it scrolls by)."""
    return (
        (_band(60, _sky_far), 0, 0.05),
        (_band(40, _sky_near), 28, 0.08),
        (_band(64, _mountains), 58, 0.12),
        (_band(60, _hills), 96, 0.2),
    )


@lru_cache(maxsize=1)
def _lakes():
    return tuple(_band(72, _lake(f)) for f in range(_LAKE_FRAMES))


def backdrop(surface, cam_x: float, cam_y: float, frame: int) -> None:
    """The Green Hill sky, peaks, hills and lake, each band sliding at its own share of the camera."""
    surface.fill(_SKY)
    lift = -int(cam_y * 0.012) + 8
    for band, top, rate in _bands():
        off = int(cam_x * rate) % _BACK_W
        surface.blit(band, (-off, top + lift))
        surface.blit(band, (_BACK_W - off, top + lift))
    lake = _lakes()[frame // 8 % _LAKE_FRAMES]
    off = int(cam_x * 0.3) % _BACK_W
    surface.blit(lake, (-off, 152 + lift))
    surface.blit(lake, (_BACK_W - off, 152 + lift))


# ── decor ─────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=4)
def _decor_list(level):
    """(kind, x, foot y, level's y): everything decorative, stood on the ground under where the level puts it."""
    solid = _level_maps(level)[0]
    out = []
    for kind in ("palm", "sunflower", "flower", "totem"):
        for x, y, *_ in level.objects[kind]:
            col = solid[max(y, 0):, min(max(x, 0), level.px_w - 1)]
            out.append((kind, x, y + (int(col.argmax()) if col.any() else 0), y))
    return out


def decor_back(surface, level, cam_x, cam_y, frame) -> None:
    """Palms, swaying flowers, totems and waterfalls: what is on screen, behind the hero."""
    cx, cy = int(cam_x), int(cam_y)
    sw, sh = surface.get_size()
    wf = sprite(f"waterfall_{frame // 4 % 4}")
    for x0, y0, x1, y1 in level.waterfalls:
        if x1 < cx or x0 > cx + sw or y1 < cy or y0 > cy + sh:
            continue
        for y in range(max(y0, cy - (cy - y0) % 16), min(y1, cy + sh), 16):
            for x in range(max(x0, cx - (cx - x0) % 16), min(x1, cx + sw), 16):
                surface.blit(wf, (x - cx, y - cy))
    sway = frame // 24 % 2
    for kind, x, foot, top in _decor_list(level):
        if not (cx - 64 < x < cx + sw + 64 and cy - 96 < foot and top - 64 < cy + sh):
            continue
        if kind == "palm":
            seg = sprite("palm_trunk")
            for y in range(top + 8, foot + 2, 16):
                surface.blit(seg, (x - 6 - cx, y - cy))
            crown = sprite("palm_crown")
            surface.blit(crown, (x - 30 - cx, top - 16 - cy))
        else:
            img = sprite("totem" if kind == "totem" else f"{kind}_{(sway + x // 16) % 2}")
            surface.blit(img, (x - img.get_width() // 2 - cx, foot + 2 - img.get_height() - cy))


# ── text and screens ──────────────────────────────────────────────────────────
@lru_cache(maxsize=256)
def _text(text: str, top, low, scale: int = 1, line=(0, 0, 0)) -> pygame.Surface:
    """Chunky lettering: the 5 x 7 font, one pixel thicker and spaced to suit, two tones top to bottom,
    outlined. Letter by letter, so the thickening never joins two letters up."""
    step = 6 * scale + 2
    w, h = len(text) * step, 7 * scale
    out = pygame.Surface((w + 3, h + 3), pygame.SRCALPHA, 32)
    for i, ch in enumerate(text):
        x = i * step
        ink = nes_text(ch, scale, line)
        for dy in range(3):
            for dx in range(4):
                out.blit(ink, (x + dx, dy))
    cut = h // 2
    for i, ch in enumerate(text):
        hi, lo = nes_text(ch, scale, top), nes_text(ch, scale, low)
        for dx in (1, 2):
            out.blit(hi, (i * step + dx, 1), (0, 0, hi.get_width(), cut))
            out.blit(lo, (i * step + dx, 1 + cut), (0, cut, lo.get_width(), h - cut))
    return out


YELLOW, GOLDEN = (0xFF, 0xFF, 0x00), (0xFF, 0xB6, 0x00)
WHITE, PALE = (0xFF, 0xFF, 0xFF), (0xDB, 0xDB, 0xDB)
RED, DARK_RED = (0xFF, 0x24, 0x00), (0xB6, 0x00, 0x00)


def hud(surface, score: int, frames: int, rings: int, lives: int, flash: bool) -> None:
    """SCORE, TIME and RINGS top left, the lives icon bottom left. RINGS flashes red at zero."""
    seconds = frames // 60
    alarm = rings == 0 and flash
    rows = (("SCORE", f"{score:>6}"), ("TIME", f"{seconds // 60}:{seconds % 60:02d}"), ("RINGS", f"{rings:>3}"))
    for i, (label, value) in enumerate(rows):
        y = 8 + i * 16
        top, low = (RED, DARK_RED) if alarm and label == "RINGS" else (YELLOW, GOLDEN)
        surface.blit(_text(label, top, low), (16, y))
        surface.blit(_text(value, WHITE, PALE), (120 - 8 * len(value), y))
    surface.blit(sprite("icon_life"), (16, 200))
    surface.blit(_text("SONIC", YELLOW, GOLDEN), (34, 198))
    surface.blit(_text(f"*{lives}", WHITE, PALE), (34, 209))


def _slide(t: float, start: float, stop: float) -> float:
    """0 to 1 over start..stop of t, easing out: how far something has slid home."""
    u = min(max((t - start) / (stop - start), 0.0), 1.0)
    return 1 - (1 - u) ** 2


def _act_badge(act: int) -> pygame.Surface:
    """The ACT plate: a yellow-rimmed blue oval with the number on it."""
    s = pygame.Surface((64, 32), pygame.SRCALPHA, 32)
    pygame.draw.ellipse(s, (0, 0, 0), (0, 2, 64, 30))
    pygame.draw.ellipse(s, (0xFF, 0xDB, 0x00), (1, 3, 62, 28))
    pygame.draw.ellipse(s, (0x00, 0x24, 0xB6), (4, 5, 56, 24))
    s.blit(_text(f"ACT {act}", WHITE, PALE), (11, 12))
    return s


def title_card(surface, act: int, t: float) -> None:
    """GREEN HILL slides in from the right, ZONE from the left, the ACT plate from below; all leave at the end."""
    sw, sh = surface.get_size()
    leave = _slide(t, 0.8, 1.0)
    ease = _slide(t, 0.0, 0.2) - leave
    top = _text("GREEN HILL", WHITE, (0xFF, 0xDB, 0x49), 2)
    zone = _text("ZONE", WHITE, (0xFF, 0xDB, 0x49), 2)
    surface.blit(top, (int(sw + 8 - (sw + 8 - (sw - top.get_width()) // 2) * ease), 72))
    surface.blit(zone, (int(-zone.get_width() + (sw // 2 + 36 + zone.get_width()) * ease), 94))
    badge = _act_badge(act)
    surface.blit(badge, (int(sw - 88), int(sh + 8 - (sh + 8 - 120) * ease)))


def results(surface, act, time_bonus, ring_bonus, score, t) -> None:
    """The end of the act: who got through, then the tally rows sliding in one by one."""
    sw = surface.get_width()
    head = _text("SONIC GOT", WHITE, (0xFF, 0xDB, 0x49), 2)
    second = _text("THROUGH", WHITE, (0xFF, 0xDB, 0x49), 2)
    surface.blit(head, (int(sw + 8 - (sw + 8 - (sw - head.get_width()) // 2) * _slide(t, 0.0, 0.2)), 44))
    surface.blit(second, (int(-second.get_width() + ((sw - second.get_width()) // 2 + second.get_width()) * _slide(t, 0.05, 0.25)), 66))
    surface.blit(_act_badge(act), (int(sw - 72 + (1 - _slide(t, 0.1, 0.3)) * 120), 84))
    rows = (("SCORE", score), ("TIME BONUS", time_bonus), ("RING BONUS", ring_bonus))
    for i, (label, value) in enumerate(rows):
        x = int(64 - (1 - _slide(t, 0.2 + i * 0.1, 0.4 + i * 0.1)) * 320)
        surface.blit(_text(label, YELLOW, GOLDEN), (x, 128 + i * 16))
        v = _text(str(value), WHITE, PALE)
        surface.blit(v, (x + 192 - v.get_width(), 128 + i * 16))


@lru_cache(maxsize=1)
def _emblem() -> pygame.Surface:
    """The title's badge: a banded blue disc in a gold ring, a blue ribbon with pointed ends across it."""
    a = _Art(240, 136)
    cx, cy = 120, 64
    a.paint(a.ell(cx, cy, 62, 62), _ramp(a.y, "bcdc", (24.0, 48.0, 82.0)), "K")
    a.paint(a.ell(cx, cy, 62, 62) & ~a.ell(cx, cy, 58, 58), _ramp(a.y, "YTYt", (24.0, 40.0, 96.0)), None)
    ribbon = a.poly([(2, 82), (34, 70), (206, 70), (238, 82), (206, 116), (34, 116)])
    a.paint(ribbon, _ramp(a.y, "cba", (84.0, 104.0)), "K")
    a.paint(a.poly([(34, 72), (206, 72), (206, 74), (34, 74)]), "Y", None)
    a.paint(a.poly([(34, 112), (206, 112), (206, 114), (34, 114)]), "t", None)
    for s in (-1, 1):                                               # the ribbon's stars
        _star(a, cx + s * 100, 92, 7, "W", "t")
    return art_png.to_surface(a.rows(), PAL)


def title_screen(surface, frame) -> None:
    """The word SONIC on a ribbon across the badge, the hero stepping out behind it, PRESS START blinking."""
    sw = surface.get_width()
    backdrop(surface, frame * 2.0, 700.0, frame)
    emblem = _emblem()
    x, y = (sw - emblem.get_width()) // 2, 12
    surface.blit(emblem, (x, y))
    him = pygame.transform.scale2x(sprite(f"wait_{frame // 20 % 2}"))
    surface.blit(him, ((sw - him.get_width()) // 2, y - 2), (0, 0, him.get_width(), 84))   # the ribbon hides his legs
    surface.blit(emblem, (x, y + 72), (0, 72, emblem.get_width(), 64))
    word = _text("SONIC", WHITE, (0xFF, 0xDB, 0x00), 4, (0x00, 0x00, 0x6D))
    surface.blit(word, ((sw - word.get_width()) // 2, y + 80))
    if frame // 30 % 2 == 0:
        press = _text("PRESS START", WHITE, PALE)
        surface.blit(press, ((sw - press.get_width()) // 2, 176))


def all_sprites() -> dict:
    """name -> rows: what art_png.py exports."""
    return dict(SPRITES)
