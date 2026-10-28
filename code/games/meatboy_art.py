"""Super Meat Boy's World 1 (The Forest) look for meatboy_core, drawn in the colours the game shows
(sampled from a walkthrough of 1-1 Hello World): a misty sky with rock and pine silhouettes, dark
rock under a moss cap, Meat Boy's red cube and Bandage Girl in pink. Original drawings, no assets.
"""
from __future__ import annotations

import math
import random
from functools import lru_cache

import pygame

SKY_TOP, SKY_LOW = (196, 210, 214), (178, 196, 200)
MIST = [(168, 186, 188), (150, 170, 172), (128, 150, 150)]     # far to near silhouettes
ROCK, ROCK_HI, ROCK_LO = (44, 36, 28), (76, 58, 48), (30, 24, 18)
MOSS, MOSS_HI, MOSS_LO = (50, 62, 26), (70, 88, 36), (34, 44, 16)
CRUMBLE, CRUMBLE_LO = (176, 150, 112), (120, 98, 70)
SPIKE, SPIKE_HI = (120, 124, 130), (200, 204, 210)
FIRE_LO, FIRE, FIRE_HI = (196, 36, 14), (240, 112, 24), (255, 214, 80)
MEAT, MEAT_HI, MEAT_LO = (190, 32, 26), (226, 70, 56), (120, 14, 12)
BLOOD = (150, 20, 18)
GIRL, GIRL_LO, BANDAGE = (232, 160, 168), (190, 116, 126), (246, 236, 214)
INK = (16, 10, 8)


@lru_cache(maxsize=8)
def backdrop(w: int, h: int) -> pygame.Surface:
    """The sky and three layers of misty rocks and pines, made once per window size."""
    s = pygame.Surface((w, h))
    for y in range(h):
        t = y / max(1, h - 1)
        s.fill(tuple(int(a + (b - a) * t) for a, b in zip(SKY_TOP, SKY_LOW)), (0, y, w, 1))
    rng = random.Random(7)
    for i, col in enumerate(MIST):
        base = int(h * (0.55 + 0.13 * i))
        x = -40
        while x < w + 40:
            if rng.random() < 0.5:                                  # a pine
                th, tw = rng.randint(60, 140) - 20 * (2 - i), rng.randint(18, 30)
                for k in range(4):
                    yy = base - th + k * th // 4
                    pygame.draw.polygon(s, col, [(x, yy), (x - tw // 2 - k * 3, yy + th // 3), (x + tw // 2 + k * 3, yy + th // 3)])
                pygame.draw.rect(s, col, (x - 2, base - th // 4, 5, th // 4))
            else:                                                   # a rock
                rw, rh = rng.randint(40, 110), rng.randint(30, 90)
                pygame.draw.ellipse(s, col, (x - rw // 2, base - rh, rw, rh * 2))
            x += rng.randint(30, 80)
        pygame.draw.rect(s, col, (0, base, w, h - base))
    return s


@lru_cache(maxsize=4096)
def ground(ts: int, top: bool, left: bool, right: bool, bottom: bool, seed: int) -> pygame.Surface:
    """A rock tile; top/left/right/bottom: that side faces open air (moss on top, a dark rim elsewhere)."""
    s = pygame.Surface((ts, ts))
    s.fill(ROCK)
    rng = random.Random(seed)
    for _ in range(2):                                              # brown boulders in the rock
        r = rng.randint(ts // 5, ts // 3)
        cx, cy = rng.randint(0, ts), rng.randint(0, ts)
        pygame.draw.circle(s, ROCK_HI, (cx, cy), r)
        pygame.draw.circle(s, ROCK_LO, (cx, cy), r, 2)
    rim = max(2, ts // 10)
    for side, rect in ((left, (0, 0, rim, ts)), (right, (ts - rim, 0, rim, ts)), (bottom, (0, ts - rim, ts, rim))):
        if side:
            s.fill(ROCK_LO, rect)
    if top:                                                         # the moss cap and its blades
        cap = ts // 3
        s.fill(MOSS, (0, 0, ts, cap))
        s.fill(MOSS_LO, (0, cap, ts, 2))
        for x in range(1, ts, 4):
            hh = rng.randint(2, 5)
            s.fill(MOSS_HI, (x, 0, 2, hh))
    return s


@lru_cache(maxsize=8)
def crumble(ts: int) -> pygame.Surface:
    s = pygame.Surface((ts, ts))
    s.fill(CRUMBLE)
    pygame.draw.rect(s, CRUMBLE_LO, (0, 0, ts, ts), 2)
    for a, b in (((4, 6), (14, 15)), ((14, 15), (10, 26)), ((14, 15), (27, 12)), ((22, 22), (28, 29))):
        pygame.draw.line(s, CRUMBLE_LO, (a[0] * ts // 32, a[1] * ts // 32), (b[0] * ts // 32, b[1] * ts // 32), 2)
    return s


@lru_cache(maxsize=8)
def spikes(ts: int) -> pygame.Surface:
    s = pygame.Surface((ts, ts), pygame.SRCALPHA)
    n = 3
    for i in range(n):
        x0, w = i * ts // n, ts // n
        pygame.draw.polygon(s, SPIKE, [(x0, ts), (x0 + w // 2, ts // 4), (x0 + w, ts)])
        pygame.draw.line(s, SPIKE_HI, (x0 + w // 2, ts // 4), (x0 + w // 2 - 2, ts - 2), 1)
    return s


@lru_cache(maxsize=32)
def fire(ts: int, phase: int, top: bool) -> pygame.Surface:
    """Fire (1-11 Fired): a wall of flame. A tile under more fire is solid flame; the top tile throws
    tongues half a tile up, which kill too. phase 0-3 flickers it."""
    h = ts + (ts // 2 if top else 0)
    s = pygame.Surface((ts, h), pygame.SRCALPHA)
    s.fill(FIRE_LO, (0, h - ts, ts, ts))
    if not top:
        pygame.draw.rect(s, FIRE, (0, 0, ts, ts), 0)
        for k in range(3):                                         # embers rising through the flame
            pygame.draw.circle(s, FIRE_HI, ((k * 11 + phase * 7) % ts, (k * 13 + phase * 9) % ts), 3)
        return s
    rng = random.Random(phase)
    for layer, col, scale in ((0, FIRE_LO, 1.0), (1, FIRE, 0.8), (2, FIRE_HI, 0.5)):
        for i in range(3):                                         # three tongues per tile, swaying with the phase
            w = ts / 3
            x0 = i * w - w * 0.25
            tip = (1 - scale) * h + rng.randint(0, ts // 4)
            sway = rng.randint(-ts // 8, ts // 8)
            pygame.draw.polygon(s, col, [(x0, h), (x0 + w * 0.75 + sway, tip), (x0 + w * 1.5, h)])
    return s


def _cube(s: pygame.Surface, rect: pygame.Rect, body, lo, hi) -> None:
    pygame.draw.rect(s, INK, rect.inflate(4, 4), border_radius=5)
    pygame.draw.rect(s, body, rect, border_radius=4)
    pygame.draw.rect(s, lo, (rect.x, rect.bottom - 4, rect.w, 4), border_bottom_left_radius=4, border_bottom_right_radius=4)
    pygame.draw.rect(s, hi, (rect.x + 3, rect.y + 2, rect.w - 6, 2))


@lru_cache(maxsize=64)
def meat_boy(w: int, h: int, right: bool, pose: int) -> pygame.Surface:
    """Meat Boy: a red cube with black eyes, a toothy grin and stubby legs. pose 0 stand, 1/2 run,
    3 in the air, 4 on a wall."""
    s = pygame.Surface((w + 8, h + 8), pygame.SRCALPHA)
    legs = 5
    body = pygame.Rect(4, 4, w, h - legs)
    if pose == 3:
        body.inflate_ip(-2, 2)
    _cube(s, body, MEAT, MEAT_LO, MEAT_HI)
    ex = body.centerx + (3 if right else -3)
    for dx in (-5, 3):                                              # eyes, looking where he goes
        pygame.draw.rect(s, INK, (ex + dx, body.y + 5, 3, 5))
    my = body.y + body.h - 9
    pygame.draw.rect(s, INK, (ex - 5, my, 11, 4))                   # the grin
    pygame.draw.rect(s, (240, 236, 226), (ex - 4, my, 9, 2))
    lx = [(-5, 2), (-7, 4), (-3, 0), (-6, 3), (-5, 2)][pose]
    for x in lx:
        pygame.draw.rect(s, INK, (body.centerx + x * (1 if right else -1) - 2 + (0 if x < 0 else 2), body.bottom, 4, legs + 2))
    return s


@lru_cache(maxsize=8)
def bandage_girl(ts: int) -> pygame.Surface:
    """Bandage Girl: pink cube, a bandage across her head, a white flower, pigtail bumps."""
    size = int(ts * 0.8)
    s = pygame.Surface((size + 8, size + 8), pygame.SRCALPHA)
    body = pygame.Rect(4, 6, size, size - 4)
    _cube(s, body, GIRL, GIRL_LO, (246, 196, 202))
    pygame.draw.rect(s, BANDAGE, (body.x, body.y + 3, body.w, 4))
    pygame.draw.rect(s, (200, 186, 160), (body.x + body.w // 2 - 1, body.y + 3, 3, 4))
    for dx in (-5, 3):
        pygame.draw.rect(s, INK, (body.centerx + dx, body.y + 10, 3, 4))
    pygame.draw.rect(s, INK, (body.centerx - 3, body.bottom - 7, 7, 2))
    fx, fy = body.right - 3, body.y + 1                            # the flower
    for a in range(5):
        pygame.draw.circle(s, (250, 250, 250), (fx + int(3 * math.cos(a * 1.26)), fy + int(3 * math.sin(a * 1.26))), 2)
    pygame.draw.circle(s, (240, 200, 60), (fx, fy), 1)
    return s


def timer_sign(surface: pygame.Surface, seconds: float) -> None:
    """The game's timer: a white sign hung on two ropes at the top left, the attempt's seconds in red."""
    from .modules.System.nes_font import nes_text      # a bitmap font: no font object to outlive pygame.quit()
    board = pygame.Rect(56, 18, 92, 30)
    for x0, x1 in ((24, board.left + 6), (180, board.right - 6)):
        pygame.draw.line(surface, INK, (x0, 0), (x1, board.top), 3)
    pygame.draw.rect(surface, INK, board.inflate(6, 6), border_radius=4)
    pygame.draw.rect(surface, (244, 242, 236), board, border_radius=3)
    text = nes_text(f"{seconds:.2f}", 2, (190, 24, 24))
    surface.blit(text, text.get_rect(center=board.center))


def lives(surface: pygame.Surface, n: int) -> None:
    """The study's lives under the timer sign: Meat Boy and the count (the game itself has none)."""
    from .modules.System.nes_font import nes_text
    plate = pygame.Surface((96, 38), pygame.SRCALPHA)
    pygame.draw.rect(plate, (16, 10, 8, 170), plate.get_rect(), border_radius=6)
    surface.blit(plate, (52, 56))
    surface.blit(meat_boy(24, 24, True, 0), (56, 60))
    surface.blit(nes_text(f"*{n}", 3, (244, 242, 236), shadow=INK), (92, 66))   # "*" is the font's times sign
