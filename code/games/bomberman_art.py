"""Bomberman II (NES) look for bomberman_core: hand-drawn 16 x 16 pixel art in the colours the
console shows (sampled from a normal-mode playthrough and snapped to the NES palette), drawn at
2x so one sprite pixel is 2 x 2 world pixels on the 32 px tile. Original drawings in that style,
not ripped sprites. Every sprite is a list of strings, one character per pixel ('.' transparent).
"""
from __future__ import annotations

from functools import lru_cache

import pygame

# Colours as the console shows them (2C02 palette entries in brackets)
PAL = {
    "K": (0, 0, 0),          # $0D outline
    "W": (248, 248, 248),    # $20
    "G": (160, 160, 160),    # $3D light grey
    "g": (120, 120, 120),
    "H": (88, 88, 88),       # $00 frame grey
    "h": (56, 56, 56),       # $2D
    "R": (136, 16, 0),       # $16-ish brick red
    "r": (192, 40, 8),
    "O": (248, 120, 48),     # $27 face, balloon
    "o": (200, 76, 12),
    "B": (16, 152, 248),     # $21 suit blue
    "b": (0, 88, 200),
    "P": (236, 88, 180),     # $25 pom-pom pink
    "Y": (232, 216, 128),    # $37 flame core
    "F": (248, 144, 0),      # $27 flame
    "f": (224, 48, 0),       # $16 flame edge
    "L": (104, 88, 0),       # $18 floor
    "l": (84, 70, 0),
    "C": (60, 188, 252),     # $2C light blue
    "c": (0, 120, 248),
    "E": (76, 208, 32),      # $2A green
    "e": (0, 120, 0),
    "p": (248, 164, 192),    # $35 pig pink
    "S": (188, 188, 188),
    "U": (148, 0, 132),      # purple
    "u": (216, 0, 204),
}
FLOOR = PAL["L"]
SCALE = 2

# ── terrain ──────────────────────────────────────────────────────────────────
PILLAR = [
    "................",
    ".KKKKKKKKKKKKKK.",
    ".KWWWWWWWWWWWWK.",
    ".KWGGGGGGGGGGWK.",
    ".KWGWWWWWWWWGWK.",
    ".KWGWGGGGGGWGWK.",
    ".KWGGGGGGGGGGWK.",
    ".KWWWWWWWWWWWWK.",
    ".KKKKKKKKKKKKKK.",
    ".KGGGGGGGGGGGGK.",
    ".KWWWWWWWWWWWWK.",
    ".KGGGGGGGGGGGGK.",
    ".KgggggggggggggK"[:16],
    ".KKKKKKKKKKKKKK.",
    "..hhhhhhhhhhhhh.",
    "................",
]
BRICK = [
    "KKKKKKKKKKKKKKKK",
    "KrRRRRRKrRRRRRRK",
    "KRRRRRRKRRRRRRRK",
    "KRRRRRRKRRRRRRRK",
    "KKKKKKKKKKKKKKKK",
    "KrRRKrRRRRRRKrRK",
    "KRRRKRRRRRRRKRRK",
    "KRRRKRRRRRRRKRRK",
    "KKKKKKKKKKKKKKKK",
    "KrRRRRRKrRRRRRRK",
    "KRRRRRRKRRRRRRRK",
    "KRRRRRRKRRRRRRRK",
    "KKKKKKKKKKKKKKKK",
    "KrRRKrRRRRRRKrRK",
    "KRRRKRRRRRRRKRRK",
    "KKKKKKKKKKKKKKKK",
]
DOOR = [
    "................",
    ".KKKKKKKKKKKKKK.",
    ".KGGGGGGKGGGGGK.",
    ".KGWWWWGKGWWWWK.",
    ".KGWGGGGKGGGGWK.",
    ".KGWGGGGKGGGGWK.",
    ".KGWGGGGKGGGGWK.",
    ".KGWGGGWKWGGGWK.",
    ".KGWGGGWKWGGGWK.",
    ".KGWGGGGKGGGGWK.",
    ".KGWGGGGKGGGGWK.",
    ".KGWGGGGKGGGGWK.",
    ".KGGGGGGKGGGGGK.",
    ".KggggggKgggggK.",
    ".KKKKKKKKKKKKKK.",
    "................",
]

# ── Bomberman: white helmet, orange face, blue suit ──────────────────────────
_HEAD_FRONT = [
    "...........PP...",
    ".....KKKKKKPPK..",
    "...KKWWWWWWKKK..",
    "..KWWWWWWWWWWK..",
    ".KWWOOOOOOOOWWK.",
    ".KWOOKOOOOKOOWK.",
    ".KWOOKOOOOKOOWK.",
    ".KWOOOOOOOOOOWK.",
    "..KWWWWWWWWWWK..",
]
_HEAD_BACK = [
    "...........PP...",
    ".....KKKKKKPPK..",
    "...KKWWWWWWKKK..",
    "..KWWWWWWWWWWK..",
    ".KWWWWWWWWWWWWK.",
    ".KWWWWWWWWWWWWK.",
    ".KWWWWWWWWWWWWK.",
    ".KWWWWWWWWWWWWK.",
    "..KWWWWWWWWWWK..",
]
_HEAD_SIDE = [               # facing right; mirrored for left
    "....PP..........",
    "...KPPKKKKK.....",
    "...KKWWWWWWKK...",
    "..KWWWWWWWWWWK..",
    ".KWWWWWOOOOOOWK.",
    ".KWWWWOOOKOOOWK.",
    ".KWWWWOOOKOOOWK.",
    ".KWWWWWOOOOOOWK.",
    "..KWWWWWWWWWWK..",
]
_BODY_STAND = [
    "...KKBBBBBBKK...",
    "..KWKBBBBBBKWK..",
    "..KWKBBWWBBKWK..",
    "...KKBBBBBBKK...",
    "....KBBKKBBK....",
    "...KWWWKKWWWK...",
    "...KKKK..KKKK...",
]
_BODY_WALK = [
    "...KKBBBBBBKK...",
    "..KWKBBBBBBKWK..",
    "...KKBBWWBBKWK..",
    "....KBBBBBBKK...",
    "....KBBKKBBBK...",
    "...KWWWK.KWWWK..",
    "...KKKK...KKKK..",
]
_BODY_SIDE = [
    "....KBBBBBBK....",
    "...KBBBBWWKWK...",
    "...KBBBBBBKWK...",
    "....KBBBBBBK....",
    "....KBBKBBK.....",
    "...KWWWKWWWK....",
    "...KKKKKKKKK....",
]
_BODY_SIDE_WALK = [
    "....KBBBBBBK....",
    "...KBBBBWWKWK...",
    "...KBBBBBBKWK...",
    "....KBBBBBBK....",
    "...KBBK.KBBK....",
    "..KWWWK..KWWWK..",
    "..KKKK....KKKK..",
]


def _mirror(rows):
    return [r[::-1] for r in rows]


PLAYER = {   # (facing, frame) -> 16 rows; facing 1 up, 2 right, 3 down, 4 left
    (3, 0): _HEAD_FRONT + _BODY_STAND, (3, 1): _HEAD_FRONT + _BODY_WALK, (3, 2): _HEAD_FRONT + _mirror(_BODY_WALK),
    (1, 0): _HEAD_BACK + _BODY_STAND, (1, 1): _HEAD_BACK + _BODY_WALK, (1, 2): _HEAD_BACK + _mirror(_BODY_WALK),
    (2, 0): _HEAD_SIDE + _BODY_SIDE, (2, 1): _HEAD_SIDE + _BODY_SIDE_WALK, (2, 2): _HEAD_SIDE + _BODY_SIDE,
}
for _k in ((0,), (1,), (2,)):
    PLAYER[(4, _k[0])] = _mirror(PLAYER[(2, _k[0])])
PLAYER_DEATH = [   # the console's death: he blinks white-hot, then crumples
    _HEAD_FRONT + _BODY_STAND,
    [r.replace("B", "W").replace("O", "Y") for r in _HEAD_FRONT + _BODY_STAND],
    ["................"] * 4 + [r.replace("B", "F").replace("O", "f") for r in (_HEAD_FRONT + _BODY_STAND)[::2][:12]],
    ["................"] * 9 + ["....KffffffK....", "...KfFFYYFFfK...", "..KfFYYYYYYFfK..",
                               "...KfFFYYFFfK...", "....KKKKKKKK....", "................", "................"],
]

# ── bombs and flames ─────────────────────────────────────────────────────────
BOMB = [   # three sizes: the bomb throbs while it burns down
    [
        "................",
        "...........KK...",
        "..........KWYK..",
        ".........KK.F...",
        ".....KKKKKK.....",
        "...KKKKKKKKKK...",
        "..KKWWKKKKKKKK..",
        "..KWWKKKKKKKKK..",
        ".KKWKKKKKKKKKKK.",
        ".KKKKKKKKKKKKKK.",
        ".KKKKKKKKKKKKKK.",
        ".KKKKKKKKKKKKKK.",
        "..KKKKKKKKKKKK..",
        "..KKKKKKKKKKKK..",
        "...KKKKKKKKKK...",
        ".....KKKKKK.....",
    ],
    [
        "................",
        "................",
        "..........KYK...",
        ".........KK.....",
        ".....KKKKKK.....",
        "....KKKKKKKK....",
        "...KWWKKKKKKK...",
        "...KWKKKKKKKK...",
        "..KKKKKKKKKKKK..",
        "..KKKKKKKKKKKK..",
        "..KKKKKKKKKKKK..",
        "..KKKKKKKKKKKK..",
        "...KKKKKKKKKK...",
        "...KKKKKKKKKK...",
        "....KKKKKKKK....",
        "................",
    ],
]


def _flame(shape: str, level: int):
    """Flame pieces built from bands: shape is 'c' centre, 'h' / 'v' arm, or an end 'u','d','l','r'.
    level 0 thin .. 2 full."""
    w = (4, 6, 8)[level]                       # band width
    rows = [["."] * 16 for _ in range(16)]

    def band(horizontal: bool, start: int, stop: int):
        for i in range(start, stop):
            for j in range(16):
                d = abs(j - 7.5)
                ch = "Y" if d < w / 4 else "F" if d < w / 2 else "f" if d < w / 2 + 1 else None
                if ch:
                    y, x = (j, i) if horizontal else (i, j)
                    rows[y][x] = ch

    if shape in ("c", "h"):
        band(True, 0, 16)
    if shape in ("c", "v"):
        band(False, 0, 16)
    if shape == "r":
        band(True, 0, 14)
    if shape == "l":
        band(True, 2, 16)
    if shape == "d":
        band(False, 0, 14)
    if shape == "u":
        band(False, 2, 16)
    return ["".join(r) for r in rows]


# ── enemies ──────────────────────────────────────────────────────────────────
BALLOON = [
    [
        "................",
        ".....KKKKKK.....",
        "...KKOOOOOOKK...",
        "..KOOOOOOOOOOK..",
        ".KOOOOOOOOOOOOK.",
        ".KOOKKOOOOKKOOK.",
        "KOOOKKOOOOKKOOOK",
        "KOOOKWOOOOKWOOOK",
        "KOOOOOOOOOOOOOOK",
        ".KOOOOOOOOOOOOK.",
        ".KOOOOOKKOOOOOK.",
        "..KOOOOOOOOOOK..",
        "...KKOOOOOOKK...",
        ".....KKOOKK.....",
        "......KWWK......",
        ".......KK.......",
    ],
    [
        "................",
        "................",
        ".....KKKKKK.....",
        "...KKOOOOOOKK...",
        "..KOOOOOOOOOOK..",
        ".KOOKKOOOOKKOOK.",
        "KOOOKKOOOOKKOOOK",
        "KOOOKWOOOOKWOOOK",
        "KOOOOOOOOOOOOOOK",
        "KOOOOOOOOOOOOOOK",
        ".KOOOOOKKOOOOOK.",
        "..KOOOOOOOOOOK..",
        "...KKOOOOOOKK...",
        ".....KKOOKK.....",
        "......KWWK......",
        ".......KK.......",
    ],
]


@lru_cache(maxsize=64)
def frame_tile(left: bool, right: bool, up: bool, down: bool, vent: bool) -> tuple:
    """A piece of the arena's metal frame. left/right/up/down: which sides face the arena (the
    bevel and black lip go on those sides); vent: a grille slot on a top or bottom edge."""
    rows = []
    for y in range(16):
        row = []
        for x in range(16):
            # distance to the nearest side that faces the arena (16 = none)
            d = min([15 - x] if right else [16], default=16)
            d = min(d, *([x] if left else [16]), *([15 - y] if down else [16]), *([y] if up else [16]))
            if d == 0:
                ch = "K"                                        # the lip onto the floor
            elif d == 1:
                ch = "W"
            elif d == 2:
                ch = "G"
            elif d <= 13:
                ch = "h" if (x + y) % 4 == 0 else "H"           # hatched plate
            else:
                ch = "G"
            if vent and (up or down) and 4 <= x <= 11 and 5 <= y <= 10 and x % 2 == 0:
                ch = "K"                                        # a vent slot
            row.append(ch)
        rows.append("".join(row))
    return tuple(rows)


@lru_cache(maxsize=512)
def sprite(rows: tuple, flip: bool = False, scale: int = SCALE) -> pygame.Surface:
    """One bitmap as a surface, transparent where '.', each pixel scale x scale."""
    h, w = len(rows), max(len(r) for r in rows)
    s = pygame.Surface((w * scale, h * scale), pygame.SRCALPHA)
    for y, row in enumerate(rows):
        for x, ch in enumerate(row[::-1] if flip else row):
            if ch != "." and ch in PAL:
                s.fill(PAL[ch], (x * scale, y * scale, scale, scale))
    return s


def get(rows, flip: bool = False) -> pygame.Surface:
    return sprite(tuple(rows), flip)


BURNING_BRICK = [   # a brick caught by the flames, flickering until they die
    [r.replace("R", "F").replace("r", "Y").replace("K", "f") for r in BRICK],
    [r.replace("R", "f").replace("r", "F").replace("K", "R") for r in BRICK],
]

DEATH_FACE = [   # an enemy caught in a blast: a startled face before it vanishes
    "................",
    ".....KKKKKK.....",
    "...KKOOOOOOKK...",
    "..KOOOOOOOOOOK..",
    ".KOKKKOOOOKKKOK.",
    ".KOKWKOOOOKWKOK.",
    "KOOKKKOOOOKKKOOK",
    "KOOOOOOOOOOOOOOK",
    "KOOOOOKKKKOOOOOK",
    ".KOOOKKKKKKOOOK.",
    ".KOOOKKKKKKOOOK.",
    "..KOOOKKKKOOOK..",
    "...KKOOOOOOKK...",
    ".....KKKKKK.....",
    "................",
    "................",
]

# Items under the bricks (filled in from the Bomberman II reference): glyph -> icon
_ITEM_FRAME = [
    "KKKKKKKKKKKKKKKK",
    "KWWWWWWWWWWWWWWK",
    "KW............WK",
] + ["KW............WK"] * 10 + [
    "KW............WK",
    "KWWWWWWWWWWWWWWK",
    "KKKKKKKKKKKKKKKK",
]


def _icon(picture: list, back: str = "f") -> list:
    """An item: the console's bordered panel with a picture on it."""
    rows = [list(r) for r in _ITEM_FRAME]
    for y in range(3, 13):
        for x in range(2, 14):
            rows[y][x] = back
    for y, line in enumerate(picture):
        for x, ch in enumerate(line):
            if ch != ".":
                rows[3 + y][2 + x] = ch
    return ["".join(r) for r in rows]


ITEM_ICONS = {
    "c": _icon(["............", "........KY..", ".......K....", "....KKKK....", "...KKKKKK...", "..KWKKKKKK..",
                "..KKKKKKKK..", "..KKKKKKKK..", "...KKKKKK...", "....KKKK...."], "R"),          # bomb up
    "f": _icon(["............", ".....FF.....", "....FYYF....", "...FYYYYF...", "..FYYWWYYF..", "..FYWWWWYF..",
                "..FYWWWWYF..", "...FYYYYF...", "....FFFF....", "............"], "R"),          # fire up
    "s": _icon(["............", "...KKKK.....", "...KCCK.....", "...KCCKKKK..", "...KCCCCCCK.", "...KCCCCCCK.",
                "...KKKKKKKK.", "...K.K..K.K.", "....K....K..", "............"], "R"),          # speed (skate)
}
BOYON = [   # blue blob with eyes on stalks; it stretches tall and squashes flat
    [
        "................",
        "..KKK......KKK..",
        ".KWWWK....KWWWK.",
        ".KWKWKKKKKKWKWK.",
        ".KWWWBBBBBBWWWK.",
        "..KKBBBBBBBBKK..",
        "..KBBBBKKBBBBK..",
        "..KBBBBKWKBBBBK.",
        "..KBBBBKWKBBBK..",
        ".KBBBBBKKBBBBBK.",
        ".KBBBBKWKBBBBBK.",
        ".KBBBBKWKBBBBBK.",
        ".KBBBBBKKBBBBBK.",
        "KBBBBBBBBBBBBBBK",
        "KBBBBBBBBBBBBBBK",
        ".KKKKKKKKKKKKKK.",
    ],
    [
        "................",
        "................",
        "................",
        "................",
        "................",
        "................",
        "..KKK......KKK..",
        ".KWWWK....KWWWK.",
        ".KWKWKKKKKKWKWK.",
        ".KWWWBBBBBBWWWK.",
        ".KKBBBBBBBBBBKK.",
        "KBBBBBKWWKBBBBBK",
        "KBBBBBBKKBBBBBBK",
        "KBBBBBBBBBBBBBBK",
        "KBBBBBBBBBBBBBBK",
        ".KKKKKKKKKKKKKK.",
    ],
]
TERUPYO = [   # green, a white grin across its face, eyes on top
    [
        "................",
        "...KKKKKKKKKK...",
        "..KEWWEEEEWWEK..",
        "..KWKWEEEEWKWK..",
        "..KEWWEEEEWWEK..",
        "..KEEEEEEEEEEK..",
        "..KWWWWWWWWWWK..",
        "..KWKKKKKKKKWK..",
        "..KWWWWWWWWWWK..",
        ".KEEEEEEEEEEEEK.",
        "KEEEEeEEEEeEEEEK",
        "KEEEeEEEEEEeEEEK",
        "KEEEEEEEEEEEEEEK",
        ".KEEEK.KK.KEEEK.",
        ".KKKK......KKKK.",
        "................",
    ],
    [
        "................",
        "...KKKKKKKKKK...",
        "..KEWWEEEEWWEK..",
        "..KWKWEEEEWKWK..",
        "..KEWWEEEEWWEK..",
        "..KEEEEEEEEEEK..",
        "..KWWWWWWWWWWK..",
        "..KWKKKKKKKKWK..",
        "..KWWWWWWWWWWK..",
        ".KEEEEEEEEEEEEK.",
        "KEEEEeEEEEeEEEEK",
        "KEEEeEEEEEEeEEEK",
        "KEEEEEEEEEEEEEEK",
        "..KEEK.KK.KEEK..",
        "..KKKK....KKKK..",
        "................",
    ],
]
NAGACHAM = [   # a white ghost under an orange cap
    [
        "................",
        ".....KKKKKK.....",
        "...KKOOOOOOKK...",
        "..KOOOOOOOOOOK..",
        ".KOOOOOOOOOOOOK.",
        ".KKKKKKKKKKKKKK.",
        ".KWWWWWWWWWWWWK.",
        "KWWKKWWWWWWKKWWK",
        "KWWKKWWWWWWKKWWK",
        "KWWWWWWWWWWWWWWK",
        "KWWWWWWKKWWWWWWK",
        "KWWWWWWWWWWWWWWK",
        "KWWWWWWWWWWWWWWK",
        "KWWWKWWWWWWKWWWK",
        "KWWK.KWWWWK.KWWK",
        ".KK...KKKK...KK.",
    ],
    [
        "................",
        ".....KKKKKK.....",
        "...KKOOOOOOKK...",
        "..KOOOOOOOOOOK..",
        ".KOOOOOOOOOOOOK.",
        ".KKKKKKKKKKKKKK.",
        ".KWWWWWWWWWWWWK.",
        "KWWKKWWWWWWKKWWK",
        "KWWKKWWWWWWKKWWK",
        "KWWWWWWWWWWWWWWK",
        "KWWWWWWKKWWWWWWK",
        "KWWWWWWWWWWWWWWK",
        "KWWWWWWWWWWWWWWK",
        "KWWWWKWWWWKWWWWK",
        ".KWWK.KWWK.KWWK.",
        "..KK...KK...KK..",
    ],
]
PASS = [   # an orange bear that hunts you
    [
        "................",
        "..KKK......KKK..",
        ".KOOOK....KOOOK.",
        ".KOoOKKKKKKOoOK.",
        "..KOOOOOOOOOOK..",
        ".KOOOOOOOOOOOOK.",
        ".KOOKKOOOOKKOOK.",
        ".KOOKWOOOOKWOOK.",
        ".KOOOOOKKOOOOOK.",
        ".KOOOOWWWWOOOOK.",
        "..KOOOOOOOOOOK..",
        "...KKOOOOOOKK...",
        "..KOOKKKKKKOOK..",
        "..KOOK....KOOK..",
        "..KKKK....KKKK..",
        "................",
    ],
]
PONTAN = [   # a spinning coin-like face out of a bombed door
    [r.replace("O", "Y").replace("o", "F") for r in BALLOON[0]],
    [r.replace("O", "F").replace("o", "f") for r in BALLOON[1]],
]
POMORI = [[r.replace("O", "C").replace("o", "c") for r in PASS[0]]]   # a slow blue cat

ENEMIES = {"E": BALLOON, "b": BOYON, "t": TERUPYO, "n": NAGACHAM, "K": PASS, "o": PONTAN, "p": POMORI,
           "k": [[r.replace("O", "C") for r in f] for f in BALLOON],
           "M": [[r.replace("O", "p") for r in f] for f in BALLOON],
           "B": [[r.replace("O", "U") for r in f] for f in BALLOON]}
# an enemy caught in a blast: its own colours behind the startled face
ENEMY_DEATH = {k: [r.replace("O", c) for r in DEATH_FACE] for k, c in
               (("E", "O"), ("b", "B"), ("t", "E"), ("n", "W"), ("K", "O"), ("o", "Y"), ("p", "C"),
                ("k", "C"), ("M", "p"), ("B", "U"))}

ITEM_ICONS.update({
    "r": _icon(["............", "..KK..KK....", ".KffKKffK...", ".KfffffffK..", ".KffffffK.KK", "..KffffK.KYK",
                "...KffK.KKKK", "....KK.KKKKK", "........KKK.", "............"], "C"),    # remote: heart and bomb
    "w": _icon(["............", "KKKKKKKKKKKK", "RRRKRRRRKRRR", "KKKKKKKKKKKK", "RKRRRRKRRRRK", "KKKKKKKKKKKK",
                "RRRKRRRRKRRR", "KKKKKKKKKKKK", "RKRRRRKRRRRK", "KKKKKKKKKKKK"], "G"),    # wall pass
    "x": _icon(["............", ".......KY...", "..WW..K.....", "....KKKK....", "WW.KKKKKK...", "..KKKKKKKK..",
                "WW.KKKKKK...", "....KKKK....", "..WW........", "............"], "C"),    # bomb pass
    "v": _icon(["............", "...BB..BB...", "..BBBBBBBB..", "..BWBBBBWB..", "..BBBBBBBB..", "..BBBWWBBB..",
                "..BBBBBBBB..", "..BBBBBBBB..", "...BBBBBB...", "............"], "R"),    # vest
    "d": _icon(["............", ".....YY.....", "...YYYYYY...", "..YY.YY.....", "...YYYYY....", ".....YY.YY..",
                "..YYYYYYY...", ".....YY.....", "............", "............"], "E"),    # dollar
    "q": _icon(["............", "...WWWWWW...", "..WW....WW..", "........WW..", ".....WWWW...", ".....WW.....",
                ".....WW.....", "............", ".....WW.....", "............"], "U"),    # mystery
})
