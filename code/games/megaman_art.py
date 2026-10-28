"""Mega Man (NES) look for megaman_core when the ripped art (megaman-data/) is not there: original
pixel drawings in the Cut Man stage's colours (sampled from a playthrough), not ripped sprites.

Same interface as megaman_assets.Assets: sprite(name, flip_x, flip_y), backdrop(...), play(events).
Every frame has the size megaman_assets.FRAMES gives it, so the core's anchors fit; like the sheets,
Mega Man and the enemies face left and Cut Man faces right (his walk frames face left).
A sprite is a list of strings, one character per NES pixel ('.' transparent).
"""
from __future__ import annotations

import pygame

from . import art_png
from .megaman_assets import FRAMES, terrain

PAL = {
    "K": (0, 0, 0),
    "W": (252, 252, 252),
    "B": (0, 112, 236),      # Mega Man's blue
    "C": (60, 188, 252),     # his light blue
    "S": (252, 216, 168),    # skin
    "R": (216, 40, 96),      # Cut Man's red
    "P": (252, 116, 180),    # his pink; the Big Eye, the Beak
    "M": (168, 0, 80),       # dark magenta
    "G": (168, 168, 168),    # greys
    "g": (108, 108, 108),
    "L": (228, 228, 228),
    "E": (76, 208, 132),     # brick green
    "e": (0, 136, 56),
    "D": (136, 20, 0),       # girder red
    "O": (248, 120, 48),     # orange
    "Y": (252, 228, 160),    # pale yellow: energy, shots
    "U": (60, 188, 252),
}
SKY = (68, 200, 216)

def _pad(rows, w):
    return [r.ljust(w, ".")[:w] for r in rows]


def _shift(rows, dx, w):
    return [("." * dx + r).ljust(w, ".")[:w] for r in rows]


# ── Mega Man ─────────────────────────────────────────────────────────────────
# Drawn facing right, body centred on column 12 of 24, then mirrored: the sheets face left.
# Head (rows 0-11), torso (12-17) and legs (18-23) meet on the same outline columns.
_HEAD = [
    ".........KKKKKK.........",
    ".......KKCCCCCCKK.......",
    "......KCCCCCCBBBBK......",
    ".....KCCCCCBBBBBBBK.....",
    ".....KCCCBBBBBBBBBK.....",
    "....KBCCBBBKKKKKKBBK....",
    "....KBBBBBKSSSSSSKBK....",
    "....KBCCBKSSSSWWKSKK....",
    "....KBCCBKSSSSWKKSK.....",
    "....KBBBBKSSSSSSSSK.....",
    ".....KBBBKSSSKKKSSK.....",
    "......KKKKKSSSSSSK......",
]
_HEAD_BLINK = _HEAD[:7] + [
    "....KBCCBKSSSSSSSSKK....",
    "....KBCCBKSSSSKKKSK.....",
] + _HEAD[9:]
_HEAD_OUCH = _HEAD[:7] + [         # eyes squeezed shut, mouth open
    "....KBCCBKSSSKSSKSKK....",
    "....KBCCBKSSSSKKSSK.....",
    "....KBBBBKSSSSSSSSK.....",
    ".....KBBBKSSSKKKKSK.....",
    "......KKKKKSSKWWKK......",
]
_TORSO = [                         # arms at his sides
    ".......KKCCKKKKKKK......",
    ".....KKBCCCCCCCCKKK.....",
    "....KBBKCCCCCCCCKBBK....",
    "....KBBKCCCCCCCCKBBK....",
    "....KCCKBBBBBBBBKCCK....",
    ".....KKKBBBBBBBBKKK.....",
]
_TORSO_GUN = [                     # the front arm out as the buster
    ".......KKCCKKKKKKK..........",
    ".....KKBCCCCCCCCKKKKKKKKKK..",
    "....KBBKCCCCCCCCBBBBBBBBBCK.",
    "....KBBKCCCCCCCCBBBBBBBBBCK.",
    "....KCCKBBBBBBBBKKKKKKKKKK..",
    ".....KKKBBBBBBBBK...........",
]
_LEGS = {
    "stand": [
        ".......KCCCKKCCCK.......",
        "......KCCCK..KCCCK......",
        ".....KBBBBK..KBBBBK.....",
        "....KBBBBBK..KBBBBBK....",
        "....KBBBBBK..KBBBBBK....",
        "....KKKKKKK..KKKKKKK....",
    ],
    "inch": [
        "........KCCCKCCCK.......",
        ".......KCCCKKCCCK.......",
        "......KBBBBKKBBBBK......",
        ".....KBBBBBKKBBBBBK.....",
        ".....KBBBBBKKBBBBBK.....",
        ".....KKKKKKKKKKKKKK.....",
    ],
    "pass": [
        ".........KCCCCK.........",
        ".........KCCCCK.........",
        "........KBBBBBK.........",
        "........KBBBBBBK........",
        "........KBBBBBBK........",
        "........KKKKKKKK........",
    ],
    "step": [
        ".......KCCCKKCCK........",
        "......KCCCK.KCCCK.......",
        ".....KBBBK...KBBBK......",
        "....KBBBBK...KBBBBK.....",
        "....KBBBBK....KBBBK.....",
        "....KKKKKK....KKKKK.....",
    ],
}
_STRIDE = [                        # 32 wide, centred on 16: the long step
    "..........KKCCCKKCCCKK..........",
    "........KKCCCK....KCCCKK........",
    "......KKBBBBK......KBBBBKK......",
    "....KKBBBBBK........KBBBBBKK....",
    "....KBBBBBK..........KBBBBBK....",
    "....KKKKKK............KKKKKK....",
]


def _put(canvas, part, x, y):
    """Paint part onto canvas (lists of strings) at (x, y); '.' in part leaves the canvas alone."""
    out = list(canvas)
    for j, row in enumerate(part):
        line = list(out[y + j])
        for i, ch in enumerate(row):
            if ch != "." and 0 <= x + i < len(line):
                line[x + i] = ch
        out[y + j] = "".join(line)
    return out


def _blank(w, h):
    return ["." * w] * h


def _man(w, h, x, head=_HEAD, torso=_TORSO, legs=_LEGS["stand"], top=None):
    """Head, torso and legs stacked from the bottom of a w x h frame, the 24-wide body at column x."""
    y = h - 24 if top is None else top
    c = _put(_blank(w, h), legs, x if len(legs[0]) <= 24 else 0, y + 18)
    c = _put(c, torso, x, y + 12)
    return _put(c, head, x, y)


_RIGHT = {
    "mm_idle": _man(24, 24, 0),
    "mm_blink": _man(24, 24, 0, head=_HEAD_BLINK),
    "mm_inch": _man(24, 24, 0, legs=_LEGS["inch"]),
    "mm_walk0": _man(32, 24, 4, legs=_STRIDE),
    "mm_walk1": _man(24, 24, 0, legs=_LEGS["pass"]),
    "mm_walk2": _man(24, 24, 0, legs=_LEGS["step"]),
    # shooting: the body's centre 12 px from the back edge (the core anchors it there), the gun reaching on
    "mm_idle_shoot": _man(32, 24, 0, torso=_TORSO_GUN),
    "mm_walk1_shoot": _man(32, 24, 0, torso=_TORSO_GUN, legs=_LEGS["pass"]),
    "mm_walk2_shoot": _man(32, 24, 0, torso=_TORSO_GUN, legs=_LEGS["step"]),
}
_RIGHT["mm_walk0_shoot"] = _put(_put(_blank(32, 24), _STRIDE, -4, 18),
                                _put(_put(_blank(28, 18), _TORSO_GUN, 0, 12), _HEAD, 0, 0), 0, 0)

_ARM_UP = [                        # a raised arm, fist on top
    ".KK.",
    "KCCK",
    "KCCK",
    "KBBK",
    "KBBK",
    "KBBK",
    "KBBK",
    "KBBK",
]
_TORSO_AIR = [                     # the back arm raised (drawn on separately), the front fist forward
    "..KBBKKKCCKKKKKKK.......",
    "..KBBKKCCCCCCCCCKKKK....",
    "...KBBKCCCCCCCCCBBBCK...",
    "....KKKCCCCCCCCCKKKK....",
    ".....KKBBBBBBBBBBKK.....",
    "......KBBBBBBBBBBK......",
]
_TORSO_UP = [                      # both arms raised: the shoulders run into them
    "..KBBKKKCCKKKKKKKKKBBK..",
    "..KBBKKCCCCCCCCCCKKBBK..",
    "...KBBKCCCCCCCCCCKBBK...",
    "....KKKCCCCCCCCCCKKK....",
    ".....KKBBBBBBBBBBKK.....",
    "......KBBBBBBBBBBK......",
]
_LEGS_AIR = [                      # the back leg hangs, the front knee comes up
    "......KCCCKKKKCCCK......",
    ".....KCCCK...KCCCCK.....",
    ".....KCCCK....KCCCKKKK..",
    "....KBBBBK.....KBBBBBBK.",
    "....KBBBBK.....KBBBBBBK.",
    "...KBBBBBK.....KKKKKKKK.",
    "...KBBBBBK..............",
    "...KKKKKKK..............",
]


def _jump(head=_HEAD, torso=_TORSO_AIR, both_arms=False):
    c = _put(_blank(32, 32), _LEGS_AIR, 4, 24)
    c = _put(c, torso, 4, 18)
    c = _put(c, _ARM_UP, 5, 10)
    if both_arms:
        c = _put(c, [r[::-1] for r in _ARM_UP], 23, 10)
    return _put(c, head, 4, 6)


_RIGHT["mm_jump"] = _jump()
_RIGHT["mm_jump_shoot"] = _jump(torso=_TORSO_GUN)
_RIGHT["mm_hurt"] = _jump(head=_HEAD_OUCH, torso=_TORSO_UP, both_arms=True)

_CLIMB = [                         # 16 wide, his back, the right hand up on the rung
    "...........KK...",
    "..........KCCK..",
    "....KKKKKKKCCK..",
    "..KKBBCCBBKBBK..",
    ".KBBBBCCBBBKBK..",
    ".KBBBBCCBBBKBK..",
    ".KCCBBCCBBCKBK..",
    ".KCCBBBBBBCKBK..",
    ".KBBBBBBBBBKBK..",
    "..KBBBBBBBKKBK..",
    "...KKKKKKKKBBK..",
    "..KBKCCCCCCKK...",
    ".KBBKCCCCCCCK...",
    ".KBBKCCCCCCCK...",
    ".KCCKBBBBBBBK...",
    "..KKKBBBBBBK....",
    "....KCCCKCCCK...",
    "....KCCCKCCCK...",
    "....KCCK.KCCK...",
    "....KBBK.KCCK...",
    "...KBBBK.KBBK...",
    "...KBBBK.KBBBK..",
    "...KKKKK.KKKKK..",
]
_RIGHT["mm_climb0"] = _put(_blank(16, 32), _CLIMB, 0, 9)
_RIGHT["mm_climb1"] = [r[::-1] for r in _RIGHT["mm_climb0"]]
# shooting from a ladder: side on, the gun out, centred 8 px from the back edge
_RIGHT["mm_climb_shoot"] = _put(_put(_put(_blank(24, 32), _CLIMB[16:], -4, 25),
                                     [r[4:] for r in _TORSO_GUN], 0, 19), [r[4:] for r in _HEAD], 0, 7)

MM = {name: [r[::-1] for r in rows] for name, rows in _RIGHT.items()}
MM["shot"] = [
    "..KKKK..",
    ".KYYYYK.",
    "KYYWWYYK",
    "KYWWWWYK",
    "KYWWWWYK",
    "KYYWWYYK",
    ".KYYYYK.",
    "..KKKK..",
]

# ── terrain (16 x 16) ────────────────────────────────────────────────────────
BRICK = [
    "KKKKKKKKKKKKKKKK",
    "EEEEEEEKEEEEEEEK",
    "EEEEEEEKEEEEEEEK",
    "EEEEEEEKEEEEEEEK",
    "eeeeeeeKeeeeeeeK",
    "KKKKKKKKKKKKKKKK",
    "EEEKEEEEEEEKEEEE",
    "EEEKEEEEEEEKEEEE",
    "EEEKEEEEEEEKEEEE",
    "eeeKeeeeeeeKeeee",
    "KKKKKKKKKKKKKKKK",
    "EEEEEEEKEEEEEEEK",
    "EEEEEEEKEEEEEEEK",
    "EEEEEEEKEEEEEEEK",
    "eeeeeeeKeeeeeeeK",
    "KKKKKKKKKKKKKKKK",
]
LEDGE = [                      # a brick whose top is open: the girder lip, as on the stage's platforms
    "EEEEEEEEEEEEEEEE",
    "DDDDDDDDDDDDDDDD",
    "DKDDDKDDDDKDDDKD",
    "KKKKKKKKKKKKKKKK",
] + BRICK[4:]
LADDER = [
    ".KDK........KDK.",
    ".KDKKKKKKKKKKDK.",
    ".KDWWWWWWWWWWDK.",
    ".KDWWWWWWWWWWDK.",
    ".KDKKKKKKKKKKDK.",
    ".KDDDDDDDDDDDDK.",
    ".KDDDDDDDDDDDDK.",
    ".KDK........KDK.",
    ".KDK........KDK.",
    ".KDKKKKKKKKKKDK.",
    ".KDWWWWWWWWWWDK.",
    ".KDWWWWWWWWWWDK.",
    ".KDKKKKKKKKKKDK.",
    ".KDDDDDDDDDDDDK.",
    ".KDDDDDDDDDDDDK.",
    ".KDK........KDK.",
]
SPIKE = [
    ".......K........",
    "......KWK.......",
    "......KWK.......",
    ".....KWLGK......",
    ".....KWLGK......",
    "....KWLLGGK.....",
    "....KWLLGGK.....",
    "...KWLLLGGgK....",
    "...KWLLLGGgK....",
    "..KWLLLLGGggK...",
    "..KWLLLLGGggK...",
    ".KWLLLLLGGgggK..",
    ".KWLLLLLGGgggK..",
    "KWLLLLLLGGggggK.",
    "KKKKKKKKKKKKKKKK",
    "gggggggggggggggg",
]
SPIKE = [r[:8] + r[:8] for r in [row[4:12] for row in SPIKE]]   # two narrower spikes per tile
DOOR = [
    "KKKKKKKKKKKKKKKK",
    "KLLLLLLLLLLLLLLK",
    "KGGGGGGGGGGGGGGK",
    "KggggggggggggggK",
    "KKKKKKKKKKKKKKKK",
    "KLLLLLLLLLLLLLLK",
    "KGGGGGGGGGGGGGGK",
    "KggggggggggggggK",
    "KKKKKKKKKKKKKKKK",
    "KLLLLLLLLLLLLLLK",
    "KGGGGGGGGGGGGGGK",
    "KggggggggggggggK",
    "KKKKKKKKKKKKKKKK",
    "KLLLLLLLLLLLLLLK",
    "KGGGGGGGGGGGGGGK",
    "KggggggggggggggK",
]
WALL = [                       # the boss rooms' tiled wall (background)
    "GGGGGGGGGGGGGGGg",
    "GLLLLLLLLLLLLLGg",
    "GLGGGGGGGGGGGGGg",
    "GLGGGGGGGGGGGGGg",
    "GLGGGGGGGGGGGGGg",
    "GLGGGGGGGGGGGGGg",
    "GLGGGGGGGGGGGGGg",
    "GLGGGGGGGGGGGGGg",
    "GLGGGGGGGGGGGGGg",
    "GLGGGGGGGGGGGGGg",
    "GLGGGGGGGGGGGGGg",
    "GLGGGGGGGGGGGGGg",
    "GLGGGGGGGGGGGGGg",
    "GLGGGGGGGGGGGGGg",
    "GGGGGGGGGGGGGGGg",
    "gggggggggggggggg",
]


TILES = {"tile_brick": BRICK, "tile_ledge": LEDGE, "tile_ladder": LADDER, "tile_spike": SPIKE,
         "tile_door": DOOR, "tile_wall": WALL}


def _surface(name: str, rows) -> pygame.Surface:
    """The sprite's PNG (art_png.py) when there is one, else drawn from its text."""
    png = art_png.load("megaman", name)
    return png if png is not None else art_png.to_surface(rows, PAL)


def _framed(name: str) -> list:
    """A sprite's rows in the size megaman_assets.FRAMES gives it, standing on its bottom edge."""
    rows = SPRITES[name]
    w, h = FRAMES[name][3:] if name in FRAMES else (max(map(len, rows)), len(rows))
    return ["." * w] * max(0, h - len(rows)) + [r.ljust(w, ".")[:w] for r in rows][-h:]


def all_sprites() -> dict:
    """name -> rows for every sprite and tile: what art_png.py exports."""
    return {**{name: _framed(name) for name in SPRITES}, **TILES}


class DrawnAssets:
    """The Assets interface over the drawings above, at the core's scale (tile_size / 16)."""

    def __init__(self, scale: float) -> None:
        self.scale = scale
        self._sprites: dict = {}
        self._screens: dict = {}
        self._tiles = {k: _surface(name, TILES[name]) for k, name in
                       {"#": "tile_brick", "ledge": "tile_ledge", "H": "tile_ladder", "^": "tile_spike",
                        "D": "tile_door", "wall": "tile_wall"}.items()}

    def sprite(self, name: str, flip_x: bool = False, flip_y: bool = False):
        key = (name, flip_x, flip_y)
        if key not in self._sprites:
            out = None
            if name in SPRITES:
                img = _surface(name, _framed(name))
                out = pygame.transform.scale(img, (int(img.get_width() * self.scale), int(img.get_height() * self.scale)))
                if flip_x or flip_y:
                    out = pygame.transform.flip(out, flip_x, flip_y)
            self._sprites[key] = out
        return self._sprites[key]

    def backdrop(self, level_file: str, entry: dict, grid, sx: int, sy: int, cols: int, rows: int):
        key = (id(grid), sx, sy)
        if key not in self._screens:
            if len(self._screens) > 256:
                self._screens.clear()
            self._screens[key] = self._paint(grid, sx, sy, cols, rows)
        return self._screens[key]

    def _paint(self, grid, sx, sy, cols, rows):
        # Past the first shutter on this row of screens is the boss corridor and room: tiled wall, not sky.
        doors = [c for r in range(sy * rows, (sy + 1) * rows) if r < len(grid)
                 for c, ch in enumerate(grid[r]) if terrain(ch) == "D"]
        indoors = bool(doors) and sx * cols > min(doors)
        out = pygame.Surface((cols * 16, rows * 16))
        out.fill(SKY)
        for r in range(rows):
            for c in range(cols):
                row, col = sy * rows + r, sx * cols + c
                kind = terrain(grid[row][col]) if row < len(grid) and col < len(grid[row]) else " "
                at = (c * 16, r * 16)
                if indoors and kind != "#":
                    out.blit(self._tiles["wall"], at)
                if kind == "#":
                    above = terrain(grid[row - 1][col]) if row > 0 and col < len(grid[row - 1]) else "#"
                    out.blit(self._tiles["ledge" if above in ".H" and not indoors else "#"], at)
                elif kind in ("H", "^", "D"):
                    out.blit(self._tiles[kind], at)
                elif kind == " ":
                    out.fill((0, 0, 0), (*at, 16, 16))
        return pygame.transform.scale(out, (int(cols * 16 * self.scale), int(rows * 16 * self.scale)))

    def play(self, events) -> None:
        """No sounds: the ripped effects are not ours to ship."""


SPRITES: dict = dict(MM)


def _flip(rows):
    return [r[::-1] for r in rows]


def _rot(rows):
    """A quarter turn clockwise."""
    return ["".join(row[i] for row in reversed(rows)) for i in range(len(rows[0]))]


# ── Cut Man (facing right; scissors on his head) ─────────────────────────────
_SCISSORS = [                  # 24 wide, 6 rows: the blades open in a V
    ".....KK......KK.........",
    ".....KWK....KWK.........",
    "......KWK..KWK..........",
    ".......KWKKWK...........",
    "........KWWK............",
    ".......KRKKRK...........",
]
_CUT_HEAD = [
    "........KKKKKKK.........",
    "......KKRRRRRRRKK.......",
    ".....KRRPPRRRRRRRK......",
    "....KRRPPRRRRRRRRRK.....",
    "....KRRRRRKKKKKKRRK.....",
    "....KRRRRKSSSSSSKRK.....",
    "....KRRRKSSSSKWWKRK.....",
    "....KRRRKSSSSKWKKK......",
    ".....KRRKSSSSSSSSK......",
    "......KKKSSSKKKSK.......",
    ".......KKSSSSSSK........",
]
_CUT_TORSO = [
    "......KRRKKKKKKRK.......",
    ".....KRRRRPPPPRRRK......",
    "....KRRKRRPPPPRRKRRK....",
    "....KRKKRRRRRRRRKKRK....",
    "....KSSKKRRRRRRKKKSSK...",
    ".....KKK.KKKKKK..KKK....",
]
_CUT_LEGS = {
    "stand": [
        "........KRRRRRRK........",
        ".......KRRPKKPRRK.......",
        "......KRRRK..KRRRK......",
        ".....KRRRRK..KRRRRK.....",
        "....KRRRRRK..KRRRRRK....",
        "....KKKKKKK..KKKKKKK....",
    ],
    "stride": [
        ".........KRRRRRRK.......",
        "........KRRPKKPRRK......",
        "......KKRRRK..KRRRKK....",
        "....KKRRRRK....KRRRRKK..",
        "...KRRRRRK......KRRRRRK.",
        "...KKKKKK........KKKKKK.",
    ],
    "pass": [
        "........KRRRRRRK........",
        "........KRRPPRRK........",
        "........KRRKKRRK........",
        ".......KRRRKKRRRK.......",
        "......KRRRRRRRRRRK......",
        "......KKKKKKKKKKKK......",
    ],
    "air": [
        "........KRRRRRRK........",
        ".......KRRPKKPRRK.......",
        "......KRRRK..KRRRKK.....",
        ".....KRRRK....KRRRRK....",
        ".....KRRK......KKKKK....",
        ".....KKKK...............",
    ],
}
_CUT_ARM_UP = [                # throwing: the arm over his head
    "......KRRKKKKKKRKKK.....",
    ".....KRRRRPPPPRRKSSK....",
    "....KRRKRRPPPPRRKRRK....",
    "....KRKKRRRRRRRRKRK.....",
    "....KSSKKRRRRRRKKK......",
    ".....KKK.KKKKKK.........",
]
_CUT_ARM_OUT = [               # just thrown: the arm reaching forward
    "......KRRKKKKKKRK.......",
    ".....KRRRRPPPPRRRKKKKK..",
    "....KRRKRRPPPPRRRRRRSSK.",
    "....KRKKRRRRRRRRKKKKKK..",
    "....KSSKKRRRRRRKK.......",
    ".....KKK.KKKKKK.........",
]


def _cut(legs, torso=_CUT_TORSO, scissors=True, w=24, dx=0):
    rows = (_SCISSORS if scissors else []) + _CUT_HEAD + torso + _CUT_LEGS[legs]
    return _shift(rows, dx, w)


CUT = {
    "cut_idle0": _cut("stand"), "cut_bare_idle0": _cut("stand", scissors=False),
    "cut_jump": _cut("air"), "cut_bare_jump": _cut("air", scissors=False),
    "cut_throw": _cut("stand", _CUT_ARM_UP),
    "cut_thrown": _cut("stand", _CUT_ARM_OUT, scissors=False, w=32, dx=2),
    # the walk frames face left on his sheet
    "cut_walk0": _flip(_cut("stride", w=32, dx=4)), "cut_walk1": _flip(_cut("pass", w=28, dx=2)),
    "cut_walk2": _flip(_cut("stand", w=26, dx=1)),
    "cut_bare_walk0": _flip(_cut("stride", scissors=False, w=32, dx=4)),
    "cut_bare_walk1": _flip(_cut("pass", scissors=False)), "cut_bare_walk2": _flip(_cut("stand", scissors=False)),
}
_BLINK = ["....KRRRKSSSSKKKKRK.....", "....KRRRKSSSSSSSKK......"]
CUT["cut_idle1"] = CUT["cut_idle0"][:12] + _BLINK + CUT["cut_idle0"][14:]
CUT["cut_bare_idle1"] = CUT["cut_bare_idle0"][:6] + _BLINK + CUT["cut_bare_idle0"][8:]
_ROLL = [                      # the Rolling Cutter: his scissors, spinning
    "................",
    "..KK........KK..",
    ".KWWK......KWWK.",
    ".KWLWK....KWLWK.",
    "..KWLWK..KWLWK..",
    "...KWLWKKWLWK...",
    "....KWLWWLWK....",
    ".....KWWWWK.....",
    "....KRKKKKRK....",
    "...KRRK..KRRK...",
    "..KRK......KRK..",
    "..KRK......KRK..",
    "...KRRK..KRRK...",
    "....KKK..KKK....",
    "................",
    "................",
]
for _i in range(4):
    CUT[f"rolling{_i}"] = _ROLL
    _ROLL = _rot(_ROLL)

# ── the stage's enemies (facing left) ────────────────────────────────────────
ENEMIES = {
    "blader0": [
        ".KKKKKKKKKKKKKK.",
        "......KGK.......",
        "......KGK.......",
        "....KKKKKKK.....",
        "...KCCCCCCCK....",
        "..KCBBBBBBBCK...",
        ".KCBBBBBBBBBCK..",
        ".KBKWKBBBKWKBK..",
        ".KBKKKBBBKKKBK..",
        ".KBBBBBBBBBBBK..",
        ".KKWKWKWKWKWKK..",
        "..KWKWKWKWKWK...",
        "...KKKKKKKKK....",
    ],
    "octopus2": [              # eye open
        "KKKKKKKKKKKKKKKK",
        "KOOOOOOOOOOOOOOK",
        "KOKKKKKKKKKKKKOK",
        "KOKWWWWWWWWWWKOK",
        "KOKWWWWKKWWWWKOK",
        "KOKWWWKKKKWWWKOK",
        "KOKWWWKKKKWWWKOK",
        "KOKWWWWKKWWWWKOK",
        "KOKWWWWWWWWWWKOK",
        "KOKKKKKKKKKKKKOK",
        "KOOOOOOOOOOOOOOK",
        "KDDDDDDDDDDDDDDK",
        "KDKKDDKKDDKKDDKK",
        "KDDDDDDDDDDDDDDK",
        "KDDDDDDDDDDDDDDK",
        "KKKKKKKKKKKKKKKK",
    ],
    "flea_sit": [
        "....KKKKKKK.....",
        "...KBBBBBBBK....",
        "..KBCCBBBBBBK...",
        "..KBCKWBBKWBK...",
        "..KBBKKBBKKBK...",
        "..KBBBBBBBBBK...",
        "...KBBKKKBBK....",
        "...KBK...KBK....",
        "..KBBK...KBBK...",
        "..KKKK...KKKK...",
    ],
    "bigeye_low": [
        "..........KKKKKKKKKKK...........",
        "........KKPPPPPPPPPPPKK.........",
        "......KKPPPPPPPPPPPPPPPKK.......",
        ".....KPPPPPKKKKKKPPPPPPPPK......",
        "....KPPPPKKWWWWWWKKPPPPPPPK.....",
        "...KPPPPKWWWWWWWWWWKPPPPPPPK....",
        "...KPPPKWWWWKKKKWWWWKPPPPPPK....",
        "..KPPPPKWWWKKKKKKWWWKPPPPPPPK...",
        "..KPPPKWWWKKKKKKKKWWWKPPPPPPK.KK",
        "..KPPPKWWWKKKKKKKKWWWKPPPPPPKKWK",
        "..KPPPKWWWKKKKKKKKWWWKPPPPPPKW.K",
        "..KPPPPKWWWKKKKKKWWWKPPPPPPPKW.K",
        "..KPPPPKWWWWKKKKWWWWKPPPPPPPKWK.",
        "..KPPPPPKWWWWWWWWWWKPPPPPPPPKK..",
        "..KMPPPPPKKWWWWWWKKPPPPPPPPMK...",
        "..KMMPPPPPPKKKKKKPPPPPPPPPMMK...",
        "..KMMMPPPPPPPPPPPPPPPPPPPMMMK...",
        "...KMMMMPPPPPPPPPPPPPPPMMMMK....",
        "....KMMMMMMMMMMMMMMMMMMMMMK.....",
        ".....KKMMMMMMMMMMMMMMMMMKK......",
        ".......KKKKKKKKKKKKKKKKK........",
        "............KWWWWWK.............",
        "............KLLLLLK.............",
        "............KWWWWWK.............",
        "............KLLLLLK.............",
        "...........KKKKKKKKK............",
        "..........KGGGGGGGGGK...........",
        ".........KGLLLLLLLLLGK..........",
        "........KGGGGGGGGGGGGGK.........",
        "........KKKKKKKKKKKKKKK.........",
    ],
    "shell_shut": [
        "......KKKK......",
        "....KKOOOOKK....",
        "...KOOYYOOOOK...",
        "..KOOYOOOOOOOK..",
        "..KOOOOOOOOOOK..",
        "..KKKKKKKKKKKK..",
        "..KGGGGGGGGGGK..",
        "...KGGGGGGGGK...",
        "....KKGGGGKK....",
        "......KKKK......",
    ],
    "cutter0": [
        "....KKKKKKK.....",
        "..KKWWWWWWWKK...",
        ".KWWLLLLLLLWWK..",
        ".KWLKKKKKKKLWK..",
        "KWLK.......KWK..",
        "KWLK............",
        "KWLK............",
        "KWLK............",
        "KWLK.......KWK..",
        ".KWLKKKKKKKLWK..",
        ".KWWLLLLLLLWWK..",
        "..KKWWWWWWWKK...",
        "....KKKKKKK.....",
        "................",
        "................",
        "................",
    ],
    "screw_shut": [
        "................",
        "................",
        "......KKKK......",
        "....KKOOOOKK....",
        "...KOOYYOOOOK...",
        "..KOOOOOOOOOOK..",
        ".KKKKKKKKKKKKKK.",
        ".KGGGGGGGGGGGGK.",
    ],
    "bullet": [
        "..KKKK..",
        ".KPPPPK.",
        "KPPWWPPK",
        "KPWWWWPK",
        "KPWWWWPK",
        "KPPWWPPK",
        ".KPPPPK.",
        "..KKKK..",
    ],
    "life_small": [
        "....KKKKKKKK....",
        "...KYYYYYYYYK...",
        "...KYWWYYYYYK...",
        "...KKKKKKKKKK...",
        "...KOOOOOOOOK...",
        "...KOOOOOOOOK...",
        "....KKKKKKKK....",
    ],
    "life_large": [
        "..KKKKKKKKKKKK..",
        ".KYYYYYYYYYYYYK.",
        ".KYWWWYYYYYYYYK.",
        ".KYWYYYYYYYYYYK.",
        ".KYYYYYYYYYYYYK.",
        ".KKKKKKKKKKKKKK.",
        ".KOOOOOOOOOOOOK.",
        ".KOOOOOOOOOOOOK.",
        ".KOOOOOOOOOOOOK.",
        ".KDDDDDDDDDDDDK.",
        ".KDDDDDDDDDDDDK.",
        "..KKKKKKKKKKKK..",
    ],
    "pearl": [
        "..KKKK..",
        ".KWWWLK.",
        "KWWLLLGK",
        "KWLLLLGK",
        "KLLLLGGK",
        "KLLLGGgK",
        ".KGGggK.",
        "..KKKK..",
    ],
    "clear": [
        "....KKKKKKKK....",
        "..KKWWWWWWWWKK..",
        ".KWWWYYYYYYWWWK.",
        ".KWYYYOOOOYYYWK.",
        "KWWYYOOOOOOYYWWK",
        "KWYYOOOWWOOOYYWK",
        "KWYOOOWWWWOOOYWK",
        "KWYOOOWWWWOOOYWK",
        "KWYYOOOWWOOOYYWK",
        "KWWYYOOOOOOYYWWK",
        ".KWYYYOOOOYYYWK.",
        ".KWWWYYYYYYWWWK.",
        "..KKWWWWWWWWKK..",
        "....KKKKKKKK....",
    ],
    "boom": [
        "...........W............",
        "....W......W......W.....",
        ".....W.....W.....W......",
        "......W...WWW...W.......",
        ".......W.WWWWW.W........",
        "........WWWWWWW.........",
        "........WWWYWWW.........",
        ".W.WWWWWWWYYYWWWWWWW.W..",
        "........WWWYWWW.........",
        "........WWWWWWW.........",
        ".......W.WWWWW.W........",
        "......W...WWW...W.......",
        ".....W.....W.....W......",
        "....W......W......W.....",
        "...........W............",
    ],
}
ENEMIES["blader1"] = ["....KKKKKKK....."] + ENEMIES["blader0"][1:]
ENEMIES["flea_jump"] = ENEMIES["flea_sit"][:7] + [
    "...KBK...KBK....",
    "...KBK...KBK....",
    "..KBK.....KBK...",
    "..KBK.....KBK...",
    ".KBK.......KBK..",
    ".KBK.......KBK..",
    "KBBK.......KBBK.",
    "KKKK.......KKKK.",
]
ENEMIES["bigeye_high"] = ENEMIES["bigeye_low"][:21] + ENEMIES["bigeye_low"][21:25] * 3 + ENEMIES["bigeye_low"][25:]
_eye = ENEMIES["octopus2"]
ENEMIES["octopus1"] = _eye[:3] + [_eye[2]] * 2 + _eye[5:7] + [_eye[2]] * 2 + _eye[9:]   # half shut
ENEMIES["octopus0"] = _eye[:3] + [_eye[2]] * 6 + _eye[9:]                                # shut
ENEMIES["shell_open"] = ENEMIES["shell_shut"][:5] + [
    "................",
    ".....KKKKKK.....",
    "....KWWKKWWK....",
    "....KWKKKKWK....",
    ".....KKKKKK.....",
    "................",
] + ENEMIES["shell_shut"][5:]
ENEMIES["cutter1"] = _flip(_rot(ENEMIES["cutter0"])) + _rot(ENEMIES["cutter0"])
_DOME = ENEMIES["screw_shut"]
for _i, _top in enumerate(["....KKKKKKKK....", "...KYKOOOOKYK...", "...KOYKOOKYOK...", "...KOOYKKYOOK..."]):
    ENEMIES[f"screw{_i}"] = ["................"] * 4 + [_top, "......KOOK......", "......KOOK......",
                                                       "....KKOOOOKK...."] + _DOME[4:]
_BEAK = [                      # shut: a shell on the wall to its right
    "........KKKKKKKK",
    "......KKMMMMMMGK",
    ".....KMMPPPPMMGK",
    "....KMPPPPPPPMGK",
    "...KMPPPPPPPPMGK",
    "..KMPPPPPPPPPMGK",
    "..KMPPPPPPPPPMGK",
    ".KMPPPPPPPPPPMGK",
    ".KMPPPPPPPPPPMGK",
    "..KMPPPPPPPPPMGK",
    "..KMPPPPPPPPPMGK",
    "...KMPPPPPPPPMGK",
    "....KMPPPPPPPMGK",
    ".....KMMPPPPMMGK",
    "......KKMMMMMMGK",
    "........KKKKKKKK",
]
ENEMIES["beak0"] = _BEAK
ENEMIES["beak1"] = [r[8:] for r in _BEAK]
ENEMIES["beak2"] = _BEAK[:6] + ["..KKKKKKKKKKKKGK", "..KgggKKKKKKKKGK", "..KgggKKKKKKKKGK",
                                "..KKKKKKKKKKKKGK"] + _BEAK[10:]
ENEMIES["beak3"] = _BEAK[:4] + ["..KKKKKKKKKKKMGK", "KKKKKKKKKKKKKKGK", "KgggggKKKWWKKKGK", "KgggggKKKWKKKKGK",
                                "KgggggKKKKKKKKGK", "KgggggKKKKKKKKGK", "KKKKKKKKKKKKKKGK",
                                "..KKKKKKKKKKKMGK"] + _BEAK[12:]

ENEMIES["weapon_small"] = [r.replace("Y", "C").replace("O", "B") for r in ENEMIES["life_small"]]
ENEMIES["weapon_large"] = [r.replace("Y", "C").replace("O", "B").replace("D", "B") for r in ENEMIES["life_large"]]
ENEMIES["oneup"] = _HEAD[:11] + ["......KKKKKKKKK........."]   # Mega Man's head

SPRITES.update(CUT)
SPRITES.update(ENEMIES)
