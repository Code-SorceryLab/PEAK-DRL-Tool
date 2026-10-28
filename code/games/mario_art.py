"""Our Mario's world: magic-themed NES-style pixel art around the game's own sorcerer sprite
(code/games/assets/*.png, untouched). Original drawings.

Every sprite is a list of strings, one character per pixel ('.' transparent), drawn at half the game's
size (a 32 px tile is 16 x 16 here) and shown at 2x. The game's rules are untouched: each sprite stands
in for one thing the game already has (the imp is the Goomba, the rune tortoise the Koopa, the mana gem
the coin, ...). The imp and the tortoise face left; the game mirrors them.
The PNGs in assets/art/mario/ (art_png.py) are drawn in place of the text when they are there.
"""
from __future__ import annotations

import pygame

from . import art_png

PAL = {
    "K": (20, 16, 32),       # outline
    "W": (248, 248, 248),
    "P": (124, 64, 208),     # robe, hat
    "p": (76, 36, 140),
    "R": (220, 48, 40),      # the fire sorcerer's robe
    "r": (140, 20, 20),
    "Y": (252, 216, 88),     # gold: stars, trim, runes
    "y": (200, 148, 24),
    "S": (252, 216, 168),    # the mushroom's stem
    "G": (224, 224, 236),    # silver
    "g": (150, 150, 172),
    "N": (140, 92, 44),      # staff wood
    "C": (96, 224, 252),     # mana: orbs, crystals
    "c": (32, 140, 220),
    "O": (252, 140, 32),     # flame
    "M": (196, 56, 136),     # imp
    "m": (120, 24, 80),
    "E": (72, 184, 72),      # tortoise, moss
    "e": (24, 112, 48),
    "L": (120, 120, 160),    # stone
    "l": (80, 80, 116),
    "V": (150, 110, 190),    # brick
    "v": (98, 70, 136),
}
SKY = (56, 60, 132)          # twilight

SPRITES: dict = {}

# ── the imp (Goomba) ─────────────────────────────────────────────────────────
SPRITES["imp_0"] = [
    "..K.......K..",
    ".KMK.....KMK.",
    ".KMMKKKKKMMK.",
    "KMMMMMMMMMMMK",
    "KMYKMMMMMYKMK",
    "KMYYMMMMMYYMK",
    "KMMMMKKKMMMMK",
    ".KMMMWKWMMMK.",
    "..KmMMMMMmK..",
    ".KKK.....KKK.",
]
SPRITES["imp_1"] = SPRITES["imp_0"][:9] + ["..KKK...KKK.."]

# ── the rune tortoise (Koopa): walking, winged, its crystal shell ─────────────
SPRITES["tortoise_0"] = [
    "..KKK......",
    ".KEEEK.....",
    "KEWKEK.....",
    "KEEEEK.....",
    ".KEEK.KKKK.",
    "..KEKKCCCCK",
    "..KEKCGCCCK",
    ".KEEKCCYCCK",
    ".KEEKCYYYcK",
    "..KKKCCYCcK",
    "...KCCCCccK",
    "...KKKKKKK.",
    "...KEK.KEK.",
    "..KEEK.KEEK",
    "..KKKK.KKKK",
]
SPRITES["tortoise_1"] = SPRITES["tortoise_0"][:12] + ["...KEKKEK..", "..KEEKKEEK.", "..KKKKKKKK."]
def _put(rows, part, x, y):
    """part painted over rows at (x, y); its '.' leave rows alone."""
    out = list(rows)
    for j, line in enumerate(part):
        row = list(out[y + j])
        for i, ch in enumerate(line):
            if ch != ".":
                row[x + i] = ch
        out[y + j] = "".join(row)
    return out


SPRITES["tortoise_fly_0"] = _put(SPRITES["tortoise_0"], ["..KK.", ".KWWK", "KWWWK", "KWWK."], 6, 0)   # wing up
SPRITES["tortoise_fly_1"] = _put(SPRITES["tortoise_0"], ["KWWK.", "KWWWK", ".KWWK", "..KK."], 6, 2)   # wing down
SPRITES["shell_0"] = [
    "...KKKKK...",
    "..KCCGCCK..",
    ".KCGCCCCCK.",
    ".KCCCYCCcK.",
    "KCCCYYYCccK",
    "KCCCCYCCccK",
    "KCCCCCCcccK",
    "KKKKKKKKKKK",
    "KEEEEEEEEEK",
    ".KeeeeeeeK.",
    "..KKKKKKK..",
]
SPRITES["shell_1"] = [r.replace("Y", "W") for r in SPRITES["shell_0"]]   # it glints as it slides

# ── blocks and ground (16 x 16) ──────────────────────────────────────────────
SPRITES["ground"] = [
    "LLLLLLLKLLLLLLLK",
    "LGLLLLLKLGLLLLlK",
    "LLLLLLlKLLLLLLlK",
    "LLLLLLlKLLLLLLlK",
    "llllllKKllllllKK",
    "KKKKKKKKKKKKKKKK",
    "LLLKLLLLLLLKLLLL",
    "LGLKLGLLLLlKLGLL",
    "LLLKLLLLLLlKLLLL",
    "llKKllllllKKllll",
    "KKKKKKKKKKKKKKKK",
    "LLLLLLLKLLLLLLLK",
    "LGLLLLLKLGLLLLlK",
    "LLLLLLlKLLLLLLlK",
    "llllllKKllllllKK",
    "KKKKKKKKKKKKKKKK",
]
SPRITES["ground_top"] = [
    "EEEEEEEEEEEEEEEE",
    "EeEEEEeEEEEeEEEE",
    "eKeEEeKeEEeKeEeE",
    "KLKeeKLKeeKLKeKe",
] + SPRITES["ground"][4:]
SPRITES["brick"] = [
    "KKKKKKKKKKKKKKKK",
    "VVVVVVVKVVVVVVVK",
    "VYVVVVvKVVVVVVvK",
    "VVVVVVvKVVVVVVvK",
    "vvvvvvvKvvvvvvvK",
    "KKKKKKKKKKKKKKKK",
    "VVVKVVVVVVVKVVVV",
    "VVvKVVVVVVvKVVVV",
    "VVvKVVVVVVvKVVVV",
    "vvvKvvvvvvvKvvvv",
    "KKKKKKKKKKKKKKKK",
    "VVVVVVVKVVVVVVVK",
    "VVVVVVvKVVVVVYvK",
    "VVVVVVvKVVVVVVvK",
    "vvvvvvvKvvvvvvvK",
    "KKKKKKKKKKKKKKKK",
]
_BLOCK = [                     # gold, riveted, bevelled: the ? block before its mark
    "KKKKKKKKKKKKKKKK",
    "KYYYYYYYYYYYYYyK",
    "KYKYYYYYYYYYYKyK",
] + ["KYYYYYYYYYYYYYyK"] * 10 + [
    "KYKYYYYYYYYYYKyK",
    "KyyyyyyyyyyyyyyK",
    "KKKKKKKKKKKKKKKK",
]
_QUESTION = [                  # the mark, outlined, 6 x 10
    ".KKKKK",
    "KWWWWWK",
    "KWKKKWWK",
    "KKKYYKWWK",
    "....KWWK",
    "...KWWK",
    "...KWWK",
    "...KKK",
    "...KWWK",
    "...KKKK",
]
SPRITES["question_block_0"] = _put(_BLOCK, _QUESTION, 4, 3)
SPRITES["question_block_1"] = [r.replace("W", "C") for r in SPRITES["question_block_0"]]   # the mark glows
SPRITES["question_block_used"] = [r.replace("Y", "l").replace("y", "K") for r in _BLOCK]
SPRITES["crystal_spikes"] = [
    "................",
    "...K.......K....",
    "..KCK.....KCK...",
    "..KCK..K..KCK...",
    ".KCGcK.KK.KCGcK.",
    ".KCGcKKCK.KCGcK.",
    ".KCGcKKCcKKCGcK.",
    "KCCGccKCGcKCCGcK",
    "KCCGccKCGcKCCGcK",
    "KCCGcKCCGccKCGcK",
    "KCGGcKCCGccKCGcK",
    "KCGcccKCGcKCCGcK",
    "KKKKKKKKKKKKKKKK",
    "LLLLLLLLLLLLLLLL",
    "llllllllllllllll",
    "KKKKKKKKKKKKKKKK",
]
SPRITES["staff_pole"] = [
    "......KGGK......",
    "......KGgK......",
    "......KYgK......",
    "......KGYK......",
    "......KGgK......",
    "......KGgK......",
    "......KYgK......",
    "......KGYK......",
    "......KGgK......",
    "......KGgK......",
    "......KYgK......",
    "......KGYK......",
    "......KGgK......",
    "......KGgK......",
    "......KYgK......",
    "......KGYK......",
]
SPRITES["staff_top"] = [
    "......KKKK......",
    "....KKCCCCKK....",
    "...KCCWWCCccK...",
    "...KCWWCCCccK...",
    "...KCCCCCCccK...",
    "...KCCCCCcccK...",
    "....KKcccccK....",
    "......KKKK......",
    ".....KYYYYK.....",
    "......KGgK......",
    "......KYgK......",
    "......KGYK......",
    "......KGgK......",
    "......KGgK......",
    "......KYgK......",
    "......KGYK......",
]

# ── pickups and spells ───────────────────────────────────────────────────────
SPRITES["mana_gem_0"] = [
    "...KK...",
    "..KCWK..",
    ".KCWCcK.",
    "KCCCCccK",
    "KcCCCccK",
    ".KcCccK.",
    "..KccK..",
    "...KK...",
]
SPRITES["mana_gem_1"] = [r.replace("W", "C") for r in SPRITES["mana_gem_0"][:3]] + SPRITES["mana_gem_0"][3:]
# The power-ups stay what they are, in the world's darker, gothic colours.
SPRITES["mushroom"] = [        # crimson cap, glinting spots, a bone-pale stem
    "...KKKK...",
    ".KKrRRrKK.",
    "KrRRWRRRrK",
    "KRWCRRRWRK",
    "KRRRRrRCWK",
    "KrRWRRRRrK",
    "KKKKKKKKKK",
    ".KGKGGKGK.",
    ".KgGGGGgK.",
    "..KKKKKK..",
]
SPRITES["one_up"] = [r.replace("R", "E").replace("r", "e") for r in SPRITES["mushroom"]]
SPRITES["fire_flower"] = [     # flame petals round a white-hot core, a thorny stem
    "..KK..KK..",
    ".KOOKKOOK.",
    "KORYYYYROK",
    ".KYWWWWYK.",
    "KOYWCCWYOK",
    ".KYWWWWYK.",
    "KORYYYYROK",
    ".KOOKKOOK.",
    ".KeeKeKeeK",
    "..KKKeKKK.",
]
SPRITES["star"] = [            # gold, its right half in shadow, a mana glint at its heart
    "....KK....",
    "...KWYK...",
    "KKKKYyKKKK",
    "KWYYYYYyyK",
    ".KYYKYKyK.",
    "..KYYYyK..",
    ".KYYCYyyK.",
    ".KYYKKyyK.",
    "KYYK..KyyK",
    "KKK....KKK",
]
SPRITES["magic_bolt_0"] = [
    "..KK..",
    ".KCWK.",
    "KCWWCK",
    "KcCCcK",
    ".KccK.",
    "..KK..",
]
SPRITES["magic_bolt_1"] = [
    "K.KK.K",
    ".KWCK.",
    "KCWCcK",
    "KCCcCK",
    ".KcCK.",
    "K.KK.K",
]
SPRITES["flame_orb"] = [       # one link of a fire bar
    "..KKK..",
    ".KPWPK.",
    "KPWWWPK",
    "KPWYWPK",
    "KpPWPpK",
    ".KpPpK.",
    "..KKK..",
]


SPRITES["rune_platform"] = [   # one 32 x 16 piece of a moving lift
    "KKKKKKKKKKKKKKKK",
    "KCCCCCCCCCCCCCCK",
    "KCWCCCCCCCCCWCcK",
    "KccccccccccccccK",
    "KVVVVVVVVVVVVVVK",
    "KvVYvVVvYVVvYVvK",
    ".KvvvvvvvvvvvvK.",
    "..KKKKKKKKKKKK..",
]

_scaled: dict = {}


def image(name: str, w: int, h: int, flip: bool = False) -> pygame.Surface:
    """The sprite stretched to w x h (its object's box), mirrored if asked: its PNG when there is one."""
    key = (name, int(w), int(h), flip)
    if key not in _scaled:
        img = art_png.load("mario", name)
        if img is None:
            img = art_png.to_surface(SPRITES[name], PAL)
        img = pygame.transform.scale(img, (int(w), int(h)))
        _scaled[key] = pygame.transform.flip(img, True, False) if flip else img
    return _scaled[key]


def draw(surface: pygame.Surface, name: str, x: float, y: float, w: float, h: float, flip: bool = False) -> None:
    surface.blit(image(name, w, h, flip), (int(x), int(y)))


def tick(frames: int, ms: int) -> int:
    """Which of `frames` animation frames shows now, each lasting `ms` (drawing only; the game's clock is untouched)."""
    return pygame.time.get_ticks() // ms % frames


def all_sprites() -> dict:
    """name -> rows: what art_png.py exports."""
    return dict(SPRITES)
