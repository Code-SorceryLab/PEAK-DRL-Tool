"""Optional art and sound for megaman, read from a local, git-ignored folder (megaman-data/).

Nothing here is needed to play or train: without the folder the core draws its flat placeholder
shapes and stays silent. The folder holds copyrighted material (the ROM, sprite sheets ripped by
Mister Mike, sound effects ripped by Nai255), so only the *coordinates* of the frames live in
this file.

  * sprites  - boxes on the sheets; the sheets' green and purple backing becomes transparent
  * tiles    - one backdrop per 16 x 15 screen, cropped from the stage map image. A cell whose
               terrain no longer matches the ROM's (the level was edited) is repainted with the
               most common tile of its new kind, so an edited level still reads correctly
  * sounds   - event name -> .wav
"""
from __future__ import annotations

import os
from collections import Counter

import pygame

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SHEET = "sprites/NES - Mega Man - {}.png"
MM, CUT, ENEMY, ITEM, WEAPON = ("Playable Characters - Mega Man", "Robot Masters - Cut Man",
                                "Enemies & Other Bosses - Enemies", "Miscellaneous - Items & Miscellaneous",
                                "Miscellaneous - Weapons")

# name: (sheet, x, y, w, h). Mega Man and the enemies face left on their sheets; Cut Man faces right.
FRAMES = {
    "mm_idle": (MM, 1, 21, 24, 24), "mm_blink": (MM, 26, 21, 24, 24), "mm_inch": (MM, 55, 21, 24, 24),
    "mm_walk0": (MM, 87, 21, 32, 24), "mm_walk1": (MM, 120, 21, 24, 24), "mm_walk2": (MM, 145, 21, 24, 24),
    "mm_jump": (MM, 174, 13, 32, 32), "mm_climb0": (MM, 211, 13, 16, 32), "mm_climb1": (MM, 228, 13, 16, 32),
    "mm_hurt": (MM, 266, 13, 32, 32),
    "mm_idle_shoot": (MM, 1, 78, 32, 24), "mm_walk0_shoot": (MM, 38, 78, 32, 24),
    "mm_walk1_shoot": (MM, 71, 78, 32, 24), "mm_walk2_shoot": (MM, 104, 78, 32, 24),
    "mm_jump_shoot": (MM, 141, 70, 32, 32), "mm_climb_shoot": (MM, 178, 70, 24, 32),
    "shot": (WEAPON, 220, 11, 8, 8),

    "cut_idle0": (CUT, 1, 21, 24, 32), "cut_idle1": (CUT, 26, 21, 24, 32),
    "cut_walk0": (CUT, 55, 24, 32, 29), "cut_walk1": (CUT, 88, 22, 28, 31), "cut_walk2": (CUT, 117, 24, 26, 29),
    "cut_jump": (CUT, 148, 21, 24, 32), "cut_throw": (CUT, 182, 21, 24, 32), "cut_thrown": (CUT, 207, 29, 32, 24),
    "cut_bare_idle0": (CUT, 1, 78, 24, 24), "cut_bare_idle1": (CUT, 26, 78, 24, 24),
    "cut_bare_walk0": (CUT, 55, 78, 32, 24), "cut_bare_walk1": (CUT, 88, 78, 24, 24),
    "cut_bare_walk2": (CUT, 113, 78, 24, 24), "cut_bare_jump": (CUT, 142, 78, 24, 24),
    "rolling0": (CUT, 39, 117, 16, 16), "rolling1": (CUT, 56, 117, 16, 16),
    "rolling2": (CUT, 73, 117, 16, 16), "rolling3": (CUT, 90, 117, 16, 16),

    "blader0": (ENEMY, 486, 19, 16, 24), "blader1": (ENEMY, 503, 19, 16, 24),
    "beak0": (ENEMY, 312, 28, 16, 16), "beak1": (ENEMY, 329, 28, 8, 16),
    "beak2": (ENEMY, 338, 28, 16, 16), "beak3": (ENEMY, 355, 28, 16, 16),
    "flea_sit": (ENEMY, 166, 173, 16, 16), "flea_jump": (ENEMY, 183, 165, 16, 24),
    "octopus0": (ENEMY, 1, 19, 16, 16), "octopus1": (ENEMY, 18, 19, 16, 16), "octopus2": (ENEMY, 35, 19, 16, 16),
    "bigeye_low": (ENEMY, 143, 19, 32, 40), "bigeye_high": (ENEMY, 110, 11, 32, 48),
    "shell_shut": (ENEMY, 301, 181, 16, 16), "shell_open": (ENEMY, 319, 173, 16, 24),
    "cutter0": (ENEMY, 383, 220, 16, 16), "cutter1": (ENEMY, 400, 212, 16, 32),
    "screw_shut": (ENEMY, 156, 259, 16, 8), "screw0": (ENEMY, 173, 251, 16, 16), "screw1": (ENEMY, 190, 251, 16, 16),
    "screw2": (ENEMY, 207, 251, 16, 16), "screw3": (ENEMY, 224, 251, 16, 16),
    "bullet": (ENEMY, 372, 15, 8, 8),

    "life_small": (ITEM, 1, 1, 16, 16), "life_large": (ITEM, 18, 1, 16, 16), "pearl": (ITEM, 78, 22, 8, 8),
    "clear": (ITEM, 1, 18, 16, 16), "boom": (ITEM, 1, 39, 24, 24),
}

SOUNDS = {
    "start": "04 - MegamanWarp", "shoot": "05 - MegaBuster", "land": "06 - MegamanLand",
    "hurt": "07 - MegamanDamage", "death": "08 - MegamanDefeat", "enemy_hit": "09 - EnemyDamage",
    "enemy_shoot": "10 - EnemyShoot", "deflect": "11 - Dink", "big_eye": "12 - BigEye",
    "explosion": "13 - Explosion", "rolling_cutter": "18 - RollingCutter", "heal": "24 - EnergyFill",
    "pearl": "26 - BonusBall", "gate": "30 - BossGate",
}

_BACKING = ((0, 128, 0), (128, 0, 128))     # the sheets' green page and purple sprite cells
_MAP_ORIGIN, _MAP_PITCH = 1, 256            # the stage map draws each screen on a 256 px grid
_cache: dict = {}


def terrain(glyph: str) -> str:
    """The class of a level glyph as far as tile art goes."""
    return {"#": "#", "H": "H", "^": "^", "D": "D", "d": "D", " ": " "}.get(glyph, ".")


class Assets:
    """Everything loaded for one (folder, scale). Share it: `Assets.get(cfg, scale)`."""

    @classmethod
    def get(cls, cfg: dict | None, scale: float) -> "Assets | None":
        cfg = cfg or {}
        root = os.environ.get("PEAK_MM1_ASSETS") or cfg.get("dir") or "megaman-data"
        root = root if os.path.isabs(root) else os.path.join(_REPO, root)
        key = (root, scale)
        if key not in _cache:
            if os.path.isdir(os.path.join(root, "sprites")):
                _cache[key] = cls(root, cfg, scale)
            else:                               # no ripped art here: our own drawings (megaman_art.py)
                from .megaman_art import DrawnAssets
                _cache[key] = DrawnAssets(scale)
        return _cache[key]

    def __init__(self, root: str, cfg: dict, scale: float) -> None:
        self.root, self.cfg, self.scale = root, cfg, scale
        self._sheets: dict = {}
        self._sprites: dict = {}
        self._stages: dict = {}
        self._sources: dict = {}
        self._sounds: dict | None = None

    # ── sprites ──────────────────────────────────────────────────────────────
    def _sheet(self, name: str):
        if name not in self._sheets:
            path = os.path.join(self.root, _SHEET.format(name))
            self._sheets[name] = pygame.image.load(path) if os.path.exists(path) else None
        return self._sheets[name]

    def sprite(self, name: str, flip_x: bool = False, flip_y: bool = False):
        """A frame at game scale with a transparent background, or None if its sheet is missing."""
        if name not in FRAMES:
            return None
        key = (name, flip_x, flip_y)
        if key not in self._sprites:
            sheet, x, y, w, h = FRAMES[name]
            src = self._sheet(sheet)
            out = None
            if src is not None:
                out = pygame.Surface((w, h), pygame.SRCALPHA, 32)
                out.blit(src, (0, 0), (x, y, w, h))
                rgb, alpha = pygame.surfarray.pixels3d(out), pygame.surfarray.pixels_alpha(out)
                for colour in _BACKING:
                    alpha[(rgb[:, :, 0] == colour[0]) & (rgb[:, :, 1] == colour[1]) & (rgb[:, :, 2] == colour[2])] = 0
                del rgb, alpha
                out = pygame.transform.scale(out, (int(w * self.scale), int(h * self.scale)))
                if flip_x or flip_y:
                    out = pygame.transform.flip(out, flip_x, flip_y)
            self._sprites[key] = out
        return self._sprites[key]

    # ── tiles ────────────────────────────────────────────────────────────────
    def backdrop(self, level_file: str, entry: dict, grid, sx: int, sy: int, cols: int, rows: int):
        """The art for screen (sx, sy) of this level, or None when the level names no stage.
        `entry` is the level's megaman_levels.yaml entry: `stage`, and for a slice the `origin`
        (in screens) of its top-left corner on that stage."""
        name = entry.get("stage")
        source = self._source(name) if name else None
        if source is None:
            return None
        stamp = (level_file, os.path.getmtime(level_file))
        if stamp not in self._stages:
            if len(self._stages) > 16:
                self._stages.clear()
            self._stages[stamp] = {}                # an edited level repaints from scratch
        screens = self._stages[stamp]
        if (sx, sy) not in screens:
            ox, oy = entry.get("origin") or (0, 0)
            screens[(sx, sy)] = self._paint(source, grid, sx, sy, int(ox), int(oy), cols, rows)
        return screens[(sx, sy)]

    def _source(self, name: str):
        """The stage's map image and its untouched grid, straight from the ROM."""
        if name not in self._sources:
            source = None
            image = os.path.join(self.root, (self.cfg.get("maps") or {}).get(name, ""))
            rom = os.path.join(self.root, self.cfg.get("rom", ""))
            if os.path.isfile(image) and os.path.isfile(rom):
                from .tools.mm1_extract_level import STAGES, extract
                with open(rom, "rb") as f:
                    stock, _ = extract(f.read(), STAGES.index(name))
                source = {"map": pygame.image.load(image), "stock": stock, "typical": None}
            self._sources[name] = source
        return self._sources[name]

    def _cell(self, stage, col: int, row: int, cols: int, rows: int) -> pygame.Rect:
        return pygame.Rect(_MAP_ORIGIN + (col // cols) * _MAP_PITCH + (col % cols) * 16,
                           _MAP_ORIGIN + (row // rows) * _MAP_PITCH + (row % rows) * 16, 16, 16)

    def _typical(self, stage, cols: int, rows: int) -> dict:
        """The most common 16 px tile of each terrain kind in the untouched stage."""
        if stage["typical"] is None:
            seen: dict = {}
            image, stock = stage["map"], stage["stock"]
            for row, line in enumerate(stock):
                for col, glyph in enumerate(line):
                    kind = terrain(glyph)
                    if kind == " ":
                        continue
                    rect = self._cell(stage, col, row, cols, rows)
                    if image.get_rect().contains(rect):
                        data = pygame.image.tostring(image.subsurface(rect), "RGB")
                        seen.setdefault(kind, Counter())[data] += 1
            stage["typical"] = {kind: pygame.image.fromstring(count.most_common(1)[0][0], (16, 16), "RGB")
                                for kind, count in seen.items()}
        return stage["typical"]

    def _paint(self, stage, grid, sx: int, sy: int, ox: int, oy: int, cols: int, rows: int):
        image, stock = stage["map"], stage["stock"]
        out = pygame.Surface((cols * 16, rows * 16))
        src = pygame.Rect(_MAP_ORIGIN + (sx + ox) * _MAP_PITCH, _MAP_ORIGIN + (sy + oy) * _MAP_PITCH,
                          cols * 16, rows * 16)
        if image.get_rect().contains(src):
            out.blit(image, (0, 0), src)
        for r in range(rows):
            for c in range(cols):
                col, row = sx * cols + c, sy * rows + r
                now = terrain(grid[row][col]) if row < len(grid) and col < len(grid[row]) else " "
                col, row = col + ox * cols, row + oy * rows          # the same cell on the full stage
                was = terrain(stock[row][col]) if row < len(stock) and col < len(stock[row]) else " "
                if now != was:
                    tile = self._typical(stage, cols, rows).get(now)
                    if tile is not None:
                        out.blit(tile, (c * 16, r * 16))
                    else:
                        out.fill((0, 0, 0), (c * 16, r * 16, 16, 16))
        return pygame.transform.scale(out, (int(cols * 16 * self.scale), int(rows * 16 * self.scale)))

    # ── sounds ───────────────────────────────────────────────────────────────
    def play(self, events) -> None:
        """Play this frame's sound events (once each). Silent if the mixer or the files are missing."""
        if self._sounds is None:
            self._sounds = {}
            try:
                if not pygame.mixer.get_init():
                    pygame.mixer.init()
                for name, stem in SOUNDS.items():
                    path = os.path.join(self.root, "sound", stem + ".wav")
                    if os.path.exists(path):
                        self._sounds[name] = pygame.mixer.Sound(path)
                        self._sounds[name].set_volume(0.5)
            except pygame.error:
                self._sounds = {}
        for name in dict.fromkeys(events):
            if name in self._sounds:
                self._sounds[name].play()
