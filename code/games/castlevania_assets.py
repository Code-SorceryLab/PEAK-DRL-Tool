"""Optional art for Castlevania: the reference maps' own tile layers, drawn behind the game, and
the reference sprite sheets for Simon, the enemies, the candles and the items, and its sounds.

The folder (castlevania-data/, git-ignored) holds a C++ remake's content: Tiled maps (a tile
sheet and a grid of tile numbers each) and sprite sheets with an atlas file naming each frame.
Without the folder the game draws flat placeholder shapes.
"""
from __future__ import annotations

import os
import re

import pygame

_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


class Backdrop:
    _cache: dict = {}

    @classmethod
    def get(cls, cfg: dict | None, tile: int) -> "Backdrop | None":
        cfg = cfg or {}
        if not cfg.get("enabled", True):
            return None
        root = os.environ.get("PEAK_CV_ASSETS") or os.path.join(_REPO, cfg.get("dir", "castlevania-data"))
        folder = os.path.join(root, cfg.get("maps", ""))
        key = (folder, tile)
        if key not in cls._cache:
            try:
                cls._cache[key] = cls(folder, cfg.get("backdrops") or [], tile)
            except (OSError, ValueError, pygame.error):
                cls._cache[key] = None
        return cls._cache[key]

    def __init__(self, folder: str, maps: list, tile: int) -> None:
        self.tile = tile
        self.pieces = []                # (surface, world x, world y)
        self.cells = set()              # every (col, row) the art covers
        for m in maps:
            with open(os.path.join(folder, m["file"]), encoding="utf-8") as f:
                text = f.read()
            sheet = re.search(r'<tileset firstgid="(\d+)"[^>]*tilewidth="(\d+)"[^>]*columns="(\d+)">\s*<image source="([^"]+)"', text)
            layer = re.search(r'<layer [^>]*width="(\d+)" height="(\d+)".*?<data encoding="csv">(.*?)</data>', text, re.S)
            if not sheet or not layer:
                raise ValueError(f"{m['file']}: no tile layer")
            first, size, columns = int(sheet.group(1)), int(sheet.group(2)), int(sheet.group(3))
            image = pygame.image.load(os.path.join(folder, sheet.group(4)))
            w, h = int(layer.group(1)), int(layer.group(2))
            gids = [int(v) for v in layer.group(3).replace("\n", "").split(",") if v.strip()]
            art = pygame.Surface((w * size, h * size))
            for i, gid in enumerate(gids):
                if gid >= first:
                    n = gid - first
                    art.blit(image, (i % w * size, i // w * size), (n % columns * size, n // columns * size, size, size))
            per = size // 32                                    # the maps are drawn at 32 px per stage cell
            rows = min(int(m.get("rows", h * per)), h * per)
            art = art.subsurface((0, 0, w * size, rows * 32))
            if tile != 32:
                art = pygame.transform.scale(art, (w * per * tile, rows * tile))
            self.pieces.append((art, int(m["col"]) * tile, int(m["row"]) * tile))
            self.cells |= {(int(m["col"]) + c, int(m["row"]) + r) for c in range(w * per) for r in range(rows)}

    def covers(self, col: int, row: int) -> bool:
        return (col, row) in self.cells

    def draw(self, surface, camx: int, camy: int) -> None:
        for art, x, y in self.pieces:
            surface.blit(art, (x - camx, y - camy))


class Sprites:
    """Frames cut from the reference sheets, by name ("simon/walk_01", "zombie/walk_02", "heart").
    The sheets are drawn at two screen px per NES px, which is the game's own scale at 32 px tiles."""
    _cache: dict = {}
    # name prefix -> sheet (an atlas beside it names the frames; a sheet without one is a single frame)
    SHEETS = {"simon": "Characters/Players/Simon", "zombie": "Characters/Enemies/Zombie",
              "panther": "Characters/Enemies/Panther", "bat": "Characters/Enemies/VampireBat",
              "fishman": "Characters/Enemies/Fishman", "boss": "Characters/Enemies/GiantBat",
              "candle": "Items/Candle", "brazier": "Items/Brazier", "whip": "Items/Whip", "money": "Items/Money_Bag",
              "orb": "Items/Crystal_Ball", "flame": "Effects/Flame", "heart": "Items/Small_Heart",
              "big_heart": "Items/Large_Heart", "whip_item": "Items/Whip_Powerup", "fireball": "Weapons/Fireball"}

    @classmethod
    def get(cls, cfg: dict | None, tile: int) -> "Sprites | None":
        cfg = cfg or {}
        if not cfg.get("enabled", True) or not cfg.get("sprites"):
            return None
        root = os.environ.get("PEAK_CV_ASSETS") or os.path.join(_REPO, cfg.get("dir", "castlevania-data"))
        key = (os.path.join(root, cfg["sprites"]), tile)
        if key not in cls._cache:
            try:
                cls._cache[key] = cls(*key)
            except (OSError, ValueError, pygame.error):
                cls._cache[key] = None
        return cls._cache[key]

    def __init__(self, folder: str, tile: int) -> None:
        self.frames: dict = {}
        self._flipped: dict = {}
        scale = tile / 32.0
        for prefix, sheet in self.SHEETS.items():
            image = pygame.image.load(os.path.join(folder, sheet + ".png"))
            if image.get_alpha() is None and image.get_colorkey() is None:
                image.set_colorkey(image.get_at((0, 0)))
            atlas = os.path.join(folder, sheet + ".atlas.xml")
            cuts = {}
            if os.path.exists(atlas):
                with open(atlas, encoding="utf-8") as f:
                    for m in re.finditer(r'<Sprite ID="([^"]+)">\s*<SpriteFrame Left="(-?\d+)" Top="(-?\d+)" Width="(\d+)" Height="(\d+)"', f.read()):
                        cuts[m.group(1)] = pygame.Rect(*(int(v) for v in m.groups()[1:]))
            for name, rect in (cuts or {"": image.get_rect()}).items():
                rect = rect.clip(image.get_rect())
                if not rect.width or not rect.height:
                    continue
                frame = image.subsurface(rect)
                if scale != 1.0:
                    frame = pygame.transform.scale(frame, (max(1, int(rect.width * scale)), max(1, int(rect.height * scale))))
                self.frames[f"{prefix}/{name}" if name else prefix] = frame

    def frame(self, name: str, flip: bool = False):
        img = self.frames.get(name)
        if img is None or not flip:
            return img
        if name not in self._flipped:
            self._flipped[name] = pygame.transform.flip(img, True, False)
        return self._flipped[name]


class Sounds:
    """The reference sound effects and music, by event name (castlevania_config.yaml, assets.sounds).
    Silent if the mixer or the files are missing."""
    _cache: dict = {}

    @classmethod
    def get(cls, cfg: dict | None) -> "Sounds | None":
        cfg = cfg or {}
        if not cfg.get("enabled", True) or not cfg.get("sounds"):
            return None
        root = os.environ.get("PEAK_CV_ASSETS") or os.path.join(_REPO, cfg.get("dir", "castlevania-data"))
        folder = os.path.join(root, cfg.get("sprites", ""))
        if folder not in cls._cache:
            cls._cache[folder] = cls(folder, cfg["sounds"], cfg.get("music") or {})
        return cls._cache[folder]

    def __init__(self, folder: str, sounds: dict, music: dict) -> None:
        self.effects: dict = {}
        self.music = {k: os.path.join(folder, v) for k, v in music.items() if os.path.exists(os.path.join(folder, v))}
        self.playing = None
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            for name, file in sounds.items():
                path = os.path.join(folder, file)
                if os.path.exists(path):
                    self.effects[name] = pygame.mixer.Sound(path)
                    self.effects[name].set_volume(0.5)
        except pygame.error:
            self.effects, self.music = {}, {}

    def play(self, events) -> None:
        for name in dict.fromkeys(events):          # each event once a frame
            if name in self.effects:
                self.effects[name].play()

    def track(self, name: str | None) -> None:
        """Loop this piece of music (None: silence). Asking for the one already playing does nothing."""
        if name == self.playing or (name and name not in self.music):
            return
        try:
            pygame.mixer.music.stop()
            if name:
                pygame.mixer.music.load(self.music[name])
                pygame.mixer.music.set_volume(0.35)
                pygame.mixer.music.play(-1)
            self.playing = name
        except pygame.error:
            self.music = {}
