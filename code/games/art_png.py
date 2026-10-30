"""Our own sprites as PNG files, one per sprite, to edit in any pixel editor (Aseprite, Piskel, ...).

The art modules draw each sprite as text, rows of palette letters (bomberman_art.py, megaman_art.py,
mario_art.py), or as shapes in code (meatboy_art.py). This writes them out at one pixel per art pixel, and the games draw a sprite's PNG in place
of its text whenever the file is there: edit the PNG and the game shows the edit.

    python -m code.games.art_png                    # write the PNGs that are not there yet
    python -m code.games.art_png --force            # rewrite them all from the text (loses PNG edits)

Files: code/games/assets/art/<game>/<name>.png. Keep the size; transparent pixels stay transparent.
"""
from __future__ import annotations

import argparse
import os
from functools import lru_cache

import pygame

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "art")


def named(prefix: str, value, out: dict) -> dict:
    """Every sprite (a list of strings) inside value, named by where it sits: PLAYER[(2, 1)] -> player_2_1."""
    if isinstance(value, (list, tuple)) and value and all(isinstance(r, str) for r in value):
        out.setdefault(prefix, list(value))
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            named(f"{prefix}_{i}", v, out)
    elif isinstance(value, dict):
        for k, v in value.items():
            key = "_".join(map(str, k)) if isinstance(k, tuple) else str(k)
            # one letter keys differ only by case on some games ("b" vs "B"): keep them apart on disk
            named(f"{prefix}_{key}{'_up' if key.isalpha() and key.isupper() and len(key) == 1 else ''}", v, out)
    return out


def index(sprites: dict) -> dict:
    """rows -> name, the first name for each distinct picture (an alias shares its file)."""
    out = {}
    for name, rows in sprites.items():
        out.setdefault(tuple(rows), name)
    return out


def path(game: str, name: str) -> str:
    return os.path.join(ROOT, game, f"{name}.png")


@lru_cache(maxsize=1024)
def load(game: str, name: str):
    """The sprite's PNG at one pixel per art pixel, or None when there is no file."""
    p = path(game, name)
    return pygame.image.load(p) if name and os.path.isfile(p) else None


def to_surface(rows, pal: dict) -> pygame.Surface:
    w = max(len(r) for r in rows)
    out = pygame.Surface((w, len(rows)), pygame.SRCALPHA, 32)
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch != "." and ch in pal:
                out.set_at((x, y), pal[ch])
    return out


def export(game: str, sprites: dict, pal: dict, force: bool = False) -> int:
    os.makedirs(os.path.join(ROOT, game), exist_ok=True)
    n = 0
    for rows, name in index(sprites).items():
        if force or not os.path.isfile(path(game, name)):
            pygame.image.save(to_surface(rows, pal), path(game, name))
            n += 1
    return n


def export_surfaces(game: str, surfaces: dict, force: bool = False) -> int:
    """The same, for art drawn as images rather than text (meatboy_art.py)."""
    os.makedirs(os.path.join(ROOT, game), exist_ok=True)
    n = 0
    for name, img in surfaces.items():
        if force or not os.path.isfile(path(game, name)):
            pygame.image.save(img, path(game, name))
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="rewrite every PNG from the text, losing edits")
    args = ap.parse_args()
    from . import bomberman_art, mario_art, megaman_art, speedy_art
    for game, mod in (("bomberman", bomberman_art), ("megaman", megaman_art), ("mario", mario_art), ("speedy", speedy_art)):
        n = export(game, mod.all_sprites(), mod.PAL, args.force)
        print(f"{game}: {n} written to {os.path.join(ROOT, game)}")
    from . import meatboy_art
    n = export_surfaces("meatboy", meatboy_art.all_surfaces(), args.force)
    print(f"meatboy: {n} written to {os.path.join(ROOT, 'meatboy')}")


if __name__ == "__main__":
    main()
