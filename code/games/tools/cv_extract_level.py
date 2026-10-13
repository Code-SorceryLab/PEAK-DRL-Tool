#!/usr/bin/env python3
"""Rebuild Castlevania (NES) stage 1 as a PEAK ASCII level, from your own copy of the reference maps.

    python -m code.games.tools.cv_extract_level            # reads castlevania-data/, writes the stage file

The reference is the Tiled maps of a C++ remake of block 1 (github.com/NearHuscarl/Castlevania,
kept locally in castlevania-data/code): the courtyard, the great hall and the underground. Each
map is a set of collision rectangles, stair triggers and object positions on a 32 px grid (one
NES tile at 2x). This tool stitches the three maps into one grid of 16 x 12-tile screens:

    row 0   courtyard (3 screens) | great hall (11 screens, two doors)
    row 1                           the underground, below the hall's middle room

The underground sits two tiles right of the hall's origin, which is where its two staircases
meet the hall's; the rest of its three screens is filled solid. The courtyard is one tile lower
than in its own map so that its ground meets the hall's.

Only the layout and the placements are read: no graphics. What the game does not model yet
(sub-weapons and their pickups, the hidden money bags, the breakable walls) is left out: a candle
that held one of those drops a small heart, a breakable block is plain floor.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

TILE = 32
SCREEN_COLS, SCREEN_ROWS = 16, 12
MAPS = "code/src/GameCuaTao/Castlevania/Content/TiledMaps/Stage_01"
# map -> (file, column, row) of its top-left corner in the stitched grid
PLACE = {"courtyard": ("Courtyard.tmx", 0, 1), "hall": ("Great_Hall.tmx", 48, 0), "underground": ("Underground.tmx", 146, 12)}
COLS, ROWS = 224, 24
ITEM_GLYPH = {"SmallHeart": "c", "LargeHeart": "C", "WhipPowerup": "w",
              "WhiteMoneyBag": "m", "RedMoneyBag": "m", "BlueMoneyBag": "m"}   # anything else: a small heart
# Where the enemies that arrive in waves come from: (map, column, row) of each spawner, at the
# middle of the remake's spawn area (castlevania_config.yaml says how far each one reaches).
SPAWNERS = [("Z", "hall", 12, 9), ("Z", "hall", 83, 9), ("Z", "hall", 140, 9),
            ("B", "hall", 111, 1), ("F", "underground", 15, 8)]


def objects(text: str) -> dict[str, list[dict]]:
    """{layer name: [object]} with x, y, width, height as numbers and the custom properties inline."""
    out: dict[str, list[dict]] = {}
    for group in re.finditer(r'<objectgroup[^>]*name="([^"]+)"[^>]*>(.*?)</objectgroup>', text, re.S):
        items = out.setdefault(group.group(1), [])
        for obj in re.finditer(r'<object ([^>]*?)/?>(.*?)(?=<object |\Z)', group.group(2), re.S):
            o = dict(re.findall(r'(\w+)="([^"]*)"', obj.group(1)))
            o.update(dict(re.findall(r'<property name="([^"]+)"[^>]*value="([^"]*)"', obj.group(2))))
            for k in ("x", "y", "width", "height"):
                o[k] = float(o.get(k, 0))
            items.append(o)
    return out


def build(root: Path) -> list[str]:
    grid = [[" "] * COLS for _ in range(ROWS)]

    def fill(c0, r0, c1, r1, ch):                       # inclusive cells, clipped to the grid
        for r in range(max(0, r0), min(ROWS, r1 + 1)):
            for c in range(max(0, c0), min(COLS, c1 + 1)):
                grid[r][c] = ch

    fill(0, 0, COLS - 1, SCREEN_ROWS - 1, ".")          # the top row of screens: courtyard and hall
    fill(0, 10, COLS - 1, 11, "#")                      # their ground, two tiles thick
    fill(144, 12, 191, 23, "#")                         # the three screens the underground sits in
    stairs = []                                         # (bottom x, bottom y, +1 rising right / -1 left, cells)

    for name, (file, ox, oy) in PLACE.items():
        layers = objects((root / MAPS / file).read_text(encoding="utf-8"))
        rows = 12 if name != "courtyard" else 11        # the courtyard's last row is under its ground
        if name == "underground":
            fill(ox, oy, ox + 31, oy + 9, ".")
            fill(ox, oy + 10, ox + 31, oy + 11, "~")    # water
        for o in layers.get("Bounds", []):
            if o["width"] < 16:                         # the hairline borders at a map's edge
                continue
            c0, r0 = int(o["x"] // TILE), int(o["y"] // TILE)
            c1, r1 = int((o["x"] + o["width"] - 1) // TILE), int((o["y"] + o["height"] - 1) // TILE)
            fill(ox + c0, oy + r0, ox + c1, oy + min(r1, rows - 1), "#")
        for o in layers.get("Entities", []):
            col = ox + int((o["x"] + o["width"] / 2) // TILE)
            top = oy + int((o["y"] - o["height"]) // TILE)          # a Tiled tile object hangs from its bottom-left corner
            row = oy + int((o["y"] - 1) // TILE)
            mid = oy + int((o["y"] - min(o["height"], 32) / 2) // TILE)   # the cell its lower half is in
            kind = o.get("type")
            if kind in ("Candle", "Brazier"):
                grid[mid][col] = ITEM_GLYPH.get(o.get("Item", ""), "c")
            elif kind in ("BreakableBlock", "BreakableWall"):
                fill(col, top, col, row, "#")
            elif kind == "Door":
                fill(col, top, col, row, "D")
            elif kind == "SpawnPoint":
                grid[mid][col] = {"Panther": "p", "GiantBat": "X"}[o["SpawnObject"]]
        ups = [o for o in layers.get("Triggers", []) if o.get("name") == "StairUp"]
        downs = [o for o in layers.get("Triggers", []) if o.get("name") == "StairDown"]
        tops = {(d["x"] + 16, d["y"] + 32) for d in downs}
        used = set()
        for u in ups:                                   # a flight runs from its foot to the matching head
            way = 1 if u.get("Facing") == "Right" else -1
            bx, by = u["x"] + 16, u["y"] + 32
            n = next((k for k in range(1, 13) if (bx + way * k * TILE, by - k * TILE) in tops), None)
            if n is None:                               # no head in this map: it leaves through the top
                n = int(by // TILE)
            else:
                used.add((bx + way * n * TILE, by - n * TILE))
            stairs.append((ox + int(bx // TILE), oy + int(by // TILE), way, n))
        for d in downs:                                 # a head with no foot: it leaves through the floor
            tx, ty = d["x"] + 16, d["y"] + 32
            if (tx, ty) not in used:
                way = -1 if d.get("Facing") == "Right" else 1      # it descends the way it faces
                n = rows - int(ty // TILE)
                stairs.append((ox + int(tx // TILE) - way * n, oy + int(ty // TILE) + n, way, n))
        entry = next((o for o in layers.get("Locations", []) if o.get("name") == "Entrypoint"), None)
        if name == "courtyard" and entry:
            grid[oy + int((entry["y"] - 1) // TILE)][ox + int((entry["x"] + entry["width"] / 2) // TILE)] = "P"

    for bc, br, way, n in stairs:                       # (bc, br): the grid corner the foot stands on
        for k in range(n):
            c = bc + k if way > 0 else bc - 1 - k
            grid[br - 1 - k][c] = "/" if way > 0 else "\\"
    fill(47, 0, 47, 7, "#")                             # the castle wall, and the way in under it
    fill(47, 8, 47, 9, "D")
    for glyph, name, col, row in SPAWNERS:
        _, ox, oy = PLACE[name]
        if grid[oy + row][ox + col] != ".":
            raise SystemExit(f"spawner {glyph} at {name} ({col}, {row}) is not in open air")
        grid[oy + row][ox + col] = glyph
    return ["".join(r).rstrip() if any(ch != " " for ch in r) else "" for r in grid]


def main() -> None:
    repo = Path(__file__).resolve().parents[3]
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=str(repo / "castlevania-data"), help="the reference folder (default: castlevania-data/)")
    ap.add_argument("--out", default=str(repo / "code/games/levels/castlevania/01_entrance.txt"))
    args = ap.parse_args()
    root = Path(args.data)
    if not (root / MAPS).is_dir():
        sys.exit(f"{root / MAPS} not found — the reference maps are not in this repository")
    rows = build(root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"wrote {out} ({COLS} x {ROWS} tiles, {COLS // SCREEN_COLS} x {ROWS // SCREEN_ROWS} screens)")


if __name__ == "__main__":
    main()
