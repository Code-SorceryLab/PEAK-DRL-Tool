#!/usr/bin/env python3
"""Rebuild a Mega Man (NES) stage layout as a PEAK ASCII level, from your own ROM.

    python -m code.games.tools.mm1_extract_level --rom "megaman-data/Mega Man (USA).nes"

The ROM stores a stage as screens of 8 x 8 blocks (32 px each, column-major); a block is four
16 px tiles whose top two bits give the collision class. The stage is a chain of "stripes": runs
of screens scrolled horizontally, linked to the next run by going up, down, or through a shutter.
This tool stitches the screens into one grid at their real relative positions, so a vertical link
puts the next screen directly above or below. Cells outside every screen stay blank (void).

Table addresses and formats follow https://github.com/TheRealQuantam/mm1dasm (globals.inc,
bank7.s, bank6.s). Only the layout and the enemy placement are read: no graphics. Nothing from
the ROM is copied except the tile classes and enemy types this writes out as glyphs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PRG_BANK = 0x4000
HEADER = 16
ROOM_BLOCK_DATA = 0x8000      # 4 bytes per 32 px block: one per 16 px tile, class in bits 7-6
ROOM_ORDER_TABLE = 0x8C00     # screen number -> room index
ROOM_POINTER_TABLE = 0x8C30   # room index -> pointer to its 64 block bytes
ROOM_LAYOUT_TABLE = 0x8C70    # one byte per stripe, starting at +1

ENEMY_BANK = 6
ENEMY_DATA_POINTERS = 0xA452  # per stage: pointer to (screen, x, y, type) entries, ended by $FF

SCREEN_COLS, SCREEN_ROWS = 16, 15   # 256 x 240 px; the 16th tile row of a room is never shown

LINK_UP, LINK_DOWN, LINK_SHUTTER = 0x80, 0x40, 0x20

STAGES = ["cutman", "iceman", "bombman", "fireman", "elecman", "gutsman"]   # CurrentStage 0..5
BOSS_SCREEN = [0x17, 0x17, 0x17, 0x13, 0x17, 0x12]                           # RoomBossOrder
# BlockTransparencyMap (bank7.s): what each stage's four collision classes mean.
#   0 air · 1 solid · 2 ladder · 3 spikes · 4 ice · 5 and 6 are stage specials (not handled here)
TILE_KINDS = [
    (0, 1, 2, 3),   # Cut Man
    (0, 1, 4, 5),   # Ice Man
    (0, 1, 2, 3),   # Bomb Man
    (0, 1, 2, 6),   # Fire Man
    (0, 1, 2, 0),   # Elec Man
    (0, 1, 3, 0),   # Guts Man
]
GLYPH = {0: ".", 1: "#", 2: "H", 3: "^", 4: "#"}
# Object type (EnemyAIaddr, bank5.s) -> glyph. A glyph marks the 16 px cell the object's centre
# falls in; one whose cell is terrain (it sits on that floor, or starts inside a wall) moves up.
ENEMY_GLYPH = {
    0x00: "b",   # Blader
    0x02: "f",   # Flea
    0x04: "v",   # Octopus Battery, moving vertically
    0x21: "h",   # Octopus Battery, moving horizontally
    0x05: "w",   # Screw Bomber on the floor
    0x16: "W",   # Screw Bomber on the ceiling
    0x09: "e",   # Big Eye
    0x0A: "s",   # Flying Shell spawner
    0x0B: "k",   # Beak on a right-hand wall, firing left
    0x22: "j",   # ... the same, 38 frames further into its cycle
    0x17: "K",   # Beak on a left-hand wall, firing right
    0x23: "J",   # ... the same, 38 frames further into its cycle
    0x1C: "c",   # Super Cutter spawner
    0x40: "+",   # large life energy capsule
}


def extract(rom: bytes, stage: int) -> tuple[list[str], dict]:
    if rom[:4] != b"NES\x1a" or len(rom) < HEADER + 8 * PRG_BANK:
        raise SystemExit("not an iNES Mega Man ROM (expected a 16-byte header and 8 PRG banks)")
    bank = rom[HEADER + stage * PRG_BANK: HEADER + (stage + 1) * PRG_BANK]

    def rd(addr: int) -> int:
        return bank[addr - 0x8000]

    n_screens = BOSS_SCREEN[stage] + 1
    kinds = TILE_KINDS[stage]
    unknown = set()

    def room(screen: int) -> list[list[str]]:
        idx = rd(ROOM_ORDER_TABLE + screen)
        ptr = rd(ROOM_POINTER_TABLE + idx * 2) | rd(ROOM_POINTER_TABLE + idx * 2 + 1) << 8
        g = [["."] * SCREEN_COLS for _ in range(16)]
        for bx in range(8):
            for by in range(8):
                block = rd(ptr + bx * 8 + by)
                for dx in range(2):
                    for dy in range(2):
                        kind = kinds[rd(ROOM_BLOCK_DATA + block * 4 + dy + dx * 2) >> 6]
                        if kind not in GLYPH:
                            unknown.add(kind)
                        g[by * 2 + dy][bx * 2 + dx] = GLYPH.get(kind, ".")
        return g[:SCREEN_ROWS]

    # Walk the stripes: where each screen sits, and which screens end at a shutter.
    pos, shutters = {}, []
    x = y = screen = 0
    addr = ROOM_LAYOUT_TABLE + 1
    while screen < n_screens:
        link = rd(addr)
        addr += 1
        for _ in range((link & 0x1F) + 1):
            pos[screen] = (x, y)
            screen += 1
            x += 1
        if link & LINK_UP:
            x -= 1; y -= 1
        elif link & LINK_DOWN:
            x -= 1; y += 1
        elif link & LINK_SHUTTER:
            shutters.append(screen - 1)
        else:
            break
    if len(pos) != n_screens:
        raise SystemExit(f"layout table gave {len(pos)} screens, expected {n_screens}")

    min_x = min(p[0] for p in pos.values())
    min_y = min(p[1] for p in pos.values())
    pos = {s: (px - min_x, py - min_y) for s, (px, py) in pos.items()}
    width = (max(p[0] for p in pos.values()) + 1) * SCREEN_COLS
    height = (max(p[1] for p in pos.values()) + 1) * SCREEN_ROWS
    grid = [[" "] * width for _ in range(height)]
    for s, (sx, sy) in pos.items():
        for r, row in enumerate(room(s)):
            grid[sy * SCREEN_ROWS + r][sx * SCREEN_COLS: (sx + 1) * SCREEN_COLS] = row

    def stand_cell(screen: int, col: int) -> tuple[int, int]:
        """The air cell a body dropped down this screen column comes to rest in."""
        sx, sy = pos[screen]
        c = sx * SCREEN_COLS + col
        rows = range(sy * SCREEN_ROWS, (sy + 1) * SCREEN_ROWS)
        air = [r for r in rows if grid[r][c] == "."]
        for r in air:
            if r + 1 in rows and grid[r + 1][c] in "#H" and (r - 1 not in rows or r - 1 in air or r == air[0]):
                return r, c
        raise SystemExit(f"screen {screen:#x}: nowhere to stand in column {col}")

    # The player beams in at x = $80 of the first screen ...
    r, c = stand_cell(0, 8)
    grid[r][c] = "P"
    # ... where the boss drops in at x = $C0, y = $B4 (bank7.s, 1DBC4).
    sx, sy = pos[n_screens - 1]
    grid[sy * SCREEN_ROWS + 0xB4 // 16][sx * SCREEN_COLS + 0xC0 // 16] = "X"

    # A shutter fills the doorway in the last column of the screen before it.
    for s in shutters:
        sx, sy = pos[s]
        c = (sx + 1) * SCREEN_COLS - 1
        top, bottom = sy * SCREEN_ROWS, (sy + 1) * SCREEN_ROWS
        r = top
        while r < bottom:
            if grid[r][c] != ".":
                r += 1
                continue
            end = r
            while end < bottom and grid[end][c] == ".":
                end += 1
            if r > top and end < bottom:   # a gap walled above and below: the doorway
                for rr in range(r, end):
                    grid[rr][c] = "D"
            r = end

    # Enemies: four bytes each, sorted by screen.
    data = rom[HEADER + ENEMY_BANK * PRG_BANK: HEADER + (ENEMY_BANK + 1) * PRG_BANK]
    ptr = data[ENEMY_DATA_POINTERS - 0x8000 + stage * 2] | data[ENEMY_DATA_POINTERS - 0x8000 + stage * 2 + 1] << 8
    placed, skipped = 0, []
    while data[ptr - 0x8000] != 0xFF:
        screen, ex, ey, kind = data[ptr - 0x8000: ptr - 0x8000 + 4]
        ptr += 4
        if screen not in pos:
            continue                                   # belongs to the Wily stage sharing this table
        sx, sy = pos[screen]
        c, r = sx * SCREEN_COLS + ex // 16, sy * SCREEN_ROWS + min(ey // 16, SCREEN_ROWS - 1)
        while r > sy * SCREEN_ROWS and grid[r][c] != ".":
            r -= 1
        if kind in ENEMY_GLYPH and grid[r][c] == ".":
            grid[r][c] = ENEMY_GLYPH[kind]
            placed += 1
        else:
            skipped.append(f"{kind:02X}@{screen:02X}")

    info = {"screens": n_screens, "cols": width, "rows": height, "shutters": len(shutters),
            "unknown_kinds": sorted(unknown), "enemies": placed, "skipped": skipped}
    return ["".join(row) for row in grid], info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rom", required=True, help="path to your Mega Man (USA) .nes file")
    ap.add_argument("--stage", default="cutman", choices=STAGES)
    ap.add_argument("--out", default=None, help="output .txt (default: code/games/levels/megaman/NN_<stage>.txt)")
    args = ap.parse_args()

    stage = STAGES.index(args.stage)
    lines, info = extract(Path(args.rom).read_bytes(), stage)
    out = Path(args.out) if args.out else \
        Path(__file__).resolve().parent.parent / "levels" / "megaman" / f"{stage + 1:02d}_{args.stage}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")   # no trailing newline: the editor reads one as an extra row
    print(f"{args.stage}: {info['screens']} screens -> {info['cols']} x {info['rows']} tiles, "
          f"{info['shutters']} shutters, {info['enemies']} enemies and items -> {out}")
    if info["skipped"]:
        print(f"warning: no glyph or no free cell for {', '.join(info['skipped'])} (type@screen)", file=sys.stderr)
    if info["unknown_kinds"]:
        print(f"warning: tile kinds {info['unknown_kinds']} have no glyph yet and were written as air",
              file=sys.stderr)


if __name__ == "__main__":
    main()
