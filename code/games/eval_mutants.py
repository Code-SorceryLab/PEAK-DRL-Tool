"""The AST evaluation mutants: five one-change copies of each game's evaluation level.

    python -m code.games.eval_mutants                 # write Mario's, Meat Boy's, Bomberman's
    python -m code.games.eval_mutants --list          # the evaluation set of every game

    operator             what it changes                                   class       agents
    seal_route           a full-height wall across the route               breaking    retrained
    widen_gap            a wider gap (Bomberman add_enemy: one more enemy)  degrading   retrained
    jump_velocity_down   jump x0.85 (Bomberman fuse_down: a shorter fuse)   degrading   retrained
    off_route            a tile no player can reach                        neutral     retrained
    cosmetic_tile        '.' written as ' ': the level parses the same     equivalent  replayed only (RQ2)

The changes are fixed by hand (CHANGES), not searched for. Every cell names the character it
expects and every value the number the game uses now; a level or config that no longer matches is
refused, never mutated in the wrong place. Each mutant is then loaded through the game's own hook
(code/games/mutants.py) and checked before it is written: the game plays the change, a cosmetic
copy parses to the same level, an off-route tile is walled in on all four sides, and a flood fill
over the open tiles reaches the exit for every mutant except seal_route.

Meat Boy's cells differ from the spec, whose coordinates fit the June layout of The Gap (42 x 14,
flat floor, 14-tile gap). Kevin redrew it on 13 Oct (33 x 16, a cave with a 12-tile gap); the same
five changes are placed on that layout, and each record's `note` says how.
Mega Man's five are segment 3 (Ledges) of code/games/megaman_mutants.py plus its global jump change.
Mutants go to runs/<game>/mutants/<id>/ (mutant.json, level.txt, level.yaml when the level has one).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from contextlib import contextmanager

from . import mutants

LEVELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "levels")

# The evaluation level of each game: (level id, what to call it)
EVAL_LEVELS = {
    "mario": ("Mario1-1", "Super Mario Bros. 1-1"),
    "meatboy": ("2", "Super Meat Boy 1-3 The Gap"),
    "bomberman": ("12", "Bomberman II 1-1"),
    "megaman": ("CutMan", "Mega Man, the Cut Man stage"),
}
MEGAMAN_SET = ["s3-seal_route", "s3-widen_gap", "m-jump_velocity_down", "s3-off_route", "s3-cosmetic_tile"]


def column(col: int, rows: range, before: str, after: str) -> list:
    return [(r, col, b, after) for r, b in zip(rows, before)]


# Level changes are cells (row, col, before, after), counted from 0 as in the spec; mechanics
# changes are config values (dotted key, before, after). id, class, change, description, note.
CHANGES = {
    "mario": ("platformer/world1_1.txt", [
        ("seal_route", "breaking", column(190, range(14), "." * 14, "#"),
         "A wall down column 190, rows 0-13, between the last staircase and the flagpole: the flag "
         "can no longer be reached", None),
        ("widen_gap", "degrading",
         [(14, c, "#", ".") for c in (89, 90)] + [(r, c, "#", "O") for r in range(15, 22) for c in (89, 90)],
         "The second pit (columns 86-88) widened by 2 to the right: 3 tiles -> 5", None),
        ("jump_velocity_down", "degrading",
         [("player.jump.min_velocity", -495.0, -420.75), ("player.jump.max_velocity", -800.0, -680.0)],
         "Jump velocity x0.85: take-off -495 -> -420.75 px/s, the cap -800 -> -680",
         "The spec edits physics.jump_velocity in world1_1.yaml, but the game never reads that block "
         "(LevelLoader takes only `dynamics` from it), so that edit would change nothing. This scales "
         "the jump values the game does read; the small run-speed bonus (up to 57 px/s) is unchanged. "
         "A full running jump rises 3.75 tiles (5 before). 1-1's pipes at columns 46 and 57 are 4 tall, "
         "as in SMB: the 3-tall pipe at 38 is a step up onto 46, but nothing steps up onto 57 (even "
         "x0.9 cannot clear it from the ground), so only a Goomba bounce may get past it."),
        ("off_route", "neutral", [(19, 75, "#", ".")],
         "The ground tile at row 19, column 75 hollowed out, 5 tiles under the surface", None),
        ("cosmetic_tile", "neutral", [(2, 10, ".", " ")],
         "The sky cell at row 2, column 10 written as ' ' instead of '.': the same tile", None),
    ]),
    "meatboy": ("meatboy/world1_03_the_gap.txt", [
        ("seal_route", "breaking", column(28, range(6, 11), ".....", "#"),
         "A wall down column 28, rows 6-10, from the ceiling to the floor just before the goal: "
         "the goal is closed in",
         "Spec: column 33, rows 0-8, on the June layout. Column 28 is the last one before the goal "
         "and meets the roof at row 5, so no wall jump can climb over it."),
        ("widen_gap", "degrading",
         [(11, 8, "#", "."), (11, 9, "#", "."), (12, 7, "#", "."), (12, 8, "#", "."), (13, 6, "#", "."),
          (13, 7, "#", "."), (14, 5, "#", "."), (14, 6, "#", "."), (15, 5, "#", "."), (15, 6, "#", ".")],
         "The take-off side of the gap cut back by 2 tiles in every floor row: 12 tiles -> 14 at the surface",
         "Spec: columns 12-13, rows 9-13 (gap 14 -> 16) on the June layout. Here the gap's left wall "
         "steps out one tile per row, so each row loses the 2 tiles next to the gap."),
        ("jump_velocity_down", "degrading", [("jump.jump_vel", 480.0, 408.0)],
         "Jump velocity x0.85: jump.jump_vel 480 -> 408",
         "The spec edits meatboy_config.yaml and reverts it; the mutant sets the value only in the "
         "process that plays it, so the shared config never changes."),
        ("off_route", "neutral", [(12, 5, "#", ".")],
         "The floor tile at row 12, column 5 hollowed out, under the start ledge and closed on all four sides",
         "The spec's cell. On this layout it lies 1 tile under the surface (3 on the June one); the "
         "agents' rays and pit probe stop at the first solid tile, so they never reach it."),
        ("cosmetic_tile", "neutral", [(0, 6, ".", " ")],
         "The air cell at row 0, column 6 written as ' ' instead of '.': the same tile",
         "Spec: row 0, column 0, a wall tile on this layout. Column 6 is the top row's first air cell."),
    ]),
    "bomberman": ("bomberman/bm2_1_1.txt", [
        ("seal_route", "breaking", [(5, 9, "?", "#"), (5, 11, ".", "#")],
         "Hard walls at row 5, columns 9 and 11 (a brick and a floor tile) close the exit into its pocket",
         "The exit is under the brick at row 5, column 10; pillars stand above and below it."),
        ("add_enemy", "degrading", [(6, 7, ".", "E")],
         "A fourth Ballom at row 6, column 7, the floor tile nearest the centre; every enemy must die before the exit opens", None),
        ("fuse_down", "degrading", [("levels.12.fuse", 150, 94)],
         "Bomb fuse 150 -> 94 frames on this level (the same x0.625 as before)", None),
        ("off_route", "neutral", [(0, 0, "#", ".")],
         "The frame's top-left corner hollowed out, closed on all four sides", None),
        ("cosmetic_tile", "neutral", [(1, 2, ".", " ")],
         "The floor cell at row 1, column 2 written as ' ' instead of '.': the same tile", None),
    ]),
}

# What the running game uses for each mechanics key, read from a built core
EFFECTIVE = {
    "player.jump.min_velocity": lambda core: core.physics_manager.context.JUMP_VEL_MIN,
    "player.jump.max_velocity": lambda core: core.physics_manager.context.JUMP_VEL_MAX,
    "jump.jump_vel": lambda core: core._jump["jump_vel"],
    "levels.8.fuse": lambda core: core.fuse_frames,
    "levels.12.fuse": lambda core: core.fuse_frames,
    "bomb.fuse_frames": lambda core: core.fuse_frames,
}
EMPTY = (".", " ")                  # air or floor in every game's level text
BOMBABLE = set("?@CFSRWXVDQ")      # Bomberman: a brick, or the exit or a power-up under one; a bomb opens it
GAP_FILL = set(".  O*%^")           # what may fill a gap: air, Mario's pit, Meat Boy's saws and spikes
FUSE_SCALE = 150 / 240              # Bomberman's fuse_down scale (94 of 150), applied to every level's own fuse


def _effective(key: str):
    """The reader for a mechanics key; every level's own fuse reads like level 8's."""
    return EFFECTIVE["levels.8.fuse" if key.startswith("levels.") and key.endswith(".fuse") else key]


def out_dir(game: str) -> str:
    return os.path.join("runs", game, "mutants")


def level_file(game: str, level: str) -> str:
    """The level's file under LEVELS_DIR, as the game's config names it (disabled levels too)."""
    import yaml
    games = os.path.dirname(LEVELS_DIR)
    if game == "mario":
        with open(os.path.join(games, "game_config.yaml"), encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return {**(data.get("disabled_levels") or {}), **(data.get("levels") or {})}[level]["file"]
    with open(os.path.join(games, f"{game}_config.yaml"), encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    entry = ((data.get("levels") or []) + (data.get("disabled_levels") or []))[int(level)]
    return entry if isinstance(entry, str) else entry["file"]


def level_name(game: str, level: str) -> str:
    """A level as people call it, from its file name: "Super Meat Boy 1-6 Bladecatcher"."""
    if game not in CHANGES:
        return level
    stem = os.path.splitext(os.path.basename(level_file(game, level)))[0]
    if game == "mario":
        return f"Super Mario Bros. {level.replace('Mario', '')}"
    if game == "meatboy":
        world, n, *rest = stem.replace("world", "").split("_")
        return f"Super Meat Boy {int(world)}-{int(n)} {' '.join(rest).title()}"
    n, *rest = stem.split("_")
    return f"Bomberman {n} {' '.join(rest).title()}"


def other_levels(game: str) -> list[str]:
    """Every enabled level but the evaluation one: their mutants are placed by program (place())."""
    from code.neuro.adapters import list_levels
    return [lv for lv in list_levels(game) if lv != EVAL_LEVELS[game][0]]


# ── the game as an agent meets it ─────────────────────────────────────────────────────────────
@contextmanager
def _named(folder: str | None):
    """PEAK_MUTANT set to this folder (or unset) while the block builds a game."""
    saved = {k: os.environ.pop(k, None) for k in (mutants.ENV, mutants.OLD_ENV)}
    if folder:
        os.environ[mutants.ENV] = folder
    try:
        yield
    finally:
        os.environ.pop(mutants.ENV, None)
        os.environ.update({k: v for k, v in saved.items() if v is not None})


def _load(game: str, folder: str | None = None, level: str | None = None):
    from code.neuro.adapters import make_adapter
    with _named(folder):
        return make_adapter(game, level or EVAL_LEVELS[game][0], 600, 0.0)


def _layout(adapter):
    """What a player meets: every tile, the start, and how many of each kind of object."""
    ld = adapter.core.level_data
    counts = {k: len(getattr(ld, k)) for k in ("enemies", "coins", "qblocks", "goals", "saws", "spawns")
              if isinstance(getattr(ld, k, None), list)}
    start = getattr(ld, "player_start", None) or getattr(ld, "start", None)
    return [list(r) for r in ld.grid], tuple(start), counts


def _tile_open(adapter, c: int, r: int) -> bool:
    ts = adapter.tile_size
    return not adapter.solid_at((c + 0.5) * ts, (r + 0.5) * ts)


def _goal_tiles(adapter) -> set:
    core, ts = adapter.core, adapter.tile_size
    ld = core.level_data
    if hasattr(ld, "exit"):                                     # Bomberman
        return {tuple(ld.exit)}
    if getattr(core, "_goal_tiles", None):                       # Meat Boy
        return set(map(tuple, core._goal_tiles))
    return {(c, r) for g in ld.goals for c in range(int(g.gObj.x // ts), int((g.gObj.x + g.gObj.width - 1) // ts) + 1)
            for r in range(int(g.gObj.y // ts), int((g.gObj.y + g.gObj.height - 1) // ts) + 1)}


def _passable(adapter, c: int, r: int) -> bool:
    """Open, or (Bomberman) a tile a bomb opens: a brick, or the exit or a power-up under one."""
    ld = adapter.core.level_data
    return _tile_open(adapter, c, r) or (hasattr(ld, "exit") and ld.grid[r][c] in BOMBABLE)


def _start(adapter) -> tuple[int, int]:
    return int(adapter.x // adapter.tile_size), int(adapter.y // adapter.tile_size)


def _flood(adapter, walls=frozenset()) -> set:
    """The tiles a flood fill reaches from the start over passable tiles, `walls` taken as solid."""
    ld = adapter.core.level_data
    seen = {_start(adapter)}
    todo = list(seen)
    while todo:
        c, r = todo.pop()
        for nc, nr in ((c + 1, r), (c - 1, r), (c, r + 1), (c, r - 1)):
            if (0 <= nc < ld.cols and 0 <= nr < ld.rows and (nc, nr) not in seen and (nc, nr) not in walls
                    and _passable(adapter, nc, nr)):
                seen.add((nc, nr))
                todo.append((nc, nr))
    return seen


def exit_reachable(adapter, walls=frozenset()) -> bool:
    """A flood fill over the passable tiles inside the level, from the start. It ignores gravity and
    jump height, so 'no' proves the exit cut off; 'yes' only says nothing walls it off."""
    return bool(_goal_tiles(adapter) & _flood(adapter, walls))


# ── the five operators placed by program, on every level but the evaluation one ─────────────────
def _ch(rows: list, c: int, r: int) -> str | None:
    return rows[r][c] if 0 <= r < len(rows) and 0 <= c < len(rows[r]) else None


def _seal(game: str, rows: list, base, start, goals):
    """The wall nearest the goal that cuts it off: '#' over every empty cell of one column (or row)
    between start and goal, the axis they lie further apart on first. Bomberman: hard walls on the
    floor and bombable cells next to the exit."""
    if game == "bomberman":
        cells = sorted({(c, r) for gc, gr in goals for c, r in ((gc + 1, gr), (gc - 1, gr), (gc, gr + 1), (gc, gr - 1))
                        if _ch(rows, c, r) in set(".") | BOMBABLE - {"@"} and (c, r) != start})
        if cells and not exit_reachable(base, frozenset(cells)):
            return [(r, c, rows[r][c], "#") for c, r in cells], \
                f"Hard walls on the {len(cells)} tiles next to the exit close it in"
        return None, "walls on the exit's neighbours do not close it in"
    (sc, sr), gc, gr = start, sorted(c for c, _ in goals), sorted(r for _, r in goals)
    cols = range(gc[0] - 1, sc, -1) if gc[0] > sc else range(gc[-1] + 1, sc) if gc[-1] < sc else range(0)
    rws = range(gr[0] - 1, sr, -1) if gr[0] > sr else range(gr[-1] + 1, sr) if gr[-1] < sr else range(0)
    lines = [("column", n, [(n, r) for r in range(len(rows)) if _ch(rows, n, r) in EMPTY]) for n in cols]
    lines_r = [("row", n, [(c, n) for c in range(len(rows[n])) if _ch(rows, c, n) in EMPTY]) for n in rws]
    if abs(gr[0] - sr) > abs(gc[0] - sc):
        lines, lines_r = lines_r, lines
    for axis, n, cells in lines + lines_r:
        if cells and not set(cells) & goals and not exit_reachable(base, frozenset(cells)):
            return [(r, c, rows[r][c], "#") for c, r in cells], \
                f"A wall along {axis} {n}, {len(cells)} tiles: the goal can no longer be reached"
    return None, "no wall of one column or row between start and goal cuts the goal off"


def _widen(rows: list, base, start, goals):
    """The gap nearest the start widened by 2 tiles on its far side: a run of 1-8 open tiles in a
    floor row with a drop under it and a ledge on each side; the 2 ledge tiles take the gap edge's
    contents down to the gap's floor, and the ledge keeps at least one tile."""
    def solid(c, r):
        return _ch(rows, c, r) is not None and not _tile_open(base, c, r)

    (sc, sr), found = start, []
    for r in range(1, len(rows) - 1):
        c = 0
        while c < len(rows[r]):
            if solid(c, r) or _ch(rows, c, r) is None:
                c += 1
                continue
            c1 = c
            while c < len(rows[r]) and not solid(c, r):
                c += 1
            c2 = c - 1
            if (c2 - c1 < 8 and solid(c1 - 1, r) and solid(c2 + 1, r)
                    and all(not solid(x, r - 1) for x in range(c1 - 1, c2 + 2))
                    and all(not solid(x, r + 1) for x in range(c1, c2 + 1))):
                found.append((abs((c1 + c2) / 2 - sc) + abs(r - sr), c1, c2, r))
    for _, c1, c2, r in sorted(found):
        right = (c1 + c2) / 2 >= sc
        edge, new, keep = (c2, (c2 + 1, c2 + 2), c2 + 3) if right else (c1, (c1 - 1, c1 - 2), c1 - 3)
        if not solid(keep, r) or any(solid(n, r - 1) for n in new):
            continue
        change, ok, rr = [], True, r
        while ok and rr < len(rows) and not solid(edge, rr):
            fill = rows[rr][edge]
            for n in new:
                if (n, rr) == start or (n, rr) in goals or fill not in GAP_FILL:
                    ok = False
                elif solid(n, rr):
                    change.append((rr, n, rows[rr][n], fill))
            rr += 1
        if ok and change:
            return change, (f"The gap at row {r}, columns {c1}-{c2}, widened by 2 to the "
                            f"{'right' if right else 'left'}: {c2 - c1 + 1} tiles -> {c2 - c1 + 3}")
    return None, "no gap with a ledge on each side and a drop under it"


def _add_enemy(rows: list, base, start, goals):
    """One more Ballom on the floor tile nearest the level's centre, at least 4 tiles from the start."""
    (sc, sr), reach = start, _flood(base)
    floor = [(c, r) for c, r in reach if _ch(rows, c, r) == "." and abs(c - sc) + abs(r - sr) >= 4
             and (c, r) not in goals]
    if not floor:
        return None, "no floor tile 4 tiles from the start"
    mc, mr = (len(rows[0]) - 1) / 2, (len(rows) - 1) / 2
    c, r = min(floor, key=lambda t: ((t[0] - mc) ** 2 + (t[1] - mr) ** 2, t[1], t[0]))
    return [(r, c, ".", "E")], f"A Ballom at row {r}, column {c}"


def _off_route(rows: list, base, start, goals):
    """The '#' tile nearest the start whose four neighbours are solid (and, in Bomberman, not
    bombable), hollowed out. Outside the level is what the game says (solid in Bomberman, so a
    corner of a one-tile border qualifies; open where a platformer's player can fall out)."""
    (sc, sr), ld = start, base.core.level_data

    def closed(c, r):
        inside = 0 <= c < ld.cols and 0 <= r < ld.rows
        return not (_passable(base, c, r) if inside else _tile_open(base, c, r))

    cells = [(c, r) for r in range(len(rows)) for c in range(len(rows[r]))
             if rows[r][c] == "#" and closed(c, r)
             and all(closed(c + dc, r + dr) for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)))]
    if not cells:
        return None, "no solid tile closed on all four sides"
    c, r = min(cells, key=lambda t: ((t[0] - sc) ** 2 + (t[1] - sr) ** 2, t[1], t[0]))
    return [(r, c, "#", ".")], f"The tile at row {r}, column {c} hollowed out, closed on all four sides"


def _cosmetic(rows: list, base, start, goals):
    """The first '.' cell, row by row, written as ' '."""
    cell = next(((r, c) for r in range(len(rows)) for c in range(len(rows[r])) if rows[r][c] == "."), None)
    if cell is None:
        return None, "no '.' cell"
    return [(cell[0], cell[1], ".", " ")], f"The cell at row {cell[0]}, column {cell[1]} written as ' ' instead of '.': the same tile"


def place(game: str, level: str, text: list, base) -> tuple[list, list]:
    """The evaluation level's five operators placed on another level by rule, in CHANGES' format
    (id, class, change, description, note), and the operators no rule could place, with why."""
    rows, start, goals = [list(r) for r in text], _start(base), _goal_tiles(base)
    specs, skipped = [], []
    for mid, cls, change, description, _ in CHANGES[game][1]:
        if mid == "jump_velocity_down":                     # one game-wide value: the same change
            specs.append((mid, cls, change, description, None))
            continue
        if mid == "fuse_down":
            old = base.core.fuse_frames
            specs.append((mid, cls, [(_fuse_key(level), old, round(old * FUSE_SCALE))],
                          f"Bomb fuse {old} -> {round(old * FUSE_SCALE)} frames on this level", None))
            continue
        rule = {"seal_route": lambda: _seal(game, rows, base, start, goals), "widen_gap": lambda: _widen(rows, base, start, goals),
                "add_enemy": lambda: _add_enemy(rows, base, start, goals), "off_route": lambda: _off_route(rows, base, start, goals),
                "cosmetic_tile": lambda: _cosmetic(rows, base, start, goals)}[mid]
        cells, why = rule()
        if cells is None:
            skipped.append({"operator": mid, "reason": why})
        else:
            specs.append((mid, cls, cells, why, "placed by rule (eval_mutants.place)"))
    return specs, skipped


def _fuse_key(level: str) -> str:
    """Bomberman: a level written as {file: ...} takes its own fuse; a bare path plays bomb.fuse_frames."""
    import yaml
    with open(os.path.join(os.path.dirname(LEVELS_DIR), "bomberman_config.yaml"), encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    entry = ((data.get("levels") or []) + (data.get("disabled_levels") or []))[int(level)]
    return f"levels.{level}.fuse" if isinstance(entry, dict) else "bomb.fuse_frames"


# ── generating ───────────────────────────────────────────────────────────────────────────────
def build(game: str, out: str | None = None, level: str | None = None) -> dict:
    """Write the game's five mutants, each checked through the game itself. Raises on a mismatch.
    The evaluation level takes CHANGES' hand-placed cells; any other `level` takes place()'s, and
    its mutants are named <operator>@<level>."""
    out = out or out_dir(game)
    level = level or EVAL_LEVELS[game][0]
    hand = level == EVAL_LEVELS[game][0]
    file, specs = CHANGES[game] if hand else (level_file(game, level), None)
    path = os.path.join(LEVELS_DIR, file)
    with open(path, encoding="utf-8") as f:
        text = f.read().split("\n")
    while text and not text[-1]:
        text.pop()
    sidecar = os.path.splitext(path)[0] + ".yaml"
    base = _load(game, level=level)
    base_layout, base_reach = _layout(base), exit_reachable(base)
    if not base_reach:
        raise SystemExit(f"{game} {level}: the flood fill cannot reach the exit of the unmutated level")
    skipped = []
    if not hand:
        specs, skipped = place(game, level, text, base)
    records = []
    for op, cls, change, description, note in specs:
        mid = op if hand else f"{op}@{level}"
        folder = os.path.join(out, mid)
        shutil.rmtree(folder, ignore_errors=True)           # only what this run writes
        os.makedirs(folder)
        record = {"id": mid, "game": game, "stage": level, "section": None, "operator": op, "class": cls,
                  "direction": "none" if cls == "neutral" else "harder", "description": description,
                  "retrain": op != "cosmetic_tile", "base_file": file}
        if isinstance(change[0][0], str):                   # mechanics: config values
            olds = {k: _effective(k)(base.core) for k, _, _ in change}
            wrong = [f"{k} is {olds[k]}, the spec expects {b}" for k, b, _ in change if olds[k] != b]
            if wrong:
                raise SystemExit(f"{game} {mid}: refused, " + "; ".join(wrong))
            record.update({"family": "mechanics", "level_file": None, "cells": [], "location": None,
                           "config": {k: a for k, _, a in change},
                           "parameter": {"key": change[0][0], "from": change[0][1], "to": change[0][2]}})
        else:                                                # level: cells of the level file
            rows = [list(r) for r in text]
            wrong = [f"({r}, {c}) is {rows[r][c] if r < len(rows) and c < len(rows[r]) else 'outside the level'!r}, "
                     f"the spec expects {b!r}" for r, c, b, _ in change
                     if not (r < len(rows) and c < len(rows[r]) and rows[r][c] == b)]
            if wrong or len({(r, c) for r, c, _, _ in change}) != len(change):
                raise SystemExit(f"{game} {mid}: refused, " + ("; ".join(wrong) or "a cell is listed twice"))
            for r, c, _, a in change:
                rows[r][c] = a
            with open(os.path.join(folder, "level.txt"), "w", encoding="utf-8", newline="\n") as f:
                f.write("\n".join("".join(r) for r in rows) + "\n")
            if os.path.exists(sidecar):                     # the level's saws and other dynamics
                shutil.copyfile(sidecar, os.path.join(folder, "level.yaml"))
            cells = sorted((c, r, b, a) for r, c, b, a in change)
            mid_cell = cells[len(cells) // 2]
            record.update({"family": "level", "level_file": "level.txt", "config": None,
                           "cells": [list(x) for x in cells], "location": {"col": mid_cell[0], "row": mid_cell[1]},
                           "parameter": {"tiles": len(cells)}})
        if note:
            record["note"] = note
        with open(os.path.join(folder, "mutant.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(record, f, indent=2)
        try:
            record["checks"] = check(game, record, folder, base, base_layout)
        except SystemExit:
            shutil.rmtree(folder, ignore_errors=True)       # never leave a mutant that looks written
            raise
        with open(os.path.join(folder, "mutant.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(record, f, indent=2)
        records.append(record)
    index = {"game": game, "stage": level, "base_file": file, "mutants": records, "skipped": skipped}
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "mutants.json" if hand else f"mutants@{level}.json"), "w", encoding="utf-8",
              newline="\n") as f:
        json.dump(index, f, indent=2)
    return index


def check(game: str, m: dict, folder: str, base, base_layout) -> dict:
    """Load the mutant through the game's hook and confirm it is what its class says."""
    played = _load(game, folder, m["stage"])
    if played.core.mutant is None or played.core.mutant["id"] != m["id"]:
        raise SystemExit(f"{game} {m['id']}: the game did not pick up the mutant")
    checks: dict = {"exit_reachable": exit_reachable(played)}
    if m["family"] == "mechanics":
        now = {k: _effective(k)(played.core) for k in m["config"]}
        if now != m["config"]:
            raise SystemExit(f"{game} {m['id']}: the game plays {now}, not {m['config']}")
        checks["values"] = now
    else:
        same = _layout(played) == base_layout
        checks["same_level"] = same
        if same != (m["operator"] == "cosmetic_tile"):
            raise SystemExit(f"{game} {m['id']}: the loaded level is {'the same as' if same else 'not'} "
                             "the original" + ("" if same else " (a cosmetic change must parse the same)"))
        if m["operator"] == "off_route":
            walled = all(not _tile_open(base, c + dc, r + dr) for c, r, _, _ in m["cells"]
                         for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)))
            checks["walled_in"] = walled
            if not walled:
                raise SystemExit(f"{game} {m['id']}: the hollowed tile touches an open one")
    if checks["exit_reachable"] == (m["class"] == "breaking"):
        raise SystemExit(f"{game} {m['id']}: the flood fill {'reaches' if checks['exit_reachable'] else 'misses'} "
                         f"the exit, which a {m['class']} mutant must not")
    return checks


# ── the evaluation set ──────────────────────────────────────────────────────────────────────────
def eval_set(game: str) -> list[dict]:
    """The game's five evaluation mutants as written on disk, each with `dir` and `retrain_levels`
    (the levels whose agents are retrained on it: Mega Man retrains only the section it changes)."""
    ids = MEGAMAN_SET if game == "megaman" else [s[0] for s in CHANGES[game][1]]
    out = []
    for mid in ids:
        folder = os.path.join(out_dir(game), mid)
        try:
            with open(os.path.join(folder, "mutant.json"), encoding="utf-8") as f:
                m = json.load(f)
        except OSError:
            continue
        m["dir"] = folder
        m.setdefault("retrain", m["operator"] != "cosmetic_tile")
        if game == "megaman":
            from code.neuro.relay import stage_legs
            m["retrain_levels"] = [m["section"]] if m["section"] else [leg.level for leg in stage_legs("CutMan")]
        else:
            m["retrain_levels"] = [m["stage"]]
        out.append(m)
    return out


def placed_set(game: str) -> list[dict]:
    """Every other level's placed mutants as written on disk, level by level in operator order, each
    with `dir` and `retrain_levels` (its own level)."""
    if game not in CHANGES:
        return []
    out = []
    for level in other_levels(game):
        for op, *_ in CHANGES[game][1]:
            folder = os.path.join(out_dir(game), f"{op}@{level}")
            try:
                with open(os.path.join(folder, "mutant.json"), encoding="utf-8") as f:
                    m = json.load(f)
            except OSError:
                continue
            m.update({"dir": folder, "retrain_levels": [level]})
            m.setdefault("retrain", op != "cosmetic_tile")
            out.append(m)
    return out


def all_mutants(game: str) -> list[dict]:
    """The evaluation level's mutants, then every other level's placed ones."""
    return eval_set(game) + placed_set(game)


def table(game: str) -> str:
    rows = eval_set(game)
    lines = [f"{EVAL_LEVELS[game][1]}  ({out_dir(game)})"]
    if not rows:
        return lines[0] + "\n  no mutants yet: " + ("python -m code.games.megaman_mutants" if game == "megaman"
                                                     else "python -m code.games.eval_mutants")
    for m in rows:
        lines.append(f"  {m['id']:<22}{m['class']:<11}{'retrain' if m['retrain'] else 'replay ':<9}{m['description']}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Write the AST evaluation mutants")
    ap.add_argument("--game", action="append", choices=list(CHANGES), help="only this game (repeatable)")
    ap.add_argument("--list", action="store_true", help="print the evaluation set of every game and exit")
    ap.add_argument("--all-levels", action="store_true",
                    help="also place the five operators on every other level (<operator>@<level>)")
    args = ap.parse_args()
    if not args.list:
        for game in args.game or list(CHANGES):
            build(game)
            for level in other_levels(game) if args.all_levels else []:
                index = build(game, level=level)
                print(f"{game} {level}: {len(index['mutants'])} placed"
                      + "".join(f"; {s['operator']} skipped ({s['reason']})" for s in index["skipped"]))
    for game in (args.game or list(EVAL_LEVELS)):
        print(table(game), "\n")


if __name__ == "__main__":
    main()
