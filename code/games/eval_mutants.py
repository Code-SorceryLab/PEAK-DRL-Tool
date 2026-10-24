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
    "bomberman": ("8", "Bomberman 09 Two Balloms"),
    "megaman": ("CutMan", "Mega Man, the Cut Man stage"),
}
MEGAMAN_SET = ["s3-seal_route", "s3-widen_gap", "m-jump_velocity_down", "s3-off_route", "s3-cosmetic_tile"]


def column(col: int, rows: range, before: str, after: str) -> list:
    return [(r, col, b, after) for r, b in zip(rows, before)]


# Level changes are cells (row, col, before, after), counted from 0 as in the spec; mechanics
# changes are config values (dotted key, before, after). id, class, change, description, note.
CHANGES = {
    "mario": ("platformer/world1_1.txt", [
        ("seal_route", "breaking", column(90, range(14), "......=.......", "#"),
         "A wall down column 90, rows 0-13 (it takes the place of the = tile at row 6): the flag "
         "can no longer be reached", None),
        ("widen_gap", "degrading",
         [(14, c, "#", " ") for c in (86, 87)] + [(r, c, "#", "O") for r in range(15, 22) for c in (86, 87)],
         "The second pit (columns 83-85) widened by 2 to the right: 3 tiles -> 5", None),
        ("jump_velocity_down", "degrading",
         [("player.jump.min_velocity", -495.0, -420.75), ("player.jump.max_velocity", -800.0, -680.0)],
         "Jump velocity x0.85: take-off -495 -> -420.75 px/s, the cap -800 -> -680",
         "The spec edits physics.jump_velocity in world1_1.yaml, but the game never reads that block "
         "(LevelLoader takes only `dynamics` from it), so that edit would change nothing. This scales "
         "the jump values the game does read; the small run-speed bonus (up to 57 px/s) is unchanged. "
         "A full jump still rises 4 tiles (5 before) and the tallest step on the route is 3, so 1-1 "
         "stays solvable."),
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
    "bomberman": ("bomberman/09_two_balloms.txt", [
        ("seal_route", "breaking", [(9, 12, ".", "#"), (8, 13, ".", "#")],
         "Walls at row 9, column 12 and row 8, column 13 close the exit into its corner", None),
        ("add_enemy", "degrading", [(7, 7, ".", "E")],
         "A third Ballom at row 7, column 7; every enemy must die before the exit opens", None),
        ("fuse_down", "degrading", [("levels.8.fuse", 240, 150)],
         "Bomb fuse 240 -> 150 frames on this level", None),
        ("off_route", "neutral", [(11, 5, "#", ".")],
         "The wall tile at row 11, column 5 hollowed out, closed on all four sides", None),
        ("cosmetic_tile", "neutral", [(5, 5, ".", " ")],
         "The floor cell at row 5, column 5 written as ' ' instead of '.': the same tile", None),
    ]),
}

# What the running game uses for each mechanics key, read from a built core
EFFECTIVE = {
    "player.jump.min_velocity": lambda core: core.physics_manager.context.JUMP_VEL_MIN,
    "player.jump.max_velocity": lambda core: core.physics_manager.context.JUMP_VEL_MAX,
    "jump.jump_vel": lambda core: core._jump["jump_vel"],
    "levels.8.fuse": lambda core: core.fuse_frames,
}


def out_dir(game: str) -> str:
    return os.path.join("runs", game, "mutants")


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


def _load(game: str, folder: str | None = None):
    from code.neuro.adapters import make_adapter
    with _named(folder):
        return make_adapter(game, EVAL_LEVELS[game][0], 600, 0.0)


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


def exit_reachable(adapter) -> bool:
    """A flood fill over the open tiles inside the level, from the start. It ignores gravity and
    jump height, so 'no' proves the exit cut off; 'yes' only says nothing walls it off."""
    ld, ts = adapter.core.level_data, adapter.tile_size
    goals = _goal_tiles(adapter)
    start = (int(adapter.x // ts), int(adapter.y // ts))
    seen, todo = {start}, [start]
    while todo:
        c, r = todo.pop()
        if (c, r) in goals:
            return True
        for nc, nr in ((c + 1, r), (c - 1, r), (c, r + 1), (c, r - 1)):
            if 0 <= nc < ld.cols and 0 <= nr < ld.rows and (nc, nr) not in seen and _tile_open(adapter, nc, nr):
                seen.add((nc, nr))
                todo.append((nc, nr))
    return False


# ── generating ───────────────────────────────────────────────────────────────────────────────
def build(game: str, out: str | None = None) -> dict:
    """Write the game's five mutants, each checked through the game itself. Raises on a mismatch."""
    out = out or out_dir(game)
    file, specs = CHANGES[game]
    level = EVAL_LEVELS[game][0]
    path = os.path.join(LEVELS_DIR, file)
    with open(path, encoding="utf-8") as f:
        text = f.read().split("\n")
    while text and not text[-1]:
        text.pop()
    sidecar = os.path.splitext(path)[0] + ".yaml"
    base = _load(game)
    base_layout, base_reach = _layout(base), exit_reachable(base)
    if not base_reach:
        raise SystemExit(f"{game}: the flood fill cannot reach the exit of the unmutated level")
    records = []
    for mid, cls, change, description, note in specs:
        folder = os.path.join(out, mid)
        shutil.rmtree(folder, ignore_errors=True)           # only what this run writes
        os.makedirs(folder)
        record = {"id": mid, "game": game, "stage": level, "section": None, "operator": mid, "class": cls,
                  "direction": "none" if cls == "neutral" else "harder", "description": description,
                  "retrain": mid != "cosmetic_tile", "base_file": file}
        if isinstance(change[0][0], str):                   # mechanics: config values
            olds = {k: EFFECTIVE[k](base.core) for k, _, _ in change}
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
    index = {"game": game, "stage": level, "base_file": file, "mutants": records, "skipped": []}
    with open(os.path.join(out, "mutants.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(index, f, indent=2)
    return index


def check(game: str, m: dict, folder: str, base, base_layout) -> dict:
    """Load the mutant through the game's hook and confirm it is what its class says."""
    played = _load(game, folder)
    if played.core.mutant is None or played.core.mutant["id"] != m["id"]:
        raise SystemExit(f"{game} {m['id']}: the game did not pick up the mutant")
    checks: dict = {"exit_reachable": exit_reachable(played)}
    if m["family"] == "mechanics":
        now = {k: EFFECTIVE[k](played.core) for k in m["config"]}
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
    args = ap.parse_args()
    if not args.list:
        for game in args.game or list(CHANGES):
            build(game)
    for game in (args.game or list(EVAL_LEVELS)):
        print(table(game), "\n")


if __name__ == "__main__":
    main()
