"""SPEEDY's AST mutants: the same five operators as the other games, on each act of Green Hill.

    python -m code.games.eval_mutants --game speedy --all-levels

The other 2D games edit cells of a text grid; a SPEEDY level is polygons and object lists, so its
mutants edit the level's yaml. Each is loaded through the game's own hook (code/games/mutants.py)
and checked before it is written, and a mutant that fails its check is not left on disk.
"""
from __future__ import annotations

import json
import os
import re
import shutil

import numpy as np

from . import mutants, speedy_level as lv
from .speedy_core import SpeedyCore

JUMP = 0x680                # the jump the config holds now; a config that differs is refused
JUMP_DOWN = int(JUMP * 0.85)


def _pocket(level: lv.Level) -> tuple:
    """A 16 px cell in the middle of solid-on-every-side dirt, 32 px from anything else."""
    raw = np.frombuffer(level.masks[0], np.uint8).reshape(-1, level.px_w)
    for y in range(level.px_h - 48, 48, -32):
        for x in range(64, level.px_w - 64, 32):
            if (raw[y - 40:y + 40, x - 40:x + 40] == 1).all():
                return x, y
    raise SystemExit(f"{level.name}: no solid pocket for off_route")


def _edits(level: lv.Level) -> dict:
    """operator -> (text to change the level file, record fields)."""
    goal_x, goal_y = (level.sign or level.capsule)[:2]
    wall = goal_x - 64
    x0, x1, y = level.bridges[0]
    px, py = _pocket(level)
    return {
        "seal_route": (f'block: "{wall},0,{wall + 16},{level.px_h}"\n', {"cells": [[wall // 32, goal_y // 32]]}),
        "widen_gap": (None, {"bridge": [x0, x1, y], "cells": [[(x0 + x1) // 64, y // 32]]}),
        "off_route": (f'pocket: "{px},{py},{px + 16},{py + 16}"\n', {"cells": [[px // 32, py // 32]]}),
        "cosmetic_tile": ("# cosmetic_tile: a comment, the level parses the same\n", {"cells": []}),
    }


def _text(base: str, op: str, add: str | None, fields: dict) -> str:
    if op == "widen_gap":                       # the logs go, the gap they spanned stays
        x0, x1, y = fields["bridge"]
        lines = base.split("\n")
        at = next(i for i, l in enumerate(lines) if l.startswith("bridges:"))
        rest = re.findall(r"\[\d+, \d+, \d+\]", lines[at])[1:]
        lines[at] = f"bridges: [{', '.join(rest)}]\ngaps: [[{x0}, {x1}, {y}]]"
        return "\n".join(lines)
    return base + add


def _check(op: str, level: str, record: dict, folder: str, base: SpeedyCore) -> dict:
    os.environ[mutants.ENV] = folder
    try:
        core = SpeedyCore(level=level)
    finally:
        del os.environ[mutants.ENV]
    L = core.level
    if op == "seal_route":
        ok = all(L.solid(record["wall_x"] + 8, y) for y in range(0, L.px_h, 8))
    elif op == "widen_gap":
        ok = len(L.bridges) == len(base.level.bridges) - 1 and not L.solid((record["bridge"][0] + record["bridge"][1]) // 2,
                                                                         record["bridge"][2] + 8)
    elif op == "jump_velocity_down":
        ok = core.jump == JUMP_DOWN and core.level is base.level
    elif op == "off_route":
        ok = L.masks[0] != base.level.masks[0]
    else:
        ok = L.masks == base.level.masks and L.objects == base.level.objects
    if not ok:
        raise SystemExit(f"speedy {record['id']}: the game did not take the change")
    return {"plays": True}


def build(specs: list, out: str, level: str, hand: bool) -> dict:
    """Write the five mutants of one act into `out`; <operator>@<level> on any act but the evaluation one."""
    file = f"speedy/{level}.yaml"
    with open(os.path.join(lv.LEVEL_DIR, f"{level}.yaml"), encoding="utf-8") as f:
        base_text = f.read()
    base = SpeedyCore(level=level)
    if int(base.cfg["physics"]["jump"]) != JUMP:
        raise SystemExit(f"speedy: physics.jump is {base.cfg['physics']['jump']}, the spec expects {JUMP}")
    edits = _edits(base.level)
    records = []
    for op, cls, _, description, note in specs:
        mid = op if hand else f"{op}@{level}"
        folder = os.path.join(out, mid)
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder)
        record = {"id": mid, "game": "speedy", "stage": level, "section": None, "operator": op, "class": cls,
                  "direction": "none" if cls == "neutral" else "harder", "description": description,
                  "retrain": op != "cosmetic_tile", "base_file": file, "base_sha1": mutants.source_hash(file)}
        if op == "jump_velocity_down":
            record.update({"family": "mechanics", "level_file": None, "cells": [], "location": None,
                           "config": {"physics.jump": JUMP_DOWN},
                           "parameter": {"key": "physics.jump", "from": JUMP, "to": JUMP_DOWN}})
        else:
            add, fields = edits[op]
            with open(os.path.join(folder, "level.yaml"), "w", encoding="utf-8", newline="\n") as f:
                f.write(_text(base_text, op, add, fields))
            cells = fields["cells"]
            record.update({"family": "level", "level_file": "level.yaml", "config": None, "cells": cells,
                           "location": {"col": cells[0][0], "row": cells[0][1]} if cells else None,
                           "parameter": {"tiles": len(cells)}})
            if op == "widen_gap":
                record["bridge"] = fields["bridge"]
            if op == "seal_route":
                record["wall_x"] = (base.level.sign or base.level.capsule)[0] - 64
        if note:
            record["note"] = note
        with open(os.path.join(folder, "mutant.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(record, f, indent=2)
        try:
            record["checks"] = _check(op, level, record, folder, base)
        except SystemExit:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        with open(os.path.join(folder, "mutant.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(record, f, indent=2)
        records.append(record)
    index = {"game": "speedy", "stage": level, "base_file": file, "mutants": records, "skipped": []}
    with open(os.path.join(out, "mutants.json" if hand else f"mutants@{level}.json"), "w", encoding="utf-8",
              newline="\n") as f:
        json.dump(index, f, indent=2)
    return index
