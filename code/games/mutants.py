"""The game mutant this process plays, if PEAK_MUTANT names one (code/games/eval_mutants.py and
code/games/megaman_mutants.py make them).

A mutant is a folder holding mutant.json and, for a level change, its own copy of the level
(level.txt, plus level.yaml when the level has a sidecar). Each core asks here when it loads its
config and its level, so level names and every command stay the same:

    PEAK_MUTANT=runs/mario/mutants/seal_route python -m code.neuro.trainer --game mario --level Mario1-1
"""
from __future__ import annotations

import json
import os

ENV = "PEAK_MUTANT"
OLD_ENV = "PEAK_MM1_MUTANT"   # Mega Man's first name for it; its notes and commands still use it


def active(game: str) -> dict | None:
    """The named mutant's record, with its folder in `dir`; None when none is named or it is
    another game's (a process that builds several games changes only the mutant's own)."""
    path = os.environ.get(ENV) or os.environ.get(OLD_ENV)
    if not path:
        return None
    if os.path.isdir(path):
        path = os.path.join(path, "mutant.json")
    with open(path, encoding="utf-8") as f:
        record = json.load(f)
    if record.get("game") != game:
        return None
    record["dir"] = os.path.dirname(os.path.abspath(path))
    return record


def level_path(record: dict | None, file: str) -> str | None:
    """The mutant's own copy of this level file, or None when the mutant leaves it alone."""
    if record and record.get("level_file") and record.get("base_file") == file:
        return os.path.join(record["dir"], record["level_file"])
    return None


def apply_config(cfg, record: dict | None) -> None:
    """Write the mutant's values into a loaded config, in place. Keys are dotted paths and a number
    indexes a list: {"jump.jump_vel": 408.0, "levels.8.fuse": 150}. A wrong path raises."""
    for key, value in ((record or {}).get("config") or {}).items():
        node = cfg
        *parents, leaf = key.split(".")
        for k in parents:
            node = node[int(k)] if isinstance(node, list) else node[k]
        node[int(leaf) if isinstance(node, list) else leaf] = value
