#!/usr/bin/env python3
"""Split a Mega Man 1 stage into training sections: one level per stretch of the stage, so the
curriculum climbs a ladder instead of facing all 24 screens at once.

    python -m code.games.tools.mm1_make_slices

A section is not a separate file. It is the full stage file with its own starting point, and it
is won on reaching the screens where the next section begins. So the world, the enemies and
everything an agent senses are exactly those of the full stage, and a network trained on a
section behaves the same when it plays its part of the whole (code/neuro/relay.py).

Reads the stage (levels/megaman/01_cutman.txt — edited or straight from the ROM) to find where
each section is entered, and lists the sections in megaman_levels.yaml ahead of the full stage.
Run it again after editing the stage.
"""
from __future__ import annotations

from pathlib import Path

import yaml

GAMES_DIR = Path(__file__).resolve().parent.parent
LEVELS_YAML = GAMES_DIR / "megaman_levels.yaml"
COLS, ROWS = 16, 15

# name, screens as (column, row) on the stage's screen grid, how you get in, how you get out
CUTMAN = {
    "stage": "cutman",
    "source": "megaman/01_cutman.txt",
    "full_id": "CutMan",
    "slices": [
        ("CutMan-1-Street",   [(0, 8), (1, 8), (2, 8), (3, 8)],         "start", "up"),
        ("CutMan-2-Climb",    [(3, 7), (3, 6), (3, 5)],                 "below", "up"),
        ("CutMan-3-Ledges",   [(3, 4), (4, 4), (5, 4)],                 "below", "up"),
        ("CutMan-4-Shaft",    [(5, 3), (5, 2), (5, 1)],                 "below", "up"),
        ("CutMan-5-Rooftop",  [(5, 0), (6, 0), (7, 0)],                 "below", "down"),
        ("CutMan-6-Descent",  [(7, 1), (7, 2), (7, 3), (8, 3)],         "above", "door"),
        ("CutMan-7-Corridor", [(9, 3), (10, 3), (11, 3)],               "left",  "door"),
        ("CutMan-8-Boss",     [(12, 3)],                                "left",  "boss"),
    ],
}


def entry_cell(grid: list[list[str]], first: tuple, enter: str) -> tuple[int, int, bool]:
    """Where the stage comes into this screen: (column, row, on a ladder?)."""
    def cell(c, r):
        return grid[r][c] if 0 <= r < len(grid) and 0 <= c < len(grid[r]) else " "

    fc, fr = first[0] * COLS, first[1] * ROWS           # top-left cell of the section's first screen
    if enter == "start":
        c, r = next((c, r) for r in range(len(grid)) for c in range(len(grid[r])) if grid[r][c] == "P")
        return c, r, False
    if enter == "below":                                # climbing in through the floor
        c = next(c for c in range(fc, fc + COLS) if cell(c, fr + ROWS - 1) == "H" and cell(c, fr + ROWS) == "H")
        return c, fr + ROWS - 1, True
    if enter == "above":                                # dropping in through the ceiling
        c = next(c for c in range(fc, fc + COLS) if cell(c, fr) in ".H" and cell(c, fr - 1) in ".H")
        return c, fr, False
    r = max(r for r in range(fr, fr + ROWS) if cell(fc - 1, r) == "D")      # "left": through the shutter
    return fc + 1, r, False


def main(spec: dict = CUTMAN) -> None:
    src = GAMES_DIR / "levels" / spec["source"]
    grid = [list(line) for line in src.read_text(encoding="utf-8").split("\n")]
    width = max(len(r) for r in grid)
    grid = [r + [" "] * (width - len(r)) for r in grid]

    data = yaml.safe_load(LEVELS_YAML.read_text(encoding="utf-8")) or {}
    old = data.get("levels") or {}
    parts = spec["slices"]
    starts = [entry_cell(grid, screens[0], enter) for _, screens, enter, _ in parts]
    made = {}
    for i, (name, screens, enter, leave) in enumerate(parts):
        c, r, climbing = starts[i]
        section = {"start": [c, r], "climbing": climbing, "screens": [list(s) for s in screens]}
        if i + 1 < len(parts):                          # won on reaching the next section's screens
            section["goal_screens"] = [list(s) for s in parts[i + 1][1]]
            section["end"] = list(starts[i + 1][:2])
        made[name] = {"file": spec["source"], "stage": spec["stage"], "section": section}
        print(f"{name:<20} {len(screens)} screens, starts at {(c, r)}{' on the ladder' if climbing else ''}")
    full = dict(old.get(spec["full_id"]) or {"file": spec["source"]})
    full["stage"] = spec["stage"]
    # sections first (the ladder), then the full stage, then anything else already listed
    data["levels"] = {**made, spec["full_id"]: full,
                      **{k: v for k, v in old.items() if k not in made and k != spec["full_id"]}}
    LEVELS_YAML.write_text(yaml.dump(data, default_flow_style=None, sort_keys=False, width=120), encoding="utf-8")
    print(f"listed {len(made)} sections + {spec['full_id']} in {LEVELS_YAML.name}")


if __name__ == "__main__":
    main()
