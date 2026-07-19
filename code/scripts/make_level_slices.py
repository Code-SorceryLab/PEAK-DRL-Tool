"""Slice the original SMB levels into thirds ("rule of thirds" curriculum).

Each full level is cut into 3 overlapping column ranges. Every slice is a
genuine excerpt of the ORIGINAL level geometry (not a synthetic ramp): the
first slice keeps the real spawn, the last keeps the real goal, and missing
spawn/goal markers are placed on real ground at the slice edges.

Why: short levels = short episodes = frequent terminals. The win bonus is
reachable within a few hundred decisions on a slice, so credit assignment
works from the first PPO updates, and the curriculum walks the agent from
slice mastery to the full level. (Same idea as ML-Agents' curriculum +
episode-design guidance: keep episodes short, escalate difficulty.)

Usage:
    python -m code.scripts.make_level_slices          # writes *_s{1,2,3}.txt/.yaml
    python -m code.scripts.make_level_slices --check  # dry run, print summary
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

TILE = 32
OVERLAP = 4          # columns of overlap between adjacent slices
SOLID = {"#", "="}
LEVELS_DIR = Path(__file__).resolve().parents[1] / "games" / "levels" / "platformer"
SOURCES = ["world1_1", "world1_2"]


def _load_rows(path: Path) -> list[str]:
    rows = path.read_text().splitlines()
    width = max(len(r) for r in rows)
    return [r.ljust(width, ".") for r in rows]


def _ground_row(rows: list[str], col: int) -> int | None:
    """Bottom-up first solid tile in a column (skips ceilings)."""
    for r in range(len(rows) - 1, -1, -1):
        if rows[r][col] in SOLID:
            # the cell above must be free so an entity can stand there
            if r > 0 and rows[r - 1][col] in (".", " "):
                return r
    return None


def _place(rows: list[str], col_range, char: str, skip_cols=()) -> tuple[int, int]:
    """Place `char` on the first standable ground column in col_range."""
    for c in col_range:
        if c in skip_cols:
            continue
        r = _ground_row(rows, c)
        if r is not None:
            row = rows[r - 1]
            rows[r - 1] = row[:c] + char + row[c + 1:]
            return r - 1, c
    raise RuntimeError(f"no standable ground found for '{char}' in {col_range}")


def _remove(rows: list[str], char: str) -> None:
    for i, row in enumerate(rows):
        if char in row:
            rows[i] = row.replace(char, ".")


def _spawn_reachable(level_name: str) -> bool:
    """Load the written slice with the real engine and check the spawn tile
    has a valid Dijkstra distance to the goal. Guards against placing the
    spawn inside a dead-end pocket (e.g. world1_2 cols 58-61) that the
    solver's goal flood-fill cannot reach."""
    from code.games.platformer_core import PlatformerCore
    core = PlatformerCore(render_mode="none", world=level_name,
                          curriculum_enabled=False)
    core.reset()
    _, _, _, _, info = core.step([0, 0, 0])
    return float(info.get("dijkstra_dist", -1.0)) >= 0.0


def _config_name(source: str, slice_idx: int) -> str | None:
    """Map (source stem, 1-based slice index) → the game_config level key,
    e.g. ("world1_1", 2) → "Mario1-1b". None for unregistered sources."""
    base = {"world1_1": "Mario1-1", "world1_2": "Mario1-2"}.get(source)
    return None if base is None else base + "abc"[slice_idx - 1]


def _slice_bounds(width: int) -> list[tuple[int, int]]:
    third = width // 3
    return [
        (0, third + OVERLAP),
        (third - OVERLAP, 2 * third + OVERLAP),
        (2 * third - OVERLAP, width),
    ]


def _slice_yaml(src_yaml: Path, c0: int, c1: int) -> dict | None:
    """Carry over sidecar dynamics that fall inside the slice, x-shifted."""
    if not src_yaml.exists():
        return None
    data = yaml.safe_load(src_yaml.read_text()) or {}
    out: dict = {}
    plats = ((data.get("dynamics") or {}).get("moving_platforms")) or []
    kept = []
    for p in plats:
        x = float(p["start"][0])
        if c0 * TILE <= x < c1 * TILE:
            q = dict(p)
            q["start"] = [p["start"][0] - c0 * TILE, p["start"][1]]
            q["end"] = [p["end"][0] - c0 * TILE, p["end"][1]]
            kept.append(q)
    if kept:
        out["dynamics"] = {"moving_platforms": kept}
    if data.get("physics"):
        out["physics"] = data["physics"]   # not positional — copy through
    return out or None


def make_slices(check: bool = False) -> list[str]:
    made = []
    for name in SOURCES:
        src = LEVELS_DIR / f"{name}.txt"
        rows = _load_rows(src)
        width = len(rows[0])
        for i, (c0, c1) in enumerate(_slice_bounds(width), start=1):
            sliced = [r[c0:c1] for r in rows]
            w = c1 - c0
            has_p = any("P" in r for r in sliced)
            has_g = any("G" in r for r in sliced)
            p_col = None
            if not has_p:
                # spawn near the left edge, but not at col 0 (OOB-left margin)
                _, p_col = _place(sliced, range(2, w // 2), "P")
            if not has_g:
                # goal near the right edge (goal guard needs x > TILE_SIZE)
                _place(sliced, range(w - 3, w // 2, -1), "G")
            out_txt = LEVELS_DIR / f"{name}_s{i}.txt"
            side = _slice_yaml(src.with_suffix(".yaml"), c0, c1)
            if check:
                print(f"{out_txt.name}: cols [{c0},{c1}) w={w} "
                      f"P={'orig' if has_p else 'added'} G={'orig' if has_g else 'added'} "
                      f"sidecar={'yes' if side else 'no'}")
                made.append(out_txt.name)
                continue

            out_txt.write_text("\n".join(sliced) + "\n")
            if side:
                out_txt.with_suffix(".yaml").write_text(
                    yaml.safe_dump(side, sort_keys=False))

            # Self-validate with the real engine: if the spawn tile can't
            # reach the goal per the Dijkstra solver (dead-end pocket), move
            # P rightward to the next standable column and retry.
            level_name = _config_name(name, i)
            if has_p or level_name is None:
                pass  # original spawn (or unregistered) — trust the source
            else:
                tried: set[int] = {p_col}   # seed with the initial placement
                for _ in range(30):
                    if _spawn_reachable(level_name):
                        break
                    _remove(sliced, "P")
                    _, c = _place(sliced, range(2, w - 4), "P", skip_cols=tried)
                    tried.add(c)
                    out_txt.write_text("\n".join(sliced) + "\n")
                else:
                    raise RuntimeError(f"{out_txt.name}: no reachable spawn found")
            made.append(out_txt.name)
    return made


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="dry run")
    args = ap.parse_args()
    made = make_slices(check=args.check)
    if not args.check:
        print("wrote:", ", ".join(made))
    sys.exit(0)
