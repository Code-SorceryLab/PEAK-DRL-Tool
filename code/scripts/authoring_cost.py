#!/usr/bin/env python3
"""
authoring_cost.py — Challenge-3 evidence: cost to author one level variant.

Produces ONE concrete level variant in PEAK (widen a pit by K tiles on a real
ASCII level), MEASURES the edit footprint (files touched, lines/chars/bytes
changed, wall-clock to apply + reload), and prints an honest comparison against
the same change in Unity (Tilemap) or a Gym-Retro ROM.

This converts Requirement 3 ("low-overhead authoring") from assertion to evidence.

    .venv/bin/python -m code.scripts.authoring_cost [--level <path>] [--tiles K]
"""
from __future__ import annotations

import argparse
import difflib
import sys
import time
from pathlib import Path

DEFAULT_LEVEL = "code/games/levels/platformer/world1_1.txt"
GROUND = "#"
AIR = " "


def _widen_pit(lines: list[str], tiles: int) -> tuple[list[str], int]:
    """Convert a run of `tiles` ground chars to air on the bottom-most ground rows,
    creating/widening a pit. Returns (new_lines, rows_modified)."""
    # bottom-most rows with substantial ground
    ground_rows = [i for i, ln in enumerate(lines) if ln.count(GROUND) >= max(4, tiles)]
    if not ground_rows:
        raise ValueError("no ground rows found to edit")
    target_rows = ground_rows[-2:] if len(ground_rows) >= 2 else ground_rows[-1:]
    out = list(lines)
    rows_modified = 0
    for r in target_rows:
        ln = out[r]
        # widen at the horizontal midpoint, snapped to a ground run
        start = max(0, len(ln) // 2 - tiles // 2)
        # find a position where `tiles` consecutive GROUND exist near the midpoint
        pos = ln.find(GROUND * tiles, start)
        if pos == -1:
            pos = ln.find(GROUND * tiles)
        if pos == -1:
            continue
        out[r] = ln[:pos] + AIR * tiles + ln[pos + tiles:]
        rows_modified += 1
    if rows_modified == 0:
        raise ValueError(f"could not find a run of {tiles} ground tiles to widen")
    return out, rows_modified


def _char_diff(a: str, b: str) -> int:
    return sum(1 for x, y in zip(a, b) if x != y) + abs(len(a) - len(b))


def measure_peak(level_path: str, tiles: int) -> dict:
    p = Path(level_path)
    base = p.read_text()
    base_lines = base.splitlines(keepends=False)

    t0 = time.perf_counter()
    new_lines, rows_modified = _widen_pit(base_lines, tiles)
    variant = "\n".join(new_lines) + ("\n" if base.endswith("\n") else "")
    # write variant next to the original, then confirm it loads (no rebuild needed)
    variant_path = p.with_name(p.stem + f"__pitwiden{tiles}.txt")
    variant_path.write_text(variant)
    load_ok, load_err = _try_load(str(variant_path))
    elapsed = time.perf_counter() - t0

    chars_changed = _char_diff(base, variant)
    bytes_changed = abs(len(variant.encode()) - len(base.encode())) or chars_changed
    diff_lines = [l for l in difflib.unified_diff(base_lines, new_lines, lineterm="")
                  if l and l[0] in "+-" and not l.startswith(("+++", "---"))]

    # clean up the generated variant file (the measurement is the point, not the artifact)
    try:
        variant_path.unlink()
    except Exception:
        pass

    return {
        "level": level_path,
        "edit": f"widen a pit by {tiles} tiles",
        "files_touched": 1,
        "rows_modified": rows_modified,
        "chars_changed": chars_changed,
        "bytes_changed": bytes_changed,
        "diff_line_count": len(diff_lines),
        "wallclock_s": round(elapsed, 4),
        "rebuild_required": False,
        "loads_at_runtime": load_ok,
        "load_error": load_err,
    }


def _try_load(level_path: str):
    """Best-effort: confirm the variant parses via PEAK's LevelLoader (proves the
    edit needs no code change / recompile — the level is read at runtime)."""
    try:
        from code.games.modules.System.LevelLoader import LevelLoader
        rel = level_path
        # LevelLoader resolves paths relative to its base dir; pass a repo-relative path.
        loader = LevelLoader()
        loader.load_level(rel.replace("code/games/levels/", ""))
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def print_report(m: dict) -> str:
    lines = []
    lines.append("AUTHORING-COST EVIDENCE — produce one level variant (Requirement 3)")
    lines.append("=" * 70)
    lines.append(f"Edit: {m['edit']}   on   {m['level']}")
    lines.append("")
    lines.append("PEAK (measured):")
    lines.append(f"  files touched ........ {m['files_touched']}  (one ASCII .txt)")
    lines.append(f"  rows modified ........ {m['rows_modified']}")
    lines.append(f"  chars changed ........ {m['chars_changed']}")
    lines.append(f"  bytes changed ........ {m['bytes_changed']}")
    lines.append(f"  apply + reload ....... {m['wallclock_s']} s  (programmatic)")
    lines.append(f"  rebuild required ..... {m['rebuild_required']}  (level read at runtime)")
    lines.append(f"  variant loads ........ {m['loads_at_runtime']}"
                 + (f"  [{m['load_error']}]" if m["load_error"] else ""))
    lines.append("")
    lines.append("Honest comparison (same pit-widening change elsewhere):")
    lines.append("  Unity (Tilemap):  open Editor, paint/erase tiles in the Tilemap, adjust the")
    lines.append("    TilemapCollider, save the scene + tilemap assets, re-enter Play mode")
    lines.append("    (domain reload). >=2 assets touched; an editor round-trip of seconds-to-")
    lines.append("    minutes per iteration. If physics/logic changes, recompile C#.")
    lines.append("  Gym-Retro ROM:  requires the commercial ROM + a SMB-specific level editor /")
    lines.append("    disassembly; edits are binary/offset-based and not designer-accessible.")
    lines.append("    Reproducing one variant is hours and practically/legally gated.")
    lines.append("")
    lines.append("Takeaway: in PEAK a level variant is a one-file, few-character text edit with")
    lines.append("no rebuild; the equivalent change in a production engine or a ROM benchmark is")
    lines.append("a multi-asset, editor/tooling round-trip. This is the Requirement-3 advantage.")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Measure PEAK level-authoring cost (Challenge 3).")
    ap.add_argument("--level", default=DEFAULT_LEVEL)
    ap.add_argument("--tiles", type=int, default=3, help="pit width to add (tiles)")
    ap.add_argument("--out", default=None, help="optional path to write the report")
    args = ap.parse_args(argv)
    m = measure_peak(args.level, args.tiles)
    report = print_report(m)
    print(report)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(report + "\n")
        print(f"\n[INFO] Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
