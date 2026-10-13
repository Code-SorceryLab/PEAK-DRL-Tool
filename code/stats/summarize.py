"""Balance metrics for one episode log: episodes.csv -> metrics.json (+ a terminal summary).

Training runs and manual-play sessions call this when they stop. It also works on any run
directory after the fact:

    python -m code.stats.summarize runs/mario_Mario1-1          # a training run
    python -m code.stats.summarize runs/manual/mario/Mario1-1    # manual play
"""
from __future__ import annotations

import argparse
import json
import os

from . import episode_log, registry


def summarize_rows(rows: list[dict]) -> dict:
    """{game: {level: {persona: metrics}}} for every (game, level, persona) in the rows."""
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for r in rows:
        groups.setdefault((r.get("game") or "?", str(r.get("level")), r.get("persona") or "?"), []).append(r)
    out: dict = {}
    for (game, level, persona), eps in sorted(groups.items()):
        m = registry.compute_level(eps, game)
        m["_episodes"] = len(eps)
        out.setdefault(game, {}).setdefault(level, {})[persona] = m
    for game, levels in out.items():  # skill-tier metrics wherever the tier personas share a level
        for level, by_persona in levels.items():
            cross = registry.compute_cross(by_persona, game)
            if any(v is not None for v in cross.values()):
                by_persona["_skill_tiers"] = cross
    return out


def summarize_file(csv_path: str, out_path: str | None = None, verbose: bool = True) -> str | None:
    rows = episode_log.read(csv_path)
    if not rows:
        return None
    result = summarize_rows(rows)
    out_path = out_path or os.path.join(os.path.dirname(csv_path), "metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"episodes": len(rows), "games": result}, f, indent=2)
    if verbose:
        print(f"\nbalance metrics ({len(rows)} episodes) -> {out_path}")
        for game, levels in result.items():
            for level, by_persona in levels.items():
                for persona, m in by_persona.items():
                    if persona.startswith("_"):
                        continue
                    print(registry.format_table({f"{level} · {persona}": m}, game))
        print(flush=True)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Balance metrics for episode logs")
    ap.add_argument("paths", nargs="+", help="run / session directories or episodes.csv files")
    args = ap.parse_args()
    for p in args.paths:
        csv_path = p if p.endswith(".csv") else os.path.join(p, "episodes.csv")
        if summarize_file(csv_path) is None:
            print(f"no episodes in {csv_path}")


if __name__ == "__main__":
    main()
