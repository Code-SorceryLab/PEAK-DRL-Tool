"""Game mutants of a Mega Man stage: copies of the game that differ by one small change.

    python -m code.games.megaman_mutants                 # -> runs/megaman/mutants/
    python -m code.games.megaman_mutants --list          # what is there

A mutant is a record (mutant.json) plus what it changes: its own copy of the stage file (level
mutants) or new values for megaman_config.yaml (mechanics mutants). The game code is not touched.
Each mutant makes exactly one change, and each level operator is applied once in every section of
the stage where it has something to act on; where it has not (no ladder in the boss room) the
section is skipped and the reason recorded.

    Level operators      seal_route  remove_ladder  widen_gap  add_hazard
                         add_enemy   remove_enemy   cosmetic_tile  off_route
    Mechanics operators  one config value moved by a fixed step (jump, gravity, speeds, energy, ...)

Every mutant gets its class before any agent plays it, from the route field
(megaman_route.py) searched from the stage's true start:

    breaking   the exit can no longer be reached
    degrading  still reachable, but something on the route changed
    neutral    nothing a player can reach or sense changed (checked, not assumed)

The route field approximates the physics and is cautious, so spot-check a sample by hand.

To play a mutant, name it in the environment; every command then runs on that copy of the game:

    PEAK_MM1_MUTANT=runs/megaman/mutants/s3-seal_route python -m code.neuro.relay --episodes 20 \\
        --out runs/megaman/mutants/s3-seal_route/relay
"""
from __future__ import annotations

import argparse
import json
import math
import os

import yaml

from . import megaman_enemies as en
from .megaman_core import AIR, CONFIG_PATH, DOOR, DOOR_SHUT, GOAL, LADDER, LEVELS_PATH, SOLID, SPIKE, START, VOID, parse_level
from .megaman_route import DRIFT_CELLS, JUMP_CELLS, route_field

GAME = "megaman"
OUT = os.path.join("runs", GAME, "mutants")
LEVELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "levels")
LEVEL_OPERATORS = ["seal_route", "remove_ladder", "widen_gap", "add_hazard", "add_enemy", "remove_enemy",
                   "cosmetic_tile", "off_route"]
INF = float("inf")
SOLIDS = (SOLID, DOOR_SHUT)
KEEP = START + GOAL + DOOR + DOOR_SHUT + en.ENEMY_GLYPHS + en.ITEM_GLYPHS + en.BOSS_GLYPH   # never overwritten by a wall
# Mechanics mutants: (id, config key, new value from the old one, what a player would notice)
MECHANICS = [
    ("jump_velocity_down", "player.jump_velocity", lambda v: round(v * 0.85, 3), "harder"),
    ("jump_velocity_low", "player.jump_velocity", lambda v: round(v * 0.70, 3), "harder"),
    ("jump_velocity_up", "player.jump_velocity", lambda v: round(v * 1.10, 3), "easier"),
    ("gravity_up", "player.gravity", lambda v: round(v * 1.25, 4), "harder"),
    ("run_speed_down", "player.run_speed", lambda v: round(v * 0.8, 4), "harder"),
    ("air_speed_down", "player.air_speed", lambda v: round(v * 0.8, 4), "harder"),
    ("ladder_speed_down", "player.ladder_speed", lambda v: round(v * 2 / 3, 4), "harder"),
    ("energy_down", "player.hp", lambda v: int(v) - 8, "harder"),
    ("max_shots_down", "buster.max_shots", lambda v: 1, "harder"),
    ("time_limit_down", "time_limit", lambda v: int(v * 0.8), "harder"),
]


# ── the stage, as text and as the game reads it ──────────────────────────────────────────────────
class Stage:
    def __init__(self, stage: str) -> None:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        with open(LEVELS_PATH, encoding="utf-8") as f:
            levels = (yaml.safe_load(f) or {}).get("levels") or {}
        if stage not in levels:
            raise SystemExit(f"unknown level '{stage}'")
        self.name, self.file = stage, levels[stage]["file"]
        self.sc, self.sr = int(self.cfg["screen"]["cols"]), int(self.cfg["screen"]["rows"])
        self.tile = int(self.cfg["tile_size"])
        with open(os.path.join(LEVELS_DIR, self.file), encoding="utf-8") as f:
            self.text = [r.rstrip("\n") for r in f]
        while self.text and not self.text[-1].strip():
            self.text.pop()
        self.sections = [(name, e["section"]) for name, e in levels.items()
                         if name != stage and e.get("file") == self.file and (e.get("section") or {}).get("screens")]
        self.ld = self.parse(self.text)
        self.cost = route_field(self.ld)

    def parse(self, text: list[str]):
        return parse_level(text, self.sc, self.sr, self.tile, self.file)

    def edit(self, cells: dict[tuple[int, int], str]) -> list[str]:
        """A copy of the stage text with these cells rewritten: {(col, row): glyph}."""
        rows = [list(r) for r in self.text]
        for (c, r), ch in cells.items():
            if len(rows[r]) <= c:
                rows[r] += [VOID] * (c + 1 - len(rows[r]))
            rows[r][c] = ch
        return ["".join(r) for r in rows]

    def char(self, c: int, r: int) -> str:
        return self.text[r][c] if 0 <= r < len(self.text) and 0 <= c < len(self.text[r]) else VOID

    def solid(self, c: int, r: int) -> bool:
        ld = self.ld
        return not (0 <= r < ld.rows and 0 <= c < ld.cols) or ld.grid[r][c] in SOLIDS

    def open(self, c: int, r: int) -> bool:
        """A cell a body may pass through: inside a screen, not solid, not spikes."""
        ld = self.ld
        return 0 <= r < ld.rows and 0 <= c < ld.cols and ld.grid[r][c] in (AIR, LADDER)

    def reachable(self, text: list[str], jump: int = JUMP_CELLS, drift: int = DRIFT_CELLS):
        """(can the exit be reached from the true start, cost of the route from there, the field)."""
        ld = self.parse(text)
        cost = route_field(ld, jump, drift)
        start = float(cost[ld.start[1], ld.start[0]])
        return math.isfinite(start), start, cost


def route(stage: Stage, sec: dict) -> list[tuple[int, int]]:
    """The cells of the cheapest route through a section, from its start to where it ends."""
    cost = stage.cost
    own = {tuple(s) for s in sec["screens"]}
    c, r = sec["start"]
    end = sec.get("end")
    stop = float(cost[end[1], end[0]]) if end else 0.0
    path = [(c, r)]
    while cost[r, c] > stop and len(path) < 4000:
        here, best = float(cost[r, c]), None
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if (dr or dc) and 0 <= nr < cost.shape[0] and 0 <= nc < cost.shape[1] and cost[nr, nc] < here:
                    off = abs(float(cost[nr, nc]) + (1.0 if not dr or not dc else 1.4) - here)   # the move the field took
                    if best is None or (off, float(cost[nr, nc])) < best[0]:
                        best = ((off, float(cost[nr, nc])), nc, nr)
        if best is None:
            break
        c, r = best[1], best[2]
        path.append((c, r))
    return [p for p in path if (p[0] // stage.sc, p[1] // stage.sr) in own]


def middle_out(n: int, margin: int = 3) -> list[int]:
    """Indices lo..hi of a route, starting from its middle and alternating outward."""
    idx = list(range(margin, max(margin, n - margin)))
    mid = len(idx) // 2
    order = []
    for k in range(len(idx)):
        j = mid + ((k + 1) // 2) * (1 if k % 2 else -1)
        if 0 <= j < len(idx):
            order.append(idx[j])
    return order


# ── level operators: each returns (cells to rewrite, parameter, description, direction) or a reason ──
def seal_route(st: Stage, sec: dict, path: list, own) -> tuple | str:
    """A wall across the passage, at the first place along the route where it cuts the exit off."""
    for i in middle_out(len(path) - 1)[:60]:
        (c, r), (nc, nr) = path[i], path[i + 1]
        across = (0, 1) if nc != c and nr == r else (1, 0)     # walking: a wall floor to ceiling; climbing: a lid
        bar = [(c, r)]
        for sign in (-1, 1):
            x, y = c + sign * across[0], r + sign * across[1]
            while st.open(x, y) and own(x, y) and len(bar) < 16:
                bar.append((x, y))
                x, y = x + sign * across[0], y + sign * across[1]
        if any(st.char(x, y) in KEEP for x, y in bar) or any((x, y) == tuple(sec["start"]) for x, y in bar):
            continue
        cells = {p: SOLID for p in bar}
        if not st.reachable(st.edit(cells))[0]:
            return cells, {"tiles": len(bar), "wall": "vertical" if across == (0, 1) else "horizontal"}, \
                f"A {len(bar)}-tile wall across the route at ({c}, {r})", "harder"
    return "no wall of up to 16 tiles cuts the exit off (there is another way round)"


def remove_ladder(st: Stage, sec: dict, path: list, own) -> tuple | str:
    """The first ladder the route climbs, taken out whole."""
    seen = set()
    runs = []
    for c, r in path:
        if st.char(c, r) != LADDER or (c, r) in seen:
            continue
        top = bottom = r
        while st.char(c, top - 1) == LADDER:
            top -= 1
        while st.char(c, bottom + 1) == LADDER:
            bottom += 1
        cells = [(c, y) for y in range(top, bottom + 1)]
        seen.update(cells)
        runs.append(cells)
    if not runs:
        return "the route uses no ladder here"
    inside = [run for run in runs if all(own(*p) for p in run)]
    if not inside:
        return "the only ladders on the route here are shared with the next section or the one before"
    run = inside[0]
    c = run[0][0]
    return {p: AIR for p in run}, {"col": c, "top": run[0][1], "bottom": run[-1][1], "tiles": len(run)}, \
        f"The {len(run)}-tile ladder in column {c}, rows {run[0][1]}-{run[-1][1]}, removed", "harder"


def widen_gap(st: Stage, sec: dict, path: list, own, width: int = 2) -> tuple | str:
    """The ledge the route jumps from, cut back by two tiles."""
    for i in middle_out(len(path) - 1, margin=2):
        (c, r), (nc, nr) = path[i], path[i + 1]
        s = nc - c
        if not s or not st.solid(c, r + 1) or st.char(c, r + 1) != SOLID or st.solid(c + s, r + 1):
            continue                                            # not the last tile of a floor before open space
        cells = {}
        for k in range(width):
            x, y = c - s * k, r + 1
            while st.char(x, y) == SOLID and own(x, y):         # the whole thickness of the ledge
                cells[(x, y)] = AIR
                y += 1
        if len({x for x, _ in cells}) < width or tuple(sec["start"]) in {(x, y - 1) for x, y in cells}:
            continue
        return cells, {"col": c, "row": r + 1, "width": width, "side": "right" if s > 0 else "left", "tiles": len(cells)}, \
            f"The ledge at ({c}, {r + 1}) cut back by {width} tiles, widening the gap to its {'right' if s > 0 else 'left'}", "harder"
    return "the route crosses no gap here"


def add_hazard(st: Stage, sec: dict, path: list, own) -> tuple | str:
    """Spikes standing on a floor tile the route walks over: they have to be jumped."""
    fallback = None
    for i in middle_out(len(path), margin=4):
        c, r = path[i]
        flat = all(st.char(x, r + 1) == SOLID and st.open(x, r) and st.ld.grid[r][x] == AIR for x in (c - 1, c, c + 1))
        if not flat or not own(c, r) or st.char(c, r) in KEEP or (c, r) == tuple(sec["start"]):
            continue
        cells = {(c, r): SPIKE}
        made = (cells, {"col": c, "row": r, "glyph": SPIKE}, f"Spikes on the floor at ({c}, {r})", "harder")
        if st.reachable(st.edit(cells))[0]:
            return made
        fallback = fallback or made
    return fallback or "the route crosses no flat floor three tiles wide here"


def add_enemy(st: Stage, sec: dict, path: list, own, glyph: str = "b") -> tuple | str:
    """A Blader (it flies, so it needs no wall or floor) placed above the route."""
    spawns = [(c, r) for _, c, r in st.ld.spawns]
    for i in middle_out(len(path), margin=6):
        c, r = path[i]
        for y in (r - 2, r - 1, r):
            if st.ld.grid[y][c] == AIR and st.char(c, y) in (AIR, VOID) and own(c, y) \
                    and all(abs(c - sc) + abs(y - sr) >= 4 for sc, sr in spawns):
                return {(c, y): glyph}, {"col": c, "row": y, "glyph": glyph, "enemy": "Blader"}, \
                    f"A Blader added at ({c}, {y}), on the route", "harder"
    return "no open cell on the route at least four tiles from every other enemy"


def remove_enemy(st: Stage, sec: dict, path: list, own) -> tuple | str:
    """The enemy closest to the route, removed."""
    mine = [(g, c, r) for g, c, r in st.ld.spawns if g in en.ENEMY_GLYPHS and own(c, r)]
    if not mine or not path:
        return "no enemy in this section"
    g, c, r = min(mine, key=lambda s: (min(abs(s[1] - pc) + abs(s[2] - pr) for pc, pr in path), s[2], s[1]))
    return {(c, r): AIR}, {"col": c, "row": r, "glyph": g}, f"The enemy '{g}' at ({c}, {r}) removed", "easier"


def cosmetic_tile(st: Stage, sec: dict, path: list, own) -> tuple | str:
    """One air cell on the route spelled the other way ('.' and a blank are the same tile)."""
    for i in middle_out(len(path), margin=2):
        c, r = path[i]
        ch = st.char(c, r)
        if ch in (AIR, VOID) and st.ld.grid[r][c] == AIR:
            new = VOID if ch == AIR else AIR
            if new == VOID and c == len(st.text[r].rstrip()) - 1:
                continue                                        # a trailing blank could be trimmed away
            return {(c, r): new}, {"col": c, "row": r, "from": ch, "to": new}, \
                f"The air cell at ({c}, {r}) written as {new!r} instead of {ch!r}: the same tile", "none"
    return "no plain air cell on the route"


def off_route(st: Stage, sec: dict, path: list, own) -> tuple | str:
    """One tile hollowed out deep inside a wall, where nothing can reach or see it."""
    if not path:
        return "no route here"
    mc, mr = path[len(path) // 2]
    for depth in (2, 1):
        buried = [(c, r) for sx, sy in sorted(tuple(s) for s in sec["screens"])
                  for r in range(sy * st.sr, (sy + 1) * st.sr) for c in range(sx * st.sc, (sx + 1) * st.sc)
                  if all(st.char(c + dc, r + dr) == SOLID and own(c + dc, r + dr)
                         for dr in range(-depth, depth + 1) for dc in range(-depth, depth + 1))]
        if buried:
            c, r = min(buried, key=lambda p: (abs(p[0] - mc) + abs(p[1] - mr), p[1], p[0]))
            return {(c, r): AIR}, {"col": c, "row": r, "depth": depth}, \
                f"The solid tile at ({c}, {r}), {depth} tile{'s' if depth > 1 else ''} deep inside a wall, hollowed out", "none"
    return "no wall thick enough to hide a change in"


OPERATORS = {f.__name__: f for f in (seal_route, remove_ladder, widen_gap, add_hazard, add_enemy, remove_enemy,
                                     cosmetic_tile, off_route)}
NEUTRAL_BY_DESIGN = {"cosmetic_tile", "off_route"}


def classify(st: Stage, operator: str, text: list[str]) -> tuple[str, dict]:
    """The mutant's class, from the route field searched from the stage's true start."""
    ok, start, cost = st.reachable(text)
    before = float(st.cost[st.ld.start[1], st.ld.start[0]])
    checks = {"exit_reachable": ok, "route_cost_before": round(before, 1), "route_cost_after": round(start, 1) if ok else None}
    if not ok:
        return "breaking", checks
    if operator in NEUTRAL_BY_DESIGN:
        ld = st.parse(text)
        same_game = ld.grid == st.ld.grid and ld.spawns == st.ld.spawns
        import numpy as np
        reach = np.isfinite(st.cost)
        same_routes = bool(np.array_equal(np.isfinite(cost), reach) and np.allclose(cost[reach], st.cost[reach]))
        checks.update({"same_game": same_game, "same_routes": same_routes})
        if same_game or same_routes:
            return "neutral", checks
    return "degrading", checks


def jump_reach(player: dict) -> tuple[int, int]:
    """(cells a jump lifts the feet, cells it carries sideways) for these movement values."""
    v, g = float(player["jump_velocity"]), float(player["gravity"])
    height, frames = 0.0, 0
    while v > 0:
        height += v
        v -= g
        frames += 1
    return int(height // 16), int(2 * frames * float(player["air_speed"]) // 16)


# ── generating ───────────────────────────────────────────────────────────────────────────────
def generate(stage: str = "CutMan", out: str = OUT) -> dict:
    st = Stage(stage)
    mutants, skipped = [], []
    for n, (name, sec) in enumerate(st.sections, 1):
        screens = {tuple(s) for s in sec["screens"]}

        def own(c, r, screens=screens):
            return (c // st.sc, r // st.sr) in screens

        path = route(st, sec)
        for op in LEVEL_OPERATORS:
            made = OPERATORS[op](st, sec, path, own)
            if isinstance(made, str):
                skipped.append({"section": name, "operator": op, "reason": made})
                continue
            cells, parameter, description, direction = made
            text = st.edit(cells)
            cls, checks = classify(st, op, text)
            changed = sorted(cells)
            record = {"id": f"s{n}-{op}", "game": GAME, "stage": stage, "section": name, "family": "level", "operator": op,
                      "class": cls, "direction": direction, "description": description, "parameter": parameter,
                      "location": {"col": changed[len(changed) // 2][0], "row": changed[len(changed) // 2][1]},
                      "cells": [[c, r, st.char(c, r), cells[(c, r)]] for c, r in changed],     # col, row, before, after
                      "base_file": st.file, "level_file": "level.txt", "config": None, "checks": checks}
            mutants.append((record, text))
    base_jump = jump_reach(st.cfg["player"])
    for mid, key, change, direction in MECHANICS:
        node = st.cfg
        *parents, leaf = key.split(".")
        for k in parents:
            node = node[k]
        old, new = node[leaf], change(node[leaf])
        player = dict(st.cfg["player"], **({leaf: new} if parents == ["player"] else {}))
        jump, drift = jump_reach(player)
        # the field's own reach where the values leave the jump as it was; the mutated reach where they do not
        jump, drift = (JUMP_CELLS if jump == base_jump[0] else jump), (DRIFT_CELLS if drift == base_jump[1] else drift)
        ok, start, _ = st.reachable(st.text, jump, drift)
        before = float(st.cost[st.ld.start[1], st.ld.start[0]])
        record = {"id": f"m-{mid}", "game": GAME, "stage": stage, "section": None, "family": "mechanics", "operator": mid,
                  "class": "breaking" if not ok else "degrading", "direction": direction,
                  "description": f"{key} changed from {old} to {new}", "parameter": {"key": key, "from": old, "to": new},
                  "location": None, "cells": [], "base_file": st.file, "level_file": None, "config": {key: new},
                  "checks": {"exit_reachable": ok, "route_cost_before": round(before, 1),
                             "route_cost_after": round(start, 1) if ok else None, "jump_cells": jump, "drift_cells": drift}}
        mutants.append((record, None))
    os.makedirs(out, exist_ok=True)
    for record, text in mutants:                    # a mutant's folder also holds its results later: only these two files are written
        folder = os.path.join(out, record["id"])
        os.makedirs(folder, exist_ok=True)
        if text is not None:
            with open(os.path.join(folder, "level.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(text) + "\n")
        with open(os.path.join(folder, "mutant.json"), "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
    index = {"game": GAME, "stage": stage, "base_file": st.file, "mutants": [r for r, _ in mutants], "skipped": skipped}
    with open(os.path.join(out, "mutants.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    return index


def load(out: str = OUT) -> dict | None:
    try:
        with open(os.path.join(out, "mutants.json"), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def table(index: dict) -> str:
    lines = [f"{'MUTANT':<26}{'CLASS':<11}{'SECTION':<20}WHAT CHANGED"]
    for m in index["mutants"]:
        lines.append(f"{m['id']:<26}{m['class']:<11}{m['section'] or 'whole game':<20}{m['description']}")
    by = {}
    for m in index["mutants"]:
        by[m["class"]] = by.get(m["class"], 0) + 1
    lines += ["", f"{len(index['mutants'])} mutants: " + ", ".join(f"{v} {k}" for k, v in sorted(by.items()))]
    if index["skipped"]:
        lines += ["", "Skipped (the operator has nothing to act on there):"]
        lines += [f"  {s['section']:<20}{s['operator']:<16}{s['reason']}" for s in index["skipped"]]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate game mutants of a Mega Man stage")
    ap.add_argument("--stage", default="CutMan", help="the full-stage level id")
    ap.add_argument("--out", default=OUT, help="where the mutants go (default: runs/megaman/mutants)")
    ap.add_argument("--list", action="store_true", help="print the mutants already generated and exit")
    args = ap.parse_args()
    index = load(args.out) if args.list else generate(args.stage, args.out)
    if index is None:
        raise SystemExit(f"no mutants in {args.out} — run without --list first")
    print(table(index))
    if not args.list:
        print(f"\nwrote {args.out}/<mutant>/mutant.json (+ level.txt) and {args.out}/mutants.json")


if __name__ == "__main__":
    main()
