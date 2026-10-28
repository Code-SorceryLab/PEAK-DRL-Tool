"""The AST evaluation metrics, over playtest attempts (code/neuro/ast_eval.py play) and people at
the keyboard (manual_play --record runs/ast/<game>/<variant>/human).

    python -m code.stats.ast_metrics                   # every game: tables, runs/ast/metrics.json
    python -m code.stats.ast_metrics --game mario

    metric            what it is                                                         RQ
    completion        solved by at least one agent (yes/no), the strongest signal       1
    win rate          wins / attempts: agents per persona (steps of 1/3; Mega Man: its
                      relay's attempts), people per player, both pooled                  1, 3
    progress          median and max `progress` over the attempts that did not win      1, 3
    end location      the tile where each failed attempt ended                          1 (localization), 3
    death cluster     the region holding most failed attempts: the mode of the end
                      tiles, merged within 3 tiles                                      1, 3
    cause mix         share of failed attempts per `cause`                              1, 3
    time to finish    median `time_s` over wins                                         1 (degrading changes)
    play coverage     the tiles the routes pass through, over all attempts              1 (why a mutant is missed)
    effort            median jumps, kills, pickups, damage taken (Mega Man), bombs
                      placed (Bomberman), wall jumps (Meat Boy)                         1 (how play changed)
    segment reached   Mega Man: the segment each attempt ended in, time and damage
                      per segment                                                       1 (by segment), 3

The rule (fixed 2026-10-25, after the pilot runs; see paper-ast/ast_rule.md). A mutant is
compared with the original level on four signals: completion lost, a change in win rate (Fisher's
exact test on the pooled agent attempts, p < 0.05), the death cluster moved more than 3 tiles, and
a new dominant cause of death; any one of them flags it. A comparison only counts when agents
solve the original (an unsolved original cannot lose completion and has no route to break). Each
counted comparison gets one verdict: broken (completion lost), changed (any other signal) or
unchanged. Reported beside it, not part of the rule: whether the mutant's win rate falls outside
the spread of the original's per-seed win rates (the outline's alternative to Fisher).

Two modes per mutant: retrain (agents.csv, agents trained on the mutant) and replay
(agents_replay.csv, the original's frozen agents on it). The cosmetic mutant is only replayed: it
passes the determinism check (RQ2) when every agent's attempt matches its attempt on the original,
field for field. base-reseeded, the original retrained with three new seeds, is compared with the
original like a mutant: every signal it raises is a false alarm. Mega Man's mutants are compared
with base-retrained, the original retrained section by section exactly as they were; base, its
relay of hand-picked section agents, stays on the page as context.

Every other level of the 2D games has the same variants, named <id>@<level>: base@<level> (that
level's baseline agents), base-reseeded@<level> and the mutants placed there by rule
(code/games/eval_mutants.py place()). Each is compared with base@<level>.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
from collections import Counter

from code.stats import episode_log

ROOT = os.path.join("runs", "ast")
GAMES = ["mario", "meatboy", "bomberman", "megaman"]
RETRAINED = "base-retrained"            # Mega Man's original retrained like its mutants (code/neuro/ast_eval.py)
RESEEDED = "base-reseeded"              # the original retrained with new seeds: the false-alarm check
ORIGINALS = ("base", RETRAINED)
SIGNALS = ("completion_lost", "win_rate_changed", "cluster_moved", "cause_changed")
CLUSTER_TILES = 3
ALPHA = 0.05
# (label, column) per effort count; a column a game does not log is left out
EFFORT = [("jumps", "jumps"), ("kills", "kills"), ("pickups", "coins"), ("pickups", "powerups"),
          ("damage taken", "damage_taken"), ("bombs placed", "bombs_placed"), ("wall jumps", "wall_jumps")]
SAME_ATTEMPT = ("status", "cause", "frames", "end_x", "end_y", "route")


# ── reading ──────────────────────────────────────────────────────────────────────────────────
MODES = {"retrain": "agents.csv", "replay": "agents_replay.csv", "random": "agents_random.csv"}


def load(game: str, variant: str, mode: str = "retrain") -> list[dict]:
    """Every attempt at this variant: the agents' playtest and any people's recorded play; or the
    original's frozen agents on it (replay), or the random-action baseline (random)."""
    base = os.path.join(ROOT, game, variant)
    if mode != "retrain":
        path = os.path.join(base, MODES[mode])
        return episode_log.read(path) if os.path.exists(path) else []
    rows = []
    for path in [os.path.join(base, "agents.csv")] + sorted(glob.glob(os.path.join(base, "human", "**", "episodes.csv"),
                                                                     recursive=True)):
        if os.path.exists(path):
            rows += episode_log.read(path)
    return rows


def kind(v: str) -> str:
    """What a variant is, without its level or section: "base-reseeded@5 @ x" -> "base-reseeded"."""
    return v.split(" @ ")[0].split("@")[0]


def is_original(v: str) -> bool:
    """base, base-retrained, or another level's base (base@<level>)."""
    return v.split(" @ ")[0] in ORIGINALS or kind(v) == "base"


def variants(game: str) -> list[str]:
    """The originals first, then every variant that has attempts on disk (mutant ids in the evaluation
    order), then each other level's: its base, its reseeded base and its placed mutants."""
    try:
        from code.games.eval_mutants import eval_set, placed_set
        order = [m["id"] for m in eval_set(game)]
        for m in placed_set(game):
            order += [v for v in (f"base@{m['stage']}", f"{RESEEDED}@{m['stage']}") if v not in order] + [m["id"]]
    except Exception:  # noqa: BLE001 — the tables still work from what is on disk
        order = []
    files = glob.glob(os.path.join(ROOT, game, "*", "agents.csv"))
    files += glob.glob(os.path.join(ROOT, game, "*", "human", "**", "episodes.csv"), recursive=True)
    found = {os.path.relpath(p, os.path.join(ROOT, game)).split(os.sep)[0] for p in files}
    rest = sorted(found - set(order) - set(ORIGINALS) - {RESEEDED})
    head = [v for v in order if "@" not in v]
    return [v for v in [*ORIGINALS, RESEEDED, *head, *rest, *(v for v in order if "@" in v)] if v in found]


# ── per attempt ──────────────────────────────────────────────────────────────────────────────
def _tile_size(row: dict) -> float:
    return float(row.get("tile") or 32)


def end_tile(row: dict) -> tuple[int, int] | None:
    if row.get("end_x") is None or row.get("end_y") is None:
        return None
    t = _tile_size(row)
    return int(float(row["end_x"]) // t), int(float(row["end_y"]) // t)


def route_tiles(row: dict) -> set[tuple[int, int]]:
    """The tiles a route passes through. Points are 8 frames apart, so each step between two is
    walked in quarter-tile increments to fill the tiles between them."""
    pts, t = row.get("route") or [], _tile_size(row)
    out = set()
    for (x0, y0), (x1, y1) in zip(pts, pts[1:] or pts):
        n = max(1, int(max(abs(x1 - x0), abs(y1 - y0)) / (t / 4)))
        out |= {(int((x0 + (x1 - x0) * k / n) // t), int((y0 + (y1 - y0) * k / n) // t)) for k in range(n + 1)}
    return out


def _won(r: dict) -> bool:
    return r.get("status") == "WON"


def _median(vals: list) -> float | None:
    vals = [float(v) for v in vals if v is not None]
    return round(statistics.median(vals), 3) if vals else None


# ── over a set of attempts ───────────────────────────────────────────────────────────────────
def death_cluster(tiles: list[tuple[int, int]], radius: int = CLUSTER_TILES) -> dict | None:
    """The end tile with the most failed attempts within `radius` tiles (Chebyshev) of it."""
    if not tiles:
        return None
    count = Counter(tiles)
    near = {t: sum(n for u, n in count.items() if max(abs(u[0] - t[0]), abs(u[1] - t[1])) <= radius) for t in count}
    centre = max(near, key=lambda t: (near[t], count[t], -t[0], -t[1]))
    return {"col": centre[0], "row": centre[1], "attempts": near[centre], "share": round(near[centre] / len(tiles), 3)}


def win_rate(rows: list[dict], key: str = "persona") -> dict:
    """{group: [wins, attempts]} plus "pooled"."""
    out: dict = {}
    for r in rows:
        w = out.setdefault(str(r.get(key) or "?"), [0, 0])
        w[0] += _won(r)
        w[1] += 1
    out["pooled"] = [sum(_won(r) for r in rows), len(rows)]
    return out


def seed_win_rates(rows: list[dict]) -> dict:
    """{training seed: [wins, attempts]}, the seed read from the run dir (".../Mario1-1_1234");
    empty where the attempts do not name one run per seed (Mega Man's relay)."""
    out: dict = {}
    for r in rows:
        tail = os.path.basename(str(r.get("run") or "")).rsplit("_", 1)
        if len(tail) == 2 and tail[1].isdigit():
            w = out.setdefault(tail[1], [0, 0])
            w[0] += _won(r)
            w[1] += 1
    return out


def route_lines(rows: list[dict], points: int = 80) -> list[dict]:
    """Every attempt's route in tile units, at most `points` points, for the map: {"w": won, "p": [[x, y], ...]}."""
    out = []
    for r in rows:
        pts, t = r.get("route") or [], _tile_size(r)
        if len(pts) >= 2:
            step = max(1, len(pts) // points)
            out.append({"w": _won(r), "p": [[round(x / t, 2), round(y / t, 2)] for x, y in pts[::step] + pts[-1:]]})
    return out


def summarize(rows: list[dict]) -> dict:
    """The metrics of one set of attempts (one variant; agents or people, never mixed)."""
    failed = [r for r in rows if not _won(r)]
    ends = [t for t in map(end_tile, failed) if t]
    causes = Counter(r.get("cause") or "?" for r in failed)
    coverage = set().union(*map(route_tiles, rows)) if rows else set()
    progress = [r.get("progress") for r in failed if r.get("progress") is not None]
    out = {
        "attempts": len(rows),
        "completion": any(map(_won, rows)),
        "win_rate": win_rate(rows),
        "seed_win_rates": seed_win_rates(rows),
        "progress_median": _median(progress),
        "progress_max": round(max(map(float, progress)), 3) if progress else None,
        "end_tiles": sorted(ends),
        "death_cluster": death_cluster(ends),
        "causes": {c: round(n / len(failed), 3) for c, n in causes.most_common()},
        "time_to_finish": _median([r.get("time_s") for r in rows if _won(r)]),
        "coverage": sorted(coverage),
        "routes": route_lines(rows),
        "effort": {},
    }
    for label, col in EFFORT:
        if any(r.get(col) is not None for r in rows) and not (col == "jumps" and rows[0].get("game") == "bomberman"):
            out["effort"][label] = _median([r.get(col) for r in rows])   # Bomberman's "jump" is the bomb button
    if any(r.get("sections") for r in rows):                             # Mega Man
        ended = Counter(r.get("section") or "?" for r in rows)
        per = {}
        for r in rows:
            for s in r.get("sections") or []:
                d = per.setdefault(s["section"], {"time": [], "damage": []})
                d["time"].append(s["frames"] / 60.0)
                d["damage"].append(s["damage"])
        out["segments"] = {"ended_in": dict(ended),
                           "per_segment": {k: {"entered": len(v["time"]), "time_median": _median(v["time"]),
                                               "damage_median": _median(v["damage"])} for k, v in per.items()}}
    return out


# ── comparing a mutant with the original ─────────────────────────────────────────────────────
def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided p of the 2x2 table [[a, b], [c, d]]: the summed chance of every table with the
    same margins that is no more likely than this one."""
    n1, n2, k = a + b, c + d, a + c
    n = n1 + n2
    if n == 0:
        return 1.0

    def prob(x: int) -> float:
        return math.comb(n1, x) * math.comb(n2, k - x) / math.comb(n, k)

    here = prob(a)
    return min(1.0, sum(p for x in range(max(0, k - n2), min(k, n1) + 1) if (p := prob(x)) <= here * (1 + 1e-9)))


def compare(base: dict, mut: dict) -> dict:
    """The four signals a mutant can raise against the original level, whether the comparison
    counts (the original is solved), the verdict, and the seed-spread check beside them."""
    bw, bn = base["win_rate"]["pooled"]
    mw, mn = mut["win_rate"]["pooled"]
    p = fisher_exact(bw, bn - bw, mw, mn - mw)
    bc, mc = base["death_cluster"], mut["death_cluster"]
    moved = bool(bc and mc and max(abs(bc["col"] - mc["col"]), abs(bc["row"] - mc["row"])) > CLUSTER_TILES)
    new_cluster = bool(mc and not bc)
    top = [{c for c, s in d.items() if s == max(d.values())} for d in (base["causes"], mut["causes"])]  # ties: all
    signals = {
        "completion_lost": base["completion"] and not mut["completion"],
        "win_rate_p": round(p, 4),
        "win_rate_changed": p < ALPHA,
        "cluster_moved": moved or new_cluster,
        "cause_changed": bool(top[1] and not top[0] & top[1]),
    }
    rates = [w / n for w, n in (base.get("seed_win_rates") or {}).values() if n]
    signals["win_rate_beyond_seeds"] = (not min(rates) <= mw / mn <= max(rates)) if len(rates) >= 2 and mn else None
    signals["counted"] = bool(base["completion"])
    signals["detected"] = signals["counted"] and any(signals[k] for k in SIGNALS)
    signals["verdict"] = ("not counted" if not signals["counted"] else "broken" if signals["completion_lost"]
                          else "changed" if signals["detected"] else "unchanged")
    return signals


def determinism(base_rows: list[dict], mut_rows: list[dict]) -> dict:
    """RQ2: does every agent replay the cosmetic copy exactly as it played the original?"""
    def key(r):
        return r.get("run") or f"{r.get('persona')}/{r.get('agent')}/{r.get('seed')}"
    base = {key(r): r for r in base_rows if r.get("source") == "agent"}
    pairs = [(base.get(key(r)), r) for r in mut_rows if r.get("source") == "agent"]
    differ = [key(m) for b, m in pairs if b is None or any(b.get(f) != m.get(f) for f in SAME_ATTEMPT)]
    return {"compared": len(pairs), "identical": len(pairs) - len(differ), "differ": differ,
            "passed": bool(pairs) and not differ}


# ── one game, every variant ──────────────────────────────────────────────────────────────────
def references(game: str) -> dict:
    """{level: (name, agent attempts)}: the original each variant is compared with over that level,
    base-retrained where it played the level, else base (another level's: base@<level>)."""
    from code.games.eval_mutants import EVAL_LEVELS
    main = EVAL_LEVELS[game][0]
    refs: dict = {}
    others = [v for v in variants(game) if "@" in v and is_original(v)]
    for ref in [ORIGINALS[0], *others, ORIGINALS[1]]:  # the retrained original replaces the relay where it played
        rows: dict = {}
        for r in load(game, ref):
            if r.get("source") == "agent":
                rows.setdefault(str(r.get("level")), []).append(r)
        refs.update({level: (ref if level == main or "@" in ref else f"{ref} @ {level}", rs) for level, rs in rows.items()})
    return refs


def _alike(ref: list[dict], rows: list[dict]) -> list[dict]:
    """The original's attempts by the personas that played `rows`: a variant played by fewer personas
    (Mega Man's base-reseeded: experienced only) is never judged against personas it lacks."""
    personas = {r.get("persona") for r in rows}
    return [r for r in ref if r.get("persona") in personas]


def evaluate(game: str) -> dict:
    """{variant: {"agents": metrics, "humans": metrics or None, "vs_base": signals, "vs": the
    original it was compared with, ...}}. A variant is compared with the original played over the
    same level, by base-retrained where it played that level. Mega Man's section mutants are played
    from their section's start, and the originals and the global mutant over each section too,
    listed as "<variant> @ <section>". Every persona's attempts are pooled, in every game."""
    from code.games.eval_mutants import EVAL_LEVELS
    main = EVAL_LEVELS[game][0]
    refs = references(game)
    rand_ref: dict = {}
    for ref in ["base"] + [v for v in variants(game) if "@" in v and kind(v) == "base"]:   # each level's own original
        for r in load(game, ref, "random"):
            rand_ref.setdefault(str(r.get("level")), []).append(r)
    out = {}
    for v in variants(game):
        by_level: dict = {}
        for r in load(game, v):
            by_level.setdefault(str(r.get("level")), []).append(r)
        replayed: dict = {}
        for r in load(game, v, "replay"):
            replayed.setdefault(str(r.get("level")), []).append(r)
        randoms: dict = {}
        for r in load(game, v, "random"):
            randoms.setdefault(str(r.get("level")), []).append(r)
        for level, rs in sorted(by_level.items(), key=lambda kv: (kv[0] != main, kv[0])):
            one = not is_original(v) and len(by_level) == 1      # a section mutant keeps its own name
            key = v if level == main or one or "@" in v else f"{v} @ {level}"
            agents = [r for r in rs if r.get("source") == "agent"]
            humans = [r for r in rs if r.get("source") != "agent"]
            entry = {"agents": summarize(agents) if agents else None, "level": level,
                     "humans": dict(summarize(humans), win_rate=win_rate(humans)) if humans else None}
            vs, ref = refs.get(level, (None, None))
            if not is_original(v) and ref and agents and _alike(ref, agents):
                # only the personas both played: either side may lack one (Mega Man trains section by section)
                entry["vs_base"] = compare(summarize(_alike(ref, agents)), summarize(_alike(agents, ref)))
                entry["vs"] = vs
                if kind(v).endswith("cosmetic_tile"):            # placed copies too (cosmetic_tile@<level>)
                    entry["determinism"] = determinism(_alike(ref, agents), _alike(agents, ref))
                if replayed.get(level) and _alike(ref, replayed[level]):
                    rep = summarize(replayed[level])
                    entry["replay"] = {"agents": rep, "vs_base": compare(summarize(_alike(ref, replayed[level])),
                                                                         summarize(_alike(replayed[level], ref)))}
            if not is_original(v) and randoms.get(level) and rand_ref.get(level):
                rnd = summarize(randoms[level])          # the random baseline against itself on the original
                entry["random"] = {"agents": rnd, "vs_base": compare(summarize(rand_ref[level]), rnd)}
            out[key] = entry
    return out


def false_alarms(results: dict) -> dict:
    """How often each signal fires on base-reseeded, the original against itself trained with new
    seeds, over every counted comparison (a game's level, or a Mega Man section)."""
    nulls = [e["vs_base"] for res in results.values() for k, e in res.items()
             if kind(k) == RESEEDED and (e.get("vs_base") or {}).get("counted")]
    fired = {s: sum(bool(n[s]) for n in nulls) for s in (*SIGNALS, "detected")}
    fired["strict"] = sum(bool(n["completion_lost"] or n["win_rate_changed"]) for n in nulls)   # completion or Fisher only
    spread = [n["win_rate_beyond_seeds"] for n in nulls if n.get("win_rate_beyond_seeds") is not None]
    return {"comparisons": len(nulls), "fired": fired,
            "beyond_seeds": [sum(spread), len(spread)]}


def table(game: str, results: dict) -> str:
    """The text table the menu prints: one line per variant."""
    width = max(map(len, results), default=20) + 2
    lines = [f"{game}", f"  {'VARIANT':<{width}}{'WINS':>8}{'SOLVED':>8}{'PROG':>7}{'TIME':>7}  {'CLUSTER':<14}{'CAUSE':<12}"
                        f"{'VERDICT':<13}{'REPLAY':<20}SIGNALS"]
    for v, e in results.items():
        a = e["agents"]
        if not a:
            continue
        w, n = a["win_rate"]["pooled"]
        c = a["death_cluster"]
        cause = next(iter(a["causes"]), "-")
        sig = e.get("vs_base") or {}
        flags = [k for k in SIGNALS if sig.get(k)]
        if "determinism" in e:
            d = e["determinism"]
            flags.append(f"replay {d['identical']}/{d['compared']} identical")
        rep = e.get("replay")
        replay = (f"{rep['agents']['win_rate']['pooled'][0]}/{rep['agents']['win_rate']['pooled'][1]} "
                  f"{rep['vs_base']['verdict']}") if rep else "-"
        prog = "-" if a["progress_median"] is None else f"{100 * a['progress_median']:.0f}%"
        tff = "-" if a["time_to_finish"] is None else f"{a['time_to_finish']:.1f}s"
        where = f"({c['col']},{c['row']}) {c['share']:.0%}" if c else "-"
        lines.append(f"  {v:<{width}}{f'{w}/{n}':>8}{'yes' if a['completion'] else 'no':>8}{prog:>7}{tff:>7}  "
                     f"{where:<14}{cause:<12}{sig.get('verdict', '-'):<13}{replay:<20}"
                     f"{', '.join(flags) or ('-' if v.startswith('base') else 'none')}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="AST evaluation metrics over the playtest attempts")
    ap.add_argument("--game", action="append", choices=GAMES, help="only this game (repeatable)")
    ap.add_argument("--page", action="store_true",
                    help="also rebuild the Command Center (runs/site/), its Mutations tab included")
    args = ap.parse_args()
    games = args.game or GAMES
    results = {g: evaluate(g) for g in (GAMES if args.page else games)}
    if args.page:
        from code.neuro import command_center
        print(f"wrote {len(command_center.build_site())} files to {command_center.SITE_DIR}")
    os.makedirs(ROOT, exist_ok=True)
    path = os.path.join(ROOT, "metrics.json")
    old = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            old = json.load(f)
    with episode_log.atomic_open(path, "w", encoding="utf-8") as f:     # the next run reads and merges it
        json.dump({**old, **results}, f, indent=1, default=list)
    for g in games:
        print(table(g, results[g]) if results[g] else f"{g}\n  no playtests yet: python -m code.neuro.ast_eval play --game {g}")
        print()
    fa = false_alarms(results)
    if fa["comparisons"]:
        print(f"false alarms ({RESEEDED} vs the original, {fa['comparisons']} comparisons): "
              + ", ".join(f"{k} {v}" for k, v in fa["fired"].items())
              + f", beyond the seed spread {fa['beyond_seeds'][0]}/{fa['beyond_seeds'][1]}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
