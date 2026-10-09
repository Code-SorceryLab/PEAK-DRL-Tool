"""The balance metrics: every metric defined once, used by every consumer.

balance.py (probe reports), the trainer and manual play (metrics.json per run / session), the
Balance Command report and the Streamlit dashboard all call compute_level() on the same
episode rows (code/stats/episode_stats.py), so a metric can't mean one thing in the report and
another on the dashboard.

Metrics follow the team's concrete-metrics table: eight balance dimensions, shared metrics plus
game-specific ones. A metric that doesn't apply to a game is not computed for it; one that
applies but has no data (no wins yet, no coins in the level) is None, shown as "N/A".

Adding a metric: record its raw input in the game's EpisodeStats.to_dict(), then add one
Metric(...) entry below. Threshold bands, if any, go in code/stats/thresholds/<game>.yaml.
"""
from __future__ import annotations

import math
import os
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

DIMENSIONS = {
    1: "Challenge calibration",
    2: "Punishment severity",
    3: "Triangularity (risk vs reward)",
    4: "Path / strategy diversity",
    5: "Skill expression",
    6: "Progression fit",
    7: "Emergent complexity",
    8: "Reward density",
}

# Skill tiers -> the persona that stands in for them. "expert" is the speedrunner: it sprints and
# is paid for time left on the clock, the closest persona to expert play.
# skill tier -> persona. The novice–expert gaps compare novice with expert (= experienced); speedrunner
# is its own tier, listed by "completion per skill" but not one end of a gap.
TIERS = {"novice": "novice", "expert": "experienced", "speedrunner": "speedrunner"}

SCROLLERS = frozenset({"mario", "megaman", "sonic"})
ALL_GAMES = frozenset({"mario", "megaman", "sonic", "meatboy", "bomberman"})
FAILED = ("DEAD", "STUCK")
ENTROPY_BINS = 10
ROUTE_BINS = 50        # side-scroller route profile: mean height in 50 slices across the level
ROUTE_POINTS = 64      # maze route profile: 64 points evenly spaced along the path
THRESHOLDS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "thresholds")


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    dims: tuple[int, ...]            # balance dimensions it belongs to (empty = supporting metric)
    fn: Callable[..., Any]           # (episodes, ctx) -> value | None; persona scope: (by_tier) -> value
    games: frozenset = ALL_GAMES
    fmt: str = "num"                 # pct | num | num2 | s | ratio | bits | count | dist | grid | tiers
    tip: str = ""
    scope: str = "level"             # level: one (level, persona); persona: across the skill tiers
    hidden: bool = False             # computed for charts, not shown as a stat


@dataclass
class Ctx:
    game: str
    route_threshold: float = 96.0
    cache: dict = field(default_factory=dict)


# ── helpers ──────────────────────────────────────────────────────────────────

def _won(eps):
    return [e for e in eps if e.get("status") == "WON"]


def _failed(eps):
    return [e for e in eps if e.get("status") in FAILED]


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return statistics.fmean(vals) if vals else None


def _has(eps, key):
    return [e for e in eps if e.get(key) is not None]


def _entropy_bits(counts) -> float:
    total = sum(counts)
    return abs(sum((n / total) * math.log2(n / total) for n in counts if n)) if total else 0.0


def _progress_hist(eps) -> list[int]:
    hist = [0] * ENTROPY_BINS
    for e in _failed(eps):
        p = e.get("progress")
        if p is not None:
            hist[min(int(p * ENTROPY_BINS), ENTROPY_BINS - 1)] += 1
    return hist


def _grid(points, eps):
    """Tile-binned counts of (x, y) pixel points: {"tile", "cols", "rows", "cells": [[tx, ty, n], ...]}."""
    ref = next((e for e in eps if e.get("tile")), None)
    if ref is None:
        return None
    ts = ref["tile"]
    cols = max(1, int(math.ceil((ref.get("level_w") or ts) / ts)))
    rows = max(1, int(math.ceil((ref.get("level_h") or ts) / ts)))
    c = Counter((int(x // ts), int(y // ts)) for x, y in points)
    return {"tile": ts, "cols": cols, "rows": rows, "cells": [[tx, ty, n] for (tx, ty), n in sorted(c.items())]}


# ── route clustering (strategy diversity) ────────────────────────────────────

def _profile_scroller(route, x_range):
    import numpy as np
    xs = np.array([p[0] for p in route], float)
    ys = np.array([p[1] for p in route], float)
    edges = np.linspace(x_range[0], x_range[1], ROUTE_BINS + 1)
    prof = np.full(ROUTE_BINS, np.nan)
    for i in range(ROUTE_BINS):
        m = (xs >= edges[i]) & (xs < edges[i + 1])
        if m.any():
            prof[i] = ys[m].mean()
    valid = ~np.isnan(prof)
    if valid.sum() < 2:
        return None
    return np.interp(np.arange(ROUTE_BINS), np.where(valid)[0], prof[valid])


def _profile_maze(route):
    """Arc-length resampling: two routes through a maze match when they pass the same places in
    the same order, whatever their speed."""
    import numpy as np
    traj = np.asarray(route, float)
    seg = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    dist = np.concatenate([[0.0], np.cumsum(seg)])
    if dist[-1] <= 0:
        return None
    t = np.linspace(0.0, dist[-1], ROUTE_POINTS)
    return np.stack([np.interp(t, dist, traj[:, 0]), np.interp(t, dist, traj[:, 1])], axis=1)


def _strategies(eps, ctx: Ctx) -> dict | None:
    """Greedy centroid clustering of winning routes; one result shared by the four route metrics."""
    if "strategies" in ctx.cache:
        return ctx.cache["strategies"]
    out = None
    won = [e for e in _won(eps) if isinstance(e.get("route"), list) and len(e["route"]) >= 2]
    if len(won) >= 2:
        import numpy as np
        if ctx.game == "meatboy":
            profs = [(e, _profile_maze(e["route"])) for e in won]
            dist = lambda a, b: float(np.mean(np.linalg.norm(a - b, axis=1)))  # noqa: E731
        else:
            xs = [p[0] for e in won for p in e["route"]]
            x_range = (min(xs), max(xs))
            profs = [(e, _profile_scroller(e["route"], x_range)) for e in won] if x_range[1] - x_range[0] >= 1 else []
            dist = lambda a, b: float(np.mean(np.abs(a - b)))  # noqa: E731
        profs = [(e, p) for e, p in profs if p is not None]
        if len(profs) >= 2:
            clusters, cents = [[0]], [profs[0][1].copy()]
            for i in range(1, len(profs)):
                d = [dist(profs[i][1], c) for c in cents]
                best = int(np.argmin(d))
                if d[best] <= ctx.route_threshold:
                    clusters[best].append(i)
                    cents[best] = np.mean([profs[j][1] for j in clusters[best]], axis=0)
                else:
                    clusters.append([i])
                    cents.append(profs[i][1].copy())
            sizes = [len(c) for c in clusters]
            times = [statistics.fmean(profs[j][0]["time_s"] for j in c) for c in clusters]
            out = {"count": len(clusters), "dominant": max(sizes) / len(profs),
                   "diversity": _entropy_bits(sizes),
                   "safe_fast": (max(times) / max(min(times), 0.01)) if len(times) >= 2 else None}
    ctx.cache["strategies"] = out
    return out


def _strat(key):
    def fn(eps, ctx):
        s = _strategies(eps, ctx)
        return None if s is None else s[key]
    return fn


# ── metric functions ─────────────────────────────────────────────────────────

def completion_rate(eps, ctx):
    return len(_won(eps)) / len(eps) if eps else None


def mean_completion_time(eps, ctx):
    return _mean(e.get("time_s") for e in _won(eps))


def completion_time_stddev(eps, ctx):
    t = [e["time_s"] for e in _won(eps) if e.get("time_s") is not None]
    return statistics.stdev(t) if len(t) > 1 else (0.0 if t else None)


def progress_at_death(eps, ctx):
    return _mean(e.get("progress") for e in _failed(eps))


def exit_found_ratio(eps, ctx):
    have = _has(eps, "exit_found")
    return statistics.fmean(e["exit_found"] for e in have) if have else None


def enemies_killed_ratio(eps, ctx):
    return _mean(e["kills"] / e["level_enemies"] for e in _failed(eps)
                 if e.get("level_enemies") and e.get("kills") is not None)


def deaths_per_run(eps, ctx):
    """Failed attempts per win (every episode is one life). No wins: every death counts."""
    return len(_failed(eps)) / max(len(_won(eps)), 1) if eps else None


def death_cause_distribution(eps, ctx):
    return dict(Counter(e.get("cause") or "?" for e in _failed(eps)).most_common())


def death_location_heatmap(eps, ctx):
    pts = [(e["end_x"], e["end_y"]) for e in _failed(eps) if e.get("end_x") is not None and e.get("end_y") is not None]
    return _grid(pts, eps) if pts else None


def death_progress_hist(eps, ctx):
    return _progress_hist(eps)


def death_cluster_entropy(eps, ctx):
    """Normalized Shannon entropy of death progress over 10 bins: 0 = one hotspot, 1 = uniform."""
    hist = _progress_hist(eps)
    return _entropy_bits(hist) / math.log2(ENTROPY_BINS) if sum(hist) else None


def powerup_loss_rate(eps, ctx):
    return _mean(e.get("powerups_lost") for e in eps)


def _coin_split(eps):
    have = [e for e in eps if e.get("coins") is not None and e.get("level_coins")]
    return [e for e in have if e["coins"] > 0], [e for e in have if e["coins"] == 0]


def coin_time_cost(eps, ctx):
    collectors, direct = _coin_split(eps)
    a, b = mean_completion_time(collectors, ctx), mean_completion_time(direct, ctx)
    return a / b if a is not None and b else None


def coin_death_premium(eps, ctx):
    collectors, direct = _coin_split(eps)
    if not collectors or not direct:
        return None
    return len(_failed(collectors)) / len(collectors) - len(_failed(direct)) / len(direct)


def coin_collection_rate(eps, ctx):
    return _mean(min(e["coins"] / e["level_coins"], 1.0) for e in eps
                 if e.get("level_coins") and e.get("coins") is not None)


def _no_bandages(eps, ctx):
    return None  # Meat Boy levels have no bandage pickups yet


def bomb_kill_rate(eps, ctx):
    bombs = sum(e.get("bombs_placed") or 0 for e in eps)
    return sum(e.get("kills") or 0 for e in eps) / bombs if bombs else None


def bombs_placed(eps, ctx):
    return _mean(e.get("bombs_placed") for e in eps)


def bomb_locations(eps, ctx):
    pts = [((tx + 0.5) * e["tile"], (ty + 0.5) * e["tile"]) for e in eps
           for tx, ty in (e.get("bomb_drops") or []) if e.get("tile")]
    return _grid(pts, eps) if pts else None


def time_to_first_kill(eps, ctx):
    return _mean(e.get("first_kill_s") for e in eps)


def blocks_before_exit(eps, ctx):
    return _mean(e.get("bricks_before_exit") for e in eps)


def wall_jump_rate(eps, ctx):
    return _mean(e.get("wall_jumps") for e in eps)


def chain_reaction_rate(eps, ctx):
    return _mean(e.get("chain_bombs") for e in eps)


def powerup_collection_rate(eps, ctx):
    return _mean(min(e["powerups"] / e["level_powerups"], 1.0) for e in eps
                 if e.get("level_powerups") and e.get("powerups") is not None)


def learning_gain(eps, ctx):
    """GA reading of learning: win rate of the last third of generations minus the first third."""
    if any(e.get("source") == "manual" for e in eps):
        return None
    gens = sorted({e.get("gen") or 0 for e in eps})
    if len(gens) < 2:
        return None
    third = max(1, len(gens) // 3)
    early = [e for e in eps if (e.get("gen") or 0) in set(gens[:third])]
    late = [e for e in eps if (e.get("gen") or 0) in set(gens[-third:])]
    return completion_rate(late, ctx) - completion_rate(early, ctx)


# cross-persona (skill tiers), from the per-persona level metrics
def novice_expert_gap(by_tier):
    n, x = by_tier.get("novice", {}).get("completion_rate"), by_tier.get("expert", {}).get("completion_rate")
    return x - n if n is not None and x is not None else None


def novice_expert_time_gap(by_tier):
    n = by_tier.get("novice", {}).get("mean_completion_time")
    x = by_tier.get("expert", {}).get("mean_completion_time")
    return n - x if n is not None and x is not None else None


def completion_rate_per_skill(by_tier):
    out = {t: m.get("completion_rate") for t, m in by_tier.items() if m.get("completion_rate") is not None}
    return {t: out[t] for t in TIERS if t in out} or None


# ── the registry ─────────────────────────────────────────────────────────────

_M, _MB, _BM = frozenset({"mario"}), frozenset({"meatboy"}), frozenset({"bomberman"})
_PROGRESS = SCROLLERS | _MB
_ROUTES = _M | _MB

METRICS: list[Metric] = [
    # 1 — challenge calibration
    Metric("completion_rate", "Completion", (1,), completion_rate, fmt="pct",
           tip="Share of attempts that reach the goal."),
    Metric("mean_completion_time", "Mean win time", (1,), mean_completion_time, fmt="s",
           tip="Average time of the runs that succeeded."),
    Metric("completion_time_stddev", "Win time spread", (1,), completion_time_stddev, fmt="s",
           tip="Standard deviation of the winning runs' times."),
    Metric("progress_at_death", "Progress at death", (1,), progress_at_death, games=_PROGRESS, fmt="pct",
           tip="How far failing attempts get (0% = start, 100% = goal). Side-scrollers: the death point "
               "between start and goal horizontally. Meat Boy: share of the BFS shortest path covered."),
    Metric("exit_found_ratio", "Exit found", (1,), exit_found_ratio, games=_BM, fmt="pct",
           tip="Runs that found the exit: a hidden exit bombed open, or a visible exit reached."),
    Metric("enemies_killed_ratio", "Enemies killed", (1,), enemies_killed_ratio, games=_BM, fmt="pct",
           tip="Enemies killed ÷ enemies in the level, averaged over the failed runs."),
    # 2 — punishment severity
    Metric("deaths_per_run", "Deaths per win", (2,), deaths_per_run, fmt="num2",
           tip="Failed attempts per win (every attempt is one life). With no wins, all deaths count."),
    Metric("death_cause_distribution", "Death causes", (2,), death_cause_distribution, fmt="dist",
           tip="What killed the agent, and how many times. Stall = no progress for too long."),
    Metric("death_location_heatmap", "Death map", (2,), death_location_heatmap, fmt="grid",
           tip="Where on the level the deaths happened, per tile."),
    Metric("death_progress_hist", "Deaths along the level", (2,), death_progress_hist, hidden=True, fmt="count",
           tip="Deaths in 10 bins from start to goal."),
    Metric("death_cluster_entropy", "Death spread", (2, 7), death_cluster_entropy, fmt="num2",
           tip="Normalized entropy of where deaths happen along the level (10 bins): "
               "0 = one learnable chokepoint, 1 = deaths spread evenly (feels random)."),
    Metric("powerup_loss_rate", "Power-ups lost", (2,), powerup_loss_rate, games=_M, fmt="num2",
           tip="Hits that cost a power-up (big / fire → smaller), per run."),
    # 3 — triangularity
    Metric("coin_time_cost", "Coin time cost", (3,), coin_time_cost, games=_M, fmt="ratio",
           tip="Mean win time of runs that collected coins ÷ mean win time of direct runs (0 coins)."),
    Metric("coin_death_premium", "Coin death premium", (3,), coin_death_premium, games=_M, fmt="pct",
           tip="Failure rate of runs that collected coins minus that of direct runs."),
    Metric("coin_collection_rate", "Coin rate", (3, 8), coin_collection_rate, games=_M, fmt="pct",
           tip="Coins collected ÷ coins in the level (N/A when the level has none)."),
    Metric("bandage_time_cost", "Bandage time cost", (3,), _no_bandages, games=_MB, fmt="ratio",
           tip="N/A: Meat Boy levels have no bandage pickups yet."),
    Metric("bandage_death_premium", "Bandage death premium", (3,), _no_bandages, games=_MB, fmt="pct",
           tip="N/A: Meat Boy levels have no bandage pickups yet."),
    Metric("bandage_collection_rate", "Bandage rate", (3, 8), _no_bandages, games=_MB, fmt="pct",
           tip="N/A: Meat Boy levels have no bandage pickups yet."),
    Metric("bomb_kill_rate", "Kills per bomb", (3,), bomb_kill_rate, games=_BM, fmt="num2",
           tip="Enemies killed per bomb placed."),
    # 4 — path / strategy diversity
    Metric("strategy_count", "Strategies", (4, 7), _strat("count"), games=_ROUTES, fmt="count",
           tip="Clearly different routes among the winning runs (needs 2+ wins)."),
    Metric("dominant_path_share", "Dominant path", (4,), _strat("dominant"), games=_ROUTES, fmt="pct",
           tip="Share of winning runs on the most common route. Near 100% = one true path."),
    Metric("path_diversity_index", "Path diversity", (4,), _strat("diversity"), games=_ROUTES, fmt="bits",
           tip="Shannon entropy (bits) of how winning runs spread across routes. 0 = one route."),
    Metric("safe_vs_fast_path_ratio", "Safe vs fast", (4,), _strat("safe_fast"), games=_ROUTES, fmt="ratio",
           tip="Mean time of the slowest route ÷ the fastest (needs 2+ routes)."),
    Metric("bombs_placed", "Bombs placed", (4,), bombs_placed, games=_BM, fmt="num",
           tip="Bombs placed per run."),
    Metric("bomb_locations", "Bomb map", (4,), bomb_locations, games=_BM, fmt="grid",
           tip="Where bombs were placed, per tile."),
    Metric("time_to_first_kill", "First kill", (4,), time_to_first_kill, games=_BM, fmt="s",
           tip="Time until the first enemy dies, over runs with a kill."),
    Metric("blocks_destroyed_before_finding_exit", "Blocks before exit", (4,), blocks_before_exit,
           games=_BM, fmt="num", tip="Bricks destroyed before the exit was found, over runs that found it."),
    # 5, 6 — across skill tiers
    Metric("novice_expert_gap", "Novice–expert gap", (5,), novice_expert_gap, fmt="pct", scope="persona",
           tip="Expert completion rate − novice completion rate (expert = experienced persona)."),
    Metric("novice_expert_time_gap", "Novice–expert time gap", (5,), novice_expert_time_gap, fmt="s",
           scope="persona", tip="Novice mean win time − expert (experienced) mean win time."),
    Metric("completion_rate_per_skill", "Completion per skill", (6,), completion_rate_per_skill, fmt="tiers",
           scope="persona", tip="Completion rate for each skill tier: novice, expert (experienced), speedrunner."),
    # 7 — emergent complexity (strategy_count and death_cluster_entropy are listed above)
    Metric("wall_jump_utilization_rate", "Wall jumps", (7,), wall_jump_rate, games=_MB, fmt="num",
           tip="Wall jumps per run."),
    Metric("chain_reaction_rate", "Chain reactions", (7,), chain_reaction_rate, games=_BM, fmt="num2",
           tip="Bombs set off by another bomb's blast, per run."),
    # 8 — reward density (coin / bandage rates are listed above)
    Metric("powerup_collection_rate", "Power-up rate", (8,), powerup_collection_rate, games=_BM, fmt="pct",
           tip="Power-ups collected ÷ power-ups in the level (N/A when the level has none)."),
    # supporting (no dimension): the GA's own learning signal
    Metric("learning_gain", "Learning gain", (), learning_gain, fmt="pct",
           tip="Win rate in the last third of generations minus the first third: how much the population learned."),
]
BY_KEY = {m.key: m for m in METRICS}

# which banded metrics make up each dimension's verdict pill
VERDICT_METRICS = {1: ["completion_rate", "mean_completion_time"],
                   2: ["deaths_per_run", "death_cluster_entropy"],
                   4: ["strategy_count", "dominant_path_share"]}


def metrics_for(game: str, scope: str = "level", include_hidden: bool = True) -> list[Metric]:
    return [m for m in METRICS if game in m.games and m.scope == scope and (include_hidden or not m.hidden)]


def _jsonable(v):
    if isinstance(v, float):
        return round(v, 4)
    if hasattr(v, "item"):  # numpy scalar
        return _jsonable(v.item())
    return v


def compute_level(episodes: list[dict], game: str, route_threshold: float | None = None) -> dict:
    """Every applicable level-scope metric for one (level, persona) set of episodes."""
    if route_threshold is None:
        route_threshold = float(load_thresholds(game).get("route_cluster_threshold", 96))
    ctx = Ctx(game=game, route_threshold=route_threshold)
    return {m.key: _jsonable(m.fn(episodes, ctx)) for m in metrics_for(game)}


def compute_cross(by_persona: dict[str, dict], game: str) -> dict:
    """Skill-tier metrics from {persona: level metrics}; tiers map to personas via TIERS."""
    by_tier = {t: by_persona[p] for t, p in TIERS.items() if p in by_persona}
    return {m.key: _jsonable(m.fn(by_tier)) for m in metrics_for(game, "persona")}


# ── thresholds + verdicts ────────────────────────────────────────────────────

def load_thresholds(game: str | None = None) -> dict:
    """default.yaml, overlaid by <game>.yaml: {"bands": {key: {target, warning}}, ...}."""
    import yaml
    out: dict = {"bands": {}}
    for name in ("default", game):
        if not name:
            continue
        path = os.path.join(THRESHOLDS_DIR, f"{name}.yaml")
        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except FileNotFoundError:
            continue
        bands = data.pop("bands", {}) or {}
        out.update(data)
        out["bands"].update(bands)
    return out


def in_band(value, band) -> bool | None:
    if value is None or not band:
        return None
    return abs(value - band["target"]) <= band["warning"]


def verdicts(values: dict, game: str, thresholds: dict | None = None) -> list[tuple[int, str]]:
    """[(dimension, "ok" | "partial" | "bad")] for each dimension with at least one banded value."""
    bands = (thresholds or load_thresholds(game))["bands"]
    out = []
    for dim, keys in VERDICT_METRICS.items():
        hits = [in_band(values.get(k), bands.get(k)) for k in keys if game in BY_KEY[k].games]
        hits = [h for h in hits if h is not None]
        if hits:
            out.append((dim, "ok" if all(hits) else ("partial" if any(hits) else "bad")))
    return out


# ── display ──────────────────────────────────────────────────────────────────

def fmt(key: str, v) -> str:
    """Human-readable value for a stat card or a terminal table."""
    if v is None:
        return "N/A"
    kind = BY_KEY[key].fmt if key in BY_KEY else "num"
    if kind == "pct":
        return f"{v:+.0%}" if key in ("novice_expert_gap", "coin_death_premium", "learning_gain") else f"{v:.0%}"
    if kind == "s":
        return f"{v:+.1f}s" if key == "novice_expert_time_gap" else f"{v:.1f}s"
    if kind == "ratio":
        return f"{v:.2f}×"
    if kind == "bits":
        return f"{v:.2f} bits"
    if kind == "count":
        return str(v) if not isinstance(v, list) else f"{sum(v)} deaths"
    if kind == "num2":
        return f"{v:.2f}"
    if kind == "dist":
        total = sum(v.values()) or 1
        return ", ".join(f"{k} {n / total:.0%}" for k, n in v.items()) or "none"
    if kind == "grid":
        return f"{sum(c[2] for c in v['cells'])} on {len(v['cells'])} tiles"
    if kind == "tiers":
        return " · ".join(f"{t} {r:.0%}" for t, r in v.items())
    return f"{v:.1f}" if isinstance(v, float) else str(v)


def format_table(by_level: dict[str, dict], game: str) -> str:
    """Terminal summary: one block per level, metrics grouped by dimension."""
    lines = []
    for level, vals in by_level.items():
        lines.append(f"\n  {game} · {level}  ({vals.get('_episodes', '?')} episodes)")
        for dim, name in DIMENSIONS.items():
            ms = [m for m in metrics_for(game, include_hidden=False) if m.dims and m.dims[0] == dim
                  and m.fmt not in ("grid",)]
            if not ms:
                continue
            cells = [f"{m.label} {fmt(m.key, vals.get(m.key))}" for m in ms]
            lines.append(f"    {dim} {name:<30} " + " · ".join(cells))
    return "\n".join(lines)
