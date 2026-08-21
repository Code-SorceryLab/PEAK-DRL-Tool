"""GA hyperparameter ablation: one knob at a time against literature-grounded bounds.

The baseline is `GAConfig()`; every axis in AXES varies exactly one field (the rest stay at
baseline) so each axis yields a clean "metric vs value" curve and the per-axis winners compose
into a recommended config per game (`best_config`, confirmed with `--confirm`).

Probes reuse balance.probe() and land under runs/gasweep/<game>/<persona>/<tag>/<level>_<seed>/
with tag = p<pop>g<gens>[_grid]_b<sig>_<axis>-<value>  (baseline: _base, composite: _best; b<sig>
hashes the baseline so a sweep around new defaults lands beside the old one, not over it), and are
aggregated into runs/balance/gasweep_<game>_<persona>_<tag>.json — a separate root and prefix so
the main Balance Command pages never see 23 extra configs.

    python -m code.neuro.gasweep --game mario --gens 40 --axes hidden memory --confirm
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import statistics
import time

from .balance import (BALANCE_DIR, config_tag, fmt_hms, probe_dir, run_jobs, write_report)
from .evolution import GAConfig
from .personas import PERSONAS, get_persona
from .sensors import SENSOR_MODES

GASWEEP_ROOT = os.path.join("runs", "gasweep")

# value lists: [low, baseline, high] (architecture axes: baseline first). Sources in docs/BALANCE.md.
AXES: dict[str, list] = {
    "hidden":          [8, 16, 32, 64],
    "pop_size":        [10, 30, 100],
    "elite":           [1, 4, 6],
    "tournament_k":    [2, 5, 8],
    "crossover_rate":  [0.0, 0.7, 0.95],
    "mutation_rate":   [0.005, 0.15, 0.5],
    "mutation_sigma":  [0.02, 0.15, 0.5],
    "anneal_factor":   [0.3, 0.5, 1.0],
    "init_sigma":      [0.25, 0.5, 1.0],
    "action_feedback": [False, True],
    "memory":          [0, 2, 3],
}

AXIS_DOC: dict[str, str] = {
    "hidden":          "tanh units — 8 / 32 / 64 vs 16; is a bigger brain faster to evolve?",
    "pop_size":        "De Jong 50–100, Grefenstette 30, Such et al. 1000 — 30 and 100 vs 10 (×3 / ×10 compute)",
    "elite":           "Such et al. keep exactly 1 elite; classic 1–10 % — 1 and 6 vs 4",
    "tournament_k":    "k=2 is the weakest pressure; Harik et al. k=4–8 converge faster — 2 and 8 vs 5",
    "crossover_rate":  "De Jong 0.6, Grefenstette 0.95, Such et al. none — 0 and 0.95 vs 0.7",
    "mutation_rate":   "binary-GA optimum ~1/L (≈0.003), real-valued NE mutates every gene — 0.005 and 0.5 vs 0.15",
    "mutation_sigma":  "Such et al. σ 0.002–0.005 on deep nets, tiny nets 0.1–0.5 — 0.02 and 0.5 vs 0.15",
    "anneal_factor":   "post-first-win mutation shrink; Such ≈0.45×, 1.0 = off — 0.3 and off vs 0.5 (win rate only; 0.5 beat 0.8 in 6/6 sweeps)",
    "init_sigma":      "Xavier gives σ≈0.27 for 14 inputs — 0.25 and 1.0 vs 0.5",
    "action_feedback": "previous (move, jump) fed back as 2 inputs — timing memory for wall-jumps",
    "memory":          "Jordan memory units: 2 or 3 extra outputs looped back as inputs next frame",
}


def _fmt(v) -> str:
    return str(int(v)) if isinstance(v, bool) else str(v)


def sweep_configs(axes: list[str] | None = None, base: GAConfig | None = None) -> list[tuple[str, dict]]:
    """[("base", {}), ("hidden-8", {"hidden": 8}), ...] — baseline values deduped, invalid
    combinations (elite >= pop, tournament > pop) dropped."""
    base = base or GAConfig()
    out: list[tuple[str, dict]] = [("base", {})]
    for axis in (axes if axes is not None else list(AXES)):
        for v in AXES[axis]:
            if v == getattr(base, axis):
                continue
            ov = {axis: v}
            pop = ov.get("pop_size", base.pop_size)
            if ov.get("elite", base.elite) >= pop or ov.get("tournament_k", base.tournament_k) > pop:
                continue
            out.append((f"{axis}-{_fmt(v)}", ov))
    return out


def base_sig(base: GAConfig | None = None) -> str:
    """5-hex fingerprint of the baseline's sweepable fields — part of every sweep tag."""
    base = base or GAConfig()
    key = json.dumps({a: getattr(base, a) for a in AXES}, sort_keys=True)
    return "b" + hashlib.sha1(key.encode()).hexdigest()[:5]


def sweep_tag(pop_size: int, gens: int, sensors: str, suffix: str, base: GAConfig | None = None) -> str:
    return f"{config_tag(pop_size, gens, sensors)}_{base_sig(base)}_{suffix}"


_SIG_RE = re.compile(r"^b[0-9a-f]{5}_")


def _tag_rest(tag: str) -> tuple[str | None, str] | None:
    """-> (baseline sig or None for pre-sig tags, '<axis>-<value>' | 'base' | 'best')."""
    parts = tag.split("_", 1)
    if len(parts) < 2:
        return None
    rest = parts[1]
    if rest.startswith("grid_"):
        rest = rest[len("grid_"):]
    m = _SIG_RE.match(rest)
    return (m.group(0)[:-1] if m else None), (rest[m.end():] if m else rest)


def tag_base_sig(tag: str) -> str | None:
    r = _tag_rest(tag)
    return r[0] if r else None


def parse_sweep_tag(tag: str) -> tuple[str, object] | None:
    """'p10g40_grid_b1a2c3_hidden-8' -> ('hidden', 8); '..._base' -> ('base', None); else None."""
    r = _tag_rest(tag)
    if r is None:
        return None
    rest = r[1]
    if rest in ("base", "best"):
        return rest, None
    axis, sep, val = rest.rpartition("-")
    if not sep or axis not in AXES:
        return None
    kind = type(AXES[axis][0])
    try:
        return axis, (bool(int(val)) if kind is bool else kind(val))
    except ValueError:
        return None


def cost_multiplier(configs: list[tuple[str, dict]], base: GAConfig | None = None) -> float:
    """Episodes relative to one baseline probe set (pop_size is the only knob that changes cost)."""
    base = base or GAConfig()
    return sum(ov.get("pop_size", base.pop_size) / base.pop_size for _, ov in configs)


# ── results ──────────────────────────────────────────────────────────────────

def sweep_point(report: dict) -> tuple[float, float]:
    """(mean win rate, mean first-win gen) over every cell; unsolved cells are censored at the
    gens budget so a never-winning config sits at the top instead of vanishing."""
    budget = report.get("gens_budget") or 0
    cells = [c for lst in (report.get("cells") or {}).values() for c in lst]
    if not cells:
        return 0.0, float(budget)
    wr = statistics.fmean(float(c.get("win_rate") or 0.0) for c in cells)
    fw = statistics.fmean(float(c.get("first_win_gen") or budget) for c in cells)
    return wr, fw


def paired_delta(variant: dict, base: dict) -> tuple[float, float, int, int, int]:
    """Variant vs baseline on the same (level, seed) cells: (mean Δ win rate, 95 % CI, cells up,
    cells down, n). Paired because every config shares seeds, so the per-cell difference
    cancels level/seed luck."""
    from .balance import mean_ci
    b = {(lvl, c["seed"]): c for lvl, lst in (base.get("cells") or {}).items() for c in lst}
    d = [float(c.get("win_rate") or 0) - float(b[(lvl, c["seed"])].get("win_rate") or 0)
         for lvl, lst in (variant.get("cells") or {}).items() for c in lst if (lvl, c["seed"]) in b]
    m, ci = mean_ci(d)
    return m, ci, sum(1 for x in d if x > 1e-9), sum(1 for x in d if x < -1e-9), len(d)


def best_config(reports: list[dict]) -> dict:
    """Per axis, the value with the highest win rate (ties: earliest first win), as GAConfig
    overrides vs the baseline. Empty dict = the baseline already wins every axis.
    OFAT composite — it ignores interactions; run it with --confirm to check."""
    base_pt = None
    by_axis: dict[str, list[tuple[object, tuple[float, float]]]] = {}
    for r in reports:
        parsed = parse_sweep_tag(r.get("tag") or "")
        if not parsed:
            continue
        axis, val = parsed
        if axis == "base":
            base_pt = sweep_point(r)
        elif axis != "best":
            by_axis.setdefault(axis, []).append((val, sweep_point(r)))
    out: dict = {}
    for axis, pts in by_axis.items():
        cands = ([(None, base_pt)] if base_pt else []) + pts   # base first: it wins ties
        val, _ = max(cands, key=lambda p: (round(p[1][0], 6), -p[1][1]))
        if val is not None:
            out[axis] = val
    return out


def load_reports(out_dir: str = BALANCE_DIR, game: str | None = None,
                 persona: str | None = None) -> list[dict]:
    reports = []
    for path in sorted(glob.glob(os.path.join(out_dir, "gasweep_*.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if (game is None or data.get("game") == game) and (persona is None or data.get("persona") == persona):
            reports.append(data)
    return reports


def rebuild(out_dir: str = BALANCE_DIR) -> list[str]:
    """Regenerate every gasweep_*.json from the probe dirs under runs/gasweep/."""
    written = []
    for d in sorted(glob.glob(os.path.join(GASWEEP_ROOT, "*", "*", "p*g*"))):
        tag = os.path.basename(d)
        persona = os.path.basename(os.path.dirname(d))
        game = os.path.basename(os.path.dirname(os.path.dirname(d)))
        if parse_sweep_tag(tag) is None:
            continue
        path = write_report(game, persona, tag, out_dir, root=GASWEEP_ROOT, prefix="gasweep")
        if path:
            written.append(path)
    return written


def _order(report: dict) -> tuple:
    axis, val = parse_sweep_tag(report["tag"]) or ("", None)
    rank = {"base": -1, "best": 99}.get(axis, list(AXES).index(axis) if axis in AXES else 98)
    return rank, (val if isinstance(val, (int, float)) else 0)


def format_sweep(reports: list[dict]) -> str:
    from .net import make_net
    rows = [f"{'config':<24}{'params':>7}  {'solved':>7}  {'first win':>9}  {'win rate':>8}  {'time':>8}"]
    for r in sorted(reports, key=_order):
        wr, fw = sweep_point(r)
        ga = r.get("ga_config")
        n_params = make_net(GAConfig(**ga)).n_params if ga else 0
        levels = r.get("levels") or []
        solved = sum(1 for lv in levels if lv.get("solved_by"))
        name = (_tag_rest(r["tag"]) or (None, r["tag"]))[1]
        rows.append(f"{name:<24}{n_params:>7}  {solved:>3}/{len(levels):<3}  {fw:>9.1f}  {wr:>7.0%}  "
                    f"{fmt_hms(float(r.get('train_time_s') or 0)):>8}")
    ov = best_config(reports)
    rec = " ".join(f"{k}={_fmt(v)}" for k, v in ov.items()) or "baseline (no axis beat it)"
    rows.append(f"\nrecommended config: {rec}")
    return "\n".join(rows)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="GA hyperparameter ablation sweep (one knob at a time)")
    ap.add_argument("--game", default="mario")
    ap.add_argument("--persona", default="experienced", choices=sorted(PERSONAS))
    ap.add_argument("--gens", type=int, default=40, help="generation budget per probe")
    ap.add_argument("--seeds", nargs="*", type=int, default=[1234, 2025, 31337])
    ap.add_argument("--levels", nargs="*", default=None, help="default: every enabled level")
    ap.add_argument("--workers", type=int, default=None,
                    help="probe processes; default = min(jobs, cores-1), 1 = sequential")
    ap.add_argument("--sensors", default="rays", choices=SENSOR_MODES)
    ap.add_argument("--axes", nargs="*", default=None, choices=list(AXES),
                    help="axes to sweep (default: all). `--axes` alone = none (use with --confirm)")
    ap.add_argument("--out", default=BALANCE_DIR)
    ap.add_argument("--confirm", action="store_true",
                    help="after the sweep, probe the per-axis-winner composite once as '<tag>_best'")
    ap.add_argument("--rebuild", action="store_true",
                    help="regenerate gasweep JSONs from runs/gasweep and exit (no training)")
    args = ap.parse_args()

    if args.rebuild:
        for path in rebuild(args.out):
            print(f"rebuilt {path}")
        return

    from .adapters import list_levels, validate_level
    persona = get_persona(args.persona)
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    pygame.init()

    levels = args.levels or list_levels(args.game)
    if not levels:
        raise SystemExit(f"no levels found for game '{args.game}'")
    for lvl in args.levels or []:
        validate_level(args.game, lvl)

    def run_configs(configs: list[tuple[str, dict]], what: str) -> None:
        jobs = []
        for suffix, ov in configs:
            cfg = GAConfig(sensors=args.sensors, **ov)
            tag = sweep_tag(cfg.pop_size, args.gens, args.sensors, suffix)   # sig = current GAConfig()
            jobs += [(args.game, lvl, seed, args.gens, persona, args.sensors, ov,
                      probe_dir(args.game, persona.name, tag, lvl, seed, root=GASWEEP_ROOT))
                     for lvl in levels for seed in args.seeds]
        jobs.sort(key=lambda j: -j[6].get("pop_size", 0))   # heavy configs first: better pool tail
        workers = args.workers or min(len(jobs), max(1, (os.cpu_count() or 2) - 1))
        print(f"GA sweep [{what}]: {len(configs)} configs x {len(levels)} levels x {len(args.seeds)} seeds "
              f"= {len(jobs)} probes, budget {args.gens} gens each, ≈{cost_multiplier(configs):.0f}x one "
              f"balance sweep, {workers} worker(s)  [{args.game} · {persona.name} · {args.sensors}]\n", flush=True)
        t0 = time.time()
        run_jobs(jobs, workers, label=lambda j: f"{os.path.basename(os.path.dirname(j[7]))} · {j[1]} seed {j[2]}")
        rebuild(args.out)
        print(f"\n{fmt_hms(time.time() - t0)} wall", flush=True)

    def this_sweep() -> list[dict]:
        return [r for r in load_reports(args.out, args.game, persona.name)
                if r.get("gens_budget") == args.gens
                and (r.get("ga_config") or {}).get("sensors", "rays") == args.sensors
                and tag_base_sig(r.get("tag") or "") == base_sig()]   # only this baseline's sweep

    configs = sweep_configs(args.axes) if args.axes is None or args.axes else []
    if configs:
        run_configs(configs, "ablation")
        print("\n" + format_sweep(this_sweep()), flush=True)

    if args.confirm:
        ov = best_config(this_sweep())
        if not ov:
            print("\nconfirm: the baseline already wins every axis — nothing to run", flush=True)
        else:
            print(f"\nconfirm: probing the composite {ov}", flush=True)
            run_configs([("best", ov)], "confirm")
            print("\n" + format_sweep(this_sweep()), flush=True)


if __name__ == "__main__":
    main()
