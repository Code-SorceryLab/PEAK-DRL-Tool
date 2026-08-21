"""Level-balance report: multi-seed neuroevolution probes per level.

The GA successor to the paper matrix (run_paper_matrix / case_study, archived) and the
stats subsystem's failure-mode taxonomy: for each (level x seed), evolve a fresh
population and measure how learnable the level is. Difficulty is read the same way the
paper read it — win rate, generations-to-first-win, and the death-cause mix that names
WHY a level is hard (Enemy pressure vs Pit gaps vs Spike density vs Stall dead-ends).

Run:  python -m code.neuro.balance --game mario [--levels L1 L2 ...]
      [--seeds 1234 2025 31337] [--gens 25] [--out runs/balance]
      python -m code.neuro.balance --rebuild      # regenerate report JSONs from runs/probes

      python -m code.neuro.balance --game mario --sensors grid   # tile-grid sensor ablation
      python -m code.neuro.balance --game mario --compare         # rays vs grid, side by side

Layout: runs/probes/<game>/<persona>/<tag>/<level>_<seed>/   tag = p<pop_size>g<gens>[_<sensors>]
        runs/balance/report_<game>_<persona>_<tag>.json
Different configs (persona, population, generation budget, sensors) never overwrite each
other; re-running the same config on a level replaces that level's probes.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import multiprocessing as mp
import os
import re
import statistics
import time

from .evolution import GAConfig
from .personas import PERSONAS, Persona, get_persona
from .sensors import SENSOR_MODES

PROBES_ROOT = os.path.join("runs", "probes")
BALANCE_DIR = os.path.join("runs", "balance")

# two-sided 95% t critical values for small n (paper used the same approach)
T_CRIT = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365}
WIN_WINDOW = 10  # generations after the first win used to measure a stable win rate


def mean_ci(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    m = statistics.fmean(vals)
    if len(vals) < 2:
        return m, 0.0
    sd = statistics.stdev(vals)
    return m, T_CRIT.get(len(vals), 1.96) * sd / math.sqrt(len(vals))


def config_tag(pop_size: int, gens: int, sensors: str = "rays") -> str:
    return f"p{pop_size}g{gens}" + ("" if sensors == "rays" else f"_{sensors}")


def probe_dir(game: str, persona: str, tag: str, level: str, seed: int,
              root: str = PROBES_ROOT) -> str:
    return os.path.join(root, game, persona, tag, f"{level}_{seed}".replace(" ", "_"))


def probe(game: str, level: str, seed: int, gens: int, persona: Persona | None = None,
          sensors: str = "rays", overrides: dict | None = None, run_dir: str | None = None) -> dict:
    """One (level, seed) cell: evolve a fresh population, stop WIN_WINDOW gens after
    the first win (or at the gen budget), and summarize what happened.
    `overrides` are extra GAConfig fields (the GA sweep varies one at a time)."""
    from .trainer import Trainer  # lazy: pulls in pygame
    persona = persona or PERSONAS["experienced"]
    cfg = GAConfig(seed=seed, sensors=sensors, **(overrides or {}))
    run_dir = run_dir or probe_dir(game, persona.name, config_tag(cfg.pop_size, gens, sensors), level, seed)
    trainer = Trainer(game, level, cfg, run_dir=run_dir, persona=persona)
    pop = trainer.pop

    first_win: int | None = None
    while pop.generation < gens:
        trainer.run(max_gens=pop.generation + 1, verbose=False)
        row = pop.history[-1]
        if first_win is None and row["wins"] > 0:
            first_win = row["gen"]
        if first_win is not None and pop.generation >= first_win + WIN_WINDOW:
            break
    return summarize(pop.history, cfg.pop_size, level, seed, pop.best_fitness, pop.best_gen)


def _improvement_rate(hist: list[dict], level_len: float) -> float | None:
    """Slope of best progress per generation (fraction of the level / gen), measured
    from gen 1 up to the generation the peak was first reached (post-win plateau excluded)."""
    xs = [r.get("best_x") or 0.0 for r in hist]
    if len(xs) < 2:
        return None
    cut = xs.index(max(xs)) + 1
    if cut < 2:
        return 0.0
    gens = list(range(1, cut + 1))
    return statistics.linear_regression(gens, xs[:cut]).slope / level_len


def summarize(hist: list[dict], pop_size: int, level: str, seed: int,
              best_fitness: float, best_gen: int) -> dict:
    """Fold one probe's per-generation history into a cell (pure; also used by --rebuild)."""
    first_win = next((r["gen"] for r in hist if r["wins"] > 0), None)
    tail = hist[-WIN_WINDOW:]
    episodes = len(tail) * pop_size
    causes: dict[str, int] = {}
    stuck = 0

    # Amr's balance-metric table: completion, punishment severity, triangularity
    all_eps = [e for r in hist for e in r["envs"]]
    total_eps = len(all_eps)
    won_eps = [e for e in all_eps if e["status"] == "WON"]
    dead_eps = [e for e in all_eps if e["status"] in ("DEAD", "STUCK")]
    level_len = max((e.get("level_len") or 0.0) for e in all_eps) or 1.0
    win_times = [e["frames"] / 60.0 for e in won_eps]
    death_hist = [0] * 10  # death-location heatmap, 10 bins across the level
    progress_at_death = []
    for e in dead_eps:
        p = min(max((e.get("end_x") or 0.0) / level_len, 0.0), 1.0)
        progress_at_death.append(p)
        death_hist[min(int(p * 10), 9)] += 1
    total_deaths = sum(death_hist)
    entropy = 0.0
    if total_deaths:
        for n in death_hist:
            if n:
                q = n / total_deaths
                entropy -= q * math.log(q)
        entropy /= math.log(len(death_hist))  # normalized 0 (one hotspot) .. 1 (uniform)
    level_coins = max((e.get("level_coins") or 0) for e in all_eps)
    coin_rate = (statistics.fmean(min(e["coins"] / level_coins, 1.0) for e in all_eps)
                 if level_coins else 1.0)  # "automatic 1 if there are 0 bandages"
    third = max(1, len(hist) // 3)
    novice_wr = sum(r["wins"] for r in hist[:third]) / (third * pop_size)
    expert_wr = sum(r["wins"] for r in hist[-third:]) / (third * pop_size)

    # Failure causes: a stall (STUCK) is a cause like any death — "Stall dead-ends" name
    # why a level is hard just as much as enemies or pits do.
    for r in hist:
        stuck += r["stuck"]
        for e in r["envs"]:
            if e["status"] == "STUCK":
                causes["Stall"] = causes.get("Stall", 0) + 1
            elif e["status"] == "DEAD":
                c = e.get("cause") or "?"
                causes[c] = causes.get(c, 0) + 1

    early = statistics.fmean(r["avg"] for r in hist[:third])
    late = statistics.fmean(r["avg"] for r in hist[-third:])
    trend = "IMPROVING" if late > early * 1.15 else ("DECLINING" if late < early * 0.85 else "FLAT")
    rate = _improvement_rate(hist, level_len)

    return {
        "level": level,
        "seed": seed,
        "pop_size": pop_size,
        "gens_run": len(hist),
        "first_win_gen": first_win,
        "best_gen": best_gen or None,
        "win_rate": sum(r["wins"] for r in tail) / episodes if episodes else 0.0,
        "best": round(best_fitness, 1),
        "best_x": max(r["best_x"] for r in hist),
        "stuck_frac": stuck / (len(hist) * pop_size),
        "causes": causes,
        "trend": trend,
        "improvement_rate": round(rate, 4) if rate is not None else None,
        "train_time_s": round(sum(r.get("duration") or 0.0 for r in hist), 1),
        # Amr's table
        "completion_rate": round(len(won_eps) / total_eps, 3) if total_eps else 0.0,
        "mean_completion_time": round(statistics.fmean(win_times), 1) if win_times else None,
        "completion_time_stddev": round(statistics.stdev(win_times), 1) if len(win_times) > 1 else 0.0,
        "progress_at_death": round(statistics.fmean(progress_at_death), 3) if progress_at_death else None,
        "deaths_per_run": round(len(dead_eps) / max(len(hist), 1), 2),
        "death_hist": death_hist,
        "death_cluster_entropy": round(entropy, 3),
        "coin_collection_rate": round(coin_rate, 3),
        "novice_expert_gap": round(expert_wr - novice_wr, 3),
        "curve": [[round(r["best"], 1), round(r["avg"], 1)] for r in hist],
    }


def _pool_init() -> None:
    """Each worker process gets its own headless pygame."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    pygame.init()


def _probe_job(job: tuple) -> tuple[tuple, dict]:
    """job = the positional args of probe(); returns (job, cell) so callers can label output."""
    return job, probe(*job)


def run_jobs(jobs: list[tuple], workers: int, label=None) -> None:
    """Run probe jobs sequentially (workers <= 1) or in a headless process pool, printing one
    line per finished cell. `label(job)` names the cell (default: "<level> seed <seed>")."""
    label = label or (lambda j: f"{j[1]} seed {j[2]}")

    def _record(i: int, job: tuple, cell: dict) -> None:
        bg = f"best gen {cell['best_gen']}" if cell["best_gen"] else "no progress"
        print(f"[{i}/{len(jobs)}] {label(job)}: {bg}, win rate {cell['win_rate']:.0%}, "
              f"best_x {cell['best_x']}, {fmt_hms(cell['train_time_s'])}", flush=True)

    if workers <= 1:
        for i, job in enumerate(jobs, 1):
            _record(i, *_probe_job(job))
    else:
        with mp.Pool(workers, initializer=_pool_init) as pool:
            for i, (job, cell) in enumerate(pool.imap_unordered(_probe_job, jobs), 1):
                _record(i, job, cell)


def aggregate(cells: list[dict]) -> dict:
    """Fold one level's per-seed cells into a paper-style row (mean +- 95% CI)."""
    n = len(cells)
    solved = [c for c in cells if c["first_win_gen"] is not None]
    wr_m, wr_ci = mean_ci([c["win_rate"] for c in cells])
    fw_m, fw_ci = mean_ci([float(c["first_win_gen"]) for c in solved])
    bg = [float(c["best_gen"]) for c in cells if c.get("best_gen")]
    bg_m, bg_ci = mean_ci(bg)
    causes: dict[str, int] = {}
    for c in cells:
        for k, v in c["causes"].items():
            causes[k] = causes.get(k, 0) + v
    total_deaths = sum(causes.values()) or 1
    dom = max(causes.items(), key=lambda kv: kv[1]) if causes else ("-", 0)
    death_hist = [sum(c["death_hist"][i] for c in cells) for i in range(10)]
    comp_times = [c["mean_completion_time"] for c in cells if c["mean_completion_time"] is not None]
    comp_sd = [c["completion_time_stddev"] for c in cells if c["mean_completion_time"] is not None]
    pad = [c["progress_at_death"] for c in cells if c["progress_at_death"] is not None]
    rates = [c["improvement_rate"] for c in cells if c.get("improvement_rate") is not None]
    return {
        "level": cells[0]["level"],
        "seeds": n,
        "pop_size": cells[0].get("pop_size"),
        "solved_by": len(solved),
        "first_win_mean": round(fw_m, 1) if solved else None,
        "first_win_ci": round(fw_ci, 1) if solved else None,
        "best_gen_mean": round(bg_m, 1) if bg else None,
        "best_gen_ci": round(bg_ci, 1) if bg else None,
        "win_rate_mean": round(wr_m, 3),
        "win_rate_ci": round(wr_ci, 3),
        "dominant_cause": dom[0],
        "dominant_cause_frac": round(dom[1] / total_deaths, 2),
        "causes": causes,
        "best_x_mean": round(statistics.fmean(c["best_x"] for c in cells), 1),
        "stuck_frac_mean": round(statistics.fmean(c["stuck_frac"] for c in cells), 3),
        "trends": [c["trend"] for c in cells],
        "improvement_rate_mean": round(statistics.fmean(rates), 4) if rates else None,
        "train_time_s": round(sum(c.get("train_time_s") or 0.0 for c in cells), 1),
        # Amr's table, aggregated
        "completion_rate_mean": round(statistics.fmean(c["completion_rate"] for c in cells), 3),
        "mean_completion_time": round(statistics.fmean(comp_times), 1) if comp_times else None,
        "completion_time_stddev": round(statistics.fmean(comp_sd), 1) if comp_sd else None,
        "progress_at_death_mean": round(statistics.fmean(pad), 3) if pad else None,
        "deaths_per_run_mean": round(statistics.fmean(c["deaths_per_run"] for c in cells), 2),
        "death_hist": death_hist,
        "death_cluster_entropy_mean": round(statistics.fmean(c["death_cluster_entropy"] for c in cells), 3),
        "coin_collection_rate_mean": round(statistics.fmean(c["coin_collection_rate"] for c in cells), 3),
        "novice_expert_gap_mean": round(statistics.fmean(c["novice_expert_gap"] for c in cells), 3),
    }


def difficulty_rank(rows: list[dict]) -> list[dict]:
    """Hardest first: unsolved levels top, then by how long the GA needed to crack them."""
    return sorted(rows, key=lambda r: (r["solved_by"], -(r["first_win_mean"] or 9999)))


def format_report(rows: list[dict], game: str) -> str:
    out = [f"LEVEL BALANCE REPORT - {game}  (hardest first; win rate measured over the "
           f"{WIN_WINDOW} generations after the first win)"]
    hdr = (f"{'LEVEL':<14}{'SOLVED':>7}{'FIRST WIN':>14}{'BEST GEN':>13}{'WIN RATE':>16}"
           f"{'DOMINANT CAUSE':>17}{'BEST X':>9}{'STUCK':>7}{'RATE':>9}{'TIME':>9}  TREND")
    out.append(hdr)
    out.append("-" * len(hdr))
    for r in difficulty_rank(rows):
        fw = (f"gen {r['first_win_mean']}±{r['first_win_ci']}" if r["first_win_mean"] is not None
              else "never")
        bg = (f"gen {r['best_gen_mean']}±{r['best_gen_ci']}" if r.get("best_gen_mean") is not None
              else "-")
        wr = f"{r['win_rate_mean']:.0%} ±{r['win_rate_ci']:.0%}"
        cause = f"{r['dominant_cause']} ({r['dominant_cause_frac']:.0%})"
        rate = (f"{r['improvement_rate_mean']:+.1%}/g" if r.get("improvement_rate_mean") is not None
                else "-")
        out.append(f"{r['level']:<14}{r['solved_by']}/{r['seeds']:<5}{fw:>14}{bg:>13}{wr:>16}"
                   f"{cause:>17}{r['best_x_mean']:>9}{r['stuck_frac_mean']:>7.0%}{rate:>9}"
                   f"{fmt_hms(r.get('train_time_s') or 0):>9}  {'/'.join(r['trends'])}")
    return "\n".join(out)


def fmt_hms(sec: float) -> str:
    s = int(sec)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


# ── rebuild from probes ──────────────────────────────────────────────────────

def _natural(name: str):  # "2" < "10", "MM-Stage2" < "MM-Stage10"
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def _parse_tag(tag: str) -> tuple[int, int] | None:
    try:
        p, g = tag.split("_")[0][1:].split("g")
        return int(p), int(g)
    except ValueError:
        return None


def format_compare(by_mode: dict[str, list[dict]], game: str) -> str:
    """Side-by-side sensor ablation: one row per level, one column group per sensor mode."""
    modes = list(by_mode)
    out = [f"SENSOR ABLATION - {game}  (first win = generations to first win ± 95% CI; "
           f"win rate over the {WIN_WINDOW} gens after it)"]
    hdr = f"{'LEVEL':<14}" + "".join(f"{m.upper() + ' FIRST WIN':>20}{m.upper() + ' WIN RATE':>18}" for m in modes)
    out += [hdr, "-" * len(hdr)]
    levels = sorted({r["level"] for rows in by_mode.values() for r in rows}, key=_natural)
    for lvl in levels:
        line = f"{lvl:<14}"
        for m in modes:
            r = next((r for r in by_mode[m] if r["level"] == lvl), None)
            if r is None:
                line += f"{'-':>20}{'-':>18}"
                continue
            fw = (f"gen {r['first_win_mean']}±{r['first_win_ci']}" if r["first_win_mean"] is not None
                  else "never")
            line += f"{fw:>20}{r['win_rate_mean']:>10.0%} ±{r['win_rate_ci']:<6.0%}"
        out.append(line)
    return "\n".join(out)


def compare(game: str, persona: str, gens: int, out_dir: str = BALANCE_DIR) -> str:
    by_mode: dict[str, list[dict]] = {}
    for mode in SENSOR_MODES:
        path = os.path.join(out_dir, f"report_{game}_{persona}_{config_tag(GAConfig().pop_size, gens, mode)}.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                by_mode[mode] = json.load(f)["levels"]
    missing = [m for m in SENSOR_MODES if m not in by_mode]
    if missing:
        return (f"no {'/'.join(missing)} report for {game}/{persona} at {gens} gens — run "
                + " and ".join(f"--sensors {m}" for m in missing) + " first")
    return format_compare(by_mode, game)


def load_probe_cells(game: str, persona: str, tag: str,
                     root: str = PROBES_ROOT) -> dict[str, list[dict]]:
    """Re-summarize every probe dir under one (game, persona, tag) from its state.json."""
    cells: dict[str, list[dict]] = {}
    for sp in sorted(glob.glob(os.path.join(root, game, persona, tag, "*", "state.json"))):
        try:
            with open(sp, encoding="utf-8") as f:
                st = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        hist = st.get("history") or []
        if not hist:
            continue
        cfg = st.get("config", {})
        cell = summarize(hist, int(cfg.get("pop_size", 10)), str(hist[-1].get("level")),
                         int(cfg.get("seed", 0)), float(st.get("best_fitness", 0.0)),
                         int(st.get("best_gen", 0)))
        cell["dir"] = os.path.dirname(sp)
        # The trainer anneals mutation in place after a level's first win and saves that;
        # report the values the sweep was configured with.
        cfg = dict(cfg)
        f = cfg.get("anneal_factor", 1.0)
        if st.get("annealed") and f not in (0.0, 1.0):
            cfg["mutation_rate"] = round(cfg["mutation_rate"] / f, 6)
            cfg["mutation_sigma"] = round(cfg["mutation_sigma"] / f, 6)
        cell["_config"] = cfg
        cells.setdefault(cell["level"], []).append(cell)
    for lst in cells.values():
        lst.sort(key=lambda c: c["seed"])
    return dict(sorted(cells.items(), key=lambda kv: _natural(kv[0])))


def write_report(game: str, persona: str, tag: str, out_dir: str = BALANCE_DIR,
                 root: str = PROBES_ROOT, prefix: str = "report") -> str | None:
    """Aggregate one (game, persona, tag) probe set into <out_dir>/<prefix>_*.json."""
    cells = load_probe_cells(game, persona, tag, root)
    if not cells:
        return None
    parsed = _parse_tag(tag) or (next(iter(cells.values()))[0]["pop_size"], 0)
    seeds = sorted({c["seed"] for lst in cells.values() for c in lst})
    ga_config = next(iter(cells.values()))[0].get("_config")  # same GA settings for every cell
    for lst in cells.values():
        for c in lst:
            c.pop("_config", None)
    rows = [aggregate(c) for c in cells.values()]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{prefix}_{game}_{persona}_{tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"game": game, "persona": persona, "tag": tag, "pop_size": parsed[0],
                   "gens_budget": parsed[1], "seeds": seeds,
                   "train_time_s": round(sum(r["train_time_s"] for r in rows), 1),
                   "ga_config": ga_config,
                   "levels": rows, "cells": cells},
                  f, indent=2, default=float)  # numpy scalars leak from some cores
    return out_path


def rebuild(out_dir: str = BALANCE_DIR) -> list[str]:
    """Regenerate every report JSON from the probe dirs on disk (new layout only)."""
    written = []
    for d in sorted(glob.glob(os.path.join(PROBES_ROOT, "*", "*", "p*g*"))):
        tag = os.path.basename(d)
        persona = os.path.basename(os.path.dirname(d))
        game = os.path.basename(os.path.dirname(os.path.dirname(d)))
        if _parse_tag(tag) is None:
            continue
        path = write_report(game, persona, tag, out_dir)
        if path:
            written.append(path)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-seed level-balance report")
    ap.add_argument("--game", default="mario")
    ap.add_argument("--levels", nargs="*", default=None, help="default: every enabled level")
    ap.add_argument("--seeds", nargs="*", type=int, default=[1234, 2025, 31337])
    ap.add_argument("--gens", type=int, default=25, help="generation budget per probe")
    ap.add_argument("--persona", default="experienced", choices=sorted(PERSONAS),
                    help="player type the probe agents imitate")
    ap.add_argument("--out", default=BALANCE_DIR)
    ap.add_argument("--workers", type=int, default=None,
                    help="probe processes; default = min(jobs, cores-1), 1 = sequential")
    ap.add_argument("--sensors", default="rays", choices=SENSOR_MODES,
                    help="agent exteroception: rays (default) or the 3x11x11 tile grid")
    ap.add_argument("--compare", action="store_true",
                    help="print rays-vs-grid table from existing reports and exit (no training)")
    ap.add_argument("--rebuild", action="store_true",
                    help="regenerate report JSONs from runs/probes and exit (no training)")
    args = ap.parse_args()

    if args.rebuild:
        for path in rebuild(args.out):
            print(f"rebuilt {path}")
        return
    if args.compare:
        print(compare(args.game, args.persona, args.gens, args.out))
        return

    from .adapters import list_levels, validate_level
    persona = get_persona(args.persona)

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    pygame.init()

    levels = args.levels or list_levels(args.game)
    if not levels:
        raise SystemExit(f"no levels found for game '{args.game}'")
    if args.levels:
        for lvl in args.levels:
            validate_level(args.game, lvl)
    tag = config_tag(GAConfig().pop_size, args.gens, args.sensors)
    jobs = [(args.game, lvl, seed, args.gens, persona, args.sensors) for lvl in levels for seed in args.seeds]
    workers = args.workers or min(len(jobs), max(1, (os.cpu_count() or 2) - 1))
    print(f"balance probe: {len(levels)} levels x {len(args.seeds)} seeds = {len(jobs)} jobs, "
          f"budget {args.gens} gens each, {workers} worker(s)  [{persona.name} · {tag}]\n", flush=True)

    t0 = time.time()
    run_jobs(jobs, workers)

    # The probe dirs are the source of truth: re-aggregate everything under this config,
    # so earlier levels probed with the same config stay and re-probed levels are replaced.
    out_path = write_report(args.game, persona.name, tag, args.out)
    with open(out_path, encoding="utf-8") as f:
        rows = json.load(f)["levels"]
    print("\n" + format_report([r for r in rows if r["level"] in levels], args.game), flush=True)
    print(f"\nsaved {out_path}  ({fmt_hms(time.time() - t0)} wall)", flush=True)


if __name__ == "__main__":
    main()
