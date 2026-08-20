"""Level-balance report: multi-seed neuroevolution probes per level.

The GA successor to the paper matrix (run_paper_matrix / case_study, archived) and the
stats subsystem's failure-mode taxonomy: for each (level x seed), evolve a fresh
population and measure how learnable the level is. Difficulty is read the same way the
paper read it — win rate, generations-to-first-win, and the death-cause mix that names
WHY a level is hard (Enemy pressure vs Pit gaps vs Spike density vs Stall dead-ends).

Run:  python -m code.neuro.balance --game mario [--levels L1 L2 ...]
      [--seeds 1234 2025 31337] [--gens 25] [--out runs/balance]
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import statistics
import time

from .adapters import list_levels
from .evolution import GAConfig, Population
from .net import NeuralNet
from .personas import PERSONAS, Persona, get_persona
from .trainer import Trainer

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


def probe(game: str, level: str, seed: int, gens: int, run_root: str,
          persona: Persona | None = None) -> dict:
    """One (level, seed) cell: evolve a fresh population, stop WIN_WINDOW gens after
    the first win (or at the gen budget), and summarize what happened."""
    cfg = GAConfig(seed=seed)
    run_dir = os.path.join(run_root, f"{level}_{seed}".replace(" ", "_"))
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

    hist = pop.history
    tail = hist[-WIN_WINDOW:]
    episodes = len(tail) * cfg.pop_size
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
    novice_wr = sum(r["wins"] for r in hist[:third]) / (third * cfg.pop_size)
    expert_wr = sum(r["wins"] for r in hist[-third:]) / (third * cfg.pop_size)

    for r in hist:
        stuck += r["stuck"]
        for e in r["envs"]:
            if e["status"] == "DEAD":
                c = e.get("cause") or "?"
                causes[c] = causes.get(c, 0) + 1

    early = statistics.fmean(r["avg"] for r in hist[:third])
    late = statistics.fmean(r["avg"] for r in hist[-third:])
    trend = "IMPROVING" if late > early * 1.15 else ("DECLINING" if late < early * 0.85 else "FLAT")

    return {
        "level": level,
        "seed": seed,
        "gens_run": pop.generation,
        "first_win_gen": first_win,
        "win_rate": sum(r["wins"] for r in tail) / episodes if episodes else 0.0,
        "best": round(pop.best_fitness, 1),
        "best_x": max(r["best_x"] for r in hist),
        "stuck_frac": stuck / (len(hist) * cfg.pop_size),
        "causes": causes,
        "trend": trend,
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


def _probe_job(job: tuple) -> tuple[str, int, dict]:
    game, lvl, seed, gens, run_root, persona = job
    return lvl, seed, probe(game, lvl, seed, gens, run_root, persona)


def aggregate(cells: list[dict]) -> dict:
    """Fold one level's per-seed cells into a paper-style row (mean +- 95% CI)."""
    n = len(cells)
    solved = [c for c in cells if c["first_win_gen"] is not None]
    wr_m, wr_ci = mean_ci([c["win_rate"] for c in cells])
    fw_m, fw_ci = mean_ci([float(c["first_win_gen"]) for c in solved])
    causes: dict[str, int] = {}
    for c in cells:
        for k, v in c["causes"].items():
            causes[k] = causes.get(k, 0) + v
    total_deaths = sum(causes.values()) or 1
    dom = max(causes.items(), key=lambda kv: kv[1]) if causes else ("-", 0)
    death_hist = [sum(c["death_hist"][i] for c in cells) for i in range(10)]
    comp_times = [c["mean_completion_time"] for c in cells if c["mean_completion_time"] is not None]
    pad = [c["progress_at_death"] for c in cells if c["progress_at_death"] is not None]
    return {
        "level": cells[0]["level"],
        "seeds": n,
        "solved_by": len(solved),
        "first_win_mean": round(fw_m, 1) if solved else None,
        "first_win_ci": round(fw_ci, 1) if solved else None,
        "win_rate_mean": round(wr_m, 3),
        "win_rate_ci": round(wr_ci, 3),
        "dominant_cause": dom[0],
        "dominant_cause_frac": round(dom[1] / total_deaths, 2),
        "causes": causes,
        "best_x_mean": round(statistics.fmean(c["best_x"] for c in cells), 1),
        "stuck_frac_mean": round(statistics.fmean(c["stuck_frac"] for c in cells), 3),
        "trends": [c["trend"] for c in cells],
        # Amr's table, aggregated
        "completion_rate_mean": round(statistics.fmean(c["completion_rate"] for c in cells), 3),
        "mean_completion_time": round(statistics.fmean(comp_times), 1) if comp_times else None,
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
    hdr = (f"{'LEVEL':<14}{'SOLVED':>7}{'FIRST WIN':>14}{'WIN RATE':>16}"
           f"{'DOMINANT CAUSE':>17}{'BEST X':>9}{'STUCK':>7}  TREND")
    out.append(hdr)
    out.append("-" * len(hdr))
    for r in difficulty_rank(rows):
        fw = (f"gen {r['first_win_mean']}±{r['first_win_ci']}" if r["first_win_mean"] is not None
              else "never")
        wr = f"{r['win_rate_mean']:.0%} ±{r['win_rate_ci']:.0%}"
        cause = f"{r['dominant_cause']} ({r['dominant_cause_frac']:.0%})"
        out.append(f"{r['level']:<14}{r['solved_by']}/{r['seeds']:<5}{fw:>14}{wr:>16}"
                   f"{cause:>17}{r['best_x_mean']:>9}{r['stuck_frac_mean']:>7.0%}  "
                   f"{'/'.join(r['trends'])}")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-seed level-balance report")
    ap.add_argument("--game", default="mario")
    ap.add_argument("--levels", nargs="*", default=None, help="default: every enabled level")
    ap.add_argument("--seeds", nargs="*", type=int, default=[1234, 2025, 31337])
    ap.add_argument("--gens", type=int, default=25, help="generation budget per probe")
    ap.add_argument("--persona", default="experienced", choices=sorted(PERSONAS),
                    help="player type the probe agents imitate")
    ap.add_argument("--out", default=os.path.join("runs", "balance"))
    ap.add_argument("--workers", type=int, default=None,
                    help="probe processes; default = min(jobs, cores-1), 1 = sequential")
    args = ap.parse_args()
    persona = get_persona(args.persona)

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    pygame.init()

    levels = args.levels or list_levels(args.game)
    if not levels:
        raise SystemExit(f"no levels found for game '{args.game}'")
    if args.levels:
        from .adapters import validate_level
        for lvl in args.levels:
            validate_level(args.game, lvl)
    run_root = os.path.join("runs", "probes", args.game)
    jobs = [(args.game, lvl, seed, args.gens, run_root, persona)
            for lvl in levels for seed in args.seeds]
    workers = args.workers or min(len(jobs), max(1, (os.cpu_count() or 2) - 1))
    print(f"balance probe: {len(levels)} levels x {len(args.seeds)} seeds = {len(jobs)} jobs, "
          f"budget {args.gens} gens each, {workers} worker(s)\n", flush=True)

    cells: dict[str, list[dict]] = {lvl: [] for lvl in levels}
    t0 = time.time()

    def _record(i: int, lvl: str, seed: int, cell: dict) -> None:
        cells[lvl].append(cell)
        fw = f"first win gen {cell['first_win_gen']}" if cell["first_win_gen"] else "never won"
        print(f"[{i}/{len(jobs)}] {lvl} seed {seed}: {fw}, "
              f"win rate {cell['win_rate']:.0%}, best_x {cell['best_x']}", flush=True)

    if workers <= 1:
        for i, job in enumerate(jobs, 1):
            lvl, seed, cell = _probe_job(job)
            _record(i, lvl, seed, cell)
    else:
        with mp.Pool(workers, initializer=_pool_init) as pool:
            for i, (lvl, seed, cell) in enumerate(pool.imap_unordered(_probe_job, jobs), 1):
                _record(i, lvl, seed, cell)
        seed_order = {s: i for i, s in enumerate(args.seeds)}
        for lvl in cells:  # deterministic JSON regardless of completion order
            cells[lvl].sort(key=lambda c: seed_order.get(c["seed"], 99))

    rows = [aggregate(c) for c in cells.values()]
    report = format_report(rows, args.game)
    print("\n" + report, flush=True)

    os.makedirs(args.out, exist_ok=True)
    # one file per (game, persona) — personas must never overwrite each other's rows
    out_path = os.path.join(args.out, f"report_{args.game}_{persona.name}.json")
    # Merge into any existing report so partial probes extend rather than clobber it.
    merged_rows, merged_cells = {}, {}
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            old = json.load(f)
        merged_rows = {r["level"]: r for r in old.get("levels", [])}
        merged_cells = old.get("cells", {})
    merged_rows.update({r["level"]: r for r in rows})
    merged_cells.update(cells)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"game": args.game, "gens_budget": args.gens, "seeds": args.seeds,
                   "persona": persona.name, "elapsed_s": round(time.time() - t0, 1),
                   "levels": list(merged_rows.values()), "cells": merged_cells},
                  f, indent=2, default=float)  # numpy scalars leak from some cores
    print(f"\nsaved {out_path}  ({time.time() - t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
