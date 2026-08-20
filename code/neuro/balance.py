"""Level-balance report: multi-seed neuroevolution probes per level.

The GA successor to the paper matrix (run_paper_matrix / case_study, archived) and the
stats subsystem's failure-mode taxonomy: for each (level x seed), evolve a fresh
population and measure how learnable the level is. Difficulty is read the same way the
paper read it — win rate, generations-to-first-win, and the death-cause mix that names
WHY a level is hard (Enemy pressure vs Pit gaps vs Spike density vs Stall dead-ends).

Run:  python -m code.neuro.balance --game mario [--levels L1 L2 ...]
      [--seeds 1234 2025 31337] [--gens 25] [--out balance]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time

from .adapters import list_levels
from .evolution import GAConfig, Population
from .net import NeuralNet
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


def probe(game: str, level: str, seed: int, gens: int, run_root: str) -> dict:
    """One (level, seed) cell: evolve a fresh population, stop WIN_WINDOW gens after
    the first win (or at the gen budget), and summarize what happened."""
    cfg = GAConfig(seed=seed)
    run_dir = os.path.join(run_root, f"{level}_{seed}".replace(" ", "_"))
    trainer = Trainer(game, level, cfg, run_dir=run_dir)
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
    for r in hist:
        stuck += r["stuck"]
        for e in r["envs"]:
            if e["status"] == "DEAD":
                c = e.get("cause") or "?"
                causes[c] = causes.get(c, 0) + 1

    third = max(1, len(hist) // 3)
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
    }


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
    ap.add_argument("--out", default="balance")
    args = ap.parse_args()

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    pygame.init()

    levels = args.levels or list_levels(args.game)
    if not levels:
        raise SystemExit(f"no levels found for game '{args.game}'")
    run_root = os.path.join(args.out, "probes", args.game)
    jobs = [(lvl, seed) for lvl in levels for seed in args.seeds]
    print(f"balance probe: {len(levels)} levels x {len(args.seeds)} seeds = {len(jobs)} jobs, "
          f"budget {args.gens} gens each\n", flush=True)

    cells: dict[str, list[dict]] = {lvl: [] for lvl in levels}
    t0 = time.time()
    for i, (lvl, seed) in enumerate(jobs, 1):
        cell = probe(args.game, lvl, seed, args.gens, run_root)
        cells[lvl].append(cell)
        fw = f"first win gen {cell['first_win_gen']}" if cell["first_win_gen"] else "never won"
        print(f"[{i}/{len(jobs)}] {lvl} seed {seed}: {fw}, "
              f"win rate {cell['win_rate']:.0%}, best_x {cell['best_x']}", flush=True)

    rows = [aggregate(c) for c in cells.values()]
    report = format_report(rows, args.game)
    print("\n" + report, flush=True)

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"report_{args.game}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"game": args.game, "gens_budget": args.gens, "seeds": args.seeds,
                   "elapsed_s": round(time.time() - t0, 1),
                   "levels": rows, "cells": cells}, f, indent=2)
    print(f"\nsaved {out_path}  ({time.time() - t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
