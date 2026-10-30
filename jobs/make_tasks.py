"""Write the Narval task files, one shell command per line, for jobs/run_tasks.sh (order: jobs/README.md).

    .venv/bin/python -m jobs.make_tasks            # every stage into jobs/tasks/<stage>.txt
    .venv/bin/python -m jobs.make_tasks ast_play   # one stage (the AST stages read the mutants on disk)

$PY (the venv's python) and $W (the task's cores) are set by run_tasks.sh.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from code.games.eval_mutants import EVAL_LEVELS
from code.neuro import ast_eval
from code.neuro.adapters import list_levels
from code.neuro.ast_eval import PERSONA_ORDER as PERSONAS
from code.neuro.relay import stage_legs

GAMES = os.environ.get("PEAK_GAMES", "mario meatboy bomberman megaman").split()    # PEAK_GAMES=speedy: only that game
GAMES_2D = tuple(g for g in GAMES if g != "megaman")
SEEDS = "1234 2025 31337"
SECTIONS = [leg.level for leg in stage_legs(ast_eval.MEGAMAN_STAGE)]
QUICK = "GENS=5 SEEDS=1234"     # the derisking sweep: every stage of the real one, a few generations


def test() -> list[str]:
    """Step 6: one small training job per game, into runs/_test (not the real tree)."""
    level = dict(EVAL_LEVELS, megaman=("CutMan-3-Ledges",))
    return [f"$PY -m code.neuro.balance --game {g} --persona experienced --levels {level[g][0]} --gens 5 --seeds 1234 "
            f"--workers $W --out runs/_test/balance --root runs/_test/probes" for g in GAMES]


def quick_sweep() -> list[str]:
    """Step 7, run in its own copy (PEAK_DIR=$SCRATCH/peak_quick): the GA-sweep pipeline at 5 gens."""
    return ([f"GAMES={g} PERSONAS={p} {QUICK} bash run_all.sh sweeps" for g in GAMES_2D for p in PERSONAS]
            + [f"$PY -m code.neuro.balance --game megaman --persona {p} --gens 5 --seeds 1234 --workers $W && "
               f"$PY -m code.neuro.gasweep --game megaman --persona {p} --levels {' '.join(SECTIONS)} --gens 5 "
               f"--seeds 1234 --workers $W --confirm" for p in PERSONAS])


def quick_best() -> list[str]:
    """Step 7, after quick_final's ga_best.yaml: every persona on its GA winners, every level."""
    return [f"$PY -m code.neuro.balance --game {g} --persona {p} --gens 5 --seeds {SEEDS} --workers $W --best"
            for g in GAMES for p in PERSONAS]


def sweep2d() -> list[str]:
    """Step 8: 40-gen probes, sensor ablation and the GA sweep with its confirm, one game x persona a task."""
    return [f"GAMES={g} PERSONAS={p} bash run_all.sh sweeps" for g in GAMES_2D for p in PERSONAS]


def sweep_mm() -> list[str]:
    """Step 8: Mega Man's GA sweep, one persona x section a task (each writes only its own probe dirs)."""
    return [f"$PY -m code.neuro.gasweep --game megaman --persona {p} --levels {s} --gens 40 --seeds {SEEDS} --workers $W"
            for p in PERSONAS for s in SECTIONS]


def confirm_mm() -> list[str]:
    """Step 8, after gabest has rebuilt every section's reports: each persona's composite on all sections."""
    return [f"$PY -m code.neuro.gasweep --game megaman --persona {p} --levels {' '.join(SECTIONS)} --gens 40 "
            f"--seeds {SEEDS} --workers $W --axes --confirm" for p in PERSONAS]


def full() -> list[str]:
    """Step 8: 200-gen agents of every 2D game, persona and level, on the default GA and on its winners
    (ast_eval.baseline_tag keeps whichever won more on each level). Reports go to a stage folder;
    the reports stage rebuilds runs/balance from the probe dirs once every task has ended."""
    return [f"$PY -m code.neuro.balance --game {g} --persona {p} --levels {lv} --gens 200 --seeds {SEEDS} "
            f"--workers $W --out runs/balance_stage{best}"
            for g in GAMES_2D for p in PERSONAS for lv in list_levels(g) for best in ("", " --best")]


def ast_retrain() -> list[str]:
    """Step 8: every retrained variant of every game, one persona (and Mega Man section) a task. Mega Man's
    base-retrained is its full training (on --best, ast_eval.retrain_job)."""
    out = []
    for g in GAMES:
        for v in ast_eval.variants(g):
            m = ast_eval.mutant(g, v)
            if not (m and m["retrain"]):
                continue
            for p in PERSONAS:
                if g == "megaman":
                    out += [f"$PY -m code.neuro.ast_eval retrain --game megaman --mutant {v} --persona {p} --level {s} "
                            f"--workers $W" for s in m["retrain_levels"]]
                else:
                    out.append(f"$PY -m code.neuro.ast_eval retrain --game {g} --mutant {v} --persona {p} --workers $W")
    return out


def ast_play() -> list[str]:
    """Step 8: one playtest task per variant and mode, so no two tasks write the same CSV."""
    out = []
    for g in GAMES:
        for v in ast_eval.variants(g):
            if g == "megaman" and v == "base":      # the hand-picked relay of an older pipeline: not in a fresh runs/
                continue
            m, kind = ast_eval.mutant(g, v), v.split("@")[0]
            cmd = f"$PY -m code.neuro.ast_eval play --game {g} --variant {v}"
            out.append(cmd)
            if m and m["retrain"] and kind not in (ast_eval.RESEEDED, ast_eval.RETRAINED):
                out.append(cmd + " --replay")
            if g != "megaman" and kind != ast_eval.RESEEDED:
                out.append(cmd + " --random")
    return out


ONE_LINERS = {
    "quick_final": ["$PY -m code.neuro.gasweep --rebuild && $PY -m code.neuro.balance --rebuild"],
    "mutants": ["$PY -m code.games.eval_mutants --all-levels " + " ".join(f"--game {g}" for g in GAMES_2D)
                + (" && $PY -m code.games.megaman_mutants" if "megaman" in GAMES else "")],
    "gabest": ["$PY -m code.neuro.gasweep --rebuild"],
    "reports": ["$PY -m code.neuro.balance --rebuild"],
    "final": ["$PY -m code.stats.ast_metrics"],
}
STAGES = {f.__name__: f for f in (test, quick_sweep, quick_best, sweep2d, sweep_mm, confirm_mm, full, ast_retrain, ast_play)}


def main(names: list[str]) -> None:
    os.makedirs(os.path.join("jobs", "tasks"), exist_ok=True)
    for name in names or [*STAGES, *ONE_LINERS]:
        lines = ONE_LINERS[name] if name in ONE_LINERS else STAGES[name]()
        with open(os.path.join("jobs", "tasks", f"{name}.txt"), "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + "\n")
        print(f"{name:<12}{len(lines):>5} tasks  --array=0-{len(lines) - 1}")


if __name__ == "__main__":
    main(sys.argv[1:])
