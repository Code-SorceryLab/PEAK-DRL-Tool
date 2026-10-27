"""Menu P (curriculum) without Mega Man, with a per-game budget of 200 generations per enabled level.

Same runs P builds: one population per game x persona x seed walks every enabled level in order
(3 wins in a generation -> next level), run dirs <game>_<persona>_curriculum_seed<N>, seed by seed.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import menu  # noqa: E402
from code.neuro.adapters import list_levels  # noqa: E402

GAMES = ["mario", "meatboy", "bomberman"]
SEEDS = ["101", "202", "303"]
WORKERS = 10
GENS = {g: 200 * len(list_levels(g)) for g in GAMES}          # mario 800, meatboy 2200, bomberman 2400

runs = []
for seed in SEEDS:                                             # seed outermost, as P does
    for game in GAMES:
        for persona in menu.PERSONA_CHOICES:
            run_dir = menu._run_dir(game, None, persona, "rays", f"curriculum_seed{seed}")
            cmd = menu._trainer_cmd(game, None, GENS[game], turbo=True, serve=False, run_dir=run_dir,
                                    extra=("--persona", persona, "--seed", seed, "--best"))
            runs.append((f"{game} | {persona} | seed {seed} | all levels | {GENS[game]} gens", cmd, run_dir))

print(f"budgets: {GENS} | {len(runs)} runs, {WORKERS} at once", flush=True)
done = menu.execute_parallel_runs(runs, WORKERS)
print(f"FINISHED: {done}/{len(runs)} runs succeeded", flush=True)
