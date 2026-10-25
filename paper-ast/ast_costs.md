# AST evaluation: what a check costs (RQ2)

Measured 2026-10-25 from the committed runs.

- **Retrain** times are `train_time_s` from `runs/ast/<game>/<variant>/report_*.json` (and, for the
  originals, `runs/balance/report_*` rows of the evaluation level). That is the summed wall time of
  every seed's generations, one core per seed, so it is CPU time.
  - The runs were on Narval (Alliance), 3 cores per task.
  - Training stops early once it wins, so an unsolvable mutant (a sealed route) costs the full 200
    generations.
- **Replay** times are wall time on the PC (AMD Ryzen 5 5600X, one core, Idle priority), including
  start-up.
  - The command was `python -m code.neuro.ast_eval play --game <g> --replay`, one attempt per frozen
    agent on every retrained mutant.
  - Mega Man's replay also picks the best of three runs per section, as its relay does.

## Retrain: CPU-hours per mutant (Narval)

| Game | seal_route | widen_gap / add_enemy | jump_velocity_down / fuse_down | off_route | all four | the original, same level |
|---|---:|---:|---:|---:|---:|---:|
| Mario 1-1 | 7.07 | 0.57 | 1.01 | 0.57 | 9.22 | 0.56 |
| Meat Boy 1-3 | 0.83 | 0.15 | 0.19 | 0.23 | 1.40 | 0.23 |
| Bomberman 09 | 7.26 | 2.76 | 1.16 | 2.61 | 13.79 | 2.65 |
| Mega Man Cut Man | 1.24 | 0.93 | 7.46 (all 8 sections) | 1.54 | 11.17 | 9.27 (`base-retrained`, 8 sections) |

- Total: **35.6 CPU-hours** for the 16 mutants, **44.9** with Mega Man's retrained original.
- On a 6-core desktop that would be about 6–8 hours of wall time; on one laptop core, about two
  days.

## Replay: wall time on one PC core

| Game | mutants | attempts | total | per mutant |
|---|---:|---:|---:|---:|
| Mario 1-1 | 4 | 72 | 55 s | ~14 s |
| Meat Boy 1-3 | 4 | 72 | 5 s | ~1 s |
| Bomberman 09 | 4 | 72 | 15 s | ~4 s |
| Mega Man Cut Man | 4 | 240 (the jump mutant over the full stage and 8 sections) | 244 s | 20 attempts per section |

Replaying every mutant of all four games takes **under 6 minutes on one core**. Retraining them
takes about 36 CPU-hours.

## Not measured yet

- Playtests of the retrained agents ran on Narval (job 7130198); their logs were not copied home.
- `base-reseeded`, the false-alarm check, should cost about what the originals did: about
  13 CPU-hours (0.56 + 0.23 + 2.65 + 9.27).
- The outline's "one laptop, no GPU" holds for replay. For retraining it holds only as a few
  hours on a 6-core machine; these runs used the cluster.
