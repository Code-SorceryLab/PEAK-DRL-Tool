# The AST detection rule

Fixed on 2026-10-25 in `code/stats/ast_metrics.py` (`compare`, `false_alarms`). It was written on
2026-10-24, after the pilot replays had been seen (sealed, wider-gap and weaker-jump mutants), so
it is **not pre-registered**. The paper should say so. Its false-alarm rate is measured, not
assumed: the original level is retrained with three new seeds (`base-reseeded`) and compared with
itself exactly as a mutant is.

## Unit

One comparison = one level (a Mega Man section counts as a level) played by every agent once:

- Mario 1-1, Meat Boy 1-3, Bomberman 09: 6 personas x 3 training seeds = 18 attempts per variant.
- Mega Man Cut Man: the experienced persona's relay, 20 attempts per section or full stage.

A comparison **counts only if the original's agents solve that level at least once.** An
unsolved original can lose nothing, so it is reported but left out of every tally.

## Signals

A mutant is flagged when any of these holds against the original over the same level:

| Signal | Test |
|---|---|
| completion lost | the original's agents win at least once, the mutant's never |
| win rate | Fisher's exact test on pooled wins vs losses, two-sided, p < .05 |
| death cluster moved | the cluster of failed end tiles (Chebyshev radius 3) moves more than 3 tiles, or appears |
| new top cause | the most common cause of failure differs |

There is no correction for testing four signals; the false-alarm check measures what the OR of
the four costs instead.

## Verdict

Each counted comparison gets one verdict:

- **broken**: completion lost
- **changed**: any other signal
- **unchanged**: no signal
- **not counted**: the original never wins on this level

The expected verdicts come from the mutant's class, labelled before any agent ran
(`code/games/eval_mutants.py`: breaking, degrading, neutral; the flood-fill check confirms that
breaking mutants have no route and the others still do). A breaking or degrading mutant is
caught when flagged; a neutral one is right when unchanged.

## Reported beside the rule

- **Win rate outside the seed spread** (the outline's alternative to Fisher): the mutant's pooled
  win rate falls outside the range of the original's per-seed win rates. It needs at least two
  seeds, so it is not available for Mega Man's relay, which plays one picked agent per section.
- **Two modes per mutant**: *retrain* (agents trained on the mutant, `agents.csv`) and *replay*
  (the original's frozen agents on it, `agents_replay.csv`), judged by the same rule.
- **Cosmetic mutants** are replayed only and must match the original attempt for attempt (RQ2).
