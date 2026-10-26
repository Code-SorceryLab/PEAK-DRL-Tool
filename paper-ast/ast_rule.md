# The AST detection rule

Fixed on 2026-10-25 in `code/stats/ast_metrics.py` (`compare`, `false_alarms`). It was written on
2026-10-24, after the pilot replays had been seen (sealed, wider-gap and weaker-jump mutants), so
it is **not pre-registered**. The paper should say so. Its false-alarm rate is measured, not
assumed: the original level is retrained with three new seeds (`base-reseeded`) and compared with
itself exactly as a mutant is.

## Who answers what (decided 2026-10-25)

- **Replay is the detector.** The original's frozen agents play the mutant and are judged by the
  rule below against their own play on the original. Frozen agents replay attempt for attempt
  (the cosmetic copies are identical in every game), so no seed noise reaches the comparison: a
  flag can only come from the change itself.
- **Retraining answers one question: can the mutant still be finished?** Agents are trained on the
  mutant from scratch, the way the original's were. A breaking mutant should no longer be solved,
  and every other mutant still should.
- **Retraining is not used as a detector**, because it is too noisy. Retrained on the unchanged
  original with new seeds, the rule fires in 6 of 10 comparisons (Mega Man sections included),
  mostly through the death cluster and the top cause. Its completion signal, the one "still
  solvable" uses, never fired on an unchanged 2D level. In Mega Man sections training from new seeds
  sometimes fails outright (Street and Climb on the PC, Street on Kevin's machine), so a Mega Man
  "no longer solvable" needs that caveat.

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
the four costs for retrained agents. Replay has no seed noise for it to cost.

## Verdict

Each counted comparison gets one verdict:

- **broken**: completion lost
- **changed**: any other signal
- **unchanged**: no signal
- **not counted**: the original never wins on this level

The expected verdicts come from the mutant's class, labelled before any agent ran
(`code/games/eval_mutants.py`: breaking, degrading, neutral). The flood-fill check confirms that
breaking mutants have no route and the others still do. For the detector, a breaking or degrading
mutant is caught when flagged, and a neutral one is right when unchanged.

## Reported beside the rule

- **Win rate outside the seed spread** (the outline's alternative to Fisher): the mutant's pooled
  win rate falls outside the range of the original's per-seed win rates. It needs at least two
  seeds, so it is not available for Mega Man's relay, which plays one picked agent per section.
- **Strict count** (completion lost or Fisher only), for the retrained false-alarm check.
- **Cosmetic mutants** are replayed only and must match the original attempt for attempt (RQ2).
