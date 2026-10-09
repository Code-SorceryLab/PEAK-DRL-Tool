# AST 2027 paper — task list

- **Paper:** Agent Playthroughs as Regression Tests: An Engine for Automated Game Testing
- **Deadline:** 30 October 2026 (AoE). Outline and context: `paper.md` (section numbers below refer
  to it).
- **Pipeline everyone is building toward:** design the game → train the agents → play (manual or
  automated) → measure telemetry metrics. A regression check runs it twice and compares.
- **Owners:** Kevin = games, Al = NN training, Amr = telemetry.
- Dates are proposals; adjust them together.

## Milestones

| Date | What must work |
|---|---|
| Wed 14 Oct | Pilot on one level (Mario 1-1): train → test run → telemetry → compare, with about ten hand-made mutants |
| Mon 19 Oct | All 14 levels playable and trainable; baseline telemetry for agents; human recordings started |
| Fri 23 Oct | Full mutant runs finished; human baseline finished |
| Mon 26 Oct | Tables and figures ready |
| Fri 30 Oct | Submission |

---

## Kevin — games (pipeline stage 1, Design)

Goal: the 14 evaluation levels exist, are deterministic, and can be mutated automatically.

- [ ] **Super Mario Bros., World 1 (4 levels).** 1-1 and 1-2 are active; enable 1-3; build 1-4.
      Decide with Cristiano whether 1-4 includes the boss or ends at the goal.
- [ ] **Super Meat Boy, first chapter (6 levels).** Check the first six existing levels against the
      original. Confirm which chapter we clone: The Forest (what the repository has) or Chipper
      Grove (the link Cristiano sent).
- [ ] **Bomberman NES, stages 1–4.** The current 12 levels are a teaching ladder, not the NES
      stages. Decide with Cristiano: seeded generator (bricks, exit, power-up, and enemy start
      positions random per seed, as in the original) or fixed layouts per stage.
- [ ] **Register the 14 levels** in the configs so the trainer and manual play list exactly them.
- [ ] **Determinism.** Same level + same seed + same actions gives the same episode, on all three
      games. Remove or seed every unseeded `random` call in the cores (the Mario core has some in
      its curriculum code). Add a test per game.
- [ ] **Mutant generator** (section 3.7). A command that takes a level file or a config and writes
      one mutant per operator and location, each to its own file, with a small record of what was
      changed and where (operator, tiles or key, before and after).
  - Level operators: seal a route, widen a gap, move the goal, add a hazard or enemy on a path,
    remove a platform, ladder, or spring.
  - Mechanics operators: jump velocity, gravity, run speed, enemy speed, time limit, each by a
    fixed step.
  - Neutral operators: cosmetic tile, platform off every route, moved coin.
- [ ] **Ground truth for each mutant** (section 4.3): label it *breaking*, *degrading*, or
      *neutral* before any agent runs, from the reachability check plus playing it once.
- [ ] **Level figures** for the paper: one clean render per game.

Done when: `list_levels` shows the 14 levels; the determinism tests pass; the generator produces
about 10 mutants and 3 neutral changes per level, each with a label.

---

## Al — NN training (pipeline stage 2, Train, and the automated half of stage 3, Play)

Goal: trained agents for every level, persona, and seed, and a way to make them play without
learning.

- [ ] **Test-run command** (section 3.4). Load saved models (`best.npz`), play a given level
      headless at full speed for a fixed number of attempts, one life each, with a given game seed,
      and hand each finished episode to Amr's telemetry writer. No learning, no looping, no
      advancing to the next level. The current `--replay` is a viewer and does none of this.
- [ ] **Seed everything in a test run.** The *bad* persona's input slips use an unseeded random
      generator in replay; give it a seed so its runs repeat.
- [ ] **Who plays in a test run:** the best agent of each training seed (10 seeds → 10 agents).
- [ ] **Training plan.** 14 levels × personas × 10 seeds. Personas (section 3.3): novice, bad,
      experienced, good, and speedrunner on all three games; killer and collector on Mario and
      Bomberman only. Time one level first and report the cost before launching everything.
- [ ] **Baseline training** on the 14 levels. Report which levels no seed solves; those cannot be
      used for detection.
- [ ] **Regression check, two modes** (section 3.6):
  - *Replay:* the baseline agents play a changed level (no training).
  - *Retrain:* new agents are trained on the changed level with the same seeds, then play.
- [ ] **Mutant runs.** Replay on every mutant; retrain where the replay differs from the baseline.
      Keep replay and retrain results apart.
- [ ] **Repeatability and cost** (RQ2): run the same level and seed twice and confirm identical
      telemetry, per game; record steps per second and wall-clock time per check.
- [ ] **Training data stays separate** from telemetry: fitness per generation, generations to first
      win, and settings are for cost and debugging only.

Done when: one command trains a level, one command plays the saved agents and produces telemetry,
and the pilot (Mario 1-1, ten mutants) runs end to end in both modes.

---

## Amr — telemetry (the recording half of stage 3, Play, and stage 4, Measure)

Goal: one telemetry format for agents and humans, the metrics computed from it, and the comparison
that produces a verdict.

- [ ] **Separate telemetry from training data** (section 3.6 of the outline, "Telemetry and
      training data"). Today the trainer writes an episode row for every genome in every
      generation, and win rate is computed from training generations. Telemetry must come only from
      test runs and human play. Start from `code/neuro/episodes.py` (the shared writer) and
      `code/games/tools/manual_play.py --record`.
- [ ] **Telemetry row.** Keep the current columns (persona or player, game, level, cause of death,
      jumps, pickups, speed, progress, route, kills, bricks, time) and add what is missing to
      identify a run: source (agent or human), model or player id, game seed, level version or
      mutant id, attempt number.
- [ ] **Fix progress.** It is horizontal position over level width, which is wrong for levels whose
      goal is not at the right edge (a Super Meat Boy win can show 45%). Use path progress to the
      goal.
- [ ] **Metrics per level and persona** (section 3.5): completion, win rate, furthest progress,
      death locations, cause-of-death mix, time, route, and the per-episode counts.
- [ ] **Play coverage.** From the routes, the set of tiles the agents visited on a level. Needed to
      explain which mutants are missed.
- [ ] **Comparison command** (section 3.6). Two telemetry sets in (baseline and changed game),
      one verdict per level and persona out: *unchanged*, *changed but playable* (with the
      differences listed), or *broken*; non-zero exit when something is flagged. Agree the flagging
      rule with Cristiano before the mutant runs and do not change it afterwards.
- [ ] **Localization.** For a flagged level, report whether the new death cluster falls on the
      changed tiles (uses Kevin's mutant record).
- [ ] **Human recordings.** Make sure `--record` works with a real keyboard on all three games;
      write a one-page protocol (levels, attempts per level, one life per attempt) and collect the
      files. Report attempts as a distribution, first attempts separately.
- [ ] **Human vs agent comparison** (RQ4): completion, cause of death, death locations on the level
      map, ranking of levels by difficulty.
- [ ] **Tables and figures:** detection per operator and game, false alarms, replay vs retrain,
      coverage vs detection, death maps for humans and agents.

Done when: a test run and a human session produce files the same metrics code reads, and the
comparison command gives a verdict for the pilot mutants.

---

## Where the three parts meet

| Hand-off | From | To | Agree on |
|---|---|---|---|
| Level and mutant files, with the record of what changed | Kevin | Al, Amr | file layout and the mutant record format |
| Finished episodes from a test run | Al | Amr | the telemetry row and where files are written |
| Verdicts and metrics | Amr | paper | the flagging rule |

- Agree these three formats first (by Mon 12 Oct); everything else can then proceed in parallel.

## Not assigned yet

- [ ] Who plays the human baseline (about 10 per game) and when.
- [ ] Cost of adding Bomberman from the version history (RQ3): clean the preliminary count.
- [ ] Read the closest related work in full (Feldmeier and Fraser, ASE 2022) and fix the difference
      statement in `paper.md` 6.4.
- [ ] Double-blind: the README expands PEAK with author names and the repository is public.
- [ ] Remove `.DS_Store` from the repository and ignore it.
- [ ] Writing: turn `paper.md` into prose, then LaTeX.

## Open decisions (Cristiano)

- Headline finding (see "Main goal and headline finding" in `paper.md`); the pilot decides.
- Super Meat Boy chapter; Mario 1-4 boss; Bomberman generator or fixed layouts.
- Personas per game; number of seeds (proposal: 10).
- The flagging rule.
