# PEAK for AST 2027 — working outline

Brainstorm outline in bullets. This file is the source of truth until we move to LaTeX
(`main.tex` keeps the IEEEtran skeleton, `references.bib` the verified entries).

- **Venue:** AST 2027, regular paper, 10 pages + 2 pages of references, IEEEtran, double-blind.
- **Deadline:** 30 October 2026 (AoE).
- **Conventions:** citations are `[@key]` from `references.bib`; **[TODO: ...]** is an open task;
  **[VERIFY: ...]** is a number or claim not yet re-checked (run data for Mario, Super Meat Boy,
  and Bomberman is back in `runs/` since the pull of 2026-10-09, so these can now be checked);
  citation keys are the Better BibTeX keys from Zotero.

## Positioning

- **One sentence:** PEAK is test infrastructure: a game engine designed for testability, in which
  agent playthroughs act as repeatable regression tests.
- **Judged on:** oracle, repeatability, cost, and fault detection. The editor and the ML integration
  are means to that end, not the contribution.
- **Core evidence:** game mutation (mutants of levels and mechanics, not of code).
- **Not claimed:** the editor as a novelty; a better learning algorithm; balance (extended paper);
  that the tool transfers to commercial engines (the requirements transfer, the tool does not).

## Main goal and headline finding

- **Main goal (one question):** can playing agents serve as regression tests for game content?
- **Answer the paper aims to give:** yes, if the engine is built for it, and here is how well it
  works and where it stops.
- **Candidate headline findings** (hypotheses until the runs are done; pick the one the data
  supports best):
  1. *Trained agents are brittle tests; retraining is what makes them regression tests.* Replaying
     old agents on a changed game flags many harmless changes, but an identical replay safely
     skips retraining. Software-testing readers know this as test fragility and test selection.
     Nobody has measured it for game agents. Publishable whichever way it comes out.
  2. *Play coverage predicts detection.* Mutants on tiles the agents visit are detected; mutants
     elsewhere are missed. Gives playtests an adequacy measure, the way code coverage does for
     unit tests. Computable from the routes already in the telemetry.
  3. *Different personas detect different changes.* A single best agent misses changes that only
     affect a weaker or differently motivated player. The test-diversity argument.
  4. *Agents fail where people fail.* Supports using agents before human playtests.
- **Extra contributions worth adding:**
  - *A benchmark.* The levels, the labeled game mutants, the trained agents, and the agent and
    human telemetry, released together. Lets others test their own agents on the same mutants.
  - *Play coverage* as a measure (finding 2), reported next to the detection rate.
  - *Operators tied to real defects.* Map each mutation operator to a category in a game-bug
    taxonomy [@lewisWhatWentWrong2010], so the mutants are not arbitrary.
- **[TODO: run a pilot on one level (Mario 1-1, about ten mutants, the saved agents) to see which
  finding holds before committing the paper to it.]**

## Title

- **Agent Playthroughs as Regression Tests: An Engine for Automated Game Testing**

## Abstract

- **[TODO: write after the evaluation is settled.]**

## Keywords

- game testing, regression testing, game mutation, testability, test infrastructure, playing
  agents

---

## 1. Introduction

### 1.1 Problem

- Software teams rerun a test suite after every change. For games this is rare and costly:
  checking that a level still plays as intended needs someone to play it. Where regression
  testing of games exists, it is built for one large commercial game
  [@yuGameRTSRegressionTesting2023; @wuRegressionTestingMassively2020].
- Game testing is still largely manual [@politowskiSurveyVideoGame2021; @politowskiAutomatedVideoGame2022]; a change to a level, a
  mechanic, or the engine code is rechecked by human playtesting, if at all.
- A *playability regression*: after a change, the game still runs, but a player can no longer do
  what they could before (a goal that became unreachable, a jump that no longer clears, a route
  that got sealed).
- These are silent failures: nothing crashes, no error is logged, and code-level tests still pass.
  Only playing the level reveals them.
- Since replaying every level after every change is too expensive, they surface late: at the next
  scheduled playtest, or after release. Consequences:
  - **costlier fixes:** other content has been built on top of the broken level or mechanic, and
    the change that caused it is hard to trace back;
  - **wasted playtests:** human testers hit a blocker and the session yields no feedback on what
    it was meant to evaluate;
  - **defects reach players:** levels that cannot be finished, patched after launch
    [@politowskiSurveyVideoGame2021; @trueloveWellFixIt2021; @linStudyingUrgentUpdates2017;
    @zhangWhyAreWe2016];
  - **fear of change:** designers avoid adjusting shared mechanics late in production, because
    nobody can recheck every level that depends on them.
- **[TODO: the first, second, and fourth consequences are our argument; look for support in
  postmortem data [@politowskiDatasetVideoGame2020] or state them as argument.]**
- Playing agents can be the test driver, but only if the game is built to be driven by them.

### 1.2 Why existing options fall short

- Production engines: built for authoring and rendering; attaching agents needs substantial
  integration work [@gillbergTechnicalChallengesDeploying2023].
- Fixed benchmarks and emulators: easy to train on, but levels and mechanics cannot be edited, so
  there is nothing to regress against.
- Agent-based testing work studies the agent (find crashes, play like humans)
  [@bergdahlAugmentingAutomatedGame2020; @zhengWujiAutomaticOnline2019; @ariyurekAutomatedVideoGame2019; @holmgardAutomatedPlaytestingProcedural2019], less what the
  game must provide for agent runs to count as tests.
- Learned agents are flaky test drivers: results vary across seeds and runs
  [@hendersonDeepReinforcementLearning2018; @parrySurveyFlakyTests2021], so a changed outcome may
  be the run, not the game.
- Closest work evolves neural networks as tests for the code of small games
  [@feldmeierNeuroevolutionBasedGenerationTests2022]; it targets program coverage, not changes to
  levels and mechanics.
- No accepted way to say how good such a test is: there is no fault-detection measure for agent
  playthroughs comparable to a mutation score.

### 1.3 Key idea

- Treat the game as the system under test and an agent playthrough as a test case.
- Regression check: keep the agents and seeds fixed, change the game, compare the outcomes. The
  oracle is the previous version of the same game [@barrOracleProblemSoftware2015].
- This is an engine-design problem first: the engine must make playthroughs repeatable, cheap, and
  comparable.
- Measure test quality with mutants, as software testing does. We separate two notions:
  - **Software mutation** (the established one): make small changes to the program code and check
    whether a fixed test suite fails. Every non-equivalent mutant is a fault.
  - **Game mutation** (ours): make small automatic changes to the game's content, a level layout
    or a mechanics value, and check whether agent playthroughs behave differently. The code is
    untouched.
- Game mutation differs in what counts as a fault: a game mutant may break a level, make it
  harder, or change nothing, so we count detections on the first two and false alarms on the
  third.
- How the agents learn to play is a pluggable feature of the engine, not the contribution.
- Pipeline: design the game → train the agents → play the game (manual or automated) → measure
  telemetry metrics. A regression check runs it twice and compares the metrics.

### 1.4 Approach: PEAK

- What we built, and what each part is for in testing terms:

  | What PEAK has | Role in testing |
  |---|---|
  | Level editor; levels and mechanics as text files | Cheap versions of the system under test; faults can be injected systematically |
  | Deterministic, headless game cores | Repeatable test execution; no flaky runs |
  | One adapter interface between games and agents | Test drivers survive changes to the game |
  | Agent training (neural networks) with debug views | Automatic test drivers; failure diagnosis (where and why the agent died) |
  | Seven player personas: skill (novice, bad, experienced, good) and play style (killer, collector, speedrunner) | Diverse test inputs; a change can break a level for one persona only |
  | Per-episode records and reports | Structured test outcomes that can be compared across versions |
  | Clones of the opening levels of three known games | Test subjects readers can judge; evidence the harness generalizes |
  | Recorded human play in the same format | Calibration of agent testers against people |

### 1.5 Evidence preview

- Fault detection: **[TODO: N]** mutants of levels and mechanics, **[TODO: X%]** detected,
  **[TODO: Y]** false alarms on neutral changes.
- Repeatability: same seed gives an identical run on every game.
- Cost: **[VERIFY: minutes per check on a laptop, no GPU.]**
- Extensibility: adding a game of a different genre changed **[VERIFY: 33 lines]** of the shared
  training loop.
- Calibration: agents and human players fail in the same places on **[TODO]** of the levels.

### 1.6 Contributions

- **Requirements:** what a game engine must provide for agent playthroughs to work as regression
  tests.
- **Game mutation:** a definition of mutants for game content (levels and mechanics), set apart
  from software mutation; operators for 2D games; and a mutation-based measure of how well agent
  playthroughs detect changes.
- **PEAK:** an open-source engine that meets the requirements, with several games, an adapter
  interface, pluggable agent training, and reporting.
- **Evaluation:** fault detection, repeatability, cost, extensibility, and agreement with human
  play.

---

## 2. Requirements for agent-driven regression testing

- Framing: a short scenario, then the requirements the engine section answers one by one.
- Scenario: a designer moves a platform two tiles; nobody replays the level; the goal is now out of
  jump range; the defect ships to the next playtest.
- **R1 Determinism:** the same seed and the same game give the same run, so a changed outcome means
  the game changed.
- **R2 Headless execution:** runs need no display and no real-time clock, so they fit a build
  server and run faster than real time.
- **R3 Stable agent interface:** agents see the game only through a fixed interface, so a test
  driver survives edits to levels and mechanics.
- **R4 Content as data:** levels and mechanics are files that can be diffed, versioned, and mutated
  without touching code.
- **R5 Structured outcomes:** each playthrough yields a record (outcome, cause, position, route)
  that can be compared across versions.
- **R6 Low cost per run:** a check must finish in minutes on a developer machine, or it will not
  be rerun after each change.
- **R7 Player variety:** more than one skill level and play style, since a change can break the
  level for one kind of player only.
- Grounding: R1, R3, R5 are controllability and observability from the testability literature
  [@voasSoftwareTestabilityNew1995; @binderDesignTestabilityObjectoriented1994]; R1 also answers
  flaky tests [@luoEmpiricalAnalysisFlaky2014]; R2, R3, R6 answer the deployment costs reported for
  production games [@gillbergTechnicalChallengesDeploying2023]; R7 follows persona-based
  playtesting [@holmgardAutomatedPlaytestingProcedural2019]. **[TODO: confirm each mapping against
  the papers.]**

---

## 3. PEAK: engine design

### 3.1 The testing pipeline

- Four stages; the rest of this section follows them in order.

  ```mermaid
  flowchart LR
      D["1. Design<br/>levels + mechanics"] --> T["2. Train<br/>agents learn the game"]
      T --> P["3. Play<br/>manual | automated"]
      D --> P
      P --> M["4. Measure<br/>telemetry metrics"]
      M -.->|change the game, run again, compare| D
  ```

  | Stage | Who | Input | Output |
  |---|---|---|---|
  | 1. Design | Designer, in the editor or a text file | — | Level files and mechanics config (the game under test) |
  | 2. Train | The learner | The game, a persona, a seed | Trained agents + training data |
  | 3. Play | Human players (manual) or trained agents (automated) | The game | Telemetry: one row per episode |
  | 4. Measure | The engine | Telemetry | Metrics per level and persona; report |

- Manual play skips stage 2: a person needs no training. Both kinds of play write the same
  telemetry, so stage 4 does not know who played.
- A regression check is the pipeline run twice, before and after a change, with stage 4 comparing
  the two sets of metrics (3.6).
- Python and Pygame; runs locally and offline; no GPU.
- Table: each requirement R1–R7 mapped to the stage and component that meets it.

### 3.2 Stage 1, Design: the game under test (R1, R2, R4)

- Three games in this paper, each one core module, each a clone of the opening of a known game:

  | Game | Levels in the evaluation | In the repository today |
  |---|---|---|
  | Super Mario Bros. | 4: World 1 (1-1 to 1-4) | 1-1 and 1-2 enabled; 1-3 present but disabled; 1-4 missing |
  | Super Meat Boy | 6: the first six levels of the first chapter | 11 levels present, named after the original game's first chapter (The Forest) |
  | Bomberman (NES) | 4: stages 1–4 | hand-made single-screen levels (a teaching ladder of 12 active levels, not the NES stages) |
  | **Total** | **14** | |

- Fallback if time runs short: 2 Mario (1-1, 1-2), 4 Super Meat Boy, 3 Bomberman (9 levels).

- Sonic and Megaman clones exist but are archived in the code (not in the active game list);
  they join the evaluation only if time allows.
- Two genres: side-scrolling platformers and a top-down maze game with a different action space.
- **Random content in the NES original** (from memory; check against a reliable source): the
  pillar grid is fixed, but the bricks, the exit, and the power-up hidden under bricks are placed
  at random on each play, and enemies start at random positions. Enemy types and counts and the
  power-up type are fixed per stage.
- **The clone today:** layouts are fixed text files (bricks, exit, and enemy start positions are
  authored); only enemy movement is random, and it is seeded. The arena is one screen (15 × 13
  tiles); the original scrolls over a wider arena.
- **[TODO: decide how to clone it.**
  - *Seeded generator (faithful):* a stage is a set of parameters (brick density, enemy types and
    counts, power-up) plus a seed. Shows the engine handles random content deterministically, and
    game mutants can target the parameters. Needs a generator and a wider arena.
  - *Fixed layouts (what exists):* a few layouts per NES stage, with that stage's enemies. Less
    work, same pipeline as the platformers, but not how the original works.**]**
- **[TODO: Super Meat Boy chapter. The link given is Chipper Grove, which I believe is the first
  chapter of *Super Meat Boy Forever*; the 11 levels in the repository carry names from the
  original game's first chapter, The Forest. Decide which one.]**
- **[TODO: Mario 1-4 ends in a boss; decide whether the clone includes it or ends at the goal.]**
- Levels are plain-text tile maps, one character per tile; the level editor writes the same files.
- Mechanics (gravity, jump velocity, run speed, time limits) and the enabled level list are YAML.
- An edit takes effect without a restart: create a level and play it right away.
- A change to a level or a mechanic is a text diff, reviewable like code.
- Each core has `reset`, a fixed-timestep `step`, optional `render`, and episode state (alive, won,
  cause of death).
- Deterministic: one seed drives level setup and every game step.
- Headless mode skips rendering entirely; the same code path runs with or without a display.

### 3.3 Stage 2, Train: preparing the automated players (R3, R6, R7)

- Adapter interface between a game and the learner: one protocol with about 15 members (player
  state, `reset`, `step`, world queries, `fitness`, `episode_stats`, `set_level`). The learner
  imports the adapter, never a game.
- Optional adapter hooks for games that do not fit the platformer shape: number of actions, a
  game-owned sensor vector, a game-defined progress measure.
- Fitness lives in the adapter, not in the game, so the game needs no knowledge of how it is
  tested.
- Sensors: 14 numbers from raycasts and body state by default, or a 368-cell tile window.
- Default learner: a genetic algorithm over the weights of a small fixed network (14 → 16 → 3, 291
  weights) [@suchDeepNeuroevolutionGenetic2017]. No gradients, replay buffer, or GPU; fully seeded.
- Earlier learner: deep reinforcement learning (PPO) through standard ML libraries. **[TODO: decide
  how to present it; it is on the archive branch and does not use the current adapter.]**
- Player personas, as documented in the project (README, *Player personas*). Every persona plays
  the same game with the same network; they differ in *hands* (skill) and in *event weights*
  (play style):

  | Persona | Hands | Paid for | Question it answers as a test |
  |---|---|---|---|
  | novice | walk only, senses every 3rd frame | reaching the goal | can a new player still finish? |
  | bad | walk only, senses every 4th frame, a 0.2 s mis-press about every 0.5 s | reaching the goal | where does a clumsy player now die? |
  | experienced | walk, senses every frame (default) | reaching the goal | the baseline |
  | good | sprint, senses every frame, clean inputs | reaching the goal | does skill still carry through? |
  | killer | as experienced | goal + share of enemies killed | is fighting still possible and worth it? |
  | collector | as experienced | goal + share of coins or power-ups picked up | are the pickups still reachable? |
  | speedrunner | sprint, senses every frame | goal + 25 per second left on a win | is the fast route still there? |

- Event weights are shares of the goal (killing every enemy pays as much as finishing), capped per
  level, following procedural personas [@holmgardAutomatedPlaytestingProcedural2019].
- A level without the event gives a plain player: Super Meat Boy has no enemies or pickups, so
  killer and collector play like experienced there.
- Reported in the README for Mario 1-1 **[VERIFY against `runs/`]**: killer wins 100% and kills
  7.0 enemies against 2.7 for experienced; bad wins 20%.
- The persona is saved with the trained agent and reused when it plays.
- Why personas matter for testing: a change can break a level for one kind of player only (a gap
  that needs sprint, a route that needs a kill, a pickup that became unreachable).
- **[TODO: personas per game in the evaluation; each one multiplies the runs. Candidate: the four
  skill personas and speedrunner on all three games; killer and collector on Mario and Bomberman
  only.]**
- **[TODO: the bad persona's slips use an unseeded random generator in replay; seed it, or its
  runs are not repeatable.]**
  good, speedrunner, killer.]**
- Output 1: trained agents (saved weights), one set per level, persona, and seed.
- Output 2: training data (fitness per generation, generations to first win, learner settings).
  Used for cost reporting and for debugging the learner, never as a test result.
- Debug views for this stage: live dashboard with sensor values, fitness over generations, manual
  takeover.

### 3.4 Stage 3, Play: manual or automated (R5)

- **Manual:** a person plays with the keyboard; each attempt is recorded.
- **Automated:** the trained agents play the level in a test run, with no learning.
- Both write the same telemetry through one writer: one row per episode with player or persona,
  game, level, outcome, cause of death, progress, sampled route, jumps, coins, kills, time.
- One life per attempt in both modes, so attempts are comparable.
- Random-action player available through the same path, as a floor.
- **[TODO: the code mixes stages 2 and 3 today. Episode rows are written during training, for every
  genome in every generation, and win rate is computed from the last training generations. Add the
  test run: after training, the saved agents play and only those episodes become telemetry.]**
- **[TODO: decide who plays in a test run. One frozen agent on a fixed level repeats the same
  episode, so a rate needs several agents: the best agent of each seed (recommended), or the top-k
  of the final population.]**

### 3.5 Stage 4, Measure: telemetry metrics (R5)

- Computed per level and per persona (or per human player group), from telemetry only:
  - completion (solved by at least one player);
  - win rate;
  - furthest progress and death locations;
  - cause-of-death mix;
  - time to finish and route taken;
  - jumps, pickups collected, enemies killed, bricks destroyed (per episode).
- Human attempts are reported as a distribution over attempts, with first attempts shown
  separately, since people learn the level.
- Report: one self-contained HTML page per set of runs, with death locations and routes drawn on
  the level, and a replay of the best agent.

### 3.6 Regression check: the pipeline run twice

- Baseline: stages 1–4 on the current game; keep the metrics.
- Change the game (stage 1).
- Run again, in one of two ways, both deterministic:
  - **Replay:** skip stage 2; the agents trained on the baseline play the changed game. Seconds per
    level. Answers "does the known solution still work?"
  - **Retrain:** redo stage 2 on the changed game from the same seeds, then play. Minutes per
    level. Answers "can the level still be solved?"
- Compare the stage 4 metrics of the two runs. Signals, strongest first:
  - completion flips (solved → unsolved, or the reverse);
  - win rate moves beyond the spread across seeds;
  - furthest progress drops, and deaths cluster at a new location;
  - cause-of-death mix shifts (e.g., pit deaths replace enemy deaths);
  - still solved, but slower or by another route.
- An unchanged game gives identical telemetry, so any difference comes from the change.
- What the check answers, for the person who made the change: *did my change alter how the game
  plays, for whom, and where?*

  | Step | Question | Possible answers |
  |---|---|---|
  | Replay (old agents, changed game) | Did the change touch play at all? | no → stop; yes → continue |
  | Retrain (new agents, changed game) | Can each kind of player still get through? | yes; yes, but fewer of them; no |
  | Play (new agents, changed game) | How do they play it now? | telemetry |
  | Measure and compare with the baseline | What is different? | where they die, how long it takes, which route, which personas |

- Verdict per level and persona, one of three:
  - *unchanged:* replay identical;
  - *changed but playable:* still completed, with the differences listed (harder, slower, new
    route, one persona affected);
  - *broken:* no longer completed by any agent.
- The check reports that play changed, not that the change is wrong: a harder level may be what
  the designer wanted. Only *broken* is a defect without knowing the intent.
- Proposed roles (to confirm with data):
  - **Retrain is the test.** Its verdict is about the game: can a player of this kind still get
    through, and how hard is it.
  - **Replay is triage.** A trained agent reacts to what it senses; it is not a recorded input
    sequence, so it does not fail on every change. But it was trained on one level and may be
    brittle, so a replay failure alone cannot separate "the level broke" from "this agent's
    habit broke". Its reliable outcome is the other one: identical telemetry means the change did
    not touch play, and retraining can be skipped. This is test selection
    [@yuGameRTSRegressionTesting2023; @yooRegressionTestingMinimization2012].
- **[TODO: measure how brittle replay is: share of neutral and degrading mutants on which the
  baseline agents fail. If most fail, replay is only a filter; if few do, it is a cheap test.]**
- **[TODO: build the comparison command (two run directories in, flagged levels and the signal
  that fired out, non-zero exit on a flag). Today the comparison is done by reading the report.]**

### 3.7 Game mutation

- **Definition.** A *game mutant* is a copy of the game that differs from the original by one
  small, automatic change to its content: a level layout or a mechanics value. The game code is not
  changed. *Game mutation* is generating such mutants and checking whether the tests detect them.
- **Software mutation**, for contrast, changes the program code and checks whether a fixed test
  suite fails [@jiaAnalysisSurveyDevelopment2011; @papadakisMutationTestingAdvances2019].
- **Purpose.** Same as software mutation: measure how well a set of tests
  detects changes. Here the tests are agent playthroughs. We use mutants to evaluate the testing
  approach, not to improve a test suite.
- **How the two differ:**

  | | Software mutation | Game mutation (this paper) |
  |---|---|---|
  | What is mutated | Source code | Level layouts and mechanics values (data) |
  | The tests | Fixed test cases with assertions | Agent playthroughs: agents, personas, seeds |
  | Mutant detected ("killed") when | An assertion fails | Telemetry metrics differ from the baseline beyond a fixed threshold |
  | Is every mutant a fault? | Yes, unless equivalent | No. A mutant may break the level, make it harder, or change nothing |
  | Mutants that change nothing | Equivalent mutants, a nuisance to filter out | Neutral mutants, generated on purpose to measure false alarms |

- **Mutant classes**, assigned before looking at agent results (ground truth):
  - *breaking:* the level can no longer be completed (e.g., goal out of reach);
  - *degrading:* still completable, but playability changed (harder jump, longer route, new
    hazard);
  - *neutral:* no effect on play (cosmetic tile, platform off every route).
- **Expected outcome per class:** breaking mutants must be flagged; degrading mutants should be
  flagged, with a signal that says how play changed; neutral mutants must not be flagged.
- **Fixed or regenerated tests.** Replay (3.6) is the classic setting: the tests stay fixed and the
  subject changes. Retrain has no classic counterpart: the tests are regenerated for each mutant,
  so a flag means "no agent could solve it", not "the old solution broke".
- **What a mutant stands for.** A plausible designer mistake: a small edit that looked harmless.
  Whether a real change is a fault depends on design intent, which the engine does not know; it
  reports that play changed and where.
- **Operators on levels** (text edits):
  - seal a route (add solid tiles across a passage);
  - widen a gap (remove ground tiles);
  - move the goal;
  - add a hazard or an enemy on a path;
  - remove a platform, ladder, or spring.
- **Operators on mechanics** (YAML edits):
  - change jump velocity, gravity, or run speed by a fixed step;
  - change enemy speed;
  - shorten a time limit.
- **Neutral operators:** cosmetic tiles; a platform off every route; a coin moved.
- **[TODO: optional third family, code mutants: apply a standard Python mutation tool to the game
  code (physics, collision) and count how many the agent playthroughs kill. This is mutation
  testing in the classic sense and would speak directly to AST reviewers; costs extra runs.]**
- **[TODO: build the mutant generator (level or config in, one mutant per operator and location
  out).]**
- **[TODO: wording check. A game is also software, so if code mutants are added, call them
  "software mutants of the game code" to keep the two terms apart.]**

### 3.8 Self-tests

- 80 automated tests: determinism under a seed, sensor dimensions, adapter contracts, level
  loading, and reachability of each level's goal.

### 3.9 Running example (all numbers illustrative)

- Use one example through the paper: Mario 1-1, persona *good*, 10 seeds.
- **1. Design.** The level is a text file; mechanics are in the config (jump velocity, gravity).
- **2. Train.** 10 training runs, one per seed, each saves its best agent. Output: 10 agents plus
  training data (kept aside).
- **3. Play.** Each of the 10 agents plays the level once, with no learning (one play is enough:
  the agent and this level are deterministic; a game with random events needs several game
  seeds). One human also plays 10 attempts, since a person varies and learns between attempts.
  Output: 10 + 10 telemetry rows in the same format.
- **4. Measure.** Shown here with three metrics only; the full set is in 3.5 (completion, win
  rate, progress, death locations, cause of death, time, route, plus jumps, pickups, and kills).
  Agents: solved, win rate 10/10, median time 41 s, deaths none. Human: 8/10, two deaths at the
  second pit.
- **Change.** A mutant widens the second pit by two tiles (class: breaking for walkers, degrading
  for sprinters; decided before the run).
- **Play again (replay).** The same 10 agents on the mutant: 0/10, all die "Pit" at 34% progress.
- **Measure and compare.** Completion flips, win rate 100% → 0%, deaths cluster on the mutated
  tiles. Flag raised, located at the pit.
- **Confirm (retrain).** 10 new agents trained on the mutant: 7/10 win, by a sprint jump. Verdict:
  the level is still solvable, but the old way through is gone and it is harder.
- **Neutral change.** A decorative tile added above the path: replay gives identical telemetry;
  no flag.
- **Validation in one line each:**
  - RQ1: over all mutants, how many breaking and degrading ones were flagged, how many neutral
    ones were flagged by mistake, and how often the death cluster is on the mutated tiles.
  - RQ2: rerun the baseline with the same seeds and check the telemetry is identical; time it.
  - RQ3: lines changed to add Bomberman.
  - RQ4: do the human's deaths and failures fall where the agents' do, on the baseline and on a
    sample of mutants.

---

## 4. Evaluation

### 4.1 Research questions

- **RQ1 (fault detection):** how many injected faults in levels and mechanics do agent playthroughs
  detect, and how many neutral changes do they flag by mistake?
- **RQ2 (repeatability and cost):** are runs identical under a seed, and cheap enough to rerun after
  each change?
- **RQ3 (extensibility):** what does it take to add a game to the engine?
- **RQ4 (agreement with human play):** do agents and human players succeed and fail in the same
  places?

### 4.2 Common setup

- Subjects: 14 levels in three games (3.2): Mario World 1 (4), the first six levels of Super
  Meat Boy's first chapter (6), and Bomberman stages 1–4 (4).
- A level enters RQ1 only if the agents solve it at baseline; levels never solved are reported as
  a limitation and their mutants are not counted.
- Expected size: about 10 mutants and 3 neutral changes per level, so about 180 mutants.
- Expected cost: **[VERIFY: about 5 CPU-hours per level with 10 seeds and 3 personas in retrain
  mode, about 70 CPU-hours in total; time one level first.]**
- If Bomberman uses a seeded generator: 3 generated layouts per stage, 12 layouts, 22 levels in
  total.
- Agents: default configuration, fixed for the whole evaluation; personas per game as chosen in
  3.3.
- Seeds: **[TODO: 10–30 per level and persona instead of the current 3]**, following guidance for
  randomized algorithms [@arcuriPracticalGuideUsing2011].
- Telemetry from test runs only, same format for agents and humans; training data reported
  separately as cost.
- Hardware: one laptop, no GPU. **[TODO: state the model.]**

### 4.3 RQ1: fault detection by game mutation (primary)

- Baseline: levels the agents solve, run with fixed seeds.
- Generate mutants with the operators in 3.7, one change per mutant.
- Rerun each mutant with the same seeds and agents.
- A mutant is detected when the flagging rule fires on the signals in 3.6. **[TODO: fix the rule
  before running, e.g., solved → unsolved, or completion drop beyond the seed interval, or a shift
  in the cause-of-death mix.]**
- Report replay and retrain separately: detection rate, false alarms, and time for each.
- Localization: does the new death cluster fall on the mutated tiles? Report the share of detected
  level mutants where it does.
- Ground truth for each mutant: its class (breaking, degrading, neutral; 3.7), decided by a
  reachability check and by one author playing it, before looking at agent results.
- Report:
  - detection rate on breaking and on degrading mutants, per operator and per game;
  - false alarms on neutral mutants;
  - mutants missed, with the reason (agent cannot sense it, alternative route exists);
  - which persona detected which mutant, to show whether personas add detection power.
- Supporting cases: real defects the engine caught during development **[VERIFY: unreachable goal,
  unusable ladder path, sealed route in the Bomberman levels.]**
- Baseline to beat: a random-action agent and a static reachability check, to show what learned
  agents add. **[TODO: decide whether to include the static check.]**

### 4.4 RQ2: repeatability and cost

- Repeatability: run the same level and seed twice, compare the full episode records; report
  identical or not, per game. Zero flaky runs is the claim.
- Across machines: **[TODO: same check on a second machine or OS, if time allows.]**
- Throughput: environment steps per second, headless, per game (**[VERIFY: 5,000+]** in the README).
- Cost of one regression check: wall-clock time for a level over all seeds (**[VERIFY: about six
  CPU-minutes for two Mario levels × three seeds.]**)
- Cost of the full game-mutation experiment: total time for all mutants.
- Remaining variance: spread of completion across seeds, to justify the seed count.

### 4.5 RQ3: extensibility

- Case: adding Bomberman, a different genre with a different action space.
- Measure from the version history: files and lines added for the game versus lines changed in
  shared engine code.
- Preliminary count from `daa3dcc9~1..aa6eaa0a`, to be cleaned of unrelated sweep changes:

  | Part | Lines added | Lines removed |
  |---|---|---|
  | All code (18 files) | 1,573 | 56 |
  | New game core | 662 | — |
  | Adapter | 233 | — |
  | Tests | 166 | — |
  | Shared training loop | 33 | 11 |

- What had to change in the shared code and why (action space, sensing hook, stall rule).
- What came for free: dashboard, reports, recording, level editor, mutation operators on levels.
- **[TODO: calendar time is one day in the history; say so only if it is honest about prior design
  work.]**

### 4.6 RQ4: agreement with human play (secondary)

- Human baseline: about 10 per game. **[TODO: confirm whether this is 10 players or 10 recorded
  attempts, and who plays; authors only is a calibration, not a user study.]**
- Protocol: fixed level set, fixed number of attempts per level, one life per attempt, recorded with
  the engine:
  `python -m code.games.tools.manual_play --game mario --level Mario1-1 --record --player p1 --episodes 10`
- Compare humans and agents per level:
  - completion rate and progress reached;
  - cause-of-death mix;
  - death locations and routes overlaid on the level map;
  - rank order of levels by difficulty (rank correlation).
- Humans on a sample of mutants: do people fail on the mutants the agents flagged? **[TODO: only if
  time allows.]**
- Random-action baseline through the same recorder (`--random --record`), as a floor.
- Report disagreements as findings: levels humans clear and agents do not point to what the agent
  cannot sense or do.
- Figure: one level per game with human and agent deaths on the same map.
- Precedent for the design: human testers compared with agents on seeded bugs [@ariyurekAutomatedVideoGame2019].

### 4.7 Threats to validity

- Mutation operators are designed by the authors; real regressions may be subtler. For programs,
  mutants are known to correlate with real faults [@justAreMutantsValid2014]; no such evidence
  exists for game content.
- The games were written for the engine; external games are not evaluated. The requirements are
  stated independently of PEAK so they can be checked against other engines.
- Reactive agents give a lower bound on what a player can do; an unsolved level is not proof that
  it is unsolvable.
- Ground truth for mutants depends on one author's judgment plus a reachability check.
- Players who are authors know the levels.
- Progress is horizontal position in the platformers, which under-reports levels whose goal is not
  at the right edge.

### 4.8 Future work hinted here

- The same episode records, collected from many players, as telemetry for balance analysis
  (extended paper).
- The regression check in continuous integration, per commit.
- Human play as oversight of agent testers (fits the AST 2027 theme; one sentence, not a claim).

---

## 5. Lessons learned

- **What the agent can sense matters more than the learner.** Bomberman agents died to their own
  bombs in every episode until the sensor vector said which neighboring tile burns next; after
  that, **[VERIFY: first win at generation 4.]**
- **A missed mutant is usually a sensing gap.** **[TODO: confirm from RQ1.]**
- **Determinism has to be designed in.** One seed for level setup, game steps, and the learner;
  checkpoints store the random state. **[TODO: list the concrete sources of non-determinism we had
  to remove.]**
- **Dense progress beats sparse success.** Fitness that improves the moment the world improves
  (path cost to the goal) made levels learnable that distance alone did not.
- **Small learners keep the test cheap.** No GPU and minutes per check is what makes rerunning
  after each change realistic.
- **Give every failure a name.** Named causes of death turn a failed check into a diagnosis.

---

## 5b. Implications

- Placement: a short subsection of the discussion. The contribution list in 1.6 stays technical;
  this says who can use it and for what.

### For game developers

- **A regression check for content.** After editing a level or a mechanic, a verdict per level and
  persona (unchanged, changed but playable, broken) in minutes on a laptop, without a GPU. Within
  reach of small teams.
- **Actionable output.** The verdict says where players now die, which kind of player is affected,
  and what changed in time and route, not only a pass or fail.
- **Human playtests spent on what needs people.** Blockers are caught before the session, so
  testers evaluate feel and fun instead of reporting that level 4 cannot be finished.
- **A checklist for their own engine.** The requirements (determinism, headless runs, a stable
  agent interface, content as data, structured telemetry) apply to any engine; this is the part
  that transfers, not the tool.
- **A way to judge their own bots.** Game mutation answers "how good is my automated playtest?"
  for any agent, learned or scripted.
- Limits to state: 2D tile-based games on our own engine; not a plugin for a commercial engine.

### For testing educators

- **Testing concepts made visible.** A mutant is a moved platform a student can see; a killed
  mutant is an agent falling into the pit. The same holds for regression testing, the oracle
  problem (the previous version as oracle), flaky tests and determinism, and test selection
  (replay as triage).
- **Design for testability as an exercise.** Adding a game through the adapter makes students
  decide what a system must expose to be tested.
- **Small and open.** Python, runs offline on a laptop, a few thousand lines; fits a lab session.
- **Possible assignments:** write a mutation operator and measure its detection rate; design a
  level and predict which personas complete it; find a change the agents miss and explain why;
  compare recorded human play with agent telemetry.
- **Motivation.** Games are already used to teach programming and testing
  [@stahlbauerTestingScratchPrograms2019; @feldmeierPlayTestGamifiedTest2023].
- **[TODO: this is potential use, with no classroom evidence. Claim it as an implication only,
  unless the tool is used in a course before the camera-ready; if it is, say which and how.]**

---

## 6. Related work

- Library: Zotero collection "AST2027 PEAK - related work" (82 items, 8 subcollections), exported
  to `references.bib`. The threads below follow the subcollections. Not every item needs citing.

### 6.1 Game testing practice and bugs

- Testing is largely manual; automation is rare and hard [@politowskiSurveyVideoGame2021;
  @politowskiAutomatedVideoGame2022; @albaghajatiVideoGameAutomated2020].
- Bugs reach players and are patched after release [@lewisWhatWentWrong2010;
  @trueloveWellFixIt2021; @linStudyingUrgentUpdates2017; @zhangWhyAreWe2016].
- Development problems reported in postmortems [@politowskiDatasetVideoGame2020].
- Use: supports 1.1 (problem and consequences of late detection).

### 6.2 Agents for game testing

- Learned agents that explore and find bugs [@bergdahlAugmentingAutomatedGame2020;
  @zhengWujiAutomaticOnline2019; @gordilloImprovingPlaytestingCoverage2021a;
  @sestiniAutomatedGameplayTesting2024; @luGoExploreComplex3D2024;
  @liuInspectorPixelbasedAutomated2022; @tufanoUsingReinforcementLearning2022].
- Synthetic and human-like testers on seeded bugs [@ariyurekAutomatedVideoGame2019].
- Agent frameworks and reusable tools [@prasetyaAgentbasedApproachAutomated2022;
  @chenMIMICPyExtensibleTool2026; @chenMIMICIntegratingDiverse2025;
  @wangLeveragingLLMAgents2025].
- Cost of deploying agents in production games [@gillbergTechnicalChallengesDeploying2023].
- Benchmark of game bugs [@liGBGalleryBenchmarkFramework2022].
- Difference: these study the agent or attach it to an existing game; we design the game side and
  measure detection with game mutation.

### 6.3 Regression testing of games

- Industrial regression testing: test selection for a commercial game
  [@yuGameRTSRegressionTesting2023]; regression testing of online role-playing games
  [@wuRegressionTestingMassively2020].
- Learned agents as regression tests: temporal-logic specifications
  [@gutierrez-sanchezReinforcementLearningTemporal2023;
  @gutierrez-sanchezProgressBasedAlgorithmInterpretable2024]; behavior specifications on a
  platformer clone [@mastainBehaviordrivenDevelopmentReinforcement2024;
  @mastainEnhancingAutomatedVideo2026] (prior workshop work; cite in the third person);
  code-change-guided agents [@muSynergizingCodeCoverage2025].
- Difference: they select or specify tests for a given game; we ask what the engine must provide,
  and add determinism, several games, and a detection measure.
- **Consequence for 1.1:** regression testing of games exists in large studios. Reword "game teams
  mostly cannot" to say it is rare, costly, and tied to one game.

### 6.4 Neuroevolution and search-based game testing

- **Closest work.** Neural networks evolved to play Scratch games as test cases, robust to
  randomness, with program mutants used to evaluate them
  [@feldmeierNeuroevolutionBasedGenerationTests2022; @feldmeierManyObjectiveNeuroevolutionTesting2025a;
  @feldmeierCombiningNeuroevolutionSearch2024]; human gameplay traces used to speed up the search
  [@feldmeierLearningViewingGenerating2023]; the underlying test framework
  [@stahlbauerTestingScratchPrograms2019].
- Difference, to confirm by reading the full papers: they generate tests for the *code* of a game
  (coverage-driven, white-box) and mutate the program; we test the *content* (levels and
  mechanics), mutate the content, and design the engine for it.
- Evolution as a seedable alternative to gradient-based learning
  [@suchDeepNeuroevolutionGenetic2017].
- **[TODO: read the 2022 paper in full before writing 1.2 and 3.7; a reviewer from this group is
  likely at AST.]**

### 6.5 Platforms and engines for agents

- Engines and toolkits that expose games to learning agents [@julianiUnityGeneralPlatform2020;
  @beechingGodotReinforcementLearning2021; @gomesAI4UToolGame2020a;
  @jayaramireddySurveyReinforcementLearning2023].
- Fixed benchmarks [@bellemareArcadeLearningEnvironment2013;
  @torradoDeepReinforcementLearning2018].
- Difference: built for training agents, not for testing the game; determinism and comparable
  records across game versions are not their goal.

### 6.6 Testing foundations

- Mutation testing [@jiaAnalysisSurveyDevelopment2011; @papadakisMutationTestingAdvances2019];
  mutants as stand-ins for real faults [@justAreMutantsValid2014].
- Flaky tests [@luoEmpiricalAnalysisFlaky2014; @parrySurveyFlakyTests2021]; variance of learned
  agents [@hendersonDeepReinforcementLearning2018].
- Regression testing [@yooRegressionTestingMinimization2012].
- Test oracles [@barrOracleProblemSoftware2015]; ours is differential, the previous version of the
  same game.
- Testability [@voasSoftwareTestabilityNew1995; @binderDesignTestabilityObjectoriented1994]:
  grounding for the requirements in Section 2 (controllability, observability).
- Statistics for randomized algorithms [@arcuriPracticalGuideUsing2011]: grounding for the seed
  count and the flagging rule.

### 6.7 Simulation-based testing in other domains

- Scenario-based testing of driving systems in simulators [@zhongSurveyScenarioBasedTesting2021];
  recent AST papers on robotic and driving systems [@krishnanACTAutomatedCPS2026;
  @liUnifiedBenchmarkOutofDistribution2026].
- Use: the closest software-testing analogue. There the simulator is the tool and the controller
  is under test; here the game world is under test and the agent is the tool.

### 6.8 Humans, personas, and telemetry

- Personas and play styles as automated playtesters [@holmgardAutomatedPlaytestingProcedural2019;
  @ariyurekPlaytestingWhatPersonas2023; @guerrero-romeroMAPElitesGenerateTeam2021;
  @zhangGeneratingGameLevels2022].
- Human-like agents and comparison with human players [@zhaoWinningNotEverything2020;
  @stahlkeArtificialPlayfulnessTool2019; @novaChartingUnchartedGUR2022;
  @goodmanCaseStudyAiassisted2023].
- Agents that check whether a level can be played [@shakerEvolvingPlayableContent2013]; a level
  editor with built-in synthetic testers [@suetakeInteractiveDesignExploration2020].
- Gameplay telemetry and its visualization [@wallnerVisualizationbasedAnalysisGameplay2013;
  @hullettDataAnalyticsGame2011].
- Our earlier work on balance with agents [@politowskiAssessingVideoGame2023] (cite in the third
  person).
- Difference: these use agents to evaluate design; we use the same kind of data as a regression
  oracle and measure how well it detects changes.

---

## 7. Conclusion

- Regression testing for games needs an engine built for it: deterministic, headless, with a
  stable agent interface, content as data, and structured records.
- PEAK shows these fit in one small open-source engine with three games in two genres evaluated
  (two more are archived in the code).
- Game mutation gives agent playthroughs a fault-detection measure. **[TODO: fill the numbers
  after the runs.]**
- Next: the same records as player telemetry for balance analysis (extended paper); the check in
  continuous integration; applying the requirements to an external engine.

---

## Reviewer objections and answers

- **"Toy games written for your own engine."** Clones of the opening levels of three well-known games, so readers can judge the levels
  themselves; measured cost of adding
  a game; requirements stated independently of the tool.
- **"No real bugs."** Systematic mutants with ground truth, plus the real defects caught during
  development.
- **"Agents are not players."** Human calibration in RQ4; agents are claimed as a lower bound.
- **"Learned agents are flaky."** Identical reruns under a seed (RQ2).
- **"This is an AI paper."** The learner is a default, not a contribution; no algorithm comparison.

## Work plan to 30 October

1. Test run (3.4), comparison command (3.6), and mutant generator (3.7), then the RQ1 runs.
2. Determinism and cost numbers (RQ2).
3. Human baseline recordings (RQ4).
4. Finish the level sets (Mario 1-3 and 1-4; Bomberman NES stages; check Super Meat Boy against
   the original) . Sonic and Megaman only if time allows.
5. Clean the Bomberman cost count (RQ3).
6. Read the closest papers in full (6.4 first, then 6.3) and fix the difference statements.

## Out of scope (extended paper)

- Balance metrics and intent bands, difficulty ranking across personas, GA and sensor ablations.

## Open questions

- **DRL:** the main branch is neuroevolution only; PPO lives on the archive branch. Present both as
  backends, or neuroevolution with DRL as the prior version?
- **Wording:** "engine" vs. "framework" vs. "testbed".
- **Flagging rule** for RQ1, fixed before the runs.
- **Super Meat Boy chapter:** The Forest (in the repository) or Chipper Grove (the link).
- **Bomberman:** seeded generator or fixed layouts.
- **Human baseline:** who plays, and how many attempts.
- **Name:** the README expands PEAK with author names and the repository is public; rename or
  anonymize for double-blind review.

## References

- `references.bib` is exported from the Zotero collection "AST2027 PEAK - related work" with Better
  BibTeX (abstracts and local file paths removed). Re-export after adding items; do not edit by hand.
