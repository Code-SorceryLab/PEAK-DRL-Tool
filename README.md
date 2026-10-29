<p align="center">
  <img src="docs/img/PEAK_LOGO.png" alt="PEAK: Platformer Engine by Al and Kevin" width="520">
</p>

<h3 align="center">Small evolved neural networks that playtest 2D games, and a test of whether a playtest notices a one-tile change</h3>

**PEAK** is a research tool from Ontario Tech University (Master's program). It plays 2D games with
small neural networks that learn by evolution (a genetic algorithm, no gradients, no GPU). It then
reports how hard each level is, for six kinds of player. It also makes small, controlled changes to a
level ("mutants") and checks whether a playtest notices the change. People can play the same levels
and mutants, so human and agent play can be compared.

This README explains the whole system in simple words, and gives every setting, every experiment,
every result we have now, and where each data file is. It is written so that the paper can be written
from it.

---

## Contents

1. [The short version](#1-the-short-version)
2. [The four games and their levels](#2-the-four-games-and-their-levels)
3. [The agent: what it sees, what it does, how it learns](#3-the-agent-what-it-sees-what-it-does-how-it-learns)
4. [The six personas](#4-the-six-personas)
5. [Words used in this README](#5-words-used-in-this-readme)
6. [The experiments (the Nibi run of 2026-10-28)](#6-the-experiments-the-nibi-run-of-2026-10-28)
7. [Mutation testing (AST): how it works](#7-mutation-testing-ast-how-it-works)
8. [The pilot user study](#8-the-pilot-user-study)
9. [Results](#9-results)
10. [Known limits and caveats](#10-known-limits-and-caveats)
11. [Where the data is](#11-where-the-data-is)
12. [How to run it and reproduce the results](#12-how-to-run-it-and-reproduce-the-results)
13. [Status on 2026-10-29](#13-status-on-2026-10-29)
14. [Authors](#14-authors)

---

## 1. The short version

- **Games.** Four hand-written games in Python and Pygame: Mario (Super Mario Bros. 1-1 to 1-4),
  Meat Boy (Super Meat Boy world 1), Bomberman (a ladder of levels plus Bomberman II 1-1 to 1-4) and
  Mega Man (the Cut Man stage, in 8 sections). All four are deterministic: the same seed gives the
  same game, frame by frame.
- **Agents.** Each agent is a tiny neural network (for Mario: 14 inputs, 16 hidden units, 3 outputs,
  291 weights). A genetic algorithm (GA) evolves a population of 10 networks. There is no gradient
  training and no GPU.
- **Personas.** Six player types: novice, clumsy, experienced, speedrunner, killer and collector.
  They differ in their "hands" (reaction time, sprint, mistakes) and in what they are paid for
  (the goal, enemies, coins, time).
- **Balance measurement.** For every level, persona and seed, a fresh population trains for up to 200
  generations. We record whether it ever wins (solved), how soon (first-win generation) and how often
  it wins after that (win rate).
- **GA settings.** A GA sweep changed one GA setting at a time and picked the best settings for each
  game (`code/neuro/ga_best.yaml`). Training with them is called `--best`.
- **Sensor ablation.** The normal sensor (a few raycasts, 14 to 26 numbers) was compared with a full
  tile grid (368 numbers). The raycasts win clearly.
- **Mutation testing (AST).** Each level gets five one-change copies: a sealed route, a harder gap
  or enemy, a weaker jump or shorter bomb fuse, a tile nobody can reach, and a cosmetic change. Agents
  are retrained on each copy, and the original agents replay each copy, to see whether play changes.
- **Pilot user study.** Five people played one game each: the evaluation level 3 times, then its
  five mutants once each in a random order.
- **Where we are.** All training and the GA sweep are done. The retraining on mutants is done for
  Mario, Meat Boy and Bomberman and is finishing for Mega Man. The playtests on mutants (`ast_play`)
  and the final detection metrics (`final`) have not run yet (see [section 13](#13-status-on-2026-10-29)).

---

## 2. The four games and their levels

Every game reads its levels from ASCII text files in `code/games/levels/<game>/`. The **evaluation
level** is the one level of each game that the mutation testing and the user study use.

| Game | Levels (as the code names them) | Evaluation level | Network inputs (rays mode) | Outputs | Weights |
|---|---|---|---|---|---|
| Mario | `Mario1-1`, `Mario1-2`, `Mario1-3`, `Mario1-4` (4 levels) | `Mario1-1` (Super Mario Bros. 1-1) | 14 | 3: left, right, jump | 291 |
| Meat Boy | `0` to `10` (11 levels, files `world1_01_hello_world.txt` to `world1_11_fired.txt`) | `2` (Super Meat Boy 1-3 "The Gap", file `world1_03_the_gap.txt`) | 21 | 3: left, right, jump | 403 |
| Bomberman | `0` to `15` (16 levels: `0` to `11` are files `01_open_floor.txt` to `12_chasers.txt`; `12` to `15` are Bomberman II 1-1 to 1-4, files `bm2_1_1.txt` to `bm2_1_4.txt`) | `12` (Bomberman II 1-1, file `bm2_1_1.txt`) | 16 | 5: left, right, up, down, bomb | 357 |
| Mega Man | `CutMan-1-Street`, `-2-Climb`, `-3-Ledges`, `-4-Shaft`, `-5-Rooftop`, `-6-Descent`, `-7-Corridor`, `-8-Boss` (8 sections of one stage) | `CutMan` (the whole Cut Man stage) | 26 | 6: left, right, jump, up, down, fire | 534 |

Level indices start at 0 in Meat Boy and Bomberman ("level 2" is the third file).

**Mega Man is trained one section at a time.** Each of the 8 sections is a separate level for
training. The whole stage is played by a **relay**: the agent of section 1 plays section 1, then
the agent of section 2 takes over, and so on (`code/neuro/relay.py`).

The level files, the game rules and the game settings:

| What | Where |
|---|---|
| Level files | `code/games/levels/<game>/*.txt`, glyphs in `code/games/levels/common/ASCII_TILEMAP.md` |
| Level lists and game settings | `code/games/game_config.yaml` (Mario), `code/games/meatboy_config.yaml`, `code/games/bomberman_config.yaml`, `code/games/megaman_config.yaml` and `code/games/megaman_levels.yaml` |
| Game engines | `code/games/*_core.py` |
| Mutant levels | `code/games/levels/mutants/<game>/<mutant id>/` |

---

## 3. The agent: what it sees, what it does, how it learns

### 3.1 The network

Each agent is a fixed-shape network: inputs, one hidden layer of 16 units with `tanh`, and outputs
(`code/neuro/net.py`). The weights are one flat list of numbers. Evolution changes these numbers;
nothing else is learned. The number of weights is `inputs x 16 + 16 + 16 x outputs + outputs`
(14 inputs and 3 outputs give 291).

### 3.2 What the agent sees (sensors)

There are two sensor modes (`code/neuro/sensors.py`, `--sensors`):

- **rays** (the default, used in every main result): a few raycasts plus a few body values. In
  Mario: six raycasts (forward, forward-up and forward-down at 30 and 60 degrees, back), an enemy
  corridor, a pit probe, the velocity, "on the ground" and "can jump", and nearby question blocks:
  14 numbers. Meat Boy (21), Bomberman (16) and Mega Man (26) have their own sensor lists; each slot
  has a name in the adapter's `SENSOR_LABELS` (`code/neuro/adapters.py`).
- **grid**: an 11 x 11 tile window around the agent with 3 channels (solid, collectible, hazard),
  plus 5 body values: 368 numbers. Used only in the sensor ablation.

### 3.3 What the agent does

Every frame the network reads its sensors and picks its buttons: left, right and jump in Mario and
Meat Boy; plus up, down and bomb in Bomberman; plus fire in Mega Man.

### 3.4 Fitness (what evolution rewards)

Fitness is the furthest progress toward the exit, plus 5000 for a win, plus the persona's extras
(section 4). Progress is the furthest x position in Mario; the path progress toward the exit in
Meat Boy; the Dijkstra cost to the exit on a 0 to 1000 scale in Bomberman (a brick on the way costs
6, so a useful bomb pays at once). The code is the `fitness()` method of each adapter in
`code/neuro/adapters.py`.

### 3.5 How one training run works

A **population** of 10 networks starts with random weights. In each **generation**, every network
plays the level once (one episode). An episode ends at a win, a death, the frame budget (3600 frames,
60 seconds of game time) or a stall (300 frames with no new progress). Then the GA makes the next
population:

1. **Elitism**: the 4 best networks are copied unchanged.
2. **Tournament selection**: each parent is the best of 5 networks picked at random.
3. **Uniform crossover** (probability 0.7): each weight comes from one of two parents.
4. **Gaussian mutation**: each weight changes with probability 0.15 by a normal step with sigma 0.15.
5. **Annealing**: after the first win on a level, the mutation rate and sigma are multiplied by the
   anneal factor (0.5 by default), so a winning population changes more slowly.

One seed sets the random start, every GA choice and every game step, so a run is fully repeatable.

### 3.6 The default GA settings

From `GAConfig` in `code/neuro/evolution.py`:

| Setting | Value | Setting | Value |
|---|---|---|---|
| Population size | 10 | Initial weight sigma | 0.5 |
| Elite (copied unchanged) | 4 | Anneal factor after the first win | 0.5 |
| Tournament size k | 5 | Frame budget per episode | 3600 (60 s) |
| Crossover rate | 0.7 | Stall limit | 300 frames with no new progress |
| Mutation rate | 0.15 | Win bonus | 5000 |
| Mutation sigma | 0.15 | Hidden units | 16 (`tanh`), no memory units, no action feedback |

### 3.7 The `--best` GA settings (from the GA sweep)

The GA sweep (section 6.2) wrote these per-game settings to `code/neuro/ga_best.yaml` (generated
2026-10-28 08:07; a copy is `runs/ga_best_2026-10-28.yaml`). `--best` uses them; every setting not
named stays at the default.

| Game | `--best` changes |
|---|---|
| Mario | anneal factor 0.3 |
| Meat Boy | anneal factor 0.3 |
| Bomberman | population 30, tournament k 8, anneal factor 0.3 |
| Mega Man | population 30, elite 1, mutation rate 0.5, anneal factor 0.3 |

Important: for Mario and Meat Boy, `--best` changes only the anneal factor, which acts only after
the first win. So `--best` cannot change whether or when a level is first solved in those two games;
it can only change the win rate after that.

---

## 4. The six personas

A persona is the kind of player the agents imitate (`code/neuro/personas.py`). All personas use the
same network. They differ in two ways: their **hands** (how often they get fresh senses, whether they
can sprint, whether they make mistakes) and **what they are paid for**.

| Persona | Sprint | Reacts every | Mistakes | Paid for |
|---|---|---|---|---|
| `novice` | no | 3rd frame | none | the goal |
| `clumsy` | no | 4th frame | a 0.2 s (12-frame) mis-press with probability 0.03 per frame (about every 0.5 s) | the goal |
| `experienced` | yes | every frame | none | the goal |
| `speedrunner` | yes | every frame | none | the goal + 25 per second left on a win |
| `killer` | no | every frame | none | the goal + enemies killed |
| `collector` | no | every frame | none | the goal + coins and power-ups picked up |

**Extras are shares of the goal.** For the killer, killing every enemy in the level pays as much as
the goal; killing 3 of 21 pays 3/21 of it. The share is capped at 1, so kills cannot be farmed.
A level with no enemies (or no coins) gives a plain player: the killer and the collector then play
for the goal alone. Clumsy's mistakes are seeded, so a replay makes the same mistakes.

---

## 5. Words used in this README

| Word | Meaning |
|---|---|
| **seed** | The number that fixes all randomness of one run. The paper seeds are **1234, 2025 and 31337**. |
| **cell** | One training run: one game, level, persona, GA setting and seed. |
| **generation** | One round in which each of the 10 networks plays the level once. |
| **solved** | The cell won at least once, in any generation. |
| **first-win generation** | The generation of the first win. |
| **win rate** | The share of episodes won in the **10 generations after the first win** (100 episodes with a population of 10). A cell that never won has a win rate of 0. |
| **budget** | The most generations a cell may run (200 in the main training, 40 in the sweeps). A cell stops 10 generations after its first win (`WIN_WINDOW = 10` in `code/neuro/balance.py`), so a solved cell usually runs fewer generations than the budget. |
| **probe** | One cell's run folder: the best network (`best.npz`), the run state (`state.json`) and the episode log (`episodes.csv`). |
| **report** | One JSON file that sums up the cells of one game, persona and setting (`runs/balance/report_*.json`). |
| **tag** | The name of a setting in file names: `p10g200` = population 10, 200 generations; `p30g200_best` = population 30, 200 generations, `--best`. |
| **mutant** | A copy of a level (or of a game setting) with exactly one change. |
| **operator** | The kind of change a mutant makes (for example `seal_route`). |
| **variant** | What the AST plays: `base` (the original), a mutant id, `base-reseeded` or `base-retrained`. |
| **evaluation level** | The one level per game that the mutation testing and the user study use (section 2). |

---

## 6. The experiments (the Nibi run of 2026-10-28)

All results in this README come from one pipeline run on the **Nibi** cluster (Digital Research
Alliance of Canada), started 2026-10-28 at 04:05 and resumed 2026-10-29 at 12:37. Every task used
3 CPU cores. The code was commit `79fc29691` (branch `USER-STUDY+SPRITIFY`). The pipeline script
runs these stages in order; each stage waits for the one before it.

| # | Stage | What it does | Tasks |
|---|---|---|---|
| 1 | `test` | A 5-generation smoke test per game | 4 |
| 2 | `mutants` | Writes every mutant of every level (`eval_mutants --all-levels`, `megaman_mutants`) | 1 |
| 3 | `sweep2d` | For Mario, Meat Boy, Bomberman and each persona: the 40-generation balance sweep, the sensor ablation and the GA sweep (`run_all.sh sweeps`) | 18 |
| 4 | `sweep_mm` | The Mega Man GA sweep, one section at a time (6 personas x 8 sections) | 48 |
| 5 | `gabest` | Writes `ga_best.yaml` from all sweeps (`gasweep --rebuild`) | 1 |
| 6 | `confirm_mm` | Confirms the Mega Man winners over all 8 sections | 6 |
| 7 | `gabest` | Writes `ga_best.yaml` again | 1 |
| 8 | `full` | **The main training: 200 generations**, every persona and level, default GA and `--best` | 372 |
| 9 | `reports` | Rebuilds every report from the probe folders (`balance --rebuild`) | 1 |
| 10 | `ast_retrain` | Retrains every persona's 3 seeds on every mutant (200 generations) | 1038 |
| 11 | `ast_play` | Playtests, replays and random baselines on the original and every mutant | 511 |
| 12 | `final` | Computes the AST metrics and verdicts (`runs/ast/metrics.json`) | 1 |

### 6.1 Main training (stage `full`)

- **What:** every level of Mario (4), Meat Boy (11) and Bomberman (16), with each of the 6
  personas, with the default GA and with `--best`, on seeds 1234, 2025 and 31337.
- **Budget:** 200 generations per cell, stopped 10 generations after the first win.
- **Size:** 2 settings x 6 personas x 31 levels x 3 seeds = 1116 cells.
- **Mega Man** has no 200-generation run in this stage. Its 200-generation agents come from the AST
  stage (`runs/ast/megaman/base-retrained`, section by section, with `--best`).
- **Output:** `runs/balance/report_<game>_<persona>_<tag>.json` and the probe folders in
  `runs/probes/<game>/<persona>/<tag>/<level>_<seed>/`.

### 6.2 GA sweep (stages `sweep2d`, `sweep_mm`, `gabest`, `confirm_mm`)

- **Question:** which GA setting helps, per game and persona?
- **Method:** one setting at a time is moved to a low or a high value from the literature (De Jong,
  Grefenstette, Miller and Goldberg, Such et al.), and everything else stays at the default. The
  run is compared with the default run on the same seeds and levels. 22 variants per sweep:

  | Setting | Values tried (default in brackets) |
  |---|---|
  | population size | 30, 100 (10) |
  | elite | 1, 6 (4) |
  | tournament k | 2, 8 (5) |
  | crossover rate | 0.0, 0.95 (0.7) |
  | mutation rate | 0.005, 0.5 (0.15) |
  | mutation sigma | 0.02, 0.5 (0.15) |
  | initial sigma | 0.25, 1.0 (0.5) |
  | anneal factor | 0.3, 1.0 (0.5) |
  | hidden units | 8, 32, 64 (16) |
  | memory units | 2, 3 (0) |
  | action feedback | on (off) |

- **Budget:** 40 generations, seeds 1234, 2025, 31337, all levels of the game (Mega Man: each
  section on its own).
- **Choice:** the winning values of one sweep are combined and probed again (`--confirm`). A
  setting goes into a game's `--best` only if it wins a **strict majority** of that game's sweeps
  (one sweep per persona), so one lucky persona cannot move it.
- **Output:** `runs/gasweep/<game>/<persona>/<tag>/...`, the reports
  `runs/balance/gasweep_*.json` and `code/neuro/ga_best.yaml`.

### 6.3 Sensor ablation (in stage `sweep2d`)

- **Question:** does the agent need to see more?
- **Method:** the same GA, seeds and levels, with `rays` sensors and with the `grid` sensor.
- **Budget:** 40 generations, all personas, Mario, Meat Boy and Bomberman.
- **Output:** `runs/ablation/...` and `runs/balance/ablation_*.json`.

### 6.4 Mutation testing (stages `mutants`, `ast_retrain`, `ast_play`, `final`)

Section 7 explains it in full.

---

## 7. Mutation testing (AST): how it works

The question: **does a playtest notice one small change to a level?** Code:
`code/games/eval_mutants.py`, `code/games/megaman_mutants.py`, `code/neuro/ast_eval.py`,
`code/stats/ast_metrics.py`. The detection rule is written down in `paper-ast/ast_rule.md`.

### 7.1 The five operators

Each level gets five mutants. Each mutant makes exactly one change. Its class is set before any
agent plays it.

| Operator | What it changes | Class | Agents |
|---|---|---|---|
| `seal_route` | A wall across the route: the exit can no longer be reached | breaking | retrained |
| `widen_gap` (Bomberman: `add_enemy`) | A gap 2 tiles wider (Bomberman: one more enemy) | degrading | retrained |
| `jump_velocity_down` (Bomberman: `fuse_down`) | Jump velocity x0.85 (Bomberman: bomb fuse x0.625) | degrading | retrained |
| `off_route` | One tile changed where no player can reach or sense it | neutral | retrained |
| `cosmetic_tile` | A `.` written as a space: the level parses the same | equivalent | replayed only |

### 7.2 Where the mutants are

- **On the evaluation level** the changes are placed by hand (`CHANGES` in
  `code/games/eval_mutants.py`). Examples on Mario 1-1: a wall down column 190 before the flagpole;
  the second pit widened from 3 to 5 tiles; take-off jump velocity -495 to -420.75 px/s.
  On Bomberman II 1-1: hard walls at row 5, columns 9 and 11 close the exit; a fourth Ballom at
  row 6, column 7; the fuse 150 to 94 frames.
- **On every other level** the same five operators are placed **by rule** (`place()` in
  `eval_mutants.py`): the wall nearest the goal that cuts it off; the gap nearest the start widened by
  2; one more enemy on the floor tile nearest the level's centre; the solid tile nearest the start
  whose four neighbours are solid, hollowed out; the first `.` written as a space. These mutants are
  named `<operator>@<level>` (for example `add_enemy@2`). Where no rule fits, the operator is skipped
  and the reason is recorded.
- **Mega Man** uses five mutants: `s3-seal_route`, `s3-widen_gap`, `s3-off_route`,
  `s3-cosmetic_tile` (all in section 3, Ledges) and `m-jump_velocity_down` (the jump value of the
  whole game). `megaman_mutants.py` also writes 59 more mutants for other sections and settings; the
  evaluation uses the five above.
- **Checks before writing:** each mutant is loaded in the game. The change must play; a cosmetic copy
  must parse to the same level; an off-route tile must be walled in on all four sides; a flood fill
  must find the exit for every mutant except `seal_route`.
- Mutants on disk now: Mario 23, Meat Boy 58, Bomberman 96, Mega Man 64
  (`code/games/levels/mutants/<game>/`).

### 7.3 Retraining (stage `ast_retrain`)

For every mutant except the cosmetic one, each persona's 3 seeds are trained from scratch on the
mutant, exactly as the original was trained (200 generations, the same GA setting that won more on
that level; Mega Man always `--best`, section by section). Two special variants are trained on the
**unchanged** level:

- `base-reseeded`: the original with **three new seeds**. It measures how much retraining alone
  changes the results (the false-alarm check).
- `base-retrained` (Mega Man only): the original with every section retrained exactly as the
  mutants were. Mega Man's mutants are compared with it.

**Retraining answers one question only: can the mutant still be finished?** A breaking mutant
should no longer be solved; every other mutant still should.

### 7.4 Playtests (stage `ast_play`)

A playtest gives **each agent one attempt with learning off**: the best network of each of the 3
seeds of each persona, so a persona's win rate moves in steps of 1/3. For one level this is
6 personas x 3 seeds = 18 attempts. The games and agents are deterministic, so one attempt is the
answer. Mega Man plays its relay of section agents, 20 attempts per scope, for each persona. The
scopes are the full stage and each section for `base-retrained` and `m-jump_velocity_down`, and
section 3 alone for the `s3-` mutants (most full-stage attempts end before section 3).

Three kinds of playtest are written to `runs/ast/<game>/<variant>/`:

| File | Who plays |
|---|---|
| `agents.csv` | The agents trained on that variant |
| `agents_replay.csv` | **The original's frozen agents** on the mutant (`play --replay`) |
| `agents_random.csv` | A random-action player, as many attempts as the agents (2D games only) |

### 7.5 The detection rule (stage `final`)

**Replay is the detector.** The original's frozen agents play the mutant and are compared with their
own play on the original. Because they are frozen and deterministic, any difference comes from the
change, not from seed noise. A mutant is **flagged** when any of these holds:

| Signal | Test |
|---|---|
| completion lost | the original's agents win at least once; on the mutant they never win |
| win rate | Fisher's exact test on wins vs losses, two-sided, p < 0.05 |
| death cluster moved | the cluster of failed end tiles (Chebyshev radius 3) moves more than 3 tiles, or appears |
| new top cause | the most common cause of failure changes |

Each comparison gets one **verdict**: **broken** (completion lost), **changed** (any other signal),
**unchanged** (no signal), or **not counted** (the original's agents never win on this level, so
nothing can be lost). A breaking or degrading mutant is **caught** when flagged; a neutral one is
**right** when unchanged. The cosmetic copy must match the original attempt for attempt.

Reported beside the rule: the mutant's win rate against the spread of the original's per-seed win
rates; a strict count (completion lost or Fisher only); and the false-alarm rate of the rule on
`base-reseeded`. The rule was fixed on 2026-10-25, after pilot replays had been seen, so it is **not
pre-registered** (`paper-ast/ast_rule.md`).

### 7.6 The metrics

`code/stats/ast_metrics.py` reads the attempts of agents and people and reports: completion, win
rate, progress, end location, death cluster, cause mix, time to finish, play coverage, effort counts
(jumps, coins, kills) and, for Mega Man, the section reached. `python -m code.stats.ast_paper` writes
the paper's tables and figures to `paper-ast/`.

---

## 8. The pilot user study

The study runs from the menu (`python menu.py`, option `u`) in one pop-up window, so the participant
sees only the games (`study_session` in `menu.py`). For each participant:

1. Type the participant ID and pick a game.
2. **Warm-up:** 5 minutes of normal play of the game. It is **not recorded**.
3. **Baseline:** the evaluation level, **3 runs**, each with the game's usual **3 lives**.
4. **Mutation:** the evaluation level's **5 mutants, once each, in a random order**. The
   participant is not told which mutant is which.

Each run is stored as `runs/user_study/<participant>/<game>/<phase>/run<k>/episodes.csv` (one row per
life) and `inputs.csv` (the button presses). Mutation rows also store `mutant` and `mutant_class`.
A run quit before its last life is kept and labelled `quit`, and it can be played again.

Data so far (pilot, committed by Kevin on 2026-10-29):

| Participant | Game | Baseline runs | Mutation runs |
|---|---|---|---|
| p1 | Mario | 3 | 4 |
| p2 | Mega Man | 3 | 5 |
| p3 | Mario | 3 | 5 |
| p4 | Mega Man | 3 | 4 |
| p5 | Mario | 3 | 5 |
| Test1, test111 | Meat Boy | 3 each | 5 each (test sessions, not participants) |

---

## 9. Results

All numbers come from the files in `runs/` on this branch (the Nibi run). Seeds 1234, 2025, 31337.

### 9.1 Main training, 200 generations (stage `full`)

| Game | GA | Cells solved | Levels solved by any seed | Levels solved by all 18 (6 personas x 3 seeds) | Mean win rate | Median first-win generation | Evaluation level solved |
|---|---|---|---|---|---|---|---|
| Mario | default | 44 / 72 (61%) | 4 / 4 | 1 | 22.7% | 10 | 18 / 18 |
| Mario | `--best` | 44 / 72 (61%) | 4 / 4 | 1 | 27.4% | 10 | 18 / 18 |
| Meat Boy | default | 92 / 198 (46%) | 11 / 11 | 1 | 20.6% | 34 | 18 / 18 |
| Meat Boy | `--best` | 92 / 198 (46%) | 11 / 11 | 1 | 25.2% | 34 | 18 / 18 |
| Bomberman | default | 103 / 288 (36%) | 10 / 16 | 2 | 16.6% | 21 | **0 / 18** |
| Bomberman | `--best` | 133 / 288 (46%) | 12 / 16 | 3 | 21.9% | 12 | **0 / 18** |

"Mean win rate" counts a never-solved cell as 0. "Median first-win generation" is over solved cells.

<p align="center">
  <img src="docs/img/results_per_level.png" alt="Per level and persona: seeds that solved the level (of 3) and mean win rate, default GA and --best, for all four games" width="900">
</p>
<p align="center"><sub>Every level, persona and GA setting: how many of the 3 seeds solved it, and the mean win rate. Mega Man here is the 40-generation GA-sweep baseline, per section.</sub></p>

<p align="center">
  <img src="docs/img/results_per_persona.png" alt="Per game and persona: mean win rate, levels solved by any seed and by all three seeds" width="720">
</p>

What the training shows:

- **Mario.** 1-1 is solved by every persona and seed (18/18). 1-3 is the wall: only 2 of 18 cells
  (experienced and speedrunner, 1 seed each). 1-2 is 14/18 and 1-4 is 10/18.
- **Meat Boy.** Level 2 (The Gap, the evaluation level) is 18/18 and level 0 is 16/18. Levels 9 and
  10 are the hardest (2/18 each), then level 3 (3/18).
- **Bomberman.** Levels 0 and 4 are 18/18. With the default GA, levels 8, 10, 12, 13, 14 and 15 are
  never solved. `--best` (population 30) solves more cells (133 vs 103), first wins earlier (median
  12 vs 21) and also solves levels 8 (2/18) and 10 (5/18). **The four Bomberman II levels (12 to 15),
  the evaluation level 12 included, are never solved with either setting.**
- **Mega Man** (40-generation GA-sweep baseline, each section on its own): only 5 Rooftop and
  7 Corridor are solved (18/18 each); the other 6 sections are 0/18 at 40 generations.
- **Personas.** Clumsy is the weakest persona in every game (mean win rate 1% to 9%). In Meat Boy,
  experienced and speedrunner give exactly the same results, and so do killer and collector. Meat Boy
  has no enemies or coins, which explains killer = collector; why the speedrunner's time bonus makes
  no difference there has not been checked.
- **`--best`** raises the mean win rate in all three games (Mario +4.7, Meat Boy +4.6, Bomberman
  +5.3 points). In Mario and Meat Boy it cannot change the first wins (section 3.7), and it did not.

### 9.2 GA sweep, 40 generations

Win rate of the default GA (baseline) and of the confirmed combination of the sweep's winners, per
game and persona (from `code/neuro/ga_best.yaml`):

| Game | Persona | Baseline | Confirmed winners | Game | Persona | Baseline | Confirmed winners |
|---|---|---|---|---|---|---|---|
| Mario | novice | 30.4% | 35.3% | Bomberman | novice | 14.7% | 16.0% |
| Mario | clumsy | 5.4% | 6.5% | Bomberman | clumsy | 9.0% | 8.0% |
| Mario | experienced | 21.7% | 30.7% | Bomberman | experienced | 12.7% | 11.3% |
| Mario | speedrunner | 19.9% | 22.8% | Bomberman | speedrunner | 13.2% | 13.2% |
| Mario | killer | 23.3% | 7.3% | Bomberman | killer | 10.9% | 13.7% |
| Mario | collector | 29.8% | 31.9% | Bomberman | collector | 10.9% | 13.7% |
| Meat Boy | novice | 7.8% | 18.4% | Mega Man | novice | 21.4% | 18.7% |
| Meat Boy | clumsy | 0.6% | 1.4% | Mega Man | clumsy | 14.4% | 6.9% |
| Meat Boy | experienced | 19.7% | 31.0% | Mega Man | experienced | 20.8% | 19.4% |
| Meat Boy | speedrunner | 19.7% | 31.0% | Mega Man | speedrunner | 20.6% | 23.0% |
| Meat Boy | killer | 12.8% | 15.2% | Mega Man | killer | 22.4% | 22.4% |
| Meat Boy | collector | 12.8% | 15.2% | Mega Man | collector | 20.8% | 19.4% |

- A lower anneal factor (0.3) is the one change that wins in most sweeps of every game; it is in the
  `--best` of all four games.
- The full combination of one sweep's winners can be worse than the baseline (Mario killer 23.3% to
  7.3%; Mega Man clumsy 14.4% to 6.9%). This is why `--best` keeps only the settings that win a
  strict majority of a game's sweeps.

### 9.3 Sensor ablation, 40 generations

| Game | Sensors | Cells solved | Mean win rate |
|---|---|---|---|
| Mario | rays (14) | 37 / 72 (51%) | 21.7% |
| Mario | grid (368) | 5 / 72 (7%) | 1.8% |
| Meat Boy | rays (21) | 54 / 198 (27%) | 12.3% |
| Meat Boy | grid (368) | 9 / 198 (5%) | 1.4% |
| Bomberman | rays (16) | 62 / 288 (22%) | 11.9% |
| Bomberman | grid (368) | 27 / 288 (9%) | 6.6% |

The few raycasts beat the full tile grid in every game. More inputs means many more weights to
evolve, not more useful information.

### 9.4 Mutation testing: can the mutant still be finished? (stage `ast_retrain`)

Share of retrained cells (persona x seed) that solved the mutant. "Evaluation level" is the
hand-placed mutant (18 cells); "other levels" are the mutants placed by rule on every other level.

| Game | Variant | Evaluation level | Other levels |
|---|---|---|---|
| Mario | `base-reseeded` (no change, new seeds) | 18 / 18 | 28 / 54 |
| Mario | `seal_route` (breaking) | 0 / 18 | 0 / 54 |
| Mario | `widen_gap` (degrading) | 18 / 18 | 27 / 54 |
| Mario | `jump_velocity_down` (degrading) | **0 / 18** | 16 / 54 |
| Mario | `off_route` (neutral) | 18 / 18 | 12 / 36 |
| Meat Boy | `base-reseeded` | 16 / 18 | 51 / 180 |
| Meat Boy | `seal_route` (breaking) | 0 / 18 | 0 / 180 |
| Meat Boy | `widen_gap` (degrading) | 6 / 18 | 15 / 36 |
| Meat Boy | `jump_velocity_down` (degrading) | 6 / 18 | 63 / 180 |
| Meat Boy | `off_route` (neutral) | 18 / 18 | 77 / 180 |
| Bomberman | `base-reseeded` | 0 / 18 | 124 / 270 |
| Bomberman | `seal_route` (breaking) | 0 / 18 | 0 / 270 |
| Bomberman | `add_enemy` (degrading) | 0 / 18 | 59 / 270 |
| Bomberman | `fuse_down` (degrading) | 0 / 18 | 113 / 270 |
| Bomberman | `off_route` (neutral) | 0 / 18 | 138 / 270 |

- **Every `seal_route` mutant is unsolvable** for the retrained agents (0 of 666 cells), as its class
  says.
- **Neutral `off_route` stays solvable** where the original is solvable (Mario 1-1 and Meat Boy The
  Gap: 18/18).
- **Degrading mutants are harder:** Bomberman `add_enemy` 59/270 against 124/270 for the reseeded
  original; Meat Boy `widen_gap` on The Gap 6/18 against 16/18.
- **Mario `jump_velocity_down` on 1-1 is 0/18.** It is classed degrading, but no retrained agent
  finishes 1-1 with the weaker jump. The mutant's own note explains it: with jump x0.85 the 4-tall
  pipe at column 57 can no longer be cleared from the ground, so only a Goomba bounce gets past it.
  For agents this mutant acts as a breaking one.
- **Bomberman's evaluation level is 0/18 for every variant**, the original included, so all its
  comparisons there are "not counted" (section 7.5). Bomberman's evidence comes from the other levels.
- **Mega Man** retraining was still running when this data was copied; its numbers are not final.
- The verdicts of the detection rule (replay) need `ast_play` and `final`, which have not run yet.

---

## 10. Known limits and caveats

1. **Bomberman's evaluation level (12, Bomberman II 1-1) was never solved**, in training and in every
   retrain. Its mutation comparisons are all "not counted". (An older version of this project used
   level 8, "09 Two Balloms", as the evaluation level; `paper-ast/ast_rule.md` still says
   "Bomberman 09". The code and this run use level 12.)
2. **Mario `jump_velocity_down` behaves as breaking for agents** on 1-1 (section 9.4).
3. **"Solved" is a low bar for clumsy.** 37 solved cells have a win rate of 0% (rounded), all clumsy:
   the seed won once, then almost never in the next 10 generations.
4. **Duplicate personas in Meat Boy:** experienced = speedrunner and killer = collector in every
   cell, so Meat Boy has 4 distinct agent sets, not 6.
5. **`--best` for Mario and Meat Boy only changes the anneal factor**, which acts after the first
   win; first wins are the same as the default in every cell.
6. **The detection rule is not pre-registered** (fixed on 2026-10-25, after pilot replays).
7. **Retraining is not a detector.** In an earlier run, the rule fired in 6 of 10 comparisons of
   `base-reseeded` against the unchanged original (`paper-ast/ast_rule.md`). The `final` stage
   measures this again on this run's data.
8. **Mega Man** sections trained from new seeds sometimes fail outright (Street and Climb in earlier
   runs), so a Mega Man "no longer solvable" needs that caveat.
9. **The replay-wall videos** in `runs/videos/` were recorded from older agents, not from this run's.
10. **Pilot study:** p1 and p4 have 4 of the 5 mutation runs; Test1 and test111 are test sessions.
11. **The win-rate window** is 10 generations (100 episodes), so a win rate is a short-run measure.

---

## 11. Where the data is

### 11.1 Folders

| Folder | What is in it | Written by |
|---|---|---|
| `runs/balance/` | Every report JSON: `report_*` (training), `gasweep_*` (GA sweep), `ablation_*` (sensor ablation) | `code/neuro/balance.py`, `gasweep.py` |
| `runs/probes/<game>/<persona>/<tag>/<level>_<seed>/` | Main training: one probe folder per cell | `balance.py` |
| `runs/gasweep/<game>/<persona>/<tag>/<level>_<seed>/` | GA sweep probes | `gasweep.py` |
| `runs/ablation/<game>/<persona>/<tag>/<level>_<seed>/` | Sensor ablation probes | `balance.py --ablation` |
| `runs/ast/<game>/<variant>/` | AST: the variant's report(s), `agents*.csv` playtests, and `probes/` (retrained agents) | `code/neuro/ast_eval.py` |
| `runs/ast/metrics.json` | AST metrics and verdicts (after the `final` stage) | `code/stats/ast_metrics.py` |
| `runs/user_study/<participant>/<game>/<phase>/run<k>/` | Pilot study: `episodes.csv` and `inputs.csv` | `menu.py` (option `u`) |
| `runs/manual/<game>/<player>/` | Free manual play with `--record` | `code/games/tools/manual_play.py` |
| `runs/logs/` | The Slurm job logs of every pipeline stage of the Nibi run | the cluster |
| `runs/videos/<game>/<persona>.webm` | Replay-wall videos | `code/neuro/wall_video.py` |
| `runs/site/` | The built Command Center website (not committed; rebuilt on demand) | `code/neuro/command_center.py` |
| `code/neuro/ga_best.yaml` | The `--best` GA settings per game, with every sweep's numbers | `gasweep.py` |
| `code/games/levels/mutants/<game>/<id>/` | Each mutant: `mutant.json` (what changed, class, why) and its level file | `eval_mutants.py`, `megaman_mutants.py` |
| `paper-ast/` | The AST paper draft, the rule, tables and figures | `code/stats/ast_paper.py` and by hand |

### 11.2 Files

**`best.npz`** holds the best network of a run (its weights). **`state.json`** holds the run state:
the GA settings (`config`), the persona, the best fitness, and the history of every generation
(wins, average fitness, best progress, and each episode's result).

**Report JSON** (`runs/balance/report_*.json`): `game`, `persona`, `tag`, `pop_size`,
`gens_budget`, `seeds`, `ga_config` (the GA settings used), `train_time_s`, `levels` (one summary per
level) and `cells` (one entry per level and seed). Main fields:

| Field | Meaning |
|---|---|
| `cells.<level>[].first_win_gen` | first-win generation, or `null` if never solved |
| `cells.<level>[].win_rate` | win rate in the 10 generations after the first win |
| `cells.<level>[].gens_run` | generations actually run |
| `cells.<level>[].best` | best fitness |
| `cells.<level>[].causes` | count of each cause of failure (for example `Stall`, `Enemy`, `Pit`) |
| `cells.<level>[].completion_rate`, `deaths_per_win`, `progress_at_death`, `stuck_frac`, `death_hist`, `death_cluster_entropy`, `coin_collection_rate`, `novice_expert_gap`, `improvement_rate`, `trend` | balance metrics of the cell |
| `levels[].solved_by`, `seeds` | how many seeds solved the level, of how many |
| `levels[].first_win_mean`, `first_win_ci` | mean first-win generation and its 95% confidence interval |
| `levels[].win_rate_mean`, `win_rate_ci` | mean win rate and its 95% confidence interval |

**Agent `episodes.csv`** (one row per episode, in each probe folder): `persona`, `game`, `world`
(the level), `cause_of_death` (`success` for a win), `jump_count`, `coins_collected`, `avg_vx`,
`progress_ratio`, `route` (sampled positions), `enemies_killed`, `bricks_destroyed`, `elapsed_time`.

**Human and playtest files** (`runs/user_study/.../episodes.csv`, `runs/manual/...`, and the
`agents*.csv` playtests; one row per attempt or life; written by `code/stats/episode_log.py`). Every
file starts with the same 12 columns: `persona` (the participant ID for people), `game`, `level`,
`source` (`human` or agent), `session`, `gen`, `agent`, `status` (`WON` for a win), `cause`,
`frames`, `time_s`, `progress`. Each game then adds its own columns; for Mario these are `jumps`,
`coins`, `kills`, `avg_speed`, `end_x`, `end_y`, `tile`, `bricks`, `run_result`, `lives_used`,
`life`, `recorded_at` and `route`. Rows of the study's mutation phase also have `mutant` and
`mutant_class`.

---

## 12. How to run it and reproduce the results

### 12.1 Set up

```bash
git clone https://github.com/Code-SorceryLab/PEAK-DRL-Tool.git
cd PEAK-DRL-Tool
pip install -r requirements.txt
python menu.py
```

The menu drives everything: train, watch, play, edit levels, run the sweeps, run the mutation
testing, run the user study and open the Command Center.

### 12.2 The commands behind each experiment

```bash
# Main training (one game, persona and level; 200 generations; the paper seeds)
python -m code.neuro.balance --game mario --persona novice --levels Mario1-1 --gens 200 --seeds 1234 2025 31337
python -m code.neuro.balance --game mario --persona novice --levels Mario1-1 --gens 200 --seeds 1234 2025 31337 --best

# GA sweep and sensor ablation (40 generations), as the cluster ran them
GAMES=mario PERSONAS=novice bash run_all.sh sweeps
python -m code.neuro.gasweep --game megaman --persona novice --levels CutMan-1-Street --gens 40 --seeds 1234 2025 31337
python -m code.neuro.gasweep --rebuild                       # rewrite code/neuro/ga_best.yaml from the sweeps

# Mutants
python -m code.games.eval_mutants --all-levels               # Mario, Meat Boy, Bomberman: every level
python -m code.games.megaman_mutants                         # Mega Man
python -m code.games.eval_mutants --list                     # the evaluation set of every game

# AST: retrain, playtest, replay, random baseline, metrics, paper tables
python -m code.neuro.ast_eval retrain --game mario --mutant seal_route --persona novice
python -m code.neuro.ast_eval retrain --game megaman --mutant base-retrained --persona novice --level CutMan-1-Street
python -m code.neuro.ast_eval play --game mario --variant base
python -m code.neuro.ast_eval play --game mario --variant seal_route --replay
python -m code.neuro.ast_eval play --game mario --variant base --random
python -m code.stats.ast_metrics                             # runs/ast/metrics.json
python -m code.stats.ast_paper                               # paper-ast/ tables and figures

# Rebuild reports from the probe folders, and build the website
python -m code.neuro.balance --rebuild
python -m code.neuro.command_center --open

# Watch an agent, or play a level or a mutant yourself
python -m code.neuro.trainer --game mario --replay runs/probes/mario/novice/p10g200/Mario1-1_1234/best.npz
PEAK_MUTANT=code/games/levels/mutants/mario/seal_route python -m code.games.tools.manual_play --game mario --level Mario1-1
```

### 12.3 On a cluster

`run_all.sh` holds the sweep and Mega Man jobs (`./run_all.sh sweeps | megaman | ast-megaman | test`).
The Nibi run used a pipeline script that runs the stages of section 6 as Slurm job arrays, one stage
after the other, and resumes from any stage. Each stage's task list is one command per line, as in
section 12.2.

### 12.4 Tests

```bash
python -m pytest code/tests -q
```

---

## 13. Status on 2026-10-29

| Part | State |
|---|---|
| Main training (200 generations, 1116 cells) | Done |
| GA sweep and `ga_best.yaml` | Done |
| Sensor ablation | Done |
| Mutants (all levels, all games) | Done |
| Retraining on mutants: Mario, Meat Boy, Bomberman | Done |
| Retraining on mutants: Mega Man | Finishing on Nibi; the data here is a copy taken while it ran |
| Playtests, replays, random baselines (`ast_play`) | Not run yet |
| Detection metrics and verdicts (`final`, `runs/ast/metrics.json`) | Not run yet |
| Pilot user study | 5 participants (Mario 3, Mega Man 2) |
| Replay-wall videos | Older agents; need a new recording |

When `ast_play` and `final` finish, copy `runs/ast/` and `runs/logs/` again from the cluster, then
rebuild the metrics and the website.

---

## 14. Authors

**Al (AI-Scripting)** · [LinkedIn](https://www.linkedin.com/in/al-mohamed-shifan-5266b924b/)
**Kevin Chu** · [LinkedIn](https://www.linkedin.com/in/kevincchua/)

Ontario Tech University, Master's program. Game engines, the evolution system and the dashboards are
built from scratch in Python, Pygame and numpy. The statistics-dashboard metrics were designed with
Amr Abdalla's statistics-observer work. The earlier deep-RL version (stable-baselines3 PPO) is kept
on the `archive/drl-sb3-final` branch.
