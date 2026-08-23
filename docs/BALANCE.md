# PEAK Balance Metrics & Tooling

*The balance-measurement system: every metric, where it comes from, and how to read the
report. This implements the team balance-metrics table (Amr's doc) on the neuroevolution
probes.*

## The pipeline

1. **Probe** — `python -m code.neuro.balance --game <g>` (menu **[13] Full Sweep**).
   For each (level × seed) a fresh population evolves until shortly after its first win
   (or the generation budget). Default seeds `1234 2025 31337` — the same set the paper
   matrix used, keep them for comparability.
2. **Aggregate** — per level, means ± 95% t-CIs across seeds, merged into
   `runs/balance/report_<game>_<persona>_<tag>.json` (tag = `p<pop>g<gens>[_grid]`; new probes
   extend the file; they don't clobber it).
   **Sensor ablation:** `--sensors grid` swaps the 6 rays for the 3×11×11 tile window
   (solid / collectible / hazard, Dijkstra dropped, 368 inputs). Menu 14 runs both arms with
   `--ablation`: probes land under `runs/ablation/` and aggregate into `ablation_*.json`, so the
   ablation never touches the real sweeps in `runs/probes/`; `--compare --ablation` prints both
   side by side (generations-to-first-win ± CI, win rate) per level.
   **GA hyperparameter ablation (menu 15, `python -m code.neuro.gasweep`):** one `GAConfig`
   knob at a time is moved to a literature-grounded low and high bound while everything else
   stays at the baseline (the `GAConfig` defaults) — population 10 · 30 · 100 (Grefenstette 1986,
   De Jong 1975), elite 1 · 4 · 6 (Such et al. 2017 keep one elite), tournament 2 · 5 · 8
   (Miller & Goldberg 1995; Harik et al.), crossover 0 · 0.7 · 0.95, mutation rate
   0.005 (≈ 1/L) · 0.15 · 0.5, mutation σ 0.02 · 0.15 · 0.5, anneal 0.3 · 0.5 · off, init σ
   0.25 (Xavier) · 0.5 · 1.0, hidden 8 · 16 · 32 · 64, last-action feedback, 2–3 memory units:
   23 configs, each probed on every enabled level × 3 seeds. Probes land under
   `runs/gasweep/<game>/<persona>/p<pop>g<gens>[_grid]_b<sig>_<axis>-<value>/<level>_<seed>/` and aggregate into
   `runs/balance/gasweep_*.json`; the **GA sweep** page shows a generations-to-first-win vs
   parameter-count capacity curve with a flat / improves / degrades verdict, a bound track per
   axis, Δ tables and learning-curve overlays, and the per-axis-winner **recommended config**
   per game (`--confirm` probes that composite once — one-factor-at-a-time ignores
   interactions). Caveat: trajectories are bit-identical until the first win, so the anneal
   axis can only move win rate, never generations-to-first-win. The tag carries a 6-char
   fingerprint of the baseline (`_b1a2c3_`), so re-sweeping after changing `GAConfig` defaults
   lands beside the previous sweep instead of overwriting it; the page shows each baseline as its
   own group. First sweep (2026-08-20, 2 games × 3 personas × 23 configs, 32.6 h): capacity flat
   in 6/6, anneal 0.5 > 0.8 in 6/6 (now the default), pop 100 worse in 6/6, elite 6 / σ 0.02 /
   crossover 0 hurt Meat Boy, memory units and action feedback help only the sprinting persona.
3. **Report** — `python -m code.neuro.report --open` renders every game's JSON into one
   self-contained page, `runs/balance/report.html`: ranked difficulty table, death-location
   heatmaps, failure-mode bars, per-seed fitness curves. Menu [12] opens it; [13], [14] and [15] offer to.

## The metric table

### Challenge Calibration
| Metric | Definition | Status |
|---|---|---|
| `completion_rate` | wins / all episodes across the probe (0–1) | ✅ table + JSON |
| `mean_completion_time` | avg time (s) of winning episodes | ✅ |
| `completion_time_stddev` | σ of completion time | ✅ (JSON) |
| `progress_at_death` | end-x / level length for dying episodes (0 = start, 1 = goal) | ✅ |
| generations-to-first-win ± CI | GA-native difficulty: how long evolution needs | ✅ (new) |

### Punishment Severity
| Metric | Definition | Status |
|---|---|---|
| `deaths_per_run` | deaths per generation of 10 attempts | ✅ |
| `death_location_heatmap` | deaths binned across the level (10 bins, start → goal) | ✅ report heatmap strip |
| `death_cluster_entropy` | 0 = one hotspot, 1 = deaths spread evenly | ✅ |
| failure-mode mix | Enemy / Pit / Spike / Saw / OOB / Stall shares (paper taxonomy) | ✅ stacked bars |

### Triangularity (Risk/Reward)
| Metric | Definition | Status |
|---|---|---|
| `coin_collection_rate` | collected / available coins ("bandages"); 1.0 when a level has none | ✅ |
| `bandage_time_cost`, `bandage_death_premium` | need forced-collector agents | ⏳ future (needs coin term in fitness) |

### Skill Expression / Progression Fit
| Metric | Definition | Status |
|---|---|---|
| `novice_expert_gap` | win rate of last third of generations − first third (GA reading of novice vs expert) | ✅ |
| `completion_rate_per_skill` | per-budget completion | ⏳ run probes at different `--gens` |

### Path / Strategy Diversity & Emergent Complexity
`strategy_count`, `dominant_path_share`, `path_diversity_index`, `safe_vs_fast_path_ratio`,
`wall_jump_utilization_rate` — ⏳ future work: these need per-episode trajectory logging
(x,y polylines). The hooks exist (adapters already record end positions); the next step is
sampling positions every N frames into the env rows.

## Levels

Levels are ASCII tilemaps (`code/games/levels/`, legend in `levels/common/ASCII_TILEMAP.md` —
note: use `G` for goals, never `D`). Enabled levels register in `code/games/game_config.yaml`
(Mario at root, megaman/sonic nested) and `meatboy_config.yaml` (ordered list). New
game-faithful levels added:

| Level | Style | Probe verdict (2 seeds, 20-gen budget) |
|---|---|---|
| `Green Hill 3` | Sonic act: rolling slopes, ring arcs, springs, badniks | solved 2/2, first win gen 3.5 ±6.4, 50% win rate; Enemy 75% — easy/medium |
| `Mario1-3` | SMB 1-3 athletic: platform islands over pits, koopas | solved 2/2, first win gen 9 ±0, 16% win rate; Koopa 92% — medium |
| `MM-Stage2` | Mega Man: jumpable pillar corridor, spike troughs, mets + bats (ladders are manual-play flavor) | solved 2/2, first win gen 14 ±12.7, 7% win rate; Spike 94% — hard |
| `world1_11_salt_factory` | Meat Boy: wall-jump shafts, saws, crumble bridges | unsolved at 20 gens (75% BFS progress, still improving) — hardest; BFS-verified reachable |

**Design constraint the probes exposed:** agents act with left/right/jump only — Mega Man's
climb action is never pressed, so **ladders are unusable by agents** (manual play only).
Agent-facing megaman levels must have a jumpable critical path. The first MM-Stage2 draft
violated this (0/2 seeds, Spike 71%) and was redesigned; the first Green Hill 3 draft had
its goal floating above the running surface (agents sprinted underneath it — OOB 78%) and
got a full-height goal column instead. This is the tool working as intended: **probe every
new level before shipping it.**

### The Bomberman ladder — a campaign designed as a measuring instrument

Bomberman's authored ladder is fifteen levels (`code/games/levels/bomberman/`, list in
`bomberman_config.yaml`); **twelve are enabled** — 13-15 sit under `disabled_levels` because they
stayed 0/3 at every budget, so they cost sweep time without ranking anything. The rungs were
authored as a graded ladder rather than a game: each rung adds exactly one demand, so an unsolved
level names the skill that failed instead of just "hard".

| # | Level | New demand |
|---|---|---|
| 1 | `01_open_floor` | walk to the exit |
| 2 | `02_first_bomb` | one brick blocks a doorway — bomb it and retreat |
| 3 | `03_dead_end` | the same, with the retreat pointing away from the exit |
| 4 | `04_one_wall` | a brick wall spanning the arena; any one brick opens it |
| 5 | `05_sparse_bricks` | bricks as scenery — a clear lane exists, bombing is optional |
| 6 | `06_hidden_exit` | the exit is under a brick |
| 7 | `07_arena` | a small arena and one Ballom: the exit only opens on a clear arena |
| 8 | `08_corridor` | the same kill, in the full arena |
| 9 | `09_two_balloms` | two enemies |
| 10–15 | `10_brick_maze` … `15_gauntlet` | dense mazes, hidden exits, chasers, brick-walkers, power-ups, three enemies |

`code/tests/test_bomberman.py` pins each rung's *intent* — whether the level is beatable with no
bomb at all — so a level can't silently drift. It caught the first `05_sparse_bricks` draft, whose
scattered bricks happened to seal every route to the exit; the probe had reported it as
mysteriously unsolvable, one rung after a level agents beat in a single generation.

**Design constraints the probes exposed (Bomberman):**

- **The agent must be able to see where is safe.** With eight wall-distance rays and one "this tile
  is about to burn" scalar, every genome in every generation died on its own bomb — the vector said
  a blast was coming but not which way was out. Four per-direction burn timers took first win on the
  brick levels from *never in 60 generations* to generation 4.
- **Waiting is play, not stalling.** The trainer's stuck rule comes from the platformers, where
  standing still is always wasted time. Standing clear of a live fuse is the core Bomberman skill,
  so adapters may declare themselves `busy` and pause the rule.
- **A kill must not be worth less on a crowded level.** Scoring kills purely as a share of the
  enemies present paid 500 for the only enemy but 250 each for two — least reward where the task is
  hardest. Kills now also pay a flat per-kill bonus.
- **Self-inflicted blasts, not difficulty, were the ceiling.** Levels 8-14 went 0/3 in every
  sweep, and the dominant death on *every* level was the agent's own bomb (55-97 % of episodes).
  A 2.5 s fuse is not enough time for a reactive policy to walk clear, so any genome that learned
  to bomb also learned to die. Lengthening the fuse to 4 s (`bomb.fuse_frames: 240`) was the single
  biggest win in the sweep: `06_hidden_exit` went 0/3 → 2/3 and `02`/`03`/`08_corridor` all gained
  a seed. Measured, not guessed — every alternative below was probed the same way.
- **A range-1 bomb cannot be aimed, but widening it is worse.** Bombing a wandering Ballom
  point-blank killed it **0 times in 200 trials** (unchanged at fuses of 45, 60, 90 and 150
  frames); at range 2 the same shot lands **100 %** of the time. Raising the blast anyway made
  the ladder *worse* — `08_corridor` fell 3/3 → 0/3 and `09_two_balloms` 2/3 → 1/3 — because a
  bigger cross needs a longer run to escape and the agent has to leave from inside it. What did
  unlock the two-enemy wall was slowing the enemies to 0.7× so a blast has time to matter:
  `09_two_balloms` went 0/3 → 2/3, the first wins that level has ever recorded.
- **Do not gate a maze on kills.** `10_brick_maze` … `15_gauntlet` required a cleared arena, but a
  blast almost never catches a wandering enemy in a corridor maze, so the exit could not open.
  Those six now set `requires_clear: false`: their lesson is routing and bomb survival, while the
  open arenas (`07`-`09`) stay gated, where clearing really is the lesson.
- **Brick density has a playable ceiling near 20 %.** Every level the probes solve sits at 0-19 %
  bricks as a share of walkable space; every level they never solve sat at 28-57 %, with 5-11 bombs
  needed in sequence to reach the exit. Thinning the six mazes to a graded 21-32 % and their routes
  to at most three bombs lifted mean progress on them from 0.06-0.10 to 0.29-0.56 and bricks opened
  per episode from 0 to 0.7-2.5 — real movement, but they still do not finish.
- **`11_buried_exit` … `15_gauntlet` remain unsolved, and that is the honest hard end.** They ask
  for three or more bombs chained through a maze while two enemies hunt you, and 80 generations
  does not get there either. The ladder now reads: 10 of 15 rungs cleared by at least one
  seed, a smooth ramp through `09_two_balloms`, then a cliff — which is exactly the kind of
  step this tool exists to locate, now that it is a real one rather than an unaimable bomb.

- **The agent cannot tell a wall from a brick — and every fix for that made things worse.** The
  four rays return distance to whatever blocks the way; `solid()` treats permanent wall and
  destructible brick alike, so a maze reads exactly like a dead end (verified: the east ray is
  0.064 either way). Only one aggregate slot, "bricks a bomb here would open", carries the
  difference, and it has no direction. Two fixes were probed and both lost: **four extra
  "that one is breakable" slots plus a buried-exit flag** (16 → 21 inputs) dropped the ladder from
  57 seed-solves to 31, and folding the same bit into the **sign of the existing rays** (no extra
  inputs) dropped it to 8. The information is real; paying for it in input width costs more search
  than it returns at pop 10, and overloading the distance slots corrupts a signal the policy
  already leans on. Left as-is deliberately — the same shape of result as "bigger nets don't win
  sooner".

**Per-level overrides (`bomberman_config.yaml`).** A `levels:` entry is either a bare path or
`{file: ..., range: N, fuse: F, enemy_speed: X, requires_clear: false}`. `range` and `fuse` set the
blast arms and fuse the player starts that level with, `enemy_speed` scales every enemy on it, and
`requires_clear: false` opens the exit on arrival. The knobs are how the ladder's difficulty is
stated explicitly, and how each claim above was A/B-probed.

### Best hyperparameters per game — `code/neuro/ga_best.yaml`

Every sweep (`python -m code.neuro.gasweep`, and `--rebuild`) rewrites `code/neuro/ga_best.yaml`:
per game, the OFAT winner for each knob, kept only when it wins a **strict majority** of that
game's (persona × budget × sensors) sweeps — one persona's luck cannot move a default. The file
also records each sweep's `baseline_win_rate` and the `confirmed_win_rate` of the composite that
`--confirm` actually probed, so the OFAT-ignores-interactions caveat stays visible next to the
recommendation.

Consume it with `--best` on the trainer or the probe, or `GAConfig.for_game(game)` in code.
Explicit flags always beat the file, and probes run with `--best` are tagged `_best` so they never
land on baseline probe directories.

**Selecting levels live:** the dashboard's Run panel has a level dropdown — picking one
switches the whole population at the next generation boundary (the curriculum resumes from
there). `--level` on the CLI locks a single level; menu Train Single prompts for one.

## Player personas

Probes and training runs imitate a chosen player type (`--persona`, or the persona prompt
in menu Train Single / Full Sweep). A persona is a *capability + objective* profile,
not a reward function:

| Persona | Capabilities | Objective |
|---|---|---|
| `novice` | walk speed only, fresh senses every **3rd** frame (reaction lag) | reach the goal |
| `experienced` | walk speed, full reactions — the default | reach the goal |
| `speedrunner` | **sprint unlocked**, full reactions | goal **+ 25 fitness per second left on the clock** — finishing fast is selected for |

The persona is stored with the run (`state.json`), shown on the dashboard's Run panel, and
replay automatically uses the persona the genome was trained with. Balancing per persona is
the real version of the table's *skill expression* metrics: probe the same level with
`--persona novice` and `--persona experienced` and compare completion rates — that gap IS
`novice_expert_gap` across true player types (the in-run early/late-generation gap is the
within-persona approximation).

## Plug-and-play authoring loop

The intended workflow for level designers:

1. **Design** — menu **[9] Level Editor** (paint tiles, place entities; save writes the
   `.txt` + sidecar `.yaml`). Register the level in the game's config (or use Toggle Levels
   if it's already registered).
2. **Play it yourself** — menu **[5] Play Manually** (unlisted files work too:
   `python -m code.games.tools.manual_play --game platformer --file <path>.txt`).
3. **Train agents on it** — menu **[2] Train Single**, pick the level and a persona, watch
   on the dashboard (or pick the level live from the Run panel's dropdown).
4. **Judge the balance** — menu **[13] Full Sweep** on the game (optionally per
   persona), then the web report shows completion, death heatmap, and failure modes.

## The debug overlay

With **Sensors** on, the watched feed draws the classic PEAK debug view: red raycasts with a
white dot at each hit point, a red outline on the exact tile each ray stopped on, and
translucent blue boxes on the tile column the pit probe scans. **Hitboxes** and **Grid**
toggles add the engine's own overlays. In manual play (menu [5]) the F-keys do the same
in-window: F1 rays, F2 free cam, F3 slow motion, F4 hitboxes, F5 agent vision.

## Related work & positioning

*(Citation years/venues flagged for verification against the originals before submission.)*

**Why a GA probe instead of DRL.** Measured in this repo on the same levels: the GA reproduces
the PPO matrix's failure-mode taxonomy at roughly 1/100th the compute, bit-exact under a seed,
with no reward shaping or per-game tuning (fitness = distance, 291 params at the default 16
hidden units — the GA sweep page reports 147–1,155-param variants, 4 engines unchanged).
The trade: a reactive policy is a *lower bound* on difficulty — it cannot express memory or
planning, and it is not human-like. For balance CI that trade is correct: the probe must be
frozen, cheap, and deterministic so metric deltas isolate the level change. DRL is the better
player; the GA is the better measuring instrument for the inner design loop.

**Agents as difficulty probes.** Isaksen et al., *Exploring Game Space* (2015) — simple agents
mapping Flappy Bird difficulty; the closest ancestor of "cheap probe, many runs."
Gudmundsson et al. (King, 2018), *Human-Like Playtesting with Deep Learning* — production
difficulty prediction for Candy Crush. Roohi et al. (~2020) — difficulty/churn prediction
combining an RL agent with a player-population model.

**Procedural personas.** Holmgård, Liapis, Togelius & Yannakakis (2014–2018) — the canonical
line on encoding player archetypes as differently-parameterized agents; Mugrai et al. for
match-3. Citation anchor for our novice / experienced / speedrunner tiers.

**RL for game testing in industry.** Bergdahl et al. (EA SEED, 2020), *Augmenting Automated
Game Testing with Deep RL*; Gisslén et al. (2021), adversarial RL for content validation;
Zheng et al., *Wuji* (2019), evolutionary+RL hybrid that found real bugs at NetEase;
Ariyurek et al. (2021), synthetic human-like testers; Ubisoft La Forge and modl.ai ship this
commercially — evidence of industry pull.

**Automated balancing.** Volz et al. (GECCO 2016), *Demonstrating the Feasibility of Automatic
Game Balancing*; Fontaine et al. (2019), MAP-Elites over Hearthstone decks; Pfau et al. (2020),
*Dungeons & Replicants*, player-behavior clones for MMO class balance; Tomašev et al. (2020),
AlphaZero assessing chess-variant balance.

**Neuroevolution in platformers.** Togelius et al.'s Mario AI competition line (2009+);
SethBling's MarI/O (2015, NEAT); Such et al. (Uber, 2017) and Salimans et al. (OpenAI, 2017)
showing GA/ES competitive with DRL — legitimizing the probe choice.

**The gap PEAK occupies.** Studio tools are proprietary, single-game, and DRL-heavy — too
expensive per level edit. Academic difficulty work mostly scores finished levels offline.
The assembled loop — deterministic sub-minute probes, persona tiers, intent bands,
report-not-gate, inside the designer's edit loop, across multiple engines, on a laptop —
is the contribution: "SonarQube for playtesting" as an open, reproducible pipeline. The GA
probe is what makes the CI economics work.

**Use case in one sentence:** a designer edits a saw speed and two minutes later gets a
diffable report saying which balance dimensions moved out of intent — no scheduled human
playtest, no GPU.

## Episode CSVs & the stats dashboard

Every training run and balance probe now appends one row per finished episode to
`<run_dir>/episodes.csv` in the stats-dashboard schema (`persona, game, world,
cause_of_death, jump_count, coins_collected, avg_vx, progress_ratio, route,
enemies_killed, elapsed_time` — `route` is a sampled `(x, y)` trace, ~7.5 points/s).
Amr's Streamlit dashboard (`code/stats/dashboard/`, from the statistics-observer line)
reads those CSVs recursively from `runs/` (configured in
`code/stats/MarioThresholds.yaml`) and computes B1 challenge calibration, B2 punishment
severity, and B3 strategy diversity (route clustering) with per-level target bands and a
route-overlay view on the level grid. Launch: `streamlit run code/stats/dashboard/app.py`
(not in the menu).

Speed: `balance.py --workers N` runs probes in parallel processes (default cores-1;
each (level x seed) cell is independent, so wall clock divides by the worker count).
The trainer also skips all frame JPEG encoding while no dashboard tab is connected.

**Per-config reports:** probe checkpoints live under
`runs/probes/<game>/<persona>/<tag>/<level>_<seed>/` (sensor ablation: `runs/ablation/…`, GA sweep:
`runs/gasweep/…`) where `tag = p<pop_size>g<gens>[_grid]` (e.g. `p10g40`),
so different personas, population sizes, and generation budgets never overwrite each other;
re-probing a level with the same config replaces just that level. The Balance Command
(`python -m code.neuro.report --open`, menu 12) re-aggregates every probe on disk into
`runs/balance/report_<game>_<persona>_<tag>.json` each time it opens — no finished sweep
needed — and renders one section per (game, tag). Each probe cell records `best_gen`
(generation the record genome appeared), `improvement_rate` (slope of best progress per
generation until the peak, as a share of the level), win-time std, and wall-clock train time;
a `STUCK` episode counts as the `Stall` cause alongside Enemy/Pit/Saw/Spike.
