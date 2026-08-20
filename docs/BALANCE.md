# PEAK Balance Metrics & Tooling

*The balance-measurement system: every metric, where it comes from, and how to read the
report. This implements the team balance-metrics table (Amr's doc) on the neuroevolution
probes.*

## The pipeline

1. **Probe** — `python -m code.neuro.balance --game <g>` (menu **[14] Balance Report**).
   For each (level × seed) a fresh population evolves until shortly after its first win
   (or the generation budget). Default seeds `1234 2025 31337` — the same set the paper
   matrix used, keep them for comparability.
2. **Aggregate** — per level, means ± 95% t-CIs across seeds, merged into
   `runs/balance/report_<game>_<persona>.json` (new probes extend the file; they don't clobber it).
3. **Report** — `python -m code.neuro.report --open` renders every game's JSON into one
   self-contained page, `runs/balance/report.html`: ranked difficulty table, death-location
   heatmaps, failure-mode bars, per-seed fitness curves. Menu [12] and [14] offer to open it.

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

**Selecting levels live:** the dashboard's Run panel has a level dropdown — picking one
switches the whole population at the next generation boundary (the curriculum resumes from
there). `--level` on the CLI locks a single level; menu Train Single prompts for one.

## Player personas

Probes and training runs imitate a chosen player type (`--persona`, or the persona prompt
in menu Train Single / Balance Report). A persona is a *capability + objective* profile,
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
4. **Judge the balance** — menu **[14] Balance Report** on the level (optionally per
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
with no reward shaping or per-game tuning (fitness = distance, 291 params, 4 engines unchanged).
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
route-overlay view on the level grid. Launch: menu → Analyze / Balance → "stats
dashboard", or `streamlit run code/stats/dashboard/app.py`.

Speed: `balance.py --workers N` runs probes in parallel processes (default cores-1;
each (level x seed) cell is independent, so wall clock divides by the worker count).
The trainer also skips all frame JPEG encoding while no dashboard tab is connected.

**Per-persona reports:** balance probes write `runs/balance/report_<game>_<persona>.json` — one file
per (game, persona) so tiers never overwrite each other's rows. The web report renders every
file it finds, grouped by game with the persona shown in the section header. Probe checkpoints
live under `runs/probes/<game>/<level>_<seed>/`.
