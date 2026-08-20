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
   `balance/report_<game>.json` (new probes extend the file; they don't clobber it).
3. **Report** — `python -m code.neuro.report --open` renders every game's JSON into one
   self-contained page, `balance/report.html`: ranked difficulty table, death-location
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

| Level | Style | First probe verdict |
|---|---|---|
| `Mario1-3` | SMB 1-3 athletic: platform islands over pits, koopas | solved 2/2 seeds, gen 9±0; Koopa 92% of deaths |
| `MM-Stage2` | Mega Man tower: ladder climbs, spike floor, mets + bats | see `balance/report_megaman.json` |
| `Green Hill 3` | Sonic act: rolling slopes, ring arcs, springs, badniks | see `balance/report_sonic.json` |
| `world1_11_salt_factory` | Meat Boy: wall-jump shafts, saws, crumble bridges | see `balance/report_meatboy.json` |

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
