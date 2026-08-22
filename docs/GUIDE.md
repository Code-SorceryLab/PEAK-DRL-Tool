# PEAK Neuroevolution — The Complete Guide

*How the system works, what changed from the DRL era, and how to read everything on the dashboard.*

---

## 1. What this is

PEAK trains neural networks to play five hand-written Pygame games — four platformers
(Mario-style, Megaman, Sonic, Meat Boy) and a top-down Bomberman — using **neuroevolution**: no gradients, no reward shaping, no RL
framework. Ten small networks play the same level simultaneously; the ones that get furthest
breed the next generation. Repeat until the population completes the level, then the
curriculum moves everyone to the next one.

Quickstart: `python menu.py` (classic PEAK ENGINE menu) or
`python -m code.neuro.trainer --game mario`, then open
**http://127.0.0.1:8000/mario/index.html**.

---

## 2. DRL vs neuroevolution — what actually changed

| | Old (DRL / PPO) | New (neuroevolution / GA) |
|---|---|---|
| Learning signal | Per-step **reward** (shaped by "persona" functions, PBRS over Dijkstra distance) | Per-episode **fitness** (furthest x reached + win bonus) |
| Optimizer | Gradient descent on policy/value networks (stable-baselines3 PPO, torch) | Selection + crossover + mutation on raw weights (numpy only) |
| Network | CNN over 4×21×21 observation grids + 20 scalars (~hundreds of thousands of params) | 14 sensors → 16 tanh → 3 outputs (**~290 parameters**) |
| Observation | Grid channels (solids/collectibles/hazards/Dijkstra) rebuilt every step | 6 raycasts + 8 scalar sensors |
| Parallelism | SubprocVecEnv (12 subprocesses; caused the cv2/fork crashes) | 10 envs in **one process** — those crashes are structurally gone |
| Exploration | Entropy bonus, stochastic policy | Population diversity via mutation |
| Training feel | Hours, opaque, tensorboard curves | Minutes, watchable live, generation by generation |
| Determinism | Effectively unreproducible | **Bit-exact reproducible** from a seed |

**Why the DRL Mario struggled and this doesn't:** PPO had to discover credit assignment
through millions of gradient updates on a huge input. The GA searches a tiny, well-shaped
policy space (290 weights over 14 meaningful signals), and "went further = better" is exactly
the gradient the game offers. First level completion now happens around generation 6.

### What stayed the same (everything that made PEAK, PEAK)

- All four **game engines**, physics, levels, ASCII tilemaps, `game_config.yaml` — untouched
  except two safe patches (a `skip_obs` fast path and headless camera).
- The **player sprite art** — and it renders in training now, which the DRL era never did
  (headless mode silently skipped sprite loading since day one; the red rectangle you
  remember was a fallback).
- The **debug overlays** (hitboxes, tile grid, F-keys in manual play), now also toggleable
  from the dashboard.
- The **level editor** (v3.1, all 1,962 lines) and **manual play** with per-game controls.
- The **PEAK ENGINE menu** — logo, sections, toggle pickers, victory chime.
- The **metrics**: everything the old CSV logger captured per episode (x, max_x, level,
  event, cause) plus score/coins/kills/time it never captured. RL-only metrics (reward
  breakdowns, obs-sanity stats) died with the reward system.
- The old DRL stack itself is preserved on branch `archive/drl-sb3-final`.

---

## 3. The brain: sensors → network → actions

Every frame, each agent reads **14 sensors** (all normalized to roughly [-1, 1]):

| # | Sensor | Meaning |
|---|---|---|
| 0 | FWD ray | Distance to the nearest solid tile straight ahead (max 250px) |
| 1–2 | UP 30 / UP 60 rays | Solids diagonally up-forward |
| 3–4 | DN 30 / DN 60 rays | Solids diagonally down-forward (ledges, floors ahead) |
| 5 | BACK ray | Solids behind |
| 6 | ENEMY | Distance to nearest enemy in a ±1-tile corridor straight ahead |
| 7 | PIT | 1 if no ground within 4 tiles below a probe 1.5 tiles ahead |
| 8 | GND | On the ground right now |
| 9–10 | VX / VY | Velocity (the dashboard draws these as center-zero bars) |
| 11 | JUMP | Jump available (coyote frames left, not mid-jump) |
| 12 | QBLK | Question blocks within 5 tiles (Mario only; 0 elsewhere) |
| 13 | bias | Constant 1.0 |

Rays flip direction when the agent moves left. The whole vector feeds a
14 → 16 (tanh) → 3 (sigmoid) network; outputs are **left, right, jump** (left/right conflict
resolves by whichever is stronger, jump fires above 0.5). The hidden size is a `GAConfig` knob
(`--hidden`), and two optional carries exist for timing problems such as wall-jumps:
`--action-feedback` appends the previous (move, jump) as two extra inputs, `--memory N` adds N
Jordan memory units (extra outputs looped back as inputs next frame). Both are swept by menu 15.

### Is the THREAT panel accurate?

Mostly, with two honest caveats:

- **ENEMY** only sees enemies in a horizontal corridor (±1 tile of the agent's height,
  straight ahead). An enemy above or below goes undetected until it's level with you.
  Accurate for what it claims; narrow by design — 14 inputs is the point.
- **PIT** means "no landing surface within 4 tiles below a point just ahead." When the agent
  is high mid-jump, it reads 1.0 even over safe ground — during a jump there genuinely is no
  floor nearby, but read it as "no landing below" rather than "danger." When grounded, it is
  a reliable pit detector.

Sensor bars show *proximity* (bar grows as the obstacle nears); the raw value in the right
column is the normalized distance (1.00 = clear).

### A game can own its sensor vector

The table above is the side-scroller layout. A game whose geometry doesn't fit it declares its
own: an adapter that defines `sense()` returns whatever vector it likes, its length goes in
`N_INPUTS_BY_GAME`, and `SENSOR_LABELS` tells the dashboard how to draw it. Bomberman reads
**16**, because moving on two axes and outrunning your own bomb needs different information:

| # | Sensor | Meaning |
|---|---|---|
| 0–3 | N / E / S / W rays | Distance to the nearest wall, brick or solid bomb in each direction the agent can move |
| 4–7 | !N / !E / !S / !W | How soon the neighbouring tile burns (1 = burning now, 0 = safe) |
| 8 | BOOM | How soon *this* tile burns |
| 9–11 | NMY / NX / NY | Nearest living enemy: distance, and its bearing on each axis |
| 12 | BMB | Bombs left to drop |
| 13 | BRK | Bricks a bomb dropped here would open (of 4 arms) |
| 14–15 | EX / EY | Bearing to the exit |

Outputs grow to five (`N_OUTPUTS_BY_GAME`): left, right, **bomb**, up, down.

Slots 4–7 are the whole reason the game is learnable. Without them the agent knows a blast is
coming but not which way is out, and every genome in every generation dies on its own bomb —
that was the measured behaviour before they existed. With them, "step to the neighbour that
burns latest" walks out of the cross one tile at a time, and a reactive net can express it.

---

## 4. Is there a reward system?

No per-step rewards — that's the deepest change from DRL. There is a **fitness function**,
evaluated once per episode:

```
fitness = furthest x reached (max_x_seen)  [+ 5000 win bonus if the level was completed
                                            + time left × persona time_rate (speedrunner: 25/s)]
```

(Meat Boy's levels are 2-D mazes, so it uses BFS-distance-to-goal progress scaled to ~0–1000
instead of x. Bomberman scores Dijkstra cost-to-exit on the same 0–1000 scale, with bricks priced
at 6 so blowing open the right one counts as progress the moment it happens, plus a share for
enemies killed — its exit stays sealed until the arena is clear.) The old persona reward functions (`adept`, `speedrunner`, `enemy_hunter`, …)
are gone with the RL stack — under a GA, "which behaviors got further" replaces "which
actions earned points." Score and coins are **recorded** per episode as metrics, but they do
not influence evolution. If you ever want coin-hunting behavior, the lever is adding coins
into the fitness function, not a reward.

An agent that stops improving its fitness for 300 frames (5 seconds) is killed as STUCK —
that's the anti-stall, replacing the old core watchdog.

---

## 5. The generation loop and the GA

1. Each of the 10 nets gets one game instance on the **same level**; all step simultaneously
   (that's why the population wall is live).
2. An episode ends on death (enemy/pit/spike), win, STUCK, or the 3,600-frame (60s) budget.
3. When all 10 are done: fitness is computed, and the next generation is bred —
   the **4 best survive unchanged** (elitism); the other 6 come from tournament selection
   (best of 5 random picks), uniform crossover (70% chance, genes mixed 50/50), and gaussian
   mutation (each weight has a 15% chance of a ±0.15σ nudge; mutation is ×0.5 after a level's
   first win — the GA sweep's most robust finding).
4. Checkpoint written to `runs/<name>/` every generation.

All knobs live in `GAConfig` (`code/neuro/evolution.py`); its defaults are the baseline the GA
sweep (menu 15, `docs/BALANCE.md`) measures every knob against. Everything is seeded — same seed,
same run, bit for bit. That's also why manual play takes a real-time lock and why the core's
jump-arc overlay stays off during training (rendering it perturbs game state).

---

## 6. Levels: auto plays ALL of them

- **`--level <name>`** locks training to that single level forever.
- **No `--level` (auto)** walks **every enabled level in config order**. When a generation
  produces **3 winners** (`GAConfig.advance_wins`), the whole population carries its weights
  to the next level. The results table records which level each generation ran on, and
  `--resume` continues on the level you left off.

Measured from seed 42: Mario1-1a solved by gen 13 → 1-1b by 19 → 1-1c by 20 → **full
Mario1-1 by gen 23** → into world 1-2. Skills transfer: after advancing, the population
usually wins on the new level within a few generations.

Enable/disable levels with menu option **[10] Toggle Levels** (edits `game_config.yaml`).
One caveat: the header's all-time best spans levels, so a long early level can outrank a
short later one — read per-level bests from the results table.

---

## 7. Turbo vs real-time

- **Real-time (default):** the loop is paced to 60 fps — the games run at natural speed,
  feeds are smooth (thumbnails 5 fps, watched env 20 fps). Best for watching behavior.
- **Turbo:** no pacing — the simulation runs as fast as the CPU allows (a generation takes
  ~2–8s instead of ~60s). Frame encoding throttles down (thumbnails 1 fps) to keep CPU on
  simulation; the watched env stays live. Best for actually training.

Toggle it live from the dashboard or start with `--turbo`. Manual play forces real-time
while you hold the controls. Turbo does not change game physics or outcomes — only pacing —
so runs stay deterministic either way.

---

## 8. Reading the dashboard

- **Header:** generation counter, all-time best, last-gen best, average fitness, steps/s,
  win rate over the last 10 generations, elapsed time. Buttons: Turbo, Sensors, Hitboxes,
  Grid, Take control.
- **Watched specimen:** big live feed with the raycast overlay (gold = solid hit,
  red = enemy, white dot = hit point). The header line shows x, fitness, score, coins,
  time left, and the death cause once it dies. "Take control" hands you the keys
  (arrows/WASD, space) while the other nine keep evolving.
- **Sensor telemetry:** the 13 live sensor bars, grouped RAYS / THREAT / BODY / LOOT.
- **Fitness record:** gold = generation best (area), blue = average.
- **Run:** level (updates as the curriculum advances), population, elite, mutation, mode,
  wins, total frames.
- **Population wall:** one card per env — live feed with rays, square status pip, X and
  fitness, and a gold rail scaled to the generation's best. Dead specimens grey out. Click
  a card to watch it. The census strip counts running/stuck/dead/won.
- **Results table:** one row per generation — level, best/avg/median fitness, wins, stuck,
  dead, best x, avg score, coins, duration. Newest first. The same table prints via
  `--results runs/<name>` and at the end of every run; full data (per-env records with
  death causes) persists in `runs/<name>/state.json`.

---

## 9. Balancing games with it — the balance report

PEAK's purpose is balancing levels with agents. The paper pipeline did this with PPO
matrices (win rate ± CI + failure-mode taxonomy, fed by the stats subsystem); the GA
successor is `python -m code.neuro.balance --game mario` (menu **[13] Full Sweep**).
For each (level × seed) it evolves a fresh population, stops shortly after the first win,
and aggregates with 95% t-CIs: **win rate, generations-to-first-win, dominant death cause,
stuck rate, best x, learning trend** — hardest levels first. JSON lands in
`runs/balance/report_<game>_<persona>_<tag>.json` (tag = `p<pop>g<gens>[_grid]`).

First real run (4 levels × 3 seeds, 12.4 min total, all seeds solved everything):

| Level | First win | Win rate | Dominant cause |
|---|---|---|---|
| Mario1-1a | gen 10.0 ±4.3 | 18% ±10% | Enemy (51%) |
| Mario1-1b | gen 7.0 ±5.0 | 33% ±40% | OOB (54%) |
| Mario1-1 (full) | gen 4.7 ±5.7 | 36% ±12% | Enemy (65%) |
| Mario1-1c | gen 1.3 ±1.4 | 83% ±18% | Enemy (67%) |

Immediate balance insight: the 1-1a slice is *harder than the full level* (slice boundaries
invert difficulty), and full Mario1-2 is a wall — the curriculum population reached x≈4990
and won zero episodes in 25 generations there.

## 10. Files and tools

```
menu.py                     PEAK ENGINE hub (train / play / watch / tools)
code/neuro/                 the neuroevolution system (net, evolution, sensors,
                            adapters, trainer, server, web dashboard)
code/games/                 the five engines + levels + assets (four platformers + bomberman)
code/games/tools/           level_editor.py, manual_play.py, make_level_slices.py
runs/<name>/                gen_state.npz (weights) + state.json (config, RNG,
                            full history) + best.npz (all-time best genome)
```

Replay the best genome: `python -m code.neuro.trainer --game mario --replay runs/mario/best.npz`.
Resume training: `--resume runs/mario`. Tests: `python -m pytest code/tests/ -q` (76 tests:
GA determinism, net shapes, sensor geometry, adapter/trainer smoke tests for every game,
balance aggregation, GA-sweep tags and verdicts).
