# PEAK Neuroevolution

Neural networks learn to play custom Pygame platformers (Mario-style, Megaman, Sonic, Meat Boy)
through evolution — no gradients, no reward shaping, no RL framework. Built from scratch:
sensors, network, population system, fitness evaluation, and a live browser dashboard.

> The previous deep-RL (stable-baselines3 PPO) version of this repo is preserved on the
> `archive/drl-sb3-final` branch.

## How it works

Each agent observes the game through **14 sensors** — six raycasts (forward, back, ±30°, ±60°),
forward enemy distance, pit detection, velocity, grounded/jump state, and nearby question blocks.
A tiny fixed-topology MLP (14 → 16 tanh → 3 sigmoid, ~290 weights, numpy only) maps sensors to
actions: move left, move right, jump.

Ten networks play the **same level simultaneously**, each in its own game instance. When all of
them die, get stuck, or win, the generation ends: fitness = furthest x reached (+ a win bonus).
The best two survive unchanged (elitism); the rest are bred by tournament selection, uniform
crossover, and gaussian mutation. Repeat. Early generations are chaos; within ~20 generations
on the first level the population produces a level-completing run.

Everything is deterministic under a seed — same seed, same evolution, step for step.

## Quickstart

```bash
pip install -r requirements.txt
python menu.py                            # interactive launcher
python -m code.neuro.trainer --game mario # or the CLI directly
```

Then open **http://127.0.0.1:8000/mario/index.html** — you get the live dashboard:

- grid of all 10 environments with per-env HUD (X, fitness, RUNNING/STUCK/DEAD/WON)
- a large view of the watched env with raycast overlay + live sensor readouts
- master stats bar (generation, all-time best, last gen best, avg fitness, steps/s)
- fitness-over-generations chart
- **Turbo** toggle (max-speed headless vs real-time) and **Sensors** debug toggle
- click any thumbnail to watch that env
- **manual play**: click "Take control" and drive the watched env yourself
  (arrows/WASD, space to jump) while the other nine keep evolving
- **debug overlays** from classic PEAK: hitboxes and tile grid, toggleable live
- **results table**: every generation's full metrics (best/avg/median fitness, wins,
  stuck/dead counts, best x, score, coins, duration) — also printed at end of run,
  via `--results runs/<name>`, and persisted as JSON in the run dir (no more CSVs)

### CLI

```bash
python -m code.neuro.trainer --game sonic          # also: megaman, meatboy
python -m code.neuro.trainer --game mario --level Mario1-2 --turbo   # start in turbo
python -m code.neuro.trainer --resume runs/mario                     # continue a run
python -m code.neuro.trainer --replay runs/mario/best.npz            # watch the all-time best
python -m code.neuro.trainer --results runs/mario                    # print the results table
python -m code.neuro.trainer --no-serve --gens 50                    # headless, no dashboard
python code/games/tools/level_editor.py --game platformer            # PEAK level editor
```

Checkpoints (population weights, RNG state, fitness history, best genome) land in `runs/<game>/`
after every generation.

## Layout

```
code/
  neuro/            the neuroevolution system
    net.py          numpy MLP
    evolution.py    GAConfig + Population (elitism/tournament/crossover/mutation, save/load)
    sensors.py      game-agnostic raycast + scalar sensors
    adapters.py     GameAdapter protocol + per-game adapters
    trainer.py      generational loop + CLI
    server.py       dashboard websocket + static HTTP server
    web/index.html  the dashboard (vanilla JS, no build step)
  games/            the four hand-written Pygame engines + ASCII levels + assets
  tests/            pytest suite (GA determinism, sensors, adapter smoke tests)
```

## What a dev can tweak

| Knob | Where | What it changes |
|---|---|---|
| GA hyperparameters | `GAConfig` in `code/neuro/evolution.py` | population size (10), elite (2), tournament k, crossover rate (0.7), mutation rate/σ (0.15/0.3), episode frame budget (3600), stuck kill (300 frames), curriculum `advance_wins` (3), win bonus (5000), seed |
| Player personas | `code/neuro/personas.py` | who the agents play like: sprint capability, sensor reaction period, time-left fitness bonus — add your own profile in one dataclass |
| Network shape | `code/neuro/net.py` | hidden size (16), outputs (add e.g. a climb action for Megaman ladders) |
| Sensors | `code/neuro/sensors.py` | ray angles/count, max distance (250px), march step, velocity normalization, pit-probe depth (4 tiles) |
| Levels | `code/games/levels/*/*.txt` + registration in `code/games/game_config.yaml` / `meatboy_config.yaml` | ASCII tilemaps — legend in `levels/common/ASCII_TILEMAP.md` (goal char is `G`, never `D`); or paint them in the level editor (menu 9) |
| Game feel / physics | per-game blocks in `game_config.yaml`, `meatboy_config.yaml` | gravity, jump velocity, run speed, coyote frames, wall-jump forces, per-level `time_limit` |
| Balance probes | `code/neuro/balance.py` | seed set (default 1234/2025/31337 — keep for paper comparability), gens budget, post-win measurement window |
| Dashboard | `--port` (HTTP, default 8000), ws 8765, frame cadences in `code/neuro/trainer.py` | streaming rates: thumbnails 5 fps (1 in turbo), watched env 20 fps |

Deep dives: `docs/GUIDE.md` (how the whole system works) and `docs/BALANCE.md` (every balance
metric + the author → play → train → balance loop).

Known wart: the top-level package is named `code`, which shadows a stdlib module. Renaming it
touches every import in `code/games/` and hasn't been worth the churn.

## Tests

```bash
python -m pytest code/tests/ -q
```
