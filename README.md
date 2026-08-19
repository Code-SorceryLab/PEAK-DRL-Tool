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
python -m code.neuro.trainer --game mario
```

Then open **http://127.0.0.1:8000/mario/index.html** — you get the live dashboard:

- grid of all 10 environments with per-env HUD (X, fitness, RUNNING/STUCK/DEAD/WON)
- a large view of the watched env with raycast overlay + live sensor readouts
- master stats bar (generation, all-time best, last gen best, avg fitness, steps/s)
- fitness-over-generations chart
- **Turbo** toggle (max-speed headless vs real-time) and **Sensors** debug toggle
- click any thumbnail to watch that env

### CLI

```bash
python -m code.neuro.trainer --game mario --level Mario1-2 --turbo   # start in turbo
python -m code.neuro.trainer --resume runs/mario                     # continue a run
python -m code.neuro.trainer --replay runs/mario/best.npz            # watch the all-time best
python -m code.neuro.trainer --no-serve --gens 50                    # headless, no dashboard
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

GA hyperparameters live in `GAConfig` (`code/neuro/evolution.py`): population 10, elite 2,
mutation 0.15/weight (σ=0.3), crossover 0.7, 3600-frame episode budget, 300-frame stuck kill.

Known wart: the top-level package is named `code`, which shadows a stdlib module. Renaming it
touches every import in `code/games/` and hasn't been worth the churn.

## Tests

```bash
python -m pytest code/tests/ -q
```
