# Bomberman in PEAK — wiring plan

Status: **plan, not started.** Date: 2026-08-22. Owner: Al.

Goal: a fifth game — a top-down Bomberman clone — that runs end to end through the existing
neuroevolution pipeline: `menu.py` → trainer → sensors (rays **and** grid) → balance probes →
command center → README figures. Same personas, same GA, same reports. No new dependencies.

---

## 0. Branch

`main` (b6d31081 "meatboy redone") is the **old SB3/PPO stack** — 46 commits behind
`ReVAMP-AL-AUG19` and has no `code/neuro/`. Two options:

- **A (recommended):** merge `ReVAMP-AL-AUG19` → `main` (fast-forward), then
  `git checkout -b bomberman main`.
- **B:** `git checkout -b bomberman ReVAMP-AL-AUG19`, merge to main later.

Either way the Bomberman branch must contain `code/neuro/`.

## 1. What the pipeline assumes today (and where Bomberman breaks it)

| Seam | Today | Bomberman |
|---|---|---|
| `NeuralNet.act()` → `(move_x, jump)`, `N_OUTPUTS = 3` | left / right / jump | needs **left / right / up / down / bomb** (5 outputs) |
| `GameAdapter.step(move_x, jump)` | 1-D movement | 2-D movement + bomb |
| `sensors.RAY_DIRS` | forward, ±30°, ±60°, back; flips with `vx` | top-down: 4 cardinals + 4 diagonals, no "facing" |
| 14-ray semantics | enemy corridor, pit probe, grounded, can_jump, q-blocks | blast danger, bomb available, bricks in range |
| `core._obs()["grids"]` (3 × 11 × 11: solid / collectible / hazard) | platformer window | **natural fit** — bricks/walls/bombs, power-ups/exit, blast cells/enemies |
| Fitness = max-x (Mario) or BFS-to-goal (Meat Boy) | linear progress | BFS-to-exit through bricks with cost; exit hidden under a brick |
| Stall kill: fitness must rise by >1 within `stuck_frames` | works | works with BFS fitness (destroying a brick on the path lowers cost) |
| Levels: ASCII tilemaps + `GLYPHS` in report.py | side-view | top-down ASCII, same glyph categories |
| `game == "meatboy"` special cases (menu, report, manual_play) | indexed levels | generalise to an `INDEXED_GAMES` set |

Everything else (GA, personas, balance/gasweep/report/figures, dashboard) is game-agnostic.

## 2. Level research — what "Bomberman" means here

Reference: NES *Bomberman* (Hudson, 1985) and *Super Bomberman* (SNES, 1993).

- **Arena:** outer wall of indestructible blocks; indestructible pillars on every even (x, y);
  the rest is floor or destructible brick. NES stages are 31 × 13 (horizontal scroll);
  SNES battle/normal stages are 13–15 × 11–13 (single screen).
- **Player:** 4-directional, tile-aligned movement with corner-cutting; drops bombs on its own
  tile; starts with 1 bomb, blast range 1; walks through its own bomb until it leaves the tile.
- **Bombs:** ~2.5 s fuse (150 frames @ 60 fps); blast is a cross, stopped by walls, destroys
  the first brick in each arm; chain-detonates other bombs; kills player and enemies.
- **Enemies (NES set):** Ballom (slow, random), Onil (faster, chases), Dahl (bounces),
  Minvo (fast), Doria (slow, passes bricks), Ovape (passes bricks), Pass (fast chaser),
  Pontan (fast, passes bricks). Touch = death.
- **Goal:** the exit is under a random brick; it opens once every enemy is dead. Bombing the
  exit spawns more enemies. Time limit 200 s.
- **Power-ups** (under bricks): bomb-up, fire-up, speed, wall-pass, detonator, bomb-pass,
  flame-pass, mystery.

**For PEAK (balancing tool, deterministic probes):**
- **Hand-authored levels**, single screen **15 × 13**, 32 px tiles → 480 × 416 px feed, no
  camera scroll (`camera = (0, 0)`). ASCII files in `code/games/levels/bomberman/`.
- **Win = stand on the exit with all enemies dead** (the "reach the flag" analogue). Exit is
  *visible* in authored levels (glyph `G`) unless a level file marks it hidden (`X` = exit under a
  brick) — both are level-design knobs the tool should be able to compare.
- **8 levels, ascending:** `01_open_floor` (no enemies, exit across open floor — sanity),
  `02_one_brick` (one brick wall between you and the exit), `03_corridor` (pillars only, 1 Ballom),
  `04_brick_maze` (dense bricks, exit visible, 2 Balloms), `05_hidden_exit` (exit under a brick,
  2 enemies), `06_chaser` (Onil + Pass), `07_brick_passers` (Doria/Ovape, bricks don't protect),
  `08_gauntlet` (NES stage-8 density, 4 enemies, power-ups). Level 1 must be solved by seed
  ~gen 1–3; level 8 should be unsolved by novice at 80 gens — that spread is the tool's signal.
- **Procedural NES-style stages** (31 × 13, seeded brick density + enemy roster by stage
  number) are a follow-up, not v1.

Glyphs (extend `ASCII_TILEMAP.md`): `#` wall (indestructible), `%` pillar (indestructible,
same category as `#`), `?` brick (destructible — reuses the "question block" category so the
route canvas draws it orange), `P` start, `G` exit, `X` exit under a brick, `E` enemy (Ballom),
`k` Onil, `K` Pass, `M` Doria, `B` Ovape, `C` power-up under a brick (`C` then the brick type in
a sidecar `.yaml`, like `world1_06_bladecatcher.yaml` for Meat Boy).

## 3. Design decisions (the ones that touch shared code)

1. **5-output net, per-game.** `GAConfig.n_outputs: int = 3`; trainer sets `5` for
   bomberman; `make_net(cfg)` uses it. `NeuralNet.act()` returns `(move_x, jump, move_y)` —
   `move_y` is `0` for 3-output nets. `adapter.step(move_x, jump, move_y=0)`; platformer
   adapters ignore `move_y`; Bomberman reads `jump` as **bomb**. Genome size differs per game
   exactly like rays vs grid already does — `n_outputs` lives in `state.json["config"]` so
   `report._sweep_point` computes `n_params` correctly.
2. **Adapter-owned senses.** `sensors.read_sensors()` gains one hook: if the adapter defines
   `sense()` it returns the 14-float vector + overlay rays/tiles itself. Bomberman's 14 slots:
   8 rays (N NE E SE S SW W NW, distance to wall/brick/bomb, enemy hit flagged like today),
   blast-danger on own tile (1 − time-to-boom, 0 if safe), nearest-enemy distance, bombs
   available (0/1), bricks within blast range (count/4), vx, vy → 14 + bias handled by the net
   as today. Trainer publishes `sensor_labels` in stats; the dashboard telemetry panel uses
   them when present (falls back to FWD/U30/…).
3. **Grid mode needs no new code** beyond `BombermanCore._obs()` returning
   `{"grids": (3, window, window)}` with `core.window` like Meat Boy: ch0 solid = walls +
   pillars + bricks + bombs, ch1 collectible = power-ups + exit, ch2 hazard = enemies + cells
   that will be in a blast within 30 frames. `_fit_window()` already handles `core.window`.
4. **Fitness (BFS with brick cost).** Dijkstra from the player's tile to the exit where floor
   costs 1 and a brick costs 6; `progress = 1 − best_dist / start_dist`, scaled ×1000 like Meat
   Boy; + 60 per brick destroyed, + 150 per enemy killed, win = 1000 + win_bonus +
   time bonus (persona `time_rate`). Death ends the episode (fitness frozen). This keeps the
   stall rule and `progress_at_death` / `death_hist` (bins by BFS progress, not x) meaningful.
5. **Personas map cleanly:** `sprint` → speed power-up behaviour (1.5× walk speed);
   `sensor_period` → reaction time (already generic); `time_rate` → time-left bonus.
6. **Indexed levels like Meat Boy**: `bomberman_config.yaml` lists the level files; menu /
   report / manual_play special-cases become `game in INDEXED_GAMES`.

## 4. Work packages (in order; each leaves `pytest` green)

| # | Package | Files | Check |
|---|---|---|---|
| 1 | **Branch + config** | `git checkout -b bomberman`, `code/games/bomberman_config.yaml` (tile 32, fuse 150, range 1, speeds, enemy table), `code/games/levels/bomberman/01…08.txt`, `ASCII_TILEMAP.md` glyph rows | files parse; 8 levels listed |
| 2 | **Core** `code/games/bomberman_core.py` | plain class like `MeatboyCore`: `level_data` (grid, width, height, start, exit, enemies), `player` (x, y, vx, vy, width, height, bombs_max, range, speed), `bombs`, `blasts`, `enemies` (per-type movers incl. brick-passers), `step(action)` with tile-aligned movement + corner cutting, chain detonation, enemy AI, exit rule, `score`, `timer`, `death_cause`, `won`, `_obs()` + `window`, `render(surface)` (flat-colour tiles + sprites; assets optional) | unit tests: bomb destroys brick and stops at wall, chain reaction, player dies in blast, exit opens only when enemies dead, enemy brick-pass, determinism under a seed |
| 3 | **Manual play** | `manual_play.py` key map (arrows/WASD + SPACE bomb), `menu._PLAY_CONTROLS` | `python -m code.games.tools.manual_play --game bomberman --level 3` playable to the exit |
| 4 | **Shared-code changes** | `evolution.GAConfig.n_outputs`, `net.act()` 3-tuple, `adapters.GameAdapter.step(move_y=0)` + all four adapters, `trainer` pass-through + `n_outputs` by game + `sensor_labels`, `sensors.read_sensors()` adapter hook | existing tests still pass; Mario 5-gen smoke identical fitness to before (determinism guard) |
| 5 | **Adapter** `BombermanAdapter` in `adapters.py` + `_ADAPTERS` + `list_levels`/`validate_level` | x/y/vx/vy, `grounded=True`, `can_jump` = bomb available, `camera=(0,0)`, `solid_at` (wall/pillar/brick/bomb), `enemy_positions`, `qblock_count_near` = bricks in range, `sense()`, fitness, `episode_stats` (`level_len` = start BFS distance so Reach % works), `set_level` | `test_adapters.py`: reset/step/win/death, sensor dims 14 and 368 + 5, net 5-output decode |
| 6 | **Menu / report / figures** | `menu.GAMES`, `get_levels_for_game`, `CONFIG_KEY`; `report._level_file/_level_grid/_level_config/_game_icon`; `figures.GNAME`; `INDEXED_GAMES` set shared by all three | `python menu.py` → Train Single → bomberman shows 8 levels; report renders a bomberman section with the route canvas |
| 7 | **End-to-end** | `python -m code.neuro.trainer --game bomberman --level 0 --gens 5 --turbo` (rays) and `--sensors grid`; dashboard telemetry shows Bomberman labels, rays drawn in 8 directions, progress bar moves; `python -m code.neuro.balance --game bomberman --gens 20` on levels 0–2; `python -m code.neuro.report` | level 0 solved by gen ≤ 3 on all seeds; level 1 needs a bomb (unsolvable without it); command center level dialog opens with verdict + routes |
| 8 | **Docs** | README game list + engines strip, GUIDE (adapter contract: 5 outputs), BALANCE (Bomberman metrics: bricks destroyed, enemies killed replace coin rate) | — |

Deferred (not v1): procedural NES stages, power-up roster beyond bomb-up/fire-up/speed,
sprites/animation (flat colours first), detonator/remote bombs, multiplayer battle mode.

## 5. Risks

- **Action-space change touches every game.** Mitigated by the `move_y=0` default and a
  determinism guard test (Mario fitness per gen unchanged before/after).
- **5-output genome ≠ 3-output genome** — a bomberman `best.npz` can't replay in Mario and vice
  versa; `run_meta` already tags game/level, so the menu's Watch picker stays correct.
- **Death by own bomb dominates early evolution.** The blast-danger sensor + hazard channel
  exist for exactly this; if novice still can't clear level 2, widen the fuse (config knob), not
  the GA.
- **BFS each frame is O(tiles)** — 195 tiles, trivial; recompute only when a brick/bomb changes.
- **Stall rule vs. waiting for a fuse:** 2.5 s of no fitness change is well under
  `stuck_frames` (20 s) — fine.

## 6. Definition of done

`python menu.py` → 13 Full Sweep with games = `mario, meatboy, bomberman` runs to completion;
the command center shows a Bomberman section whose level dialog has a verdict sentence, route
canvas over the arena, death heat-map by BFS progress; `16 README Figures` includes Bomberman in
`fig_difficulty`; 37 + new tests green.

---

## 7. As built — where the plan was wrong

Three things the plan got wrong, all found by running the thing:

**The 14-slot vector was the wrong shape.** The plan reused the ray-mode budget with eight compass
rays and a single "this tile is about to burn" scalar. Every agent, in every episode, on every
level, died to its own bomb: the vector said a blast was coming but never which way was out. The
adapter now owns a 16-slot vector (`N_INPUTS_BY_GAME`, new `GAConfig.n_inputs` seam) whose slots
4–7 are per-direction burn timers. First win on the brick levels went from *never in 60
generations* to generation 4.

**Eight levels was too coarse a ladder.** Levels 1 → 2 jumped from "walk to the exit" to "bomb
through a wall" to "kill an enemy" with nothing in between. The campaign is now 15 levels, one new
demand per rung, and `test_bomberman.py` pins each rung's intent (walkable vs bomb-required) so a
level can't silently seal itself — which is exactly what the first `05_sparse_bricks` draft did.

**The safe-bomb bonus needed splitting.** One counter can't teach two lessons. Paying for *any*
survived bomb let agents farm the tight arena with litter (first win gen 59); paying only for
*productive* bombs starved the exploration that finds the first bomb at all (gen 26). Surviving
your first bomb now pays once, and bombs that open a brick or land on an enemy pay per use.
