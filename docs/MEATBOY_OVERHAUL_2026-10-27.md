# Meat Boy overhaul, 2026-10-27 (branch USER-STUDY)

A handoff for the next session. Meat Boy now plays like Super Meat Boy's World 1. Every value, its source
and the comparison with the real game: `docs/MEATBOY_SMB_RESEARCH.md`. Nothing here is committed yet.

## What changed

| Area | Change | Files |
|---|---|---|
| Physics | Walk and run (Shift) with the measured speeds; instant stop on the ground; turning at 2x; no air friction; run let go drops to walk speed at once; jump 470 px/s, gravity 918, let go after 8 frames stops the climb; no jump buffer or coyote time; fall cap 500 | `meatboy_config.yaml`, `modules/Objects/MeatboyPlayer.py`, `modules/System/MeatboyPhysicsManager.py` |
| Walls | Slide (180 px/s) only while held into the wall, falls at full speed when let go; 200 ms stick when pulling away; wall jump 500 up / 500 away, no lockout; the player counts its wall jumps | `MeatboyPlayer.py`, `code/stats/episode_stats.py` |
| Body and view | 22 x 22 px (0.7 tile, the game's proportion); a 1280 x 720 window (the game's own, 40 x 22.5 tiles); a level smaller than the view sits centred | `meatboy_core.py` |
| Elements | Crumble 0.75 s after a touch (back on the next attempt); `^` is fire (World 1 has no spikes); blood stays across attempts; the hanging timer sign at the top left | `meatboy_core.py`, `meatboy_art.py` |
| Levels | 1-1, 1-2, 1-3 redrawn from AL's copy of the game; 1-8, 1-10 and 1-11 from the 100 % video; 1-11 is now Fired (the old 1-11 "Salt Factory" was not a World 1 level); "1-1 Hello World" lost its trailing "World" in `level_name` (fixed) | `levels/meatboy/`, `eval_mutants.py` |
| Agents | Meat Boy's own `sense()`: the 14 shared rays plus wall left/right, ceiling, nearest saw (dx, dy, distance to the blade) and a rising jump: **21 inputs** (`N_INPUTS_BY_GAME`); the shared rays moved into `sensors.shared_rays()` | `code/neuro/adapters.py`, `code/neuro/sensors.py` |
| Mutants | 1-3's five mutants re-placed on the new layout; all Meat Boy mutants regenerated (`--all-levels`); a crumbling block counts as passable in the flood fill (it breaks), as a brick does for Bomberman; the stale `widen_gap@9` (old 1-10) removed | `code/games/eval_mutants.py`, `runs/meatboy/mutants/` |
| Debug panel | F1 to F7 and the side panel as Mario's: F1 shows the agents' real sensor rays and the jump arc, F4 the saw circles; two guards in the shared overlays for a core without Mario's physics manager | `meatboy_core.py`, `debugging_mods/overlays.py`, `menu.py` |
| Manual play, study | 0.3 s splat, then straight back; the game's own timer sign instead of the TIME pill (TIME LEFT in the study stays); the study keeps `--lives 3` (AL's choice); play-test from the level editor works (`--file`) | `code/games/tools/manual_play.py` |
| Level editor | `--game meatboy`: the seven glyphs, saws in the sidecar (B: place, drag, `Dia - +`, Delete), never rewrites `meatboy_config.yaml` | `code/games/tools/level_editor.py` |
| Tests | 13 Meat Boy tests (tuned values, speeds, jumps, gap, walls, crumble, fire, saws, sensors); 4 editor tests | `code/tests/test_meatboy.py`, `code/tests/test_level_editor_meatboy.py`, `test_adapters.py` |
| Git | `meatboy-data/` is git-ignored: screenshots of AL's game and the playthrough video live there | `.gitignore` |

## What needs retraining

Everything Meat Boy: the physics, the body, the view, the levels and the network input (14 to 21) all
changed, so no old Meat Boy genome loads or plays the same.

- `runs/meatboy_*` (the evaluation and curriculum runs of every persona)
- `runs/ast/meatboy/` (baselines, mutant retrains, replays) and the AST verdicts built on them
- The balance reports and the GA-sweep rows for Meat Boy (`runs/balance`, `code/neuro/ga_best.yaml`'s
  meatboy entry was tuned on the old game)
- The Meat Boy replay videos on the wall and in the game-select clips (`runs/videos`)

## Smoke training (27 Oct, 1-3, `--best`, seed 1, 30 agents)

| Persona | First win | Wins out of 30 |
|---|---|---|
| experienced (runs) | generation 3 | 27 at generation 8, 24 at 15 |
| novice (cannot run) | generation 3 | 30 at generation 10 |

The novice clears the gap without the run: a walking jump falls short, and the agent wall-jumps up the far
ledge's face (the speedrunners' "cornerjump"). The game allows the same, so the layout keeps it.

## How to check it

```bash
SDL_VIDEODRIVER=dummy .venv/bin/python -m pytest -q -p no:cacheprovider code/tests/test_meatboy.py code/tests/test_level_editor_meatboy.py
.venv/bin/python -m code.games.tools.manual_play --game meatboy --level 2
.venv/bin/python -m code.games.tools.level_editor --game meatboy code/games/levels/meatboy/world1_03_the_gap.txt
```

## Still open

1. 1-4 to 1-7 and 1-9 are Kevin's 13 Oct drawings, not traced from the game. The video shows 1-5 and
   1-7 only in part (the camera scrolls) and not 1-4 or 1-9; AL's copy has 1-1 to 1-13 unlocked, so
   screenshots of each level's start screen are the quickest source.
2. The wall slide speed (180 px/s) and the crumble time (0.75 s) are estimates; no source gives them.
3. The run top speed (600 px/s, about 27 u/s) comes from the 1-3 record and Kevin's walkthrough
   measurement, not a direct measurement.
4. The debug panel makes the manual-play window 1630 px wide (as Mario's); F7 closes it. The study shows it
   too, as it does for Mario.
5. Editor: air painted in the editor saves as a space (the same tile); a saw's `period` cannot be edited
   there; the platform tool still shows for Meat Boy.
6. `docs/BALANCE.md` still names the old `world1_11_salt_factory` in its history table.
