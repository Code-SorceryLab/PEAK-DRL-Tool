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

### Second round (AL's play-test, same evening)

| Area | Change | Files |
|---|---|---|
| Start | Meat Boy spawns standing on the floor of his `P` tile, in its middle (he dropped 10 px before) | `meatboy_core.py` |
| Zoom | Each level is framed as the game frames it: a level that fits the window fills it (1-1 and 1-2 at 1.32x); a bigger one scrolls at 30 x 17 tiles (4/3, the view of 1-4 in the game). The world is drawn at 1:1 and scaled; F1, F4 and F6 draw on the unscaled view, so they line up. The dashboard, the camera wall and the replay videos scale their rays by `core.zoom` | `meatboy_core.py`, `debugging_mods/manager.py` (`game_overlays=False`), `trainer.py`, `cameras.py`, `wall_video.py` |
| Lives | The study's lives show under the timer sign (Meat Boy and "x3"); free play shows the default 3 too; none in the warm-up | `meatboy_core.py` (`hud_lives`), `meatboy_art.py` |
| Fire | A wall of flame: 1-11's fire is 3 tiles deep, its top throws tongues half a tile up, and the tongues kill too | `meatboy_art.py`, `meatboy_core.py`, `world1_11_fired.txt` |
| Saws | Set into walls and floors, not floating: 1-6 (the third saw bigger, as in the game; one corridor opened to 3 tiles to jump the second), 1-7 (one size, on the wall faces), 1-8 (12 saws edge to edge in the pit floor), 1-9 (one saw every 2 tiles, the right ones sunk in 2 new rock columns, the floating pocket emptied), 1-10 (the shelf saw about twice the floor saw, both half sunk) | `levels/meatboy/` |
| Progress | Meat Boy's progress is the share of the start's shortest path to Bandage Girl covered: 0 standing still, 1 on a win. Before, an agent that never moved on 1-3 scored 551 of 1000 ("reach from the farthest cell", the audit's open item) and the CSV rows used x over the level's width. Agents (fitness, `episodes.csv`) and the study rows now use the same measure | `adapters.py` (`progress`, `reach`), `trainer.py`, `manual_play.py` |
| Menu speed | The menu took about 31 s to draw (and again after every action), because the run scan walked all 205,810 files under `runs/gasweep` before skipping them; it now skips that folder during the walk: about 3 s. A "Loading PEAK ENGINE" bar shows meanwhile | `menu.py` |
| Picker freeze | After a game closed, the reopened game-select window had no keyboard focus on macOS (it looked frozen); it now takes the focus. A loading bar shows until the game's window is up | `game_select.py`, `manual_play.py` (`PEAK_READY_FILE`) |
| Levels, round 2 | 1-4 Nutshell (the spiral, a closed pit of saws under it), 1-6 Bladecatcher (three shelves in a zig-zag, a saw at each open end, the top one bigger), 1-7 Diverge (the cliff outside and the tunnel inside, three saws) and 1-9 Safety Third (the crumbling wall in front of a saw wall, a crumbling floor over a saw pit) redrawn from the start screens of AL's copy; 1-5 got the start shed | `levels/meatboy/` |
| NES pad | macOS gives the USB NES pad to one process at a time, so a game started from the picker could not read it; the picker now lets go of it while a game runs | `gamepad.py` (`release_pad`), `game_select.py` |

### Third round (same night)

| Area | Change | Files |
|---|---|---|
| Levels | All ten levels but 1-3 redrawn as whole levels: one-screen levels from the wiki's 1920 x 1080 captures, scrolling ones stitched from a 1080p60 World 1 playthrough (YouTube g5nzkzyZmAc); sources and confidence per level in `MEATBOY_SMB_RESEARCH.md`. Slopes become steps; the 1-5 warp portal is not built. Only flood-fill reachability is checked: the long climbs (1-5, the 1-8 pillar, 1-9, 1-11) are not yet proven solvable in PEAK | `levels/meatboy/`, `runs/meatboy/mutants/` |
| Quit | A game started by hand goes back to SELECT GAME on quit; the game select, the study, recordings, demos, random play and the editor's play-test keep their own flow | `manual_play.py` |
| Loading | `menu.run_with_loading`: a bar in the terminal until the window or dashboard is up (the program touches `PEAK_READY_FILE`): game select, study, Watch Agent (opens the browser then), Watch All, Watch Random, Level Editor, Play a Mutant | `menu.py`, `game_select.py`, `level_editor.py`, `trainer.py` |
| Terminal | The menu turns newlines back on before each prompt (Enter had shown `^M`); the clip decoders keep off the terminal | `menu.py`, `game_select.py` |
| Music | A new original tune for every game in its style (no real melody copied: the repo is public); sawtooth and FM voices, chord arpeggios, swing | `modules/System/music.py`, `code/tests/test_music.py` |

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

1. 1-5 is still Kevin's climb (with the start shed): only its bottom room is on the start screen. 1-4, 1-7
   and 1-9 scroll, so the parts past their start screens (Bandage Girl's corner) follow the wiki, not the game.
2. The wall slide speed (180 px/s) and the crumble time (0.75 s) are estimates; no source gives them.
3. The run top speed (600 px/s, about 27 u/s) comes from the 1-3 record and Kevin's walkthrough
   measurement, not a direct measurement.
4. The debug panel makes the manual-play window 1630 px wide (as Mario's); F7 closes it. The study shows it
   too, as it does for Mario.
5. Editor: air painted in the editor saves as a space (the same tile); a saw's `period` cannot be edited
   there; the platform tool still shows for Meat Boy.
6. `docs/BALANCE.md` still names the old `world1_11_salt_factory` in its history table.
