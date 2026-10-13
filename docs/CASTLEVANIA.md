# Castlevania (NES), stage 1

`castlevania` is the second game played as a stage in sections, after `megaman`. It uses the
same framework end to end: a standalone core with `megaman_core`'s interface, a route field, a
training adapter, the relay, the telemetry, the mutant generator and the pipeline pages.

## What is in the game

The stage (`code/games/levels/castlevania/01_entrance.txt`, 14 x 2 screens of 16 x 12 tiles) is
the courtyard, the great hall with its two doors, the underground below the hall's middle room,
and the Phantom Bat's room. It is built by `code/games/tools/cv_extract_level.py` from the Tiled
maps of a C++ remake of block 1 kept locally in `castlevania-data/code`
(github.com/NearHuscarl/Castlevania). The movement numbers in `castlevania_config.yaml` come
from the same remake, not from the ROM; the ones it leaves to its animations (whip timing, the
boss's rests) are marked approximate.

| | |
|---|---|
| Simon | walk, a committed jump (no steering in the air), duck, a hit throws him back until he lands |
| Whip | a wind-up, then the lash; three levels (two upgrades in the courtyard); he cannot walk while it is out |
| Stairs | flights walked diagonally, entered only at the foot (up) or the head (down); two go through the floor to the underground |
| Enemies | zombies, bats and fishmen arrive in waves from spawners (`Z`, `B`, `F`); panthers (`p`) sit until he is near |
| Boss | the Phantom Bat (`X`): rests, swoops down beside him and across; 16 energy; the orb it leaves ends the stage |
| Candles | drop hearts, money or a whip upgrade (`c`, `C`, `m`, `w`) |
| Deaths | energy, water, the clock (300 s) |

Not modelled: sub-weapons and their pickups (a candle that held one drops a small heart), the
breakable walls (so no pork chop: nothing in the stage refills energy), the hidden money bags.

Art and sound come from `castlevania-data/` when it is there (`castlevania_assets.py`): the
reference maps' tile layers as the backdrop, the sprite sheets for Simon, the whip, the enemies,
the candles and the items, twelve sound effects and the stage and boss music (only a person
playing hears them). Without the folder the game draws flat shapes and is silent.

Two knowing departures from the console, both to keep a learning agent from standing still:
down ducks only with no direction held, and pressing up or down within a tile of a flight's end
walks him to it and on (the console does the second too).

## How it differs from Mega Man in the framework

- **Route** (`castlevania_route.py`). A jump cannot be steered, so the route is searched over
  the places a body can rest, joined by walks, drops, stairs and jumps whose landing is found by
  playing the jump out with the game's numbers. The field records the move out of every cell.
- **Compass** (sensors 18 and 19). It gives that move as a direction. Up without stairs means
  jump, and is given only from a spot where the jump lands where the route goes.
- **Progress** counts only with a foot down or on the stairs.
- **Fire** does not repeat while the output is held: one swing per press.
- **Sensors** 15 to 17 and 23 read the stairs, 24 the whip (ladders in Mega Man).
- **Training** varies the starting whip as well as the starting energy on later attempts, pays
  25 points per whip upgrade picked up, and pays a winner up to 1000 points for the share of its
  starting energy it kept (without that term the sections were won by walking through the
  enemies, and the full stage was lost for lack of energy).
- **Stalls**: an attempt with no progress for 240 frames ends, except during the boss fight.

A scripted player that only follows the compass (`code/neuro/compass_bot.py`: left / right from
18, up and jump when 19 is negative, down when it is positive, whip an enemy ahead) wins every
section and the full stage, so the stage is known to be completable with what the agents sense.
Its attempts can be saved as demonstrations and a network fitted to them
(`code/neuro/imitate.py`), as a starting population for a section evolution does not crack from
random networks. It is a check and a seed, not a player: it is not in the telemetry.

## Sections

| # | Level id | Screens | Starts |
|---|---|---|---|
| 1 | `Entrance-1-Courtyard` | 3 | on foot |
| 2 | `Entrance-2-Hall` | 3 | on foot |
| 3 | `Entrance-3-Panthers` | 3 | on foot |
| 4 | `Entrance-4-Descent` | 1 | on foot, past the first door |
| 5 | `Entrance-5-Crypt` | 3 (underground) | on the stairs, coming down |
| 6 | `Entrance-6-Ascent` | 1 | on the stairs, coming up |
| 7 | `Entrance-7-Gallery` | 2 | on foot, past the second door |
| 8 | `Entrance-8-Boss` | 1 | on foot |

`Entrance` is the full stage.

## Commands

```bash
# Rebuild the stage file from the reference maps
python -m code.games.tools.cv_extract_level

# Play it yourself (A / D move, W / S stairs and duck, Space jump, Z or J whip)
python -m code.games.tools.manual_play --game castlevania
python -m code.games.tools.manual_play --game castlevania --level Entrance-5-Crypt --record runs/manual --player me --warmup 3

# Train one section (dashboard at http://127.0.0.1:8000/castlevania/index.html), or headless
python -m code.neuro.trainer --game castlevania --level Entrance-2-Hall --seed 1 --hidden 32 --memory 2
python -m code.neuro.trainer --game castlevania --level Entrance-2-Hall --seed 101 --hidden 32 --memory 2 \
  --gens 60 --no-serve --run-dir runs/castlevania/Entrance-2-Hall/seed101

# The scripted compass player: check the stage, and save demonstrations to seed a hard section
python -m code.neuro.compass_bot
python -m code.neuro.compass_bot --demos runs/demos
python -m code.neuro.imitate --game castlevania --demos runs/demos/castlevania/compass-bot --memory 2 \
  --fire-window 0 --epochs 1500
python -m code.neuro.trainer --game castlevania --level Entrance-8-Boss --seed 201 --gens 60 --no-serve \
  --seed-from runs/castlevania/Entrance-8-Boss/imitation --run-dir runs/castlevania/Entrance-8-Boss/seed201

# Pick the best run per section, then play the full stage or one section with the picked networks
python -m code.neuro.relay --game castlevania --pick
python -m code.neuro.relay --game castlevania --episodes 20
python -m code.neuro.relay --game castlevania --section Entrance-5-Crypt --episodes 20
python -m code.neuro.relay --game castlevania --watch

# Telemetry page for the results
python -m code.stats.telemetry runs/castlevania/relay --serve --open

# Mutants: generate, list, play one
python -m code.games.mutants --game castlevania
python -m code.games.mutants --game castlevania --list
PEAK_CV_MUTANT=runs/castlevania/mutants/s3-seal_route python -m code.neuro.relay --game castlevania \
  --episodes 20 --out runs/castlevania/mutants/s3-seal_route/relay

# Pipeline pages (the game links at the top right switch between Mega Man and Castlevania)
python -m code.pipeline --game castlevania --open

# Tests
python -m pytest code/tests/test_castlevania.py code/tests/test_mutants.py code/tests/test_pipeline.py -q
```
