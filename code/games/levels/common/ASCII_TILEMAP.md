# ASCII Tile Map Sheet

Use these characters in `.txt` level files and in the level editor.

## Shared

| Char | Meaning |
| --- | --- |
| `.` | Air / empty space (a space character also reads as air) |
| `P` | Player start |
| `#` | Solid ground |
| `=` | One-way platform |
| `^` | Spike / hazard |
| `G` | Goal |

## Mario / Platformer

| Char | Meaning |
| --- | --- |
| `O` | Pit |
| `?` | Question block with coin |
| `>` | Question block with star |
| `<` | Question block with mushroom |
| `F` | Question block with fire flower |
| `L` | Question block with 1-up |
| `C` | Coin |
| `E` | Enemy |
| `k` | Koopa |
| `K` | Flying Koopa |

## Mega Man (NES) — `megaman`

A stage is a grid of 16 × 15-tile screens. A screen block left entirely blank does not exist:
walking off a row of screens is blocked, and falling out of a screen with none below is a pit.
Inside a screen, `.` and a blank both read as air.

| Char | Meaning |
| --- | --- |
| `#` | Solid block |
| `H` | Ladder (the top rung is a floor you can stand on and climb down through) |
| `^` | Spikes (instant death) |
| `D` | Shutter: opens once from the left, seals behind the player, and ends the camera's run |
| `G` | Stage clear on touch (for a level without a boss) |
| `X` | Boss (Cut Man): entering his room starts the fight; the orb he leaves ends the stage |
| `P` | Player start |
| `b` | Blader — flies at the player and swoops |
| `k` `j` | Beak on a right-hand wall, firing left (`j` starts 38 frames into its cycle); armoured while shut |
| `K` `J` | Beak on a left-hand wall, firing right |
| `c` | Super Cutter launcher (invisible): lobs a cutter every 32 frames while the player is within 80 px |
| `f` | Flea — hops at the player |
| `v` `h` | Octopus Battery — slides vertically / horizontally between walls |
| `s` | Flying Shell — one crosses the screen from the right edge, forever; armoured while shut |
| `e` | Big Eye — 20 shots to kill, 10 damage on contact |
| `w` `W` | Screw Bomber on the floor below / the ceiling above this cell |
| `+` | Large life energy (+10) |

Enemies appear when the screen edge uncovers them, and come back only after their spot has left
the screen. Their behaviour is in `code/games/megaman_enemies.py`.

Art and sound are optional: with the git-ignored `megaman-data/` folder present (ROM, sprite
sheets, stage map, sound effects — see `assets:` in `megaman_config.yaml`) the game draws the
real tiles and sprites; without it, flat placeholder shapes. A cell edited away from the ROM's
terrain is drawn with the most common tile of its new kind.

Training walks sections of the stage before the whole of it. A section is the same stage file
with its own starting point; `python -m code.games.tools.mm1_make_slices` works out where each one
starts (so it follows your edits) and lists them in `megaman_levels.yaml`.

Rebuild a stage from your own ROM with `python -m code.games.tools.mm1_extract_level --rom <file>`.

## Meat Boy

| Char | Meaning |
| --- | --- |
| `*` | Static saw blade (2 tiles in diameter, centred in the tile) |
| `%` | Crumble block (dissolves ~0.5 s after the player touches it) |
| `G` | Bandage Girl (goal) |

Moving or custom-sized saws go in the level's sidecar YAML:

```yaml
dynamics:
  saws:
    - x: 416          # centre, world px
      y: 240
      diameter: 96    # optional (default 64)
      end: [416, 80]  # optional -> ping-pong moving saw
      period: 4.0     # optional, seconds per full cycle
```

Notes:
- `.` is the recommended empty tile for hand-editing levels.
- The editor now filters files by game, so each game only shows its own levels.

## Bomberman (top-down)

Arena = outer wall + a pillar on every even (x, y); everything else is floor or brick.
Single screen, 15 × 13 tiles. The exit opens once every enemy is dead.

| Char | Meaning |
| --- | --- |
| `#` | Wall / pillar (indestructible) |
| `?` | Brick (destructible — one bomb arm removes it) |
| `G` | Exit, visible |
| `@` | Exit hidden under a brick |
| `E` | Ballom — slow, wanders |
| `k` | Onil — medium, chases when it sees you |
| `K` | Pass — fast chaser |
| `M` | Doria — slow, walks through bricks |
| `B` | Ovape — medium, walks through bricks |
| `C` | Bomb-up power-up under a brick (+1 bomb) |
| `F` | Fire-up power-up under a brick (+1 blast range) |
| `S` | Speed power-up under a brick |
