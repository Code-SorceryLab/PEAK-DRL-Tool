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

## Mega Man

| Char | Meaning |
| --- | --- |
| `O` | Pit / kill zone |
| `H` | Ladder |
| `M` | Met enemy |
| `B` | Bat enemy |
| `D` | Boss door / exit door (goal-coloured tile only — does **not** complete the level; use `G`) |
| `X` | Boss spawn |

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

## Sonic

| Char | Meaning |
| --- | --- |
| `C` | Ring |
| `E` | Badnik spawn |
| `S` | Spring |
| `/` | Steep slope up |
| `\` | Steep slope down |
| `(` | Gentle slope up |
| `)` | Gentle slope down |
| `[` | Gentle slope up top |
| `]` | Gentle slope down top |
| `U` | Concave slope |
| `n` | Convex slope |
| `G` | Goal / act clear |
| `P` | Player start |

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
