# Sonic (archived 2026-10-28)

Sonic left the active roster on 2026-08-22 (it cost probe budget; see `ARCHIVED_GAMES` in
`code/neuro/adapters.py`) and its code left the game set on 2026-10-28. Everything Sonic-only is here, at the
path it had under the repo root, so `git mv archive/sonic/<path> <path>` puts a file back:

- `code/games/sonic_core.py`, `sonic_config.yaml`: the engine and its settings
- `code/games/game_config.sonic.yaml`: the `sonic:` section cut from `code/games/game_config.yaml`
- `code/games/modules/Objects/`: `SonicPlayer`, `Badnik`, `Ring`, `Spring`, `SlopeTile`
- `code/games/modules/System/`: `SonicPhysicsManager`, `SlopePhysics`, and `music_sonic.py` (its tune, cut
  from `music.py`'s `TUNES`)
- `code/games/modules/Parameters/Sonic_*_parameters.py`
- `code/games/levels/sonic/green_hill_1..3.txt`

To restore it also: put the slope (`/ \ ( ) [ ] U n`) and spring (`S`) branches back into
`LevelLoader._parse_ascii_map` (they are in git before this commit), the `sonic` entries back into
`manual_play.py`, `game_select.py`, `level_editor.py`, `menu.py` and `code/games/__init__.py`.
`code/neuro/adapters.py` still has `SonicAdapter`; it fails on import until the core is back.

## Level glyphs
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
