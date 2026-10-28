"""Level editor, every game: open a level, place a tile, save, re-open, and the game itself loads the edit."""
import os
import shutil
from pathlib import Path

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from code.games.tools import level_editor as le


def _game_reads(game: str, path: str):
    """The tile the game itself sees in the saved file: grid[row][col] as its own loader builds it."""
    if game in ("platformer", "meatboy"):
        from code.games.modules.System.LevelLoader import LevelLoader
        return LevelLoader().load_level(path).grid
    if game == "bomberman":
        from code.games.bomberman_core import BombermanCore
        core = BombermanCore()
        core.cfg["levels"], core._level_idx = [path], 0      # as manual_play --file does
        core.reset()
        return core.level_data.grid
    from code.games.megaman_core import MegamanCore
    return MegamanCore(level_file=path).level_data.grid


@pytest.mark.parametrize("game", ["platformer", "meatboy", "bomberman", "megaman"])
def test_open_edit_save_reopen_and_the_game_loads_it(game, tmp_path, monkeypatch):
    monkeypatch.setattr(le.Level, "TRIM", game not in le.GAME_SCREEN_GRID and game not in le.SAW_GAMES)  # as main()
    monkeypatch.setattr(le.Level, "SAVE_PHYSICS", game not in le.NO_PHYSICS_GAMES)
    assert "#" in [t[0] for t in le.tile_entries_for(game)]
    gc = le.GameConfig(game)
    src = Path(gc.get_level_file_path(gc.level_ids[0]))
    txt = tmp_path / src.name
    for f in (src, src.with_suffix(".yaml")):
        if f.exists():
            shutil.copy(f, tmp_path / f.name)

    unchanged = tmp_path / ("same_" + src.name)          # a save with no edit changes nothing, byte for byte
    le.Level.load(str(txt)).save(str(unchanged))
    assert unchanged.read_bytes() == txt.read_bytes()

    ed = le.Editor(le.Level.load(str(txt)), gc)
    r, c = next((r, c) for r in range(1, ed.level.rows) for c in range(1, ed.level.cols)
                if ed.level.get(r, c) in (" ", ".") and ed.level.get(r - 1, c) in (" ", "."))
    er, ec = next((r2, c2) for r2 in range(1, ed.level.rows - 1) for c2 in range(1, ed.level.cols - 1)
                  if ed.level.get(r2, c2) == "#")                 # and a tile taken out, as the eraser does
    ed.paint_cell(r, c, "#")
    ed.paint_cell(er, ec, " ")                                    # the eraser's brush
    le.do_action("SAVE", ed, None, None)

    again = le.Level.load(str(txt))
    assert again.grid == ed.level.grid and again.get(r, c) == "#" and again.get(er, ec) == " "
    assert again.saws == ed.level.saws
    seen = _game_reads(game, str(txt))[r][c]
    assert seen == "#" if game in ("bomberman", "megaman") else seen == le.TILE_GROUND
