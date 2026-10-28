"""Level editor, Meat Boy: the palette, the sidecar saws, and a load/save round trip that changes nothing."""
import os
import shutil
from pathlib import Path

import pygame
import pytest
import yaml

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from code.games.tools import level_editor as le

BLADECATCHER = Path(le.__file__).resolve().parent.parent / "levels" / "meatboy" / "world1_06_bladecatcher.txt"


@pytest.fixture
def meatboy(monkeypatch):
    """The Level switches main() sets for --game meatboy."""
    monkeypatch.setattr(le.Level, "TRIM", False)
    monkeypatch.setattr(le.Level, "SAVE_PHYSICS", False)


def test_palette_is_the_meatboy_glyphs():
    assert le.parse_args(["--game", "meatboy"]).game == "meatboy"
    assert sorted(t[0] for t in le.tile_entries_for("meatboy")) == sorted(" #%*^GP")


def test_bladecatcher_shows_its_saws():
    saws = yaml.safe_load(BLADECATCHER.with_suffix(".yaml").read_text())["dynamics"]["saws"]
    pygame.init()
    screen = pygame.display.set_mode((le.WINDOW_W, le.WINDOW_H))
    ed = le.Editor(le.Level.load(str(BLADECATCHER)), le.GameConfig("meatboy"))
    ed.show_grid = False
    ed.zoom = 0.5; ed._center()                          # room for a level redrawn bigger
    assert saws and len(ed.level.saws) == len(saws)
    le.draw_viewport(screen, ed, None, None)
    for i, s in enumerate(ed.level.saws):               # each centre handle is drawn, and grabs its saw
        centre = ed.wp2s(s["x"], s["y"])
        assert screen.get_at(centre)[:3] == le.HANDLE_START
        assert ed.ht_saw(*centre) == (i, "start")
    pygame.quit()


def test_level_list_is_read_by_index_and_never_rewritten(tmp_path):
    gc = le.GameConfig("meatboy")
    files = yaml.safe_load(gc.config_path.read_text())["levels"]
    assert gc.level_ids == [str(i) for i in range(len(files))]
    assert [gc.levels[lid]["file"] for lid in gc.level_ids] == files
    real, gc.config_path = gc.config_path, tmp_path / gc.config_path.name   # a copy: the real file is never at risk
    shutil.copy(real, gc.config_path)
    before = gc.config_path.read_bytes()
    assert not gc.assign_stage_to_level(str(BLADECATCHER), "99")
    assert not gc.delete_level_from_config("0")
    assert gc.config_path.read_bytes() == before


def test_save_round_trip_keeps_text_and_saws(tmp_path, meatboy):
    text = BLADECATCHER.read_text()
    side = yaml.safe_load(BLADECATCHER.with_suffix(".yaml").read_text())
    side["dynamics"]["saws"].append({"x": 400, "y": 208, "diameter": 96, "end": [400, 80], "period": 3.5})
    side["dynamics"]["firebars"] = []                   # keys the editor does not own...
    side["note"] = "kept"                               # ...go back as they were
    txt = tmp_path / BLADECATCHER.name
    txt.write_text(text)
    txt.with_suffix(".yaml").write_text(yaml.dump(side, sort_keys=False))
    ed = le.Editor(le.Level.load(str(txt)), le.GameConfig("meatboy"))
    le.do_action("SAVE", ed, None, None)                # bakes the (hidden) physics sliders, as the button does
    assert txt.read_text() == text
    assert yaml.safe_load(txt.with_suffix(".yaml").read_text()) == side
