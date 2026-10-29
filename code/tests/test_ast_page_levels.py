"""The Mutations tab and the paper's results with every level (code/stats/ast_page.py,
code/stats/ast_paper.py): each level's variants beside the evaluation level's, judged alike."""
import json
import os
import re

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from code.games import eval_mutants as em
from code.stats import ast_metrics as am
from code.stats import ast_page, ast_paper, episode_log


def row(status, level="5", **kw):
    r = {"source": "agent", "game": "meatboy", "persona": "novice", "level": level, "status": status,
         "cause": "" if status == "WON" else "Stall", "end_x": 900.0, "end_y": 400.0, "tile": 32, "time_s": 9.0,
         "frames": 600, "progress": 1.0 if status == "WON" else 0.4, "jumps": 3, "route": [[16.0, 96.0], [900.0, 400.0]]}
    r.update(kw)
    return r


@pytest.fixture(scope="module")
def mutants_dir(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("mutants"))
    em.build("meatboy", out, level="5")
    return out


@pytest.fixture
def results(mutants_dir, tmp_path, monkeypatch):
    real = em.out_dir
    monkeypatch.setattr(em, "out_dir", lambda game: mutants_dir if game == "meatboy" else real(game))
    monkeypatch.setattr(am, "ROOT", str(tmp_path))
    won = [row("WON", seed=s, run=f"r/{s}") for s in (1, 2, 3)]
    dead = [row("DEAD", seed=s, run=f"r/{s}") for s in (1, 2, 3)]
    episode_log.append(str(tmp_path / "meatboy" / "base" / "agents.csv"), [row("WON", level="2", seed=1, run="r/1")])
    for v, rows in (("base@5", won), ("base-reseeded@5", won), ("seal_route@5", dead)):
        episode_log.append(str(tmp_path / "meatboy" / v / "agents.csv"), [dict(r) for r in rows])
    episode_log.append(str(tmp_path / "meatboy" / "seal_route@5" / "agents_replay.csv"), [dict(r) for r in dead])
    return {"meatboy": am.evaluate("meatboy")}


def test_every_level_has_a_readable_name():
    assert em.level_name("meatboy", "5") == "Super Meat Boy 1-6 Bladecatcher"
    assert em.level_name("bomberman", "3") == "Bomberman 04 One Wall"
    assert em.level_name("mario", "Mario1-2") == "Super Mario Bros. 1-2"
    assert em.level_name("meatboy", "2") == em.EVAL_LEVELS["meatboy"][1]


def test_the_board_adds_a_row_for_every_other_level(results):
    board = ast_page.board(results)
    assert "other levels" in board
    sealed = re.search(r'data-slot="Sealed route"><span class="v">(.*?)</span>', board.split("other levels")[1]).group(1)
    assert "right 1/1" in sealed
    assert ast_page.scores(results)["good"] >= 1


def test_the_game_section_maps_each_level_from_its_own_file(results):
    html = ast_page._game_section("meatboy", results["meatboy"])
    data = json.loads(re.search(r'id="data-meatboy">(.*?)</script>', html).group(1))
    with open(os.path.join(em.LEVELS_DIR, em.level_file("meatboy", "5")), encoding="utf-8") as f:
        assert data["maps"]["5"] == [r.rstrip("\n") for r in f if r.strip()]
    assert data["level"]["seal_route@5"] == "5" and data["cells"]["seal_route@5"]
    assert '<option value="5">' in html and 'data-level="5"' in html


def test_the_paper_counts_every_level_beside_the_evaluation_level(results, tmp_path):
    rows = ast_paper.judgments(results)
    sealed = next(r for r in rows if r["id"] == "seal_route@5")
    assert sealed["level"] == "5" and ast_paper.where(sealed["game"], sealed["level"]) == "Meat Boy 1-6"
    assert sealed["replay"] is True
    written = ast_paper.write(results, str(tmp_path / "paper"))
    md = open(next(p for p in written if p.endswith("ast_results.md")), encoding="utf-8").read()
    assert "## Every level" in md and "Meat Boy 1-6" in md
    assert any(p.endswith("detection_levels.tex") for p in written)
