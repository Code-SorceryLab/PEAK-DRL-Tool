"""The PEAK Command Center (code/neuro/command_center.py and the pages it gathers): only the paper
seeds reach it, every game gets every tab, and the site is static."""
import os
import re
import shutil
import subprocess

import pytest

from code.neuro import balance, cameras, report
from code.stats import ast_metrics, ast_page

PAPER = {"1234", "2025", "31337"}


def test_reports_keep_only_the_seeds_asked_for(tmp_path):
    for seed in ("1234", "8"):                  # a paper seed and one of the stray seeds committed on 2026-09-21
        src = os.path.join(balance.PROBES_ROOT, "mario", "novice", "p10g40", f"Mario1-1_{seed}", "state.json")
        dst = tmp_path / "mario" / "novice" / "p10g40" / f"Mario1-1_{seed}"
        dst.mkdir(parents=True)
        shutil.copy(src, dst / "state.json")
    cells = balance.load_probe_cells("mario", "novice", "p10g40", str(tmp_path), seeds=balance.PAPER_SEEDS)
    assert [c["seed"] for c in cells["Mario1-1"]] == [1234]
    assert {c["seed"] for c in balance.load_probe_cells("mario", "novice", "p10g40", str(tmp_path))["Mario1-1"]} == {8, 1234}


def test_the_wall_and_the_route_traces_skip_extra_seeds():
    # on the committed probes: Bomberman experienced level 1, seed 8 wins 57 %, the paper seeds 0 %
    models = dict(cameras.pick_models("bomberman", "experienced"))
    assert models and all(d.rsplit("_", 1)[1] in PAPER for d in models.values())
    routes, best = report._collect_routes("bomberman", "experienced", "1", "p10g40")
    assert best is None or best["seed"] in PAPER


def test_mega_man_has_a_drawn_icon_not_a_letter():
    html = report._game_icon("megaman")
    assert "<img" in html and "svg" in html and "gletter" not in html


def test_no_page_offers_a_watch_replay():
    assert not hasattr(report, "serve") and not hasattr(report, "_watch_cmd")


def test_the_old_name_is_gone():
    hits = subprocess.run(["git", "grep", "-l", "-i", "balance command", "--", "code", "menu.py", "README.md", "docs",
                           ":!code/tests/test_command_center.py"], capture_output=True, text=True).stdout.split()
    assert [h for h in hits if not h.startswith("docs/superpowers/")] == []


GAMES = ["mario", "meatboy", "bomberman", "megaman"]


def test_every_game_gets_its_four_sub_pages_without_a_rebuild(monkeypatch):
    from code.neuro import gasweep
    monkeypatch.setattr(balance, "rebuild", lambda *a, **k: pytest.fail("rebuild called"))
    monkeypatch.setattr(gasweep, "rebuild", lambda *a, **k: pytest.fail("gasweep rebuild called"))
    data = report.load_data()
    for g in GAMES:
        for page in (report.balance_page(g, data), report.gasweep_page(g, data), report.ablation_page(g, data)):
            assert page.startswith("<!doctype html>") and "<nav" not in page
            assert not re.search(r'href="(report|mutations|gasweep|ablation|instructions)\.html|"/wall"|"/watch', page)
    assert "<nav" not in report.instructions_page(data)


def test_mega_man_shows_its_sections_and_says_what_is_not_run():
    data = report.load_data()
    assert all(f"CutMan-{i}-" in report.balance_page("megaman", data) for i in range(1, 9))
    assert "No sensor probes yet" in report.ablation_page("megaman", data)
    if not any(key[0] == "megaman" for key in data["gasweep"]):    # until jobs 7434840/7434841 come home
        assert "No GA sweep yet" in report.gasweep_page("megaman", data)


def test_the_mutations_sub_page_keeps_the_totals_for_every_game():
    res = {g: ast_metrics.evaluate(g) for g in GAMES}
    page = ast_page.render(res, only="meatboy", nav=False)
    sc = ast_page.scores(res)
    assert f'{sc["good"]}</span>' in page and 'id="meatboy"' in page and 'id="mario"' not in page and "<nav" not in page
