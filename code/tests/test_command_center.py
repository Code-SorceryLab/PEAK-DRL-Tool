"""The PEAK Command Center (code/neuro/command_center.py and the pages it gathers): only the paper
seeds reach it, every game gets every tab, and the site is static."""
import os
import shutil

from code.neuro import balance, cameras, report

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
