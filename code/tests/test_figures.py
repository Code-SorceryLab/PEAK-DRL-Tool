"""README figure generator on synthetic report dicts — writes PNGs to tmp_path, no probes needed."""
import os

from code.neuro import figures


def _rep(game, persona, tag, levels, gens=40):
    return {"game": game, "persona": persona, "tag": tag, "pop_size": 10, "gens_budget": gens, "seeds": [1, 2],
            "levels": [{"level": lv, "win_rate_mean": wr, "win_rate_ci": 0.1, "first_win_mean": 5.0, "solved_by": 2}
                       for lv, wr in levels],
            "cells": {lv: [{"seed": s, "win_rate": wr, "first_win_gen": 5} for s in (1, 2)] for lv, wr in levels}}


def test_difficulty_and_sensors_figures(tmp_path):
    reports = [_rep("mario", p, "p10g80", [("Mario1-1", 0.6), ("Mario1-2", 0.2)], gens=80) for p in ("novice", "experienced")]
    abl = [_rep("mario", "novice", "p10g40", [("Mario1-1", 0.6)]), _rep("mario", "novice", "p10g40_grid", [("Mario1-1", 0.4)])]
    assert figures.fig_difficulty(reports, str(tmp_path / "d.png"))
    assert figures.fig_sensors(abl, str(tmp_path / "s.png"))
    assert figures.fig_sensors(abl[:1], str(tmp_path / "unpaired.png")) is False  # one arm only → skipped
    assert os.path.getsize(tmp_path / "d.png") > 1000 and os.path.getsize(tmp_path / "s.png") > 1000


def test_readme_stamp_rewrites_between_markers(tmp_path, monkeypatch):
    readme = tmp_path / "README.md"
    readme.write_text(f"head\n{figures.MARK_OPEN}\nold\n{figures.MARK_CLOSE}\ntail\n", encoding="utf-8")
    monkeypatch.setattr(figures, "README", str(readme))
    assert figures.update_readme(["fig_knobs.png"], {"report": 1, "ablation": 2, "gasweep": 3})
    text = readme.read_text(encoding="utf-8")
    assert text.startswith("head\n") and text.endswith("\ntail\n") and "old" not in text and "fig_knobs.png" in text
    readme.write_text("no markers", encoding="utf-8")
    assert figures.update_readme([], {"report": 0, "ablation": 0, "gasweep": 0}) is False
