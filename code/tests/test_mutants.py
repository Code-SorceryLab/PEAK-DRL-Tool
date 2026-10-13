"""Game mutants (code/games/mutants.py): one change each, a class the route check backs,
and a game that loads them when PEAK_MM1_MUTANT names one."""
import os

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from code.games import mutants as mm
from code.games.megaman_core import MUTANT_ENV, MegamanCore


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("mutants"))
    return out, mm.generate("CutMan", out), mm.Stage("CutMan")


def test_one_mutant_per_operator_and_section_or_a_reason(generated):
    _, index, stage = generated
    level = [(m["section"], m["operator"]) for m in index["mutants"] if m["family"] == "level"]
    skipped = [(s["section"], s["operator"]) for s in index["skipped"]]
    assert len(set(level)) == len(level) and not set(level) & set(skipped)
    assert set(level) | set(skipped) == {(name, op) for name, _ in stage.sections for op in stage.game.level_operators}
    assert all(s["reason"] for s in index["skipped"])
    assert {m["class"] for m in index["mutants"]} == {"breaking", "degrading", "neutral"}


def test_a_level_mutant_differs_from_the_stage_only_in_its_recorded_cells(generated):
    out, index, stage = generated
    for m in index["mutants"]:
        if m["family"] != "level":
            assert m["level_file"] is None and len(m["config"]) == 1          # one value, no level change
            continue
        with open(os.path.join(out, m["id"], "level.txt"), encoding="utf-8") as f:
            text = f.read().split("\n")
        diff = {(c, r) for r in range(max(len(text), len(stage.text))) for c in range(stage.ld.cols)
                if (text[r][c] if r < len(text) and c < len(text[r]) else " ") != stage.char(c, r)}
        assert diff == {(c, r) for c, r, _, _ in m["cells"]} and diff, m["id"]
        assert all(stage.char(c, r) == before and before != after for c, r, before, after in m["cells"]), m["id"]


def test_the_class_is_what_the_route_check_says(generated):
    out, index, stage = generated
    for m in index["mutants"]:
        if m["family"] != "level":
            continue
        with open(os.path.join(out, m["id"], "level.txt"), encoding="utf-8") as f:
            text = [r.rstrip("\n") for r in f]
        reachable = stage.reachable(text)[0]
        assert (m["class"] == "breaking") == (not reachable), m["id"]
        if m["operator"] == "seal_route":
            assert m["class"] == "breaking", m["id"]
        if m["operator"] in mm.NEUTRAL_BY_DESIGN:
            assert m["class"] == "neutral", m["id"]
        if m["operator"] == "cosmetic_tile":                                   # the same game, tile for tile
            assert stage.parse(text).grid == stage.ld.grid


def test_generating_twice_gives_the_same_mutants(generated, tmp_path):
    _, index, _ = generated
    assert mm.generate("CutMan", str(tmp_path))["mutants"] == index["mutants"]


def test_the_game_plays_the_mutant_named_in_the_environment(generated, monkeypatch):
    out, index, _ = generated
    base = MegamanCore(level="CutMan")
    assert base.mutant is None
    sealed = next(m for m in index["mutants"] if m["operator"] == "seal_route")
    monkeypatch.setenv(MUTANT_ENV, os.path.join(out, sealed["id"]))
    core = MegamanCore(level=sealed["section"])                               # a section of the mutated stage
    c, r, _, after = sealed["cells"][0]
    assert core.level_file().startswith(out) and core.level_data.grid[r][c] == after != base.level_data.grid[r][c]
    slow = next(m for m in index["mutants"] if m["id"] == "m-gravity_up")
    monkeypatch.setenv(MUTANT_ENV, os.path.join(out, slow["id"], "mutant.json"))
    core = MegamanCore(level="CutMan")
    assert core.cfg["player"]["gravity"] == slow["parameter"]["to"] != base.cfg["player"]["gravity"]
    assert core.level_data.grid == base.level_data.grid
