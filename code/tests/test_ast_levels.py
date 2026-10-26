"""AST evaluation on every level (code/neuro/ast_eval.py): a variant named <id>@<level> is retrained,
played and compared on its own level, against the baseline agents of that level."""
import os

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from code.games import eval_mutants as em
from code.neuro import ast_eval, balance


@pytest.fixture(scope="module")
def on_disk(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("mutants"))
    em.build("meatboy", out, level="5")
    return out


@pytest.fixture
def placed(on_disk, monkeypatch):
    monkeypatch.setattr(em, "out_dir", lambda game: on_disk if game == "meatboy" else os.path.join("runs", game, "mutants"))


def test_every_level_with_mutants_gets_its_own_base_and_false_alarm_check(placed):
    vs = ast_eval.variants("meatboy")
    assert vs[:2] == ["base", "base-reseeded"]
    assert {"base@5", "base-reseeded@5", "seal_route@5", "cosmetic_tile@5"} <= set(vs)
    assert vs.index("base@5") < vs.index("base-reseeded@5") < vs.index("seal_route@5")
    assert ast_eval.level_of("meatboy", "seal_route@5") == "5" and ast_eval.level_of("meatboy", "seal_route") == "2"


def test_a_placed_mutant_retrains_on_its_own_level_with_that_levels_ga_setting(placed):
    cmd, env = ast_eval.retrain_job("meatboy", "seal_route@5", "experienced")
    assert cmd[cmd.index("--levels") + 1:cmd.index("--seeds")] == ["5"]
    assert env["PEAK_MUTANT"].endswith("seal_route@5")
    assert cmd[cmd.index("--root") + 1] == os.path.join("runs", "ast", "meatboy", "seal_route@5", "probes")
    assert ("--best" in cmd) == ast_eval.baseline_tag("meatboy", "experienced", "5").endswith("_best")


def test_the_base_of_a_level_is_that_levels_baseline_agents(placed):
    tag = ast_eval.baseline_tag("meatboy", "experienced", "5")
    assert ast_eval.agent_dirs("meatboy", "base@5", "experienced") == [
        balance.probe_dir("meatboy", "experienced", tag, "5", s) for s in ast_eval.SEEDS]
    m = ast_eval.mutant("meatboy", "base-reseeded@5")
    assert m["retrain_levels"] == ["5"] and m["seeds"] == ast_eval.RESEEDS
    assert ast_eval.mutant("meatboy", "base@5") is None


def test_an_unknown_level_variant_is_refused(placed):
    with pytest.raises(SystemExit):
        ast_eval.mutant("meatboy", "seal_route@99")


def test_a_level_variant_is_played_on_its_level_and_replayed_by_that_levels_base(placed, monkeypatch, tmp_path):
    seen = []

    def fake(game, run, level, seed=0):
        seen.append((os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(run)))), level, os.path.basename(run)))
        return {"status": "WON", "cause": None, "progress": 1.0, "time_s": 1.0, "persona": "experienced"}
    monkeypatch.setattr(ast_eval, "play_agent", fake)
    monkeypatch.setattr(ast_eval, "ROOT", str(tmp_path))
    monkeypatch.setattr(ast_eval.os.path, "exists", lambda p: True)
    monkeypatch.setattr(ast_eval.episode_log, "reset", lambda p: None)
    monkeypatch.setattr(ast_eval.episode_log, "append", lambda p, rows: None)
    ast_eval.play("meatboy", "base@5", ["experienced"])
    ast_eval.play("meatboy", "seal_route@5", ["experienced"], replay=True)
    assert {lv for _, lv, _ in seen} == {"5"}
    assert all(run.startswith("5_") for _, _, run in seen) and len(seen) == 2 * len(ast_eval.SEEDS)
    assert ast_eval.play("meatboy", "base-reseeded@5", ["experienced"], replay=True).startswith("skip")


def test_the_level_tasks_retrain_every_persona_on_each_retrained_level_variant(placed):
    tasks = [t for t in ast_eval.level_tasks(["meatboy"])]
    variants = {v for _, v, _ in tasks}
    assert "base-reseeded@5" in variants and "seal_route@5" in variants
    assert not {"base@5", "cosmetic_tile@5", "seal_route", "base-reseeded"} & variants
    assert len(tasks) == len(variants) * len(ast_eval.PERSONA_ORDER)
