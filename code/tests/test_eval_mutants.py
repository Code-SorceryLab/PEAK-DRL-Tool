"""The AST evaluation mutants (code/games/eval_mutants.py): the spec's cells or nothing, and a game
that plays the copy named in PEAK_MUTANT (code/games/mutants.py)."""
import os

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from code.games import eval_mutants as em
from code.games import mutants


@pytest.fixture(scope="module", params=["mario", "meatboy", "bomberman"])
def built(request, tmp_path_factory):
    out = str(tmp_path_factory.mktemp(request.param))
    return request.param, out, em.build(request.param, out)


def test_five_mutants_one_per_operator_with_the_spec_class(built):
    game, _, index = built
    got = [(m["id"], m["class"], m["retrain"]) for m in index["mutants"]]
    assert got == [(mid, cls, mid != "cosmetic_tile") for mid, cls, *_ in em.CHANGES[game][1]]
    assert [m["class"] for m in index["mutants"]].count("breaking") == 1


def test_a_level_mutant_changes_exactly_its_recorded_cells(built):
    game, out, index = built
    with open(os.path.join(em.LEVELS_DIR, em.CHANGES[game][0]), encoding="utf-8") as f:
        base = f.read().rstrip("\n").split("\n")
    for m in index["mutants"]:
        if m["family"] != "level":
            assert m["level_file"] is None and m["config"]
            continue
        with open(os.path.join(out, m["id"], "level.txt"), encoding="utf-8") as f:
            text = f.read().rstrip("\n").split("\n")
        assert [len(r) for r in text] == [len(r) for r in base]
        diff = {(c, r) for r in range(len(base)) for c in range(len(base[r])) if text[r][c] != base[r][c]}
        assert diff == {(c, r) for c, r, _, _ in m["cells"]}, m["id"]
        assert all(base[r][c] == before != after == text[r][c] for c, r, before, after in m["cells"])


def test_the_checks_back_each_class(built):
    _, _, index = built
    for m in index["mutants"]:
        ck = m["checks"]
        assert ck["exit_reachable"] == (m["class"] != "breaking"), m["id"]
        if m["operator"] == "cosmetic_tile":
            assert ck["same_level"] is True
        if m["operator"] == "off_route":
            assert ck["walled_in"] is True


def test_a_level_that_no_longer_matches_the_spec_is_refused(monkeypatch, tmp_path):
    file, specs = em.CHANGES["bomberman"]
    wrong = [("off_route", "neutral", [(11, 5, ".", "#")], "a cell that is not what the spec says", None)]
    monkeypatch.setitem(em.CHANGES, "bomberman", (file, wrong))
    with pytest.raises(SystemExit, match="refused"):
        em.build("bomberman", str(tmp_path))


def test_the_games_play_the_mutant_named_in_the_environment(built, monkeypatch):
    game, out, _ = built
    from code.neuro.adapters import make_adapter
    level = em.EVAL_LEVELS[game][0]
    mech = next(m for m in em.CHANGES[game][1] if isinstance(m[2][0][0], str))
    monkeypatch.setenv(mutants.ENV, os.path.join(out, mech[0]))
    core = make_adapter(game, level, 600, 0.0).core
    assert {k: em.EFFECTIVE[k](core) for k, _, _ in mech[2]} == {k: after for k, _, after in mech[2]}
    monkeypatch.setenv(mutants.ENV, os.path.join(out, "seal_route"))
    sealed = make_adapter(game, level, 600, 0.0)
    assert sealed.core.mutant["id"] == "seal_route" and not em.exit_reachable(sealed)
    monkeypatch.delenv(mutants.ENV)
    assert make_adapter(game, level, 600, 0.0).core.mutant is None


def test_another_games_mutant_is_ignored(built, monkeypatch):
    game, out, _ = built
    monkeypatch.setenv(mutants.ENV, os.path.join(out, "seal_route"))
    other = "bomberman" if game != "bomberman" else "meatboy"
    assert mutants.active(game)["id"] == "seal_route" and mutants.active(other) is None


def test_config_paths_index_lists_and_refuse_typos():
    cfg = {"jump": {"jump_vel": 480.0}, "levels": [{"file": "a"}, {"file": "b"}]}
    mutants.apply_config(cfg, {"config": {"jump.jump_vel": 408.0, "levels.1.fuse": 150}})
    assert cfg == {"jump": {"jump_vel": 408.0}, "levels": [{"file": "a"}, {"file": "b", "fuse": 150}]}
    with pytest.raises(KeyError):
        mutants.apply_config(cfg, {"config": {"jumps.jump_vel": 1.0}})
