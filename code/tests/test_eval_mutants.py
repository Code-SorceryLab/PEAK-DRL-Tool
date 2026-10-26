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


# ── the same five operators placed by program on every other level ─────────────────────────────
OTHER = {"mario": "Mario1-2", "meatboy": "5", "bomberman": "3"}


@pytest.fixture(scope="module", params=list(OTHER))
def placed(request, tmp_path_factory):
    game = request.param
    out = str(tmp_path_factory.mktemp(f"{game}-levels"))
    return game, OTHER[game], out, em.build(game, out, level=OTHER[game])


def test_a_bomberman_exit_behind_bricks_counts_as_reachable():
    from code.neuro.adapters import make_adapter
    assert em.exit_reachable(make_adapter("bomberman", "3", 600, 0.0))     # 04 One Wall: bricks across the route


def test_another_level_gets_the_five_operators_placed_and_checked(placed):
    game, level, _, index = placed
    ops = [m["operator"] for m in index["mutants"]] + [s["operator"] for s in index["skipped"]]
    assert sorted(ops) == sorted(mid for mid, *_ in em.CHANGES[game][1])
    for m in index["mutants"]:
        assert m["id"] == f"{m['operator']}@{level}" and m["stage"] == level, m["id"]
        assert m["checks"]["exit_reachable"] == (m["class"] != "breaking"), m["id"]
    assert all(s["reason"] for s in index["skipped"])


def test_a_placed_mutant_changes_only_its_recorded_cells(placed):
    game, level, out, index = placed
    with open(os.path.join(em.LEVELS_DIR, em.level_file(game, level)), encoding="utf-8") as f:
        base = f.read().rstrip("\n").split("\n")
    for m in index["mutants"]:
        if m["family"] != "level":
            continue
        with open(os.path.join(out, m["id"], "level.txt"), encoding="utf-8") as f:
            text = f.read().rstrip("\n").split("\n")
        diff = {(c, r) for r in range(len(base)) for c in range(len(base[r])) if text[r][c] != base[r][c]}
        assert diff == {(c, r) for c, r, _, _ in m["cells"]}, m["id"]


def test_the_game_plays_a_placed_mutant_on_its_own_level(placed, monkeypatch):
    game, level, out, index = placed
    from code.neuro.adapters import make_adapter
    seal = next(m for m in index["mutants"] if m["operator"] == "seal_route")
    monkeypatch.setenv(mutants.ENV, os.path.join(out, seal["id"]))
    played = make_adapter(game, level, 600, 0.0)
    assert played.core.mutant["id"] == seal["id"] and not em.exit_reachable(played)


def test_every_other_level_gets_a_wall_a_mechanics_change_and_a_cosmetic_copy(tmp_path):
    mechanics = {"mario": "jump_velocity_down", "meatboy": "jump_velocity_down", "bomberman": "fuse_down"}
    for game in OTHER:
        levels = em.other_levels(game)
        assert levels and em.EVAL_LEVELS[game][0] not in levels
        for level in levels:
            got = {m["operator"] for m in em.build(game, str(tmp_path / game), level=level)["mutants"]}
            assert {"seal_route", mechanics[game], "cosmetic_tile"} <= got, (game, level, got)
