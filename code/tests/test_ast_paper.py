"""The AST paper's analyses (code/stats/ast_paper.py): distances to the change, what play reached,
and the tallies the tables print."""
from code.stats import ast_metrics as am
from code.stats import ast_paper as ap


def test_distances_to_the_change_are_chebyshev_in_tiles():
    cells = [(10, 5), (11, 5)]
    assert ap.nearest(cells, (14, 7)) == 3 and ap.nearest(cells, None) is None and ap.nearest([], (1, 1)) is None
    assert ap.share_near(cells, [(12, 5), (13, 8), (30, 5), (11, 9)]) == 0.5
    assert ap.reached(cells, [(12, 6)]) and not ap.reached(cells, [(14, 5)]) and ap.reached([], [(1, 1)]) is None


def test_a_method_is_right_when_it_flags_a_change_and_leaves_a_neutral_alone():
    assert ap.right("degrading", True) and not ap.right("degrading", False)
    assert ap.right("neutral", False) and not ap.right("neutral", True) and ap.right("breaking", None) is None
    rows = [{"class": "breaking", "operator": "seal_route", "m": True},
            {"class": "degrading", "operator": "widen_gap", "m": False},
            {"class": "neutral", "operator": "off_route", "m": True},
            {"class": "neutral", "operator": "cosmetic_tile", "m": True},       # judged by replay identity, not here
            {"class": "degrading", "operator": "fuse_down", "m": None}]          # not judged
    assert ap.tally(rows, "m") == {"caught": [1, 2], "quiet": [1, 1]}


def test_the_strict_check_keeps_completion_and_fisher_on_counted_levels():
    s = {"counted": True, "completion_lost": False, "win_rate_changed": False, "cluster_moved": True}
    m = {"id": "m-jump"}
    res = {"m-jump @ A": {"vs_base": s}, "m-jump @ B": {"vs_base": dict(s, counted=False, completion_lost=True)}}
    assert ap.strict_flag(m, res, "retrain") is False          # B does not count, A only moved its cluster
    res["m-jump @ A"]["vs_base"] = dict(s, win_rate_changed=True)
    assert ap.strict_flag(m, res, "retrain") is True
    assert ap.strict_flag(m, res, "replay") is None           # no replay played
    assert ap.strict_flag({"id": "x"}, {"x": {"vs_base": dict(s, completion_lost=True)}}, "retrain") is True


def test_a_persona_is_judged_alone_by_its_worst_counted_section(tmp_path, monkeypatch):
    from code.stats import episode_log
    from code.tests.test_ast_metrics import row
    monkeypatch.setattr(am, "ROOT", str(tmp_path))
    monkeypatch.setattr(ap.eval_mutants, "eval_set", lambda game: [{"id": "m-jump", "class": "degrading", "operator": "m-jump"}])

    def put(variant, persona, level, status):
        episode_log.append(str(tmp_path / "megaman" / variant / "agents.csv"),
                           [row(status, persona=persona, level=level, game="megaman")] * 6)
    put("base-retrained", "novice", "A", "WON")
    put("base-retrained", "novice", "B", "DEAD")               # B unsolved by novice: does not count
    put("base-retrained", "killer", "A", "WON")
    put("m-jump", "novice", "A", "DEAD")                       # completion lost on A
    put("m-jump", "novice", "B", "DEAD")
    put("m-jump", "killer", "A", "WON")
    r = ap.personas({"megaman": am.evaluate("megaman")})[0]
    assert r[("retrain", "novice")] == "broken" and r[("retrain", "killer")] == "unchanged"
    assert ("replay", "novice") not in r                       # nothing replayed
    assert ap.worst(["unchanged", "not counted", "changed"]) == "changed" and ap.worst(["not counted"]) == "not counted"


def test_the_random_baseline_is_compared_with_itself_on_the_original(tmp_path, monkeypatch):
    from code.stats import episode_log
    from code.tests.test_ast_metrics import row
    monkeypatch.setattr(am, "ROOT", str(tmp_path))
    episode_log.append(str(tmp_path / "mario" / "base" / "agents.csv"), [row("WON", level="Mario1-1")] * 3)
    episode_log.append(str(tmp_path / "mario" / "base" / "agents_random.csv"), [row(level="Mario1-1")] * 3)
    episode_log.append(str(tmp_path / "mario" / "seal_route" / "agents.csv"), [row(level="Mario1-1")] * 3)
    episode_log.append(str(tmp_path / "mario" / "seal_route" / "agents_random.csv"),
                       [row(level="Mario1-1", cause="Enemy", end=(900.0, 96.0))] * 3)
    rnd = am.evaluate("mario")["seal_route"]["random"]["vs_base"]
    assert rnd["cause_changed"] and not rnd["counted"]          # it never wins, so it never counts


def test_persona_flags_leave_out_comparisons_that_do_not_count():
    per = [{"class": "breaking", ("replay", "clumsy"): "not counted", ("replay", "killer"): "broken"},
           {"class": "breaking", ("replay", "clumsy"): "broken", ("replay", "killer"): "unchanged"},
           {"class": "neutral", ("replay", "clumsy"): "changed"}]          # neutral: a false alarm, not a flag
    # ast_rule.md: an unsolved original ("not counted") is left out of every tally, its denominator too
    assert ap.persona_flags(per, "replay", ["clumsy", "killer"]) == {"clumsy": (1, 1), "killer": (1, 2)}
