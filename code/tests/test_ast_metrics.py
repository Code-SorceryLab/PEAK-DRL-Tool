"""The AST evaluation metrics (code/stats/ast_metrics.py) and the runner's choice of agents
(code/neuro/ast_eval.py)."""
import json
import os

from code.stats import ast_metrics as am


def row(status="DEAD", end=(320.0, 96.0), cause="Pit", persona="novice", **kw):
    r = {"source": "agent", "game": "mario", "persona": persona, "status": status, "cause": "" if status == "WON" else cause,
         "end_x": end[0], "end_y": end[1], "tile": 32, "time_s": 40.0, "frames": 2400, "progress": 1.0 if status == "WON" else 0.5,
         "jumps": 10, "kills": 1, "coins": 2, "route": [[16.0, 96.0], [end[0], end[1]]], "run": f"r/{persona}/{kw.get('seed', 1)}"}
    r.update(kw)
    return r


def test_fisher_exact_matches_the_hypergeometric_tail():
    assert abs(am.fisher_exact(3, 0, 0, 3) - 0.1) < 1e-9          # 1/20 + 1/20
    assert am.fisher_exact(2, 1, 2, 1) == 1.0
    assert am.fisher_exact(0, 0, 0, 0) == 1.0
    assert am.fisher_exact(18, 0, 0, 18) < 1e-9


def test_the_death_cluster_merges_end_tiles_within_three():
    c = am.death_cluster([(10, 5), (11, 5), (13, 7), (30, 5)])
    assert (c["attempts"], c["share"]) == (3, 0.75) and abs(c["col"] - 11) <= 1
    assert am.death_cluster([]) is None


def test_a_route_covers_the_tiles_between_its_samples():
    tiles = am.route_tiles({"tile": 32, "route": [[0.0, 0.0], [128.0, 0.0]]})
    assert {(c, 0) for c in range(5)} <= tiles


def test_summarize_reads_the_spec_metrics():
    rows = [row("WON", time_s=30.0), row("WON", time_s=50.0), row(cause="Pit"), row(cause="Enemy", end=(330.0, 100.0))]
    s = am.summarize(rows)
    assert s["completion"] and s["win_rate"]["pooled"] == [2, 4] and s["win_rate"]["novice"] == [2, 4]
    assert s["time_to_finish"] == 40.0 and s["progress_median"] == 0.5
    assert s["causes"] == {"Pit": 0.5, "Enemy": 0.5} and s["end_tiles"] == [(10, 3), (10, 3)]
    assert s["effort"] == {"jumps": 10.0, "kills": 1.0, "pickups": 2.0}
    assert [r["w"] for r in s["routes"]] == [True, True, False, False] and s["routes"][2]["p"][-1] == [10.0, 3.0]
    bomb = am.summarize([dict(row(), game="bomberman", bombs_placed=3)])
    assert "jumps" not in bomb["effort"] and bomb["effort"]["bombs placed"] == 3.0


def test_compare_raises_the_signals_a_sealed_level_should():
    base = am.summarize([row("WON") for _ in range(6)] + [row(end=(64.0, 96.0))])
    sealed = am.summarize([row(cause="Stall", end=(2880.0, 384.0)) for _ in range(7)])
    s = am.compare(base, sealed)
    assert s["completion_lost"] and s["win_rate_changed"] and s["cluster_moved"] and s["cause_changed"] and s["detected"]
    assert not am.compare(base, base)["detected"]


def test_the_cosmetic_replay_must_match_attempt_for_attempt():
    base = [row(seed=1), row("WON", seed=2)]
    assert am.determinism(base, [dict(r) for r in base])["passed"]
    off = [dict(base[0], frames=2401), dict(base[1])]
    d = am.determinism(base, off)
    assert not d["passed"] and d["identical"] == 1


def test_the_baseline_is_the_200_generation_setting_that_won_more(tmp_path, monkeypatch):
    from code.neuro import ast_eval, balance
    for tag, wr in (("p10g200", 0.4), ("p30g200_best", 0.6), ("p10g40", 0.9)):
        with open(tmp_path / f"report_bomberman_novice_{tag}.json", "w", encoding="utf-8") as f:
            json.dump({"tag": tag, "levels": [{"level": "8", "win_rate_mean": wr}]}, f)
    monkeypatch.setattr(balance, "BALANCE_DIR", str(tmp_path))
    assert ast_eval.baseline_tag("bomberman", "novice") == "p30g200_best"
    dirs = ast_eval.agent_dirs("bomberman", "base", "novice")
    assert dirs == [os.path.join("runs", "probes", "bomberman", "novice", "p30g200_best", f"8_{s}") for s in ast_eval.SEEDS]


def test_each_variant_is_compared_with_the_original_played_over_the_same_level(tmp_path, monkeypatch):
    from code.stats import episode_log
    monkeypatch.setattr(am, "ROOT", str(tmp_path))
    stage = [row(level="CutMan", seed=s, game="megaman") for s in (1, 2)]
    ledges = [row(level="CutMan-3-Ledges", seed=s, game="megaman", end=(1200.0, 1100.0)) for s in (1, 2)]
    episode_log.append(str(tmp_path / "megaman" / "base" / "agents.csv"), stage + ledges)
    episode_log.append(str(tmp_path / "megaman" / "s3-cosmetic_tile" / "agents.csv"), [dict(r) for r in ledges])
    episode_log.append(str(tmp_path / "megaman" / "s3-seal_route" / "human" / "megaman" / "al" / "episodes.csv"),
                       [dict(ledges[0], source="human", persona="al")])
    res = am.evaluate("megaman")
    assert {"base", "base @ CutMan-3-Ledges", "s3-cosmetic_tile", "s3-seal_route"} <= set(res)
    assert res["base"]["agents"]["attempts"] == 2 and res["base @ CutMan-3-Ledges"]["agents"]["attempts"] == 2
    assert res["s3-cosmetic_tile"]["determinism"]["passed"]            # matched against the section's own run
    assert res["s3-seal_route"]["agents"] is None and res["s3-seal_route"]["humans"]["win_rate"]["al"] == [0, 1]
