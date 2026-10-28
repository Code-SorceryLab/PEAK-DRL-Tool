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
    from code.games.eval_mutants import EVAL_LEVELS
    lv = EVAL_LEVELS["bomberman"][0]
    for tag, wr in (("p10g200", 0.4), ("p30g200_best", 0.6), ("p10g40", 0.9)):
        with open(tmp_path / f"report_bomberman_novice_{tag}.json", "w", encoding="utf-8") as f:
            json.dump({"tag": tag, "levels": [{"level": lv, "win_rate_mean": wr}]}, f)
    monkeypatch.setattr(balance, "BALANCE_DIR", str(tmp_path))
    assert ast_eval.baseline_tag("bomberman", "novice") == "p30g200_best"
    dirs = ast_eval.agent_dirs("bomberman", "base", "novice")
    assert dirs == [os.path.join("runs", "probes", "bomberman", "novice", "p30g200_best", f"{lv}_{s}") for s in ast_eval.SEEDS]


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


def test_mega_man_mutants_meet_the_retrained_original_section_by_section(tmp_path, monkeypatch):
    from code.stats import ast_page, episode_log
    monkeypatch.setattr(am, "ROOT", str(tmp_path))

    def rows(level, status="DEAD"):
        return [row(status, level=level, seed=s, game="megaman", end=(1200.0, 1100.0)) for s in (1, 2, 3)]

    def write(variant, *parts):
        episode_log.append(str(tmp_path / "megaman" / variant / "agents.csv"), [r for p in parts for r in p])
    write("base", rows("CutMan"), rows("CutMan-3-Ledges", "WON"))           # the relay's hand-picked legs
    write("base-retrained", rows("CutMan"), rows("CutMan-1-Street", "WON"), rows("CutMan-3-Ledges"))
    write("s3-off_route", rows("CutMan-3-Ledges"))
    write("m-jump_velocity_down", rows("CutMan"), rows("CutMan-1-Street"), rows("CutMan-3-Ledges"))
    res = am.evaluate("megaman")
    off = res["s3-off_route"]                                              # quiet: the relay's 3/3 is not its baseline
    assert off["vs"] == "base-retrained @ CutMan-3-Ledges" and not off["vs_base"]["detected"]
    street = res["m-jump_velocity_down @ CutMan-1-Street"]
    assert street["vs"] == "base-retrained @ CutMan-1-Street" and street["vs_base"]["completion_lost"]
    assert res["m-jump_velocity_down"]["vs"] == "base-retrained" and "vs_base" not in res["base @ CutMan-3-Ledges"]
    jump = {"id": "m-jump_velocity_down", "class": "degrading", "operator": "m-jump_velocity_down"}
    assert ast_page.judge(jump, res) == ("caught in 1/1 sections", "good")   # Ledges: its original never wins
    assert not off["vs_base"]["counted"] and ast_page.verdict(
        {"class": "neutral", "operator": "s3-off_route", "id": "s3-off_route", "game": "megaman"}, off) == ("original unsolved", "none")


def test_a_variant_meets_only_the_personas_that_played_it(tmp_path, monkeypatch):
    from code.stats import episode_log
    monkeypatch.setattr(am, "ROOT", str(tmp_path))

    def rows(persona, status):
        return [row(status, persona=persona, level="CutMan-4-Shaft", seed=s, game="megaman", end=(1200.0, 1100.0))
                for s in (1, 2, 3)]
    episode_log.append(str(tmp_path / "megaman" / "base-retrained" / "agents.csv"),
                       rows("experienced", "DEAD") + rows("killer", "WON"))      # only killer solves Shaft
    episode_log.append(str(tmp_path / "megaman" / "base-reseeded" / "agents.csv"), rows("experienced", "DEAD"))
    res = am.evaluate("megaman")
    s = (res.get("base-reseeded @ CutMan-4-Shaft") or res["base-reseeded"])["vs_base"]
    assert not s["counted"] and s["verdict"] == "not counted"    # experienced never solved it: no false alarm


def test_a_comparison_counts_only_on_a_solved_original_and_gets_one_verdict():
    solved = am.summarize([row("WON") for _ in range(6)] + [row(end=(64.0, 96.0))])
    unsolved = am.summarize([row(end=(64.0, 96.0)) for _ in range(7)])
    moved = am.summarize([row("WON") for _ in range(6)] + [row(end=(640.0, 96.0))])
    assert am.compare(solved, unsolved)["verdict"] == "broken"
    assert am.compare(solved, moved)["verdict"] == "changed"
    assert am.compare(solved, solved)["verdict"] == "unchanged"
    s = am.compare(unsolved, am.summarize([row(cause="Enemy", end=(900.0, 96.0)) for _ in range(7)]))
    assert s["cause_changed"] and (s["counted"], s["detected"], s["verdict"]) == (False, False, "not counted")


def test_the_seed_spread_check_reads_each_agents_training_seed():
    def agent(status, seed):
        return row(status, run=f"runs/probes/mario/novice/p10g200/Mario1-1_{seed}")
    base = am.summarize([agent("WON", 1234), agent("WON", 1234), agent("WON", 2025), agent("DEAD", 2025)])
    assert base["seed_win_rates"] == {"1234": [2, 2], "2025": [1, 2]}       # 100% and 50%
    assert am.compare(base, am.summarize([agent("DEAD", 7)] * 4))["win_rate_beyond_seeds"]
    assert not am.compare(base, am.summarize([agent("WON", 7)] * 3 + [agent("DEAD", 7)]))["win_rate_beyond_seeds"]
    assert am.compare(am.summarize([row("WON")]), base)["win_rate_beyond_seeds"] is None   # one seed: no spread


def test_false_alarms_count_the_reseeded_original_where_its_original_is_solved():
    fired = {"counted": True, "completion_lost": False, "win_rate_changed": False, "cluster_moved": True,
             "cause_changed": False, "detected": True, "win_rate_beyond_seeds": False}
    quiet = dict(fired, cluster_moved=False, detected=False, win_rate_beyond_seeds=None)
    results = {"mario": {"base-reseeded": {"vs_base": fired}, "seal_route": {"vs_base": fired}},
               "megaman": {"base-reseeded @ CutMan-1-Street": {"vs_base": quiet},
                           "base-reseeded @ CutMan-4-Shaft": {"vs_base": dict(fired, counted=False)}}}
    fa = am.false_alarms(results)
    assert fa["comparisons"] == 2 and fa["fired"]["cluster_moved"] == 1 and fa["fired"]["detected"] == 1
    assert fa["beyond_seeds"] == [0, 1]


def test_replays_sit_beside_the_retrained_attempts(tmp_path, monkeypatch):
    from code.stats import episode_log
    monkeypatch.setattr(am, "ROOT", str(tmp_path))
    episode_log.append(str(tmp_path / "mario" / "base" / "agents.csv"), [row("WON", level="Mario1-1") for _ in range(3)])
    episode_log.append(str(tmp_path / "mario" / "widen_gap" / "agents.csv"), [row("WON", level="Mario1-1") for _ in range(3)])
    episode_log.append(str(tmp_path / "mario" / "widen_gap" / "agents_replay.csv"), [row(level="Mario1-1") for _ in range(3)])
    e = am.evaluate("mario")["widen_gap"]
    assert e["vs_base"]["verdict"] == "unchanged" and e["replay"]["vs_base"]["verdict"] == "broken"
    assert e["agents"]["attempts"] == 3 and e["replay"]["agents"]["attempts"] == 3


def test_the_reseeded_original_trains_with_new_seeds():
    from code.neuro import ast_eval
    m = ast_eval.mutant("mario", ast_eval.RESEEDED)
    assert m["retrain"] and m["dir"] is None and m["retrain_levels"] == ["Mario1-1"]
    assert not set(m["seeds"]) & set(ast_eval.SEEDS)
    cmd, env = ast_eval.retrain_job("mario", ast_eval.RESEEDED, "novice")
    assert cmd[cmd.index("--seeds") + 1:][:3] == list(map(str, ast_eval.RESEEDS)) and env["PEAK_MUTANT"] == ""
    assert len(ast_eval.mutant("megaman", ast_eval.RESEEDED)["retrain_levels"]) == 8


def test_mega_mans_original_is_retrained_with_the_mutants_command():
    from code.neuro import ast_eval
    m = ast_eval.mutant("megaman", "base-retrained")
    assert m["dir"] is None and len(m["retrain_levels"]) == 8                # every section of the stage
    cmd, env = ast_eval.retrain_job("megaman", "base-retrained", "experienced", levels=["CutMan-3-Ledges"])
    arg = {k: cmd[cmd.index(k) + 1] for k in ("--levels", "--gens", "--root")}
    assert env["PEAK_MUTANT"] == "" and cmd[cmd.index("--seeds") + 1:][:3] == list(map(str, ast_eval.SEEDS))
    assert arg == {"--levels": "CutMan-3-Ledges", "--gens": "200",
                   "--root": os.path.join("runs", "ast", "megaman", "base-retrained", "probes")}


def test_a_level_variant_is_compared_with_the_base_of_its_own_level(tmp_path, monkeypatch):
    from code.stats import episode_log
    monkeypatch.setattr(am, "ROOT", str(tmp_path))
    won = [row("WON", game="meatboy", level="5", seed=s) for s in (1, 2, 3)]
    dead = [row(cause="Stall", end=(900.0, 400.0), game="meatboy", level="5", seed=s) for s in (1, 2, 3)]
    episode_log.append(str(tmp_path / "meatboy" / "base" / "agents.csv"), [row("WON", game="meatboy", level="2")])
    episode_log.append(str(tmp_path / "meatboy" / "base@5" / "agents.csv"), won)
    episode_log.append(str(tmp_path / "meatboy" / "base-reseeded@5" / "agents.csv"), [dict(r) for r in won])
    episode_log.append(str(tmp_path / "meatboy" / "seal_route@5" / "agents.csv"), dead)
    res = am.evaluate("meatboy")
    assert ["base", "base@5", "base-reseeded@5", "seal_route@5"] == [v for v in res]
    assert res["seal_route@5"]["vs"] == "base@5" and res["seal_route@5"]["vs_base"]["verdict"] == "broken"
    assert res["base-reseeded@5"]["vs"] == "base@5" and "vs_base" not in res["base@5"]
    assert am.false_alarms({"meatboy": res})["comparisons"] == 1
