"""GA-sweep page helpers on synthetic report dicts — no disk, no pygame."""
from dataclasses import asdict

from code.neuro.evolution import GAConfig
from code.neuro.gasweep import sweep_tag
from code.neuro.report import _gasweep_groups, _hidden_verdict, _sweep_point


def _rep(suffix, ov, wr, fw, seeds=(1, 2)):
    return {"game": "mario", "persona": "experienced", "tag": sweep_tag(10, 40, "rays", suffix),
            "gens_budget": 40, "seeds": list(seeds), "train_time_s": 5.0,
            "ga_config": asdict(GAConfig(**ov)),
            "levels": [{"level": "L", "solved_by": int(wr > 0), "best_gen_mean": 3.0,
                        "improvement_rate_mean": 0.01, "win_rate_mean": wr}],
            "cells": {"L": [{"seed": s, "win_rate": wr, "first_win_gen": fw, "curve": [[1, 0]]} for s in seeds]}}


def test_groups_put_baseline_first_in_axis_order():
    datas = [_rep("memory-2", {"memory": 2}, 0.1, 5), _rep("hidden-8", {"hidden": 8}, 0.2, 4),
             _rep("base", {}, 0.3, 3), {"game": "mario", "tag": "p10g40"}]   # last one: not a sweep tag
    groups = _gasweep_groups(datas)
    from code.neuro.gasweep import base_sig
    assert list(groups) == [("mario", "experienced", 40, "rays", base_sig())]
    from code.neuro.gasweep import parse_sweep_tag
    assert [parse_sweep_tag(d["tag"])[0] for d in groups[list(groups)[0]]] == ["base", "hidden", "memory"]


def test_point_censors_unsolved_at_budget_and_counts_params():
    p = _sweep_point(_rep("hidden-8", {"hidden": 8}, 0.0, None))
    assert p["fw"] == 40 and p["wr"] == 0 and p["n_params"] == 147 and p["label"] == "h8"
    assert p["solved"] == 0 and p["n_levels"] == 1


def test_hidden_verdict():
    flat = [_sweep_point(_rep(s, ov, 0.5, 10)) for s, ov in
            [("base", {}), ("hidden-8", {"hidden": 8}), ("hidden-64", {"hidden": 64})]]
    assert _hidden_verdict(flat)["word"] == "flat"
    better = [_sweep_point(_rep(s, ov, 0.5, fw)) for (s, ov), fw in
              zip([("hidden-8", {"hidden": 8}), ("base", {}), ("hidden-64", {"hidden": 64})], (30, 15, 5))]
    assert _hidden_verdict(better)["word"] == "improves"
    worse = [_sweep_point(_rep(s, ov, 0.5, fw)) for (s, ov), fw in
             zip([("hidden-8", {"hidden": 8}), ("base", {}), ("hidden-64", {"hidden": 64})], (5, 15, 30))]
    assert _hidden_verdict(worse)["word"] == "degrades"
    assert _hidden_verdict(flat[:2])["word"] == "insufficient"


# ── indexed-game level entries: bare path or {file: ..., overrides} ──────────────────

def test_indexed_entry_handles_both_entry_shapes():
    """Bomberman levels 6+ carry per-level overrides as dicts; 0-5 are bare paths.
    Reading only the bare form silently dropped the route/config panels for 6 onward."""
    from code.neuro.report import _indexed_entry, _level_file, _level_grid
    from code.neuro.adapters import list_levels
    for lvl in list_levels("bomberman", include_disabled=True):
        entry = _indexed_entry("bomberman", lvl)
        assert entry.get("file", "").endswith(".txt"), f"level {lvl} has no file: {entry}"
        assert _level_file("bomberman", lvl), f"level {lvl} resolved to no path"
        assert _level_grid("bomberman", lvl), f"level {lvl} drew no grid"


def test_indexed_entry_reaches_disabled_levels_and_rejects_bad_ids():
    """Disabled levels are a suffix, so ids stay stable and old reports still draw them."""
    from code.neuro.report import _indexed_entry
    from code.neuro.adapters import list_levels
    enabled = len(list_levels("bomberman"))
    every = len(list_levels("bomberman", include_disabled=True))
    if every > enabled:                       # a retired rung is still addressable
        assert _indexed_entry("bomberman", str(every - 1)).get("file", "").endswith(".txt")
    assert _indexed_entry("bomberman", str(every)) == {}
    assert _indexed_entry("bomberman", "-1") == {}
    assert _indexed_entry("meatboy", "0").get("file", "").endswith(".txt")   # bare-path game


# ── route traces: episodes.csv where a probe kept one, else the cluster's routes.json sample ──

def test_collect_routes_merges_routes_json_and_csv_and_finds_the_fastest_win(tmp_path, monkeypatch):
    """200-generation probes come back from the cluster with only a compact routes.json; their maps
    must still show wins, deaths and the fastest win (from best_time_s, or a CSV's smallest elapsed_time)."""
    import json
    from code.neuro import report
    monkeypatch.setattr(report, "PROBES_ROOT", str(tmp_path))
    probe = tmp_path / "mario" / "novice" / "p10g200"
    won = [[[0, 400], [100, 380], [200, 400]], [[0, 400], [150, 300], [300, 400]]]
    lost = [[[0, 400], [40, 410]], [[0, 400], [60, 420], [61, 440]]]
    for seed, best, t in (("1234", won[0], 31.8), ("2025", won[1], 24.5)):
        (probe / f"Mario1-1_{seed}").mkdir(parents=True)
        (probe / f"Mario1-1_{seed}" / "routes.json").write_text(json.dumps(
            {"level": "Mario1-1", "persona": "novice", "won": won, "lost": lost, "best": best, "best_time_s": t}))
    routes, best = report._collect_routes("mario", "novice", "Mario1-1", "p10g200")
    assert [r["p"] for r in routes if r["w"]] == won + won           # wins first, merged over seeds in order
    assert [r["p"] for r in routes if not r["w"]] == lost + lost
    assert best == {"p": won[1], "t": 24.5, "seed": "2025"}          # the faster of the two seeds' bests

    # a seed with an episode log is read from it; its quicker win becomes the fastest
    (probe / "Mario1-1_31337").mkdir()
    (probe / "Mario1-1_31337" / "episodes.csv").write_text(
        "persona,game,world,cause_of_death,route,elapsed_time\n"
        'novice,mario,Mario1-1,Success,"[(0.0, 400.0), (90.4, 350.2)]",20.3\n'
        'novice,mario,Mario1-1,Pit,"[(0.0, 400.0), (30.0, 470.0)]",3.0\n', encoding="utf-8")
    routes, best = report._collect_routes("mario", "novice", "Mario1-1", "p10g200")
    assert sum(r["w"] for r in routes) == 5 and len(routes) == 10
    assert best == {"p": [[0, 400], [90, 350]], "t": 20.3, "seed": "31337"}
    assert report._collect_routes("mario", "killer", "Mario1-1", "p10g200") == ([], None)   # nothing on disk
    assert report._pack([[3, 4], [5, 1], [5, 1]]) == [3, 4, 2, -3, 0, 0]   # what the page embeds: steps


def test_mutations_is_a_balance_command_page():
    """AL: the Command Center and the mutation page are one site: every page's top bar links the
    Mutations page, and that page carries the same top bar (marked as the current page)."""
    from code.neuro import report
    from code.stats import ast_page
    for page in ("report", "ablation", "gasweep", "instructions"):
        assert 'href="mutations.html"' in report._nav(["mario"], True, None, page)
    html = ast_page.render({})                       # no playtests yet: the page still renders
    assert report._nav([], False, None, "mutations") in html
    assert 'href="mutations.html" aria-current="page"' in html


# ── verdicts, CIs and the ablation / GA-sweep summaries ──────────────────────────────

def test_b2_reads_deaths_per_win_and_an_old_report_gets_no_b2_verdict():
    from code.neuro.report import _verdicts
    row = {"completion_rate_mean": 0.7, "mean_completion_time": 20, "death_cluster_entropy_mean": 0.4,
           "deaths_per_run_mean": 5.0, "deaths_per_win_mean": 1.0}      # 5 deaths a generation, 1 per win
    assert "B2 punishment: in band" in _verdicts(row, None, {})
    old = {k: v for k, v in row.items() if k != "deaths_per_win_mean"}
    assert "B2" not in _verdicts(old, None, {})


def test_luck_dependent_sentence_quotes_the_winners_own_rate():
    from code.neuro.report import _verdict_sentence
    row = {"win_rate_mean": 0.2, "solved_by": 1, "seeds": 3, "death_hist": [0] * 10,   # seed rates 0.6, 0, 0
           "dominant_cause": "Pit", "dominant_cause_frac": 1.0}
    word, _, sent = _verdict_sentence(row, None)
    assert word == "luck-dependent" and "winners then win 60%" in sent


def test_one_seed_shows_no_interval_rather_than_zero():
    from code.neuro import report
    from code.neuro.balance import aggregate, summarize
    hist = [{"gen": g, "best": 900.0, "avg": 450.0, "best_x": 900.0, "duration": 1.0, "wins": 3, "stuck": 0,
             "envs": [{"status": "WON", "end_x": 900.0, "level_len": 1000.0, "coins": 0, "level_coins": 0,
                       "frames": 600}] * 3} for g in range(1, 12)]
    html = report._metric_table([aggregate([summarize(hist, 3, "L", 1, 900.0, 1)])])
    assert "None" not in html and "<td>1.0</td><td>1.0</td>" in html and "100% ±" not in html   # first win, best gen
    datas = [_rep("base", {}, 0.5, 10, seeds=(1,)), _rep("hidden-8", {"hidden": 8}, 0.6, 8, seeds=(1,))]
    key, group = next(iter(_gasweep_groups(datas).items()))
    html = report._gasweep_section(key, group, 0) + report._gasweep_overview({key: group})
    assert "None" not in html and "± 0" not in html


def test_ablation_first_win_means_count_unsolved_seeds_at_the_budget():
    from code.neuro import report

    def rep(fws, tag):   # one level, three seeds; None = never won in the 40-gen budget
        solved = [fw for fw in fws if fw]
        return {"tag": tag, "gens_budget": 40, "seeds": [1, 2, 3], "pop_size": 10,
                "levels": [{"level": "L", "seeds": 3, "solved_by": len(solved), "first_win_mean": sum(solved) / len(solved),
                            "first_win_ci": 0.0, "win_rate_mean": 0.5, "win_rate_ci": 0.1,
                            "dominant_cause": "Pit", "dominant_cause_frac": 1.0}],
                "cells": {"L": [{"seed": s, "win_rate": 0.5, "first_win_gen": fw} for s, fw in zip((1, 2, 3), fws)]}}
    by_mode = {"rays": rep([5, None, None], "p10g40"), "grid": rep([5, 20, 20], "p10g40_grid")}
    face = report._abl_face(by_mode["rays"], by_mode["grid"])
    assert "gen 28.3" in face and "gen 15.0" in face           # (5 + 40 + 40) / 3 vs (5 + 20 + 20) / 3
    html = report._ablation_section("mario", "novice", "p10g40", by_mode, 0)
    assert "gen 28.3" in html and "+10.0 gens" not in html      # 5.0 vs 15.0 would compare different seeds


def test_the_sweep_overview_caption_claims_no_seed_count_or_budget():
    datas = [_rep("base", {}, 0.5, 10, seeds=(1, 2, 3, 4, 5)), _rep("hidden-8", {"hidden": 8}, 0.6, 8, seeds=(1, 2, 3, 4, 5))]
    from code.neuro.report import _gasweep_overview
    html = _gasweep_overview(_gasweep_groups(datas))
    assert "3 seeds" not in html and "gens/probe" not in html
