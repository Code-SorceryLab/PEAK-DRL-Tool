"""summarize() folds probe history into a cell: stall counts as a cause, best_gen + improvement rate."""
from code.neuro.balance import _parse_tag, aggregate, config_tag, summarize


def _row(gen, statuses, best_x, cause="Enemy"):
    envs = [{"status": s, "cause": cause if s == "DEAD" else "", "end_x": best_x / 2,
             "level_len": 1000.0, "coins": 0, "level_coins": 0, "frames": 600} for s in statuses]
    return {"gen": gen, "best": best_x, "avg": best_x / 2, "best_x": best_x, "duration": 1.5,
            "wins": statuses.count("WON"), "stuck": statuses.count("STUCK"), "envs": envs}


def test_summarize_counts_stall_as_cause_and_tracks_best_gen():
    hist = [_row(1, ["STUCK", "STUCK", "DEAD"], 100.0),
            _row(2, ["STUCK", "DEAD", "WON"], 500.0),
            _row(3, ["STUCK", "WON", "WON"], 1000.0)]
    cell = summarize(hist, pop_size=3, level="L", seed=7, best_fitness=6000.0, best_gen=3)
    assert cell["causes"] == {"Stall": 4, "Enemy": 2}
    assert cell["first_win_gen"] == 2 and cell["best_gen"] == 3
    assert cell["train_time_s"] == 4.5
    assert abs(cell["improvement_rate"] - 0.45) < 1e-6  # slope of 100,500,1000 over gens 1..3 / 1000
    row = aggregate([cell])
    assert row["dominant_cause"] == "Stall" and row["best_gen_mean"] == 3.0
    assert row["train_time_s"] == 4.5


def test_config_tag_roundtrip():
    assert _parse_tag(config_tag(50, 40)) == (50, 40)
    assert config_tag(50, 40, "grid") == "p50g40_grid" and _parse_tag("p50g40_grid") == (50, 40)
    assert _parse_tag("legacy") is None


# ── GA hyperparameter sweep (code/neuro/gasweep.py) ──────────────────────────

def test_sweep_configs_one_factor_at_a_time():
    from code.neuro.evolution import GAConfig
    from code.neuro.gasweep import AXES, sweep_configs, parse_sweep_tag, sweep_tag
    cfgs = sweep_configs()
    assert cfgs[0] == ("base", {})
    assert len(cfgs) == 1 + sum(len(vs) - 1 for vs in AXES.values()) == 23
    base = GAConfig()
    for suffix, ov in cfgs[1:]:
        assert len(ov) == 1                                  # exactly one knob moves
        (axis, val), = ov.items()
        assert val != getattr(base, axis)                    # baseline value deduped
        assert parse_sweep_tag(sweep_tag(10, 40, "rays", suffix)) == (axis, val)


def test_sweep_elite_guard():
    from code.neuro.evolution import GAConfig
    from code.neuro.gasweep import sweep_configs
    cfgs = sweep_configs(["elite", "tournament_k"], base=GAConfig(pop_size=4, elite=2, tournament_k=3))
    assert [s for s, _ in cfgs] == ["base", "elite-1", "tournament_k-2"]   # elite 4,6 >= pop; k 5,8 > pop


def test_sweep_tag_roundtrip():
    from code.neuro.gasweep import base_sig, parse_sweep_tag, sweep_tag, tag_base_sig
    from code.neuro.evolution import GAConfig
    sig = base_sig()
    t = sweep_tag(10, 40, "grid", "mutation_rate-0.5")
    assert t == f"p10g40_grid_{sig}_mutation_rate-0.5" and len(sig) == 6
    assert _parse_tag(t) == (10, 40)
    assert parse_sweep_tag(t) == ("mutation_rate", 0.5) and tag_base_sig(t) == sig
    assert base_sig(GAConfig(anneal_factor=0.9)) != sig          # a new baseline gets a new sig
    assert parse_sweep_tag("p10g40_base") == ("base", None)      # pre-sig tags still parse
    assert tag_base_sig("p10g40_hidden-8") is None
    assert parse_sweep_tag("p10g40_best") == ("best", None)
    assert parse_sweep_tag("p10g40_action_feedback-1") == ("action_feedback", True)
    assert parse_sweep_tag("p10g40") is None
    assert parse_sweep_tag("p10g40_grid") is None
    assert parse_sweep_tag("p10g40_bogus-3") is None


def _report(suffix, ov, wr, fw):
    from dataclasses import asdict
    from code.neuro.evolution import GAConfig
    from code.neuro.gasweep import sweep_tag
    return {"game": "mario", "persona": "experienced", "tag": sweep_tag(10, 40, "rays", suffix),
            "gens_budget": 40, "ga_config": asdict(GAConfig(**ov)),
            "levels": [{"level": "L", "win_rate_mean": wr, "first_win_mean": fw}],
            "cells": {"L": [{"win_rate": wr, "first_win_gen": fw}]}}


def test_best_config_picks_per_axis_winner():
    from code.neuro.gasweep import best_config
    reports = [_report("base", {}, 0.5, 10),
               _report("hidden-8", {"hidden": 8}, 0.5, 8),          # ties win rate, faster
               _report("hidden-64", {"hidden": 64}, 0.4, 4),
               _report("elite-1", {"elite": 1}, 0.2, None),        # unsolved -> censored at 40
               _report("elite-6", {"elite": 6}, 0.9, 12)]
    assert best_config(reports) == {"hidden": 8, "elite": 6}
    assert best_config([_report("base", {}, 0.5, 10)]) == {}
