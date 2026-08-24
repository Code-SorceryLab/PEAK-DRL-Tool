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
    for lvl in list_levels("bomberman"):
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
