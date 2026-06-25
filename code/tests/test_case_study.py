"""Tests for the multi-seed case-study aggregation."""
import math
from code.scripts.case_study import aggregate, mean_ci, _KEY_TO_LETTER


def _result(ext, persona, skill, seed, wr11, wr12, cause11, cause12, train_s=100.0):
    return {
        "job": {"ext": ext, "persona": persona, "skill": skill, "seed": seed},
        "train_seconds": train_s,
        "levels": {
            "Mario1-1": {"n": 10, "wins": int(wr11 * 10), "win_rate": wr11, "by_cause": cause11},
            "Mario1-2": {"n": 10, "wins": int(wr12 * 10), "win_rate": wr12, "by_cause": cause12},
        },
    }


def test_mean_ci_small_n_uses_t():
    m, ci = mean_ci([0.5, 0.6, 0.7])
    assert m == 0.6
    # sd=0.1, t(95%, df=2)=4.303 -> ci = 4.303*0.1/sqrt(3)
    assert math.isclose(ci, 4.303 * 0.1 / math.sqrt(3), rel_tol=1e-3)


def test_mean_ci_single_seed_has_no_ci():
    m, ci = mean_ci([0.42])
    assert m == 0.42 and ci is None


def test_aggregate_maps_run_letter_and_computes_winrate_ci():
    # config = (lightmobile, platformer_adept, Novice) = Run C
    results = [
        _result("lightmobile", "platformer_adept", "Novice", s, wr, 0.1,
                {"pit": 7, "stall": 3}, {"pit": 8, "stall": 2})
        for s, wr in [(1, 0.5), (2, 0.6), (3, 0.7)]
    ]
    agg = aggregate(results)
    assert "C" in agg
    c = agg["C"]
    assert c["n_seeds"] == 3
    l1 = c["levels"]["Mario1-1"]
    assert math.isclose(l1["win_rate_mean"], 0.6, rel_tol=1e-9)
    assert l1["win_rate_ci"] is not None and l1["win_rate_ci"] > 0
    # taxonomy: pit dominates on 1-1
    assert l1["dominant_failure"] == "pit"
    assert math.isclose(l1["taxonomy"]["pit"]["mean"], 0.7, rel_tol=1e-9)


def test_aggregate_handles_missing_config():
    # only Run C present; aggregate should not invent others
    results = [_result("lightmobile", "platformer_adept", "Novice", 1, 0.5, 0.1,
                       {"pit": 5}, {"stall": 5})]
    agg = aggregate(results)
    assert set(agg.keys()) == {"C"}


def test_run_design_covers_all_eight():
    assert len(_KEY_TO_LETTER) == 8
    assert ("spatialattention", "platformer_adept", "Expert") in _KEY_TO_LETTER
    assert _KEY_TO_LETTER[("spatialattention", "platformer_adept", "Expert")] == "H"
