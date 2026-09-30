"""The balance-metric registry on synthetic episode rows — no pygame, no disk."""
from code.stats import registry


def _ep(status, time_s=10.0, progress=0.5, cause="Enemy", **kw):
    return {"status": status, "cause": "" if status == "WON" else cause, "time_s": time_s,
            "progress": 1.0 if status == "WON" else progress, "tile": 32, "level_w": 320, "level_h": 64,
            "end_x": 40.0, "end_y": 40.0, **kw}


def test_shared_metrics():
    eps = [_ep("WON", 10), _ep("WON", 14), _ep("DEAD", progress=0.05), _ep("DEAD", progress=0.95, cause="Pit"),
           _ep("STUCK", progress=0.55, cause="Stall")]
    m = registry.compute_level(eps, "mario")
    assert m["completion_rate"] == 0.4
    assert m["mean_completion_time"] == 12.0
    assert m["deaths_per_run"] == 1.5                       # 3 failed attempts per 2 wins
    assert m["death_cause_distribution"] == {"Enemy": 1, "Pit": 1, "Stall": 1}
    assert m["death_progress_hist"] == [1, 0, 0, 0, 0, 1, 0, 0, 0, 1]
    assert m["death_cluster_entropy"] == round(__import__("math").log2(3) / __import__("math").log2(10), 4)
    assert m["death_location_heatmap"]["cells"] == [[1, 1, 3]]


def test_game_specific_metrics_only_where_they_apply():
    eps = [_ep("DEAD", bricks=3, bombs_placed=2, kills=1, level_enemies=2)]
    bm = registry.compute_level(eps, "bomberman")
    mario = registry.compute_level(eps, "mario")
    assert "bomb_kill_rate" in bm and "bomb_kill_rate" not in mario
    assert "progress_at_death" in mario and "progress_at_death" not in bm
    assert bm["bomb_kill_rate"] == 0.5 and bm["enemies_killed_ratio"] == 0.5
    assert registry.compute_level(eps, "meatboy")["bandage_collection_rate"] is None   # no bandages yet


def test_missing_data_is_none_not_zero():
    m = registry.compute_level([_ep("DEAD", coins=0, level_coins=0)], "mario")
    assert m["mean_completion_time"] is None
    assert m["coin_collection_rate"] is None              # the level has no coins: N/A, not 100%
    assert m["strategy_count"] is None                    # needs 2+ wins


def test_coin_triangularity():
    eps = [_ep("WON", 20, coins=3, level_coins=4), _ep("WON", 10, coins=0, level_coins=4),
           _ep("DEAD", coins=2, level_coins=4), _ep("WON", 10, coins=0, level_coins=4)]
    m = registry.compute_level(eps, "mario")
    assert m["coin_time_cost"] == 2.0
    assert m["coin_death_premium"] == 0.5                 # collectors fail 1/2, direct runs 0/2
    assert m["coin_collection_rate"] == round((0.75 + 0 + 0.5 + 0) / 4, 4)


def test_strategies_split_high_and_low_routes():
    low = [[x, 400.0] for x in range(0, 1000, 20)]
    high = [[x, 100.0] for x in range(0, 1000, 20)]
    eps = [_ep("WON", 10, route=low), _ep("WON", 11, route=low), _ep("WON", 30, route=high)]
    m = registry.compute_level(eps, "mario")
    assert m["strategy_count"] == 2
    assert m["dominant_path_share"] == round(2 / 3, 4)
    assert m["safe_vs_fast_path_ratio"] == round(30 / 10.5, 4)
    assert 0 < m["path_diversity_index"] < 1


def test_skill_tiers_map_to_personas():
    by_persona = {"novice": {"completion_rate": 0.2, "mean_completion_time": 30.0},
                  "experienced": {"completion_rate": 0.5, "mean_completion_time": 25.0},
                  "speedrunner": {"completion_rate": 0.7, "mean_completion_time": 18.0}}
    c = registry.compute_cross(by_persona, "mario")
    assert c["novice_expert_gap"] == 0.5
    assert c["novice_expert_time_gap"] == 12.0
    assert c["completion_rate_per_skill"] == {"novice": 0.2, "mid": 0.5, "expert": 0.7}
    assert registry.compute_cross({"novice": by_persona["novice"]}, "mario")["novice_expert_gap"] is None


def test_verdicts_use_bands():
    th = {"bands": {"completion_rate": {"target": 0.7, "warning": 0.2},
                    "mean_completion_time": {"target": 20, "warning": 4},
                    "deaths_per_run": {"target": 2, "warning": 1}}}
    v = dict(registry.verdicts({"completion_rate": 0.6, "mean_completion_time": 40, "deaths_per_run": 2}, "mario", th))
    assert v == {1: "partial", 2: "ok"}                   # dimension 4 has no values -> no pill


def test_default_thresholds_file_loads():
    th = registry.load_thresholds("mario")
    assert th["bands"]["completion_rate"] == {"target": 0.7, "warning": 0.2}
    assert th["route_cluster_threshold"] == 96
