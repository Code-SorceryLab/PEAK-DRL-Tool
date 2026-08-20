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
