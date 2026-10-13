"""Telemetry: a Mega Man 1 episode row, and the dashboard's metrics computed from rows alone."""
import json
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["PEAK_MM1_ASSETS"] = "/nonexistent"
os.environ["PEAK_MM1_ATTEMPTS"] = "1"

from code.games.megaman_core import MegamanCore
from code.stats import episode_log, telemetry
from code.stats.episode_stats import make_stats


def _play(level, frames, action, source="agent"):
    core = MegamanCore(level=level)
    stats = make_stats("megaman", core, level=level, persona="tester", source=source, agent=0)
    for _ in range(frames):
        _, _, done, _, _ = core.step(action)
        stats.on_step(bool(action[2]))
        if done:
            break
    stats.finish("WON" if core.won else "DEAD" if not core.alive else "STUCK")
    return stats.to_dict()


def test_megaman_episode_row():
    row = _play("CutMan", 240, (1, 0, 0, 0))                       # walk right for four seconds
    assert row["game"] == "megaman" and row["level"] == "CutMan" and row["status"] == "STUCK"
    assert 0.0 < row["progress"] < 0.1 and row["frames"] == 240 and row["time_s"] == 4.0
    assert row["section"] == "CutMan-1-Street" and row["sections_reached"] == 1
    assert row["sections"][0]["frames"] == 240 and row["sections"][0]["hp_in"] == 28
    assert {"kills", "coins", "jumps", "hp", "damage_taken", "shots", "route", "end_x", "end_y"} <= set(row)
    won = _play("CutMan-7-Corridor", 900, (1, 0, 0, 0))             # the corridor: walking right wins it
    assert won["status"] == "WON" and won["progress"] == 1.0 and won["cause"] == ""


def test_dashboard_metrics_come_from_rows_alone(tmp_path):
    rows = [_play("CutMan-7-Corridor", 900, (1, 0, 0, 0)),
            _play("CutMan-7-Corridor", 60, (0, 0, 0, 0)),
            _play("CutMan-7-Corridor", 60, (-1, 0, 1, 0), source="random")]
    path = tmp_path / "run" / "episodes.csv"
    episode_log.append(str(path), rows)
    data = telemetry.build([str(tmp_path)])
    assert data["episodes"] == 3 and len(data["groups"]) == 2       # agent and random are separate players
    g = next(x for x in data["groups"] if x["player"] == "tester")
    assert g["completed"] and g["wins"] == 1 and g["win_rate"] == 0.5 and g["furthest"] == 1.0
    assert g["time_to_finish"] > 0 and g["causes"] == [{"cause": "Stall", "n": 1}]
    assert sum(g["progress_hist"]) == 1 and g["geometry"]["solid"] and g["sections"][0]["reached"] == 2
    assert {m["key"] for m in g["means"]} >= {"jumps", "kills", "coins"}
    json.dumps(data)                                                # the page embeds it as JSON
