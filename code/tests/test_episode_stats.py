"""Per-game EpisodeStats on real cores, and the episodes.csv round trip."""
import os

import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from code.neuro.adapters import make_adapter
from code.stats import episode_log
from code.stats.episode_stats import make_stats


def _play(game, level, frames=300):
    pygame.init()
    ad = make_adapter(game, level, frames, 5000.0)
    ad.reset()
    st = make_stats(game, ad.core, level=level, persona="experienced", source="probe", agent=0, gen=1)
    i = 0
    while ad.alive:
        jump = i % 20 < 5
        ad.step(1, jump, 0)
        st.on_step(jump)
        i += 1
    st.finish(ad.status)
    return st.to_dict()


def test_mario_row_schema():
    row = _play("mario", "Mario1-1")
    assert row["status"] in ("WON", "DEAD", "STUCK") and row["cause"]
    assert 0.0 <= row["progress"] <= 1.0
    assert row["frames"] <= 300 and row["time_s"] == round(row["frames"] / 60, 2)
    assert "powerups_lost" in row and "bricks" not in row      # game-specific keys only where they apply
    assert len(row["route"]) >= 2


def test_bomberman_row_schema():
    row = _play("bomberman", "1")
    for k in ("bombs_placed", "bomb_drops", "chain_bombs", "exit_found", "level_enemies", "powerups"):
        assert k in row, k
    assert row["bombs_placed"] == len(row["bomb_drops"])
    assert "coins" not in row and "powerups_lost" not in row


def test_meatboy_progress_is_bfs_share():
    row = _play("meatboy", "0")
    assert "wall_jumps" in row
    assert row["progress"] is None or 0.0 <= row["progress"] <= 1.0


def test_episode_log_round_trip(tmp_path):
    path = str(tmp_path / "episodes.csv")
    episode_log.append(path, [{"persona": "p", "game": "meatboy", "level": "0", "status": "DEAD",
                               "progress": None, "time_s": 1.5, "route": [[1.0, 2.0], [3.0, 4.0]]}])
    episode_log.append(path, [{"persona": "p", "game": "meatboy", "level": "0", "status": "WON",
                               "progress": 1.0, "time_s": 2.0, "wall_jumps": 3, "route": []}])  # new column
    rows = episode_log.read(path)
    assert rows[0]["level"] == "0"                 # level ids stay text
    assert rows[0]["progress"] is None and rows[0]["route"] == [[1.0, 2.0], [3.0, 4.0]]
    assert rows[0]["wall_jumps"] is None and rows[1]["wall_jumps"] == 3
    episode_log.reset(path)
    assert episode_log.read(path) == []
