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
    st = make_stats(game, ad.core, level=level, persona="experienced", source="watch", agent=0, gen=1)
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


# The counters below are observed from outside the game, frame to frame — the game code keeps
# no metric-only state. These tests set the situation up directly and check what was seen.

def test_bomberman_drops_and_chain_are_observed():
    from code.games.bomberman_core import FLOOR, Bomb
    pygame.init()
    ad = make_adapter("bomberman", "0", 600, 5000.0)
    ad.reset()
    core = ad.core
    st = make_stats("bomberman", core, level="0", persona="t", source="t")
    p = core.player
    px, py = core._center_tile(p.x, p.y, p.width, p.height)
    rng = core.player.blast_range
    # two adjacent floor tiles well out of blast reach of the player: the first one's blast
    # sets the second off long before its own fuse would
    a = next((x, y) for y in range(core.level_data.rows) for x in range(core.level_data.cols - 1)
             if core.tile(x, y) == FLOOR and core.tile(x + 1, y) == FLOOR
             and abs(x - px) + abs(y - py) > rng + 3 and abs(x + 1 - px) + abs(y - py) > rng + 3)
    core.bombs += [Bomb(a[0], a[1], 3, rng), Bomb(a[0] + 1, a[1], 200, rng)]
    for _ in range(20):
        ad.step(0, False, 0)
        st.on_step(False)
    st.finish("STUCK")
    row = st.to_dict()
    assert row["bomb_drops"] == [[a[0], a[1]], [a[0] + 1, a[1]]]
    assert row["chain_bombs"] == 1      # the second went off with fuse left; the first on its own fuse


def test_mario_powerups_lost_is_observed():
    pygame.init()
    ad = make_adapter("mario", "Mario1-1", 300, 5000.0)
    ad.reset()
    st = make_stats("mario", ad.core, level="Mario1-1", persona="t", source="t")
    pm = ad.core.player.power_machine

    def idle():
        ad.step(0, False, 0)
        st.on_step(False)

    idle()
    pm.collect_flower()             # SMALL -> BIG -> FIRE: gaining tiers is not a loss
    idle()
    pm.take_hit()                   # FIRE -> BIG
    idle()
    pm._iframes_timer = 0.0         # skip the i-frame window instead of waiting it out
    pm.take_hit()                   # BIG -> SMALL
    idle()
    st.finish("STUCK")
    assert st.to_dict()["powerups_lost"] == 2


def test_meatboy_wall_jumps_are_observed():
    from types import SimpleNamespace
    from code.stats.episode_stats import MeatboyStats

    player = SimpleNamespace(x=0.0, y=0.0, width=10, height=10, vx=0.0, air_lockout=0)
    core = SimpleNamespace(player=player, _bfs_norm_dist=lambda: 0.5)
    st = MeatboyStats(core, game="meatboy", level="0", persona="t", source="t")
    # a wall jump raises air_lockout (6, read as 5 after control()'s decrement); it only falls otherwise
    for lockout in (0, 5, 4, 3, 5, 4, 3, 2, 1, 0, 0, 5, 4):
        player.air_lockout = lockout
        st.on_step(False)
    st.finish("STUCK")
    assert st.to_dict()["wall_jumps"] == 3


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
