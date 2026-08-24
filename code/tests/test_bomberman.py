"""Bomberman: level-file integrity, blast rules, and the adapter/sensor contract."""
import os

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from code.games.bomberman_core import BRICK, EXIT, EXIT_HIDDEN, POWERUPS, WALL, Bomb, BombermanCore
from code.neuro.adapters import N_INPUTS_BY_GAME, list_levels, make_adapter
from code.neuro.sensors import read_sensors, sensor_dim

LEVELS = list_levels("bomberman")
LEGAL = set("#.?GE kKMB") | set(POWERUPS) | {EXIT_HIDDEN}


@pytest.fixture(scope="module")
def core():
    return BombermanCore(render_mode="none")


@pytest.mark.parametrize("level", LEVELS)
def test_level_file_is_well_formed(core, level):
    """Every level: rectangular, walled, one start, one reachable exit, no unfair spawn."""
    core._level_idx = int(level)
    core.won = False
    core.reset()
    ld = core.level_data
    with open(core.level_file(), encoding="utf-8") as f:
        rows = [r.rstrip("\n") for r in f if r.strip()]
    assert len({len(r) for r in rows}) == 1, "ragged rows"
    assert set("".join(rows)) - {"P"} <= LEGAL, f"unknown glyph in {core.level_file()}"
    assert "".join(rows).count("P") == 1
    assert sum(r.count(EXIT) + r.count(EXIT_HIDDEN) for r in rows) == 1
    assert rows[0] == rows[-1] == WALL * len(rows[0]), "top/bottom must be solid wall"
    assert all(r[0] == r[-1] == WALL for r in rows), "sides must be solid wall"
    assert core.start_cost() < float("inf"), "exit unreachable even by bombing"
    px, py = core._center_tile(core.player.x, core.player.y, core.player.width, core.player.height)
    for e in core.enemies:
        ex, ey = core._center_tile(e.x, e.y, e.width, e.height)
        assert abs(ex - px) + abs(ey - py) > 2, "enemy spawns on top of the player"


# What each rung of the campaign is for. "walk" = beatable with no bomb at all (the bricks are
# scenery); "bomb" = the exit is sealed until something is blown open. A level that drifts from its
# intent is a design bug the probe would otherwise report as mysterious difficulty.
LADDER = ["walk", "bomb", "bomb", "bomb", "walk", "bomb", "walk", "walk",
          "walk", "bomb", "bomb", "bomb", "bomb", "bomb", "bomb"]


def _walk_only_cost(core) -> int | None:
    """Shortest path treating bricks as solid — None means the level cannot be beaten without a bomb."""
    import heapq
    sx, sy = core.level_data.start
    goal = core.level_data.exit
    seen, queue = {(sx, sy)}, [(0, sx, sy)]
    while queue:
        d, x, y = heapq.heappop(queue)
        if (x, y) == goal:
            return d
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            nxt = (x + dx, y + dy)
            if nxt in seen or core.solid(*nxt):
                continue
            seen.add(nxt)
            heapq.heappush(queue, (d + 1, *nxt))
    return None


@pytest.mark.parametrize("level", LEVELS)
def test_level_matches_its_rung(level):
    core = BombermanCore(render_mode="none", level_idx=int(level))
    want = LADDER[int(level)]
    walkable = _walk_only_cost(core) is not None
    assert walkable == (want == "walk"), f"level {level} is meant to be a '{want}' level"


def test_ladder_ramps_up():
    """The campaign is graded: level 1 is a plain walk, the last level is the gauntlet."""
    costs = [(BombermanCore(render_mode="none", level_idx=int(lv)).start_cost(),
              len(BombermanCore(render_mode="none", level_idx=int(lv)).enemies)) for lv in LEVELS]
    assert costs[0] == (22.0, 0), "level 1 is a plain walk to the exit"
    assert costs[1][0] > costs[0][0], "level 2 must cost a bomb"
    assert len(LADDER) == len(LEVELS), "every level needs a rung"
    assert max(n for _, n in costs) == costs[-1][1] >= 3, "the last level holds the most enemies"


def test_blast_stops_at_wall_and_ends_on_brick(core):
    core._level_idx, core.won = 3, False   # 04_one_wall: a full brick column at x=7
    core.reset()
    cells = core.blast_cells(Bomb(1, 1, 0, 3))
    assert (1, 0) not in cells and (0, 1) not in cells, "blast must not cross the border wall"
    core.level_data.grid[1][3] = BRICK
    assert (3, 1) in core.blast_cells(Bomb(1, 1, 0, 3)), "the arm ends on the brick, not before it"
    assert (4, 1) not in core.blast_cells(Bomb(1, 1, 0, 3)), "and stops there"


def test_chain_reaction_and_brick_reveal(core):
    core._level_idx, core.won = 5, False   # 06_hidden_exit: the exit sits under a brick
    core.reset()
    hx, hy = core.level_data.exit
    assert core.tile(hx, hy) == EXIT_HIDDEN
    core.bombs = [Bomb(hx - 1, hy, 1, 1), Bomb(hx - 2, hy, 999, 1)]
    core._explode(core.bombs[0])
    assert not core.bombs, "the neighbouring bomb must chain, not wait out its fuse"
    assert core.tile(hx, hy) == EXIT, "a bombed brick reveals what was under it"


def test_exit_only_opens_on_a_clear_arena():
    """Standing on the exit with an enemy alive is not a win; killing it makes it one."""
    core = BombermanCore(render_mode="none", level_idx=7)   # 08_corridor: one Ballom
    ts = core.tile_size
    ex, ey = core.level_data.exit
    core.player.x, core.player.y = ex * ts + 4, ey * ts + 4
    core.step((0, 0, 0))
    assert not core.won and core.enemies
    for e in core.enemies:
        e.alive = False
    core.step((0, 0, 0))
    assert core.won


def test_determinism():
    a, b = (BombermanCore(render_mode="none", level_idx=11) for _ in range(2))
    for _ in range(240):
        a.step((1, 1, 0))
        b.step((1, 1, 0))
    assert (a.player.x, a.player.y, a.score) == (b.player.x, b.player.y, b.score)
    assert [(e.x, e.y, e.alive) for e in a.enemies] == [(e.x, e.y, e.alive) for e in b.enemies]


@pytest.mark.parametrize("mode,dim", [("rays", N_INPUTS_BY_GAME["bomberman"]), ("grid", sensor_dim("grid"))])
def test_sensor_dims(mode, dim):
    adapter = make_adapter("bomberman", "9", max_frames=300, win_bonus=100.0)
    vec, _rays, _tiles = read_sensors(adapter, mode)
    assert vec.shape == (dim,)
    assert vec.min() >= -1.0 and vec.max() <= 1.0


def test_danger_slots_point_the_way_out():
    """Drop a bomb: the tile the agent stands on reads dangerous, two tiles away reads safe."""
    adapter = make_adapter("bomberman", "0", max_frames=600, win_bonus=100.0)
    adapter.step(0, True)                      # bomb on the start tile
    vec, _, _ = read_sensors(adapter, "rays")
    assert vec[8] > 0.0, "own tile must read dangerous"
    assert vec[5] > 0.0 and vec[6] > 0.0, "east and south neighbours are in the cross"
    for _ in range(60):                        # walk two tiles east, out of the blast
        adapter.step(1, False)
    vec, _, _ = read_sensors(adapter, "rays")
    assert vec[8] == 0.0 and max(vec[4:8]) == 0.0, "two tiles clear of a range-1 bomb is safe"


def test_win_and_death_are_reported():
    adapter = make_adapter("bomberman", "0", max_frames=1200, win_bonus=500.0)
    while adapter.alive:
        adapter.step(1, False, 1)              # 01_open_floor: walk the diagonal to the exit
    assert adapter.won and adapter.status == "WON"
    assert adapter.fitness() > 1000.0
    st = adapter.episode_stats()
    assert st["cause"] in ("", "Win", None) or st["won"]

    adapter.reset()
    assert adapter.alive and adapter.core._level_idx == 0, "training stays on its level after a win"
    adapter.step(0, True)
    while adapter.alive:
        adapter.step(0, False)                 # stand on the bomb
    assert adapter.status == "DEAD" and adapter.core.death_cause == "Bomb"
