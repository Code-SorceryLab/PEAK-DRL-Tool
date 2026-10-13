"""Castlevania stage 1: level-file integrity, the game's rules, the route field, and the
adapter/sensor/telemetry contract it shares with Mega Man."""
import os

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["PEAK_CV_ASSETS"] = "/nonexistent"      # placeholder art: the tests never need castlevania-data
os.environ["PEAK_CV_ATTEMPTS"] = "1"

from code.games import castlevania_core as cv
from code.games.castlevania_core import CastlevaniaCore
from code.games.castlevania_route import route_field
from code.neuro.adapters import N_INPUTS_BY_GAME, STAGE_GAMES, list_levels, make_adapter
from code.neuro.sensors import read_sensors

LEVELS = list_levels("castlevania")
STAGE = STAGE_GAMES["castlevania"]
LEGAL = set(" .#/\\~DGXP") | set(cv.ENEMY_GLYPHS) | set(cv.CANDLES)
IDLE, RIGHT, LEFT, UP, DOWN, JUMP, FIRE = ((0, 0, 0, 0), (1, 0, 0, 0), (-1, 0, 0, 0), (0, -1, 0, 0), (0, 1, 0, 0),
                                           (0, 0, 1, 0), (0, 0, 0, 1))


def place(core, col, row):
    """Stand the player in a cell, feet on its bottom edge."""
    p, ts = core.player, core.tile_size
    p.x, p.y = col * ts + (ts - p.width) / 2.0, (row + 1) * ts - p.height
    p.vx = p.vy = 0.0
    p.grounded, p.climbing, p.stair = False, False, None
    core.screen = (col // core.screen_cols, row // core.screen_rows)
    core.camera_x, core.camera_y = core._camera_target()


def feet(core):
    return (core.player.y + core.player.height) / core.tile_size


@pytest.mark.parametrize("level", LEVELS)
def test_level_file_is_well_formed(level):
    core = CastlevaniaCore(level=level)
    with open(core.level_file(), encoding="utf-8") as f:
        rows = f.read().split("\n")
    assert set("".join(rows)) <= LEGAL
    assert "".join(rows).count("P") == 1
    section = core.level_entry().get("section")
    if section:
        c, r = section["start"]
        if section["climbing"]:
            assert rows[r][c] in "/\\" and core.player.climbing, "a section entered on the stairs starts on them"
        else:
            assert rows[r][c] in ".P" and rows[r + 1][c] == "#", "a section starts standing on a floor"
        assert tuple(section["screens"][0]) == core.screen
    adapter = make_adapter("castlevania", level, 600, 5000.0)
    assert np.isfinite(adapter._start_cost), "the exit is unreachable from the start"
    for _ in range(30):
        core.step(IDLE)
    assert core.alive and core.player.hp == core.player.hp_max and (core.player.grounded or core.player.climbing)


def test_sections_chain_from_the_start_to_the_boss():
    core = CastlevaniaCore(level=STAGE)
    cost = route_field(core.level_data)
    sections = [e["section"] for _, e in core.levels if e.get("section")]
    assert sections[0]["start"] == list(core.level_data.start)
    costs = [float(cost[s["start"][1], s["start"][0]]) for s in sections]
    assert all(np.isfinite(costs)) and costs == sorted(costs, reverse=True), costs
    for a, b in zip(sections, sections[1:]):
        assert a["end"] == b["start"] and [b["start"][0] // 16, b["start"][1] // 12] in a["goal_screens"]


def test_walk_and_a_committed_jump():
    core = CastlevaniaCore(level=STAGE)
    p, s = core.player, core.scale
    core.step(IDLE)
    x0 = p.x
    for _ in range(60):
        core.step(RIGHT)
    assert abs((p.x - x0) / s - 60 * 1.0417) < 1e-6
    y0, top, x0 = p.y, p.y, p.x
    core.step((1, 0, 1, 0))
    for _ in range(80):                                 # the pad pushed the other way: the jump does not turn
        core.step(LEFT if not p.grounded else IDLE)
        top = min(top, p.y)
    assert 38.0 < (y0 - top) / s < 42.0 and p.grounded and p.y == y0, (y0 - top) / s
    assert 2.3 < (p.x - x0) / core.tile_size < 2.9, (p.x - x0) / core.tile_size


def test_stairs_up_and_down():
    core = CastlevaniaCore(level=STAGE)
    p = core.player
    place(core, 87, 9)                                  # the foot of the first flight in the hall
    core.step(IDLE)
    assert core.flight_at(-1) and not core.flight_at(1)
    for _ in range(20):                                 # up: he walks to the first step, then onto it
        core.step(UP)
    assert p.climbing
    core.step(JUMP)                                     # no jumping off the stairs
    assert p.climbing
    for _ in range(300):
        core.step(UP)
    assert not p.climbing and p.grounded and feet(core) == 6.0 and p.x + p.width / 2 == 91 * 32
    for _ in range(5):
        core.step(IDLE)
    assert p.grounded and feet(core) == 6.0, "the head of a flight is a place to stand"
    for _ in range(20):
        core.step(DOWN)
    assert p.climbing
    for _ in range(300):
        core.step(LEFT)                                 # left is down on a flight that rises to the right
    assert not p.climbing and p.grounded and feet(core) == 10.0


def test_stairs_through_the_floor_lead_to_the_underground():
    core = CastlevaniaCore(level=STAGE)
    p = core.player
    place(core, 146, 9)
    for _ in range(40):
        core.step(RIGHT)                                # walks over the head of the flight without falling in
    assert p.grounded and feet(core) == 10.0
    place(core, 147, 9)                                 # half a tile from the head: down walks him to it, then on
    core.step(IDLE)
    for _ in range(20):
        core.step(DOWN)
    assert p.climbing
    for _ in range(700):
        core.step(DOWN)
    assert core.screen == (9, 1) and core.camera_y == core.HEIGHT
    assert not p.climbing and p.grounded and feet(core) == 15.0
    core.step(UP)
    for _ in range(700):
        core.step(UP)
    assert core.screen == (9, 0) and p.grounded and feet(core) == 10.0


@pytest.mark.parametrize("level, col, row, press, into", [("Entrance-5-Crypt", 147, 9, DOWN, (9, 1)),
                                                         ("Entrance-6-Ascent", 174, 16, UP, (10, 0))])
def test_a_section_entered_on_the_stairs_starts_where_the_stage_hands_it_over(level, col, row, press, into):
    stage = CastlevaniaCore(level=STAGE)
    p = stage.player
    place(stage, col, row)
    p.x -= 16                                           # on the corner the flight starts from
    stage.step(IDLE)
    for _ in range(2000):
        stage.step(press)
        if stage.screen == into:
            break
    assert stage.screen == into and p.climbing
    q = CastlevaniaCore(level=level).player
    assert q.climbing and abs(q.x - p.x) < 1.0 and abs(q.y - p.y) < 1.0 and q.facing == p.facing


def test_water_and_doors():
    core = CastlevaniaCore(level=STAGE)
    place(core, 160, 16)                                # the first gap over the water
    for _ in range(200):
        core.step(IDLE)
    assert not core.alive and core.death_cause == "Water"

    core = CastlevaniaCore(level=STAGE)
    p, ts = core.player, core.tile_size
    place(core, 45, 9)
    for _ in range(200):
        core.step(RIGHT)
    assert core.tile(47, 9) == cv.DOOR_SHUT and core.screen == (3, 0)
    for _ in range(300):
        core.step(LEFT)
    assert p.x == 48 * ts and core.camera_x == 48 * ts, "the door is shut and the camera stays in the hall"


def test_whip_kills_and_candles_drop():
    core = CastlevaniaCore(level=STAGE)
    p = core.player
    place(core, 11, 9)                                  # three tiles left of the first whip upgrade: out of reach
    core.step(IDLE)
    core.step(FIRE)
    assert p.attack == 0 and core.whip_box() is None    # the wind-up cannot hit
    for _ in range(40):
        core.step(RIGHT)                                # rooted while the whip is out, then walks
    assert len(core.candles) == len(core.level_data.candles)
    place(core, 13, 9)
    core.step(IDLE)
    core.step(FIRE)
    for _ in range(120):
        core.step(RIGHT if p.attack < 0 else IDLE)
    assert len(core.candles) == len(core.level_data.candles) - 1 and p.whip_level == 2

    core = CastlevaniaCore(level="Entrance-2-Hall", seed=3)
    p = core.player
    kills = 0
    for i in range(1500):                               # stand and whip: zombies come in from the screen's edge
        z = min((e for e in core.enemies if e.kind == "zombie"), key=lambda e: abs(e.x - core.nes_player()[0]), default=None)
        act = IDLE
        if z is not None and abs(z.x - core.nes_player()[0]) < 44:
            want = 1 if z.x > core.nes_player()[0] else -1
            act = FIRE if p.facing == want else ((want, 0, 0, 0) if i % 2 else IDLE)
        core.step(act if i % 2 else (act[0], 0, 0, 0))
        kills = core.kills_total
    assert kills >= 3 and core.alive, (kills, p.hp)


def test_a_hit_throws_him_back_and_blinks():
    core = CastlevaniaCore(level="Entrance-2-Hall", seed=3)
    p = core.player
    for _ in range(600):
        core.step(IDLE)
        if p.hp < p.hp_max:
            break
    assert p.hp == p.hp_max - 2 and p.knock and p.blink > 100 and not p.grounded
    for _ in range(100):
        core.step(RIGHT)
    assert p.knock == 0 and p.grounded and p.hp == p.hp_max - 2, "nothing hurts twice during the blink"


def test_panther_wakes_and_leaps():
    core = CastlevaniaCore(level=STAGE)
    place(core, 84, 9)
    core.step(IDLE)
    cat = next(e for e in core.enemies if e.kind == "panther")
    assert cat.state == "sit"
    states = set()
    for _ in range(200):
        core.step(RIGHT)
        if cat in core.enemies:
            states.add(cat.state)
    assert {"run", "jump"} <= states


def test_boss_fight_and_the_orb():
    core = CastlevaniaCore(level="Entrance-8-Boss", seed=5)
    p = core.player
    assert core.boss is None or core.boss.state == "sleep"
    for _ in range(120):
        core.step(RIGHT)
    assert core.boss and core.boss.state != "sleep" and core.boss_lock
    for _ in range(400):
        core.step(LEFT)
    assert p.x == core.boss_lock[0], "the boss's screen is shut"
    states = set()
    for _ in range(600):
        states.add(core.boss.state)
        core.step(IDLE)
    assert {"rest", "dive", "rise"} <= states and p.hp < p.hp_max
    core = CastlevaniaCore(level="Entrance-8-Boss", seed=5, invincible=True)
    p = core.player
    for _ in range(120):
        core.step(RIGHT)
    core.boss.hp = 1
    for i in range(3000):
        if core.boss is None:
            break
        b = core.boss
        core.step((0, 0, 0, i % 2) if abs(b.y - core.nes_player()[1]) < 30 else IDLE)
    assert core.boss_dead and core.boss_damage == 1
    orb = next(it for it in core.items if it.kind == "clear")
    for _ in range(600):
        _, _, done, _, _ = core.step((1 if orb.x * core.scale > p.x + p.width / 2 else -1, 0, 0, 0))
        if done:
            break
    assert core.won and core.render().get_size() == (core.WIDTH, core.HEIGHT)


def test_sensors_and_a_scripted_win():
    adapter = make_adapter("castlevania", LEVELS[0], 3600, 5000.0)
    vec, rays, tiles = read_sensors(adapter, "rays")
    assert vec.shape == (N_INPUTS_BY_GAME["castlevania"],) == (26,)
    assert np.all(np.abs(vec) <= 1.0 + 1e-6) and vec[25] == 1.0 and vec[24] == 1.0
    labelled = sorted(r["i"] for g in adapter.SENSOR_LABELS for r in g["rows"])
    assert labelled == list(range(25))
    assert vec[18] > 0 and vec[19] == 0, "the compass points along the courtyard"
    n = 0
    while adapter.alive and n < 3000:
        adapter.step(1, False, 0, False)
        n += 1
    assert adapter.won and adapter.status == "WON" and adapter.fitness() > 5000

    adapter = make_adapter("castlevania", "Entrance-3-Panthers", 3600, 5000.0)
    core = adapter.core
    place(core, 127, 9)                                 # two tiles short of the flight up to the door
    core.step(IDLE)
    vec = read_sensors(adapter, "rays")[0]
    assert vec[16] == 0 and 0 < vec[23] < 1
    place(core, 129, 9)
    core.step(IDLE)
    vec = read_sensors(adapter, "rays")[0]
    assert vec[16] == 1 and abs(vec[23]) > 0.9 and vec[19] < 0, "at the foot: up steps on, and the route goes up"


def test_a_held_fire_output_swings_once_and_down_with_a_direction_walks():
    adapter = make_adapter("castlevania", LEVELS[0], 3600, 5000.0)
    core = adapter.core
    for _ in range(120):
        adapter.step(0, False, 0, True)
    assert core.attacks_total == 1
    adapter.step(0, False, 0, False)
    for _ in range(40):
        adapter.step(0, False, 0, True)
    assert core.attacks_total == 2
    x0 = core.player.x
    for _ in range(30):
        adapter.step(1, False, 1, False)
    assert core.player.x > x0 + 30 and not core.player.ducking
    for _ in range(10):
        adapter.step(0, False, 1, False)
    assert core.player.ducking


def test_progress_counts_only_with_a_foot_down():
    adapter = make_adapter("castlevania", "Entrance-2-Hall", 3600, 5000.0)
    for _ in range(5):
        adapter.step(0, False, 0, False)
    before = adapter.progress()
    adapter.step(1, True, 0, False)
    for _ in range(10):
        adapter.step(1, True, 0, False)
    assert not adapter.core.player.grounded and adapter.progress() == pytest.approx(before, abs=0.01)
    for _ in range(60):
        adapter.step(0, False, 0, False)
    assert adapter.progress() > before + 0.02


def test_telemetry_row():
    from code.stats.episode_stats import make_stats
    core = CastlevaniaCore(level=STAGE)
    stats = make_stats("castlevania", core, level=STAGE, persona="tester", source="agent", agent=0)
    for i in range(1600):
        act = (1, 0, 0, int(i % 120 == 0))
        _, _, term, trunc, info = core.step(act)
        stats.on_step(False)
        if term or trunc:
            break
    stats.finish("STUCK")
    row = stats.to_dict()
    assert row["game"] == "castlevania" and 0.15 < row["progress"] < 0.5
    assert row["sections_reached"] == 2 and row["sections"][0]["section"] == "Entrance-1-Courtyard"
    assert row["section"] == "Entrance-2-Hall" and row["shots"] == core.attacks_total > 10
