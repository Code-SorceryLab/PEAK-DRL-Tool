"""Meat Boy plays like Super Meat Boy: the measured values (docs/MEATBOY_SMB_RESEARCH.md) and the rules
built on them. u = one Meat Boy height = 22.4 px."""
import os

import pygame
import pytest
import yaml

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from code.games import meatboy_core
from code.games.meatboy_core import MeatboyCore
from code.neuro.adapters import make_adapter

U = 22.4
TS = 32


def test_the_tuned_values_stay_put():
    """The values and where each comes from are in docs/MEATBOY_SMB_RESEARCH.md; change both together."""
    with open(meatboy_core._CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["player"] == {"width": 22, "height": 22}
    assert cfg["physics"] == {"gravity": 918.0, "max_fall_speed": 500.0}
    assert cfg["movement"] == {"walk_accel": 784.0, "max_walk_speed": 403.0, "run_accel": 1500.0,
                               "max_run_speed": 600.0, "turn_factor": 2.0}
    assert cfg["jump"] == {"jump_vel": 470.0, "min_hold_frames": 8}
    assert cfg["wall"] == {"slide_max_speed": 180.0, "stick_frames": 12, "wall_jump_vy": 500.0, "wall_jump_push": 500.0}
    assert cfg["crumble_seconds"] == 0.75
    assert (MeatboyCore.WIDTH, MeatboyCore.HEIGHT) == (1280, 720)


def _core(tmp_path, rows):
    """A core on a level of our own: rows of the level text, '#' rock, '%' crumble, 'P' the start."""
    pygame.init()
    path = tmp_path / "level.txt"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    core = MeatboyCore()
    core.levels, core._level_idx, core.current_level = [str(path)], 0, None
    core.reset()
    return core


def _flat(tmp_path):
    core = _core(tmp_path, ["#" + "." * 198 + "#"] * 20 + ["#....P" + "." * 193 + "#", "#" * 200])
    for _ in range(20):                                     # land
        core.step([0, 0, 0])
    return core


def test_walk_and_run_reach_their_top_speeds_in_the_measured_time(tmp_path):
    core = _flat(tmp_path)
    for _ in range(31):                                     # 35 u/s^2 to 18 u/s: 0.52 s
        core.step([2, 0, 0])
    assert core.player.vx == pytest.approx(18 * U, abs=1)
    core = _flat(tmp_path)
    for _ in range(24):                                     # 0 -> 600 px/s in 0.4 s
        core.step([2, 1, 0])
    assert core.player.vx == 600.0


def test_standing_still_he_rests_on_the_floor_without_sinking_or_shaking(tmp_path):
    core = _flat(tmp_path)
    seen = set()
    for _ in range(60):
        core.step([0, 0, 0])
        seen.add((core.player.y, core.player.vy, core.player.on_ground))
    assert len(seen) == 1 and seen.pop()[1:] == (0.0, True)


def test_no_direction_stops_him_dead_on_the_ground_but_not_in_the_air(tmp_path):
    core = _flat(tmp_path)
    for _ in range(30):
        core.step([2, 1, 0])
    core.step([0, 1, 0])
    assert core.player.vx == 0.0
    for _ in range(30):
        core.step([2, 0, 0])
    core.step([2, 0, 1])                                    # take off at walk speed, then let go of the direction
    for _ in range(10):
        core.step([0, 0, 1])
    assert not core.player.on_ground and core.player.vx == pytest.approx(18 * U, abs=1)


def test_letting_go_of_run_in_the_air_drops_to_walk_speed_at_once(tmp_path):
    core = _flat(tmp_path)
    for _ in range(30):
        core.step([2, 1, 0])
    core.step([2, 1, 1])
    core.step([2, 0, 1])
    assert core.player.vx == pytest.approx(18 * U, abs=1)


def _jump(core, hold, move=0, run=0):
    """Frames in the air, the height reached and the distance covered by one jump."""
    x0, y0, top, f = core.player.x, core.player.y, core.player.y, 0
    while True:
        f += 1
        core.step([move, run, 1 if f <= hold else 0])
        top = min(top, core.player.y)
        if core.player.on_ground and f > 3:
            return f, y0 - top, core.player.x - x0


def test_a_full_jump_and_a_tap_rise_as_measured(tmp_path):
    frames, height, _ = _jump(_flat(tmp_path), 999)
    assert height / U == pytest.approx(5.55, abs=0.4) and frames / 60 == pytest.approx(1.05, abs=0.05)
    frames, height, _ = _jump(_flat(tmp_path), 1)        # released at once: the climb still lasts the minimum hold
    assert height / U == pytest.approx(2.21, abs=0.15) and frames / 60 == pytest.approx(0.467, abs=0.05)


def test_a_running_jump_clears_the_gap_of_1_3_and_a_walking_jump_does_not(tmp_path):
    for run, (lo, hi) in ((1, (16, 21)), (0, (12, 14))):
        core = _flat(tmp_path)
        for _ in range(60):
            core.step([2, run, 0])
        _, _, dist = _jump(core, 999, move=2, run=run)
        assert lo < dist / TS < hi, (run, dist / TS)     # the gap is 15 tiles; a walking jump went 18.6 u


def test_jump_is_ignored_while_it_is_held_from_before(tmp_path):
    core = _flat(tmp_path)
    for _ in range(80):                                     # held from the air down to the floor
        core.step([0, 0, 1])
    assert core.player.on_ground and core.player.vy < 20      # no buffered jump: let go and press again


def _shaft(tmp_path):
    """A tall room with Meat Boy beside the right wall."""
    return _core(tmp_path, ["#" + "." * 30 + "#" * 9] * 60 + ["#" + "." * 20 + "P" + "." * 9 + "#" * 9, "#" * 40])


def test_he_climbs_one_wall_by_wall_jumps_while_he_steers_back(tmp_path):
    for run, tiles in ((0, 10), (1, 20)):
        core = _shaft(tmp_path)
        for _ in range(20):
            core.step([0, 0, 0])
        y0, hold = core.player.y, 0
        for _ in range(600):
            p = core.player
            if hold == 0 and (p.on_ground or p.wall_dir == 1):
                hold = 20
            core.step([2, run, 1 if hold > 2 else 0])
            hold = max(0, hold - 1)
        assert core.player.wall_jumps > 5 and (y0 - core.player.y) / TS > tiles, (run, core.player.y)


def test_he_slides_only_while_held_into_the_wall_and_sticks_200_ms_when_pulling_away(tmp_path):
    core = _shaft(tmp_path)
    core.player.gObj.x, core.player.gObj.y = 31 * TS - 22, 20 * TS
    for _ in range(40):
        core.step([2, 0, 0])
    assert core.player.sliding and core.player.vy == 180.0
    x = core.player.x
    for _ in range(12):
        core.step([1, 0, 0])
    assert core.player.x == pytest.approx(x, abs=1)         # 12 frames on the wall
    for _ in range(8):                                      # then he goes
        core.step([1, 0, 0])
    assert core.player.x < x - 1 and not core.player.wall_dir
    core.player.gObj.x, core.player.gObj.y, core.player.vy = 31 * TS - 22, 20 * TS, 0.0
    for _ in range(5):
        core.step([2, 0, 0])
    for _ in range(40):                                     # let go: he falls at full speed
        core.step([0, 0, 0])
    assert core.player.vy == 500.0 and core.player.wall_dir == 1


def test_a_crumbling_block_breaks_after_a_touch_and_is_back_on_the_next_attempt(tmp_path):
    core = _core(tmp_path, ["#" + "." * 10 + "#"] * 5 + ["#..P.......#", "#%%%%%%%%%%#", "#..........#", "############"])
    assert core.level_data.grid[6][3] == 7
    for _ in range(int(60 * 0.75) + 20):                    # lands, then 0.75 s
        core.step([0, 0, 0])
    assert core.level_data.grid[6][3] == 0
    core.reset()
    assert core.level_data.grid[6][3] == 7


def test_fire_and_saws_kill_and_the_blood_stays_for_the_next_attempt(tmp_path):
    core = _core(tmp_path, ["#" + "." * 10 + "#"] * 5 + ["#..P.......#", "#..^.......#", "############"])
    for _ in range(30):
        core.step([0, 0, 0])
    assert not core.alive and core.death_cause == "Fire"
    core.render(pygame.Surface((core.WIDTH, core.HEIGHT)))
    blood = set(core._blood)
    core.reset()
    assert core.alive and core._blood == blood
    a = make_adapter("meatboy", "5", 600, 0.0)              # 1-6 Bladecatcher has saws
    saw = a.core.level_data.saws[0]
    a.core.player.gObj.x, a.core.player.gObj.y = saw.cx - 11, saw.cy - 11
    a.step(0, False)
    assert a.status == "DEAD" and a.core.death_cause == "Saw"


def test_the_gap_is_won_by_a_run_and_a_jump_from_the_ledge():
    a = make_adapter("meatboy", "2", 600, 0.0)
    a.sprint = True
    while a.alive:
        a.step(1, a.x > 380)
    assert a.won and a.core._steps / 60 < 1.6                 # the record is 1.37 s


def test_the_agents_see_walls_saws_and_a_rising_jump():
    from code.neuro.sensors import read_sensors
    a = make_adapter("meatboy", "2", 600, 0.0)
    vec, rays, _ = read_sensors(a)
    assert vec.shape == (21,)
    near = vec[19]
    a.core.player.gObj.x, a.core.player.gObj.y = 300.0, 330.0   # under the left saw
    vec, _, _ = read_sensors(a)
    assert vec[19] < near and vec[18] < 0                      # nearer, and above him
