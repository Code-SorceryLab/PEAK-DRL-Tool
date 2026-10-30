"""SPEEDY: the numbers of the 1991 game's physics, the set pieces of Green Hill, and the training wiring."""
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pytest

from code.games import speedy_level as lv
from code.games.speedy_core import AIR_CAP, JUMP_CUT, TOP, SpeedyCore
from code.games.speedy_objects import Monitor, SpringUp
from code.neuro.adapters import list_levels, make_adapter
from code.neuro.sensors import read_sensors, sensor_dim


def core(level="green_hill_1"):
    return SpeedyCore(level=level)


def run(c, frames, move=1, jump=0, down=0):
    for _ in range(frames):
        c.step((move, jump, down))


def teleport(c, x, y):
    p = c.player
    p.x, p.y, p.air, p.vx, p.vy, p.gsp = x << 8, y << 8, True, 0, 0x100, 0
    c.camera_x, c.camera_y = max(0, x - 160), max(0, y - 96)
    for _ in range(40):
        c.step((0, 0, 0))
        if not p.air:
            return


def test_levels_build_and_list():
    assert list_levels("speedy") == ["green_hill_1", "green_hill_2", "green_hill_3"]
    for act, name in enumerate(list_levels("speedy"), 1):
        L = lv.load(name)
        assert L.act == act and len(L.loops) == 1 and len(L.coins) > 100
        assert L.solid(L.start[0], L.start[1] + 24)         # ground under the start
        assert not L.solid(*L.start)
        assert all(len(a) == len(L.angles[0][0]) for p in L.angles for a in p)


def test_run_tops_out_at_six_pixels_a_frame():
    c = core()
    run(c, 160)
    assert c.player.gsp == TOP
    run(c, 30)
    assert c.player.gsp == TOP


def test_jump_height_and_release():
    c = core()
    run(c, 2, 0, 0)
    y0 = c.player.py
    top = y0
    c.step((0, 1, 0))
    assert c.player.air and c.player.rolling and c.player.vy < 0
    for _ in range(60):
        c.step((0, 1, 0))
        top = min(top, c.player.py)
    assert 88 <= y0 - top <= 104                     # 6.5 px/frame against 0.21875 gravity
    c = core()
    run(c, 2, 0, 0)
    c.step((0, 1, 0))
    c.step((0, 0, 0))
    assert c.player.vy >= JUMP_CUT


def test_full_speed_runs_the_loop():
    c = core()
    teleport(c, 5120, 660)
    seen = set()
    for _ in range(240):
        c.step((1, 0, 0))
        seen.add(c.player.plane)
    assert c.player.px > 5640 and seen == {0, 1}     # up the right side on plane 0, down the left on plane 1
    assert not c.player.air


def test_tunnels_force_a_roll():
    c = core()
    teleport(c, 6100, 600)
    rolled = False
    for _ in range(200):
        c.step((1, 0, 0))
        rolled = rolled or (c.player.rolling and c._in_tube(c.player))
    assert rolled


def test_rings_pay_and_a_hit_scatters_them():
    c = core()
    p = c.player
    for x, y in c.rings[:3]:
        p.x, p.y, p.vx, p.vy, p.air = x << 8, y << 8, 0, 0, True
        c.step((0, 0, 0))
    assert c.rings_held == c.coins_total == 3
    c.hurt(p, p.px + 8)
    assert p.routine == "hurt" and p.vx == -0x200 and p.vy == -0x400 and c.rings_held == 0
    assert sum(hasattr(e, "last") or type(e).__name__ == "RingScatter" for e in c.extra) == 3
    c.step((0, 0, 0))
    c.player.invuln = 0
    p.routine = "control"
    c.rings_held = 0
    c.hurt(p, p.px)
    assert p.routine == "dead" and c.dead and c.death_cause == "Enemy"


def test_a_spike_hits_through_the_flashing():
    c = core()
    p = c.player
    p.invuln, c.rings_held = 60, 5
    c.hurt(p, p.px, spikes=False)
    assert p.routine == "control"                    # enemies respect the flashing
    c.hurt(p, p.px, spikes=True)
    assert p.routine == "hurt"                       # the 1991 spikes do not


def test_monitor_breaks_under_a_roll_and_pays_ten_rings():
    c = core()
    m = next(o for o in c.objs if isinstance(o, Monitor))
    p = c.player
    teleport(c, m.x, m.y - 80)
    p.rolling, p.air, p.vy = True, True, 0x300
    p.h, p.w = 14, 7
    run(c, 60, 0)
    assert m.broken and c.coins_total == 10


def test_spring_launches_at_ten():
    c = core()
    s = next(o for o in c.objs if isinstance(o, SpringUp))
    p = c.player
    teleport(c, s.x - 40, s.y - 10)
    p.x, p.y, p.air, p.vy = s.x << 8, (s.y - 40) << 8, True, 0x100
    c.camera_x = s.x - 160
    fastest = 0
    for _ in range(30):
        c.step((0, 0, 0))
        fastest = min(fastest, p.vy)
    assert fastest <= -0xA00 + 0x38


def test_rolling_kills_a_badnik_and_chain_pays_more():
    c = core()
    p = c.player
    run(c, 1, 0)
    moto = next(o for o in c.objs if type(o).__name__ == "Moto")
    moto.wake(c)
    moto.x, moto.y = p.px, p.py + 10
    p.rolling, p.h, p.w, p.vy, p.air = True, 14, 7, 0x200, True
    moto.touch(c, p)
    assert not moto.alive and c.kills_total == 1 and c.score == 100 and p.vy < 0


def test_a_ledge_gives_way_for_this_attempt_only():
    c = core()
    x0, y0, x1, y1 = c.level.objects["ledges"][0]
    p = c.player
    teleport(c, (x0 + x1) // 2, y0 - 30)
    assert not p.air
    run(c, 40, 0)
    assert p.air and c.collapsed
    assert not c.solid_at((x0 + x1) // 2, y0 + 20)
    c.reset()
    assert c.solid_at((x0 + x1) // 2, y0 + 20) and not c.collapsed


def test_time_limit_kills():
    c = core()
    c.frames = 10 * 3600 - 2
    run(c, 2, 0)
    assert c.dead and c.death_cause == "Time"


def test_same_inputs_same_game():
    def play():
        c = core()
        for f in range(1200):
            c.step((1, int(f % 50 < 30), 0))
        p = c.player
        return p.x, p.y, p.gsp, c.rings_held, c.score, c.coins_total
    assert play() == play()


def jumpy_bot(c, frames):
    """Hold right, and jump for most of the time when slow. Damage is off: this checks the map, not a player."""
    jt = 0
    for _ in range(frames):
        p = c.player
        p.invincible = 1000
        if jt > 0:
            jt -= 1
        elif jt < 0:
            jt += 1
        elif not p.air and abs(p.gsp) < 0x180:
            jt = 34
        else:
            jt = -8
        _, _, t, tr, _ = c.step((1, int(jt > 0), 0))
        if t or tr:
            return


def test_act_two_runs_to_the_sign():
    c = core("green_hill_2")
    jumpy_bot(c, 20000)
    assert c.won


def test_boss_takes_eight_hits_then_the_capsule_ends_act_three():
    c = core("green_hill_3")
    p = c.player
    p.x, p.y, p.air = 10760 << 8, 900 << 8, True
    c.camera_x, c.camera_y, c.bottom = 10592, 768, 768
    p.invincible = 100000                        # the ball is not what is under test
    run(c, 700, 0)                               # the camera locks and the machine lowers its ball
    b = c.boss
    assert b is not None and c.locked and b.chain == 128
    for _ in range(8):
        for _ in range(40):
            p.x, p.y, p.vx, p.vy, p.air, p.rolling = int(b.x) << 8, int(b.y + 20) << 8, 0, -0x100, True, True
            p.invuln = 60
            c.step((0, 0, 0))
    assert b.hits == 0 and b.state == "dying"
    run(c, 200, 0)
    assert not c.locked and c.cam_right > 10592
    cap = next(o for o in c.objs if type(o).__name__ == "Capsule")
    p.x, p.y, p.vx, p.vy, p.air, p.rolling = cap.x << 8, (cap.y - 60) << 8, 0, 0, True, False
    c.camera_x = c.cam_right
    run(c, 40, 0)
    assert c.won


@pytest.mark.parametrize("sensors", ["rays", "grid"])
def test_adapter_runs_and_ends(sensors):
    a = make_adapter("speedy", "green_hill_1", 3600, 1000.0)
    assert a.core.max_steps >= 10800
    vec, _, _ = read_sensors(a, sensors)
    assert vec.shape == (sensor_dim(sensors),)
    while a.alive:
        a.step(0, False, 0)
    assert a.status == "STUCK" and a.core.death_cause == "Stall"
    assert a.episode_stats()["level_coins"] == 153
    a.reset()
    assert a.alive and a.core.frames == 0 and a.core.rings_held == 0


def test_a_held_jump_output_repeats():
    a = make_adapter("speedy", "green_hill_1", 3600, 1000.0)
    jumps = 0
    was_air = False
    for _ in range(300):
        a.step(0, True, 0)
        jumps += a.core.player.air and not was_air
        was_air = a.core.player.air
    assert jumps >= 3
