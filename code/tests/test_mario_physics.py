import os

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from code.games.modules.Parameters import Movement_parameters as MP
from code.games.platformer_core import PlatformerCore

TS = 32
IDLE, WALK, RUN = 0, 3, 4

FLAT = ["." * 80] * 13 + ["...P" + "." * 76] + ["#" * 80] * 2


def _core(tmp_path, rows):
    """A PlatformerCore on a small throwaway level (same injection manual_play uses)."""
    path = tmp_path / "level.txt"
    path.write_text("\n".join(rows) + "\n")
    core = PlatformerCore(render_mode="none", lock_level=True, curriculum_enabled=False,
                          skip_obs=True, dijkstra_enabled=False)
    core.config_manager.yaml_data["levels"]["__test__"] = {"file": str(path), "time_limit": 300}
    core.level_order.append("__test__")
    core.world = core.locked_level = "__test__"
    core.max_lives = 1
    core.reset()
    for _ in range(30):  # settle onto the floor
        core.step([IDLE, 0, 0])
    assert core.player.on_ground
    return core


def _top_speed(core, move):
    for _ in range(180):
        core.step([move, 0, 0])
    return abs(core.player.vx)


def _jump_height(core, hold_frames, move=IDLE, runup=0):
    """Peak height in tiles of one jump, with jump held for hold_frames."""
    for _ in range(runup):
        core.step([move, 0, 0])
    y0 = peak = core.player.gObj.y
    for t in range(150):
        core.step([move, 1 if t < hold_frames else 0, 0])
        peak = min(peak, core.player.gObj.y)
        if t > 5 and core.player.on_ground:
            break
    return (y0 - peak) / TS


def test_top_speeds_equal_the_constants(tmp_path):
    assert _top_speed(_core(tmp_path, FLAT), WALK) == pytest.approx(MP.MAX_WALK_SPEED)
    assert _top_speed(_core(tmp_path, FLAT), RUN) == pytest.approx(MP.MAX_RUN_SPEED)


def test_tap_and_hold_jumps_differ(tmp_path):
    tap = _jump_height(_core(tmp_path, FLAT), 2)
    hold = _jump_height(_core(tmp_path, FLAT), 120)
    assert hold - tap >= 2.0
    assert 3.5 <= hold <= 4.5


def test_holding_jump_gives_one_jump(tmp_path):
    core = _core(tmp_path, FLAT)
    jumps, was_grounded = 0, True
    for _ in range(300):
        core.step([IDLE, 1, 0])
        if was_grounded and not core.player.on_ground:
            jumps += 1
        was_grounded = core.player.on_ground
    assert jumps == 1


def test_running_jump_is_about_a_tile_higher(tmp_path):
    standing = _jump_height(_core(tmp_path, FLAT), 120)
    running = _jump_height(_core(tmp_path, FLAT), 120, move=RUN, runup=90)
    assert 0.6 <= running - standing <= 1.4


def test_releasing_direction_in_air_keeps_speed(tmp_path):
    core = _core(tmp_path, FLAT)
    for _ in range(90):
        core.step([RUN, 0, 0])
    core.step([RUN, 1, 0])
    for _ in range(3):
        core.step([RUN, 1, 0])
    takeoff = core.player.vx
    for _ in range(15):
        core.step([IDLE, 1, 0])
    assert not core.player.on_ground
    assert core.player.vx == pytest.approx(takeoff)


def test_running_jump_can_turn_back_in_the_air(tmp_path):
    """NES air control: a full run reverses within one held jump."""
    core = _core(tmp_path, FLAT)
    for _ in range(90):
        core.step([RUN, 0, 0])
    while core.player.on_ground or core.player.vx > 0:
        core.step([2, 1, 0])   # sprint left, jump held
        assert not core.player.on_ground or core.player.vx > 0, "landed before turning back"


def test_releasing_run_mid_jump_keeps_momentum(tmp_path):
    core = _core(tmp_path, FLAT)
    for _ in range(90):
        core.step([RUN, 0, 0])
    for _ in range(4):
        core.step([RUN, 1, 0])
    takeoff = core.player.vx
    for _ in range(15):
        core.step([WALK, 1, 0])   # still holding forward, B let go
    assert not core.player.on_ground
    assert core.player.vx == pytest.approx(takeoff)


def test_a_coin_block_pays_out_when_hit(tmp_path):
    """The coin counts the moment the block is bumped, once, and its pop then disappears."""
    core = _core(tmp_path, FLAT[:10] + ["...?" + "." * 76] + FLAT[11:])
    hit_at = None
    for t in range(90):
        core.step([IDLE, 1 if t < 30 else 0, 0])
        if hit_at is None and core.level_data.qblocks[0].hit:
            hit_at = t
            assert core.coins_total == 1
    assert hit_at is not None, "never reached the block"
    assert core.coins_total == 1
    assert not any(c.flyup for c in core.level_data.coins), "the pop never ended"


def test_big_mario_hit_standing_still_shrinks_in_place(tmp_path):
    """No hop: the feet stay on the floor, x stays put, and i-frames start."""
    core = _core(tmp_path, FLAT[:13] + ["...P...E" + "." * 72] + FLAT[14:])
    p = core.player
    p.power_machine.collect_mushroom()
    for _ in range(10):   # grow, then settle on the floor
        core.step([IDLE, 0, 0])
    assert p.powered_up and p.on_ground
    feet, x = p.gObj.y + p.gObj.height, p.gObj.x
    core.level_data.enemies[0].vx = -abs(core.level_data.enemies[0].vx) or -60.0   # walk into Mario
    for _ in range(240):
        core.step([IDLE, 0, 0])
        assert p.gObj.y + p.gObj.height == pytest.approx(feet, abs=2), "feet left the floor"   # resting jitter < 1 px; the hop was 32
        assert p.gObj.x == pytest.approx(x)
        if not p.powered_up:
            break
    assert not p.powered_up and core.alive and p.invincible_timer > 0


def test_player_drops_into_a_one_tile_gap(tmp_path):
    rows = FLAT[:14] + ["#" * 10 + "." + "#" * 69] * 2
    core = _core(tmp_path, rows)
    floor_y = core.player.gObj.y
    for _ in range(120):
        core.step([WALK, 0, 0])
        if core.player.gObj.y > floor_y + TS / 2 or not core.alive:
            break
    assert core.player.gObj.y > floor_y + TS / 2 or not core.alive


@pytest.mark.parametrize("drop_px", range(40, 72, 3))
def test_falling_onto_a_grounded_enemy_kills_it(tmp_path, drop_px):
    rows = FLAT[:13] + ["...P.....E" + "." * 70] + FLAT[14:]
    core = _core(tmp_path, rows)
    enemy = core.level_data.enemies[0]
    enemy.vx = 0.0  # stays put under the player
    p = core.player
    p.gObj.x = enemy.gObj.x + enemy.gObj.width / 2 - p.gObj.width / 2
    p.gObj.y = enemy.gObj.y - p.gObj.height - drop_px
    p.vx = p.vy = 0.0
    for _ in range(60):
        core.step([IDLE, 0, 0])
        if not enemy.gObj.active or not core.alive:
            break
    assert core.alive
    assert not enemy.gObj.active
