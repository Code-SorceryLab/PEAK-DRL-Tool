import os

import pygame
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from code.games.modules.Objects.FireBar import FireBar
from code.games.platformer_core import PlatformerCore
from code.neuro.adapters import list_levels, make_adapter

FLAT = ["." * 40] * 13 + ["...P" + "." * 36] + ["#" * 40] * 2


def _core(tmp_path, firebars):
    (tmp_path / "level.txt").write_text("\n".join(FLAT) + "\n")
    lines = ["dynamics:", "  firebars:"] + [f"  - {{x: {x}, y: {y}, angle: {a}, period: {p}}}" for x, y, a, p in firebars]
    (tmp_path / "level.yaml").write_text("\n".join(lines) + "\n")
    core = PlatformerCore(render_mode="none", lock_level=True, curriculum_enabled=False,
                          skip_obs=True, dijkstra_enabled=False)
    core.config_manager.yaml_data["levels"]["__test__"] = {"file": str(tmp_path / "level.txt"), "time_limit": 300}
    core.level_order.append("__test__")
    core.world = core.locked_level = "__test__"
    core.max_lives = 1
    core.reset()
    return core


def test_bar_geometry_and_rotation():
    bar = FireBar(cx=100.0, cy=100.0, length=6, period=4.0, angle=0.0)
    assert bar.balls()[0] == (100.0, 100.0) and bar.balls()[-1] == pytest.approx((180.0, 100.0))
    for _ in range(60):                       # a quarter turn clockwise: now pointing down
        bar.update(1 / 60.0)
    assert bar.balls()[-1] == pytest.approx((100.0, 180.0))
    assert bar.hits_rect(pygame.Rect(90, 170, 20, 20))
    assert not bar.hits_rect(pygame.Rect(170, 90, 20, 20))


def test_fire_bar_kills_small_mario(tmp_path):
    # Pivot 2 tiles right of the start, bar pointing straight at Mario's head height and not moving.
    core = _core(tmp_path, [(3 * 32 + 90, 13 * 32 + 16, 180, 1e9)])
    assert len(core.level_data.firebars) == 1
    for _ in range(40):
        core.step([0, 0, 0])
    assert not core.alive and core.death_cause == "Fire"


def test_fire_bar_out_of_reach_is_harmless(tmp_path):
    core = _core(tmp_path, [(20 * 32, 5 * 32, 0, 3.4)])
    for _ in range(300):
        core.step([0, 0, 0])
    assert core.alive


def test_world_1_4_is_listed_and_has_fire_bars():
    assert "Mario1-4" in list_levels("mario", include_disabled=True)
    adapter = make_adapter("mario", "Mario1-4", max_frames=600, win_bonus=0.0)
    adapter.reset()
    assert len(adapter.core.level_data.firebars) == 7
    assert adapter.core.level_data.goals, "the axe is the goal"
    assert adapter._count_enemies() == 0 and len(adapter.enemy_positions()) == 42   # fireballs are sensed, not counted
    for _ in range(30):
        adapter.step(0, False)
    assert adapter.alive and adapter.grounded
