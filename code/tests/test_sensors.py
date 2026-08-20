import numpy as np

from code.neuro.sensors import HIT_ENEMY, HIT_SOLID, RAY_MAX_DIST, read_sensors


class FakeAdapter:
    """32px tiles: solid wall at x >= 160, floor at y >= 96, one enemy 100px ahead."""

    tile_size = 32
    x, y = 64.0, 64.0
    vx, vy = 50.0, 0.0
    grounded = True
    can_jump = True

    def solid_at(self, wx: float, wy: float) -> bool:
        return wx >= 160.0 or wy >= 96.0

    def enemy_positions(self):
        return [(164.0, 64.0), (500.0, 64.0)]

    def qblock_count_near(self, r_tiles: int) -> int:
        return 2


def test_forward_ray_hits_wall():
    vec, rays, _tiles = read_sensors(FakeAdapter())
    assert vec[0] < 1.0  # forward ray blocked by the wall at 96px
    assert abs(vec[0] * RAY_MAX_DIST - 96.0) <= 8.0  # within one march step
    x1, y1, x2, y2, hit = rays[0]
    assert hit == HIT_SOLID and x2 > x1 and y1 == y2


def test_enemy_corridor_picks_nearest():
    vec, rays, _tiles = read_sensors(FakeAdapter())
    assert abs(vec[6] * RAY_MAX_DIST - 100.0) < 1e-3
    assert any(r[4] == HIT_ENEMY for r in rays)


def test_scalars_and_ranges():
    vec, _, _t = read_sensors(FakeAdapter())
    assert vec[7] == 0.0        # floor below the probe point -> no pit
    assert vec[8] == 1.0        # grounded
    assert vec[11] == 1.0       # can_jump
    assert vec[12] == 2 / 5     # qblocks
    assert vec[13] == 1.0       # bias
    assert vec.shape == (14,)
    assert np.all(vec >= -1.0) and np.all(vec <= 1.0)


def test_pit_detected_when_no_floor():
    class PitAdapter(FakeAdapter):
        def solid_at(self, wx: float, wy: float) -> bool:
            return False

    vec, _, _t = read_sensors(PitAdapter())
    assert vec[7] == 1.0
    assert vec[0] == 1.0  # all rays clear


def test_rays_flip_when_moving_left():
    class LeftAdapter(FakeAdapter):
        vx = -50.0

    _, rays, _t = read_sensors(LeftAdapter())
    x1, _, x2, _, _ = rays[0]  # forward ray now points left
    assert x2 < x1
