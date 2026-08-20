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


class FakeCore:
    """Meatboy-style core: builds a `window`-sized obs with the agent at the centre cell,
    a solid cell 2 tiles ahead, a hazard directly below, and an oracle channel 3."""

    window = 21  # the cores' default; the grid sensor resizes it

    def _obs(self):
        n, c = self.window, self.window // 2
        g = np.zeros((4, n, n), dtype=np.float32)
        g[0, c, c + 2] = 1.0   # solid, (row, col) = centre + 2 columns
        g[2, c + 1, c] = -1.0  # hazard directly below
        g[3, :, :] = 0.7       # dijkstra oracle: must NOT reach the vector
        return {"grids": g, "scalars": np.zeros(20, dtype=np.float32)}


def test_grid_mode_resizes_window_and_drops_dijkstra():
    from code.neuro.sensors import GRID_HALF, GRID_N, TILE_HIT, sensor_dim

    a = FakeAdapter()
    a.core = FakeCore()
    vec, rays, tiles = read_sensors(a, "grid")
    assert a.core.window == GRID_N == 11  # core now builds 11x11 directly, no crop
    assert vec.shape == (sensor_dim("grid"),) == (368,)
    win = vec[:363].reshape(3, 11, 11)
    assert win[0, GRID_HALF, GRID_HALF + 2] == 1.0 and win[2, GRID_HALF + 1, GRID_HALF] == -1.0
    assert win.max() <= 1.0 and win.min() >= -1.0  # the 0.7 oracle channel is gone
    assert vec[-1] == 1.0 and vec[-5] == 1.0  # bias, grounded
    assert rays == [] and len(tiles) == 1 and tiles[0][3] == TILE_HIT
    # the one solid cell (2 tiles ahead) outlines the tile at x = agent_tile + 2
    assert tiles[0][0] == (64.0 // 32) * 32 + 2 * 32


def test_sensor_dim_rejects_unknown_mode():
    import pytest
    from code.neuro.sensors import sensor_dim

    assert sensor_dim("rays") == 14
    with pytest.raises(ValueError):
        sensor_dim("pixels")
