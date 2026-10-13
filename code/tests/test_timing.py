import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from code.games.platformer_core import PlatformerCore


def _mario(render_mode):
    core = PlatformerCore(render_mode=render_mode, lock_level=True, curriculum_enabled=False,
                          skip_obs=True, dijkstra_enabled=False)
    core.reset()
    return core


def test_mario_windowed_and_headless_share_physics():
    """Manual play (windowed) must run the same fixed 1/60 s step agents train on."""
    headless, windowed = _mario("none"), _mario("human")
    for t in range(300):
        action = [3, 1 if t % 50 < 15 else 0, 0]  # walk right, hop now and then
        headless.step(action)
        windowed.step(action)
    assert windowed.dt == headless.dt == 1 / 60.0
    a, b = headless.player, windowed.player
    assert (a.gObj.x, a.gObj.y, a.vx, a.vy) == (b.gObj.x, b.gObj.y, b.vx, b.vy)
