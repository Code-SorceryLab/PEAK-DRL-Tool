import os

import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from code.games.meatboy_core import MeatboyCore
from code.games.platformer_core import PlatformerCore


def _actions(nvec, n=600):
    """Fixed pseudo-random action list (no RNG state shared with the cores)."""
    return [[(t * 7 + i * 3 + (t // 13)) % k for i, k in enumerate(nvec)] for t in range(n)]


def test_mario_same_level_seed_actions_same_run():
    def run():
        core = PlatformerCore(render_mode="none", lock_level=True, curriculum_enabled=False,
                              skip_obs=True, dijkstra_enabled=False)
        core.reset(seed=7)
        trace = []
        for a in _actions((5, 2, 2)):
            core.step(a)
            p = core.player
            trace.append((p.gObj.x, p.gObj.y, p.vx, p.vy, core.score, core.alive,
                          tuple((e.gObj.x, e.gObj.y, e.gObj.active) for e in core.level_data.enemies)))
        return trace

    assert run() == run()


def test_meatboy_same_level_seed_actions_same_run():
    pygame.init()

    def run(level_idx):
        core = MeatboyCore(render_mode="none")
        core._level_idx = level_idx
        core.reset(seed=7)
        trace = []
        for a in _actions((3, 2, 2)):
            _, _, terminated, truncated, _ = core.step(a)
            p = core.player
            trace.append((p.x, p.y, p.vx, p.vy, core.alive, core.won,
                          tuple((s.cx, s.cy) for s in core.level_data.saws)))
            if terminated or truncated:
                core.won = False  # replay the same level instead of advancing
                core.reset(seed=7)
        return trace

    for level_idx in (0, 5, 8):
        assert run(level_idx) == run(level_idx)
