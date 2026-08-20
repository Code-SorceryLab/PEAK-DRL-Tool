import os

import numpy as np
import pygame
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from code.neuro.adapters import make_adapter
from code.neuro.sensors import read_sensors


@pytest.mark.parametrize("game", ["mario", "megaman", "sonic", "meatboy"])
def test_adapter_smoke(game):
    adapter = make_adapter(game, None, max_frames=300, win_bonus=100.0)
    adapter.reset()
    assert adapter.alive and not adapter.won
    assert adapter.status == "RUNNING"
    assert adapter.tile_size > 0

    vec, rays, _tiles = read_sensors(adapter)
    assert vec.shape == (14,)
    assert not np.any(np.isnan(vec))
    assert len(rays) >= 6

    for _ in range(60):  # run right + occasional jump for a second
        adapter.step(1, True)
        if not adapter.alive:
            break
    assert adapter.fitness() >= 0.0

    cam = adapter.camera
    assert len(cam) == 2

    surf = pygame.Surface((adapter.core.WIDTH, adapter.core.HEIGHT))
    adapter.render(surf)
    assert pygame.surfarray.array3d(surf).std() > 0  # something got drawn

    adapter.reset()  # second reset stays on the same pinned level
    assert adapter.alive and adapter.status == "RUNNING"
