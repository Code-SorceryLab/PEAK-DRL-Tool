import os

import numpy as np
import pygame
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from code.neuro.adapters import make_adapter
from code.neuro.sensors import read_sensors


@pytest.fixture(scope="module")
def adapter():
    return make_adapter("mario", None, max_frames=600, win_bonus=5000.0)


def test_reset_and_state(adapter):
    adapter.reset()
    assert adapter.alive and not adapter.won
    assert adapter.status == "RUNNING"
    assert adapter.core.lives == 1
    assert adapter.x > 0 and adapter.y > 0


def test_step_and_fitness_progress(adapter):
    adapter.reset()
    for _ in range(120):  # run right for 2 seconds
        adapter.step(1, False)
        if not adapter.alive:
            break
    assert adapter.fitness() > 0.0


def test_sensors_on_real_level(adapter):
    adapter.reset()
    vec, rays, _tiles = read_sensors(adapter)
    assert vec.shape == (14,)
    assert not np.any(np.isnan(vec))
    assert len(rays) >= 6


def test_episode_terminates(adapter):
    adapter.reset()
    frames = 0
    while adapter.alive and frames < 700:
        adapter.step(0, False)  # stand still until the frame budget kills it
        frames += 1
    assert not adapter.alive
    assert adapter.status in ("DEAD", "STUCK")


def test_render_to_surface(adapter):
    adapter.reset()
    surf = pygame.Surface((adapter.core.WIDTH, adapter.core.HEIGHT))
    adapter.render(surf)
    px = pygame.surfarray.array3d(surf)
    assert px.std() > 0  # something got drawn


def test_unknown_game_raises():
    with pytest.raises(ValueError):
        make_adapter("tetris", None, 100, 0.0)
