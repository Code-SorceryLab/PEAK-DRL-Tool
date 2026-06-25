"""Dijkstra obs-channel ablation: dijkstra_enabled=False zeros channel 3."""
import numpy as np
from code.games.platformer_core import PlatformerCore


def _first_obs(core):
    core.reset()
    obs, *_ = core.step([0, 0, 0])
    return obs


def test_dijkstra_channel_zeroed_when_disabled():
    core = PlatformerCore(render_mode="none", dijkstra_enabled=False, world="Mario1-1")
    assert core.dijkstra_enabled is False
    obs = _first_obs(core)
    grids = obs["grids"]
    assert grids.shape == (4, 21, 21)            # shape preserved (extractor unchanged)
    assert np.all(grids[3] == 0.0)               # channel 3 (Dijkstra) zeroed


def test_dijkstra_channel_active_by_default_and_carries_signal():
    core = PlatformerCore(render_mode="none", world="Mario1-1")  # enabled default
    assert core.dijkstra_enabled is True
    obs = _first_obs(core)
    grids = obs["grids"]
    assert grids.shape == (4, 21, 21)
    # the enabled Dijkstra channel carries a real (non-zero) navigational signal,
    # so the ablation above actually removes information.
    assert not np.all(grids[3] == 0.0)
