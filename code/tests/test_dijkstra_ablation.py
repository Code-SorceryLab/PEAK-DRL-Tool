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


# ── Scalar-leak regression ────────────────────────────────────────────────
# The historical bug: dijkstra_enabled=False zeroed grid channel 3 but the
# tracking scalars (global 17=dijkstra_dist, 18=step_dx, 19=step_dy) still
# filled from the live solver, handing the policy the exact unit
# gradient-to-goal — every "without Dijkstra" ablation run was silently
# invalid.

def test_ablation_masks_dijkstra_scalars():
    core = PlatformerCore(render_mode="none", dijkstra_enabled=False, world="Mario1-1")
    obs = _first_obs(core)
    scalars = obs["scalars"]
    assert scalars.shape == (20,)
    assert scalars[17] == 1.0                          # dijkstra_dist neutral default
    assert scalars[18] == 0.0 and scalars[19] == 0.0   # step_dx / step_dy masked


def test_enabled_scalars_carry_signal():
    core = PlatformerCore(render_mode="none", world="Mario1-1")
    obs = _first_obs(core)
    scalars = obs["scalars"]
    # With the solver on, the step-direction unit vector must be non-zero
    # (there is always a cheaper neighbouring tile away from spawn).
    assert (scalars[18] != 0.0) or (scalars[19] != 0.0)
    assert 0.0 <= scalars[17] < 1.0                    # a real normalised distance


def test_ablation_keeps_reward_path_alive():
    """Obs ablation must NOT ablate the reward: the adept persona's PBRS and
    alignment terms read the solver via _info(), which stays live."""
    core = PlatformerCore(render_mode="none", dijkstra_enabled=False, world="Mario1-1")
    core.reset()
    _, _, _, _, info = core.step([3, 0, 0])            # move right one frame
    assert core.dijkstra is not None                   # solver still built
    assert info.get("dijkstra_dist", -1.0) >= 0.0      # reward path still sees it
