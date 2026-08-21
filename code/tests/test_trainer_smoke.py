"""One headless generation with every GA-sweep architecture knob on: the carry has the right
shape, the checkpoint round-trips, and the saved genome fits the net rebuilt from its config
(what replay() does)."""
import os

import numpy as np
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from code.neuro.evolution import GAConfig, Population
from code.neuro.net import make_net
from code.neuro.trainer import Trainer


def test_one_generation_with_feedback_and_memory(tmp_path):
    pygame.init()
    cfg = GAConfig(pop_size=2, max_frames=60, hidden=8, action_feedback=True, memory=2, seed=1)
    trainer = Trainer("mario", "Mario1-1", cfg, run_dir=str(tmp_path))
    trainer.run(max_gens=1, verbose=False)

    slot = trainer.slots[0]
    assert slot.net.carry.shape == (4,)          # 2 feedback + 2 memory
    assert slot.last_sensors.shape == (14,)      # sensor vector untouched by the carry

    loaded = Population.load(str(tmp_path))
    net = make_net(loaded.cfg)
    assert net.n_params == loaded.n_params == 8 * (14 + 2 + 2 + 1) + (8 + 1) * 5
    net.set_weights(np.load(tmp_path / "best.npz")["weights"])   # replay-style rebuild fits
