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


def test_resume_keeps_the_runs_own_persona_and_locked_level(tmp_path, monkeypatch):
    import sys
    from code.neuro import trainer as tr
    pop = Population(GAConfig(), 10)
    pop.persona, pop.locked_level = "novice", "Mario1-3"       # a locked novice probe
    pop.save(str(tmp_path))
    seen = {}

    class Fake:
        def __init__(self, game, level, cfg, run_dir, state=None, population=None, persona=None):
            seen.update(level=level, persona=persona.name)

        def run(self, gens):
            pass
    monkeypatch.setattr(tr, "Trainer", Fake)
    monkeypatch.setattr(sys, "argv", ["trainer", "--game", "mario", "--resume", str(tmp_path), "--no-serve"])
    tr.main()                                                   # what menu "Watch All Models" runs
    assert seen == {"level": "Mario1-3", "persona": "novice"}
