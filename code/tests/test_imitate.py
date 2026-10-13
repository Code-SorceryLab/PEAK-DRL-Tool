"""Imitation: labels from held buttons, the fit, the saved run folder, and seeding a population."""
import json
import os

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["PEAK_MM1_ASSETS"] = "/nonexistent"
os.environ["PEAK_MM1_ATTEMPTS"] = "1"

from code.neuro import imitate
from code.neuro.adapters import N_INPUTS_BY_GAME
from code.neuro.evolution import GAConfig, Population
from code.neuro.net import make_net
from code.neuro.relay import load_net


def test_labels_spread_a_fire_tap_and_split_the_stick():
    held = np.zeros((40, 4), dtype=np.int8)
    held[5:10, 0] = -1                                   # left
    held[12:15, 2] = -1                                  # up
    held[20, 3] = 1                                      # one tap of fire
    y = imitate.labels(held, fire_window=8)
    assert y[5:10, 0].all() and not y[:, 1].any()        # left, never right
    assert y[12:15, 3].all() and not y[:, 4].any()       # up, never down
    assert y[12:29, 5].all() and not y[11, 5] and not y[29, 5]   # the tap, 8 frames either side


def test_only_sections_the_player_got_through_are_learned():
    first, second = {(0, 0)}, {(1, 0)}
    def demo(screens, status):
        n = len(screens)
        return {"meta": {"status": status}, "sensors": np.zeros((n, 26), np.float32),
                "held": np.tile(np.array([1, 0, 0, 0], np.int8), (n, 1)),      # holding right throughout
                "where": np.array([(sx, sy, 0, 0, 28) for sx, sy in screens])}
    through = demo([(0, 0)] * 100 + [(1, 0)] * 80, "DEAD")   # cleared the first, died in the second
    stuck = demo([(0, 0)] * 100, "DEAD")
    x, _, counts = imitate.section_frames([through, stuck], first, second)
    assert counts == [100] and len(x) == 100
    assert imitate.section_frames([through], second, set())[0] is None
    assert imitate.section_frames([through], second, set(), include_losses=True)[2] == [80]
    won = demo([(0, 0)] * 100 + [(1, 0)], "WON")           # a section win ends on its first frame next door
    assert imitate.section_frames([won], second, set())[0] is None
    waiting = demo([(0, 0)] * 100 + [(1, 0)] * 80, "DEAD")
    waiting["held"][:30] = 0                               # 30 frames standing still with nothing held
    assert imitate.section_frames([waiting], first, second)[2] == [70]   # ...are not copied


def test_fit_then_seed_a_population(tmp_path):
    rng = np.random.default_rng(0)
    n_in = N_INPUTS_BY_GAME["megaman"]
    x = rng.uniform(-1, 1, (3000, n_in)).astype(np.float32)
    y = np.zeros((3000, 6), np.float32)
    y[:, 1] = x[:, 18] > 0                               # go right when the compass says ahead
    y[:, 3] = x[:, 16] > 0.5                             # press up on a ladder
    params, report = imitate.fit(x, y, hidden=16, epochs=300)
    assert report["accuracy"] > 0.97 and report["outputs"]["up"]["recall"] > 0.9
    run = imitate.to_run(params, str(tmp_path / "imitation"), hidden=16, memory=2, report=report)
    net, sensors, _ = load_net(run)                      # the relay, replay and --seed-from all read this
    assert sensors == "rays" and net.memory == 2
    probe = np.zeros(n_in, np.float32)
    probe[16], probe[18] = 1.0, 1.0
    move_x, jump, move_y, fire = net.act(probe)
    assert move_x == 1 and move_y == -1

    cfg = GAConfig(pop_size=6, hidden=16, memory=2, n_inputs=n_in, n_outputs=6)
    pop = Population(cfg, make_net(cfg).n_params)
    seed = np.load(os.path.join(run, "best.npz"))["weights"]
    pop.seed_from(seed)
    assert np.array_equal(pop.weights[0], seed) and not np.array_equal(pop.weights[1], seed)
    assert np.mean(pop.weights[1] != seed) < 0.5         # copies differ in a share of weights only
    with pytest.raises(ValueError):
        Population(GAConfig(pop_size=6, hidden=8, n_inputs=n_in, n_outputs=6),
                   make_net(GAConfig(hidden=8, n_inputs=n_in, n_outputs=6)).n_params).seed_from(seed)
    assert json.load(open(os.path.join(run, "state.json")))["imitation"]["frames"] == 3000
