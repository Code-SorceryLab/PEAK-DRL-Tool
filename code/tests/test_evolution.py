import numpy as np

from code.neuro.evolution import GAConfig, Population
from code.neuro.net import NeuralNet


def test_net_param_roundtrip():
    net = NeuralNet()
    flat = np.arange(net.n_params, dtype=np.float32)
    net.set_weights(flat)
    out = net.forward(np.ones(net.n_inputs, dtype=np.float32))
    assert out.shape == (net.n_outputs,)
    assert np.all((out >= 0.0) & (out <= 1.0))


def test_act_conflict_resolves_by_argmax():
    net = NeuralNet(n_inputs=2, n_hidden=2, n_outputs=3)
    net.set_weights(np.zeros(net.n_params, dtype=np.float32))
    net._b2 = np.array([4.0, 6.0, -4.0], dtype=np.float32)  # left and right both fire, right wins
    move_x, jump, _ = net.act(np.zeros(2, dtype=np.float32))
    assert move_x == 1
    assert jump is False


def test_evolution_deterministic_with_seed():
    def evolve_once(seed: int) -> np.ndarray:
        pop = Population(GAConfig(seed=seed), n_params=50)
        pop.evolve(list(range(pop.cfg.pop_size)))
        return pop.weights

    a, b = evolve_once(7), evolve_once(7)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, evolve_once(8))


def test_elitism_preserves_best():
    pop = Population(GAConfig(seed=1), n_params=20)
    fitnesses = [float(i) for i in range(pop.cfg.pop_size)]
    best_before = pop.weights[np.argmax(fitnesses)].copy()
    pop.evolve(fitnesses)
    assert np.array_equal(pop.weights[0], best_before)
    assert np.array_equal(pop.best_weights, best_before)
    assert pop.best_fitness == max(fitnesses)


def test_save_load_roundtrip(tmp_path):
    pop = Population(GAConfig(seed=3), n_params=30)
    pop.evolve([float(i) for i in range(pop.cfg.pop_size)])
    pop.save(str(tmp_path))
    loaded = Population.load(str(tmp_path))
    assert loaded.generation == pop.generation
    assert np.array_equal(loaded.weights, pop.weights)
    assert np.array_equal(loaded.best_weights, pop.best_weights)
    # RNG state restored: both produce the identical next generation
    pop.evolve([1.0] * pop.cfg.pop_size)
    loaded.evolve([1.0] * loaded.cfg.pop_size)
    assert np.array_equal(loaded.weights, pop.weights)


# ── GA-sweep architecture knobs: hidden size, action feedback, memory units ──

def test_n_params_formula():
    from code.neuro.net import make_net
    for h, af, mem in [(16, False, 0), (8, False, 0), (32, False, 0), (64, False, 0),
                       (16, True, 0), (16, False, 2), (16, False, 3), (8, True, 3)]:
        net = make_net(GAConfig(hidden=h, action_feedback=af, memory=mem))
        assert net.n_params == (14 + 2 * af + mem + 1) * h + (h + 1) * (3 + mem)
    assert make_net(GAConfig()).n_params == 291
    assert [make_net(GAConfig(hidden=h)).n_params for h in (8, 32, 64)] == [147, 579, 1155]
    assert make_net(GAConfig(action_feedback=True)).n_params == 323
    assert [make_net(GAConfig(memory=m)).n_params for m in (2, 3)] == [357, 390]


def test_act_carries_feedback_and_memory():
    net = NeuralNet(n_inputs=2, n_hidden=2, n_outputs=3, feedback=True, memory=2)
    assert net.carry.shape == (4,) and not net.carry.any()
    net.set_weights(np.zeros(net.n_params, dtype=np.float32))
    net._b2[1] = 5.0    # right
    net._b2[2] = 5.0    # jump
    net._b2[4] = 5.0    # memory unit 1 fires
    move_x, jump, _ = net.act(np.zeros(2, dtype=np.float32))
    assert (move_x, jump) == (1, True)
    out = net.forward(np.zeros(6, dtype=np.float32))   # 2 sensors + 2 feedback + 2 memory
    assert np.allclose(net.carry[:2], [1.0, 1.0])          # decoded action fed back
    assert np.allclose(net.carry[2:], out[3:])             # memory outputs fed back
    net.reset()
    assert not net.carry.any()


def test_save_load_keeps_net_fields(tmp_path):
    import json
    pop = Population(GAConfig(seed=3, hidden=8, memory=2), n_params=50)
    pop.save(str(tmp_path))
    loaded = Population.load(str(tmp_path))
    assert loaded.cfg.hidden == 8 and loaded.cfg.memory == 2 and loaded.cfg.action_feedback is False
    meta = json.loads(str(np.load(tmp_path / "best.npz")["meta"]))
    assert meta["hidden"] == 8 and meta["memory"] == 2 and meta["action_feedback"] is False
