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
    move_x, jump = net.act(np.zeros(2, dtype=np.float32))
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
