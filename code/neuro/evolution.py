"""Population management: elitism + tournament selection + uniform crossover + gaussian mutation."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict

import numpy as np

from .net import N_HIDDEN, NeuralNet
from .sensors import sensor_dim


@dataclass
class GAConfig:
    # These defaults are the GA-sweep baseline (menu 15 / code/neuro/gasweep.py): every sweep
    # axis varies exactly one of them against a literature-grounded low/high bound.
    pop_size: int = 20
    elite: int = 3          # raised from 2: population win rate plateaued at 23%
    tournament_k: int = 5   # raised from 3: stronger selection pressure toward winners
    crossover_rate: float = 0.7
    mutation_rate: float = 0.15
    mutation_sigma: float = 0.15
    init_sigma: float = 0.5
    max_frames: int = 6000       # per-episode frame budget (60s at 60fps)
    stuck_frames: int = 300      # frames without max_x gain before an env is marked STUCK
    advance_wins: int = 3        # wins in one generation before the curriculum advances a level
    anneal_factor: float = 0.5   # × mutation after a level's first win (1.0 = off). 2026-08-20 GA sweep: 0.5 beat 0.8 in 6/6 game×persona groups
    win_bonus: float = 5000.0
    seed: int = 42
    sensors: str = "rays"        # exteroception: "rays" (14 inputs) or "grid" (3x11x11 + body)
    # Network shape (see net.NeuralNet): hidden tanh units, previous action fed back as 2 inputs,
    # and N Jordan memory units (extra outputs looped back as inputs next frame).
    hidden: int = 16
    action_feedback: bool = False
    memory: int = 0

def _proto_for(sensors: str, n_params: int) -> NeuralNet | None:
    """Rebuild the net shape a saved weight vector came from, or None if it no longer fits.

    The output width is per-game now (adapters.BUTTONS), and state.json does not record
    the game, so recover it from the length instead:
        n_params = n_in*n_hid + n_hid + n_out*(n_hid + 1)
    A remainder or a non-positive n_out means the file predates a topology change, so the
    caller falls back to unscaled sigmas rather than guessing. param_scale() only touches
    the input layer, so an off-by-one in n_out could never be silently absorbed here.
    """
    n_in = sensor_dim(sensors)
    n_out, rem = divmod(n_params - n_in * N_HIDDEN - N_HIDDEN, N_HIDDEN + 1)
    if rem or n_out <= 0:
        return None
    return NeuralNet(n_in, N_HIDDEN, n_out)


class Population:
    """Holds pop_size flat weight vectors and evolves them from fitness scores."""

    def __init__(self, cfg: GAConfig, n_params: int, param_scale: np.ndarray | None = None) -> None:
        self.cfg = cfg
        self.n_params = n_params
        # Per-parameter sigma multiplier from NeuralNet.param_scale(); ones (= old
        # behaviour) when the caller has no net, as in the unit tests.
        self.scale = (np.ones(n_params, dtype=np.float32) if param_scale is None
                      else param_scale.astype(np.float32))
        self.rng = np.random.default_rng(cfg.seed)
        self.weights = (self.rng.normal(0.0, cfg.init_sigma, (cfg.pop_size, n_params))
                        * self.scale).astype(np.float32)
        self.generation = 0
        self.best_fitness = -np.inf
        self.best_weights = self.weights[0].copy()
        self.best_gen = 0  # generation (1-based, matches history rows) the all-time best was found
        self.best_level: str | None = None  # level the all-time best was earned on (set by the trainer)
        self.annealed = False  # mutation halved after the current level's first win (trainer-managed)
        # Per-gen result rows. evolve() writes best/avg; the trainer appends the
        # rest (median, wins, statuses, per-env episode stats, duration, ...).
        self.history: list[dict] = []

    def _tournament(self, fitnesses: np.ndarray) -> int:
        idx = self.rng.integers(0, self.cfg.pop_size, self.cfg.tournament_k)
        return int(idx[np.argmax(fitnesses[idx])])

    def evolve(self, fitnesses: list[float]) -> None:
        fit = np.asarray(fitnesses, dtype=np.float64)
        gen_best = int(np.argmax(fit))
        if fit[gen_best] > self.best_fitness:
            self.best_fitness = float(fit[gen_best])
            self.best_weights = self.weights[gen_best].copy()
            self.best_gen = self.generation + 1
        self.history.append({"best": float(fit.max()), "avg": float(fit.mean())})

        order = np.argsort(fit)[::-1]
        next_w = np.empty_like(self.weights)
        next_w[:self.cfg.elite] = self.weights[order[:self.cfg.elite]]

        for i in range(self.cfg.elite, self.cfg.pop_size):
            a = self.weights[self._tournament(fit)]
            if self.rng.random() < self.cfg.crossover_rate:
                b = self.weights[self._tournament(fit)]
                mask = self.rng.random(self.n_params) < 0.5
                child = np.where(mask, a, b)
            else:
                child = a.copy()
            mut = self.rng.random(self.n_params) < self.cfg.mutation_rate
            child = child + mut * self.rng.normal(0.0, self.cfg.mutation_sigma, self.n_params) * self.scale
            next_w[i] = child.astype(np.float32)

        self.weights = next_w
        self.generation += 1

    # ── persistence ────────────────────────────────────────────────────────

    def save(self, run_dir: str) -> None:
        os.makedirs(run_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(run_dir, "gen_state.npz"),
            weights=self.weights,
            best_weights=self.best_weights,
        )
        state = {
            "generation": self.generation,
            "best_fitness": self.best_fitness,
            "best_gen": self.best_gen,
            "best_level": self.best_level,
            "annealed": self.annealed,
            "persona": getattr(self, "persona", None),
            "config": asdict(self.cfg),
            "rng_state": self.rng.bit_generator.state,
            "history": self.history,
        }
        with open(os.path.join(run_dir, "state.json"), "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        # The tags travel inside the model file too, so a copied best.npz stays identifiable.
        meta = {
            "game": getattr(self, "game", None),
            "level": self.best_level,
            "persona": getattr(self, "persona", None),
            "fitness": round(float(self.best_fitness), 1),
            "generation": self.generation,
            "seed": self.cfg.seed,
            "sensors": self.cfg.sensors,
            "n_params": self.n_params,
        }
        np.savez_compressed(os.path.join(run_dir, "best.npz"),
                            weights=self.best_weights, meta=np.array(json.dumps(meta)))

    @classmethod
    def load(cls, run_dir: str) -> "Population":
        with open(os.path.join(run_dir, "state.json"), encoding="utf-8") as f:
            state = json.load(f)
        cfg = GAConfig(**state["config"])
        data = np.load(os.path.join(run_dir, "gen_state.npz"))
        n_params = int(data["weights"].shape[1])
        # Rebuild the sigma scale from the saved sensor mode; skip it if the file
        # predates a topology change and no longer matches this net.
        proto = _proto_for(cfg.sensors, n_params)
        pop = cls(cfg, n_params, proto.param_scale() if proto is not None else None)
        pop.weights = data["weights"].astype(np.float32)
        pop.best_weights = data["best_weights"].astype(np.float32)
        pop.generation = int(state["generation"])
        pop.best_fitness = float(state["best_fitness"])
        pop.best_gen = int(state.get("best_gen", 0))
        pop.best_level = state.get("best_level")
        pop.annealed = state.get("annealed", False)
        pop.history = state.get("history", [])
        pop.rng.bit_generator.state = state["rng_state"]
        return pop
