"""Population management: elitism + tournament selection + uniform crossover + gaussian mutation."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class GAConfig:
    pop_size: int = 10
    elite: int = 2
    tournament_k: int = 3
    crossover_rate: float = 0.7
    mutation_rate: float = 0.15
    mutation_sigma: float = 0.3
    init_sigma: float = 0.5
    max_frames: int = 3600       # per-episode frame budget (60s at 60fps)
    stuck_frames: int = 300      # frames without max_x gain before an env is marked STUCK
    win_bonus: float = 5000.0
    seed: int = 42


class Population:
    """Holds pop_size flat weight vectors and evolves them from fitness scores."""

    def __init__(self, cfg: GAConfig, n_params: int) -> None:
        self.cfg = cfg
        self.n_params = n_params
        self.rng = np.random.default_rng(cfg.seed)
        self.weights = self.rng.normal(0.0, cfg.init_sigma, (cfg.pop_size, n_params)).astype(np.float32)
        self.generation = 0
        self.best_fitness = -np.inf
        self.best_weights = self.weights[0].copy()
        self.history: list[dict[str, float]] = []  # per-gen {best, avg}

    def _tournament(self, fitnesses: np.ndarray) -> int:
        idx = self.rng.integers(0, self.cfg.pop_size, self.cfg.tournament_k)
        return int(idx[np.argmax(fitnesses[idx])])

    def evolve(self, fitnesses: list[float]) -> None:
        fit = np.asarray(fitnesses, dtype=np.float64)
        gen_best = int(np.argmax(fit))
        if fit[gen_best] > self.best_fitness:
            self.best_fitness = float(fit[gen_best])
            self.best_weights = self.weights[gen_best].copy()
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
            child = child + mut * self.rng.normal(0.0, self.cfg.mutation_sigma, self.n_params)
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
            history_best=np.array([h["best"] for h in self.history]),
            history_avg=np.array([h["avg"] for h in self.history]),
        )
        state = {
            "generation": self.generation,
            "best_fitness": self.best_fitness,
            "config": asdict(self.cfg),
            "rng_state": self.rng.bit_generator.state,
        }
        with open(os.path.join(run_dir, "state.json"), "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        np.savez_compressed(os.path.join(run_dir, "best.npz"), weights=self.best_weights)

    @classmethod
    def load(cls, run_dir: str) -> "Population":
        with open(os.path.join(run_dir, "state.json"), encoding="utf-8") as f:
            state = json.load(f)
        cfg = GAConfig(**state["config"])
        data = np.load(os.path.join(run_dir, "gen_state.npz"))
        pop = cls(cfg, int(data["weights"].shape[1]))
        pop.weights = data["weights"].astype(np.float32)
        pop.best_weights = data["best_weights"].astype(np.float32)
        pop.generation = int(state["generation"])
        pop.best_fitness = float(state["best_fitness"])
        pop.history = [
            {"best": float(b), "avg": float(a)}
            for b, a in zip(data["history_best"], data["history_avg"])
        ]
        pop.rng.bit_generator.state = state["rng_state"]
        return pop
