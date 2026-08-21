"""Population management: elitism + tournament selection + uniform crossover + gaussian mutation."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class GAConfig:
    # These defaults are the GA-sweep baseline (menu 15 / code/neuro/gasweep.py): every sweep
    # axis varies exactly one of them against a literature-grounded low/high bound.
    pop_size: int = 10
    elite: int = 4          # raised from 2: population win rate plateaued at 23%
    tournament_k: int = 5   # raised from 3: stronger selection pressure toward winners
    crossover_rate: float = 0.7
    mutation_rate: float = 0.15
    mutation_sigma: float = 0.15
    init_sigma: float = 0.5
    max_frames: int = 3600       # per-episode frame budget (60s at 60fps)
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
            "hidden": self.cfg.hidden,
            "action_feedback": self.cfg.action_feedback,
            "memory": self.cfg.memory,
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
        pop = cls(cfg, int(data["weights"].shape[1]))
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
