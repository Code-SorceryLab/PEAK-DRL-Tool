"""Seed plumbing regression.

HISTORICAL BUG: run_paper_matrix.py passed `seed={N}` for its 3-seed matrix
(and grid.yaml defines `seed: 1234`), but train.py never consumed it — no
set_random_seed call, no seed kwarg to the SB3 algo — so all "seeds" ran
byte-identical configs and any per-seed CI computed over them was fabricated
variance. These tests lock the plumbing so the paper's multi-seed claims are
backed by genuinely different runs.
"""
import inspect

from omegaconf import OmegaConf

import code.scripts.train as train_mod
from code.scripts.train import _resolve_run_seed


# ── Resolver semantics ────────────────────────────────────────────────────

def test_resolver_reads_int_seed():
    assert _resolve_run_seed(OmegaConf.create({"seed": 42})) == 42


def test_resolver_reads_string_seed():
    # Hydra CLI overrides can arrive as strings.
    assert _resolve_run_seed(OmegaConf.create({"seed": "31337"})) == 31337


def test_resolver_grid_default():
    # grid.yaml ships seed: 1234 — the default run must be seeded.
    grid = OmegaConf.load("code/conf/grid.yaml")
    assert _resolve_run_seed(grid) == 1234


def test_resolver_allows_unseeded():
    assert _resolve_run_seed(OmegaConf.create({})) is None
    assert _resolve_run_seed(OmegaConf.create({"seed": None})) is None
    assert _resolve_run_seed(OmegaConf.create({"seed": "none"})) is None


# ── Wiring guards (fail if the plumbing is ever removed again) ────────────

def test_main_consumes_seed():
    src = inspect.getsource(train_mod.main)
    assert "_resolve_run_seed" in src, "main() no longer resolves cfg seed"
    assert "set_random_seed" in src, "main() no longer seeds global RNGs"
    assert 'train_kwargs["seed"]' in src, (
        "seed no longer passed to the SB3 algo — SB3 seeds the action space "
        "and VecEnv from this kwarg; without it multi-seed runs are identical"
    )


def test_matrix_seeds_are_distinct():
    # The paper matrix must sweep genuinely different seeds.
    from code.scripts.run_paper_matrix import SEEDS
    assert len(SEEDS) >= 3
    assert len(set(SEEDS)) == len(SEEDS)
