"""Drift-guard: PPO discount (ppo.yaml gamma) must equal the PBRS shaping
discount (POTENTIAL_GAMMA in train_platformer.py).

Potential-based reward shaping is policy-invariant ONLY when the shaping
gamma equals the RL gamma (Ng, Harada & Russell 1999). If the two ever
drift apart, the shaping silently biases the learned policy. This test
fails loudly if anyone edits one without the other.
"""
from pathlib import Path

import yaml

# Repo root = three parents up from this file (code/tests/<file> -> repo root).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PPO_YAML = _REPO_ROOT / "code" / "conf" / "algo" / "ppo.yaml"


def _load_ppo_gamma() -> float:
    """Read the raw `gamma` scalar from code/conf/algo/ppo.yaml."""
    with open(_PPO_YAML, "r") as f:
        cfg = yaml.safe_load(f)
    return float(cfg["gamma"])


def test_ppo_gamma_is_the_retuned_value():
    # Phase-1 retune target. Update this AND ppo.yaml together if it ever changes.
    assert _load_ppo_gamma() == 0.997


def test_ppo_yaml_other_hyperparams_retuned():
    with open(_PPO_YAML, "r") as f:
        cfg = yaml.safe_load(f)
    assert float(cfg["gae_lambda"]) == 0.97
    assert float(cfg["vf_coef"]) == 0.5
    assert int(cfg["n_epochs"]) == 8


def test_potential_gamma_constant_matches_ppo_gamma():
    """The module-level shaping discount must equal the PPO gamma."""
    from code.rewards.train_platformer import POTENTIAL_GAMMA
    assert POTENTIAL_GAMMA == _load_ppo_gamma()


def test_no_persona_redefines_potential_gamma_locally():
    """Guard against re-introducing a per-persona local POTENTIAL_GAMMA that
    could drift from the module constant. There must be exactly ONE assignment
    `POTENTIAL_GAMMA =` in the source (the module-level one)."""
    src = (_REPO_ROOT / "code" / "rewards" / "train_platformer.py").read_text()
    assignments = [
        ln for ln in src.splitlines()
        if ln.strip().startswith("POTENTIAL_GAMMA") and "=" in ln and "==" not in ln
    ]
    assert len(assignments) == 1, (
        f"Expected exactly one POTENTIAL_GAMMA assignment (module-level), "
        f"found {len(assignments)}: {assignments}"
    )
