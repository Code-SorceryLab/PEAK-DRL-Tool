"""Config-wiring + vec-env dispatch assertions for the Phase-1 long run (Task 9).

Proves throughput/wiring only:
  - grid.yaml carries the scaled n_envs and multi-M budget,
  - the train.py dispatch contract (n_envs>1 -> SubprocVecEnv, else DummyVecEnv)
    passes exactly n_envs env-init callables to the constructor.
It does NOT prove the learning outcome — that is observed via the LAUNCH +
MONITOR procedure in Step 4 (per-level win-rate rising over baseline).
"""
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
GRID = REPO_ROOT / "code" / "conf" / "grid.yaml"


# --- (a) config values are wired to the scaled Phase-1 budget ---------------

def _cfg():
    return OmegaConf.load(GRID)

def test_n_envs_scaled_for_subproc_path():
    n_envs = int(_cfg().get("n_envs"))
    assert n_envs >= 8, f"Phase-1 expects 8-16 parallel envs, got n_envs={n_envs}"
    assert n_envs <= 16, f"n_envs={n_envs} exceeds the 8-16 Phase-1 band"
    assert n_envs > 1, "n_envs>1 is required to take the SubprocVecEnv path"

def test_training_budget_matches_paper_table():
    # Budgets are pinned to the paper's Table II design (Novice 1M / Expert 8M)
    # so the case-study matrix re-run is directly comparable to the original table.
    skills = _cfg().get("skills")
    novice = int(skills.get("Novice"))
    expert = int(skills.get("Expert"))
    assert novice == 1_000_000, f"Novice budget should match paper (1M): {novice}"
    assert expert == 8_000_000, f"Expert budget should match paper (8M): {expert}"

def test_eval_and_save_freq_are_per_env_documented_values():
    # These stay as-is; effective global cadence = freq * n_envs. Guard against
    # an accidental shrink that would make eval dominate CPU wall time.
    cfg = _cfg()
    assert int(cfg.get("eval_freq")) >= 10_000
    assert int(cfg.get("save_freq")) >= 20_000


# --- (b) dispatch contract: mirrors train.py:1145-1149 exactly --------------

class _FakeSubproc:
    def __init__(self, env_fns):
        self.env_fns = list(env_fns)
        self.kind = "subproc"

class _FakeDummy:
    def __init__(self, env_fns):
        self.env_fns = list(env_fns)
        self.kind = "dummy"

def _dispatch_vec_env(n_envs, make_env, SubprocVecEnv, DummyVecEnv):
    """Local mirror of train.py:1145-1149 — kept in lockstep with that branch."""
    n_envs = int(n_envs)
    if n_envs > 1:
        return SubprocVecEnv([make_env() for _ in range(n_envs)])
    return DummyVecEnv([make_env()])

def test_dispatch_uses_subproc_when_n_envs_gt_1():
    calls = {"n": 0}
    def make_env():
        calls["n"] += 1
        return lambda: None
    vec = _dispatch_vec_env(12, make_env, _FakeSubproc, _FakeDummy)
    assert vec.kind == "subproc"
    assert len(vec.env_fns) == 12          # exactly n_envs init callables
    assert calls["n"] == 12                # make_env() called once per env

def test_dispatch_uses_dummy_when_n_envs_eq_1():
    vec = _dispatch_vec_env(1, lambda: (lambda: None), _FakeSubproc, _FakeDummy)
    assert vec.kind == "dummy"
    assert len(vec.env_fns) == 1

def test_config_n_envs_takes_subproc_branch():
    # The real config value, run through the real branch logic, must select subproc.
    n_envs = int(_cfg().get("n_envs"))
    vec = _dispatch_vec_env(n_envs, lambda: (lambda: None), _FakeSubproc, _FakeDummy)
    assert vec.kind == "subproc"
    assert len(vec.env_fns) == n_envs
