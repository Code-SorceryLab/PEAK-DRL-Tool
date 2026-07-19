"""Time-horizon & experience-buffer parameter regression tests.

Three knobs became per-run parameters:
- +gamma= / +n_steps= / +batch_size= ... (algo overrides — previously locked
  inside conf/algo/ppo.yaml, unreachable from the CLI). gamma is the time
  horizon (effective ≈ 1/(1-γ) decisions); n_steps × n_envs is the
  experience buffer per PPO update.
- PEAK_POTENTIAL_GAMMA env var syncs the PBRS shaping discount to the PPO
  gamma (Ng et al. policy invariance requires they match), read LAZILY so
  SubprocVecEnv workers and in-process eval envs both see the override.
- +time_limit= overrides each level's in-game clock (episode horizon in
  game-clock units).
"""
import inspect

import pytest

from code.rewards.train_platformer import _potential_gamma, _ScoreTracker, adept


def test_potential_gamma_default():
    assert _potential_gamma() == pytest.approx(0.99)


def test_potential_gamma_env_override_is_lazy(monkeypatch):
    monkeypatch.setenv("PEAK_POTENTIAL_GAMMA", "0.997")
    assert _potential_gamma() == pytest.approx(0.997)


def test_adept_uses_synced_gamma(monkeypatch):
    """The persona must pick up the overridden gamma at call time — a stale
    module-level constant would silently break PBRS policy invariance for
    any +gamma= run."""
    monkeypatch.setenv("PEAK_POTENTIAL_GAMMA", "0.997")
    t = _ScoreTracker()
    for d in (0.5, 0.4):
        info = {"lives": 3, "score": 0, "goal_dist": 10.0,
                "dijkstra_dist": d, "x_position": 100.0}
        t.step(info)
    comps = adept._core_fn(False, False, info, 0)
    g, scale = 0.997, 0.3
    expected = ((0.5 - g * 0.4) - (1.0 - g) * 0.5) * scale
    assert comps["potential"] == pytest.approx(expected)


def test_train_main_wires_overrides():
    import code.scripts.train as train_mod
    src = inspect.getsource(train_mod.main)
    assert "_ALGO_OVERRIDABLE" in src, "algo CLI-override merge removed"
    for key in ("gamma", "n_steps", "batch_size"):
        assert f'"{key}"' in src
    assert "PEAK_POTENTIAL_GAMMA" in src, "PBRS gamma sync removed"
    assert 'time_limit=cfg.get("time_limit"' in src, "time_limit plumb removed"


def test_time_limit_override_beats_level_config():
    from code.games.platformer_core import PlatformerCore
    core = PlatformerCore(render_mode="none", world="Mario1-1", time_limit=123)
    core.reset()
    assert core.timer == pytest.approx(123.0)
    assert core.timer_seconds == 123          # obs normalisation follows


def test_time_limit_default_unchanged():
    # curriculum off so world= actually pins the FULL level (with curriculum
    # on, reset loads the first slice Mario1-1a @ 120 — that's intended).
    from code.games.platformer_core import PlatformerCore
    core = PlatformerCore(render_mode="none", world="Mario1-1",
                          curriculum_enabled=False)
    core.reset()
    assert core.timer == pytest.approx(300.0)  # game_config.yaml per-level value


def test_slice_uses_short_time_limit():
    from code.games.platformer_core import PlatformerCore
    core = PlatformerCore(render_mode="none", world="Mario1-1a",
                          curriculum_enabled=False)
    core.reset()
    assert core.timer == pytest.approx(120.0)  # slices get a short horizon
