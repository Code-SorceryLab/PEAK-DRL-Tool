"""Unit tests for code.scripts.train._build_vecnorm_kwargs.

Verifies the TRAIN VecNormalize wrapper normalizes reward with a clip_reward
strictly above the 5.0 win bonus, while the EVAL wrapper leaves reward
un-normalized for interpretable evaluation. Also checks that the Dict-obs
norm_obs_keys handling is preserved for both train and eval.
"""
import gymnasium.spaces as spaces
import numpy as np
import pytest

from code.scripts.train import _build_vecnorm_kwargs

WIN_BONUS = 5.0


def _box_dict_space():
    return spaces.Dict({
        "image": spaces.Box(low=0, high=1, shape=(4, 8, 8), dtype=np.float32),
        "vec":   spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32),
        "flag":  spaces.Discrete(2),  # non-Box: must be excluded from norm_obs_keys
    })


def test_train_normalizes_reward_and_clip_above_win_bonus():
    kw = _build_vecnorm_kwargs(uses_dict_obs=False, obs_space=None, training=True)
    assert kw["norm_obs"] is True
    assert kw["norm_reward"] is True, "TRAIN must normalize reward"
    assert kw["clip_obs"] == 10.0
    assert kw["clip_reward"] > WIN_BONUS, (
        f"clip_reward={kw['clip_reward']} must exceed win bonus {WIN_BONUS} "
        "so the win signal is not clipped away"
    )


def test_eval_does_not_normalize_reward():
    kw = _build_vecnorm_kwargs(uses_dict_obs=False, obs_space=None, training=False)
    assert kw["norm_obs"] is True
    assert kw["norm_reward"] is False, "EVAL must leave reward un-normalized"
    assert kw["clip_obs"] == 10.0


def test_dict_obs_keys_preserved_for_train_and_eval():
    space = _box_dict_space()
    for training in (True, False):
        kw = _build_vecnorm_kwargs(uses_dict_obs=True, obs_space=space, training=training)
        assert kw["norm_obs_keys"] == ["image", "vec"], (
            "only Box subspaces should be normalized; Discrete must be excluded"
        )


def test_non_dict_obs_has_no_norm_obs_keys():
    kw = _build_vecnorm_kwargs(uses_dict_obs=False, obs_space=None, training=True)
    assert "norm_obs_keys" not in kw
