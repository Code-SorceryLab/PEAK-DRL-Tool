"""Unit tests for code.scripts.train._resolve_device.

Mocks torch.cuda.is_available and torch.backends.mps.is_available/is_built
to cover all 4 specs × 4 availability scenarios + invalid spec + env-var
side effect + stdout format.
"""
import os
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from code.scripts.train import _resolve_device


@contextmanager
def _mock_backends(cuda_ok=False, mps_ok=False):
    with patch("code.scripts.train.torch.cuda.is_available", return_value=cuda_ok), \
         patch("code.scripts.train.torch.backends.mps.is_available", return_value=mps_ok), \
         patch("code.scripts.train.torch.backends.mps.is_built", return_value=mps_ok):
        yield


# --- auto ---

def test_auto_picks_cuda_when_available():
    with _mock_backends(cuda_ok=True, mps_ok=False):
        assert _resolve_device("auto", verbose=False) == "cuda"


def test_auto_picks_mps_when_only_mps_available():
    with _mock_backends(cuda_ok=False, mps_ok=True):
        assert _resolve_device("auto", verbose=False) == "mps"


def test_auto_falls_back_to_cpu_when_no_backend():
    with _mock_backends(cuda_ok=False, mps_ok=False):
        assert _resolve_device("auto", verbose=False) == "cpu"


def test_auto_prefers_cuda_over_mps():
    with _mock_backends(cuda_ok=True, mps_ok=True):
        assert _resolve_device("auto", verbose=False) == "cuda"


# --- cuda ---

def test_cuda_returns_cuda_when_available():
    with _mock_backends(cuda_ok=True):
        assert _resolve_device("cuda", verbose=False) == "cuda"


def test_cuda_falls_back_to_cpu_when_unavailable():
    with _mock_backends(cuda_ok=False):
        assert _resolve_device("cuda", verbose=False) == "cpu"


# --- mps ---

def test_mps_returns_mps_when_available():
    with _mock_backends(mps_ok=True):
        assert _resolve_device("mps", verbose=False) == "mps"


def test_mps_falls_back_to_cpu_when_unavailable():
    with _mock_backends(mps_ok=False):
        assert _resolve_device("mps", verbose=False) == "cpu"


# --- cpu ---

def test_cpu_always_returns_cpu():
    with _mock_backends(cuda_ok=True, mps_ok=True):
        assert _resolve_device("cpu", verbose=False) == "cpu"


# --- invalid spec ---

def test_unknown_spec_falls_back_to_cpu():
    with _mock_backends(cuda_ok=True):
        assert _resolve_device("xpu", verbose=False) == "cpu"


def test_none_spec_treated_as_auto():
    # cfg.get("device", "auto") usually returns "auto", but defensive: None → cpu (safe default).
    with _mock_backends(cuda_ok=False, mps_ok=False):
        assert _resolve_device(None, verbose=False) == "cpu"


# --- env-var side effect ---

def test_mps_sets_fallback_env_var():
    with _mock_backends(mps_ok=True):
        assert "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ
        _resolve_device("mps", verbose=False)
        assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"


def test_cpu_does_not_set_fallback_env_var():
    with _mock_backends(mps_ok=True):
        _resolve_device("cpu", verbose=False)
        assert "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ


def test_cuda_does_not_set_fallback_env_var():
    with _mock_backends(cuda_ok=True):
        _resolve_device("cuda", verbose=False)
        assert "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ


# --- resolution stdout ---

def test_resolution_message_for_auto_mps(capsys):
    with _mock_backends(mps_ok=True):
        _resolve_device("auto", verbose=True)
    out = capsys.readouterr().out
    assert "Device requested: 'auto'" in out
    assert "resolved: 'mps'" in out
    assert "Apple Silicon" in out


def test_resolution_message_for_cuda_fallback(capsys):
    with _mock_backends(cuda_ok=False):
        _resolve_device("cuda", verbose=True)
    out = capsys.readouterr().out
    assert "Device requested: 'cuda'" in out
    assert "resolved: 'cpu'" in out
    assert "CUDA not available" in out
