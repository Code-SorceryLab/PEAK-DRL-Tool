"""Test fixtures shared across the suite."""
import os
import pytest


@pytest.fixture(autouse=True)
def _clear_mps_fallback_env(monkeypatch):
    """Each test starts with no PYTORCH_ENABLE_MPS_FALLBACK set."""
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
