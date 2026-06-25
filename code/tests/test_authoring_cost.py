"""Tests for the authoring-cost edit measurement."""
import pytest
from code.scripts.authoring_cost import _widen_pit, _char_diff


def test_char_diff_counts_changes_and_length_delta():
    assert _char_diff("####", "####") == 0
    assert _char_diff("####", "#  #") == 2
    assert _char_diff("####", "##") == 2  # length delta


def test_widen_pit_converts_ground_to_air():
    grid = [
        "....O....",
        "#########",
        "#########",
    ]
    new, rows = _widen_pit(grid, tiles=3)
    assert rows == 2  # bottom two ground rows
    # exactly 3 ground tiles per modified row became air
    for orig, mod in zip(grid[1:], new[1:]):
        assert mod.count(" ") - orig.count(" ") == 3
        assert mod.count("#") == orig.count("#") - 3


def test_widen_pit_raises_without_ground():
    with pytest.raises(ValueError):
        _widen_pit(["....", "...."], tiles=3)
