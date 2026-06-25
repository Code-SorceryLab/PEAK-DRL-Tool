"""Guard: the platformer trains on the ORIGINAL Super Mario Bros levels only
(Mario1-1, Mario1-2) — NOT synthetic stage_* difficulty ramps.

Rationale: an earlier change registered hand-authored stage_1..stage_14 ramps
into the training curriculum. Per the project owner, training must use the real
SMB levels (and, separately, the real Super Meat Boy levels for meatboy), so the
synthetic stages must NOT appear in the platformer curriculum. This test fails on
purpose if they are ever re-registered.
"""
from code.games.platformer_core import PlatformerCore


def test_platformer_trains_on_original_smb_levels_only():
    core = PlatformerCore(render_mode="none")
    assert core.level_order == ["Mario1-1", "Mario1-2"], (
        f"platformer should train only on the original SMB levels, got {core.level_order}"
    )


def test_no_synthetic_stage_levels_registered():
    core = PlatformerCore(render_mode="none")
    stages = [lvl for lvl in core.level_order if str(lvl).startswith("stage_")]
    assert not stages, f"synthetic stage_* levels must not be in the curriculum: {stages}"
