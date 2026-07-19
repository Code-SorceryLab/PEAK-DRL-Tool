"""Regression: on_ground must be stable for a resting player.

Bug: on_ground was reset False each frame and only re-set True when a floor
collision *resolved* that frame. A resting player re-penetrates the floor by
only ~0.36px/frame, landing on pygame's exclusive integer edge every other
frame → on_ground flickered 0,1,0,1 at 50% while genuinely standing on flat
ground. That corrupted BOTH the on_ground obs scalar (frame_skip=4 phase-locks
the sampled value) AND ground-vs-air friction/acceleration. Fixed by a
purely-additive 2px ground probe (_settle_grounded).
"""
from code.games.platformer_core import PlatformerCore


def _settle(core, n=30):
    for _ in range(n):
        core.step([0, 0, 0])


def test_on_ground_stable_while_standing():
    core = PlatformerCore(render_mode="none", world="Mario1-1",
                          curriculum_enabled=False)
    core.reset()
    _settle(core)
    flags = [int(core.step([0, 0, 0])[-1].get("on_ground",
             core.player.on_ground)) for _ in range(24)]
    # was 50% before the fix; must be solidly grounded now.
    assert sum(flags) == len(flags), f"on_ground flickered while standing: {flags}"


def test_on_ground_scalar_matches():
    """The obs scalar [4] the agent learns from must agree with reality."""
    core = PlatformerCore(render_mode="none", world="Mario1-1",
                          curriculum_enabled=False)
    core.reset()
    _settle(core)
    for _ in range(12):
        obs, *_ = core.step([0, 0, 0])
        assert obs["scalars"][4] == 1.0, "on_ground obs scalar False while grounded"


def test_jump_still_leaves_ground():
    """The fix is additive — it must not glue the player to the floor."""
    core = PlatformerCore(render_mode="none", world="Mario1-1",
                          curriculum_enabled=False)
    core.reset()
    _settle(core)
    left = False
    for _ in range(30):
        core.step([3, 1, 0])            # run-right + jump
        if not core.player.on_ground and core.player.vy < 0:
            left = True
            break
    assert left, "player never became airborne after jumping"


def test_relands_after_jump():
    core = PlatformerCore(render_mode="none", world="Mario1-1",
                          curriculum_enabled=False)
    core.reset()
    _settle(core)
    for _ in range(30):
        core.step([3, 1, 0])
    landed = any(core.step([3, 0, 0]) and core.player.on_ground for _ in range(120))
    assert landed, "player never landed after a jump"


def test_not_grounded_at_jump_apex():
    """While genuinely airborne (mid-jump, near apex) on_ground must be False —
    guards against the probe causing infinite jumps."""
    core = PlatformerCore(render_mode="none", world="Mario1-1",
                          curriculum_enabled=False)
    core.reset()
    _settle(core)
    saw_airborne_false = False
    for _ in range(40):
        core.step([3, 1, 0])
        if abs(core.player.vy) < 50 and core.player.gObj.y < 400:  # near apex, up high
            assert core.player.on_ground is False
            saw_airborne_false = True
    assert saw_airborne_false, "test never observed an apex frame"
