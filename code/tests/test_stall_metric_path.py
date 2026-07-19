"""Path-based anti-stall metric regression tests.

The legacy watchdog measures progress as EUCLIDEAN (straight-line) distance
to the goal. On Mario1-2, correct play — riding a vertical platform, climbing
a shaft — does not shrink straight-line distance, so the watchdog executed
the agent for progressing (measured: stall = 14 of 28 eval failures).
stall_metric="path" measures along the Dijkstra route instead, with a
Euclidean fallback when the tile reading is invalid, so genuine stalls
still die.
"""
from code.games.platformer_core import PlatformerCore


class _FakeDijkstra:
    """get_dist returns scripted values (a new one per call)."""
    def __init__(self, values):
        self.values = list(values)

    def get_dist(self, px, py):
        return self.values.pop(0) if len(self.values) > 1 else self.values[0]


def _core(**kwargs):
    core = PlatformerCore(render_mode="none", world="Mario1-1", **kwargs)
    core.reset()
    core.dt = 1.0                      # 1 simulated second per watchdog tick
    core.stall_window = 2.0
    return core


def _tick(core, n):
    windows_before = core.stall_windows_count
    for _ in range(n):
        core._update_stall_metrics()
    return core.stall_windows_count - windows_before


def test_default_metric_is_euclid():
    core = _core()
    assert core.stall_metric == "euclid"
    core.close() if hasattr(core, "close") else None


def test_kwarg_selects_path_metric():
    core = _core(stall_metric="path")
    assert core.stall_metric == "path"


def test_path_progress_resets_stall_even_without_euclid_progress():
    """Vertical play: path distance falls step by step while the player's
    position (and thus Euclidean distance) is pinned — the watchdog must NOT
    accrue stall windows."""
    core = _core(stall_metric="path")
    # Path distance improves by 1 cost-unit per tick (> the 0.5 threshold).
    core.dijkstra = _FakeDijkstra([100 - i for i in range(200)])
    # Freeze Euclidean progress entirely.
    core._goal_dist_cache = core.best_dist_to_goal
    assert _tick(core, 10) == 0            # no stall windows while pathing


def test_path_stagnation_still_dies():
    """Genuine stall (constant path distance) must accrue windows exactly like
    the legacy metric — the watchdog stays live."""
    core = _core(stall_metric="path")
    core.dijkstra = _FakeDijkstra([50.0])  # never improves
    core._goal_dist_cache = core.best_dist_to_goal
    # Tick 1 establishes the anchor (inf → 50 counts as progress, same as
    # the legacy metric's init); the remaining 9 stagnant ticks accrue one
    # window per stall_window=2.0 sim-seconds at dt=1.0 → 4 windows.
    assert _tick(core, 10) == 4


def test_invalid_path_reading_falls_back_to_euclid():
    """Mid-air / unreachable tile (get_dist < 0): the Euclidean test must take
    over, so Euclid progress still resets the timer."""
    core = _core(stall_metric="path")
    core.dijkstra = _FakeDijkstra([-1.0])  # tile never resolvable
    # Euclidean distance improves 20px per tick (> 16px threshold).
    core.best_dist_to_goal = 10_000.0
    windows = 0
    for i in range(10):
        core._goal_dist_cache = 10_000.0 - 20.0 * (i + 1)
        core._update_stall_metrics()
        windows = core.stall_windows_count
    assert windows == 0


def test_euclid_mode_ignores_dijkstra():
    """Legacy mode must not consult the solver at all — behavior unchanged."""
    core = _core(stall_metric="euclid")
    class _Boom:
        def get_dist(self, px, py):
            raise AssertionError("euclid mode must not call the solver")
    core.dijkstra = _Boom()
    core._goal_dist_cache = core.best_dist_to_goal   # no euclid progress
    assert _tick(core, 4) == 2                       # normal stall accrual
