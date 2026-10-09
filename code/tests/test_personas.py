import numpy as np

from code.neuro.adapters import make_adapter
from code.neuro.personas import PERSONAS, Hands


def _bomberman(weights=None):
    a = make_adapter("bomberman", "8", max_frames=300, win_bonus=100.0, weights=weights)
    a.reset()
    return a


def test_event_weight_is_a_share_of_the_goal():
    plain, killer, collector = _bomberman(), _bomberman(PERSONAS["killer"].weights), \
        _bomberman(PERSONAS["collector"].weights)
    n = killer.level_events()["kills"]
    assert n > 0
    for a in (plain, killer, collector):
        a.core.kills_total = 1
    base = plain.fitness()
    # killing 1 of n enemies pays 1/n of the full progress scale
    assert abs(killer.fitness() - (base + killer.progress_scale() / n)) < 1e-6
    # this level holds no power-ups: the collector's weight is dead, so it scores like `experienced`
    if collector.level_events()["coins"] == 0:
        assert collector.fitness() == base


def test_kill_share_is_capped_and_mario_counts_kills():
    a = make_adapter("mario", None, max_frames=60, win_bonus=0.0, weights={"kills": 1.0})
    a.reset()
    assert a.events()["kills"] == 0 and a.level_events()["kills"] > 0
    a.kills = 10_000
    assert a.fitness() == a._progress_fitness() + a.progress_scale()


def test_meatboy_has_no_kill_events():
    a = make_adapter("meatboy", "0", max_frames=60, win_bonus=0.0, weights={"kills": 1.0})
    a.reset()
    assert a.level_events()["kills"] == 0


def test_only_the_bad_player_slips_and_slips_last():
    act = (1, False, 0)
    good = Hands(PERSONAS["good"], np.random.default_rng(0))
    assert all(good(act) == act for _ in range(500))
    bad = Hands(PERSONAS["bad"], np.random.default_rng(0))
    out = [bad(act) != act for _ in range(6000)]
    share = sum(out) / len(out)
    assert 0.1 < share < 0.4  # ~0.03 starts/frame x 12 frames held, minus no-op draws
    runs = [len(r) for r in "".join("x" if o else "." for o in out).split(".") if r]
    assert max(runs) >= PERSONAS["bad"].slip_frames // 2  # a slip is held, not a 1-frame flicker
