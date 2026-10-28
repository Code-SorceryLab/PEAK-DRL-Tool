# code/tests/test_music.py: every tune renders a whole, clean loop.
import numpy as np

from code.games.modules.System import music


def test_every_game_the_selectors_launch_has_a_tune():
    assert {"menu", "platformer", "megaman", "meatboy", "bomberman"} <= music.TUNES.keys()


def test_every_tune_renders_its_bars_as_one_clean_loop():
    for name, tune in music.TUNES.items():
        bars = sum(len(tune[k][0]) for k in tune["form"])
        assert bars >= 16, name
        y = music.render(name, 8000)
        assert len(y) == int(bars * 240 / tune["bpm"] * 8000), name
        assert np.isfinite(y).all() and np.abs(y).max() <= 0.95 + 1e-9, name


def test_the_loop_seam_matches_playing_it_twice(monkeypatch):
    # platformer has swing and the doubled harmony; drums off, as their hiss differs pass to pass
    tune = dict(music.TUNES["platformer"], gains={"drums": 0.0})
    monkeypatch.setitem(music.TUNES, "once", tune)
    monkeypatch.setitem(music.TUNES, "twice", dict(tune, form=tune["form"] * 2))
    y, y2 = music.render("once", 8000), music.render("twice", 8000)
    n = len(y)
    assert np.allclose(np.concatenate([y[-500:], y[:500]]), y2[n - 500:n + 500], atol=1e-6)


def test_chord_degrees():
    assert music._chord("f#m7") == (6, {"1": 0, "2": 2, "3": 3, "4": 5, "5": 7, "6": 8, "7": 10, "8": 12})
    assert music._chord("bb")[0] == 10 and music._chord("bm")[0] == 11
    assert music._midi("a4") == 69 and music._midi("bb3") == 58
    assert music._below(72, *music._chord("c")) == 67 and music._below(76, *music._chord("c")) == 72
