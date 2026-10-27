# code/tests/test_music.py: every tune renders a whole, clean loop.
import numpy as np

from code.games.modules.System import music


def test_every_tune_renders_its_bars_as_one_clean_loop():
    for name, tune in music.TUNES.items():
        bars = sum(len(tune[k][0]) for k in tune["form"])
        y = music.render(name, 8000)
        assert len(y) == int(bars * 240 / tune["bpm"] * 8000), name
        assert np.isfinite(y).all() and np.abs(y).max() <= 0.95 + 1e-9, name


def test_chord_degrees():
    assert music._chord("f#m7") == (6, {"1": 0, "3": 3, "5": 7, "7": 10, "8": 12})
    assert music._chord("bb")[0] == 10 and music._chord("bm")[0] == 11
    assert music._midi("a4") == 69 and music._midi("bb3") == 58
