"""PEAK's pixel art (code/neuro/pixel_art.py): clean grids, a font whose 2 is never an S, and a
stylesheet that carries what the grids say."""
from code.neuro import pixel_art as pa


def test_every_grid_is_rectangular_and_uses_known_colours():
    for name, rows in pa.ICONS.items():
        assert len({len(r) for r in rows}) == 1, name
        assert set("".join(rows)) <= set(pa.COLORS) | {"."}, name
    for ch, rows in pa.FONT.items():
        assert len({len(r) for r in rows}) == 1 and len(rows) <= 9, ch
    assert set(pa.PERSONA_ICON.values()) <= set(pa.ICONS)


def test_the_digits_do_not_pass_for_letters():
    for a, b in (("2", "S"), ("2", "Z"), ("5", "S"), ("0", "O"), ("1", "I"), ("8", "B")):
        assert pa.FONT[a] != pa.FONT[b], (a, b)
    assert pa.FONT["2"][-1] == "#####"                  # the 2 stands on a flat base; an S does not
    assert pa.FONT["S"][-1] != "#####"


def test_the_font_builds_the_same_every_time_and_maps_its_glyphs():
    import io

    from fontTools.ttLib import TTFont
    data = pa.build_font()
    assert data == pa.build_font()
    cmap = TTFont(io.BytesIO(data)).getBestCmap()
    assert all(ord(ch) in cmap for ch in pa.FONT) and 32 in cmap


def test_the_stylesheet_holds_the_current_art():
    with open(pa.THEME, encoding="utf-8") as f:
        theme = f.read()
    assert pa.css() in theme, "run `python -m code.neuro.pixel_art` after editing the grids"
