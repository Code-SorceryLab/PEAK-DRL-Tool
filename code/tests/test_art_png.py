import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

from code.games import art_png, bomberman_art


def test_export_writes_each_sprite_as_drawn_and_keeps_edits(tmp_path, monkeypatch):
    pygame.init()
    monkeypatch.setattr(art_png, "ROOT", str(tmp_path))
    assert art_png.export("bomberman", bomberman_art.all_sprites(), bomberman_art.PAL) > 0

    # one PNG pixel per art pixel, in the art's colours
    brick = pygame.image.load(art_png.path("bomberman", "brick"))
    assert brick.get_size() == (16, 16)
    assert brick.get_at((1, 1)) == pygame.Color(*bomberman_art.PAL[bomberman_art.BRICK[1][1]])

    # a second export leaves existing files (someone's edits) alone
    assert art_png.export("bomberman", bomberman_art.all_sprites(), bomberman_art.PAL) == 0


def test_the_games_draw_an_edited_png(tmp_path, monkeypatch):
    pygame.init()
    monkeypatch.setattr(art_png, "ROOT", str(tmp_path))
    art_png.export("bomberman", bomberman_art.all_sprites(), bomberman_art.PAL)
    for game, name in (("bomberman", "brick"),):
        img = pygame.image.load(art_png.path(game, name))
        img.set_at((1, 1), (1, 2, 3, 255))
        pygame.image.save(img, art_png.path(game, name))
    art_png.load.cache_clear()
    bomberman_art.sprite.cache_clear()
    assert bomberman_art.get(bomberman_art.BRICK).get_at((2, 2))[:3] == (1, 2, 3)     # drawn 2x
    art_png.load.cache_clear()
    bomberman_art.sprite.cache_clear()
