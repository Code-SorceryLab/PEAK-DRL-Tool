import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

from code.games import art_png, bomberman_art, mario_art, megaman_art


def test_export_writes_each_sprite_as_drawn_and_keeps_edits(tmp_path, monkeypatch):
    pygame.init()
    monkeypatch.setattr(art_png, "ROOT", str(tmp_path))
    assert art_png.export("bomberman", bomberman_art.all_sprites(), bomberman_art.PAL) > 0
    assert art_png.export("megaman", megaman_art.all_sprites(), megaman_art.PAL) > 0

    # one PNG pixel per art pixel, in the art's colours
    brick = pygame.image.load(art_png.path("bomberman", "brick"))
    assert brick.get_size() == (16, 16)
    assert brick.get_at((1, 1)) == pygame.Color(*bomberman_art.PAL[bomberman_art.BRICK[1][1]])
    assert pygame.image.load(art_png.path("megaman", "mm_idle")).get_size() == (24, 24)

    # a second export leaves existing files (someone's edits) alone
    assert art_png.export("megaman", megaman_art.all_sprites(), megaman_art.PAL) == 0


def test_the_games_draw_an_edited_png(tmp_path, monkeypatch):
    pygame.init()
    monkeypatch.setattr(art_png, "ROOT", str(tmp_path))
    art_png.export("bomberman", bomberman_art.all_sprites(), bomberman_art.PAL)
    art_png.export("megaman", megaman_art.all_sprites(), megaman_art.PAL)
    art_png.export("mario", mario_art.all_sprites(), mario_art.PAL)
    for game, name in (("bomberman", "brick"), ("megaman", "mm_idle"), ("mario", "brick")):
        img = pygame.image.load(art_png.path(game, name))
        img.set_at((1, 1), (1, 2, 3, 255))
        pygame.image.save(img, art_png.path(game, name))
    art_png.load.cache_clear()
    bomberman_art.sprite.cache_clear()
    assert bomberman_art.get(bomberman_art.BRICK).get_at((2, 2))[:3] == (1, 2, 3)     # drawn 2x
    assert megaman_art.DrawnAssets(1.0).sprite("mm_idle").get_at((1, 1))[:3] == (1, 2, 3)
    mario_art._scaled.clear()
    assert mario_art.image("brick", 32, 32).get_at((2, 2))[:3] == (1, 2, 3)               # drawn 2x
    mario_art._scaled.clear()
    art_png.load.cache_clear()
    bomberman_art.sprite.cache_clear()


def test_meat_boy_draws_its_pngs(tmp_path, monkeypatch):
    from code.games import meatboy_art
    pygame.init()
    monkeypatch.setattr(art_png, "ROOT", str(tmp_path))
    assert art_png.export_surfaces("meatboy", meatboy_art.all_surfaces()) == 5 + 1 + 64 + 2 + 8   # Meat Boy, Bandage Girl, rock, crumble + spikes, fire
    img = pygame.image.load(art_png.path("meatboy", "crumble"))
    img.set_at((5, 5), (1, 2, 3, 255))
    pygame.image.save(img, art_png.path("meatboy", "crumble"))
    art_png.load.cache_clear()
    meatboy_art.crumble.cache_clear()
    assert meatboy_art.crumble(32).get_at((5, 5))[:3] == (1, 2, 3)
    art_png.load.cache_clear()
    meatboy_art.crumble.cache_clear()
