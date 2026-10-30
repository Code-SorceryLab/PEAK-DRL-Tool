import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame
import pytest

from code.games import speedy_art, speedy_level


def _n(stem, count):
    return [f"{stem}_{i}" for i in range(count)]


NAMES = [
    "idle_0", "lookup_0", "duck_0", "spring_0", "hurt_0", "dead_0",
    *_n("wait", 2), *_n("walk", 6), *_n("run", 4), *_n("roll", 5), *_n("skid", 2), *_n("push", 4), *_n("balance", 2),
    *_n("moto", 3), *_n("moto_smoke", 2), *_n("crab", 3), "crab_fire_0", "crab_ball",
    *_n("buzz", 2), "buzz_fire_0", *_n("buzz_flame", 2), "buzz_shot", *_n("chopper", 2),
    *_n("newtron_b", 3), *_n("newtron_g", 3), *_n("newtron_fly", 2), "newtron_shot",
    *_n("explode", 5), *_n("freed", 4), "points_100", "points_200", "points_500", "points_1000",
    *_n("sparkle", 4), *_n("shield", 2), *_n("star", 4),
    *_n("ring", 4), *_n("bigring", 2), "monitor", "monitor_broken",
    "icon_rings", "icon_shield", "icon_invincible", "icon_shoes", "icon_life",
    *_n("spring_y", 2), *_n("spring_r", 2), *_n("spring_side_y", 2), *_n("spring_side_r", 2),
    "spike", "rock", *_n("lamp", 2), "sign_robot", *_n("sign_spin", 2), "sign_hero", "log", "platform", "ledge_piece",
    "swing_link", "swing_anchor", "swing_plat", *_n("boss_ship", 2), "boss_hurt", "boss_ball", "chain_link",
    "capsule", "capsule_open", "palm_crown", "palm_trunk", *_n("sunflower", 2), *_n("flower", 2), "totem", *_n("waterfall", 4),
]


@pytest.fixture(scope="module", autouse=True)
def _pygame():
    pygame.init()


def test_every_sprite_is_there_and_has_pixels():
    for name in NAMES:
        img = speedy_art.sprite(name)
        assert img.get_flags() & pygame.SRCALPHA and img.get_bounding_rect().w > 0, name
        assert speedy_art.sprite(name, True).get_size() == img.get_size()


def test_no_two_frames_are_the_same_picture():
    assert len({tuple(rows) for rows in speedy_art.SPRITES.values()}) == len(speedy_art.SPRITES)


def test_palette_keeps_to_three_bits_per_channel():
    steps = {0x00, 0x24, 0x49, 0x6D, 0x92, 0xB6, 0xDB, 0xFF}
    assert all(set(rgb) <= steps for rgb in speedy_art.PAL.values())
    assert all(c == "." or c in speedy_art.PAL for rows in speedy_art.SPRITES.values() for r in rows for c in r)


def test_standing_frames_put_the_feet_at_centre_plus_19():
    for name in ("idle_0", "walk_0", "run_0", "push_0", "duck_0"):
        img = speedy_art.sprite(name)
        assert img.get_height() == 40 and img.get_bounding_rect().bottom == 40, name


def test_hero_turns_in_eighths():
    for rot in range(8):
        assert speedy_art.hero("walk_0", rot).get_bounding_rect().w > 0
    assert speedy_art.hero("walk_0", 0) is speedy_art.sprite("walk_0")
    assert speedy_art.hero("walk_0", 2).get_size() == speedy_art.sprite("walk_0").get_size()[::-1]


def test_terrain_chunk_is_clear_only_where_nothing_stands():
    level = speedy_level.load("green_hill_1")
    img = speedy_art.terrain_chunk(level, 0, 3)
    assert img.get_size() == (256, 256) and img.get_flags() & pygame.SRCALPHA
    solid = (level.masks_np[0] | level.masks_np[1])[768:1024, :256]
    alpha = pygame.surfarray.array_alpha(img).T
    assert np.all(alpha[solid] == 255) and np.all(alpha[~solid] == 0)


def test_the_screens_draw_headless():
    level = speedy_level.load("green_hill_1")
    s = pygame.Surface((320, 224))
    speedy_art.backdrop(s, 100.0, 800.0, 3)
    speedy_art.decor_back(s, level, 0, 760, 3)
    speedy_art.hud(s, 100, 3725, 0, 3, True)
    speedy_art.title_card(s, 2, 0.5)
    speedy_art.results(s, 2, 3000, 500, 12300, 0.9)
    speedy_art.title_screen(s, 0)
