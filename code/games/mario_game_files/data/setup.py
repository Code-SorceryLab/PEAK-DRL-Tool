import os
import pygame as pg
from . import tools
from . import constants as c

ORIGINAL_CAPTION = c.ORIGINAL_CAPTION

PKG_DIR = os.path.dirname(os.path.abspath(__file__))          # .../mario_game_files/data
GAME_DIR = os.path.dirname(PKG_DIR)                            # .../mario_game_files
RES_DIR  = os.path.join(GAME_DIR, "resources")

os.environ['SDL_VIDEO_CENTERED'] = '1'
pg.init()
pg.event.set_allowed([pg.KEYDOWN, pg.KEYUP, pg.QUIT])
pg.display.set_caption(c.ORIGINAL_CAPTION)
SCREEN = pg.display.set_mode(c.SCREEN_SIZE)
SCREEN_RECT = SCREEN.get_rect()

FONTS = tools.load_all_fonts(os.path.join(RES_DIR, "fonts"))
MUSIC = tools.load_all_music(os.path.join(RES_DIR, "music"))
GFX   = tools.load_all_gfx(os.path.join(RES_DIR, "graphics"))
SFX   = tools.load_all_sfx(os.path.join(RES_DIR, "sound"))
