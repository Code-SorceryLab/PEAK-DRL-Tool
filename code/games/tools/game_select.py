# code/games/tools/game_select.py — an NES-style screen to pick a game and level for manual play.
# Each game runs in its own manual_play window; quitting it comes back here. ESC here exits.
import argparse
import os
import subprocess
import sys

os.environ["SDL_VIDEO_CENTERED"] = "1"

import pygame

import menu
from code.games.modules.System.nes_font import nes_text
from code.games.tools import gamepad

parser = argparse.ArgumentParser()
parser.add_argument("--record", action="store_true", help="record every attempt like an agent run")
parser.add_argument("--player", default=None, help="player label for --record")
args = parser.parse_args()

NAMES = {"mario": "MARIO", "sonic": "SONIC", "megaman": "MEGA MAN", "meatboy": "MEAT BOY", "bomberman": "BOMBERMAN"}
SIZE = (800, 600)
ROWS = 8                          # list rows shown at once; longer lists scroll
GOLD, GREY = (252, 188, 60), (188, 188, 188)


def _play(game: str, level: str | None) -> None:
    """Run one game in its own window until it is quit."""
    cmd = [sys.executable, "-m", "code.games.tools.manual_play", "--game", menu.CONFIG_KEY.get(game, game), "--fps", "60"]
    if level:
        cmd += ["--level", level]
    if args.record:
        cmd += ["--record"] + (["--player", args.player] if args.player else [])
    env = {k: v for k, v in os.environ.items() if k not in ("SDL_VIDEODRIVER", "MARIO_MANUAL_PLAY")}
    pygame.display.quit()          # one window at a time
    subprocess.run(cmd, env=env)
    pygame.display.init()


def _draw(screen, title: str, items: list, choice: int, hint: str) -> None:
    screen.fill((0, 0, 0))
    w, h = SIZE
    pygame.draw.rect(screen, (252, 252, 252), (16, 16, w - 32, h - 32), 4)
    centre = lambda img, y: screen.blit(img, ((w - img.get_width()) // 2, y))
    centre(nes_text("PEAK", 3, GREY), 48)
    centre(nes_text(title, 5), 84)
    top = min(max(0, choice - ROWS // 2), max(0, len(items) - ROWS))
    shown = items[top:top + ROWS]
    x = (w - max(nes_text(i, 4).get_width() for i in shown)) // 2
    blink = (pygame.time.get_ticks() // 400) % 2 == 0
    for row, item in enumerate(shown):
        y = 170 + row * 42
        screen.blit(nes_text(item, 4), (x, y))
        if top + row == choice and blink:
            screen.blit(nes_text(">", 4, GOLD), (x - 40, y))
    if top > 0:
        centre(nes_text("-", 3, GREY), 150)
    if top + ROWS < len(items):
        centre(nes_text("-", 3, GREY), 170 + ROWS * 42 - 6)
    centre(nes_text(hint, 2, GREY), h - 64)


def main() -> None:
    pygame.init()
    gamepad.open_pad(quiet=True)
    clock = pygame.time.Clock()
    games = menu.get_available_games()
    game, choice = None, 0                 # game None: picking a game; else picking its level
    prev, armed, back_prev = (0, True), False, True
    screen = None
    while True:
        if screen is None:
            screen = pygame.display.set_mode(SIZE)
            pygame.display.set_caption("PEAK  Manual Play")
            prev, armed, back_prev = (0, True), False, True   # wait for keys from the last game to be let go
        clock.tick(60)
        back = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_BACKSPACE):
                if game is None and event.key == pygame.K_ESCAPE:
                    return
                back = True
            elif event.type in (pygame.JOYDEVICEADDED, pygame.JOYDEVICEREMOVED):
                gamepad.open_pad(quiet=True)

        if game is None:
            items = [NAMES.get(g, g) for g in games]
        else:
            items = ["AUTO"] + [f"LEVEL {int(l) + 1}" if l.isdigit() else l for l in levels]

        dy, confirm = gamepad.menu_input()
        b_held = gamepad.held()[3]
        back |= b_held and not back_prev
        back_prev = b_held
        if dy and dy != prev[0]:
            choice = (choice + dy) % len(items)
        armed |= confirm and not prev[1]
        prev = (dy, confirm)

        if back and game is not None:      # B / Backspace / ESC: back to the games
            choice, game = games.index(game), None
        elif armed and not confirm:        # act on release, so the press never reaches the game
            armed = False
            if game is None:
                game, levels, choice = games[choice], menu.get_levels_for_game(games[choice]), 0
            else:
                _play(game, None if choice == 0 else levels[choice - 1])
                screen = None
                continue

        if game is None:
            _draw(screen, "SELECT GAME", items, choice, "A  PLAY      ESC  EXIT")
        else:
            _draw(screen, NAMES.get(game, game), items, choice, "A  PLAY      B  BACK")
        pygame.display.flip()


if __name__ == "__main__":
    main()
    pygame.quit()
