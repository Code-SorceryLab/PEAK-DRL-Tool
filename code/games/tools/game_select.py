# code/games/tools/game_select.py — an NES-style screen to pick a game and level for manual play.
# Each game runs in its own manual_play window; quitting it comes back to SELECT GAME. ESC here exits.
# --study: the whole User Study runs in this window (menu.study_session), so the participant never needs the terminal.
import argparse
import os
import subprocess
import sys

os.environ["SDL_VIDEO_CENTERED"] = "1"

import pygame

import menu
from code.games.modules.System import music
from code.games.modules.System.nes_font import nes_text
from code.games.tools import gamepad

parser = argparse.ArgumentParser()
parser.add_argument("--record", action="store_true", help="record every attempt like an agent run")
parser.add_argument("--player", default=None, help="player label for --record")
parser.add_argument("--study", action="store_true", help="run the User Study in this window")
args = parser.parse_args()

NAMES = {"mario": "MARIO", "sonic": "SONIC", "megaman": "MEGA MAN", "meatboy": "MEAT BOY", "bomberman": "BOMBERMAN"}
SIZE = (800, 600)
ROWS = 8                          # list rows shown at once; longer lists scroll
GOLD, GREY = (252, 188, 60), (188, 188, 188)


def run_game(cmd: list, env: dict) -> None:
    """Run one game in its own window until it is quit, then come back to this one."""
    pygame.display.quit()          # one window at a time
    music.stop()
    subprocess.run(cmd, env=env)
    pygame.display.init()
    music.play("menu")


def _play(game: str, level: str | None) -> None:
    cmd = [sys.executable, "-m", "code.games.tools.manual_play", "--game", menu.CONFIG_KEY.get(game, game), "--fps", "60"]
    if level:
        cmd += ["--level", level]
    if args.record:
        cmd += ["--record"] + (["--player", args.player] if args.player else [])
    run_game(cmd, {k: v for k, v in os.environ.items() if k not in ("SDL_VIDEODRIVER", "MARIO_MANUAL_PLAY")})


def _draw(title: str, items: list, choice: int | None, hint: str, size: int = 4) -> None:
    """The framed screen: a title, a list (with a blinking cursor unless choice is None) and a hint."""
    screen = pygame.display.get_surface()
    if screen is None:                 # the first screen, or the first after a game closed the window
        screen = pygame.display.set_mode(SIZE)
        pygame.display.set_caption("PEAK  User Study" if args.study else "PEAK  Manual Play")
    screen.fill((0, 0, 0))
    w, h = SIZE
    pygame.draw.rect(screen, (252, 252, 252), (16, 16, w - 32, h - 32), 4)
    centre = lambda img, y: screen.blit(img, ((w - img.get_width()) // 2, y))
    centre(nes_text("PEAK", 3, GREY), 48)
    centre(nes_text(title, 5), 84)
    at = choice or 0
    top = min(max(0, at - ROWS // 2), max(0, len(items) - ROWS))
    shown = items[top:top + ROWS]
    x = (w - max(nes_text(i, size).get_width() for i in shown)) // 2
    blink = (pygame.time.get_ticks() // 400) % 2 == 0
    for row, item in enumerate(shown):
        y = 170 + row * 42
        screen.blit(nes_text(item, size), (x, y))
        if choice is not None and top + row == choice and blink:
            screen.blit(nes_text(">", size, GOLD), (x - 40, y))
    if top > 0:
        centre(nes_text("-", 3, GREY), 150)
    if top + ROWS < len(items):
        centre(nes_text("-", 3, GREY), 170 + ROWS * 42 - 6)
    centre(nes_text(hint, 2, GREY), h - 64)
    pygame.display.flip()


def ask(title: str, items: list, choice: int | None = 0, hint: str = "A  OK", size: int = 4,
        b_backs: bool = True) -> int | None:
    """Show a list until A picks a row (its index), or ESC (or B / Backspace, if b_backs) backs out (None).
    choice None: a message with no cursor, A goes on (0). Closing the window quits."""
    clock = pygame.time.Clock()
    prev, armed, back_prev = (0, True), False, True   # wait for keys from the last screen to be let go
    while True:
        clock.tick(60)
        back = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return None
            if event.type == pygame.KEYDOWN and event.key == pygame.K_BACKSPACE:
                back = True
            elif event.type in (pygame.JOYDEVICEADDED, pygame.JOYDEVICEREMOVED):
                gamepad.open_pad(quiet=True)
        dy, confirm = gamepad.menu_input()
        b_held = gamepad.held()[3]
        back |= b_held and not back_prev
        back_prev = b_held
        if back and b_backs:
            return None
        if dy and dy != prev[0] and choice is not None:
            choice = (choice + dy) % len(items)
        armed |= confirm and not prev[1]
        prev = (dy, confirm)
        if armed and not confirm:          # act on release, so the press never reaches the game
            return choice or 0
        _draw(title, items, choice, hint, size)


def ask_text(title: str, hint: str = "TYPE IT      ENTER  OK      ESC  BACK") -> str | None:
    """Letters, digits, - and _ typed on the keyboard (16 at most); None on ESC."""
    clock, text = pygame.time.Clock(), ""
    while True:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit
            if event.type != pygame.KEYDOWN:
                continue
            if event.key == pygame.K_ESCAPE:
                return None
            if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER) and text:
                return text
            if event.key == pygame.K_BACKSPACE:
                text = text[:-1]
            elif event.unicode.isascii() and (event.unicode.isalnum() or event.unicode in ("-", "_")) and len(text) < 16:
                text += event.unicode
        _draw(title, [text + ("_" if (pygame.time.get_ticks() // 400) % 2 == 0 else " ")], None, hint)


def main() -> None:
    pygame.init()
    gamepad.open_pad(quiet=True)
    music.play("menu")
    if args.study:
        menu.study_session(sys.modules[__name__])
        return
    games, pick = menu.get_available_games(), 0
    while True:
        pick = ask("SELECT GAME", [NAMES.get(g, g) for g in games], pick, "A  PLAY      ESC  EXIT", b_backs=False)
        if pick is None:
            return
        levels = menu.get_levels_for_game(games[pick])
        row = ask(NAMES.get(games[pick], games[pick]),
                  ["AUTO"] + [f"LEVEL {int(l) + 1}" if l.isdigit() else l for l in levels], 0, "A  PLAY      B  BACK")
        if row is not None:                # B / Backspace / ESC: back to the games
            _play(games[pick], None if row == 0 else levels[row - 1])   # then back to SELECT GAME


if __name__ == "__main__":
    main()
    pygame.quit()
