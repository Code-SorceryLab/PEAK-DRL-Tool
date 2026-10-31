# code/games/tools/game_select.py — an NES-style screen to pick a game and level for manual play.
# Each game runs in its own manual_play window; quitting it comes back to SELECT GAME. ESC here exits.
# --study: the whole User Study runs in this window (menu.study_session), so the participant never needs the terminal.
import argparse
import faulthandler
import json
import os
import subprocess
import sys
import tempfile
import threading

os.environ["SDL_VIDEO_CENTERED"] = "1"
# macOS may not focus this window when a game closes; the pad must still work (SDL drops unfocused pad input).
os.environ["SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS"] = "1"

import pygame
from pygame._sdl2.video import Window

import menu
from code.games.modules.System import music
from code.games.modules.System.nes_font import nes_text
from code.games.tools import gamepad

parser = argparse.ArgumentParser()
parser.add_argument("--record", action="store_true", help="record every attempt like an agent run")
parser.add_argument("--player", default=None, help="player label for --record")
parser.add_argument("--study", action="store_true", help="run the User Study in this window")
args = parser.parse_known_args()[0]   # menu.NO_RESTORE rides along on macOS

NAMES = {"mario": "MARIO", "megaman": "MEGA MAN", "meatboy": "MEAT BOY", "bomberman": "BOMBERMAN", "sonic": "SONIC"}
SIZE = (800, 600)
ROWS = 8                          # list rows shown at once; longer lists scroll
GOLD, GREY = (252, 188, 60), (188, 188, 188)
CLIP_BOX, CLIP_FPS, CLIP_S = (240, 200), 12, 6   # the gameplay clip: its decoded size, rate and length
game_shown = None                # once the study has a game, its name heads every screen
_window = None                   # see _draw
_clips, _frames = {}, {}         # game -> (size, raw RGB frames) from the decoder thread; -> Surfaces


def run_game(cmd: list, env: dict) -> None:
    """Run one game in its own window until it is quit, then come back to this one. A loading bar shows
    until the game's window is up (manual_play touches PEAK_READY_FILE then)."""
    music.stop()
    gamepad.release_pad()          # macOS gives the USB NES pad to one process at a time: the game's turn
    ready = os.path.join(tempfile.gettempdir(), f"peak_ready_{os.getpid()}")
    if os.path.exists(ready):
        os.remove(ready)
    proc = subprocess.Popen(cmd + menu.NO_RESTORE, env={**env, "PEAK_READY_FILE": ready})
    clock, start = pygame.time.Clock(), pygame.time.get_ticks()
    while proc.poll() is None and not os.path.exists(ready):
        pygame.event.pump()
        _loading(min(0.95, (pygame.time.get_ticks() - start) / 3000))   # about 3 s to the first frame
        clock.tick(30)
    pygame.display.quit()          # one window at a time
    proc.wait()
    if os.path.exists(ready):
        os.remove(ready)
    pygame.display.init()
    gamepad.open_pad(quiet=True)
    music.play("menu")


def _loading(done: float) -> None:
    """LOADING and a bar while a game starts."""
    _draw("LOADING", [" "], None, "THE GAME OPENS IN ITS OWN WINDOW")
    screen = pygame.display.get_surface()
    w, h = SIZE
    pygame.draw.rect(screen, GREY, (w // 2 - 200, h // 2, 400, 28), 3)
    pygame.draw.rect(screen, GOLD, (w // 2 - 194, h // 2 + 6, int(388 * done), 16))
    pygame.display.flip()


def _play(game: str, level: str | None) -> None:
    cmd = [sys.executable, "-m", "code.games.tools.manual_play", "--game", menu.CONFIG_KEY.get(game, game), "--fps", "60"]
    if level:
        cmd += ["--level", level]
    if args.record:
        cmd += ["--record"] + (["--player", args.player] if args.player else [])
    run_game(cmd, {k: v for k, v in os.environ.items() if k not in ("SDL_VIDEODRIVER", "MARIO_MANUAL_PLAY")})


def show_game(game: str) -> None:
    """Head every later screen with this game's name and play its clip there."""
    global game_shown
    game_shown = game


def _decode_clip(game: str) -> None:
    """CLIP_S seconds of a trained agent's replay (runs/videos/<game>, menu 22), from a level other than the
    study's own where there is one, so the clip shows the game without the route. None without the video or PyAV.
    PyAV, not the ffmpeg command: Windows Smart App Control blocks a downloaded ffmpeg.exe."""
    from code.games.eval_mutants import EVAL_LEVELS
    video = menu.RUNS_DIR / "videos" / game / "experienced.webm"
    try:
        import av
        meta = json.loads(video.with_suffix(".json").read_text())
        tries = meta["attempts"]          # [start frame, level, outcome] per attempt
        i = next((k for k, a in enumerate(tries) if a[1] != EVAL_LEVELS.get(game, ("",))[0]), 0)
        start = tries[i][0] / meta["fps"]
        length = min(CLIP_S, (tries[i + 1][0] / meta["fps"] - start) if i + 1 < len(tries) else CLIP_S)
        frames, due = [], start
        with av.open(str(video)) as box:
            s = box.streams.video[0]
            scale = min(CLIP_BOX[0] / s.width, CLIP_BOX[1] / s.height)
            size = (int(s.width * scale) // 2 * 2, int(s.height * scale) // 2 * 2)
            box.seek(int(start / s.time_base), stream=s)           # the keyframe before start
            for f in box.decode(s):
                if f.time is None or f.time < due:
                    continue
                if f.time >= start + length:
                    break
                frames.append(f.to_ndarray(width=size[0], height=size[1], format="rgb24",
                                           interpolation="POINT").tobytes())
                due += 1 / CLIP_FPS
    except Exception:                     # no PyAV, no video or a bad one: the screen just has no clip
        return
    _clips[game] = (size, frames)


def _draw(title: str, items: list, choice: int | None, hint: str, size: int = 4, game: str | None = None) -> None:
    """The framed screen: the game (or PEAK), a title, a list (with a blinking cursor unless choice is None),
    the game's clip (beside a list with a cursor, under a message) and a hint."""
    screen = pygame.display.get_surface()
    if screen is None:                 # the first screen, or the first after a game closed the window
        screen = pygame.display.set_mode(SIZE)
        pygame.display.set_caption("PEAK  User Study" if args.study else "PEAK  Manual Play")
        if sys.platform == "darwin":       # macOS leaves a reopened window without the keyboard
            # held in a global: SDL keeps a raw pointer to this wrapper and pygame touches it on every window
            # event, so a wrapper dropped here corrupts memory (on Windows it frees the window: a crash)
            global _window
            _window = Window.from_display_module()
            _window.focus()
        if os.environ.get("PEAK_READY_FILE"):  # up: the menu's loading bar stops
            open(os.environ["PEAK_READY_FILE"], "a").close()
    screen.fill((0, 0, 0))
    w, h = SIZE
    pygame.draw.rect(screen, (252, 252, 252), (16, 16, w - 32, h - 32), 4)
    centre = lambda img, y: screen.blit(img, ((w - img.get_width()) // 2, y))
    centre(nes_text(NAMES.get(game_shown, game_shown), 3, GOLD) if game_shown else nes_text("PEAK", 3, GREY), 48)
    centre(nes_text(title, 5), 84)
    at = choice or 0
    top = min(max(0, at - ROWS // 2), max(0, len(items) - ROWS))
    shown = items[top:top + ROWS]
    if game in _clips:                  # decoded in the background: turn it into Surfaces once
        dims, raw = _clips.pop(game)
        _frames[game] = [pygame.image.frombytes(f, dims, "RGB") for f in raw]
    clip = _frames.get(game)
    widest = max(nes_text(i, size).get_width() for i in shown)
    side = bool(clip) and choice is not None and widest <= w // 2 - 80
    x = (w // 2 - widest) // 2 + 30 if side else (w - widest) // 2
    blink = (pygame.time.get_ticks() // 400) % 2 == 0
    step = 10 * size + 2
    for row, item in enumerate(shown):
        y = 170 + row * step
        screen.blit(nes_text(item, size), (x, y))
        if choice is not None and top + row == choice and blink:
            screen.blit(nes_text(">", size, GOLD), (x - 40, y))
    if top > 0:
        centre(nes_text("-", 3, GREY), 150)
    if top + ROWS < len(items):
        centre(nes_text("-", 3, GREY), 170 + ROWS * step - 6)
    if clip:
        img = clip[pygame.time.get_ticks() * CLIP_FPS // 1000 % len(clip)]
        cx, y0 = (w * 3 // 4 - 20, 170) if side else (w // 2, 170 + len(shown) * step + 20)
        fit = min((w // 2 - 60 if side else w - 120) / img.get_width(), (h - 84 - y0) / img.get_height())
        img = pygame.transform.scale(img, (int(img.get_width() * fit), int(img.get_height() * fit)))
        box = img.get_rect(midtop=(cx, y0))
        screen.blit(img, box)
        pygame.draw.rect(screen, GREY, box.inflate(8, 8), 2)
    centre(nes_text(hint, 2, GREY), h - 64)
    pygame.display.flip()


RED, SHELL, FACE = (206, 32, 40), (196, 196, 200), (36, 36, 40)   # the controller's colours


def _dpad(screen, c, arm: int, colour=(12, 12, 14)) -> None:
    for r in (pygame.Rect(0, 0, arm * 3, arm), pygame.Rect(0, 0, arm, arm * 3)):
        r.center = c
        pygame.draw.rect(screen, colour, r, border_radius=max(1, arm // 6))


def _pill(screen, c, w: int, colour=(12, 12, 14)) -> None:
    pygame.draw.rect(screen, colour, (c[0] - w // 2, c[1] - w // 6, w, w // 3), border_radius=w // 6)


def _button(screen, c, r: int, letter: str | None = None) -> None:
    pygame.draw.circle(screen, RED, c, r)
    pygame.draw.circle(screen, (120, 14, 20), c, r, 2)
    if letter:
        img = nes_text(letter, 2, (252, 252, 252))
        screen.blit(img, (c[0] - img.get_width() // 2 + 1, c[1] - img.get_height() // 2 + 1))


def _pad_screen(title: str, controls: dict, hint: str) -> None:
    """An NES controller, and under it each button's job in this game (controls: dpad, a, b, select,
    start -> what it does). Drawn, not a picture: grey shell, black face, the red A and B."""
    screen = pygame.display.get_surface()
    if screen is None:
        screen = pygame.display.set_mode(SIZE)
    screen.fill((0, 0, 0))
    w, h = SIZE
    pygame.draw.rect(screen, (252, 252, 252), (16, 16, w - 32, h - 32), 4)
    centre = lambda img, y: screen.blit(img, ((w - img.get_width()) // 2, y))
    centre(nes_text(NAMES.get(game_shown, game_shown), 3, GOLD) if game_shown else nes_text("PEAK", 3, GREY), 48)
    centre(nes_text(title, 5), 84)

    body = pygame.Rect(0, 0, 380, 150)
    body.center = (w // 2, 236)
    pygame.draw.rect(screen, SHELL, body, border_radius=10)
    pygame.draw.rect(screen, (120, 120, 128), body, 3, border_radius=10)
    face = body.inflate(-28, -34)
    pygame.draw.rect(screen, FACE, face, border_radius=6)
    band = pygame.Rect(face.x + 118, face.y + 16, 100, face.h - 32)
    for k in range(3):                                         # the grey bands in the middle
        pygame.draw.rect(screen, (150, 150, 156), (band.x, band.y + k * 18, band.w, 12), border_radius=3)
    _dpad(screen, (face.x + 58, face.centery), 20)
    sel, start = (band.x + 26, band.bottom - 12), (band.x + 74, band.bottom - 12)
    for c, name in ((sel, "SELECT"), (start, "START")):
        _pill(screen, c, 30)
        img = nes_text(name, 1, RED)
        screen.blit(img, (c[0] - img.get_width() // 2, c[1] - 22))
    for x, letter in ((face.right - 88, "B"), (face.right - 38, "A")):
        c = (x, face.centery + 6)
        pygame.draw.rect(screen, (150, 150, 156), (c[0] - 22, c[1] - 22, 44, 44), border_radius=6)
        _button(screen, c, 17)
        img = nes_text(letter, 2, RED)
        screen.blit(img, (c[0] - img.get_width() // 2, c[1] + 26))

    # the legend: an icon of the button, its name, what it does here
    rows = [("dpad", "D-PAD"), ("a", "A"), ("b", "B"), ("select", "SELECT"), ("start", "START")]
    x0, y = w // 2 - 200, 336
    for key, name in rows:
        if not controls.get(key):
            continue
        icon = (x0 + 18, y + 9)
        if key == "dpad":                                     # grey: black would vanish on the black screen
            _dpad(screen, icon, 8, (170, 170, 176))
        elif key in ("a", "b"):
            _button(screen, icon, 12, name)
        else:
            _pill(screen, icon, 26, (170, 170, 176))
        screen.blit(nes_text(name, 2, GOLD), (x0 + 48, y + 2))
        screen.blit(nes_text(controls[key], 2), (x0 + 170, y + 2))
        y += 32
    centre(nes_text(hint, 2, GREY), h - 64)
    pygame.display.flip()


def show_controls(title: str, controls: dict, hint: str = "A  START") -> int | None:
    """The controller screen until A goes on (0) or ESC (None); the same keys as ask()."""
    clock = pygame.time.Clock()
    prev, armed = (0, True), False                    # wait for keys from the last screen to be let go
    while True:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return None
            if event.type in (pygame.JOYDEVICEADDED, pygame.JOYDEVICEREMOVED):
                gamepad.open_pad(quiet=True)
        dy, confirm = gamepad.menu_input()
        armed |= confirm and not prev[1]
        prev = (dy, confirm)
        if armed and not confirm:
            return 0
        _pad_screen(title, controls, hint)


def ask(title: str, items: list, choice: int | None = 0, hint: str = "A  OK", size: int = 4,
        b_backs: bool = True, games: list | None = None) -> int | None:
    """Show a list until A picks a row (its index), or ESC (or B / Backspace, if b_backs) backs out (None).
    choice None: a message with no cursor, A goes on (0). games: each row's game, whose clip plays while
    it is highlighted (else the shown game's). Closing the window quits."""
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
        _draw(title, items, choice, hint, size, games[choice] if games else game_shown)


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
    for g in menu.get_available_games():   # the clips decode in the background; a screen shows one when ready
        threading.Thread(target=_decode_clip, args=(g,), daemon=True).start()
    if args.study:
        while menu.study_session(sys.modules[__name__]):
            show_game(None)                # the next participant starts at the ID screen
        return
    games, pick = menu.get_available_games(), 0
    while True:
        pick = ask("SELECT GAME", [NAMES.get(g, g) for g in games], pick, "A  PLAY      ESC  EXIT", b_backs=False,
                   games=games)
        if pick is None:
            return
        levels = menu.get_levels_for_game(games[pick])
        row = ask(NAMES.get(games[pick], games[pick]),
                  ["AUTO"] + [f"LEVEL {int(l) + 1}" if l.isdigit() else l for l in levels], 0, "A  PLAY      B  BACK")
        if row is not None:                # B / Backspace / ESC: back to the games
            _play(games[pick], None if row == 0 else levels[row - 1])   # then back to SELECT GAME


if __name__ == "__main__":
    main()
    # shortcut: a pygame.quit() that hangs (seen once with a USB pad, not reproduced) is cut off after 5 s with
    # a stack dump, so the menu comes back; drop this once the dump shows the cause.
    faulthandler.dump_traceback_later(5, exit=True)
    pygame.quit()
