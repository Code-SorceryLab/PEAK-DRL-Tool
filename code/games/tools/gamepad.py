# code/games/tools/gamepad.py — the game controller for manual play and its game selection screen.
import sys
import threading

import pygame
from pygame._sdl2 import controller as sdl_controller

pad = None                       # the first connected game controller, if any
# SDL's GameController layout gives every common pad (Xbox, PlayStation, Switch, 8BitDo...) Xbox-style
# names: A is Cross on a PlayStation pad, B is Circle, X is Square.
PAD_JUMP = (pygame.CONTROLLER_BUTTON_A,)
PAD_FIRE = (pygame.CONTROLLER_BUTTON_B,)
PAD_RUN = (pygame.CONTROLLER_BUTTON_B, pygame.CONTROLLER_BUTTON_X, pygame.CONTROLLER_BUTTON_RIGHTSHOULDER)  # B runs, as on a NES
PAD_DEADZONE = 0.5
# The USB NES pad (0810:e501). On macOS SDL reads it through Apple's controller framework, which loses its
# A, B and d-pad, so it is read straight from USB (hidapi): its last raw report, while it is plugged in.
NES_PAD = (0x0810, 0xE501)
_nes = _nes_dev = None
_NES_BITS = {(5, 0x20): pygame.CONTROLLER_BUTTON_A, (5, 0x10): pygame.CONTROLLER_BUTTON_B,   # HID buttons 1, 0
             (6, 0x10): pygame.CONTROLLER_BUTTON_BACK, (6, 0x20): pygame.CONTROLLER_BUTTON_START}   # 8, 9
_NES_DPAD = ((3, pygame.CONTROLLER_BUTTON_DPAD_LEFT, pygame.CONTROLLER_BUTTON_DPAD_RIGHT),
             (4, pygame.CONTROLLER_BUTTON_DPAD_UP, pygame.CONTROLLER_BUTTON_DPAD_DOWN))   # 00, 7f at rest, ff


def _nes_held(report=None) -> set:
    """The SDL controller buttons held in a raw report of the NES pad (the last one read by default)."""
    r = report or _nes
    if not r:
        return set()
    held = {b for (i, bit), b in _NES_BITS.items() if r[i] & bit}
    return held | {low if r[i] < 0x40 else high for i, low, high in _NES_DPAD if not 0x40 <= r[i] <= 0xbf}


def _read_nes(dev) -> None:
    global _nes, _nes_dev
    try:
        while not _release.is_set():
            _nes = dev.read(64, 100) or _nes
        dev.close()              # released: another process may open it now
    except OSError:              # unplugged: the next open_pad looks for it again
        pass
    _nes = _nes_dev = None


_release = threading.Event()


def release_pad() -> None:
    """Let go of the USB NES pad: macOS gives it to one process at a time, so a game launched from here
    could not read it. open_pad() takes it back."""
    reader = next((t for t in threading.enumerate() if t.name == "nes-pad"), None)
    if reader:
        _release.set()
        reader.join(1.0)
        _release.clear()


def _open_nes() -> bool:
    """macOS: start reading the NES pad from USB if it is plugged in and hidapi is installed."""
    global _nes_dev
    if sys.platform == "darwin" and _nes_dev is None:
        try:
            import hid
            dev = hid.device()
            dev.open(*NES_PAD)
        except (ImportError, OSError):
            return False
        _nes_dev = dev
        threading.Thread(target=_read_nes, args=(dev,), daemon=True, name="nes-pad").start()
    return _nes_dev is not None


def open_pad(quiet: bool = False) -> None:
    """(Re)open the first controller; called at start and whenever one is plugged or unplugged."""
    global pad
    sdl_controller.init()
    nes = _open_nes()            # then SDL's copy of it (Apple's framework, name "usb gamepad") is left out
    idx = next((i for i in range(sdl_controller.get_count()) if sdl_controller.is_controller(i)
                and not (nes and sdl_controller.name_forindex(i).strip() == "usb gamepad")), None)
    # shortcut: a pad missing from SDL's mapping database is ignored, add a raw-joystick fallback if one turns up
    pad = sdl_controller.Controller(idx) if idx is not None else None
    if pad and pad.as_joystick().get_guid() == "03007e411008000001e5000000000000":
        # USB NES pad (0810:e501): SDL's SNES-clone mapping wants A on b2, which this pad lacks.
        # Swap a/b here if A and B feel reversed.
        pad.set_mapping({"a": "b1", "b": "b0", "back": "b8", "start": "b9", "leftx": "a0", "lefty": "a1",
                         "dpleft": "-a0", "dpright": "+a0", "dpup": "-a1", "dpdown": "+a1"})
    if (pad or nes) and not quiet:
        print(f"[Play] Controller: {pad.name if pad else 'USB NES pad'} (stick / d-pad move, A/Cross jump, "
              f"B/Circle fire, B, X/Square or RB run; Mega Man fires with X/Square too)")


def button(btn) -> bool:
    return bool(pad and pad.get_button(btn)) or btn in _nes_held()


def pad_state() -> tuple[int, int, bool, bool, bool]:
    """(dx, dy, jump, fire, run) from the controller: left stick or d-pad, plus the face buttons."""
    if pad is None and not _nes:
        return 0, 0, False, False, False
    held = lambda ids: any(button(b) for b in ids)
    x = pad.get_axis(pygame.CONTROLLER_AXIS_LEFTX) / 32767 if pad else 0.0
    y = pad.get_axis(pygame.CONTROLLER_AXIS_LEFTY) / 32767 if pad else 0.0
    dx = (held((pygame.CONTROLLER_BUTTON_DPAD_RIGHT,)) - held((pygame.CONTROLLER_BUTTON_DPAD_LEFT,))
          or (x > PAD_DEADZONE) - (x < -PAD_DEADZONE))
    dy = (held((pygame.CONTROLLER_BUTTON_DPAD_DOWN,)) - held((pygame.CONTROLLER_BUTTON_DPAD_UP,))
          or (y > PAD_DEADZONE) - (y < -PAD_DEADZONE))
    return dx, dy, held(PAD_JUMP), held(PAD_FIRE), held(PAD_RUN)


def held(jump_keys=()) -> tuple:
    """(dx, dy, a, b, select, start) held on the pad or keyboard; `jump_keys` (W in Mario) count as A, not up."""
    k = pygame.key.get_pressed()
    px, py, pjump, pfire, prun = pad_state()
    up = any(k[key] for key in (pygame.K_w, pygame.K_UP) if key not in jump_keys)
    dx = int(k[pygame.K_d] or k[pygame.K_RIGHT]) - int(k[pygame.K_a] or k[pygame.K_LEFT]) or px
    dy = int(k[pygame.K_s] or k[pygame.K_DOWN]) - int(up) or py
    a = bool(k[pygame.K_SPACE] or pjump or any(k[key] for key in jump_keys))
    b = bool(k[pygame.K_LSHIFT] or k[pygame.K_RSHIFT] or k[pygame.K_j] or k[pygame.K_z] or pfire or prun)
    return dx, dy, a, b, button(pygame.CONTROLLER_BUTTON_BACK), button(pygame.CONTROLLER_BUTTON_START)


def menu_input() -> tuple:
    """(dy, confirm) for a menu: d-pad / arrows / W S, and A / Start / Space / Enter."""
    _, dy, a, _, _, start = held()
    return dy, bool(a or start or pygame.key.get_pressed()[pygame.K_RETURN])
