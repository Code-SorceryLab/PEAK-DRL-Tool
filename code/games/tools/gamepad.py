# code/games/tools/gamepad.py — the game controller for manual play and its game selection screen.
import pygame
from pygame._sdl2 import controller as sdl_controller

pad = None                       # the first connected game controller, if any
# SDL's GameController layout gives every common pad (Xbox, PlayStation, Switch, 8BitDo...) Xbox-style
# names: A is Cross on a PlayStation pad, B is Circle, X is Square.
PAD_JUMP = (pygame.CONTROLLER_BUTTON_A,)
PAD_FIRE = (pygame.CONTROLLER_BUTTON_B,)
PAD_RUN = (pygame.CONTROLLER_BUTTON_B, pygame.CONTROLLER_BUTTON_X, pygame.CONTROLLER_BUTTON_RIGHTSHOULDER)  # B runs, as on a NES
PAD_DEADZONE = 0.5


def open_pad(quiet: bool = False) -> None:
    """(Re)open the first controller; called at start and whenever one is plugged or unplugged."""
    global pad
    sdl_controller.init()
    idx = next((i for i in range(sdl_controller.get_count()) if sdl_controller.is_controller(i)), None)
    # shortcut: a pad missing from SDL's mapping database is ignored, add a raw-joystick fallback if one turns up
    pad = sdl_controller.Controller(idx) if idx is not None else None
    if pad and pad.as_joystick().get_guid() == "03007e411008000001e5000000000000":
        # USB NES pad (0810:e501): SDL's SNES-clone mapping wants A on b2, which this pad lacks.
        # Swap a/b here if A and B feel reversed.
        pad.set_mapping({"a": "b1", "b": "b0", "back": "b8", "start": "b9", "leftx": "a0", "lefty": "a1",
                         "dpleft": "-a0", "dpright": "+a0", "dpup": "-a1", "dpdown": "+a1"})
    if pad and not quiet:
        print(f"[Play] Controller: {pad.name} (stick / d-pad move, A/Cross jump, B/Circle fire, "
              f"B, X/Square or RB run; Mega Man fires with X/Square too)")


def button(btn) -> bool:
    return bool(pad and pad.get_button(btn))


def pad_state() -> tuple[int, int, bool, bool, bool]:
    """(dx, dy, jump, fire, run) from the controller: left stick or d-pad, plus the face buttons."""
    if pad is None:
        return 0, 0, False, False, False
    held = lambda ids: any(pad.get_button(b) for b in ids)
    x = pad.get_axis(pygame.CONTROLLER_AXIS_LEFTX) / 32767
    y = pad.get_axis(pygame.CONTROLLER_AXIS_LEFTY) / 32767
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
