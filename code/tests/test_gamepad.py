import pygame

from code.games.tools import gamepad

# Raw reports of the USB NES pad (0810:e501) captured on the Mac with hidapi.
REST = [0x01, 0x80, 0x80, 0x7f, 0x7f, 0x0f, 0x00, 0x00]


def _report(**at):
    r = list(REST)
    for i, v in at.items():
        r[int(i[1:])] = v
    return r


def test_the_nes_pad_report_reads_as_sdl_buttons():
    assert gamepad._nes_held(REST) == set()
    assert gamepad._nes_held(_report(b5=0x2f)) == {pygame.CONTROLLER_BUTTON_A}       # HID button 1, as Kevin mapped it
    assert gamepad._nes_held(_report(b5=0x1f)) == {pygame.CONTROLLER_BUTTON_B}       # HID button 0
    assert gamepad._nes_held(_report(b6=0x10)) == {pygame.CONTROLLER_BUTTON_BACK}    # Select, HID button 8
    assert gamepad._nes_held(_report(b6=0x20)) == {pygame.CONTROLLER_BUTTON_START}   # Start, HID button 9
    assert gamepad._nes_held(_report(b3=0x00, b4=0xff)) == {pygame.CONTROLLER_BUTTON_DPAD_LEFT,
                                                          pygame.CONTROLLER_BUTTON_DPAD_DOWN}
    assert gamepad._nes_held(_report(b3=0xff, b4=0x00)) == {pygame.CONTROLLER_BUTTON_DPAD_RIGHT,
                                                          pygame.CONTROLLER_BUTTON_DPAD_UP}


def test_the_game_reads_the_nes_pad_without_sdl(monkeypatch):
    monkeypatch.setattr(gamepad, "pads", [])
    monkeypatch.setattr(gamepad, "_nes", _report(b3=0xff, b5=0x2f, b6=0x20))   # right, A, Start
    assert gamepad.pad_state() == (1, 0, True, False, False)
    assert gamepad.button(pygame.CONTROLLER_BUTTON_START)


class _Pad:
    def __init__(self, buttons=(), axes=None):
        self.buttons, self.axes = buttons, axes or {}

    def get_button(self, btn):
        return btn in self.buttons

    def get_axis(self, axis):
        return self.axes.get(axis, 0)


def test_any_connected_pad_plays(monkeypatch):
    """A PS5 pad works while the NES pad is plugged in too, whichever SDL lists first."""
    ps5 = _Pad({pygame.CONTROLLER_BUTTON_A}, {pygame.CONTROLLER_AXIS_LEFTX: -32767})
    monkeypatch.setattr(gamepad, "_nes", None)
    monkeypatch.setattr(gamepad, "pads", [_Pad(), ps5])
    assert gamepad.pad_state() == (-1, 0, True, False, False)
    monkeypatch.setattr(gamepad, "pads", [ps5, _Pad()])
    assert gamepad.pad_state() == (-1, 0, True, False, False)


def test_released_pads_open_again_after_a_game():
    gamepad.release_pad()
    assert gamepad.pads == [] and not pygame.joystick.get_init()
    gamepad.open_pad(quiet=True)
    assert pygame.joystick.get_init()
