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
    monkeypatch.setattr(gamepad, "pad", None)
    monkeypatch.setattr(gamepad, "_nes", _report(b3=0xff, b5=0x2f, b6=0x20))   # right, A, Start
    assert gamepad.pad_state() == (1, 0, True, False, False)
    assert gamepad.button(pygame.CONTROLLER_BUTTON_START)
