"""manual_play.py is a script: importing it opens the game. These tests lift one top-level statement
at a time out of its source and run it against stand-ins."""
import ast
import os
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

from code.games.tools import gamepad

SRC = Path(__file__).resolve().parents[1] / "games" / "tools" / "manual_play.py"


def lift(marker: str, **env) -> dict:
    """Run the top-level statement of manual_play.py whose source holds `marker`, in `env`."""
    text = SRC.read_text(encoding="utf-8")
    node = next(n for n in ast.parse(text).body if marker in ast.get_source_segment(text, n))
    exec(compile(ast.Module([node], []), str(SRC), "exec"), env)
    return env


def press(monkeypatch, *keys) -> None:
    monkeypatch.setattr(pygame.key, "get_pressed", lambda: defaultdict(bool, dict.fromkeys(keys, True)))


def test_a_meatboy_time_out_is_recorded_as_a_timeout():
    """Meat Boy stops a run at max_steps without a cause: a player still alive then timed out."""
    from code.games.meatboy_core import MeatboyCore
    core = MeatboyCore(max_steps=5)
    rows = []
    env = lift("class _Recorder:", args=SimpleNamespace(player="p", random=False, game="meatboy", record="unused"),
               core_game=core, level_id=None, level_file=None, os=os, Path=Path,
               append_rows=lambda folder, new: rows.extend(new) or folder)
    recorder = env["_Recorder"]()
    while not core.step([0, 0, 0])[3]:
        pass
    assert core.alive and not core.death_cause
    recorder.end()
    core.alive = False
    recorder.end()
    assert [row[3] for row in rows] == ["Timeout", "Dead"]


def test_inputs_log_a_jump_key_as_a_not_up(monkeypatch):
    """W jumps in Mario and Meat Boy (the up arrow too in Meat Boy): it is logged as A."""
    keys = lift("JUMP_KEYS =", pygame=pygame)["JUMP_KEYS"]
    monkeypatch.setattr(gamepad, "pad", None)

    def held(game, key):
        press(monkeypatch, key)
        _, dy, a, *_ = gamepad.held(keys.get(game, ()))
        return dy, a

    assert held("megaman", pygame.K_w) == (-1, False)          # Mega Man climbs on W
    assert held("platformer", pygame.K_w) == (0, True)
    assert held("platformer", pygame.K_UP) == (-1, False)      # the up arrow does nothing in Mario: up, as held
    assert held("meatboy", pygame.K_w) == held("meatboy", pygame.K_UP) == (0, True)


def test_a_recorded_run_ignores_the_debug_keys(monkeypatch):
    """F3 halves Mario's dt and F2 freezes him on the free camera: in a recorded run they do nothing."""
    from code.games.modules.System.debugging_mods.manager import DebugManager
    pygame.init()

    def after_f2_f3(record):
        dm = DebugManager(print_help=False)
        dm._prev_keys = defaultdict(bool)
        lift("update_input", args=SimpleNamespace(record=record), core_game=SimpleNamespace(debug_manager=dm))
        press(monkeypatch, pygame.K_F2, pygame.K_F3)
        dm.update_input()
        return dm.free_cam_active, dm.slow_motion

    assert after_f2_f3(None) == (True, True)                   # practice: the debug keys work
    assert after_f2_f3("runs/manual") == (False, False)
