"""Checks for the 2026-10-28 audit follow-up: stale mutants, kills counted once, Mario's pits, left
edge and goal, empty levels, and run_all.sh failing when a step fails."""
import os
import shutil
import subprocess
import types

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from code.games import mutants
from code.games.modules.Objects.GameObject import GameObject
from code.tests.test_mario_physics import IDLE, _core

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LEFT = 1


def test_a_mutant_made_from_an_older_level_is_refused(tmp_path):
    base = "meatboy/world1_02_upward.txt"
    record = {"game": "meatboy", "dir": str(tmp_path), "level_file": "level.txt", "base_file": base,
              "base_sha1": mutants.source_hash(base)}
    assert mutants.level_path(dict(record), base) == os.path.join(str(tmp_path), "level.txt")
    with pytest.raises(SystemExit, match="older"):
        mutants.level_path(dict(record, base_sha1="0" * 40), base)


def test_an_enemy_is_one_kill_however_often_it_is_hit():
    core, koopa = types.SimpleNamespace(kills_step=0), GameObject(0, 0, 16, 16)
    for _ in range(3):                      # wings, shell, a revived shell
        koopa.count_kill(core)
    GameObject(0, 0, 16, 16).count_kill(core)
    assert core.kills_step == 2


def test_a_respawned_mega_man_enemy_is_no_new_kill():
    from code.games import megaman_enemies as en
    core = types.SimpleNamespace(score=0, kills_total=0, killed_spawns=set(), items=[],
                                 rng=types.SimpleNamespace(randrange=lambda n: 127))
    for _ in range(2):                      # shot down, then back from its spawn point
        en.kill(en.make("b", 5, 5, spawn=3), core)
    en.kill(en.make("b", 9, 5, spawn=4), core)
    assert core.kills_total == 2


def test_mario_stops_at_the_level_left_edge(tmp_path):
    core = _core(tmp_path, ["." * 40] * 13 + ["P" + "." * 39] + ["#" * 40] * 2)
    for _ in range(120):
        core.step([LEFT, 0, 0])
    assert core.alive and core.player.gObj.x == 0


def test_mario_dies_in_an_o_pit_not_48_px_under_the_level(tmp_path):
    rows = ["." * 40] * 13 + ["...P" + "." * 36] + ["#" * 8 + "OO" + "#" * 30] * 2
    core = _core(tmp_path, rows)
    core.player.gObj.x = 8 * 32 + 5                       # over the gap
    for _ in range(60):
        core.step([IDLE, 0, 0])
        if not core.alive:
            break
    assert not core.alive and core.player.gObj.y < core.level_data.height


def test_an_empty_level_file_says_so(tmp_path):
    from code.games.bomberman_core import BombermanCore
    from code.games.modules.System.LevelLoader import LevelLoader
    path = tmp_path / "level.txt"
    path.write_text("\n\n")
    with pytest.raises(ValueError, match="empty level"):
        LevelLoader(tile_size=32).load_level(str(path))
    core = BombermanCore.__new__(BombermanCore)
    core.level_file = lambda: str(path)
    with pytest.raises(ValueError, match="empty level"):
        core._load_level()


@pytest.mark.skipif(not shutil.which("bash"), reason="needs bash")
def test_run_all_fails_when_a_step_fails():
    env = dict(os.environ, PY="false", GAMES="mario", PERSONAS="novice", SLURM_ARRAY_TASK_ID="0")
    r = subprocess.run(["bash", "run_all.sh", "sweeps"], cwd=ROOT, env=env, capture_output=True, text=True)
    assert r.returncode == 1 and "ALL DONE" in r.stdout


def test_the_grid_sensor_sees_a_koopa_and_no_floor_over_a_pit(tmp_path):
    rows = ["." * 40] * 13 + ["...P...k" + "." * 32] + ["#" * 12 + "OO" + "#" * 26] * 2
    core = _core(tmp_path, rows)
    solid, _, hazard, r0, c0 = core._grid_obs_window()
    k = next(e for e in core.level_data.enemies if type(e).__name__ == "Koopa").gObj
    assert hazard[int(k.y // 32) - r0, int(k.x // 32) - c0] == 1.0
    assert hazard[13 - r0, 12 - c0] == -0.5                # the air over the `O` gap reads as a pit
