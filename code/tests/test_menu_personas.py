import sys

import pytest

import menu


def test_train_all_personas_locks_each_game_to_its_evaluation_level(monkeypatch):
    picks = iter([["mario  ·  Mario1-1", "meatboy  ·  2"], ["novice", "clumsy"]])
    answers = iter(["200", "7", "1"])                      # generations, the seed, one run at a time
    cmds = []
    monkeypatch.setattr(menu, "_refresh_screen", lambda: None)
    monkeypatch.setattr(menu, "toggle_select", lambda *a, **k: next(picks))
    monkeypatch.setattr("builtins.input", lambda *a: next(answers, "y"))
    monkeypatch.setattr(menu, "execute_training_run", lambda cmd: cmds.append(cmd) or True)
    monkeypatch.setattr(menu, "play_chime", lambda: None)
    menu.train_all_personas()
    assert len(cmds) == 4
    for cmd, (game, level, persona) in zip(cmds, [("mario", "Mario1-1", "novice"), ("mario", "Mario1-1", "clumsy"),
                                                  ("meatboy", "2", "novice"), ("meatboy", "2", "clumsy")]):
        assert cmd[cmd.index("--level") + 1] == level      # locked: every generation on the evaluation level
        assert cmd[cmd.index("--game") + 1] == game and cmd[cmd.index("--persona") + 1] == persona
        assert cmd[cmd.index("--seed") + 1] == "7" and cmd[cmd.index("--gens") + 1] == "200"
        assert cmd[cmd.index("--run-dir") + 1].endswith(f"{game}_{level}_{persona}_eval_seed7")


def test_train_all_personas_finishes_each_seed_before_the_next(monkeypatch):
    picks = iter([["mario  ·  Mario1-1"], ["novice", "clumsy"]])
    answers = iter(["200", "7 8", "1"])                    # generations, two seeds, one run at a time
    cmds = []
    monkeypatch.setattr(menu, "_refresh_screen", lambda: None)
    monkeypatch.setattr(menu, "toggle_select", lambda *a, **k: next(picks))
    monkeypatch.setattr("builtins.input", lambda *a: next(answers, "y"))
    monkeypatch.setattr(menu, "execute_training_run", lambda cmd: cmds.append(cmd) or True)
    monkeypatch.setattr(menu, "play_chime", lambda: None)
    menu.train_all_personas()
    order = [(c[c.index("--seed") + 1], c[c.index("--persona") + 1]) for c in cmds]
    assert order == [("7", "novice"), ("7", "clumsy"), ("8", "novice"), ("8", "clumsy")]
    for cmd in cmds:
        seed, persona = cmd[cmd.index("--seed") + 1], cmd[cmd.index("--persona") + 1]
        assert cmd[cmd.index("--run-dir") + 1].endswith(f"mario_Mario1-1_{persona}_eval_seed{seed}")


def test_train_all_personas_parallel_runs_headless_in_seed_order(monkeypatch):
    picks = iter([["mario  ·  Mario1-1", "meatboy  ·  2"], ["novice"]])
    answers = iter(["200", "7 8", "3"])                    # generations, two seeds, three runs at once
    calls = []
    monkeypatch.setattr(menu, "_refresh_screen", lambda: None)
    monkeypatch.setattr(menu, "toggle_select", lambda *a, **k: next(picks))
    monkeypatch.setattr("builtins.input", lambda *a: next(answers, "y"))
    monkeypatch.setattr(menu, "execute_parallel_runs", lambda runs, workers: calls.append((runs, workers)) or len(runs))
    monkeypatch.setattr(menu, "execute_training_run", lambda cmd: pytest.fail("parallel mode must not run one by one"))
    monkeypatch.setattr(menu, "play_chime", lambda: None)
    menu.train_all_personas()
    (runs, workers), = calls
    assert workers == 3
    cmds = [cmd for _, cmd, _ in runs]
    order = [(c[c.index("--seed") + 1], c[c.index("--game") + 1]) for c in cmds]
    assert order == [("7", "mario"), ("7", "meatboy"), ("8", "mario"), ("8", "meatboy")]
    assert all("--no-serve" in c for c in cmds)            # parallel runs can't share the dashboard port


def test_execute_parallel_runs_caps_concurrency_and_logs_each_run(tmp_path):
    marker = tmp_path / "live"
    marker.mkdir()
    # each job records itself as live, notes the peak number of live jobs, then finishes; job 2 fails
    script = ("import os, sys, time, pathlib; d = pathlib.Path(sys.argv[1]); me = d / sys.argv[2]; me.touch(); "
              "n = len(list(d.iterdir())); (d.parent / ('peak' + sys.argv[2])).write_text(str(n)); "
              "print('hello from', sys.argv[2]); time.sleep(1.5); me.unlink(); sys.exit(1 if sys.argv[2] == '2' else 0)")
    runs = [(f"job {i}", [sys.executable, "-c", script, str(marker), str(i)], tmp_path / f"run{i}") for i in range(5)]
    assert menu.execute_parallel_runs(runs, workers=2) == 4
    assert max(int((tmp_path / f"peak{i}").read_text()) for i in range(5)) <= 2
    for i in range(5):
        assert f"hello from {i}" in (tmp_path / f"run{i}" / "train.log").read_text()


def test_evaluation_levels_exist_and_are_enabled():
    for game, level in menu.EVAL_LEVELS.items():
        assert level in menu.get_levels_for_game(game), f"{game}: {level} is not an enabled level"
