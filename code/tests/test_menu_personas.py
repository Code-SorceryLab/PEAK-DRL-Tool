import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import menu


def test_train_all_personas_runs_one_curriculum_per_game_and_persona(monkeypatch):
    picks = iter([["mario", "meatboy"], ["novice", "clumsy"]])
    answers = iter(["800", "7", "1"])                      # generations, the seed, one run at a time
    cmds = []
    monkeypatch.setattr(menu, "_refresh_screen", lambda: None)
    monkeypatch.setattr(menu, "toggle_select", lambda *a, **k: next(picks))
    monkeypatch.setattr("builtins.input", lambda *a: next(answers, "y"))
    monkeypatch.setattr(menu, "execute_training_run", lambda cmd: cmds.append(cmd) or True)
    monkeypatch.setattr(menu, "play_chime", lambda: None)
    menu.train_all_personas()
    assert len(cmds) == 4
    for cmd, (game, persona) in zip(cmds, [("mario", "novice"), ("mario", "clumsy"),
                                           ("meatboy", "novice"), ("meatboy", "clumsy")]):
        assert "--level" not in cmd                        # no level: the trainer walks them all, nets carry over
        assert "--best" in cmd                             # the GA sweep's recommended config for the game
        assert cmd[cmd.index("--game") + 1] == game and cmd[cmd.index("--persona") + 1] == persona
        assert cmd[cmd.index("--seed") + 1] == "7" and cmd[cmd.index("--gens") + 1] == "800"
        assert cmd[cmd.index("--run-dir") + 1].endswith(f"{game}_{persona}_curriculum_seed7")


def test_train_all_personas_finishes_each_seed_before_the_next(monkeypatch):
    picks = iter([["mario"], ["novice", "clumsy"]])
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
        assert cmd[cmd.index("--run-dir") + 1].endswith(f"mario_{persona}_curriculum_seed{seed}")


def test_train_all_personas_parallel_runs_headless_in_seed_order(monkeypatch):
    picks = iter([["mario", "meatboy"], ["novice"]])
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


def test_manual_play_records_like_an_agent_when_asked(monkeypatch):
    picks = iter(["mario", "Mario1-1"])
    answers = iter(["y", "Kevin C!"])                      # record? then the player name
    cmds = []
    monkeypatch.setattr(menu, "_refresh_screen", lambda: None)
    monkeypatch.setattr(menu, "pick_one", lambda *a, **k: next(picks))
    monkeypatch.setattr("builtins.input", lambda *a: next(answers))
    monkeypatch.setattr(menu.subprocess, "run", lambda cmd, **k: cmds.append(cmd))
    menu.run_manual_play()
    (cmd,) = cmds
    assert cmd[cmd.index("--record") + 1:cmd.index("--record") + 3] == ["--player", "Kevin_C_"]  # safe folder name


def test_manual_play_records_nothing_by_default(monkeypatch):
    picks = iter(["mario", "Mario1-1"])
    cmds = []
    monkeypatch.setattr(menu, "_refresh_screen", lambda: None)
    monkeypatch.setattr(menu, "pick_one", lambda *a, **k: next(picks))
    monkeypatch.setattr("builtins.input", lambda *a: "")
    monkeypatch.setattr(menu.subprocess, "run", lambda cmd, **k: cmds.append(cmd))
    menu.run_manual_play()
    assert "--record" not in cmds[0]


def _study_popup(played: list, ask=lambda *a, **k: 0, runs=()) -> SimpleNamespace:
    """A fake game select pop-up: A on every screen (or ask). Each attempt records one row per life, the causes
    taken from runs in order (("Pit",) = died once, then quit), then a cleared level ("Success",)."""
    import csv
    runs = list(runs)

    def run_game(cmd, env):
        if "--record" not in cmd:                          # the warm-up
            return
        played.append(env.get("PEAK_MUTANT"))
        out = Path(cmd[cmd.index("--record") + 1]) / "mario" / "p1" / "episodes.csv"
        out.parent.mkdir(parents=True)
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, ["persona", "game", "world", "cause_of_death", "jump_count", "coins_collected",
                                   "avg_vx", "progress_ratio", "route", "enemies_killed", "bricks_destroyed",
                                   "elapsed_time"])
            w.writeheader()
            for cause in (runs.pop(0) if runs else ("Success",)):
                w.writerow({"world": "Mario1-1", "cause_of_death": cause, "route": "[(1.0, 2.0)]", "elapsed_time": 3})

    return SimpleNamespace(NAMES={}, ask_text=lambda title: "P1", ask=ask, show_game=lambda game: None,
                           run_game=run_game)


def test_user_study_plays_baseline_then_every_mutant_once_shuffled(monkeypatch, tmp_path):
    import csv
    from code.games.eval_mutants import eval_set
    played = []                                            # PEAK_MUTANT per recorded attempt
    monkeypatch.setattr(menu, "STUDY_DIR", tmp_path)
    menu.study_session(_study_popup(played))
    ids = {m["dir"]: m["id"] for m in eval_set("mario")}
    assert played[:3] == [None] * 3                        # baseline: the unchanged level
    assert sorted(played[3:]) == sorted(ids) and len(played) == 8
    for run, folder in enumerate(played[3:], 1):
        (row,) = csv.DictReader(open(tmp_path / "P1" / "mario" / "mutation" / f"run{run}" / "episodes.csv"))
        assert row["mutant"] == ids[folder] and row["session"] == "P1/mutation"
    assert len(list((tmp_path / "P1" / "mario" / "baseline").glob("run*/episodes.csv"))) == 3


def test_esc_or_b_on_a_study_screen_ends_the_session_only_when_confirmed(monkeypatch, tmp_path):
    monkeypatch.setattr(menu, "STUDY_DIR", tmp_path)
    asked, played = [], []
    answers = {"BASELINE": [None], "END SESSION?": [0]}    # ESC (the games' quit key) on attempt 1, then NO

    def ask(title, items, choice=0, hint="", size=4, b_backs=True, games=None):
        asked.append((title, b_backs))
        return answers[title].pop(0) if answers.get(title) else 0

    menu.study_session(_study_popup(played, ask))
    assert len(played) == 8                                # the session went on: 3 baseline + 5 mutants
    assert not any(b for title, b in asked)                # B runs and fires in the games: never "back" here
    played.clear()
    answers.update({"BASELINE": [None], "END SESSION?": [1]})   # ESC, then YES: nothing more is played
    menu.study_session(_study_popup(played, ask))
    assert played == []
    assert menu.study_session(SimpleNamespace(ask_text=lambda title: None)) is False   # ESC on the ID: close the pop-up


def _asker(asked: list, answers: dict):
    def ask(title, items, choice=0, hint="", size=4, b_backs=True, games=None):
        asked.append(title)
        return answers[title].pop(0) if answers.get(title) else 0
    return ask


def test_a_run_quit_before_its_last_life_is_kept_as_quit_and_offered_again(monkeypatch, tmp_path):
    import csv
    monkeypatch.setattr(menu, "STUDY_DIR", tmp_path)
    asked, played = [], []
    menu.study_session(_study_popup(played, _asker(asked, {}), runs=[("Pit",), ("Pit", "Pit", "Pit")]))
    assert asked.count("NOT FINISHED") == 1                # only the quit run is offered again (TRY AGAIN)
    run1 = list(csv.DictReader(open(tmp_path / "P1" / "mario" / "baseline" / "run1" / "episodes.csv")))
    assert [(r["run_result"], r["lives_used"]) for r in run1] == [("quit", "1")] + [("game_over", "3")] * 3
    assert len(played) == 9                                # 8 attempts and the retry


def test_end_session_on_the_retry_screen_ends_the_session(monkeypatch, tmp_path):
    monkeypatch.setattr(menu, "STUDY_DIR", tmp_path)
    played = []
    ask = _asker([], {"NOT FINISHED": [None], "END SESSION?": [1]})   # ESC there, then YES, END IT
    assert menu.study_session(_study_popup(played, ask, runs=[("Pit",)])) is True
    assert len(played) == 1


def test_ga_sweep_passes_the_chosen_worker_count(monkeypatch):
    picks = iter(["rays", "stop"])                  # sensors, no --confirm
    toggles = iter([["gens_dummy"], ["meatboy"], ["experienced"]])   # GA axes, games, personas
    answers = iter(["", "", "10", "y"])             # default gens, default seeds, 10 at once, proceed
    cmds = []
    monkeypatch.setattr(menu, "_refresh_screen", lambda: None)
    monkeypatch.setattr(menu, "pick_one", lambda *a, **k: next(picks))
    monkeypatch.setattr(menu, "toggle_select", lambda *a, **k: next(toggles))
    monkeypatch.setattr("builtins.input", lambda *a: next(answers))
    monkeypatch.setattr(menu.subprocess, "run", lambda cmd, **k: cmds.append(cmd))
    monkeypatch.setattr(menu, "play_chime", lambda: None)
    monkeypatch.setattr(menu, "_open_command_center", lambda *a: None)
    import code.neuro.gasweep as gasweep
    monkeypatch.setattr(gasweep, "sweep_configs", lambda axes: [("base", {})])
    menu.run_ga_sweep()
    (cmd,) = cmds
    assert cmd[cmd.index("--workers") + 1] == "10"


def test_deleting_everything_under_runs_keeps_the_user_study(monkeypatch, tmp_path):
    (tmp_path / "user_study" / "P1").mkdir(parents=True)
    (tmp_path / "probes" / "mario").mkdir(parents=True)
    (tmp_path / "notes.txt").write_text("x")
    monkeypatch.setattr(menu, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(menu, "STUDY_DIR", tmp_path / "user_study")
    monkeypatch.setattr(menu, "_RUN_GROUPS", [])
    monkeypatch.setattr(menu, "get_run_dirs", lambda: [])
    monkeypatch.setattr(menu, "_refresh_screen", lambda: None)
    monkeypatch.setattr(menu, "toggle_select", lambda *a, **k: ["EVERYTHING under runs/"])
    monkeypatch.setattr("builtins.input", lambda *a: "y")
    menu.delete_logs_and_models()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["user_study"]   # people's play cannot be run again
