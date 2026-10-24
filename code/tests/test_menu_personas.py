import menu


def test_train_all_personas_runs_one_curriculum_per_game_and_persona(monkeypatch):
    picks = iter([["mario", "meatboy"], ["novice", "clumsy"]])
    answers = iter(["800", "7"])                           # generations, then the seed
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
        assert cmd[cmd.index("--game") + 1] == game and cmd[cmd.index("--persona") + 1] == persona
        assert cmd[cmd.index("--seed") + 1] == "7" and cmd[cmd.index("--gens") + 1] == "800"
        assert cmd[cmd.index("--run-dir") + 1].endswith(f"{game}_{persona}_curriculum_seed7")
