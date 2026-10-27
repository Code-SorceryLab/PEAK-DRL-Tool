"""Every menu choice goes through the checklist screen; pick_one keeps a single tick."""
import menu


def answers(monkeypatch, *typed):
    it = iter(typed)
    monkeypatch.setattr("builtins.input", lambda *a: next(it))


def test_pick_one_starts_on_the_default_and_a_number_moves_the_tick(monkeypatch):
    answers(monkeypatch, "")
    assert menu.pick_one("GAME", ["mario", "meatboy", "bomberman"], default="meatboy") == "meatboy"
    answers(monkeypatch, "3", "1", "")
    assert menu.pick_one("GAME", ["mario", "meatboy", "bomberman"], default="meatboy") == "mario"
    answers(monkeypatch, "0")
    assert menu.pick_one("GAME", ["mario", "meatboy"]) is None


def test_toggle_select_still_toggles_many(monkeypatch):
    answers(monkeypatch, "2,3", "")
    assert menu.toggle_select("GAMES", ["mario", "meatboy", "bomberman"]) == ["mario", "meatboy", "bomberman"]


def test_no_menu_function_uses_the_old_numbered_prompt():
    src = open(menu.__file__, encoding="utf-8").read()
    assert "ask_index" not in src
