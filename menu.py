"""Interactive launcher for the neuroevolution trainer.

Run:  python menu.py
"""
from __future__ import annotations

import os
import subprocess
import sys

GAMES = ["mario", "megaman", "sonic", "meatboy"]
LEVEL_HINTS = {
    "mario": "e.g. Mario1-1a, Mario1-1, Mario1-2 (blank = first level)",
    "megaman": "e.g. MM-Train4, MM-Stage1 (blank = default)",
    "sonic": "e.g. Green Hill 1, Green Hill 2 (blank = default)",
    "meatboy": "level index 0-4 (blank = 0)",
}


def ask(prompt: str, default: str = "") -> str:
    raw = input(f"{prompt}{f' [{default}]' if default else ''}: ").strip()
    return raw or default


def pick_game() -> str:
    for i, g in enumerate(GAMES, 1):
        print(f"  {i}. {g}")
    choice = ask("game", "1")
    return GAMES[int(choice) - 1] if choice.isdigit() else choice


def run_trainer(*args: str) -> None:
    cmd = [sys.executable, "-m", "code.neuro.trainer", *args]
    print("+", " ".join(cmd))
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        pass


def main() -> None:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    while True:
        print("\n── PEAK Neuroevolution ─────────────────────")
        print("  1. train (with dashboard)")
        print("  2. resume a run")
        print("  3. replay all-time best")
        print("  4. level editor")
        print("  5. show results table")
        print("  6. run tests")
        print("  7. quit")
        choice = ask("choice", "1")
        if choice == "1":
            game = pick_game()
            level = ask(f"level  ({LEVEL_HINTS[game]})")
            turbo = ask("start in turbo? y/n", "n").lower().startswith("y")
            args = ["--game", game]
            if level:
                args += ["--level", level]
            if turbo:
                args += ["--turbo"]
            print("dashboard: http://127.0.0.1:8000/mario/index.html  (Ctrl+C to stop)")
            run_trainer(*args)
        elif choice == "2":
            run_trainer("--resume", ask("run dir", "runs/mario"))
        elif choice == "3":
            game = pick_game()
            run_trainer("--game", game, "--replay", ask("best.npz path", f"runs/{game}/best.npz"))
        elif choice == "4":
            game = ask("editor game (platformer/megaman/sonic)", "platformer")
            subprocess.run([sys.executable, "code/games/tools/level_editor.py", "--game", game])
        elif choice == "5":
            run_trainer("--results", ask("run dir", "runs/mario"))
        elif choice == "6":
            subprocess.run([sys.executable, "-m", "pytest", "code/tests/", "-q"])
        else:
            return


if __name__ == "__main__":
    main()
