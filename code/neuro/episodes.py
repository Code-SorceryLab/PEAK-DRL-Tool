# code/neuro/episodes.py — the per-episode CSV shared by agent runs and recorded manual play.
import csv
import os

# Stats-dashboard schema (Amr's): one row per finished episode.
HEADER = ["persona", "game", "world", "cause_of_death", "jump_count",
          "coins_collected", "avg_vx", "progress_ratio", "route",
          "enemies_killed", "bricks_destroyed", "elapsed_time"]


def append_rows(run_dir: str, rows: list[list]) -> str:
    """Append rows to <run_dir>/episodes.csv, writing the header on first use."""
    path = os.path.join(run_dir, "episodes.csv")
    os.makedirs(run_dir, exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(HEADER)
        w.writerows(rows)
    return path
