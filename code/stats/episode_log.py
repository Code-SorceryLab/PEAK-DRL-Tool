"""episodes.csv: one row per finished episode, written from EpisodeStats.to_dict().

Nested values (routes, bomb drops) are stored as JSON in their cell, None as an empty cell.
Every game has its own columns; a run directory only ever holds one game, and a file that
gains a column (a new metric) is rewritten with the wider header instead of dropping it.
"""
from __future__ import annotations

import csv
import json
import os

LEAD = ["persona", "game", "level", "source", "session", "gen", "agent", "status", "cause",
        "frames", "time_s", "progress"]
TEXT = {"persona", "game", "level", "source", "session", "status", "cause", "agent"}


def _cell(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (list, tuple, dict)):
        return json.dumps(v, separators=(",", ":"), default=float)  # numpy scalars leak from some cores
    return v


def _parse(key: str, raw: str):
    if raw is None or raw == "":
        return None
    if key in TEXT:
        return raw
    if raw[0] in "[{":
        try:
            return json.loads(raw)
        except ValueError:
            return raw
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def _header(rows: list[dict]) -> list[str]:
    keys: list[str] = [k for k in LEAD if any(k in r for r in rows)]
    for r in rows:
        keys += [k for k in r if k not in keys and k != "route"]
    if any("route" in r for r in rows):
        keys.append("route")  # widest cell last, so the file stays readable in a spreadsheet
    return keys


def read(path: str) -> list[dict]:
    try:
        with open(path, encoding="utf-8", newline="") as f:
            return [{k: _parse(k, v) for k, v in row.items() if k is not None}
                    for row in csv.DictReader(f)]
    except FileNotFoundError:
        return []


def append(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    existing_header: list[str] | None = None
    if os.path.exists(path):
        with open(path, encoding="utf-8", newline="") as f:
            existing_header = next(csv.reader(f), None)
    new_keys = {k for r in rows for k in r}
    if existing_header is not None and not new_keys <= set(existing_header):
        old = read(path)  # a new column: rewrite once with the union header
        rows = old + rows
        existing_header = None
    header = existing_header or _header(rows)
    mode = "a" if existing_header is not None else "w"
    with open(path, mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header, restval="", extrasaction="ignore")
        if mode == "w":
            w.writeheader()
        for r in rows:
            w.writerow({k: _cell(v) for k, v in r.items()})


def reset(path: str) -> None:
    """A fresh population starts a fresh log (a re-probed level must not mix with the old one)."""
    if os.path.exists(path):
        os.remove(path)
