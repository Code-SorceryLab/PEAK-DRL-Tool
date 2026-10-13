"""A scripted player for Castlevania that does nothing but follow the route compass.

    python -m code.neuro.compass_bot                          # play every section, print how it did
    python -m code.neuro.compass_bot --level Entrance         # the full stage
    python -m code.neuro.compass_bot --demos runs/demos       # also save what it sensed and pressed

Its whole policy, on the 26 sensor values an agent gets and nothing else:

    left / right    the compass's horizontal part (sensor 18)
    up and jump     while its vertical part (19) points up; down while it points down
    whip            while an enemy is ahead at whip height and near (6); the boss from further off
    face the boss   when the compass has nowhere left to send it (20, 7, 21)

It serves two purposes. It checks the stage: if it gets through, the stage can be finished with
what the agents sense, so a section no agent learns is a training problem and not a level
problem. And with --demos it writes demonstrations in the format of manual_play --demo, which
`python -m code.neuro.imitate --game castlevania` fits a network to: a starting population for a
section evolution does not crack from random networks. It is not a person, and it is not
entered in the telemetry.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .adapters import make_adapter, list_levels
from .sensors import RAY_MAX_DIST, read_sensors

GAME = "castlevania"
NMY, NX, GX, GY, BOSS, FACE = 6, 7, 18, 19, 20, 21      # sensor slots (CastlevaniaAdapter.SENSOR_LABELS)
WHIP_AT, WHIP_BOSS_AT = 44.0, 90.0                       # NES px to the enemy ahead at which it swings


def act(vec) -> tuple[int, bool, int, bool]:
    """(move_x, jump, move_y, fire) from one sensor reading."""
    mx, my = int(np.sign(vec[GX])), int(np.sign(vec[GY]))
    boss = vec[BOSS] > 0
    if boss and mx == 0 and vec[NX] < 0:
        mx = -int(vec[FACE])                             # turn round: the bat is behind
    near = float(vec[NMY]) * RAY_MAX_DIST / 2.0          # the sensor is in world px, two per NES px
    return mx, my < 0, my, near < (WHIP_BOSS_AT if boss else WHIP_AT)


def play(level: str, seed: int, max_frames: int = 3600):
    """One attempt. Returns (adapter, sensors, held, where): the last three one row per frame."""
    adapter = make_adapter(GAME, level, max_frames, 5000.0)
    adapter.attempts = 1
    adapter.core._seed = seed
    adapter.reset()
    adapter.jitter(seed or None)                         # the first attempt as the level is; the rest with an unsteady hand
    xs, held, where = [], [], []
    core = adapter.core
    while adapter.alive:
        vec = read_sensors(adapter, "rays")[0]
        mx, jump, my, fire = act(vec)
        xs.append(vec)
        held.append((mx, int(jump), my, int(fire)))
        where.append((*core.screen, round(adapter.x), round(adapter.y), core.player.hp))
        adapter.step(mx, jump, my, fire)
    return adapter, np.asarray(xs, np.float32), np.asarray(held, np.int8), np.asarray(where, np.int32)


def main() -> None:
    ap = argparse.ArgumentParser(description="Play Castlevania by following the route compass")
    ap.add_argument("--level", default=None, help="one level id (default: every section, then the full stage)")
    ap.add_argument("--attempts", type=int, default=5)
    ap.add_argument("--demos", default=None, metavar="DIR",
                    help="save each attempt as a demonstration under DIR/castlevania/compass-bot/<level>/")
    args = ap.parse_args()
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("PEAK_CV_ATTEMPTS", "1")
    levels = [args.level] if args.level else list_levels(GAME)
    print(f"{'LEVEL':<22}{'WON':>8}{'MEAN PROGRESS':>15}  HOW THE REST ENDED")
    for level in levels:
        full = level == levels[-1] and not args.level
        wins, progress, ends = 0, 0.0, {}
        for k in range(args.attempts):
            adapter, xs, held, where = play(level, k, 18000 if full or level == "Entrance" else 3600)
            core = adapter.core
            wins += adapter.won
            progress += adapter.progress()
            if not adapter.won:
                ends[core.death_cause or adapter.status] = ends.get(core.death_cause or adapter.status, 0) + 1
            if args.demos:
                out = os.path.join(args.demos, GAME, "compass-bot", level)
                os.makedirs(out, exist_ok=True)
                status = "WON" if adapter.won else "DEAD"
                np.savez_compressed(
                    os.path.join(out, f"{k:03d}_{status.lower()}.npz"), sensors=xs, held=held, sent=held, where=where,
                    meta=json.dumps({"game": GAME, "level": level, "player": "compass-bot", "status": status,
                                     "cause": core.death_cause, "progress": round(adapter.progress(), 4),
                                     "frames": len(xs), "invincible": False, "compass": "screen", "seed": k,
                                     "columns": {"held": ["move_x", "jump", "move_y", "fire"],
                                                 "where": ["screen_x", "screen_y", "x", "y", "hp"]}}))
        rest = ", ".join(f"{v} x {c}" for c, v in sorted(ends.items(), key=lambda kv: -kv[1]))
        print(f"{level:<22}{wins:>4} / {args.attempts:<2}{100 * progress / args.attempts:>13.0f}%  {rest}", flush=True)


if __name__ == "__main__":
    main()
