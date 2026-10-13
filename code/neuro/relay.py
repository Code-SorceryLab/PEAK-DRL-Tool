"""Relay: play a whole stage with one specialist network per slice.

Each slice of the stage was trained as its own level; here they run back to back on the full
stage, the network changing hands as the player crosses from one slice's screens into the next.
That gives the end-to-end attempt a playtest needs, with the metrics split by section.
The game is Mega Man unless `--game castlevania` says otherwise; every path below then reads
runs/castlevania/... and the sections are that game's.

    python -m code.neuro.relay --pick                 # choose the best trained run for each slice
    python -m code.neuro.relay --episodes 20          # play the full stage, print and save metrics
    python -m code.neuro.relay --watch                # the same, live on the dashboard
    python -m code.neuro.relay --section CutMan-3-Ledges --episodes 20   # one slice alone, from its start

`--pick` tests the best trained runs of each section (runs/megaman/<section>/<run>/) over a dozen
varied attempts, keeps the one that wins most often, and writes runs/megaman/relay.json (edit
it to choose by hand). Episodes are seeded `--seed`, `--seed + 1`, ..., so a batch is repeatable.
Results: runs/megaman/relay/episodes.csv, one row per attempt in the shared telemetry schema
(code/stats/), with the split by section in its `sections` column, and metrics.json beside it.
`--section` plays one slice on its own with its picked network, from the slice's start at full
energy, one attempt per episode; results go to runs/megaman/relay/sections/<section>/, so the
relay's telemetry page lists each section played this way next to the full stage.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass

import numpy as np
import yaml

from .adapters import N_INPUTS_BY_GAME, N_OUTPUTS_BY_GAME, OWN_LEVEL_FILES, STAGE_GAMES, make_adapter
from .evolution import GAConfig
from .net import make_net
from .personas import PERSONAS, Hands
from .sensors import read_sensors

GAME = "megaman"             # the game in play; --game (or set_game) switches it
GAMES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "games")


def set_game(game: str) -> str:
    """Play this game's stage from here on (megaman, castlevania). Returns its full-stage level id."""
    global GAME
    if game not in STAGE_GAMES:
        raise SystemExit(f"'{game}' is not played as a stage in sections (those are: {', '.join(STAGE_GAMES)})")
    GAME = game
    return STAGE_GAMES[game]


@dataclass
class Leg:
    """One slice of the stage and the network that plays it."""
    level: str
    screens: set            # the stage screens (sx, sy) this slice covers
    run_dir: str | None = None
    net: object = None
    sensors: str = "rays"
    persona: object = None


def stage_legs(stage_level: str, game: str | None = None) -> list[Leg]:
    """The slices cut from this stage, in order, each with the screens it owns."""
    with open(os.path.join(GAMES_DIR, OWN_LEVEL_FILES[game or GAME]), encoding="utf-8") as f:
        levels = (yaml.safe_load(f) or {}).get("levels") or {}
    stage = (levels.get(stage_level) or {}).get("stage")
    legs = []
    for name, entry in levels.items():
        section = entry.get("section") or {}
        if name != stage_level and entry.get("stage") == stage and section.get("screens"):
            legs.append(Leg(name, {tuple(sc) for sc in section["screens"]}))
    if not legs:
        raise SystemExit(f"no sections of '{stage_level}' in {OWN_LEVEL_FILES[game or GAME]}")
    return legs


def run_summary(run_dir: str) -> dict | None:
    """How a training run did: wins over its last ten generations, and its best fitness."""
    try:
        with open(os.path.join(run_dir, "state.json"), encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, ValueError):
        return None
    hist = meta.get("history") or []
    if not hist or not os.path.exists(os.path.join(run_dir, "best.npz")):
        return None
    last = hist[-10:]
    episodes = sum(len(g.get("envs") or []) for g in last) or 1
    return {"run": run_dir, "gens": len(hist), "best": float(meta.get("best_fitness") or 0.0),
            "win_rate": sum(g.get("wins", 0) for g in last) / episodes,
            "first_win": next((g["gen"] for g in hist if g.get("wins")), None),
            "reach": max(g.get("best_x", 0.0) for g in hist) / 1000.0}


def load_net(run_dir: str):
    """(network, sensor mode, persona) of a training run's best genome."""
    with open(os.path.join(run_dir, "state.json"), encoding="utf-8") as f:
        meta = json.load(f)
    saved = meta.get("config") or {}
    cfg = GAConfig(pop_size=1, **{k: saved[k] for k in ("sensors", "hidden", "action_feedback", "memory")
                                  if k in saved})
    cfg.n_outputs = max(cfg.n_outputs, N_OUTPUTS_BY_GAME.get(GAME, 3))
    if cfg.sensors == "rays":
        cfg.n_inputs = N_INPUTS_BY_GAME.get(GAME, 0)
    net = make_net(cfg)
    net.set_weights(np.load(os.path.join(run_dir, "best.npz"))["weights"].astype(np.float32))
    return net, cfg.sensors, PERSONAS.get(meta.get("persona") or "", PERSONAS["experienced"])


def evaluate(run_dir: str, level: str, attempts: int) -> tuple[float, float, float]:
    """Test a run's best genome on its section: (share of attempts won, mean progress, mean fitness) over
    `attempts` tries with varied seeds and starting energy — conditions it was not selected on."""
    net, sensors, persona = load_net(run_dir)
    adapter = make_adapter(GAME, level, 3600, 5000.0)
    adapter.attempts = attempts
    adapter._episode = 4242                             # fixed, so every candidate faces the same tries
    adapter.reset()
    net.reset()
    hands = Hands(persona, np.random.default_rng(0))
    frames, vec = 0, None
    while adapter.alive and frames < attempts * 4000:
        if vec is None or frames % persona.sensor_period == 0:
            vec = read_sensors(adapter, sensors)[0]
        adapter.step(*hands(net.act(vec)))
        frames += 1
    return adapter.attempt_wins / attempts, adapter.progress(), float(adapter.fitness())


def pick(runs_root: str, stage_level: str, attempts: int = 12, shortlist: int = 6) -> dict:
    """The strongest run for each section, chosen by testing: the best few by training record
    are each played `attempts` times and the one that wins most often is kept."""
    chosen = {}
    for leg in stage_legs(stage_level):
        found = [s for s in map(run_summary, sorted(glob.glob(os.path.join(runs_root, leg.level, "*")))) if s]
        found.sort(key=lambda s: (s["win_rate"], s["best"]), reverse=True)
        for s in found[:shortlist]:
            s["tested_win_rate"], s["tested_progress"], s["tested_fitness"] = evaluate(s["run"], leg.level, attempts)
            s["candidates"] = len(found)
        if found:                                       # between equal winners, the fitter: faster, and with more energy left
            chosen[leg.level] = max(found[:shortlist], key=lambda s: (s["tested_win_rate"], s["tested_progress"], s["tested_fitness"]))
    return chosen


def load_legs(manifest: str, stage_level: str) -> list[Leg]:
    with open(manifest, encoding="utf-8") as f:
        chosen = json.load(f)["legs"]
    legs = stage_legs(stage_level)
    for leg in legs:
        entry = chosen.get(leg.level)
        if not entry:
            raise SystemExit(f"{manifest} has no run for section '{leg.level}' — train it, then --pick again")
        leg.run_dir = entry["run"]
        leg.net, leg.sensors, leg.persona = load_net(leg.run_dir)
    return legs


def play(legs: list[Leg], stage_level: str, seed: int, max_frames: int, stall: int, on_frame=None,
         session: str = "") -> dict:
    """One attempt at the full stage (or at one slice of it). Returns the episode's telemetry row (code/stats/episode_stats.py):
    the shared columns plus the split by section."""
    from code.stats.episode_stats import make_stats
    adapter = make_adapter(GAME, stage_level, max_frames, 5000.0)
    core = adapter.core
    adapter.attempts = 1                                # a slice trains on several tries per episode; a playtest is one
    core._seed = seed
    adapter.reset()
    rng = np.random.default_rng(seed)
    adapter.jitter(seed)                                # what makes attempts differ (see jitter)
    hands = {leg.level: Hands(leg.persona, rng) for leg in legs}
    by_screen = {s: i for i, leg in enumerate(legs) for s in leg.screens}
    stats = make_stats(GAME, core, level=stage_level, persona=legs[0].persona.name, source="agent",
                       agent=f"relay-{seed}", session=session)
    cur = -1
    vec, rays, tiles = None, [], []
    frames = 0
    best, since_gain = adapter.fitness(), 0
    status = None
    while adapter.alive:
        idx = by_screen.get(core.screen, cur)
        if idx != cur:                                  # the baton changes hands
            cur = idx
            legs[cur].net.reset()
            vec = None
        leg = legs[cur]
        if vec is None or frames % leg.persona.sensor_period == 0:
            vec, rays, tiles = read_sensors(adapter, leg.sensors)
        move_x, jump, move_y, *extra = hands[leg.level](leg.net.act(vec))
        adapter.step(move_x, jump, move_y, *extra)
        stats.on_step(bool(jump))
        frames += 1
        fit = adapter.fitness()
        if fit > best + 1e-6 or adapter.busy:
            best, since_gain = max(best, fit), 0
        else:
            since_gain += 1
            if stall and since_gain >= stall:           # the trainer's rule: no progress, no episode
                status = "STUCK"
                break
        if on_frame:
            on_frame(adapter, leg, rays, tiles, vec)
    stats.finish(status or adapter.status)
    row = stats.to_dict()
    row["seed"] = seed
    return row


def _ended_in(e: dict) -> str:
    return e.get("section") or "?"


def report(episodes: list[dict], legs: list[Leg], what: str = "the full stage") -> str:
    n = len(episodes)
    won = [e for e in episodes if e["status"] == "WON"]
    out = [f"{n} attempts at {what}: {len(won)} won, "
           f"mean distance along it {100 * sum(e['progress'] or 0 for e in episodes) / n:.0f}%, "
           f"mean time {sum(e['time_s'] for e in episodes) / n:.0f} s", "",
           f"{'SECTION':<20}{'REACHED':>8}{'CLEARED':>9}{'ENDED HERE':>12}{'AVG S':>8}{'AVG DMG':>9}{'AVG KILLS':>11}"]
    for i, leg in enumerate(legs):
        secs = [s for e in episodes for s in e.get("sections") or [] if s["section"] == leg.level]
        cleared = sum(1 for e in episodes if (e.get("sections_reached") or 0) > i + 1
                      or (e["status"] == "WON" and any(s["section"] == leg.level for s in e.get("sections") or [])))
        ended = sum(1 for e in episodes if _ended_in(e) == leg.level and e["status"] != "WON")
        m = len(secs) or 1
        out.append(f"{leg.level:<20}{len(secs):>8}{cleared:>9}{ended:>12}"
                   f"{sum(s['frames'] for s in secs) / m / 60:>8.1f}{sum(s['damage'] for s in secs) / m:>9.1f}"
                   f"{sum(s['kills'] for s in secs) / m:>11.1f}")
    causes: dict = {}
    for e in episodes:
        if e["status"] != "WON":
            key = f"{e['cause']} in {_ended_in(e)}"
            causes[key] = causes.get(key, 0) + 1
    if causes:
        out += ["", "How the failed attempts ended:"]
        out += [f"  {v:>3} x {k}" for k, v in sorted(causes.items(), key=lambda kv: -kv[1])]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Play a full stage with one specialist per slice")
    ap.add_argument("--game", default=GAME, choices=sorted(STAGE_GAMES), help="which game (default: megaman)")
    ap.add_argument("--stage", default=None, help="the full-stage level id (default: the game's stage)")
    ap.add_argument("--runs", default=None, help="where the slice runs live (default: runs/<game>)")
    ap.add_argument("--manifest", default=None, help="which run plays which slice (default: <runs>/relay.json)")
    ap.add_argument("--pick", action="store_true", help="choose the best run per slice and write the manifest")
    ap.add_argument("--section", default=None, help="play this one slice alone, from its start, instead of the full stage")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-frames", type=int, default=18000, help="frame budget per attempt (5 min)")
    ap.add_argument("--stall", type=int, default=300, help="end an attempt after this many frames without progress (0 = never)")
    ap.add_argument("--out", default=None, help="results dir (default: <runs>/relay)")
    ap.add_argument("--watch", action="store_true", help="play live on the dashboard instead of in a batch")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    args.stage = args.stage or set_game(args.game)
    set_game(args.game)
    args.runs = args.runs or os.path.join("runs", GAME)
    manifest = args.manifest or os.path.join(args.runs, "relay.json")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    if args.pick:
        chosen = pick(args.runs, args.stage)
        legs = stage_legs(args.stage)
        print(f"{'SECTION':<20}{'TESTED WINS':>12}{'TESTED REACH':>14}{'RUNS':>6}  PICKED RUN")
        for leg in legs:
            s = chosen.get(leg.level)
            if s is None:
                print(f"{leg.level:<20}{'— no trained run found —':>40}")
                continue
            print(f"{leg.level:<20}{100 * s['tested_win_rate']:>11.0f}%{100 * s['tested_progress']:>13.0f}%"
                  f"{s['candidates']:>6}  {s['run']}")
        with open(manifest, "w", encoding="utf-8") as f:
            json.dump({"game": GAME, "stage": args.stage, "legs": chosen}, f, indent=2)
        print(f"\nwrote {manifest}")
        return

    if not os.path.exists(manifest):
        raise SystemExit(f"{manifest} not found — run with --pick first")
    legs = load_legs(manifest, args.stage)
    level = args.stage
    if args.section:
        legs = [leg for leg in legs if leg.level == args.section]
        if not legs:
            raise SystemExit(f"'{args.section}' is not a section of '{args.stage}'")
        level = args.section

    if args.watch:
        import pygame

        from .trainer import Controls, SharedState, Trainer
        from .server import start_server
        pygame.init()
        state = SharedState(Controls())
        start_server(state, http_port=args.port)
        print(f"dashboard: http://127.0.0.1:{args.port}/{GAME}/index.html   (Ctrl+C to stop)", flush=True)
        trainer = Trainer(GAME, level, GAConfig(pop_size=1),     # only its dashboard plumbing is used
                          run_dir=os.path.join(args.runs, "_relay_watch"), state=state)
        clock = pygame.time.Clock()
        slot = trainer.slots[0]

        def show(adapter, leg, rays, tiles, vec):
            slot.adapter, slot.net = adapter, leg.net
            slot.last_sensors, slot.last_rays, slot.last_tiles = vec, rays, tiles
            trainer._publish_frames(encode_thumbs=True)
            trainer._publish_stats([adapter.status], [adapter.fitness()], 60.0)
            clock.tick(60)

        seed = args.seed
        while True:
            e = play(legs, level, seed, args.max_frames, args.stall, on_frame=show)
            print(f"seed {seed}: {e['status']} {e['cause']} in {_ended_in(e)}, "
                  f"{100 * (e['progress'] or 0):.0f}% of the way in {e['time_s']:.0f} s", flush=True)
            seed += 1

    from code.stats import episode_log, summarize
    out = args.out or os.path.join(args.runs, "relay", *(("sections", args.section) if args.section else ()))
    csv_path = os.path.join(out, "episodes.csv")
    episode_log.reset(csv_path)                         # one batch per file: rerunning replaces it
    session = os.path.basename(manifest)
    episodes = []
    for i in range(args.episodes):
        e = play(legs, level, args.seed + i, args.max_frames, args.stall, session=session)
        episodes.append(e)
        print(f"  {i + 1:>3}/{args.episodes}  seed {e['seed']}  {e['status']:<6}{e['cause']:<14} in {_ended_in(e):<20}"
              f"{100 * (e['progress'] or 0):>4.0f}% of the way  {e['time_s']:>6.1f} s  hp {e['hp']:>2}  "
              f"kills {e['kills']}", flush=True)
    print()
    print(report(episodes, legs, args.section or "the full stage"))
    episode_log.append(csv_path, episodes)              # the shared telemetry: one row per episode
    summarize.summarize_file(csv_path, verbose=False)   # -> metrics.json beside it
    with open(os.path.join(out, "legs.json"), "w", encoding="utf-8") as f:
        json.dump({"game": GAME, "stage": args.stage, "section": args.section, "manifest": manifest, "legs": {leg.level: leg.run_dir for leg in legs}}, f, indent=2)
    print(f"\nsaved {csv_path} (+ metrics.json, legs.json)")
    print(f"dashboard: python -m code.stats.telemetry {out}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nstopped", flush=True)
        raise SystemExit(130)
