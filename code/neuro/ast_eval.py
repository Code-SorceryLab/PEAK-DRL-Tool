"""AST evaluation: agents retrained on each mutant, playtests of the original level and of every
mutant, and the metrics of code/stats/ast_metrics.py over them.

    python -m code.neuro.ast_eval retrain --game mario --mutant seal_route [--persona novice ...]
    python -m code.neuro.ast_eval play --game mario [--variant base seal_route ...] [--persona ...]
    python -m code.neuro.ast_eval play --game megaman [--episodes 20]      # the relay of section agents
    python -m code.neuro.ast_eval retrain --game megaman --mutant base-retrained [--level CutMan-1-Street]

A playtest gives each agent one attempt with learning off: the best genome of each of the three
seeds of a persona, under the rules it trained with (frame budget, stall rule, the persona's
reaction time and slips). Agents and games are deterministic, so one attempt is the answer and a
persona's win rate moves in steps of 1/3. Mega Man is played by the relay of section agents
(code/neuro/relay.py) over --episodes varied attempts. Its original's relay (runs/megaman/relay.json)
holds hand-picked legs of an older pipeline, so the mutants are compared with `base-retrained`:
the original level, every section retrained exactly as the mutants were.

    runs/ast/<game>/<variant>/agents.csv     one row per attempt (code/stats/episode_stats.py)
    runs/ast/<game>/<variant>/human/         people at the keyboard (manual_play --record)
    runs/ast/<game>/<mutant>/probes/         agents trained on the mutant (balance.py --root)

`variant` is `base` or a mutant id (code/games/eval_mutants.py). Baseline agents are the
200-generation balance probes in runs/probes, on whichever GA setting won more often on the
evaluation level; a mutant is retrained with that setting, budget and seeds. The cosmetic mutant
is never retrained: the baseline agents replay it, and its rows must match theirs (RQ2).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np

from code.games import eval_mutants
from code.games.eval_mutants import EVAL_LEVELS
from code.stats import episode_log

from . import balance
from .adapters import N_INPUTS_BY_GAME, N_OUTPUTS_BY_GAME, make_adapter
from .evolution import GAConfig
from .net import make_net
from .personas import Hands, get_persona
from .sensors import read_sensors

SEEDS = (1234, 2025, 31337)
GENS = 200
ROOT = os.path.join("runs", "ast")
PERSONA_ORDER = ["novice", "clumsy", "experienced", "speedrunner", "killer", "collector"]
MEGAMAN_STAGE, MEGAMAN_FRAMES, MEGAMAN_STALL = "CutMan", 18000, 300     # relay.py's defaults
RETRAINED = "base-retrained"    # Mega Man's original, every section retrained as its mutants were
RESEEDED = "base-reseeded"      # the original retrained with new seeds: how often a signal fires by chance
RESEEDS = (1111, 2222, 3333)


def variant_dir(game: str, variant: str) -> str:
    return os.path.join(ROOT, game, variant)


def variants(game: str) -> list[str]:
    return (["base"] + ([RETRAINED] if game == "megaman" else []) + [RESEEDED]
            + [m["id"] for m in eval_mutants.eval_set(game)])


def mutant(game: str, variant: str) -> dict | None:
    if variant == "base":
        return None
    if variant == RESEEDED or (game == "megaman" and variant == RETRAINED):   # the original, retrained
        from .relay import stage_legs
        levels = [leg.level for leg in stage_legs(MEGAMAN_STAGE)] if game == "megaman" else [EVAL_LEVELS[game][0]]
        return {"id": variant, "dir": None, "retrain": True, "section": None, "retrain_levels": levels,
                "seeds": RESEEDS if variant == RESEEDED else SEEDS}
    m = next((m for m in eval_mutants.eval_set(game) if m["id"] == variant), None)
    if m is None:
        raise SystemExit(f"{game} has no mutant '{variant}' on disk: python -m code.games.eval_mutants")
    return m


def baseline_tag(game: str, persona: str) -> str:
    """The 200-generation GA setting (default or --best) that won more on the evaluation level;
    a tie keeps the default."""
    level, best = EVAL_LEVELS[game][0], None
    for path in sorted(glob.glob(os.path.join(balance.BALANCE_DIR, f"report_{game}_{persona}_p*g{GENS}*.json"))):
        with open(path, encoding="utf-8") as f:
            rep = json.load(f)
        row = next((r for r in rep.get("levels") or [] if str(r.get("level")) == level), None)
        if row and (best is None or row["win_rate_mean"] > best[0]):
            best = (row["win_rate_mean"], rep["tag"])
    if best is None:
        raise SystemExit(f"no {GENS}-generation balance report for {game} / {persona} in {balance.BALANCE_DIR}")
    return best[1]


def agent_dirs(game: str, variant: str, persona: str) -> list[str]:
    """The run dirs of the agents that play this variant: the baseline's, or those retrained on it."""
    m, tag, level = mutant(game, variant), baseline_tag(game, persona), EVAL_LEVELS[game][0]
    if m is None or not m["retrain"]:
        return [balance.probe_dir(game, persona, tag, level, s) for s in SEEDS]
    root = os.path.join(variant_dir(game, variant), "probes")
    tags = sorted(os.path.basename(d) for d in glob.glob(os.path.join(root, game, persona, "*")))
    tag = tag if tag in tags or not tags else tags[0]       # a shorter test budget leaves its own tag
    return [balance.probe_dir(game, persona, tag, level, s, root) for s in m.get("seeds", SEEDS)]


# ── retraining ───────────────────────────────────────────────────────────────────────────────
def retrain_job(game: str, variant: str, persona: str, gens: int = GENS, workers: int | None = None,
                levels: list[str] | None = None):
    """(command, environment) that trains this persona's three seeds on the mutant, as balance.py
    trained the baseline. Mega Man retrains only the sections the mutant changes (`levels`)."""
    m = mutant(game, variant)
    if m is None or not m["retrain"]:
        raise SystemExit(f"{game} {variant} is not retrained (the baseline agents replay it)")
    cmd = [sys.executable, "-m", "code.neuro.balance", "--game", game, "--persona", persona,
           "--levels", *(levels or m["retrain_levels"]), "--seeds", *map(str, m.get("seeds", SEEDS)), "--gens", str(gens),
           "--root", os.path.join(variant_dir(game, variant), "probes"), "--out", variant_dir(game, variant)]
    if game != "megaman" and baseline_tag(game, persona).endswith("_best"):
        cmd.append("--best")
    if workers:
        cmd += ["--workers", str(workers)]
    env = dict(os.environ, PEAK_MUTANT=m["dir"] or "", SDL_VIDEODRIVER="dummy")     # "" plays the original
    env.pop("PEAK_MM1_MUTANT", None)
    return cmd, env


# ── playing ──────────────────────────────────────────────────────────────────────────────────
def load_agent(run_dir: str, game: str):
    """(network, GA config, persona) of a run's best genome, as the trainer would replay it."""
    with open(os.path.join(run_dir, "state.json"), encoding="utf-8") as f:
        meta = json.load(f)
    saved = {k: v for k, v in (meta.get("config") or {}).items() if k in GAConfig.__dataclass_fields__}
    cfg = GAConfig(**dict(saved, pop_size=1))
    cfg.n_outputs = max(cfg.n_outputs, N_OUTPUTS_BY_GAME.get(game, 3))
    if cfg.sensors == "rays":
        cfg.n_inputs = N_INPUTS_BY_GAME.get(game, 0)
    net = make_net(cfg)
    net.set_weights(np.load(os.path.join(run_dir, "best.npz"))["weights"].astype(np.float32))
    return net, cfg, get_persona(meta.get("persona") or "experienced")


def play_agent(game: str, run_dir: str, level: str, seed: int = 0) -> dict:
    """One attempt by a run's best genome with learning off. Returns its telemetry row."""
    net, cfg, persona = load_agent(run_dir, game)
    return _attempt(game, level, cfg, persona, net.act, net.reset, seed, os.path.basename(run_dir), run_dir)


def play_random(game: str, level: str, seed: int = 0) -> dict:
    """One attempt by an agent whose network outputs are uniform noise, in the body the trained
    agents use (the experienced persona's hands and reaction time, the baseline's GA config): the
    floor a learned agent has to beat."""
    _, cfg, _ = load_agent(agent_dirs(game, "base", "experienced")[0], game)
    rng = np.random.default_rng(10_000 + seed)                   # apart from the hands' own stream

    def act(_vec):
        # shortcut: Net.act's decoding for 3-output nets only (left, right, jump); Mega Man has no random baseline
        left, right, jump = rng.random(3)
        return (0 if max(left, right) <= 0.5 else -1 if left > right else 1), bool(jump > 0.5), 0
    return _attempt(game, level, cfg, get_persona("experienced"), act, lambda: None, seed,
                    f"random_{seed}", f"random/{seed}", persona_name="random")


def _attempt(game: str, level: str, cfg, persona, act, reset, seed: int, agent: str, run: str,
             persona_name: str | None = None) -> dict:
    """One attempt: `act(sensors)` gives the network outputs, the persona's hands turn them into
    buttons, the trainer's stall rule ends a run that stops making progress."""
    from code.stats.episode_stats import make_stats
    adapter = make_adapter(game, level, cfg.max_frames, cfg.win_bonus, sprint=persona.sprint,
                           time_rate=persona.time_rate, weights=persona.weights)
    adapter.reset()
    reset()
    hands = Hands(persona, np.random.default_rng(seed))
    stats = make_stats(game, adapter.core, level=level, persona=persona_name or persona.name, source="agent",
                       agent=agent)
    anchor, still, frame, vec, status = adapter.fitness(), 0, 0, None, None
    while adapter.alive:
        if vec is None or frame % persona.sensor_period == 0:   # the persona's reaction time
            vec = read_sensors(adapter, cfg.sensors)[0]
        move_x, jump, move_y, *extra = hands(act(vec))
        adapter.step(move_x, jump, move_y, *extra)
        stats.on_step(bool(jump))
        frame += 1
        fit = adapter.fitness()                                  # the trainer's stall rule
        if fit > anchor + 1.0:
            anchor, still = fit, 0
        elif getattr(adapter, "busy", False):
            still = 0
        else:
            still += 1
            if still >= cfg.stuck_frames:
                status = "STUCK"
                break
    stats.finish(status or adapter.status)
    row = stats.to_dict()
    row.update({"seed": seed, "run": run.replace("\\", "/")})
    return row


def _best_retrained(variant: str, section: str) -> str | None:
    """The run trained on this section of the variant that wins most often (relay.py's test)."""
    from .relay import evaluate
    root = os.path.join(variant_dir("megaman", variant), "probes")
    runs = [os.path.dirname(p) for p in glob.glob(os.path.join(root, "megaman", "*", "*", f"{section}_*", "best.npz"))]
    scored = [(evaluate(r, section, 12), r) for r in sorted(runs)]
    return max(scored)[1] if scored else None


def play_megaman(variant: str, episodes: int = 20, replay: bool = False) -> list[dict]:
    """The relay over the full stage, or over the one section a level mutant changes, from that
    section's start (most full-stage attempts end before segment 3, so they would never meet a
    change there). A global mutant and base-retrained are played over the full stage and over each
    section alone, so every mutant meets the retrained original over the same ground. The relay's
    hand-picked legs (base) are kept as context; the cosmetic copy is replayed by base-retrained's,
    and so is any mutant with `replay` (the frozen original's agents, not those retrained on it)."""
    from . import relay
    m = mutant("megaman", variant)
    if m is None:
        scopes = [MEGAMAN_STAGE] + sorted({x["section"] for x in eval_mutants.eval_set("megaman") if x["section"]})
    else:
        scopes = [m["section"]] if m["section"] else [MEGAMAN_STAGE] + m["retrain_levels"]
    with eval_mutants._named(m["dir"] if m else None):
        legs = relay.load_legs(os.path.join("runs", "megaman", "relay.json"), MEGAMAN_STAGE)
        for leg in legs:
            if m and leg.level in m["retrain_levels"]:
                trained = m["id"] if m["retrain"] and not replay else RETRAINED
                run = _best_retrained(trained, leg.level)
                if run is None:
                    raise SystemExit(f"no agent retrained on {trained} for {leg.level}: retrain it first")
                leg.run_dir = run
                leg.net, leg.sensors, leg.persona = relay.load_net(run)
        rows = []
        for scope in scopes:
            use = legs if scope == MEGAMAN_STAGE else [leg for leg in legs if leg.level == scope]
            for i in range(episodes):
                row = relay.play(use, scope, 1234 + i, MEGAMAN_FRAMES, MEGAMAN_STALL, session=variant)
                rows.append(dict(row, variant=variant))
                print(f"  {variant:<22}{scope:<18} seed {row['seed']}  {row['status']:<6}{row['cause'] or '':<12}"
                      f"{100 * (row['progress'] or 0):>4.0f}%  {row.get('section') or ''}", flush=True)
    return rows


def play(game: str, variant: str, personas: list[str] | None = None, episodes: int = 20,
         replay: bool = False, random: bool = False) -> str:
    """Playtest one variant and write its agents.csv (rows of personas not replayed are kept).
    `replay` plays the original's frozen agents on a retrained mutant instead, into agents_replay.csv;
    `random` the random-action baseline, as many attempts as the agents make, into agents_random.csv."""
    personas = personas or PERSONA_ORDER
    m = mutant(game, variant)
    if random:
        if game == "megaman" or variant in (RETRAINED, RESEEDED):
            return f"skip {variant}: the random baseline plays the 2D games' original and mutants"
        with eval_mutants._named(m["dir"] if m else None):
            rows = [dict(play_random(game, EVAL_LEVELS[game][0], s), variant=variant)
                    for s in range(len(PERSONA_ORDER) * len(SEEDS))]
        path = os.path.join(variant_dir(game, variant), "agents_random.csv")
        episode_log.reset(path)
        episode_log.append(path, rows)
        return f"wrote {path} ({sum(r['status'] == 'WON' for r in rows)}/{len(rows)} won)"
    if replay and (m is None or not m["retrain"] or variant in (RETRAINED, RESEEDED)):
        return f"skip {variant}: only a retrained mutant has a replay of its own"
    name = "agents_replay.csv" if replay else "agents.csv"
    if game == "megaman":
        rows, keep = play_megaman(variant, episodes, replay), []
    else:
        level, rows = EVAL_LEVELS[game][0], []
        with eval_mutants._named(m["dir"] if m else None):
            for persona in personas:
                for run in agent_dirs(game, "base" if replay else variant, persona):
                    if not os.path.exists(os.path.join(run, "best.npz")):
                        print(f"  {variant:<20}{persona:<13}missing {run}", flush=True)
                        continue
                    row = dict(play_agent(game, run, level), variant=variant)
                    rows.append(row)
                    print(f"  {variant:<20}{persona:<13}{row['status']:<6}{row['cause'] or '':<10}"
                          f"{100 * (row['progress'] or 0):>4.0f}%  {row['time_s']:>6.1f} s", flush=True)
        path = os.path.join(variant_dir(game, variant), name)
        keep = [r for r in (episode_log.read(path) if os.path.exists(path) else []) if r.get("persona") not in personas]
    path = os.path.join(variant_dir(game, variant), name)
    if not rows:
        return f"nothing for {variant}: no agents to play it"
    episode_log.reset(path)
    episode_log.append(path, keep + rows)
    return f"wrote {path}"


def main() -> None:
    ap = argparse.ArgumentParser(description="AST evaluation: retrain on mutants, playtest, measure")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("retrain", help="train each persona's three seeds on one mutant")
    r.add_argument("--game", required=True, choices=list(eval_mutants.CHANGES) + ["megaman"])
    r.add_argument("--mutant", required=True)
    r.add_argument("--persona", nargs="*", default=None, help="default: all six (Mega Man: each section's own)")
    r.add_argument("--gens", type=int, default=GENS)
    r.add_argument("--workers", type=int, default=None)
    r.add_argument("--level", nargs="*", default=None,
                   help="Mega Man: only these sections (default: every section the mutant changes)")
    p = sub.add_parser("play", help="one attempt per agent on the original level and the mutants")
    p.add_argument("--game", required=True, choices=list(EVAL_LEVELS))
    p.add_argument("--variant", nargs="*", default=None, help="base and/or mutant ids (default: all)")
    p.add_argument("--persona", nargs="*", default=None, help="default: all six")
    p.add_argument("--episodes", type=int, default=20, help="Mega Man relay attempts per variant")
    p.add_argument("--replay", action="store_true",
                   help="the original's frozen agents on each retrained mutant (agents_replay.csv)")
    p.add_argument("--random", action="store_true",
                   help="the random-action baseline on the original and each mutant (agents_random.csv)")
    args = ap.parse_args()
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    if args.cmd == "retrain":
        if args.game == "megaman":                       # each section with the persona its relay leg plays
            from .relay import load_legs
            m = mutant("megaman", args.mutant)
            jobs: dict = {}
            for leg in load_legs(os.path.join("runs", "megaman", "relay.json"), MEGAMAN_STAGE):
                if leg.level in (m or {}).get("retrain_levels", []) and (not args.level or leg.level in args.level):
                    jobs.setdefault(leg.persona.name, []).append(leg.level)
        else:
            jobs = {p: None for p in args.persona or PERSONA_ORDER}
        for persona, levels in jobs.items():
            cmd, env = retrain_job(args.game, args.mutant, persona, args.gens, args.workers, levels)
            print(">>> PEAK_MUTANT=" + env["PEAK_MUTANT"] + " " + " ".join(cmd), flush=True)
            if subprocess.run(cmd, env=env).returncode:
                raise SystemExit(1)
        return
    for variant in args.variant or variants(args.game):
        print(play(args.game, variant, args.persona, args.episodes, args.replay, args.random), flush=True)


if __name__ == "__main__":
    main()
