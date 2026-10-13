"""Imitation: fit an agent's network to a person's demonstrations, as a head start for evolution.

    python -m code.neuro.imitate --demos runs/demos/megaman/human               # every section with demos
    python -m code.neuro.imitate --demos runs/demos/megaman/human --section CutMan-3-Ledges
    python -m code.neuro.imitate --game castlevania --demos runs/demos/castlevania/compass-bot

A demonstration (manual_play --demo) is, per frame, the 26 sensor values an agent would have seen
and the buttons the player held. Here a network of the agent's own shape is fitted to predict the
buttons from the sensors — one network per stage section, from the frames played inside that
section — and saved as a run folder (runs/megaman/<section>/imitation/) that the trainer can
start a population from:

    python -m code.neuro.trainer --game megaman --level CutMan-3-Ledges --seed-from runs/megaman/CutMan-3-Ledges/imitation

Only the baseline level is ever demonstrated. The fitted network is a starting point: evolution
still has to make it win, and on a changed level only the agents play.

The fit is plain numpy (one hidden tanh layer, sigmoid outputs, cross-entropy, Adam): the same
arithmetic as code/neuro/net.py, so the fitted weights drop straight into an agent.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from .adapters import N_INPUTS_BY_GAME, N_OUTPUTS_BY_GAME
from .evolution import GAConfig
from .net import make_net

GAME = "megaman"
OUTPUTS = ["left", "right", "jump", "up", "down", "fire"]
POS_WEIGHT_CAP = 8.0     # a rare button (up, jump) counts for more, up to this much
VX, VY = 12, 13          # the sensor slots holding Mega Man's velocity
GX, FACE = 18, 21        # the route compass (horizontal) and the facing
MIN_FRAMES = 60          # fewer frames than this inside a section is passing through, not playing it


def load_demos(root: str) -> list[dict]:
    """Every demonstration under `root`, except invincible ones (they show routes an agent cannot
    survive). Which of their frames are learned from is decided per section (section_frames)."""
    out = []
    for path in sorted(glob.glob(os.path.join(root, "**", "*.npz"), recursive=True)):
        d = np.load(path)
        if "sensors" not in d or "held" not in d:
            continue
        meta = json.loads(str(d["meta"]))
        if meta.get("invincible"):
            continue
        sensors = d["sensors"].astype(np.float32)
        if meta.get("compass") != "screen":     # recorded when the compass was relative to facing
            sensors[:, GX] *= sensors[:, FACE]
        out.append({"path": path, "meta": meta, "sensors": sensors,
                    "held": d["held"].astype(np.int8), "where": d["where"]})
    return out


def labels(held: np.ndarray, fire_window: int = 8) -> np.ndarray:
    """The six network outputs a player's held buttons stand for. People tap fire where a
    network holds its output, so a tap marks the frames around it as "wants to fire"."""
    mx, jump, my, fire = (held[:, i] for i in range(4))
    wants_fire = np.zeros(len(fire), dtype=bool)
    for t in np.nonzero(fire)[0]:
        wants_fire[max(0, t - fire_window): t + fire_window + 1] = True
    return np.stack([mx < 0, mx > 0, jump > 0, my < 0, my > 0, wants_fire], axis=1).astype(np.float32)


def section_frames(demos: list[dict], screens: set, later: set, include_losses: bool = False, fire_window: int = 8):
    """(sensors, labels, frames per demo) for the frames played inside one section. A section
    counts from a demo only if the player got through it: won there, or went on to a later one."""
    xs, ys, counts = [], [], []
    for d in demos:
        where = [tuple(s) for s in d["where"][:, :2].tolist()]
        inside = np.array([s in screens for s in where])
        if inside.sum() < MIN_FRAMES:      # e.g. the one frame in the next section that ends a win
            continue
        last = int(np.nonzero(inside)[0][-1])
        cleared = any(s in later for s in where[last + 1:]) or \
            (d["meta"].get("status") == "WON" and last == len(where) - 1)
        if not (cleared or include_losses):
            continue
        y = labels(d["held"], fire_window)
        # Standing still with nothing held (getting ready, looking at the screen) is not a
        # decision to copy: a network that reacts only to what it senses would learn to stand
        # in that spot forever. Waiting in mid-air or mid-step is kept.
        still = (d["held"] == 0).all(axis=1) & (np.abs(d["sensors"][:, VX]) < 1e-6) & (np.abs(d["sensors"][:, VY]) < 1e-6)
        keep = inside & ~still
        xs.append(d["sensors"][keep])
        ys.append(y[keep])
        counts.append(int(keep.sum()))
    if not xs:
        return None, None, []
    return np.concatenate(xs), np.concatenate(ys), counts


def fit(x: np.ndarray, y: np.ndarray, hidden: int = 16, epochs: int = 600, lr: float = 0.01, seed: int = 0):
    """Fit sensors -> buttons. Returns (w1, b1, w2, b2) and a report (accuracy, per-output recall)
    measured on frames held out from the fit."""
    rng = np.random.default_rng(seed)
    n = len(x)
    order = rng.permutation(n)
    n_val = max(1, n // 7)
    val, tr = order[:n_val], order[n_val:]
    xt, yt, xv, yv = x[tr], y[tr], x[val], y[val]
    pos = yt.mean(axis=0)
    pos_w = np.clip((1.0 - pos) / np.maximum(pos, 1e-6), 1.0, POS_WEIGHT_CAP).astype(np.float32)
    params = [rng.normal(0, 0.3, (x.shape[1], hidden)).astype(np.float32), np.zeros(hidden, np.float32),
              rng.normal(0, 0.3, (hidden, y.shape[1])).astype(np.float32), np.zeros(y.shape[1], np.float32)]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]

    def forward(inp, p):
        h = np.tanh(inp @ p[0] + p[1])
        return h, 1.0 / (1.0 + np.exp(-(h @ p[2] + p[3])))

    for step in range(1, epochs + 1):
        h, out = forward(xt, params)
        w = np.where(yt > 0.5, pos_w, 1.0)
        dz2 = w * (out - yt) / len(xt)                      # d loss / d pre-sigmoid
        dh = (dz2 @ params[2].T) * (1.0 - h * h)
        grads = [xt.T @ dh, dh.sum(0), h.T @ dz2, dz2.sum(0)]
        for i, g in enumerate(grads):                       # Adam
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            params[i] -= lr * (m[i] / (1 - 0.9 ** step)) / (np.sqrt(v[i] / (1 - 0.999 ** step)) + 1e-8)
    pred = forward(xv, params)[1] > 0.5
    truth = yv > 0.5
    report = {"frames": int(n), "held_out": int(n_val), "accuracy": float((pred == truth).mean()),
              "outputs": {name: {"share_held": round(float(y[:, i].mean()), 3),
                                 "recall": round(float(pred[:, i][truth[:, i]].mean()), 3) if truth[:, i].any() else None,
                                 "precision": round(float(truth[:, i][pred[:, i]].mean()), 3) if pred[:, i].any() else None}
                          for i, name in enumerate(OUTPUTS)}}
    return params, report


def to_run(params, out_dir: str, hidden: int, memory: int, report: dict, seed: int = 0) -> str:
    """Save the fitted network as a run folder: best.npz (the agent's flat weight vector) and
    state.json (its shape), which --seed-from, --replay and the relay all read."""
    cfg = GAConfig(pop_size=1, hidden=hidden, memory=memory)
    cfg.n_outputs = N_OUTPUTS_BY_GAME[GAME]
    cfg.n_inputs = N_INPUTS_BY_GAME[GAME]
    net = make_net(cfg)
    w1, b1, w2, b2 = params
    if memory:                                              # memory units start blank: no input weight,
        rng = np.random.default_rng(seed)                   # a small random output of their own
        w1 = np.concatenate([w1, np.zeros((memory, hidden), np.float32)])
        w2 = np.concatenate([w2, rng.normal(0, 0.1, (hidden, memory)).astype(np.float32)], axis=1)
        b2 = np.concatenate([b2, np.zeros(memory, np.float32)])
    flat = np.concatenate([p.ravel() for p in (w1, b1, w2, b2)]).astype(np.float32)
    net.set_weights(flat)                                   # raises if the shape is not the agent's
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "best.npz"), weights=flat)
    with open(os.path.join(out_dir, "state.json"), "w", encoding="utf-8") as f:
        json.dump({"config": {"sensors": "rays", "hidden": hidden, "action_feedback": False, "memory": memory},
                   "persona": "experienced", "history": [], "best_fitness": 0.0, "imitation": report}, f, indent=2)
    return out_dir


def record_agent(run_dir: str, level: str, attempts: int = 3, max_frames: int = 3600) -> list[dict]:
    """Demonstrations played by a trained agent instead of a person, in the same shape as
    manual_play --demo writes. For checking the fit: a clone of a good agent should play like it."""
    from .adapters import make_adapter
    from .relay import load_net
    from .sensors import read_sensors
    net, sensors, _ = load_net(run_dir)
    demos = []
    for k in range(attempts):
        adapter = make_adapter(GAME, level, max_frames, 5000.0)
        adapter.attempts = 1
        adapter.core._seed = k
        adapter.reset()
        net.reset()
        xs, held, where = [], [], []
        while adapter.alive:
            vec = read_sensors(adapter, sensors)[0]
            move_x, jump, move_y, fire = net.act(vec)
            xs.append(vec)
            held.append((move_x, int(jump), move_y, int(fire)))
            where.append((*adapter.core.screen, round(adapter.x), round(adapter.y), adapter.core.player.hp))
            adapter.step(move_x, jump, move_y, fire)
        demos.append({"path": f"{run_dir}#{k}", "meta": {"status": adapter.status, "level": level, "compass": "screen"},
                      "sensors": np.asarray(xs, np.float32), "held": np.asarray(held, np.int8),
                      "where": np.asarray(where, np.int32)})
    return demos


def main() -> None:
    global GAME
    from . import relay
    from .relay import evaluate, stage_legs
    ap = argparse.ArgumentParser(description="Fit agent networks to recorded demonstrations")
    ap.add_argument("--demos", required=True, help="folder of demonstrations (searched for .npz files)")
    ap.add_argument("--game", default=GAME, choices=sorted(relay.STAGE_GAMES), help="which game (default: megaman)")
    ap.add_argument("--stage", default=None, help="the full-stage level id whose sections to fit (default: the game's stage)")
    ap.add_argument("--section", default=None, help="fit only this section (default: every one with demos)")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--memory", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--fire-window", type=int, default=8,
                    help="frames either side of a fire press that count as wanting to fire: a person taps; "
                         "use 0 for a scripted player, which holds fire exactly when it means it")
    ap.add_argument("--include-losses", action="store_true", help="also learn from attempts that did not get through")
    ap.add_argument("--runs", default=None, help="where the section run folders live (default: runs/<game>)")
    ap.add_argument("--test", type=int, default=12, help="varied attempts to test each fitted network on (0 = skip)")
    args = ap.parse_args()
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    GAME = args.game
    args.stage = args.stage or relay.set_game(GAME)
    relay.set_game(GAME)
    args.runs = args.runs or os.path.join("runs", GAME)

    demos = load_demos(args.demos)
    if not demos:
        raise SystemExit(f"no demonstrations under {args.demos}")
    legs = stage_legs(args.stage)
    print(f"{len(demos)} demonstrations, {sum(len(d['sensors']) for d in demos)} frames")
    print(f"{'SECTION':<20}{'DEMOS':>6}{'FRAMES':>8}{'ACCURACY':>10}{'TESTED WINS':>13}  SAVED TO")
    for i, leg in enumerate(legs):
        if args.section and leg.level != args.section:
            continue
        later = set().union(*[other.screens for other in legs[i + 1:]]) if i + 1 < len(legs) else set()
        x, y, counts = section_frames(demos, leg.screens, later, args.include_losses, args.fire_window)
        if x is None:
            print(f"{leg.level:<20}{'—':>6}{'no demonstration got through this section':>48}")
            continue
        params, report = fit(x, y, args.hidden, args.epochs)
        report["demos"] = len(counts)
        out = to_run(params, os.path.join(args.runs, leg.level, "imitation"), args.hidden, args.memory, report)
        tested = f"{100 * evaluate(out, leg.level, args.test)[0]:.0f}%" if args.test else "—"
        print(f"{leg.level:<20}{len(counts):>6}{report['frames']:>8}{100 * report['accuracy']:>9.0f}%{tested:>13}  {out}")


if __name__ == "__main__":
    main()
