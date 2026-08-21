"""Generational neuroevolution trainer + CLI.

Run:  python -m code.neuro.trainer --game mario [--level Mario1-1a] [--turbo] [--serve]
      python -m code.neuro.trainer --resume runs/mario
      python -m code.neuro.trainer --replay runs/mario/best.npz

All population members play simultaneously (round-robin stepped) so the dashboard
can show a live grid. The trainer thread owns every pygame surface; the server
thread only ever sees encoded frame bytes.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import threading
import time

import numpy as np
import pygame

from .adapters import GameAdapter, list_levels, make_adapter
from .evolution import GAConfig, Population
from .personas import PERSONAS, Persona, get_persona
from .net import NeuralNet, make_net
from .sensors import N_BODY, SENSOR_MODES, read_sensors

THUMB_SCALE = 0.35          # thumbnail downscale factor
THUMB_INTERVAL = 0.20       # s between thumbnail encodes (real-time mode)
THUMB_INTERVAL_TURBO = 1.0
WATCH_INTERVAL = 0.05       # s between watched-env encodes
STATS_INTERVAL = 0.5


class Controls:
    """Flags the dashboard flips; the trainer polls them once per tick."""

    def __init__(self, turbo: bool = False) -> None:
        self.turbo = turbo
        self.watch_env = 0
        self.sensors_on = True
        self.manual = False  # human plays the watched env instead of its net
        self.keys = {"left": False, "right": False, "jump": False}
        self.hitboxes = False  # classic PEAK debug overlays, drawn by the game cores
        self.grid = False
        self.level_request: str | None = None  # dashboard level pick; applied at the next gen boundary


class SharedState:
    """Thread-safe snapshot bridge between trainer and websocket server."""

    def __init__(self, controls: Controls) -> None:
        self.controls = controls
        self._lock = threading.Lock()
        self._stats: dict = {}
        self._envs: list[dict] = []
        self._viewers = 0  # connected dashboard clients; frames aren't encoded at 0

    def add_viewer(self) -> None:
        with self._lock:
            self._viewers += 1

    def remove_viewer(self) -> None:
        with self._lock:
            self._viewers = max(0, self._viewers - 1)

    @property
    def has_viewers(self) -> bool:
        return self._viewers > 0

    def publish(self, stats: dict | None = None, envs: list[dict] | None = None) -> None:
        with self._lock:
            if stats is not None:
                self._stats = stats
            if envs is not None:
                self._envs = envs

    def snapshot(self) -> tuple[dict, list[dict]]:
        with self._lock:
            return self._stats, self._envs


class EnvSlot:
    def __init__(self, adapter: GameAdapter, net: NeuralNet) -> None:
        self.adapter = adapter
        self.net = net
        self.frames = 0
        self.stuck_anchor_x = 0.0
        self.stuck_frames = 0
        self.last_rays: list = []
        self.last_tiles: list = []
        self.last_sensors: np.ndarray = np.zeros(net.n_inputs, dtype=np.float32)
        # per-episode telemetry for the balance CSV (Amr's stats schema)
        self.route: list[tuple[float, float]] = []
        self.jump_count = 0
        self.prev_jump = False
        self.vx_sum = 0.0


def _encode(surface: pygame.Surface, scale: float = 1.0) -> bytes:
    if scale != 1.0:
        w, h = surface.get_size()
        surface = pygame.transform.smoothscale(surface, (int(w * scale), int(h * scale)))
    buf = io.BytesIO()
    try:
        pygame.image.save(surface, buf, "frame.jpg")
    except pygame.error:
        pygame.image.save(surface, buf, "frame.png")
    return buf.getvalue()


class Trainer:
    def __init__(self, game: str, level: str | None, cfg: GAConfig, run_dir: str,
                 state: SharedState | None = None, population: Population | None = None,
                 persona: Persona | None = None) -> None:
        self.game = game
        self.cfg = cfg
        self.run_dir = run_dir
        self.state = state
        self.persona = persona or PERSONAS["experienced"]
        self.net_proto = make_net(cfg)
        self.pop = population or Population(cfg, self.net_proto.n_params)
        self.pop.persona = self.persona.name  # persisted so replay matches capabilities
        self.pop.game = game  # tags embedded in best.npz

        # Level curriculum: an explicit --level locks that one level; otherwise the
        # trainer walks every enabled level in config order, advancing once a
        # generation produces cfg.advance_wins winners.
        if level is not None:
            self.levels: list[str] | None = None
            self.level: str | None = level
        else:
            self.levels = list_levels(game) or None
            idx = 0
            for row in reversed(self.pop.history):  # resume on the level we left off
                if self.levels and row.get("level") in self.levels:
                    idx = self.levels.index(row["level"])
                    break
            self.level = self.levels[idx] if self.levels else None

        self.slots = [
            EnvSlot(make_adapter(game, self.level, cfg.max_frames, cfg.win_bonus,
                                 sprint=self.persona.sprint, time_rate=self.persona.time_rate),
                    make_net(cfg))
            for _ in range(cfg.pop_size)
        ]
        self._surfaces: list[pygame.Surface] | None = None
        # Base mutation settings, restored when the curriculum moves to an unsolved level.
        # If resuming an already-annealed run, recover the originals by inverting the factor.
        if self.pop.annealed and cfg.anneal_factor not in (0.0, 1.0):
            self._base_mutation = (cfg.mutation_rate / cfg.anneal_factor,
                                   cfg.mutation_sigma / cfg.anneal_factor)
        else:
            self._base_mutation = (cfg.mutation_rate, cfg.mutation_sigma)
        self._step_accum = 0  # cumulative env-steps, sampled by _steps_per_sec
        self._steps_window: list[tuple[float, int]] = []  # (timestamp, cumulative steps)
        self._start_time = time.time()

    # ── serving helpers ──────────────────────────────────────────────────

    def _surface_for(self, i: int) -> pygame.Surface:
        if self._surfaces is None:
            core = self.slots[0].adapter.core  # type: ignore[attr-defined]
            self._surfaces = [pygame.Surface((core.WIDTH, core.HEIGHT)) for _ in self.slots]
        return self._surfaces[i]

    def _publish_stats(self, statuses: list[str], fitnesses: list[float], sps: float) -> None:
        if self.state is None:
            return
        hist = self.pop.history
        last10 = hist[-10:]
        episodes10 = len(last10) * self.cfg.pop_size
        self.state.publish(stats={
            "gen": self.pop.generation,
            "all_time_best": round(self.pop.best_fitness if hist else 0.0, 1),
            "last_gen_best": round(hist[-1]["best"], 1) if hist else 0.0,
            "avg_fitness": round(hist[-1]["avg"], 1) if hist else 0.0,
            "elite": self.cfg.elite,
            "mut_rate": self.cfg.mutation_rate,
            "pop_size": self.cfg.pop_size,
            "hidden": self.cfg.hidden,
            "sensors": self.cfg.sensors,
            "level": self.level or "auto",
            "levels": self.levels or ([self.level] if self.level else []),
            "persona": self.persona.name,
            "game": self.game,
            "sps": round(sps),
            "turbo": self.state.controls.turbo,
            "manual": self.state.controls.manual,
            "history": [[round(h["best"], 1), round(h["avg"], 1)] for h in hist[-400:]],
            "live_fitness": [round(f, 1) for f in fitnesses],
            "statuses": statuses,
            "results": [
                {k: r.get(k) for k in ("gen", "level", "best", "avg", "median", "wins", "stuck",
                                       "dead", "best_x", "avg_score", "coins", "duration")}
                for r in hist[-60:]
            ],
            "win_rate10": round(100 * sum(r.get("wins", 0) for r in last10) / episodes10, 1)
                          if episodes10 else 0.0,
            "elapsed": int(time.time() - self._start_time),
            "train_time": round(sum(r.get("duration") or 0 for r in hist), 1),  # cumulative, survives --resume
            "total_frames": self._step_accum,
        })

    def _publish_frames(self, encode_thumbs: bool) -> None:
        if self.state is None or not self.state.has_viewers:
            return
        ctrl = self.state.controls
        for slot in self.slots:  # classic PEAK overlays drawn by the cores themselves
            dm = getattr(slot.adapter.core, "debug_manager", None)  # type: ignore[attr-defined]
            if dm is not None:
                dm.show_hitboxes = ctrl.hitboxes
                dm.show_grid = ctrl.grid
                # NOTE: dm.show_sensors stays off — the core's jump-arc overlay
                # perturbs game state when rendered (breaks determinism); the
                # dashboard draws its own rays from sensors.py instead.
        watch = ctrl.watch_env % len(self.slots)
        envs = []
        for i, slot in enumerate(self.slots):
            entry: dict = {
                "id": i,
                "x": round(slot.adapter.x, 1),
                "fitness": round(slot.adapter.fitness(), 1),
                "status": slot.adapter.status,
                "watched": i == watch,
            }
            if encode_thumbs or i == watch:
                surf = self._surface_for(i)
                slot.adapter.render(surf)
                cam_x, cam_y = slot.adapter.camera
                entry["jpg"] = _encode(surf, 1.0 if i == watch else THUMB_SCALE)
                entry["scale"] = 1.0 if i == watch else THUMB_SCALE
                if self.state.controls.sensors_on:
                    entry["rays"] = [
                        [round(x1 - cam_x, 1), round(y1 - cam_y, 1),
                         round(x2 - cam_x, 1), round(y2 - cam_y, 1), hit]
                        for x1, y1, x2, y2, hit in slot.last_rays
                    ]
                    entry["tiles"] = [
                        [round(tx - cam_x, 1), round(ty - cam_y, 1), ts, kind]
                        for tx, ty, ts, kind in slot.last_tiles
                    ]
                    if i == watch:  # grid mode: the 363 cells are drawn as tiles, bars show body only
                        shown = slot.last_sensors if self.cfg.sensors == "rays" else slot.last_sensors[-N_BODY:]
                        entry["sensors"] = [round(float(v), 2) for v in shown]
                if i == watch:
                    entry["live"] = slot.adapter.episode_stats()
            envs.append(entry)
        self.state.publish(envs=envs)

    def _append_episode_csv(self, env_rows: list[dict]) -> None:
        """One row per finished episode, in the stats dashboard's CSV schema."""
        path = os.path.join(self.run_dir, "episodes.csv")
        os.makedirs(self.run_dir, exist_ok=True)
        new = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["persona", "game", "world", "cause_of_death", "jump_count",
                            "coins_collected", "avg_vx", "progress_ratio", "route",
                            "enemies_killed", "elapsed_time"])
            for slot, rec in zip(self.slots, env_rows):
                cause = "Success" if rec["status"] == "WON" else (rec.get("cause") or rec["status"].title())
                level_len = rec.get("level_len") or 1.0
                progress = min(max((rec.get("end_x") or 0.0) / level_len, 0.0), 1.0)
                w.writerow([self.persona.name, self.game, self.level or "auto", cause,
                            slot.jump_count, rec.get("coins", 0),
                            round(slot.vx_sum / max(slot.frames, 1), 2), round(progress, 3),
                            repr(slot.route), rec.get("kills", 0),
                            round(slot.frames / 60.0, 2)])

    def _steps_per_sec(self) -> float:
        now = time.time()
        self._steps_window.append((now, self._step_accum))
        while len(self._steps_window) > 1 and now - self._steps_window[0][0] > 2.0:
            self._steps_window.pop(0)
        t0, c0 = self._steps_window[0]
        return (self._step_accum - c0) / max(now - t0, 1e-6)

    # ── core loop ────────────────────────────────────────────────────────

    def run_generation(self) -> list[float]:
        for i, slot in enumerate(self.slots):
            slot.net.set_weights(self.pop.weights[i])
            slot.net.reset()   # clear action-feedback / memory carry at the episode boundary
            slot.adapter.reset()
            slot.frames = 0
            slot.stuck_anchor_x = 0.0
            slot.stuck_frames = 0
            slot.route = [(round(slot.adapter.x, 1), round(slot.adapter.y, 1))]
            slot.jump_count = 0
            slot.prev_jump = False
            slot.vx_sum = 0.0

        clock = pygame.time.Clock()
        last_thumb = last_watch = last_stats = 0.0

        ctrl = self.state.controls if self.state else None
        while any(s.adapter.alive for s in self.slots):
            manual_idx = ctrl.watch_env % len(self.slots) if ctrl and ctrl.manual else -1
            for i, slot in enumerate(self.slots):
                if not slot.adapter.alive:
                    continue
                # Persona reaction time: the novice only gets fresh senses every Nth frame
                if slot.frames % self.persona.sensor_period == 0:
                    vec, rays, tiles = read_sensors(slot.adapter, self.cfg.sensors)
                    slot.last_sensors, slot.last_rays, slot.last_tiles = vec, rays, tiles
                else:
                    vec = slot.last_sensors
                if i == manual_idx:
                    k = ctrl.keys
                    move_x = (1 if k["right"] else 0) - (1 if k["left"] else 0)
                    jump = k["jump"]
                else:
                    move_x, jump = slot.net.act(vec)
                slot.adapter.step(move_x, jump)
                slot.frames += 1
                self._step_accum += 1
                if jump and not slot.prev_jump:
                    slot.jump_count += 1
                slot.prev_jump = bool(jump)
                slot.vx_sum += slot.adapter.vx
                if slot.frames % 8 == 0:  # ~7.5 route samples/s, matches Amr's density
                    slot.route.append((round(slot.adapter.x, 1), round(slot.adapter.y, 1)))
                # trainer-side stuck kill (faster than the core's 20s stall watchdog)
                fit = slot.adapter.fitness()
                if fit > slot.stuck_anchor_x + 1.0:
                    slot.stuck_anchor_x = fit
                    slot.stuck_frames = 0
                else:
                    slot.stuck_frames += 1
                    if slot.stuck_frames >= self.cfg.stuck_frames and slot.adapter.alive:
                        slot.adapter.alive = False
                        slot.adapter.status = "STUCK"
                        slot.adapter._end_xy = (slot.adapter.x, slot.adapter.y)  # type: ignore[attr-defined]

            if self.state is not None:
                now = time.time()
                turbo = self.state.controls.turbo
                thumb_iv = THUMB_INTERVAL_TURBO if turbo else THUMB_INTERVAL
                if now - last_watch >= WATCH_INTERVAL:
                    encode_thumbs = now - last_thumb >= thumb_iv
                    self._publish_frames(encode_thumbs)
                    last_watch = now
                    if encode_thumbs:
                        last_thumb = now
                if now - last_stats >= STATS_INTERVAL:
                    self._publish_stats(
                        [s.adapter.status for s in self.slots],
                        [s.adapter.fitness() for s in self.slots],
                        self._steps_per_sec(),
                    )
                    last_stats = now
                if not turbo or manual_idx >= 0:  # manual play is always real-time
                    clock.tick(60)
            # headless without server: no pacing, run flat out

        return [s.adapter.fitness() for s in self.slots]

    def run(self, max_gens: int | None = None, verbose: bool = True) -> None:
        while max_gens is None or self.pop.generation < max_gens:
            t0 = time.time()
            fitnesses = self.run_generation()
            statuses = [s.adapter.status for s in self.slots]
            env_rows = []
            for i, slot in enumerate(self.slots):
                rec = slot.adapter.episode_stats()
                rec.update({"env": i, "fit": round(fitnesses[i], 1),
                            "status": statuses[i], "frames": slot.frames})
                env_rows.append(rec)
            self._append_episode_csv(env_rows)
            # Post-first-win annealing: once this level is solved, drop mutation so the
            # population exploits the winning lineage instead of re-exploring forever.
            if (statuses.count("WON") > 0 and not self.pop.annealed
                    and self.cfg.anneal_factor not in (0.0, 1.0)):
                self.cfg.mutation_rate *= self.cfg.anneal_factor
                self.cfg.mutation_sigma *= self.cfg.anneal_factor
                self.pop.annealed = True
                if verbose:
                    print(f"first win on [{self.level or 'auto'}] — mutation annealed to "
                          f"{self.cfg.mutation_rate:.3f}/{self.cfg.mutation_sigma:.2f}", flush=True)
            prev_best = self.pop.best_fitness
            self.pop.evolve(fitnesses)
            if self.pop.best_fitness > prev_best:
                self.pop.best_level = self.level  # replay defaults to the level the record was set on
            fits = np.asarray(fitnesses)
            self.pop.history[-1].update({
                "gen": self.pop.generation,
                "level": self.level or "auto",
                "persona": self.persona.name,
                "median": round(float(np.median(fits)), 1),
                "min": round(float(fits.min()), 1),
                "wins": statuses.count("WON"),
                "stuck": statuses.count("STUCK"),
                "dead": statuses.count("DEAD"),
                "best_x": max(r["x"] for r in env_rows),
                "avg_score": round(sum(r["score"] for r in env_rows) / len(env_rows), 1),
                "coins": sum(r["coins"] for r in env_rows),
                "frames": sum(r["frames"] for r in env_rows),
                "duration": round(time.time() - t0, 1),
                "envs": env_rows,
            })
            self.pop.save(self.run_dir)
            h = self.pop.history[-1]
            if verbose:
                print(f"gen {self.pop.generation:4d}  [{self.level or 'auto'}]  "
                      f"best {h['best']:8.1f}  avg {h['avg']:8.1f}  "
                      f"all-time {self.pop.best_fitness:8.1f}  wins {h['wins']}  ({h['duration']}s)",
                      flush=True)
            if self.state is not None:
                self._publish_stats(statuses, fitnesses, 0.0)

            # Re-read the enabled-level list each generation so levels added or
            # toggled in game_config.yaml appear live in the dropdown/curriculum.
            if self.levels is not None:
                try:
                    fresh = list_levels(self.game)
                    if fresh and fresh != self.levels:
                        self.levels = fresh
                        if verbose:
                            print(f"level list refreshed: {', '.join(fresh)}", flush=True)
                except Exception:
                    pass  # a half-saved config edit shouldn't kill training

            # Dashboard level pick wins over the curriculum, applied at the gen boundary.
            requested = self.state.controls.level_request if self.state else None
            if requested and self.levels and requested in self.levels and requested != self.level:
                self.level = requested
                for slot in self.slots:
                    slot.adapter.set_level(self.level)
                self.cfg.mutation_rate, self.cfg.mutation_sigma = self._base_mutation
                self.pop.annealed = False
                self.state.controls.level_request = None
                if verbose:
                    print(f"level switched to [{self.level}] from the dashboard "
                          f"at gen {self.pop.generation}", flush=True)
            # Curriculum: enough winners this generation -> move the whole
            # population to the next enabled level (nets carry over).
            elif (self.levels and h["wins"] >= self.cfg.advance_wins
                    and self.level in self.levels  # current level may have been toggled off
                    and self.levels.index(self.level) < len(self.levels) - 1):
                self.level = self.levels[self.levels.index(self.level) + 1]
                for slot in self.slots:
                    slot.adapter.set_level(self.level)
                # fresh unsolved level -> back to full exploration
                self.cfg.mutation_rate, self.cfg.mutation_sigma = self._base_mutation
                self.pop.annealed = False
                if verbose:
                    print(f"curriculum: {h['wins']} wins -> advancing to level "
                          f"[{self.level}] at gen {self.pop.generation}", flush=True)
        if verbose:
            print(format_results(self.pop.history), flush=True)


def format_results(history: list[dict], tail: int | None = None) -> str:
    """Aligned per-generation results table (replaces the old CSV logs)."""
    rows = history[-tail:] if tail else history
    cols = [("GEN", "gen", 4), ("LEVEL", "level", 12), ("BEST", "best", 9), ("AVG", "avg", 9),
            ("MEDIAN", "median", 9), ("MIN", "min", 8), ("WINS", "wins", 4), ("STUCK", "stuck", 5),
            ("DEAD", "dead", 4), ("BEST X", "best_x", 8), ("AVG SCORE", "avg_score", 9),
            ("COINS", "coins", 5), ("DUR S", "duration", 6)]
    def cell(v) -> str:
        return f"{v:.1f}" if isinstance(v, float) else str(v)

    out = ["  ".join(h.rjust(w) for h, _, w in cols)]
    for r in rows:
        out.append("  ".join(cell(r.get(k, "-")).rjust(w) for _, k, w in cols))
    return "\n".join(out)


def replay(path: str, game: str, level: str | None, state: SharedState | None) -> None:
    if not os.path.exists(path):
        print(f"replay: '{path}' not found — train first, or pass a valid best.npz", flush=True)
        return
    persona = PERSONAS["experienced"]
    net_cfg: dict = {}  # sensors / hidden / action_feedback / memory the run was trained with
    state_path = os.path.join(os.path.dirname(path), "state.json")
    if os.path.exists(state_path):
        with open(state_path, encoding="utf-8") as f:
            meta = json.load(f)
        persona = PERSONAS.get(meta.get("persona") or "", persona)
        saved = meta.get("config") or {}
        net_cfg = {k: saved[k] for k in ("sensors", "hidden", "action_feedback", "memory") if k in saved}
        if level is None and meta.get("best_level"):  # default to the record's level
            level = meta["best_level"]
            print(f"replaying on level [{level}] as [{persona.name}] "
                  f"(where the record was set)", flush=True)
    weights = np.load(path)["weights"]
    cfg = GAConfig(pop_size=1, **net_cfg)
    sensors = cfg.sensors
    net = make_net(cfg)
    net.set_weights(weights.astype(np.float32))
    adapter = make_adapter(game, level, cfg.max_frames, cfg.win_bonus,
                           sprint=persona.sprint, time_rate=persona.time_rate)
    clock = pygame.time.Clock()
    trainer = Trainer(game, level, cfg, run_dir="runs/_replay", state=state)
    trainer.slots[0].adapter = adapter
    trainer.slots[0].net = net
    all_levels = list_levels(game)
    cur_level = level if level is not None else (all_levels[0] if all_levels else None)
    while True:
        adapter.reset()
        net.reset()
        while adapter.alive:
            vec, rays, tiles = read_sensors(adapter, sensors)
            slot0 = trainer.slots[0]
            slot0.last_sensors, slot0.last_rays, slot0.last_tiles = vec, rays, tiles
            move_x, jump = net.act(vec)
            adapter.step(move_x, jump)
            if state is not None:
                trainer._publish_frames(encode_thumbs=True)
                trainer._publish_stats([adapter.status], [adapter.fitness()], 60.0)
            clock.tick(60)
        print(f"replay episode done: {adapter.status}, fitness {adapter.fitness():.1f}", flush=True)
        # A win advances the replay to the next enabled level (wrapping), like the game would.
        if getattr(adapter, "won", False) and all_levels and cur_level in all_levels:
            cur_level = all_levels[(all_levels.index(cur_level) + 1) % len(all_levels)]
            adapter.set_level(cur_level)
            trainer.level = cur_level  # dashboard header follows
            print(f"replay: level cleared — advancing to [{cur_level}]", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Neuroevolution trainer")
    ap.add_argument("--game", default="mario")
    ap.add_argument("--level", default=None, help="level name from game_config.yaml (default: first)")
    ap.add_argument("--gens", type=int, default=None, help="stop after N generations (default: run forever)")
    ap.add_argument("--turbo", action="store_true", help="start in max-speed mode")
    ap.add_argument("--no-serve", action="store_true", help="disable the browser dashboard")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--run-dir", default=None, help="checkpoint dir (default: runs/<game>)")
    ap.add_argument("--resume", default=None, help="resume from a run dir")
    ap.add_argument("--replay", default=None, help="path to a best.npz to watch")
    ap.add_argument("--results", default=None, help="print the results table for a run dir and exit")
    ap.add_argument("--persona", default="experienced", choices=sorted(PERSONAS),
                    help="player type the agents imitate")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sensors", default="rays", choices=SENSOR_MODES,
                    help="exteroception: 6 rays + probes (14 inputs) or a 3x11x11 tile grid (368)")
    ap.add_argument("--hidden", type=int, default=16, help="hidden tanh units")
    ap.add_argument("--action-feedback", action="store_true",
                    help="feed the previous action (move, jump) back in as 2 extra inputs")
    ap.add_argument("--memory", type=int, default=0,
                    help="Jordan memory units: extra outputs looped back as inputs next frame")
    args = ap.parse_args()

    if args.results:
        print(format_results(Population.load(args.results).history))
        return

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    state: SharedState | None = None
    if not args.no_serve:
        from .server import start_server
        state = SharedState(Controls(turbo=args.turbo))
        start_server(state, http_port=args.port)
        print(f"dashboard: http://127.0.0.1:{args.port}/mario/index.html", flush=True)

    if args.replay:
        replay(args.replay, args.game, args.level, state)
        return

    run_dir = args.run_dir or args.resume or os.path.join("runs", args.game)
    if args.resume:
        pop = Population.load(args.resume)
        cfg = pop.cfg
        print(f"resumed {args.resume} at gen {pop.generation}", flush=True)
    else:
        pop = None
        cfg = GAConfig(seed=args.seed, sensors=args.sensors, hidden=args.hidden,
                       action_feedback=args.action_feedback, memory=args.memory)

    if args.level is not None:
        from .adapters import validate_level
        validate_level(args.game, args.level)
    trainer = Trainer(args.game, args.level, cfg, run_dir, state=state, population=pop,
                      persona=get_persona(args.persona))
    trainer.run(args.gens)


if __name__ == "__main__":
    main()
