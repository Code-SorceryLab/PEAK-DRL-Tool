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
import io
import os
import threading
import time

import numpy as np
import pygame

from .adapters import GameAdapter, make_adapter
from .evolution import GAConfig, Population
from .net import NeuralNet
from .sensors import read_sensors

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


class SharedState:
    """Thread-safe snapshot bridge between trainer and websocket server."""

    def __init__(self, controls: Controls) -> None:
        self.controls = controls
        self._lock = threading.Lock()
        self._stats: dict = {}
        self._envs: list[dict] = []

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
        self.last_sensors: np.ndarray = np.zeros(14, dtype=np.float32)


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
                 state: SharedState | None = None, population: Population | None = None) -> None:
        self.game = game
        self.level = level
        self.cfg = cfg
        self.run_dir = run_dir
        self.state = state
        self.net_proto = NeuralNet()
        self.pop = population or Population(cfg, self.net_proto.n_params)
        self.slots = [
            EnvSlot(make_adapter(game, level, cfg.max_frames, cfg.win_bonus), NeuralNet())
            for _ in range(cfg.pop_size)
        ]
        self._surfaces: list[pygame.Surface] | None = None
        self._step_accum = 0  # cumulative env-steps, sampled by _steps_per_sec
        self._steps_window: list[tuple[float, int]] = []  # (timestamp, cumulative steps)

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
        self.state.publish(stats={
            "gen": self.pop.generation,
            "all_time_best": round(self.pop.best_fitness if hist else 0.0, 1),
            "last_gen_best": round(hist[-1]["best"], 1) if hist else 0.0,
            "avg_fitness": round(hist[-1]["avg"], 1) if hist else 0.0,
            "elite": self.cfg.elite,
            "mut_rate": self.cfg.mutation_rate,
            "pop_size": self.cfg.pop_size,
            "level": self.level or "auto",
            "game": self.game,
            "sps": round(sps),
            "turbo": self.state.controls.turbo,
            "history": [[round(h["best"], 1), round(h["avg"], 1)] for h in hist[-400:]],
            "live_fitness": [round(f, 1) for f in fitnesses],
            "statuses": statuses,
        })

    def _publish_frames(self, encode_thumbs: bool) -> None:
        if self.state is None:
            return
        watch = self.state.controls.watch_env % len(self.slots)
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
                    if i == watch:
                        entry["sensors"] = [round(float(v), 2) for v in slot.last_sensors]
            envs.append(entry)
        self.state.publish(envs=envs)

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
            slot.adapter.reset()
            slot.frames = 0
            slot.stuck_anchor_x = 0.0
            slot.stuck_frames = 0

        clock = pygame.time.Clock()
        last_thumb = last_watch = last_stats = 0.0

        while any(s.adapter.alive for s in self.slots):
            for slot in self.slots:
                if not slot.adapter.alive:
                    continue
                vec, rays = read_sensors(slot.adapter)
                slot.last_sensors, slot.last_rays = vec, rays
                move_x, jump = slot.net.act(vec)
                slot.adapter.step(move_x, jump)
                slot.frames += 1
                self._step_accum += 1
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
                if not turbo:
                    clock.tick(60)
            # headless without server: no pacing, run flat out

        return [s.adapter.fitness() for s in self.slots]

    def run(self, max_gens: int | None = None) -> None:
        while max_gens is None or self.pop.generation < max_gens:
            t0 = time.time()
            fitnesses = self.run_generation()
            self.pop.evolve(fitnesses)
            self.pop.save(self.run_dir)
            h = self.pop.history[-1]
            print(f"gen {self.pop.generation:4d}  best {h['best']:8.1f}  avg {h['avg']:8.1f}  "
                  f"all-time {self.pop.best_fitness:8.1f}  ({time.time() - t0:.1f}s)", flush=True)
            if self.state is not None:
                self._publish_stats([s.adapter.status for s in self.slots], fitnesses, 0.0)


def replay(path: str, game: str, level: str | None, state: SharedState | None) -> None:
    weights = np.load(path)["weights"]
    cfg = GAConfig(pop_size=1)
    net = NeuralNet()
    net.set_weights(weights.astype(np.float32))
    adapter = make_adapter(game, level, cfg.max_frames, cfg.win_bonus)
    clock = pygame.time.Clock()
    trainer = Trainer(game, level, cfg, run_dir="runs/_replay", state=state)
    trainer.slots[0].adapter = adapter
    trainer.slots[0].net = net
    while True:
        adapter.reset()
        while adapter.alive:
            vec, rays = read_sensors(adapter)
            trainer.slots[0].last_sensors, trainer.slots[0].last_rays = vec, rays
            move_x, jump = net.act(vec)
            adapter.step(move_x, jump)
            if state is not None:
                trainer._publish_frames(encode_thumbs=True)
                trainer._publish_stats([adapter.status], [adapter.fitness()], 60.0)
            clock.tick(60)
        print(f"replay episode done: {adapter.status}, fitness {adapter.fitness():.1f}", flush=True)


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
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

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
        cfg = GAConfig(seed=args.seed)

    trainer = Trainer(args.game, args.level, cfg, run_dir, state=state, population=pop)
    trainer.run(args.gens)


if __name__ == "__main__":
    main()
