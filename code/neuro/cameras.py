"""Replay wall: every trained game x persona plays live, like a wall of security cameras.

    python -m code.neuro.cameras            # http://127.0.0.1:8010/

Each camera plays the saved full-sweep models of one game and persona (runs/probes/<game>/<persona>/
p10g40/<level>_<seed>/best.npz), level after level in config order, with the best seed of each
level; Mega Man plays the full Cut Man stage as a relay (runs/megaman/relay.json). One process per
game steps its cameras at 30 frames per second and sends small JPEGs; the camera being watched is
sent at full size with its rays and sensor values. The page also shows the balance pages
(runs/balance/) under its tabs.
"""
from __future__ import annotations

import argparse
import base64
import functools
import glob
import json
import multiprocessing as mp
import os
import queue
import re
import threading
import time
import urllib.parse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import pygame

from . import relay
from .adapters import N_INPUTS_BY_GAME, N_OUTPUTS_BY_GAME, list_levels, make_adapter
from .balance import BALANCE_DIR, PAPER_SEEDS, PROBES_ROOT
from .evolution import GAConfig
from .net import make_net
from .personas import PERSONAS, Hands
from .sensors import read_sensors
from .trainer import THUMB_SCALE, _encode

GAMES = ["mario", "meatboy", "bomberman"]
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "theme.css"), encoding="utf-8") as _f:
    COLORS = dict(re.findall(r"--p-(\w+):\s*(#[0-9a-fA-F]{6})", _f.read()))   # the dashboards' persona colours
ORDER = ["novice", "clumsy", "experienced", "speedrunner", "killer", "collector"]
TAG = "p10g40"              # the full sweep's run tag (rays, pop 10, 40 generations)
FPS = 30
THUMB_EVERY = 4             # a small view refreshes every 4th frame (about 7 per second)
STALL = 300                 # end an attempt after 10 s without progress, as the relay does
MEGAMAN_ID = len(GAMES) * len(ORDER)
WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
NET_KEYS = ("sensors", "hidden", "action_feedback", "memory")


def pick_models(game: str, persona: str, root: str = PROBES_ROOT, seeds=PAPER_SEEDS) -> list[tuple[str, str]]:
    """[(level, run dir)] in level order: per level, the seed (of `seeds`) with the best training
    record. A level without a saved model is left out."""
    keep = {str(s) for s in seeds}
    out = []
    for level in list_levels(game, include_disabled=True):  # every level that has trained agents
        dirs = [d for d in sorted(glob.glob(os.path.join(root, game, persona, TAG, f"{level}_*")))
                if d.rsplit("_", 1)[-1] in keep]
        runs = [s for s in map(relay.run_summary, dirs) if s]
        if runs:
            out.append((level, max(runs, key=lambda s: (s["win_rate"], s["best"]))["run"]))
    return out


def _load(run_dir: str, game: str):
    """(net, sensor mode, frame limit, seed) of a run's best genome."""
    with open(os.path.join(run_dir, "state.json"), encoding="utf-8") as f:
        saved = json.load(f).get("config") or {}
    cfg = GAConfig(pop_size=1, **{k: saved[k] for k in NET_KEYS if k in saved})
    cfg.n_outputs = max(cfg.n_outputs, N_OUTPUTS_BY_GAME.get(game, 3))
    if cfg.sensors == "rays":
        cfg.n_inputs = N_INPUTS_BY_GAME.get(game, 0)
    net = make_net(cfg)
    net.set_weights(np.load(os.path.join(run_dir, "best.npz"))["weights"].astype(np.float32))
    return net, cfg.sensors, int(saved.get("max_frames", 3600)), int(saved.get("seed", 0))


class Camera:
    """One game x persona, playing its levels in a loop the way it trained: same reaction time,
    same hands. After an attempt the result stays on screen for `hold` frames."""

    def __init__(self, game: str, persona: str, models: list[tuple[str, str]], hold: int = FPS) -> None:
        self.game, self.persona, self.hold = game, PERSONAS[persona], hold
        self.models = models
        self.nets = {level: _load(d, game) for level, d in models}
        self.i = self.wins = self.attempts = self.hold_left = 0
        self.last = ""
        p = self.persona
        _, _, max_frames, seed = self.nets[self.level]
        self.adapter = make_adapter(game, self.level, max_frames, 5000.0,
                                    sprint=p.sprint, time_rate=p.time_rate, weights=p.weights)
        self.hands = Hands(p, np.random.default_rng(seed))
        self._start()

    @property
    def level(self) -> str:
        return self.models[self.i][0]

    def _start(self) -> None:
        self.adapter.reset()
        self.nets[self.level][0].reset()
        self.hands.left = 0
        self.frame, self.vec, self.rays, self.tiles = 0, None, [], []
        self.best, self.since_gain = self.adapter.fitness(), 0

    def step(self) -> None:
        if self.hold_left:
            self.hold_left -= 1
            if not self.hold_left:
                self._next()
            return
        net, sensors, _, _ = self.nets[self.level]
        if self.vec is None or self.frame % self.persona.sensor_period == 0:
            self.vec, self.rays, self.tiles = read_sensors(self.adapter, sensors)
        self.frame += 1
        self.adapter.step(*self.hands(net.act(self.vec)))
        fit = self.adapter.fitness()
        if fit > self.best + 1e-6:
            self.best, self.since_gain = fit, 0
        else:
            self.since_gain += 1
        if not self.adapter.alive or self.since_gain >= STALL:
            won = bool(getattr(self.adapter, "won", False))
            self.attempts += 1
            self.wins += won
            self.last = "WON" if won else (self.adapter.episode_stats().get("cause") or "Stall")
            if self.hold:
                self.hold_left = self.hold
            else:
                self._next()

    def _next(self) -> None:
        self.i = (self.i + 1) % len(self.models)
        self.adapter.set_level(self.level)
        self._start()


class MegamanCamera:
    """The Cut Man stage played as a relay (one network per section). relay.play runs in its own
    thread and pauses after every frame, so the worker steps it like any other camera."""

    game = "megaman"
    persona = PERSONAS["experienced"]

    def __init__(self, manifest: str, stage: str = "CutMan", hold: int = FPS) -> None:
        self.legs, self.stage, self.hold = relay.load_legs(manifest, stage), stage, hold
        self.wins = self.attempts = self.hold_left = 0
        self.last, self.level = "", self.legs[0].level
        self.adapter, self.rays, self.tiles, self.vec = None, [], [], None
        self._go, self._done = threading.Semaphore(0), threading.Semaphore(0)
        threading.Thread(target=self._run, daemon=True).start()
        self._done.acquire()                            # the first frame

    def _pause(self) -> None:
        self._done.release()
        self._go.acquire()

    def _on_frame(self, adapter, leg, rays, tiles, vec) -> None:
        self.adapter, self.level, self.rays, self.tiles, self.vec = adapter, leg.level, rays, tiles, vec
        self._pause()

    def _run(self) -> None:
        seed = 1234
        while True:
            e = relay.play(self.legs, self.stage, seed, 18000, STALL, on_frame=self._on_frame)
            self.attempts += 1
            self.wins += e["status"] == "WON"
            self.last = "WON" if e["status"] == "WON" else (e.get("cause") or e["status"])
            self.hold_left = self.hold
            seed += 1
            self._pause()

    def step(self) -> None:
        if self.hold_left:
            self.hold_left -= 1
            return
        self._go.release()
        self._done.acquire()


def frame_message(cam, full: bool) -> dict:
    """What the page needs for one view: the picture, and for the watched view the overlay."""
    core = cam.adapter.core
    surf = getattr(cam, "_surf", None)
    if surf is None or surf.get_size() != (core.WIDTH, core.HEIGHT):
        surf = cam._surf = pygame.Surface((core.WIDTH, core.HEIGHT))
    cam.adapter.render(surf)
    msg = {"game": cam.game, "persona": "relay" if cam.game == "megaman" else cam.persona.name,
           "level": cam.level, "status": cam.last if cam.hold_left else "", "wins": cam.wins,
           "attempts": cam.attempts,
           "jpg": base64.b64encode(_encode(surf, 1.0 if full else THUMB_SCALE)).decode("ascii")}
    if full:
        cx, cy = cam.adapter.camera
        msg["rays"] = [[round(x1 - cx, 1), round(y1 - cy, 1), round(x2 - cx, 1), round(y2 - cy, 1), hit]
                       for x1, y1, x2, y2, hit in cam.rays]
        msg["tiles"] = [[round(tx - cx, 1), round(ty - cy, 1), ts, kind] for tx, ty, ts, kind in cam.tiles]
        msg["sensors"] = [round(float(v), 2) for v in (cam.vec if cam.vec is not None else [])]
        msg["sensor_labels"] = getattr(cam.adapter, "SENSOR_LABELS", None)
    return msg


def layout(root: str = PROBES_ROOT, manifest: str = os.path.join("runs", "megaman", "relay.json")) -> list[dict]:
    """Every camera on the wall, with the levels it will play (empty = no model, a grey view)."""
    cams = [{"id": g * len(ORDER) + p, "game": game, "persona": persona,
             "levels": [lvl for lvl, _ in pick_models(game, persona, root)]}
            for g, game in enumerate(GAMES) for p, persona in enumerate(ORDER)]
    mm = [leg.level for leg in relay.stage_legs("CutMan")] if os.path.exists(manifest) else []
    return cams + [{"id": MEGAMAN_ID, "game": "megaman", "persona": "relay", "levels": mm}]


def _worker(game: str, cams: list[dict], out_q, watch, paused, root: str, manifest: str) -> None:
    parent = os.getppid()
    pygame.init()
    live = []
    for c in cams:
        if not c["levels"]:
            continue
        cam = (MegamanCamera(manifest) if game == "megaman"
               else Camera(game, c["persona"], pick_models(game, c["persona"], root)))
        live.append((c["id"], cam))
    tick = 0
    while live and os.getppid() == parent:            # the server is gone (even killed): stop, do not orphan
        t0 = time.perf_counter()
        for _, cam in live:
            cam.step()
        if not paused.value:
            w = watch.value
            for cid, cam in live:
                full = cid == w
                if full or (tick + cid) % THUMB_EVERY == 0:
                    msg = frame_message(cam, full)
                    msg.update(id=cid, full=full)
                    try:
                        out_q.put_nowait(msg)
                    except queue.Full:
                        pass                            # the page is behind: drop, the next frame replaces it
        tick += 1
        time.sleep(max(0.0, 1.0 / FPS - (time.perf_counter() - t0)))


class Wall:
    """The worker processes, the latest message per camera, and the watched / paused flags."""

    def __init__(self, root: str = PROBES_ROOT, manifest: str = os.path.join("runs", "megaman", "relay.json")) -> None:
        self.root, self.manifest = root, manifest
        self.cams = layout(root, manifest)
        self.watch, self.paused = mp.Value("i", -1), mp.Value("b", 0)
        self.out_q = mp.Queue(maxsize=400)
        self.latest: dict[int, dict] = {}
        self.version = 0
        self.lock = threading.Lock()
        self.procs: dict[str, tuple[mp.Process, int]] = {}
        for game in GAMES + ["megaman"]:
            self._spawn(game, 0)
        threading.Thread(target=self._collect, daemon=True).start()
        threading.Thread(target=self._guard, daemon=True).start()

    def _spawn(self, game: str, restarts: int) -> None:
        cams = [c for c in self.cams if c["game"] == game]
        p = mp.Process(target=_worker, args=(game, cams, self.out_q, self.watch, self.paused, self.root, self.manifest),
                       daemon=True, name=f"cameras-{game}")
        p.start()
        self.procs[game] = (p, restarts)

    def _guard(self) -> None:
        while True:
            time.sleep(2)
            for game, (p, restarts) in list(self.procs.items()):
                if p.is_alive() or p.exitcode == 0:
                    continue
                if restarts < 3:
                    print(f"cameras: the {game} worker stopped (exit {p.exitcode}), restarting", flush=True)
                    self._spawn(game, restarts + 1)
                else:
                    print(f"cameras: the {game} worker failed 4 times; its views stay offline", flush=True)
                    self.procs.pop(game)

    def _collect(self) -> None:
        while True:
            msg = self.out_q.get()
            with self.lock:
                self.version += 1
                msg["v"] = self.version
                if msg.pop("full"):
                    self.latest[-1] = msg                     # the large view
                    msg = {k: v for k, v in msg.items() if k not in ("rays", "tiles", "sensors", "sensor_labels")}
                self.latest[msg["id"]] = msg

    def since(self, v: int) -> list[dict]:
        with self.lock:
            return [m for m in self.latest.values() if m["v"] > v]


class _Handler(SimpleHTTPRequestHandler):
    """/ -> web/cameras.html, /balance/* -> runs/balance/*, anything else from web/."""

    def translate_path(self, path: str) -> str:
        clean = path.split("?", 1)[0].split("#", 1)[0]
        if clean.startswith("/balance/"):
            root = os.path.abspath(BALANCE_DIR)
            target = os.path.abspath(os.path.join(root, urllib.parse.unquote(clean[len("/balance/"):])))
            return target if target.startswith(root + os.sep) else os.path.join(root, "404")   # no ../ out of it
        return super().translate_path("/cameras.html" if clean in ("", "/") else path)

    def do_GET(self) -> None:
        if self.path.split("?", 1)[0] == "/wall":     # the balance pages' "Replay wall" link, clicked inside a tab
            self.send_response(302)
            self.send_header("Location", "/")
            self.end_headers()
            return
        super().do_GET()

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")   # a reload always shows the current page and reports
        super().end_headers()

    def log_message(self, *args) -> None:
        pass


async def _serve_ws(wall: Wall, port: int) -> None:
    import asyncio

    import websockets

    async def handler(ws) -> None:
        await ws.send(json.dumps({"type": "layout", "cams": wall.cams, "colors": COLORS}))
        seen = 0

        async def recv() -> None:
            async for raw in ws:
                try:
                    m = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if m.get("cmd") == "watch":
                    wall.watch.value = int(m.get("id", -1))
                elif m.get("cmd") == "close":
                    wall.watch.value = -1
                elif m.get("cmd") == "pause":
                    wall.paused.value = int(bool(m.get("on")))

        task = asyncio.create_task(recv())
        try:
            while not task.done():
                fresh = wall.since(seen)
                if fresh:
                    seen = max(m["v"] for m in fresh)
                    await ws.send(json.dumps({"type": "frames", "frames": fresh}))
                await asyncio.sleep(1.0 / FPS)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            task.cancel()

    import signal

    stop = asyncio.get_running_loop().create_future()
    for sig in (signal.SIGTERM, signal.SIGINT):        # wakes the loop even with no page connected
        try:
            asyncio.get_running_loop().add_signal_handler(sig, lambda: stop.done() or stop.set_result(None))
        except NotImplementedError:                    # Windows: Ctrl+C reaches asyncio.run as KeyboardInterrupt
            break
    async with websockets.serve(handler, "127.0.0.1", port, max_size=None):
        await stop


def main() -> None:
    import asyncio

    ap = argparse.ArgumentParser(description="Replay wall: every trained game x persona, live")
    ap.add_argument("--port", type=int, default=8010, help="HTTP port; the websocket uses port + 765")
    ap.add_argument("--probes", default=PROBES_ROOT)
    ap.add_argument("--manifest", default=os.path.join("runs", "megaman", "relay.json"))
    args = ap.parse_args()
    try:
        http = ThreadingHTTPServer(("127.0.0.1", args.port), functools.partial(_Handler, directory=WEB_DIR))
    except OSError:
        raise SystemExit(f"port {args.port} is in use: pass --port with a free one") from None
    wall = Wall(args.probes, args.manifest)
    threading.Thread(target=http.serve_forever, daemon=True).start()
    n = sum(1 for c in wall.cams if c["levels"])
    print(f"replay wall: {n} of {len(wall.cams)} cameras have models · http://127.0.0.1:{args.port}/   (Ctrl+C to stop)",
          flush=True)
    try:
        asyncio.run(_serve_ws(wall, args.port + 765))
    except OSError:
        raise SystemExit(f"websocket port {args.port + 765} is in use: pass --port with a free one") from None


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nstopped", flush=True)
