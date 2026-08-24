"""Dashboard server: static HTTP for the web UI + a websocket pushing trainer state.

Runs entirely on daemon threads; the trainer thread never blocks on it. The server
only ever reads encoded frame bytes from SharedState — never pygame surfaces.
"""
from __future__ import annotations

import asyncio
import base64
import functools
import json
import os
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import websockets

from .trainer import SharedState

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
PUSH_INTERVAL = 0.1  # s between websocket pushes


class _Handler(SimpleHTTPRequestHandler):
    """Serves web/ and accepts /mario/... (or any /<game>/...) as an alias for /."""

    def translate_path(self, path: str) -> str:
        parts = path.lstrip("/").split("/", 1)
        if len(parts) == 2 and "." not in parts[0]:  # strip a single game-name prefix
            path = "/" + parts[1]
        self.path = path or "/index.html"
        return super().translate_path(self.path)

    def log_message(self, *args) -> None:  # silence per-request stderr spam
        pass


def _run_http(port: int) -> None:
    handler = functools.partial(_Handler, directory=WEB_DIR)
    ThreadingHTTPServer(("127.0.0.1", port), handler).serve_forever()


async def _ws_handler(ws, state: SharedState) -> None:
    async def recv_loop() -> None:
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            cmd = msg.get("cmd")
            if cmd == "watch":
                state.controls.watch_env = int(msg.get("env", 0))
            elif cmd == "turbo":
                state.controls.turbo = bool(msg.get("on", False))
            elif cmd == "sensors":
                state.controls.sensors_on = bool(msg.get("on", True))
            elif cmd == "manual":
                state.controls.manual = bool(msg.get("on", False))
                if not state.controls.manual:
                    state.controls.keys = {"left": False, "right": False, "jump": False}
            elif cmd == "hitboxes":
                state.controls.hitboxes = bool(msg.get("on", False))
            elif cmd == "grid":
                state.controls.grid = bool(msg.get("on", False))
            elif cmd == "level":
                state.controls.level_request = str(msg.get("level", "")) or None
            elif cmd == "keys":
                state.controls.keys = {
                    "left": bool(msg.get("left")),
                    "right": bool(msg.get("right")),
                    "jump": bool(msg.get("jump")),
                    "up": bool(msg.get("up")),
                    "down": bool(msg.get("down")),
                }

    async def send_loop() -> None:
        try:
            while True:
                stats, envs = state.snapshot()
                payload = {"type": "update", "stats": stats, "envs": []}
                for e in envs:
                    out = {k: v for k, v in e.items() if k != "jpg"}
                    if "jpg" in e:
                        out["img"] = base64.b64encode(e["jpg"]).decode("ascii")
                    payload["envs"].append(out)
                await ws.send(json.dumps(payload))
                await asyncio.sleep(PUSH_INTERVAL)
        except websockets.exceptions.ConnectionClosed:
            return  # browser tab closed or reloaded — a normal event, not an error

    state.add_viewer()  # trainer skips frame encoding while nobody is connected
    try:
        recv = asyncio.create_task(recv_loop())
        send = asyncio.create_task(send_loop())
        done, pending = await asyncio.wait({recv, send}, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        for t in done:
            t.exception()  # retrieve so asyncio never logs "exception was never retrieved"
    finally:
        state.remove_viewer()


def _run_ws(state: SharedState, port: int) -> None:
    async def serve() -> None:
        async with websockets.serve(lambda ws: _ws_handler(ws, state), "127.0.0.1", port):
            await asyncio.Future()

    asyncio.run(serve())


def start_server(state: SharedState, http_port: int = 8000, ws_port: int = 8765) -> None:
    threading.Thread(target=_run_http, args=(http_port,), daemon=True).start()
    threading.Thread(target=_run_ws, args=(state, ws_port), daemon=True).start()
