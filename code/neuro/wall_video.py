"""The replay wall, recorded: one looping WebM per game and persona for the Command Center.

    python -m code.neuro.wall_video                                  # every game and persona -> runs/videos/
    python -m code.neuro.wall_video --game megaman --persona novice

Each video is one pass of the live wall's camera (code/neuro/cameras.py): the persona's agents play
every level once, in level order, with the paper seed that trained best on it, and the result stays
on screen for a second. Mega Man plays the Cut Man stage as a relay: the hand-picked legs for
experienced, each section's best retrained run for the other personas. The page loops the video, so
the wall never stops. Recording refuses the ripped Mega Man art (megaman-data/), which must never
reach the public site; without it the game draws its own flat shapes.
"""
from __future__ import annotations

import argparse
import os
import subprocess
from typing import Iterable, Iterator

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from . import cameras

VIDEOS_DIR = os.path.join("runs", "videos")
GAMES = ["mario", "meatboy", "bomberman", "megaman"]
RIPPED = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "megaman-data")


def frames(cam, max_frames: int = 5400) -> Iterator[pygame.Surface]:
    """One picture per camera step until the camera has finished one pass (2D: an attempt at each
    of its levels; Mega Man: one stage attempt) and its result hold is over, or `max_frames`.
    The same surface is yielded each time: use it before taking the next."""
    target = len(cam.models) if hasattr(cam, "models") else 1
    surf = None
    for _ in range(max_frames):
        cam.step()
        core = cam.adapter.core
        if surf is None or surf.get_size() != (core.WIDTH, core.HEIGHT):
            surf = pygame.Surface((core.WIDTH, core.HEIGHT))
        cam.adapter.render(surf)
        yield surf
        if cam.attempts >= target and not cam.hold_left:
            return


def write_webm(surfaces: Iterable[pygame.Surface], path: str, fps: int = cameras.FPS, width: int = 480) -> int:
    """Pipe the pictures to ffmpeg as a VP9 WebM `width` px wide (pixels kept sharp); the frame count."""
    proc, n = None, 0
    try:
        for surf in surfaces:
            if proc is None:
                w, h = surf.get_size()
                proc = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                         "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
                                         "-vf", f"scale={width}:-2:flags=neighbor", "-c:v", "libvpx-vp9", "-b:v", "0",
                                         "-crf", "40", "-pix_fmt", "yuv420p", "-an", path], stdin=subprocess.PIPE)
            proc.stdin.write(pygame.image.tobytes(surf, "RGB"))
            n += 1
    finally:
        if proc is not None:
            proc.stdin.close()
            if proc.wait():
                raise SystemExit(f"ffmpeg failed writing {path}")
    return n


def camera(game: str, persona: str):
    """The live wall's camera for this game and persona, or None when it has no agents."""
    if game == "megaman":
        from .ast_eval import persona_legs
        if persona == "experienced":
            return cameras.MegamanCamera(os.path.join("runs", "megaman", "relay.json"))
        legs = persona_legs(persona)
        return cameras.MegamanCamera(legs=legs) if legs else None
    models = cameras.pick_models(game, persona)
    return cameras.Camera(game, persona, models) if models else None


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Record the replay wall: one looping WebM per game and persona")
    ap.add_argument("--game", nargs="*", default=GAMES, choices=GAMES)
    ap.add_argument("--persona", nargs="*", default=cameras.ORDER, choices=cameras.ORDER)
    ap.add_argument("--out", default=VIDEOS_DIR)
    ap.add_argument("--max-frames", type=int, default=5400, help="cap per video (30 fps: 5400 = 3 minutes)")
    args = ap.parse_args(argv)
    if os.environ.get("PEAK_MM1_ASSETS") or os.path.isdir(RIPPED):
        raise SystemExit("refusing to record: the ripped Mega Man art (megaman-data/ or PEAK_MM1_ASSETS) would end up "
                         "on the public site; unset PEAK_MM1_ASSETS and move megaman-data/ away first")
    pygame.init()
    for game in args.game:
        for persona in args.persona:
            cam = camera(game, persona)
            if cam is None:
                print(f"skip {game} {persona}: no trained agents", flush=True)
                continue
            path = os.path.join(args.out, game, f"{persona}.webm")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            part = path[:-len(".webm")] + ".part.webm"   # a half-written video never replaces a good one
            n = write_webm(frames(cam, args.max_frames), part)
            os.replace(part, path)
            print(f"wrote {path} ({n} frames, {n / cameras.FPS:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
