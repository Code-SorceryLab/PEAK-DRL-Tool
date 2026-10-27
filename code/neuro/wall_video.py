"""The replay wall, recorded: one looping WebM per game and persona for the Command Center.

    python -m code.neuro.wall_video                                  # every game and persona -> runs/videos/
    python -m code.neuro.wall_video --game megaman --persona novice

Each video is one pass of the live wall's camera (code/neuro/cameras.py): the persona's agents play
every level once, in level order, with the paper seed that trained best on it (an attempt that
outruns its level's share of the video ends as a stall), and the result stays on screen for a second.
Mega Man plays the Cut Man stage as a relay of each section's best paper-seed run. The page loops the
video, so the wall never stops. Recording refuses the ripped Mega Man art (megaman-data/), which must
never reach the public site; without it the game draws its own flat shapes.
"""
from __future__ import annotations

import argparse
import json
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
NO_ART = os.path.join(os.path.dirname(RIPPED), ".no-megaman-art")   # never created: the game finds no art there


def frames(cam, max_frames: int = 5400, track: dict | None = None, width: int = 480) -> Iterator[pygame.Surface]:
    """One picture per camera step until the camera has finished one pass (2D: an attempt at each
    of its levels; Mega Man: an attempt at each of its sections) and its result hold is over, or
    `max_frames`. Each level gets an equal share of the frames: an attempt that runs past its share
    ends as a stall, so every level appears before the video loops. With `track`, also fills
    track["attempts"] ([start frame, level, result] per attempt, for the captions) and, every 2nd
    frame, track["frames"] ([rays relative to the view at `width` px, sensor values], for the large
    view), with track["labels"] naming the sensors.
    The same surface is yielded each time: use it before taking the next."""
    target = len(cam.models) if hasattr(cam, "models") else 1
    share = max(1, max_frames // target - getattr(cam, "hold", 0))
    if track is not None:
        track.update(labels=list(getattr(cam.adapter, "SENSOR_LABELS", None) or []), attempts=[], frames=[])
    surf, played, attempts, opened, cut = None, 0, cam.attempts, False, False
    for i in range(max_frames):
        holding = cam.hold_left                 # a step that only runs down the result hold is not play
        if track is not None and not holding and not opened:
            track["attempts"].append([i, str(cam.level), None])
            opened = True
        cam.step()
        if cam.attempts == attempts and not holding:
            played += 1
            if played >= share and hasattr(cam, "end_attempt"):
                cam.end_attempt()              # a 2D camera counts it now, Mega Man's on its next step
                cut = True
        if cam.attempts != attempts:
            attempts, played = cam.attempts, 0
            if track is not None and opened:          # "cut": the video's share ran out, not the game
                track["attempts"][-1][2] = "cut" if cut else (cam.last or "Stall")
                opened = False
            cut = False
        core = cam.adapter.core
        if surf is None or surf.get_size() != (core.WIDTH, core.HEIGHT):
            surf = pygame.Surface((core.WIDTH, core.HEIGHT))
        cam.adapter.render(surf)
        if track is not None and i % 2 == 0:
            track["frames"].append(_sample(cam, width / core.WIDTH))
        yield surf
        if cam.attempts >= target and not cam.hold_left:
            return


def _sample(cam, scale: float) -> list:
    """[rays as [x1, y1, x2, y2, hit] in video pixels relative to the view, sensor values]."""
    cx, cy = cam.adapter.camera
    rays = [[round((x1 - cx) * scale), round((y1 - cy) * scale), round((x2 - cx) * scale), round((y2 - cy) * scale), int(hit)]
            for x1, y1, x2, y2, hit in (cam.rays or [])]
    return [rays, [round(float(v), 2) for v in (cam.vec if cam.vec is not None else [])]]


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
                                         "-crf", "40", "-deadline", "realtime", "-cpu-used", "8", "-row-mt", "1",
                                         "-pix_fmt", "yuv420p", "-an", path], stdin=subprocess.PIPE)   # realtime: ~10x faster
            proc.stdin.write(pygame.image.tobytes(surf, "RGB"))
            n += 1
    finally:
        if proc is not None:
            proc.stdin.close()
            if proc.wait():
                raise SystemExit(f"ffmpeg failed writing {path}")
    return n


def no_ripped_art() -> None:
    """Refuse while the ripped Mega Man art is around, then point the game's art folder at nowhere,
    so even a config naming another folder (megaman_config.yaml assets.dir) draws flat shapes."""
    if os.environ.get("PEAK_MM1_ASSETS") not in (None, "", NO_ART) or os.path.isdir(RIPPED):
        raise SystemExit("refusing to record: the ripped Mega Man art (megaman-data/ or PEAK_MM1_ASSETS) would end up "
                         "on the public site; unset PEAK_MM1_ASSETS and move megaman-data/ away first")
    os.environ["PEAK_MM1_ASSETS"] = NO_ART


def camera(game: str, persona: str):
    """The live wall's camera for this game and persona, or None when it has no agents."""
    if game == "megaman":
        from .ast_eval import persona_legs
        legs = persona_legs(persona)          # each section's best paper-seed run, every persona (AL: paper seeds only)
        return cameras.MegamanCamera(legs=legs, scopes=[leg.level for leg in legs]) if legs else None
    models = cameras.pick_models(game, persona)
    return cameras.Camera(game, persona, models) if models else None


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Record the replay wall: one looping WebM per game and persona")
    ap.add_argument("--game", nargs="*", default=GAMES, choices=GAMES)
    ap.add_argument("--persona", nargs="*", default=cameras.ORDER, choices=cameras.ORDER)
    ap.add_argument("--out", default=VIDEOS_DIR)
    ap.add_argument("--max-frames", type=int, default=None,
                    help="cap per video (30 fps); default 5400 (3 minutes), Mega Man 7200 (900 per section)")
    args = ap.parse_args(argv)
    no_ripped_art()
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
            track, width = {}, 480
            cap = args.max_frames or (7200 if game == "megaman" else 5400)
            n = write_webm(frames(cam, cap, track, width), part, width=width)
            os.replace(part, path)
            base = path[:-len(".webm")]
            with open(base + ".json", "w", encoding="utf-8") as f:          # the tiles' captions
                json.dump({"fps": cameras.FPS, "width": width, "attempts": track["attempts"]}, f, separators=(",", ":"))
            with open(base + ".sensors.json", "w", encoding="utf-8") as f:  # the large view's rays and bars
                json.dump({"every": 2, "width": width, "labels": track["labels"], "frames": track["frames"]}, f,
                          separators=(",", ":"))
            print(f"wrote {path} ({n} frames, {n / cameras.FPS:.0f} s) with its timeline and sensor track", flush=True)


if __name__ == "__main__":
    main()
