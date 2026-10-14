"""Replay wall (code/neuro/cameras.py): which saved model plays each level, the level loop, the frames."""
import json
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np

from code.neuro.cameras import Camera, frame_message, pick_models
from code.neuro.evolution import GAConfig
from code.neuro.net import make_net


def _run(root, game, persona, level, seed, wins, max_frames=3600):
    """A fake probe run: state.json with `wins` per generation and random best.npz weights."""
    d = os.path.join(root, game, persona, "p10g40", f"{level}_{seed}")
    os.makedirs(d)
    cfg = GAConfig(pop_size=1, max_frames=max_frames, seed=seed)
    net = make_net(cfg)
    np.savez(os.path.join(d, "best.npz"), weights=np.random.default_rng(seed).normal(size=net.n_params))
    hist = [{"gen": g, "wins": wins, "best_x": 100.0, "envs": [{}] * 10} for g in range(3)]
    with open(os.path.join(d, "state.json"), "w") as f:
        json.dump({"persona": persona, "best_fitness": 100.0 + wins, "history": hist,
                   "config": {"seed": seed, "sensors": "rays", "hidden": 16, "max_frames": max_frames}}, f)
    return d


def test_each_level_plays_its_best_seed_and_a_level_without_a_model_is_skipped(tmp_path):
    root = str(tmp_path)
    _run(root, "mario", "clumsy", "Mario1-1", 1, wins=1)
    best = _run(root, "mario", "clumsy", "Mario1-1", 2, wins=7)
    _run(root, "mario", "clumsy", "Mario1-1", 3, wins=0)
    assert pick_models("mario", "clumsy", root) == [("Mario1-1", best)]   # Mario1-2 has no model
    assert pick_models("mario", "novice", root) == []


def test_the_camera_loops_through_the_levels(tmp_path):
    root = str(tmp_path)
    for level in ("Mario1-1", "Mario1-2"):
        _run(root, "mario", "experienced", level, 1, wins=1, max_frames=20)
    cam = Camera("mario", "experienced", pick_models("mario", "experienced", root), hold=0)
    seen = [cam.level]
    for _ in range(200):
        cam.step()
        if cam.level != seen[-1]:
            seen.append(cam.level)
    assert seen[:3] == ["Mario1-1", "Mario1-2", "Mario1-1"]
    assert cam.attempts >= 2


def test_a_frame_carries_the_picture_and_for_the_watched_camera_the_rays(tmp_path):
    root = str(tmp_path)
    _run(root, "mario", "good", "Mario1-1", 1, wins=1)
    cam = Camera("mario", "good", pick_models("mario", "good", root))
    for _ in range(5):
        cam.step()
    thumb = frame_message(cam, full=False)
    full = frame_message(cam, full=True)
    assert thumb["jpg"] and thumb["level"] == "Mario1-1" and thumb["persona"] == "good" and "rays" not in thumb
    assert len(full["jpg"]) > len(thumb["jpg"]) and full["rays"] and len(full["sensors"]) > 0


def test_encoding_frames_does_not_keep_them_in_memory():
    import tracemalloc

    import pygame

    from code.neuro.trainer import _encode
    surf = pygame.Surface((320, 240))
    _encode(surf)
    tracemalloc.start()
    before = tracemalloc.get_traced_memory()[0]
    for _ in range(500):
        _encode(surf)
    grown = tracemalloc.get_traced_memory()[0] - before
    tracemalloc.stop()
    assert grown < 200_000          # pygame.image.save into a BytesIO kept every frame: about 5 MB here


def test_the_balance_route_cannot_leave_runs_balance():
    from code.neuro.balance import BALANCE_DIR
    from code.neuro.cameras import _Handler
    h = _Handler.__new__(_Handler)
    h.directory = "."
    root = os.path.abspath(BALANCE_DIR)
    assert h.translate_path("/balance/report.html") == os.path.join(root, "report.html")
    for evil in ("/balance/../../code/neuro/personas.py", "/balance/%2e%2e/%2e%2e/menu.py"):
        assert h.translate_path(evil).startswith(root + os.sep)
