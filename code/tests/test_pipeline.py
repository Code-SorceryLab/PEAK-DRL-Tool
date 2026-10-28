"""The pipeline pages (code/pipeline.py) build from what is on disk, with or without any runs."""
import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("PEAK_MM1_ATTEMPTS", "1")

from code import pipeline


def test_every_page_builds_and_links_the_stages():
    pages = {"hub": pipeline.hub("CutMan"), "design": pipeline.design("CutMan"), "mutate": pipeline.mutate("CutMan"),
             "train": pipeline.train("CutMan"), "play": pipeline.play("CutMan"), "measure": pipeline.measure()}
    for name, page in pages.items():
        assert page.startswith("<!doctype html>"), name
        for key, _ in pipeline.STAGES:                      # every page links every stage, in order
            assert f'href="/{key}"' in page, (name, key)
    assert "CutMan-8-Boss" in pages["design"] and "<svg" in pages["design"]
    assert 'value="human"' in pages["play"] and "Play it yourself" in pages["play"]


def test_a_job_reports_its_last_line_and_stops(tmp_path, monkeypatch):
    import sys
    import time
    monkeypatch.setattr(pipeline, "JOB_LOGS", str(tmp_path))
    jobs = pipeline.Jobs()
    jobs.start("echo", [sys.executable, "-c", "print('first'); print('last')"])
    slow = jobs.start("sleep", [sys.executable, "-c", "import time; time.sleep(60)"], group="g")
    jobs.start("replaces it", [sys.executable, "-c", "pass"], group="g")      # same group: the sleeper is stopped
    assert slow["proc"].poll() is not None
    for _ in range(50):
        view = {j["label"]: j for j in jobs.view()}
        if view["echo"]["status"] == "done" and view["replaces it"]["status"] == "done":
            break
        time.sleep(0.1)
    assert view["echo"]["line"] == "last" and view["sleep"]["status"] == "stopped"


def test_a_second_train_press_gets_its_own_seed_folders(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "RUNS", str(tmp_path))
    monkeypatch.setattr(pipeline, "MANIFEST", str(tmp_path / "relay.json"))
    dirs = []
    jobs = SimpleNamespace(start=lambda label, cmd: dirs.append(cmd[cmd.index("--run-dir") + 1]))
    for _ in range(2):                              # pressed again before any trainer saved its first generation
        pipeline.start_training(jobs, "CutMan-1-Street", 2, 5)
    assert len(set(dirs)) == 4 and all(os.path.isdir(d) for d in dirs)
    assert pipeline.section_runs("CutMan-1-Street") == []          # a claimed, empty folder is no run yet


def test_re_pick_does_not_overwrite_a_malformed_relay_json(tmp_path, monkeypatch):
    bad = tmp_path / "relay.json"
    bad.write_text('{"legs": {"CutMan-2-Climb": ')                # the other sections' picks, cut off
    monkeypatch.setattr(pipeline, "MANIFEST", str(bad))
    monkeypatch.setattr(pipeline, "section_runs", lambda section: [{"run": "r", "win_rate": 1.0, "best": 1.0}])
    monkeypatch.setattr(pipeline.relay, "evaluate", lambda run, section, attempts: (1.0, 1.0))
    with pytest.raises(ValueError):
        pipeline.pick_section("CutMan-1-Street", "CutMan")
    assert bad.read_text() == '{"legs": {"CutMan-2-Climb": '
