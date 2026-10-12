"""The pipeline pages (code/pipeline.py) build from what is on disk, with or without any runs."""
import os

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
