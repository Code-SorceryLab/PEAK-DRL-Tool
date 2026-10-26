"""The PEAK Command Center (code/neuro/command_center.py and the pages it gathers): only the paper
seeds reach it, every game gets every tab, and the site is static."""
import os
import re
import shutil
import subprocess

import pytest

from code.neuro import balance, cameras, report
from code.stats import ast_metrics, ast_page

PAPER = {"1234", "2025", "31337"}


def test_reports_keep_only_the_seeds_asked_for(tmp_path):
    for seed in ("1234", "8"):                  # a paper seed and one of the stray seeds committed on 2026-09-21
        src = os.path.join(balance.PROBES_ROOT, "mario", "novice", "p10g40", f"Mario1-1_{seed}", "state.json")
        dst = tmp_path / "mario" / "novice" / "p10g40" / f"Mario1-1_{seed}"
        dst.mkdir(parents=True)
        shutil.copy(src, dst / "state.json")
    cells = balance.load_probe_cells("mario", "novice", "p10g40", str(tmp_path), seeds=balance.PAPER_SEEDS)
    assert [c["seed"] for c in cells["Mario1-1"]] == [1234]
    assert {c["seed"] for c in balance.load_probe_cells("mario", "novice", "p10g40", str(tmp_path))["Mario1-1"]} == {8, 1234}


def test_the_wall_and_the_route_traces_skip_extra_seeds():
    # on the committed probes: Bomberman experienced level 1, seed 8 wins 57 %, the paper seeds 0 %
    models = dict(cameras.pick_models("bomberman", "experienced"))
    assert models and all(d.rsplit("_", 1)[1] in PAPER for d in models.values())
    routes, best = report._collect_routes("bomberman", "experienced", "1", "p10g40")
    assert best is None or best["seed"] in PAPER


def test_mega_man_has_a_drawn_icon_not_a_letter():
    html = report._game_icon("megaman")
    assert "<img" in html and "svg" in html and "gletter" not in html


def test_no_page_offers_a_watch_replay():
    assert not hasattr(report, "serve") and not hasattr(report, "_watch_cmd")


def test_the_old_name_is_gone():
    hits = subprocess.run(["git", "grep", "-l", "-i", "balance command", "--", "code", "menu.py", "README.md", "docs",
                           ":!code/tests/test_command_center.py"], capture_output=True, text=True).stdout.split()
    assert [h for h in hits if not h.startswith("docs/superpowers/")] == []


GAMES = ["mario", "meatboy", "bomberman", "megaman"]


def test_every_game_gets_its_four_sub_pages_without_a_rebuild(monkeypatch):
    from code.neuro import gasweep
    monkeypatch.setattr(balance, "rebuild", lambda *a, **k: pytest.fail("rebuild called"))
    monkeypatch.setattr(gasweep, "rebuild", lambda *a, **k: pytest.fail("gasweep rebuild called"))
    data = report.load_data()
    for g in GAMES:
        for page in (report.balance_page(g, data), report.gasweep_page(g, data), report.ablation_page(g, data)):
            assert page.startswith("<!doctype html>") and "<nav" not in page
            assert not re.search(r'href="(report|mutations|gasweep|ablation|instructions)\.html|"/wall"|"/watch', page)
    assert "<nav" not in report.instructions_page(data)


def test_mega_man_shows_its_sections_and_says_what_is_not_run():
    data = report.load_data()
    assert all(f"CutMan-{i}-" in report.balance_page("megaman", data) for i in range(1, 9))
    assert "No sensor probes yet" in report.ablation_page("megaman", data)
    if not any(key[0] == "megaman" for key in data["gasweep"]):    # until jobs 7434840/7434841 come home
        assert "No GA sweep yet" in report.gasweep_page("megaman", data)


def test_the_mutations_sub_page_keeps_the_totals_for_every_game():
    res = {g: ast_metrics.evaluate(g) for g in GAMES}
    page = ast_page.render(res, only="meatboy", nav=False)
    sc = ast_page.scores(res)
    assert f'{sc["good"]}</span>' in page and 'id="meatboy"' in page and 'id="mario"' not in page and "<nav" not in page


def test_the_site_has_every_tab_of_every_game(tmp_path):
    from code.neuro import command_center
    (tmp_path / "videos" / "mario").mkdir(parents=True)
    (tmp_path / "videos" / "mario" / "novice.webm").write_bytes(b"x")
    written = command_center.build_site(str(tmp_path / "site"), str(tmp_path / "videos"))
    names = {os.path.relpath(p, tmp_path / "site") for p in written}
    assert {"index.html", "howitworks.html", ".nojekyll", "videos/mario/novice.webm"} <= names
    assert {f"{g}-{t}.html" for g in command_center.GAMES for t in ("balance", "mutations", "gasweep", "ablation")} <= names


def test_the_wall_says_which_videos_are_missing():
    from code.neuro import command_center
    html = command_center.shell({"mario": {"novice": True, "clumsy": False}})
    assert 'data-src="videos/mario/novice.webm"' in html and "not recorded yet" in html
    assert "PEAK Command Center" in html and "Balance Command" not in html
    assert "loop" in html and "muted" in html and "playsinline" in html


class FakeCam:
    """Camera's surface: `models` (one per level), `attempts`, `hold_left`, `step()`, `adapter.render(surf)`
    and `adapter.core.WIDTH/HEIGHT`. Each attempt takes `steps` steps, then `hold` steps of hold."""

    def __init__(self, levels, steps, hold):
        self.models, self.steps, self.hold = [None] * levels, steps, hold
        self.attempts = self.hold_left = self.frame = 0
        core = type("Core", (), {"WIDTH": 64, "HEIGHT": 48})()
        self.adapter = type("Adapter", (), {"core": core, "render": lambda self, surf: surf.fill((200, 30, 20))})()

    def step(self):
        if self.hold_left:
            self.hold_left -= 1
            return
        self.frame += 1
        if self.frame == self.steps:
            self.end_attempt()

    def end_attempt(self):
        self.attempts, self.frame, self.hold_left = self.attempts + 1, 0, self.hold


def test_a_video_stops_after_one_pass_and_its_hold():
    from code.neuro import wall_video
    assert sum(1 for _ in wall_video.frames(FakeCam(levels=2, steps=3, hold=2))) == 2 * (3 + 2)
    assert sum(1 for _ in wall_video.frames(FakeCam(levels=2, steps=10_000, hold=2), max_frames=50)) == 50


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_webm_frames_are_written(tmp_path):
    import pygame
    from code.neuro import wall_video
    surfaces = [pygame.Surface((64, 48)) for _ in range(10)]
    assert wall_video.write_webm(surfaces, str(tmp_path / "v.webm")) == 10
    assert (tmp_path / "v.webm").stat().st_size > 0


def test_recording_never_uses_ripped_art(monkeypatch):
    from code.neuro import wall_video
    monkeypatch.setenv("PEAK_MM1_ASSETS", "/somewhere")
    with pytest.raises(SystemExit, match="megaman-data"):
        wall_video.main(["--game", "megaman"])


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def tmp_repo(tmp_path):
    """A repo with committed results, a one-commit gh-pages (the old landing page), a built site
    holding a macOS ._ file, and a bare origin."""
    repo, origin = tmp_path / "repo", tmp_path / "origin.git"
    git(tmp_path, "init", "--bare", "-q", str(origin))
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.email", "t@example.com")
    git(repo, "config", "user.name", "T")
    (repo / "runs" / "balance").mkdir(parents=True)
    (repo / "runs" / "balance" / "report_x.json").write_text("{}")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "results")
    git(repo, "switch", "-q", "--orphan", "gh-pages")
    (repo / "landing.html").write_text("old")
    git(repo, "add", "landing.html")
    git(repo, "commit", "-q", "-m", "landing page")
    git(repo, "switch", "-q", "main")
    git(repo, "remote", "add", "origin", str(origin))
    (repo / "site" / "videos").mkdir(parents=True)
    (repo / "site" / "index.html").write_text("new")
    (repo / "site" / "._index.html").write_text("macOS metadata")
    return repo


def test_publish_refuses_uncommitted_results(tmp_repo):
    from code.neuro import command_center
    (tmp_repo / "runs" / "balance" / "report_x.json").write_text("changed")
    with pytest.raises(SystemExit, match="uncommitted"):
        command_center.publish(str(tmp_repo / "site"), repo=str(tmp_repo))


def test_publish_makes_one_commit_and_keeps_the_old_page_as_a_tag(tmp_repo):
    from code.neuro import command_center
    old = git(tmp_repo, "rev-parse", "gh-pages")
    command_center.publish(str(tmp_repo / "site"), repo=str(tmp_repo))
    assert git(tmp_repo, "rev-list", "--count", "gh-pages") == "1"
    assert git(tmp_repo, "ls-tree", "--name-only", "gh-pages").split() == ["index.html"]     # no ._ file
    assert git(tmp_repo, "rev-parse", "landing-page-2026-03^{commit}") == old
    command_center.publish(str(tmp_repo / "site"), repo=str(tmp_repo), push=True)          # a second publish
    assert git(tmp_repo, "rev-parse", "landing-page-2026-03^{commit}") == old
    assert git(tmp_repo, "ls-remote", "origin", "refs/heads/gh-pages").split()[0] == git(tmp_repo, "rev-parse", "gh-pages")
    assert git(tmp_repo, "ls-remote", "origin", "refs/tags/landing-page-2026-03").split()[0] == old


def test_the_menu_lists_the_command_center_and_its_two_new_items():
    import menu
    assert menu.DISPATCH["12"][0] == "command_center"
    assert menu.DISPATCH["22"][0] == "record_wall" and menu.DISPATCH["23"][0] == "publish_site"


# ── the final review's fixes ─────────────────────────────────────────────────────────────────────
def test_publish_ignores_macos_metadata_in_the_results(tmp_repo):
    from code.neuro import command_center
    (tmp_repo / "runs" / "balance" / "._report_x.json").write_text("macOS metadata")   # exFAT writes these
    command_center.publish(str(tmp_repo / "site"), repo=str(tmp_repo))
    assert git(tmp_repo, "rev-list", "--count", "gh-pages") == "1"


def test_publish_keeps_the_live_landing_page_even_without_a_local_copy(tmp_repo):
    from code.neuro import command_center
    old = git(tmp_repo, "rev-parse", "gh-pages")
    git(tmp_repo, "push", "-q", "origin", "gh-pages")                  # the live site
    git(tmp_repo, "branch", "-q", "-D", "gh-pages")                    # a clone that never fetched it
    git(tmp_repo, "update-ref", "-d", "refs/remotes/origin/gh-pages")
    command_center.publish(str(tmp_repo / "site"), repo=str(tmp_repo), push=True)
    assert git(tmp_repo, "ls-remote", "origin", "refs/tags/landing-page-2026-03").split()[0] == old


def test_a_long_pass_still_shows_every_level():
    from code.neuro import wall_video
    cam = FakeCam(levels=3, steps=10_000, hold=2)
    n = sum(1 for _ in wall_video.frames(cam, max_frames=60))
    assert cam.attempts == 3 and n <= 60


def test_a_camera_can_end_an_attempt_as_a_stall(tmp_path):
    from code.tests.test_cameras import _run
    for level in ("Mario1-1", "Mario1-2"):
        _run(str(tmp_path), "mario", "experienced", level, 1234, wins=1)
    cam = cameras.Camera("mario", "experienced", cameras.pick_models("mario", "experienced", str(tmp_path)), hold=2)
    cam.step()
    cam.end_attempt()
    assert (cam.attempts, cam.last, cam.hold_left) == (1, "Stall", 2)
    cam.step()
    cam.step()
    assert cam.level == "Mario1-2"


def test_a_partial_sweep_keeps_every_paper_seed_in_its_report(tmp_path, monkeypatch):
    import json
    import sys
    seen = {}

    def write_report(*a, **k):
        seen.update(k)
        path = tmp_path / "r.json"
        path.write_text(json.dumps({"levels": []}))
        return str(path)
    monkeypatch.setattr(balance, "run_jobs", lambda *a, **k: None)
    monkeypatch.setattr(balance, "write_report", write_report)
    monkeypatch.setattr(sys, "argv", ["balance", "--game", "mario", "--persona", "novice", "--levels", "Mario1-1",
                                      "--seeds", "1234", "--gens", "1", "--out", str(tmp_path)])
    balance.main()
    assert tuple(seen["seeds"]) == balance.PAPER_SEEDS          # not just the one seed this run probed


def test_recording_hides_any_mega_man_art(tmp_path, monkeypatch):
    from code.games import megaman_assets
    from code.neuro import wall_video
    monkeypatch.setenv("PEAK_MM1_ASSETS", "x")
    monkeypatch.delenv("PEAK_MM1_ASSETS")                       # restored (unset) after the test
    (tmp_path / "art" / "sprites").mkdir(parents=True)          # a config pointing at ripped art elsewhere
    wall_video.no_ripped_art()
    assert megaman_assets.Assets.get({"dir": str(tmp_path / "art")}, 1.0) is None


def test_the_experienced_mega_man_video_uses_the_paper_seed_agents(monkeypatch):
    from code.neuro import ast_eval, wall_video
    calls = []
    monkeypatch.setattr(ast_eval, "persona_legs", lambda p, *a, **k: calls.append(p) or ["legs"])
    monkeypatch.setattr(cameras, "MegamanCamera", lambda *a, legs=None, **k: ("camera", legs))
    assert wall_video.camera("megaman", "experienced") == ("camera", ["legs"]) and calls == ["experienced"]


def test_the_sub_page_titles_carry_no_long_dashes():
    data = report.load_data()
    pages = [report.balance_page("mario", data), report.gasweep_page("mario", data), report.ablation_page("mario", data),
             report.instructions_page(data),
             ast_page.render({g: ast_metrics.evaluate(g) for g in GAMES}, only="mario", nav=False)]
    titles = [re.search(r"<title>([^<]*)</title>", p).group(1) for p in pages]
    assert not any("\u2014" in t or "\u2013" in t for t in titles), titles


def test_a_hidden_wall_stays_hidden_whatever_its_display():
    from code.neuro import command_center
    html = command_center.shell({"mario": {"novice": True}, "megaman": {"novice": True}})
    css = re.sub(r"\s+", "", command_center.CSS)
    assert "[hidden]{display:none!important}" in css and ".wall{display:grid" in css    # .wall's grid beat [hidden]
    assert html.count('class="wall') == 2 and html.count("hidden>") >= 2


# ── round 2 ───────────────────────────────────────────────────────────────────────────────────────
def test_the_top_bar_and_tabs_are_the_old_pages_pixel_buttons_with_icons():
    from code.neuro import command_center
    html = command_center.shell({"mario": {"novice": True}})
    assert "<nav" in html and 'class="brand"' in html and ".tab.active" in html      # the old pages' styles
    tabs = re.findall(r'<button class="tab"[^>]*data-tab="(\w+)"[^>]*>(.*?)</button>', html)
    assert [t for t, _ in tabs][:5] == ["wall", "balance", "mutations", "gasweep", "ablation"]
    assert tabs and all('<i class="ico ' in body for _, body in tabs)


def test_the_balance_tab_lists_the_levels_no_agent_trained_on():
    data = report.load_data()
    bomberman, megaman = report.balance_page("bomberman", data), report.balance_page("megaman", data)
    assert all(f'data-untrained="{lv}"' in bomberman for lv in ("12", "13", "14")) and 'data-untrained="8"' not in bomberman
    assert 'data-untrained="CutMan-4-Shaft-Gap"' in megaman and 'data-untrained="CutMan-3-Ledges"' not in megaman
    assert "not trained yet" in bomberman
