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
    vids = tmp_path / "videos" / "mario"
    vids.mkdir(parents=True)
    (vids / "novice.webm").write_bytes(b"x")
    (vids / "novice.json").write_text('{"fps": 30, "width": 480, "attempts": [[0, "Mario1-1", "Pit"]]}')
    (vids / "novice.sensors.json").write_text('{"every": 2, "width": 480, "labels": ["a"], "frames": [[[], [0.5]]]}')
    written = command_center.build_site(str(tmp_path / "site"), str(tmp_path / "videos"))
    names = {os.path.relpath(p, tmp_path / "site") for p in written}
    assert {"index.html", "howitworks.html", ".nojekyll", "videos/mario/novice.webm"} <= names
    track = (tmp_path / "site" / "videos" / "mario" / "novice.sensors.js").read_text()   # a script: works from file://
    assert track.startswith('PEAK_TRACK("mario/novice",') and '"labels":["a"]' in track
    assert '"Mario1-1"' in (tmp_path / "site" / "index.html").read_text()               # the timeline, embedded
    assert {f"{g}-{t}.html" for g in command_center.GAMES for t in ("balance", "mutations", "gasweep", "ablation", "reproduce")} <= names
    assert {f"agents/{g}.zip" for g in command_center.GAMES} <= names


def test_the_wall_says_which_videos_are_missing():
    from code.neuro import command_center
    html = command_center.shell({"mario": {"novice": True, "clumsy": False}})
    assert 'data-src="videos/mario/novice.webm"' in html and "not recorded yet" in html
    assert "PEAK Command Center" in html and "Balance Command" not in html
    assert "loop" in html and "muted" in html and "playsinline" in html


class FakeCam:
    """Camera's surface: `models` (one per level), `level`, `last`, `attempts`, `hold_left`, `rays`, `vec`,
    `step()`, `end_attempt()`, `adapter.render(surf)`, `adapter.camera` and `adapter.core.WIDTH/HEIGHT`.
    Each attempt takes `steps` steps, then `hold` steps of hold, then the next level."""

    def __init__(self, levels, steps, hold):
        self.models, self.steps, self.hold = [None] * levels, steps, hold
        self.attempts = self.hold_left = self.frame = self.i = 0
        self.last, self.rays, self.vec = "", [(10.0, 20.0, 74.0, 20.0, 1)], [0.5, 0.25]
        core = type("Core", (), {"WIDTH": 64, "HEIGHT": 48})()
        self.adapter = type("Adapter", (), {"core": core, "camera": (2.0, 0.0), "SENSOR_LABELS": ["a", "b"],
                                            "render": lambda self, surf: surf.fill((200, 30, 20))})()

    @property
    def level(self):
        return f"L{self.i % len(self.models)}"

    def step(self):
        if self.hold_left:
            self.hold_left -= 1
            if not self.hold_left:
                self.i += 1
            return
        self.frame += 1
        if self.frame == self.steps:
            self.end_attempt()

    def end_attempt(self):
        self.attempts, self.frame, self.hold_left, self.last = self.attempts + 1, 0, self.hold, "Pit"


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
    calls, leg = [], type("Leg", (), {"level": "CutMan-1-Street"})()
    monkeypatch.setattr(ast_eval, "persona_legs", lambda p, *a, **k: calls.append(p) or [leg])
    monkeypatch.setattr(cameras, "MegamanCamera", lambda *a, legs=None, scopes=None, **k: ("camera", legs, scopes))
    assert wall_video.camera("megaman", "experienced") == ("camera", [leg], ["CutMan-1-Street"]) and calls == ["experienced"]


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


def test_a_recording_keeps_a_timeline_and_a_sensor_track():
    from code.neuro import wall_video
    track = {}
    n = sum(1 for _ in wall_video.frames(FakeCam(levels=2, steps=3, hold=2), track=track))
    assert n == 10 and track["attempts"] == [[0, "L0", "Pit"], [5, "L1", "Pit"]]
    assert len(track["frames"]) == 5 and track["labels"] == ["a", "b"]         # every 2nd frame
    rays, sensors = track["frames"][0]
    assert sensors == [0.5, 0.25] and rays == [[60, 150, 540, 150, 1]]            # view-relative, at 480 px wide


def test_mega_man_plays_one_attempt_per_section():
    from code.neuro import relay, wall_video
    legs = relay.load_legs(os.path.join("runs", "megaman", "relay.json"), "CutMan")     # quick to load; any legs do
    cam = cameras.MegamanCamera(legs=legs, scopes=["CutMan-5-Rooftop", "CutMan-7-Corridor"], hold=5)
    track = {}
    n = sum(1 for _ in wall_video.frames(cam, max_frames=200, track=track))
    assert cam.attempts == 2 and n <= 200
    assert [a[1] for a in track["attempts"]] == ["CutMan-5-Rooftop", "CutMan-7-Corridor"]


def test_the_recorder_writes_the_timeline_and_the_sensor_track_beside_the_video(tmp_path, monkeypatch):
    import json
    from code.neuro import wall_video
    monkeypatch.setenv("PEAK_MM1_ASSETS", "x")
    monkeypatch.delenv("PEAK_MM1_ASSETS")                       # the guard's own setting is undone after the test
    monkeypatch.setattr(wall_video, "camera", lambda game, persona: FakeCam(levels=2, steps=3, hold=2))
    monkeypatch.setattr(wall_video, "write_webm", lambda surfaces, path, **k: (open(path, "wb").close(), sum(1 for _ in surfaces))[1])
    wall_video.main(["--game", "mario", "--persona", "novice", "--out", str(tmp_path)])
    timeline = json.load(open(tmp_path / "mario" / "novice.json"))
    track = json.load(open(tmp_path / "mario" / "novice.sensors.json"))
    assert (tmp_path / "mario" / "novice.webm").exists() and timeline["fps"] == 30 and timeline["width"] == 480
    assert timeline["attempts"] == [[0, "L0", "Pit"], [5, "L1", "Pit"]]
    assert track["every"] == 2 and track["labels"] == ["a", "b"] and len(track["frames"]) == 5



def test_recording_twice_in_one_process_is_allowed(monkeypatch):
    from code.neuro import wall_video
    monkeypatch.setenv("PEAK_MM1_ASSETS", "x")
    monkeypatch.delenv("PEAK_MM1_ASSETS")
    wall_video.no_ripped_art()
    wall_video.no_ripped_art()                                  # its own setting is not ripped art



def test_the_wall_has_captions_and_a_large_view_with_rays_and_sensor_bars():
    from code.neuro import command_center
    html = command_center.shell({"mario": {"novice": {"fps": 30, "width": 480, "attempts": [[0, "Mario1-1", "WON"]]},
                                           "clumsy": True}})
    assert html.count('class="cap"') == 2 and '<dialog id="big"' in html and '<canvas class="rays"' in html
    assert 'class="sensors"' in html and "PEAK_TRACK" in html and "sensors.js" in html


def test_the_new_modules_compile_without_warnings():
    import warnings
    for path in ("code/neuro/command_center.py", "code/neuro/wall_video.py"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            compile(open(path, encoding="utf-8").read(), path, "exec")
        assert not [w for w in caught if issubclass(w.category, SyntaxWarning)], path


def test_hosted_agents_load_from_their_folder_and_skip_stray_seeds(tmp_path):
    import json
    from code.neuro import command_center
    agents = command_center.host_agents(str(tmp_path / "agents"), games=["mario"])
    d = tmp_path / "agents" / "probes" / "mario" / "novice" / "p10g40" / "Mario1-1_1234"
    st = json.load(open(d / "state.json"))
    assert (d / "best.npz").exists() and "history" not in st and {"persona", "best_level", "config"} <= set(st)
    assert "seed" in st["config"] and set(st["config"]) <= {"seed", "sensors", "hidden", "action_feedback", "memory",
                                                            "pop_size", "max_frames"}   # what trainer.replay reads, and the budget
    assert not (tmp_path / "agents" / "probes" / "mario" / "novice" / "p10g40" / "Mario1-1_8").exists()
    assert (tmp_path / "agents" / "mario.zip").stat().st_size > 0
    assert agents and all(str(a["seed"]) in PAPER for a in agents)
    cam = cameras.Camera("mario", "novice", [("Mario1-1", str(d))])                 # the hosted folder loads and plays
    for _ in range(5):
        cam.step()


def test_the_reproduce_tab_gives_the_commands_for_every_agent():
    from code.neuro import command_center
    agents = [{"game": "mario", "source": "probes", "variant": None, "persona": "novice", "tag": "p10g40", "level": "Mario1-1",
               "seed": 1234, "path": "agents/probes/mario/novice/p10g40/Mario1-1_1234", "best_fitness": 1.0}]
    html = command_center.reproduce_page("mario", agents)
    assert "pip install -r requirements.txt" in html and 'href="agents/mario.zip"' in html
    assert "--replay agents/probes/mario/novice/p10g40/Mario1-1_1234/best.npz" in html
    assert "python -m code.neuro.balance --game mario --persona novice --levels Mario1-1 --seeds 1234 --gens 40" in html
    body = html.split("</style>", 1)[1]                        # the shared styles' comments are not page text
    assert "<nav" not in body and "\u2014" not in body and "\u2013" not in body


def test_the_tabs_run_in_the_spec_order_with_reproduce_before_how_it_works():
    from code.neuro import command_center
    html = command_center.shell({"mario": {"novice": True}})
    assert re.findall(r'<button class="tab"[^>]*data-tab="(\w+)"', html) == [
        "wall", "balance", "mutations", "gasweep", "ablation", "reproduce", "howitworks"]


def test_the_untrained_cards_are_not_a_collapsible_game_section():
    # the report's script wires a collapse button (.chev) on every section.game; these cards have none
    data = report.load_data()
    block = report._untrained("bomberman", data["games"]["bomberman"])
    assert block and not re.search(r'<section class="[^"]*\bgame\b', block)
