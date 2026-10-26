# PEAK Command Center Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the five-page Balance Command and its live replay wall with one PEAK Command Center page (game dropdown, six tabs, looping recorded videos) that menu 12 opens locally and that GitHub Pages hosts.

**Architecture:** `code/neuro/command_center.py` writes a static site to `runs/site/`: a shell `index.html` (top bar, dropdown, tab bar, the wall's `<video>` grid) and one sub-page per game and tab, built by the existing page functions in `report.py` and `ast_page.py` with their nav removed. The shell shows a sub-page in an `<iframe>` per tab, so each sub-page keeps its own CSS and JS unchanged. Videos are recorded once by `code/neuro/wall_video.py` (the replay wall's cameras, piped to ffmpeg) into a git-ignored cache. A publish step puts `runs/site/` on `gh-pages` as one commit with no history.

**Tech Stack:** Python 3.12 (`.venv`), pygame (headless), numpy, ffmpeg with libvpx-vp9, plain HTML/CSS/JS, git, pytest.

**Spec:** `docs/superpowers/specs/2026-10-26-command-center-design.md`

## Global Constraints

- Name: **PEAK Command Center**. The words "Balance Command" appear nowhere in code, menu, README or docs afterwards.
- Seeds: only the paper seeds `1234, 2025, 31337` reach the page (numbers, videos, route traces).
- No ripped art: never read `megaman-data/` or `PEAK_MM1_ASSETS` when recording; the published site holds no sprite sheet.
- The Command Center never calls `balance.rebuild()` or `gasweep.rebuild()` and never writes a tracked file.
- No change to `paper-ast/`, to `ast_metrics` results or to `ast_paper` output.
- Look: `code/neuro/web/theme.css` (black and red); blue only in the Mega Man icon.
- No em dashes or en dashes in code, docs or commit messages; commit messages carry no Co-Authored-By line.
- Work on branch `AL-command-center`, created from `AL-ast-eval`, in the main checkout (a second worktree would be 34,458 files and 12 GB on the exFAT drive). Narval results are committed on `AL-ast-eval` between tasks and merged in.
- Test command: `SDL_VIDEODRIVER=dummy .venv/bin/python -m pytest code/tests -q -p no:debugging`.

## Review Focus

1. A tab with no data for a game (Mega Man GA sweep until Narval jobs 7434840/7434841 come home; Mega Man ablation always): the tab shows a "not run" panel, never a blank frame or a 404. Pinned in Task 3 and Task 4.
2. A persona with no video: a "not recorded yet" tile in its place, the other five still play. Pinned in Task 4.
3. A hash that names no game or tab (`#zelda/x`, `#`, `#mario`): the page opens on Mario, Replay wall (or the named game's first tab). Pinned in Task 4 (browser check).
4. Opening `runs/site/index.html` straight from disk (`file://`, no server): every tab and video works. Pinned in Task 4 (browser check).
5. Publishing with uncommitted results, or a second publish after the tag exists: refused with a clear message, or the tag left as it is. Pinned in Task 7.

---

### Task 1: Paper seeds only

**Files:**
- Modify: `code/neuro/balance.py` (`load_probe_cells`, `write_report`, `rebuild`, `main`; remove the `MEGAMAN_ROOT` entry added in 5c46b7846)
- Modify: `code/neuro/report.py:320` (`_collect_routes`)
- Modify: `code/neuro/cameras.py:56` (`pick_models`)
- Delete (untracked file): `runs/balance/report_megaman_experienced_p10g200.json`
- Test: `code/tests/test_command_center.py` (new)

**Interfaces:**
- Produces: `balance.PAPER_SEEDS = (1234, 2025, 31337)`; `load_probe_cells(game, persona, tag, root=PROBES_ROOT, seeds=None)` and `write_report(..., seeds=None)` keep only those seeds when `seeds` is given; `report._collect_routes(..., seeds=PAPER_SEEDS)`; `cameras.pick_models(game, persona, root=PROBES_ROOT, seeds=PAPER_SEEDS)`.

- [ ] **Step 1: Create the branch**

Run: `git switch -c AL-command-center AL-ast-eval`
Expected: `Switched to a new branch 'AL-command-center'`

- [ ] **Step 2: Write the failing tests** in `code/tests/test_command_center.py`

```python
def test_reports_keep_only_the_seeds_asked_for(tmp_path):
    # two probe dirs, seeds 1234 and 8, each a state.json with a one-generation history
    ...
    cells = balance.load_probe_cells("mario", "novice", "p10g40", str(tmp_path), seeds=balance.PAPER_SEEDS)
    assert [c["seed"] for c in cells["Mario1-1"]] == [1234]
    assert {c["seed"] for c in balance.load_probe_cells("mario", "novice", "p10g40", str(tmp_path))["Mario1-1"]} == {8, 1234}

def test_the_wall_and_the_route_traces_skip_extra_seeds():
    # on the committed probes: Bomberman experienced level 1, seed 8 wins 57 %, the paper seeds 0 %
    models = dict(cameras.pick_models("bomberman", "experienced"))
    assert models and all(d.rsplit("_", 1)[1] in {"1234", "2025", "31337"} for d in models.values())
    routes, best = report._collect_routes("bomberman", "experienced", "1", "p10g40")
    assert best is None or best["seed"] in {"1234", "2025", "31337"}
```

- [ ] **Step 3: Run them to see them fail**

Run: `SDL_VIDEODRIVER=dummy .venv/bin/python -m pytest code/tests/test_command_center.py -q -p no:debugging`
Expected: FAIL (`PAPER_SEEDS` not defined; `seeds` not a parameter)

- [ ] **Step 4: Implement**

`PAPER_SEEDS` beside `PROBES_ROOT` in `balance.py`. `load_probe_cells` skips a cell whose `config.seed` is not in `seeds`; `write_report` passes `seeds` through. `rebuild()` passes `PAPER_SEEDS` for `PROBES_ROOT` only (ablation probes stay as they are) and loses the `MEGAMAN_ROOT` entry (the Command Center reads Mega Man from `runs/ast`, Task 3). `balance.main()` passes `seeds=args.seeds` to `write_report`, so a new sweep's report holds only the seeds it ran. `_collect_routes` and `pick_models` skip a `<level>_<seed>` dir whose seed is not in `seeds`. Delete the untracked `runs/balance/report_megaman_experienced_p10g200.json`.

- [ ] **Step 5: Run the tests**

Run: the test command (Global Constraints)
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add code/neuro/balance.py code/neuro/report.py code/neuro/cameras.py code/tests/test_command_center.py
git commit -m "Paper seeds only: reports, route traces and the wall skip probe seeds 1, 3, 8 and 7689798"
```

---

### Task 2: No Watch button, a Mega Man icon, the new name

**Files:**
- Modify: `code/neuro/report.py` (`_watch_cmd` and its use in `_card`; `serve()`, `WALL_PORT`, the `/wall` link in `_nav`; `_game_icon`; every "Balance Command" string)
- Modify: `code/stats/ast_page.py`, `code/stats/ast_metrics.py`, `code/stats/registry.py`, `code/neuro/gasweep.py`, `code/neuro/web/theme.css`, `menu.py`, `README.md`, `docs/BALANCE.md`, `code/tests/test_report.py` (the name)
- Test: `code/tests/test_command_center.py`

**Interfaces:**
- Produces: `report._MEGAMAN_SVG` (data URI); `report._game_icon("megaman")` returns an `<img class="gicon">`.

- [ ] **Step 1: Write the failing tests**

```python
def test_mega_man_has_a_drawn_icon_not_a_letter():
    html = report._game_icon("megaman")
    assert "<img" in html and "svg" in html and "gletter" not in html

def test_no_page_offers_a_watch_replay():
    assert not hasattr(report, "serve") and not hasattr(report, "_watch_cmd")

def test_the_old_name_is_gone():
    hits = subprocess.run(["git", "grep", "-l", "-i", "balance command", "--", "code", "menu.py", "README.md", "docs"],
                          capture_output=True, text=True).stdout.split()
    assert [h for h in hits if not h.startswith("docs/superpowers/")] == []
```

- [ ] **Step 2: Run them to see them fail**

Run: `SDL_VIDEODRIVER=dummy .venv/bin/python -m pytest code/tests/test_command_center.py -q -p no:debugging`
Expected: 3 FAIL

- [ ] **Step 3: Implement**

`_MEGAMAN_SVG`: the blue helmet of option A in the brainstorm mockup (view box `0 0 28 36`; helmet `#0070ec`, ridge and ear discs `#3cbcfc`, face `#fcd8a8`, white eyes with black pupils, body `#0070ec` with a `#3cbcfc` collar), written as a `data:image/svg+xml;utf8,` string like `_BOMBERMAN_SVG`; one-line comment that it is the only blue on the pages. Delete `_watch_cmd`, the viz block that called it in `_card`, `serve()`, `WALL_PORT`, the `/wall` nav link and the `served-only` / `file-only` / `.watchbtn` CSS that only they used. Rename "Balance Command" to "Command Center" (title text "PEAK Command Center") everywhere the grep finds it; `report.main()` loses `--serve` (Task 4 gives the site its own server).

- [ ] **Step 4: Run the whole suite**

Run: the test command
Expected: all pass (update `test_report.py` where it asserted the old name or the Watch block)

- [ ] **Step 5: Commit**

```bash
git add -A code menu.py README.md docs/BALANCE.md
git commit -m "Command Center: the new name, a drawn Mega Man icon, no Watch replay and no report server"
```

---

### Task 3: One sub-page per game and tab

**Files:**
- Modify: `code/neuro/report.py` (`build()` becomes `load_data()`; new `balance_page()`; `nav` parameter on `_gasweep_page`, `_ablation_page`, `_instructions_page`; `_game_section`'s GA-sweep link)
- Modify: `code/stats/ast_page.py` (`render`)
- Test: `code/tests/test_command_center.py`

**Interfaces:**
- Produces:
  - `report.load_data(balance_dir: str = BALANCE_DIR) -> dict` with keys `games` (game -> tag -> persona -> report), `ablation` (as `flat_abl` today), `gasweep` (as `gs_groups` today), `thresholds`; it reads `runs/balance/report_*.json` and, for Mega Man, `runs/ast/megaman/base-retrained/report_megaman_*_p10g200.json`; it never calls a rebuild.
  - `report.balance_page(game: str, data: dict) -> str`, `report.gasweep_page(game: str, data: dict) -> str`, `report.ablation_page(game: str, data: dict) -> str`, `report.instructions_page(data: dict) -> str`: full HTML documents with no top bar, each ending with its own `_page_data()` and `JS`.
  - `ast_page.render(results: dict, only: str | None = None, nav: bool = True) -> str`: with `only`, the score strip and the false-alarm card still count every game; the board row, game tabs and section are that game's only.
  - Links that crossed pages become `href="index.html#<game>/<tab>" target="_top"`.

- [ ] **Step 1: Write the failing tests**

```python
GAMES = ["mario", "meatboy", "bomberman", "megaman"]

def test_every_game_gets_its_four_sub_pages_without_a_rebuild(monkeypatch):
    monkeypatch.setattr(balance, "rebuild", lambda *a, **k: pytest.fail("rebuild called"))
    data = report.load_data()
    for g in GAMES:
        for page in (report.balance_page(g, data), report.gasweep_page(g, data), report.ablation_page(g, data)):
            assert page.startswith("<!doctype html>") and "<nav" not in page
            assert not re.search(r'href="(report|mutations|gasweep|ablation|instructions)\.html|"/wall"|"/watch', page)

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
```

- [ ] **Step 2: Run them to see them fail**

Run: `SDL_VIDEODRIVER=dummy .venv/bin/python -m pytest code/tests/test_command_center.py -q -p no:debugging`
Expected: FAIL (`load_data` not defined)

- [ ] **Step 3: Implement**

Split `build()`: the loading part (minus `rebuild()` and `rebuild_gasweep()`) becomes `load_data()`; the page parts become the four functions, each filtering its input to one game before calling today's body code (`_game_section` with `GLOSSARY` above it; `_gasweep_page` / `_ablation_page` / `_instructions_page` with `nav=False`, which drops `_nav(...)`). `_runs_section()` moves to the end of `instructions_page()`. In `ast_page.render`, `only` filters `results` for the board, tabs and sections, and `nav=False` drops `report._nav(...)`. Then point `ast_metrics.main --page` at Task 4's site build instead of `runs/balance/mutations.html`.

- [ ] **Step 4: Run the whole suite**

Run: the test command
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add code/neuro/report.py code/stats/ast_page.py code/stats/ast_metrics.py code/tests/test_command_center.py
git commit -m "Command Center sub-pages: one per game and tab, read from the committed reports, no top bar"
```

---

### Task 4: The site: shell, tabs, dropdown

**Files:**
- Create: `code/neuro/command_center.py`
- Modify: `.gitignore` (add `runs/site/` and `runs/videos/`)
- Test: `code/tests/test_command_center.py`

**Interfaces:**
- Consumes: Task 3's page functions and `ast_page.render(..., only=, nav=False)`.
- Produces:
  - `SITE_DIR = "runs/site"`, `VIDEOS_DIR = "runs/videos"`, `GAMES = ["mario", "meatboy", "bomberman", "megaman"]`, `TABS = [("wall", "Replay wall"), ("balance", "Balance"), ("mutations", "Mutations"), ("gasweep", "GA sweep"), ("ablation", "Sensor ablation"), ("howitworks", "How it works")]`, `PERSONAS` = `cameras.ORDER`.
  - `build_site(out_dir: str = SITE_DIR, videos_dir: str = VIDEOS_DIR) -> list[str]`: writes `index.html`, `<game>-<tab>.html` for balance, mutations, gasweep and ablation, `howitworks.html`, `.nojekyll`, and copies `videos_dir/<game>/<persona>.webm` to `out_dir/videos/<game>/<persona>.webm`; returns the paths written.
  - `shell(walls: dict[str, dict[str, bool]]) -> str`: the index page; `walls[game][persona]` says whether that video exists.
  - CLI: `python -m code.neuro.command_center [--out runs/site] [--open] [--serve] [--port 8001]`.

- [ ] **Step 1: Write the failing tests**

```python
def test_the_site_has_every_tab_of_every_game(tmp_path):
    (tmp_path / "videos" / "mario").mkdir(parents=True)
    (tmp_path / "videos" / "mario" / "novice.webm").write_bytes(b"x")
    written = command_center.build_site(str(tmp_path / "site"), str(tmp_path / "videos"))
    names = {os.path.relpath(p, tmp_path / "site") for p in written}
    assert {"index.html", "howitworks.html", ".nojekyll", "videos/mario/novice.webm"} <= names
    assert {f"{g}-{t}.html" for g in command_center.GAMES for t in ("balance", "mutations", "gasweep", "ablation")} <= names

def test_the_wall_says_which_videos_are_missing():
    html = command_center.shell({"mario": {"novice": True, "clumsy": False}})
    assert 'data-src="videos/mario/novice.webm"' in html and "not recorded yet" in html
    assert "PEAK Command Center" in html and "Balance Command" not in html
    assert "loop" in html and "muted" in html and "playsinline" in html
```

- [ ] **Step 2: Run them to see them fail**

Run: `SDL_VIDEODRIVER=dummy .venv/bin/python -m pytest code/tests/test_command_center.py -q -p no:debugging`
Expected: FAIL (no module `command_center`)

- [ ] **Step 3: Implement the shell**

Top bar: the PEAK wordmark, "Command Center", a `<select id="game">` with the four games (icons beside the names), then a `role="tablist"` of the six tabs. One panel per tab: the wall panel holds one `<div class="wall" data-game=...>` per game with six tiles (persona name and colour from `theme.css --p-*`; a `<video muted loop playsinline preload="none" data-src=...>` or the "not recorded yet" tile); every other panel is one `<iframe>` filling the viewport under the bar. Hash `#<game>/<tab>`; unknown parts fall back to `mario` and `wall` (a known game alone opens its `wall`). On a hash change: set the select and the active tab; for an iframe tab set `src` to `<game>-<tab>.html` (`howitworks.html` for the last) only when it differs; on the wall set each visible video's `src` from `data-src` and `play()`, and for the hidden walls `pause()`, `removeAttribute("src")`, `load()` so only six videos ever decode. Keyboard: arrow keys move between tabs (`aria-selected`). CSS and fonts from `theme.css`, inline.

- [ ] **Step 4: Implement `build_site` and the CLI**

`--serve` serves `out_dir` with `http.server.ThreadingHTTPServer` on `127.0.0.1:<port>` (static files only); `--open` opens `index.html` (served URL with `--serve`, else `file://`).

- [ ] **Step 5: Run the tests**

Run: the test command
Expected: all pass

- [ ] **Step 6: Check it in a browser**

Run: `.venv/bin/python -m code.neuro.command_center --serve --port 8001`, then in headless Chrome (CDP; the app's browser pane runs hidden and stalls timers) load `http://127.0.0.1:8001/index.html#megaman/mutations`, `#zelda/x`, `#bomberman` and `file://.../runs/site/index.html#meatboy/balance`.
Expected: Mega Man's Mutations sub-page; Mario, Replay wall; Bomberman, Replay wall; Meat Boy's Balance from disk with no console error; at most six `<video>` elements with a `src` at any time.

- [ ] **Step 7: Commit**

```bash
git add code/neuro/command_center.py .gitignore code/tests/test_command_center.py
git commit -m "PEAK Command Center: one page, a game dropdown, six tabs, iframes per game and tab"
```

---

### Task 5: Recorded replay wall

**Files:**
- Create: `code/neuro/wall_video.py`
- Modify: `code/neuro/cameras.py` (`MegamanCamera` takes `legs`)
- Modify: `code/neuro/ast_eval.py` (factor `persona_legs` out of `play_megaman`)
- Test: `code/tests/test_command_center.py`

**Interfaces:**
- Consumes: `cameras.Camera`, `cameras.pick_models(..., seeds=PAPER_SEEDS)` (Task 1).
- Produces:
  - `cameras.MegamanCamera(manifest: str | None = None, stage: str = "CutMan", hold: int = FPS, legs: list | None = None)`; `persona` is `legs[0].persona`.
  - `ast_eval.persona_legs(persona: str, variant: str = RETRAINED) -> list | None`: experienced gets `runs/megaman/relay.json`'s legs; another persona gets its best run of every section (`_best_retrained`), or `None` when a section has none. `play_megaman` uses it.
  - `wall_video.frames(cam, max_frames: int = 5400) -> Iterator[pygame.Surface]`: one surface per camera step; stops once the camera has finished one pass (2D: one attempt at each of its levels; Mega Man: one stage attempt) and its result hold is over, or at `max_frames`.
  - `wall_video.write_webm(surfaces: Iterable[pygame.Surface], path: str, fps: int = 30, width: int = 480) -> int`: frames written; pipes raw RGB to `ffmpeg -f rawvideo -pix_fmt rgb24 -s WxH -r fps -i - -vf scale=width:-2 -c:v libvpx-vp9 -b:v 0 -crf 40 -an path`.
  - `wall_video.main(argv: list[str] | None = None) -> None`, CLI `python -m code.neuro.wall_video [--game ...] [--persona ...] [--out runs/videos]`: one file per game and persona that has agents; prints what it skipped and why.

- [ ] **Step 1: Write the failing tests**

```python
class FakeCam:
    """Camera's surface: `models` (one per level), `attempts`, `hold_left`, `step()`, `adapter.render(surf)`
    and `adapter.core.WIDTH/HEIGHT`. Each attempt takes `steps` steps, then `hold` steps of hold."""

def test_a_video_stops_after_one_pass_and_its_hold():
    assert sum(1 for _ in wall_video.frames(FakeCam(levels=2, steps=3, hold=2))) == 2 * (3 + 2)
    assert sum(1 for _ in wall_video.frames(FakeCam(levels=2, steps=10_000, hold=2), max_frames=50)) == 50

@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_webm_frames_are_written(tmp_path):
    surfaces = [pygame.Surface((64, 48)) for _ in range(10)]
    assert wall_video.write_webm(surfaces, str(tmp_path / "v.webm")) == 10
    assert (tmp_path / "v.webm").stat().st_size > 0

def test_recording_never_uses_ripped_art(monkeypatch):
    monkeypatch.setenv("PEAK_MM1_ASSETS", "/somewhere")
    with pytest.raises(SystemExit, match="megaman-data"):
        wall_video.main(["--game", "megaman"])
```

- [ ] **Step 2: Run them to see them fail**

Run: `SDL_VIDEODRIVER=dummy .venv/bin/python -m pytest code/tests/test_command_center.py -q -p no:debugging`
Expected: FAIL (no module `wall_video`)

- [ ] **Step 3: Implement**

`frames()` renders with `cam.adapter.render(surface)` after each `cam.step()`, as `cameras.frame_message` does. `main()` refuses (SystemExit naming `megaman-data`) when `PEAK_MM1_ASSETS` is set or `megaman-data/` exists. Videos go to `runs/videos/<game>/<persona>.webm`, written to a temporary name and renamed at the end.

- [ ] **Step 4: Run the whole suite**

Run: the test command
Expected: all pass, the existing AST tests included (`play_megaman` now calls `persona_legs`)

- [ ] **Step 5: Record the 2D games**

Run: `SDL_VIDEODRIVER=dummy .venv/bin/python -m code.neuro.wall_video --game mario meatboy bomberman`
Expected: up to 18 files under `runs/videos/`; each plays to the end and loops in a browser.

- [ ] **Step 6: Commit**

```bash
git add code/neuro/wall_video.py code/neuro/cameras.py code/neuro/ast_eval.py code/tests/test_command_center.py
git commit -m "Replay wall videos: one looping WebM per game and persona, recorded from the wall's cameras"
```

---

### Task 6: The menu

**Files:**
- Modify: `menu.py` (item 12, `_open_command_center`, the Mutations item 21, two new items)
- Test: `code/tests/test_command_center.py`

**Interfaces:**
- Consumes: `command_center` CLI (Task 4), `wall_video` CLI (Task 5), `command_center.publish` (Task 7).
- Produces: `menu.DISPATCH` at module level (today a local of `main()`, `menu.py:1298`); item 12 `("command_center", open_command_center)` runs `python -m code.neuro.command_center --serve --open`; `_open_command_center(game: str, tab: str)` builds the site and opens `index.html#<game>/<tab>`; item 22 `("record_wall", record_wall_videos)`; item 23 `("publish_site", publish_site)`.

- [ ] **Step 1: Write the failing test**

```python
def test_the_menu_lists_the_command_center_and_its_two_new_items():
    import menu
    assert menu.DISPATCH["12"][0] == "command_center"
    assert menu.DISPATCH["22"][0] == "record_wall" and menu.DISPATCH["23"][0] == "publish_site"
```

- [ ] **Step 2: Run it to see it fail**

Run: `SDL_VIDEODRIVER=dummy .venv/bin/python -m pytest code/tests/test_command_center.py -q -p no:debugging`
Expected: FAIL

- [ ] **Step 3: Implement**

Rename `open_balance_command` to `open_command_center`. Callers of `_open_command_center(page)` pass the game and tab they ran (a sweep: `balance`; an ablation: `ablation`; a GA sweep: `gasweep`; item 21: `mutations`). Item 23 asks "Publish to https://code-sorcerylab.github.io/PEAK-DRL-Tool/? [y/N]" and runs the publish with push only on `y`.

- [ ] **Step 4: Run the whole suite**

Run: the test command
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add menu.py code/tests/test_command_center.py
git commit -m "Menu: 12 opens the Command Center, 22 records the wall, 23 publishes it"
```

---

### Task 7: Publish to GitHub Pages

**Files:**
- Modify: `code/neuro/command_center.py`
- Test: `code/tests/test_command_center.py`

**Interfaces:**
- Produces: `publish(site_dir: str = SITE_DIR, push: bool = False, repo: str = ".") -> str` returning the new `gh-pages` commit id. CLI flags `--publish` and `--push`.

- [ ] **Step 1: Write the failing tests** (fixture `tmp_repo`: `git init` in `tmp_path` with a committed `runs/balance/report_x.json`, a `gh-pages` branch of one commit whose id is `OLD_GH_PAGES_TIP`, a `site/index.html`, and an `origin` that is a bare repo beside it; `git(repo, *args)` returns stripped stdout)

```python
def test_publish_refuses_uncommitted_results(tmp_repo):
    (tmp_repo / "runs" / "balance" / "report_x.json").write_text("changed")
    with pytest.raises(SystemExit, match="uncommitted"):
        command_center.publish(str(tmp_repo / "site"), repo=str(tmp_repo))

def test_publish_makes_one_commit_and_keeps_the_old_page_as_a_tag(tmp_repo):
    first = command_center.publish(str(tmp_repo / "site"), repo=str(tmp_repo))
    assert git(tmp_repo, "rev-list", "--count", "gh-pages") == "1"
    assert git(tmp_repo, "rev-parse", "landing-page-2026-03^{commit}") == OLD_GH_PAGES_TIP
    command_center.publish(str(tmp_repo / "site"), repo=str(tmp_repo))     # a second publish
    assert git(tmp_repo, "rev-parse", "landing-page-2026-03^{commit}") == OLD_GH_PAGES_TIP
```

- [ ] **Step 2: Run them to see them fail**

Run: `SDL_VIDEODRIVER=dummy .venv/bin/python -m pytest code/tests/test_command_center.py -q -p no:debugging`
Expected: FAIL (`publish` not defined)

- [ ] **Step 3: Implement**

Refuse (SystemExit with the paths) while `git status --porcelain -- runs/balance runs/ast` prints anything. Create tag `landing-page-2026-03` at the current `gh-pages` tip only if the tag does not exist. Make the commit with a temporary index (`GIT_INDEX_FILE`), `git --work-tree=<site_dir> add -A`, `git write-tree`, `git commit-tree <tree> -m "PEAK Command Center <date>"` (no parent), `git update-ref refs/heads/gh-pages <commit>`. With `push`: `git push origin landing-page-2026-03` (if not on origin) and `git push --force origin gh-pages`.

- [ ] **Step 4: Run the whole suite**

Run: the test command
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add code/neuro/command_center.py code/tests/test_command_center.py
git commit -m "Command Center: publish the site to gh-pages as one commit, old landing page kept as a tag"
```

---

### Task 8: Mega Man videos, the full site, first publish

**Files:** none new.

- [ ] **Step 1: Bring the Narval results in**

On `AL-ast-eval`, once jobs 7425515 and 7433368 are done and their results committed there (the copy-home step outside this plan), run: `git merge AL-ast-eval` on `AL-command-center`.
Expected: a merge with no conflict in `code/`.

- [ ] **Step 2: Record Mega Man**

Run: `SDL_VIDEODRIVER=dummy .venv/bin/python -m code.neuro.wall_video --game megaman`
Expected: six files under `runs/videos/megaman/`, or a printed reason per persona that has none.

- [ ] **Step 3: Build and check the whole site**

Run: `.venv/bin/python -m code.neuro.command_center --serve --port 8001` and repeat Task 4 Step 6 for all four games and six tabs.
Expected: every tab shows data or its "not run" panel; 24 tiles, each a looping video or "not recorded yet"; no "Balance Command" anywhere; `du -sh runs/site` under 1 GB and the first view under 10 MB.

- [ ] **Step 4: Ask AL, then publish**

Show AL the local page. Only after AL says yes in chat, run: `.venv/bin/python -m code.neuro.command_center --publish --push`.
Expected: https://code-sorcerylab.github.io/PEAK-DRL-Tool/ shows the Command Center within a few minutes; tag `landing-page-2026-03` exists on origin.
