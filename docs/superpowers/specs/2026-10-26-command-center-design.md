# PEAK Command Center: one page, one game at a time, hosted on GitHub Pages

Date: 2026-10-26. Owner: AL. Status: design, agreed in chat; not built.

## Why

The dashboard that menu 12 opens (today "Balance Command") is five separate HTML pages in
`runs/balance/` plus a live replay wall served by `code/neuro/cameras.py`. Three things are wrong
with it:

- AL's professor asked for a game selector: pick one game and see only that game.
- It runs only on the machine that has the runs and a Python server, so nobody else can open it.
- Bugs: the replay (camera) wall shows over the balance pages, the ▶ Watch replay buttons do not
  work, and the live wall sometimes stops.

The goal is one page that anyone can open from a link, that shows the whole tool for one game at
a time, and that never stalls.

## Goals

1. **One page**, named **PEAK Command Center** (the name "Balance Command" goes everywhere: page
   title, top bar, menu 12, README, docs).
2. **A game dropdown** at the top: Mario, Meat Boy, Bomberman, Mega Man. The page shows only the
   chosen game.
3. **Tabs** under the dropdown, one part on screen at a time (layout A, AL's choice):
   1. **Replay wall**: one recorded video per persona (six), each looping with no end.
   2. **Balance**: what `report.html` shows today for that game (radar, configs x personas, the
      level table, the level dialog with its maps).
   3. **Mutations**: what `mutations.html` shows today for that game (map, variant buttons, the
      selected-variant card, every-variant table, wins per persona, Mega Man's segments).
   4. **GA sweep**: what `gasweep.html` shows today for that game.
   5. **Sensor ablation**: what `ablation.html` shows today for that game ("not run" where a game
      has no ablation, as Mega Man).
   6. **How it works**: today's `instructions.html` (the same for every game).
4. **A new Mega Man icon**: option A from the mockup, a blue Mega Man helmet drawn by us as an SVG,
   in the same style and size as `_BOMBERMAN_SVG` and `_MEATBOY_SVG` in `code/neuro/report.py`
   (28 x 36 view box). It replaces the letter badge "M". It is the one exception to the pages'
   "no blue" rule, and only for this icon.
5. **No ▶ Watch buttons** and no link to the live wall.
6. **Hosted on GitHub Pages**: https://code-sorcerylab.github.io/PEAK-DRL-Tool/ **replaces** the
   March 2026 landing page now on the `gh-pages` branch (AL's choice).
7. **The same page locally**: menu 12 builds it and opens it, so there is one page, not a local
   one and a hosted one.

## Non-goals

- No live agents in the browser (no Python or pygame compiled to WebAssembly, no server).
- No database server. The data rides in the page, as it does today.
- No new metrics, no change to any result, no change to `paper-ast/` outputs.

## Anti-goals (failure even if it works)

- **Hosted numbers that differ from the paper's.** `report.build()` calls `balance.rebuild()`,
  which regenerates `runs/balance/*.json` from the probe dirs; committed stray probe seeds (1, 3,
  8, 7689798 under `runs/probes/...`) turn the 3-seed reports into 7-seed ones (for example
  Bomberman experienced, level 1: win rate 0.0 becomes 0.149). The Command Center must read the
  committed report JSONs and must not rewrite tracked files.
- **Copyrighted art on the public site.** The Mega Man sprite sheets live only in the git-ignored
  `megaman-data/` (ripped Capcom sprites). They must not appear in the icon, the videos or any
  published file. Without that folder the game draws its own flat shapes; record from a machine
  without it.
- **A page that does not load.** Today the five pages weigh 20.8 MB together on disk (`gasweep.html`
  alone 7.9 MB).
- **Breaking the paper path.** `ast_metrics`, `ast_paper` and the AST runs on branch
  `AL-ast-eval` are in use until the 30 Oct 2026 deadline.

## Decided

| What | Decision | By |
|---|---|---|
| Layout | Tabs, one part at a time, under one top bar with the dropdown | AL |
| Games in the dropdown | Mario, Meat Boy, Bomberman, Mega Man | AL |
| Replay wall | Recorded videos that loop, not live play | AL |
| Mega Man icon | Option A, blue helmet, our own SVG | AL |
| Name | PEAK Command Center | AL |
| Seeds | Only the paper seeds 1234, 2025 and 31337, for the numbers, the videos and the route traces. The 300 probe runs with seeds 1, 3, 8 and 7689798 (committed 2026-09-21, "data") are left out everywhere: `cameras.pick_models`, `report._collect_routes` and any report refresh must filter them | AL |
| Watch buttons | Removed | AL |
| Hosting | GitHub Pages, replacing the `gh-pages` landing page | AL |
| Page format | One page to open (`index.html`: top bar, dropdown, tabs, the wall's videos). Every other tab shows a sub-page per game and tab (`<game>-<tab>.html`, built by today's page functions without their top bar) in an `<iframe>`, so each keeps its own CSS and JS and nothing clashes; it works from `file://` too. Plus a `videos/` folder; no JSON database. "One page" means one page for the viewer, not one file. Target: the first view (the default game, its first tab) loads under about 10 MB; today's five pages weigh 20.8 MB on disk (`gasweep.html` 7.9 MB). (Changed while planning from `data/<game>.js` scripts: the pages' JS binds at load, so injected HTML would need a rewrite of that JS.) | my call |
| Link state | The URL hash keeps the game and the tab (for example `#megaman/mutations`), so a link opens on one game | my call |
| Data source | The committed `runs/balance/*.json` and `runs/ast/**` files; no `rebuild()` during the build | my call |
| Cross-game score tiles | The Mutations tab keeps the board's totals for all games at its top (19 right, 1 wrong today), then the chosen game | my call |
| Video content | Per game and persona, the agents the live wall plays today (`cameras.pick_models`: per level, the paper seed with the best record, p10g40 full-sweep models); Mega Man plays the Cut Man stage as a relay of that persona's section agents: its best paper-seed `base-retrained` run per section, for every persona (`ast_eval.persona_legs`; changed after the final review from `runs/megaman/relay.json` for experienced, whose legs are seeds 4002 to 5201, against AL's paper-seed decision). An attempt that outruns its level's share of the 5400 frames ends as a stall, so every level appears | my call |
| Video format | WebM (VP9) through ffmpeg (installed), 30 fps, about 480 px wide, `<video autoplay muted loop playsinline>`; one attempt per level in level order, then the loop restarts | my call |
| Where videos live | A git-ignored cache (`runs/videos/<game>/<persona>.webm`) filled by its own record command (a new menu item), not by menu 12, because recording is heavy. The build copies the cache into the site; a persona without a video gets a "not recorded yet" tile | my call |
| Where the work happens | A new branch `AL-command-center` from `AL-ast-eval` (it needs the Mega Man persona code), in the main checkout. The Narval results are committed on `AL-ast-eval` between tasks and merged in. (Changed while planning from a separate worktree: a second checkout is 34,458 files and 12 GB on the exFAT drive.) | my call |
| Publishing | A script builds the site folder and puts it on `gh-pages` as one new commit with no history (the branch is replaced each time, so 50 MB of videos never pile up in every clone; `.git` is already 2.6 GB). Before the first publish, the old landing page's commit is kept as tag `landing-page-2026-03`. The script refuses to publish while `runs/balance/` or `runs/ast/` have uncommitted changes. The force-push itself is a separate step that AL approves each time | my call |
| New results | The build reads the working tree, never runs `rebuild()`, and a run's own report JSON (balance.py and gasweep.py write one at the end of every run) is how new results arrive. Menu 12 no longer folds probe folders into reports | my call |
| Mega Man data | Balance: the committed `runs/ast/megaman/base-retrained/report_megaman_<persona>_p10g200.json`. GA sweep: "not run" until Narval jobs 7434840 and 7434841 come home. Ablation: "not run". The base-retrained training for all six personas finished on Narval on 2026-10-26, so all six Mega Man videos can be recorded once those results are committed | my call |

## What exists to build on

- `code/neuro/report.py` (3,259 lines): `build()` returns `{filename: html}` for the five pages;
  `_nav()` draws the shared top bar; `_game_section()`, `_gasweep_page()`, `_ablation_page()`,
  `_instructions_page()`, `_runs_section()`; `serve()` holds the ▶ Watch and `/wall` routes; `CSS`
  and `JS`; `_game_icon()` and the icon SVGs.
- `code/stats/ast_page.py`: `render(results)` builds the Mutations page with its own styles, JS and
  per-game data blocks; `board()` and `scores()` make the cross-game totals.
- `code/neuro/web/theme.css` and `code/neuro/pixel_art.py`: the shared look (black and red, the
  PEAK Pixel font, pixel icons).
- `code/neuro/cameras.py`: `pick_models()`, `layout()` and the per-camera renderers (2D games and
  the Mega Man relay through `relay.play(..., on_frame=...)`), FPS 30. This is the code to reuse for
  recording.
- `menu.py`: item 12 (`open_balance_command`, label "Balance Command") and item 21 (Mutations page).
- GitHub Pages is already on for the repo: legacy build from branch `gh-pages`, path `/`.

## Constraints

- GitHub Pages serves static files only: a published site of at most 1 GB, files of at most
  100 MB. The page must work from `file://` too, or menu 12 serves it with a static server.
- The page keeps today's look (`theme.css`); the new top bar replaces `_nav()`'s page links.
- The repo is public and the AST paper is double-blind: the paper must not link to this site; an
  anonymous copy is a separate decision for later.
- Narval jobs 7425515, 7433368, 7434840 and 7434841 write Mega Man results that will come home on
  `AL-ast-eval`; the Command Center reads them after they are committed there and merged.

## Left to the builder

- How the six tabs share the page: today's page bodies moved under tabs, or rebuilt from the
  same functions, as long as the first view stays under the size target above.
- Exact styles of the dropdown and the tabs, inside the current theme.
- Video length details (attempt cap, a short title card per level), as long as each video loops
  with no visible stop.
- Whether `cameras.py` keeps its live server for local use or only records.
- Tests: keep `code/tests/test_report.py` and the AST tests passing; add tests where the build
  changes behaviour (no tracked file written, every game's tabs present, no ripped art).

## Round 2 (2026-10-26, after AL saw the local site)

AL asked for six more things before the first publish. AL's answers to the four questions are marked AL;
the rest is my call.

| What | Decision | By |
|---|---|---|
| Zoom | A click on a wall tile opens it large in a dialog; Esc or a click outside closes it, as the old live wall did | AL |
| Sensor data | In the large view only: the rays drawn over the video and the sensor bars beside it, in step with the video (the old wall's watched view) | AL |
| Level names | Every tile shows the level playing and the result of the last attempt, over the video | AL |
| Mega Man sections | Mega Man's videos play one attempt per section from its start (8 sections), as the 2D games play one per level, so the sections after Street appear | AL |
| Levels in the Balance tab | A level that the game has but no agent trained on (Bomberman 12 to 14, Mario's practice variants, Mega Man's full stage and practice sections) appears as a "not trained yet" card | AL |
| Pixel look | The top bar and tabs use the look of the old pages' top bar (PEAK Pixel font, pixel frames, a pixel icon per tab) | AL |
| Agents to download | Every paper-seed agent under `runs/probes` and `runs/ast`: `agents/<path as in runs>/best.npz` plus a slim `state.json` (persona, seed, network settings, level; no history), so `python -m code.neuro.trainer --game <g> --replay <dir>/best.npz` works on a download unchanged; plus `agents/<game>.zip`. About 2,000 agents, under 10 MB | AL asked; layout my call |
| Reproduce tab | A new tab before How it works: per game, setting up the repo, replaying any agent (a table of agents with download links and the replay command), retraining it (the balance command with its seed, persona, level, budget and GA setting), and rebuilding the results (the Command Center, the AST evaluation and paper tables) | AL asked; content my call |
| Video data | Next to each video: `<persona>.json`, a small timeline (level and result per attempt, with the frame it starts) for the captions, and `<persona>.sensors.json` (rays and sensor values every 2nd frame) loaded only when the large view opens | my call |
| Mega Man video length | 7,200 frames (4 minutes) for 8 sections, 900 each; the 2D games keep 5,400 | my call |
| Tab order | Replay wall, Balance, Mutations, GA sweep, Sensor ablation, Reproduce, How it works | my call |
