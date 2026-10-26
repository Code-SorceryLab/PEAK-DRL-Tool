"""The PEAK Command Center: one page, one game at a time, as a static site (runs/site/).

    python -m code.neuro.command_center                    # build runs/site/
    python -m code.neuro.command_center --serve --open     # build, serve on 127.0.0.1:8001, open it (menu 12)
    python -m code.neuro.command_center --publish --push   # build, put it on gh-pages, push it live (menu 23)

index.html holds the top bar (a game dropdown), the tabs and the replay wall: one looping video per
persona, recorded once by code/neuro/wall_video.py into runs/videos/. Every other tab shows a
sub-page in an <iframe>, one file per game and tab, built by code/neuro/report.py and
code/stats/ast_page.py; each keeps its own styles and scripts, so nothing clashes. The link keeps
the game and the tab (#megaman/mutations). Everything is read from the committed reports, so the
page carries the published numbers, and nothing needs a server: GitHub Pages hosts the same folder.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import subprocess
import tempfile
import webbrowser
from datetime import datetime

from . import report
from .cameras import ORDER as PERSONAS

SITE_DIR = os.path.join("runs", "site")
VIDEOS_DIR = os.path.join("runs", "videos")
GAMES = ["mario", "meatboy", "bomberman", "megaman"]
NAMES = {"mario": "Mario", "meatboy": "Meat Boy", "bomberman": "Bomberman", "megaman": "Mega Man"}
TABS = [("wall", "Replay wall"), ("balance", "Balance"), ("mutations", "Mutations"), ("gasweep", "GA sweep"),
        ("ablation", "Sensor ablation"), ("howitworks", "How it works")]
ICONS = {"wall": "gamepad", "balance": "trophy", "mutations": "dna", "gasweep": "gear", "ablation": "eye",
         "reproduce": "sprout", "howitworks": "qblock"}   # the theme's pixel icons, as the old top bar had
esc = html.escape


def _tile(game: str, persona: str, recorded: bool) -> str:
    head = f'<div class="who" style="--c:var(--p-{persona})"><i class="ico p-{persona}"></i>{persona}</div>'
    if not recorded:
        return f'<div class="tile frame missing">{head}<div class="none">not recorded yet</div></div>'
    return (f'<div class="tile frame">{head}<video muted loop playsinline preload="none" '
            f'data-src="videos/{game}/{persona}.webm" aria-label="{NAMES[game]}, {persona}: the agents playing"></video></div>')


def shell(walls: dict[str, dict[str, bool]]) -> str:
    """index.html. walls[game][persona]: whether that persona's video exists."""
    games = [g for g in GAMES if g in walls] or GAMES
    options = "".join(f'<option value="{g}">{NAMES[g]}</option>' for g in games)
    icons = "".join(f'<span class="gi" data-game="{g}" hidden>{report._game_icon(g)}</span>' for g in games)
    tabs = "".join(f'<button class="tab" role="tab" type="button" id="t-{t}" data-tab="{t}" aria-controls="p-{t}" '
                   f'aria-selected="false" tabindex="-1"><i class="ico {ICONS[t]}"></i>{esc(label)}</button>'
                   for t, label in TABS)
    wall = "".join(f'<div class="wall stripe g-{g}" data-game="{g}" hidden>'
                   + "".join(_tile(g, p, walls.get(g, {}).get(p, False)) for p in PERSONAS) + "</div>" for g in games)
    panels = (f'<section class="panel" role="tabpanel" id="p-wall" aria-labelledby="t-wall" hidden>{wall}</section>'
              + "".join(f'<section class="panel" role="tabpanel" id="p-{t}" aria-labelledby="t-{t}" hidden>'
                        f'<iframe title="{esc(label)}"></iframe></section>' for t, label in TABS[1:]))
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'><rect width='16' height='16' fill='%23000000'/><path d='M3 12 L8 4 L13 12' stroke='%23f83800' stroke-width='2' fill='none'/></svg>">
<title>PEAK Command Center</title><style>{report.CSS}{CSS}</style></head><body>
<nav aria-label="Command Center">
  <h1 class="brand"><span class="wordmark">PEAK</span><em>Command Center</em></h1>
  <label class="pick">{icons}<span class="sr">Game</span><select id="game">{options}</select></label>
  <div class="links" role="tablist" aria-label="Parts of the Command Center">{tabs}</div>
</nav>
<main id="main">{panels}</main>
<script>const GAMES = {json.dumps(games)}, NAMES = {json.dumps(NAMES)};{JS}</script></body></html>"""


CSS = """
[hidden]{display:none !important}
body{height:100vh;display:flex;flex-direction:column;overflow:hidden;padding:0}
nav{flex:none;position:static;margin:0}                 /* the old pages' top bar, in a full-height page */
.pick{display:flex;align-items:center;gap:8px}
.pick .gicon{height:26px;width:auto;vertical-align:middle}
.pick select{font:700 18px/1 var(--pixel);color:var(--ink);background:var(--deck-2);padding:4px 8px;cursor:pointer;
  border:3px solid transparent;border-image:var(--frame) 1 / 3px stretch;background-clip:padding-box}
.tab[aria-selected="true"]{background-color:var(--ink);color:var(--night)}
main{flex:1;min-height:0;max-width:none;margin:0}
.panel{height:100%}
.panel iframe{width:100%;height:100%;border:0;display:block;background:var(--night)}
#p-wall{overflow:auto}
.wall{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:14px;padding:16px 16px 16px 22px}
.tile{padding:8px}
.tile video{width:100%;display:block;background:#000;image-rendering:pixelated}
.who{font:700 15px var(--pixel);color:var(--c);margin:0 0 6px;display:flex;gap:6px;align-items:center}
.missing .none{display:grid;place-items:center;aspect-ratio:4/3;color:var(--mute);border:2px dashed var(--line)}
@media (max-width:640px){nav{padding:8px}.wall{grid-template-columns:1fr;padding:12px}}
"""

JS = """
const TABS = [...document.querySelectorAll('.tab')].map(b => b.dataset.tab);
const select = document.getElementById('game');
function parse(h){
  const [g, t] = h.replace(/^#/, '').split('/');
  return {game: GAMES.includes(g) ? g : GAMES[0], tab: TABS.includes(t) ? t : 'wall'};
}
function stop(v){ if (v.getAttribute('src')){ v.pause(); v.removeAttribute('src'); v.load(); } }
function show(){
  const {game, tab} = parse(location.hash);
  select.value = game;
  for (const i of document.querySelectorAll('.gi')) i.hidden = i.dataset.game !== game;
  for (const b of document.querySelectorAll('.tab')){
    const on = b.dataset.tab === tab;
    b.setAttribute('aria-selected', on); b.tabIndex = on ? 0 : -1;
    document.getElementById('p-' + b.dataset.tab).hidden = !on;
  }
  for (const w of document.querySelectorAll('.wall')){
    const on = tab === 'wall' && w.dataset.game === game;
    w.hidden = !on;
    for (const v of w.querySelectorAll('video')){        // only the six on screen ever decode
      if (on){ if (!v.getAttribute('src')) v.src = v.dataset.src; v.play().catch(() => {}); } else stop(v);
    }
  }
  if (tab !== 'wall'){
    const f = document.querySelector('#p-' + tab + ' iframe');
    const want = tab === 'howitworks' ? 'howitworks.html' : game + '-' + tab + '.html';
    if (f.getAttribute('src') !== want) f.setAttribute('src', want);
  }
  document.title = 'PEAK Command Center: ' + NAMES[game];
  if (location.hash !== '#' + game + '/' + tab) history.replaceState(null, '', '#' + game + '/' + tab);
}
const go = (game, tab) => { location.hash = '#' + game + '/' + tab; };
select.addEventListener('change', () => go(select.value, parse(location.hash).tab));
for (const b of document.querySelectorAll('.tab')) b.addEventListener('click', () => go(select.value, b.dataset.tab));
document.querySelector('[role=tablist]').addEventListener('keydown', e => {
  const step = {ArrowRight: 1, ArrowLeft: -1}[e.key];
  if (!step) return;
  const i = (TABS.indexOf(parse(location.hash).tab) + step + TABS.length) % TABS.length;
  go(select.value, TABS[i]);
  document.getElementById('t-' + TABS[i]).focus();
});
window.addEventListener('hashchange', show);
show();
"""


def build_site(out_dir: str = SITE_DIR, videos_dir: str = VIDEOS_DIR) -> list[str]:
    """Write the whole site to out_dir (replacing an earlier build there) and return the paths written."""
    from code.stats import ast_metrics, ast_page
    if os.path.isdir(out_dir) and os.listdir(out_dir) and not os.path.exists(os.path.join(out_dir, ".nojekyll")):
        raise SystemExit(f"{out_dir} is not an earlier Command Center build; not replacing it")
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
    data = report.load_data()
    results = {g: ast_metrics.evaluate(g) for g in GAMES}
    pages = {"howitworks.html": report.instructions_page(data)}
    for g in GAMES:
        pages[f"{g}-balance.html"] = report.balance_page(g, data)
        pages[f"{g}-mutations.html"] = ast_page.render(results, only=g, nav=False)
        pages[f"{g}-gasweep.html"] = report.gasweep_page(g, data)
        pages[f"{g}-ablation.html"] = report.ablation_page(g, data)
    walls: dict[str, dict[str, bool]] = {}
    written = []
    for g in GAMES:
        walls[g] = {}
        for p in PERSONAS:
            src = os.path.join(videos_dir, g, f"{p}.webm")
            walls[g][p] = os.path.isfile(src)
            if walls[g][p]:
                dst = os.path.join(out_dir, "videos", g, f"{p}.webm")
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copyfile(src, dst)
                written.append(dst)
    pages["index.html"] = shell(walls)
    pages[".nojekyll"] = ""                       # GitHub Pages: serve the files as they are
    for name, text in pages.items():
        path = os.path.join(out_dir, name)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        written.append(path)
    return written


def serve(out_dir: str, port: int, open_browser: bool) -> None:
    """Static files only, on 127.0.0.1."""
    from functools import partial
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *a) -> None:
            pass
    srv = ThreadingHTTPServer(("127.0.0.1", port), partial(Quiet, directory=out_dir))
    url = f"http://127.0.0.1:{port}/index.html"
    print(f"Command Center: {url}  (Ctrl+C to close)", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


LANDING_TAG = "landing-page-2026-03"   # the gh-pages landing page the Command Center replaced


def _git(repo: str, *args: str, env: dict | None = None, check: bool = True) -> str:
    r = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, env=env)
    if check and r.returncode:
        raise SystemExit(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout.strip()


def publish(site_dir: str = SITE_DIR, push: bool = False, repo: str = ".") -> str:
    """Put the built site on gh-pages as one commit with no history (so the videos never pile up in
    every clone), after keeping the old landing page as tag LANDING_TAG; with `push`, force-push it
    to origin (GitHub Pages serves that branch). Refuses while the results have uncommitted changes,
    so the site only ever shows committed numbers. Returns the new commit."""
    meta = (":(exclude,glob)**/._*", ":(exclude,glob)**/.DS_Store")      # macOS metadata on exFAT, never a result
    dirty = _git(repo, "status", "--porcelain", "--", "runs/balance", "runs/ast", *meta)
    if dirty:
        raise SystemExit(f"uncommitted results; commit them first so the site shows committed numbers:\n{dirty}")
    if not os.path.isfile(os.path.join(site_dir, "index.html")):
        raise SystemExit(f"no site in {site_dir}: build it first (python -m code.neuro.command_center)")
    live = ""                                                            # the gh-pages commit origin serves now
    if push:
        out = _git(repo, "ls-remote", "origin", "refs/heads/gh-pages")
        live = out.split()[0] if out else ""
        if live and not _git(repo, "ls-remote", "origin", f"refs/tags/{LANDING_TAG}"):
            _git(repo, "fetch", "-q", "origin", "refs/heads/gh-pages")      # the live page, not a stale local copy
            _git(repo, "tag", "-f", LANDING_TAG, live)
    if not _git(repo, "rev-parse", "--verify", "--quiet", f"refs/tags/{LANDING_TAG}", check=False):
        old = next((r for r in ("refs/heads/gh-pages", "refs/remotes/origin/gh-pages")
                    if _git(repo, "rev-parse", "--verify", "--quiet", r, check=False)), None)
        if old:
            _git(repo, "tag", LANDING_TAG, old)
    with tempfile.TemporaryDirectory() as tmp:          # a private index: the branch checked out is untouched
        env = dict(os.environ, GIT_INDEX_FILE=os.path.join(tmp, "index"))
        _git(repo, "--work-tree", os.path.abspath(site_dir), "add", "--force", "-A", "--", ":/",
             ":(exclude,glob)**/._*", ":(exclude,glob)**/.DS_Store", env=env)   # macOS metadata on exFAT
        tree = _git(repo, "write-tree", env=env)
    commit = _git(repo, "commit-tree", tree, "-m", f"PEAK Command Center {datetime.now():%Y-%m-%d %H:%M}")
    _git(repo, "update-ref", "refs/heads/gh-pages", commit)
    if push:
        if _git(repo, "rev-parse", "--verify", "--quiet", f"refs/tags/{LANDING_TAG}", check=False) and \
                not _git(repo, "ls-remote", "origin", f"refs/tags/{LANDING_TAG}"):
            _git(repo, "push", "origin", f"refs/tags/{LANDING_TAG}")
        # replaces exactly the commit read above: a page someone pushed meanwhile is never lost
        _git(repo, "push", f"--force-with-lease=refs/heads/gh-pages:{live}", "origin", "refs/heads/gh-pages:refs/heads/gh-pages")
    return commit


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build the PEAK Command Center (a static site)")
    ap.add_argument("--out", default=SITE_DIR)
    ap.add_argument("--videos", default=VIDEOS_DIR, help="recorded replay wall (code/neuro/wall_video.py)")
    ap.add_argument("--open", action="store_true", help="open it in the default browser")
    ap.add_argument("--serve", action="store_true", help="serve it on 127.0.0.1 (static files only)")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--publish", action="store_true", help="also put the site on the gh-pages branch (one commit)")
    ap.add_argument("--push", action="store_true", help="with --publish: force-push gh-pages to origin (goes live)")
    args = ap.parse_args(argv)
    written = build_site(args.out, args.videos)
    print(f"wrote {len(written)} files to {args.out}")
    if args.publish:
        print(f"gh-pages: {publish(args.out, push=args.push)}" + (" (pushed)" if args.push else " (local, not pushed)"))
    if args.serve:
        serve(args.out, args.port, args.open)
    elif args.open:
        webbrowser.open("file://" + os.path.abspath(os.path.join(args.out, "index.html")))


if __name__ == "__main__":
    main()
