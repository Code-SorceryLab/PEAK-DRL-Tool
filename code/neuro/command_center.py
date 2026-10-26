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


def _tile(game: str, persona: str, recorded) -> str:
    """One persona's view: its video with a caption; a click opens it large with its sensors."""
    head = f'<div class="who" style="--c:var(--p-{persona})"><i class="ico p-{persona}"></i>{persona}</div>'
    if not recorded:
        return f'<div class="tile frame missing">{head}<div class="none">not recorded yet</div></div>'
    return (f'<div class="tile frame">{head}<div class="vid" role="button" tabindex="0" data-key="{game}/{persona}" '
            f'aria-label="Watch {NAMES[game]}, {persona} large, with its rays and sensors">'
            f'<video muted loop playsinline preload="none" data-src="videos/{game}/{persona}.webm"></video>'
            f'<div class="cap"></div></div></div>')


def shell(walls: dict[str, dict]) -> str:
    """index.html. walls[game][persona]: that persona's video timeline (or True when it has none),
    falsy when there is no video."""
    games = [g for g in GAMES if g in walls] or GAMES
    timelines = {f"{g}/{p}": v for g in walls for p, v in walls[g].items() if isinstance(v, dict)}
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
<dialog id="big" aria-labelledby="bigtitle">
  <div class="bighead"><h2 id="bigtitle"></h2><span class="bigcap"></span><button class="close" type="button">Close</button></div>
  <div class="bigbody"><div class="stage"><video muted loop playsinline></video><canvas class="rays"></canvas></div>
    <div class="sensors" aria-label="Sensor values"></div></div>
</dialog>
<script>const GAMES = {json.dumps(games)}, NAMES = {json.dumps(NAMES)}, TL = {json.dumps(timelines, separators=(",", ":"))};
{JS}</script></body></html>"""


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
.vid{position:relative;cursor:zoom-in}
.cap{position:absolute;left:6px;bottom:6px;font:700 14px/1.2 var(--pixel);color:var(--ink);background:rgba(0,0,0,.72);padding:3px 7px}
.cap:empty{display:none}
#big{width:min(1240px,96vw);max-height:94vh;padding:0;color:var(--ink);background:var(--deck);
  border:3px solid transparent;border-image:var(--frame-hi) 1 / 3px stretch}
#big::backdrop{background:rgba(0,0,0,.82)}
.bighead{display:flex;align-items:center;gap:14px;padding:10px 14px;border-bottom:3px solid var(--line)}
.bighead h2{flex:1}
.bigcap{font:700 15px var(--pixel);color:var(--won)}
.bigbody{display:grid;grid-template-columns:minmax(0,1fr) 280px;gap:14px;padding:14px}
.stage{position:relative;align-self:start}
.stage video{width:100%;display:block;image-rendering:pixelated;background:#000}
.stage canvas{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
.sensors{display:flex;flex-direction:column;gap:3px;font:600 12.5px/1.3 var(--ui);color:var(--dim);overflow:auto;max-height:76vh}
.sensors h3{font-size:14px;margin:8px 0 2px;color:var(--ink)}
.srow{display:grid;grid-template-columns:44px 1fr 44px;gap:6px;align-items:center}
.sbar{height:8px;background:var(--deck-2);position:relative}
.sbar i{position:absolute;top:0;bottom:0;background:var(--c,var(--peak))}
.srow output{text-align:right}
@media (max-width:820px){.bigbody{grid-template-columns:1fr}}
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

// ── captions and the large view (the old live wall's watched view, from the recorded tracks) ──
const TRACKS = {};
window.PEAK_TRACK = (key, data) => { TRACKS[key] = data; };     // each <video>.sensors.js calls this
const PLATFORM = [   // the platformers' 14-slot ray vector (as code/neuro/web/cameras.html lays it out)
  {g: 'Proximity rays', rows: [{i: 0, l: 'FWD', d: 'Ray straight ahead', inv: true}, {i: 1, l: 'U30', d: 'Ray 30 degrees up', inv: true},
    {i: 2, l: 'U60', d: 'Ray 60 degrees up', inv: true}, {i: 3, l: 'D30', d: 'Ray 30 degrees down', inv: true},
    {i: 4, l: 'D60', d: 'Ray 60 degrees down', inv: true}, {i: 5, l: 'BCK', d: 'Ray straight back', inv: true}]},
  {g: 'Threats', rows: [{i: 6, l: 'NMY', d: 'Enemy ahead', inv: true}, {i: 7, l: 'PIT', d: 'Gap in the floor ahead'}]},
  {g: 'Kinematics', rows: [{i: 9, l: 'VX', d: 'Horizontal velocity', center: true}, {i: 10, l: 'VY', d: 'Vertical velocity', center: true},
    {i: 8, l: 'GND', d: 'On the ground'}, {i: 11, l: 'JMP', d: 'Jump held'}]},
  {g: 'Objectives', rows: [{i: 12, l: 'BLK', d: 'Question blocks ahead'}]},
];
const RAY = {0: 'rgba(255,255,255,.35)', 1: '#ef4444', 2: '#ec4899'};
const levelName = l => /^\\d+$/.test(l) ? 'Level ' + (+l + 1) : l;
function caption(key, frame, total){
  const tl = TL[key];
  if (!tl || !tl.attempts.length) return '';
  const a = tl.attempts;
  let k = 0;
  while (k + 1 < a.length && a[k + 1][0] <= frame) k++;
  const next = k + 1 < a.length ? a[k + 1][0] : total;
  return levelName(a[k][1]) + (a[k][2] && frame >= next - tl.fps ? ' \u00b7 ' + a[k][2] : '');   // the result shows in its hold
}
const big = document.getElementById('big'), bigv = big.querySelector('video'), cv = big.querySelector('canvas');
const sens = big.querySelector('.sensors');
let bigKey = null, raf = 0, rows = [];
function layout(t){
  const groups = t.labels.length ? t.labels : PLATFORM;
  sens.innerHTML = '';
  rows = [];
  for (const g of groups){
    const h = document.createElement('h3'); h.textContent = g.g; sens.appendChild(h);
    for (const r of g.rows){
      const row = document.createElement('div'); row.className = 'srow'; row.title = r.d || '';
      row.innerHTML = '<span></span><div class="sbar"><i></i></div><output></output>';
      row.firstChild.textContent = r.l;
      sens.appendChild(row);
      rows.push({r, bar: row.querySelector('i'), out: row.querySelector('output')});
    }
  }
  sens.dataset.key = bigKey;
}
function draw(){
  if (!big.open) return;
  const fps = (TL[bigKey] || {}).fps || 30, frame = Math.floor(bigv.currentTime * fps), t = TRACKS[bigKey];
  big.querySelector('.bigcap').textContent = caption(bigKey, frame, (bigv.duration || 0) * fps);
  if (t){
    if (sens.dataset.key !== bigKey) layout(t);
    const [rays, vals] = t.frames[Math.min(t.frames.length - 1, Math.floor(frame / t.every))] || [[], []];
    const w = bigv.clientWidth, h = bigv.clientHeight, k = w / t.width, ctx = cv.getContext('2d');
    if (cv.width !== w || cv.height !== h){ cv.width = w; cv.height = h; }
    ctx.clearRect(0, 0, w, h);
    ctx.lineWidth = 2;
    for (const [x1, y1, x2, y2, hit] of rays){
      ctx.strokeStyle = RAY[hit] || RAY[0];
      ctx.beginPath(); ctx.moveTo(x1 * k, y1 * k); ctx.lineTo(x2 * k, y2 * k); ctx.stroke();
    }
    for (const {r, bar, out} of rows){
      const v = Math.max(-1, Math.min(1, vals[r.i] ?? 0));
      if (r.center){ bar.style.left = (50 + Math.min(0, v) * 50) + '%'; bar.style.width = Math.abs(v) * 50 + '%'; }
      else { bar.style.left = '0'; bar.style.width = Math.max(0, r.inv ? 1 - v : v) * 100 + '%'; }
      out.textContent = (vals[r.i] ?? 0).toFixed(2);
    }
  }
  raf = requestAnimationFrame(draw);
}
function openBig(tile){
  const key = tile.dataset.key, v = tile.querySelector('video'), [g, p] = key.split('/');
  bigKey = key;
  big.querySelector('h2').textContent = NAMES[g] + ', ' + p;
  bigv.src = v.dataset.src;
  bigv.currentTime = v.currentTime || 0;
  bigv.play().catch(() => {});
  if (!TRACKS[key]){
    sens.textContent = 'Loading the sensors...';
    sens.dataset.key = '';
    const s = document.createElement('script');
    s.src = 'videos/' + key + '.sensors.js';
    s.onerror = () => { sens.textContent = 'No sensor track for this video.'; };
    document.head.appendChild(s);
  }
  big.showModal();
  cancelAnimationFrame(raf);
  raf = requestAnimationFrame(draw);
}
big.addEventListener('click', e => { if (e.target === big) big.close(); });     // a click outside the frame
big.querySelector('.close').addEventListener('click', () => big.close());
big.addEventListener('close', () => { cancelAnimationFrame(raf); stop(bigv); bigKey = null; });
for (const tile of document.querySelectorAll('.vid')){
  const v = tile.querySelector('video'), cap = tile.querySelector('.cap'), fps = (TL[tile.dataset.key] || {}).fps || 30;
  tile.addEventListener('click', () => openBig(tile));
  tile.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' '){ e.preventDefault(); openBig(tile); } });
  v.addEventListener('timeupdate', () => { cap.textContent = caption(tile.dataset.key, Math.floor(v.currentTime * fps), (v.duration || 0) * fps); });
}
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
    walls: dict[str, dict] = {}
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
                timeline, track = src[:-len(".webm")] + ".json", src[:-len(".webm")] + ".sensors.json"
                if os.path.isfile(timeline):
                    with open(timeline, encoding="utf-8") as f:
                        walls[g][p] = json.load(f)
                if os.path.isfile(track):      # a script, not JSON: a page opened from disk may not fetch files
                    with open(track, encoding="utf-8") as f:
                        data = json.dumps(json.load(f), separators=(",", ":"))
                    path = dst[:-len(".webm")] + ".sensors.js"
                    with open(path, "w", encoding="utf-8", newline="\n") as f:
                        f.write(f'PEAK_TRACK("{g}/{p}",{data});\n')
                    written.append(path)
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
