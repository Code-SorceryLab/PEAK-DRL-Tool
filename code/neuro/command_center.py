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
import glob
import html
import json
import os
import shutil
import subprocess
import tempfile
import webbrowser
import zipfile
from datetime import datetime

from . import report
from .balance import PAPER_SEEDS
from .cameras import ORDER as PERSONAS

SITE_DIR = os.path.join("runs", "site")
VIDEOS_DIR = os.path.join("runs", "videos")
GAMES = ["mario", "meatboy", "bomberman", "megaman"]
NAMES = {"mario": "Mario", "meatboy": "Meat Boy", "bomberman": "Bomberman", "megaman": "Mega Man"}
TABS = [("wall", "Replay wall"), ("balance", "Balance"), ("mutations", "Mutations"), ("gasweep", "GA sweep"),
        ("ablation", "Sensor ablation"), ("reproduce", "Reproduce"), ("howitworks", "How it works")]
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
  const result = a[k][2] === 'cut' ? 'cut short for the video' : a[k][2];
  return levelName(a[k][1]) + (result && frame >= next - tl.fps ? ' \u00b7 ' + result : '');   // the result shows in its hold
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
  cv.getContext('2d').clearRect(0, 0, cv.width, cv.height);      // no rays left from the last video
  if (!TRACKS[key]){
    sens.textContent = 'Loading the sensors...';
    sens.dataset.key = '';
    const s = document.createElement('script');
    s.src = 'videos/' + key + '.sensors.js';
    s.onerror = () => { if (bigKey === key) sens.textContent = 'No sensor track for this video.'; };   // still this video
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


STATE_KEYS = ("seed", "sensors", "hidden", "action_feedback", "memory", "pop_size", "max_frames")   # all a replay reads


def _agent_dirs(game: str):
    """(source, variant, persona, tag, level, seed, run dir) of every paper-seed agent of the game under runs/:
    the balance probes and the AST evaluation's original and mutant agents."""
    keep = {str(s) for s in PAPER_SEEDS}
    found = [("probes", None, p) for p in glob.glob(os.path.join("runs", "probes", game, "*", "*", "*", "best.npz"))]
    found += [("ast", p.split(os.sep)[3], p) for p in glob.glob(os.path.join("runs", "ast", game, "*", "probes", game, "*", "*", "*",
                                                                                "best.npz"))]
    for source, variant, path in sorted(found, key=lambda f: f[2]):
        d = os.path.dirname(path)
        level, seed = os.path.basename(d).rsplit("_", 1)
        if seed in keep and d.split(os.sep)[-3] in PERSONAS and os.path.isfile(os.path.join(d, "state.json")):
            yield source, variant, d.split(os.sep)[-3], d.split(os.sep)[-2], level, int(seed), d


def host_agents(dest: str, games: list[str] = GAMES) -> list[dict]:
    """Copy every paper-seed agent to dest/<its path under runs/>/ as best.npz plus a slim state.json (the
    keys a replay reads, no training history), so `trainer --replay <dir>/best.npz` works on a download;
    one zip per game. Returns one row per agent for the Reproduce tab."""
    rows = []
    os.makedirs(dest, exist_ok=True)
    for game in games:
        with zipfile.ZipFile(os.path.join(dest, f"{game}.zip"), "w", zipfile.ZIP_DEFLATED) as z:
            for source, variant, persona, tag, level, seed, d in _agent_dirs(game):
                rel = os.path.relpath(d, "runs")
                out = os.path.join(dest, rel)
                os.makedirs(out, exist_ok=True)
                with open(os.path.join(d, "state.json"), encoding="utf-8") as f:
                    st = json.load(f)
                slim = {"persona": st.get("persona") or persona, "best_level": st.get("best_level") or level,
                        "best_fitness": st.get("best_fitness"),
                        "config": {k: v for k, v in (st.get("config") or {}).items() if k in STATE_KEYS}}
                shutil.copyfile(os.path.join(d, "best.npz"), os.path.join(out, "best.npz"))
                with open(os.path.join(out, "state.json"), "w", encoding="utf-8") as f:
                    json.dump(slim, f, separators=(",", ":"))
                for name in ("best.npz", "state.json"):
                    z.write(os.path.join(out, name), os.path.join(rel, name))
                rows.append({"game": game, "source": source, "variant": variant, "persona": persona, "tag": tag,
                             "level": level, "seed": seed, "path": "agents/" + rel.replace(os.sep, "/"),
                             "best_fitness": slim["best_fitness"]})
    return rows


def _replay(a: dict) -> str:
    """The command that watches this agent play, on the mutant it trained on when it has one."""
    mutant = (f"PEAK_MUTANT=code/games/levels/mutants/{a['game']}/{a['variant']} "
              if a["source"] == "ast" and a["variant"] not in ("base", "base-retrained") else "")
    return f"{mutant}python -m code.neuro.trainer --game {a['game']} --replay {a['path']}/best.npz"


def _retrain(a: dict) -> str:
    """The command that trains this agent again, seed and budget as it had them."""
    if a["source"] == "ast":
        return (f"python -m code.neuro.ast_eval retrain --game {a['game']} --mutant {a['variant']} --persona {a['persona']}"
                + (f" --level {a['level']}" if a["game"] == "megaman" else ""))
    gens = a["tag"].split("g", 1)[1].split("_", 1)[0]
    return (f"python -m code.neuro.balance --game {a['game']} --persona {a['persona']} --levels {a['level']} "
            f"--seeds {a['seed']} --gens {gens}" + (" --best" if a["tag"].endswith("_best") else ""))


def reproduce_page(game: str, agents: list[dict]) -> str:
    """The Reproduce tab of one game: set up, download the agents, replay one, retrain it, rebuild the results."""
    rows = [a for a in agents if a["game"] == game]
    source = lambda a: f"AST {a['variant']}" if a["source"] == "ast" else f"probe {a['tag']}"   # noqa: E731
    personas = sorted({a["persona"] for a in rows})
    sources = sorted({source(a) for a in rows})
    table = "".join(
        f'<tr data-p="{esc(a["persona"])}" data-s="{esc(source(a))}"><td>{esc(source(a))}</td><td>{esc(a["persona"])}</td>'
        f'<td>{esc(a["level"])}</td><td class="n">{a["seed"]}</td>'
        f'<td><a href="{esc(a["path"])}/best.npz" download>best.npz</a> <a href="{esc(a["path"])}/state.json" download>state.json</a></td>'
        f'<td><code>{esc(_replay(a))}</code></td>'
        f"<td><code>{esc(_retrain(a))}</code></td></tr>" for a in rows)
    pick = lambda name, vals: (f'<label>{name} <select data-f="{name[0].lower()}"><option value="">all</option>'   # noqa: E731
                               + "".join(f"<option>{esc(v)}</option>" for v in vals) + "</select></label>")
    cmds = [(f"python -m code.neuro.ast_eval play --game {game}", "playtests: one attempt per agent"),
            (f"python -m code.neuro.ast_eval play --game {game} --replay", "the original's frozen agents on each mutant"),
            ("python -m code.stats.ast_metrics --page", "the mutation metrics, and this site"),
            ("python -m code.stats.ast_paper", "the paper's tables and figures"),
            ("python -m code.neuro.command_center --open", "this Command Center, on your machine")]
    width = max(len(c) for c, _ in cmds) + 2
    rebuild = "\n".join(esc(f"{c.ljust(width)}# {why}") for c, why in cmds)
    sha = _git(".", "rev-parse", "HEAD", check=False)       # a copy outside git (a cluster tar) has none to name
    checkout = f"\ngit checkout {sha}" if sha else ""
    steps = f"""
<h2><i class="ico sprout"></i>Reproduce: {esc(NAMES[game])}</h2>
<p class="lede">Every agent below was trained with one of the seeds {", ".join(map(str, PAPER_SEEDS))}, the seeds
behind the published numbers. Each one is a folder with its network (best.npz) and the settings it trained with
(state.json). Levels are named as the game's files and the commands name them (0 is the first level of an indexed game).</p>
<h3>1. Set up</h3>
<pre><code>git clone https://github.com/Code-SorceryLab/PEAK-DRL-Tool.git
cd PEAK-DRL-Tool{checkout}
pip install -r requirements.txt</code></pre>
<h3>2. Get the agents</h3>
<p>All {len(rows)} agents of this game: <a href="agents/{game}.zip" download>agents/{game}.zip</a>.
Unpack it into the repository: <code>unzip {game}.zip -d agents</code>. Or take one agent's two files from the table.</p>
<h3>3. Replay an agent</h3>
<p>The replay command in the table opens the live dashboard at <code>http://127.0.0.1:8000/</code> and plays the
agent with its own persona, reaction time and seed. An agent trained on a mutant replays on that mutant: its command
sets <code>PEAK_MUTANT</code> first (on Windows, run <code>set PEAK_MUTANT=...</code> on its own line before it).</p>
<h3>4. Train it again</h3>
<p>The retrain command trains the same persona, level, seed and generation budget from scratch. The AST
evaluation's agents retrain all three seeds on their mutant at once.</p>
<h3>5. Rebuild the results</h3>
<pre><code>{rebuild}</code></pre>
<h3>Every agent</h3>
<div class="filters">{pick("Persona", personas)} {pick("Source", sources)}</div>
<div class="ovwrap"><table class="agents"><thead><tr><th>Source</th><th>Persona</th><th>Level</th><th class="n">Seed</th>
<th>Files</th><th>Replay</th><th>Retrain</th></tr></thead><tbody>{table}</tbody></table></div>"""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PEAK Command Center: {esc(NAMES[game])} reproduce</title><style>{report.CSS}{REPRO_CSS}</style></head><body>
<main id="main" class="repro">{steps}</main>
<script>
const sel = [...document.querySelectorAll('.filters select')];
for (const s of sel) s.addEventListener('change', () => {{
  const [p, src] = sel.map(x => x.value);
  for (const tr of document.querySelectorAll('table.agents tbody tr'))
    tr.hidden = (p && tr.dataset.p !== p) || (src && tr.dataset.s !== src);
}});
</script></body></html>"""


REPRO_CSS = """
[hidden]{display:none !important}
.repro{padding:18px 0;display:flex;flex-direction:column;gap:10px}
.repro h3{margin-top:10px}
.repro pre{background:var(--deck);border:3px solid transparent;border-image:var(--frame) 1 / 3px stretch;padding:10px 12px;overflow:auto}
.repro .lede{color:var(--dim);max-width:80ch}
.filters{display:flex;gap:14px;flex-wrap:wrap;font:600 .9rem var(--ui)}
.filters select{background:var(--deck-2);color:var(--ink);border:2px solid var(--line);padding:2px 6px}
table.agents{border-collapse:collapse;font-size:.82rem}
table.agents td,table.agents th{padding:4px 8px;border-bottom:1px solid var(--line);vertical-align:top;text-align:left}
table.agents code{font-size:.78rem;white-space:nowrap}
"""


def build_site(out_dir: str = SITE_DIR, videos_dir: str = VIDEOS_DIR) -> list[str]:
    """Write the whole site to out_dir (replacing an earlier build there) and return the paths written."""
    from code.stats import ast_metrics, ast_page
    if os.path.isdir(out_dir) and os.listdir(out_dir) and not os.path.exists(os.path.join(out_dir, ".nojekyll")):
        raise SystemExit(f"{out_dir} is not an earlier Command Center build; not replacing it")
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
    nojekyll = os.path.join(out_dir, ".nojekyll")     # GitHub Pages: serve the files as they are
    open(nojekyll, "w").close()                       # first, so a build that fails part way is still replaced
    data = report.load_data()
    results = {g: ast_metrics.evaluate(g) for g in GAMES}
    pages = {"howitworks.html": report.instructions_page(data)}
    agents = host_agents(os.path.join(out_dir, "agents"))
    for g in GAMES:
        pages[f"{g}-reproduce.html"] = reproduce_page(g, agents)
        pages[f"{g}-balance.html"] = report.balance_page(g, data)
        pages[f"{g}-mutations.html"] = ast_page.render(results, only=g, nav=False)
        pages[f"{g}-gasweep.html"] = report.gasweep_page(g, data)
        pages[f"{g}-ablation.html"] = report.ablation_page(g, data)
    walls: dict[str, dict] = {}
    written = [nojekyll] + [os.path.join(out_dir, "agents", f"{g}.zip") for g in GAMES]
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
    so the site only ever shows committed numbers and agents, and while HEAD is on no remote branch,
    since the Reproduce tab tells readers to check it out. Returns the new commit."""
    meta = (":(exclude,glob)**/._*", ":(exclude,glob)**/.DS_Store")      # macOS metadata on exFAT, never a result
    dirty = _git(repo, "status", "--porcelain", "--", "runs/balance", "runs/ast", "runs/probes",
                 ":(glob)runs/*/state.json", *meta)                  # every run the How it works tab lists
    if dirty:
        raise SystemExit(f"uncommitted results; commit them first so the site shows committed numbers:\n{dirty}")
    if not os.path.isfile(os.path.join(site_dir, "index.html")):
        raise SystemExit(f"no site in {site_dir}: build it first (python -m code.neuro.command_center)")
    sha = _git(repo, "rev-parse", "HEAD")
    if not _git(repo, "branch", "-r", "--contains", sha):
        raise SystemExit(f"commit {sha} is on no remote branch, so readers cannot check out what the Reproduce tab "
                         "names; push it first")
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
    commit = _git(repo, "commit-tree", tree, "-m", f"PEAK Command Center {datetime.now():%Y-%m-%d %H:%M} from {sha}")
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
