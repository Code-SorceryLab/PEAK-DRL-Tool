"""Pipeline pages: one local site that walks a game's pipeline from start to finish.

    python -m code.pipeline --open            # http://127.0.0.1:8003/
    python -m code.pipeline --game castlevania --open

The games are the ones played as a stage in sections (Mega Man, Castlevania); the links at the
top of every page switch between them.

    Design  ->  Mutate | Train | Play  ->  Measure

One page per stage, linked in that order, and a hub that shows where each stage stands:

    Design    the stage map with its sections outlined (read-only; edit in the level editor)
    Mutate    the game mutants (code/games/mutants.py): what each changes and its class
    Train     per section, the runs on disk and the one the relay uses; start runs, re-pick
    Play      agents (a batch, or live on the trainer's dashboard) and a person at the keyboard
    Measure   the telemetry page (code/stats/telemetry.py) over the agents' and the person's logs

Training data stays on the Train page and playtest results on the Measure page. Everything a
button starts is a job: a subprocess whose last line of output shows on every page until it ends.
The site binds 127.0.0.1 only, starts nothing but the commands below with levels the game lists,
and stops every job it started when it is closed.
"""
from __future__ import annotations

import argparse
import glob
import html
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.parse

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from code.games import mutants  # noqa: E402
from code.neuro import relay  # noqa: E402
from code.stats import episode_log, telemetry  # noqa: E402

GAMES = {"megaman": "Mega Man", "castlevania": "Castlevania"}   # the games played as a stage in sections
HUMAN_DIR = os.path.join("runs", "manual")                  # manual play writes <dir>/<game>/<player>/episodes.csv
DASH_PORT = 8000


def use_game(game: str) -> str:
    """Point every page at this game. Returns the level id of its full stage."""
    global GAME, RUNS, MANIFEST, RELAY_OUT, JOB_LOGS
    stage = relay.set_game(game)
    GAME = game
    RUNS = os.path.join("runs", GAME)
    MANIFEST = os.path.join(RUNS, "relay.json")
    RELAY_OUT = os.path.join(RUNS, "relay")
    JOB_LOGS = os.path.join(RUNS, "pipeline_jobs")
    return stage


use_game(relay.GAME)
STAGES = [("design", "Design"), ("mutate", "Mutate"), ("train", "Train"), ("play", "Play"), ("measure", "Measure")]
esc = html.escape


# ── what is on disk ──────────────────────────────────────────────────────────────────────────
def sections(stage: str) -> list[relay.Leg]:
    return relay.stage_legs(stage)


def level_entries() -> dict:
    import yaml
    with open(os.path.join(relay.GAMES_DIR, relay.OWN_LEVEL_FILES[GAME]), encoding="utf-8") as f:
        return (yaml.safe_load(f) or {}).get("levels") or {}


def manifest() -> dict:
    try:
        with open(MANIFEST, encoding="utf-8") as f:
            return json.load(f).get("legs") or {}
    except (OSError, ValueError):
        return {}


def section_runs(section: str) -> list[dict]:
    found = [s for s in map(relay.run_summary, sorted(glob.glob(os.path.join(RUNS, section, "*")))) if s]
    return sorted(found, key=lambda s: (s["win_rate"], s["best"]), reverse=True)


def agent_log(stage: str, level: str) -> str:
    return os.path.join(RELAY_OUT, "episodes.csv") if level == stage else \
        os.path.join(RELAY_OUT, "sections", level, "episodes.csv")


def tally(rows: list[dict]) -> dict:
    return {"n": len(rows), "wins": sum(1 for r in rows if r.get("status") == "WON")}


def agent_results(stage: str, level: str) -> dict | None:
    path = agent_log(stage, level)
    if not os.path.exists(path):
        return None
    return dict(tally(episode_log.read(path)), when=os.path.getmtime(path))


def human_results() -> dict[str, dict]:
    """Per level: how many attempts people have recorded, and how many they won."""
    by: dict[str, list] = {}
    for path in glob.glob(os.path.join(HUMAN_DIR, GAME, "*", "episodes.csv")):
        for r in episode_log.read(path):
            by.setdefault(str(r.get("level")), []).append(r)
    return {level: tally(rows) for level, rows in by.items()}


def measure_paths() -> list[str]:
    return [p for p in (RELAY_OUT, os.path.join(HUMAN_DIR, GAME)) if os.path.isdir(p)]


# ── re-picking one section (what `relay --pick` does for all of them) ────────────────────────────
def pick_section(section: str, stage: str, attempts: int = 12, shortlist: int = 6) -> None:
    if section not in {leg.level for leg in sections(stage)}:
        raise SystemExit(f"'{section}' is not a section of '{stage}'")
    found = section_runs(section)
    if not found:
        raise SystemExit(f"no trained run for '{section}' under {RUNS}")
    current = (manifest().get(section) or {}).get("run")
    tested = found[:shortlist] + [s for s in found[shortlist:] if s["run"] == current]
    print(f"{section}: testing {len(tested)} of {len(found)} runs over {attempts} varied attempts", flush=True)
    for s in tested:
        s["tested_win_rate"], s["tested_progress"], s["tested_fitness"] = relay.evaluate(s["run"], section, attempts)
        s["candidates"] = len(found)
        print(f"  {os.path.basename(s['run']):<12} training wins {100 * s['win_rate']:>3.0f}%   "
              f"tested wins {100 * s['tested_win_rate']:>3.0f}%   tested reach {100 * s['tested_progress']:>3.0f}%"
              f"{'   (current pick)' if s['run'] == current else ''}", flush=True)
    best = max(tested, key=lambda s: (s["tested_win_rate"], s["tested_progress"], s["tested_fitness"]))
    try:
        with open(MANIFEST, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        data = {"game": GAME, "stage": stage, "legs": {}}
    data.setdefault("legs", {})[section] = best
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"picked {best['run']} -> {MANIFEST}", flush=True)


# ── jobs: everything a button starts ─────────────────────────────────────────────────────────────
class Jobs:
    def __init__(self) -> None:
        self.items: list[dict] = []
        self.lock = threading.Lock()

    def start(self, label: str, cmd: list[str], group: str = "", window: bool = False) -> dict:
        """Run cmd; a job in a `group` replaces the one already running in it."""
        with self.lock:
            if group:
                for j in self.items:
                    if j["group"] == group:
                        self._stop(j)
            os.makedirs(JOB_LOGS, exist_ok=True)
            n = len(self.items) + 1
            log = os.path.join(JOB_LOGS, f"{time.strftime('%Y%m%d-%H%M%S')}-{n}.log")
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            if window:
                env.pop("SDL_VIDEODRIVER", None)        # a real window, not the dummy driver
            out = open(log, "w", encoding="utf-8")
            job = {"id": n, "label": label, "group": group, "log": log, "started": time.time(),
                   "proc": subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, env=env)}
            out.close()
            self.items.append(job)
            print(f"job {n}: {label}", flush=True)
            return job

    @staticmethod
    def _stop(job: dict) -> None:
        proc = job["proc"]
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            job["stopped"] = True

    def stop(self, job_id: int) -> None:
        with self.lock:
            for j in self.items:
                if j["id"] == job_id:
                    self._stop(j)

    def stop_all(self) -> None:
        with self.lock:
            for j in self.items:
                self._stop(j)

    def running(self, group: str) -> bool:
        return any(j["group"] == group and j["proc"].poll() is None for j in self.items)

    def view(self) -> list[dict]:
        out = []
        for j in reversed(self.items[-30:]):
            code = j["proc"].poll()
            status = "running" if code is None else "stopped" if j.get("stopped") else "done" if code == 0 else "failed"
            out.append({"id": j["id"], "label": j["label"], "status": status, "line": _last_line(j["log"]),
                        "seconds": round(time.time() - j["started"]) if code is None else None})
        return out


def _last_line(path: str) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 2000))
            lines = [ln for ln in f.read().decode("utf-8", "replace").splitlines()
                     if ln.strip() and not ln.startswith(("pygame ", "Hello from the pygame"))]
        return lines[-1].strip()[:200] if lines else ""
    except OSError:
        return ""


# ── pages ────────────────────────────────────────────────────────────────────────────────────
EXTRA_CSS = """
.pipe{display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin:0 0 18px;font-size:13px}
.pipe a{padding:5px 10px;border:1px solid var(--line);border-radius:6px;background:var(--panel);color:var(--text);text-decoration:none}
.pipe a.cur{border-color:var(--s1);box-shadow:inset 0 -2px 0 var(--s1);font-weight:600}
.pipe .sep{color:var(--muted)}.pipe .home{font-weight:600}
.flow{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px;margin-bottom:16px}
.stage{display:block;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px 14px;color:var(--text);text-decoration:none}
.stage:hover{border-color:var(--s1)}.stage .num{color:var(--muted);font-size:12px}.stage h2{font-size:16px;margin:2px 0 8px}
.stage p{margin:3px 0;color:var(--text2);font-size:13px}.stage .todo{color:var(--bad)}
form.inline{display:inline-flex;gap:6px;align-items:center;flex-wrap:wrap;margin:0}
button,.btn{font:inherit;font-size:13px;padding:4px 10px;border:1px solid var(--line);border-radius:6px;background:var(--panel);color:var(--text);cursor:pointer;text-decoration:none;white-space:nowrap}
button.primary{background:var(--s1);border-color:var(--s1);color:#fff;font-weight:600}
input[type=number],input[type=text]{font:inherit;font-size:13px;padding:4px 6px;border:1px solid var(--line);border-radius:6px;background:var(--panel);color:var(--text)}
input[type=number]{width:64px}td.act{white-space:normal}td.act form{margin:2px 4px 2px 0}
code{font-size:12px;background:var(--surface);border:1px solid var(--line);border-radius:4px;padding:1px 5px}
pre{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:10px;overflow-x:auto;font-size:12px;margin:8px 0 0}
.muted{color:var(--muted)}.good{color:var(--good)}.bad{color:var(--bad)}
#jobs .job{display:grid;grid-template-columns:84px 1fr auto;gap:10px;align-items:baseline;padding:6px 0;border-bottom:1px solid var(--line)}
#jobs .job:last-child{border-bottom:0}#jobs .st{font-weight:600;font-size:12px}#jobs .ln{color:var(--text2);font-size:12px;overflow-wrap:anywhere}
#jobs .running{color:var(--s1)}#jobs .failed{color:var(--bad)}#jobs .done{color:var(--good)}#jobs .stopped{color:var(--muted)}
.sec-label{font:600 13px system-ui,sans-serif;fill:var(--text);paint-order:stroke;stroke:var(--surface);stroke-width:4px}
"""
CSS = re.search(r"<style>(.*?)</style>", telemetry.PAGE, re.S).group(1) + EXTRA_CSS     # the telemetry page's look

JOBS_HTML = '<div class="card"><h2>Jobs</h2><div id="jobs"><p class="empty">Nothing started from these pages yet.</p></div></div>'
JOBS_JS = r"""
const E = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
let lastJobs = "";
async function jobs(){
  const box = document.getElementById("jobs"); if (!box) return;
  try {
    const list = await (await fetch("/jobs", {cache: "no-store"})).json(), sig = JSON.stringify(list.map(j => [j.id, j.status]));
    if (list.length) box.innerHTML = list.map(j => `<div class="job"><span class="st ${j.status}">${j.status}${j.seconds != null ? " · " + j.seconds + " s" : ""}</span>` +
      `<span><b>${E(j.label)}</b><br><span class="ln">${E(j.line)}</span></span>` +
      (j.status == "running" ? `<form class="inline" method="post" action="/do"><input type="hidden" name="action" value="stop"><input type="hidden" name="job" value="${j.id}"><input type="hidden" name="back" value="${E(location.pathname)}"><button>Stop</button></form>` : "<span></span>") + `</div>`).join("");
    if (lastJobs && sig != lastJobs && document.body.dataset.reload && !document.querySelector("input:focus")) { location.reload(); return; }
    lastJobs = sig;
  } catch {}
  setTimeout(jobs, 2000);
}
jobs();
const who = document.getElementById("player");
if (who){ try { who.value = localStorage.getItem("peak-player") || who.value; } catch {}
  document.querySelectorAll("form.human").forEach(f => f.addEventListener("submit", () => {
    f.querySelector("[name=player]").value = who.value; f.querySelector("[name=warmup]").value = document.getElementById("warmup").value;
    try { localStorage.setItem("peak-player", who.value); } catch {} })); }
"""


def nav(cur: str) -> str:
    links = [f'<a class="home{" cur" if cur == "" else ""}" href="/">Pipeline</a>']
    for i, (key, name) in enumerate(STAGES):
        links.append(f'<a class="{"cur" if cur == key else ""}" href="/{key}">{i + 1}. {name}</a>')
    games = "".join(f'<a class="{"cur" if g == GAME else ""}" href="/game?g={g}">{esc(name)}</a>' for g, name in GAMES.items())
    return '<nav class="pipe" aria-label="Pipeline stages">' + links[0] + '<span class="sep">·</span>' + \
        '<span class="sep">→</span>'.join(links[1:]) + f'<span class="sep" style="margin-left:auto">Game</span>{games}</nav>'


def page(cur: str, title: str, sub: str, body: str, reload_on_change: bool = False) -> str:
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>PEAK pipeline — {esc(title)}</title><style>{CSS}</style></head>"
            f'<body{" data-reload=1" if reload_on_change else ""}><main>{nav(cur)}<h1>{esc(title)}</h1><p class="sub">{sub}</p>'
            f"{body}{JOBS_HTML}</main><script>{JOBS_JS}</script></body></html>")


def post(action: str, back: str, label: str, fields: dict | None = None, extra: str = "", cls: str = "",
         primary: bool = False, new_tab: bool = False) -> str:
    hidden = "".join(f'<input type="hidden" name="{esc(k)}" value="{esc(str(v))}">'
                     for k, v in {"action": action, "back": back, **(fields or {})}.items())
    return (f'<form class="inline {cls}" method="post" action="/do"{" target=_blank" if new_tab else ""}>{hidden}{extra}'
            f'<button{" class=primary" if primary else ""}>{label}</button></form>')


def when(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))


def hub(stage: str) -> str:
    legs, picks, people = sections(stage), manifest(), human_results()
    full = agent_results(stage, stage)
    alone = [agent_results(stage, leg.level) for leg in legs]
    n_runs = sum(len(glob.glob(os.path.join(RUNS, leg.level, "*", "best.npz"))) for leg in legs)
    paths = measure_paths()
    data = telemetry.build(paths) if paths else {"episodes": 0, "groups": []}
    cards = [
        ("design", "Design", [f"Stage <b>{esc(stage)}</b>, {len(legs)} sections",
                              f"{sum(len(leg.screens) for leg in legs)} screens"]),
        ("mutate", "Mutate", mutate_status()),
        ("train", "Train", [f"{n_runs} runs on disk", f"{sum(1 for leg in legs if leg.level in picks)} of {len(legs)} sections have a pick"]),
        ("play", "Play", [f"Full stage, agents: <b>{full['wins']} of {full['n']}</b> won" if full else "No full-stage batch yet",
                          f"Sections played alone: {sum(1 for a in alone if a)} of {len(legs)}",
                          f"Human attempts: {sum(v['n'] for v in people.values())}"]),
        ("measure", "Measure", [f"{data['episodes']} episodes", f"{len(data['groups'])} level and player groups"]),
    ]
    flow = '<div class="flow">' + "".join(
        f'<a class="stage" href="/{key}"><div class="num">Stage {i + 1}</div><h2>{name}</h2>' +
        "".join(f"<p>{ln}</p>" for ln in lines) + "</a>" for i, (key, name, lines) in enumerate(cards)) + "</div>"
    return page("", "PEAK pipeline", f"Game <code>{GAME}</code>, stage <code>{esc(stage)}</code>. "
                "Design the level, then mutate, train and play it, then measure.", flow)


def design(stage: str) -> str:
    legs, entries = sections(stage), level_entries()
    path = telemetry.level_file(GAME, stage)
    geo = telemetry.geometry(path, 1, GAME)
    up, lethal = ("Stairs", "Water") if GAME == "castlevania" else ("Ladder", "Spikes")
    body = ""
    if geo:
        st = mutants.Stage(stage, GAME)
        sc, sr = st.sc, st.sr                             # a screen, in tiles (the game's config)
        svg = [f'<svg class="map" viewBox="0 0 {geo["cols"]} {geo["rows"]}" role="img" aria-label="Stage map with its sections outlined">']
        svg += [f'<rect x="{c}" y="{r}" width="{n}" height="1" fill="var(--geo)"/>' for c, r, n in geo["solid"]]
        svg += [f'<rect x="{c + .3}" y="{r}" width="{n - .6}" height="1" fill="var(--geo2)"/>' for c, r, n in geo["ladder"]]
        svg += [f'<rect x="{c}" y="{r + .5}" width="{n}" height=".5" fill="var(--s2)"/>' for c, r, n in geo["spike"]]
        for i, leg in enumerate(legs):
            for sx, sy in sorted(leg.screens):
                svg.append(f'<rect x="{sx * sc + .2}" y="{sy * sr + .2}" width="{sc - .4}" height="{sr - .4}" fill="none" '
                           f'stroke="var(--s1)" stroke-width=".35" stroke-opacity=".8"/>')
            sx, sy = min(leg.screens, key=lambda s: (s[1], s[0]))
            svg.append(f'<text class="sec-label" x="{sx * sc + 1}" y="{sy * sr + 3.2}" style="font-size:2.6px;stroke-width:.8px">{i + 1}</text>')
            start = (entries.get(leg.level) or {}).get("section", {}).get("start")
            if start:
                svg.append(f'<circle cx="{start[0] + .5}" cy="{start[1] + .5}" r=".9" fill="var(--s3)" stroke="var(--surface)" stroke-width=".25"/>')
        body += ('<div class="card"><h2>Stage map</h2><div class="legend"><span><i class="line" style="background:var(--s1)"></i>Section (numbered)</span>'
                 '<span><i style="background:var(--s3)"></i>Section start</span><span><i style="background:var(--geo2)"></i>' + up + '</span>'
                 '<span><i style="background:var(--s2)"></i>' + lethal + '</span></div>' + "".join(svg) + "</svg></div>")
    rows = ""
    for i, leg in enumerate(legs):
        sec = (entries.get(leg.level) or {}).get("section") or {}
        rows += (f"<tr><td>{i + 1}</td><td class='l'>{esc(leg.level)}</td><td>{len(leg.screens)}</td>"
                 f"<td>{tuple(sec.get('start') or ()) or '—'}</td><td class='l'>{('on the ' + up.lower()) if sec.get('climbing') else 'on foot'}</td>"
                 f"<td>{tuple(sec.get('end') or ()) or 'stage goal'}</td></tr>")
    body += ('<div class="card"><h2>Sections</h2><div class="scroll"><table><tr><th>#</th><th class="l">Section</th><th>Screens</th>'
             '<th>Start tile</th><th class="l">Entered</th><th>Ends at tile</th></tr>' + rows + "</table></div></div>")
    body += ('<div class="card"><h2>Editing</h2><p>This page is read-only. The stage is the file '
             f"<code>{esc(os.path.relpath(path) if path else '?')}</code>; its sections are listed in "
             f"<code>code/games/{relay.OWN_LEVEL_FILES[GAME]}</code>.</p>"
             + ("<pre>python code/games/tools/level_editor.py        # edit the stage\n"
                "python -m code.games.tools.mm1_make_slices     # cut it into sections again</pre></div>" if GAME == "megaman" else
                "<p>The stage file is plain text and can be edited by hand; the sections are edited in the level list. "
                "To rebuild the stage from the reference maps:</p>"
                "<pre>python -m code.games.tools.cv_extract_level</pre></div>"))
    return page("design", "1. Design", "The stage as the agents and players get it.", body)


def mutate_status() -> list[str]:
    index = mutants.load(game_name=GAME)
    if not index:
        return ['<span class="todo">No mutants generated yet</span>']
    by: dict = {}
    for m in index["mutants"]:
        by[m["class"]] = by.get(m["class"], 0) + 1
    return [f"<b>{len(index['mutants'])}</b> mutants", ", ".join(f"{v} {k}" for k, v in sorted(by.items()))]


def mutant_map(m: dict, st) -> str:
    """The stage around a level mutant's change, before and after, the changed cells outlined."""
    c0, r0 = m["location"]["col"], m["location"]["row"]
    x0, y0, w, h = max(0, c0 - 12), max(0, r0 - 8), 25, 17
    after = {(c, r): ch for c, r, _, ch in m["cells"]}

    def draw(changed: dict) -> str:
        out = [f'<svg class="map" viewBox="{x0} {y0} {w} {h}" role="img" style="max-width:420px">']
        for r in range(y0, min(y0 + h, len(st.text))):
            for c in range(x0, x0 + w):
                ch = changed.get((c, r), st.char(c, r))
                if ch in "#d":
                    out.append(f'<rect x="{c}" y="{r}" width="1.02" height="1.02" fill="var(--geo)"/>')
                elif ch in st.game.climb:
                    out.append(f'<rect x="{c + .3}" y="{r}" width=".4" height="1.02" fill="var(--geo2)"/>')
                elif ch == st.game.hazard:
                    out.append(f'<rect x="{c}" y="{r + .4}" width="1" height=".6" fill="var(--s2)"/>')
                elif ch in st.game.enemy_glyphs:
                    out.append(f'<circle cx="{c + .5}" cy="{r + .5}" r=".38" fill="var(--bad)"/>')
        for c, r in after:
            out.append(f'<rect x="{c}" y="{r}" width="1" height="1" fill="none" stroke="var(--s1)" stroke-width=".12"/>')
        return "".join(out) + "</svg>"

    return ('<div class="grid" style="margin:8px 0 0"><div><div class="muted">Before</div>' + draw({}) +
            '</div><div><div class="muted">After</div>' + draw(after) + "</div></div>")


def mutate(stage: str) -> str:
    index = mutants.load(game_name=GAME)
    spec = mutants.game(GAME)
    make = post("mutants", "/mutate", "Generate mutants" if not index else "Generate again", primary=not index)
    if not index:
        return page("mutate", "2. Mutate", "Deliberate changes to the stage, to see whether the pipeline notices them.",
                    f'<div class="card"><p class="empty">No mutants yet.</p>{make}</div>', reload_on_change=True)
    st = mutants.Stage(stage, GAME)
    tone = {"breaking": "bad", "neutral": "good", "degrading": ""}
    by: dict = {}
    for m in index["mutants"]:
        by[m["class"]] = by.get(m["class"], 0) + 1
    tiles = '<div class="tiles">' + "".join(
        f'<div class="tile"><div class="k">{esc(k.capitalize())}</div><div class="v">{by.get(k, 0)}</div><div class="n">{esc(n)}</div></div>'
        for k, n in (("breaking", "the exit can no longer be reached"), ("degrading", "reachable, but the route changed"),
                     ("neutral", "nothing reachable or sensed changed"))) + "</div>"
    groups: dict = {}
    for m in index["mutants"]:
        groups.setdefault(m["section"] or "Whole game (mechanics)", []).append(m)
    body = tiles
    for name, items in groups.items():
        rows = ""
        for m in items:
            check = m["checks"]
            detail = (f"<p class='muted' style='margin:6px 0 0'>Route check from the stage start: exit "
                      f"{'reachable' if check['exit_reachable'] else '<b>not reachable</b>'}; route cost {check['route_cost_before']} → "
                      f"{check['route_cost_after'] if check['route_cost_after'] is not None else '—'}. Expected effect on play: {esc(m['direction'])}.</p>")
            detail += mutant_map(m, st) if m["family"] == "level" else f"<pre>{esc(json.dumps(m['config']))}</pre>"
            rows += (f"<tr><td class='l'><details><summary><code>{esc(m['id'])}</code></summary>{detail}"
                     f"<pre>{spec.env}={esc(os.path.join(spec.out, m['id']))} python -m code.neuro.relay --game {GAME} --episodes 20 \\\n"
                     f"    --out {esc(os.path.join(spec.out, m['id'], 'relay'))}</pre></details></td>"
                     f"<td class='l'>{esc(m['operator'])}</td><td class='l {tone[m['class']]}'><b>{esc(m['class'])}</b></td>"
                     f"<td class='l' style='white-space:normal'>{esc(m['description'])}</td></tr>")
        body += (f'<div class="card"><h2>{esc(name)}</h2><div class="scroll"><table><tr><th class="l">Mutant</th><th class="l">Operator</th>'
                 f'<th class="l">Class</th><th class="l">What changed</th></tr>{rows}</table></div></div>')
    if index["skipped"]:
        body += ('<div class="card"><h2>Skipped</h2><p class="muted">An operator is skipped in a section where it has nothing to act on.</p>'
                 '<div class="scroll"><table><tr><th class="l">Section</th><th class="l">Operator</th><th class="l">Why</th></tr>' +
                 "".join(f"<tr><td class='l'>{esc(k['section'])}</td><td class='l'>{esc(k['operator'])}</td>"
                         f"<td class='l' style='white-space:normal'>{esc(k['reason'])}</td></tr>" for k in index["skipped"]) + "</table></div></div>")
    body += ('<div class="card"><h2>How this works</h2><p>Each mutant makes one change: its own copy of the stage file, or one value of '
             f"<code>{GAME}_config.yaml</code>. The class is assigned before any agent plays it, by searching the route from the stage's "
             "true start. Open a mutant to see the change and the command that plays it with the trained agents. Mutants are played by "
             f"agents only; people play the baseline.</p>{make}"
             f'<p class="muted" style="margin:8px 0 0">Files: <code>{esc(spec.out)}/&lt;mutant&gt;/mutant.json</code> (+ <code>level.txt</code>). '
             "Generating again rewrites those two files and leaves any results beside them.</p></div>")
    return page("mutate", "2. Mutate", "Deliberate changes to the stage, to see whether the pipeline notices them.", body,
                reload_on_change=True)


def train(stage: str) -> str:
    legs, picks = sections(stage), manifest()
    rows = ""
    for leg in legs:
        runs, pick = section_runs(leg.level), picks.get(leg.level) or {}
        actions = post("train", "/train", "Train", {"section": leg.level}, primary=True, extra=(
            '<label class="inline" style="flex-direction:row;align-items:center;gap:4px">runs <input type="number" name="seeds" value="3" min="1" max="8"></label>'
            '<label class="inline" style="flex-direction:row;align-items:center;gap:4px">generations <input type="number" name="gens" value="80" min="5" max="500"></label>'))
        actions += post("pick", "/train", "Re-pick", {"section": leg.level})
        detail = "".join(
            f"<tr><td class='l'>{esc(os.path.basename(r['run']))}{' (picked)' if r['run'] == pick.get('run') else ''}</td>"
            f"<td>{r['gens']}</td><td>{100 * r['win_rate']:.0f}%</td><td>{r['best']:.0f}</td></tr>" for r in runs)
        tested = f"{100 * pick['tested_win_rate']:.0f}%" if pick.get("tested_win_rate") is not None else "—"
        trained = f"{100 * pick['win_rate']:.0f}%" if pick.get("win_rate") is not None else "—"
        rows += (f"<tr><td class='l'><details><summary><b>{esc(leg.level)}</b> · {len(runs)} runs</summary><div class='scroll'><table>"
                 f"<tr><th class='l'>Run</th><th>Generations</th><th>Wins, last 10 generations</th><th>Best fitness</th></tr>{detail}</table></div></details></td>"
                 f"<td class='l'>{esc(os.path.basename(pick.get('run', '')) or '—')}</td><td>{pick.get('gens', '—')}</td>"
                 f"<td>{trained}</td><td>{tested}</td><td class='l act'>{actions}</td></tr>")
    body = ('<div class="card"><h2>Sections</h2><div class="scroll"><table><tr><th class="l">Section</th><th class="l">Run the relay uses</th>'
            "<th>Generations</th><th>Wins in training</th><th>Wins when tested</th><th class='l'>Actions</th></tr>" + rows + "</table></div>"
            '<p class="muted" style="margin:10px 0 0">Train starts new runs for the section, warm-started from the run the relay uses, in new '
            f"<code>seedNNN</code> folders under <code>{esc(RUNS)}/&lt;section&gt;/</code>. Re-pick tests the section's best runs over 12 varied "
            "attempts and points the relay at the one that wins most. Neither touches the other sections.</p></div>"
            '<div class="card"><h2>Watching a run train</h2><p>Runs started here are headless. To watch one live, start it from a terminal '
            f"and open the training dashboard at <code>http://127.0.0.1:{DASH_PORT}/{GAME}/index.html</code>:</p>"
            f"<pre>python -m code.neuro.trainer --game {GAME} --level {esc(legs[0].level)} --seed 1</pre></div>")
    return page("train", "3. Train", "One network per section. Training results stay on this page.", body, reload_on_change=True)


def play(stage: str) -> str:
    legs, people = sections(stage), human_results()
    rows = ""
    for level in [stage] + [leg.level for leg in legs]:
        a, h = agent_results(stage, level), people.get(level)
        agents = f"{a['wins']} of {a['n']} won <span class='muted'>· {when(a['when'])}</span>" if a else "<span class='muted'>not played</span>"
        humans = f"{h['wins']} of {h['n']} won" if h else "<span class='muted'>not played</span>"
        f = {"level": level}
        actions = (post("batch", "/play", "Run agents", f, primary=True, extra=(
                       '<label class="inline" style="flex-direction:row;align-items:center;gap:4px"><input type="number" name="episodes" value="20" min="1" max="200"> attempts</label>'))
                   + post("watch", "/play", "▶ Watch agents live", dict(f, seed=1234), new_tab=True)
                   + post("human", "/play", "Play it yourself", dict(f, player="", warmup=""), cls="human"))
        name = f"<b>{esc(level)}</b>" + (" <span class='muted'>(full stage)</span>" if level == stage else "")
        rows += f"<tr><td class='l'>{name}</td><td class='l'>{agents}</td><td class='l'>{humans}</td><td class='l act'>{actions}</td></tr>"
    body = ('<div class="card"><div class="filters" style="margin:0 0 12px"><label>Your player name (for attempts you play yourself)'
            '<input type="text" id="player" value="human" maxlength="24" pattern="[A-Za-z0-9_-]+"></label>'
            '<label>Warm-up before recording (minutes)<input type="number" id="warmup" value="3" min="0" max="30" step="0.5" style="width:90px"></label></div>'
            '<div class="scroll"><table><tr><th class="l">Level</th><th class="l">Agents, last batch</th><th class="l">People</th>'
            "<th class='l'>Actions</th></tr>" + rows + "</table></div>"
            '<p class="muted" style="margin:10px 0 0"><b>Run agents</b> replaces the level\'s last batch. <b>Watch agents live</b> opens the '
            "trainer's dashboard in a new tab. <b>Play it yourself</b> opens the game window on this machine and records one row per "
            "attempt; close the window (Esc) when done. The warm-up comes first: for that long nothing is recorded, then the level restarts and recording begins. Keys: A / D move, W / S climb, Space jump, Z or J fire. "
            'Results of all three are on the <a href="/measure">Measure</a> page.</p></div>')
    return page("play", "4. Play", "The trained agents play the stage, or a person does.", body, reload_on_change=True)


def measure() -> str:
    paths = measure_paths()
    data = telemetry.build(paths) if paths else {"episodes": 0}
    if not data["episodes"]:
        return page("measure", "5. Measure", "Telemetry from the playtests.",
                    '<div class="card"><p class="empty">No episodes yet. Play the stage on the <a href="/play">Play</a> page first.</p></div>')
    out = telemetry.render(data)
    out = out.replace("<h1>PEAK telemetry</h1>", nav("measure") + "<h1>5. Measure</h1>", 1)
    out = out.replace("</style>", EXTRA_CSS + "</style>", 1)          # the nav and jobs styles
    out = out.replace('</main><div id="tip">', JOBS_HTML + '</main><div id="tip">', 1)
    return out.replace("</script></body>", JOBS_JS + "</script></body>", 1)


# ── the server ───────────────────────────────────────────────────────────────────────────────
def serve(stage: str, port: int, open_browser: bool) -> None:
    import webbrowser
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    jobs = Jobs()
    levels = {stage} | {leg.level for leg in sections(stage)}
    section_names = levels - {stage}
    dash = f"http://127.0.0.1:{DASH_PORT}/{GAME}/index.html"

    def switch(game: str) -> None:
        """The game the pages show from now on. Jobs already running carry on."""
        nonlocal stage, levels, section_names, dash
        stage = use_game(game)
        levels = {stage} | {leg.level for leg in sections(stage)}
        section_names = levels - {stage}
        dash = f"http://127.0.0.1:{DASH_PORT}/{GAME}/index.html"

    def clamp(raw: str, lo: int, hi: int, default: int) -> int:
        try:
            return min(max(int(raw), lo), hi)
        except (TypeError, ValueError):
            return default

    def start_watch(q: dict) -> tuple[int, str]:
        found = telemetry.watch_command(q, DASH_PORT)
        if found is None:
            return 400, note("That attempt cannot be played", "<p>The level, manifest or seed in the request is not valid.</p>")
        cmd, what = found
        for j in jobs.items:                              # one live attempt at a time: free the dashboard port first
            if j["group"] == "dash":
                jobs._stop(j)
        if telemetry.port_busy(DASH_PORT):
            return 409, note("The live dashboard is already in use",
                             f"<p>Something else is running on port {DASH_PORT} (a training run or a replay started from a terminal). "
                             f"Stop it, then try again. It is at <a href='{dash}'>{dash}</a>.</p>")
        jobs.start(f"Watch live: {what}", cmd, group="dash")
        return 200, note("Starting the live attempt", f"<p><b>{esc(what)}</b></p><p id='st'>Starting…</p>"
                         "<p>The live dashboard opens in this tab as soon as it answers. Later seeds follow one after another.</p>",
                         telemetry.starting_script(dash))

    def note(title: str, body: str, script: str = "") -> str:
        return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>{esc(title)}</title><style>{CSS}</style></head>"
                f"<body><main>{nav('play')}<h1>{esc(title)}</h1>{body}</main><script>{script}</script></body></html>")

    def do(q: dict) -> tuple[int, str] | None:
        """Carry out a button press. None = done, go back to the page; otherwise a page to show."""
        action, level, section = q.get("action"), q.get("level", ""), q.get("section", "")
        if action == "stop":
            jobs.stop(clamp(q.get("job"), 0, 10 ** 6, 0))
        elif action == "train" and section in section_names:
            seeds, gens = clamp(q.get("seeds"), 1, 8, 3), clamp(q.get("gens"), 5, 500, 80)
            taken = [int(m.group(1)) for d in glob.glob(os.path.join(RUNS, section, "seed*"))
                     if (m := re.fullmatch(r"seed(\d+)", os.path.basename(d)))]
            first = (max(taken, default=0) // 100 + 1) * 100 + 1          # a new hundred: these runs belong together
            source = (manifest().get(section) or {}).get("run")
            for n in range(first, first + seeds):
                cmd = [sys.executable, "-m", "code.neuro.trainer", "--game", GAME, "--level", section, "--seed", str(n),
                       "--gens", str(gens), "--no-serve", "--run-dir", os.path.join(RUNS, section, f"seed{n}")]
                if source and os.path.exists(os.path.join(source, "best.npz")):
                    cmd += ["--seed-from", source]
                else:                                             # a first run: the network shape the picks use
                    cmd += ["--hidden", "32", "--memory", "2"]
                jobs.start(f"Train {section} seed{n}, {gens} generations", cmd)
        elif action == "mutants":
            jobs.start("Generate mutants", [sys.executable, "-m", "code.games.mutants", "--game", GAME, "--stage", stage], group="mutants")
        elif action == "pick" and section in section_names:
            jobs.start(f"Re-pick {section}", [sys.executable, "-m", "code.pipeline", "--game", GAME, "--stage", stage, "--pick-section", section],
                       group=f"pick-{section}")
        elif action == "batch" and level in levels:
            n = clamp(q.get("episodes"), 1, 200, 20)
            cmd = [sys.executable, "-m", "code.neuro.relay", "--game", GAME, "--stage", stage, "--episodes", str(n)]
            jobs.start(f"Agents play {level}, {n} attempts", cmd + (["--section", level] if level != stage else []),
                       group=f"batch-{level}")
        elif action == "watch" and level in levels:
            return start_watch({"stage": stage, "section": "" if level == stage else level, "manifest": MANIFEST,
                                "seed": q.get("seed", "1234")})
        elif action == "human" and level in levels:
            player = q.get("player") or "human"
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,24}", player):
                return 400, note("That player name cannot be used", "<p>Use letters, digits, - and _ only.</p>")
            try:
                warmup = min(max(float(q.get("warmup") or 0), 0.0), 30.0)
            except ValueError:
                warmup = 0.0
            jobs.start(f"{player} plays {level}" + (f", {warmup:g} min warm-up" if warmup else ""),
                       [sys.executable, "-m", "code.games.tools.manual_play", "--game", GAME, "--level", level,
                        "--record", HUMAN_DIR, "--player", player, "--warmup", str(warmup)],
                       group="human", window=True)
        else:
            return 400, note("That cannot be done", "<p>Unknown action or level.</p>")
        return None

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a) -> None:  # quiet
            pass

        def _send(self, status: int, body: str, ctype: str = "text/html; charset=utf-8") -> None:
            raw = body.encode()
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            u = urllib.parse.urlparse(self.path)
            try:
                if u.path == "/jobs":
                    return self._send(200, json.dumps(jobs.view()), "application/json")
                if u.path == "/game":                     # the game links in the nav
                    g = (urllib.parse.parse_qs(u.query).get("g") or [""])[0]
                    if g in GAMES:
                        switch(g)
                    self.send_response(303)
                    self.send_header("Location", "/")
                    return self.end_headers()
                if u.path == "/watch":                    # the Measure page's ▶ links
                    return self._send(*start_watch({k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}))
                pages = {"/": lambda: hub(stage), "/design": lambda: design(stage), "/mutate": lambda: mutate(stage),
                         "/train": lambda: train(stage), "/play": lambda: play(stage), "/measure": measure,
                         "/telemetry.html": measure}
                if u.path not in pages:
                    return self._send(404, note("No such page", "<p>Use the links above.</p>"))
                self._send(200, pages[u.path]())
            except Exception as e:  # noqa: BLE001 — show the error on the page instead of an empty reply
                self._send(500, note("This page could not be built", f"<pre>{esc(type(e).__name__)}: {esc(str(e))}</pre>"))

        def do_POST(self) -> None:
            if urllib.parse.urlparse(self.path).path != "/do":
                return self._send(404, "not found")
            origin = self.headers.get("Origin")           # buttons on these pages only, not another site's form
            if origin and urllib.parse.urlparse(origin).netloc != self.headers.get("Host"):
                return self._send(403, "forbidden")
            size = min(int(self.headers.get("Content-Length") or 0), 10000)
            q = {k: v[0] for k, v in urllib.parse.parse_qs(self.rfile.read(size).decode("utf-8", "replace")).items()}
            try:
                shown = do(q)
            except Exception as e:  # noqa: BLE001
                shown = 500, note("That could not be started", f"<pre>{esc(type(e).__name__)}: {esc(str(e))}</pre>")
            if shown is not None:
                return self._send(*shown)
            back = q.get("back", "/")
            self.send_response(303)
            self.send_header("Location", back if re.fullmatch(r"/[a-z]*", back) else "/")
            self.end_headers()

    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    import signal
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # reach the finally: no orphaned jobs
    url = f"http://127.0.0.1:{port}/"
    print(f"pipeline: {url}  (Ctrl+C to close; that stops every job started from the pages)", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        jobs.stop_all()


def main() -> None:
    ap = argparse.ArgumentParser(description="Pipeline pages: design, mutate, train, play, measure")
    ap.add_argument("--game", default="megaman", choices=sorted(GAMES), help="the game the pages open on")
    ap.add_argument("--stage", default=None, help="the full-stage level id (default: the game's stage)")
    ap.add_argument("--port", type=int, default=8003)
    ap.add_argument("--open", action="store_true", help="open the hub in the default browser")
    ap.add_argument("--pick-section", default=None, metavar="SECTION",
                    help="test this section's best runs, point the relay at the winner, and exit")
    args = ap.parse_args()
    stage = args.stage or use_game(args.game)
    use_game(args.game)
    if args.pick_section:
        return pick_section(args.pick_section, stage)
    serve(stage, args.port, args.open)


if __name__ == "__main__":
    main()
