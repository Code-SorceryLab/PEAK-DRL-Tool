"""Telemetry dashboard: one self-contained HTML page for a set of episode logs.

    python -m code.stats.telemetry runs/megaman/relay            # -> runs/megaman/relay/telemetry.html
    python -m code.stats.telemetry runs/manual runs/megaman/relay -o out.html
    python -m code.stats.telemetry runs/megaman/relay --serve    # the same page, with working ▶ Watch live buttons

Stage 4 of the pipeline ("Measure"): everything on the page is computed from telemetry rows
only (episodes.csv, code/stats/episode_log.py), so it does not know or care whether a person or
an agent played. Per level and per player it shows the metrics of the paper's section 3.5:

    completion · win rate · furthest progress and where attempts ended · cause-of-death mix ·
    time to finish and the routes taken · median jumps, pickups, kills, bricks, bombs, wall jumps
    per episode. Progress is over failed attempts; with 2+ human players a pooled group is added.

A game that records a split by stage section (megaman) gets that table too. The level's
geometry is drawn under the routes when its file can be found; otherwise the map is omitted.

Episodes played by the relay (code/neuro/relay.py) can be played again live: with `--serve` the
page's ▶ Watch live buttons start `relay --watch` on the same level and seed and open the
trainer's dashboard. Opened as a file the buttons are disabled.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
from collections import Counter

from . import episode_log, registry

HERE = os.path.dirname(os.path.abspath(__file__))
GAMES_DIR = os.path.join(HERE, "..", "games")
SOLID, LADDER, SPIKE = "#=?<>FLd", "H", "^"
PER_EPISODE = [("jumps", "Jumps"), ("coins", "Pickups"), ("kills", "Enemies killed"),
               ("bricks", "Bricks destroyed"), ("damage_taken", "Damage taken"), ("shots", "Shots fired"),
               ("bombs_placed", "Bombs placed"), ("wall_jumps", "Wall jumps")]
POOLED = "all humans (pooled)"


def find_logs(paths: list[str]) -> list[str]:
    out = []
    for p in paths:
        if p.endswith(".csv"):
            out.append(p)
        else:
            out += sorted(glob.glob(os.path.join(p, "**", "episodes.csv"), recursive=True))
    return list(dict.fromkeys(out))


def level_file(game: str, level: str) -> str | None:
    """The level's ASCII file, if this game's level list can be read without loading the game."""
    try:
        import yaml
        if game == "megaman":
            with open(os.path.join(GAMES_DIR, "megaman_levels.yaml"), encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            entry = (data.get("levels") or {}).get(level) or (data.get("disabled_levels") or {}).get(level) or {}
            return os.path.join(GAMES_DIR, "levels", entry["file"]) if entry.get("file") else None
        from code.neuro.report import _level_file      # the other games: the report already knows how
        return _level_file(game, level)
    except Exception:  # noqa: BLE001 — the map is optional
        return None


def geometry(path: str | None, tile: int) -> dict | None:
    """Runs of solid / ladder / spike cells, as [col, row, length], for drawing the level."""
    if not path or not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        rows = f.read().split("\n")
    out = {"tile": tile, "cols": max(map(len, rows)), "rows": len(rows), "solid": [], "ladder": [], "spike": []}
    for kind, glyphs in (("solid", SOLID), ("ladder", LADDER), ("spike", SPIKE)):
        for r, line in enumerate(rows):
            c = 0
            while c < len(line):
                if line[c] in glyphs:
                    start = c
                    while c < len(line) and line[c] in glyphs:
                        c += 1
                    out[kind].append([start, r, c - start])
                else:
                    c += 1
    return out


def summarize(rows: list[dict], game: str) -> dict:
    """The section 3.5 metrics for one (level, player) group."""
    n = len(rows)
    won = [r for r in rows if r.get("status") == "WON"]
    lost = [r for r in rows if r.get("status") != "WON"]
    prog = [r["progress"] for r in lost if r.get("progress") is not None]   # how far the failed attempts got
    times = [r["time_s"] for r in won if r.get("time_s") is not None]
    causes = Counter((r.get("cause") or "?") for r in lost)
    hist = [0] * 10
    for r in lost:
        if r.get("progress") is not None:
            hist[min(9, int(r["progress"] * 10))] += 1
    medians = [{"key": k, "label": label, "value": round(statistics.median(v), 2)}
               for k, label in PER_EPISODE if (v := [r[k] for r in rows if r.get(k) is not None])]
    # split by stage section, where the game records one
    order: list[str] = []
    for r in rows:
        for s in r.get("sections") or []:
            if s["section"] not in order:
                order.append(s["section"])
    sections = []
    for i, name in enumerate(order):
        secs = [s for r in rows for s in r.get("sections") or [] if s["section"] == name]
        ended = sum(1 for r in lost if r.get("section") == name)
        sections.append({"name": name, "reached": len(secs), "cleared": len(secs) - ended, "ended": ended,
                         "time": round(statistics.fmean(s["frames"] for s in secs) / 60, 1),
                         "damage": round(statistics.fmean(s["damage"] for s in secs), 1),
                         "kills": round(statistics.fmean(s["kills"] for s in secs), 1),
                         "hp_in": round(statistics.fmean(hp), 1) if (hp := [s["hp_in"] for s in secs
                                                                            if s["hp_in"] is not None]) else None})
    try:
        balance = registry.compute_level(rows, game)
    except Exception:  # noqa: BLE001 — a game the registry does not know yet
        balance = {}
    return {
        "episodes": n, "wins": len(won), "completed": bool(won),
        "win_rate": len(won) / n if n else 0.0,
        "furthest": max(prog) if prog else None,
        "median_progress": statistics.median(prog) if prog else None,
        "time_to_finish": statistics.median(times) if times else None,
        "causes": [{"cause": c, "n": k} for c, k in causes.most_common()],
        "progress_hist": hist, "medians": medians, "sections": sections,
        "deaths_per_win": balance.get("deaths_per_run"), "death_spread": balance.get("death_cluster_entropy"),
        "episode_rows": [{
            "agent": str(r.get("agent", "")), "seed": r.get("seed"), "status": r.get("status"), "cause": r.get("cause") or "",
            "progress": r.get("progress"), "time_s": r.get("time_s"), "jumps": r.get("jumps"),
            "kills": r.get("kills"), "coins": r.get("coins"), "section": r.get("section") or "",
            "end": [r.get("end_x"), r.get("end_y")],
            "route": [[round(x), round(y)] for x, y in (r.get("route") or [])][::2],
        } for r in rows],
    }


def relay_source(log: str) -> dict | None:
    """What the relay played to write this log (the legs.json it leaves beside it), or None."""
    try:
        with open(os.path.join(os.path.dirname(log), "legs.json"), encoding="utf-8") as f:
            legs = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not legs.get("stage") or not legs.get("manifest"):
        return None
    return {"stage": legs["stage"], "section": legs.get("section") or "", "manifest": legs["manifest"]}


def build(paths: list[str]) -> dict:
    logs = find_logs(paths)
    rows = []
    for p in logs:
        src = relay_source(p)
        for r in episode_log.read(p):
            r["_watch"] = src
            rows.append(r)
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        player = r.get("persona") or "?"
        if r.get("source") and r["source"] != "agent":
            player = f"{player} ({r['source']})"
        groups.setdefault((r.get("game") or "?", str(r.get("level")), player), []).append(r)
    humans: dict[tuple, dict[str, list[dict]]] = {}
    for (game, level, player), eps in groups.items():
        if eps[0].get("source") == "human":
            humans.setdefault((game, level), {})[player] = eps
    for (game, level), by_player in humans.items():   # pooled only adds something with 2+ people
        if len(by_player) > 1:
            groups[(game, level, POOLED)] = [r for eps in by_player.values() for r in eps]
    out = []
    for (game, level, player), eps in sorted(groups.items()):
        tile = int(next((r["tile"] for r in eps if r.get("tile")), 32))
        g = summarize(eps, game)
        g.update({"game": game, "level": level, "player": player,
                  "level_w": next((r["level_w"] for r in eps if r.get("level_w")), None),
                  "level_h": next((r["level_h"] for r in eps if r.get("level_h")), None),
                  "watch": next((r["_watch"] for r in eps if r.get("_watch") and r.get("seed") is not None), None),
                  "geometry": geometry(level_file(game, level), tile)})
        out.append(g)
    return {"logs": logs, "episodes": len(rows), "groups": out}


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>PEAK telemetry</title>
<style>
:root{color-scheme:light;--surface:#fcfcfb;--panel:#ffffff;--line:#e4e3de;--text:#0b0b0b;--text2:#52514e;--muted:#8a8983;
 --s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--geo:#d9d8d2;--geo2:#b9b8b1;--good:#0d7a3c;--bad:#b3261e}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--surface:#1a1a19;--panel:#222221;--line:#34342f;
 --text:#f0efec;--text2:#c3c2b7;--muted:#8d8c84;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--geo:#3a3a36;--geo2:#55554f;--good:#4cc38a;--bad:#f2867e}}
:root[data-theme="dark"]{color-scheme:dark;--surface:#1a1a19;--panel:#222221;--line:#34342f;--text:#f0efec;--text2:#c3c2b7;--muted:#8d8c84;
 --s1:#3987e5;--s2:#d95926;--s3:#199e70;--geo:#3a3a36;--geo2:#55554f;--good:#4cc38a;--bad:#f2867e}
*{box-sizing:border-box}
body{margin:0;background:var(--surface);color:var(--text);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:20px;margin:0 0 2px}h2{font-size:14px;margin:0 0 10px;font-weight:600}
.sub{color:var(--text2);margin:0 0 16px}
.filters{display:flex;gap:12px;flex-wrap:wrap;align-items:end;margin:0 0 16px}
label{display:flex;flex-direction:column;gap:4px;color:var(--text2);font-size:12px}
select{font:inherit;padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:var(--panel);color:var(--text)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin-bottom:16px}
.tile,.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px 14px}
.tile .k{color:var(--text2);font-size:12px}.tile .v{font-size:24px;font-weight:600;margin-top:2px}.tile .n{color:var(--muted);font-size:12px}
.status{display:inline-flex;align-items:center;gap:6px;font-size:16px;font-weight:600;white-space:nowrap}
.status.good{color:var(--good)}.status.bad{color:var(--bad)}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px}
@media (max-width:760px){.grid{grid-template-columns:1fr}}
.card{margin-bottom:12px;overflow:hidden}
.legend{display:flex;gap:14px;flex-wrap:wrap;color:var(--text2);font-size:12px;margin:0 0 8px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px;vertical-align:-1px}
.legend i.line{height:2px;border-radius:0;width:14px;vertical-align:3px}
.bar-row{display:grid;grid-template-columns:140px 1fr 84px;gap:8px;align-items:center;margin:5px 0}
.bar-row .name{color:var(--text2);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.bar-row .track{height:14px}.bar-row .fill{height:14px;background:var(--s1);border-radius:0 4px 4px 0;min-width:2px}
.bar-row .val{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.hist{display:flex;align-items:flex-end;gap:2px;height:120px;border-bottom:1px solid var(--line)}
.hist div{flex:1;background:var(--s1);border-radius:4px 4px 0 0;min-height:0}
.axis{display:flex;justify-content:space-between;color:var(--muted);font-size:11px;margin-top:4px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{text-align:right;padding:5px 8px;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child,td.l,th.l{text-align:left}th{color:var(--text2);font-weight:500;font-size:12px}
.scroll{overflow-x:auto}.map{width:100%;height:auto;display:block;background:var(--surface);border-radius:6px}
details summary{cursor:pointer;color:var(--text2)}.empty{color:var(--muted)}
#tip{position:fixed;pointer-events:none;background:var(--text);color:var(--surface);padding:6px 8px;border-radius:6px;font-size:12px;
 line-height:1.35;max-width:260px;display:none;z-index:9}
.hit{cursor:default}
.watch{display:inline-block;font:inherit;font-weight:600;padding:6px 12px;border:1px solid var(--s1);border-radius:6px;background:var(--s1);
 color:#fff;text-decoration:none;white-space:nowrap}
.watch.sm{padding:1px 7px;font-size:12px}
.watchnote{color:var(--muted);font-size:12px;align-self:center}
html.file .watch{opacity:.4;pointer-events:none}html:not(.file) .file-only{display:none}
</style></head><body><main>
<h1>PEAK telemetry</h1>
<p class="sub" id="sub"></p>
<div class="filters">
 <label>Game and level<select id="f-level"></select></label>
 <label>Player<select id="f-player"></select></label>
 <span id="watch"></span>
</div>
<div id="view"></div>
</main><div id="tip"></div>
<script>
const DATA = __DATA__;
const $ = s => document.querySelector(s), el = (t, a = {}, h = "") => { const e = document.createElement(t); for (const k in a) e.setAttribute(k, a[k]); e.innerHTML = h; return e; };
const pct = v => v == null ? "—" : Math.round(v * 100) + "%", num = v => v == null ? "—" : (Math.round(v * 10) / 10).toString();
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const tip = $("#tip");
if (location.protocol == "file:") document.documentElement.classList.add("file");  // no /watch endpoint
const watchUrl = (g, seed) => "/watch?" + new URLSearchParams({stage: g.watch.stage, section: g.watch.section, manifest: g.watch.manifest, seed});
function hover(node, html){ node.addEventListener("mousemove", e => { tip.innerHTML = html; tip.style.display = "block";
  tip.style.left = Math.min(e.clientX + 14, innerWidth - 270) + "px"; tip.style.top = (e.clientY + 14) + "px"; });
  node.addEventListener("mouseleave", () => tip.style.display = "none"); }

$("#sub").textContent = DATA.episodes + " episodes from " + DATA.logs.length + " log" + (DATA.logs.length == 1 ? "" : "s") + ": " + DATA.logs.join(", ");
const levels = [...new Set(DATA.groups.map(g => g.game + " · " + g.level))];
const fl = $("#f-level"), fp = $("#f-player");
levels.forEach(l => fl.append(el("option", {}, esc(l))));
function players(){ fp.innerHTML = ""; DATA.groups.filter(g => g.game + " · " + g.level == fl.value).forEach(g => fp.append(el("option", {}, esc(g.player)))); }
fl.onchange = () => { players(); render(); }; fp.onchange = render;

function tile(k, v, n = ""){ return `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="n">${n}</div></div>`; }

function map(g){
  const geo = g.geometry, T = geo ? geo.tile : 32;
  const W = geo ? geo.cols * T : g.level_w, H = geo ? geo.rows * T : g.level_h;
  if (!W || !H) return "";
  const r = Math.max(W, H) / 260, sw = r / 3;
  let s = `<svg class="map" viewBox="0 0 ${W} ${H}" role="img" aria-label="Level map with routes and where attempts ended">`;
  if (geo){
    s += geo.solid.map(([c, y, n]) => `<rect x="${c*T}" y="${y*T}" width="${n*T}" height="${T}" fill="var(--geo)"/>`).join("");
    s += geo.ladder.map(([c, y, n]) => `<rect x="${c*T + T*0.3}" y="${y*T}" width="${n*T - T*0.6}" height="${T}" fill="var(--geo2)"/>`).join("");
    s += geo.spike.map(([c, y, n]) => `<rect x="${c*T}" y="${y*T + T*0.5}" width="${n*T}" height="${T*0.5}" fill="var(--geo2)"/>`).join("");
  }
  g.episode_rows.forEach(e => { if (e.route.length > 1) s += `<polyline points="${e.route.map(p => p.join(",")).join(" ")}" fill="none" stroke="var(--s1)" stroke-width="${sw}" stroke-opacity=".45" stroke-linejoin="round"/>`; });
  g.episode_rows.forEach((e, i) => { if (e.end[0] == null) return; const won = e.status == "WON";
    s += `<circle class="hit" data-i="${i}" cx="${e.end[0]}" cy="${e.end[1]}" r="${r}" fill="var(--${won ? "s3" : "s2"})" stroke="var(--surface)" stroke-width="${sw}"/>`; });
  return s + "</svg>";
}

function render(){
  const g = DATA.groups.find(x => x.game + " · " + x.level == fl.value && x.player == fp.value), v = $("#view");
  if (!g){ v.innerHTML = '<p class="empty">No episodes.</p>'; $("#watch").innerHTML = ""; return; }
  const first = g.watch && g.episode_rows.find(e => e.seed != null);
  $("#watch").innerHTML = first ? `<a class="watch" target="_blank" rel="noopener" href="${esc(watchUrl(g, first.seed))}">▶ Watch live</a> <span class="watchnote file-only">start the page with --serve to enable</span>` : "";
  const status = g.completed ? '<span class="status good">✓ Completed</span>' : '<span class="status bad">✕ Not completed</span>';
  let h = '<div class="tiles">' +
    `<div class="tile"><div class="k">Completion</div><div class="v">${status}</div><div class="n">by at least one attempt</div></div>` +
    tile("Win rate", pct(g.win_rate), g.wins + " of " + g.episodes + " attempts") +
    tile("Furthest progress", pct(g.furthest), "median " + pct(g.median_progress) + " · failed attempts") +
    tile("Time to finish", g.time_to_finish == null ? "—" : num(g.time_to_finish) + " s", g.wins ? "median of wins" : "no wins yet") +
    g.medians.map(m => tile(m.label, num(m.value), "median per episode")).join("") + "</div>";

  const m = map(g);
  if (m) h += `<div class="card"><h2>Routes and where attempts ended</h2><div class="legend"><span><i class="line" style="background:var(--s1)"></i>Route taken</span><span><i style="background:var(--s2)"></i>Attempt ended (failed)</span><span><i style="background:var(--s3)"></i>Finished</span></div>${m}</div>`;

  const cmax = Math.max(1, ...g.causes.map(c => c.n)), failed = g.episodes - g.wins;
  h += '<div class="grid"><div class="card"><h2>How failed attempts ended</h2>' + (g.causes.length ? g.causes.map((c, i) =>
      `<div class="bar-row hit" data-c="${i}"><div class="name">${esc(c.cause)}</div><div class="track"><div class="fill" style="width:${100 * c.n / cmax}%"></div></div><div class="val">${c.n} · ${pct(c.n / failed)}</div></div>`).join("")
      : '<p class="empty">No failed attempts.</p>') + "</div>";
  const hmax = Math.max(1, ...g.progress_hist);
  h += '<div class="card"><h2>How far failed attempts got</h2>' + (failed ? '<div class="hist">' + g.progress_hist.map((n, i) =>
      `<div class="hit" data-h="${i}" style="height:${100 * n / hmax}%"></div>`).join("") + '</div><div class="axis"><span>start</span><span>50%</span><span>goal</span></div>'
      : '<p class="empty">No failed attempts.</p>') + "</div></div>";

  if (g.sections.length) h += '<div class="card"><h2>By stage section</h2><div class="scroll"><table><tr><th>Section</th><th>Reached</th><th>Cleared</th><th>Ended here</th><th>Avg time</th><th>Avg damage</th><th>Avg kills</th><th>Avg energy on entry</th></tr>' +
    g.sections.map(s => `<tr><td>${esc(s.name)}</td><td>${s.reached}</td><td>${s.cleared}</td><td>${s.ended}</td><td>${num(s.time)} s</td><td>${num(s.damage)}</td><td>${num(s.kills)}</td><td>${num(s.hp_in)}</td></tr>`).join("") + "</table></div></div>";

  h += '<div class="card"><details><summary>All ' + g.episodes + ' episodes</summary><div class="scroll"><table><tr><th>Attempt</th><th class="l">Outcome</th><th class="l">Cause</th><th class="l">Ended in</th><th>Progress</th><th>Time</th><th>Jumps</th><th>Kills</th><th>Pickups</th>' + (g.watch ? '<th>Live</th>' : '') + '</tr>' +
    g.episode_rows.map(e => `<tr><td>${esc(e.agent)}</td><td class="l">${esc(e.status)}</td><td class="l">${esc(e.cause)}</td><td class="l">${esc(e.section)}</td><td>${pct(e.progress)}</td><td>${num(e.time_s)} s</td><td>${e.jumps ?? "—"}</td><td>${e.kills ?? "—"}</td><td>${e.coins ?? "—"}</td>${g.watch ? `<td>${e.seed == null ? "" : `<a class="watch sm" target="_blank" rel="noopener" href="${esc(watchUrl(g, e.seed))}" aria-label="Watch attempt ${esc(e.agent)} live">▶</a>`}</td>` : ""}</tr>`).join("") + "</table></div></details></div>";
  v.innerHTML = h;

  v.querySelectorAll("circle.hit").forEach(c => { const e = g.episode_rows[+c.dataset.i];
    hover(c, `<b>${esc(e.agent)}</b><br>${e.status == "WON" ? "Finished" : esc(e.cause || e.status)}${e.section ? " in " + esc(e.section) : ""}<br>${pct(e.progress)} of the way · ${num(e.time_s)} s`); });
  v.querySelectorAll("[data-c]").forEach(b => { const c = g.causes[+b.dataset.c]; hover(b, `<b>${esc(c.cause)}</b><br>${c.n} of ${failed} failed attempts`); });
  v.querySelectorAll("[data-h]").forEach(b => { const i = +b.dataset.h; hover(b, `<b>${i * 10}–${i * 10 + 10}% of the way</b><br>${g.progress_hist[i]} failed attempts ended here`); });
}
players(); render();
</script></body></html>
"""


def render(data: dict) -> str:
    return PAGE.replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))


def watch_command(q: dict, dash_port: int = 8000) -> tuple[list[str], str] | None:
    """The `relay --watch` command for a /watch request (stage, section, manifest, seed), and a
    label for it; None when the request names something that cannot be played. Only manifests
    under runs/ and levels the game lists are accepted."""
    import re
    import sys
    stage, section, manifest = q.get("stage", ""), q.get("section", ""), q.get("manifest", "")
    try:
        from code.neuro.relay import stage_legs
        seed = int(q.get("seed", ""))
        ok = (bool(re.fullmatch(r"[\w-]+", stage)) and (not section or section in {leg.level for leg in stage_legs(stage)})
              and manifest.replace(os.sep, "/").startswith("runs/") and ".." not in manifest
              and manifest.endswith(".json") and os.path.exists(manifest))
    except (ValueError, SystemExit):
        ok = False
    if not ok:
        return None
    cmd = [sys.executable, "-m", "code.neuro.relay", "--stage", stage, "--manifest", manifest,
           "--watch", "--seed", str(seed), "--port", str(dash_port)]
    if section:
        cmd += ["--section", section]
    return cmd, f"{section or stage}, seed {seed}"


def port_busy(port: int) -> bool:
    import socket
    with socket.socket() as sock:
        sock.settimeout(0.3)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def starting_script(dash: str) -> str:
    """Page script: wait for the trainer's dashboard to answer, then go to it."""
    return f"""
const dash = {json.dumps(dash)}, t0 = Date.now(), st = document.getElementById('st');
(async function poll(){{
  try {{ await fetch(dash, {{mode: 'no-cors', cache: 'no-store'}}); location.replace(dash); return; }} catch {{}}
  const s = Math.round((Date.now() - t0) / 1000);
  st.textContent = s < 20 ? `Starting… ${{s}} s` : `Still waiting after ${{s}} s. Check the terminal that serves this page for errors.`;
  setTimeout(poll, 700);
}})();"""


def serve(paths: list[str], port: int, open_browser: bool, dash_port: int = 8000) -> None:
    """Serve the page locally so its ▶ Watch live buttons can start a live attempt.
    GET /watch?stage=S&section=X&manifest=runs/...&seed=N starts `relay --watch` (one at a time, a
    new one replaces the last) and forwards the tab to the trainer's dashboard. The page is rebuilt
    on every load. Binds 127.0.0.1 only; plays only manifests under runs/ and levels the game lists."""
    import html
    import subprocess
    import sys
    import urllib.parse
    import webbrowser
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    state: dict = {"proc": None}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a) -> None:  # quiet
            pass

        def _send(self, status: int, body: str) -> None:
            raw = body.encode()
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def _note(self, status: int, title: str, body: str, script: str = "") -> None:
            self._send(status, f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>{title}</title>"
                               "<style>body{font:15px/1.5 system-ui,sans-serif;max-width:620px;margin:12vh auto;padding:0 16px}"
                               "@media (prefers-color-scheme:dark){body{background:#1a1a19;color:#f0efec}a{color:#3987e5}}"
                               f"</style></head><body><h1>{title}</h1>{body}<p><a href='/'>← Back to the telemetry</a></p>"
                               f"<script>{script}</script></body></html>")

        def do_GET(self) -> None:
            u = urllib.parse.urlparse(self.path)
            if u.path in ("/", "/telemetry.html"):
                return self._send(200, render(build(paths)))
            if u.path != "/watch":
                return self._send(404, "not found")
            q = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}
            esc = html.escape   # the query string is reflected into HTML: escape all of it
            from code.neuro.relay import GAME
            found = watch_command(q, dash_port)
            if found is None:
                return self._note(400, "That attempt cannot be played",
                                  f"<p>Stage <code>{esc(q.get('stage', ''))}</code>, section <code>{esc(q.get('section', '')) or '—'}</code>, "
                                  f"manifest <code>{esc(q.get('manifest', ''))}</code>, seed <code>{esc(q.get('seed', ''))}</code>.</p>")
            cmd, what = found
            proc = state["proc"]
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
            dash = f"http://127.0.0.1:{dash_port}/{GAME}/index.html"
            if port_busy(dash_port):
                return self._note(409, "The live dashboard is already in use",
                                  f"<p>Something else is running on port {dash_port} (a training run or a replay started "
                                  f"from a terminal). Stop it, then press ▶ again. It is at <a href='{dash}'>{dash}</a>.</p>")
            state["proc"] = subprocess.Popen(cmd)
            print(f"watch: {what}", flush=True)
            script = starting_script(dash)
            self._note(200, "Starting the live attempt",
                       f"<p><b>{esc(what)}</b></p><p id='st'>Starting…</p>"
                       "<p>The live dashboard opens in this tab as soon as it answers. Later seeds follow one after another.</p>",
                       script)

    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    import signal
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # reach the finally: no orphaned live attempt
    url = f"http://127.0.0.1:{port}/"
    print(f"telemetry: {url}  (Ctrl+C to close)", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        if state["proc"] is not None and state["proc"].poll() is None:
            state["proc"].terminate()


def main() -> None:
    ap = argparse.ArgumentParser(description="Telemetry dashboard: one HTML page for a set of episode logs")
    ap.add_argument("paths", nargs="+", help="run / session directories (searched for episodes.csv) or .csv files")
    ap.add_argument("-o", "--out", default=None, help="output .html (default: telemetry.html in the first path)")
    ap.add_argument("--serve", action="store_true",
                    help="serve the page locally so its ▶ Watch live buttons can start a live attempt")
    ap.add_argument("--open", action="store_true", help="with --serve: open the page in the default browser")
    ap.add_argument("--port", type=int, default=8002)
    args = ap.parse_args()
    data = build(args.paths)
    if not data["episodes"]:
        raise SystemExit("no episodes found in " + ", ".join(args.paths))
    first = args.paths[0]
    out = args.out or os.path.join(first if os.path.isdir(first) else os.path.dirname(first), "telemetry.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(render(data))
    print(f"{data['episodes']} episodes, {len(data['groups'])} level/player groups -> {out}")
    if args.serve:
        serve(args.paths, args.port, args.open)


if __name__ == "__main__":
    main()
