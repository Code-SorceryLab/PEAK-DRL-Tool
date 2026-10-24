"""The mutation page, runs/ast/index.html: the AST evaluation at a glance (ast_metrics --page).

On top, the board: one cell per game and mutant, saying whether the playtest caught it (a breaking
or degrading mutant should be caught, a neutral one should stay quiet, the cosmetic copy must
replay identically). Below, per game: the level map with the change, where failed attempts ended
and the tiles the routes covered; the metric table; win rates per persona; people, when they
played; Mega Man's segments. A standalone file in the shared look (code/neuro/web/theme.css).
"""
from __future__ import annotations

import html
import json
import os

from code.games import eval_mutants

THEME = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "neuro", "web", "theme.css")
SLOTS = [("seal_route",), ("widen_gap", "add_enemy"), ("jump_velocity_down", "fuse_down"), ("off_route",), ("cosmetic_tile",)]
SLOT_NAMES = ["Sealed route", "Wider gap / extra enemy", "Weaker jump / shorter fuse", "Off-route tile", "Cosmetic tile"]
esc = html.escape


def _slot(operator: str) -> int:
    base = operator.split("-", 1)[-1] if operator.startswith(("s", "m")) and "-" in operator else operator
    return next(i for i, ops in enumerate(SLOTS) if base in ops)


def verdict(m: dict, entry: dict | None) -> tuple[str, str]:
    """(word, tone) for a mutant: tone is good, bad or none (not played yet)."""
    if not entry or not entry.get("agents"):
        trained = os.path.isdir(os.path.join("runs", "ast", m["game"], m["id"], "probes"))
        return ("not played" if trained or not m.get("retrain", True) else "not retrained"), "none"
    if m["operator"].endswith("cosmetic_tile"):
        d = entry.get("determinism")
        if not d:
            return "no baseline", "none"
        return (f"identical {d['identical']}/{d['compared']}", "good") if d["passed"] else \
            (f"differs {len(d['differ'])}/{d['compared']}", "bad")
    caught = (entry.get("vs_base") or {}).get("detected")
    if caught is None:
        return "no baseline", "none"
    if m["class"] == "neutral":
        return ("false alarm", "bad") if caught else ("quiet", "good")
    return ("caught", "good") if caught else ("missed", "bad")


def _level(ms: list[dict]) -> tuple[list[str], dict]:
    """The evaluation level's rows and each level mutant's changed cells."""
    if not ms:
        return [], {}
    path = os.path.join(eval_mutants.LEVELS_DIR, ms[0]["base_file"])
    with open(path, encoding="utf-8") as f:
        rows = [r.rstrip("\n") for r in f]
    while rows and not rows[-1].strip():
        rows.pop()
    return rows, {m["id"]: m.get("cells") or [] for m in ms}


def _signals(e: dict) -> str:
    s = e.get("vs_base") or {}
    words = [("completion lost", s.get("completion_lost")), (f"win rate (p {s.get('win_rate_p')})", s.get("win_rate_changed")),
             ("death cluster moved", s.get("cluster_moved")), ("new top cause", s.get("cause_changed"))]
    out = [w for w, on in words if on]
    if "determinism" in e:
        d = e["determinism"]
        out.append(f"replay {d['identical']}/{d['compared']} identical")
    return ", ".join(out)


def _pct(v) -> str:
    return "" if v is None else f"{100 * v:.0f}%"


def _num(v, fmt: str = "g", unit: str = "") -> str:
    return "" if v is None else format(v, fmt) + unit


def _where(c: dict | None) -> str:
    return "" if not c else f"({c['col']}, {c['row']}) {100 * c['share']:.0f}%"


def _row(v: str, a: dict, m: dict | None, tone: str, effort: list[str], signals: str) -> str:
    """One variant's line of the metric table."""
    w, n = a["win_rate"]["pooled"]
    what = f'<div class="label">{esc(m["description"])}</div>' if m else ""
    cls = m["class"] if m else "original"
    tds = [f"<td><b>{esc(v)}</b>{what}</td>", f'<td><span class="cls {esc(cls)}">{esc(cls)}</span></td>',
           f'<td class="n">{w}/{n}</td>', f'<td class="n">{"yes" if a["completion"] else "no"}</td>',
           f'<td class="n">{_pct(a["progress_median"])}<span class="label"> / {_pct(a["progress_max"])}</span></td>',
           f'<td class="n">{_num(a["time_to_finish"], ".1f", " s")}</td>', f"<td>{_where(a['death_cluster'])}</td>",
           f"<td>{esc(next(iter(a['causes']), ''))}</td>", f'<td class="n">{len(a["coverage"])}</td>']
    tds += [f'<td class="n">{_num(a["effort"].get(k))}</td>' for k in effort]
    tds.append(f'<td class="sig {tone}">{esc(signals)}</td>')
    return "<tr>" + "".join(tds) + "</tr>"


def _game_section(game: str, res: dict) -> str:
    title = eval_mutants.EVAL_LEVELS[game][1]
    ms = eval_mutants.eval_set(game)
    by_id = {m["id"]: m for m in ms}
    rows, cells = _level(ms)
    order = [v for v in ["base"] + sorted(k for k in res if k.startswith("base @")) + [m["id"] for m in ms] if v in res]
    if not order:
        return (f'<section class="frame stripe g-{game} game" id="{game}"><h2>{esc(title)}</h2>'
                f'<p class="empty"><i class="ico qblock"></i> No playtests yet. Run Agent Playtest (menu 20) or '
                f'<code>python -m code.neuro.ast_eval play --game {game}</code>.</p></section>')
    effort = sorted({k for v in order for k in ((res[v].get("agents") or {}).get("effort") or {})})
    head = "".join(f'<th class="n">{_ICON_OF.get(k, "")}{esc(k)}</th>' for k in effort)
    body = [_row(v, res[v]["agents"], by_id.get(v), verdict(by_id[v], res[v])[1] if v in by_id else "none",
                 effort, _signals(res[v])) for v in order if res[v].get("agents")]
    personas = sorted({p for v in order for p in ((res[v].get("agents") or {}).get("win_rate") or {}) if p != "pooled"})
    grid = "".join(
        f'<tr><td><i class="ico p-{esc(p)}"></i> {esc(p)}</td>' + "".join(_pips(((res[v].get("agents") or {}).get("win_rate") or {}).get(p)) for v in order) + "</tr>"
        for p in personas)
    humans = [(v, res[v]["humans"]) for v in order if res[v].get("humans")]
    human_html = ""
    if humans:
        human_html = ('<h3><i class="ico cursor"></i> People at the keyboard</h3><table class="data"><tr><th>Variant</th><th class="n">Attempts</th>'
                      '<th class="n">Wins</th><th>Per player</th><th class="n">Progress</th><th>Death cluster</th><th>Top cause</th></tr>'
                      + "".join(_human_row(v, h) for v, h in humans) + "</table>")
    seg_html = _segments(order, res) if game == "megaman" else ""
    data = {"rows": rows, "cells": cells,
            "runs": {v: {"ends": (res[v].get("agents") or {}).get("end_tiles") or [],
                         "cluster": (res[v].get("agents") or {}).get("death_cluster"),
                         "coverage": (res[v].get("agents") or {}).get("coverage") or []} for v in order}}
    buttons = "".join(f'<button class="btn{" on" if i == 0 else ""}" data-variant="{esc(v)}">{esc(v)}</button>'
                      for i, v in enumerate(order))
    return (f'<section class="frame stripe g-{game} game" id="{game}"><h2>{esc(title)}</h2>'
            f'<div class="mapbar">{buttons}</div>'
            f'<div class="map"><canvas data-game="{game}"></canvas></div>'
            '<p class="legend"><i class="k-cov"></i>tiles the routes covered <i class="k-end"></i>where a failed attempt ended '
            '<i class="k-clu"></i>death cluster (3 tiles) <i class="k-cell"></i>the mutated tiles</p>'
            f'<script type="application/json" id="data-{game}">{json.dumps(data, separators=(",", ":"))}</script>'
            f'<div class="scroll"><table class="data"><tr><th>Variant</th><th>Class</th><th class="n"><i class="ico coin"></i> Wins</th><th class="n">Solved</th>'
            f'<th class="n"><i class="ico up"></i> Progress (median / max)</th><th class="n"><i class="ico hourglass"></i> Time to finish</th><th><i class="ico skull"></i> Death cluster</th>'
            f'<th>Top cause</th><th class="n"><i class="ico eye"></i> Tiles covered</th>{head}<th>Signals</th></tr>{"".join(body)}</table></div>'
            + (f'<h3><i class="ico coin"></i> Wins per persona</h3><div class="scroll"><table class="data pips"><tr><th>Persona</th>'
               + "".join(f"<th>{esc(v)}</th>" for v in order) + f"</tr>{grid}</table></div>" if personas and game != "megaman" else "")
            + human_html + seg_html + "</section>")


def _human_row(v: str, h: dict) -> str:
    w, n = h["win_rate"]["pooled"]
    players = ", ".join(f"{p} {pw}/{pn}" for p, (pw, pn) in h["win_rate"].items() if p != "pooled")
    return (f'<tr><td>{esc(v)}</td><td class="n">{h["attempts"]}</td><td class="n">{w}/{n}</td><td>{esc(players)}</td>'
            f'<td class="n">{_pct(h["progress_median"])}</td><td>{_where(h["death_cluster"])}</td>'
            f"<td>{esc(next(iter(h['causes']), ''))}</td></tr>")


_ICON_OF = {k: f'<i class="ico {v}"></i> ' for k, v in {"jumps": "up", "kills": "sword", "pickups": "gem",
             "bombs placed": "bomb", "wall jumps": "up", "damage taken": "heartbreak"}.items()}


def _pips(wn) -> str:
    """Wins out of attempts as coins: gold for a win, grey for a loss (a persona's three agents)."""
    if not wn:
        return "<td></td>"
    w, n = wn
    if n > 6:
        return f'<td class="n">{w}/{n}</td>'
    return f'<td title="{w}/{n}">' + '<i class="ico coin"></i>' * w + '<i class="ico coin off"></i>' * (n - w) + "</td>"


def _segments(order: list[str], res: dict) -> str:
    segs = []
    for v in order:
        for s in ((res[v].get("agents") or {}).get("segments") or {}).get("per_segment", {}):
            if s not in segs:
                segs.append(s)
    if not segs:
        return ""
    head = "".join(f"<th>{esc(v)}</th>" for v in order)
    rows = []
    for s in segs:
        tds = []
        for v in order:
            seg = (res[v].get("agents") or {}).get("segments") or {}
            per = (seg.get("per_segment") or {}).get(s)
            ended = (seg.get("ended_in") or {}).get(s, 0)
            tds.append("<td></td>" if not per else
                       f'<td><b>{ended}</b> ended here<div class="label">{_num(per["time_median"], ".0f", " s")}, '
                       f'damage {_num(per["damage_median"])}</div></td>')
        rows.append(f"<tr><td>{esc(s)}</td>{''.join(tds)}</tr>")
    return (f'<h3><i class="ico flag"></i>Segments</h3><div class="scroll"><table class="data"><tr><th>Segment</th>{head}</tr>'
            f'{"".join(rows)}</table></div>')


def _icon(game: str) -> str:
    try:
        from code.neuro.report import _game_icon       # the Command center's game sprites
        return _game_icon(game)
    except Exception:  # noqa: BLE001 — the board works without them
        return ""


def board(results: dict) -> str:
    rows = []
    for game, res in results.items():
        cells = ["<td></td>"] * len(SLOTS)
        for m in eval_mutants.eval_set(game):
            word, tone = verdict(m, res.get(m["id"]))
            icon = {"good": '<i class="ico coin"></i>', "bad": '<i class="ico heartbreak"></i>'}.get(tone, "")
            cells[_slot(m["id"])] = (f'<td class="cell {tone}"><a href="#{game}"><span class="v">{icon}{esc(word)}</span>'
                                     f'<span class="label">{esc(m["class"])}</span></a></td>')
        rows.append(f'<tr><th class="g g-{game}"><a href="#{game}">{_icon(game)}{esc(eval_mutants.EVAL_LEVELS[game][1])}</a></th>'
                    f'{"".join(cells)}</tr>')
    return ('<table class="board"><tr><th></th>' + "".join(f"<th>{n}</th>" for n in SLOT_NAMES) + "</tr>"
            + "".join(rows) + "</table>")


CSS = """
main { max-width: 1320px; margin: 0 auto; padding: 22px 24px 64px; display: flex; flex-direction: column; gap: 22px; }
header.top { display: flex; align-items: center; gap: 18px; flex-wrap: wrap; }
header.top nav { margin-left: auto; display: flex; gap: 8px; }
.lede { color: var(--dim); max-width: 70ch; margin: 6px 0 0; }
.board { border-collapse: separate; border-spacing: 6px; width: 100%; table-layout: fixed; }
.board th { font: 600 12.5px/1.3 var(--ui); color: var(--dim); text-align: left; padding: 0 4px 2px; }
.board th.g { font: 700 18px/1.2 var(--pixel); padding: 0 10px; box-shadow: inset 4px 0 0 var(--stripe); width: 17%; }
.board th.g a { color: var(--ink); text-decoration: none; display: flex; align-items: center; gap: 10px; }
.gicon { height: 24px; image-rendering: pixelated; flex: none; }
.gletter { display: inline-grid; place-items: center; width: 22px; height: 22px; background: var(--stripe); color: var(--night); font: 700 18px/1 var(--pixel); }
.board td.cell { padding: 0; background: var(--deck); }
.board td.cell a { display: flex; flex-direction: column; gap: 2px; padding: 10px 12px; text-decoration: none;
  border: 3px solid transparent; border-image: var(--frame) 1 / 3px stretch; }
.board .v { font: 700 18px/1.1 var(--pixel); display: flex; align-items: center; gap: 7px; }
.board .v .ico { --ico: 18px; }
.board td.good .v { color: var(--won); }
.board td.bad .v { color: var(--dead); }
.board td.none .v { color: var(--mute); }
h3 .ico { --ico: 15px; margin-right: 4px; }
th .ico { --ico: 13px; vertical-align: -2px; }
section.game { padding: 18px 22px 22px; display: flex; flex-direction: column; gap: 12px; }
.mapbar { display: flex; gap: 6px; flex-wrap: wrap; }
.map { overflow-x: auto; background: #000; padding: 8px; width: fit-content; max-width: 100%; box-sizing: border-box; }
.map canvas { display: block; }
.legend { color: var(--dim); font-size: 13px; margin: 0; display: flex; flex-wrap: wrap; gap: 4px 16px; align-items: center; }
.legend i { display: inline-block; width: 12px; height: 12px; margin-right: 6px; vertical-align: -1px; }
.k-cov { background: rgba(248, 56, 0, .32); } .k-end { background: var(--ink); }
.k-clu { box-shadow: inset 0 0 0 2px var(--ink); } .k-cell { box-shadow: inset 0 0 0 2px var(--won); }
.scroll { overflow-x: auto; }
.cls { font-weight: 600; } .cls.breaking { color: var(--dead); } .cls.degrading { color: var(--stuck); }
.cls.neutral { color: var(--run); } .cls.base { color: var(--dim); }
td.sig.good { color: var(--won); } td.sig.bad { color: var(--dead); } td.sig.none { color: var(--dim); }
.pips td { white-space: nowrap; }
.pips .ico { --ico: 15px; margin-right: 3px; }
.empty { color: var(--dim); }
.empty .ico, td .ico { --ico: 15px; vertical-align: -3px; }
"""

JS = r"""
const TILE = { '#': '#4a4a4a', '=': '#7c7c7c', '%': '#7c7c7c', 'H': '#fca044', '^': '#e40058', 'O': '#3a0018',
  '?': '#f8d878', '<': '#f8d878', '>': '#f8d878', 'F': '#f8d878', 'L': '#f8d878', 'P': '#58d854', 'G': '#f8d878',
  'D': '#f8d878', '@': '#f8d878', 'd': '#7c7c7c' };
function draw(canvas, data, variant) {
  const rows = data.rows, cols = Math.max(...rows.map(r => r.length)), n = rows.length;
  const room = canvas.closest('section').clientWidth - 60;
  const s = Math.max(2, Math.min(16, Math.floor(room / cols)));
  canvas.width = cols * s; canvas.height = (n + 1) * s;
  const g = canvas.getContext('2d');
  g.clearRect(0, 0, canvas.width, canvas.height);
  const changed = {}; (data.cells[variant] || []).forEach(([c, r, b, a]) => changed[c + ',' + r] = a);
  for (let r = 0; r < n; r++) for (let c = 0; c < cols; c++) {
    const ch = changed[c + ',' + r] ?? (rows[r][c] || ' ');
    if (ch === '.' || ch === ' ') continue;
    g.fillStyle = TILE[ch] || (/[A-Za-z]/.test(ch) ? '#e40058' : '#7c7c7c');
    g.fillRect(c * s, r * s, s, s);
  }
  const run = data.runs[variant] || {};
  g.fillStyle = 'rgba(248,56,0,.32)';
  (run.coverage || []).forEach(([c, r]) => g.fillRect(c * s, Math.min(r, n) * s, s, s));
  g.fillStyle = '#f8f8f8';
  (run.ends || []).forEach(([c, r]) => g.fillRect(c * s + s / 4, Math.min(r, n) * s + s / 4, s / 2, s / 2));
  if (run.cluster) {
    const k = run.cluster; g.strokeStyle = '#f8f8f8'; g.lineWidth = 2;
    g.strokeRect((k.col - 3) * s, (Math.min(k.row, n) - 3) * s, 7 * s, 7 * s);
  }
  g.strokeStyle = '#f8d878'; g.lineWidth = 2;
  (data.cells[variant] || []).forEach(([c, r]) => g.strokeRect(c * s + 1, r * s + 1, s - 2, s - 2));
}
document.querySelectorAll('section.game').forEach(sec => {
  const canvas = sec.querySelector('canvas'); if (!canvas) return;
  const data = JSON.parse(document.getElementById('data-' + canvas.dataset.game).textContent);
  if (!data.rows.length) { sec.querySelector('.map').remove(); return; }
  let cur = Object.keys(data.runs)[0];
  const redraw = () => draw(canvas, data, cur);
  sec.querySelectorAll('.mapbar .btn').forEach(b => b.addEventListener('click', () => {
    sec.querySelectorAll('.mapbar .btn').forEach(x => x.classList.toggle('on', x === b));
    cur = b.dataset.variant; redraw();
  }));
  redraw(); addEventListener('resize', redraw);
});
"""


def render(results: dict) -> str:
    with open(THEME, encoding="utf-8") as f:
        theme = f.read()
    sections = "".join(_game_section(g, res) for g, res in results.items())
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>PEAK mutations</title><style>{theme}{CSS}</style></head><body><main>'
            '<header class="top"><span class="wordmark">PEAK</span><span class="banner">Mutations</span>'
            '<nav><a class="btn" href="../balance/report.html">Balance Command</a></nav></header>'
            '<div><h1>Does a playtest notice one small change?</h1><p class="lede">Each game\'s evaluation level gets five '
            'one-change copies. Agents retrained on a copy, or replaying it, are compared with agents on the original: a sealed '
            'or harder level should be caught, an off-route tile should stay quiet, and the cosmetic copy must play out identically.</p></div>'
            f'{board(results)}{sections}<div class="bricks" aria-hidden="true"></div></main><script>{JS}</script></body></html>')


def write(results: dict, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(render(results))
    return path
