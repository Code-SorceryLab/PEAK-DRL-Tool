"""The Balance Command's Mutations page, runs/balance/mutations.html: the AST evaluation at a glance
(written by `python -m code.neuro.report` with the other pages, and by ast_metrics --page).

On top, the board: one cell per game and mutant, saying whether the playtest caught it (a breaking
or degrading mutant should be caught, a neutral one should stay quiet, the cosmetic copy must
replay identically; a mutant played section by section is caught where any section catches it).
Below, per game: the level map with the change, where failed attempts ended and the tiles the
routes covered; the metric table, grouped by the level played, each mutant naming the original it
was compared with; win rates per persona; people, when they played; Mega Man's segments. One
standalone file with the Balance Command's styles and top bar (code/neuro/report.py).
"""
from __future__ import annotations

import html
import json
import os

from code.games import eval_mutants
from code.stats.ast_metrics import ORIGINALS, RETRAINED

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


def judge(m: dict, res: dict) -> tuple[str, str]:
    """verdict() on the board: a mutant also played section by section (Mega Man's global jump
    change) is judged by its sections, caught where any of them catches it."""
    parts = [e for k, e in res.items() if k.startswith(m["id"] + " @ ") and e.get("agents")]
    if not parts:
        return verdict(m, res.get(m["id"]))
    hits = sum(bool((e.get("vs_base") or {}).get("detected")) for e in parts)
    word, tone = verdict(m, {"agents": True, "vs_base": {"detected": hits > 0}})
    return f"{word} in {hits}/{len(parts)} sections", tone


def _original(game: str, name: str) -> str:
    if name == RETRAINED:
        return "The level as it ships, each section retrained exactly as the mutants were: their baseline."
    return "The level as it ships" + (", played by the relay's hand-picked section agents." if game == "megaman" else ".")


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


def _row(v: str, e: dict, m: dict | None, desc: str, tone: str, effort: list[str]) -> str:
    """One variant's line of the metric table."""
    a = e["agents"]
    w, n = a["win_rate"]["pooled"]
    cls = m["class"] if m else "original"
    vs = f'<div class="label">vs {esc(e["vs"])}</div>' if e.get("vs") else ""
    what = f'<div class="label">{esc(desc)}</div>' if desc else ""
    tds = [f"<td><b>{esc(v)}</b>{what}</td>", f'<td><span class="cls {esc(cls)}">{esc(cls)}</span></td>',
           f'<td class="n">{w}/{n}</td>', f'<td class="n">{"yes" if a["completion"] else "no"}</td>',
           f'<td class="n">{_pct(a["progress_median"])}<span class="label"> / {_pct(a["progress_max"])}</span></td>',
           f'<td class="n">{_num(a["time_to_finish"], ".1f", " s")}</td>', f"<td>{_where(a['death_cluster'])}</td>",
           f"<td>{esc(next(iter(a['causes']), ''))}</td>", f'<td class="n">{len(a["coverage"])}</td>']
    tds += [f'<td class="n">{_num(a["effort"].get(k))}</td>' for k in effort]
    tds.append(f'<td class="sig {tone}">{esc(_signals(e))}{vs}</td>')
    return "<tr>" + "".join(tds) + "</tr>"


def _summary(v: str, e: dict, m: dict | None, desc: str) -> dict:
    """What the selected-variant card shows."""
    a = e.get("agents") or {}
    word, tone = verdict(m, e) if m else ("the original", "none")
    w, n = (a.get("win_rate") or {}).get("pooled", [0, 0])
    return {"name": v, "cls": m["class"] if m else "original", "desc": desc,
            "word": word, "tone": tone, "wins": f"{w}/{n}", "solved": "yes" if a.get("completion") else "no",
            "progress": f"{_pct(a.get('progress_median')) or '-'} / {_pct(a.get('progress_max')) or '-'}",
            "tff": _num(a.get("time_to_finish"), ".1f", " s") or "-", "cluster": _where(a.get("death_cluster")) or "-",
            "cause": next(iter(a.get("causes") or {}), "-") or "-", "signals": _signals(e) or "-", "vs": e.get("vs") or "-"}


def _game_section(game: str, res: dict) -> str:
    title, main = eval_mutants.EVAL_LEVELS[game][1], eval_mutants.EVAL_LEVELS[game][0]
    ms = eval_mutants.eval_set(game)
    by_id = {m["id"]: m for m in ms}
    rows, cells = _level(ms)
    name = {k: k.split(" @ ")[0] for k in res}                  # "<variant> @ <section>" is that variant
    order = sorted((k for k in res if name[k] in ORIGINALS or name[k] in by_id),
                   key=lambda k: (res[k]["level"] != main, res[k]["level"]))     # grouped by the level played
    mut = {v: by_id.get(name[v]) for v in order}
    desc = {v: mut[v]["description"] if mut[v] else _original(game, name[v]) for v in order}
    if not order:
        return (f'<section class="frame stripe g-{game} game" id="{game}" role="tabpanel"><h2>{esc(title)}</h2>'
                f'<p class="empty"><i class="ico qblock"></i> No playtests yet. Run Agent Playtest (menu 20) or '
                f'<code>python -m code.neuro.ast_eval play --game {game}</code>.</p></section>')
    effort = sorted({k for v in order for k in ((res[v].get("agents") or {}).get("effort") or {})})
    head = "".join(f'<th class="n">{_ICON_OF.get(k, "")}{esc(k)}</th>' for k in effort)
    body, described = [], set()
    for v in (v for v in order if res[v].get("agents")):          # a variant's description on its first row only
        body.append(_row(v, res[v], mut[v], "" if name[v] in described else desc[v],
                         verdict(mut[v], res[v])[1] if mut[v] else "none", effort))
        described.add(name[v])
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
    seg_html = _segments([v for v in order if res[v]["level"] == main], res) if game == "megaman" else ""
    data = {"rows": rows, "cells": cells,
            "runs": {v: {"ends": (res[v].get("agents") or {}).get("end_tiles") or [],
                         "cluster": (res[v].get("agents") or {}).get("death_cluster"),
                         "routes": (res[v].get("agents") or {}).get("routes") or []} for v in order},
            "summary": {v: _summary(v, res[v], mut[v], desc[v]) for v in order}}
    buttons = "".join(f'<button class="btn{" on" if i == 0 else ""}" data-variant="{esc(v)}">{esc(v)}</button>'
                      for i, v in enumerate(order))
    pips = (f'<div class="card"><h3><i class="ico coin"></i> Wins per persona</h3><div class="scroll"><table class="data pips">'
            f'<tr><th>Persona</th>' + "".join(f"<th>{esc(v).replace('_', '_<wbr>')}</th>" for v in order) + f"</tr>{grid}</table></div></div>"
            if personas and game != "megaman" else "")
    side = pips or (f'<div class="card">{seg_html}</div>' if seg_html else "")
    return (f'<section class="frame stripe g-{game} game" id="{game}" role="tabpanel"><h2>{esc(title)}</h2>'
            f'<div class="mapbar">{buttons}</div>'
            f'<div class="map"><canvas data-game="{game}"></canvas></div>'
            '<p class="legend"><i class="k-won"></i>winning runs <i class="k-lost"></i>failed runs, ending where they died '
            '<i class="k-clu"></i>death cluster (3 tiles) <i class="k-cell"></i>the mutated tiles</p>'
            f'<script type="application/json" id="data-{game}">{json.dumps(data, separators=(",", ":"))}</script>'
            '<div class="duo"><div class="card vsum" aria-live="polite"></div>' + side + "</div>"
            f'<h3><i class="ico flag"></i>Every variant</h3><div class="scroll"><table class="data"><tr><th>Variant</th><th>Class</th><th class="n"><i class="ico coin"></i> Wins</th><th class="n">Solved</th>'
            f'<th class="n"><i class="ico up"></i> Progress (median / max)</th><th class="n"><i class="ico hourglass"></i> Time to finish</th><th><i class="ico skull"></i> Death cluster</th>'
            f'<th>Top cause</th><th class="n"><i class="ico eye"></i> Tiles covered</th>{head}<th>Signals</th></tr>{"".join(body)}</table></div>'
            + human_html + ("" if not pips else seg_html) + "</section>")


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
    return (f'<h3><i class="ico flag"></i>Segments of the full stage</h3><div class="scroll"><table class="data"><tr><th>Segment</th>{head}</tr>'
            f'{"".join(rows)}</table></div>')


def _icon(game: str) -> str:
    try:
        from code.neuro.report import _game_icon       # the Command center's game sprites
        return _game_icon(game)
    except Exception:  # noqa: BLE001 — the board works without them
        return ""


def scores(results: dict) -> dict:
    """How many mutants the playtests got right, got wrong, and still wait for."""
    out = {"good": 0, "bad": 0, "none": 0}
    for game, res in results.items():
        for m in eval_mutants.eval_set(game):
            out[judge(m, res)[1]] += 1
    return out


def board(results: dict) -> str:
    rows = []
    for game, res in results.items():
        cells = ["<td></td>"] * len(SLOTS)
        for m in eval_mutants.eval_set(game):
            word, tone = judge(m, res)
            icon = {"good": '<i class="ico coin"></i>', "bad": '<i class="ico heartbreak"></i>'}.get(tone, "")
            slot = _slot(m["id"])
            cells[slot] = (f'<td class="cell {tone}"><a href="#{game}" data-slot="{SLOT_NAMES[slot]}"><span class="v">{icon}{esc(word)}</span>'
                                     f'<span class="label">{esc(m["class"])}</span></a></td>')
        rows.append(f'<tr><th class="g g-{game}"><a href="#{game}">{_icon(game)}{esc(eval_mutants.EVAL_LEVELS[game][1])}</a></th>'
                    f'{"".join(cells)}</tr>')
    return ('<div class="scroll"><table class="board"><tr><th></th>' + "".join(f"<th>{n}</th>" for n in SLOT_NAMES) + "</tr>"
            + "".join(rows) + "</table></div>")


CSS = """
main { display: flex; flex-direction: column; gap: 22px; }
.lede { color: var(--dim); max-width: 70ch; margin: 6px 0 0; }
.board { border-collapse: separate; border-spacing: 6px; width: 100%; min-width: 680px; table-layout: fixed; }
.board th, .board td { text-align: left; white-space: normal; border: 0; }   /* not the report's number columns */
table.data th { white-space: normal; vertical-align: bottom; }   /* headers wrap between words, the tables fit */
table.data:not(.pips) td:first-child { white-space: normal; min-width: 16em; }   /* so do the variant descriptions */
table.data td.sig { white-space: normal; min-width: 12em; }        /* and the signal list */
table.data td { text-align: left; }                                /* words read from the left; td.n stays right */
table.data th, table.data td { padding: 7px 10px; }
.board th { font: 600 .82rem/1.3 var(--ui); color: var(--dim); padding: 0 4px 2px; }
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
section.game { padding: 18px 22px 22px; display: flex; flex-direction: column; gap: 14px; }
section.game[hidden] { display: none; }
.hero { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 18px 32px; align-items: end; }
.scorestrip { display: flex; gap: 10px; }
.scorestrip .frame { padding: 10px 16px; display: flex; flex-direction: column; gap: 4px; min-width: 110px; }
.scorestrip .num { font-size: 36px; display: flex; align-items: center; gap: 10px; }
.scorestrip .num .ico { --ico: 26px; }
.scorestrip .good .num { color: var(--won); } .scorestrip .bad .num { color: var(--dead); } .scorestrip .none .num { color: var(--mute); }
.gametabs { display: flex; gap: 8px; flex-wrap: wrap; }
.gametabs .btn { font: 700 18px/1.2 var(--pixel); padding: 8px 14px; }
.gametabs .btn img.gicon, .gametabs .btn .gletter { height: 22px; }
.duo { display: grid; grid-template-columns: minmax(260px, 1fr) minmax(0, 1.4fr); gap: 14px; align-items: start; }
.card { background: var(--night); border: 3px solid transparent; border-image: var(--frame) 1 / 3px stretch; padding: 14px 16px; min-width: 0; }
.card h3 { margin-bottom: 10px; }
.vsum .vname { font: 700 27px/1.1 var(--pixel); margin: 0 0 6px; overflow-wrap: anywhere; }
.vsum .vverdict { font: 700 18px/1.2 var(--pixel); display: flex; align-items: center; gap: 8px; margin: 10px 0; }
.vsum .vverdict .ico { --ico: 20px; }
.vsum .vverdict.good { color: var(--won); } .vsum .vverdict.bad { color: var(--dead); } .vsum .vverdict.none { color: var(--mute); }
.vsum dl { display: grid; grid-template-columns: auto 1fr; gap: 6px 16px; margin: 0; font: 500 .9rem/1.4 var(--dense); }
.vsum dt { color: var(--dim); } .vsum dd { margin: 0; color: var(--ink); }
@media (max-width: 900px) { .hero, .duo { grid-template-columns: minmax(0, 1fr); } .scorestrip { flex-wrap: wrap; }
  .board th.g { width: 22%; font: 600 .9rem/1.25 var(--ui); } .board th.g a { gap: 6px; } }   /* tablets: names fit their column */
@media (max-width: 700px) {   /* phones: each game's verdicts as a card, the slot named inside each cell */
  .board, .board tbody { display: block; min-width: 0; }
  .board tr:first-child, .board td:not(.cell) { display: none; }
  .board tr { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-bottom: 14px; }
  .board th.g { grid-column: 1 / -1; width: auto; padding: 4px 10px; }
  .board td.cell a::before { content: attr(data-slot); font: 600 .75rem/1.2 var(--ui); color: var(--dim); }
}
@media (max-width: 560px) { .scorestrip { flex-wrap: nowrap; } .scorestrip .frame { flex: 1 1 0; min-width: 0; padding: 8px 10px; }
  .scorestrip .num { font-size: 27px; } }
.mapbar { display: flex; gap: 6px; flex-wrap: wrap; }
.map { overflow-x: auto; background: #000; padding: 8px; width: fit-content; max-width: 100%; box-sizing: border-box; }
.map canvas { display: block; }
.legend { color: var(--dim); font-size: .85rem; margin: 0; display: flex; flex-wrap: wrap; gap: 4px 16px; align-items: center; }
.legend i { display: inline-block; width: 12px; height: 12px; margin-right: 6px; vertical-align: -1px; }
.k-won { height: 3px !important; background: var(--run); } .k-lost { height: 3px !important; background: var(--dead); }
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
const TILE = { '#': '#4a4a4a', '=': '#a0a0a0', '%': '#a0a0a0', 'H': '#fca044', '^': '#e40058', 'O': '#3a0018',
  '?': '#f8d878', '<': '#f8d878', '>': '#f8d878', 'F': '#f8d878', 'L': '#f8d878', 'P': '#58d854', 'G': '#f8d878',
  'D': '#f8d878', '@': '#f8d878', 'd': '#a0a0a0' };
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
    g.fillStyle = TILE[ch] || (/[A-Za-z]/.test(ch) ? '#e40058' : '#a0a0a0');
    g.fillRect(c * s, r * s, s, s);
  }
  const run = data.runs[variant] || {};
  const lw = Math.max(1.2, s / 4);
  const line = (r, color, width, alpha) => {
    g.strokeStyle = color; g.lineWidth = width; g.globalAlpha = alpha; g.lineJoin = 'round'; g.beginPath();
    r.p.forEach(([x, y], i) => i ? g.lineTo(x * s, Math.min(y, n + 0.5) * s) : g.moveTo(x * s, Math.min(y, n + 0.5) * s));
    g.stroke(); g.globalAlpha = 1;
  };
  (run.routes || []).filter(r => !r.w).forEach(r => line(r, '#e40058', lw, 0.55));   // failed runs, red
  (run.routes || []).filter(r => r.w).forEach(r => line(r, '#58d854', lw * 1.7, 0.9));   // the route the winners take, green
  if (run.cluster) {
    const k = run.cluster; g.strokeStyle = '#ededed'; g.lineWidth = 2;
    g.strokeRect((k.col - 3) * s, (Math.min(k.row, n) - 3) * s, 7 * s, 7 * s);
  }
  g.strokeStyle = '#f8d878'; g.lineWidth = 2;
  (data.cells[variant] || []).forEach(([c, r]) => g.strokeRect(c * s + 1, r * s + 1, s - 2, s - 2));
}
const ICON = {good: '<i class="ico coin"></i>', bad: '<i class="ico heartbreak"></i>', none: ''};
const esc = t => String(t).replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
function summarize(card, s) {
  card.innerHTML = `<p class="label">Selected variant</p><p class="vname">${esc(s.name)}</p>
    <span class="cls ${esc(s.cls)}">${esc(s.cls)}</span><p class="label">${esc(s.desc)}</p>
    <p class="vverdict ${s.tone}">${ICON[s.tone] || ''}${esc(s.word)}</p>
    <dl><dt>Wins</dt><dd>${esc(s.wins)}</dd><dt>Solved</dt><dd>${esc(s.solved)}</dd>
    <dt>Progress (median / max)</dt><dd>${esc(s.progress)}</dd><dt>Time to finish</dt><dd>${esc(s.tff)}</dd>
    <dt>Death cluster</dt><dd>${esc(s.cluster)}</dd><dt>Top cause</dt><dd>${esc(s.cause)}</dd>
    <dt>Signals</dt><dd>${esc(s.signals)}</dd><dt>Compared with</dt><dd>${esc(s.vs)}</dd></dl>`;
}
const redraws = {};
document.querySelectorAll('section.game').forEach(sec => {
  const canvas = sec.querySelector('canvas'); if (!canvas) return;
  const data = JSON.parse(document.getElementById('data-' + canvas.dataset.game).textContent);
  const card = sec.querySelector('.vsum');
  let cur = Object.keys(data.runs)[0];
  const redraw = () => { if (data.rows.length && !sec.hidden) draw(canvas, data, cur); if (card) summarize(card, data.summary[cur]); };
  if (!data.rows.length) sec.querySelector('.map').remove();
  sec.querySelectorAll('.mapbar .btn').forEach(b => b.addEventListener('click', () => {
    sec.querySelectorAll('.mapbar .btn').forEach(x => x.classList.toggle('on', x === b));
    cur = b.dataset.variant; redraw();
  }));
  redraws[sec.id] = redraw;
  addEventListener('resize', redraw);
});
function show(game) {
  const secs = [...document.querySelectorAll('section.game')];
  if (!secs.some(s => s.id === game)) game = secs.length ? secs[0].id : null;
  secs.forEach(s => { s.hidden = s.id !== game; });
  document.querySelectorAll('.gametabs .btn').forEach(b => {
    const on = b.dataset.game === game; b.classList.toggle('on', on); b.setAttribute('aria-selected', on);
  });
  if (redraws[game]) redraws[game]();
}
document.querySelectorAll('.gametabs .btn').forEach(b => b.addEventListener('click', () => {
  history.replaceState(null, '', '#' + b.dataset.game); show(b.dataset.game);
}));
addEventListener('hashchange', () => show(location.hash.slice(1)));
show(location.hash.slice(1));
"""


def render(results: dict) -> str:
    from code.neuro import report   # its styles and top bar: one site with the other Balance Command pages
    sections = "".join(_game_section(g, res) for g, res in results.items())
    sc = scores(results)
    strip = ('<div class="scorestrip">'
             f'<div class="frame good"><span class="label">Right</span><span class="num"><i class="ico coin"></i>{sc["good"]}</span></div>'
             f'<div class="frame bad"><span class="label">Wrong</span><span class="num"><i class="ico heartbreak"></i>{sc["bad"]}</span></div>'
             f'<div class="frame none"><span class="label">Waiting</span><span class="num"><i class="ico hourglass"></i>{sc["none"]}</span></div>'
             '</div>')
    tabs = ('<div class="gametabs" role="tablist" aria-label="Games">'
            + "".join(f'<button class="btn g-{g}" role="tab" type="button" data-game="{g}" aria-controls="{g}">{_icon(g)}'
                      f'{esc(eval_mutants.EVAL_LEVELS[g][1].split(",")[0])}</button>' for g in results) + "</div>")
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>PEAK Balance Command — Mutations</title><style>{report.CSS}{CSS}</style></head><body>'
            f'{report._nav([], False, None, "mutations")}<main id="main">'
            '<div class="hero"><div><h1>Does a playtest notice one small change?</h1><p class="lede">Each game\'s evaluation level gets five '
            'one-change copies. Agents retrained on a copy, or replaying it, are compared with agents on the original: a sealed '
            'or harder level should be caught, an off-route tile should stay quiet, and the cosmetic copy must play out identically.</p></div>'
            f'{strip}</div>{board(results)}{tabs}{sections}<div class="bricks" aria-hidden="true"></div></main><script>{JS}</script></body></html>')


def write(results: dict, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(render(results))
    return path
