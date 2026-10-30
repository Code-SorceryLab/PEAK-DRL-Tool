"""The Command Center's Mutations tab: the AST evaluation at a glance, one page per game in the
site that `python -m code.neuro.command_center` (and ast_metrics --page) writes to runs/site/.

On top, the board: one cell per game and mutant, saying whether the playtest caught it (a breaking
or degrading mutant should be caught, a neutral one should stay quiet, the cosmetic copy must
replay identically; a mutant played section by section is caught where any section catches it).
Below, per game: the level map with the change, where failed attempts ended and the tiles the
routes covered; the metric table, grouped by the level played, each mutant naming the original it
was compared with; win rates per persona; people, when they played; Mega Man's segments. One
standalone file with the Command Center's styles and top bar (code/neuro/report.py).

The 2D games' other levels carry the same five operators, placed by rule (<id>@<level>): the board
adds one row per game that totals them, and a game's section picks the level to show.
"""
from __future__ import annotations

import html
import json
import os

from code.games import eval_mutants
from code.stats.ast_metrics import RESEEDED, RETRAINED, false_alarms, is_original, kind

SLOTS = [("seal_route",), ("widen_gap", "add_enemy"), ("jump_velocity_down", "fuse_down"), ("off_route",), ("cosmetic_tile",)]
SLOT_NAMES = ["Sealed route", "Wider gap / extra enemy", "Weaker jump / shorter fuse", "Off-route tile", "Cosmetic tile"]
SHORT = {"mario": "Super Mario Bros.", "meatboy": "Super Meat Boy", "bomberman": "Bomberman"}
esc = html.escape


def _slot(operator: str) -> int:
    base = operator.split("-", 1)[-1] if operator.startswith(("s", "m")) and "-" in operator else operator
    return next(i for i, ops in enumerate(SLOTS) if base in ops)


def verdict(m: dict, entry: dict | None, replay: bool = False) -> tuple[str, str]:
    """(word, tone) for a mutant: tone is good, bad or none (not played yet, or its original is
    unsolved and so does not count). `replay` judges the frozen original agents' attempts instead."""
    if replay:
        entry = (entry or {}).get("replay")
        if not entry:
            return "not replayed", "none"
    if not entry or not entry.get("agents"):
        trained = os.path.isdir(os.path.join("runs", "ast", m["game"], m["id"], "probes"))
        return ("not played" if trained or not m.get("retrain", True) else "not retrained"), "none"
    if m["operator"].endswith("cosmetic_tile"):
        d = entry.get("determinism")
        if not d:
            return "no baseline", "none"
        return (f"identical {d['identical']}/{d['compared']}", "good") if d["passed"] else \
            (f"differs {len(d['differ'])}/{d['compared']}", "bad")
    sig = entry.get("vs_base") or {}
    if sig.get("detected") is None:
        return "no baseline", "none"
    if not sig.get("counted", True):
        return "original unsolved", "none"
    if m["class"] == "neutral":
        return ("false alarm", "bad") if sig["detected"] else ("quiet", "good")
    return ("caught", "good") if sig["detected"] else ("missed", "bad")


def judge(m: dict, res: dict, replay: bool = False) -> tuple[str, str]:
    """verdict() on the board: a mutant also played section by section (Mega Man's global jump
    change) is judged by the sections its original solves, caught where any of them catches it."""
    parts = [e for k, e in res.items() if k.startswith(m["id"] + " @ ") and e.get("agents")]
    if not parts:
        return verdict(m, res.get(m["id"]), replay)
    sigs = [(e.get("replay") or {}).get("vs_base") if replay else e.get("vs_base") for e in parts]
    counted = [s for s in sigs if s and s.get("counted")]
    if not counted:
        return ("not replayed" if replay and not any(sigs) else "original unsolved"), "none"
    hits = sum(bool(s["detected"]) for s in counted)
    word, tone = verdict(m, {"agents": True, "vs_base": {"detected": hits > 0, "counted": True}})
    return f"{word} in {hits}/{len(counted)} sections", tone


def detect(m: dict, res: dict) -> tuple[str, str]:
    """The detector's verdict: the original's frozen agents replaying the mutant (seed noise
    cannot reach them); the cosmetic copy, replayed by definition, by its determinism check."""
    return judge(m, res) if m["operator"].endswith("cosmetic_tile") else judge(m, res, replay=True)


def solvable(m: dict, res: dict) -> tuple[str, str]:
    """What retraining answers: can agents trained on the mutant still finish it? Right (good) when
    a breaking mutant is no longer solvable and any other mutant still is; judged per section for
    a global change, over the levels whose original is solved."""
    if m["operator"].endswith("cosmetic_tile"):
        return "-", "none"
    parts = [e for k, e in res.items() if k.startswith(m["id"] + " @ ") and e.get("agents")] or \
        [res.get(m["id"]) or {}]
    done = [bool(e["agents"]["completion"]) for e in parts if e.get("agents") and (e.get("vs_base") or {}).get("counted")]
    if not done:
        return ("original unsolved" if any(e.get("agents") for e in parts) else "not retrained"), "none"
    word = (("still solvable" if done[0] else "no longer solvable") if len(done) == 1
            else f"solvable in {sum(done)}/{len(done)} sections")
    return word, "good" if (m["class"] == "breaking") == (not all(done)) else "bad"


def _null_verdict(e: dict) -> tuple[str, str]:
    """base-reseeded against the original: any flag is a false alarm."""
    s = e.get("vs_base") or {}
    if not s:
        return "no baseline", "none"
    if not s.get("counted"):
        return "original unsolved", "none"
    return ("false alarm", "bad") if s["detected"] else ("quiet", "good")


def _original(game: str, name: str) -> str:
    if kind(name) == RESEEDED:
        return "The level as it ships, retrained with three new seeds: any signal it raises is a false alarm."
    if name == RETRAINED:
        return "The level as it ships, each section retrained exactly as the mutants were: their baseline."
    return "The level as it ships" + (", played by the relay's hand-picked section agents." if game == "megaman" and name == "base"
                                      else ".")


def _level(ms: list[dict]) -> tuple[list[str], dict]:
    """The evaluation level's rows and each level mutant's changed cells."""
    if not ms:
        return [], {}
    return _file_rows(ms[0]["base_file"]), {m["id"]: m.get("cells") or [] for m in ms}


def _file_rows(file: str) -> list[str]:
    with open(os.path.join(eval_mutants.LEVELS_DIR, file), encoding="utf-8") as f:
        rows = [r.rstrip("\n") for r in f]
    while rows and not rows[-1].strip():
        rows.pop()
    return rows


def _signals(e: dict) -> str:
    s = e.get("vs_base") or {}
    words = [("completion lost", s.get("completion_lost")), (f"win rate (p {s.get('win_rate_p')})", s.get("win_rate_changed")),
             ("death cluster moved", s.get("cluster_moved")), ("new top cause", s.get("cause_changed"))]
    out = [w for w, on in words if on]
    if s.get("win_rate_beyond_seeds"):
        out.append("win rate outside the seed spread")
    if "determinism" in e:
        d = e["determinism"]
        out.append(f"replay {d['identical']}/{d['compared']} identical")
    text = ", ".join(out)
    return f"{s['verdict']}: {text}" if s.get("verdict") and text else s.get("verdict") or text


def _replay(e: dict) -> str:
    """The frozen original agents on this variant: wins and verdict."""
    rep = e.get("replay")
    if not rep:
        return ""
    w, n = rep["agents"]["win_rate"]["pooled"]
    return f"{w}/{n} · {rep['vs_base']['verdict']}"


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
    cls = m["class"] if m else "null" if kind(v) == RESEEDED else "original"
    vs = f'<div class="label">vs {esc(e["vs"])}</div>' if e.get("vs") else ""
    what = f'<div class="label">{esc(desc)}</div>' if desc else ""
    tds = [f"<td><b>{esc(v)}</b>{what}</td>", f'<td><span class="cls {esc(cls)}">{esc(cls)}</span></td>',
           f'<td class="n">{w}/{n}</td>', f'<td class="n">{"yes" if a["completion"] else "no"}</td>',
           f'<td class="n">{_pct(a["progress_median"])}<span class="label"> / {_pct(a["progress_max"])}</span></td>',
           f'<td class="n">{_num(a["time_to_finish"], ".1f", " s")}</td>', f"<td>{_where(a['death_cluster'])}</td>",
           f"<td>{esc(next(iter(a['causes']), ''))}</td>", f'<td class="n">{len(a["coverage"])}</td>']
    tds += [f'<td class="n">{_num(a["effort"].get(k))}</td>' for k in effort]
    tds.append(f'<td class="sig {tone}">{esc(_signals(e))}{vs}</td>')
    tds.append(f"<td>{esc(_replay(e))}</td>")
    return f'<tr data-level="{esc(e["level"])}">' + "".join(tds) + "</tr>"


def _summary(v: str, e: dict, m: dict | None, desc: str) -> dict:
    """What the selected-variant card shows."""
    a = e.get("agents") or {}
    null = kind(v) == RESEEDED
    replayed = m and not m["operator"].endswith("cosmetic_tile")        # the detector's word for a mutant
    word, tone = verdict(m, e, replay=bool(replayed)) if m else _null_verdict(e) if null else ("the original", "none")
    w, n = (a.get("win_rate") or {}).get("pooled", [0, 0])
    return {"name": v, "cls": m["class"] if m else "null" if null else "original", "desc": desc, "replay": _replay(e) or "-",
            "word": word, "tone": tone, "wins": f"{w}/{n}", "solved": "yes" if a.get("completion") else "no",
            "progress": f"{_pct(a.get('progress_median')) or '-'} / {_pct(a.get('progress_max')) or '-'}",
            "tff": _num(a.get("time_to_finish"), ".1f", " s") or "-", "cluster": _where(a.get("death_cluster")) or "-",
            "cause": next(iter(a.get("causes") or {}), "-") or "-", "signals": _signals(e) or "-", "vs": e.get("vs") or "-"}


def _game_section(game: str, res: dict) -> str:
    title, main = eval_mutants.EVAL_LEVELS[game][1], eval_mutants.EVAL_LEVELS[game][0]
    ms = eval_mutants.all_mutants(game)
    by_id = {m["id"]: m for m in ms}
    rows = (_file_rows(eval_mutants.level_file(game, main)) if game in eval_mutants.CHANGES
            else _level(eval_mutants.eval_set(game))[0])
    cells = {m["id"]: m.get("cells") or [] for m in ms}
    name = {k: k.split(" @ ")[0] for k in res}                  # "<variant> @ <section>" is that variant
    levels = [main] + (eval_mutants.other_levels(game) if game in eval_mutants.CHANGES else [])
    rank = {lv: i for i, lv in enumerate(levels)}
    order = sorted((k for k in res if is_original(name[k]) or kind(name[k]) == RESEEDED or name[k] in by_id),
                   key=lambda k: (rank.get(res[k]["level"], len(levels)), res[k]["level"] != main, res[k]["level"]))
    shown = list(dict.fromkeys(res[v]["level"] for v in order))     # the levels played, in the game's order
    picks = [lv for lv in shown if lv in rank] if game in eval_mutants.CHANGES else []
    maps = {lv: _file_rows(eval_mutants.level_file(game, lv)) for lv in picks if lv != main}
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
                         verdict(mut[v], res[v])[1] if mut[v] else _null_verdict(res[v])[1] if kind(name[v]) == RESEEDED
                         else "none", effort))
        described.add(name[v])
    personas = sorted({p for v in order for p in ((res[v].get("agents") or {}).get("win_rate") or {}) if p != "pooled"})
    grid = "".join(
        f'<tr><td><i class="ico p-{esc(p)}"></i> {esc(p)}</td>' + "".join(
            _pips(((res[v].get("agents") or {}).get("win_rate") or {}).get(p)).replace("<td", f'<td data-level="{esc(res[v]["level"])}"', 1)
            for v in order) + "</tr>"
        for p in personas)
    humans = [(v, res[v]["humans"]) for v in order if res[v].get("humans")]
    human_html = ""
    if humans:
        human_html = ('<h3><i class="ico cursor"></i> People at the keyboard</h3><table class="data"><tr><th>Variant</th><th class="n">Attempts</th>'
                      '<th class="n">Wins</th><th>Per player</th><th class="n">Progress</th><th>Death cluster</th><th>Top cause</th></tr>'
                      + "".join(_human_row(v, h) for v, h in humans) + "</table>")
    seg_html = _segments([v for v in order if res[v]["level"] == main], res) if game == "megaman" else ""
    data = {"rows": rows, "maps": maps, "level": {v: res[v]["level"] for v in order}, "cells": cells,
            "runs": {v: {"ends": (res[v].get("agents") or {}).get("end_tiles") or [],
                         "cluster": (res[v].get("agents") or {}).get("death_cluster"),
                         "routes": (res[v].get("agents") or {}).get("routes") or []} for v in order},
            "summary": {v: _summary(v, res[v], mut[v], desc[v]) for v in order}}
    buttons = "".join(f'<button class="btn{" on" if i == 0 else ""}" data-variant="{esc(v)}" data-level="{esc(res[v]["level"])}">'
                      f'{esc(v)}</button>' for i, v in enumerate(order))
    if len(picks) > 1:                     # one level at a time: the map, its buttons, table rows and pips
        buttons = ('<select class="levelpick" aria-label="Level">' + "".join(
            f'<option value="{esc(lv)}">{esc(eval_mutants.level_name(game, lv))}</option>' for lv in picks) + "</select>" + buttons)
        title += f" and {len(picks) - 1} more levels"
    pips = (f'<div class="card"><h3><i class="ico coin"></i> Wins per persona</h3><div class="scroll"><table class="data pips">'
            f'<tr><th>Persona</th>' + "".join(f'<th data-level="{esc(res[v]["level"])}">{esc(v).replace("_", "_<wbr>")}</th>' for v in order)
            + f"</tr>{grid}</table></div></div>"
            if personas else "")
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
            f'<th>Top cause</th><th class="n"><i class="ico eye"></i> Tiles covered</th>{head}<th>Retrained: verdict and signals</th>'
            f'<th>Replay (the detector)</th></tr>{"".join(body)}</table></div>'
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
    """Wins out of attempts as coins: gold for a win, grey for a loss (a persona's three agents; a
    count for more, as Mega Man's relay attempts)."""
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
    """How many mutants the playtests got right, got wrong, and still wait for, on every level."""
    out = {"good": 0, "bad": 0, "none": 0}
    for game, res in results.items():
        for m in eval_mutants.all_mutants(game):
            out[detect(m, res)[1]] += 1
    return out


def board(results: dict) -> str:
    rows = []
    for game, res in results.items():
        cells = ["<td></td>"] * len(SLOTS)
        for m in eval_mutants.eval_set(game):
            word, tone = detect(m, res)
            icon = {"good": '<i class="ico coin"></i>', "bad": '<i class="ico heartbreak"></i>'}.get(tone, "")
            slot = _slot(m["id"])
            rep = "" if m["operator"].endswith("cosmetic_tile") else '<span class="rep {1}">retrained: {0}</span>'.format(
                *map(esc, solvable(m, res)))
            cells[slot] = (f'<td class="cell {tone}"><a href="#{game}" data-slot="{SLOT_NAMES[slot]}"><span class="v">{icon}{esc(word)}</span>'
                           f'<span class="label">{esc(m["class"])}</span>{rep}</a></td>')
        rows.append(f'<tr><th class="g g-{game}"><a href="#{game}">{_icon(game)}{esc(eval_mutants.EVAL_LEVELS[game][1])}</a></th>'
                    f'{"".join(cells)}</tr>')
        placed = eval_mutants.placed_set(game)
        if placed:
            rows.append(_others_row(game, res, placed))
    return ('<div class="scroll"><table class="board"><tr><th></th>' + "".join(f"<th>{n}</th>" for n in SLOT_NAMES) + "</tr>"
            + "".join(rows) + "</table></div>")


def _others_row(game: str, res: dict, placed: list[dict]) -> str:
    """One board row for a game's other levels: per operator, how many levels the playtest got right."""
    cells = ["<td></td>"] * len(SLOTS)
    for slot in range(len(SLOTS)):
        ms = [m for m in placed if _slot(m["operator"]) == slot]
        if not ms:
            continue
        tones = [detect(m, res)[1] for m in ms]
        good, bad = tones.count("good"), tones.count("bad")
        tone = "bad" if bad else "good" if good else "none"
        icon = {"good": '<i class="ico coin"></i>', "bad": '<i class="ico heartbreak"></i>'}.get(tone, "")
        word = f"right {good}/{good + bad}" if good + bad else "waiting"
        label = f"{good + bad or len(ms)} levels"     # levels whose original is unsolved are left out
        rep = ""
        if not ms[0]["operator"].endswith("cosmetic_tile"):
            st = [solvable(m, res)[1] for m in ms]
            sg, sb = st.count("good"), st.count("bad")
            rep = (f'<span class="rep {"bad" if sb else "good" if sg else "none"}">retrained: '
                   f'{f"right {sg}/{sg + sb}" if sg + sb else "waiting"}</span>')
        cells[slot] = (f'<td class="cell {tone}"><a href="#{game}" data-slot="{SLOT_NAMES[slot]}"><span class="v">{icon}{esc(word)}</span>'
                       f'<span class="label">{esc(label)}</span>{rep}</a></td>')
    n = len({m["stage"] for m in placed})
    return (f'<tr><th class="g g-{game}"><a href="#{game}">{_icon(game)}{esc(SHORT.get(game, game))}: other levels ({n})</a></th>'
            f'{"".join(cells)}</tr>')


CSS = """
main { display: flex; flex-direction: column; gap: 22px; }
[hidden] { display: none !important; }
.levelpick { font: 700 15px/1.2 var(--pixel); padding: 6px 8px; background: var(--panel, #111); color: inherit; border: 2px solid currentColor; }
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
.cls.neutral { color: var(--run); } .cls.base, .cls.null { color: var(--dim); }
.board .rep { font: 500 .78rem/1.3 var(--ui); color: var(--dim); margin-top: 2px; }
.board .rep.good { color: var(--won); } .board .rep.bad { color: var(--dead); }
.fa { padding: 14px 16px; } .fa table.data { max-width: 560px; }
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
  const rows = (data.maps && data.maps[data.level[variant]]) || data.rows, cols = Math.max(...rows.map(r => r.length)), n = rows.length;
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
    <dl><dt>Replay (the detector)</dt><dd>${esc(s.replay)}</dd>
    <dt>Retrained: wins</dt><dd>${esc(s.wins)}</dd><dt>Retrained agents finish it</dt><dd>${esc(s.solved)}</dd>
    <dt>Progress (median / max)</dt><dd>${esc(s.progress)}</dd><dt>Time to finish</dt><dd>${esc(s.tff)}</dd>
    <dt>Death cluster</dt><dd>${esc(s.cluster)}</dd><dt>Top cause</dt><dd>${esc(s.cause)}</dd>
    <dt>Retrained: verdict and signals</dt><dd>${esc(s.signals)}</dd>
    <dt>Compared with</dt><dd>${esc(s.vs)}</dd></dl>`;
}
const redraws = {};
document.querySelectorAll('section.game').forEach(sec => {
  const canvas = sec.querySelector('canvas'); if (!canvas) return;
  const data = JSON.parse(document.getElementById('data-' + canvas.dataset.game).textContent);
  const card = sec.querySelector('.vsum');
  let cur = Object.keys(data.runs)[0];
  const mapOf = v => (data.maps && data.maps[data.level[v]]) || data.rows;
  const redraw = () => { if (mapOf(cur).length && !sec.hidden) draw(canvas, data, cur); if (card) summarize(card, data.summary[cur]); };
  if (!data.rows.length) sec.querySelector('.map').remove();
  sec.querySelectorAll('.mapbar .btn').forEach(b => b.addEventListener('click', () => {
    sec.querySelectorAll('.mapbar .btn').forEach(x => x.classList.toggle('on', x === b));
    cur = b.dataset.variant; redraw();
  }));
  const pick = sec.querySelector('.levelpick');
  const filter = () => {                      // one level at a time: its buttons, table rows and pips
    sec.querySelectorAll('[data-level]').forEach(el => { el.hidden = el.dataset.level !== pick.value; });
    const first = [...sec.querySelectorAll('.mapbar .btn')].find(b => b.dataset.level === pick.value);
    if (first) first.click();
  };
  if (pick) { pick.addEventListener('change', filter); filter(); }
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


def render(results: dict, only: str | None = None, nav: bool = True) -> str:
    """The page; with `only`, one game's board row, tab and section (the Command Center's Mutations
    tab), while the score strip and the false-alarm card still count every game. `nav` off drops
    the top bar."""
    from code.neuro import report   # its styles and top bar: one site with the other Command Center pages
    shown = {g: res for g, res in results.items() if only in (None, g)}
    sections = "".join(_game_section(g, res) for g, res in shown.items())
    sc = scores(results)
    fa = false_alarms(results)
    strip = ('<div class="scorestrip">'
             f'<div class="frame good"><span class="label">Right</span><span class="num"><i class="ico coin"></i>{sc["good"]}</span></div>'
             f'<div class="frame bad"><span class="label">Wrong</span><span class="num"><i class="ico heartbreak"></i>{sc["bad"]}</span></div>'
             '</div>')
    tabs = ('<div class="gametabs" role="tablist" aria-label="Games">'
            + "".join(f'<button class="btn g-{g}" role="tab" type="button" data-game="{g}" aria-controls="{g}">{_icon(g)}'
                      f'{esc(eval_mutants.EVAL_LEVELS[g][1].split(",")[0])}</button>' for g in shown) + "</div>")
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>PEAK Command Center: Mutations</title><style>{report.CSS}{CSS}</style></head><body>'
            f'{report._nav([], False, None, "mutations") if nav else ""}<main id="main">'
            '<div class="hero"><div><h1>Does a playtest notice one small change?</h1><p class="lede">Every level of each game gets five '
            'one-change copies: placed by hand on the evaluation level, by rule on the others. The original\'s agents, frozen, replay each copy and are compared with their own play on the '
            'original: a sealed or harder level should be caught, an off-route tile should stay quiet, and the cosmetic copy must '
            'play out identically. Agents retrained on a copy answer one question only: can it still be finished?'
            + (f' (Retrained agents are too noisy to detect with: on the unchanged level, new seeds set off the rule '
               f'{fa["fired"]["detected"]} times in {fa["comparisons"]}.)' if fa["comparisons"] else "") + '</p></div>'
            f'{strip}</div>{board(shown)}{_false_alarm_card(results)}{tabs}{sections}<div class="bricks" aria-hidden="true"></div></main><script>{JS}</script></body></html>')


def _false_alarm_card(results: dict) -> str:
    """How often each signal fires when the original is compared with itself (base-reseeded)."""
    fa = false_alarms(results)
    if not fa["comparisons"]:
        return ('<p class="empty"><i class="ico hourglass"></i> False-alarm check not run yet: retrain the original with '
                'three new seeds (<code>python -m code.neuro.ast_eval retrain --game &lt;game&gt; --mutant base-reseeded</code>), '
                'then play it.</p>')
    n = fa["comparisons"]
    labels = (("completion_lost", "completion lost"), ("win_rate_changed", "win rate (Fisher p < .05)"),
              ("cluster_moved", "death cluster moved more than 3 tiles"), ("cause_changed", "new top cause"),
              ("detected", "any of them: the rule"), ("strict", "completion lost or win rate only (strict)"))
    rows = "".join(f'<tr><td>{esc(label)}</td><td class="n">{fa["fired"][k]}/{n}</td></tr>' for k, label in labels)
    w, m = fa["beyond_seeds"]
    if m:
        rows += f'<tr><td>win rate outside the seed spread (beside the rule)</td><td class="n">{w}/{m}</td></tr>'
    return (f'<div class="card fa"><h3><i class="ico eye"></i> Why retraining is not the detector</h3>'
            f'<p class="lede">The original retrained with three new seeds and played like a mutant, over {n} comparisons '
            '(a level, or a Mega Man section, whose original is solved). Every signal it raises is a false alarm: seed noise. '
            'Replay has none, because frozen agents replay attempt for attempt.</p>'
            f'<table class="data"><tr><th>Signal</th><th class="n">Fired</th></tr>{rows}</table></div>')


def write(results: dict, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(render(results))
    return path
