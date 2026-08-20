"""Balance web report: one self-contained HTML command center from runs/balance/report_*.json.

Structure: one section per GAME. Each opens on an Overview (all personas side by side,
one row per level), then persona tabs show that persona's expandable level cards —
click a card and it widens in place to reveal metrics, death map, causes, and curves.

Run:  python -m code.neuro.report [--dir runs/balance] [--open]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import webbrowser
from datetime import datetime

CSS = """
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
  :root{--bg:#070708;--panel:#101012;--panel2:#161618;--line:#232326;--line2:#2c2c30;
    --txt:#e8e8ea;--dim:#9a9aa0;--faint:#606066;
    --red:#ef4444;--red-dim:#b91c1c;--blue:#4a9eff;--yellow:#eab308;--green:#22c55e;
    --mono:'IBM Plex Mono',Consolas,monospace;--ui:'IBM Plex Sans','Segoe UI',sans-serif}
  *{box-sizing:border-box;margin:0}
  html{font-size:clamp(15px,1.05vw + 10px,17px)}
  body{background:var(--bg);color:var(--txt);font:1rem/1.55 var(--ui);padding:clamp(12px,3vw,32px)}
  main{max-width:1280px;margin:0 auto}

  /* command bar */
  .cmdbar{display:flex;flex-wrap:wrap;align-items:baseline;gap:12px;border:1px solid var(--line2);
    border-left:4px solid var(--red);background:var(--panel);border-radius:8px;
    padding:14px 20px;margin-bottom:22px}
  .cmdbar h1{font:700 1.3rem var(--mono);letter-spacing:.06em;color:#fff}
  .cmdbar h1 em{font-style:normal;color:var(--red)}
  .cmdbar .stamp{font:400 .78rem var(--mono);color:var(--faint);margin-left:auto}

  section.game{border:1px solid var(--line);border-radius:10px;background:var(--panel);
    padding:clamp(14px,2.5vw,24px);margin-bottom:26px}
  .gamehead{display:flex;flex-wrap:wrap;align-items:center;gap:14px;margin-bottom:16px}
  .gametag{font:700 1.05rem var(--mono);letter-spacing:.14em;text-transform:uppercase;
    color:#fff;background:var(--red-dim);border-radius:6px;padding:5px 14px}
  .gamemeta{font:400 .78rem var(--mono);color:var(--faint)}

  /* persona tabs */
  .tabs{display:flex;flex-wrap:wrap;gap:8px;margin-left:auto}
  .tab{font:600 .78rem var(--mono);letter-spacing:.06em;text-transform:uppercase;cursor:pointer;
    color:var(--dim);background:var(--panel2);border:1px solid var(--line2);border-radius:6px;
    padding:7px 14px;transition:all .15s}
  .tab:hover{color:var(--txt);border-color:var(--faint)}
  .tab.active{color:#fff;background:var(--red-dim);border-color:var(--red)}
  .tab.active.t-overview{background:#1d3a5f;border-color:var(--blue)}
  .view{display:none} .view.active{display:block}

  /* overview: one row per level, one column per persona */
  .ovwrap{overflow-x:auto;border:1px solid var(--line);border-radius:8px}
  table{border-collapse:collapse;font:400 .85rem var(--mono);font-variant-numeric:tabular-nums;width:100%}
  th{font:600 .68rem var(--ui);letter-spacing:.1em;text-transform:uppercase;color:var(--dim);
     text-align:right;padding:10px 14px;border-bottom:1px solid var(--line2);white-space:nowrap;
     background:var(--panel2)}
  td{text-align:right;padding:9px 14px;border-bottom:1px solid #1b1b1e;white-space:nowrap;color:var(--dim)}
  th:first-child,td:first-child{text-align:left}
  td:first-child{color:var(--txt);font-weight:600}
  tr:hover td{background:#141416}
  .wr-good{color:var(--green);font-weight:700}.wr-mid{color:var(--yellow);font-weight:700}
  .wr-bad{color:var(--red);font-weight:700}.wr-na{color:var(--faint)}
  .ovnote{font:400 .74rem var(--ui);color:var(--faint);margin-top:8px}

  .pill{font:600 .66rem var(--mono);border-radius:4px;padding:2px 8px;white-space:nowrap}
  .pill-ok{color:var(--green);background:rgba(34,197,94,.1);border:1px solid rgba(34,197,94,.35)}
  .pill-warn{color:var(--yellow);background:rgba(234,179,8,.1);border:1px solid rgba(234,179,8,.35)}
  .pill-bad{color:var(--red);background:rgba(239,68,68,.12);border:1px solid rgba(239,68,68,.4)}

  /* persona view: expandable cards */
  .lvlgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}
  .lvlcard{background:var(--panel2);border:1px solid var(--line2);border-radius:8px;overflow:hidden;
    transition:border-color .15s}
  .lvlcard:hover{border-color:var(--faint)}
  .lvlcard.open{grid-column:1/-1;border-color:var(--red)}
  .cardhead{display:block;width:100%;text-align:left;background:none;border:0;color:var(--txt);
    font:inherit;cursor:pointer;padding:14px 16px}
  .lvlname{font:600 .9rem var(--mono);color:#e5e5e5;margin-bottom:8px;display:flex;
    justify-content:space-between;align-items:center;gap:8px}
  .bignum{font:700 1.9rem var(--mono)}
  .lvlsub{font:400 .74rem var(--mono);color:var(--faint);margin-top:2px}
  .mini{height:4px;border-radius:2px;background:var(--line);margin-top:10px;overflow:hidden}
  .mini i{display:block;height:100%}

  .detail{display:none;padding:4px 16px 18px;border-top:1px solid var(--line2)}
  .lvlcard.open .detail{display:block;animation:slide .18s ease-out}
  @keyframes slide{from{opacity:0;transform:translateY(-6px)}to{opacity:1;transform:none}}
  .stats{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:10px;margin:14px 0 16px}
  .stat{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
  .stat .lbl{font:400 .64rem var(--ui);color:var(--faint);text-transform:uppercase;letter-spacing:.08em}
  .stat .val{font:600 1.1rem var(--mono);margin-top:2px}
  .stat .sub{font:400 .68rem var(--mono);color:var(--faint);margin-top:1px}
  .viz{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px 14px;margin-top:10px}
  .viz .vt{font:600 .76rem var(--ui);color:var(--dim);margin-bottom:8px}
  .caption{font:400 .72rem var(--ui);color:var(--faint);margin-top:6px;line-height:1.5}
  .heat{display:grid;grid-template-columns:repeat(10,1fr);gap:2px}
  .heat i{height:20px;border-radius:2px;background:var(--line)}
  .heatlbl{display:flex;justify-content:space-between;font:400 .66rem var(--mono);color:var(--faint);margin-top:3px}
  .causebar{display:flex;height:15px;gap:1px;border-radius:3px;overflow:hidden}
  .causekey{font:400 .74rem var(--mono);color:var(--dim);margin-top:6px}
  canvas.curve{width:100%;height:190px;display:block}

  .tbltitle{font:600 .78rem var(--ui);letter-spacing:.1em;text-transform:uppercase;
    color:var(--dim);margin:20px 0 8px}
  .gloss{background:var(--panel);border:1px solid var(--line);border-left:4px solid var(--blue);
    border-radius:8px;padding:14px 18px;margin-top:22px;font-size:.82rem;color:var(--dim);line-height:1.7}
  .gloss b{color:var(--txt)}
  footer{color:var(--faint);font:400 .72rem var(--mono);margin:22px 0 8px}
  @media (max-width:560px){
    .lvlgrid{grid-template-columns:1fr 1fr}
    .stats{grid-template-columns:1fr 1fr}
    .tabs{margin-left:0;width:100%}
  }
"""

CAUSE_COLORS = {"Enemy": "#ef4444", "Pit": "#4a9eff", "OOB": "#eab308", "Spike": "#a855f7",
                "Saw": "#a855f7", "Stall": "#eab308", "Timeout": "#9a9aa0", "?": "#606066"}


def _pill(row: dict) -> str:
    if row["solved_by"] == row["seeds"]:
        return '<span class="pill pill-ok">solved</span>'
    if row["solved_by"]:
        return f'<span class="pill pill-warn">{row["solved_by"]}/{row["seeds"]} seeds</span>'
    return '<span class="pill pill-bad">unsolved</span>'


def _wr_cls(row: dict) -> str:
    if row["solved_by"] == 0:
        return "wr-bad"
    return "wr-good" if row["win_rate_mean"] >= 0.5 else "wr-mid"


def _wr_var(row: dict) -> str:
    return {"wr-good": "var(--green)", "wr-mid": "var(--yellow)", "wr-bad": "var(--red)"}[_wr_cls(row)]


def _cause_name(name: str) -> str:
    return "unknown" if name in ("?", "", "-") else name


def _ci(row: dict) -> float:
    """Two-seed t-intervals explode past 100% — cap the display, the JSON keeps the raw value."""
    return min(row.get("win_rate_ci") or 0.0, 1.0)


def _heat(hist: list[int]) -> str:
    mx = max(hist) or 1
    cells = "".join(
        f'<i style="background:rgba(239,68,68,{0.15 + 0.85 * n / mx:.2f})" title="{n} deaths"></i>'
        if n else "<i></i>" for n in hist)
    return (f'<div class="heat">{cells}</div>'
            f'<div class="heatlbl"><span>start</span><span>goal</span></div>')


def _causebar(causes: dict[str, int]) -> str:
    total = sum(causes.values())
    if not total:
        return '<div class="causekey">no deaths recorded</div>'
    parts, key = [], []
    for name, n in sorted(causes.items(), key=lambda kv: -kv[1]):
        color = CAUSE_COLORS.get(name, "#606066")
        parts.append(f'<i style="width:{100 * n / total:.1f}%;background:{color}" '
                     f'title="{_cause_name(name)}: {n}"></i>')
        key.append(f'<span style="color:{color}">■</span> {_cause_name(name)} {100 * n / total:.0f}%')
    return (f'<div class="causebar">{"".join(parts)}</div>'
            f'<div class="causekey">{" · ".join(key)}</div>')


def _stat(label: str, value: str, sub: str = "") -> str:
    return (f'<div class="stat"><div class="lbl">{label}</div><div class="val">{value}</div>'
            + (f'<div class="sub">{sub}</div>' if sub else "") + "</div>")


def _card(row: dict, cells: list[dict]) -> str:
    """One expandable level card: header always visible, detail revealed in place."""
    wr = row["win_rate_mean"]
    fw = (f"first win gen {row['first_win_mean']}" if row["first_win_mean"] is not None
          else f"best progress {row.get('progress_at_death_mean') or 0:.0%}")
    color = _wr_var(row)
    mct = (f"{row['mean_completion_time']}s" if row.get("mean_completion_time") is not None else "—")
    pad = (f"{row['progress_at_death_mean']:.0%}" if row.get("progress_at_death_mean") is not None else "—")
    gap = row.get("novice_expert_gap_mean", 0.0)
    fw_full = (f"gen {row['first_win_mean']} ± {row['first_win_ci']}"
               if row["first_win_mean"] is not None else "never")
    stats = "".join([
        _stat("Win rate", f"{wr:.0%} ± {_ci(row):.0%}", "measured 10 gens after first win"),
        _stat("First win", fw_full, f"{row['solved_by']}/{row['seeds']} seeds solved"),
        _stat("Completion", f"{row.get('completion_rate_mean', 0):.0%}", "wins / all episodes"),
        _stat("Mean win time", mct, "average of winning runs"),
        _stat("Progress at death", pad, "how far failers get"),
        _stat("Deaths per gen", f"{row.get('deaths_per_run_mean', 0)}", "of 10 attempts"),
        _stat("Death spread", f"{row.get('death_cluster_entropy_mean', 0):.2f}",
              "0 = one hotspot · 1 = everywhere"),
        _stat("Dominant cause", _cause_name(row["dominant_cause"]),
              f"{row['dominant_cause_frac']:.0%} of deaths"),
        _stat("Coin rate", f"{row.get('coin_collection_rate_mean', 1):.0%}", "collected / available"),
        _stat("Skill gap", f"{gap:+.0%}", "late-gen wins − early-gen wins"),
        _stat("Stuck rate", f"{row['stuck_frac_mean']:.0%}", "episodes ending in a stall"),
    ])
    curves = json.dumps([c.get("curve", []) for c in cells])
    return f"""
    <div class="lvlcard">
      <button class="cardhead" type="button">
        <div class="lvlname">{row['level']} {_pill(row)}</div>
        <div class="bignum" style="color:{color}">{wr:.0%}</div>
        <div class="lvlsub">win rate · {fw}</div>
        <div class="mini"><i style="width:{max(2, wr * 100):.0f}%;background:{color}"></i></div>
      </button>
      <div class="detail">
        <div class="stats">{stats}</div>
        <div class="viz"><div class="vt">Where agents die (start → goal)</div>
          {_heat(row.get('death_hist') or [0] * 10)}
          <div class="caption">Each bin is 10% of the level. One bright bin = a single learnable
          chokepoint; many lit bins = difficulty spread across the level.</div></div>
        <div class="viz"><div class="vt">What kills them</div>
          {_causebar(row.get('causes') or {})}</div>
        <div class="viz"><div class="vt">Learning curves — fitness per generation, one color per seed</div>
          <canvas class="curve" width="1100" height="260" data-curves='{curves}'></canvas>
          <div class="caption">Solid line: the best genome each generation. Faint line: population
          average. A jump above the dashed line means winning runs (win bonus). Flat = stuck.</div></div>
      </div>
    </div>"""


def _metric_table(rows: list[dict]) -> str:
    trs = []
    for r in rows:
        wr = f"{r['win_rate_mean']:.0%} ±{_ci(r):.0%}"
        fw = (f"{r['first_win_mean']}±{r['first_win_ci']}" if r["first_win_mean"] is not None else "—")
        solved_cls = ("wr-good" if r["solved_by"] == r["seeds"] else
                      ("wr-mid" if r["solved_by"] else "wr-bad"))
        mct = f"{r['mean_completion_time']}s" if r.get("mean_completion_time") is not None else "—"
        pad = f"{r['progress_at_death_mean']:.0%}" if r.get("progress_at_death_mean") is not None else "—"
        gap = r.get("novice_expert_gap_mean", 0.0)
        trs.append(
            f"<tr><td>{r['level']}</td>"
            f"<td class='{solved_cls}'>{r['solved_by']}/{r['seeds']}</td>"
            f"<td>{fw}</td><td>{wr}</td>"
            f"<td>{r.get('completion_rate_mean', 0):.0%}</td><td>{mct}</td>"
            f"<td>{pad}</td><td>{r.get('deaths_per_run_mean', 0)}</td>"
            f"<td>{r.get('death_cluster_entropy_mean', 0):.2f}</td>"
            f"<td>{r.get('coin_collection_rate_mean', 1):.0%}</td>"
            f"<td>{gap:+.0%}</td>"
            f"<td>{_cause_name(r['dominant_cause'])} ({r['dominant_cause_frac']:.0%})</td>"
            f"<td>{r['stuck_frac_mean']:.0%}</td></tr>")
    return f"""
    <div class="ovwrap"><table>
      <thead><tr><th>Level</th><th>Solved</th><th>First win</th><th>Win rate ±CI</th>
        <th>Completion</th><th>Mean time</th><th>Progress@death</th><th>Deaths/gen</th>
        <th>Death spread</th><th>Coin rate</th><th>Skill gap</th>
        <th>Dominant cause</th><th>Stuck</th></tr></thead>
      <tbody>{''.join(trs)}</tbody>
    </table></div>"""


def _overview(personas: dict[str, dict]) -> str:
    """One row per level, one win-rate column per persona — the tier comparison."""
    order: list[str] = []
    for data in personas.values():
        for r in data["levels"]:
            if r["level"] not in order:
                order.append(r["level"])
    pnames = list(personas)
    ths = "".join(f"<th>{p}<br><span style='letter-spacing:0;font-weight:400'>win rate · first win</span></th>"
                  for p in pnames)
    trs = []
    for lvl in order:
        tds = [f"<td>{lvl}</td>"]
        best_pill = '<span class="pill pill-bad">unsolved</span>'
        for p in pnames:
            row = next((r for r in personas[p]["levels"] if r["level"] == lvl), None)
            if row is None:
                tds.append("<td class='wr-na'>—</td>")
                continue
            fw = (f"gen {row['first_win_mean']}" if row["first_win_mean"] is not None else "never")
            tds.append(f"<td><span class='{_wr_cls(row)}'>{row['win_rate_mean']:.0%}</span>"
                       f" <span style='color:var(--faint)'>· {fw}</span></td>")
            if row["solved_by"] == row["seeds"]:
                best_pill = '<span class="pill pill-ok">solved</span>'
            elif row["solved_by"] and "pill-ok" not in best_pill:
                best_pill = '<span class="pill pill-warn">partial</span>'
        tds.insert(1, f"<td>{best_pill}</td>")
        trs.append("<tr>" + "".join(tds) + "</tr>")
    return f"""
    <div class="ovwrap"><table>
      <thead><tr><th>Level</th><th>Status</th>{ths}</tr></thead>
      <tbody>{''.join(trs)}</tbody>
    </table></div>
    <div class="ovnote">Status = best across personas. Select a persona tab for the full
    per-level breakdown — cards expand in place with death maps, causes, and learning curves.</div>"""


def _game_section(game: str, personas: dict[str, dict], idx: int) -> str:
    metas = next(iter(personas.values()))
    tabs = [f'<button class="tab t-overview active" data-view="v{idx}_ov" type="button">Overview</button>']
    views = [f'<div class="view active" id="v{idx}_ov">{_overview(personas)}</div>']
    for pi, (pname, data) in enumerate(personas.items()):
        vid = f"v{idx}_{pi}"
        tabs.append(f'<button class="tab" data-view="{vid}" type="button">{pname}</button>')
        cell_map = data.get("cells", {})
        cards = "".join(_card(r, cell_map.get(r["level"], [])) for r in data["levels"])
        views.append(f"""
        <div class="view" id="{vid}">
          <div class="lvlgrid">{cards}</div>
          <div class="tbltitle">Full metric table — {pname}</div>
          {_metric_table(data["levels"])}
        </div>""")
    return f"""
  <section class="game">
    <div class="gamehead">
      <span class="gametag">{game}</span>
      <span class="gamemeta">seeds {metas['seeds']} · budget {metas['gens_budget']} gens/probe</span>
      <div class="tabs">{''.join(tabs)}</div>
    </div>
    {''.join(views)}
  </section>"""


JS = """
// persona tabs — one active view per game section
for (const tab of document.querySelectorAll('.tab')){
  tab.addEventListener('click', () => {
    const section = tab.closest('section.game');
    section.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    section.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    tab.classList.add('active');
    section.querySelector('#' + tab.dataset.view).classList.add('active');
  });
}

// cards expand IN PLACE (span the full grid row) — no page jump
for (const head of document.querySelectorAll('.cardhead')){
  head.addEventListener('click', () => {
    const card = head.closest('.lvlcard');
    const grid = card.closest('.lvlgrid');
    const wasOpen = card.classList.contains('open');
    grid.querySelectorAll('.lvlcard.open').forEach(c => c.classList.remove('open'));
    if (!wasOpen) card.classList.add('open');
  });
}

// learning curves with real axes
for (const cv of document.querySelectorAll('canvas.curve')){
  const seeds = JSON.parse(cv.dataset.curves || '[]').filter(c => c.length > 1);
  const ctx = cv.getContext('2d');
  if (!seeds.length) continue;
  const W = cv.width, H = cv.height, L = 62, R = 12, T = 14, B = 30;
  const maxY = Math.max(...seeds.flat().map(p => p[0]), 1) * 1.05;
  const maxX = Math.max(...seeds.map(c => c.length));
  const px = i => L + i * (W - L - R) / (maxX - 1);
  const py = v => H - B - (v / maxY) * (H - T - B);
  ctx.font = '400 12px "IBM Plex Mono", monospace';
  ctx.strokeStyle = '#222226'; ctx.fillStyle = '#77777d'; ctx.lineWidth = 1;
  for (let q = 0; q <= 4; q++){
    const v = maxY * q / 4, y = py(v);
    ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(W - R, y); ctx.stroke();
    ctx.fillText(Math.round(v).toLocaleString(), 4, y + 4);
  }
  if (maxY > 5000){
    const y = py(5000);
    ctx.setLineDash([5, 4]); ctx.strokeStyle = '#22c55e55';
    ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(W - R, y); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#22c55e99'; ctx.fillText('win bonus', W - R - 78, y - 5);
  }
  ctx.fillStyle = '#77777d';
  ctx.fillText('gen 1', L, H - 9);
  const el = 'gen ' + maxX;
  ctx.fillText(el, W - R - ctx.measureText(el).width, H - 9);
  const colors = ['#ef4444', '#4a9eff', '#eab308', '#a855f7'];
  seeds.forEach((c, si) => {
    for (const [j, w] of [[1, 1], [0, 2]]){
      ctx.strokeStyle = j ? colors[si % 4] + '44' : colors[si % 4];
      ctx.lineWidth = w; ctx.beginPath();
      c.forEach((p, i) => {
        const x = px(i), y = py(p[j]);
        i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      });
      ctx.stroke();
    }
  });
}
"""

GLOSSARY = """
  <div class="gloss">
    <b>Reading this report</b> —
    Each game opens on the <b>Overview</b>: one row per level, one column per persona, so skill
    tiers sit side by side. The number is the <b>win rate</b>: episodes won in the 10 generations
    after the population first solved the level
    (<span style="color:#22c55e">green ≥ 50%</span>,
    <span style="color:#eab308">yellow below</span>,
    <span style="color:#ef4444">red = never solved</span>).
    <b>First win</b>: generations evolution needed — the main difficulty signal.
    <b>Death spread</b>: 0 = one hotspot (learnable), 1 = deaths everywhere (scattered).
    <b>Skill gap</b>: wins in the last third of generations minus the first third.
    An <b>unsolved</b> level with identical best progress across seeds usually means a geometry
    problem (unreachable goal), not raw difficulty.
  </div>"""


def build(balance_dir: str) -> str:
    games: dict[str, dict[str, dict]] = {}
    for f in sorted(glob.glob(os.path.join(balance_dir, "report_*.json"))):
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        persona = data.get("persona") or "experienced"
        games.setdefault(data["game"], {})[persona] = data
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    sections = [_game_section(g, p, i) for i, (g, p) in enumerate(games.items())]
    body = "".join(sections) if sections else \
        "<p style='color:var(--dim)'>No balance data yet — run a Balance Report (menu 14) or Full Sweep (menu 15) first.</p>"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'><rect width='16' height='16' rx='3' fill='%23070708'/><path d='M3 12 L8 4 L13 12' stroke='%23ef4444' stroke-width='2' fill='none'/></svg>">
<title>PEAK Balance Command</title><style>{CSS}</style></head><body><main>
  <div class="cmdbar">
    <h1>PEAK <em>BALANCE COMMAND</em></h1>
    <span class="stamp">multi-seed neuroevolution probes · generated {stamp}</span>
  </div>
  {body}
  {GLOSSARY}
  <footer>PEAK ENGINE · code/neuro/report.py · data: runs/balance/report_*.json</footer>
</main><script>{JS}</script></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the balance web report")
    ap.add_argument("--dir", default=os.path.join("runs", "balance"))
    ap.add_argument("--open", action="store_true", help="open in the default browser")
    args = ap.parse_args()

    html = build(args.dir)
    os.makedirs(args.dir, exist_ok=True)
    out = os.path.join(args.dir, "report.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {out}")
    if args.open:
        webbrowser.open("file:///" + os.path.abspath(out).replace(os.sep, "/"))


if __name__ == "__main__":
    main()
