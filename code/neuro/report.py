"""Balance web report: one self-contained HTML page from balance/report_*.json.

Design follows the PEAK stats dashboard (dark, IBM Plex, card-based) and the
overview-first / details-on-demand pattern: a clean per-level card grid up top,
click a card to open its full detail panel (metrics, death map, causes, curves).

Run:  python -m code.neuro.report [--dir balance] [--open]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import webbrowser
from datetime import datetime

CSS = """
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
  :root{--bg:#1a1a1a;--card:#232323;--card2:#1e1e1e;--line:#2e2e2e;--line2:#333333;
    --txt:#e0e0e0;--dim:#9a9a9a;--faint:#666666;--accent:#4a9eff;
    --ok:#22c55e;--warn:#eab308;--bad:#ef4444;
    --mono:'IBM Plex Mono',Consolas,monospace;--ui:'IBM Plex Sans','Segoe UI',sans-serif}
  *{box-sizing:border-box;margin:0}
  body{background:var(--bg);color:var(--txt);font:15px/1.55 var(--ui);padding:28px 20px}
  main{max-width:1100px;margin:0 auto}
  .header{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:6px}
  h1{font:600 1.15rem/1.3 var(--ui);color:#e5e5e5}
  .header-meta{font:400 .78rem var(--mono);color:var(--faint)}
  .sep{border-top:1px solid var(--line);margin:18px 0}
  .section-label{font-size:.78rem;color:var(--faint);margin:18px 0 10px}
  h2{font:600 .95rem var(--ui);color:#e5e5e5} h2 b{color:var(--accent);font-weight:600}
  .gamemeta{font:400 .75rem var(--mono);color:var(--faint);margin:2px 0 14px}

  /* overview cards — one primary number each, click to inspect */
  .lvlgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:12px}
  .lvlcard{background:var(--card);border:1px solid var(--line);border-radius:6px;
    padding:12px 16px;text-align:left;color:var(--txt);cursor:pointer;font:inherit;
    transition:border-color .15s,background .15s}
  .lvlcard:hover{background:#282828;border-color:#3a3a3a}
  .lvlcard.open{border-color:var(--accent)}
  .lvlname{font:600 .82rem var(--mono);color:#cfcfcf;margin-bottom:6px;display:flex;
    justify-content:space-between;align-items:center;gap:6px}
  .pill{font:600 .62rem var(--mono);border-radius:3px;padding:2px 7px;white-space:nowrap}
  .pill-ok{color:var(--ok);background:rgba(34,197,94,.1);border:1px solid rgba(34,197,94,.35)}
  .pill-warn{color:var(--warn);background:rgba(234,179,8,.1);border:1px solid rgba(234,179,8,.35)}
  .pill-bad{color:var(--bad);background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.35)}
  .bignum{font:600 1.55rem var(--mono)}
  .lvlsub{font:400 .7rem var(--mono);color:var(--faint);margin-top:2px}
  .mini{height:3px;border-radius:2px;background:var(--line);margin-top:8px;overflow:hidden}
  .mini i{display:block;height:100%}

  /* detail panel */
  .detail{display:none;background:var(--card);border:1px solid var(--line2);border-radius:6px;
    padding:18px 20px;margin-top:14px}
  .detail.show{display:block}
  .detail h3{font:600 .95rem var(--mono);color:#e5e5e5;margin-bottom:12px;display:flex;
    justify-content:space-between;align-items:baseline}
  .detail h3 span{font:400 .72rem var(--mono);color:var(--faint)}
  .stats{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px;margin-bottom:16px}
  .stat{background:var(--card2);border:1px solid var(--line);border-radius:6px;padding:10px 12px}
  .stat .lbl{font:400 .62rem var(--ui);color:var(--faint);text-transform:uppercase;letter-spacing:.08em}
  .stat .val{font:600 1.05rem var(--mono);margin-top:2px}
  .stat .sub{font:400 .66rem var(--mono);color:var(--faint);margin-top:1px}
  .viz{background:var(--card2);border:1px solid var(--line);border-radius:6px;padding:12px 14px;margin-top:10px}
  .viz .vt{font:600 .72rem var(--ui);color:var(--dim);margin-bottom:8px}
  .caption{font:400 .7rem var(--ui);color:var(--faint);margin-top:6px;line-height:1.5}
  .heat{display:grid;grid-template-columns:repeat(10,1fr);gap:2px}
  .heat i{height:18px;border-radius:2px;background:var(--line)}
  .heatlbl{display:flex;justify-content:space-between;font:400 .64rem var(--mono);color:var(--faint);margin-top:3px}
  .causebar{display:flex;height:14px;gap:1px;border-radius:3px;overflow:hidden}
  .causekey{font:400 .7rem var(--mono);color:var(--dim);margin-top:6px}
  canvas.curve{width:100%;height:170px;display:block}

  details.tablefold{margin-top:16px}
  details.tablefold summary{cursor:pointer;font:600 .78rem var(--ui);color:var(--dim);
    padding:8px 0;list-style:none}
  details.tablefold summary::before{content:'▸ '} details[open].tablefold summary::before{content:'▾ '}
  .tablewrap{overflow-x:auto;border:1px solid var(--line);border-radius:6px}
  table{border-collapse:collapse;font:400 .74rem var(--mono);font-variant-numeric:tabular-nums;width:100%}
  th{font:600 .62rem var(--ui);letter-spacing:.08em;text-transform:uppercase;color:var(--faint);
     text-align:right;padding:8px 10px;border-bottom:1px solid var(--line2);white-space:nowrap;
     background:var(--card2);position:sticky;top:0}
  td{text-align:right;padding:6px 10px;border-bottom:1px solid #262626;white-space:nowrap;color:var(--dim)}
  th:first-child,td:first-child{text-align:left}
  td:first-child{color:var(--txt);font-weight:600}
  .good{color:var(--ok)} .bad{color:var(--bad)} .warn{color:var(--warn)}
  .gloss{background:var(--card);border:1px solid var(--line);border-radius:6px;
    padding:14px 18px;margin-top:20px;font-size:.78rem;color:var(--dim);line-height:1.7}
  .gloss b{color:var(--txt)}
  footer{color:var(--faint);font:400 .7rem var(--mono);margin:22px 0 8px}
"""

CAUSE_COLORS = {"Enemy": "#ef4444", "Pit": "#4a9eff", "OOB": "#f59e0b", "Spike": "#a855f7",
                "Saw": "#a855f7", "Stall": "#eab308", "Timeout": "#9a9a9a", "?": "#666666"}


def _pill(row: dict) -> str:
    if row["solved_by"] == row["seeds"]:
        return '<span class="pill pill-ok">solved</span>'
    if row["solved_by"]:
        return f'<span class="pill pill-warn">{row["solved_by"]}/{row["seeds"]} seeds</span>'
    return '<span class="pill pill-bad">unsolved</span>'


def _wr_color(row: dict) -> str:
    if row["solved_by"] == 0:
        return "var(--bad)"
    return "var(--ok)" if row["win_rate_mean"] >= 0.5 else "var(--warn)"


def _cause_name(name: str) -> str:
    return "unknown" if name in ("?", "", "-") else name


def _overview_card(row: dict, did: str) -> str:
    wr = row["win_rate_mean"]
    fw = (f"first win gen {row['first_win_mean']}" if row["first_win_mean"] is not None
          else f"best progress {row.get('progress_at_death_mean') or 0:.0%}")
    color = _wr_color(row)
    return f"""
    <button class="lvlcard" data-target="{did}">
      <div class="lvlname">{row['level']} {_pill(row)}</div>
      <div class="bignum" style="color:{color}">{wr:.0%}</div>
      <div class="lvlsub">win rate · {fw}</div>
      <div class="mini"><i style="width:{max(2, wr * 100):.0f}%;background:{color}"></i></div>
    </button>"""


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
        color = CAUSE_COLORS.get(name, "#666666")
        parts.append(f'<i style="width:{100 * n / total:.1f}%;background:{color}" '
                     f'title="{_cause_name(name)}: {n}"></i>')
        key.append(f'<span style="color:{color}">■</span> {_cause_name(name)} {100 * n / total:.0f}%')
    return (f'<div class="causebar">{"".join(parts)}</div>'
            f'<div class="causekey">{" · ".join(key)}</div>')


def _stat(label: str, value: str, sub: str = "") -> str:
    return (f'<div class="stat"><div class="lbl">{label}</div><div class="val">{value}</div>'
            + (f'<div class="sub">{sub}</div>' if sub else "") + "</div>")


def _detail(row: dict, cells: list[dict], did: str) -> str:
    fw = (f"gen {row['first_win_mean']} ± {row['first_win_ci']}"
          if row["first_win_mean"] is not None else "never")
    mct = (f"{row['mean_completion_time']}s" if row.get("mean_completion_time") is not None else "—")
    pad = (f"{row['progress_at_death_mean']:.0%}" if row.get("progress_at_death_mean") is not None else "—")
    gap = row.get("novice_expert_gap_mean", 0.0)
    dom = _cause_name(row["dominant_cause"])
    stats = "".join([
        _stat("Win rate", f"{row['win_rate_mean']:.0%} ± {row['win_rate_ci']:.0%}",
              "measured 10 gens after first win"),
        _stat("First win", fw, f"{row['solved_by']}/{row['seeds']} seeds solved"),
        _stat("Completion", f"{row.get('completion_rate_mean', 0):.0%}", "wins / all episodes"),
        _stat("Mean win time", mct, "average of winning runs"),
        _stat("Progress at death", pad, "how far failers get"),
        _stat("Deaths per gen", f"{row.get('deaths_per_run_mean', 0)}", "of 10 attempts"),
        _stat("Death spread", f"{row.get('death_cluster_entropy_mean', 0):.2f}",
              "0 = one hotspot · 1 = everywhere"),
        _stat("Dominant cause", dom, f"{row['dominant_cause_frac']:.0%} of deaths"),
        _stat("Coin rate", f"{row.get('coin_collection_rate_mean', 1):.0%}", "collected / available"),
        _stat("Skill gap", f"{gap:+.0%}", "late-gen wins − early-gen wins"),
        _stat("Stuck rate", f"{row['stuck_frac_mean']:.0%}", "episodes ending in a stall"),
    ])
    curves = json.dumps([c.get("curve", []) for c in cells])
    return f"""
    <div class="detail" id="{did}">
      <h3>{row['level']} <span>click the card again to close</span></h3>
      <div class="stats">{stats}</div>
      <div class="viz"><div class="vt">Where agents die (start → goal)</div>
        {_heat(row.get('death_hist') or [0] * 10)}
        <div class="caption">Each bin is 10% of the level. One bright bin = a single learnable
        chokepoint; many lit bins = difficulty spread across the level.</div></div>
      <div class="viz"><div class="vt">What kills them</div>
        {_causebar(row.get('causes') or {})}</div>
      <div class="viz"><div class="vt">Learning curves — fitness per generation, one color per seed</div>
        <canvas class="curve" width="1000" height="240" data-curves='{curves}'></canvas>
        <div class="caption">Solid line: the best genome each generation. Faint line: population
        average. A jump above the dashed line means winning runs (win bonus). Flat = stuck.</div></div>
    </div>"""


def _game_section(data: dict) -> str:
    rows = data["levels"]  # config order — the order the sweep probed them in
    cell_map = data.get("cells", {})
    cards, details = [], []
    for i, r in enumerate(rows):
        did = f"d_{data['game']}_{i}"
        cards.append(_overview_card(r, did))
        details.append(_detail(r, cell_map.get(r["level"], []), did))

    trs = []
    for r in rows:
        wr = f"{r['win_rate_mean']:.0%} ±{r['win_rate_ci']:.0%}"
        fw = (f"{r['first_win_mean']}±{r['first_win_ci']}" if r["first_win_mean"] is not None else "—")
        solved_cls = "good" if r["solved_by"] == r["seeds"] else ("warn" if r["solved_by"] else "bad")
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
  <div class="sep"></div>
  <h2>Game: <b>{data['game']}</b></h2>
  <div class="gamemeta">seeds {data['seeds']} · budget {data['gens_budget']} gens/probe ·
    persona {data.get('persona', 'experienced')} · levels in config order</div>
  <div class="section-label">levels — click a card to inspect</div>
  <div class="lvlgrid">{''.join(cards)}</div>
  {''.join(details)}
  <details class="tablefold"><summary>Full metric table</summary>
    <div class="tablewrap"><table>
      <thead><tr><th>Level</th><th>Solved</th><th>First win</th><th>Win rate ±CI</th>
        <th>Completion</th><th>Mean time</th><th>Progress@death</th><th>Deaths/gen</th>
        <th>Death spread</th><th>Coin rate</th><th>Skill gap</th>
        <th>Dominant cause</th><th>Stuck</th></tr></thead>
      <tbody>{''.join(trs)}</tbody>
    </table></div>
  </details>"""


JS = """
// overview-first: one detail panel open at a time
for (const card of document.querySelectorAll('.lvlcard')){
  card.addEventListener('click', () => {
    const target = document.getElementById(card.dataset.target);
    const wasOpen = target.classList.contains('show');
    document.querySelectorAll('.detail.show').forEach(d => d.classList.remove('show'));
    document.querySelectorAll('.lvlcard.open').forEach(c => c.classList.remove('open'));
    if (!wasOpen){
      target.classList.add('show');
      card.classList.add('open');
      target.scrollIntoView({behavior: 'smooth', block: 'nearest'});
    }
  });
}

// learning curves with real axes
function drawCurves(){
  for (const cv of document.querySelectorAll('canvas.curve')){
    const seeds = JSON.parse(cv.dataset.curves || '[]').filter(c => c.length > 1);
    const ctx = cv.getContext('2d');
    ctx.clearRect(0, 0, cv.width, cv.height);
    if (!seeds.length) continue;
    const W = cv.width, H = cv.height, L = 56, R = 10, T = 12, B = 26;
    const maxY = Math.max(...seeds.flat().map(p => p[0]), 1) * 1.05;
    const maxX = Math.max(...seeds.map(c => c.length));
    const px = i => L + i * (W - L - R) / (maxX - 1);
    const py = v => H - B - (v / maxY) * (H - T - B);
    // grid + y labels
    ctx.font = '400 11px "IBM Plex Mono", monospace';
    ctx.strokeStyle = '#2c2c2c'; ctx.fillStyle = '#777'; ctx.lineWidth = 1;
    for (let q = 0; q <= 4; q++){
      const v = maxY * q / 4, y = py(v);
      ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(W - R, y); ctx.stroke();
      ctx.fillText(Math.round(v).toLocaleString(), 4, y + 4);
    }
    // win-bonus threshold marker (fitness > 5000 implies a winning run)
    if (maxY > 5000){
      const y = py(5000);
      ctx.setLineDash([5, 4]); ctx.strokeStyle = '#22c55e55';
      ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(W - R, y); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = '#22c55e99'; ctx.fillText('win bonus', W - R - 70, y - 5);
    }
    // x labels
    ctx.fillStyle = '#777';
    ctx.fillText('gen 1', L, H - 8);
    const el = 'gen ' + maxX;
    ctx.fillText(el, W - R - ctx.measureText(el).width, H - 8);
    const colors = ['#4a9eff', '#f59e0b', '#22c55e', '#a855f7'];
    seeds.forEach((c, si) => {
      for (const [j, w] of [[1, 1], [0, 2]]){  // avg faint first, best on top
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
}
drawCurves();
"""

GLOSSARY = """
  <div class="gloss">
    <b>Reading this report</b> —
    Each card is one level, in config order. The big number is the <b>win rate</b>: the share of
    episodes won in the 10 generations after the population first solved the level
    (<span style="color:#22c55e">green ≥ 50%</span>,
    <span style="color:#eab308">yellow below</span>,
    <span style="color:#ef4444">red = never solved</span>).
    <b>First win</b>: how many generations evolution needed — the main difficulty signal.
    <b>Death spread</b>: 0 = every death at one hotspot (learnable), 1 = deaths everywhere (scattered).
    <b>Skill gap</b>: wins in the last third of generations minus the first third — how much the
    population improved.
    <b>Coin rate</b>: collected / available (100% when the level has none).
    An <b>unsolved</b> level with identical best progress across seeds usually means a geometry
    problem (unreachable goal), not raw difficulty.
  </div>"""


def build(balance_dir: str) -> str:
    files = sorted(glob.glob(os.path.join(balance_dir, "report_*.json")))
    sections = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            sections.append(_game_section(json.load(fh)))
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    body = "".join(sections) if sections else \
        "<p style='color:var(--dim)'>No balance data yet — run a Balance Report (menu 14) first.</p>"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PEAK Balance Report</title><style>{CSS}</style></head><body><main>
  <div class="header">
    <div><h1>Balance dashboard — PEAK</h1>
    <div class="header-meta">multi-seed neuroevolution probes · generated {stamp}</div></div>
  </div>
  {body}
  {GLOSSARY}
  <footer>PEAK ENGINE · code/neuro/report.py · data: balance/report_*.json</footer>
</main><script>{JS}</script></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the balance web report")
    ap.add_argument("--dir", default="balance")
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
