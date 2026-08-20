"""Balance web report: one self-contained HTML page from balance/report_*.json.

The successor to the old Dash stats dashboard — every balance metric from the
metrics table (completion, punishment severity, triangularity, skill expression)
per level, with death-location heatmaps, failure-mode bars, and fitness curves.

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
  :root{--bg:#0b0e1a;--panel:#131728;--panel2:#0f1322;--line:#242b47;--line2:#1a2038;
    --txt:#e8ecf8;--dim:#7d86a8;--faint:#4a5273;--gold:#ffb02e;--sky:#6ec6ff;
    --run:#43d17c;--stuck:#ff9d3b;--dead:#ff5470;--won:#ffd23e;
    --mono:"Cascadia Mono",Consolas,monospace;--ui:"Segoe UI",system-ui,sans-serif}
  *{box-sizing:border-box;margin:0}
  body{background:var(--bg);color:var(--txt);font:15px/1.55 var(--ui);padding:24px}
  main{max-width:1240px;margin:0 auto}
  h1{font:700 26px var(--ui)} h1 em{font-style:normal;color:var(--gold)}
  .eyebrow{font:700 11px var(--mono);letter-spacing:.2em;text-transform:uppercase;color:var(--dim)}
  .sub{color:var(--dim);margin:4px 0 20px}
  section.game{background:var(--panel);border:1px solid var(--line);padding:18px 20px;margin:18px 0;
    clip-path:polygon(0 8px,8px 0,calc(100% - 8px) 0,100% 8px,100% calc(100% - 8px),
    calc(100% - 8px) 100%,8px 100%,0 calc(100% - 8px))}
  h2{font:700 19px var(--ui);margin-bottom:4px} h2 b{color:var(--gold)}
  .meta{font:500 12.5px var(--mono);color:var(--faint);margin-bottom:14px}
  .tablewrap{overflow-x:auto;margin:10px 0 18px}
  table{border-collapse:collapse;font:500 13px var(--mono);font-variant-numeric:tabular-nums;width:100%}
  th{font:600 10.5px var(--ui);letter-spacing:.1em;text-transform:uppercase;color:var(--dim);
     text-align:right;padding:7px 10px;border-bottom:2px solid var(--line);white-space:nowrap}
  td{text-align:right;padding:6px 10px;border-bottom:1px solid var(--line2);white-space:nowrap}
  th:first-child,td:first-child{text-align:left}
  td:first-child{color:var(--sky);font-weight:700}
  .good{color:var(--run)} .bad{color:var(--dead)} .warn{color:var(--stuck)} .gold{color:var(--won)}
  .lvlgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:14px}
  .lvl{background:var(--panel2);border:1px solid var(--line2);padding:12px 14px}
  .lvl h3{font:700 14px var(--mono);color:var(--sky);margin-bottom:8px;display:flex;justify-content:space-between}
  .lvl h3 span{color:var(--dim);font-weight:500}
  .heat{display:grid;grid-template-columns:repeat(10,1fr);gap:2px;margin:8px 0 3px}
  .heat i{height:16px;background:var(--line2);position:relative}
  .heatlbl{display:flex;justify-content:space-between;font:500 10px var(--mono);color:var(--faint)}
  .causebar{display:flex;height:14px;margin:10px 0 3px;gap:1px}
  .causebar i{display:block}
  .causekey{font:500 11px var(--mono);color:var(--dim)}
  canvas.curve{width:100%;height:110px;background:var(--bg);border:1px solid var(--line2);margin-top:10px}
  footer{color:var(--faint);font:500 12px var(--mono);margin:22px 0 8px}
  .gloss{background:var(--panel);border:1px solid var(--line);padding:14px 18px;margin-top:18px;
    font-size:13.5px;color:var(--dim)}
  .gloss b{color:var(--txt)}
"""

CAUSE_COLORS = {"Enemy": "#ff5470", "Pit": "#6ec6ff", "OOB": "#ff9d3b", "Spike": "#c95cff",
                "Saw": "#c95cff", "Stall": "#ffd23e", "Timeout": "#7d86a8", "?": "#4a5273"}


def _heat(hist: list[int]) -> str:
    mx = max(hist) or 1
    cells = "".join(
        f'<i style="background:rgba(255,84,112,{0.12 + 0.88 * n / mx:.2f})" title="{n} deaths"></i>'
        if n else "<i></i>" for n in hist)
    return (f'<div class="heat">{cells}</div>'
            f'<div class="heatlbl"><span>start</span><span>death locations</span><span>goal</span></div>')


def _causebar(causes: dict[str, int]) -> str:
    total = sum(causes.values()) or 1
    parts, key = [], []
    for name, n in sorted(causes.items(), key=lambda kv: -kv[1]):
        color = CAUSE_COLORS.get(name, "#4a5273")
        parts.append(f'<i style="width:{100 * n / total:.1f}%;background:{color}" title="{name}: {n}"></i>')
        key.append(f'<span style="color:{color}">■</span> {name} {100 * n / total:.0f}%')
    return (f'<div class="causebar">{"".join(parts)}</div>'
            f'<div class="causekey">{" · ".join(key)}</div>')


def _level_card(row: dict, cells: list[dict]) -> str:
    fw = (f"first win gen {row['first_win_mean']}±{row['first_win_ci']}"
          if row["first_win_mean"] is not None else '<span class="bad">never solved</span>')
    curves = json.dumps([c.get("curve", []) for c in cells])
    cid = f"c_{abs(hash(row['level'])) % 10 ** 8}"
    return f"""
    <div class="lvl">
      <h3>{row['level']} <span>{fw}</span></h3>
      {_heat(row.get('death_hist') or [0] * 10)}
      {_causebar(row.get('causes') or {})}
      <canvas class="curve" id="{cid}" width="700" height="220" data-curves='{curves}'></canvas>
    </div>"""


def _game_section(data: dict) -> str:
    rows = sorted(data["levels"], key=lambda r: (r["solved_by"], -(r["first_win_mean"] or 9999)))
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
            f"<td class='{'good' if gap > 0 else 'dim'}'>{gap:+.0%}</td>"
            f"<td>{r['dominant_cause']} ({r['dominant_cause_frac']:.0%})</td>"
            f"<td>{r['stuck_frac_mean']:.0%}</td></tr>")
    cell_map = {lvl: cs for lvl, cs in data.get("cells", {}).items()}
    cards = "".join(_level_card(r, cell_map.get(r["level"], [])) for r in rows)
    return f"""
  <section class="game">
    <h2>Game: <b>{data['game']}</b></h2>
    <div class="meta">seeds {data['seeds']} · budget {data['gens_budget']} gens/probe ·
      probe wall-clock {data.get('elapsed_s', '?')}s · levels ranked hardest first</div>
    <div class="tablewrap"><table>
      <thead><tr><th>Level</th><th>Solved</th><th>First win</th><th>Win rate ±CI</th>
        <th>Completion</th><th>Mean time</th><th>Progress@death</th><th>Deaths/gen</th>
        <th>Death entropy</th><th>Coin rate</th><th>Skill gap</th>
        <th>Dominant cause</th><th>Stuck</th></tr></thead>
      <tbody>{''.join(trs)}</tbody>
    </table></div>
    <div class="lvlgrid">{cards}</div>
  </section>"""


JS = """
for (const cv of document.querySelectorAll("canvas.curve")){
  const seeds = JSON.parse(cv.dataset.curves || "[]").filter(c => c.length > 1);
  const ctx = cv.getContext("2d");
  if (!seeds.length) continue;
  const W = cv.width, H = cv.height, L = 6, B = 6;
  const maxY = Math.max(...seeds.flat().map(p => p[0]), 1);
  const maxX = Math.max(...seeds.map(c => c.length));
  const colors = ["#ffb02e", "#6ec6ff", "#43d17c", "#c95cff"];
  seeds.forEach((c, si) => {
    for (const [j, w] of [[1, 1], [0, 2]]){
      ctx.strokeStyle = j ? colors[si % 4] + "66" : colors[si % 4];
      ctx.lineWidth = w; ctx.beginPath();
      c.forEach((p, i) => {
        const x = L + i * (W - 2 * L) / (maxX - 1), y = H - B - (p[j] / maxY) * (H - 2 * B);
        i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      });
      ctx.stroke();
    }
  });
}
"""

GLOSSARY = """
  <div class="gloss">
    <b>Metric glossary</b> (from the balance-metrics table) —
    <b>Completion</b>: wins / all episodes across the probe.
    <b>Mean time</b>: average completion time of winning episodes (±σ in the JSON).
    <b>Progress@death</b>: how far along the level agents die, 0 = start, 100% = goal.
    <b>Deaths/gen</b>: deaths per generation of 10 attempts.
    <b>Death entropy</b>: 0 = every death at one hotspot, 1 = deaths spread evenly.
    <b>Coin rate</b>: collected coins / level coins ("bandage collection"; 100% when a level has none).
    <b>Skill gap</b>: win rate of the evolved population minus the novice population
    (last third of generations vs first third).
    <b>Heatmap</b>: death locations across the level, start → goal.
    <b>Curves</b>: per-seed fitness (bright = generation best, faint = average).
    Not yet mapped from the table: path/strategy diversity, safe-vs-fast ratio,
    wall-jump utilization — they need trajectory logging (future work).
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
  <span class="eyebrow">PEAK Engine · level balance</span>
  <h1>Balance <em>Report</em></h1>
  <p class="sub">Multi-seed neuroevolution probes per level — generated {stamp}</p>
  {body}
  {GLOSSARY}
  <footer>PEAK ENGINE · generated by code/neuro/report.py · data: balance/report_*.json</footer>
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
