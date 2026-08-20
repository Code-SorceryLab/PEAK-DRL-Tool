"""PEAK Balance Command: one self-contained HTML command center from runs/.

Merges the balance report and the stats dashboard: per-game sections with an
Overview (all personas side by side) and persona tabs whose level cards expand
in place — metrics with hover explanations, B1/B2/B3 verdicts, death maps,
cause bars, learning curves, and agent routes drawn over the level geometry.
Also lists every training run found under runs/. A separate instructions page
is generated alongside.

Run:  python -m code.neuro.report [--dir runs/balance] [--open]
"""
from __future__ import annotations

import argparse
import ast
import base64
import csv
import glob
import io
import json
import os
import webbrowser
from datetime import datetime

GAMES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "games")
LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "docs", "img", "PEAK_LOGO.png")
THRESHOLDS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stats", "MarioThresholds.yaml")
PROBES_ROOT = os.path.join("runs", "probes")
SOLID = set("#=%HS([/\\])<>?F")
HAZARD = set("^*O")

# hover blurbs for every metric (the little info bubbles)
TIPS = {
    "Win rate": "Share of episodes won in the 10 generations after the population's first win. "
                "Green means the population beats the level more often than not.",
    "First win": "How many generations evolution needed to solve the level once — "
                 "the primary difficulty signal. Lower = easier.",
    "Completion": "Wins divided by every episode of the whole probe, including early chaotic generations.",
    "Mean win time": "Average in-game time of the winning runs only.",
    "Progress at death": "How far through the level failing agents get on average (0% = start, 100% = goal).",
    "Deaths per gen": "Average deaths per generation of 10 attempts.",
    "Death spread": "Entropy of death locations: 0 = everyone dies at one learnable chokepoint, "
                    "1 = deaths scattered everywhere (feels random).",
    "Dominant cause": "The single failure mode that kills most agents: Enemy, Pit, Saw, Spike, Stall, OOB.",
    "Coin rate": "Coins collected / coins available (100% when the level has none).",
    "Skill gap": "Win rate in the last third of generations minus the first third — how much the population learned.",
    "Stuck rate": "Episodes ended by the stall detector (no forward progress for 5 seconds).",
    "Strategies": "Distinct route clusters among winning runs (needs 2+ wins with logged routes).",
    "Dominant path": "Share of winning runs that follow the most common route. Near 100% = one true path.",
    "B1": "Challenge calibration: completion rate and win time vs the target band in "
          "code/stats/MarioThresholds.yaml. Bands are designer-set, not human-calibrated yet.",
    "B2": "Punishment severity: deaths per run and death spread vs their target bands.",
    "B3": "Strategy diversity: route-cluster count and dominant-path share vs their target bands.",
}


# ── data loading ─────────────────────────────────────────────────────────────

def _load_thresholds() -> dict:
    try:
        import yaml
        with open(THRESHOLDS_PATH, encoding="utf-8") as f:
            return (yaml.safe_load(f) or {}).get("metrics", {})
    except Exception:
        return {}


def _level_file(game: str, level: str) -> str | None:
    import yaml
    try:
        if game == "meatboy":
            with open(os.path.join(GAMES_DIR, "meatboy_config.yaml"), encoding="utf-8") as f:
                lv = (yaml.safe_load(f) or {}).get("levels", [])
            i = int(level)
            return os.path.join(GAMES_DIR, "levels", lv[i]) if 0 <= i < len(lv) else None
        with open(os.path.join(GAMES_DIR, "game_config.yaml"), encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        section = {"mario": cfg, "megaman": cfg.get("megaman", {}), "sonic": cfg.get("sonic", {})}.get(game, {})
        for key in ("levels", "disabled_levels"):
            entry = (section.get(key) or {}).get(level)
            if isinstance(entry, dict) and "file" in entry:
                return os.path.join(GAMES_DIR, "levels", entry["file"])
    except Exception:
        pass
    return None


def _level_grid(game: str, level: str) -> list[str] | None:
    """Compact geometry: '#'=solid, '^'=hazard, 'G'=goal, ' '=air."""
    path = _level_file(game, level)
    if not path or not os.path.exists(path):
        return None
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f.read().splitlines():
            rows.append("".join(
                "#" if ch in SOLID else ("^" if ch in HAZARD else ("G" if ch == "G" else " "))
                for ch in line).rstrip())
    while rows and not rows[-1]:
        rows.pop()
    return rows or None


def _collect_routes(game: str, persona: str, level: str, max_routes: int = 22) -> list[dict]:
    """Route traces for one (game, persona, level) from the probe episode CSVs."""
    won, lost = [], []
    pattern = os.path.join(PROBES_ROOT, game, f"{level}_*".replace(" ", "_"), "episodes.csv")
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path, encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    if str(row.get("world")) != str(level) or row.get("persona") != persona:
                        continue
                    try:
                        pts = ast.literal_eval(row.get("route") or "[]")
                    except (ValueError, SyntaxError):
                        continue
                    if len(pts) < 2:
                        continue
                    step = max(1, len(pts) // 60)  # ≤60 points per embedded route
                    pts = [[round(x), round(y)] for x, y in pts[::step]]
                    (won if (row.get("cause_of_death") or "").lower() == "success" else lost).append(pts)
        except OSError:
            continue
    routes = [{"p": p, "w": 1} for p in won[:10]]
    routes += [{"p": p, "w": 0} for p in lost[-(max_routes - len(routes)):]]
    return routes


def _cluster_strategies(won_routes: list[list[list[float]]]) -> tuple[int, float] | None:
    """Amr's B3: 50-bin height profiles, greedy centroid clustering (threshold 96px)."""
    try:
        import numpy as np
    except ImportError:
        return None
    if len(won_routes) < 2:
        return None
    xs_all = [p[0] for r in won_routes for p in r]
    x_min, x_max = min(xs_all), max(xs_all)
    if x_max - x_min < 1:
        return None
    profiles = []
    for r in won_routes:
        xs = np.array([p[0] for p in r], float)
        ys = np.array([p[1] for p in r], float)
        edges = np.linspace(x_min, x_max, 51)
        prof = np.full(50, np.nan)
        for i in range(50):
            m = (xs >= edges[i]) & (xs < edges[i + 1])
            if m.any():
                prof[i] = ys[m].mean()
        valid = ~np.isnan(prof)
        if valid.sum() < 2:
            continue
        prof = np.interp(np.arange(50), np.where(valid)[0], prof[valid])
        profiles.append(prof)
    if len(profiles) < 2:
        return None
    clusters, cents = [[0]], [profiles[0].copy()]
    for i in range(1, len(profiles)):
        d = [float(np.mean(np.abs(profiles[i] - c))) for c in cents]
        best = int(np.argmin(d))
        if d[best] <= 96:
            clusters[best].append(i)
            cents[best] = np.mean([profiles[j] for j in clusters[best]], axis=0)
        else:
            clusters.append([i])
            cents.append(profiles[i].copy())
    dominant = max(len(c) for c in clusters) / len(profiles)
    return len(clusters), dominant


def _logo_b64() -> str | None:
    try:
        from PIL import Image
        img = Image.open(LOGO_PATH)
        img.thumbnail((160, 44))
        buf = io.BytesIO()
        img.save(buf, "PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None


# stylized Meat-Boy-ish cube of our own; nothing copyrighted
_MEATBOY_SVG = (
    "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'>"
    "<rect x='2' y='4' width='20' height='18' rx='4' fill='%23c81e1e'/>"
    "<rect x='2' y='4' width='20' height='7' rx='4' fill='%23e13030'/>"
    "<circle cx='9' cy='12' r='2.2' fill='white'/><circle cx='16' cy='12' r='2.2' fill='white'/>"
    "<circle cx='9.6' cy='12.4' r='1' fill='black'/><circle cx='16.6' cy='12.4' r='1' fill='black'/>"
    "<path d='M9 17.5 q3 2.4 6.5 0' stroke='black' stroke-width='1.3' fill='none' stroke-linecap='round'/>"
    "</svg>")

_icon_cache: dict[str, str | None] = {}


def _game_icon(game: str) -> str:
    """Small inline icon per game: the mario sprite from the engine assets, a
    stylized meat cube for meatboy, a letter badge otherwise."""
    if game in _icon_cache:
        src = _icon_cache[game]
    elif game == "mario":
        src = None
        try:
            from PIL import Image
            img = Image.open(os.path.join(GAMES_DIR, "assets", "idle1.png"))
            img.thumbnail((28, 36))
            buf = io.BytesIO()
            img.save(buf, "PNG")
            src = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception:
            pass
        _icon_cache[game] = src
    elif game == "meatboy":
        src = _MEATBOY_SVG
        _icon_cache[game] = src
    else:
        _icon_cache[game] = None
        src = None
    if src:
        return f'<img class="gicon" src="{src}" alt="">'
    return f'<span class="gicon gletter">{game[:1].upper()}</span>'


# ── verdicts (B1/B2/B3 vs threshold bands) ──────────────────────────────────

def _in_band(value, target, warn) -> bool:
    return value is not None and abs(value - target) <= warn


def _verdicts(row: dict, strat: tuple[int, float] | None, th: dict) -> str:
    out = []
    b1 = (th.get("B1_challenge_calibration") or {}).get("thresholds", {})
    cr_ok = _in_band(row.get("completion_rate_mean"), b1.get("target_completion_rate", 0.7),
                     b1.get("warning_completion_rate_difference", 0.2))
    t_ok = _in_band(row.get("mean_completion_time"), b1.get("target_mean_completion_time", 20),
                    b1.get("warning_mean_completion_time_difference", 4))
    out.append(("B1 challenge", cr_ok and t_ok, cr_ok or t_ok, TIPS["B1"]))
    b2 = (th.get("B2_punishment_severity") or {}).get("thresholds", {})
    d_ok = _in_band(row.get("deaths_per_run_mean"), b2.get("target_deaths_per_run", 2),
                    b2.get("warning_deaths_per_run", 1))
    e_ok = _in_band(row.get("death_cluster_entropy_mean"), b2.get("target_death_cluster_entropy", 0.4),
                    b2.get("warning_death_cluster_entropy", 0.1))
    out.append(("B2 punishment", d_ok and e_ok, d_ok or e_ok, TIPS["B2"]))
    if strat is not None:
        b3 = (th.get("B3_strategy_diversity") or {}).get("thresholds", {})
        s_ok = _in_band(strat[0], b3.get("target_strategy_count", 2), b3.get("warning_strategy_count", 1))
        p_ok = _in_band(strat[1], b3.get("target_dominant_path_share", 0.5),
                        b3.get("warning_dominant_path_share", 0.1))
        out.append(("B3 diversity", s_ok and p_ok, s_ok or p_ok, TIPS["B3"]))
    pills = []
    for label, ok, partial, tip in out:
        cls = "pill-ok" if ok else ("pill-warn" if partial else "pill-bad")
        word = "in band" if ok else ("partial" if partial else "off band")
        pills.append(f'<span class="pill {cls} tip" data-tip="{tip}">{label}: {word}</span>')
    return " ".join(pills)


# ── HTML pieces ──────────────────────────────────────────────────────────────

CSS = """
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
  :root{--bg:#070708;--panel:#101012;--panel2:#161618;--line:#232326;--line2:#2c2c30;
    --txt:#e8e8ea;--dim:#9a9aa0;--faint:#606066;
    --red:#ef4444;--red-dim:#b91c1c;--blue:#4a9eff;--yellow:#eab308;--green:#22c55e;
    --mono:'IBM Plex Mono',Consolas,monospace;--ui:'IBM Plex Sans','Segoe UI',sans-serif}
  *{box-sizing:border-box;margin:0}
  html{font-size:clamp(15px,1.05vw + 10px,17px);scroll-behavior:smooth}
  body{background:var(--bg);color:var(--txt);font:1rem/1.55 var(--ui);
    padding:0 clamp(12px,3vw,32px) clamp(12px,3vw,32px)}
  main{max-width:1280px;margin:0 auto}

  /* top navigation */
  nav{position:sticky;top:0;z-index:50;display:flex;flex-wrap:wrap;align-items:center;gap:14px;
    background:rgba(7,7,8,.92);backdrop-filter:blur(6px);border-bottom:1px solid var(--line2);
    padding:10px clamp(12px,3vw,32px);margin:0 calc(-1 * clamp(12px,3vw,32px)) 20px}
  nav img{height:34px;display:block}
  nav .brand{font:700 1rem var(--mono);letter-spacing:.08em;color:#fff}
  nav .brand em{font-style:normal;color:var(--red)}
  nav .links{display:flex;flex-wrap:wrap;gap:6px;margin-left:auto}
  nav a{font:600 .74rem var(--mono);letter-spacing:.08em;text-transform:uppercase;
    color:var(--dim);text-decoration:none;border:1px solid var(--line2);border-radius:6px;
    padding:6px 12px;transition:all .15s}
  nav a:hover{color:#fff;border-color:var(--red)}
  nav a.doc{color:var(--blue);border-color:#1d3a5f}

  section.game{border:1px solid var(--line);border-radius:10px;background:var(--panel);
    padding:clamp(14px,2.5vw,24px);margin-bottom:26px;scroll-margin-top:70px}
  .gamehead{display:flex;flex-wrap:wrap;align-items:center;gap:14px;margin-bottom:16px}
  .gametag{font:700 1.05rem var(--mono);letter-spacing:.14em;text-transform:uppercase;
    color:#fff;background:var(--red-dim);border-radius:6px;padding:5px 14px}
  .gamemeta{font:400 .78rem var(--mono);color:var(--faint)}
  .tabs{display:flex;flex-wrap:wrap;gap:8px;margin-left:auto}
  .tab{font:600 .78rem var(--mono);letter-spacing:.06em;text-transform:uppercase;cursor:pointer;
    color:var(--dim);background:var(--panel2);border:1px solid var(--line2);border-radius:6px;
    padding:7px 14px;transition:all .15s}
  .tab:hover{color:var(--txt);border-color:var(--faint)}
  .tab.active{color:#fff;background:var(--red-dim);border-color:var(--red)}
  .tab.active.t-overview{background:#1d3a5f;border-color:var(--blue)}
  .view{display:none} .view.active{display:block}

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

  /* hover info bubbles */
  .tip{position:relative;cursor:help}
  .tip:hover::after{content:attr(data-tip);position:absolute;left:0;bottom:calc(100% + 8px);
    z-index:60;width:min(300px,70vw);background:#1b1b1f;color:var(--txt);border:1px solid var(--line2);
    border-left:3px solid var(--blue);border-radius:6px;padding:9px 12px;
    font:400 .74rem/1.5 var(--ui);letter-spacing:0;text-transform:none;white-space:normal;
    text-align:left;box-shadow:0 8px 24px rgba(0,0,0,.5)}

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
  .verdicts{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0 2px}
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
  canvas.routes{width:100%;display:block;border-radius:4px}

  .tbltitle{font:600 .78rem var(--ui);letter-spacing:.1em;text-transform:uppercase;
    color:var(--dim);margin:20px 0 8px}
  .gloss{background:var(--panel);border:1px solid var(--line);border-left:4px solid var(--blue);
    border-radius:8px;padding:14px 18px;margin-top:22px;font-size:.82rem;color:var(--dim);line-height:1.7}
  .gloss b{color:var(--txt)}
  footer{color:var(--faint);font:400 .72rem var(--mono);margin:22px 0 8px}
  .doc-body{max-width:860px}
  .doc-body h2{font:700 1.05rem var(--mono);color:var(--red);margin:26px 0 8px;letter-spacing:.06em}
  .doc-body p,.doc-body li{color:var(--dim);font-size:.92rem}
  .doc-body code{font:600 .84rem var(--mono);color:var(--yellow);background:#151517;
    border-radius:4px;padding:1px 6px}
  .doc-body ul{padding-left:22px;margin:8px 0}
  .doc-body b{color:var(--txt)}
  /* game icons */
  .gicon{height:1.35em;width:auto;vertical-align:-0.28em;margin-right:8px;image-rendering:pixelated}
  nav a .gicon{height:1.15em;margin-right:6px}
  .gletter{display:inline-flex;align-items:center;justify-content:center;width:1.35em;height:1.35em;
    border-radius:4px;background:#333;color:#fff;font:700 .8em var(--mono)}

  /* overview split: table + radar side by side, stacking on narrow screens */
  .ovsplit{display:grid;grid-template-columns:minmax(0,1.4fr) minmax(300px,1fr);gap:14px;align-items:start}
  .radarbox{margin-top:0}
  canvas.radar{width:100%;max-width:520px;height:auto;display:block;margin:0 auto}
  @media (max-width:900px){.ovsplit{grid-template-columns:1fr}}

  /* entrance + micro animations */
  section.game{opacity:0;transform:translateY(14px);transition:opacity .5s ease,transform .5s ease}
  section.game.seen{opacity:1;transform:none}
  .mini i{width:0;transition:width .8s cubic-bezier(.2,.7,.3,1)}
  .gametag{position:relative;overflow:hidden}
  .gametag::after{content:'';position:absolute;inset:0;
    background:linear-gradient(110deg,transparent 30%,rgba(255,255,255,.18) 50%,transparent 70%);
    transform:translateX(-100%);animation:sheen 4.5s ease-in-out infinite}
  @keyframes sheen{0%,60%{transform:translateX(-100%)}80%,100%{transform:translateX(100%)}}
  @media (prefers-reduced-motion:reduce){
    *{transition:none !important;animation:none !important}
    section.game{opacity:1;transform:none}
  }

  @media (max-width:560px){
    .lvlgrid{grid-template-columns:1fr 1fr}
    .stats{grid-template-columns:1fr 1fr}
    .tabs{margin-left:0;width:100%}
    nav .links{margin-left:0}
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
    tip = TIPS.get(label, "")
    return (f'<div class="stat"><div class="lbl tip" data-tip="{tip}">{label} ⓘ</div>'
            f'<div class="val">{value}</div>'
            + (f'<div class="sub">{sub}</div>' if sub else "") + "</div>")


def _card(game: str, persona: str, row: dict, cells: list[dict], th: dict) -> str:
    wr = row["win_rate_mean"]
    fw = (f"first win gen {row['first_win_mean']}" if row["first_win_mean"] is not None
          else f"best progress {row.get('progress_at_death_mean') or 0:.0%}")
    color = _wr_var(row)
    mct = (f"{row['mean_completion_time']}s" if row.get("mean_completion_time") is not None else "—")
    pad = (f"{row['progress_at_death_mean']:.0%}" if row.get("progress_at_death_mean") is not None else "—")
    gap = row.get("novice_expert_gap_mean", 0.0)
    fw_full = (f"gen {row['first_win_mean']} ± {row['first_win_ci']}"
               if row["first_win_mean"] is not None else "never")

    routes = _collect_routes(game, persona, row["level"])
    strat = _cluster_strategies([r["p"] for r in routes if r["w"]])
    grid = _level_grid(game, row["level"])

    stats = [
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
    ]
    if strat is not None:
        stats.append(_stat("Strategies", str(strat[0]), "distinct winning routes"))
        stats.append(_stat("Dominant path", f"{strat[1]:.0%}", "share on the top route"))

    curves = json.dumps([c.get("curve", []) for c in cells])
    route_viz = ""
    if grid and routes:
        payload = json.dumps({"grid": grid, "ts": 32, "routes": routes})
        route_viz = f"""
        <div class="viz"><div class="vt">Agent routes on the level
            <span style="color:var(--green)">— wins</span>
            <span style="color:var(--red)">— deaths</span></div>
          <canvas class="routes" width="1100" height="200" data-routes='{payload}'></canvas>
          <div class="caption">Level geometry with sampled agent traces: green = winning runs,
          red = failed runs (drawn faint). Where red lines stop is where agents die.</div></div>"""

    return f"""
    <div class="lvlcard">
      <button class="cardhead" type="button">
        <div class="lvlname">{row['level']} {_pill(row)}</div>
        <div class="bignum" style="color:{color}">{wr:.0%}</div>
        <div class="lvlsub">win rate · {fw}</div>
        <div class="mini"><i style="width:{max(2, wr * 100):.0f}%;background:{color}"></i></div>
      </button>
      <div class="detail">
        <div class="verdicts">{_verdicts(row, strat, th)}</div>
        <div class="stats">{''.join(stats)}</div>
        {route_viz}
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
        trs.append(
            f"<tr><td>{r['level']}</td>"
            f"<td class='{solved_cls}'>{r['solved_by']}/{r['seeds']}</td>"
            f"<td>{fw}</td><td>{wr}</td>"
            f"<td>{r.get('completion_rate_mean', 0):.0%}</td><td>{mct}</td>"
            f"<td>{pad}</td><td>{r.get('deaths_per_run_mean', 0)}</td>"
            f"<td>{r.get('death_cluster_entropy_mean', 0):.2f}</td>"
            f"<td>{r.get('coin_collection_rate_mean', 1):.0%}</td>"
            f"<td>{r.get('novice_expert_gap_mean', 0.0):+.0%}</td>"
            f"<td>{_cause_name(r['dominant_cause'])} ({r['dominant_cause_frac']:.0%})</td>"
            f"<td>{r['stuck_frac_mean']:.0%}</td></tr>")
    tip = lambda k: f'class="tip" data-tip="{TIPS[k]}"'  # noqa: E731
    return f"""
    <div class="ovwrap"><table>
      <thead><tr><th>Level</th><th>Solved</th><th {tip("First win")}>First win</th>
        <th {tip("Win rate")}>Win rate ±CI</th><th {tip("Completion")}>Completion</th>
        <th {tip("Mean win time")}>Mean time</th><th {tip("Progress at death")}>Progress@death</th>
        <th {tip("Deaths per gen")}>Deaths/gen</th><th {tip("Death spread")}>Death spread</th>
        <th {tip("Coin rate")}>Coin rate</th><th {tip("Skill gap")}>Skill gap</th>
        <th {tip("Dominant cause")}>Dominant cause</th><th {tip("Stuck rate")}>Stuck</th></tr></thead>
      <tbody>{''.join(trs)}</tbody>
    </table></div>"""


def _overview(personas: dict[str, dict]) -> str:
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

    # balance radar: one axis per level, one polygon per persona (win rate 0..1)
    pcolors = {"experienced": "#ef4444", "novice": "#4a9eff", "speedrunner": "#eab308"}
    series = []
    for p in pnames:
        rows_by_lvl = {r["level"]: r for r in personas[p]["levels"]}
        vals = [round((rows_by_lvl.get(lvl) or {}).get("win_rate_mean", 0.0), 3) for lvl in order]
        series.append({"name": p, "color": pcolors.get(p, "#a855f7"), "vals": vals})
    radar_payload = json.dumps({"axes": [str(v) for v in order], "series": series})
    legend = " ".join(f'<span style="color:{s["color"]}">●</span> {s["name"]}' for s in series)
    radar = f"""
    <div class="viz radarbox"><div class="vt">Balance radar — win rate per level, one shape per
      persona &nbsp;<span style="font-weight:400;color:var(--faint)">{legend}</span></div>
      <canvas class="radar" width="520" height="440" data-radar='{radar_payload}'></canvas>
      <div class="caption">Each spoke is a level; distance from center = win rate (rings at 25 /
      50 / 75 / 100%). A bigger shape = an easier game for that skill tier; dents point at the
      hard levels. Personas nesting inside each other (yellow ⊃ red ⊃ blue) is healthy tier
      separation.</div></div>"""

    return f"""
    <div class="ovsplit">
      <div class="ovwrap"><table>
        <thead><tr><th>Level</th><th>Status</th>{ths}</tr></thead>
        <tbody>{''.join(trs)}</tbody>
      </table></div>
      {radar}
    </div>
    <div class="ovnote">Status = best across personas. Select a persona tab for the full
    per-level breakdown — cards expand in place with routes, death maps, causes, and curves.</div>"""


def _game_section(game: str, personas: dict[str, dict], idx: int, th: dict) -> str:
    metas = next(iter(personas.values()))
    tabs = [f'<button class="tab t-overview active" data-view="v{idx}_ov" type="button">Overview</button>']
    views = [f'<div class="view active" id="v{idx}_ov">{_overview(personas)}</div>']
    for pi, (pname, data) in enumerate(personas.items()):
        vid = f"v{idx}_{pi}"
        tabs.append(f'<button class="tab" data-view="{vid}" type="button">{pname}</button>')
        cell_map = data.get("cells", {})
        cards = "".join(_card(game, pname, r, cell_map.get(r["level"], []), th)
                        for r in data["levels"])
        views.append(f"""
        <div class="view" id="{vid}">
          <div class="lvlgrid">{cards}</div>
          <div class="tbltitle">Full metric table — {pname}</div>
          {_metric_table(data["levels"])}
        </div>""")
    return f"""
  <section class="game" id="g_{game}">
    <div class="gamehead">
      <span class="gametag">{_game_icon(game)}{game}</span>
      <span class="gamemeta">seeds {metas['seeds']} · budget {metas['gens_budget']} gens/probe</span>
      <div class="tabs">{''.join(tabs)}</div>
    </div>
    {''.join(views)}
  </section>"""


def _runs_section() -> str:
    """Every training run under runs/ (probes and balance output excluded)."""
    rows = []
    for sp in sorted(glob.glob(os.path.join("runs", "*", "state.json"))):
        name = os.path.basename(os.path.dirname(sp))
        if name in ("balance", "probes", "_replay"):
            continue
        try:
            with open(sp, encoding="utf-8") as f:
                s = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        hist = s.get("history", [])
        wins = sum(r.get("wins", 0) for r in hist)
        tsec = int(sum(r.get("duration") or 0 for r in hist))
        rows.append(f"<tr><td>{name}</td><td>{s.get('persona') or '—'}</td>"
                    f"<td>{s.get('generation', 0)}</td>"
                    f"<td>{s.get('best_fitness', 0):,.0f}</td>"
                    f"<td>{s.get('best_level') or '—'}</td>"
                    f"<td class='{'wr-good' if wins else 'wr-na'}'>{wins}</td>"
                    f"<td>{tsec // 3600}:{tsec % 3600 // 60:02d}:{tsec % 60:02d}</td></tr>")
    if not rows:
        return ""
    return f"""
  <section class="game" id="g_runs">
    <div class="gamehead"><span class="gametag" style="background:#1d3a5f">Training runs</span>
      <span class="gamemeta">every run found under runs/ · resume with
      <span style="color:var(--yellow)">--resume runs/&lt;name&gt;</span></span></div>
    <div class="ovwrap"><table>
      <thead><tr><th>Run</th><th>Persona</th><th>Gens</th><th>Best fitness</th>
        <th>Best level</th><th>Total wins</th><th>Train time</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table></div>
  </section>"""


JS = """
for (const tab of document.querySelectorAll('.tab')){
  tab.addEventListener('click', () => {
    const section = tab.closest('section.game');
    section.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    section.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    tab.classList.add('active');
    section.querySelector('#' + tab.dataset.view).classList.add('active');
  });
}
for (const head of document.querySelectorAll('.cardhead')){
  head.addEventListener('click', () => {
    const card = head.closest('.lvlcard');
    const grid = card.closest('.lvlgrid');
    const wasOpen = card.classList.contains('open');
    grid.querySelectorAll('.lvlcard.open').forEach(c => c.classList.remove('open'));
    if (!wasOpen) card.classList.add('open');
  });
}

// learning curves
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

const REDUCED = matchMedia('(prefers-reduced-motion: reduce)').matches;

// sections fade up as they enter the viewport; win-rate bars fill on reveal
const io = new IntersectionObserver(entries => {
  for (const e of entries) if (e.isIntersecting){
    e.target.classList.add('seen');
    e.target.querySelectorAll('.mini i').forEach(b => { b.style.width = b.dataset.w || b.style.width; });
    io.unobserve(e.target);
  }
}, {threshold: 0.08});
for (const s of document.querySelectorAll('section.game')){
  s.querySelectorAll('.mini i').forEach(b => { b.dataset.w = b.style.width; if (!REDUCED) b.style.width = '0'; });
  if (REDUCED) s.classList.add('seen'); else io.observe(s);
}

// balance radar — one axis per level, one animated polygon per persona
function drawRadar(cv, t){
  const d = JSON.parse(cv.dataset.radar || 'null');
  if (!d || !d.axes.length) return;
  const ctx = cv.getContext('2d');
  const W = cv.width, H = cv.height, cx = W / 2, cy = H / 2 + 6;
  const R = Math.min(W, H) / 2 - 52;
  const n = d.axes.length;
  const ang = i => -Math.PI / 2 + i * 2 * Math.PI / n;
  ctx.clearRect(0, 0, W, H);
  ctx.font = '400 12px "IBM Plex Mono", monospace';
  for (const frac of [0.25, 0.5, 0.75, 1]){
    ctx.strokeStyle = frac === 0.5 ? '#33333a' : '#222226';
    ctx.beginPath();
    for (let i = 0; i <= n; i++){
      const a = ang(i % n), r = R * frac;
      const x = cx + r * Math.cos(a), y = cy + r * Math.sin(a);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }
    ctx.stroke();
  }
  ctx.fillStyle = '#606066';
  ctx.fillText('50%', cx + 4, cy - R * 0.5 - 3);
  for (let i = 0; i < n; i++){
    const a = ang(i);
    ctx.strokeStyle = '#222226';
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + R * Math.cos(a), cy + R * Math.sin(a)); ctx.stroke();
    const lx = cx + (R + 16) * Math.cos(a), ly = cy + (R + 16) * Math.sin(a);
    ctx.fillStyle = '#9a9aa0';
    ctx.textAlign = Math.abs(Math.cos(a)) < 0.3 ? 'center' : (Math.cos(a) > 0 ? 'left' : 'right');
    ctx.fillText(d.axes[i], lx, ly + 4);
  }
  ctx.textAlign = 'left';
  for (const s of d.series){
    ctx.beginPath();
    for (let i = 0; i <= n; i++){
      const a = ang(i % n), v = Math.min(1, s.vals[i % n] || 0) * t;
      const x = cx + R * v * Math.cos(a), y = cy + R * v * Math.sin(a);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }
    ctx.closePath();
    ctx.strokeStyle = s.color; ctx.lineWidth = 2; ctx.stroke();
    ctx.fillStyle = s.color + '22'; ctx.fill();
    for (let i = 0; i < n; i++){
      const a = ang(i), v = Math.min(1, s.vals[i] || 0) * t;
      ctx.fillStyle = s.color;
      ctx.beginPath(); ctx.arc(cx + R * v * Math.cos(a), cy + R * v * Math.sin(a), 2.6, 0, 7); ctx.fill();
    }
  }
}
for (const cv of document.querySelectorAll('canvas.radar')){
  if (REDUCED){ drawRadar(cv, 1); continue; }
  const t0 = performance.now();
  const tick = now => {
    const t = Math.min(1, (now - t0) / 700);
    drawRadar(cv, 1 - Math.pow(1 - t, 3));  // ease-out cubic
    if (t < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

// agent routes over level geometry
for (const cv of document.querySelectorAll('canvas.routes')){
  const d = JSON.parse(cv.dataset.routes || 'null');
  if (!d || !d.grid.length) continue;
  const cols = Math.max(...d.grid.map(r => r.length));
  const rows = d.grid.length;
  const scale = cv.width / (cols * d.ts);
  cv.height = Math.max(90, Math.round(rows * d.ts * scale));
  const ctx = cv.getContext('2d');
  ctx.fillStyle = '#0c0c0e'; ctx.fillRect(0, 0, cv.width, cv.height);
  const cell = d.ts * scale;
  d.grid.forEach((row, r) => {
    for (let c = 0; c < row.length; c++){
      const ch = row[c];
      if (ch === ' ') continue;
      ctx.fillStyle = ch === '#' ? '#26262c' : (ch === '^' ? '#7f1d1d' : '#14532d');
      ctx.fillRect(c * cell, r * cell, cell + 0.5, cell + 0.5);
    }
  });
  const draw = (route, color, width, alpha) => {
    ctx.strokeStyle = color; ctx.globalAlpha = alpha; ctx.lineWidth = width;
    ctx.beginPath();
    route.p.forEach((pt, i) => {
      const x = pt[0] * scale, y = pt[1] * scale;
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke();
    ctx.globalAlpha = 1;
  };
  for (const r of d.routes) if (!r.w) draw(r, '#ef4444', 1, 0.30);
  for (const r of d.routes) if (r.w) draw(r, '#22c55e', 1.6, 0.85);
}
"""

GLOSSARY = """
  <div class="gloss">
    <b>Reading this report</b> —
    Each game opens on the <b>Overview</b>: one row per level, one column per persona, so skill
    tiers sit side by side. The number is the <b>win rate</b>
    (<span style="color:#22c55e">green ≥ 50%</span>,
    <span style="color:#eab308">yellow below</span>,
    <span style="color:#ef4444">red = never solved</span>).
    Hover any metric label or column header for what it means; the full manual lives on the
    <b>Instructions</b> page in the top bar.
  </div>"""


def _nav(games: list[str], has_runs: bool, logo: str | None, doc_page: bool = False) -> str:
    img = f'<img src="data:image/png;base64,{logo}" alt="PEAK">' if logo else ""
    links = "".join(f'<a href="report.html#g_{g}">{_game_icon(g)}{g}</a>' for g in games)
    if has_runs:
        links += '<a href="report.html#g_runs">runs</a>'
    links += '<a class="doc" href="instructions.html">Instructions</a>' if not doc_page \
        else '<a class="doc" href="report.html">← Back to command</a>'
    return f"""
  <nav>{img}<span class="brand">PEAK <em>BALANCE COMMAND</em></span>
    <div class="links">{links}</div>
  </nav>"""


def _instructions_page(games: list[str], logo: str | None) -> str:
    tips = "".join(f"<li><b>{k}</b> — {v}</li>" for k, v in TIPS.items())
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PEAK Balance Command — Instructions</title><style>{CSS}</style></head><body>
{_nav(games, False, logo, doc_page=True)}
<main class="doc-body">
  <h2>What this is</h2>
  <p>PEAK evolves populations of tiny neural networks (291 parameters) that play your levels
  thousands of times, then reports what happened. Nothing here blocks anything — the tool
  measures, you decide.</p>

  <h2>Running a sweep</h2>
  <ul>
    <li><code>python menu.py</code> → <b>15 Full Sweep</b>: pick games and personas; probes run
    in parallel across CPU cores and land in <code>runs/balance/</code>.</li>
    <li><b>14 Balance Report</b> probes one game × one persona with custom levels/seeds/budget.</li>
    <li>Only <b>enabled</b> levels are probed — toggle them with <b>10 Toggle Levels</b>.</li>
    <li>Regenerate this page any time: <code>python -m code.neuro.report --open</code>.</li>
  </ul>

  <h2>Personas (skill tiers)</h2>
  <ul>
    <li><b>novice</b> — walking pace, reacts every 3rd frame. "Can beginners get through?"</li>
    <li><b>experienced</b> — full reactions, walking pace. The baseline difficulty read.</li>
    <li><b>speedrunner</b> — sprint speed plus a time bonus. "What does skilled, fast play look like?"</li>
  </ul>
  <p>The Overview puts all three side by side. Healthy levels slope novice → speedrunner upward;
  inversions are design findings (a level that punishes speed, or walls out beginners).</p>

  <h2>The design loop</h2>
  <ul>
    <li>Author or edit a level (menu 9), play it yourself (menu 5).</li>
    <li>Probe it (menu 14) and read the card: win rate, first-win generation, where deaths cluster.</li>
    <li>Change one thing — double or halve a parameter — and probe again.</li>
    <li>An unsolved level whose best progress is identical across seeds is usually broken
    geometry (unreachable goal), not difficulty.</li>
  </ul>

  <h2>Every metric</h2>
  <ul>{tips}</ul>

  <h2>Reproducibility</h2>
  <p>Everything is deterministic under a seed. Default probe seeds are
  <code>1234 · 2025 · 31337</code>; each (level × seed) cell evolves a fresh population with the
  same GA settings, so numbers are comparable across sweeps as long as the GA config is unchanged.</p>
</main>
<footer style="max-width:860px;margin:22px auto">PEAK ENGINE · instructions</footer>
</body></html>"""


def build(balance_dir: str) -> tuple[str, str]:
    games: dict[str, dict[str, dict]] = {}
    for f in sorted(glob.glob(os.path.join(balance_dir, "report_*.json"))):
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        persona = data.get("persona") or "experienced"
        games.setdefault(data["game"], {})[persona] = data
    th = _load_thresholds()
    logo = _logo_b64()
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    sections = [_game_section(g, p, i, th) for i, (g, p) in enumerate(games.items())]
    runs_html = _runs_section()
    body = "".join(sections) + runs_html if sections or runs_html else \
        "<p style='color:var(--dim)'>No balance data yet — run a Balance Report (menu 14) or Full Sweep (menu 15) first.</p>"
    report = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'><rect width='16' height='16' rx='3' fill='%23070708'/><path d='M3 12 L8 4 L13 12' stroke='%23ef4444' stroke-width='2' fill='none'/></svg>">
<title>PEAK Balance Command</title><style>{CSS}</style></head><body>
{_nav(list(games), bool(runs_html), logo)}
<main>
  <div style="font:400 .78rem var(--mono);color:var(--faint);margin-bottom:16px">
    multi-seed neuroevolution probes · generated {stamp}</div>
  {body}
  {GLOSSARY}
  <footer>PEAK ENGINE · code/neuro/report.py · data: runs/balance + runs/probes + runs/*</footer>
</main><script>{JS}</script></body></html>"""
    return report, _instructions_page(list(games), logo)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the balance command center")
    ap.add_argument("--dir", default=os.path.join("runs", "balance"))
    ap.add_argument("--open", action="store_true", help="open in the default browser")
    args = ap.parse_args()

    report, instructions = build(args.dir)
    os.makedirs(args.dir, exist_ok=True)
    out = os.path.join(args.dir, "report.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(report)
    with open(os.path.join(args.dir, "instructions.html"), "w", encoding="utf-8") as f:
        f.write(instructions)
    print(f"wrote {out} (+ instructions.html)")
    if args.open:
        webbrowser.open("file:///" + os.path.abspath(out).replace(os.sep, "/"))


if __name__ == "__main__":
    main()
