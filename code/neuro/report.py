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
from .balance import BALANCE_DIR, PROBES_ROOT, WIN_WINDOW, fmt_hms, rebuild
# Level glyphs -> canvas categories (see code/games/levels/common/ASCII_TILEMAP.md)
GLYPHS = {"#": "#%([/\\])Un", "=": "=", "?": "?<>FL", "^": "^*O", "E": "EkKMBX",
          "C": "C", "G": "G", "S": "S", "H": "H", "P": "P", "D": "D"}
GLYPH_CAT = {ch: cat for cat, chars in GLYPHS.items() for ch in chars}
GLYPH_NAME = {"#": "solid", "=": "one-way platform", "?": "question block", "^": "hazard / pit / saw",
              "E": "enemy", "C": "coin / ring", "G": "goal", "S": "spring", "H": "ladder",
              "P": "player start", "D": "door"}

# hover blurbs for every metric (the little info bubbles)
TIPS = {
    "Status": "solved = every seed's population won at least once · partial (n/N seeds) = only some "
              "seeds cracked it · unsolved = no seed ever won. On the Overview it is the best across personas.",
    "Solved": "Seeds whose population produced at least one WON episode within the budget.",
    "Win rate": "Share of episodes won in the 10 generations after the population's first win. "
                "Green means the population beats the level more often than not.",
    "First win": "How many generations evolution needed to solve the level once — "
                 "the primary difficulty signal. Lower = easier.",
    "Completion": "Wins divided by every episode of the whole probe, including early chaotic generations.",
    "Mean win time": "Average in-game time of the winning runs only, ± the standard deviation "
                     "of those win times (averaged across seeds).",
    "Best gen": "Generation at which the all-time best genome was found (mean ± 95% CI across seeds). "
                "Close to First win = the population peaks as soon as it solves the level.",
    "Improvement": "Rate of improvement: slope of best progress per generation, as a share of the level, "
                   "measured from gen 1 to the generation the peak was first reached.",
    "Train time": "Wall-clock time spent evolving on this level, summed over all seeds.",
    "Progress at death": "How far through the level failing agents get on average (0% = start, 100% = goal).",
    "Deaths per gen": "Average deaths per generation (one attempt per population member).",
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
    """Level geometry + entities reduced to GLYPHS categories (' ' = air)."""
    path = _level_file(game, level)
    if not path or not os.path.exists(path):
        return None
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f.read().splitlines():
            rows.append("".join(GLYPH_CAT.get(ch, " ") for ch in line).rstrip())
    while rows and not rows[-1]:
        rows.pop()
    return rows or None


def _flatten(d, prefix: str = "") -> dict:
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def _module_constants(modname: str) -> dict:
    try:
        import importlib
        mod = importlib.import_module(modname)
        return {k.lower(): v for k, v in vars(mod).items()
                if k.isupper() and isinstance(v, (int, float, str, bool))}
    except Exception:
        return {}


def _level_config(game: str, level: str, grid: list[str] | None) -> list[tuple[str, dict]]:
    """Everything the designer set for this level: level entry, size + entity census,
    sidecar YAML (dynamics), player/physics/enemy settings of the game."""
    import yaml
    sections: list[tuple[str, dict]] = []
    try:
        if game == "meatboy":
            with open(os.path.join(GAMES_DIR, "meatboy_config.yaml"), encoding="utf-8") as f:
                g = yaml.safe_load(f) or {}
            entry = {"file": (g.get("levels") or [None] * (int(level) + 1))[int(level)]}
        else:
            with open(os.path.join(GAMES_DIR, "game_config.yaml"), encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            g = {"mario": cfg, "megaman": cfg.get("megaman", {}), "sonic": cfg.get("sonic", {})}.get(game, {})
            entry = ((g.get("levels") or {}).get(level) or (g.get("disabled_levels") or {}).get(level) or {})
    except Exception:
        return sections
    lvl = dict(_flatten(entry))
    if grid:
        cols = max((len(r) for r in grid), default=0)
        lvl.update({"size (tiles)": f"{cols} × {len(grid)}", "length (px)": cols * 32})
    sections.append(("Level", lvl))
    if grid:
        census = {}
        for row in grid:
            for ch in row:
                if ch not in " #":
                    census[GLYPH_NAME[ch]] = census.get(GLYPH_NAME[ch], 0) + 1
        if census:
            sections.append(("Entities", dict(sorted(census.items()))))
    path = _level_file(game, level)
    side = os.path.splitext(path)[0] + ".yaml" if path else None
    if side and os.path.exists(side):
        try:
            with open(side, encoding="utf-8") as f:
                sc = yaml.safe_load(f) or {}
            for k, v in sc.items():  # dynamics / physics overrides written next to the level
                flat = _flatten(v) if isinstance(v, dict) else {k: v}
                if flat:
                    sections.append((f"Level {k}", flat))
        except Exception:
            pass
    if game == "mario":
        phys = _module_constants("code.games.modules.Parameters.Movement_parameters")
        phys.update(_module_constants("code.games.modules.Parameters.Jump_parameters"))
        sections.append(("Player physics", phys))
    elif game == "meatboy":
        sections.append(("Player", _flatten(g.get("player"))))
        phys = {}
        for k in ("physics", "movement", "jump", "wall"):
            phys.update(_flatten(g.get(k), k + "."))
        sections.append(("Player physics", phys))
    else:
        sections.append(("Player", _flatten((g.get("defaults") or {}).get("player"))))
        phys = _flatten(g.get("physics"))
        enemies = {k[len("enemies."):]: v for k, v in phys.items() if k.startswith("enemies.")}
        phys = {k: v for k, v in phys.items() if not k.startswith("enemies.")}
        if phys:
            sections.append(("Physics", phys))
        if enemies:
            sections.append(("Enemies", enemies))
    return [(t, d) for t, d in sections if d]


def _config_html(sections: list[tuple[str, dict]]) -> str:
    if not sections:
        return ""
    blocks = []
    for title, d in sections:
        rows = "".join(f'<div class="kv"><b>{k}</b><span>{v}</span></div>' for k, v in d.items())
        blocks.append(f'<div class="cfgblock"><div class="cfgtitle">{title}<em>{len(d)}</em></div>{rows}</div>')
    chips = "".join(f'<span class="cfgchip">{t}</span>' for t, _ in sections)
    return f"""
        <details class="cfg"><summary><span class="sumlbl">Level config</span>{chips}
          <span class="sumhint">click to expand</span></summary>
          <div class="cfggrid">{''.join(blocks)}</div>
          <div class="caption">Read from game_config.yaml / meatboy_config.yaml, the level's sidecar
          .yaml, and the Parameters modules — what the engine used when these probes ran is whatever
          those files say now.</div>
        </details>"""


def _collect_routes(game: str, persona: str, level: str, tag: str | None,
                    max_routes: int = 22) -> list[dict]:
    """Route traces for one (game, persona, level) from the probe episode CSVs."""
    won_raw, lost_raw = [], []
    sub = (game, persona, tag) if tag else (game,)  # legacy reports pre-date the tagged layout
    pattern = os.path.join(PROBES_ROOT, *sub, f"{level}_*".replace(" ", "_"), "episodes.csv")
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path, encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    if str(row.get("world")) != str(level) or row.get("persona") != persona:
                        continue
                    (won_raw if (row.get("cause_of_death") or "").lower() == "success"
                     else lost_raw).append(row.get("route") or "[]")
        except OSError:
            continue

    def parse(raw: str) -> list | None:  # only the kept rows are parsed — the CSVs run to 10^5+ rows
        try:
            pts = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return None
        if len(pts) < 2:
            return None
        step = max(1, len(pts) // 60)  # ≤60 points per embedded route
        return [[round(x), round(y)] for x, y in pts[::step]]

    won = [p for p in map(parse, won_raw[:10]) if p]
    lost = [p for p in map(parse, lost_raw[-(max_routes - len(won)):]) if p]
    return [{"p": p, "w": 1} for p in won] + [{"p": p, "w": 0} for p in lost]


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
  nav a,nav .navbtn{display:inline-flex;align-items:center;gap:6px;line-height:1;
    font:600 .74rem var(--mono);letter-spacing:.08em;text-transform:uppercase;
    color:var(--dim);text-decoration:none;border:1px solid var(--line2);border-radius:6px;
    padding:0 12px;height:34px;transition:all .15s}
  nav a:hover{color:#fff;border-color:var(--red)}
  nav a.doc{color:var(--blue);border-color:#1d3a5f}
  nav a.doc.abl{color:var(--yellow);border-color:#5a4a10}

  /* sensor ablation page */
  .ablhero{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:22px}
  .ablmode{background:var(--panel);border:1px solid var(--line);border-top:3px solid var(--c);
    border-radius:10px;padding:16px 18px;display:grid;grid-template-columns:150px 1fr;gap:16px;align-items:center}
  .ablmode svg{width:150px;height:150px;display:block}
  .ablmode h3{font:700 .95rem var(--mono);letter-spacing:.12em;text-transform:uppercase;color:var(--c)}
  .ablmode p{font:400 .84rem/1.5 var(--ui);color:var(--dim);margin:6px 0 10px}
  .ablscore{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:0 0 22px;
    font:600 .8rem var(--mono);color:var(--dim)}
  .ablscore b{font-size:1.4rem;color:var(--c)}
  .ablscore .sep{color:var(--faint);font-weight:400}
  .dual{display:grid;grid-template-columns:minmax(110px,160px) 1fr;gap:4px 14px;align-items:center}
  .dual .lv{font:600 .82rem var(--mono);color:var(--txt);grid-row:span 2;align-self:center}
  .dual .bar{position:relative;height:16px;border-radius:3px;background:var(--line);overflow:hidden}
  .dual .bar i{display:block;height:100%;background:var(--c);border-radius:3px}
  .dual .bar em{position:absolute;inset:0;font:600 .66rem/16px var(--mono);font-style:normal;
    color:#fff;padding-left:6px;white-space:nowrap;text-shadow:0 0 4px #000}
  .dual .gap{height:8px;grid-column:1/-1}
  .delta-r{color:var(--red);font-weight:700}.delta-g{color:var(--blue);font-weight:700}.delta-0{color:var(--faint)}
  th.mr,td.mr{border-left:1px solid var(--line2)}
  th.mg,td.mg{border-left:1px solid var(--line2)}
  thead tr.modes th{text-align:center;color:#fff;background:var(--panel);letter-spacing:.16em}
  .stats.ablstats{grid-template-columns:repeat(4,1fr)}
  .curvegrid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:10px}
  .curvegrid .viz{margin-top:0}
  .curvegrid canvas.curve{height:170px}
  @media (max-width:900px){.curvegrid{grid-template-columns:1fr}}
  @media (max-width:700px){.stats.ablstats{grid-template-columns:1fr 1fr}}
  @media (max-width:900px){.ablhero{grid-template-columns:1fr}.ablmode{grid-template-columns:110px 1fr}
    .ablmode svg{width:110px;height:110px}}

  section.game{border:1px solid var(--line);border-radius:10px;background:var(--panel);
    padding:clamp(14px,2.5vw,24px);margin-bottom:26px;scroll-margin-top:70px}
  .gamehead{display:flex;flex-wrap:wrap;align-items:center;gap:14px;margin-bottom:16px}
  .gametag{font:700 1.05rem var(--mono);letter-spacing:.14em;text-transform:uppercase;
    color:#fff;background:var(--red-dim);border-radius:6px;padding:5px 14px}
  .gamemeta{font:400 .78rem var(--mono);color:var(--faint)}
  code.cmd{display:block;font:600 .8rem var(--mono);color:var(--yellow);background:#151517;
    padding:8px 10px;border-radius:6px;margin:6px 0;user-select:all;overflow-x:auto;white-space:nowrap}
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
  .lvlcard:has(dialog[open]){border-color:var(--red)}
  .cardhead{display:block;width:100%;text-align:left;background:none;border:0;color:var(--txt);
    font:inherit;cursor:pointer;padding:14px 16px}
  .lvlname{font:600 .9rem var(--mono);color:#e5e5e5;margin-bottom:8px;display:flex;
    justify-content:space-between;align-items:center;gap:8px}
  .bignum{font:700 1.9rem var(--mono)}
  .lvlsub{font:400 .74rem var(--mono);color:var(--faint);margin-top:2px}
  .mini{height:4px;border-radius:2px;background:var(--line);margin-top:10px;overflow:hidden}
  .mini i{display:block;height:100%}

  dialog.detail{margin:auto;background:var(--panel2);color:var(--txt);border:1px solid var(--line2);
    border-radius:12px;width:min(1180px,94vw);max-height:92vh;padding:6px 20px 20px;
    box-shadow:0 30px 80px rgba(0,0,0,.7)}
  dialog.detail[open]{animation:slide .18s ease-out}
  dialog.detail::backdrop{background:rgba(0,0,0,.72);backdrop-filter:blur(3px)}
  .dhead{display:flex;align-items:center;gap:14px;position:sticky;top:0;background:var(--panel2);
    padding:12px 0 8px;border-bottom:1px solid var(--line2);z-index:2}
  .dhead .lvlname{margin:0;flex:1;font-size:1rem;justify-content:flex-start}
  .dhead .bignum{font-size:1.5rem}
  .close{background:none;border:1px solid var(--line2);border-radius:6px;color:var(--dim);
    font:600 .8rem var(--mono);padding:4px 9px;cursor:pointer}
  .close:hover{color:#fff;border-color:var(--red)}
  @keyframes slide{from{opacity:0;transform:translateY(-6px)}to{opacity:1;transform:none}}
  .chev{background:none;border:0;color:var(--dim);font:400 1.1rem var(--mono);cursor:pointer;
    width:26px;transition:transform .15s}
  section.game.collapsed .chev{transform:rotate(-90deg)}
  section.game.collapsed .body,section.game.collapsed .tabs{display:none}
  section.game.collapsed .gamehead{margin-bottom:0}
  .brainbtn,.navbtn{display:inline-flex;align-items:center;gap:6px;background:none;cursor:pointer;
    border:1px solid var(--line2);border-radius:6px;color:var(--dim);font:600 .72rem var(--mono);
    letter-spacing:.06em;text-transform:uppercase;padding:5px 10px;transition:all .15s}
  .toolbar{display:flex;flex-wrap:wrap;align-items:center;gap:10px;margin:14px 0 12px}
  .toolbar .tabs{margin-left:0}
  .toolbar .brainbtn{margin-left:auto}
  .cfgview{display:none} .cfgview.active{display:block}
  .cfgbar{display:flex;flex-wrap:wrap;align-items:stretch;gap:8px;margin-top:12px}
  .cfglbl{align-self:center;font:600 .68rem var(--ui);letter-spacing:.1em;text-transform:uppercase;
    color:var(--faint);margin-right:4px}
  .cfgbtn{display:flex;flex-direction:column;align-items:flex-start;gap:2px;cursor:pointer;text-align:left;
    background:var(--panel2);border:1px solid var(--line2);border-left:3px solid var(--c);border-radius:8px;
    padding:8px 14px;color:var(--dim);transition:all .15s}
  .cfgbtn b{font:700 .8rem var(--mono);letter-spacing:.06em;color:var(--txt)}
  .cfgbtn small{font:400 .68rem var(--mono);color:var(--faint)}
  .cfgbtn:hover{border-color:var(--c)}
  .cfgbtn.active{background:color-mix(in srgb,var(--c) 12%,var(--panel2));border-color:var(--c);color:#fff}
  .chartbox{margin-top:0}
  .chartwrap{display:grid;grid-template-columns:minmax(0,1fr) minmax(200px,260px);gap:14px;align-items:center}
  @media (max-width:820px){.chartwrap{grid-template-columns:1fr}}
  canvas.chart{width:100%;max-width:720px;height:auto;display:block;margin:0 auto}
  .legend.chips{flex-direction:column;align-items:stretch;gap:4px}
  .lchip{display:flex;align-items:center;gap:8px;background:none;border:1px solid var(--line);border-radius:6px;
    padding:4px 8px;color:var(--dim);font:400 .7rem var(--mono);cursor:pointer;text-align:left}
  .lchip:hover{border-color:var(--c)}
  .lchip.off{opacity:.35}
  .lchip i{display:inline-block;width:22px;height:0;border-top:2.5px solid var(--c);flex:none}
  .lchip i.novice{border-top-style:dashed}
  .lchip i.speedrunner{border-top-style:dotted}
  .lchip small{color:var(--faint);margin-left:auto}
  .dhead em.mode{font-style:normal;font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;
    padding:2px 8px;border-radius:999px;border:1px solid currentColor;vertical-align:middle}
  .dhead em.rays{color:var(--red)} .dhead em.grid{color:var(--blue)}
  .brainbtn:hover,.navbtn:hover{color:#fff;border-color:var(--blue)}
  .brainbtn svg{color:var(--blue)}
  details.cfg{margin-top:10px;background:var(--panel);border:1px solid var(--line);border-radius:8px;
    padding:10px 14px}
  details.cfg summary{cursor:pointer;display:flex;flex-wrap:wrap;align-items:center;gap:6px;list-style:none}
  details.cfg summary::-webkit-details-marker{display:none}
  details.cfg summary::before{content:"▸";color:var(--faint);font:600 .9rem var(--mono);width:14px}
  details.cfg[open] summary::before{content:"▾"}
  .sumlbl{font:600 .76rem var(--ui);color:var(--dim);margin-right:4px}
  .cfgchip{font:400 .66rem var(--mono);color:var(--dim);border:1px solid var(--line2);border-radius:999px;padding:2px 9px}
  .sumhint{margin-left:auto;font:400 .66rem var(--mono);color:var(--faint)}
  details.cfg[open] .sumhint{display:none}
  .cfggrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px;margin-top:12px}
  .cfgblock{background:var(--panel2);border:1px solid var(--line2);border-radius:8px;padding:10px 12px;align-self:start}
  .cfgtitle{display:flex;align-items:center;gap:8px;font:600 .72rem var(--ui);letter-spacing:.08em;
    text-transform:uppercase;color:var(--dim);padding-bottom:6px;border-bottom:1px solid var(--line2);margin-bottom:4px}
  .cfgtitle em{font:600 .62rem var(--mono);font-style:normal;color:var(--faint);background:var(--panel);
    border-radius:999px;padding:1px 7px;margin-left:auto}
  .kv{display:flex;justify-content:space-between;align-items:baseline;gap:14px;padding:3px 0;
    border-bottom:1px solid var(--line);font:400 .72rem var(--mono)}
  .kv:last-child{border-bottom:0}
  .kv b{font-weight:400;color:var(--faint);flex:none}
  .kv span{color:var(--txt);text-align:right;overflow-wrap:anywhere;font-variant-numeric:tabular-nums}
  .cfgblock table{width:100%;table-layout:fixed;border-collapse:collapse;font:400 .72rem var(--mono)}
  .cfgblock td{overflow-wrap:anywhere;white-space:normal}
  .cfgblock td,.cfgblock th{padding:3px 6px;border-bottom:1px solid var(--line);text-align:left;
    vertical-align:top;word-break:break-word}
  .cfgblock td:first-child,.cfgblock th:first-child{width:45%}
  .cfgblock td:first-child{color:var(--faint)}
  .cfgblock th{color:var(--dim);font-weight:600;white-space:normal}
  .legend{display:flex;flex-wrap:wrap;gap:4px 12px;font:400 .68rem var(--mono);color:var(--dim);margin-top:6px}
  .brainrow{display:grid;grid-template-columns:minmax(300px,1.15fr) minmax(280px,1fr);gap:10px;margin-top:10px}
  .brainrow>.viz{align-self:start}
  @media (max-width:900px){.brainrow{grid-template-columns:1fr}}
  svg.net{width:100%;height:auto;display:block;margin-top:4px}
  svg.net .lbl{font:400 11px var(--mono);fill:var(--dim)}
  svg.net .lbl.out{font-weight:600;font-size:12px}
  svg.net .cap{font:600 10px var(--ui);fill:var(--faint);letter-spacing:.08em;text-transform:uppercase}
  .gastats{grid-template-columns:repeat(auto-fill,minmax(130px,1fr));margin:0}
  .gastats .val{font-size:1rem}
  .pipe{display:flex;flex-wrap:wrap;align-items:center;gap:6px}
  .pipe .step{display:flex;flex-direction:column;background:var(--panel2);border:1px solid var(--line2);
    border-radius:8px;padding:7px 12px;min-width:120px}
  .pipe .step b{font:700 .74rem var(--mono);letter-spacing:.08em;text-transform:uppercase;color:#fff}
  .pipe .step small{font:400 .68rem var(--mono);color:var(--dim)}
  .pipe .arrow{color:var(--red);font:700 1.1rem var(--mono)}
  .chips{display:flex;flex-wrap:wrap;gap:8px}
  .chip{font:400 .74rem var(--mono);color:var(--dim);border:1px solid var(--c);border-radius:999px;
    padding:5px 12px;background:color-mix(in srgb,var(--c) 10%,transparent)}
  .chip b{color:var(--c)}
  .vt .faint{font-weight:400;color:var(--faint)}
  .legend .lg i{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:4px;vertical-align:-1px}
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
  .routewrap{overflow:auto;border-radius:4px}
  .viz .vt{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
  .fsbtn{margin-left:auto}
  .fsbtn,.copybtn{background:none;border:1px solid var(--line2);border-radius:5px;
    color:var(--dim);font:600 .66rem var(--mono);letter-spacing:.06em;text-transform:uppercase;
    padding:3px 8px;cursor:pointer}
  .fsbtn:hover,.copybtn:hover{color:#fff;border-color:var(--blue)}
  .viz:fullscreen{background:var(--bg);padding:24px 24px 12px;overflow:auto;border:0;border-radius:0;
    display:flex;flex-direction:column}
  .viz:fullscreen .routewrap{flex:1;display:flex;align-items:center;justify-content:safe center}
  .cmdrow{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin-top:4px}
  .watchbtn{display:inline-flex;flex:none;align-items:center;gap:6px;background:var(--red-dim);
    color:#fff;text-decoration:none;border-radius:6px;padding:6px 14px;font:700 .74rem var(--mono);
    letter-spacing:.08em;text-transform:uppercase;transition:background .15s}
  .watchbtn:hover{background:var(--red)}
  html.file .watchbtn{opacity:.35;pointer-events:none;cursor:not-allowed}
  html.file .served-only,html:not(.file) .file-only{display:none}
  .cmdrow code.cmd{flex:1;min-width:260px;margin:6px 0}
  canvas.routes{display:block;border-radius:4px;margin:0 auto}

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
  nav a .gicon{height:1.3em;margin:0;vertical-align:middle}
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


def _npz_meta(path: str) -> dict:
    try:
        import numpy as np
        return json.loads(str(np.load(path)["meta"]))
    except Exception:
        return {}


def _watch_cmd(game: str, persona: str, level: str, cells: list[dict]) -> str:
    """Replay command for the strongest seed whose best.npz is still on disk.
    Legacy probes (pre-tag layout) live at runs/probes/<game>/<level>_<seed>."""
    for c in sorted(cells, key=lambda c: -c["best"]):
        d = c.get("dir") or os.path.join(PROBES_ROOT, game, f"{level}_{c['seed']}".replace(" ", "_"))
        npz = os.path.join(d, "best.npz")
        if os.path.exists(npz):
            break
    else:
        return ('<div class="viz"><div class="vt">Watch the best agent</div><div class="caption">'
                'No saved genome found for this level — re-probe it (menu 15) to get one.</div></div>')
    meta = _npz_meta(npz)
    note = ""
    if meta and (meta.get("persona") != persona or str(meta.get("level")) != str(level)):
        note = (f' <b style="color:var(--yellow)">Note:</b> this folder predates per-config layout and '
                f'was last written by the <b>{meta.get("persona")}</b> sweep on <b>{meta.get("level")}</b> '
                f'(fitness {meta.get("fitness"):,}) — re-probe for a genome from this persona.')
    npz = npz.replace(os.sep, "/")
    from urllib.parse import urlencode
    href = "/watch?" + urlencode({"game": game, "npz": npz})
    return (f'<div class="viz"><div class="vt">Watch this agent — best genome, seed {c["seed"]}, '
            f'fitness {c["best"]:,.0f}</div>'
            f'<div class="cmdrow"><a class="watchbtn" href="{href}" target="_blank" rel="noopener">'
            f'▶ Watch replay</a><code class="cmd">python -m code.neuro.trainer --game {game} '
            f'--replay {npz}</code><button class="copybtn" type="button">copy</button></div>'
            f'<div class="caption"><span class="served-only">The button launches the replay and opens '
            f'the live dashboard (menu 12 serves this page).</span><span class="file-only">Open the '
            f'command center via <b>menu 12</b> for a one-click button; as a plain file, run the command '
            f'from the repo root (or menu 6 → pick this probe).</span>{note}</div></div>')


def _card(game: str, persona: str, row: dict, cells: list[dict], th: dict, tag: str | None) -> str:
    wr = row["win_rate_mean"]
    fw = (f"first win gen {row['first_win_mean']}" if row["first_win_mean"] is not None
          else f"best progress {row.get('progress_at_death_mean') or 0:.0%}")
    color = _wr_var(row)
    mct = (f"{row['mean_completion_time']}s" if row.get("mean_completion_time") is not None else "—")
    pad = (f"{row['progress_at_death_mean']:.0%}" if row.get("progress_at_death_mean") is not None else "—")
    gap = row.get("novice_expert_gap_mean", 0.0)
    fw_full = (f"gen {row['first_win_mean']} ± {row['first_win_ci']}"
               if row["first_win_mean"] is not None else "never")
    bg_full = (f"gen {row['best_gen_mean']} ± {row['best_gen_ci']}"
               if row.get("best_gen_mean") is not None else "—")
    sd = row.get("completion_time_stddev")
    if sd is None and cells:  # legacy rows: derive from the per-seed cells
        sds = [c["completion_time_stddev"] for c in cells if c.get("mean_completion_time") is not None]
        sd = round(sum(sds) / len(sds), 1) if sds else None
    if mct != "—" and sd is not None:
        mct += f" ± {sd}s"
    rate = row.get("improvement_rate_mean")
    pop = row.get("pop_size") or (cells[0].get("pop_size") if cells else None)  # legacy: unknown
    ttime = row.get("train_time_s")
    if ttime is None:
        ttime = sum(c.get("train_time_s") or 0 for c in cells)

    routes = _collect_routes(game, persona, row["level"], tag)
    strat = _cluster_strategies([r["p"] for r in routes if r["w"]])
    grid = _level_grid(game, row["level"])
    cfg_html = _config_html(_level_config(game, row["level"], grid))

    stats = [
        _stat("Win rate", f"{wr:.0%} ± {_ci(row):.0%}", "measured 10 gens after first win"),
        _stat("First win", fw_full, f"{row['solved_by']}/{row['seeds']} seeds solved"),
        _stat("Best gen", bg_full, "where the record genome appeared"),
        _stat("Improvement", f"{rate:+.1%}/gen" if rate is not None else "—", "of the level per generation"),
        _stat("Completion", f"{row.get('completion_rate_mean', 0):.0%}", "wins / all episodes"),
        _stat("Mean win time", mct, "mean ± std of winning runs"),
        _stat("Progress at death", pad, "how far failers get"),
        _stat("Deaths per gen", f"{row.get('deaths_per_run_mean', 0)}", f"of {pop} attempts" if pop else "per generation"),
        _stat("Death spread", f"{row.get('death_cluster_entropy_mean', 0):.2f}",
              "0 = one hotspot · 1 = everywhere"),
        _stat("Dominant cause", _cause_name(row["dominant_cause"]),
              f"{row['dominant_cause_frac']:.0%} of deaths"),
        _stat("Coin rate", f"{row.get('coin_collection_rate_mean', 1):.0%}", "collected / available"),
        _stat("Skill gap", f"{gap:+.0%}", "late-gen wins − early-gen wins"),
        _stat("Stuck rate", f"{row['stuck_frac_mean']:.0%}", "episodes ending in a stall"),
        _stat("Train time", fmt_hms(ttime), f"all {row['seeds']} seeds, wall-clock"),
    ]
    if strat is not None:
        stats.append(_stat("Strategies", str(strat[0]), "distinct winning routes"))
        stats.append(_stat("Dominant path", f"{strat[1]:.0%}", "share on the top route"))

    curves = json.dumps([c.get("curve", []) for c in cells])
    route_viz = ""
    if grid:
        payload = json.dumps({"grid": grid, "ts": 32, "routes": routes})
        legend = " ".join(f'<span class="lg"><i style="background:{c}"></i>{GLYPH_NAME[k]}</span>'
                          for k, c in (("#", "#26262c"), ("?", "#b45309"), ("^", "#7f1d1d"),
                                       ("E", "#ef4444"), ("C", "#eab308"), ("S", "#2563eb"),
                                       ("H", "#7c3aed"), ("P", "#22c55e"), ("G", "#14532d"))
                          if any(k in r for r in grid))
        route_viz = f"""
        <div class="viz"><div class="vt">Agent routes on the level
            <span style="color:var(--green)">— wins</span>
            <span style="color:var(--red)">— deaths</span>
            <button class="fsbtn" type="button">⛶ full screen</button></div>
          <div class="routewrap"><canvas class="routes" width="1100" height="200" data-routes='{payload}'></canvas></div>
          <div class="legend">{legend}</div>
          <div class="caption">Level geometry and entities with sampled agent traces: green = winning
          runs, red = failed runs (drawn faint). Where red lines stop is where agents die.
          {'' if routes else 'No route traces logged for this level yet.'}</div></div>"""

    return f"""
    <div class="lvlcard">
      <button class="cardhead" type="button">
        <div class="lvlname">{row['level']} {_pill(row)}</div>
        <div class="bignum" style="color:{color}">{wr:.0%}</div>
        <div class="lvlsub">win rate · {fw}</div>
        <div class="mini"><i style="width:{max(2, wr * 100):.0f}%;background:{color}"></i></div>
      </button>
      <dialog class="detail">
        <div class="dhead">
          <div class="lvlname">{game} · {persona} · {row['level']} {_pill(row)}</div>
          <div class="bignum" style="color:{color}">{wr:.0%}</div>
          <button class="close" type="button" aria-label="close">✕</button>
        </div>
        <div class="verdicts">{_verdicts(row, strat, th)}</div>
        <div class="stats">{''.join(stats)}</div>
        {cfg_html}
        {_watch_cmd(game, persona, row["level"], cells)}
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
      </dialog>
    </div>"""


def _metric_table(rows: list[dict]) -> str:
    trs = []
    for r in rows:
        wr = f"{r['win_rate_mean']:.0%} ±{_ci(r):.0%}"
        fw = (f"{r['first_win_mean']}±{r['first_win_ci']}" if r["first_win_mean"] is not None else "—")
        solved_cls = ("wr-good" if r["solved_by"] == r["seeds"] else
                      ("wr-mid" if r["solved_by"] else "wr-bad"))
        mct = f"{r['mean_completion_time']}s" if r.get("mean_completion_time") is not None else "—"
        if mct != "—" and r.get("completion_time_stddev") is not None:
            mct += f" ±{r['completion_time_stddev']}"
        bg = (f"{r['best_gen_mean']}±{r['best_gen_ci']}" if r.get("best_gen_mean") is not None else "—")
        rate = (f"{r['improvement_rate_mean']:+.1%}" if r.get("improvement_rate_mean") is not None else "—")
        pad = f"{r['progress_at_death_mean']:.0%}" if r.get("progress_at_death_mean") is not None else "—"
        trs.append(
            f"<tr><td>{r['level']}</td>"
            f"<td class='{solved_cls}'>{r['solved_by']}/{r['seeds']}</td>"
            f"<td>{fw}</td><td>{bg}</td><td>{rate}</td><td>{wr}</td>"
            f"<td>{r.get('completion_rate_mean', 0):.0%}</td><td>{mct}</td>"
            f"<td>{pad}</td><td>{r.get('deaths_per_run_mean', 0)}</td>"
            f"<td>{r.get('death_cluster_entropy_mean', 0):.2f}</td>"
            f"<td>{r.get('coin_collection_rate_mean', 1):.0%}</td>"
            f"<td>{r.get('novice_expert_gap_mean', 0.0):+.0%}</td>"
            f"<td>{_cause_name(r['dominant_cause'])} ({r['dominant_cause_frac']:.0%})</td>"
            f"<td>{r['stuck_frac_mean']:.0%}</td>"
            f"<td>{fmt_hms(r.get('train_time_s') or 0)}</td></tr>")
    tip = lambda k: f'class="tip" data-tip="{TIPS[k]}"'  # noqa: E731
    return f"""
    <div class="ovwrap"><table>
      <thead><tr><th>Level</th><th {tip("Solved")}>Solved</th><th {tip("First win")}>First win</th>
        <th {tip("Best gen")}>Best gen</th><th {tip("Improvement")}>Rate/gen</th>
        <th {tip("Win rate")}>Win rate ±CI</th><th {tip("Completion")}>Completion</th>
        <th {tip("Mean win time")}>Mean time</th><th {tip("Progress at death")}>Progress@death</th>
        <th {tip("Deaths per gen")}>Deaths/gen</th><th {tip("Death spread")}>Death spread</th>
        <th {tip("Coin rate")}>Coin rate</th><th {tip("Skill gap")}>Skill gap</th>
        <th {tip("Dominant cause")}>Dominant cause</th><th {tip("Stuck rate")}>Stuck</th>
        <th {tip("Train time")}>Train time</th></tr></thead>
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
    return f"""
      <div class="ovwrap"><table>
        <thead><tr><th>Level</th><th class="tip" data-tip="{TIPS['Status']}">Status ⓘ</th>{ths}</tr></thead>
        <tbody>{''.join(trs)}</tbody>
      </table></div>
    <div class="ovnote">Status = best across personas. Select a persona tab for the full
    per-level breakdown — click a card for its window with routes, death maps, causes, and curves.</div>"""


CONFIG_COLORS = ["#ef4444", "#4a9eff", "#eab308", "#a855f7", "#22c55e", "#f97316", "#14b8a6", "#ec4899"]
PERSONA_DASH = {"experienced": [], "novice": [7, 5], "speedrunner": [2, 4]}


def _tag_label(tag: str | None, data: dict) -> str:
    if not tag:
        return "legacy"
    sensors = (data.get("ga_config") or {}).get("sensors") or ("grid" if tag.endswith("_grid") else "rays")
    return f"pop {data.get('pop_size', '?')} · {data.get('gens_budget', '?')} gens · {sensors}"


def _game_chart(game: str, by_tag: dict, idx: int) -> str:
    """One chart per game: win rate per level, one series per (config, persona).
    Colour = config, line style = persona. Radar with 3+ levels, grouped bars otherwise."""
    order: list[str] = []
    for personas in by_tag.values():
        for data in personas.values():
            for r in data["levels"]:
                if r["level"] not in order:
                    order.append(r["level"])
    series, legend = [], []
    for ti, (tag, personas) in enumerate(by_tag.items()):
        color = CONFIG_COLORS[ti % len(CONFIG_COLORS)]
        for pname, data in personas.items():
            rows = {r["level"]: r for r in data["levels"]}
            series.append({"name": f"{tag or 'legacy'} · {pname}", "tag": tag or "legacy",
                           "persona": pname, "color": color, "dash": PERSONA_DASH.get(pname, [4, 3]),
                           "vals": [round((rows.get(l) or {}).get("win_rate_mean", 0.0), 3) for l in order]})
            legend.append(f'<button class="lchip" type="button" data-series="{len(series) - 1}" '
                          f'style="--c:{color}"><i class="{pname}"></i>{tag or "legacy"} '
                          f'<small>{pname}</small></button>')
    payload = json.dumps({"axes": [str(v) for v in order], "series": series})
    kind = "radar" if len(order) >= 3 else "bars"
    how = ("Each spoke is a level; distance from centre = win rate (rings at 25 / 50 / 75 / 100%). "
           "A bigger shape = an easier game for that config; dents point at the hard levels."
           if kind == "radar" else
           "One group per level, one bar per config × persona (win rate). Radar needs 3+ levels.")
    return f"""
    <div class="viz chartbox"><div class="vt">Balance {kind} — win rate per level, every config and persona
      <span class="faint">· colour = config · line = persona (solid experienced, dashed novice, dotted speedrunner)
      · click a chip to hide/show · the selected config is highlighted</span></div>
      <div class="chartwrap"><canvas class="chart" width="720" height="520" data-kind="{kind}"
        data-chart='{payload}' data-section="{idx}"></canvas>
        <div class="legend chips">{''.join(legend)}</div></div>
      <div class="caption">{how}</div></div>"""


SENSORS = ["ray fwd", "ray fwd-up 30°", "ray fwd-up 60°", "ray fwd-down 30°", "ray fwd-down 60°",
           "ray back", "enemy distance (fwd corridor)", "pit ahead", "grounded", "vx / 300",
           "vy / 600", "can jump", "q-blocks within 5 tiles / 5", "bias (1.0)"]
_BRAIN_SVG = ("<svg viewBox='0 0 24 24' width='16' height='16' fill='none' stroke='currentColor' "
              "stroke-width='1.7' stroke-linecap='round'><circle cx='4' cy='6' r='2'/><circle cx='4' cy='18' r='2'/>"
              "<circle cx='12' cy='12' r='2.2'/><circle cx='20' cy='7' r='2'/><circle cx='20' cy='17' r='2'/>"
              "<path d='M6 7l4 4M6 17l4-4M14 11l4-3M14 13l4 3'/></svg>")


BODY = ["grounded", "vx / 300", "vy / 600", "can jump", "bias (1.0)"]


def _net_svg(mode: str = "rays") -> str:
    """Inline diagram of the evolved net. rays: 14 labelled sensors → hidden → 3 outputs.
    grid: an 11×11×3 tile window + 5 body scalars → hidden → 3 outputs."""
    from .net import N_HIDDEN, N_OUTPUTS
    from .sensors import GRID_CH, GRID_N, N_BODY
    W, H = 600, 392
    xi, xh, xo = 215, 390, 525
    yh = lambda j: 22 + j * (H - 44) / (N_HIDDEN - 1)
    yo = lambda k: H / 2 + (k - 1) * 60
    out = [f'<svg class="net" viewBox="0 0 {W} {H}" role="img" aria-label="network diagram">']
    if mode == "grid":
        # input side: a mini tile window (3 channels stacked) + the body scalars
        gx, gy, cell = 40, 24, 11
        for ch, col in enumerate(("#4a9eff", "#eab308", "#ef4444")):
            off = ch * 7
            out.append(f'<rect x="{gx + off}" y="{gy + off}" width="{GRID_N * cell}" height="{GRID_N * cell}" '
                       f'rx="3" fill="#0c0c0e" stroke="{col}" stroke-opacity=".8"/>')
        for r in range(GRID_N):
            for c in range(GRID_N):
                solid = r >= 8 or (c >= 9 and r >= 5)
                fill = "#fff" if (r, c) == (5, 5) else ("#2c2c30" if solid else "#121215")
                out.append(f'<rect x="{gx + 14 + c * cell + 1}" y="{gy + 14 + r * cell + 1}" width="{cell - 2}" '
                           f'height="{cell - 2}" rx="1" fill="{fill}"/>')
        gcy = gy + 14 + GRID_N * cell / 2
        out.append(f'<text x="{gx}" y="{gy + 14 + GRID_N * cell + 26}" class="lbl">{GRID_CH} × {GRID_N} × {GRID_N} tiles'
                   f'</text><text x="{gx}" y="{gy + 14 + GRID_N * cell + 42}" class="lbl">'
                   f'<tspan fill="#4a9eff">solid</tspan> · <tspan fill="#eab308">collectible</tspan> · '
                   f'<tspan fill="#ef4444">hazard</tspan></text>')
        out.append('<g stroke="#4a9eff" stroke-opacity=".16" stroke-width="1.2">')
        out += [f'<line x1="{gx + 14 + GRID_N * cell}" y1="{gcy:.0f}" x2="{xh}" y2="{yh(j):.0f}"/>'
                for j in range(N_HIDDEN)]
        out.append('</g>')
        yb = lambda i: H - 118 + i * 22
        out.append('<g stroke="#4a9eff" stroke-opacity=".13" stroke-width="1">')
        out += [f'<line x1="{xi}" y1="{yb(i):.0f}" x2="{xh}" y2="{yh(j):.0f}"/>'
                for i in range(N_BODY) for j in range(N_HIDDEN)]
        out.append('</g>')
        for i, name in enumerate(BODY):
            out.append(f'<text x="{xi - 12}" y="{yb(i) + 4:.0f}" text-anchor="end" class="lbl">{name}</text>'
                       f'<circle cx="{xi}" cy="{yb(i):.0f}" r="5" fill="#0b1f33" stroke="#4a9eff" stroke-width="1.5"/>')
        in_cap = f"{GRID_CH * GRID_N * GRID_N} grid + {N_BODY} body"
    else:
        yi = lambda i: 22 + i * (H - 44) / (len(SENSORS) - 1)
        out.append('<g stroke="#4a9eff" stroke-opacity=".13" stroke-width="1">')
        out += [f'<line x1="{xi}" y1="{yi(i):.0f}" x2="{xh}" y2="{yh(j):.0f}"/>'
                for i in range(len(SENSORS)) for j in range(N_HIDDEN)]
        out.append('</g>')
        for i, name in enumerate(SENSORS):
            out.append(f'<text x="{xi - 12}" y="{yi(i) + 4:.0f}" text-anchor="end" class="lbl">{name}</text>'
                       f'<circle cx="{xi}" cy="{yi(i):.0f}" r="5" fill="#0b1f33" stroke="#4a9eff" stroke-width="1.5"/>')
        in_cap = f"{len(SENSORS)} ray sensors"
    out.append('<g stroke="#ef4444" stroke-opacity=".22" stroke-width="1">')
    out += [f'<line x1="{xh}" y1="{yh(j):.0f}" x2="{xo}" y2="{yo(k):.0f}"/>'
            for j in range(N_HIDDEN) for k in range(N_OUTPUTS)]
    out.append('</g>')
    for j in range(N_HIDDEN):
        out.append(f'<circle cx="{xh}" cy="{yh(j):.0f}" r="5.5" fill="#1a1a1f" stroke="#9a9aa0" stroke-width="1.5"/>')
    for k, (name, col) in enumerate((("left", "#eab308"), ("right", "#22c55e"), ("jump", "#ef4444"))):
        out.append(f'<circle cx="{xo}" cy="{yo(k):.0f}" r="7" fill="{col}22" stroke="{col}" stroke-width="2"/>'
                   f'<text x="{xo + 14}" y="{yo(k) + 4:.0f}" class="lbl out" fill="{col}">{name}</text>')
    out.append(f'<text x="{xi if mode != "grid" else 110}" y="{H - 2}" text-anchor="middle" class="cap">{in_cap}</text>'
               f'<text x="{xh}" y="{H - 2}" text-anchor="middle" class="cap">{N_HIDDEN} tanh</text>'
               f'<text x="{xo}" y="{H - 2}" text-anchor="middle" class="cap">{N_OUTPUTS} sigmoid</text></svg>')
    return "".join(out)


GA_DOC = {  # label, plain-language subtitle, formatter
    "sensors": ("Sensors", "what the agent sees (exteroception)", lambda v: str(v)),
    "pop_size": ("Population", "genomes evaluated per generation", lambda v: f"{v}"),
    "elite": ("Elite", "best genomes copied unchanged", lambda v: f"{v}"),
    "tournament_k": ("Tournament", "random genomes per pick, fittest wins", lambda v: f"k = {v}"),
    "crossover_rate": ("Crossover", "children bred from two parents (uniform mask)", lambda v: f"{v:.0%}"),
    "mutation_rate": ("Mutation rate", "share of weights nudged per child", lambda v: f"{v:.0%}"),
    "mutation_sigma": ("Mutation σ", "gaussian nudge size", lambda v: f"{v:g}"),
    "init_sigma": ("Init σ", "N(0, σ) starting weights", lambda v: f"{v:g}"),
    "anneal_factor": ("Anneal", "× mutation after a level's first win", lambda v: f"×{v:g}"),
    "max_frames": ("Episode cap", "frames per attempt", lambda v: f"{v / 60:.0f}s"),
    "stuck_frames": ("Stall kill", "frames without progress → STUCK", lambda v: f"{v / 60:.0f}s"),
    "advance_wins": ("Advance at", "wins in one gen → next curriculum level", lambda v: f"{v} wins"),
    "win_bonus": ("Win bonus", "added to fitness on reaching the goal", lambda v: f"+{v:,.0f}"),
}


def _brain_dialog(sec_id: str, game: str, personas: dict[str, dict]) -> str:
    """GA + network hyperparameters behind one config, read from the probes' saved config."""
    from .evolution import GAConfig
    from .net import NeuralNet
    from .personas import PERSONAS
    from .sensors import GRID_CH, GRID_N, RAY_DIRS, RAY_MAX_DIST, RAY_STEP, sensor_dim
    data = next(iter(personas.values()))
    ga = data.get("ga_config")
    note = "" if ga else ("<div class='caption' style='margin:8px 0 0'>Legacy sweep: no config was saved "
                          "with these probes — showing the current GAConfig defaults.</div>")
    ga = dict(vars(GAConfig()) | (ga or {}))
    mode = ga.get("sensors") or ("grid" if (data.get("tag") or "").endswith("_grid") else "rays")
    ga["sensors"] = mode
    n_params = NeuralNet(sensor_dim(mode)).n_params
    sensing = (f"{GRID_N}×{GRID_N} tile window centred on the agent, {GRID_CH} channels "
               f"(solid · collectible · hazard) + body state — the Mario-AI-competition-style view"
               if mode == "grid" else
               f"{len(RAY_DIRS)} rays ({RAY_MAX_DIST:.0f} px reach, {RAY_STEP:.0f} px steps) + enemy corridor "
               f"+ pit probe + q-block count + body state")
    tiles = "".join(
        f'<div class="stat"><div class="lbl">{lbl}</div><div class="val">{fmt(ga[k])}</div>'
        f'<div class="sub">{sub}</div></div>' for k, (lbl, sub, fmt) in GA_DOC.items() if k in ga)
    steps = [("evaluate", f"{ga['pop_size']} genomes play the level"),
             ("elitism", f"top {ga['elite']} survive as-is"),
             ("select", f"tournament of {ga['tournament_k']}"),
             ("crossover", f"{ga['crossover_rate']:.0%} of children, uniform mask"),
             ("mutate", f"{ga['mutation_rate']:.0%} of weights · σ {ga['mutation_sigma']:g}"),
             ("anneal", f"×{ga['anneal_factor']:g} after first win")]
    pipe = '<span class="arrow">→</span>'.join(
        f'<span class="step"><b>{n}</b><small>{d}</small></span>' for n, d in steps)
    chips = "".join(
        f'<span class="chip" style="--c:{col}"><b>{p.name}</b> '
        f'{"sprint" if p.sprint else "walk"} · senses every {p.sensor_period} frame{"s" if p.sensor_period > 1 else ""}'
        f'{" · +" + format(p.time_rate, "g") + " fitness / s left" if p.time_rate else ""}</span>'
        for (n, p), col in ((kv, {"experienced": "#ef4444", "novice": "#4a9eff",
                                  "speedrunner": "#eab308"}.get(kv[0], "#a855f7"))
                            for kv in PERSONAS.items()) if n in personas)
    return f"""
    <dialog class="detail brain" id="brain_{sec_id}">
      <div class="dhead"><div class="lvlname"><span>{_BRAIN_SVG} Neuroevolution hyperparameters · {game}
        · {data.get('tag') or 'legacy'} · <em class="mode {mode}">{mode}</em></span></div>
        <button class="close" type="button" aria-label="close">✕</button></div>
      {note}
      <div class="brainrow">
        <div class="viz netbox"><div class="vt">The brain — {n_params:,} weights, no gradients
          <span class="faint">· {mode} sensing · inputs in [−1, 1] · outputs fire above 0.5 · left/right conflict → argmax</span></div>
          {_net_svg(mode)}
          <div class="caption"><b>Sensing:</b> {sensing}. <b>Fitness:</b> furthest x reached,
          +{ga['win_bonus']:,.0f} on a win (+ seconds left × persona time rate).</div></div>
        <div class="viz"><div class="vt">Genetic algorithm (GAConfig)</div>
          <div class="stats gastats">{tiles}</div></div>
      </div>
      <div class="viz"><div class="vt">One generation</div><div class="pipe">{pipe}</div></div>
      <div class="viz"><div class="vt">Personas in this config</div><div class="chips">{chips}</div></div>
    </dialog>"""


def _game_section(game: str, by_tag: dict, idx: int, th: dict) -> str:
    n_levels = len({r["level"] for p in by_tag.values() for d in p.values() for r in d["levels"]})
    total = sum(_train_time(d) for p in by_tag.values() for d in p.values())
    btns, views = [], []
    for ti, (tag, personas) in enumerate(by_tag.items()):
        data = next(iter(personas.values()))
        cid = f"c{idx}_{ti}"
        sid = _section_id(game, tag)
        color = CONFIG_COLORS[ti % len(CONFIG_COLORS)]
        ttime = sum(_train_time(d) for d in personas.values())
        btns.append(f'<button class="cfgbtn{" active" if ti == 0 else ""}" type="button" data-view="{cid}" '
                    f'data-tag="{tag or "legacy"}" style="--c:{color}"><b>{tag or "legacy"}</b>'
                    f'<small>{_tag_label(tag, data)} · seeds {len(data["seeds"])} · '
                    f'{", ".join(personas)} · {fmt_hms(ttime)}</small></button>')
        tabs = [f'<button class="tab t-overview active" data-view="{cid}_ov" type="button">Overview</button>']
        pviews = [f'<div class="view active" id="{cid}_ov">{_overview(personas)}</div>']
        for pi, (pname, pdata) in enumerate(personas.items()):
            vid = f"{cid}_{pi}"
            tabs.append(f'<button class="tab" data-view="{vid}" type="button">{pname}</button>')
            cell_map = pdata.get("cells", {})
            cards = "".join(_card(game, pname, r, cell_map.get(r["level"], []), th, tag)
                            for r in pdata["levels"])
            pviews.append(f"""
            <div class="view" id="{vid}">
              <div class="lvlgrid">{cards}</div>
              <div class="tbltitle">Full metric table — {pname}</div>
              {_metric_table(pdata["levels"])}
            </div>""")
        views.append(f"""
        <div class="cfgview{" active" if ti == 0 else ""}" id="{cid}">
          <div class="toolbar">
            <div class="tabs">{''.join(tabs)}</div>
            <button class="brainbtn tip" type="button" data-dialog="brain_{sid}"
              data-tip="GA + network hyperparameters used for this config">{_BRAIN_SVG} hyperparameters</button>
          </div>
          {''.join(pviews)}
          {_brain_dialog(sid, game, personas)}
        </div>""")
    return f"""
  <section class="game" id="g_{game}">
    <div class="gamehead">
      <button class="chev" type="button" aria-label="collapse section">▾</button>
      <span class="gametag">{_game_icon(game)}{game}</span>
      <span class="gamemeta">{len(by_tag)} config{'s' if len(by_tag) != 1 else ''} · {n_levels} levels
        · total train time {fmt_hms(total)}</span>
    </div>
    <div class="body">
      {_game_chart(game, by_tag, idx)}
      <div class="cfgbar"><span class="cfglbl">Config</span>{''.join(btns)}</div>
      {''.join(views)}
    </div>
  </section>"""


def _train_time(data: dict) -> float:
    if data.get("train_time_s") is not None:
        return float(data["train_time_s"])
    if data.get("elapsed_s"):  # legacy JSON: wall time of the last sweep only
        return float(data["elapsed_s"])
    return sum(c.get("train_time_s") or 0 for cells in data.get("cells", {}).values() for c in cells)


def _section_id(game: str, tag: str | None) -> str:
    return f"{game}_{tag}" if tag else game


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
    const scope = tab.closest('.cfgview') || tab.closest('section.game');
    scope.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    scope.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    tab.classList.add('active');
    scope.querySelector('#' + tab.dataset.view).classList.add('active');
  });
}
// config selector: one config's tables/cards at a time; the chart highlights it
for (const b of document.querySelectorAll('.cfgbtn')){
  b.addEventListener('click', () => {
    const section = b.closest('section.game');
    section.querySelectorAll('.cfgbtn').forEach(x => x.classList.remove('active'));
    section.querySelectorAll('.cfgview').forEach(v => v.classList.remove('active'));
    b.classList.add('active');
    section.querySelector('#' + b.dataset.view).classList.add('active');
    const cv = section.querySelector('canvas.chart');
    if (cv) drawChart(cv, 1);
  });
}
// cards open a mini window (native <dialog>): Esc, ✕, or a backdrop click closes it
for (const head of document.querySelectorAll('.cardhead')){
  head.addEventListener('click', () => head.closest('.lvlcard').querySelector('dialog').showModal());
}
for (const b of document.querySelectorAll('.brainbtn')){
  b.addEventListener('click', () => document.getElementById(b.dataset.dialog).showModal());
}
for (const dlg of document.querySelectorAll('dialog')){
  dlg.querySelector('.close').addEventListener('click', () => dlg.close());
  dlg.addEventListener('click', e => { if (e.target === dlg) dlg.close(); });
}
// collapsible sections + collapse/expand all
const sections = [...document.querySelectorAll('section.game')];
const toggleAll = document.getElementById('toggleAll');
const syncToggle = () => {
  if (!toggleAll) return;
  const allClosed = sections.every(s => s.classList.contains('collapsed'));
  toggleAll.textContent = allClosed ? 'Expand all' : 'Collapse all';
};
for (const s of sections){
  s.querySelector('.chev').addEventListener('click', () => { s.classList.toggle('collapsed'); syncToggle(); });
}
if (toggleAll) toggleAll.addEventListener('click', () => {
  const close = !sections.every(s => s.classList.contains('collapsed'));
  sections.forEach(s => s.classList.toggle('collapsed', close));
  syncToggle();
});
// a nav link to a collapsed section expands it
for (const a of document.querySelectorAll('nav a[href^="report.html#"]')){
  a.addEventListener('click', () => {
    const s = document.getElementById(a.getAttribute('href').split('#')[1]);
    if (s) { s.classList.remove('collapsed'); syncToggle(); }
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
  const colors = JSON.parse(cv.dataset.colors || 'null') || ['#ef4444', '#4a9eff', '#eab308', '#a855f7'];
  seeds.forEach((c, si) => {
    for (const [j, w] of [[1, 1], [0, 2]]){
      ctx.strokeStyle = j ? colors[si % colors.length] + '44' : colors[si % colors.length];
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
if (location.protocol === 'file:') document.documentElement.classList.add('file');  // no /watch endpoint

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

// balance chart — one per game: win rate per level, a series per config × persona.
// colour = config, dash = persona; legend chips toggle series; the selected config is highlighted.
const HIDDEN = new WeakMap();
function chartState(cv){
  const d = JSON.parse(cv.dataset.chart || 'null');
  if (!d || !d.axes.length) return null;
  const section = cv.closest('section.game');
  const focus = section?.querySelector('.cfgbtn.active')?.dataset.tag;
  const hidden = HIDDEN.get(cv) || new Set();
  return {d, focus, hidden};
}
function drawChart(cv, t){
  const st = chartState(cv);
  if (!st) return;
  const {d, focus, hidden} = st;
  const ctx = cv.getContext('2d');
  const W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);
  ctx.font = '400 12px "IBM Plex Mono", monospace';
  const alpha = s => (focus && s.tag !== focus) ? 0.28 : 1;
  const visible = d.series.map((s, i) => !hidden.has(i));
  if (cv.dataset.kind === 'bars'){
    const L = 48, R = 12, T = 16, B = 34, n = d.axes.length;
    const plotW = W - L - R, plotH = H - T - B;
    ctx.strokeStyle = '#222226'; ctx.fillStyle = '#77777d'; ctx.lineWidth = 1;
    for (let q = 0; q <= 4; q++){
      const y = T + plotH - plotH * q / 4;
      ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(W - R, y); ctx.stroke();
      ctx.fillText((q * 25) + '%', 6, y + 4);
    }
    const shown = d.series.map((s, i) => i).filter(i => visible[i]);
    const groupW = plotW / n, barW = Math.max(3, (groupW * 0.72) / Math.max(1, shown.length));
    d.axes.forEach((ax, k) => {
      const gx = L + k * groupW + groupW * 0.14;
      shown.forEach((si, bi) => {
        const s = d.series[si], v = Math.min(1, s.vals[k] || 0) * t, h = v * plotH;
        ctx.globalAlpha = alpha(s);
        ctx.fillStyle = s.color;
        ctx.fillRect(gx + bi * barW, T + plotH - h, barW - 1.5, h);
        if (s.dash.length){  // persona hatch on top of the bar
          ctx.strokeStyle = '#070708'; ctx.lineWidth = 1; ctx.setLineDash(s.dash);
          ctx.beginPath(); ctx.moveTo(gx + bi * barW + barW / 2, T + plotH - h); ctx.lineTo(gx + bi * barW + barW / 2, T + plotH); ctx.stroke();
          ctx.setLineDash([]);
        }
        ctx.globalAlpha = 1;
      });
      ctx.fillStyle = '#9a9aa0'; ctx.textAlign = 'center';
      ctx.fillText(ax, L + k * groupW + groupW / 2, H - 12);
      ctx.textAlign = 'left';
    });
    return;
  }
  const cx = W / 2, cy = H / 2 + 6, R = Math.min(W, H) / 2 - 60, n = d.axes.length;
  const ang = i => -Math.PI / 2 + i * 2 * Math.PI / n;
  for (const frac of [0.25, 0.5, 0.75, 1]){
    ctx.strokeStyle = frac === 0.5 ? '#33333a' : '#222226'; ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i <= n; i++){
      const a = ang(i % n), r = R * frac;
      i ? ctx.lineTo(cx + r * Math.cos(a), cy + r * Math.sin(a)) : ctx.moveTo(cx + r * Math.cos(a), cy + r * Math.sin(a));
    }
    ctx.stroke();
  }
  ctx.fillStyle = '#606066'; ctx.fillText('50%', cx + 4, cy - R * 0.5 - 3);
  for (let i = 0; i < n; i++){
    const a = ang(i);
    ctx.strokeStyle = '#222226';
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + R * Math.cos(a), cy + R * Math.sin(a)); ctx.stroke();
    ctx.fillStyle = '#9a9aa0';
    ctx.textAlign = Math.abs(Math.cos(a)) < 0.3 ? 'center' : (Math.cos(a) > 0 ? 'left' : 'right');
    ctx.fillText(d.axes[i], cx + (R + 16) * Math.cos(a), cy + (R + 16) * Math.sin(a) + 4);
  }
  ctx.textAlign = 'left';
  // draw dimmed series first so the highlighted config sits on top
  const order = d.series.map((s, i) => i).filter(i => visible[i]).sort((a, b) => alpha(d.series[a]) - alpha(d.series[b]));
  for (const si of order){
    const s = d.series[si];
    ctx.globalAlpha = alpha(s);
    ctx.setLineDash(s.dash);
    ctx.beginPath();
    for (let i = 0; i <= n; i++){
      const a = ang(i % n), v = Math.min(1, s.vals[i % n] || 0) * t;
      i ? ctx.lineTo(cx + R * v * Math.cos(a), cy + R * v * Math.sin(a)) : ctx.moveTo(cx + R * v * Math.cos(a), cy + R * v * Math.sin(a));
    }
    ctx.closePath();
    ctx.strokeStyle = s.color; ctx.lineWidth = alpha(s) < 1 ? 1.5 : 2.2; ctx.stroke();
    ctx.setLineDash([]);
    if (alpha(s) === 1){ ctx.fillStyle = s.color + '1a'; ctx.fill(); }
    for (let i = 0; i < n; i++){
      const a = ang(i), v = Math.min(1, s.vals[i] || 0) * t;
      ctx.fillStyle = s.color;
      ctx.beginPath(); ctx.arc(cx + R * v * Math.cos(a), cy + R * v * Math.sin(a), 2.6, 0, 7); ctx.fill();
    }
    ctx.globalAlpha = 1;
  }
}
for (const cv of document.querySelectorAll('canvas.chart')){
  const box = cv.closest('.chartwrap');
  for (const chip of box.querySelectorAll('.lchip')){
    chip.addEventListener('click', () => {
      const set = HIDDEN.get(cv) || new Set(), i = +chip.dataset.series;
      set.has(i) ? set.delete(i) : set.add(i);
      HIDDEN.set(cv, set); chip.classList.toggle('off', set.has(i));
      drawChart(cv, 1);
    });
  }
  if (REDUCED){ drawChart(cv, 1); continue; }
  const t0 = performance.now();
  const tick = now => {
    const t = Math.min(1, (now - t0) / 700);
    drawChart(cv, 1 - Math.pow(1 - t, 3));  // ease-out cubic
    if (t < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

// agent routes over level geometry; redrawn to fit when the map goes full screen
function drawRoutes(cv, maxW, maxH){
  const d = JSON.parse(cv.dataset.routes || 'null');
  if (!d || !d.grid.length) return;
  const cols = Math.max(...d.grid.map(r => r.length));
  const rows = d.grid.length;
  // fit the box, but never below 7px per tile (wide levels scroll instead of shrinking)
  // 7..22 px per tile: wide levels scroll instead of shrinking, tiny levels don't balloon
  const scale = Math.min(22 / d.ts, Math.max(Math.min(maxW / (cols * d.ts), maxH / (rows * d.ts)), 7 / d.ts));
  cv.width = Math.round(cols * d.ts * scale);
  cv.height = Math.max(90, Math.round(rows * d.ts * scale));
  const ctx = cv.getContext('2d');
  ctx.fillStyle = '#0c0c0e'; ctx.fillRect(0, 0, cv.width, cv.height);
  const cell = d.ts * scale;
  const BLOCK = {'#': '#26262c', '=': '#3a3a42', '?': '#b45309', '^': '#7f1d1d', 'G': '#14532d',
                 'S': '#2563eb', 'H': '#7c3aed', 'D': '#475569'};
  const MARK = {'E': '#ef4444', 'C': '#eab308', 'P': '#22c55e'};  // entities drawn as markers
  const marks = [];
  d.grid.forEach((row, r) => {
    for (let c = 0; c < row.length; c++){
      const ch = row[c];
      if (ch === ' ') continue;
      if (MARK[ch]) { marks.push([c, r, ch]); continue; }
      ctx.fillStyle = BLOCK[ch] || '#26262c';
      if (ch === '=') ctx.fillRect(c * cell, r * cell, cell + 0.5, Math.max(1, cell * 0.35));
      else ctx.fillRect(c * cell, r * cell, cell + 0.5, cell + 0.5);
    }
  });
  for (const [c, r, ch] of marks){  // enemies: ring + dot, coins: dot, start: dot — always ≥ 2.5px
    const x = (c + 0.5) * cell, y = (r + 0.5) * cell, rad = Math.max(2.5, cell * 0.45);
    ctx.fillStyle = MARK[ch]; ctx.beginPath(); ctx.arc(x, y, ch === 'C' ? rad * 0.6 : rad, 0, 7); ctx.fill();
    if (ch === 'E'){ ctx.strokeStyle = '#fca5a5'; ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(x, y, rad + 1.2, 0, 7); ctx.stroke(); }
  }
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
  const lw = Math.max(1, scale * d.ts / 7);
  for (const r of d.routes) if (!r.w) draw(r, '#ef4444', lw, 0.30);
  for (const r of d.routes) if (r.w) draw(r, '#22c55e', lw * 1.6, 0.85);
}
for (const cv of document.querySelectorAll('canvas.routes')) drawRoutes(cv, 1100, Infinity);
for (const b of document.querySelectorAll('.fsbtn')){
  b.addEventListener('click', () => {
    if (document.fullscreenElement) document.exitFullscreen();
    else b.closest('.viz').requestFullscreen();
  });
}
document.addEventListener('fullscreenchange', () => {
  const fs = document.fullscreenElement;
  for (const cv of document.querySelectorAll('canvas.routes')){
    if (fs && fs.contains(cv)) drawRoutes(cv, fs.clientWidth - 48, fs.clientHeight - 150);
    else if (cv.width !== 1100 || !fs) drawRoutes(cv, 1100, Infinity);
  }
  document.querySelectorAll('.fsbtn').forEach(x => { x.textContent = fs ? '⛶ exit full screen' : '⛶ full screen'; });
});
for (const b of document.querySelectorAll('.copybtn')){
  b.addEventListener('click', async () => {
    const code = b.previousElementSibling;
    try { await navigator.clipboard.writeText(code.textContent); b.textContent = 'copied'; }
    catch { const r = document.createRange(); r.selectNodeContents(code);
            getSelection().removeAllRanges(); getSelection().addRange(r); b.textContent = 'select + ⌘C'; }
    setTimeout(() => { b.textContent = 'copy'; }, 1500);
  });
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
    <b>Status</b>: <span class="pill pill-ok">solved</span> every seed won at least once ·
    <span class="pill pill-warn">n/N seeds</span> some did · <span class="pill pill-bad">unsolved</span>
    none did. Click a card for its mini window; hover any metric label or column header for what it
    means; formulas and episode statuses are on the <b>Instructions</b> page in the top bar.
  </div>"""


def _nav(games: list, has_runs: bool, logo: str | None, page: str = "report") -> str:
    img = f'<img src="data:image/png;base64,{logo}" alt="PEAK">' if logo else ""
    links = "" if page != "report" else "".join(
        f'<a href="report.html#g_{g}">{_game_icon(g)}{g}</a>' for g in games)
    if has_runs and page == "report":
        links += '<a href="report.html#g_runs">runs</a>'
    if page != "report":
        links += '<a class="doc" href="report.html">← Back to command</a>'
    if page != "ablation":
        links += '<a class="doc abl" href="ablation.html">Sensor ablation</a>'
    if page != "instructions":
        links += '<a class="doc" href="instructions.html">Instructions</a>'
    if page == "report":
        links += '<button class="navbtn" id="toggleAll" type="button">Collapse all</button>'
    return f"""
  <nav>{img}<span class="brand">PEAK <em>BALANCE COMMAND</em></span>
    <div class="links">{links}</div>
  </nav>"""


def _instructions_page(games: list[str], logo: str | None) -> str:
    tips = "".join(f"<li><b>{k}</b> — {v}</li>" for k, v in TIPS.items())
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PEAK Balance Command — Instructions</title><style>{CSS}</style></head><body>
{_nav(games, False, logo, page="instructions")}
<main class="doc-body">
  <h2>What this is</h2>
  <p>PEAK evolves populations of tiny neural networks (291 parameters) that play your levels
  thousands of times, then reports what happened. Nothing here blocks anything — the tool
  measures, you decide.</p>

  <h2>Running a sweep</h2>
  <ul>
    <li><code>python menu.py</code> → <b>15 Full Sweep</b>: pick games and personas; probes run
    in parallel across CPU cores and land in <code>runs/balance/</code>.</li>
    <li><b>16 Sensor Ablation</b> runs the same selection twice — raycasts and the tile grid — and
    compares them on the <b>Sensor ablation</b> page.</li>
    <li>Only <b>enabled</b> levels are probed — toggle them with <b>10 Toggle Levels</b>.</li>
    <li>Each level window has a <b>▶ Watch replay</b> button: it starts the best genome's replay
    and opens the live dashboard. It works when the page is served by <b>menu 12</b>
    (<code>python -m code.neuro.report --serve --open</code>); as a plain file, copy the command shown.</li>
    <li>Regenerate this page any time: <code>python -m code.neuro.report --open</code> — it
    re-reads every probe under <code>runs/probes/</code>, so it works without a finished sweep.</li>
    <li>Each distinct config (persona · population · generation budget) gets its own section
    and its own probe folders; re-running the same config on a level replaces that level only.</li>
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
    <li>Probe it (menu 15) and read the card: win rate, first-win generation, where deaths cluster.</li>
    <li>Change one thing — double or halve a parameter — and probe again.</li>
    <li>An unsolved level whose best progress is identical across seeds is usually broken
    geometry (unreachable goal), not difficulty.</li>
  </ul>

  <h2>Episode statuses</h2>
  <p>Every episode (one genome playing the level once) ends in exactly one status:</p>
  <ul>
    <li><b>WON</b> — reached the goal. Fitness = max x + win bonus (+ time left × persona rate).</li>
    <li><b>DEAD</b> — killed by an enemy, pit, saw, spike, or the timer; the engine's death cause
    becomes the failure cause (Enemy, Pit, Saw, Spike, OOB, Timeout…).</li>
    <li><b>STUCK</b> — no forward progress for <code>stuck_frames</code> (5 s) or the engine's stall
    watchdog fired; counted as the <b>Stall</b> cause.</li>
  </ul>
  <p>Level status rolls these up per seed: <b>solved</b> = every seed had ≥1 WON episode within the
  budget, <b>partial</b> = some seeds, <b>unsolved</b> = none. The Overview shows the best status across
  personas.</p>

  <h2>How the numbers are computed</h2>
  <ul>
    <li><b>First win</b> — the generation of the first WON episode, mean ± 95% CI over the seeds that
    solved the level (t-distribution for n &lt; 9).</li>
    <li><b>Best gen</b> — the generation whose best genome set the all-time fitness record, mean ± CI.</li>
    <li><b>Win rate</b> — WON episodes ÷ all episodes in the 10 generations after the first win
    (a probe stops 10 generations after its first win, or at the budget). Levels that were never won
    score 0.</li>
    <li><b>Completion</b> — WON ÷ all episodes of the whole probe, early chaos included.</li>
    <li><b>Mean win time ± std</b> — in-game seconds (frames ÷ 60) of WON episodes: mean, and the
    standard deviation of win times averaged across seeds.</li>
    <li><b>Improvement</b> — least-squares slope of the per-generation best x, from gen 1 to the
    generation the peak was first reached, divided by the level length (share of level per gen).</li>
    <li><b>Progress at death</b> — mean of end x ÷ level length over DEAD + STUCK episodes.</li>
    <li><b>Deaths per gen</b> — (DEAD + STUCK episodes) ÷ generations; one attempt per population member.</li>
    <li><b>Death spread</b> — entropy of the death-position histogram (10 bins), normalized to 0..1.</li>
    <li><b>Dominant cause</b> — most frequent failure cause over DEAD + STUCK episodes, with its share.</li>
    <li><b>Stuck rate</b> — STUCK episodes ÷ all episodes.</li>
    <li><b>Skill gap</b> — win rate of the last third of generations minus the first third.</li>
    <li><b>Coin rate</b> — mean over episodes of coins collected ÷ coins in the level (1 if none).</li>
    <li><b>Strategies / Dominant path</b> — winning routes clustered by shape; cluster count and the
    share on the biggest cluster (needs ≥2 logged wins).</li>
    <li><b>Train time</b> — wall-clock seconds the trainer spent per generation, summed.</li>
    <li><b>B1 / B2 / B3</b> — each metric compared with the target ± warning bands in
    <code>code/stats/MarioThresholds.yaml</code>.</li>
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


MODE_COLOR = {"rays": "#ef4444", "grid": "#4a9eff"}


def _mode_svgs() -> dict[str, str]:
    """Agent's-eye diagrams: a ray fan vs the 11x11 tile window, both centred on the agent."""
    import math
    rays = []
    for ang in (0, -30, -60, 30, 60, 180):
        a = math.radians(ang)
        x2, y2 = 75 + 62 * math.cos(a), 75 + 62 * math.sin(a)
        rays.append(f"<line x1='75' y1='75' x2='{x2:.1f}' y2='{y2:.1f}' stroke='{MODE_COLOR['rays']}' "
                    f"stroke-width='2' opacity='.9'/><circle cx='{x2:.1f}' cy='{y2:.1f}' r='3' fill='#fff'/>")
    ray_svg = (f"<svg viewBox='0 0 150 150'><rect width='150' height='150' rx='8' fill='#0c0c0e'/>"
               f"<rect x='0' y='100' width='150' height='50' fill='#1b1b1f'/>"
               f"<rect x='118' y='60' width='32' height='40' fill='#1b1b1f'/>"
               f"<rect x='88' y='92' width='6' height='8' fill='{MODE_COLOR['rays']}' opacity='.5'/>"
               f"{''.join(rays)}<rect x='69' y='63' width='12' height='24' rx='2' fill='#fff'/></svg>")
    cells = []
    for r in range(11):
        for c in range(11):
            solid = r >= 8 or (c >= 9 and r >= 5)
            fill = "#2c2c30" if solid else "#0c0c0e"
            if r == 5 and c == 5:
                fill = "#fff"
            cells.append(f"<rect x='{3 + c * 13.1:.1f}' y='{3 + r * 13.1:.1f}' width='12' height='12' "
                         f"rx='1.5' fill='{fill}'/>")
    grid_svg = (f"<svg viewBox='0 0 150 150'><rect width='150' height='150' rx='8' fill='#0c0c0e'/>"
                f"{''.join(cells)}<rect x='1.5' y='1.5' width='147' height='147' rx='7' fill='none' "
                f"stroke='{MODE_COLOR['grid']}' stroke-width='2' opacity='.9'/></svg>")
    return {"rays": ray_svg, "grid": grid_svg}


def _ablation_pairs(games: dict) -> dict[tuple[str, str, str], dict[str, dict]]:
    """Group report JSONs into (game, persona, budget-tag) -> {mode: data}."""
    pairs: dict[tuple[str, str, str], dict[str, dict]] = {}
    for (game, tag), personas in games.items():
        base, _, mode = (tag or "").partition("_")
        if not base:
            continue  # legacy untagged reports pre-date sensor modes
        for persona, data in personas.items():
            pairs.setdefault((game, persona, base), {})[mode or "rays"] = data
    return pairs


def _ablation_section(game: str, persona: str, base: str, by_mode: dict[str, dict], idx: int) -> str:
    modes = [m for m in MODE_COLOR if m in by_mode]
    rows = {m: {r["level"]: r for r in by_mode[m]["levels"]} for m in modes}
    order: list[str] = []
    for m in modes:
        for r in by_mode[m]["levels"]:
            if r["level"] not in order:
                order.append(r["level"])
    meta = by_mode[modes[0]]
    n = len(order)

    def fw(r):
        return f"gen {r['first_win_mean']}±{r['first_win_ci']}" if r["first_win_mean"] is not None else "never"

    def mean(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    stats = []
    for m in modes:
        rs = list(rows[m].values())
        solved = sum(1 for r in rs if r["solved_by"])
        mfw = mean([r["first_win_mean"] for r in rs])
        c = MODE_COLOR[m]
        stats.append(
            f'<div class="stat" style="border-top:2px solid {c}"><div class="lbl">{m} · levels solved</div>'
            f'<div class="val">{solved}<span style="color:var(--faint)">/{n}</span></div></div>'
            f'<div class="stat" style="border-top:2px solid {c}"><div class="lbl">{m} · mean first win</div>'
            f'<div class="val">{"gen %.1f" % mfw if mfw is not None else "never"}</div></div>'
            f'<div class="stat" style="border-top:2px solid {c}"><div class="lbl">{m} · mean win rate</div>'
            f'<div class="val">{mean([r["win_rate_mean"] for r in rs]) or 0:.0%}</div></div>'
            f'<div class="stat" style="border-top:2px solid {c}"><div class="lbl">{m} · probe train time</div>'
            f'<div class="val">{fmt_hms(_train_time(by_mode[m]))}</div></div>')

    dual = []
    for lvl in order:
        dual.append(f'<div class="lv">{lvl}</div>')
        for m in modes:
            r = rows[m].get(lvl)
            if r is None:
                dual.append(f'<div class="bar" style="--c:{MODE_COLOR[m]}"><em style="color:var(--faint)">not probed</em></div>')
                continue
            dual.append(f'<div class="bar" style="--c:{MODE_COLOR[m]}"><i class="mini-i" style="width:{r["win_rate_mean"] * 100:.1f}%"></i>'
                        f'<em>{r["win_rate_mean"]:.0%} · {fw(r)} · {r["solved_by"]}/{r["seeds"]} seeds</em></div>')
        if len(modes) == 1:
            dual.append('<div></div>')
        dual.append('<div class="gap"></div>')

    trs = []
    for lvl in order:
        tds = [f"<td>{lvl}</td>"]
        for m in modes:
            r = rows[m].get(lvl)
            cls = "mr" if m == "rays" else "mg"
            if r is None:
                tds.append(f"<td class='{cls} wr-na' colspan='4'>not probed</td>")
                continue
            tds.append(f"<td class='{cls}'>{r['solved_by']}/{r['seeds']}</td><td>{fw(r)}</td>"
                       f"<td><span class='{_wr_cls(r)}'>{r['win_rate_mean']:.0%}</span> ±{r['win_rate_ci']:.0%}</td>"
                       f"<td>{_cause_name(r['dominant_cause'])} {r['dominant_cause_frac']:.0%}</td>")
        if len(modes) == 2:
            a, b = rows["rays"].get(lvl), rows["grid"].get(lvl)
            if a and b:
                d = b["win_rate_mean"] - a["win_rate_mean"]
                cls = "delta-g" if d > 0.005 else "delta-r" if d < -0.005 else "delta-0"
                fa, fb = a["first_win_mean"], b["first_win_mean"]
                dg = ("—" if fa is None and fb is None else "grid only" if fa is None else "rays only" if fb is None
                      else f"{fb - fa:+.1f} gens")
                tds.append(f"<td class='{cls}'>{d:+.0%}</td><td>{dg}</td>")
            else:
                tds.append("<td class='wr-na'>—</td><td class='wr-na'>—</td>")
        trs.append("<tr>" + "".join(tds) + "</tr>")
    mode_ths = "".join(f"<th class='{'mr' if m == 'rays' else 'mg'}' colspan='4' style='color:{MODE_COLOR[m]}'>{m}</th>"
                       for m in modes)
    sub_ths = "".join(f"<th class='{'mr' if m == 'rays' else 'mg'}'>solved</th><th>first win</th><th>win rate</th><th>cause</th>"
                      for m in modes)
    delta_th = "<th colspan='2'>grid − rays</th>" if len(modes) == 2 else ""
    delta_sub = "<th>Δ win rate</th><th>Δ first win</th>" if len(modes) == 2 else ""
    shades = {"rays": ["#ef4444", "#fb923c", "#b91c1c", "#fca5a5"],
              "grid": ["#4a9eff", "#22d3ee", "#1d4ed8", "#93c5fd"]}
    curve_boxes = []
    for lvl in order:
        series, colors, legend = [], [], []
        for m in modes:
            cells = [c for c in by_mode[m].get("cells", {}).get(lvl, []) if len(c.get("curve", [])) > 1]
            if not cells:
                continue
            series += [c["curve"] for c in cells]
            colors += [shades[m][i % len(shades[m])] for i in range(len(cells))]
            legend.append(f'<span style="color:{MODE_COLOR[m]}">■ {m}</span> × {len(cells)} seeds')
        if not series:
            continue
        curve_boxes.append(
            f'<div class="viz"><div class="vt">{lvl} <span class="faint">— {" vs ".join(legend)}</span></div>'
            f'<canvas class="curve" width="560" height="220" data-curves=\'{json.dumps(series)}\' '
            f'data-colors=\'{json.dumps(colors)}\'></canvas></div>')
    curves_html = "" if not curve_boxes else (
        f'<div class="tbltitle" style="margin-top:14px">Learning curves — best fitness per generation, '
        f'<span style="color:{MODE_COLOR["rays"]}">reds</span> = ray seeds, '
        f'<span style="color:{MODE_COLOR["grid"]}">blues</span> = grid seeds</div>'
        f'<div class="curvegrid">{"".join(curve_boxes)}</div>'
        f'<div class="caption">Solid = best genome, faint = population average. The dashed green line is the '
        f'win bonus: a curve crossing it is a seed that solved the level. Steeper = faster learning.</div>')
    pending = "" if len(modes) == 2 else (
        f'<div class="ovnote">Only <b>{modes[0]}</b> probed so far for this config — run the other mode '
        f'(menu 16, or <code>balance --game {game} --persona {persona} --sensors '
        f'{"grid" if modes[0] == "rays" else "rays"}</code>) to fill the comparison.</div>')
    return f"""
  <section class="game" id="abl_{idx}">
    <div class="gamehead">
      <button class="chev" type="button" aria-label="collapse section">▾</button>
      <span class="gametag">{_game_icon(game)}{game}</span>
      <span class="gamemeta">{persona} · {base} · seeds {meta['seeds']} · pop {meta.get('pop_size', '?')} ×
        {meta.get('gens_budget', '?')} gens/probe</span>
    </div>
    <div class="stats ablstats">{''.join(stats)}</div>
    <div class="viz"><div class="vt">Win rate per level — <span style="color:{MODE_COLOR['rays']}">■ rays</span>
      vs <span style="color:{MODE_COLOR['grid']}">■ grid</span>
      <span class="faint">&nbsp;(bar = win rate over the {WIN_WINDOW} gens after the first win; label adds generations-to-first-win ± CI)</span></div>
      <div class="dual">{''.join(dual)}</div></div>
    {curves_html}
    {pending}
    <div class="tbltitle">Full comparison table</div>
    <div class="ovwrap"><table>
      <thead><tr class="modes"><th></th>{mode_ths}{delta_th}</tr>
        <tr><th>Level</th>{sub_ths}{delta_sub}</tr></thead>
      <tbody>{''.join(trs)}</tbody></table></div>
  </section>"""


def _ablation_page(games: dict, logo: str | None, stamp: str) -> str:
    from .net import NeuralNet
    from .sensors import sensor_dim
    pairs = _ablation_pairs(games)
    wins = {"rays": 0, "grid": 0, "tie": 0}
    for by_mode in pairs.values():
        if len(by_mode) == 2:
            a = {r["level"]: r for r in by_mode["rays"]["levels"]}
            for r in by_mode["grid"]["levels"]:
                if r["level"] in a:
                    d = r["win_rate_mean"] - a[r["level"]]["win_rate_mean"]
                    wins["grid" if d > 0.005 else "rays" if d < -0.005 else "tie"] += 1
    total = sum(wins.values())
    score = (f'<div class="ablscore"><span style="--c:{MODE_COLOR["rays"]}"><b>{wins["rays"]}</b> rays</span>'
             f'<span class="sep">·</span><span style="--c:{MODE_COLOR["grid"]}"><b>{wins["grid"]}</b> grid</span>'
             f'<span class="sep">·</span><span style="--c:var(--dim)"><b>{wins["tie"]}</b> ties</span>'
             f'<span class="sep">— level × persona × budget cells won on win rate, {total} paired</span></div>'
             if total else "")
    svgs = _mode_svgs()
    hero = ""
    for m, blurb in (("rays", "Six raycasts (forward, ±30°, ±60°, back) plus a forward enemy corridor, a pit probe "
                              "and a question-block count. Sees far along a few lines; blind between them."),
                     ("grid", "The cores' tile window sized to 11×11 around the agent, three channels: solid / "
                              "collectible / hazard. Sees everything nearby at tile resolution; no Dijkstra oracle.")):
        nin = sensor_dim(m)
        hero += (f'<div class="ablmode" style="--c:{MODE_COLOR[m]}">{svgs[m]}<div><h3>{m}</h3><p>{blurb}</p>'
                 f'<div class="chips"><span class="chip" style="--c:{MODE_COLOR[m]}"><b>{nin}</b> inputs</span>'
                 f'<span class="chip" style="--c:{MODE_COLOR[m]}"><b>{NeuralNet(nin).n_params:,}</b> weights</span>'
                 f'<span class="chip" style="--c:{MODE_COLOR[m]}"><b>5</b> shared body scalars</span></div></div></div>')
    sections = "".join(_ablation_section(g, p, b, by_mode, i)
                       for i, ((g, p, b), by_mode) in enumerate(sorted(pairs.items())))
    if not sections:
        sections = ('<p style="color:var(--dim)">No sensor probes yet — run <b>16 Sensor Ablation</b> from '
                    '<code>python menu.py</code> (or <code>balance --sensors rays</code> and '
                    '<code>--sensors grid</code>) and rebuild this page.</p>')
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PEAK Balance Command — Sensor Ablation</title><style>{CSS}</style></head><body>
{_nav(list(games), False, logo, page="ablation")}
<main>
  <div style="font:400 .78rem var(--mono);color:var(--faint);margin-bottom:16px">
    sensor ablation · same GA, same seeds, same levels — only what the agent sees changes · generated {stamp}</div>
  <div class="ablhero">{hero}</div>
  {score}
  {sections}
  <footer>PEAK ENGINE · code/neuro/report.py · data: runs/balance/report_*_p*g*[_grid].json</footer>
</main><script>{JS}</script></body></html>"""


def build(balance_dir: str) -> tuple[str, str, str]:
    rebuild(balance_dir)  # probe dirs are the source of truth; JSONs are regenerated from them
    flat: dict[tuple[str, str | None], dict[str, dict]] = {}  # (game, tag) -> persona -> data
    for f in sorted(glob.glob(os.path.join(balance_dir, "report_*.json"))):
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        persona = data.get("persona") or "experienced"
        flat.setdefault((data["game"], data.get("tag")), {})[persona] = data
    games: dict[str, dict] = {}  # game -> tag -> persona -> data (tagged configs first, legacy last)
    for (g, t), personas in sorted(flat.items(), key=lambda kv: (kv[0][0], kv[0][1] is None, kv[0][1] or "")):
        games.setdefault(g, {})[t] = personas
    th = _load_thresholds()
    logo = _logo_b64()
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    sections = [_game_section(g, by_tag, i, th) for i, (g, by_tag) in enumerate(games.items())]
    runs_html = _runs_section()
    total = sum(_train_time(d) for by_tag in games.values() for p in by_tag.values() for d in p.values())
    body = "".join(sections) + runs_html if sections or runs_html else \
        "<p style='color:var(--dim)'>No balance data yet — run a Full Sweep (menu 15) or Sensor Ablation (menu 16) first.</p>"
    report = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'><rect width='16' height='16' rx='3' fill='%23070708'/><path d='M3 12 L8 4 L13 12' stroke='%23ef4444' stroke-width='2' fill='none'/></svg>">
<title>PEAK Balance Command</title><style>{CSS}</style></head><body>
{_nav(list(games), bool(runs_html), logo)}
<main>
  <div style="font:400 .78rem var(--mono);color:var(--faint);margin-bottom:16px">
    multi-seed neuroevolution probes · generated {stamp} · total probe train time {fmt_hms(total)}</div>
  {body}
  {GLOSSARY}
  <footer>PEAK ENGINE · code/neuro/report.py · data: runs/balance + runs/probes + runs/*</footer>
</main><script>{JS}</script></body></html>"""
    return report, _instructions_page(list(games), logo), _ablation_page(flat, logo, stamp)


def serve(balance_dir: str, port: int, open_browser: bool) -> None:
    """Serve the command center locally so its ▶ Watch buttons can launch replays.
    GET /watch?game=G&npz=runs/... starts `trainer --replay` (one at a time) and forwards
    the tab to the trainer's dashboard. Binds 127.0.0.1 only; replays only files under runs/."""
    import subprocess
    import sys
    import urllib.parse
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

    from .adapters import _ADAPTERS
    state: dict = {"proc": None}

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw) -> None:
            super().__init__(*a, directory=balance_dir, **kw)

        def log_message(self, *a) -> None:  # quiet
            pass

        def do_GET(self) -> None:
            u = urllib.parse.urlparse(self.path)
            if u.path != "/watch":
                return super().do_GET()
            q = urllib.parse.parse_qs(u.query)
            game, npz = q.get("game", [""])[0], q.get("npz", [""])[0]
            ok = (game in _ADAPTERS and npz.startswith("runs/") and ".." not in npz
                  and npz.endswith("best.npz") and os.path.exists(npz))
            if not ok:
                return self.send_error(400, "bad replay request")
            proc = state["proc"]
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
            state["proc"] = subprocess.Popen(
                [sys.executable, "-m", "code.neuro.trainer", "--game", game, "--replay", npz])
            print(f"replay: {game} {npz}", flush=True)
            dash = f"http://127.0.0.1:8000/{game}/index.html"
            body = (f"<!doctype html><meta charset='utf-8'><meta http-equiv='refresh' content='3;url={dash}'>"
                    f"<body style='background:#070708;color:#e8e8ea;font:16px IBM Plex Mono,monospace;"
                    f"padding:40px'>Launching replay of <b>{npz}</b>… the dashboard opens in a moment "
                    f"(<a style='color:#4a9eff' href='{dash}'>open it now</a>).</body>").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    import signal
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # reach the finally: no orphaned replay
    url = f"http://127.0.0.1:{port}/report.html"
    print(f"command center: {url}  (Ctrl+C to close)", flush=True)
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
    ap = argparse.ArgumentParser(description="Generate the balance command center")
    ap.add_argument("--dir", default=BALANCE_DIR)
    ap.add_argument("--open", action="store_true", help="open in the default browser")
    ap.add_argument("--serve", action="store_true",
                    help="serve the page locally so ▶ Watch buttons can launch replays")
    ap.add_argument("--port", type=int, default=8001)
    args = ap.parse_args()

    report, instructions, ablation = build(args.dir)
    os.makedirs(args.dir, exist_ok=True)
    out = os.path.join(args.dir, "report.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(report)
    with open(os.path.join(args.dir, "instructions.html"), "w", encoding="utf-8") as f:
        f.write(instructions)
    with open(os.path.join(args.dir, "ablation.html"), "w", encoding="utf-8") as f:
        f.write(ablation)
    print(f"wrote {out} (+ instructions.html, ablation.html)")
    if args.serve:
        serve(args.dir, args.port, args.open)
    elif args.open:
        webbrowser.open("file:///" + os.path.abspath(out).replace(os.sep, "/"))


if __name__ == "__main__":
    main()
