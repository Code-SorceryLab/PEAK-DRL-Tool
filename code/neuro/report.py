"""The PEAK Command Center's sub-pages, one per game and tab (code/neuro/command_center.py
gathers them into one site): Balance, GA sweep, Sensor ablation and How it works.

Balance holds per-game sections with an Overview (all personas side by side) and persona tabs
whose level cards expand in place: metrics with hover explanations, B1/B2/B3 verdicts, death maps,
cause bars, learning curves, and agent routes drawn over the level geometry. How it works also
lists every training run found under runs/. Everything is read from the committed report JSONs
(load_data); nothing here rebuilds them.
"""
from __future__ import annotations

import ast
import base64
import csv
import glob
import html as _html
import io
import json
import math
import os
import re
from datetime import datetime

GAMES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "games")
LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "docs", "img", "PEAK_LOGO.png")
THRESHOLDS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stats", "MarioThresholds.yaml")
from .adapters import INDEXED_GAMES, OWN_LEVEL_FILES, list_levels
from .balance import BALANCE_DIR, PAPER_SEEDS, PROBES_ROOT, WIN_WINDOW, fmt_ci, fmt_hms, mean_ci
from .gasweep import AXES, AXIS_DOC, best_config, paired_delta, parse_sweep_tag, sweep_point, tag_base_sig
# Level glyphs -> canvas categories (see code/games/levels/common/ASCII_TILEMAP.md)
GLYPHS = {"#": "#%([/\\])Un", "=": "=", "?": "?<>FL", "^": "^*O", "E": "EkKMBX",
          "C": "C", "G": "G", "S": "S", "H": "H", "P": "P", "D": "D"}
GLYPH_CAT = {ch: cat for cat, chars in GLYPHS.items() for ch in chars}
# Bomberman hides things under bricks: the exit and every power-up are solid until bombed.
GLYPH_CAT_BOMBERMAN = {**GLYPH_CAT, "@": "?", "C": "?", "F": "?", "S": "?"}
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
    "Deaths per win": "Failed attempts (deaths and stalls) per win over the whole probe; with no wins, every one counts.",
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
    "B2": "Punishment severity: deaths per win and death spread vs their target bands.",
    "B3": "Strategy diversity: route-cluster count and dominant-path share vs their target bands.",
}


def _tip_key(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")


def _tip_attrs(label: str) -> str:
    """A TIPS bubble by reference: its text sits once per page (TIP_CSS + the hidden TIP_DESC),
    not in each of the thousands of tiles that show it."""
    k = _tip_key(label)
    return f'data-k="{k}" tabindex="0" aria-describedby="tip-{k}"'


# each bubble's text, once: shown by CSS on hover / focus / tap, read out through aria-describedby
TIP_CSS = "".join(f'.tip[data-k="{_tip_key(k)}"]:is(:hover,:focus-visible,.open)::after{{content:{json.dumps(v, ensure_ascii=False)}}}'
                  for k, v in TIPS.items())
TIP_DESC = ('<div hidden>' + "".join(f'<span id="tip-{_tip_key(k)}">{_html.escape(v, quote=False)}</span>'
                                     for k, v in TIPS.items()) + "</div>")


def _json(obj) -> str:
    return json.dumps(obj, separators=(",", ":"))


_TPL: dict[str, str] = {}                                # <template id> -> markup a dialog stamps in when it opens
_STORE: dict[str, dict] = {"grids": {}, "curves": {}}   # page-level JSON that canvases reference by key


def _slot(tid: str, html: str) -> str:
    """Shared markup (a level's config, the net diagram) emitted once per page: an empty slot that
    the page JS fills from <template id=tid> when its dialog opens."""
    if not html:
        return ""
    _TPL[tid] = html
    return f'<div data-tpl="{tid}"></div>'


def _page_data() -> str:
    """The templates and JSON stores the page's slots and canvases point at; resets them for the next page."""
    out = "".join(f'<template id="{k}">{v}</template>' for k, v in _TPL.items())
    for k, v in _STORE.items():
        if v:
            out += f'<script type="application/json" id="{k}">' + _json(v).replace("<", "\\u003c") + "</script>"
    _TPL.clear()
    for v in _STORE.values():
        v.clear()
    return out


# ── data loading ─────────────────────────────────────────────────────────────

def _load_thresholds() -> dict:
    try:
        import yaml
        with open(THRESHOLDS_PATH, encoding="utf-8") as f:
            return (yaml.safe_load(f) or {}).get("metrics", {})
    except Exception:
        return {}


def _indexed_entry(game: str, level: str) -> dict:
    """One level of an indexed game (meatboy / bomberman) as a dict.

    An entry is either a bare path or {file: ..., <per-level overrides>}; disabled levels are a
    suffix of the enabled ones, so ids stay stable and a report can still draw a retired level.
    """
    import yaml
    try:
        with open(os.path.join(GAMES_DIR, f"{game}_config.yaml"), encoding="utf-8") as f:
            g = yaml.safe_load(f) or {}
        lv = list(g.get("levels") or []) + list(g.get("disabled_levels") or [])
        i = int(level)
        if not 0 <= i < len(lv):
            return {}
        entry = lv[i]
        return dict(entry) if isinstance(entry, dict) else {"file": entry}
    except Exception:
        return {}


def _own_entry(game: str, level: str) -> dict:
    """One level of a game that keeps named levels in its own file (megaman)."""
    import yaml
    try:
        with open(os.path.join(GAMES_DIR, OWN_LEVEL_FILES[game]), encoding="utf-8") as f:
            g = yaml.safe_load(f) or {}
        return dict((g.get("levels") or {}).get(level) or (g.get("disabled_levels") or {}).get(level) or {})
    except Exception:
        return {}


def _level_file(game: str, level: str) -> str | None:
    import yaml
    try:
        if game in OWN_LEVEL_FILES:
            f = _own_entry(game, level).get("file")
            return os.path.join(GAMES_DIR, "levels", f) if f else None
        if game in INDEXED_GAMES:
            f = _indexed_entry(game, level).get("file")
            return os.path.join(GAMES_DIR, "levels", f) if f else None
        with open(os.path.join(GAMES_DIR, "game_config.yaml"), encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        section = {"mario": cfg, "sonic": cfg.get("sonic", {})}.get(game, {})
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
    cat = GLYPH_CAT_BOMBERMAN if game == "bomberman" else GLYPH_CAT
    with open(path, encoding="utf-8") as f:
        for line in f.read().splitlines():
            rows.append("".join(cat.get(ch, " ") for ch in line).rstrip())
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
        if game in OWN_LEVEL_FILES:
            with open(os.path.join(GAMES_DIR, f"{game}_config.yaml"), encoding="utf-8") as f:
                g = yaml.safe_load(f) or {}
            entry = _own_entry(game, level)
        elif game in INDEXED_GAMES:
            with open(os.path.join(GAMES_DIR, f"{game}_config.yaml"), encoding="utf-8") as f:
                g = yaml.safe_load(f) or {}
            entry = _indexed_entry(game, level)
        else:
            with open(os.path.join(GAMES_DIR, "game_config.yaml"), encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            g = {"mario": cfg, "sonic": cfg.get("sonic", {})}.get(game, {})
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
    elif game == "megaman":
        sections.append(("Player (NES px, px/frame)", _flatten(g.get("player"))))
        sections.append(("Buster", _flatten(g.get("buster"))))
    elif game == "bomberman":
        sections.append(("Player", _flatten(g.get("player"))))
        sections.append(("Bombs", _flatten(g.get("bomb"))))
        sections.append(("Enemies", {f"{v.get('name', k)}": f"{v.get('speed')} px/s · chase {v.get('chase')}"
                                     + (" · walks through bricks" if v.get("passes_bricks") else "")
                                     for k, v in (g.get("enemies") or {}).items()}))
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
                    max_routes: int = 22, seeds=PAPER_SEEDS) -> tuple[list[dict], dict | None]:
    """Route traces for one (game, persona, level), merged over `seeds`, and the fastest win.

    A probe dir gives its episodes.csv rows when it has one, else the compact routes.json sampled
    on the cluster ({won, lost, best, best_time_s}: routes already ≤60 integer points).
    Returns (routes, best): routes = [{"p": [[x, y], ...], "w": 1 | 0}], ≤10 wins then losses;
    best = {"p": route, "t": seconds or None, "seed": str} or None.
    """
    won_raw, lost_raw, fastest = [], [], []   # fastest: (seconds or None, raw route, seed)
    sub = (game, persona, tag) if tag else (game,)  # legacy reports pre-date the tagged layout
    for d in sorted(glob.glob(os.path.join(PROBES_ROOT, *sub, f"{level}_*".replace(" ", "_")))):
        seed = os.path.basename(d).rsplit("_", 1)[-1]
        if seed not in {str(s) for s in seeds}:
            continue
        if os.path.exists(os.path.join(d, "episodes.csv")):
            try:
                with open(os.path.join(d, "episodes.csv"), encoding="utf-8", newline="") as f:
                    for row in csv.DictReader(f):
                        if str(row.get("world")) != str(level) or row.get("persona") != persona:
                            continue
                        raw = row.get("route") or "[]"
                        if (row.get("cause_of_death") or "").lower() != "success":
                            lost_raw.append(raw)
                            continue
                        won_raw.append(raw)
                        try:
                            fastest.append((float(row["elapsed_time"]), raw, seed))
                        except (KeyError, TypeError, ValueError):   # older CSVs have no elapsed_time
                            pass
            except OSError:
                continue
        elif os.path.exists(os.path.join(d, "routes.json")):
            try:
                with open(os.path.join(d, "routes.json"), encoding="utf-8") as f:
                    js = json.load(f)
            except (OSError, ValueError):
                continue
            if not isinstance(js, dict) or js.get("persona", persona) != persona:
                continue
            won_raw += js["won"] if isinstance(js.get("won"), list) else []
            lost_raw += js["lost"] if isinstance(js.get("lost"), list) else []
            t = js.get("best_time_s")
            if js.get("best"):
                fastest.append((float(t) if isinstance(t, (int, float)) else None, js["best"], seed))

    def parse(raw) -> list | None:  # only the kept rows are parsed — the CSVs run to 10^5+ rows
        try:
            pts = ast.literal_eval(raw) if isinstance(raw, str) else raw
            if len(pts) < 2:
                return None
            step = max(1, len(pts) // 60)  # ≤60 points per embedded route
            return [[round(x), round(y)] for x, y in pts[::step]]
        except (ValueError, SyntaxError, TypeError, OverflowError):
            return None

    won = [p for p in map(parse, won_raw[:10]) if p]
    lost = [p for p in map(parse, lost_raw[-(max_routes - len(won)):]) if p]
    timed = [f for f in fastest if f[0] is not None and math.isfinite(f[0])]
    pick = min(timed, key=lambda f: f[0]) if timed else (fastest[0] if fastest else None)
    best = None
    if pick and (pts := parse(pick[1])):
        best = {"p": pts, "t": round(pick[0], 1) if timed else None, "seed": pick[2]}
    return [{"p": p, "w": 1} for p in won] + [{"p": p, "w": 0} for p in lost], best


def _pack(route: list) -> list[int]:
    """[[x, y], ...] -> [x0, y0, dx1, dy1, ...] for the page: flat, and the steps between points are
    short numbers (a third fewer bytes than absolute pairs). drawRoutes adds them back up."""
    out, px, py = [], 0, 0
    for x, y in route:
        out += [x - px, y - py]
        px, py = x, y
    return out


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
_BOMBERMAN_SVG = (  # white helmet, red suit (no blue on these pages), a lit bomb
    "data:image/svg+xml;utf8," + "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 28 36'>"
    "<circle cx='12' cy='11' r='9' fill='%23f8f8f8'/><rect x='5' y='10' width='14' height='5' rx='2' fill='%23f7c9bf'/>"
    "<circle cx='9.5' cy='12.5' r='1.2' fill='%23000000'/><circle cx='14.5' cy='12.5' r='1.2' fill='%23000000'/>"
    "<rect x='5' y='19' width='14' height='11' rx='4' fill='%23f83800'/>"
    "<circle cx='23' cy='27' r='5' fill='%231c1c1c'/><path d='M24 22 l2 -4' stroke='%23bcbcbc' stroke-width='1.6'/>"
    "<circle cx='26.4' cy='17.6' r='1.6' fill='%23ffc83c'/></svg>"
)
_MEATBOY_SVG = (
    "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'>"
    "<rect x='2' y='4' width='20' height='18' rx='4' fill='%23c81e1e'/>"
    "<rect x='2' y='4' width='20' height='7' rx='4' fill='%23e13030'/>"
    "<circle cx='9' cy='12' r='2.2' fill='white'/><circle cx='16' cy='12' r='2.2' fill='white'/>"
    "<circle cx='9.6' cy='12.4' r='1' fill='black'/><circle cx='16.6' cy='12.4' r='1' fill='black'/>"
    "<path d='M9 17.5 q3 2.4 6.5 0' stroke='black' stroke-width='1.3' fill='none' stroke-linecap='round'/>"
    "</svg>")

_MEGAMAN_SVG = (  # a blue helmet of our own: the one blue on these pages, so Mega Man reads at a glance
    "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 28 36'>"
    "<path d='M3 17 Q3 3 14 3 Q25 3 25 17 L25 22 L3 22 Z' fill='%230070ec'/>"
    "<path d='M11 3.4 Q14 2.6 17 3.4 L16 11 L12 11 Z' fill='%233cbcfc'/>"
    "<rect x='7' y='12' width='14' height='10' rx='3' fill='%23fcd8a8'/>"
    "<rect x='9.5' y='14' width='3' height='4' rx='1' fill='white'/><rect x='15.5' y='14' width='3' height='4' rx='1' fill='white'/>"
    "<rect x='10.6' y='15' width='1.6' height='2.6' fill='black'/><rect x='16.6' y='15' width='1.6' height='2.6' fill='black'/>"
    "<circle cx='4' cy='17' r='2.6' fill='%233cbcfc'/><circle cx='24' cy='17' r='2.6' fill='%233cbcfc'/>"
    "<rect x='6' y='23' width='16' height='10' rx='4' fill='%230070ec'/><rect x='10' y='23' width='8' height='4' rx='2' fill='%233cbcfc'/>"
    "</svg>")

_icon_cache: dict[str, str | None] = {}


def _game_icon(game: str) -> str:
    """Small inline icon per game: the mario sprite from the engine assets, our own SVGs for
    meatboy (a meat cube), bomberman and megaman (a helmet), a letter badge otherwise."""
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
    elif game == "bomberman":
        src = _BOMBERMAN_SVG
        _icon_cache[game] = src
    elif game == "megaman":
        src = _MEGAMAN_SVG
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
    out.append(("B1 challenge", cr_ok and t_ok, cr_ok or t_ok, "B1"))
    if row.get("deaths_per_win_mean") is not None:   # older reports hold only deaths per generation: no verdict
        b2 = (th.get("B2_punishment_severity") or {}).get("thresholds", {})
        d_ok = _in_band(row["deaths_per_win_mean"], b2.get("target_deaths_per_run", 2),
                        b2.get("warning_deaths_per_run", 1))
        e_ok = _in_band(row.get("death_cluster_entropy_mean"), b2.get("target_death_cluster_entropy", 0.4),
                        b2.get("warning_death_cluster_entropy", 0.1))
        out.append(("B2 punishment", d_ok and e_ok, d_ok or e_ok, "B2"))
    if strat is not None:
        b3 = (th.get("B3_strategy_diversity") or {}).get("thresholds", {})
        s_ok = _in_band(strat[0], b3.get("target_strategy_count", 2), b3.get("warning_strategy_count", 1))
        p_ok = _in_band(strat[1], b3.get("target_dominant_path_share", 0.5),
                        b3.get("warning_dominant_path_share", 0.1))
        out.append(("B3 diversity", s_ok and p_ok, s_ok or p_ok, "B3"))
    pills = []
    for label, ok, partial, tip in out:
        cls = "pill-ok" if ok else ("pill-warn" if partial else "pill-bad")
        word = "in band" if ok else ("partial" if partial else "off band")
        pills.append(f'<span class="pill {cls} tip" {_tip_attrs(tip)}>{label}: {word}</span>')
    return " ".join(pills)


def _verdict_sentence(row: dict, strat: tuple[int, float] | None) -> tuple[str, str, str]:
    """(word, colour, one plain-English sentence) a designer can act on without reading the tiles."""
    wr, solved, seeds = row["win_rate_mean"], row["solved_by"], row["seeds"]
    hist = row.get("death_hist") or []
    peak = max(range(len(hist)), key=lambda i: hist[i]) if hist and max(hist) else None
    cause = _cause_name(row["dominant_cause"])
    frac = row["dominant_cause_frac"]
    where = f"in the {peak * 10}–{peak * 10 + 10} % stretch" if peak is not None else "with no single hotspot"
    if solved == 0:
        word, color = "too hard", "var(--red)"
        pad = row.get("progress_at_death_mean")
        sent = (f"No seed ever reached the goal — the best agents die at {pad:.0%} of the level, "
                f"<b>{cause}</b> accounts for {frac:.0%} of deaths {where}. "
                "Ease that stretch or give the player a tool for it.") if pad is not None else \
               f"No seed ever reached the goal; <b>{cause}</b> accounts for {frac:.0%} of deaths {where}."
    elif solved < seeds:
        word, color = "luck-dependent", "var(--yellow)"
        wr = wr * seeds / solved   # the winners' own mean: a seed that never won has win rate 0
        sent = (f"Only {solved} of {seeds} seeds cracked it and winners then win {wr:.0%} of the time — "
                f"<b>{cause}</b> kills {frac:.0%} {where}. A level some playtesters never beat is a spike, not a curve.")
    elif wr >= 0.5:
        word, color = "learnable", "var(--green)"
        sent = (f"Every seed wins and keeps winning ({wr:.0%} after the first win). "
                f"<b>{cause}</b> is still the main killer ({frac:.0%}, {where})"
                + (f"; {strat[0]} distinct winning routes." if strat else "."))
    else:
        word, color = "inconsistent", "var(--yellow)"
        sent = (f"Every seed wins at least once, but only {wr:.0%} of the time afterwards — "
                f"<b>{cause}</b> ({frac:.0%} of deaths, {where}) keeps catching agents that already know the way. "
                "That is a precision demand, not a discovery problem.")
    return word, color, sent


# ── HTML pieces ──────────────────────────────────────────────────────────────

# The Night Cartridge look every PEAK page shares (code/neuro/web/theme.css), inlined so the pages stay standalone.
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "theme.css"), encoding="utf-8") as _f:
    THEME_CSS = _f.read()

CSS = THEME_CSS + """
  /* ── Command Center on the Night Cartridge tokens above ─────────────────── */
  :root{  /* the report's older names, kept so existing markup keeps working */
    --bg:var(--night);--panel:var(--deck);--panel2:var(--deck-2);--line2:var(--line);
    --txt:var(--ink);--faint:var(--mute);--mono:var(--ui);
    --green:var(--run);--yellow:var(--stuck);--red:var(--dead);--red-dim:var(--deck-3);
    --blue:#cfcfcf}   /* the second series (grid sensing, "bigger helps"): grey beside the red */
  *{box-sizing:border-box;margin:0}
  html{font-size:clamp(16px,1.05vw + 10px,18px);scroll-behavior:smooth}
  body{font-size:1rem;line-height:1.55;padding:0 clamp(12px,3vw,32px) clamp(12px,3vw,32px)}
  canvas.chart,canvas.curve,canvas.scatter{image-rendering:auto}   /* scaled charts, not game frames */
  .skip{position:absolute;left:-999px;top:8px;z-index:99;background:var(--ink);color:var(--night);
    padding:8px 12px;font-weight:600}
  .skip:focus{left:8px}
  main{max-width:1280px;margin:0 auto}
  .faint{color:var(--mute)}
  .sr{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}

  /* top bar: the wordmark, the page name, every page as a pixel button */
  nav{position:sticky;top:0;z-index:50;display:flex;flex-wrap:wrap;align-items:center;gap:10px 18px;
    background:var(--night);border-bottom:3px solid var(--line);
    padding:12px clamp(12px,3vw,32px);margin:0 calc(-1 * clamp(12px,3vw,32px)) 20px}
  nav .brand{display:flex;align-items:baseline;gap:14px}
  nav .brand em{font:700 18px/1 var(--pixel);font-style:normal;color:var(--dim)}
  nav .links{display:flex;flex-wrap:wrap;gap:6px;margin-left:auto}
  nav a .gicon{height:1.3em;margin:0;vertical-align:middle}

  /* pixel buttons: theme .btn, on the controls the report already has */
  nav a,.navbtn,.brainbtn,.tab,.cfgbtn,.close,.fsbtn,.copybtn,details.more summary{
    display:inline-flex;align-items:center;gap:8px;padding:5px 12px;cursor:pointer;white-space:nowrap;
    font:600 .87rem/1.2 var(--ui);color:var(--ink);text-decoration:none;
    border:3px solid transparent;border-image:var(--frame) 1 / 3px stretch;
    background:var(--deck-2);background-clip:padding-box}
  nav a:hover,.navbtn:hover,.brainbtn:hover,.tab:hover,.cfgbtn:hover,.close:hover,
  .fsbtn:hover,.copybtn:hover,details.more summary:hover{background-color:var(--deck-3)}
  nav a:active,.navbtn:active,.brainbtn:active,.tab:active,.cfgbtn:active,
  .close:active,.fsbtn:active,.copybtn:active{transform:translateY(2px)}
  nav a[aria-current],.tab.active{background-color:var(--ink);color:var(--night)}
  .tab[style*="--c"]::before{content:"";width:8px;height:8px;background:var(--c);flex:none}  /* persona colour */
  .brainbtn svg,.navbtn svg{color:var(--blue)}
  .fsbtn,.copybtn{padding:2px 9px;font-size:.78rem}
  .close{padding:3px 10px}

  /* panels: pixel-notched frames (theme .frame). Containers sit on the deck, read-outs on the night */
  .viz,.stat,.lvlcard,.gloss,details.cfg,.cfgblock,.axiscard,.dverdict,.ablmode,dialog.detail,.card{
    border:3px solid transparent;border-image:var(--frame) 1 / 3px stretch;
    background:var(--night);background-clip:padding-box}
  .gloss,.ablmode,dialog.detail,.card{background-color:var(--deck)}

  /* game sections: .frame .stripe .g-<game> in the markup */
  section.game{padding:clamp(14px,2.5vw,24px);padding-left:calc(clamp(14px,2.5vw,24px) + 4px);
    margin-bottom:26px;scroll-margin-top:84px}
  .gamehead{display:flex;flex-wrap:wrap;align-items:center;gap:12px 14px;margin-bottom:16px}
  .gametag{display:inline-flex;align-items:center;font-size:1.4rem;color:var(--ink)}
  .gametag .gicon{margin-right:10px}
  .pchip{font:600 .85rem/1.2 var(--ui);color:var(--c);border:2px solid var(--c);padding:2px 10px}
  .gamemeta{font:400 .85rem var(--ui);color:var(--dim)}
  .chev{background:none;border:0;color:var(--dim);font:400 1.1rem var(--ui);cursor:pointer;width:26px;
    transition:transform .15s}
  section.game.collapsed .chev{transform:rotate(-90deg)}
  section.game.collapsed{cursor:pointer}   /* a collapsed section opens from a click anywhere on it */
  section.game.collapsed .body,section.game.collapsed .tabs{display:none}
  section.game.collapsed .gamehead{margin-bottom:0}
  code.cmd{display:block;font:500 .82rem/1.4 var(--code);color:var(--ink);background:var(--deck);
    border:2px solid var(--line);padding:7px 10px;margin:6px 0;user-select:all;overflow-x:auto;white-space:nowrap}

  /* persona tabs, config buttons */
  .toolbar{display:flex;flex-wrap:wrap;align-items:center;gap:10px;margin:14px 0 12px}
  .tabs{display:flex;flex-wrap:wrap;gap:6px;margin-left:auto}
  .toolbar .tabs{margin-left:0}
  .toolbar .brainbtn{margin-left:auto}
  .view{display:none} .view.active{display:block}
  .cfgview{display:none} .cfgview.active{display:block}
  .cfgnow{display:inline-flex;align-items:baseline;gap:8px;font:400 .8rem var(--ui);color:var(--dim);margin-right:4px}
  .cfgnow::before{content:"";width:8px;height:8px;background:var(--c);align-self:center;flex:none}
  .cfgnow b{font:600 .9rem var(--dense);color:var(--ink)}
  .cfgbtn{align-items:baseline;text-align:left;color:var(--dim);padding:6px 12px}
  .cfgbtn::before{content:"";width:8px;height:8px;background:var(--c);align-self:center;flex:none}
  .cfgbtn b{font:600 .9rem var(--dense);color:var(--ink)}
  .cfgbtn small{font:400 .78rem var(--ui)}
  .cfgbtn.active{border-image-source:var(--frame-hi);background-color:var(--deck-3);color:var(--ink)}

  /* tables: table.data's look, numbers right-aligned in tabular figures */
  .ovwrap{overflow-x:auto;background:var(--night)}
  table{border-collapse:collapse;width:100%;font:500 .93rem/1.35 var(--dense);font-variant-numeric:tabular-nums}
  th{font:600 .83rem/1.3 var(--ui);color:var(--dim);text-align:right;padding:9px 12px;
    border-bottom:2px solid var(--line);white-space:nowrap}
  td{text-align:right;padding:8px 12px;border-bottom:1px solid var(--deck-3);white-space:nowrap;color:var(--dim)}
  th:first-child,td:first-child{text-align:left}
  td:first-child{color:var(--ink);font-weight:600}
  tr:hover td{background:var(--deck-2)}
  table.metrics th:first-child,table.metrics td:first-child{position:sticky;left:0;z-index:1;background:var(--night)}
  table.metrics tr:hover td:first-child{background:var(--deck-2)}
  table.metrics th[data-sort]{cursor:pointer;user-select:none}
  table.metrics th[data-sort]:hover{color:var(--ink)}
  table.metrics th[aria-sort="ascending"]::after{content:" ▴";color:var(--won)}
  table.metrics th[aria-sort="descending"]::after{content:" ▾";color:var(--won)}
  table.metrics tbody tr{cursor:pointer}
  table.metrics tbody tr:focus-visible td{background:var(--deck-3)}
  .wr-good{color:var(--run);font-weight:700}.wr-mid{color:var(--stuck);font-weight:700}
  .wr-bad{color:var(--dead);font-weight:700}.wr-na{color:var(--mute)}
  .ovnote{font:400 .8rem var(--ui);color:var(--mute);margin-top:8px}
  .tbltitle{font:600 .95rem var(--ui);color:var(--ink);margin:20px 0 8px}

  /* status words and verdicts */
  .pill{display:inline-flex;align-items:center;gap:4px;font:600 .75rem/1.3 var(--ui);padding:0 6px;
    border:2px solid currentColor;white-space:nowrap;vertical-align:middle}
  .pill .ico{--ico:11px;margin:0;vertical-align:0}   /* solved trophy · partial half star · unsolved skull */
  .pill-ok{color:var(--run)}.pill-warn{color:var(--stuck)}.pill-bad{color:var(--dead)}
  .verdicts{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0 2px}
  .verdict{display:inline-flex;align-items:center;gap:8px;font:600 .85rem/1.2 var(--ui);color:var(--c);
    border:2px solid var(--c);padding:3px 10px}
  .verdict::before{content:"";width:8px;height:8px;background:var(--c);flex:none}
  .vsent{font:400 .95rem/1.5 var(--ui);color:var(--ink);margin-top:10px}
  .vsent b{color:var(--c)}
  .dverdict{--c:var(--dim);display:grid;grid-template-columns:auto 1fr;gap:8px 14px;align-items:start;
    padding:12px 14px 12px 18px;margin:12px 0 0;box-shadow:inset 4px 0 0 var(--c)}
  .dverdict .verdict{align-self:center}
  .dverdict .vsent{margin:0}
  .dverdict .verdicts{grid-column:1/-1;margin:2px 0 0}

  /* info bubbles: hover, keyboard focus, or tap */
  .tip{position:relative;cursor:help}
  .tip:hover::after,.tip:focus-visible::after,.tip.open::after{content:attr(data-tip);position:absolute;left:0;
    bottom:calc(100% + 8px);z-index:60;width:min(300px,70vw);padding:8px 11px;
    border:3px solid transparent;border-image:var(--frame-hi) 1 / 3px stretch;
    background:var(--night);background-clip:padding-box;color:var(--ink);
    font:400 .8rem/1.5 var(--ui);white-space:normal;text-align:left}

  /* level cards and their windows */
  .lvlgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}
  .lvlcard{display:flex;flex-direction:column;background-color:var(--deck-2)}
  .lvlcard.untrained .cardbody{padding:10px 12px}
  .untrainedsec{margin-top:18px;padding:14px 18px}
  .lvlcard:hover,.lvlcard:has(dialog[open]){border-image-source:var(--frame-hi)}
  .lvlcard:has(.cardhead:focus-visible){border-image-source:var(--frame-hi);outline:2px solid var(--won);outline-offset:2px}
  .cardhead{display:block;width:100%;text-align:left;background:none;border:0;color:var(--ink);
    font:inherit;cursor:pointer;padding:14px 16px;flex:1}
  .cardhead:focus-visible{outline:none}
  .cardfoot{display:flex;align-items:center;gap:10px;padding:8px 16px 10px;border-top:2px solid var(--deck-3)}
  .cardfoot small{font:400 .78rem var(--ui);color:var(--dim)}
  .lvlname{font:700 18px/1.2 var(--pixel);color:var(--ink);margin-bottom:8px;display:flex;
    justify-content:space-between;align-items:center;gap:8px}
  .bignum{font:700 36px/1 var(--pixel)}
  .bignum .ico{--ico:.8em;margin-right:8px;vertical-align:0}
  .stat .lbl .ico{--ico:13px;vertical-align:-2px}
  .lvlsub{font:400 .8rem var(--ui);color:var(--dim);margin-top:6px}
  .mini{height:6px;background:var(--deck-3);margin-top:12px;overflow:hidden}
  .mini i{display:block;height:100%}
  dialog.detail{margin:auto;color:var(--ink);border-image-source:var(--frame-hi);
    width:min(1180px,94vw);max-height:92vh;padding:6px 20px 20px}
  dialog.detail::backdrop{background:rgba(0,0,0,.82)}
  .dhead{display:flex;align-items:center;gap:14px;position:sticky;top:0;z-index:2;background:var(--deck);
    padding:12px 0 10px;border-bottom:2px solid var(--line)}
  .dhead .lvlname{margin:0;flex:1;font-size:1.15rem;justify-content:flex-start;flex-wrap:wrap}
  .dhead .bignum{font-size:27px}
  .dhead em.mode{font:600 .75rem/1.3 var(--ui);font-style:normal;padding:0 6px;border:2px solid currentColor;
    vertical-align:middle}
  .dhead em.rays{color:#f87858} .dhead em.grid{color:var(--blue)}   /* MODE_COLOR */
  details.more{margin:6px 0 10px}
  details.more summary{list-style:none}
  details.more summary::-webkit-details-marker{display:none}
  details.more summary::before{content:"▸";color:var(--mute)}
  details.more[open] summary::before{content:"▾"}
  /* the level window: the route map large on the left, the read-outs beside it, history below */
  .dgrid{display:grid;grid-template-columns:minmax(0,1.65fr) minmax(300px,1fr);grid-template-areas:"main side" "curve more";
    gap:12px;margin-top:12px;align-items:start}
  .dgrid>*{min-width:0}
  .dmain,.dside,.dmore{display:flex;flex-direction:column;gap:12px}
  .dmain{grid-area:main}.dside{grid-area:side}.dcurve{grid-area:curve}.dmore{grid-area:more}
  .dgrid .viz,.dmore details.more,.dmore details.cfg{margin:0}
  .dside .stats.headline{grid-template-columns:1fr 1fr;gap:8px;margin:0}
  .dside .stats.headline .stat .val{font-size:18px;white-space:normal}
  .dmore .stats{grid-template-columns:repeat(auto-fill,minmax(150px,1fr));margin:10px 0 0}
  @media (max-width:900px){.dgrid{grid-template-columns:minmax(0,1fr);grid-template-areas:"main" "side" "curve" "more"}}

  /* read-outs: stats, charts, maps */
  .stats{display:grid;grid-template-columns:repeat(auto-fill,minmax(175px,1fr));gap:10px;margin:14px 0 16px}
  .stats.headline{grid-template-columns:repeat(auto-fit,minmax(175px,1fr));margin:12px 0 10px}
  .stats.headline .stat .val{font-size:1.3rem}
  .stat{padding:10px 12px;min-width:0}
  .stat .lbl{font:500 .83rem/1.3 var(--ui);color:var(--dim)}
  .stat .val{font:700 18px/1.25 var(--pixel);margin-top:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .stat .sub{font:400 .78rem var(--ui);color:var(--mute);margin-top:2px}
  .viz{padding:12px 14px;margin-top:10px}
  .viz .vt{display:flex;align-items:center;gap:8px;flex-wrap:wrap;font:600 .9rem/1.35 var(--ui);
    color:var(--ink);margin-bottom:8px}
  .vt .faint{font-weight:400}
  .caption{font:400 .8rem/1.5 var(--ui);color:var(--mute);margin-top:6px}
  .legend{display:flex;flex-wrap:wrap;gap:4px 12px;font:400 .8rem var(--ui);color:var(--dim);margin-top:6px}
  .legend .lg i{display:inline-block;width:9px;height:9px;margin-right:4px;vertical-align:-1px}
  .heat{display:grid;grid-template-columns:repeat(10,1fr);gap:3px}
  .heat i{height:20px;background:var(--deck-2)}
  .heatlbl{display:flex;justify-content:space-between;font:400 .78rem var(--ui);color:var(--mute);margin-top:4px}
  .causebar{display:flex;height:16px;gap:2px;overflow:hidden}
  .causekey{font:400 .82rem var(--ui);color:var(--dim);margin-top:6px}
  canvas.curve{width:100%;height:190px;display:block}
  .routewrap{overflow:auto}
  canvas.routes{display:block;margin:0 auto}
  .fsbtn{margin-left:auto}
  .viz:fullscreen{background:var(--night);border:0;border-image:none;padding:24px 24px 12px;overflow:auto;
    display:flex;flex-direction:column}
  .viz:fullscreen .routewrap{flex:1;display:flex;align-items:center;justify-content:safe center}
  .cmdrow{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin-top:4px}
  .cmdrow code.cmd{flex:1;min-width:min(260px,100%);margin:6px 0}
  .chartbox{margin-top:0}
  /* a game's chart beside its series matrix: a row per config (the config picker), a column per persona */
  .chartwrap{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:16px 22px;align-items:start}
  .chartmain,.seriesbox{min-width:0}
  .seriesbox{border-left:2px solid var(--deck-3);padding-left:20px}
  .seriesbox .caption{max-width:46ch}
  @media (max-width:1080px){.chartwrap{grid-template-columns:minmax(0,1fr)}
    .seriesbox{border-left:0;padding-left:0;border-top:2px solid var(--deck-3);padding-top:12px}}
  canvas.chart{width:100%;max-width:720px;height:auto;display:block;margin:0 auto}
  .chips{display:flex;flex-wrap:wrap;gap:8px}
  .chip{font:400 .82rem var(--ui);color:var(--dim);border:2px solid var(--c);padding:3px 10px}
  .chip b{font-family:var(--pixel);font-weight:700;color:var(--c)}
  table.series{width:auto;border-collapse:separate;border-spacing:4px 3px;margin-left:-4px}
  table.series th,table.series td{padding:0;border:0;text-align:center;vertical-align:middle;white-space:nowrap}
  table.series tr:hover td{background:none}
  table.series thead th{font:500 .74rem/1 var(--dense);color:var(--dim);padding:0 0 6px;vertical-align:bottom}
  table.series thead th .ico{display:block;margin:0 auto 6px}
  table.series thead th span{display:inline-block;writing-mode:vertical-rl;transform:rotate(180deg)}
  table.series tbody th{text-align:left}
  table.series .cfgbtn{display:grid;grid-template-columns:8px auto;gap:2px 8px;align-items:center;width:100%;
    padding:4px 10px;white-space:normal}
  table.series .cfgbtn::before{grid-row:1 / 3}
  table.series .cfgbtn small{font-size:.72rem;max-width:150px}
  table.series button:focus-visible{outline-offset:-1px}
  .lchip{display:flex;align-items:center;justify-content:center;width:36px;height:28px;padding:0;cursor:pointer;
    background:none;border:2px solid var(--deck-3);color:var(--c)}
  .lchip:hover{border-color:var(--c)}
  .lchip[aria-pressed="true"]{background:var(--deck-2)}
  .lchip.off{opacity:.35}
  .lchip svg{display:block}

  /* level config */
  details.cfg{margin-top:10px;padding:10px 14px}
  details.cfg summary{cursor:pointer;display:flex;flex-wrap:wrap;align-items:center;gap:6px;list-style:none}
  details.cfg summary::-webkit-details-marker{display:none}
  details.cfg summary::before{content:"▸";color:var(--mute);font:600 .9rem var(--ui);width:14px}
  details.cfg[open] summary::before{content:"▾"}
  .sumlbl{font:600 .9rem var(--ui);color:var(--ink);margin-right:4px}
  .cfgchip{font:500 .75rem var(--ui);color:var(--dim);border:2px solid var(--deck-3);padding:0 8px}
  .sumhint{margin-left:auto;font:400 .78rem var(--ui);color:var(--mute)}
  details.cfg[open] .sumhint{display:none}
  .cfggrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px;margin-top:12px}
  .cfgblock{padding:10px 12px;align-self:start}
  .cfgtitle{display:flex;align-items:center;gap:8px;font:600 .85rem var(--ui);color:var(--ink);
    padding-bottom:6px;border-bottom:2px solid var(--line);margin-bottom:4px}
  .cfgtitle em{font:700 .8rem var(--pixel);font-style:normal;color:var(--mute);margin-left:auto}
  .kv{display:flex;justify-content:space-between;align-items:baseline;gap:14px;padding:3px 0;
    border-bottom:1px solid var(--deck-3);font:500 .8rem var(--dense)}
  .kv:last-child{border-bottom:0}
  .kv b{font-weight:400;color:var(--mute);flex:none}
  .kv span{color:var(--ink);text-align:right;overflow-wrap:anywhere}

  /* the network window */
  .brainrow{display:grid;grid-template-columns:minmax(300px,1.15fr) minmax(280px,1fr);gap:10px;margin-top:10px}
  .brainrow>.viz{align-self:start}
  @media (max-width:900px){.brainrow{grid-template-columns:1fr}}
  svg.net{width:100%;height:auto;display:block;margin-top:4px}
  svg.net .lbl{font:500 11px var(--dense);fill:var(--dim)}
  svg.net .lbl.out{font-weight:600;font-size:12px}
  svg.net .cap{font:600 11px var(--ui);fill:var(--mute)}
  .gastats{grid-template-columns:repeat(auto-fill,minmax(130px,1fr));margin:0}
  .gastats .val{font-size:1rem}
  .pipe{display:flex;flex-wrap:wrap;align-items:center;gap:6px}
  .pipe .step{display:flex;flex-direction:column;border:2px solid var(--line);padding:6px 12px;min-width:120px}
  .pipe .step b{font:600 .88rem var(--ui);color:var(--ink)}
  .pipe .step small{font:400 .78rem var(--ui);color:var(--dim)}
  .pipe .arrow{color:var(--mute);font:600 1.1rem var(--ui)}

  /* sensor ablation page */
  .ablhero{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:22px}
  .ablhero.scored{grid-template-columns:minmax(0,1fr) minmax(180px,auto) minmax(0,1fr);align-items:stretch}
  .ablmode{display:grid;grid-template-columns:110px 1fr;gap:14px;align-items:center;padding:12px 16px 12px 20px;
    box-shadow:inset 4px 0 0 var(--c)}
  .ablmode svg{width:110px;height:110px;display:block}
  .ablmode h3{font-size:1.3rem;color:var(--c)}
  .ablmode p{font:400 .9rem/1.5 var(--ui);color:var(--dim);margin:6px 0 10px}
  /* the versus board between the two senses: paired level cells each one won */
  .vsboard{display:flex;flex-direction:column;align-items:center;justify-content:center;gap:8px;text-align:center;padding:8px 4px}
  .vsscore{display:flex;align-items:center;gap:16px}
  .vsn{display:flex;flex-direction:column;align-items:center;gap:6px;color:var(--c)}
  .vsn .num{font-size:36px}
  .vsn small{font:600 .8rem var(--ui)}
  .vsword{font:700 27px/1 var(--pixel);color:var(--ink);text-shadow:3px 3px 0 var(--peak)}
  .vsties{font:400 .85rem var(--ui);color:var(--dim)}
  .vsties .num{font-size:18px;margin-right:4px}
  .vsboard p{font:400 .75rem/1.4 var(--ui);color:var(--mute);max-width:22ch}
  /* a comparison card's face: rays | Δ and winner | grid */
  .vs{display:grid;grid-template-columns:minmax(0,1fr) auto minmax(0,1fr);gap:8px;margin-top:14px}
  .vsside{display:flex;flex-direction:column;gap:3px;min-width:0;padding:10px 12px;background:var(--night);
    border:2px solid var(--deck-3);box-shadow:inset 0 3px 0 var(--c)}
  .vsside.win{border-color:var(--c)}
  .vsmode{font:600 .8rem var(--ui);color:var(--c)}
  .vsside .num{font-size:27px;color:var(--c);margin:2px 0 4px}
  .vsside small{font:400 .78rem/1.35 var(--ui);color:var(--dim)}
  .vsmid{display:flex;flex-direction:column;align-items:center;justify-content:center;gap:3px;text-align:center}
  .vsmid .num{font-size:18px}
  .vsmid small{font:400 .72rem var(--ui);color:var(--mute)}
  .vsmid .verdict{font-size:.75rem;padding:1px 7px;margin-top:4px}
  .dual{display:grid;grid-template-columns:minmax(110px,160px) 1fr;gap:4px 14px;align-items:center}
  .dual .lv{font:600 .88rem var(--dense);color:var(--ink);grid-row:span 2;align-self:center}
  .dual .bar{position:relative;height:16px;background:var(--deck-2);overflow:hidden}
  .dual .bar i{display:block;height:100%;background:var(--c)}
  .dual .bar em{position:absolute;inset:0;font:600 .72rem/16px var(--dense);font-style:normal;color:var(--ink);
    padding-left:6px;white-space:nowrap;text-shadow:1px 1px 0 var(--night)}
  .dual .gap{height:8px;grid-column:1/-1}
  .delta-r{color:var(--dead);font-weight:700}.delta-g{color:var(--blue);font-weight:700}.delta-0{color:var(--mute)}
  th.mr,td.mr,th.mg,td.mg{border-left:2px solid var(--line)}
  thead tr.modes th{text-align:center}
  .stats.ablstats{grid-template-columns:repeat(4,1fr)}
  .curvegrid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:10px}
  .curvegrid .viz{margin-top:0}
  .curvegrid canvas.curve{height:170px}
  @media (max-width:900px){.curvegrid,.ablhero,.ablhero.scored{grid-template-columns:1fr}}
  @media (max-width:700px){.stats.ablstats{grid-template-columns:1fr 1fr}}

  /* GA sweep page */
  .gshero{display:grid;grid-template-columns:minmax(0,1.6fr) minmax(280px,1fr);gap:14px;align-items:stretch}
  .gshero .viz{margin-top:0}
  canvas.scatter{width:100%;height:auto;display:block;margin-top:4px}
  .gsrec{box-shadow:inset 4px 0 0 var(--run);padding-left:18px}
  .gsrec .recrow{display:flex;flex-wrap:wrap;gap:8px;margin:6px 0 10px}
  .recchip{display:inline-flex;align-items:baseline;gap:6px;font:400 .8rem var(--ui);color:var(--dim);
    border:2px solid var(--line);padding:4px 10px}
  .recchip b{color:var(--ink);font-weight:600}
  .recchip s{color:var(--mute)}
  .recchip em{font-style:normal;color:var(--run);font-weight:600}
  .recchip em.neg{color:var(--dead)}
  .gsaxes{display:grid;grid-template-columns:repeat(auto-fill,minmax(290px,1fr));gap:12px;margin-top:12px}
  .axiscard{padding:12px 14px 10px;display:flex;flex-direction:column}
  .axiscard h4{font:600 .95rem var(--ui);color:var(--ink);display:flex;align-items:center;gap:8px}
  .axiscard h4 small{margin-left:auto;font:500 .78rem var(--ui);color:var(--mute)}
  .axiscard p{font:400 .8rem/1.45 var(--ui);color:var(--mute);margin:4px 0 0;min-height:2.9em}
  .rail{position:relative;height:96px;margin:14px 30px 4px}
  .rail::before{content:"";position:absolute;left:-12px;right:-12px;top:43px;height:3px;background:var(--line)}
  .mk{position:absolute;top:44px;left:var(--x);transform:translate(-50%,-50%);width:0;height:0}
  .mk i{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);width:var(--s);height:var(--s);background:var(--c)}
  .mk.best i{outline:2px solid var(--won);outline-offset:2px}
  .mk.none i{background:none;border:2px dashed var(--mute)}
  .mk b{position:absolute;bottom:18px;left:50%;transform:translateX(-50%);font:600 .78rem var(--dense);
    color:var(--c);white-space:nowrap}
  .mk.none b{color:var(--mute);font-weight:400}
  .mk small{position:absolute;top:18px;left:50%;transform:translateX(-50%);font:500 .72rem/1.3 var(--dense);
    color:var(--dim);white-space:nowrap;text-align:center}
  .mk small strong{display:block;color:var(--ink);font-weight:600}
  .mk.best small strong::after{content:" ★";color:var(--won)}
  .axisnote{font:400 .75rem var(--ui);color:var(--mute);margin-top:6px}
  table.gst td.base{color:var(--stuck)}
  @media (max-width:900px){.gshero{grid-template-columns:1fr}}
  /* page heads (GA sweep, sensor ablation) and the GA sweep's hero row of five numbers */
  .pagehead{margin:6px 0 18px}
  .pagehead h2{font-size:27px}
  .pagehead p{font:400 .9rem/1.5 var(--ui);color:var(--dim);margin-top:10px;max-width:90ch}
  .kpis{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px;margin:0 0 18px}
  .kpis .stat{padding:12px 14px}
  .kpis .stat .val{font-size:27px;margin-top:8px}
  .kpis .stat.accent{box-shadow:inset 4px 0 0 var(--c);padding-left:16px}
  .kpis .stat.accent .val{color:var(--c)}
  @media (max-width:1000px){.kpis{grid-template-columns:repeat(3,minmax(0,1fr))}.kpis .stat.accent{grid-column:span 2}}
  @media (max-width:700px){.kpis{grid-template-columns:repeat(2,minmax(0,1fr))}.kpis .stat.accent{grid-column:1/-1}}
  @media (max-width:560px){.kpis .stat .val{font-size:18px}}
  /* the knob map: a row per sweep, a column per knob, each cell shaded by its winning bound's paired Δ */
  .gsov{margin-bottom:18px}
  table.knobmap{border-collapse:separate;border-spacing:3px;width:100%}
  table.knobmap th,table.knobmap td{border:0;padding:6px 7px;text-align:center;white-space:nowrap;vertical-align:middle}
  table.knobmap thead th{white-space:normal;vertical-align:bottom;font:600 .74rem/1.2 var(--ui);min-width:60px}
  table.knobmap tbody th{position:sticky;left:0;z-index:1;background:var(--deck);text-align:left;
    box-shadow:inset 3px 0 0 var(--stripe,var(--line));padding-left:12px}
  .knobmap .gsjump{display:flex;flex-direction:column;gap:2px;color:var(--ink);text-decoration:none;font:600 .85rem/1.2 var(--ui)}
  .knobmap .gsjump:hover span{text-decoration:underline;text-underline-offset:3px}
  .knobmap .gsjump small{font:500 .75rem var(--dense);color:var(--dim)}
  .knobmap .gsjump small .ico{--ico:12px;margin-right:4px}
  table.knobmap td.kbase{background:var(--night);color:var(--ink);font:600 .85rem var(--dense)}
  table.knobmap td.kbase small{display:block;color:var(--mute);font-weight:400}
  table.knobmap td.k{background-color:var(--deck-2);color:var(--ink);font:500 .82rem/1.25 var(--dense)}
  table.knobmap td.k b{display:block;font-weight:700}
  table.knobmap td.k small{display:flex;align-items:center;justify-content:center;gap:3px;font-size:.72rem}
  table.knobmap td.k .ico{--ico:11px}
  table.knobmap td.k-none{background:none;color:var(--mute)}
  table.knobmap td.k-base{color:var(--mute)}
  table.knobmap td.k-sig{box-shadow:inset 0 0 0 2px var(--won)}
  table.knobmap .verdict{font-size:.75rem;padding:1px 7px}
  .kmlegend{display:flex;flex-wrap:wrap;align-items:center;gap:6px 18px;margin-top:10px;font:400 .78rem var(--ui);color:var(--dim)}
  .kmlegend span{display:inline-flex;align-items:center;gap:6px}
  .kmlegend i:not(.ico){width:22px;height:14px;background:var(--deck-2);flex:none}
  .kmlegend i.sig{box-shadow:inset 0 0 0 2px var(--won)}
  .kmlegend b{color:var(--mute)}
  /* a sweep card's face: baseline -> composite, and the capacity verdict */
  .gsface{display:flex;flex-wrap:wrap;align-items:flex-end;gap:10px 22px;margin-top:14px}
  .gsnum{display:flex;flex-direction:column;gap:3px}
  .gsnum small{font:400 .75rem var(--ui);color:var(--mute)}
  .gsnum .num{font-size:27px;color:var(--ink)}
  .gsface .verdict{margin-left:auto;font-size:.78rem}
  /* sweep and comparison cards: a grid, and an opened card takes a full row of its own */
  .cardgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(340px,100%),1fr));gap:14px;grid-auto-flow:row dense}
  .cardgrid>section.game{margin:0;min-width:0}
  .cardgrid>section.game:not(.collapsed){grid-column:1/-1}
  .cardgrid>section.game.collapsed:hover{border-image-source:var(--frame-hi)}
  .cardgrid section.game .gamemeta{flex-basis:100%}
  .cardgrid section.game.collapsed .gamemeta{display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}

  /* how-to-read box, instructions page, footer */
  .gloss{padding:12px 18px;margin:0 0 18px;font-size:.92rem;color:var(--dim);line-height:1.7}
  .gloss b{color:var(--ink)}
  .gloss summary{cursor:pointer;font-weight:600;color:var(--ink);list-style:none;display:flex;align-items:center;gap:8px}
  .gloss summary::-webkit-details-marker{display:none}
  .gloss summary::before{content:"▸";color:var(--mute)}
  .gloss[open] summary::before{content:"▾"}
  .gloss[open] summary{margin-bottom:6px}
  footer{color:var(--mute);font:400 .8rem var(--ui);margin:22px 0 8px}
  /* instructions: a sticky contents rail beside the text */
  main.doc{display:grid;grid-template-columns:minmax(170px,220px) minmax(0,1fr);gap:clamp(20px,4vw,52px);
    max-width:1120px;align-items:start}
  .toc{position:sticky;top:88px;padding:14px 12px;max-height:calc(100vh - 110px);overflow:auto}
  .toc h2{font-size:18px;margin:0 0 10px 6px}
  .toc ul{list-style:none;padding:0;margin:0}
  .toc a{display:flex;align-items:center;gap:8px;padding:6px;color:var(--dim);text-decoration:none;font:500 .88rem/1.3 var(--ui)}
  .toc a:hover{color:var(--ink);background:var(--deck-2)}
  .toc .ico{--ico:13px}
  .doc-body{max-width:72ch;min-width:0}
  .doc-body section{scroll-margin-top:90px}
  .doc-body h2{font-size:18px;color:var(--ink);margin:34px 0 12px;display:flex;align-items:center;gap:10px}
  .doc-body section:first-child h2{margin-top:4px}
  .doc-body h2 .ico{--ico:18px;margin:0}
  .doc-body p,.doc-body ul{max-width:72ch}
  .doc-body p,.doc-body li{color:var(--dim);font-size:.95rem}
  .doc-body p + p,.doc-body ul + p,.doc-body p + ul{margin-top:10px}
  .doc-body code{color:var(--ink);background:var(--deck-2);padding:1px 6px}
  .doc-body ul{padding-left:22px;margin:8px 0}
  .doc-body li + li{margin-top:4px}
  .doc-body ul.icolist{list-style:none;padding-left:0}
  .doc-body b:not([class]){color:var(--ink)}
  .doc-body li .ico{vertical-align:-2px}
  .docfoot{max-width:1120px;margin:22px auto 8px}
  @media (max-width:900px){main.doc{grid-template-columns:minmax(0,1fr)}.toc{position:static;max-height:none}}

  /* game icons: the sprites stay pixel-crisp */
  .gicon{height:1.35em;width:auto;vertical-align:-0.28em;margin-right:8px;image-rendering:pixelated}
  .gletter{display:inline-flex;align-items:center;justify-content:center;width:1.35em;height:1.35em;
    background:var(--deck-3);color:var(--ink);font:700 .8em/1 var(--pixel)}

  @media (prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
  @media (max-width:560px){
    nav{position:static}
    nav .links{flex-wrap:nowrap;overflow-x:auto;width:100%;padding-bottom:6px}   /* one swipeable row, not five */
    nav .links>*{flex:none}
    .lvlgrid{grid-template-columns:1fr 1fr}
    .stats{grid-template-columns:1fr 1fr}
    .tabs{margin-left:0;width:100%}
    nav .links{margin-left:0}
    table.series .cfgbtn small{display:none}
  }
""" + TIP_CSS

CAUSE_COLORS = {"Enemy": "#e0442e", "Pit": "#ededed", "OOB": "#f8b800", "Spike": "#f878f8", "Bomb": "#e45c10",
                "Saw": "#f878f8", "Stall": "#fca044", "Timeout": "#cfcfcf", "?": "#a0a0a0"}  # NES; Stall = --stuck


STATUS_ICON = {"ok": "trophy", "warn": "halfstar", "bad": "skull"}   # solved / partial / unsolved


def _status_pill(kind: str, text: str) -> str:
    return f'<span class="pill pill-{kind}"><i class="ico {STATUS_ICON[kind]}"></i>{text}</span>'


def _pill(row: dict) -> str:
    if row["solved_by"] == row["seeds"]:
        return _status_pill("ok", "solved")
    if row["solved_by"]:
        return _status_pill("warn", f'{row["solved_by"]}/{row["seeds"]} seeds')
    return _status_pill("bad", "unsolved")


def _wr_cls(row: dict) -> str:
    if row["solved_by"] == 0:
        return "wr-bad"
    return "wr-good" if row["win_rate_mean"] >= 0.5 else "wr-mid"


def _wr_var(row: dict) -> str:
    return {"wr-good": "var(--green)", "wr-mid": "var(--yellow)", "wr-bad": "var(--red)"}[_wr_cls(row)]


def _cause_name(name: str) -> str:
    return "unknown" if name in ("?", "", "-") else name


def _ci(row: dict, sep: str = " ± ") -> str:
    ci = row.get("win_rate_ci")
    return fmt_ci(None if ci is None else min(ci, 1.0), ".0%", sep)


def _heat(hist: list[int]) -> str:
    mx = max(hist) or 1
    cells = "".join(
        f'<i style="background:rgba(228,0,88,{0.15 + 0.85 * n / mx:.2f})" title="{n} deaths"></i>'
        if n else "<i></i>" for n in hist)
    return (f'<div class="heat">{cells}</div>'
            f'<div class="heatlbl"><span>start</span><span>goal</span></div>')


def _causebar(causes: dict[str, int]) -> str:
    total = sum(causes.values())
    if not total:
        return '<div class="causekey">no deaths recorded</div>'
    parts, key = [], []
    for name, n in sorted(causes.items(), key=lambda kv: -kv[1]):
        color = CAUSE_COLORS.get(name, "#a0a0a0")
        parts.append(f'<i style="width:{100 * n / total:.1f}%;background:{color}" '
                     f'title="{_cause_name(name)}: {n}"></i>')
        key.append(f'<span style="color:{color}">■</span> {_cause_name(name)} {100 * n / total:.0f}%')
    return (f'<div class="causebar">{"".join(parts)}</div>'
            f'<div class="causekey">{" · ".join(key)}</div>')


def _stat(label: str, value: str, sub: str = "", icon: str = "") -> str:
    ico = f'<i class="ico {icon}"></i> ' if icon else ""
    return (f'<div class="stat"><div class="lbl tip" {_tip_attrs(label)}>{ico}{label} ⓘ</div>'
            f'<div class="val">{value}</div>'
            + (f'<div class="sub">{sub}</div>' if sub else "") + "</div>")


def _npz_meta(path: str) -> dict:
    try:
        import numpy as np
        return json.loads(str(np.load(path)["meta"]))
    except Exception:
        return {}


def _card(game: str, persona: str, row: dict, cells: list[dict], th: dict, tag: str | None) -> str:
    wr = row["win_rate_mean"]
    fw = (f"first win gen {row['first_win_mean']}" if row["first_win_mean"] is not None
          else f"best progress {row.get('progress_at_death_mean') or 0:.0%}")
    color = _wr_var(row)
    mct = (f"{row['mean_completion_time']}s" if row.get("mean_completion_time") is not None else "—")
    pad = (f"{row['progress_at_death_mean']:.0%}" if row.get("progress_at_death_mean") is not None else "—")
    gap = row.get("novice_expert_gap_mean", 0.0)
    fw_full = (f"gen {row['first_win_mean']}{fmt_ci(row['first_win_ci'], sep=' ± ')}"
               if row["first_win_mean"] is not None else "never")
    bg_full = (f"gen {row['best_gen_mean']}{fmt_ci(row['best_gen_ci'], sep=' ± ')}"
               if row.get("best_gen_mean") is not None else "—")
    sd = row.get("completion_time_stddev")
    if sd is None and cells:  # legacy rows: derive from the per-seed cells
        sds = [c["completion_time_stddev"] for c in cells if c.get("mean_completion_time") is not None]
        sd = round(sum(sds) / len(sds), 1) if sds else None
    if mct != "—" and sd is not None:
        mct += f" ± {sd}s"
    rate = row.get("improvement_rate_mean")
    ttime = row.get("train_time_s")
    if ttime is None:
        ttime = sum(c.get("train_time_s") or 0 for c in cells)

    routes, best = _collect_routes(game, persona, row["level"], tag)
    strat = _cluster_strategies([r["p"] for r in routes if r["w"]])
    gkey = f"{game}|{row['level']}"   # one grid and one config per level and page, however many dialogs show them
    grid = _STORE["grids"].get(gkey) or _level_grid(game, row["level"])
    tid = "cfg_" + re.sub(r"\W", "_", f"{game}_{row['level']}")
    cfg_slot = _slot(tid, _TPL.get(tid) or _config_html(_level_config(game, row["level"], grid)))

    headline = [  # the four a designer reads first; the rest sit behind "all metrics"
        _stat("Win rate", f"{wr:.0%}{_ci(row)}", "measured 10 gens after first win", icon="coin"),
        _stat("First win", fw_full, f"{row['solved_by']}/{row['seeds']} seeds solved", icon="up"),
        _stat("Dominant cause", _cause_name(row["dominant_cause"]), icon="heartbreak", sub=
              f"{row['dominant_cause_frac']:.0%} of deaths"),
        _stat("Death spread", f"{row.get('death_cluster_entropy_mean', 0):.2f}",
              "0 = one hotspot · 1 = everywhere", icon="skull"),
    ]
    stats = [
        _stat("Best gen", bg_full, "where the record genome appeared"),
        _stat("Improvement", f"{rate:+.1%}/gen" if rate is not None else "—", "of the level per generation"),
        _stat("Completion", f"{row.get('completion_rate_mean', 0):.0%}", "wins / all episodes"),
        _stat("Mean win time", mct, "mean ± std of winning runs"),
        _stat("Progress at death", pad, "how far failers get"),
        _stat("Deaths per win", f"{row.get('deaths_per_win_mean', 'N/A')}", "failed attempts per win"),
        _stat("Coin rate", f"{row.get('coin_collection_rate_mean', 1):.0%}", "collected / available"),
        _stat("Skill gap", f"{gap:+.0%}", "late-gen wins − early-gen wins"),
        _stat("Stuck rate", f"{row['stuck_frac_mean']:.0%}", "episodes ending in a stall"),
        _stat("Train time", fmt_hms(ttime), f"all {row['seeds']} seeds, wall-clock"),
    ]
    if strat is not None:
        stats.append(_stat("Strategies", str(strat[0]), "distinct winning routes"))
        stats.append(_stat("Dominant path", f"{strat[1]:.0%}", "share on the top route"))
    vword, vcolor, vsent = _verdict_sentence(row, strat)
    did = "d_" + "".join(ch if ch.isalnum() else "_" for ch in f"{game}_{persona}_{tag or 'legacy'}_{row['level']}")

    curves = _json([c.get("curve", []) for c in cells])
    route_viz = ""
    if grid:
        _STORE["grids"][gkey] = grid
        payload = ({"grid": gkey, "ts": 32, "routes": [{"p": _pack(r["p"]), "w": r["w"]} for r in routes]}
                   | ({"best": best | {"p": _pack(best["p"])}} if best else {}))
        legend = " ".join(f'<span class="lg"><i style="background:{c}"></i>{GLYPH_NAME[k]}</span>'
                          for k, c in (("#", "#3a3a3a"), ("?", "#ac7c00"), ("^", "#a81000"),   # = drawRoutes
                                       ("E", "#e0442e"), ("C", "#f8d878"), ("S", "#f878f8"),
                                       ("H", "#fca044"), ("P", "#58d854"), ("G", "#006800"))
                          if any(k in r for r in grid))
        n_win = sum(1 for r in routes if r["w"])
        best_note = ""
        if best:
            best_note = (f" Gold = the fastest win: seed {best['seed']}"
                         + (f", {best['t']:.1f} s" if best["t"] is not None else "") + ".")
        route_viz = f"""
        <div class="viz routebox"><div class="vt">Agent routes on the level
            <span style="color:var(--green)">— wins</span>
            <span style="color:var(--red)">— deaths</span>
            {'<span style="color:var(--won)">— fastest win</span>' if best else ''}
            <button class="fsbtn" type="button">⛶ Full screen</button></div>
          <div class="routewrap"><canvas class="routes" width="1100" height="200" data-routes='{_json(payload)}' role="img"
            aria-label="{row['level']} drawn as tiles with {n_win} winning and {len(routes) - n_win} failed agent routes{', and the fastest win in gold' if best else ''}"></canvas></div>
          <div class="legend">{legend}</div>
          <div class="caption">Level geometry and entities with sampled agent traces: green = winning
          runs, red = failed runs (drawn faint). Where red lines stop is where agents die.{best_note}
          {'' if routes or best else 'No route traces logged for this level yet.'}</div></div>"""

    # the level window: the route map large on the left, the read-outs beside it, history below
    return f"""
    <div class="lvlcard">
      <button class="cardhead" type="button" aria-haspopup="dialog" aria-controls="{did}" data-dialog="{did}">
        <div class="lvlname">{row['level']} {_pill(row)}</div>
        <div class="bignum" style="color:{color}"><i class="ico coin"></i>{wr:.0%}</div>
        <div class="lvlsub">win rate · {fw}</div>
        <div class="mini"><i style="width:{max(2, wr * 100):.0f}%;background:{color}"></i></div>
      </button>
      <dialog class="detail" id="{did}" aria-labelledby="{did}_t">
        <div class="dhead">
          <h3 class="lvlname" id="{did}_t"><span>{game} · <i class="ico p-{persona}"></i>{persona} · {row['level']}</span> {_pill(row)}</h3>
          <div class="bignum" style="color:{color}">{wr:.0%}</div>
          <button class="close" type="button" aria-label="Close level details">✕</button>
        </div>
        <div class="dverdict" style="--c:{vcolor}">
          <span class="verdict">{vword.capitalize()}</span>
          <p class="vsent">{vsent}</p>
          <div class="verdicts">{_verdicts(row, strat, th)}</div>
        </div>
        <div class="dgrid">
          <div class="dmain">
            {route_viz}
            <div class="viz"><div class="vt">Where agents die (start → goal)</div>
              {_heat(row.get('death_hist') or [0] * 10)}
              <div class="caption">Each bin is 10% of the level. One bright bin = a single learnable
              chokepoint; many lit bins = difficulty spread across the level.</div></div>
          </div>
          <div class="dside">
            <div class="stats headline">{''.join(headline)}</div>
            <div class="viz"><div class="vt">What kills them</div>
              {_causebar(row.get('causes') or {})}</div>
          </div>
          <div class="viz dcurve"><div class="vt">Learning curves — fitness per generation, one color per seed</div>
            <canvas class="curve" width="1100" height="260" data-curves='{curves}' role="img"
              aria-label="Fitness per generation for each seed on {row['level']}"></canvas>
            <div class="caption">Solid line: the best genome each generation. Faint line: population
            average. A jump above the dashed line means winning runs (win bonus). Flat = stuck.</div></div>
          <div class="dmore">
            <details class="more"><summary>All {len(stats) + 4} metrics</summary>
              <div class="stats">{''.join(stats)}</div></details>
            {cfg_slot}
          </div>
        </div>
      </dialog>
    </div>"""


def _metric_table(rows: list[dict]) -> str:
    trs = []
    for r in rows:
        wr = f"{r['win_rate_mean']:.0%}{_ci(r, ' ±')}"
        fw = (f"{r['first_win_mean']}{fmt_ci(r['first_win_ci'])}" if r["first_win_mean"] is not None else "-")
        solved_cls = ("wr-good" if r["solved_by"] == r["seeds"] else
                      ("wr-mid" if r["solved_by"] else "wr-bad"))
        mct = f"{r['mean_completion_time']}s" if r.get("mean_completion_time") is not None else "—"
        if mct != "—" and r.get("completion_time_stddev") is not None:
            mct += f" ±{r['completion_time_stddev']}"
        bg = (f"{r['best_gen_mean']}{fmt_ci(r['best_gen_ci'])}" if r.get("best_gen_mean") is not None else "-")
        rate = (f"{r['improvement_rate_mean']:+.1%}" if r.get("improvement_rate_mean") is not None else "—")
        pad = f"{r['progress_at_death_mean']:.0%}" if r.get("progress_at_death_mean") is not None else "—"
        trs.append(
            f"<tr data-level=\"{r['level']}\" tabindex=\"0\" title=\"Open {r['level']}\"><td>{r['level']}</td>"
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
    tip = lambda k: f'scope="col" data-sort class="tip" {_tip_attrs(k)}'  # noqa: E731
    return f"""
    <div class="ovwrap"><table class="metrics">
      <thead><tr><th scope="col" data-sort>Level</th><th {tip("Solved")}>Solved</th><th {tip("First win")}>First win</th>
        <th {tip("Best gen")}>Best gen</th><th {tip("Improvement")}>Rate/gen</th>
        <th {tip("Win rate")}>Win rate ±CI</th><th {tip("Completion")}>Completion</th>
        <th {tip("Mean win time")}>Mean time</th><th {tip("Progress at death")}>Progress@death</th>
        <th {tip("Deaths per gen")}>Deaths/gen</th><th {tip("Death spread")}>Death spread</th>
        <th {tip("Coin rate")}>Coin rate</th><th {tip("Skill gap")}>Skill gap</th>
        <th {tip("Dominant cause")}>Dominant cause</th><th {tip("Stuck rate")}>Stuck</th>
        <th {tip("Train time")}>Train time</th></tr></thead>
      <tbody>{''.join(trs)}</tbody>
    </table></div>
    <div class="ovnote">Click a column to sort, a row to open that level.</div>"""


def _overview(personas: dict[str, dict]) -> str:
    order: list[str] = []
    for data in personas.values():
        for r in data["levels"]:
            if r["level"] not in order:
                order.append(r["level"])
    pnames = list(personas)
    ths = "".join(f"<th scope='col' style='color:{PERSONA_COLOR.get(p, 'var(--dim)')}'>{p.capitalize()}<br>"
                  f"<span style='font-weight:400;color:var(--dim)'>Win rate · first win</span></th>"
                  for p in pnames)
    trs = []
    for lvl in order:
        tds = [f"<td>{lvl}</td>"]
        best = "bad"
        for p in pnames:
            row = next((r for r in personas[p]["levels"] if r["level"] == lvl), None)
            if row is None:
                tds.append("<td class='wr-na'>—</td>")
                continue
            fw = (f"gen {row['first_win_mean']}" if row["first_win_mean"] is not None else "never")
            tds.append(f"<td><span class='{_wr_cls(row)}'>{row['win_rate_mean']:.0%}</span>"
                       f" <span style='color:var(--faint)'>· {fw}</span></td>")
            if row["solved_by"] == row["seeds"]:
                best = "ok"
            elif row["solved_by"] and best != "ok":
                best = "warn"
        tds.insert(1, f"<td>{_status_pill(best, {'ok': 'solved', 'warn': 'partial', 'bad': 'unsolved'}[best])}</td>")
        trs.append("<tr>" + "".join(tds) + "</tr>")
    return f"""
      <div class="ovwrap"><table>
        <thead><tr><th scope="col">Level</th><th scope="col" class="tip" {_tip_attrs("Status")}>Status ⓘ</th>{ths}</tr></thead>
        <tbody>{''.join(trs)}</tbody>
      </table></div>
    <div class="ovnote">Status = best across personas. Select a persona tab for the full
    per-level breakdown — click a card for its window with routes, death maps, causes, and curves.</div>"""


CONFIG_COLORS = ["#e0442e", "#cfcfcf", "#f8d878", "#58d854", "#f878f8", "#fca044", "#e40058", "#f87858"]  # NES, red first
PERSONA_DASH = {"experienced": [], "novice": [7, 5], "speedrunner": [2, 4]}
# chart series per persona: theme.css --p-* as hex (a canvas can't read CSS variables)
PERSONA_COLOR = dict(re.findall(r"--p-(\w+):\s*(#[0-9a-fA-F]{6})", THEME_CSS))


def _persona_chip(persona: str) -> str:
    return (f'<span class="pchip" style="--c:{PERSONA_COLOR.get(persona, "var(--dim)")}">'
            f'<i class="ico p-{persona}"></i> {persona.capitalize()}</span>')


def _tag_label(tag: str | None, data: dict) -> str:
    if not tag:
        return "legacy"
    sensors = (data.get("ga_config") or {}).get("sensors") or ("grid" if tag.endswith("_grid") else "rays")
    return f"pop {data.get('pop_size', '?')} · {data.get('gens_budget', '?')} gens · {sensors}"


def _dash_svg(dash: list) -> str:
    """A series' line style as drawn on the chart, for its toggle in the series matrix."""
    da = f' stroke-dasharray="{" ".join(map(str, dash))}"' if dash else ""
    return (f'<svg viewBox="0 0 26 6" width="26" height="6" aria-hidden="true">'
            f'<path d="M0 3h26" stroke="currentColor" stroke-width="3"{da}/></svg>')


def _game_chart(game: str, by_tag: dict, idx: int) -> str:
    """One chart per game: win rate per level, one series per (config, persona), beside the matrix
    that drives it: a row per config (its button picks the config shown below), a column per
    persona, each cell the series' toggle. Colour = config, line style = persona.
    Radar with 3+ levels, grouped bars otherwise."""
    order: list[str] = []
    pnames: list[str] = []
    for personas in by_tag.values():
        for pname, data in personas.items():
            if pname not in pnames:
                pnames.append(pname)
            for r in data["levels"]:
                if r["level"] not in order:
                    order.append(r["level"])
    series, mrows = [], []
    by_persona = len(by_tag) == 1  # one config: persona is the only variable, so it owns the colour
    for ti, (tag, personas) in enumerate(by_tag.items()):
        ccolor = CONFIG_COLORS[ti % len(CONFIG_COLORS)]
        cells = {}
        for pname, data in personas.items():
            color = PERSONA_COLOR.get(pname, "#cfcfcf") if by_persona else ccolor
            dash = [] if by_persona else PERSONA_DASH.get(pname, [4, 3])
            rows = {r["level"]: r for r in data["levels"]}
            name = f"{tag or 'legacy'} · {pname}"
            series.append({"name": name, "tag": tag or "legacy",
                           "persona": pname, "color": color, "dash": dash,
                           "vals": [round((rows.get(l) or {}).get("win_rate_mean", 0.0), 3) for l in order]})
            cells[pname] = (f'<button class="lchip" type="button" data-series="{len(series) - 1}" aria-pressed="true" '
                            f'style="--c:{color}" title="{name}" aria-label="{name}">{_dash_svg(dash)}</button>')
        data = next(iter(personas.values()))
        btn = (f'<button class="cfgbtn{" active" if ti == 0 else ""}" type="button" data-view="c{idx}_{ti}" '
               f'data-tag="{tag or "legacy"}" style="--c:{ccolor}"><b>{tag or "legacy"}</b>'
               f'<small>{_tag_label(tag, data)} · {len(data["seeds"])} seeds</small></button>')
        mrows.append(f'<tr><th scope="row">{btn}</th>' + "".join(f"<td>{cells.get(p, '')}</td>" for p in pnames) + "</tr>")
    heads = "".join(f'<th scope="col" title="{p}"><i class="ico p-{p}"></i><span>{p}</span></th>' for p in pnames)
    payload = _json({"axes": [str(v) for v in order], "series": series})
    kind = "radar" if len(order) >= 3 else "bars"
    enc = "colour = persona" if by_persona else "colour = config, line style = persona"
    return f"""
    <div class="viz chartbox">
      <div class="chartwrap">
        <div class="chartmain"><div class="vt">Win rate per level <span class="faint">— {kind}</span></div>
          <canvas class="chart" width="720" height="400" data-kind="{kind}"
            data-chart='{payload}' data-section="{idx}" role="img"
            aria-label="Win rate per level for {game}, one series per persona and config"></canvas></div>
        <div class="seriesbox"><div class="vt">Configs × personas</div>
          <div class="ovwrap"><table class="series">
            <thead><tr><th scope="col"><span class="sr">Config</span></th>{heads}</tr></thead>
            <tbody>{''.join(mrows)}</tbody></table></div>
          <div class="caption">Pick a config to show its cards and tables below; a cell shows or hides
          that series on the {kind} ({enc}).</div></div>
      </div></div>"""


SENSORS = ["ray fwd", "ray fwd-up 30°", "ray fwd-up 60°", "ray fwd-down 30°", "ray fwd-down 60°",
           "ray back", "enemy distance (fwd corridor)", "pit ahead", "grounded", "vx / 300",
           "vy / 600", "can jump", "q-blocks within 5 tiles / 5", "bias (1.0)"]
_DIAL_SVG = ("<svg viewBox='0 0 24 24' width='16' height='16' fill='none' stroke='currentColor' "
             "stroke-width='1.7' stroke-linecap='round'><path d='M3 12h4M17 12h4M12 3v4M12 17v4'/>"
             "<circle cx='12' cy='12' r='5'/><path d='M12 12l3-3'/></svg>")
_BRAIN_SVG = ("<svg viewBox='0 0 24 24' width='16' height='16' fill='none' stroke='currentColor' "
              "stroke-width='1.7' stroke-linecap='round'><circle cx='4' cy='6' r='2'/><circle cx='4' cy='18' r='2'/>"
              "<circle cx='12' cy='12' r='2.2'/><circle cx='20' cy='7' r='2'/><circle cx='20' cy='17' r='2'/>"
              "<path d='M6 7l4 4M6 17l4-4M14 11l4-3M14 13l4 3'/></svg>")


BODY = ["grounded", "vx / 300", "vy / 600", "can jump", "bias (1.0)"]


def _net_svg(mode: str = "rays", n_hidden: int = 16) -> str:
    """Inline diagram of the evolved net. rays: 14 labelled sensors → hidden → 3 outputs.
    grid: an 11×11×3 tile window + 5 body scalars → hidden → 3 outputs."""
    from .net import N_OUTPUTS
    from .sensors import GRID_CH, GRID_N, N_BODY
    W, H = 600, 392
    xi, xh, xo = 215, 390, 525
    yh = lambda j: 22 + j * (H - 44) / (n_hidden - 1)
    yo = lambda k: H / 2 + (k - 1) * 60
    out = [f'<svg class="net" viewBox="0 0 {W} {H}" role="img" aria-label="network diagram">']
    if mode == "grid":
        # input side: a mini tile window (3 channels stacked) + the body scalars
        gx, gy, cell = 40, 24, 11
        for ch, col in enumerate(("#cfcfcf", "#f8d878", "#e0442e")):
            off = ch * 7
            out.append(f'<rect x="{gx + off}" y="{gy + off}" width="{GRID_N * cell}" height="{GRID_N * cell}" '
                       f'fill="#000000" stroke="{col}" stroke-opacity=".8"/>')
        for r in range(GRID_N):
            for c in range(GRID_N):
                solid = r >= 8 or (c >= 9 and r >= 5)
                fill = "#ededed" if (r, c) == (5, 5) else ("#4a4a4a" if solid else "#0e0e0e")
                out.append(f'<rect x="{gx + 14 + c * cell + 1}" y="{gy + 14 + r * cell + 1}" width="{cell - 2}" '
                           f'height="{cell - 2}" fill="{fill}"/>')
        gcy = gy + 14 + GRID_N * cell / 2
        out.append(f'<text x="{gx}" y="{gy + 14 + GRID_N * cell + 26}" class="lbl">{GRID_CH} × {GRID_N} × {GRID_N} tiles'
                   f'</text><text x="{gx}" y="{gy + 14 + GRID_N * cell + 42}" class="lbl">'
                   f'<tspan fill="#cfcfcf">solid</tspan> · <tspan fill="#f8d878">collectible</tspan> · '
                   f'<tspan fill="#e0442e">hazard</tspan></text>')
        out.append('<g stroke="#cfcfcf" stroke-opacity=".16" stroke-width="1.2">')
        out += [f'<line x1="{gx + 14 + GRID_N * cell}" y1="{gcy:.0f}" x2="{xh}" y2="{yh(j):.0f}"/>'
                for j in range(n_hidden)]
        out.append('</g>')
        yb = lambda i: H - 118 + i * 22
        out.append('<g stroke="#cfcfcf" stroke-opacity=".13" stroke-width="1">')
        out += [f'<line x1="{xi}" y1="{yb(i):.0f}" x2="{xh}" y2="{yh(j):.0f}"/>'
                for i in range(N_BODY) for j in range(n_hidden)]
        out.append('</g>')
        for i, name in enumerate(BODY):
            out.append(f'<text x="{xi - 12}" y="{yb(i) + 4:.0f}" text-anchor="end" class="lbl">{name}</text>'
                       f'<circle cx="{xi}" cy="{yb(i):.0f}" r="5" fill="#0e0e0e" stroke="#cfcfcf" stroke-width="1.5"/>')
        in_cap = f"{GRID_CH * GRID_N * GRID_N} grid + {N_BODY} body"
    else:
        yi = lambda i: 22 + i * (H - 44) / (len(SENSORS) - 1)
        out.append('<g stroke="#cfcfcf" stroke-opacity=".13" stroke-width="1">')
        out += [f'<line x1="{xi}" y1="{yi(i):.0f}" x2="{xh}" y2="{yh(j):.0f}"/>'
                for i in range(len(SENSORS)) for j in range(n_hidden)]
        out.append('</g>')
        for i, name in enumerate(SENSORS):
            out.append(f'<text x="{xi - 12}" y="{yi(i) + 4:.0f}" text-anchor="end" class="lbl">{name}</text>'
                       f'<circle cx="{xi}" cy="{yi(i):.0f}" r="5" fill="#0e0e0e" stroke="#cfcfcf" stroke-width="1.5"/>')
        in_cap = f"{len(SENSORS)} ray sensors"
    out.append('<g stroke="#e0442e" stroke-opacity=".22" stroke-width="1">')
    out += [f'<line x1="{xh}" y1="{yh(j):.0f}" x2="{xo}" y2="{yo(k):.0f}"/>'
            for j in range(n_hidden) for k in range(N_OUTPUTS)]
    out.append('</g>')
    for j in range(n_hidden):
        out.append(f'<circle cx="{xh}" cy="{yh(j):.0f}" r="{5.5 if n_hidden <= 24 else 3.2}" fill="#1c1c1c" stroke="#cfcfcf" stroke-width="1.5"/>')
    for k, (name, col) in enumerate((("left", "#f8d878"), ("right", "#58d854"), ("jump", "#e0442e"))):
        out.append(f'<circle cx="{xo}" cy="{yo(k):.0f}" r="7" fill="{col}22" stroke="{col}" stroke-width="2"/>'
                   f'<text x="{xo + 14}" y="{yo(k) + 4:.0f}" class="lbl out" fill="{col}">{name}</text>')
    out.append(f'<text x="{xi if mode != "grid" else 110}" y="{H - 2}" text-anchor="middle" class="cap">{in_cap}</text>'
               f'<text x="{xh}" y="{H - 2}" text-anchor="middle" class="cap">{n_hidden} tanh</text>'
               f'<text x="{xo}" y="{H - 2}" text-anchor="middle" class="cap">{N_OUTPUTS} sigmoid</text></svg>')
    return "".join(out)


GA_DOC = {  # label, plain-language subtitle, formatter
    "sensors": ("Sensors", "what the agent sees (exteroception)", lambda v: str(v)),
    "pop_size": ("Population", "genomes evaluated per generation", lambda v: f"{v}"),
    "elite": ("Elite", "best genomes copied unchanged", lambda v: f"{v}"),
    "tournament_k": ("Tournament", "random genomes per pick, fittest wins", lambda v: f"k = {v}"),
    "crossover_rate": ("Crossover", "children bred from two parents (uniform mask)", lambda v: f"{v:.0%}"),
    "mutation_rate": ("Mutation rate", "share of weights nudged per child", lambda v: f"{v:.1%}" if v < 0.01 else f"{v:.0%}"),
    "mutation_sigma": ("Mutation σ", "gaussian nudge size", lambda v: f"{v:g}"),
    "init_sigma": ("Init σ", "N(0, σ) starting weights", lambda v: f"{v:g}"),
    "anneal_factor": ("Anneal", "× mutation after a level's first win", lambda v: f"×{v:g}"),
    "max_frames": ("Episode cap", "frames per attempt", lambda v: f"{v / 60:.0f}s"),
    "stuck_frames": ("Stall kill", "frames without progress → STUCK", lambda v: f"{v / 60:.0f}s"),
    "advance_wins": ("Advance at", "wins in one gen → next curriculum level", lambda v: f"{v} wins"),
    "win_bonus": ("Win bonus", "added to fitness on reaching the goal", lambda v: f"+{v:,.0f}"),
    "hidden": ("Hidden", "tanh units in the hidden layer", lambda v: f"{v}"),
    "action_feedback": ("Action feedback", "last move + jump fed back as 2 inputs", lambda v: "on" if v else "off"),
    "memory": ("Memory", "recurrent units looped back next frame", lambda v: f"{v}" if v else "off"),
}


def _ray_layout(game: str) -> str:
    """How the "rays" mode is wired for this game — a top-down game owns its own layout."""
    from .sensors import RAY_DIRS, RAY_MAX_DIST, RAY_STEP
    if game == "bomberman":
        return ("4 rays (N · E · S · W) + how soon each neighbouring tile burns + own-tile blast timer "
                "+ nearest-enemy bearing + bombs left + bricks a bomb here would open + exit bearing")
    if game == "megaman":
        return (f"{len(RAY_DIRS)} rays relative to facing + enemy ahead + nearest-enemy bearing + nearest shot "
                "+ pit probe + body + ladder here / below / overhead / ahead + route compass + boss energy; "
                "outputs: left, right, jump, up, down, fire")
    return (f"{len(RAY_DIRS)} rays ({RAY_MAX_DIST:.0f} px reach, {RAY_STEP:.0f} px steps) + enemy corridor "
            f"+ pit probe + q-block count + body state")


def _brain_dialog(sec_id: str, game: str, personas: dict[str, dict]) -> str:
    """GA + network hyperparameters behind one config, read from the probes' saved config."""
    from .evolution import GAConfig
    from .net import make_net
    from .personas import PERSONAS
    from .sensors import GRID_CH, GRID_N, RAY_DIRS, RAY_MAX_DIST, RAY_STEP
    data = next(iter(personas.values()))
    ga = data.get("ga_config")
    note = "" if ga else ("<div class='caption' style='margin:8px 0 0'>Legacy sweep: no config was saved "
                          "with these probes — showing the current GAConfig defaults.</div>")
    ga = dict(vars(GAConfig()) | (ga or {}))
    mode = ga.get("sensors") or ("grid" if (data.get("tag") or "").endswith("_grid") else "rays")
    ga["sensors"] = mode
    n_params = make_net(GAConfig(**{k: v for k, v in ga.items() if k in vars(GAConfig())})).n_params
    extra_in = (" + previous action" if ga.get("action_feedback") else "") + \
               (f" + {ga['memory']} memory units" if ga.get("memory") else "")
    sensing = (f"{GRID_N}×{GRID_N} tile window centred on the agent, {GRID_CH} channels "
               f"(solid · collectible · hazard) + body state — the Mario-AI-competition-style view"
               if mode == "grid" else _ray_layout(game))
    tiles = "".join(
        f'<div class="stat"><div class="lbl">{lbl}</div><div class="val">{fmt(ga[k])}</div>'
        f'<div class="sub">{sub}</div></div>' for k, (lbl, sub, fmt) in GA_DOC.items() if k in ga)
    steps = [("Evaluate", f"{ga['pop_size']} genomes play the level"),
             ("Elitism", f"top {ga['elite']} survive as-is"),
             ("Select", f"tournament of {ga['tournament_k']}"),
             ("Crossover", f"{ga['crossover_rate']:.0%} of children, uniform mask"),
             ("Mutate", f"{ga['mutation_rate']:.0%} of weights · σ {ga['mutation_sigma']:g}"),
             ("Anneal", f"×{ga['anneal_factor']:g} after first win")]
    pipe = '<span class="arrow">→</span>'.join(
        f'<span class="step"><b>{n}</b><small>{d}</small></span>' for n, d in steps)
    net_id = f"net_{mode}_{ga.get('hidden') or 16}"   # the diagram is the same for every config of that shape
    chips = "".join(
        f'<span class="chip" style="--c:{col}"><b>{p.name}</b> '
        f'{"sprint" if p.sprint else "walk"} · senses every {p.sensor_period} frame{"s" if p.sensor_period > 1 else ""}'
        f'{" · +" + format(p.time_rate, "g") + " fitness / s left" if p.time_rate else ""}</span>'
        for (n, p), col in ((kv, PERSONA_COLOR.get(kv[0], "#cfcfcf"))
                            for kv in PERSONAS.items()) if n in personas)
    return f"""
    <dialog class="detail brain" id="brain_{sec_id}" aria-labelledby="brain_{sec_id}_t">
      <div class="dhead"><h3 class="lvlname" id="brain_{sec_id}_t"><span>{_BRAIN_SVG} Neuroevolution hyperparameters · {game}
        · {data.get('tag') or 'legacy'} · <em class="mode {mode}">{mode.capitalize()}</em></span></h3>
        <button class="close" type="button" aria-label="Close hyperparameters">✕</button></div>
      {note}
      <div class="brainrow">
        <div class="viz netbox"><div class="vt">The brain — {n_params:,} weights, no gradients
          <span class="faint">· {mode} sensing · inputs in [−1, 1] · outputs fire above 0.5 · left/right conflict → argmax</span></div>
          {_slot(net_id, _TPL.get(net_id) or _net_svg(mode, ga.get("hidden") or 16))}
          <div class="caption"><b>Sensing:</b> {sensing}{extra_in}. <b>Fitness:</b> furthest x reached,
          +{ga['win_bonus']:,.0f} on a win (+ seconds left × persona time rate).</div></div>
        <div class="viz"><div class="vt">Genetic algorithm (GAConfig)</div>
          <div class="stats gastats">{tiles}</div></div>
      </div>
      <div class="viz"><div class="vt">One generation</div><div class="pipe">{pipe}</div></div>
      <div class="viz"><div class="vt">Personas in this config</div><div class="chips">{chips}</div></div>
    </dialog>"""


def _game_section(game: str, by_tag: dict, idx: int, th: dict, ga_anchor: str | None = None) -> str:
    ga_link = (f'<a class="navbtn tip" href="index.html#{game}/gasweep" target="_top" '
               f'data-tip="GA hyperparameter sweep for {game}: which knobs matter, recommended config">'
               f'{_DIAL_SVG} GA sweep</a>' if ga_anchor else "")
    n_levels = len({r["level"] for p in by_tag.values() for d in p.values() for r in d["levels"]})
    total = sum(_train_time(d) for p in by_tag.values() for d in p.values())
    views = []   # the config buttons are the rows of the chart's series matrix (_game_chart)
    for ti, (tag, personas) in enumerate(by_tag.items()):
        data = next(iter(personas.values()))
        cid = f"c{idx}_{ti}"
        sid = _section_id(game, tag)
        color = CONFIG_COLORS[ti % len(CONFIG_COLORS)]
        tabs = [f'<button class="tab t-overview active" role="tab" aria-selected="true" aria-controls="{cid}_ov" '
                f'data-view="{cid}_ov" type="button">Overview</button>']
        pviews = [f'<div class="view active" role="tabpanel" id="{cid}_ov">{_overview(personas)}</div>']
        for pi, (pname, pdata) in enumerate(personas.items()):
            vid = f"{cid}_{pi}"
            tabs.append(f'<button class="tab" role="tab" aria-selected="false" aria-controls="{vid}" data-view="{vid}" '
                        f'type="button" style="--c:{PERSONA_COLOR.get(pname, "var(--dim)")}"><i class="ico p-{pname}"></i>{pname.capitalize()}</button>')
            cell_map = pdata.get("cells", {})
            cards = "".join(_card(game, pname, r, cell_map.get(r["level"], []), th, tag)
                            for r in pdata["levels"])
            pviews.append(f"""
            <div class="view" role="tabpanel" id="{vid}">
              <div class="lvlgrid">{cards}</div>
              <div class="tbltitle">Full metric table — {pname}</div>
              {_metric_table(pdata["levels"])}
            </div>""")
        views.append(f"""
        <div class="cfgview{" active" if ti == 0 else ""}" id="{cid}">
          <div class="toolbar">
            <span class="cfgnow" style="--c:{color}"><b>{tag or "legacy"}</b>{_tag_label(tag, data)}</span>
            <div class="tabs" role="tablist" aria-label="Persona">{''.join(tabs)}</div>
            <button class="brainbtn tip" type="button" data-dialog="brain_{sid}"
              data-tip="GA + network hyperparameters used for this config">{_BRAIN_SVG} Hyperparameters</button>
            {ga_link}
          </div>
          {''.join(pviews)}
          {_brain_dialog(sid, game, personas)}
        </div>""")
    return f"""
  <section class="game frame stripe g-{game}" id="g_{game}" aria-labelledby="g_{game}_t">
    <div class="gamehead">
      <button class="chev" type="button" aria-expanded="true" aria-controls="g_{game}_body" aria-label="Collapse {game}">▾</button>
      <h2 class="gametag" id="g_{game}_t">{_game_icon(game)}{game.capitalize()}</h2>
      <span class="gamemeta">{len(by_tag)} config{'s' if len(by_tag) != 1 else ''} · {n_levels} levels
        · total train time {fmt_hms(total)}</span>
    </div>
    <div class="body" id="g_{game}_body">
      {_game_chart(game, by_tag, idx)}
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
        if name in ("balance", "probes", "gasweep", "_replay"):
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
  <section class="game frame stripe" id="g_runs" aria-labelledby="g_runs_t">
    <div class="gamehead">
      <button class="chev" type="button" aria-expanded="true" aria-label="Collapse training runs">▾</button>
      <h2 class="gametag" id="g_runs_t">Training runs</h2>
      <span class="gamemeta">every run found under runs/ · resume with
      <code>--resume runs/&lt;name&gt;</code></span></div>
    <div class="body"><div class="ovwrap"><table>
      <thead><tr><th scope="col">Run</th><th scope="col">Persona</th><th scope="col">Gens</th><th scope="col">Best fitness</th>
        <th scope="col">Best level</th><th scope="col">Total wins</th><th scope="col">Train time</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table></div></div>
  </section>"""


JS = """
const FONT = '500 12px "Barlow Semi Condensed", Barlow, sans-serif';  // canvas labels: the theme's --dense
const REDUCED = matchMedia('(prefers-reduced-motion: reduce)').matches;

// nothing is drawn before it can be seen: a dialog paints when it opens, any other canvas when it
// scrolls into view (a report holds hundreds of level windows; drawing them all at load took seconds)
const STORES = {};   // page-level JSON (level grids, GA-sweep curves), parsed on first use
const store = id => STORES[id] ??= JSON.parse(document.getElementById(id)?.textContent || '{}');
const PAINTED = new WeakSet();
const fitW = cv => cv.parentElement.clientWidth || 1100;
function paintCanvas(cv){
  if (PAINTED.has(cv)) return;
  PAINTED.add(cv);
  if (cv.classList.contains('routes')) drawRoutes(cv, fitW(cv), Infinity);
  else if (cv.classList.contains('curve')) drawCurve(cv);
  else if (cv.classList.contains('scatter')) drawScatter(cv);
  else if (cv.classList.contains('chart')) introChart(cv);
}
function paint(root){
  for (const s of root.querySelectorAll('[data-tpl]:empty')){   // shared blocks: a level's config, the net diagram
    const t = document.getElementById(s.dataset.tpl);
    if (t) s.replaceChildren(t.content.cloneNode(true));
  }
  root.querySelectorAll('canvas').forEach(paintCanvas);
}
function openDialog(dlg){ dlg.showModal(); paint(dlg); }

for (const tab of document.querySelectorAll('.tab')){
  tab.addEventListener('click', () => {
    const scope = tab.closest('.cfgview') || tab.closest('section.game');
    scope.querySelectorAll('.tab').forEach(t => { t.classList.remove('active'); t.setAttribute('aria-selected', 'false'); });
    scope.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    tab.classList.add('active'); tab.setAttribute('aria-selected', 'true');
    scope.querySelector('#' + tab.dataset.view).classList.add('active');
  });
}
// info bubbles open on tap/Enter too, not just hover (capture: before a header's own Enter-to-sort)
document.addEventListener('keydown', e => {
  const t = e.target.closest?.('.tip[tabindex]');
  if (!t) return;
  if (e.key === 'Enter' || e.key === ' '){ e.preventDefault(); t.classList.toggle('open'); }
  if (e.key === 'Escape') t.classList.remove('open');
}, true);
document.addEventListener('focusout', e => { if (e.target.matches?.('.tip[tabindex]')) e.target.classList.remove('open'); });
// full metric tables: click a header to sort, a row to open that level's window
const num = s => { const m = s.replace(/,/g, '').match(/-?\\d+(\\.\\d+)?/); return m ? +m[0] : -Infinity; };
for (const table of document.querySelectorAll('table.metrics')){
  const ths = [...table.querySelectorAll('th[data-sort]')];
  ths.forEach((th, col) => {
    const sort = () => {
      const dir = th.getAttribute('aria-sort') === 'descending' ? 'ascending' : 'descending';
      ths.forEach(x => x.removeAttribute('aria-sort')); th.setAttribute('aria-sort', dir);
      const tb = table.tBodies[0], rows = [...tb.rows];
      const key = r => col ? num(r.cells[col].textContent) : r.cells[0].textContent;
      rows.sort((a, b) => (key(a) > key(b) ? 1 : key(a) < key(b) ? -1 : 0) * (dir === 'ascending' ? 1 : -1));
      rows.forEach(r => tb.appendChild(r));
    };
    th.addEventListener('click', e => { if (!e.target.classList.contains('open')) sort(); });
    th.addEventListener('keydown', e => { if (e.key === 'Enter' && !th.classList.contains('open')) sort(); });
  });
  const open = r => { const card = [...r.closest('.view').querySelectorAll('.lvlcard')]
      .find(c => c.querySelector('.lvlname').firstChild.textContent.trim() === r.dataset.level);
    card?.querySelector('.cardhead').click(); };
  for (const r of table.tBodies[0].rows){
    r.addEventListener('click', () => open(r));
    r.addEventListener('keydown', e => { if (e.key === 'Enter') open(r); });
  }
}
// config selector (the rows of the series matrix): one config's tables/cards at a time; the chart highlights it
const HIDDEN = new WeakMap();  // per-chart set of hidden series indices
function focusConfig(section, tag){
  // the chart shows the selected config's personas; other configs stay one cell-click away
  const cv = section.querySelector('canvas.chart');
  if (!cv) return;
  const d = JSON.parse(cv.dataset.chart || 'null');
  if (!d) return;
  const set = new Set();
  d.series.forEach((s, i) => { if (s.tag !== tag) set.add(i); });
  HIDDEN.set(cv, set);
  cv.closest('.chartwrap')?.querySelectorAll('.lchip').forEach(c => { const off = set.has(+c.dataset.series); c.classList.toggle('off', off); c.setAttribute('aria-pressed', !off); });
}
const redraw = cv => { PAINTED.add(cv); drawChart(cv, 1); };   // a user change: no intro animation later
for (const b of document.querySelectorAll('.cfgbtn')){
  b.addEventListener('click', () => {
    const section = b.closest('section.game');
    section.querySelectorAll('.cfgbtn').forEach(x => x.classList.remove('active'));
    section.querySelectorAll('.cfgview').forEach(v => v.classList.remove('active'));
    b.classList.add('active');
    section.querySelector('#' + b.dataset.view).classList.add('active');
    focusConfig(section, b.dataset.tag);
    const cv = section.querySelector('canvas.chart');
    if (cv) redraw(cv);
  });
}
for (const s of document.querySelectorAll('section.game')){
  const active = s.querySelector('.cfgbtn.active');
  if (active) focusConfig(s, active.dataset.tag);
}
// cards open a mini window (native <dialog>): Esc, ✕, or a backdrop click closes it
// level windows: the URL hash remembers which one is open (#d_<game>_<persona>_<config>_<level>)
function openLevel(dlg, push){
  if (!dlg || dlg.open) return;
  const section = dlg.closest('section.game'), view = dlg.closest('.view'), cfg = dlg.closest('.cfgview');
  if (section){ section.classList.remove('collapsed'); syncChev(section); }
  if (cfg && !cfg.classList.contains('active')) section.querySelector(`.cfgbtn[data-view="${cfg.id}"]`)?.click();
  if (view && !view.classList.contains('active')) cfg.querySelector(`.tab[data-view="${view.id}"]`)?.click();
  openDialog(dlg);
  if (push) history.replaceState(null, '', '#' + dlg.id);
}
for (const head of document.querySelectorAll('.cardhead')){
  head.addEventListener('click', () => openLevel(document.getElementById(head.dataset.dialog), true));
}
for (const b of document.querySelectorAll('.brainbtn')){
  b.addEventListener('click', () => openDialog(document.getElementById(b.dataset.dialog)));
}
for (const dlg of document.querySelectorAll('dialog')){
  dlg.querySelector('.close').addEventListener('click', () => dlg.close());
  dlg.addEventListener('click', e => { if (e.target === dlg) dlg.close(); });
  dlg.addEventListener('close', () => { if (location.hash === '#' + dlg.id) history.replaceState(null, '', location.pathname); });
}
// collapsible sections + collapse/expand all; a collapsed card opens from a click anywhere on it
const sections = [...document.querySelectorAll('section.game')];
const toggleAll = document.getElementById('toggleAll');
const syncChev = s => {
  const c = s.querySelector('.chev'), open = !s.classList.contains('collapsed');
  const name = s.querySelector('.gametag')?.textContent.trim() || 'section';
  c.setAttribute('aria-expanded', open); c.setAttribute('aria-label', (open ? 'Collapse ' : 'Expand ') + name);
};
const syncToggle = () => {
  sections.forEach(syncChev);
  if (!toggleAll) return;
  const allClosed = sections.every(s => s.classList.contains('collapsed'));
  toggleAll.textContent = allClosed ? 'Expand all' : 'Collapse all';
};
const expand = s => {   // in a card grid an opened card takes a full row of its own: follow it there
  s.classList.remove('collapsed'); syncToggle();
  if (s.parentElement.classList.contains('cardgrid')) s.scrollIntoView({block: 'start', behavior: REDUCED ? 'auto' : 'smooth'});
};
for (const s of sections){
  s.querySelector('.chev').addEventListener('click', () => {
    if (s.classList.contains('collapsed')) expand(s);
    else { s.classList.add('collapsed'); syncToggle(); }
  });
  s.addEventListener('click', e => { if (s.classList.contains('collapsed') && !e.target.closest('a, button')) expand(s); });
}
syncToggle();
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

// overview links on the GA-sweep page open the (collapsed) section they point at
for (const a of document.querySelectorAll('a.gsjump')){
  a.addEventListener('click', () => {
    const s = document.getElementById(a.getAttribute('href').slice(1));
    if (s) { s.classList.remove('collapsed'); syncToggle(); }
  });
}

// capacity scatter (GA sweep): generations to first win vs network weights
function drawScatter(cv){
  const pts = JSON.parse(cv.dataset.points || '[]');
  if (!pts.length) return;
  const fit = JSON.parse(cv.dataset.fit || 'null');
  const ctx = cv.getContext('2d');
  const W = cv.width, H = cv.height, L = 58, R = 26, T = 26, B = 44;
  const lx = v => Math.log2(v);
  const xs = pts.map(p => lx(p.x));
  const x0 = Math.min(...xs) - 0.45, x1 = Math.max(...xs) + 0.45;
  const maxY = Math.max(...pts.map(p => p.y + (p.ci || 0)), 1) * 1.18;
  const px = v => L + (lx(v) - x0) / (x1 - x0) * (W - L - R);
  const py = v => H - B - (Math.max(v, 0) / maxY) * (H - T - B);
  ctx.font = FONT; ctx.lineWidth = 1;
  for (let q = 0; q <= 4; q++){
    const v = maxY * q / 4, y = py(v);
    ctx.strokeStyle = '#2e2e2e'; ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(W - R, y); ctx.stroke();
    ctx.fillStyle = '#a0a0a0'; ctx.fillText(maxY < 10 ? v.toFixed(1) : v.toFixed(0), 8, y + 4);
  }
  let lastEnd = -1e9;
  for (const p of [...pts].sort((a, b) => a.x - b.x)){
    const x = px(p.x), t = p.x.toLocaleString(), w = ctx.measureText(t).width;
    ctx.strokeStyle = '#1c1c1c'; ctx.beginPath(); ctx.moveTo(x, T); ctx.lineTo(x, H - B); ctx.stroke();
    if (x - w / 2 > lastEnd + 6){ ctx.fillStyle = '#a0a0a0'; ctx.fillText(t, x - w / 2, H - B + 16); lastEnd = x + w / 2; }
  }
  ctx.fillStyle = '#a0a0a0';
  const xl = 'weights (log scale)'; ctx.fillText(xl, W - R - ctx.measureText(xl).width, H - 6);
  ctx.save(); ctx.translate(14, (T + H - B) / 2); ctx.rotate(-Math.PI / 2);
  const yl = 'gens to first win'; ctx.fillText(yl, -ctx.measureText(yl).width / 2, 0); ctx.restore();
  if (fit){
    const hx = pts.filter(p => p.kind === 'hidden').map(p => p.x);
    const xa = Math.min(...hx), xb = Math.max(...hx);
    ctx.setLineDash([6, 5]); ctx.strokeStyle = fit.color || '#58d85499'; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(px(xa), py(fit.m * lx(xa) + fit.b)); ctx.lineTo(px(xb), py(fit.m * lx(xb) + fit.b));
    ctx.stroke(); ctx.setLineDash([]); ctx.lineWidth = 1;
  }
  for (const p of pts){
    const x = px(p.x), y = py(p.y);
    if (p.ci){
      ctx.strokeStyle = p.color + '99'; ctx.beginPath();
      ctx.moveTo(x, py(p.y - p.ci)); ctx.lineTo(x, py(p.y + p.ci)); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(x - 4, py(p.y - p.ci)); ctx.lineTo(x + 4, py(p.y - p.ci));
      ctx.moveTo(x - 4, py(p.y + p.ci)); ctx.lineTo(x + 4, py(p.y + p.ci)); ctx.stroke();
    }
    ctx.fillStyle = p.color; ctx.beginPath();
    if (p.kind === 'arch'){ ctx.moveTo(x, y - 7); ctx.lineTo(x + 7, y); ctx.lineTo(x, y + 7); ctx.lineTo(x - 7, y); ctx.closePath(); }
    else ctx.arc(x, y, p.star ? 7 : 5.5, 0, Math.PI * 2);
    ctx.fill();
    if (p.star){ ctx.strokeStyle = '#ededed'; ctx.lineWidth = 1.5; ctx.stroke(); ctx.lineWidth = 1; }
  }
  // labels last, nudged apart when points sit on top of each other
  const placed = [];
  for (const p of [...pts].sort((a, b) => a.x - b.x || a.y - b.y)){
    let x = px(p.x) + 10, y = py(p.y) - 8;
    const w = ctx.measureText(p.label).width;
    for (let k = 0; k < 6 && placed.some(q => Math.abs(q.y - y) < 13 && x < q.x + q.w + 6 && x + w + 6 > q.x); k++) y -= 13;
    placed.push({x, y, w});
    ctx.fillStyle = '#ededed'; ctx.fillText(p.label, x, y);
  }
}
// learning curves: best (solid) and population average (faint) per generation, one colour per seed
function drawCurve(cv){
  const seeds = (cv.dataset.cref ? JSON.parse(cv.dataset.cref).flatMap(k => store('curves')[k] || [])
                                 : JSON.parse(cv.dataset.curves || '[]')).filter(c => c.length > 1);
  if (!seeds.length) return;
  // drawn at the size it is shown, so the labels stay crisp in any column width
  const dpr = window.devicePixelRatio || 1, W = cv.clientWidth || cv.width, H = cv.clientHeight || cv.height;
  cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
  const ctx = cv.getContext('2d');
  ctx.scale(dpr, dpr);
  const L = 62, R = 12, T = 14, B = 30;
  const maxY = Math.max(...seeds.flat().map(p => p[0]), 1) * 1.05;
  const maxX = Math.max(...seeds.map(c => c.length));
  const px = i => L + i * (W - L - R) / (maxX - 1);
  const py = v => H - B - (v / maxY) * (H - T - B);
  ctx.font = FONT;
  ctx.strokeStyle = '#2e2e2e'; ctx.fillStyle = '#a0a0a0'; ctx.lineWidth = 1;
  for (let q = 0; q <= 4; q++){
    const v = maxY * q / 4, y = py(v);
    ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(W - R, y); ctx.stroke();
    ctx.fillText(Math.round(v).toLocaleString(), 4, y + 4);
  }
  if (maxY > 5000){
    const y = py(5000);
    ctx.setLineDash([5, 4]); ctx.strokeStyle = '#58d85455';
    ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(W - R, y); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#58d85499'; ctx.fillText('win bonus', W - R - 78, y - 5);
  }
  ctx.fillStyle = '#a0a0a0';
  ctx.fillText('gen 1', L, H - 9);
  const el = 'gen ' + maxX;
  ctx.fillText(el, W - R - ctx.measureText(el).width, H - 9);
  const colors = JSON.parse(cv.dataset.colors || 'null') || ['#e0442e', '#f8d878', '#cfcfcf', '#f878f8'];
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
// colour = config, dash = persona; matrix cells toggle series; the selected config is highlighted.
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
  const shown = Math.round(cv.getBoundingClientRect().width);   // phones: a square radar at the size shown, not 720 x 400 shrunk
  if (cv.dataset.kind === 'radar' && shown && shown < 560 && cv.width !== shown){ cv.width = shown; cv.height = shown; }
  const W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);
  ctx.font = FONT;
  const alpha = s => (focus && s.tag !== focus) ? 0.28 : 1;
  const visible = d.series.map((s, i) => !hidden.has(i));
  if (cv.dataset.kind === 'bars'){
    const L = 48, R = 12, T = 16, B = 34, n = d.axes.length;
    const plotW = W - L - R, plotH = H - T - B;
    ctx.strokeStyle = '#2e2e2e'; ctx.fillStyle = '#a0a0a0'; ctx.lineWidth = 1;
    for (let q = 0; q <= 4; q++){
      const y = T + plotH - plotH * q / 4;
      ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(W - R, y); ctx.stroke();
      ctx.fillText((q * 25) + '%', 6, y + 4);
    }
    const shown = d.series.map((s, i) => i).filter(i => visible[i]);
    // few levels: cap the group width and centre the groups instead of stretching two bars across the canvas
    const groupW = Math.min(plotW / n, 60 + 34 * Math.max(1, shown.length)), off = (plotW - groupW * n) / 2;
    const barW = Math.max(3, (groupW * 0.72) / Math.max(1, shown.length));
    d.axes.forEach((ax, k) => {
      const gx = L + off + k * groupW + groupW * 0.14;
      shown.forEach((si, bi) => {
        const s = d.series[si], v = Math.min(1, s.vals[k] || 0) * t, h = v * plotH;
        ctx.globalAlpha = alpha(s);
        ctx.fillStyle = s.color;
        ctx.fillRect(gx + bi * barW, T + plotH - h, barW - 1.5, h);
        if (s.dash.length){  // persona hatch on top of the bar
          ctx.strokeStyle = '#000000'; ctx.lineWidth = 1; ctx.setLineDash(s.dash);
          ctx.beginPath(); ctx.moveTo(gx + bi * barW + barW / 2, T + plotH - h); ctx.lineTo(gx + bi * barW + barW / 2, T + plotH); ctx.stroke();
          ctx.setLineDash([]);
        }
        ctx.globalAlpha = 1;
      });
      ctx.fillStyle = '#cfcfcf'; ctx.textAlign = 'center';
      ctx.fillText(ax, L + off + k * groupW + groupW / 2, H - 12);
      ctx.textAlign = 'left';
    });
    return;
  }
  const cx = W / 2, cy = H / 2 + 6, R = Math.min(W, H) / 2 - 60, n = d.axes.length;
  const ang = i => -Math.PI / 2 + i * 2 * Math.PI / n;
  for (const frac of [0.25, 0.5, 0.75, 1]){
    ctx.strokeStyle = frac === 0.5 ? '#4a4a4a' : '#2e2e2e'; ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i <= n; i++){
      const a = ang(i % n), r = R * frac;
      i ? ctx.lineTo(cx + r * Math.cos(a), cy + r * Math.sin(a)) : ctx.moveTo(cx + r * Math.cos(a), cy + r * Math.sin(a));
    }
    ctx.stroke();
  }
  ctx.fillStyle = '#a0a0a0'; ctx.fillText('50%', cx + 4, cy - R * 0.5 - 3);
  for (let i = 0; i < n; i++){
    const a = ang(i);
    ctx.strokeStyle = '#2e2e2e';
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + R * Math.cos(a), cy + R * Math.sin(a)); ctx.stroke();
    ctx.fillStyle = '#cfcfcf';
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
    if (alpha(s) === 1){ ctx.fillStyle = s.color + '0d'; ctx.fill(); }   // a faint fill: six series overlap
    for (let i = 0; i < n; i++){
      const a = ang(i), v = Math.min(1, s.vals[i] || 0) * t;
      ctx.fillStyle = s.color;
      ctx.beginPath(); ctx.arc(cx + R * v * Math.cos(a), cy + R * v * Math.sin(a), 2.6, 0, 7); ctx.fill();
    }
    ctx.globalAlpha = 1;
  }
}
function introChart(cv){   // the first draw grows the series out from the centre
  if (REDUCED) return drawChart(cv, 1);
  const t0 = performance.now();
  const tick = now => {
    const t = Math.min(1, (now - t0) / 700);
    drawChart(cv, 1 - Math.pow(1 - t, 3));  // ease-out cubic
    if (t < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}
for (const cv of document.querySelectorAll('canvas.chart')){
  for (const chip of cv.closest('.chartwrap').querySelectorAll('.lchip')){
    chip.addEventListener('click', () => {
      const set = HIDDEN.get(cv) || new Set(), i = +chip.dataset.series;
      set.has(i) ? set.delete(i) : set.add(i);
      HIDDEN.set(cv, set); chip.classList.toggle('off', set.has(i)); chip.setAttribute('aria-pressed', !set.has(i));
      redraw(cv);
    });
  }
}

// agent routes over level geometry; redrawn to fit when the map goes full screen
function drawRoutes(cv, maxW, maxH, cap = 1){
  const d = JSON.parse(cv.dataset.routes || 'null');
  const grid = d && (typeof d.grid === 'string' ? store('grids')[d.grid] : d.grid);   // a key: one grid per level per page
  if (!grid || !grid.length) return;
  const cols = Math.max(...grid.map(r => r.length));
  const rows = grid.length;
  // fit the box: never below 7 px per tile (wide levels scroll instead of shrinking), at most `cap` x the level's own size
  const scale = Math.min(cap, Math.max(Math.min(maxW / (cols * d.ts), maxH / (rows * d.ts)), 7 / d.ts));
  cv.width = Math.round(cols * d.ts * scale);
  cv.height = Math.max(90, Math.round(rows * d.ts * scale));
  const ctx = cv.getContext('2d');
  ctx.fillStyle = '#000000'; ctx.fillRect(0, 0, cv.width, cv.height);
  const cell = d.ts * scale;
  const BLOCK = {'#': '#3a3a3a', '=': '#5a5a5a', '?': '#ac7c00', '^': '#a81000', 'G': '#006800',
                 'S': '#f878f8', 'H': '#fca044', 'D': '#a0a0a0'};   // NES; the legend in _card matches
  const MARK = {'E': '#e0442e', 'C': '#f8d878', 'P': '#58d854'};  // entities drawn as markers
  const marks = [];
  grid.forEach((row, r) => {
    for (let c = 0; c < row.length; c++){
      const ch = row[c];
      if (ch === ' ') continue;
      if (MARK[ch]) { marks.push([c, r, ch]); continue; }
      ctx.fillStyle = BLOCK[ch] || '#2e2e2e';
      if (ch === '=') ctx.fillRect(c * cell, r * cell, cell + 0.5, Math.max(1, cell * 0.35));
      else ctx.fillRect(c * cell, r * cell, cell + 0.5, cell + 0.5);
    }
  });
  for (const [c, r, ch] of marks){  // enemies: ring + dot, coins: dot, start: dot — always ≥ 2.5px
    const x = (c + 0.5) * cell, y = (r + 0.5) * cell, rad = Math.max(2.5, cell * 0.45);
    ctx.fillStyle = MARK[ch]; ctx.beginPath(); ctx.arc(x, y, ch === 'C' ? rad * 0.6 : rad, 0, 7); ctx.fill();
    if (ch === 'E'){ ctx.strokeStyle = '#f87858'; ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(x, y, rad + 1.2, 0, 7); ctx.stroke(); }
  }
  const draw = (route, color, width, alpha) => {
    ctx.strokeStyle = color; ctx.globalAlpha = alpha; ctx.lineWidth = width;
    ctx.beginPath();
    let x = 0, y = 0;   // packed (_pack): the first point, then each step from the one before
    for (let i = 0; i < route.p.length; i += 2){
      x += route.p[i]; y += route.p[i + 1];
      i ? ctx.lineTo(x * scale, y * scale) : ctx.moveTo(x * scale, y * scale);
    }
    ctx.stroke();
    ctx.globalAlpha = 1;
  };
  const lw = Math.max(1, scale * d.ts / 7);
  for (const r of d.routes) if (!r.w) draw(r, '#e40058', lw, 0.30);   // --dead
  for (const r of d.routes) if (r.w) draw(r, '#58d854', lw * 1.6, 0.85);   // --run
  if (d.best){   // the fastest win on top: coin gold on a black edge, like the pixel icons
    const bw = Math.max(2.5, lw * 2.2);
    ctx.lineJoin = 'round';
    draw(d.best, '#000000', bw + 2, 0.85);
    draw(d.best, '#f8d878', bw, 1);   // --won
  }
}
for (const b of document.querySelectorAll('.fsbtn')){
  b.addEventListener('click', () => {
    if (document.fullscreenElement) document.exitFullscreen();
    else b.closest('.viz').requestFullscreen();
  });
}
let fsViz = null;   // only the map going in or out of full screen is redrawn
document.addEventListener('fullscreenchange', () => {
  const fs = document.fullscreenElement, viz = fs || fsViz, cv = viz?.querySelector('canvas.routes');
  if (cv){ PAINTED.add(cv); fs ? drawRoutes(cv, fs.clientWidth - 48, fs.clientHeight - 150, 2) : drawRoutes(cv, fitW(cv), Infinity); }
  const b = viz?.querySelector('.fsbtn');
  if (b) b.textContent = fs ? '⛶ Exit full screen' : '⛶ Full screen';
  fsViz = fs;
});
for (const b of document.querySelectorAll('.copybtn')){
  b.addEventListener('click', async () => {
    const code = b.previousElementSibling;
    try { await navigator.clipboard.writeText(code.textContent); b.textContent = 'Copied'; }
    catch { const r = document.createRange(); r.selectNodeContents(code);
            getSelection().removeAllRanges(); getSelection().addRange(r); b.textContent = 'Select + ⌘C'; }
    setTimeout(() => { b.textContent = 'Copy'; }, 1500);
  });
}

// on load: every canvas outside a dialog paints as it nears the viewport; open the level window the URL names
const lazy = 'IntersectionObserver' in window ? new IntersectionObserver(es => {
  for (const e of es) if (e.isIntersecting){ lazy.unobserve(e.target); paintCanvas(e.target); }
}, {rootMargin: '200px 0px'}) : null;
for (const cv of document.querySelectorAll('canvas')) if (!cv.closest('dialog')) lazy ? lazy.observe(cv) : paintCanvas(cv);
if (location.hash.startsWith('#d_')) openLevel(document.getElementById(location.hash.slice(1)), false);
"""

GLOSSARY = f"""
  <details class="gloss">
    <summary>How to read this page</summary>
    Each game opens on its chart: win rate per level, one series per config × persona, toggled from the
    matrix beside it, whose rows pick the config shown below. The <b>Overview</b> tab has one row per level,
    one column per persona, so skill tiers sit side by side. The number is the <b>win rate</b>
    (<span class="wr-good">green ≥ 50%</span>,
    <span class="wr-mid">orange below</span>,
    <span class="wr-bad">red = never solved</span>).
    <b>Status</b>: {_status_pill("ok", "solved")} every seed won at least once ·
    {_status_pill("warn", "n/N seeds")} some did · {_status_pill("bad", "unsolved")}
    none did. Click a card for its window — it opens on a one-line verdict, then the routes the agents
    took, the fastest win in gold. Hover, tap, or focus any <b>ⓘ</b> label for the definition; formulas
    and episode statuses are on the <b>Instructions</b> page.
  </details>"""


def _nav(games: list, has_runs: bool, logo: str | None, page: str = "report") -> str:
    # `logo` stays in the signature, unused: the theme's pixel wordmark replaces the raster logo (off-palette glow)
    links = "" if page != "report" else "".join(
        f'<a href="report.html#g_{g}">{_game_icon(g)}{g.capitalize()}</a>' for g in games)
    if has_runs and page == "report":
        links += '<a href="report.html#g_runs"><i class="ico hourglass"></i>Runs</a>'
    if page != "report":
        links += '<a class="doc" href="report.html">Balance report</a>'
    links += ('<a class="doc" href="mutations.html"' + (' aria-current="page"' if page == "mutations" else "")
              + '><i class="ico dna"></i>Mutations</a>')
    links += ('<a class="doc" href="ablation.html"' + (' aria-current="page"' if page == "ablation" else "")
              + '><i class="ico eye"></i>Sensor ablation</a>')
    links += ('<a class="doc" href="gasweep.html"' + (' aria-current="page"' if page == "gasweep" else "")
              + '><i class="ico gear"></i>GA sweep</a>')
    links += ('<a class="doc" href="instructions.html"' + (' aria-current="page"' if page == "instructions" else "")
              + '><i class="ico qblock"></i>Instructions</a>')
    if page == "report":
        links += '<button class="navbtn" id="toggleAll" type="button">Collapse all</button>'
    return f"""
  <a class="skip" href="#main">Skip to content</a>
  <nav aria-label="Pages"><h1 class="brand"><span class="wordmark">PEAK</span><em>Command Center</em></h1>
    <div class="links">{links}</div>
  </nav>"""


def _instructions_page(games: list[str], logo: str | None, nav: bool = True, after: str = "") -> str:
    tips = "".join(f"<li><b>{k}</b> — {v}</li>" for k, v in TIPS.items())
    persona = lambda p, text: f'<li><b><i class="ico p-{p}"></i> {p}</b> — {text}</li>'  # noqa: E731
    sections = [  # (anchor, icon, heading, body): the contents rail lists them in this order
        ("what", "qblock", "What this is", """
  <p>PEAK evolves populations of tiny neural networks (291 parameters at the default 16 hidden units;
  147–1,155 across the GA sweep) that play your levels
  thousands of times, then reports what happened. Nothing here blocks anything — the tool
  measures, you decide.</p>"""),
        ("sweep", "gamepad", "Running a sweep", """
  <ul>
    <li><code>python menu.py</code> → <b>13 Full Sweep</b>: pick games and personas; probes run
    in parallel across CPU cores and land in <code>runs/balance/</code>.</li>
    <li><b>14 Sensor Ablation</b> runs the same selection twice — raycasts and the tile grid — and
    compares them on the <b>Sensor ablation</b> page.</li>
    <li><b>15 GA Sweep</b> moves one GA knob at a time (population, elite, tournament, crossover,
    mutation, anneal, init σ, hidden size, action feedback, memory units) to its literature low and
    high bound and reports which knobs matter per game on the <b>GA sweep</b> page — with a
    recommended config and a first-win-vs-parameters capacity curve.</li>
    <li>Only <b>enabled</b> levels are probed — toggle them with <b>10 Toggle Levels</b>.</li>
    <li>Regenerate the Command Center any time: <code>python -m code.neuro.command_center --open</code>
    (menu 12). It reads the report each probe run wrote (paper seeds only), so it works without a finished sweep.</li>
    <li>Each distinct config (persona · population · generation budget · sensors) gets its own section
    and its own probe folders; re-running the same config on a level replaces that level only.</li>
  </ul>"""),
        ("personas", "p-experienced", "Personas (skill tiers)", f"""
  <ul class="icolist">
    {persona("novice", 'walking pace, reacts every 3rd frame. "Can beginners get through?"')}
    {persona("experienced", "full reactions, walking pace. The baseline difficulty read.")}
    {persona("speedrunner", 'sprint speed plus a time bonus. "What does skilled, fast play look like?"')}
  </ul>
  <p>The Overview puts all three side by side. Healthy levels slope novice → speedrunner upward;
  inversions are design findings (a level that punishes speed, or walls out beginners).</p>"""),
        ("loop", "up", "The design loop", """
  <ul>
    <li>Author or edit a level (menu 9), play it yourself (menu 5).</li>
    <li>Probe it (menu 13) and read the card: win rate, first-win generation, where deaths cluster.</li>
    <li>Change one thing — double or halve a parameter — and probe again.</li>
    <li>An unsolved level whose best progress is identical across seeds is usually broken
    geometry (unreachable goal), not difficulty.</li>
  </ul>"""),
        ("statuses", "flag", "Episode statuses", f"""
  <p>Every episode (one genome playing the level once) ends in exactly one status:</p>
  <ul class="icolist">
    <li><b class="c-WON"><i class="ico coin"></i> WON</b> — reached the goal. Fitness = max x + win bonus (+ time left × persona rate).</li>
    <li><b class="c-DEAD"><i class="ico skull"></i> DEAD</b> — killed by an enemy, pit, saw, spike, or the timer; the engine's death cause
    becomes the failure cause (Enemy, Pit, Saw, Spike, OOB, Timeout…).</li>
    <li><b class="c-STUCK"><i class="ico hourglass"></i> STUCK</b> — no forward progress for <code>stuck_frames</code> (5 s) or the engine's stall
    watchdog fired; counted as the <b>Stall</b> cause.</li>
  </ul>
  <p>Level status rolls these up per seed:</p>
  <ul class="icolist">
    <li>{_status_pill("ok", "solved")} every seed had ≥1 WON episode within the budget</li>
    <li>{_status_pill("warn", "partial")} some seeds did (the cards say how many, n/N)</li>
    <li>{_status_pill("bad", "unsolved")} none did</li>
  </ul>
  <p>The Overview shows the best status across personas.</p>"""),
        ("numbers", "gear", "How the numbers are computed", """
  <ul>
    <li><b>First win</b> — the generation of the first WON episode, mean ± 95% CI over the seeds that
    solved the level (t-distribution; no interval from a single seed).</li>
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
    <li><b>Deaths per win</b>: (DEAD + STUCK episodes) ÷ WON episodes over the whole probe; every death
    when nothing won. The B2 band reads this one.</li>
    <li><b>Deaths per gen</b> — (DEAD + STUCK episodes) ÷ generations; one attempt per population member.</li>
    <li><b>Death spread</b> — entropy of the death-position histogram (10 bins), normalized to 0..1.</li>
    <li><b>Dominant cause</b> — most frequent failure cause over DEAD + STUCK episodes, with its share.</li>
    <li><b>Stuck rate</b> — STUCK episodes ÷ all episodes.</li>
    <li><b>Skill gap</b> — win rate of the last third of generations minus the first third.</li>
    <li><b>Coin rate</b> — mean over episodes of coins collected ÷ coins in the level (1 if none).</li>
    <li><b>Strategies / Dominant path</b> — winning routes clustered by shape; cluster count and the
    share on the biggest cluster (needs ≥2 logged wins).</li>
    <li><b>Fastest win</b> — the gold route on a level's map: the winning episode with the shortest
    in-game time, over every seed (from the episode log, or the route sample a cluster probe keeps).</li>
    <li><b>Train time</b> — wall-clock seconds the trainer spent per generation, summed.</li>
    <li><b>B1 / B2 / B3</b> — each metric compared with the target ± warning bands in
    <code>code/stats/MarioThresholds.yaml</code>.</li>
  </ul>"""),
        ("metrics", "eye", "Every metric", f"""
  <ul>{tips}</ul>"""),
        ("repro", "dna", "Reproducibility", """
  <p>Everything is deterministic under a seed. Default probe seeds are
  <code>1234 · 2025 · 31337</code>; each (level × seed) cell evolves a fresh population with the
  same GA settings, so numbers are comparable across sweeps as long as the GA config is unchanged.</p>"""),
    ]
    toc = "".join(f'<li><a href="#{a}"><i class="ico {i}"></i>{h}</a></li>' for a, i, h, _ in sections)
    body = "".join(f'<section id="{a}" aria-labelledby="{a}_t"><h2 id="{a}_t"><i class="ico {i}"></i>{h}</h2>{b}</section>'
                   for a, i, h, b in sections)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PEAK Command Center: Instructions</title><style>{CSS}</style></head><body>
{_nav(games, False, logo, page="instructions") if nav else ""}
<main id="main" class="doc">
  <div class="toc frame" role="navigation" aria-label="On this page"><h2>Contents</h2><ul>{toc}</ul></div>
  <div class="doc-body">{body}{after}</div>
</main>
<footer class="docfoot">PEAK engine · instructions</footer>
<div class="bricks" aria-hidden="true"></div>
</body></html>"""


MODE_COLOR = {"rays": "#f87858", "grid": "#cfcfcf"}   # NES $26 / $22 (CSS: .dhead em.rays, --blue)


def _mode_svgs() -> dict[str, str]:
    """Agent's-eye diagrams: a ray fan vs the 11x11 tile window, both centred on the agent."""
    import math
    rays = []
    for ang in (0, -30, -60, 30, 60, 180):
        a = math.radians(ang)
        x2, y2 = 75 + 62 * math.cos(a), 75 + 62 * math.sin(a)
        rays.append(f"<line x1='75' y1='75' x2='{x2:.1f}' y2='{y2:.1f}' stroke='{MODE_COLOR['rays']}' "
                    f"stroke-width='2' opacity='.9'/><circle cx='{x2:.1f}' cy='{y2:.1f}' r='3' fill='#ededed'/>")
    ray_svg = (f"<svg viewBox='0 0 150 150'><rect width='150' height='150' fill='#000000'/>"
               f"<rect x='0' y='100' width='150' height='50' fill='#2e2e2e'/>"
               f"<rect x='118' y='60' width='32' height='40' fill='#2e2e2e'/>"
               f"<rect x='88' y='92' width='6' height='8' fill='{MODE_COLOR['rays']}' opacity='.5'/>"
               f"{''.join(rays)}<rect x='69' y='63' width='12' height='24' fill='#ededed'/></svg>")
    cells = []
    for r in range(11):
        for c in range(11):
            solid = r >= 8 or (c >= 9 and r >= 5)
            fill = "#4a4a4a" if solid else "#0e0e0e"
            if r == 5 and c == 5:
                fill = "#ededed"
            cells.append(f"<rect x='{3 + c * 13.1:.1f}' y='{3 + r * 13.1:.1f}' width='12' height='12' "
                         f"fill='{fill}'/>")
    grid_svg = (f"<svg viewBox='0 0 150 150'><rect width='150' height='150' fill='#000000'/>"
                f"{''.join(cells)}<rect x='1' y='1' width='148' height='148' fill='none' "
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
        return f"gen {r['first_win_mean']}{fmt_ci(r['first_win_ci'])}" if r["first_win_mean"] is not None else "never"

    def mean(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    def agg(m):
        rs = list(rows[m].values())
        solved = sum(1 for r in rs if r["solved_by"])
        return {"solved": f'{solved}<span class="faint">/{n}</span>',
                "fw": f"gen {sweep_point(by_mode[m])[1]:.1f}" if solved else "never",   # unsolved seeds at the budget
                "wr": f'{mean([r["win_rate_mean"] for r in rs]) or 0:.0%}',
                "time": fmt_hms(_train_time(by_mode[m]))}
    A = {m: agg(m) for m in modes}

    def vs(key):   # one tile shows both modes side by side
        return " <span class='faint'>vs</span> ".join(
            f'<span style="color:{MODE_COLOR[m]}">{A[m][key]}</span>' for m in modes)
    stats = [f'<div class="stat"><div class="lbl">{lbl}</div><div class="val">{vs(k)}</div></div>'
             for k, lbl in (("solved", "Levels solved"), ("fw", "Mean first win"), ("wr", "Mean win rate"), ("time", "Probe train time"))]

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
                       f"<td><span class='{_wr_cls(r)}'>{r['win_rate_mean']:.0%}</span>{fmt_ci(r['win_rate_ci'], '.0%', ' ±')}</td>"
                       f"<td>{_cause_name(r['dominant_cause'])} {r['dominant_cause_frac']:.0%}</td>")
        if len(modes) == 2:
            a, b = rows["rays"].get(lvl), rows["grid"].get(lvl)
            if a and b:
                d = b["win_rate_mean"] - a["win_rate_mean"]
                cls = "delta-g" if d > 0.005 else "delta-r" if d < -0.005 else "delta-0"
                fa, fb = a["first_win_mean"], b["first_win_mean"]
                dg = ("—" if fa is None and fb is None else "grid only" if fa is None else "rays only" if fb is None
                      else f"{fb - fa:+.1f} gens" if a["solved_by"] == a["seeds"] and b["solved_by"] == b["seeds"]
                      else "partial")   # means over the solved seeds compare like with like only when every seed solved
                tds.append(f"<td class='{cls}'>{d:+.0%}</td><td>{dg}</td>")
            else:
                tds.append("<td class='wr-na'>—</td><td class='wr-na'>—</td>")
        trs.append("<tr>" + "".join(tds) + "</tr>")
    mode_ths = "".join(f"<th class='{'mr' if m == 'rays' else 'mg'}' colspan='4' style='color:{MODE_COLOR[m]}'>{m.capitalize()}</th>"
                       for m in modes)
    sub_ths = "".join(f"<th class='{'mr' if m == 'rays' else 'mg'}'>Solved</th><th>First win</th><th>Win rate</th><th>Cause</th>"
                      for m in modes)
    delta_th = "<th colspan='2'>Grid − rays</th>" if len(modes) == 2 else ""
    delta_sub = "<th>Δ win rate</th><th>Δ first win</th>" if len(modes) == 2 else ""
    shades = {"rays": ["#f87858", "#e0442e", "#fca044", "#f0d0b0"],   # NES reds / greys
              "grid": ["#cfcfcf", "#a0a0a0", "#d8d8d8", "#5a5a5a"]}
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
            f'<canvas class="curve" width="560" height="220" data-curves=\'{_json(series)}\' '
            f'data-colors=\'{_json(colors)}\' role="img" aria-label="Learning curves on {lvl}: rays vs grid seeds"></canvas></div>')
    curves_html = "" if not curve_boxes else (
        f'<div class="tbltitle" style="margin-top:14px">Learning curves — best fitness per generation, '
        f'<span style="color:{MODE_COLOR["rays"]}">reds</span> = ray seeds, '
        f'<span style="color:{MODE_COLOR["grid"]}">greys</span> = grid seeds</div>'
        f'<div class="curvegrid">{"".join(curve_boxes)}</div>'
        f'<div class="caption">Solid = best genome, faint = population average. The dashed green line is the '
        f'win bonus: a curve crossing it is a seed that solved the level. Steeper = faster learning.</div>')
    pending = "" if len(modes) == 2 else (
        f'<div class="ovnote">Only <b>{modes[0]}</b> probed so far for this config — run the other mode '
        f'(menu 14, or <code>balance --game {game} --persona {persona} --sensors '
        f'{"grid" if modes[0] == "rays" else "rays"}</code>) to fill the comparison.</div>')
    sid = f"abl{idx}"
    legend = " vs ".join(f'<span style="color:{MODE_COLOR[m]}">■ {m}</span>' for m in modes)
    face = _abl_face(by_mode["rays"], by_mode["grid"]) if len(modes) == 2 else ""
    return f"""
  <section class="game frame stripe g-{game} collapsed" id="abl_{idx}" aria-labelledby="abl_{idx}_t">
    <div class="gamehead">
      <button class="chev" type="button" aria-expanded="false" aria-label="Expand {game} {persona}">▾</button>
      <h2 class="gametag" id="abl_{idx}_t">{_game_icon(game)}{game.capitalize()}</h2>{_persona_chip(persona)}
      <span class="gamemeta">{base} · seeds {meta['seeds']} · pop {meta.get('pop_size', '?')} ×
        {meta.get('gens_budget', '?')} gens/probe</span>
    </div>
    {face}
    <div class="body">
      <div class="toolbar">
        <div class="tabs"><button class="tab active" data-view="{sid}_s" type="button">Simple</button>
          <button class="tab" data-view="{sid}_d" type="button">In-depth</button></div>
      </div>
      <div class="view active" id="{sid}_s">
        <div class="stats ablstats">{''.join(stats)}</div>
        <div class="viz"><div class="vt">Win rate per level — {legend}
          <span class="faint">&nbsp;(bar = win rate over the {WIN_WINDOW} gens after the first win · label: first win ± CI · seeds that won)</span></div>
          <div class="dual">{''.join(dual)}</div></div>
        {pending}
      </div>
      <div class="view" id="{sid}_d">
        <div class="tbltitle">Full comparison table</div>
        <div class="ovwrap"><table>
          <thead><tr class="modes"><th></th>{mode_ths}{delta_th}</tr>
            <tr><th>Level</th>{sub_ths}{delta_sub}</tr></thead>
          <tbody>{''.join(trs)}</tbody></table></div>
        {curves_html}
      </div>
    </div>
  </section>"""


def _abl_face(r: dict, g: dict) -> str:
    """The face of one rays-vs-grid card, side by side: each mode's mean win rate, first win (unsolved seeds
    at the budget, as in the GA sweep) and levels solved, the paired Δ between them (identical level × seed
    cells, ★ = clears its 95 % CI) and the winner."""
    m, ci, up, dn, n = paired_delta(g, r)   # grid − rays
    sig = abs(m) > (ci or 0) > 0
    winner = ("tie" if abs(m) < 0.005 else "grid" if m > 0 else "rays")

    def agg(d):
        rs = d["levels"]
        solved = sum(1 for x in rs if x["solved_by"])
        return (sum(x["win_rate_mean"] for x in rs) / len(rs) if rs else 0.0,
                sweep_point(d)[1] if solved else None, solved, len(rs))

    def side(mode, wr, fw, solved, nl):
        return (f'<div class="vsside{" win" if winner == mode else ""}" style="--c:{MODE_COLOR[mode]}">'
                f'<span class="vsmode">{mode.capitalize()}</span><b class="num">{wr:.0%}</b>'
                f'<small>{"gen %.1f" % fw if fw is not None else "never"}</small>'
                f'<small>{solved}/{nl} levels solved</small></div>')
    rw, rf, rsv, nl = agg(r)
    gw, gf, gsv, _ = agg(g)
    dcls = "delta-g" if m > 0.005 else "delta-r" if m < -0.005 else "delta-0"
    return (f'<div class="vs">{side("rays", rw, rf, rsv, nl)}'
            f'<div class="vsmid"><b class="num {dcls}" title="{up} cells grid better · {dn} rays better · {n} paired">{m:+.0%}</b>'
            f'<small>{fmt_ci(ci, ".0%", "± ")}{" ★" if sig else ""}</small><small>grid − rays</small>'
            f'<span class="verdict" style="--c:{MODE_COLOR.get(winner, "var(--faint)")}">{winner.capitalize()}</span></div>'
            f'{side("grid", gw, gf, gsv, nl)}</div>')


def _ablation_page(games: dict, logo: str | None, stamp: str, nav: bool = True) -> str:
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
    # the scoreboard between the two senses, like a versus screen
    score = (f'<div class="vsboard"><div class="vsscore">'
             f'<span class="vsn" style="--c:{MODE_COLOR["rays"]}"><b class="num">{wins["rays"]}</b><small>rays</small></span>'
             f'<span class="vsword" aria-hidden="true">VS</span>'
             f'<span class="vsn" style="--c:{MODE_COLOR["grid"]}"><b class="num">{wins["grid"]}</b><small>grid</small></span></div>'
             f'<div class="vsties"><b class="num">{wins["tie"]}</b> ties</div>'
             f'<p>level × persona × budget cells won on win rate, {total} paired</p></div>'
             if total else "")
    svgs = _mode_svgs()
    hero = []
    for m, blurb in (("rays", "A handful of raycasts plus the scalars that game needs — a forward enemy corridor, "
                              "a pit probe, a question-block count. Sees far along a few lines; blind between them."),
                     ("grid", "The cores' tile window sized to 11×11 around the agent, three channels: solid / "
                              "collectible / hazard. Sees everything nearby at tile resolution; no Dijkstra oracle.")):
        nin = sensor_dim(m)
        hero.append(f'<div class="ablmode" style="--c:{MODE_COLOR[m]}">{svgs[m]}<div><h3>{m.capitalize()}</h3><p>{blurb}</p>'
                    f'<div class="chips"><span class="chip" style="--c:{MODE_COLOR[m]}"><b>{nin}</b> inputs</span>'
                    f'<span class="chip" style="--c:{MODE_COLOR[m]}"><b>{NeuralNet(nin).n_params:,}</b> weights</span>'
                    f'<span class="chip" style="--c:{MODE_COLOR[m]}"><b>5</b> shared body scalars</span></div></div></div>')
    paired = [(k, v) for k, v in sorted(pairs.items()) if len(v) == 2]
    unpaired = [(k, next(iter(v))) for k, v in sorted(pairs.items()) if len(v) == 1]
    sections = "".join(_ablation_section(g, p, b, by_mode, i) for i, ((g, p, b), by_mode) in enumerate(paired))
    if sections:
        sections = (f'<div class="tbltitle">Rays vs grid, one card per sweep <span class="faint">— win rate over the {WIN_WINDOW} '
                    f'gens after the first win; Δ = grid − rays, paired on identical level × seed cells, ★ = clears its 95 % CI. '
                    f'Open a card for the level-by-level details.</span></div><div class="cardgrid">{sections}</div>')
    if unpaired:
        items = ", ".join(f"{g} · {p} · {b} ({m} only)" for (g, p, b), m in unpaired)
        sections += (f'<div class="ovnote" style="margin:16px 0">Not yet paired — probed with one sensor mode only: {items}. '
                     f'Run the other mode (menu 14) to add them to the comparison.</div>')
    if not paired and not unpaired:
        sections = ('<p style="color:var(--dim)">No sensor probes yet — run <b>14 Sensor Ablation</b> from '
                    '<code>python menu.py</code> (or <code>balance --ablation --sensors rays</code> and '
                    '<code>--ablation --sensors grid</code>) and rebuild this page.</p>')
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PEAK Command Center: Sensor ablation</title><style>{CSS}</style></head><body>
{_nav(list(games), False, logo, page="ablation") if nav else ""}
<main id="main">
  <div class="pagehead"><h2><i class="ico eye"></i>Sensor ablation</h2>
    <p>Same GA, same seeds, same levels — only what the agent sees changes · generated {stamp}</p></div>
  <div class="ablhero{" scored" if score else ""}">{hero[0]}{score}{hero[1]}</div>
  {sections}
  <footer>Generated by PEAK from the sensor-ablation probes on disk. Regenerate from the menu (12) after a new ablation.</footer>
</main><div class="bricks" aria-hidden="true"></div>{_page_data()}<script>{JS}</script></body></html>"""



# ── GA hyperparameter sweep page (data: runs/balance/gasweep_*.json) ──────────

_AXIS_SHORT = {"hidden": lambda v: f"h{v}", "pop_size": lambda v: f"pop {v}", "elite": lambda v: f"elite {v}",
               "tournament_k": lambda v: f"k{v}", "crossover_rate": lambda v: f"xo {v:g}",
               "mutation_rate": lambda v: f"mut {v:g}", "mutation_sigma": lambda v: f"σ {v:g}",
               "anneal_factor": lambda v: f"anneal {v:g}", "init_sigma": lambda v: f"init {v:g}",
               "action_feedback": lambda v: "+feedback", "memory": lambda v: f"+mem{v}"}
_VERDICT = {"flat": ("var(--green)", "Flat"), "improves": ("var(--blue)", "Bigger helps"),
            "degrades": ("var(--red)", "Bigger hurts"), "insufficient": ("var(--faint)", "Not measured")}


def _gs_short(axis: str, val) -> str:
    return {"base": "base", "best": "best"}.get(axis) or _AXIS_SHORT[axis](val)


def _gs_order(data: dict) -> tuple:
    axis, val = parse_sweep_tag(data.get("tag") or "") or ("", None)
    rank = {"base": -1, "best": 99}.get(axis, list(AXES).index(axis) if axis in AXES else 98)
    return rank, (float(val) if isinstance(val, (int, float)) else 0.0)


def _gasweep_groups(datas: list[dict]) -> dict[tuple, list[dict]]:
    """(game, persona, gens_budget, sensors) -> reports, baseline first then AXES order."""
    groups: dict[tuple, list[dict]] = {}
    for d in datas:
        if not parse_sweep_tag(d.get("tag") or ""):
            continue
        key = (d["game"], d.get("persona") or "experienced", d.get("gens_budget"),
               (d.get("ga_config") or {}).get("sensors", "rays"), tag_base_sig(d["tag"]) or "")
        groups.setdefault(key, []).append(d)
    for lst in groups.values():
        lst.sort(key=_gs_order)
    return dict(sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2] or 0, kv[0][3], kv[0][4])))


def _gs_baseline_label(ga: dict) -> str:
    return (f"pop {ga['pop_size']} · elite {ga['elite']} · k{ga['tournament_k']} · xo {ga['crossover_rate']:g} · "
            f"mut {ga['mutation_rate']:g} / σ {ga['mutation_sigma']:g} · anneal {ga['anneal_factor']:g} · "
            f"init {ga['init_sigma']:g} · h{ga['hidden']}"
            + (" · +feedback" if ga.get("action_feedback") else "") + (f" · +mem{ga['memory']}" if ga.get("memory") else ""))


def _gasweep_overview(groups: dict[tuple, list[dict]]) -> str:
    """One table: a column per sweep, a row per knob (winning value; Δ shown only when it clears
    its CI), plus baseline / composite / capacity rows. Sections below are collapsed; the
    column headers open them."""
    from .evolution import GAConfig
    cols = []
    for i, (key, datas) in enumerate(groups.items()):
        game, persona, budget, sensors, _sig = key
        pts = [_sweep_point(d) for d in datas]
        base = next((q for q in pts if q["axis"] == "base"), None)
        cols.append({"i": i, "game": game, "persona": persona, "budget": budget, "sensors": sensors, "pts": pts,
                     "base": base, "best": next((q for q in pts if q["axis"] == "best"), None),
                     "ga": base["ga"] if base else dict(vars(GAConfig())), "verdict": _hidden_verdict(pts)})
    if not cols:
        return ""
    star = '<i class="ico star" role="img" aria-label="clears its 95 % CI"></i>'
    rows = []
    for c in cols:   # a row per sweep: its baseline, the winning bound of every knob, the composite, the capacity verdict
        tds = [f'<th scope="row" class="g-{c["game"]}"><a class="gsjump" href="#gs_{c["i"]}"><span>{_game_icon(c["game"])}'
               f'{c["game"].capitalize()}</span><small><i class="ico p-{c["persona"]}"></i>{c["persona"]}</small></a></th>',
               f'<td class="kbase">{c["base"]["wr"]:.0%} <small>gen {c["base"]["fw"]:.0f}</small></td>' if c["base"]
               else '<td class="kbase">—</td>']
        for axis in AXES:
            ap = ([c["base"]] if c["base"] else []) + [q for q in c["pts"] if q["axis"] == axis]
            if len(ap) < 2:
                tds.append('<td class="k k-none">—</td>')
                continue
            best = _axis_best(ap)
            if best["axis"] == "base":
                tds.append('<td class="k k-base">base</td>')
                continue
            m, ci, up, dn, n = paired_delta(best["data"], c["base"]["data"]) if c["base"] else (0.0, 0.0, 0, 0, 0)
            sig = abs(m) > (ci or 0) > 0
            tds.append(f'<td class="k{" k-sig" if sig else ""}" style="{_effect(m)}" title="{GA_DOC[axis][0]}: Δ win rate {m:+.0%}{fmt_ci(ci, ".0%", " ± ")} · {up} cells up, {dn} down of {n}">'
                       f'<b>{GA_DOC[axis][2](best["val"])}</b>{f"<small>{m:+.0%} {star}</small>" if sig else ""}</td>')
        if c["best"] and c["base"]:
            m, ci, up, dn, n = paired_delta(c["best"]["data"], c["base"]["data"])
            sig = abs(m) > (ci or 0) > 0
            tds.append(f'<td class="k{" k-sig" if sig else ""}" style="{_effect(m)}" title="{up} cells up, {dn} down of {n} · first win gen {c["best"]["fw"]:.1f}">'
                       f'<b>{c["best"]["wr"]:.0%}</b><small>{m:+.0%}{fmt_ci(ci, ".0%", " ± ")}{f" {star}" if sig else ""}</small></td>')
        else:
            tds.append('<td class="k k-none">not run</td>')
        tds.append(f'<td><span class="verdict" style="--c:{_VERDICT[c["verdict"]["word"]][0]}">{_VERDICT[c["verdict"]["word"]][1]}</span></td>')
        rows.append("<tr>" + "".join(tds) + "</tr>")
    ths = "".join(f'<th scope="col">{GA_DOC[a][0]}</th>' for a in AXES)
    baselines = {_gs_baseline_label(c["ga"]) for c in cols}
    note = (f'Baseline: <span style="color:var(--txt)">{next(iter(baselines))}</span> · ' if len(baselines) == 1 else "")
    return f"""
  <div class="viz gsov"><div class="vt">Best value per knob <span class="faint">— per sweep, the bound that beat the baseline on
    win rate ("base" = none did), shaded by its paired Δ; the Δ is printed only when it clears its 95 % CI. Click a sweep to open it.</span></div>
    <div class="caption" style="margin:-2px 0 8px">{note}paired on identical level × seed cells</div>
    <div class="ovwrap"><table class="knobmap"><thead><tr><th scope="col">Sweep</th><th scope="col">Baseline win rate</th>{ths}
      <th scope="col">Composite (confirmed)</th><th scope="col">Capacity</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>
    <div class="kmlegend"><span><i style="{_effect(-0.3)}"></i>worse than the baseline</span><span><i></i>about the same</span>
      <span><i style="{_effect(0.3)}"></i>better</span><span><i class="sig"></i>{star} Δ clears its 95 % CI</span>
      <span><b>base</b> no bound beat the baseline</span><span><b>—</b> not run</span></div>
  </div>"""


def _effect(m: float) -> str:
    """Knob-map shade for a paired Δ win rate vs the baseline: green better, red worse, deeper with size
    (±30 % saturates; capped so the white text on it stays readable)."""
    if abs(m) < 0.005:
        return ""
    return f"background:rgba({'88,216,84' if m > 0 else '228,0,88'},{0.14 + 0.41 * min(1.0, abs(m) / 0.3):.2f})"


def _sweep_point(data: dict) -> dict:
    """One config → the numbers every chart on the page uses. Unsolved cells are censored at the
    gens budget (a never-winning config sits at the top of the capacity curve, not off it)."""
    from .evolution import GAConfig
    from .net import make_net
    axis, val = parse_sweep_tag(data["tag"])
    budget = data.get("gens_budget") or 0
    cells = [c for lst in (data.get("cells") or {}).values() for c in lst]
    wr, wr_ci = mean_ci([float(c.get("win_rate") or 0.0) for c in cells])
    fw, fw_ci = mean_ci([float(c.get("first_win_gen") or budget) for c in cells])
    levels = data.get("levels") or []
    bg = [r["best_gen_mean"] for r in levels if r.get("best_gen_mean") is not None]
    ir = [r["improvement_rate_mean"] for r in levels if r.get("improvement_rate_mean") is not None]
    ga = {k: v for k, v in (dict(vars(GAConfig()) | (data.get("ga_config") or {}))).items() if k in vars(GAConfig())}
    return {"axis": axis, "val": val, "label": _gs_short(axis, val), "wr": wr, "wr_ci": wr_ci,
            "fw": fw, "fw_ci": fw_ci, "solved": sum(1 for r in levels if r.get("solved_by")),
            "n_levels": len(levels), "best_gen": sum(bg) / len(bg) if bg else None,
            "rate": sum(ir) / len(ir) if ir else None, "time": _train_time(data),
            "n_params": make_net(GAConfig(**ga)).n_params, "ga": ga, "data": data}


def _hidden_verdict(points: list[dict]) -> dict:
    """OLS of first-win vs log2(weights) over the hidden-size axis (+ baseline).
    flat = the whole 8→64 range moves first-win by less than the noise floor."""
    import math
    import statistics
    pts = sorted((p for p in points if p["axis"] in ("base", "hidden")), key=lambda p: p["n_params"])
    if len(pts) < 3:
        return {"word": "insufficient", "change": 0.0, "ci": 0.0, "m": 0.0, "b": 0.0, "n": len(pts)}
    xs = [math.log2(p["n_params"]) for p in pts]
    ys = [p["fw"] for p in pts]
    if len(set(xs)) < 2:
        return {"word": "insufficient", "change": 0.0, "ci": 0.0, "m": 0.0, "b": 0.0, "n": len(pts)}
    m, b = statistics.linear_regression(xs, ys)
    change = m * (max(xs) - min(xs))
    ci = statistics.fmean([p["fw_ci"] for p in pts if p["fw_ci"] is not None] or [0.0])
    word = "flat" if abs(change) < max(ci, 1.0) else ("improves" if change < 0 else "degrades")
    return {"word": word, "change": change, "ci": ci, "m": m, "b": b, "n": len(pts)}


def _axis_best(pts: list[dict]) -> dict:
    return max(pts, key=lambda p: (round(p["wr"], 6), -p["fw"]))


def _gasweep_section(key: tuple, datas: list[dict], idx: int) -> str:
    game, persona, budget, sensors, _sig = key
    pts = [_sweep_point(d) for d in datas]
    by_axis: dict[str, list[dict]] = {}
    base = next((p for p in pts if p["axis"] == "base"), None)
    best_pt = next((p for p in pts if p["axis"] == "best"), None)
    for p in pts:
        if p["axis"] in AXES:
            by_axis.setdefault(p["axis"], []).append(p)
    sid = f"gs{idx}"
    seeds = sorted({c["seed"] for d in datas for lst in d.get("cells", {}).values() for c in lst})
    n_levels = max((p["n_levels"] for p in pts), default=0)
    total = sum(p["time"] for p in pts)
    from .evolution import GAConfig
    base_ga = base["ga"] if base else dict(vars(GAConfig()))

    # ── capacity curve ──
    verdict = _hidden_verdict(pts)
    vcol, vlabel = _VERDICT[verdict["word"]]
    scatter_pts = []
    for p in pts:
        if p["axis"] in ("base", "hidden", "action_feedback", "memory", "best"):
            kind = "hidden" if p["axis"] in ("base", "hidden") else "arch"
            color = "#e0442e" if p["axis"] == "base" else ("#cfcfcf" if kind == "hidden" else
                                                           "#f8b800" if p["axis"] != "best" else "#58d854")
            scatter_pts.append({"label": p["label"], "x": p["n_params"], "y": round(p["fw"], 2),
                                "ci": None if p["fw_ci"] is None else round(p["fw_ci"], 2), "kind": kind, "color": color, "star": p["axis"] == "base"})
    fit = ({"m": verdict["m"], "b": verdict["b"], "color": vcol.replace("var(--green)", "#58d85499")
            .replace("var(--blue)", "#cfcfcf99").replace("var(--red)", "#e4005899")}
           if verdict["word"] not in ("insufficient",) else None)
    hs = [p for p in pts if p["axis"] in ("base", "hidden")]
    rng = (f"{min(p['n_params'] for p in hs):,}→{max(p['n_params'] for p in hs):,} weights" if len(hs) > 1 else "")
    sentence = {
        "flat": f"Going from {rng} moves generations-to-first-win by <b>{verdict['change']:+.1f}</b> — inside the "
                f"±{max(verdict['ci'], 1.0):.1f} noise floor. <b>Bigger doesn't help.</b> The bottleneck is the level, not the brain.",
        "improves": f"Bigger nets win sooner: <b>{abs(verdict['change']):.1f} fewer generations</b> across {rng} "
                    f"(noise ±{verdict['ci']:.1f}). Capacity is part of the story on this game.",
        "degrades": f"Bigger nets win later: <b>{verdict['change']:+.1f} generations</b> across {rng} "
                    f"(noise ±{verdict['ci']:.1f}). More weights to search, no payoff.",
        "insufficient": "Run the <b>hidden</b> axis (8 / 16 / 32 / 64) to measure capacity against first win.",
    }[verdict["word"]]
    hero = f"""
      <div class="gshero">
        <div class="viz"><div class="vt">Capacity curve <span class="faint">— generations to first win (mean ± 95 % CI,
          unsolved = budget {budget}) vs network weights · <span style="color:#e0442e">●</span> baseline
          <span style="color:#cfcfcf">●</span> hidden size <span style="color:#f8b800">◆</span> feedback / memory
          {'<span style="color:#58d854">◆</span> recommended' if best_pt else ''}</span></div>
          <canvas class="scatter" width="860" height="360" data-points='{_json(scatter_pts)}'
            data-fit='{_json(fit)}' role="img"
            aria-label="Capacity curve for {game} {persona}: generations to first win against network weights"></canvas></div>
        <div class="viz" style="--c:{vcol}"><div class="vt">Verdict</div>
          <span class="verdict" style="--c:{vcol}">{vlabel}</span>
          <div class="vsent" style="--c:{vcol}">{sentence}</div>
          <div class="caption" style="margin-top:10px">Dashed line = least-squares fit of first-win on log₂(weights) over the
          hidden-size points. "Flat" means the fitted change across the range is smaller than the mean CI (or one generation).</div>
        </div>
      </div>"""

    # ── recommended config ──
    rec = best_config(datas)
    chips = []
    for axis in AXES:
        if axis not in rec:
            continue
        lbl, _, fmt = GA_DOC[axis]
        bp = next((p for p in by_axis[axis] if p["val"] == rec[axis]), None)
        d_wr = (bp["wr"] - base["wr"]) if (bp and base) else 0.0
        chips.append(f'<span class="recchip">{lbl} <s>{fmt(base_ga[axis])}</s> <b>{fmt(rec[axis])}</b>'
                     f'<em class="{"neg" if d_wr < 0 else ""}">{d_wr:+.0%} win rate</em></span>')
    cli = (f"python -m code.neuro.gasweep --game {game} --persona {persona} --gens {budget}"
           f"{' --sensors grid' if sensors == 'grid' else ''} --axes --confirm")
    if best_pt:
        best_line = (f'<div class="caption" style="margin-top:8px"><b style="color:var(--green)">Confirmed:</b> the composite '
                     f'won {best_pt["solved"]}/{best_pt["n_levels"]} levels · win rate <b>{best_pt["wr"]:.0%}</b> '
                     f'(baseline {base["wr"] if base else 0:.0%}) · first win gen <b>{best_pt["fw"]:.1f}</b> '
                     f'(baseline {base["fw"] if base else 0:.1f}) · {best_pt["n_params"]:,} weights</div>')
    else:
        best_line = ('<div class="caption" style="margin-top:8px">Not confirmed yet — the composite assumes the axes '
                     'do not interact. Probe it once:</div>'
                     f'<div class="cmdrow"><code class="cmd">{cli}</code><button class="copybtn" type="button">Copy</button></div>') if rec else ""
    rec_html = f"""
      <div class="viz gsrec"><div class="vt">Recommended config for {game} <span class="faint">— per-axis winners on win rate
        (ties → earliest first win), everything else at baseline</span></div>
        <div class="recrow">{''.join(chips) if chips else
          '<span class="recchip"><b>Baseline holds</b> — no axis beat it on win rate</span>'}</div>
        {best_line}</div>"""

    # ── axis cards (simple) + tables / curves (in-depth) ──
    cards, tables, curves = [], [], []
    for axis, vals in AXES.items():
        if axis not in by_axis:
            continue
        lbl, sub_, fmt = GA_DOC[axis]
        allpts = ([base] if base else []) + by_axis[axis]
        best = _axis_best(allpts)
        ref_wr = max(p["wr"] for p in allpts) or 1.0
        marks = []
        for i, v in enumerate(vals):
            x = 50 if len(vals) == 1 else i * 100 / (len(vals) - 1)
            p = next((q for q in allpts if q["val"] == v), None) if v != base_ga.get(axis) else base
            if p is None:
                marks.append(f'<span class="mk none" style="--x:{x:.0f}%;--s:12px;--c:var(--faint)"><i></i>'
                             f'<b>{fmt(v)}</b><small>not run</small></span>')
                continue
            is_base = p["axis"] == "base"
            col = "#e0442e" if is_base else "#cfcfcf"
            size = 10 + 16 * (p["wr"] / ref_wr)
            marks.append(f'<span class="mk{" best" if p is best else ""}" style="--x:{x:.0f}%;--s:{size:.0f}px;--c:{col}">'
                         f'<i></i><b>{fmt(v)}</b><small><strong>{p["wr"]:.0%}</strong>gen {p["fw"]:.1f}</small></span>')
        note = AXIS_DOC[axis] + (' First win is identical by construction — annealing only starts after a win; judged on win rate.'
                                 if axis == "anneal_factor" else "")
        cards.append(f"""
        <div class="axiscard"><h4>{lbl}<small>best: {fmt(best['val']) if best['axis'] != 'base' else fmt(base_ga[axis]) + ' (base)'}</small></h4>
          <p>{note}</p>
          <div class="rail">{''.join(marks)}</div>
          <div class="axisnote">low · base · high — marker size = win rate · ★ = best · label = win rate / mean first-win gen</div>
        </div>""")
        # in-depth table
        trs = []
        for p in sorted(allpts, key=lambda q: (q["axis"] == "base" and -1, q["val"] if isinstance(q["val"], (int, float)) else 0)):
            is_base = p["axis"] == "base"
            d_wr, d_ci, up, dn, _n = paired_delta(p["data"], base["data"]) if base and not is_base else (0.0, 0.0, 0, 0, 0)
            d_fw = p["fw"] - base["fw"] if base and not is_base else 0.0
            cls_wr = "delta-0" if abs(d_wr) < 0.005 else ("delta-g" if d_wr > 0 else "delta-r")
            cls_fw = "delta-0" if abs(d_fw) < 0.05 else ("delta-g" if d_fw < 0 else "delta-r")
            trs.append(f"<tr><td class='{'base' if is_base else ''}'>{fmt(base_ga[axis]) if is_base else fmt(p['val'])}"
                       f"{' (base)' if is_base else ''}{' ★' if p is best else ''}</td>"
                       f"<td>{p['n_params']:,}</td><td>{p['solved']}/{p['n_levels']}</td>"
                       f"<td>{p['fw']:.1f}{fmt_ci(p['fw_ci'], '.1f', ' ± ')}</td><td>{p['wr']:.0%}{fmt_ci(p['wr_ci'], '.0%', ' ± ')}</td>"
                       f"<td>{p['best_gen']:.1f}</td><td>{(p['rate'] or 0) * 100:.1f}%</td><td>{fmt_hms(p['time'])}</td>"
                       f"<td class='{cls_wr if not is_base else ''}'>{'-' if is_base else f'{d_wr:+.0%}' + fmt_ci(d_ci, '.0%', ' ± ')}"
                       f"{' ★' if not is_base and abs(d_wr) > (d_ci or 0) > 0 else ''}</td>"
                       f"<td>{'—' if is_base else f'{up}/{dn}'}</td>"
                       f"<td class='{cls_fw if not is_base else ''}'>{'—' if is_base else f'{d_fw:+.1f}'}</td></tr>")
        tables.append(f"""
        <div class="tbltitle">{lbl} <span class="faint">— {sub_}</span></div>
        <div class="ovwrap"><table class="gst"><thead><tr><th>Value</th><th>Weights</th><th>Levels solved</th>
          <th>First win (gen ± CI)</th><th>Win rate ± CI</th><th>Best gen</th><th>Progress / gen</th><th>Train time</th>
          <th>Δ win rate <small class="faint">paired ± CI</small></th><th>Cells ↑/↓</th><th>Δ first win</th></tr></thead><tbody>{''.join(trs)}</tbody></table></div>""")
        # learning curves per level, one colour per value (repeated per seed); baseline first, in the baseline red
        palette = ["#e0442e", "#cfcfcf", "#f8b800", "#f878f8", "#58d854", "#f87858"]
        boxes = []
        for lvl in sorted({l for p in allpts for l in p["data"].get("cells", {})}):
            refs, colors, legend = [], [], []   # the baseline's curves sit in every knob's box: stored once, referenced
            for ci_, p in enumerate(allpts):
                col = palette[ci_ % len(palette)]
                cs = p["data"].get("cells", {}).get(lvl, [])
                series = [c["curve"] for c in cs if c.get("curve")]
                if series:
                    ref = f"{idx}|{p['data'].get('tag')}|{lvl}"
                    _STORE["curves"][ref] = series
                    refs.append(ref)
                    colors += [col] * len(series)
                if cs:
                    legend.append(f'<span style="color:{col}">{p["label"]}</span>')
            if refs:
                boxes.append(f'<div class="viz"><div class="vt">{lvl} <span class="faint">— {" · ".join(legend)}</span></div>'
                             f'<canvas class="curve" width="560" height="220" data-cref=\'{_json(refs)}\' '
                             f'data-colors=\'{_json(colors)}\' role="img" aria-label="Learning curves on {lvl}, one colour per {lbl} value"></canvas></div>')
        if boxes:
            curves.append(f'<details class="cfg"><summary><span class="sumlbl">Learning curves — {lbl}</span>'
                          f'<span class="sumhint">best fitness per generation, one colour per value</span></summary>'
                          f'<div class="curvegrid">{"".join(boxes)}</div></details>')

    personas = {persona: base["data"] if base else datas[0]}
    # the card face: baseline -> confirmed composite, and the capacity verdict
    face_base = (f'<b class="num">{base["wr"]:.0%}</b><small>first win gen {base["fw"]:.1f}</small>' if base
                 else '<b class="num">—</b><small>no baseline</small>')
    if best_pt and base:
        cm, cci, *_ = paired_delta(best_pt["data"], base["data"])
        face_best = (f'<b class="num">{best_pt["wr"]:.0%}</b>'
                     f'<small>{cm:+.0%}{fmt_ci(cci, ".0%", " ± ")}{" ★" if abs(cm) > (cci or 0) > 0 else ""} vs baseline</small>')
    else:
        face_best = '<b class="num">—</b><small>not confirmed yet</small>'
    face = (f'<div class="gsface"><div class="gsnum"><small>Baseline win rate</small>{face_base}</div>'
            f'<div class="gsnum"><small>Composite</small>{face_best}</div>'
            f'<span class="verdict" style="--c:{vcol}">{vlabel}</span></div>')
    return f"""
  <section class="game frame stripe g-{game} collapsed" id="gs_{idx}" aria-labelledby="gs_{idx}_t">
    <div class="gamehead">
      <button class="chev" type="button" aria-expanded="false" aria-label="Expand {game} {persona}">▾</button>
      <h2 class="gametag" id="gs_{idx}_t">{_game_icon(game)}{game.capitalize()}</h2>{_persona_chip(persona)}
      <span class="gamemeta">{len(pts)} configs · {n_levels} levels · seeds {seeds} · {budget} gens/probe
        · {sensors} · train time {fmt_hms(total)} · baseline {_gs_baseline_label(base_ga)}</span>
    </div>
    {face}
    <div class="body">
      <div class="toolbar">
        <div class="tabs"><button class="tab active" data-view="{sid}_s" type="button" aria-selected="true">Simple</button>
          <button class="tab" data-view="{sid}_d" type="button" aria-selected="false">In-depth</button></div>
        <button class="brainbtn tip" type="button" data-dialog="brain_{sid}"
          data-tip="the baseline GAConfig every axis is measured against">{_BRAIN_SVG} Baseline hyperparameters</button>
      </div>
      <div class="view active" id="{sid}_s">
        {hero}
        {rec_html}
        <div class="tbltitle" style="margin-top:14px">One knob at a time <span class="faint">— each card moves one GAConfig
          field to its literature low / high bound; everything else stays at baseline (red)</span></div>
        <div class="gsaxes">{''.join(cards)}</div>
      </div>
      <div class="view" id="{sid}_d">
        {''.join(tables)}
        {''.join(curves)}
      </div>
      {_brain_dialog(sid, game, personas)}
    </div>
  </section>"""


def _gasweep_page(groups: dict[tuple, list[dict]], games: list[str], logo: str | None, stamp: str,
                  nav: bool = True) -> str:
    n_cfg = sum(len(v) for v in groups.values())
    n_probes = sum(len(lst) for v in groups.values() for d in v for lst in d.get("cells", {}).values())
    total = sum(_train_time(d) for v in groups.values() for d in v)
    verdicts = {}
    for key, datas in groups.items():
        verdicts[key] = _hidden_verdict([_sweep_point(d) for d in datas])["word"]
    flat = sum(1 for w in verdicts.values() if w == "flat")
    measured = sum(1 for w in verdicts.values() if w != "insufficient")
    vcol = "var(--green)" if measured and flat == measured else ("var(--yellow)" if measured else "var(--faint)")
    kpi = lambda icon, lbl, val, sub: (f'<div class="stat"><div class="lbl"><i class="ico {icon}"></i> {lbl}</div>'  # noqa: E731
                                       f'<div class="val">{val}</div><div class="sub">{sub}</div></div>')
    kpis = (f'<div class="kpis">'
            + kpi("gear", "Sweeps", len(groups), "game × persona × budget")
            + kpi("dna", "Configs", n_cfg, "baseline + one-knob variants")
            + kpi("gamepad", "Probes", n_probes, "level × seed cells")
            + kpi("hourglass", "Train time", fmt_hms(total), "all probes")
            + f'<div class="stat accent" style="--c:{vcol}"><div class="lbl">Capacity verdict</div>'
            f'<div class="val">{flat}/{measured} flat</div><div class="sub">sweeps where a bigger net did not help</div></div>'
            f'</div>') if groups else ""
    sections = _gasweep_overview(groups) + (
        '<div class="tbltitle">One card per sweep <span class="faint">— open one for its capacity curve, recommended '
        'config and every knob</span></div><div class="cardgrid">'
        + "".join(_gasweep_section(key, datas, i) for i, (key, datas) in enumerate(groups.items())) + "</div>")
    if not groups:
        sections = ('<div class="viz"><div class="vt">No GA sweep yet</div><p style="color:var(--dim)">Run <b>15 GA Sweep</b> from '
                    '<code>python menu.py</code>, or <code>python -m code.neuro.gasweep --game mario --gens 40 --confirm</code>. '
                    'Each config is probed on every enabled level × 3 seeds; results land here as one section per game × persona.</p></div>')
    axes_doc = "".join(f'<li><b>{GA_DOC[a][0]}</b> <code>{" · ".join(map(str, vs))}</code> — {AXIS_DOC[a]}</li>'
                       for a, vs in AXES.items())
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'><rect width='16' height='16' fill='%23000000'/><circle cx='8' cy='8' r='4' stroke='%2358d854' stroke-width='2' fill='none'/></svg>">
<title>PEAK Command Center: GA sweep</title><style>{CSS}</style></head><body>
{_nav(games, False, logo, page="gasweep") if nav else ""}
<main id="main">
  <div class="pagehead"><h2><i class="ico gear"></i>GA sweep</h2>
    <p>GA hyperparameter ablation · one knob at a time against literature bounds · same seeds, same levels · generated {stamp}</p></div>
  {kpis}
  {sections}
  <details class="cfg" style="margin-top:18px"><summary><span class="sumlbl">The axes and where the bounds come from</span>
    <span class="sumhint">De Jong 1975 · Grefenstette 1986 · Schaffer 1989 · Miller &amp; Goldberg 1995 · Such et al. 2017</span></summary>
    <ul style="font:400 .8rem/1.6 var(--ui);color:var(--dim);margin:10px 0 0 18px">{axes_doc}</ul>
    <p class="caption" style="margin-top:8px">One-factor-at-a-time isolates each knob's effect but ignores interactions —
    the recommended config is the per-axis composite; <code>--confirm</code> probes it once to check.</p></details>
  <footer>Generated by PEAK from the GA-sweep cells on disk. Regenerate from the menu (12) after a new sweep.</footer>
</main><div class="bricks" aria-hidden="true"></div>{_page_data()}<script>{JS}</script></body></html>"""


MEGAMAN_REPORTS = os.path.join("runs", "ast", "megaman", "base-retrained")   # Mega Man's probes: every section retrained


def load_data(balance_dir: str = BALANCE_DIR) -> dict:
    """What the Command Center's sub-pages show, read from the committed report JSONs; nothing is
    rebuilt, so the pages carry the published numbers. games: game -> tag -> persona -> report."""
    def _flat(*patterns: str) -> dict[tuple[str, str | None], dict[str, dict]]:  # (game, tag) -> persona -> data
        out: dict[tuple[str, str | None], dict[str, dict]] = {}
        for pattern in patterns:
            for data in _load_json_glob(pattern):
                out.setdefault((data["game"], data.get("tag")), {})[data.get("persona") or "experienced"] = data
        return out
    flat = _flat(os.path.join(balance_dir, "report_*.json"), os.path.join(MEGAMAN_REPORTS, "report_megaman_*.json"))
    games: dict[str, dict] = {}  # tagged configs first, legacy last
    for (g, t), personas in sorted(flat.items(), key=lambda kv: (kv[0][0], kv[0][1] is None, kv[0][1] or "")):
        games.setdefault(g, {})[t] = personas
    return {"games": games, "ablation": _flat(os.path.join(balance_dir, "ablation_*.json")),
            "gasweep": _gasweep_groups(_load_json_glob(os.path.join(balance_dir, "gasweep_*.json"))),
            "thresholds": _load_thresholds(), "stamp": datetime.now().strftime("%Y-%m-%d %H:%M")}


def _scoped(html: str) -> str:
    """Every <th> on these pages heads a column; label the ones the emitters didn't."""
    return re.sub(r"<th(?=[\s>])(?![^>]*\bscope=)", '<th scope="col"', html)


def balance_page(game: str, data: dict) -> str:
    """The Balance tab of one game: what the balance report showed for it, with no top bar."""
    by_tag = data["games"].get(game)
    has_sweep = any(key[0] == game for key in data["gasweep"])
    body = (GLOSSARY + _game_section(game, by_tag, 0, data["thresholds"], "gasweep" if has_sweep else None) if by_tag else
            f"<p style='color:var(--dim)'>No balance data for {game} yet: run a Full Sweep (menu 13) first.</p>")
    body += _untrained(game, by_tag or {})
    return _scoped(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PEAK Command Center: {game} balance</title><style>{CSS}</style></head><body>
<main id="main">
  {body}
</main><div class="bricks" aria-hidden="true"></div>{TIP_DESC}{_page_data()}<script>{JS}</script></body></html>""")


def _untrained(game: str, by_tag: dict) -> str:
    """A card for every level the game has (practice and disabled ones too) that no agent trained on."""
    trained = {str(r["level"]) for personas in by_tag.values() for d in personas.values() for r in d["levels"]}
    levels = [lv for lv in map(str, list_levels(game, include_disabled=True)) if lv not in trained]
    if not levels:
        return ""
    cards = "".join(f'<div class="lvlcard untrained" data-untrained="{_html.escape(lv)}"><div class="cardbody">'
                    f'<div class="lvlname">{_html.escape(lv)} {_status_pill("bad", "not trained yet")}</div>'
                    f'<div class="lvlsub">no agent has trained on this level</div></div></div>' for lv in levels)
    return (f'<section class="untrainedsec frame stripe g-{game}" aria-label="Levels not trained yet"><div class="body">'
            f'<div class="tbltitle">Not trained yet<span class="faint">: {len(levels)} level{"s" if len(levels) != 1 else ""} '
            f'of this game with no agent trained on them</span></div><div class="lvlgrid">{cards}</div></div></section>')


def gasweep_page(game: str, data: dict) -> str:
    """The GA sweep tab of one game."""
    groups = {k: v for k, v in data["gasweep"].items() if k[0] == game}
    return _scoped(_gasweep_page(groups, [game], None, data["stamp"], nav=False))


def ablation_page(game: str, data: dict) -> str:
    """The Sensor ablation tab of one game."""
    arms = {k: v for k, v in data["ablation"].items() if k[0] == game}
    return _scoped(_ablation_page(arms, None, data["stamp"], nav=False))


def instructions_page(data: dict) -> str:
    """The How it works tab: the instructions, then every training run on this machine."""
    return _scoped(_instructions_page(list(data["games"]), None, nav=False, after=_runs_section()))


def _load_json_glob(pattern: str) -> list[dict]:
    out = []
    for f in sorted(glob.glob(pattern)):
        try:
            with open(f, encoding="utf-8") as fh:
                out.append(json.load(fh))
        except (OSError, json.JSONDecodeError):
            continue
    return out
