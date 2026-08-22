"""README figures from runs/balance/*.json (menu 16).

    python -m code.neuro.figures            # docs/img/fig_{difficulty,capacity,sensors,knobs}.png
    python -m code.neuro.figures --readme   # + refresh the stamp between the README figure markers

Same data the command center renders: report_* (Full Sweep), ablation_* (Sensor Ablation),
gasweep_* (GA Sweep). Figures that have no data yet are skipped, not faked.
"""
import argparse
import glob
import os
import re
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from .balance import BALANCE_DIR, mean_ci
from .gasweep import AXES, paired_delta
from .report import GA_DOC, _ablation_pairs, _gasweep_groups, _gs_baseline_label, _load_json_glob, _sweep_point

IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "docs", "img")
README = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "README.md")
MARK_OPEN, MARK_CLOSE = "<!-- figures:auto -->", "<!-- /figures:auto -->"

BG, PANEL, TXT, DIM, FAINT, GRID = "#070708", "#101012", "#e8e8ea", "#9a9aa0", "#606066", "#232326"
PCOL = {"novice": "#4a9eff", "experienced": "#ef4444", "speedrunner": "#22c55e"}
PORDER = ["novice", "experienced", "speedrunner"]
GNAME = {"mario": "Mario", "meatboy": "Meat Boy", "megaman": "Mega Man", "sonic": "Sonic"}

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
    "axes.edgecolor": GRID, "axes.labelcolor": DIM, "xtick.color": DIM, "ytick.color": DIM,
    "text.color": TXT, "grid.color": GRID, "axes.grid": True, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11,
    "legend.frameon": False, "legend.labelcolor": TXT,
})


def _level_label(game: str, level: str) -> str:
    if level.isdigit():
        return f"L{level}"
    return re.sub(rf"^{game}", "", level, flags=re.I) or level


def _personas(datas: list[dict]) -> list[str]:
    seen = {d.get("persona") or "experienced" for d in datas}
    return [p for p in PORDER if p in seen] + sorted(seen - set(PORDER))


# ── 1. difficulty: win rate per level per persona ───────────────────────────

def fig_difficulty(reports: list[dict], out: str) -> bool:
    rays = [d for d in reports if "_" not in (d.get("tag") or "")]
    if not rays:
        return False
    by_game: dict[str, list[dict]] = {}
    for d in rays:
        by_game.setdefault(d["game"], []).append(d)
    panels = []
    for game, ds in by_game.items():
        budget = max(d.get("gens_budget") or 0 for d in ds)  # the deepest sweep per game
        panels.append((game, [d for d in ds if (d.get("gens_budget") or 0) == budget]))
    personas = _personas([d for _, ds in panels for d in ds])
    widths = [max(1, len({r["level"] for d in ds for r in d["levels"]})) for _, ds in panels]
    fig, axes = plt.subplots(1, len(panels), figsize=(min(16, 3 + 0.95 * sum(widths)), 6),
                             gridspec_kw={"width_ratios": widths}, squeeze=False, sharey=True)
    w = 0.8 / max(1, len(personas))
    for ax, (game, ds) in zip(axes[0], panels):
        levels = sorted({r["level"] for d in ds for r in d["levels"]}, key=lambda s: [int(t) if t.isdigit() else t
                                                                                  for t in re.split(r"(\d+)", s)])
        for j, persona in enumerate(personas):
            rows = {r["level"]: r for d in ds if (d.get("persona") or "experienced") == persona for r in d["levels"]}
            xs = [i + (j - (len(personas) - 1) / 2) * w for i in range(len(levels))]
            ys = [rows[lv]["win_rate_mean"] * 100 if lv in rows else 0 for lv in levels]
            es = [rows[lv]["win_rate_ci"] * 100 if lv in rows else 0 for lv in levels]
            ax.bar(xs, ys, w, color=PCOL.get(persona, DIM), label=persona, yerr=es,
                   error_kw={"ecolor": DIM, "elinewidth": 1, "capsize": 0})
        ax.set_xticks(range(len(levels)), [_level_label(game, lv) for lv in levels])
        ax.set_title(GNAME.get(game, game), loc="left", fontweight="bold", color=TXT)
        ax.set_xlabel("level")
        ax.set_ylim(0, 100)
        ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel("win rate after first win")
    axes[0][0].set_yticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100 %"])
    d0 = panels[0][1][0]
    fig.suptitle("How hard is each level?", x=0.02, ha="left", fontsize=16, fontweight="bold")
    fig.text(0.02, 0.91, f"population {d0.get('pop_size')} · {d0.get('gens_budget')} generations · "
             f"{len(d0.get('seeds') or [])} seeds · whiskers = 95 % CI", color=DIM, fontsize=10)
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="upper right", ncol=len(personas), bbox_to_anchor=(0.99, 0.99))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return True


# ── 2. capacity: generations to first win vs hidden size ────────────────────

def fig_capacity(groups: dict, out: str) -> bool:
    series = []
    for (game, persona, budget, sensors, _sig), datas in groups.items():
        pts = sorted((p for p in map(_sweep_point, datas) if p["axis"] in ("base", "hidden")), key=lambda p: p["n_params"])
        if len(pts) >= 3:
            series.append((game, persona, budget, pts))
    if not series:
        return False
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    budget = max(s[2] or 0 for s in series)
    ax.axhline(budget, ls=":", color=FAINT, lw=1)
    ax.text(0.99, budget, f"budget ({budget} gens)", ha="right", va="bottom", color=DIM, fontsize=10, transform=ax.get_yaxis_transform())
    games = list(dict.fromkeys(s[0] for s in series))
    styles = ["-", "--", "-.", ":"]
    ticks: dict[int, int] = {}
    for game, persona, _b, pts in series:
        xs = [p["n_params"] for p in pts]
        for p in pts:
            ticks[p["n_params"]] = p["ga"]["hidden"]
        ys, cis = [p["fw"] for p in pts], [p["fw_ci"] for p in pts]
        col = PCOL.get(persona, DIM)
        ax.plot(range(len(xs)), ys, styles[games.index(game) % 4], color=col, marker="o", lw=2,
                label=f"{GNAME.get(game, game)} · {persona}")
        ax.fill_between(range(len(xs)), [y - c for y, c in zip(ys, cis)], [y + c for y, c in zip(ys, cis)],
                        color=col, alpha=0.07, lw=0)
    xt = sorted(ticks)
    ax.set_xticks(range(len(xt)), [f"{n}\nh = {ticks[n]}" for n in xt])
    ax.set_xlabel("network weights (hidden size)")
    ax.set_ylabel("generations to first win (mean over level × seed)")
    ax.set_ylim(0, budget * 1.08)
    ax.set_title("Bigger nets don't win sooner", loc="left", fontsize=16, fontweight="bold", pad=14)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=10)
    style_names = {"-": "solid", "--": "dashed", "-.": "dash-dot", ":": "dotted"}
    fig.text(0.02, 0.015, " · ".join(f"{style_names[styles[i % 4]]} = {GNAME.get(g, g)}" for i, g in enumerate(games))
             + " · unsolved cells censored at the budget · shaded = 95 % CI", color=DIM, fontsize=9.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return True


# ── 3. sensors: paired Δ win rate, grid − rays ──────────────────────────────

def fig_sensors(ablations: list[dict], out: str) -> bool:
    flat: dict = {}
    for d in ablations:
        flat.setdefault((d["game"], d.get("tag")), {})[d.get("persona") or "experienced"] = d
    paired = [(k, v) for k, v in sorted(_ablation_pairs(flat).items()) if len(v) == 2]
    if not paired:
        return False
    rows = []
    for (game, persona, _base), by_mode in paired:
        m, ci, _up, _dn, _n = paired_delta(by_mode["grid"], by_mode["rays"])
        rows.append((f"{GNAME.get(game, game)} · {persona}", m * 100, ci * 100, PCOL.get(persona, DIM)))
    rows.sort(key=lambda r: (r[0].split(" · ")[0], PORDER.index(r[0].split(" · ")[1]) if r[0].split(" · ")[1] in PORDER else 9))
    fig, ax = plt.subplots(figsize=(9.5, 1.1 + 0.62 * len(rows)))
    ys = range(len(rows))[::-1]
    ax.barh(list(ys), [r[1] for r in rows], 0.55, color=[r[3] for r in rows],
            xerr=[r[2] for r in rows], error_kw={"ecolor": DIM, "elinewidth": 1.3, "capsize": 4})
    ax.axvline(0, color=TXT, lw=1)
    ax.set_yticks(list(ys), [r[0] for r in rows])
    ax.xaxis.set_major_formatter(lambda v, _p: f"{v:+.0f}%" if v else "0%")
    ax.set_xlabel("Δ win rate, grid − rays  (paired, ± 95 % CI)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Rays (14 inputs) vs tile grid (368 inputs)", loc="center", fontsize=16, fontweight="bold", pad=24)
    ax.text(0, 1.02, "◀ rays better", transform=ax.transAxes, color=PCOL["experienced"], fontsize=10, va="bottom")
    ax.text(1, 1.02, "grid better ▶", transform=ax.transAxes, color=PCOL["speedrunner"], fontsize=10, va="bottom", ha="right")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return True


# ── 4. knobs: heat-map of paired Δ win rate per GA knob value ────────────────

def fig_knobs(groups: dict, out: str) -> bool:
    cols = []
    for (game, persona, budget, _sensors, _sig), datas in groups.items():
        pts = [_sweep_point(d) for d in datas]
        base = next((p for p in pts if p["axis"] == "base"), None)
        if base:
            cols.append((game, persona, budget, base, {(p["axis"], p["val"]): p for p in pts}))
    if not cols:
        return False
    base_ga = cols[0][3]["ga"]
    rows = [(axis, v) for axis, vals in AXES.items() for v in vals if v != base_ga.get(axis)]
    rows = [r for r in rows if any(r in c[4] for c in cols)]
    if not rows:
        return False
    vals = [[None] * len(cols) for _ in rows]
    for i, key in enumerate(rows):
        for j, (_g, _p, _b, base, pts) in enumerate(cols):
            if key in pts:
                m, ci, _up, _dn, _n = paired_delta(pts[key]["data"], base["data"])
                vals[i][j] = (m, ci)
    vmax = max((abs(v[0]) for row in vals for v in row if v), default=0.1) or 0.1
    cmap = LinearSegmentedColormap.from_list("peak", [PCOL["experienced"], "#7a2326", "#2a1214", BG, "#0f2a18", "#176b36",
                                                      PCOL["speedrunner"]])
    fig, ax = plt.subplots(figsize=(2.2 + 1.15 * len(cols) + 2.4, 1.6 + 0.42 * len(rows)))
    grid = [[(v[0] if v else float("nan")) for v in row] for row in vals]
    im = ax.imshow(grid, cmap=cmap, norm=TwoSlopeNorm(0, -vmax, vmax), aspect="auto")
    for i, row in enumerate(vals):
        for j, v in enumerate(row):
            if v is None:
                ax.text(j, i, "—", ha="center", va="center", color=FAINT)
                continue
            m, ci = v
            sig = abs(m) > ci > 0
            ax.text(j, i, f"{m:+.0%}" + ("★" if sig else ""), ha="center", va="center", fontsize=10.5,
                    color=TXT, fontweight="bold" if sig else "normal")
    ax.set_xticks(range(len(cols)), [f"{GNAME.get(g, g)}\n{p}" for g, p, *_ in cols], fontsize=10)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(rows)), [f"{GA_DOC[a][0]}  {GA_DOC[a][2](v)}" for a, v in rows], fontsize=10.5)
    ax.grid(False)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([x - 0.5 for x in range(1, len(cols))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(rows))], minor=True)
    ax.grid(which="minor", color=BG, lw=3)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cb.set_ticks([-vmax, 0, vmax], labels=[f"{-vmax:+.0%}", "0", f"{vmax:+.0%}"])
    cb.outline.set_visible(False)
    fig.suptitle("One knob at a time — Δ win rate vs the baseline", fontsize=16, fontweight="bold", y=0.985)
    fig.text(0.5, 0.012, f"Paired on identical level × seed cells · ★ = clears its 95 % CI · {cols[0][2]} gens · "
             f"baseline: {_gs_baseline_label(base_ga)}", ha="center", color=DIM, fontsize=9.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return True


# ── README stamp ─────────────────────────────────────────────────────────────

def update_readme(made: list[str], counts: dict[str, int]) -> bool:
    with open(README, encoding="utf-8") as f:
        text = f.read()
    if MARK_OPEN not in text or MARK_CLOSE not in text:
        return False
    stamp = (f"{MARK_OPEN}\nEverything below comes straight out of `runs/balance/*.json` — the same data the command "
             f"center renders. Figures regenerated {date.today().isoformat()} from {counts['report']} sweep reports, "
             f"{counts['ablation']} ablation arms and {counts['gasweep']} GA-sweep configs "
             f"(`python menu.py` → 16: {', '.join(made) or 'none'}).\n{MARK_CLOSE}")
    head, _, rest = text.partition(MARK_OPEN)
    _, _, tail = rest.partition(MARK_CLOSE)
    with open(README, "w", encoding="utf-8", newline="") as f:
        f.write(head + stamp + tail)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="README figures from runs/balance")
    ap.add_argument("--dir", default=BALANCE_DIR)
    ap.add_argument("--out", default=IMG_DIR)
    ap.add_argument("--readme", action="store_true", help="refresh the stamp between the README figure markers")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    reports = _load_json_glob(os.path.join(args.dir, "report_*.json"))
    ablations = _load_json_glob(os.path.join(args.dir, "ablation_*.json"))
    gasweeps = _load_json_glob(os.path.join(args.dir, "gasweep_*.json"))
    groups = _gasweep_groups(gasweeps)
    made = []
    for name, fn, data in (("difficulty", fig_difficulty, reports), ("capacity", fig_capacity, groups),
                           ("sensors", fig_sensors, ablations), ("knobs", fig_knobs, groups)):
        path = os.path.join(args.out, f"fig_{name}.png")
        if fn(data, path):
            made.append(f"fig_{name}.png")
            print(f"wrote {os.path.relpath(path)}")
        else:
            print(f"skipped fig_{name}.png — no data in {args.dir}")
    if args.readme:
        ok = update_readme(made, {"report": len(reports), "ablation": len(ablations), "gasweep": len(gasweeps)})
        print("updated README.md" if ok else f"README.md has no {MARK_OPEN} … {MARK_CLOSE} markers — stamp not written")


if __name__ == "__main__":
    main()
