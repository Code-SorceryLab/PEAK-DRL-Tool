"""The AST paper's tables and figures (paper-ast/): what retraining, replay, a random-action agent
and the static reachability check catch; which persona catches what; whether failures land on the
change; whether play reached it; and why each miss was missed.

    python -m code.stats.ast_paper      # paper-ast/ast_results.md, tables/*.tex, figures/*.pdf|png

Everything is read from the playtests on disk through code/stats/ast_metrics.py, so rerun it
after any new playtest. Figures are drawn on white for print (the dashboards stay black and red).
"""
from __future__ import annotations

import os

from code.games import eval_mutants
from code.stats import ast_metrics as am
from code.stats import ast_page

OUT = "paper-ast"
GAME = {"mario": "Mario 1-1", "meatboy": "Meat Boy 1-3", "bomberman": "Bomberman 09", "megaman": "Mega Man Cut Man"}
NEAR = 3                                   # tiles: the death cluster's own radius
MODES = ("static", "random", "replay")              # the detectors compared; retraining only answers "still solvable?"
HEAD = {"static": "Static", "random": "Random", "replay": "Replay (detector)", "solvable": "Still solvable (retrained)"}


# ── small pieces (tested) ────────────────────────────────────────────────────────────────────
def nearest(cells: list, tile) -> int | None:
    """Chebyshev distance in tiles from `tile` to the nearest changed cell."""
    if not cells or tile is None:
        return None
    return min(max(abs(c[0] - tile[0]), abs(c[1] - tile[1])) for c in cells)


def share_near(cells: list, tiles: list, radius: int = NEAR) -> float | None:
    """The share of failed attempts that ended within `radius` tiles of a changed cell."""
    if not cells or not tiles:
        return None
    return round(sum(nearest(cells, t) <= radius for t in tiles) / len(tiles), 3)


def reached(cells: list, coverage: list, radius: int = 1) -> bool | None:
    """Did the original's routes pass within `radius` tiles of the change? None for mechanics."""
    if not cells:
        return None
    seen = {tuple(t) for t in coverage}
    return any((c[0] + dx, c[1] + dy) in seen for c in cells
               for dx in range(-radius, radius + 1) for dy in range(-radius, radius + 1))


def right(cls: str, flagged: bool | None) -> bool | None:
    """Did a method get this mutant right: a play-changing one flagged, a neutral one left alone?"""
    if flagged is None:
        return None
    return flagged if cls != "neutral" else not flagged


def tally(rows: list[dict], mode: str) -> dict:
    """{caught: [n, of], quiet: [n, of]} for one method over the mutants it judged."""
    out = {"caught": [0, 0], "quiet": [0, 0]}
    for r in rows:
        ok = r[mode]
        if ok is None or r["operator"].endswith("cosmetic_tile"):
            continue
        k = "quiet" if r["class"] == "neutral" else "caught"
        out[k][0] += bool(ok)
        out[k][1] += 1
    return out


# ── the analyses ─────────────────────────────────────────────────────────────────────────────
def judgments(results: dict) -> list[dict]:
    """One row per mutant: whether each method got it right, and the words the page shows."""
    rows = []
    for game, res in results.items():
        for m in eval_mutants.eval_set(game):
            e = res.get(m["id"]) or {}
            row = {"game": game, "id": m["id"], "class": m["class"], "operator": m["operator"],
                   "slot": ast_page.SLOT_NAMES[ast_page._slot(m["id"])]}
            row["static"] = right(m["class"], not (m.get("checks") or {}).get("exit_reachable", True))
            rnd = (e.get("random") or {}).get("vs_base")
            row["random"] = right(m["class"], any(rnd[k] for k in am.SIGNALS)) if rnd else None   # never counted: it never wins
            for mode in ("replay", "retrain"):
                word, tone = ast_page.judge(m, res, replay=mode == "replay")
                row[mode] = None if tone == "none" else tone == "good"
                row[mode + "_word"] = word
                row[mode + "_strict"] = right(m["class"], strict_flag(m, res, mode))
            word, tone = ast_page.solvable(m, res)
            row["solvable"], row["solvable_word"] = (None if tone == "none" else tone == "good"), word
            if m["operator"].endswith("cosmetic_tile"):
                row["replay"], row["replay_word"] = None, "-"
                row["replay_strict"] = row["retrain_strict"] = None
            rows.append(row)
    return rows


def solvability(rows: list[dict]) -> dict:
    """{breaking mutants found no longer solvable, other mutants found still solvable}: [right, of]."""
    out = {"breaking": [0, 0], "others": [0, 0]}
    for r in rows:
        if r["solvable"] is None:
            continue
        k = "breaking" if r["class"] == "breaking" else "others"
        out[k][0] += bool(r["solvable"])
        out[k][1] += 1
    return out


def strict_flag(m: dict, res: dict, mode: str) -> bool | None:
    """Flagged by the two signals that never fired on an unchanged 2D level (completion lost or
    Fisher's test), over the counted level or sections; None when nothing counts."""
    parts = [e for k, e in res.items() if k.startswith(m["id"] + " @ ")] or [res.get(m["id"]) or {}]
    sigs = [((e.get("replay") or {}).get("vs_base") if mode == "replay" else e.get("vs_base")) for e in parts]
    counted = [s for s in sigs if s and s.get("counted")]
    return any(s["completion_lost"] or s["win_rate_changed"] for s in counted) if counted else None


def personas(results: dict) -> list[dict]:
    """Per 2D game, mutant and persona: the verdict of that persona's three agents alone."""
    out = []
    for game in results:
        if game == "megaman":
            continue                        # one persona only
        base = [r for r in am.load(game, "base") if r.get("source") == "agent"]
        for m in eval_mutants.eval_set(game):
            if m["operator"].endswith("cosmetic_tile"):
                continue
            row = {"game": game, "id": m["id"], "class": m["class"]}
            for mode in ("retrain", "replay"):
                mut = [r for r in am.load(game, m["id"], mode) if r.get("source") == "agent"]
                for p in am_personas(base):
                    b = [r for r in base if r.get("persona") == p]
                    x = [r for r in mut if r.get("persona") == p]
                    if b and x:
                        row[(mode, p)] = am.compare(am.summarize(b), am.summarize(x))["verdict"]
            out.append(row)
    return out


def am_personas(rows: list[dict]) -> list[str]:
    from code.neuro.ast_eval import PERSONA_ORDER
    have = {r.get("persona") for r in rows}
    return [p for p in PERSONA_ORDER if p in have]


def localization(results: dict) -> list[dict]:
    """Level mutants: did play reach the change, and do the failures end on it?"""
    out = []
    for game, res in results.items():
        main = eval_mutants.EVAL_LEVELS[game][0]
        for m in eval_mutants.eval_set(game):
            cells = [(c[0], c[1]) for c in m.get("cells") or []]
            if not cells or m["operator"].endswith("cosmetic_tile"):
                continue
            e = res.get(m["id"]) or {}
            vs = e.get("vs") or "base"
            ref = (res.get(vs) or {}).get("agents") or {}
            row = {"game": game, "id": m["id"], "class": m["class"], "level": e.get("level") or main,
                   "reached": reached(cells, ref.get("coverage") or []),
                   "orig_near": share_near(cells, ref.get("end_tiles") or [])}
            for mode in ("retrain", "replay"):
                a = (e.get("agents") if mode == "retrain" else (e.get("replay") or {}).get("agents")) or {}
                c = a.get("death_cluster")
                row[mode + "_cluster_dist"] = nearest(cells, (c["col"], c["row"]) if c else None)
                row[mode + "_near"] = share_near(cells, a.get("end_tiles") or [])
            out.append(row)
    return out


def misses(results: dict, rows: list[dict]) -> list[str]:
    """One plain sentence per wrong call (the detector, and retraining's solvability answer)."""
    out = []
    for r in rows:
        if r["solvable"] is False:
            want = "no longer solvable" if r["class"] == "breaking" else "still solvable"
            res = results[r["game"]]
            lost = [k.split(" @ ")[1] for k, e in res.items() if k.startswith(r["id"] + " @ ") and e.get("agents")
                    and (e.get("vs_base") or {}).get("counted") and not e["agents"]["completion"]]
            null = [s for s in lost if not ((res.get(f"{am.RESEEDED} @ {s}") or {}).get("agents") or {}).get("completion", True)]
            why = (f" Unsolved after retraining: {', '.join(lost)}; the unchanged level also failed to train from new seeds "
                   f"there ({', '.join(null)}), so this is likely seed noise." if null else
                   f" Unsolved after retraining: {', '.join(lost)}." if lost else "")
            out.append(f"{GAME[r['game']]} `{r['id']}` (retrained): {r['solvable_word']}, expected {want}.{why}")
        for mode in ("replay",):
            if r[mode] is not False:
                continue
            e = results[r["game"]].get(r["id"]) or {}
            part = e if mode == "retrain" else e.get("replay") or {}
            s, a = part.get("vs_base") or {}, part.get("agents") or {}
            ref = (results[r["game"]].get(e.get("vs") or "base") or {}).get("agents") or {}
            if not s or not a or not ref:
                out.append(f"{GAME[r['game']]} {r['id']} ({mode}): judged section by section; see the page.")
                continue
            bw, bn = ref["win_rate"]["pooled"]
            mw, mn = a["win_rate"]["pooled"]
            bc, mc = ref.get("death_cluster"), a.get("death_cluster")
            moved = (max(abs(bc["col"] - mc["col"]), abs(bc["row"] - mc["row"])) if bc and mc else None)
            what = "a false alarm" if r["class"] == "neutral" else "missed"
            out.append(f"{GAME[r['game']]} `{r['id']}` ({mode}, {what}): wins {mw}/{mn} vs {bw}/{bn} on the original "
                       f"(Fisher p = {s['win_rate_p']}), death cluster moved {moved if moved is not None else '-'} "
                       f"tile(s), top cause {next(iter(ref.get('causes') or {}), '-')} -> {next(iter(a.get('causes') or {}), '-')}.")
    return out


# ── writing ──────────────────────────────────────────────────────────────────────────────────
def _mark(ok: bool | None) -> str:
    return "-" if ok is None else "yes" if ok else "no"


def _tex(s: str) -> str:
    return str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def _tex_mark(ok: bool | None) -> str:
    return "--" if ok is None else r"\checkmark" if ok else r"$\times$"


def detection_tex(rows: list[dict]) -> str:
    """Table: every mutant, what each detector got right, and whether retrained agents still finish
    it as they should. Needs booktabs and amssymb."""
    cols = (*MODES, "solvable")
    body = [f"{_tex(GAME[r['game']])} & \\texttt{{{_tex(r['id'])}}} & {r['class']} & "
            + " & ".join(_tex_mark(r[m]) for m in cols) + r" \\" for r in rows]
    tot = {m: tally(rows, m) for m in MODES}
    sv = solvability(rows)
    caught = " & ".join(f"{tot[m]['caught'][0]}/{tot[m]['caught'][1]}" for m in MODES)
    quiet = " & ".join(f"{tot[m]['quiet'][0]}/{tot[m]['quiet'][1]}" for m in MODES)
    return ("\\begin{table*}[t]\\centering\\small\n\\caption{Mutants each detector gets right: a breaking or degrading "
            "mutant flagged, a neutral one left alone (\\checkmark{}), or not ($\\times$). Replay is the original's frozen "
            "agents on the mutant. The last column asks agents retrained on the mutant whether it can still be finished: "
            "right when a breaking mutant no longer can and any other still can.}\n"
            "\\label{tab:detection}\n\\begin{tabular}{lll" + "c" * len(cols) + "}\\toprule\nGame & Mutant & Class & "
            + " & ".join(HEAD[m] for m in cols) + " \\\\\\midrule\n"
            + "\n".join(body) + "\n\\midrule\n"
            f"\\multicolumn{{3}}{{l}}{{Play-changing mutants caught}} & {caught} & -- \\\\\n"
            f"\\multicolumn{{3}}{{l}}{{Neutral mutants left alone}} & {quiet} & -- \\\\\n"
            f"\\multicolumn{{3}}{{l}}{{Solvability right (breaking / others)}} & & & & "
            f"{sv['breaking'][0]}/{sv['breaking'][1]}, {sv['others'][0]}/{sv['others'][1]} \\\\\n"
            "\\bottomrule\n\\end{tabular}\n\\end{table*}\n")


def write(results: dict, out: str = OUT) -> list[str]:
    rows = judgments(results)
    per = personas(results)
    loc = localization(results)
    miss = misses(results, rows)
    tot = {m: tally(rows, m) for m in MODES}
    os.makedirs(os.path.join(out, "tables"), exist_ok=True)
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    written = []

    md = ["# AST evaluation: results for the paper", "",
          "Generated by `python -m code.stats.ast_paper` from the playtests on disk. The rule: `ast_rule.md`; "
          "costs: `ast_costs.md`.", "",
          "## Which method gets each mutant right", "",
          "**Replay is the detector**: the original's frozen agents play the mutant and are judged by the rule "
          "against their own play on the original. Frozen agents replay attempt for attempt, so no seed noise reaches "
          "them. **Retraining answers one question**: can agents trained on the mutant still finish it? (As a detector it "
          "is too noisy: see the false alarms below.) Baselines: Static = the reachability check every mutant passed "
          "before it was written (it flags a mutant whose exit can no longer be reached); Random = an agent pressing "
          "random buttons, compared with itself on the original (it never wins, so nothing it plays counts under the "
          "rule; shown with the signals it would raise).", "",
          "| Game | Mutant | Class | " + " | ".join(HEAD[m] for m in (*MODES, "solvable")) + " |",
          "|---|---|---|" + "---|" * (len(MODES) + 1)]
    for r in rows:
        md.append(f"| {GAME[r['game']]} | `{r['id']}` | {r['class']} | " + " | ".join(
            f"{_mark(r[m])} ({r[m + '_word']})" if m in ("replay", "solvable") and r[m] is not None else _mark(r[m])
            for m in (*MODES, "solvable")) + " |")
    sv = solvability(rows)
    md += ["", "| Totals | " + " | ".join(HEAD[m] for m in MODES) + " |", "|---|" + "---|" * len(MODES),
           "| play-changing caught | " + " | ".join(f"{tot[m]['caught'][0]}/{tot[m]['caught'][1]}" for m in MODES) + " |",
           "| neutral left alone | " + " | ".join(f"{tot[m]['quiet'][0]}/{tot[m]['quiet'][1]}" for m in MODES) + " |", "",
           f"Still solvable after retraining, as expected: breaking mutants no longer solvable {sv['breaking'][0]}/"
           f"{sv['breaking'][1]}, all others still solvable {sv['others'][0]}/{sv['others'][1]}.", ""]
    cos = [r for r in rows if r["operator"].endswith("cosmetic_tile")]
    md.append(f"Cosmetic copies replayed identically: {sum(bool(r['retrain']) for r in cos)}/{len(cos)} games "
              f"({', '.join(r['retrain_word'] for r in cos)}).")
    fa = am.false_alarms(results)
    md += ["", "## False alarms: why retraining is not the detector", ""]
    if fa["comparisons"]:
        rt = tally(rows, "retrain")
        md += ["The original retrained with seeds 1111, 2222 and 3333 and compared with itself, over every level or "
               "Mega Man section whose original is solved. Every flag is a false alarm. Used as a detector, retraining "
               f"would catch {rt['caught'][0]}/{rt['caught'][1]} play-changing mutants, but it also flags the unchanged "
               f"level in {fa['fired']['detected']} of {fa['comparisons']} comparisons, so its catches cannot be told from "
               "seed noise. Replay has no such noise: its frozen agents replay attempt for attempt (cosmetic copies "
               "identical in every game), so its alarms can only come from the change itself. Completion lost, the "
               "signal behind \"still solvable\", fired only in Mega Man sections, whose training from new seeds sometimes "
               "fails outright.", "", "| Signal | Fired |", "|---|---|"]
        md += [f"| {k} | {v}/{fa['comparisons']} |" for k, v in fa["fired"].items()]
        md.append(f"| win rate outside the seed spread (beside the rule) | {fa['beyond_seeds'][0]}/{fa['beyond_seeds'][1]} |")
        nulls = [(g, k, e["vs_base"]) for g, res in results.items() for k, e in res.items()
                 if k.split(" @ ")[0] == am.RESEEDED and (e.get("vs_base") or {}).get("counted")]
        md += ["", "| Game | Level | Verdict | Signals |", "|---|---|---|---|"]
        md += [f"| {GAME[g]} | {k.split(' @ ')[1] if ' @ ' in k else eval_mutants.EVAL_LEVELS[g][0]} | {s['verdict']} | "
               + (", ".join(x for x in am.SIGNALS if s[x]) or "-") + " |" for g, k, s in nulls]
    else:
        md.append("Not run yet (`base-reseeded`).")

    md += ["", "## Which persona catches what (2D games, each persona's three agents alone)", "",
           "Verdict of one persona's agents: B broken, C changed, U unchanged, - its original is unsolved (not counted).", ""]
    plist = ["novice", "clumsy", "experienced", "speedrunner", "killer", "collector"]
    short = {"broken": "B", "changed": "C", "unchanged": "U", "not counted": "-"}
    for mode in ("retrain", "replay"):
        md += [f"**{mode}**", "", "| Game | Mutant | Class | " + " | ".join(plist) + " |", "|---|---|---|" + "---|" * len(plist)]
        for r in per:
            md.append(f"| {GAME[r['game']]} | `{r['id']}` | {r['class']} | "
                      + " | ".join(short.get(r.get((mode, p)), " ") for p in plist) + " |")
        md.append("")
        counts = {p: sum(1 for r in per if r["class"] != "neutral" and r.get((mode, p)) in ("broken", "changed")) for p in plist}
        alarms = {p: sum(1 for r in per if r["class"] == "neutral" and r.get((mode, p)) in ("broken", "changed")) for p in plist}
        n_play = sum(1 for r in per if r["class"] != "neutral")
        fa_text = ", ".join(f"{p} {alarms[p]}" for p in plist if alarms[p]) or "none"
        md.append(f"Flags per persona ({mode}): " + ", ".join(f"{p} {counts[p]}/{n_play}" for p in plist)
                  + f"; false alarms on off-route: {fa_text}.")
        md.append("")

    md += ["## Did play reach the change, and do failures end on it?", "",
           "Reached = the original's routes pass within one tile of a changed cell. Near = share of failed attempts "
           f"ending within {NEAR} tiles of a changed cell. Cluster = distance (tiles) from the death cluster to the change.", "",
           "| Game | Mutant | Class | Reached | Near (original) | Near (retrain) | Cluster (retrain) | Near (replay) | Cluster (replay) |",
           "|---|---|---|---|---|---|---|---|---|"]
    pct = lambda v: "-" if v is None else f"{100 * v:.0f}%"  # noqa: E731
    for r in loc:
        md.append(f"| {GAME[r['game']]} | `{r['id']}` | {r['class']} | {_mark(r['reached'])} | {pct(r['orig_near'])} | "
                  f"{pct(r['retrain_near'])} | {r['retrain_cluster_dist'] if r['retrain_cluster_dist'] is not None else '-'} | "
                  f"{pct(r['replay_near'])} | {r['replay_cluster_dist'] if r['replay_cluster_dist'] is not None else '-'} |")
    md += ["", "## Why each wrong call was wrong", ""] + [f"- {s}" for s in miss] + [""]

    path = os.path.join(out, "ast_results.md")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(md))
    written.append(path)
    path = os.path.join(out, "tables", "detection.tex")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(detection_tex(rows))
    written.append(path)
    written += figures(results, os.path.join(out, "figures"))
    return written


# ── figures (white, for print) ───────────────────────────────────────────────────────────────
WIN, FAIL, CHANGE, TILE = "#2e9d3a", "#c8102e", "#e0a800", "#b8b8b8"


def figures(results: dict, out: str) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white", "figure.facecolor": "white"})
    made = []

    # 1. replay vs retrain: wins on the original, with frozen agents, and after retraining
    short = {"seal_route": "sealed", "widen_gap": "wider gap", "jump_velocity_down": "weaker jump",
             "add_enemy": "extra enemy", "fuse_down": "shorter fuse"}
    bars = []
    for game, res in results.items():
        if game == "megaman":
            continue
        ref = (res.get("base") or {}).get("agents")
        for m in eval_mutants.eval_set(game):
            e = res.get(m["id"]) or {}
            if m["class"] == "neutral" or not e.get("agents") or not e.get("replay") or not ref:
                continue
            frac = lambda a: a["win_rate"]["pooled"][0] / max(1, a["win_rate"]["pooled"][1])  # noqa: E731
            bars.append((f"{GAME[game].rsplit(' ', 1)[0]}\n{short.get(m['id'], m['id'])}",
                         frac(ref), frac(e["replay"]["agents"]), frac(e["agents"])))
    if bars:
        fig, ax = plt.subplots(figsize=(7.16, 2.3))
        x = range(len(bars))
        w = 0.27
        for k, (label, col) in enumerate((("original", "#7a7a7a"), ("replay (frozen agents)", FAIL), ("retrained agents", WIN))):
            xs = [i + (k - 1) * w for i in x]
            ax.bar(xs, [b[k + 1] for b in bars], w, label=label, color=col)
            for xi, b in zip(xs, bars):
                if b[k + 1] == 0:                                    # a bar of height 0 still says something
                    ax.text(xi, 0.015, "0", ha="center", va="bottom", fontsize=6.5, color=col)
        ax.set_xticks(list(x), [b[0] for b in bars], fontsize=6.5)
        ax.set_ylabel("win rate")
        ax.set_ylim(0, 1)
        ax.legend(ncols=3, fontsize=7, loc="upper right", frameon=False)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            p = os.path.join(out, f"replay_vs_retrain.{ext}")
            fig.savefig(p, dpi=200, metadata={"CreationDate": None} if ext == "pdf" else None)   # same bytes for the same figure
            made.append(p)
        plt.close(fig)

    # 2. death maps: the original and a mutant over the same level
    picks = [("mario", "widen_gap", "retrain"), ("meatboy", "widen_gap", "replay"), ("bomberman", "add_enemy", "retrain")]
    panels = []
    for game, mid, mode in picks:
        res = results.get(game) or {}
        e = res.get(mid) or {}
        a = e.get("agents") if mode == "retrain" else (e.get("replay") or {}).get("agents")
        ref = (res.get("base") or {}).get("agents")
        if a and ref:
            panels.append((game, mid, mode, ref, a))
    if panels:
        # a level much wider than tall (Mario) gets two full-width rows, the others sit side by side
        wide = [p for p in panels if _aspect(p[0]) > 4]
        boxes = [p for p in panels if _aspect(p[0]) <= 4]
        heights = [0.35] * (2 * len(wide)) + [1.0] * len(boxes)
        fig = plt.figure(figsize=(7.16, 0.75 * len(wide) * 2 + 2.0 * len(boxes)))
        gs = fig.add_gridspec(len(heights), 2, height_ratios=heights)
        r = 0
        for game, mid, mode, ref, a in wide + boxes:
            grid, cells = ast_page._level(eval_mutants.eval_set(game))
            pair = ((f"{GAME[game]}, original", ref, []), (f"{GAME[game]}, {mid.replace('_', ' ')} ({mode})", a, cells.get(mid) or []))
            if _aspect(game) > 4:
                for title, s, changed in pair:
                    _map(fig.add_subplot(gs[r, :]), grid, s, changed, title)
                    r += 1
            else:
                for col, (title, s, changed) in enumerate(pair):
                    _map(fig.add_subplot(gs[r, col]), grid, s, changed, title)
                r += 1
        fig.tight_layout()
        for ext in ("pdf", "png"):
            p = os.path.join(out, f"death_maps.{ext}")
            fig.savefig(p, dpi=200, metadata={"CreationDate": None} if ext == "pdf" else None)   # same bytes for the same figure
            made.append(p)
        plt.close(fig)
    return made


def _aspect(game: str) -> float:
    grid, _ = ast_page._level(eval_mutants.eval_set(game))
    return max(map(len, grid), default=1) / max(1, len(grid))


def _map(ax, grid: list[str], s: dict, changed: list, title: str) -> None:
    """One level: its solid tiles, every attempt's route (red failed, green won), the death
    cluster, and the changed tiles."""
    import matplotlib.patches as mp
    n = len(grid)
    cols = max(map(len, grid), default=1)
    for r, line in enumerate(grid):
        for c, ch in enumerate(line):
            if ch not in " .":
                ax.add_patch(mp.Rectangle((c, r), 1, 1, color=TILE, lw=0))
    for route in sorted(s.get("routes") or [], key=lambda q: q["w"]):
        xs, ys = zip(*route["p"])
        ax.plot(xs, ys, color=WIN if route["w"] else FAIL, lw=0.8 if route["w"] else 0.6, alpha=0.8 if route["w"] else 0.5)
    for c in changed:
        ax.add_patch(mp.Rectangle((c[0], c[1]), 1, 1, fill=False, ec=CHANGE, lw=1.0))
    k = s.get("death_cluster")
    if k:                                   # the cluster's box, cut to the level's edges
        x0, y0 = max(0, k["col"] - NEAR), max(0, k["row"] - NEAR)
        x1, y1 = min(cols, k["col"] + NEAR + 1), min(n, k["row"] + NEAR + 1)
        ax.add_patch(mp.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="black", lw=0.8))
    w, t = s["win_rate"]["pooled"]
    ax.set_title(f"{title}: {w}/{t} won", fontsize=7, loc="left")
    ax.set_xlim(0, cols)
    ax.set_ylim(n, 0)
    ax.set_aspect("equal")
    ax.axis("off")


def main() -> None:
    results = {g: am.evaluate(g) for g in am.GAMES}
    for p in write(results):
        print("wrote", p)


if __name__ == "__main__":
    main()
