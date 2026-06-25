#!/usr/bin/env python3
"""
case_study.py — Full case-study analysis of the multi-seed paper matrix.

Ingests the per-job results written by `run_paper_matrix.py`
(`<results_dir>/*.json`, each = {job, train_seconds, levels:{<lvl>:{n,wins,win_rate,by_cause}}})
and produces the corrected, defensible tables for the revised paper:

  - Table 1: the 2x2x2 design (A-H).
  - Table 2: per run, win-rate mean +/- 95% CI per level (stochastic eval), the
    dominant failure mode, training time, and seed count.
  - Failure-mode taxonomy: per-cause fraction (mean +/- 95% CI across seeds).

Win rates are STOCHASTIC-policy means (deterministic eval is a stall-trap artifact;
see docs/paper_corrections.md). Exports markdown + CSV and prints a terminal report.

Used by the "Full Case Study Analysis" menu option, or standalone:
    .venv/bin/python -m code.scripts.case_study [--results <dir>] [--out <dir>]
"""
from __future__ import annotations

import csv
import glob
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DEFAULT_RESULTS_DIR = "/tmp/peak_matrix/results"
LEVELS = ["Mario1-1", "Mario1-2"]

# (extractor, persona, skill) -> (run letter, readable extractor, readable persona, budget)
SKILL_BUDGET = {"Novice": "1M", "Expert": "8M"}
EXTRACTOR_LABEL = {"lightmobile": "LightMobile (~18K)", "spatialattention": "SpatialAttention (~77K)"}
PERSONA_LABEL = {"platformer_simple": "Simple", "platformer_adept": "Pathfinder"}

# Paper's A-H ordering.
RUN_DESIGN: List[Tuple[str, str, str, str]] = [
    ("A", "lightmobile", "platformer_simple", "Novice"),
    ("B", "lightmobile", "platformer_simple", "Expert"),
    ("C", "lightmobile", "platformer_adept", "Novice"),
    ("D", "lightmobile", "platformer_adept", "Expert"),
    ("E", "spatialattention", "platformer_simple", "Novice"),
    ("F", "spatialattention", "platformer_simple", "Expert"),
    ("G", "spatialattention", "platformer_adept", "Novice"),
    ("H", "spatialattention", "platformer_adept", "Expert"),
]
_KEY_TO_LETTER = {(e, p, s): L for (L, e, p, s) in RUN_DESIGN}


# ── statistics ──────────────────────────────────────────────────────────────
# Two-sided 95% t critical values by sample size n (df = n-1). Honest for small n.
_T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
        8: 2.365, 9: 2.306, 10: 2.262}


def mean_ci(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    """Return (mean, 95% CI half-width). CI is None for n<2."""
    n = len(values)
    if n == 0:
        return (None, None)
    m = sum(values) / n
    if n == 1:
        return (m, None)
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    sd = math.sqrt(var)
    t = _T95.get(n, 1.96)
    return (m, t * sd / math.sqrt(n))


def _fmt_pct_ci(m: Optional[float], ci: Optional[float]) -> str:
    if m is None:
        return "—"
    if ci is None:
        return f"{m*100:.1f}%"
    return f"{m*100:.1f}% ± {ci*100:.1f}"


# ── load + aggregate ─────────────────────────────────────────────────────────
def load_results(results_dir: str) -> List[dict]:
    out = []
    for p in sorted(glob.glob(str(Path(results_dir) / "*.json"))):
        try:
            r = json.load(open(p))
        except Exception:
            continue
        if isinstance(r, dict) and "job" in r and "error" not in r:
            out.append(r)
    return out


def aggregate(results: List[dict]) -> Dict[str, dict]:
    """Group per-seed results by (extractor, persona, skill) config -> aggregate."""
    by_cfg: Dict[Tuple[str, str, str], List[dict]] = defaultdict(list)
    for r in results:
        j = r["job"]
        by_cfg[(j["ext"], j["persona"], j["skill"])].append(r)

    agg: Dict[str, dict] = {}
    for (ext, persona, skill), runs in by_cfg.items():
        letter = _KEY_TO_LETTER.get((ext, persona, skill), "?")
        row = {
            "run": letter, "extractor": ext, "persona": persona, "skill": skill,
            "n_seeds": len(runs),
            "train_seconds_mean": (sum(x.get("train_seconds", 0) for x in runs) / len(runs)) if runs else None,
            "levels": {},
        }
        for lvl in LEVELS:
            wrs = [x["levels"][lvl]["win_rate"] for x in runs
                   if lvl in x.get("levels", {}) and "win_rate" in x["levels"][lvl]]
            m, ci = mean_ci(wrs)
            # failure-mode fractions per seed, then mean across seeds
            cause_fracs: Dict[str, List[float]] = defaultdict(list)
            for x in runs:
                bc = x.get("levels", {}).get(lvl, {}).get("by_cause", {})
                tot = sum(bc.values())
                if tot <= 0:
                    continue
                for cause, cnt in bc.items():
                    cause_fracs[cause].append(cnt / tot)
            taxonomy = {}
            for cause, fracs in cause_fracs.items():
                cm, cci = mean_ci(fracs)
                if cm and cm > 0:
                    taxonomy[cause] = {"mean": cm, "ci": cci}
            dominant = max(taxonomy.items(), key=lambda kv: kv[1]["mean"])[0] if taxonomy else "—"
            row["levels"][lvl] = {
                "win_rate_mean": m, "win_rate_ci": ci, "n": len(wrs),
                "taxonomy": taxonomy, "dominant_failure": dominant,
            }
        agg[letter if letter != "?" else f"{ext}_{persona}_{skill}"] = row
    return agg


# ── formatting ───────────────────────────────────────────────────────────────
def format_table1() -> str:
    lines = ["Table 1 — 2x2x2 design (staged SMB 1-1 -> 1-2):", ""]
    lines.append(f"  {'Run':<4} {'Extractor':<24} {'Persona':<12} {'Budget':<14}")
    lines.append("  " + "-" * 56)
    for (L, e, p, s) in RUN_DESIGN:
        lines.append(f"  {L:<4} {EXTRACTOR_LABEL.get(e, e):<24} {PERSONA_LABEL.get(p, p):<12} "
                     f"{s} ({SKILL_BUDGET.get(s, '?')})")
    return "\n".join(lines)


def format_table2(agg: Dict[str, dict]) -> str:
    lines = ["Table 2 — results (stochastic win-rate, mean ± 95% CI over seeds):", ""]
    lines.append(f"  {'Run':<4} {'Seeds':<6} {'1-1 Win%':<16} {'1-2 Win%':<16} "
                 f"{'1-1 fail':<10} {'1-2 fail':<10} {'Train(s)':<9}")
    lines.append("  " + "-" * 78)
    for (L, e, p, s) in RUN_DESIGN:
        row = agg.get(L)
        if not row:
            lines.append(f"  {L:<4} {'—':<6} {'(no data)':<16}")
            continue
        l1 = row["levels"].get("Mario1-1", {})
        l2 = row["levels"].get("Mario1-2", {})
        ts = row.get("train_seconds_mean")
        lines.append(
            f"  {L:<4} {row['n_seeds']:<6} "
            f"{_fmt_pct_ci(l1.get('win_rate_mean'), l1.get('win_rate_ci')):<16} "
            f"{_fmt_pct_ci(l2.get('win_rate_mean'), l2.get('win_rate_ci')):<16} "
            f"{l1.get('dominant_failure','—'):<10} {l2.get('dominant_failure','—'):<10} "
            f"{('%.0f' % ts) if ts else '—':<9}"
        )
    return "\n".join(lines)


def format_taxonomy(agg: Dict[str, dict]) -> str:
    lines = ["Failure-mode taxonomy (fraction of deaths, mean ± 95% CI across seeds):", ""]
    for (L, e, p, s) in RUN_DESIGN:
        row = agg.get(L)
        if not row:
            continue
        for lvl in LEVELS:
            tax = row["levels"].get(lvl, {}).get("taxonomy", {})
            if not tax:
                continue
            parts = []
            for cause, d in sorted(tax.items(), key=lambda kv: -kv[1]["mean"]):
                ci = f"±{d['ci']*100:.0f}" if d["ci"] is not None else ""
                parts.append(f"{cause} {d['mean']*100:.0f}%{ci}")
            lines.append(f"  Run {L} {lvl}: " + ", ".join(parts))
    return "\n".join(lines)


def export_csv(agg: Dict[str, dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "extractor", "persona", "skill", "n_seeds", "train_seconds_mean",
                    "lvl1_winrate_mean", "lvl1_winrate_ci", "lvl1_dominant_fail",
                    "lvl2_winrate_mean", "lvl2_winrate_ci", "lvl2_dominant_fail"])
        for (L, e, p, s) in RUN_DESIGN:
            row = agg.get(L)
            if not row:
                w.writerow([L, e, p, s, 0, "", "", "", "", "", "", ""])
                continue
            l1 = row["levels"].get("Mario1-1", {})
            l2 = row["levels"].get("Mario1-2", {})
            w.writerow([L, e, p, s, row["n_seeds"], row.get("train_seconds_mean"),
                        l1.get("win_rate_mean"), l1.get("win_rate_ci"), l1.get("dominant_failure"),
                        l2.get("win_rate_mean"), l2.get("win_rate_ci"), l2.get("dominant_failure")])


def build_report(agg: Dict[str, dict]) -> str:
    n_cfg = sum(1 for (L, *_ ) in RUN_DESIGN if L in agg)
    header = ("PEAK CASE STUDY — multi-seed, fixed code, trustworthy stochastic per-level eval\n"
              f"Configs with data: {n_cfg}/8\n")
    return "\n\n".join([header, format_table1(), format_table2(agg), format_taxonomy(agg)])


def run_case_study(results_dir: str = DEFAULT_RESULTS_DIR,
                   out_dir: Optional[str] = None) -> Dict[str, dict]:
    results = load_results(results_dir)
    agg = aggregate(results)
    report = build_report(agg)
    print(report)
    if not results:
        print(f"\n[!] No matrix results found in {results_dir}. Run run_paper_matrix.py first.")
    if out_dir:
        od = Path(out_dir)
        od.mkdir(parents=True, exist_ok=True)
        (od / "case_study_report.md").write_text(report + "\n")
        export_csv(agg, od / "case_study_table2.csv")
        print(f"\n[INFO] Exported: {od/'case_study_report.md'}  and  {od/'case_study_table2.csv'}")
    return agg


def main(argv: Optional[list] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Full case-study analysis of the paper matrix.")
    ap.add_argument("--results", default=DEFAULT_RESULTS_DIR, help="dir of run_paper_matrix *.json results")
    ap.add_argument("--out", default="case-study/regenerated", help="output dir for report + csv")
    args = ap.parse_args(argv)
    run_case_study(args.results, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
