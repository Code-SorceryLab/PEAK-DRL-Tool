#!/usr/bin/env python3
"""
run_paper_matrix.py — Re-run the paper's 2x2x2 case-study matrix x N seeds with the
FIXED code (working PBRS, capped Simple reward, norm_reward, trustworthy eval) and
measure each config on the trustworthy per-level eval (eval_level.py, stochastic).

Produces, per (extractor x persona x budget x seed) job:
  - training wall-clock seconds
  - per-level (Mario1-1, Mario1-2) win-rate + death-cause distribution

Design:
  - Each job writes to an ISOLATED out_root (/tmp/peak_matrix/<jid>) via the new
    `+out_root=` override, so concurrent runs never clobber models/csv.
  - Up to MAX_CONCURRENT jobs run at once (this Mac is CPU-bound; n_envs=2 each).
  - Resumable: a job whose results/<jid>.json exists is skipped.

Run from repo root with the venv python:
  .venv/bin/python -m code.scripts.run_paper_matrix          # full 24-job matrix
  .venv/bin/python -m code.scripts.run_paper_matrix --smoke  # 1 tiny job (pipeline check)
  .venv/bin/python -m code.scripts.run_paper_matrix --aggregate  # build TABLE from results
"""
from __future__ import annotations

import glob
import itertools
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from code.metrics.eval_summary import summarize_eval

REPO = Path(__file__).resolve().parents[2]
PY = str(REPO / ".venv" / "bin" / "python")
MATRIX_ROOT = Path("/tmp/peak_matrix")
RESULTS = MATRIX_ROOT / "results"
LOGFILE = MATRIX_ROOT / "orchestrator.log"

MAX_CONCURRENT = 3            # ~3 cores/run on a 10-core CPU-bound Mac
EVAL_EPISODES = 30
LEVELS = ["Mario1-1", "Mario1-2"]

# Paper's 2x2x2 factors (personas: Simple, Pathfinder=adept). Paper budgets 1M/8M.
EXTRACTORS = ["lightmobile", "spatialattention"]
PERSONAS = ["platformer_simple", "platformer_adept"]
BUDGETS = {"Novice": 1_000_000, "Expert": 8_000_000}
SEEDS = [1234, 2025, 31337]

SMOKE = "--smoke" in sys.argv
if SMOKE:
    EXTRACTORS = ["lightmobile"]
    PERSONAS = ["platformer_simple"]
    BUDGETS = {"Novice": 6_000}
    SEEDS = [1234]
    EVAL_EPISODES = 3

# Single real config for a fast local "does the fixed code perform well?" check.
# spatialattention + Pathfinder(adept) + Novice(1M) is directly comparable to the
# 60%/13% baseline (the shipped adept model was a ~1M Novice run).
PERFCHECK = "--perfcheck" in sys.argv
if PERFCHECK:
    EXTRACTORS = ["spatialattention"]
    PERSONAS = ["platformer_adept"]
    BUDGETS = {"Novice": 1_000_000}
    SEEDS = [1234]
    EVAL_EPISODES = 30

# (extractor, persona) pairs to run. Default = full 2x2 (A-H over both budgets).
# --priority = the reviewer's stated minimum: C/D (LightMobile+Pathfinder) and
# E/F (SpatialAttention+Simple) only -> half the compute.
PRIORITY = "--priority" in sys.argv
if PRIORITY:
    CONFIG_PAIRS = [
        ("lightmobile", "platformer_adept"),        # C (Novice), D (Expert)
        ("spatialattention", "platformer_simple"),  # E (Novice), F (Expert)
    ]
else:
    CONFIG_PAIRS = [(e, p) for e in EXTRACTORS for p in PERSONAS]


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOGFILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOGFILE, "a") as f:
        f.write(line + "\n")


def _kill_group(proc):
    """SIGTERM then SIGKILL the process's whole group (kills SubprocVecEnv workers)."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except Exception:
            return
        try:
            proc.wait(timeout=15)
            return
        except Exception:
            continue


def run_train(cmd, log_path: Path, timeout_s: float) -> tuple:
    """Run training robustly. The trainer hangs at interpreter exit (SubprocVecEnv
    workers) AFTER it has saved the model and printed 'Done. Trained ...'. So we
    stream output to log_path, and the moment that marker appears (or the process
    exits on its own), we kill the (possibly hung) process group and return.

    Returns (ok: bool, seconds: float).
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    DONE = "Done. Trained"
    t0 = time.time()
    with open(log_path, "w") as lf:
        proc = subprocess.Popen(cmd, cwd=str(REPO), stdout=lf,
                                stderr=subprocess.STDOUT, start_new_session=True)
    try:
        while True:
            rc = proc.poll()
            if rc is not None:                     # exited on its own
                return (rc == 0, round(time.time() - t0, 1))
            try:
                done = DONE in log_path.read_text(errors="ignore")
            except Exception:
                done = False
            if done:                               # logically complete; model saved
                _kill_group(proc)
                return (True, round(time.time() - t0, 1))
            if time.time() - t0 > timeout_s:       # safety net (rarely hit)
                _kill_group(proc)
                return (False, round(time.time() - t0, 1))
            time.sleep(5)
    finally:
        if proc.poll() is None:
            _kill_group(proc)


def all_jobs():
    for (ext, persona), skill, seed in itertools.product(CONFIG_PAIRS, BUDGETS, SEEDS):
        yield {
            "ext": ext, "persona": persona, "skill": skill, "seed": seed,
            "budget": BUDGETS[skill],
            "jid": f"{ext}__{persona}__{skill}__s{seed}",
        }


def run_job(job: dict) -> tuple:
    jid = job["jid"]
    res_path = RESULTS / f"{jid}.json"
    if res_path.exists():
        log(f"SKIP {jid} (result exists)")
        return ("skip", jid)

    out_root = MATRIX_ROOT / jid
    train_cmd = [
        PY, "-m", "code.scripts.train",
        "+game=platformer", "+model=ppo",
        f"+persona={job['persona']}", f"+skill={job['skill']}", f"seed={job['seed']}",
        f"+architecture={job['ext']}",
        f"skills.Novice={BUDGETS.get('Novice', 1_000_000)}",
        f"skills.Expert={BUDGETS.get('Expert', 8_000_000)}",
        "n_envs=2", "profile=false", f"+out_root={out_root}",
    ]
    log(f"TRAIN start {jid} (budget={job['budget']})")
    train_log = out_root / "train.log"
    # Generous safety net; marker-detection ends the job right at real completion.
    timeout_s = max(2400, int(job["budget"] / 500 * 4))
    ok, train_s = run_train(train_cmd, train_log, timeout_s)
    if not ok:
        tail = ""
        try:
            tail = train_log.read_text(errors="ignore")[-1500:]
        except Exception:
            pass
        log(f"TRAIN FAIL {jid} (t={train_s}s)")
        res_path.parent.mkdir(parents=True, exist_ok=True)
        res_path.write_text(json.dumps(
            {"job": job, "error": "train_failed", "train_seconds": train_s,
             "log_tail": tail}, indent=2))
        return ("fail", jid)

    # Locate the trained model (best_model.zip preferred; else the final saved zip).
    cands = sorted(glob.glob(str(out_root / "best" / "*" / "best_model.zip"))) \
        or sorted(glob.glob(str(out_root / "*.zip")))
    if not cands:
        log(f"NO MODEL {jid}")
        res_path.parent.mkdir(parents=True, exist_ok=True)
        res_path.write_text(json.dumps(
            {"job": job, "error": "no_model", "train_seconds": train_s}, indent=2))
        return ("fail", jid)
    model = cands[0]

    levels = {}
    for lvl in LEVELS:
        out_json = out_root / f"eval_{lvl}.json"
        ev = [PY, "-m", "code.scripts.eval_level", "--model", model,
              "--level", lvl, "--episodes", str(EVAL_EPISODES), "--stochastic",
              "--out", str(out_json)]
        try:
            subprocess.run(ev, cwd=str(REPO), capture_output=True, text=True, timeout=1800)
        except subprocess.TimeoutExpired:
            log(f"EVAL TIMEOUT {jid} {lvl} (checking for partial output)")
        # eval_level writes --out before any exit-hang; trust the file if present.
        if not out_json.exists():
            levels[lvl] = {"error": "no eval output"}
            log(f"EVAL FAIL {jid} {lvl}: no output json")
            continue
        try:
            episodes = json.load(open(out_json))
            levels[lvl] = summarize_eval(episodes)
        except Exception as exc:
            levels[lvl] = {"error": f"bad eval json: {exc}"}
            log(f"EVAL PARSE FAIL {jid} {lvl}: {exc}")

    result = {"job": job, "model": model, "train_seconds": train_s, "levels": levels}
    res_path.parent.mkdir(parents=True, exist_ok=True)
    res_path.write_text(json.dumps(result, indent=2))
    wr = {lvl: levels[lvl].get("win_rate") for lvl in LEVELS}
    log(f"DONE {jid} train={train_s}s win_rate={wr}")
    return ("done", jid)


def _mean_ci(values):
    """Mean and 95% CI half-width (normal approx) for a small sample."""
    n = len(values)
    if n == 0:
        return (None, None)
    m = sum(values) / n
    if n == 1:
        return (m, None)
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    sd = math.sqrt(var)
    ci = 1.96 * sd / math.sqrt(n)
    return (m, ci)


def aggregate():
    """Aggregate per-job results into a per-config table (mean +/- 95% CI over seeds)."""
    rows = []
    by_config = {}
    for f in sorted(glob.glob(str(RESULTS / "*.json"))):
        r = json.load(open(f))
        if "error" in r:
            continue
        job = r["job"]
        key = (job["ext"], job["persona"], job["skill"])
        by_config.setdefault(key, []).append(r)

    for (ext, persona, skill), runs in sorted(by_config.items()):
        row = {"extractor": ext, "persona": persona, "budget": skill,
               "seeds": len(runs),
               "train_seconds_mean": round(sum(x["train_seconds"] for x in runs) / len(runs), 1)}
        for lvl in LEVELS:
            wrs = [x["levels"][lvl]["win_rate"] for x in runs
                   if lvl in x["levels"] and "win_rate" in x["levels"][lvl]]
            m, ci = _mean_ci(wrs)
            row[f"{lvl}_winrate_mean"] = None if m is None else round(m, 3)
            row[f"{lvl}_winrate_ci95"] = None if ci is None else round(ci, 3)
            # dominant death cause averaged across seeds
            cause_tot = {}
            for x in runs:
                bc = x["levels"].get(lvl, {}).get("by_cause", {})
                for k, v in bc.items():
                    cause_tot[k] = cause_tot.get(k, 0) + v
            row[f"{lvl}_by_cause"] = cause_tot
        rows.append(row)

    out = {"configs": rows, "n_result_files": len(glob.glob(str(RESULTS / "*.json")))}
    (MATRIX_ROOT / "TABLE.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out


def main():
    if "--aggregate" in sys.argv:
        aggregate()
        return
    jobs = list(all_jobs())
    log(f"=== matrix start: {len(jobs)} jobs, MAX_CONCURRENT={MAX_CONCURRENT}, smoke={SMOKE} ===")
    RESULTS.mkdir(parents=True, exist_ok=True)
    counts = {"done": 0, "fail": 0, "skip": 0}
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as ex:
        futs = {ex.submit(run_job, j): j["jid"] for j in jobs}
        for fut in as_completed(futs):
            status, jid = fut.result()
            counts[status] += 1
            log(f"PROGRESS {sum(counts.values())}/{len(jobs)} {counts}")
    log(f"=== matrix complete: {counts} ===")
    aggregate()


if __name__ == "__main__":
    main()
