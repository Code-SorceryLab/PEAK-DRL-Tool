"""
PEAK Agent Performance Analyzer v2
===================================
Full numpy-based analysis with per-level breakdown, episode-level stats,
learning curves, reward balance, and cross-run comparison.
"""
import pandas as pd
import glob
import os
import numpy as np
from typing import Dict, List, Any

STANDARD_COLS = {
    'step', 'total_reward', 'action', 'level', 'levels_completed',
    'x', 'y', 'vx', 'vy', 'goal_dist', 'event', 'cause',
}
OBS_SANITY_COLS = {
    'grid_player_mean','grid_player_std','grid_player_min','grid_player_max',
    'grid_solid_mean','grid_solid_std','grid_solid_min','grid_solid_max',
    'grid_hazard_mean','grid_hazard_std','grid_hazard_min','grid_hazard_max',
    'grid_collectible_mean','grid_collectible_std','grid_collectible_min','grid_collectible_max',
    'grid_dijkstra_mean','grid_dijkstra_std','grid_dijkstra_min','grid_dijkstra_max',
    'scalar_mean','scalar_std','scalar_min','scalar_max','dijkstra_val','obs_warnings',
}
_ARCH_TAGS = {
    "lightmobile": "lightmobile",
    "spatialattention": "spatialattention",
    "channelattention": "channelattention",
    "deepchannelattention": "deepchannelattention",
    "mlp": "mlp",
}
_SKILL_TAGS = {"novice", "expert", "custom"}
W = 72

def _bar(v, mx, w=20, c="█"):
    f = int((v / max(mx, 1)) * w)
    return c * f + "░" * (w - f)

def _pct(n, t): return (n / t * 100) if t > 0 else 0.0
def _sm(a): return float(np.nanmean(a)) if len(a) > 0 else 0.0
def _smed(a): return float(np.nanmedian(a)) if len(a) > 0 else 0.0
def _ssd(a): return float(np.nanstd(a)) if len(a) > 0 else 0.0

def get_all_log_files():
    files = []
    for d in [".", "csv", "mylogs", "runs"]:
        files.extend(glob.glob(os.path.join(d, "**", "training_log*.csv"), recursive=True))
    seen = set(); out = []
    for f in files:
        p = os.path.abspath(f)
        if p not in seen: seen.add(p); out.append(f)
    return sorted(out, key=os.path.getmtime, reverse=True)

def _load_csv(fp, max_rows=800_000):
    chunks = []; total = 0
    for c in pd.read_csv(fp, low_memory=False, chunksize=200_000):
        chunks.append(c); total += len(c)
    if not chunks: return None, 0
    full = pd.concat(chunks, ignore_index=True)
    if total > max_rows:
        e = max_rows // 5; l = max_rows - e
        full = pd.concat([full.iloc[:e], full.iloc[-l:]], ignore_index=True)
    return full, total

def _parse_run_id(fp):
    name = os.path.basename(fp).replace("training_log_", "").replace(".csv", "")
    parts = name.split("_")
    arch = _ARCH_TAGS[parts.pop().lower()] if parts and parts[-1].lower() in _ARCH_TAGS else None
    skill = parts.pop() if parts and parts[-1].lower() in _SKILL_TAGS else None
    game = parts[0] if len(parts) >= 1 else "?"
    persona = "_".join(parts[2:]) if len(parts) >= 3 else (parts[1] if len(parts) >= 2 else name)
    if persona.startswith(f"{game}_"): persona = persona[len(game)+1:]
    lp = [persona]
    if skill: lp.append(skill)
    if arch: lp.append(f"[{arch}]")
    return {"label": " | ".join(lp), "persona": persona, "skill": skill or "?", "arch": arch or "?", "game": game}

def _reward_cols(df):
    return [c for c in df.columns if c.lower() not in STANDARD_COLS and c.lower() not in OBS_SANITY_COLS and "unnamed" not in c.lower()]

# ═══════════════════════════════════════════════════════════════
# Episode Builder
# ═══════════════════════════════════════════════════════════════
def _build_episodes(df) -> List[Dict]:
    eps = []
    mask = df['event'].notna() & (df['event'] != "")
    eidxs = np.where(mask.values)[0]
    if len(eidxs) == 0: return eps
    prev = 0
    for ei in eidxs:
        ed = df.iloc[prev:ei+1]
        if len(ed) == 0: prev = ei+1; continue
        lr = ed.iloc[-1]
        rw = pd.to_numeric(ed['total_reward'], errors='coerce').fillna(0).values.astype(np.float32)
        xv = pd.to_numeric(ed['x'], errors='coerce').fillna(0).values.astype(np.float32)
        yv = pd.to_numeric(ed['y'], errors='coerce').fillna(0).values.astype(np.float32)
        vxv = pd.to_numeric(ed['vx'], errors='coerce').fillna(0).values.astype(np.float32)
        gd = pd.to_numeric(ed['goal_dist'], errors='coerce').fillna(0).values.astype(np.float32)
        lvls = ed['level'].dropna()
        lvl = str(lvls.mode().iloc[0]) if len(lvls) > 0 else "?"
        eps.append({
            "length": len(ed), "event": str(lr.get('event','')), "cause": str(lr.get('cause','')),
            "level": lvl, "won": str(lr.get('event',''))=="WIN", "died": str(lr.get('event',''))=="DIED",
            "reward_sum": float(rw.sum()), "reward_mean": float(rw.mean()),
            "x_max": float(xv.max()), "x_range": float(xv.max()-xv.min()),
            "y_range": float(yv.max()-yv.min()), "avg_speed": float(np.abs(vxv).mean()),
            "goal_progress": float(gd[0]-gd[-1]) if len(gd)>1 else 0.0,
            "actions": ed['action'].value_counts().to_dict(),
        })
        prev = ei+1
    return eps

# ═══════════════════════════════════════════════════════════════
# Per-Level Stats
# ═══════════════════════════════════════════════════════════════
def _per_level(episodes):
    by = {}
    for e in episodes:
        by.setdefault(e["level"], []).append(e)
    stats = {}
    for lvl, eps in sorted(by.items()):
        n = len(eps); w = sum(1 for e in eps if e["won"]); d = sum(1 for e in eps if e["died"])
        causes = {}
        for e in eps:
            if e["died"] and e["cause"]: causes[e["cause"]] = causes.get(e["cause"],0)+1
        lens = np.array([e["length"] for e in eps], dtype=np.float32)
        rwds = np.array([e["reward_sum"] for e in eps], dtype=np.float32)
        prog = np.array([e["goal_progress"] for e in eps], dtype=np.float32)
        stats[lvl] = {"visits":n,"wins":w,"deaths":d,"win_rate":_pct(w,n),"causes":causes,
            "ep_len_mean":_sm(lens),"ep_len_median":_smed(lens),"ep_len_std":_ssd(lens),
            "reward_mean":_sm(rwds),"reward_std":_ssd(rwds),"reward_median":_smed(rwds),
            "progress_mean":_sm(prog)}
    return stats

# ═══════════════════════════════════════════════════════════════
# Learning Curve
# ═══════════════════════════════════════════════════════════════
def _curve(episodes, nb=5):
    if len(episodes) < nb: nb = max(1, len(episodes))
    bs = len(episodes) // nb; bins = []
    for i in range(nb):
        s = i*bs; e = s+bs if i < nb-1 else len(episodes)
        ch = episodes[s:e]; n = len(ch); w = sum(1 for x in ch if x["won"])
        rw = np.array([x["reward_sum"] for x in ch], dtype=np.float32)
        ln = np.array([x["length"] for x in ch], dtype=np.float32)
        bins.append({"bin":i+1,"eps":n,"wr":_pct(w,n),"rwd":_sm(rw),"len":_sm(ln)})
    return bins

# ═══════════════════════════════════════════════════════════════
# Action Profile
# ═══════════════════════════════════════════════════════════════
def _actions(episodes):
    tot = {}; g = 0
    for e in episodes:
        for a,c in e["actions"].items(): tot[a]=tot.get(a,0)+c; g+=c
    return {a: _pct(tot[a],g) for a in sorted(tot, key=lambda x:-tot[x])}

# ═══════════════════════════════════════════════════════════════
# Reward Balance
# ═══════════════════════════════════════════════════════════════
def _balance(df, rcols):
    bl = {}
    for c in rcols:
        v = pd.to_numeric(df[c], errors='coerce').fillna(0).values.astype(np.float64)
        ab = float(np.abs(v).sum())
        bl[c] = {"mean":float(v.mean()),"std":float(v.std()),"min":float(v.min()),
            "max":float(v.max()),"median":float(np.median(v)),"abs_sum":ab,
            "pos_frac":float((v>0).sum()/max(1,len(v)))}
    ta = sum(b["abs_sum"] for b in bl.values()) or 1.0
    for c in bl: bl[c]["pct"] = bl[c]["abs_sum"]/ta*100
    return bl

# ═══════════════════════════════════════════════════════════════
# Single Run Analysis + Print
# ═══════════════════════════════════════════════════════════════
def _analyze_run(df, ri):
    label = ri["label"]; rcols = _reward_cols(df)
    episodes = _build_episodes(df)
    ne = len(episodes); nw = sum(1 for e in episodes if e["won"]); nd = sum(1 for e in episodes if e["died"])

    el = np.array([e["length"] for e in episodes], dtype=np.float32) if episodes else np.array([])
    er = np.array([e["reward_sum"] for e in episodes], dtype=np.float32) if episodes else np.array([])
    ep = np.array([e["goal_progress"] for e in episodes], dtype=np.float32) if episodes else np.array([])
    es = np.array([e["avg_speed"] for e in episodes], dtype=np.float32) if episodes else np.array([])

    ls = _per_level(episodes); cv = _curve(episodes, 5)
    ap = _actions(episodes); bl = _balance(df, rcols) if rcols else {}

    causes = {}
    for e in episodes:
        if e["died"] and e["cause"]: causes[e["cause"]]=causes.get(e["cause"],0)+1

    ax = pd.to_numeric(df['x'], errors='coerce').fillna(0).values

    summary = {"label":label,"run_info":ri,"total_steps":len(df),"episodes":ne,
        "wins":nw,"deaths":nd,"win_rate":_pct(nw,ne),
        "ep_len_mean":_sm(el),"ep_len_median":_smed(el),"ep_len_std":_ssd(el),
        "ep_reward_mean":_sm(er),"ep_reward_median":_smed(er),"ep_reward_std":_ssd(er),
        "ep_progress_mean":_sm(ep),"ep_speed_mean":_sm(es),"max_x":float(ax.max()),
        "level_stats":ls,"curve":cv,"action_prof":ap,"balance":bl,"causes":causes}

    # ── PRINT ─────────────────────────────────────────────────────────
    print(f"\n{'━'*W}")
    print(f"  📋  {label.upper()}")
    print(f"  🏷   persona={ri['persona']}  skill={ri['skill']}  arch={ri['arch']}")
    print(f"{'━'*W}")

    if ne == 0:
        print("  ⚠  No completed episodes."); return summary

    print(f"\n  ┌─ OVERVIEW {'─'*58}")
    print(f"  │ Steps: {len(df):>10,}    Episodes: {ne:>6,}    Levels: {len(ls)}")
    print(f"  │ Wins:  {nw:>10,}    Deaths:   {nd:>6,}    Win Rate: {_pct(nw,ne):.1f}%")
    print(f"  │ Max X: {float(ax.max()):>10.1f}    Avg Progress: {_sm(ep):>+.2f}")

    print(f"\n  ┌─ EPISODE STATS {'─'*53}")
    print(f"  │ {'':>20}  {'Mean':>10}  {'Median':>10}  {'StdDev':>10}")
    print(f"  │ {'Episode Length':>20}  {_sm(el):>10.0f}  {_smed(el):>10.0f}  {_ssd(el):>10.0f}")
    print(f"  │ {'Episode Reward':>20}  {_sm(er):>10.3f}  {_smed(er):>10.3f}  {_ssd(er):>10.3f}")
    print(f"  │ {'Avg Speed':>20}  {_sm(es):>10.3f}  {_smed(es):>10.3f}  {_ssd(es):>10.3f}")
    print(f"  │ {'Goal Progress':>20}  {_sm(ep):>10.3f}  {_smed(ep):>10.3f}  {_ssd(ep):>10.3f}")

    if causes:
        print(f"\n  ┌─ DEATH CAUSES {'─'*54}")
        for c, n in sorted(causes.items(), key=lambda x:-x[1]):
            print(f"  │ {c:<12} {n:>5}x  {_bar(_pct(n,nd),100,15)} {_pct(n,nd):>5.1f}%")

    print(f"\n  ┌─ PER-LEVEL BREAKDOWN {'─'*47}")
    print(f"  │ {'Level':<14} {'Vis':>5} {'Win':>4} {'WR%':>6} {'Die':>4} {'AvgLen':>7} {'AvgRwd':>8} {'TopCause':<10}")
    print(f"  │ {'─'*66}")
    for lvl, s in sorted(ls.items()):
        tc = max(s["causes"], key=s["causes"].get) if s["causes"] else "-"
        print(f"  │ {str(lvl):<14} {s['visits']:>5} {s['wins']:>4} {s['win_rate']:>5.1f}% "
              f"{s['deaths']:>4} {s['ep_len_mean']:>7.0f} {s['reward_mean']:>+8.2f} {tc:<10}")

    if len(cv) > 1:
        print(f"\n  ┌─ LEARNING CURVE {'─'*52}")
        print(f"  │ {'Bin':>4} {'Eps':>5} {'WR%':>7} {'AvgRwd':>9} {'AvgLen':>7}")
        print(f"  │ {'─'*36}")
        for b in cv:
            print(f"  │ {b['bin']:>4} {b['eps']:>5} {b['wr']:>6.1f}% {b['rwd']:>+9.3f} {b['len']:>7.0f}  {_bar(b['wr'],100,8)}")
        d = cv[-1]["wr"] - cv[0]["wr"]
        if d > 5: print(f"  │ 📈 IMPROVING: {cv[0]['wr']:.0f}% → {cv[-1]['wr']:.0f}%")
        elif d < -5: print(f"  │ 📉 DECLINING: {cv[0]['wr']:.0f}% → {cv[-1]['wr']:.0f}%")
        else: print(f"  │ ➡️  FLAT: {cv[0]['wr']:.0f}% → {cv[-1]['wr']:.0f}%")

    # Actions
    idle = ap.get("IDLE", 0.0)
    ra = ["RIGHT","RIGHT+JUMP","RUN+RIGHT","RUN+RIGHT+JUMP"]
    la = ["LEFT","LEFT+JUMP","RUN+LEFT","RUN+LEFT+JUMP"]
    rp = sum(ap.get(a,0) for a in ra); lp = sum(ap.get(a,0) for a in la)
    jp = sum(ap.get(a,0) for a in ap if "JUMP" in a)
    print(f"\n  ┌─ ACTION PROFILE {'─'*52}")
    print(f"  │ Right: {rp:>5.1f}%  {_bar(rp,100,12)}    Left: {lp:>5.1f}%  {_bar(lp,100,12)}")
    print(f"  │ Jump:  {jp:>5.1f}%  {_bar(jp,100,12)}    Idle: {idle:>5.1f}%  {_bar(idle,100,12)}")
    if idle > 15: print(f"  │ ⚠ High IDLE — agent may be stuck")
    print(f"  │ R/L Ratio: {rp/max(0.1,lp):.2f}x")

    if bl:
        print(f"\n  ┌─ REWARD BALANCE {'─'*52}")
        print(f"  │ {'Component':<16} {'Share':>6} {'Mean':>10} {'Std':>10} {'Pos%':>6}")
        print(f"  │ {'─'*52}")
        for c, b in sorted(bl.items(), key=lambda x:-x[1]["pct"]):
            f = " ⚠" if b["pct"]>50 else ""
            print(f"  │ {c:<16} {b['pct']:>5.1f}% {b['mean']:>+10.5f} {b['std']:>10.5f} {b['pos_frac']*100:>5.1f}%{f}")
        top = max(bl.items(), key=lambda x:x[1]["pct"])
        if top[1]["pct"] > 60: print(f"  │ ⚠ '{top[0]}' dominates at {top[1]['pct']:.0f}%")
        elif top[1]["pct"] < 40: print(f"  │ ✅ Well-balanced (top: {top[1]['pct']:.0f}%)")

    print(f"  └{'─'*W}")
    return summary

# ═══════════════════════════════════════════════════════════════
# Cross-Run Comparison
# ═══════════════════════════════════════════════════════════════
def _compare(sums):
    print(f"\n{'═'*W}")
    print("  📊  CROSS-RUN COMPARISON")
    print(f"{'═'*W}")
    print(f"  {'Run':<34} {'WR%':>6} {'Wins':>5} {'Die':>5} {'AvgEp':>7} {'AvgRwd':>8} {'MaxX':>6}")
    print(f"  {'─'*(W-2)}")
    ranked = sorted(sums.items(), key=lambda x:-x[1].get('win_rate',0))
    for n, s in ranked:
        print(f"  {n[:32]:<34} {s['win_rate']:>5.1f}% {s['wins']:>5} {s['deaths']:>5} "
              f"{s['ep_len_mean']:>7.0f} {s['ep_reward_mean']:>+8.3f} {s['max_x']:>6.1f}")
    if len(ranked) >= 2:
        print(f"\n  🏆 Best:  {ranked[0][0]} ({ranked[0][1]['win_rate']:.1f}%)")
        print(f"  💀 Worst: {ranked[-1][0]} ({ranked[-1][1]['win_rate']:.1f}%)")

    # By architecture
    ba = {}
    for n, s in sums.items():
        a = s.get("run_info",{}).get("arch","?"); ba.setdefault(a,[]).append(s)
    if len(ba) > 1:
        print(f"\n  📐 BY ARCHITECTURE:")
        for a, rs in sorted(ba.items()):
            wrs = [r["win_rate"] for r in rs]
            print(f"     {a:<12} avg WR: {np.mean(wrs):.1f}%  ({len(rs)} runs)")

    # By persona
    bp = {}
    for n, s in sums.items():
        p = s.get("run_info",{}).get("persona","?"); bp.setdefault(p,[]).append(s)
    if len(bp) > 1:
        print(f"\n  🎭 BY PERSONA:")
        for p, rs in sorted(bp.items(), key=lambda x:-np.mean([r["win_rate"] for r in x[1]])):
            wrs = [r["win_rate"] for r in rs]
            print(f"     {p:<18} avg WR: {np.mean(wrs):.1f}%  best: {np.max(wrs):.1f}%  ({len(rs)} runs)")
    print(f"\n{'═'*W}\n")

# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
def analyze_agent():
    print(f"\n{'═'*W}")
    print("  🤖  PEAK AGENT PERFORMANCE ANALYZER v2")
    print(f"{'═'*W}")
    all_f = get_all_log_files()
    if not all_f: print("  ❌ No logs found."); return
    print(f"\n  📂 Found {len(all_f)} CSV log(s):\n")
    runs = {}
    for f in all_f:
        ri = _parse_run_id(f); label = ri["label"]
        print(f"  • {label:<40}", end="")
        try:
            df, tr = _load_csv(f)
            if df is None or df.empty: print("  ⚠ empty"); continue
            print(f"  {tr:>10,} rows" + (" (sampled)" if len(df)<tr else ""))
            runs[label] = (df, ri)
        except Exception as e: print(f"  ⚠ {e}")
    if not runs: print("\n  ❌ All failed."); return
    print(f"\n  Analyzing {len(runs)} run(s)...")
    sums = {}
    for label, (df, ri) in runs.items():
        sums[label] = _analyze_run(df, ri)
    if len(sums) > 1: _compare(sums)
    print("  ✅ Analysis complete.\n")

if __name__ == "__main__":
    analyze_agent()
