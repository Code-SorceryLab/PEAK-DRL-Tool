"""
Read one or more metrics-JSON files (as saved by evaluate.py) and
plot a radar chart for the mean of each balancing dimension.

Usage
-----
python -m code.scripts.analyze_metrics \
       --files results/flappy_run-A.json results/flappy_run-B.json \
       --out   radar_flappy.png
"""

import argparse, json, itertools, pathlib, numpy as np, matplotlib.pyplot as plt

# -------- CLI ---------
p = argparse.ArgumentParser()
p.add_argument("--files", nargs="+", required=True, help="metrics JSON files")
p.add_argument("--out",   default="radar.png",    help="output image")
args = p.parse_args()

# -------- load --------
all_metrics = []
for f in args.files:
    with open(f) as fh:
        all_metrics.extend(json.load(fh))

if not all_metrics:
    raise SystemExit("No metric entries found!")

# -------- aggregate by balance dimension ----
DIM_KEYS = {
    "DIF": ["DIF_AvgGap", "DIF_dGapPerPipe"],
    "SKI": ["SKI_MeanTimingErr", "SKI_APS"],
    "PAC": ["PAC_MeanIdleRatio"],
    "LEN": ["LEN_RunTime", "LEN_PipesCleared"],
    "FAI": ["FAI_MinClearance"],
    "PRO": ["PRO_ScoreVelocity", "PRO_dDifficulty_dt"],
}

dim_means = {}
for dim, keys in DIM_KEYS.items():
    vals = [np.mean([m.get(k, 0) for k in keys]) for m in all_metrics]
    dim_means[dim] = float(np.mean(vals))

# -------- radar chart (one plot, no seaborn) ----
labels   = list(dim_means.keys())
stats    = list(dim_means.values())
angles   = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
stats   += stats[:1]          # close the loop
angles  += angles[:1]

fig = plt.figure(figsize=(6, 6))
ax  = fig.add_subplot(111, polar=True)
ax.plot(angles, stats, linewidth=2)
ax.fill(angles, stats, alpha=0.25)
ax.set_thetagrids(np.degrees(angles[:-1]), labels)
ax.set_title("Flappy Bird – Balance Radar", pad=20)
ax.grid(True)

plt.tight_layout()
fig.savefig(args.out, dpi=150)
print("✓ radar saved →", args.out)
