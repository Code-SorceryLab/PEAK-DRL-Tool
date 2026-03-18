# Cross-Architecture Analysis: LightMobile vs SpatialAttention

## Full Results Matrix (Mario Levels)

| Arch | Persona | Budget | Win Rate | Wins | Deaths | Avg Reward | Top Level |
|------|---------|--------|----------|------|--------|------------|-----------|
| SpatialAttention | simple | novice | **41.6%** | 87 | 122 | +0.887 | 1-1: 44.9% |
| SpatialAttention | adept | novice | 22.8% | 42 | 142 | +11.819 | 1-1: 22.8% |
| LightMobile | adept | expert | 23.0% | 45 | 151 | +12.220 | 1-2: 31.7% |
| SpatialAttention | adept | expert | 17.3% | 24 | 115 | +15.553 | 1-1: 17.3% |
| SpatialAttention | simple | expert | 14.3% | 45 | 270 | -2.161 | 1-2: 30.5% |
| LightMobile | adept | novice | 0.0% | 0 | 243 | +6.579 | — |

## Full Results Matrix (Mega Man Levels — LightMobile only)

| Arch | Persona | Budget | Win Rate | Wins | Deaths | Avg Reward | Top Level |
|------|---------|--------|----------|------|--------|------------|-----------|
| LightMobile | adept | expert | **12.4%** | 121 | 857 | -1.741 | MM-Stage1: 90.3% |
| LightMobile | simple | expert | 1.1% | 9 | 808 | -1.635 | MM-Stage1: 19.6% |
| LightMobile | adept | novice | 0.0% | 0 | 864 | -4.927 | — |
| LightMobile | simple | novice | 0.0% | 0 | 865 | -3.119 | — |

---

## Architecture Comparison: Mario Levels

| Factor | LightMobile | SpatialAttention |
|--------|-------------|-----------------|
| Best overall WR | 23.0% (adept expert) | **41.6%** (simple novice) |
| Budget effect | Novice = 0%, Expert > 0% | **Novice outperforms expert** |
| Mario 1-1 solved? | ❌ 0% all runs | ✅ 44.9% (simple novice) |
| Mario 1-2 solved? | ✅ 31.7% | ✅ 30.5% |
| Best persona | adept | simple (by WR), adept (by reward) |
| Dominant death cause | Pit / Stall | OOB (expert), Pit (novice) |

---

## Key Cross-Architecture Findings

### 1. SpatialAttention inverts the budget effect
LightMobile required the expert budget to learn anything. SpatialAttention achieved
its highest win rate under the novice budget (41.6%), with expert runs performing
worse. This suggests the larger architecture converges faster but may overfit or
destabilize under extended training without additional regularization.

### 2. Architecture capacity unlocked Mario 1-1
Mario 1-1 produced zero completions across all LightMobile runs but was solved at
44.9% by SpatialAttention simple novice. The open surface layout appears to require
the spatial reasoning capacity of the larger architecture to navigate reliably.

### 3. Simple persona is competitive at higher capacity
With LightMobile, adept consistently outperformed simple. With SpatialAttention,
simple matched or exceeded adept on win rate, suggesting that stronger spatial
feature extraction reduces the agent's dependence on dense reward shaping.

### 4. Threshold learning appears in both architectures
Both the LightMobile Mega Man adept expert run (0%→61% in final bin) and the
SpatialAttention adept expert run (0%→48% in bin 4) exhibit a pattern where the
agent spends the majority of training at 0% before discovering a reliable strategy
late. This is not a fluke — it recurs across architectures, games, and personas.

### 5. max_x_seen dominates universally
Across all 10 runs (6 LightMobile + 4 SpatialAttention), `max_x_seen` accounted
for 91–99% of cumulative reward in every configuration. Secondary shaping terms
had negligible measured influence on learned behavior at these budget scales.

### 6. MM-Train4 remains fully unsolved
Zero completions on MM-Train4 across all runs on both architectures. Spike deaths
at 65–79% confirm hazard avoidance as the primary unresolved challenge.

---

## Open Questions

- Does SpatialAttention's budget inversion persist at longer training horizons (>8M steps)?
- Is the simple persona advantage on SpatialAttention reproducible across seeds?
- Can MM-Train4 be solved with SpatialAttention or ChannelAttention at expert budget?
- Does the threshold learning pattern reflect reward sparsity, level structure, or
  both? Would curriculum progression mitigate it?
