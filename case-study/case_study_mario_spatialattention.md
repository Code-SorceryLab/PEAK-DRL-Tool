# Case Study: Super Mario Bros. — SpatialAttention Architecture

## Overview

This case study evaluates PEAK agents trained on Super Mario Bros. levels 1-1 and 1-2
using the SpatialAttention architecture across two personas (adept, simple) and two
training budgets (novice, expert).

---

## Results Summary

| Run | Persona | Budget | Win Rate | Wins | Deaths | Avg Reward | Max X |
|-----|---------|--------|----------|------|--------|------------|-------|
| simple \| novice \| spatialattention | simple | novice | **41.6%** | 87 | 122 | +0.887 | 195.4 |
| adept \| novice \| spatialattention | adept | novice | 22.8% | 42 | 142 | +11.819 | 194.7 |
| adept \| expert \| spatialattention | adept | expert | 17.3% | 24 | 115 | +15.553 | 194.2 |
| simple \| expert \| spatialattention | simple | expert | 14.3% | 45 | 270 | -2.161 | 195.4 |

> ⚠️ **Notable:** Budget did not follow the expected pattern. Novice runs outperformed expert runs on this architecture, with simple novice achieving the highest win rate overall at 41.6%.

---

## Per-Level Breakdown

### simple | novice
| Level | Visits | Wins | Win Rate | Deaths | Avg Length | Avg Reward | Top Death Cause |
|-------|--------|------|----------|--------|------------|------------|-----------------|
| Mario 1-1 | 136 | 61 | **44.9%** | 75 | 4,001 steps | +1.15 | Pit |
| Mario 1-2 | 73 | 26 | **35.6%** | 47 | 3,453 steps | +0.39 | Stall |

### adept | novice
| Level | Visits | Wins | Win Rate | Deaths | Avg Length | Avg Reward | Top Death Cause |
|-------|--------|------|----------|--------|------------|------------|-----------------|
| Mario 1-1 | 184 | 42 | 22.8% | 142 | 4,336 steps | +11.82 | Enemy |

### adept | expert
| Level | Visits | Wins | Win Rate | Deaths | Avg Length | Avg Reward | Top Death Cause |
|-------|--------|------|----------|--------|------------|------------|-----------------|
| Mario 1-1 | 139 | 24 | 17.3% | 115 | 5,737 steps | +15.55 | Pit |

### simple | expert
| Level | Visits | Wins | Win Rate | Deaths | Avg Length | Avg Reward | Top Death Cause |
|-------|--------|------|----------|--------|------------|------------|-----------------|
| Mario 1-1 | 174 | 2 | 1.1% | 172 | 983 steps | -4.25 | OOB |
| Mario 1-2 | 141 | 43 | 30.5% | 98 | 4,440 steps | +0.42 | Stall |

---

## Learning Curves

### simple | novice
| Bin | Episodes | Win Rate | Avg Reward | Avg Length |
|-----|----------|----------|------------|------------|
| 1 | 41 | 39.0% | +0.578 | 5,427 |
| 2 | 41 | 26.8% | -0.460 | 3,030 |
| 3 | 41 | 51.2% | +1.770 | 3,682 |
| 4 | 41 | 43.9% | +1.172 | 3,228 |
| 5 | 45 | 46.7% | +1.330 | 3,692 |

> 📈 **Trend:** IMPROVING — 39% → 47%

### adept | novice
| Bin | Episodes | Win Rate | Avg Reward | Avg Length |
|-----|----------|----------|------------|------------|
| 1 | 36 | 8.3% | +13.281 | 5,376 |
| 2 | 36 | 27.8% | +10.878 | 3,868 |
| 3 | 36 | 30.6% | +10.857 | 3,765 |
| 4 | 36 | 16.7% | +12.405 | 4,692 |
| 5 | 40 | 30.0% | +11.691 | 4,016 |

> 📈 **Trend:** IMPROVING — 8% → 30%

### adept | expert
| Bin | Episodes | Win Rate | Avg Reward | Avg Length |
|-----|----------|----------|------------|------------|
| 1 | 27 | 0.0% | +5.745 | 3,327 |
| 2 | 27 | 0.0% | -0.574 | 1,340 |
| 3 | 27 | 0.0% | +3.975 | 2,800 |
| 4 | 27 | 48.1% | +34.808 | 10,686 |
| 5 | 31 | 35.5% | +31.455 | 9,912 |

> 📈 **Trend:** IMPROVING — 0% → 35%

### simple | expert
| Bin | Episodes | Win Rate | Avg Reward | Avg Length |
|-----|----------|----------|------------|------------|
| 1 | 63 | 0.0% | -4.391 | 1,647 |
| 2 | 63 | 0.0% | -4.512 | 596 |
| 3 | 63 | 6.3% | -3.160 | 1,210 |
| 4 | 63 | 31.7% | +0.451 | 4,210 |
| 5 | 63 | 33.3% | +0.805 | 4,988 |

> 📈 **Trend:** IMPROVING — 0% → 33%

---

## Death Cause Breakdown

| Run | OOB | Stall | Pit | Enemy |
|-----|-----|-------|-----|-------|
| simple \| novice | 4.1% | 32.0% | 33.6% | 30.3% |
| adept \| novice | 3.5% | 27.5% | 30.3% | 38.7% |
| adept \| expert | 28.7% | 17.4% | 41.7% | 12.2% |
| simple \| expert | 50.7% | 17.8% | 17.4% | 14.1% |

---

## Action Profile

| Run | Right | Left | Jump | Idle | R/L Ratio |
|-----|-------|------|------|------|-----------|
| simple \| novice | 13.8% | 9.7% | 50.9% | 4.1% | 1.42x |
| adept \| novice | 11.4% | 9.4% | 50.6% | 4.6% | 1.22x |
| adept \| expert | 11.1% | 8.1% | 53.0% | 4.1% | 1.37x |
| simple \| expert | 10.9% | 8.9% | 52.4% | 4.0% | 1.23x |

---

## Key Findings

- **Budget inversion.** Unlike LightMobile, SpatialAttention novice runs outperformed
  expert runs. Simple novice achieved 41.6% — the highest Mario win rate across both
  architectures — while simple expert reached only 14.3%. This suggests the larger
  architecture found a working policy faster and may have overfit or destabilized with
  extended training.
- **Mario 1-1 solved.** Simple novice achieved 44.9% on Mario 1-1, which produced
  zero completions across all LightMobile runs. SpatialAttention's spatial reasoning
  capacity appears to be the enabling factor for this level.
- **Late-training breakthrough on adept expert.** The adept expert agent spent three
  bins at 0% before jumping to 48.1% in bin 4 and settling at 35.5%, reproducing the
  threshold learning pattern observed in the Mega Man LightMobile run.
- **OOB dominates simple expert deaths at 50.7%**, suggesting the agent learned
  aggressive rightward movement without learning to stay in bounds — a sign of
  reward-chasing without spatial constraint.
- **`max_x_seen` dominated at 98–99%** across all four runs, consistent with LightMobile
  results and confirming this is a structural property of the reward design at this scale.
