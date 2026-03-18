# Case Study: Mega Man — LightMobile Architecture

## Overview

This case study evaluates PEAK agents trained on Mega Man stages (MM-Stage1 and MM-Train4)
using the LightMobile architecture across two personas (adept, simple) and two training
budgets (novice, expert).

---

## Results Summary

| Run | Persona | Budget | Win Rate | Wins | Deaths | Avg Reward | Max X |
|-----|---------|--------|----------|------|--------|------------|-------|
| mega \| expert \| lightmobile | adept | expert | **12.4%** | 121 | 857 | -1.741 | 4,331.2 |
| simple \| expert \| lightmobile | simple | expert | 1.1% | 9 | 808 | -1.635 | 5,130.1 |
| mega \| novice \| lightmobile | adept | novice | 0.0% | 0 | 864 | -4.927 | 972.8 |
| simple \| novice \| lightmobile | simple | novice | 0.0% | 0 | 865 | -3.119 | 971.9 |

---

## Per-Level Breakdown

### adept | expert
| Level | Visits | Wins | Win Rate | Deaths | Avg Length | Avg Reward | Top Death Cause |
|-------|--------|------|----------|--------|------------|------------|-----------------|
| MM-Stage1 | 134 | 121 | **90.3%** | 13 | 4,773 steps | +22.37 | Projectile |
| MM-Train4 | 844 | 0 | 0.0% | 844 | 190 steps | -5.57 | Spike |

### simple | expert
| Level | Visits | Wins | Win Rate | Deaths | Avg Length | Avg Reward | Top Death Cause |
|-------|--------|------|----------|--------|------------|------------|-----------------|
| MM-Stage1 | 46 | 9 | 19.6% | 37 | 13,665 steps | +47.11 | Projectile |
| MM-Train4 | 771 | 0 | 0.0% | 771 | 207 steps | -4.54 | Spike |

### adept | novice
| Level | Visits | Wins | Win Rate | Deaths | Avg Length | Avg Reward | Top Death Cause |
|-------|--------|------|----------|--------|------------|------------|-----------------|
| MM-Train4 | 864 | 0 | 0.0% | 864 | 556 steps | -4.93 | Spike |

### simple | novice
| Level | Visits | Wins | Win Rate | Deaths | Avg Length | Avg Reward | Top Death Cause |
|-------|--------|------|----------|--------|------------|------------|-----------------|
| MM-Train4 | 865 | 0 | 0.0% | 865 | 852 steps | -3.12 | Spike |

---

## Learning Curves

### adept | expert
| Bin | Episodes | Win Rate | Avg Reward | Avg Length |
|-----|----------|----------|------------|------------|
| 1 | 195 | 0.0% | -6.345 | 92 |
| 2 | 195 | 0.0% | -6.280 | 99 |
| 3 | 195 | 0.0% | -6.220 | 120 |
| 4 | 195 | 0.0% | -4.339 | 366 |
| 5 | 198 | 61.1% | +14.234 | 3,372 |

> 📈 **Trend:** IMPROVING — 0% → 61%

### simple | expert
| Bin | Episodes | Win Rate | Avg Reward | Avg Length |
|-----|----------|----------|------------|------------|
| 1 | 163 | 0.0% | -5.490 | 86 |
| 2 | 163 | 0.0% | -5.245 | 128 |
| 3 | 163 | 0.0% | -5.151 | 147 |
| 4 | 163 | 0.0% | -3.775 | 316 |
| 5 | 165 | 5.5% | +11.326 | 4,110 |

> 📈 **Trend:** IMPROVING — 0% → 5%

### adept | novice
| Bin | Episodes | Win Rate | Avg Reward | Avg Length |
|-----|----------|----------|------------|------------|
| 1 | 172 | 0.0% | -6.399 | 92 |
| 2 | 172 | 0.0% | -6.122 | 120 |
| 3 | 172 | 0.0% | -6.191 | 113 |
| 4 | 172 | 0.0% | -5.482 | 198 |
| 5 | 176 | 0.0% | -0.542 | 2,217 |

> ➡️ **Trend:** FLAT — 0% → 0%

### simple | novice
| Bin | Episodes | Win Rate | Avg Reward | Avg Length |
|-----|----------|----------|------------|------------|
| 1 | 173 | 0.0% | -5.468 | 96 |
| 2 | 173 | 0.0% | -5.218 | 137 |
| 3 | 173 | 0.0% | -5.255 | 134 |
| 4 | 173 | 0.0% | -3.040 | 543 |
| 5 | 173 | 0.0% | +3.385 | 3,351 |

> ➡️ **Trend:** FLAT — 0% → 0%

---

## Death Cause Breakdown

| Run | Spike | Enemy | Projectile | Pit |
|-----|-------|-------|------------|-----|
| adept \| expert | 78.1% | 17.7% | 4.2% | — |
| simple \| expert | 72.2% | 22.8% | 4.1% | 1.0% |
| adept \| novice | 78.9% | 17.5% | 3.6% | — |
| simple \| novice | 64.9% | 26.6% | 8.6% | — |

---

## Action Profile

| Run | Right | Left | Jump | Idle | R/L Ratio |
|-----|-------|------|------|------|-----------|
| adept \| expert | 3.2% | 3.0% | 47.8% | 1.7% | 1.09x |
| simple \| expert | 2.2% | 1.4% | 33.8% | 0.9% | 1.58x |
| adept \| novice | 2.9% | 2.9% | 50.1% | 2.0% | 0.97x |
| simple \| novice | 3.5% | 2.9% | 48.9% | 1.7% | 1.23x |

---

## Key Findings

- **Budget is decisive.** All novice runs produced 0% win rates. Both expert runs showed
  late-training improvement, with adept rising sharply from 0% to 61% in the final bin.
- **Persona gap is large.** On MM-Stage1, adept achieved 90.3% vs simple's 19.6% — a
  4.6x difference — demonstrating that Dijkstra-guided shaping provides a critical
  advantage on the Mega Man stage structure.
- **MM-Train4 is unsolved.** Zero completions were recorded on MM-Train4 across all four
  runs. Spike deaths dominated at 65–79%, identifying hazard avoidance as the primary
  unresolved challenge at this architecture and budget scale.
- **Late-training breakthrough.** The adept expert agent spent the first four training
  bins at 0% before jumping to 61% in the final bin, suggesting the agent found a
  reliable strategy late rather than gradually improving. This pattern warrants
  further study with longer training runs.
- **Reward dominance.** `max_x_seen` accounted for 97–99% of cumulative reward across
  all four runs, indicating that game-specific shaping components (climb, combat,
  vertical, boss) contributed negligibly to learned behavior at this scale.
