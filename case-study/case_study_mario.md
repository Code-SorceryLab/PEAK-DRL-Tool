# Case Study: Super Mario Bros. — LightMobile Architecture

## Overview

This case study evaluates PEAK agents trained on Super Mario Bros. levels 1-1 and 1-2
using the LightMobile architecture across two personas (adept, simple) and two training
budgets (novice, expert).

---

## Results Summary

| Run | Persona | Budget | Win Rate | Wins | Deaths | Avg Reward | Max X |
|-----|---------|--------|----------|------|--------|------------|-------|
| dijkstra \| expert \| lightmobile | adept | expert | **23.0%** | 45 | 151 | +12.220 | 195.4 |
| dijkstra \| novice \| lightmobile | adept | novice | 0.0% | 0 | 243 | +6.579 | 45.0 |

---

## Per-Level Breakdown

| Level | Visits | Wins | Win Rate | Deaths | Avg Length | Avg Reward | Top Death Cause |
|-------|--------|------|----------|--------|------------|------------|-----------------|
| Mario 1-1 | 54 | 0 | 0.0% | 54 | 2,950 steps | +4.79 | OOB |
| Mario 1-2 | 142 | 45 | **31.7%** | 97 | 4,502 steps | +15.05 | Stall |

---

## Learning Curves

### adept | expert
| Bin | Episodes | Win Rate | Avg Reward | Avg Length |
|-----|----------|----------|------------|------------|
| 1 | 39 | 0.0% | +6.234 | 3,433 |
| 2 | 39 | 17.9% | +9.983 | 3,588 |
| 3 | 39 | 23.1% | +10.311 | 3,257 |
| 4 | 39 | 43.6% | +16.901 | 4,735 |
| 5 | 40 | 30.0% | +17.535 | 5,326 |

> 📈 **Trend:** IMPROVING — 0% → 30%

### adept | novice
| Bin | Episodes | Win Rate | Avg Reward | Avg Length |
|-----|----------|----------|------------|------------|
| 1 | 48 | 0.0% | +3.993 | 2,638 |
| 2 | 48 | 0.0% | +2.791 | 2,175 |
| 3 | 48 | 0.0% | +7.347 | 3,403 |
| 4 | 48 | 0.0% | +8.502 | 3,782 |
| 5 | 51 | 0.0% | +10.046 | 4,309 |

> ➡️ **Trend:** FLAT — 0% → 0%

---

## Death Cause Breakdown

### adept | expert
| Cause | Count | Share |
|-------|-------|-------|
| Pit | 50 | 33.1% |
| Stall | 48 | 31.8% |
| Enemy | 32 | 21.2% |
| OOB | 21 | 13.9% |

### adept | novice
| Cause | Count | Share |
|-------|-------|-------|
| Stall | 145 | 59.7% |
| OOB | 44 | 18.1% |
| Enemy | 37 | 15.2% |
| Pit | 17 | 7.0% |

---

## Action Profile

| Run | Right | Left | Jump | Idle | R/L Ratio |
|-----|-------|------|------|------|-----------|
| adept \| expert | 11.1% | 9.6% | 49.9% | 4.5% | 1.16x |
| adept \| novice | 8.5% | 10.7% | 50.0% | 5.3% | 0.80x |

---

## Key Findings

- **Budget is decisive.** The novice budget (800K steps) produced zero completions on
  both levels. The expert budget enabled the agent to reach a 23.0% overall win rate.
- **Level asymmetry.** Mario 1-2 was solved at 31.7% while Mario 1-1 produced zero
  completions across all runs. The open surface layout of 1-1 appears to present a
  harder exploration problem under LightMobile capacity than the constrained corridor of 1-2.
- **Failure modes.** The expert agent's deaths split across pits (33%), stalling (32%),
  enemies (21%), and OOB (14%), reflecting an agent that moves but does not yet navigate
  reliably. The novice agent stalled in 60% of deaths, indicating it never developed
  consistent forward momentum.
- **Reward dominance.** `max_x_seen` accounted for 98.7% (expert) and 91.1% (novice) of
  cumulative reward, indicating that secondary shaping terms had negligible influence on
  learned behavior at this scale.
