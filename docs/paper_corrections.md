# Paper Corrections (reviewer #3 — "free" trust fixes)

These convert presentation issues that erode reviewer trust. Code fixes are in
`code/scripts/agent_analyzer.py` (`_reward_cols` telemetry exclusion; cross-persona
reward caveat). The paper-text/figure changes below are for the manuscript.

## 1. "Max Rew" column (Table 2) — REPLACE

**Original caption:** *"Max Rew: share of accumulated reward from the `max_x_seen` component."*

**Problem:** `max_x_seen` is a monotonic horizontal-position **counter** (telemetry), not a
reward component. The analyzer's `_reward_cols` counted every non-standard CSV column as a
"reward component," so `max_x_seen` (mean ≈ 1386) dominated the computed "share" at ~99%. The
column does not measure what the caption claims and is not a quality signal.

**Fix (code):** `_reward_cols` now restricts the reward balance to the persona's true reward
components (`movement, alive, time, win, death, potential, ...`), excluding all telemetry.

**Fix (paper):** **drop the Max Rew column** from Table 2. If a reward-composition signal is
wanted, report the corrected per-component reward share for one or two configs in text, with the
explicit caveat below. Do not present a single "max_x_seen share" number.

## 2. Cross-persona Avg Rew comparison — DO NOT COMPARE

**Problem:** Pathfinder uses potential-based shaping (Ng et al. 1999), which is policy-invariant
but **inflates raw reward magnitude** relative to Simple. Comparing "Avg Rew" across personas
(Simple vs Pathfinder) is meaningless.

**Fix (paper):** State explicitly that Avg Rew is **within-persona only**. When comparing
configurations across personas, compare **win rate and failure-mode distributions**, never raw
reward. Suggested sentence:

> *Avg Rew is reported per persona and is not comparable across personas: Pathfinder's
> potential-based shaping inflates reward magnitude relative to Simple. Cross-persona comparisons
> use win rate and failure-mode distributions only.*

(The analyzer now prints this caveat in the cross-run comparison.)

## 3. Figure 2 labels

**Problem:** Fig. 2 bundles three sub-images (original SMB 1-1, the level editor, gameplay with
debug overlays) under one low-legibility caption with no guidance on what to look at.

**Fix (paper):** Label the three panels explicitly and tell the reader what each shows:

> **Figure 2.** PEAK's authoring + debugging surface. **(a)** Original *Super Mario Bros.* 1-1 for
> reference. **(b)** Its PEAK reconstruction open in the ASCII/visual level editor (tile palette,
> moving-platform paths). **(c)** The reconstructed level during play with debug overlays enabled:
> the agent-vision grid channels (solid / hazard / Dijkstra) projected onto the world, the per-step
> reward-component trace (top-left), and the predictive jump-arc (cyan). The overlays are the
> mechanism behind Requirement 2 (built-in failure diagnostics).

If feasible, replace (c) with a higher-resolution capture and call out one overlay element with an
arrow/annotation. (I can regenerate annotated screenshots from PEAK on request.)

## 4. Table 2 caption — note the methodology change (for the revised, multi-seed table)

The revised Table 2 is produced by `code/scripts/case_study.py` from the multi-seed matrix
(`run_paper_matrix.py`) with the **fixed code** and a **trustworthy stochastic per-level eval**.
Add to the caption:

> *Win rates are stochastic-policy means over 3 seeds with 95% confidence intervals; deterministic
> evaluation is reported separately as a robustness signal (a greedy policy can enter a fixed action
> loop the anti-stall watchdog terminates, which deflates deterministic win rates). Failure-mode
> percentages are seed means ± 95% CI.*
