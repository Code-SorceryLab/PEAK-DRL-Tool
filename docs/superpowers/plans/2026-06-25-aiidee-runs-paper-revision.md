# AIIDEE-Runs — Paper Revision Plan (CoG reject → resubmission)

**Branch:** `AIIDEE-Runs` (from `vLLm`). **Commits authorized on this branch only** (standing no-commit rule applies elsewhere).
**Confirmed:** 3 seeds · full A–H matrix (24 runs) · replace Max Rew · runs execute on the user's server.

Addresses the reviewers' "path to acceptance, cheapest first." The case study is re-run with the **fixed code**
(working PBRS gradient, capped Simple movement, `norm_reward=True`, trustworthy stochastic per-level eval), so the
new Table 2 is a *defensible* result rather than an illustration.

## Run design (Tables 1 & 2)

| Run | Extractor | Persona | Budget |
|---|---|---|---|
| A | LightMobile (~18K) | Simple | Novice (1M) |
| B | LightMobile | Simple | Expert (8M) |
| C | LightMobile | Pathfinder | Novice (1M) |
| D | LightMobile | Pathfinder | Expert (8M) |
| E | SpatialAttention (~77K) | Simple | Novice (1M) |
| F | SpatialAttention | Simple | Expert (8M) |
| G | SpatialAttention | Pathfinder | Novice (1M) |
| H | SpatialAttention | Pathfinder | Expert (8M) |

Each config × **3 seeds**. Personas: Simple=`platformer_simple`, Pathfinder=`platformer_adept`.

## Deliverables (cheapest-first)

### D1 — Multi-seed runs + CIs (reviewer #1, highest leverage)
- Re-run all 8 configs × 3 seeds via `code/scripts/run_paper_matrix.py` (already built + robust:
  `+out_root=` isolation, marker-detect kill for the train hang-at-exit, JSON-safe eval).
- Add a `--priority` mode (C/D + E/F = 12 runs) for the reviewer's stated minimum; keep full-8 default.
- Output per config: **win-rate mean ± 95% CI** per level (Mario1-1, Mario1-2), **failure-mode taxonomy with CIs**
  (Stall/Pit/Enemy/Spike/OOB/Timeout), **training time**. **Report STOCHASTIC eval** (deterministic is a stall-trap
  artifact; documented in methods).
- *User runs the matrix on the server; this repo provides the orchestrator + the analysis.*

### D2 — Free reporting fixes (reviewer #3)
- **Max Rew:** REPLACE the `max_x_seen` reward-share (an `agent_analyzer._balance()` artifact mixing telemetry cols)
  with a defensible metric (correct per-component reward share, or drop the column). Fix in the analysis code.
- **Cross-persona Avg Rew:** do not present Simple vs Pathfinder reward as comparable (PBRS inflates magnitude);
  report per-persona only / drop from cross-config comparison.
- **Figure 2 labels:** PDF not in repo → deliver corrected caption + label text in `docs/paper_corrections.md`
  (optionally regenerate clean annotated screenshots on request).

### D3 — Authoring-cost measurement (reviewer #2)
- `code/scripts/authoring_cost.py`: produce one PEAK level variant (e.g., widen a pit / add an enemy in an ASCII
  level), measure **edit size** (chars/lines/files changed) + **wall-clock**, framed honestly against the same change
  in Unity (tilemap + rebuild) / Gym-Retro ROM (binary edit). Emits a small Challenge-3 evidence table.

### D4 — Dijkstra on/off ablation (reviewer #4, nice-to-have)
- One strong config (SpatialAttention + Pathfinder) with the Dijkstra obs channel ON vs OFF (verify/add the config
  flag in the obs builder), comparing failure-mode distributions. Hook into the orchestrator (`--ablation`).

### D5 — "Full Case Study Analysis" menu function (primary ask)
- `code/scripts/case_study.py` (analysis module) + a new `menu.py` option.
- Ingests `run_paper_matrix.py` results (+ training logs) and emits the corrected **Table 1** (design) and **Table 2**
  (results: win-rate mean ± 95% CI per level, taxonomy with CIs, training time, fixed Max Rew), the authoring-cost
  table, and the Dijkstra ablation. Exports markdown + CSV and prints to terminal.

## Execution order
1. Branch + this plan doc (commit). 2. D2 (free, no compute). 3. D1 `--priority` mode. 4. D5 (menu + module).
5. D3. 6. D4. Commit after each milestone.

## Notes
- New Table 2 numbers will differ from the original (fixed code) — that *is* the defensible result the reviewers want.
- `code/` work from this session is already committed on `vLLm` and carried onto this branch.
