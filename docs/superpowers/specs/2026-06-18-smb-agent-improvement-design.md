# SMB / Platformer Agent Improvement — Design

- **Date:** 2026-06-18
- **Status:** Draft for review
- **Author:** Al + Claude (brainstorming session)
- **Goal:** Make the PEAK platformer (Super Mario Bros-style) RL agent actually play well, by fixing the diagnosed root causes of underperformance and standing up an imitation-learning track in parallel.

---

## 1. Background

PEAK trains a Stable-Baselines3 agent on a 2D SMB-style platformer. Observation is a gym `Dict`
(`grids` float32 `(4,21,21)` = [solid, collectible, hazard, dijkstra]; `scalars` float32 `(20,)`);
action is `MultiDiscrete([5,2,2])` = `[move, jump, fire]`. Training uses PPO (also RecurrentPPO/A2C/DQN),
a custom spatial-attention feature extractor, VecNormalize, reward "personas," and a curriculum.

The agent underperforms (best ~41.6% win rate, only on `Mario1-1`/`1-2`). A 12-agent investigation
(2026-06-18) diagnosed the causes; this spec captures the design to fix them. The user's explicit ask:
improve "by all means necessary (imitation learning, inverse DRL, etc)."

## 2. Diagnosis (grounded root causes)

| # | Root cause | Evidence | Type |
|---|---|---|---|
| 1 | **Reward is dominated by horizontal distance** (`max_x_seen` = 98–99% of total reward share); winning/coins/kills ≈ 0%. Agent learns "go right," not "win" → rushes into pits/walls. | `case-study/analysis.md:56-79`, `case-study/peak_results_full.csv` (`max_x_seen_reward_share_pct` 91–99%); `train_platformer.py:282-287` (frontier reward already removed for this reason) | Reward design |
| 2 | **Eval is non-stationary / not comparable.** Eval uses the same curriculum-enabled `PlatformerCore`, a single shared instance across runs; `reset()` mutates persistent curriculum state; goal is **not terminal** (a win chains into the next level); 25% review-jump runs a different level ~1/4 episodes. The "Mario1-2 collapse" is partly a measurement artifact. | `train.py:1129-1199`; `platformer_core.py:437-441,781-822,810-814,763-770,1080-1083`; DummyVecEnv auto-reset | Measurement bug |
| 3 | **The difficulty curriculum is dead.** Only `Mario1-1`/`Mario1-2` are registered in `game_config.yaml`; the 14 `stage_1..14` easy→hard ramp files are never loaded by the trainer. Agent trains on the hardest world with no on-ramp. | `game_config.yaml:56-70`; `config_manager.py:107-110`; `stage_*.txt` referenced only by `level_editor.py` | Config bug |
| 4 | **Curriculum thresholds untunable (double-pop bug).** `advance_threshold`/`fallback_threshold` are `kwargs.pop`'d into dead mastery-gated code first, so YAML values never reach the active batch curriculum (silently uses 0.30/0.20). `curriculum_win_rate` reads a never-written field (permanently stale). | `platformer_core.py:432-434` vs `482-483`; `:438,1694-1703,1732,1780` | Config bug |
| 5 | **Sparse terminal signal + `norm_reward=False` + short horizon.** Base reward = `score_delta` (≈0 most steps); win bonus (+5.0) ≈ passive-survival total; `gamma=0.99` ⇒ ~100-step horizon vs ~10,000-step episodes; `vf_coef` hand-lowered to 0.25 to treat the symptom. | `train.py:986`; `platformer_core.py:751-777`; `train_platformer.py:294-302,326-335`; `code/conf/algo/ppo.yaml` | Learning |
| 6 | **Under-trained.** Shipped models are ~1M steps; prior art needs ~10M for one SMB level. `n_epochs=4` (vs SB3 default 10). The "simple" model degrades with training (never positive eval reward). | `models/eval_logs/*`; `case-study/peak_results_full.csv` (800k); `grid.yaml:23-25`; `ppo.yaml` | Learning |
| 7 | **`model=rppo` crashes** — `RecurrentEvalCallback` is called but never imported. Forecloses the RecurrentPPO path. | `train.py:1166`; imports at `:26,30`; `code/callbacks/RecurrentEvalCallback.py:8` | Latent bug |

**Refinement to cause #1 (found during planning, 2026-06-18):** the "98–99% of reward" figure was
partly an *analyzer artifact* — `agent_analyzer._balance()` computes "share" over all non-standard CSV
columns, mixing logged telemetry (the `max_x_seen` position counter) with actual reward terms; `max_x`
is not a reward component and the `frontier` term is consumed by no persona. The real "go-right-not-win"
incentive is **localized to the `simple`/`default` persona**: its movement term is runtime-measured at
~94% of reward share on a non-winning rightward run, and cumulative movement (~62) out-earns the win
bonus (5.0) by ~12×. The `adept` persona is already win-dominated and its PBRS is correct. The reward
fix (Phase 1) is therefore a tight, conservative cap on the `simple` movement term, not a broad reward
overhaul.

**Imitation/inverse-RL feasibility:** Zero usable demos exist (`manual_play.py` records nothing;
the only logs lack the policy observation). The `imitation` library is version-incompatible (pins
SB3~=2.2 / gym~=0.29 / Py≤3.10 vs this repo's SB3 2.8 / gym 1.2 / Py 3.11). Therefore: a **demo
recorder + DIY behavioral cloning on the installed SB3** is the viable IL route; GAIL/AIRL are a last
resort (Dict-obs discriminator incompatibility + adversarial instability).

## 3. Decisions locked (brainstorming)

- **Sequence:** do the high-confidence fixes (Phase 0 + 1) **and** build the imitation-learning track
  (Phase 2 recorder + DIY BC) in parallel — user chose "build IL track now too."
- **Human demos:** build the recorder regardless; the user decides whether to actually record after
  Phase 1 is measured ("maybe later, build recorder anyway").
- **Target:** competent play across a registered easy→hard curriculum culminating in `Mario1-1`/`1-2`.
- **Compute:** scale `n_envs` to 8–16; multi-hour/overnight runs acceptable (CPU-pinned; MPS blocked).
- **`imitation` library:** NOT used (version conflict). DIY BC on the existing SB3 policy.

## 4. Goals / Non-Goals

**Goals**
- A trustworthy, per-level, stationary evaluation so improvement is measurable.
- The agent's reward and curriculum actually optimize for *finishing levels*, not running right.
- A measurable win-rate improvement over the current ~41.6% baseline on the trained levels.
- A working demonstration recorder + DIY behavioral-cloning warm-start pipeline (ready to use).

**Non-Goals (deferred)**
- GAIL/AIRL inverse RL (Phase 3; last resort, only if BC+PPO plateaus).
- RND/ICM exploration bonuses (Phase 3; only for proven-sparse stages / `enemy_hunter`).
- The Unity port (separate effort, paused).
- Switching RL frameworks or installing the `imitation` library.

## 5. Phase 0 — Trustworthy evaluation (measure first)

**Why first:** every before/after comparison is meaningless on the current drifting eval yardstick.

- Build the **eval env with a fixed level** (per-level eval) — disable the curriculum for eval, mirroring
  the existing megaman/sonic guard (`watch_agent.py:221-223`) for platformer.
- **Terminate the eval episode on goal** so eval measures a clean per-level win/lose outcome (do not
  chain into the next level during eval).
- **Reconstruct / hard-reset the eval env per run** so curriculum/score state can't leak across runs.
- Add a **per-level win-rate** metric (and death-cause breakdown) to eval logging.
- Keep eval `norm_reward=False` for interpretable eval reward.
- **Deliverable:** a clean baseline table (per-level win rate, avg score, death causes) for the current
  best model, produced by the fixed eval — the reference point for all later phases.

## 6. Phase 1 — High-confidence learning fixes

Each change is independently testable; retrain + re-measure on the Phase-0 eval after the batch.

1. **Reward rebalance (highest leverage).** Make goal/goal-progress dominate, not raw `max_x`. Concretely:
   cap or sharply down-weight the horizontal-distance term so the dijkstra-to-goal progress signal and
   the win bonus lead; ensure "reach the goal" beats "edge forward and survive." Target the `platformer`
   personas in `train_platformer.py`. Preserve potential-based-shaping invariance (keep `POTENTIAL_GAMMA`
   == PPO `gamma`).
2. **`norm_reward=True`** (`train.py:986`), with `clip_reward` > win bonus (5.0); keep eval
   `norm_reward=False`.
3. **Horizon/value retune (after #2):** `gamma`≈0.997, `gae_lambda`≈0.97, restore `vf_coef`→~0.5,
   `n_epochs`→~8. Update `POTENTIAL_GAMMA` in all platformer personas to match the new `gamma`.
4. **Register the real curriculum:** add the `stage_1..N` files to `game_config.yaml` (dedupe identical
   stages, verify difficulty-monotonic ordering, note vertical-climb discontinuities), so the agent gets
   an easy on-ramp before `Mario1-1`/`1-2`.
5. **Fix the curriculum double-pop bug** (`platformer_core.py:432-434` vs `482-483`) so YAML thresholds
   reach the active curriculum; wire `_level_window` so `curriculum_win_rate` is real.
6. **Fix the `rppo` NameError** (import `RecurrentEvalCallback` in `train.py`).
7. **Scale + budget:** `n_envs` 8–16 (existing SubprocVecEnv path); train toward ~5–10M steps on the
   registered curriculum. Add entropy/KL hygiene (`target_kl`, `ent_coef`) only if a plateau shows
   entropy collapse.
- **Deliverable:** a retrained agent with a measured per-level win-rate improvement over baseline.

## 7. Phase 2 — Imitation-learning track (built now, used when ready)

1. **Demonstration recorder.** Capture full `(obs['grids'], obs['scalars'], MultiDiscrete action)` per
   frame to `.npz`. Handle the VecNormalize obs-stats consistently (record raw obs + the matching
   `obs_rms`, or record post-normalization obs) so BC inputs match the policy's training distribution.
   Add to the human-play path (`manual_play.py`) and optionally the agent path (`watch_agent.py`).
2. **DIY behavioral cloning** on the **installed SB3 2.8**: build the PPO policy with `MultiInputPolicy`
   + the existing `SpatialAttentionExtractor`, train a supervised loop via `policy.evaluate_actions`
   log-prob over the `[move,jump,fire]` MultiDiscrete heads, then `load_state_dict` into a PPO model and
   fine-tune with the curriculum (lower `ent_coef`/LR early to avoid wiping the value head).
3. **Demos** are recorded by the user when they choose; until then the recorder + BC loop sit ready and
   are unit-tested with synthetic data.
- **Deliverable:** a recorder + a `behavioral_cloning.py` that turns recorded demos into a warm-started
  PPO model, verified end-to-end on synthetic demos.

## 8. Phase 3 — Last resort (out of scope for now)

AIRL with a custom Dict reward-net (BasicRewardNet can't consume Dict obs), or targeted RND for
proven-sparse stages. Only if BC+PPO plateaus and a transferable cross-stage reward is specifically
needed. Not planned in this cycle.

## 9. Measurement / testing strategy

- **Primary metric:** per-level win rate on the Phase-0 fixed eval (vs ~41.6% baseline), plus
  death-cause distribution (pit/wall/stall/enemy) and avg score.
- **Regression safety:** existing `code/tests/` stays green; new code (eval changes, recorder, BC loop)
  gets unit tests (synthetic data for the BC pipeline so it's testable without human demos).
- **Reward-change validation:** confirm potential-based-shaping invariance preserved (`POTENTIAL_GAMMA`
  == `gamma`); sanity-check that reward-share is no longer dominated by `max_x`.
- **A/B discipline:** every training change is re-measured on the *same* fixed eval; no comparisons
  against the old drifting-eval case-study numbers (a fresh baseline is established in Phase 0).

## 10. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Reward rebalance breaks training (the team has been burned by reward shaping before). | Preserve PBRS invariance; change one lever at a time; re-measure on fixed eval; keep the dijkstra signal as the dense guide. |
| Higher `gamma` destabilizes the value fn. | Do it only **after** `norm_reward=True`; short multi-seed sweep. |
| Mis-ordered curriculum causes forgetting/oscillation. | Verify difficulty-monotonic ordering; ensure VecNormalize stats persist across stage changes; track per-stage win rate. |
| BC obs distribution mismatch vs VecNormalize. | Recorder handles obs-stats explicitly; BC inputs normalized to match the policy. |
| Long CPU-bound training. | `n_envs` scaling; accept overnight runs; benchmark per-env throughput first. |
| Changing eval invalidates old case-study comparisons. | Establish a fresh baseline in Phase 0; document that old numbers are not comparable. |

## 11. Implementation phasing (for plans)

Each phase is its own spec→plan→build cycle:
- **Plan A = Phase 0 + Phase 1** (eval integrity + the high-confidence learning fixes) — highest leverage, do first.
- **Plan B = Phase 2** (recorder + DIY BC pipeline) — built in parallel; usable once demos are recorded.
- **Phase 3** is not planned this cycle.

---

### Note on process
Per the user's standing **never-commit** rule, this spec is written but **not committed**.
