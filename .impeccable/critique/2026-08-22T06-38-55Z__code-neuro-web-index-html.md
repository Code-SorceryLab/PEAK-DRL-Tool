---
target: Live training dashboard (code/neuro/web/index.html)
total_score: 23
max_score: 40
na_heuristics: 
p0_count: 1
p1_count: 2
timestamp: 2026-08-22T06-38-55Z
slug: code-neuro-web-index-html
---
Method: dual-agent (A: design review sub-agent · B: detector sub-agent). Browser overlays skipped by user request; A used headless-Chromium screenshots at 1440 / 484 plus an 8 s live websocket sample (gen 9→10, turbo on), B ran the CLI detector (regex fallback).

## Design Health Score

| # | Heuristic | Score | Key Issue |
|---|-----------|-------|-----------|
| 1 | Visibility of System Status | 3 | A confirmed level switch has no pending state — `#k_levelsel` snaps back until the gen boundary |
| 2 | Match System / Real World | 3 | Ledger gen 9: "Reach 79 %" next to "Wins 1" — a won level should read 100 % |
| 3 | User Control and Freedom | 2 | No way to cancel a sent level switch; no "follow best" |
| 4 | Consistency and Standards | 2 | Best shown three times (`#s_lvb`, `#s_lgb`, `#k_atb`); turbo stated three times; three id formats |
| 5 | Error Prevention | 2 | `#b_manual` enabled under turbo; S/T/G/H hotkeys still fire while driving |
| 6 | Recognition Rather Than Recall | 2 | `[` `]` exist only in sr-only text; fitness / win-rate definitions are `title`-only |
| 7 | Flexibility and Efficiency | 2 | No follow-best, no number-key jump, no chart keyboard scrub; `median`/`duration`/`kills` pushed but dropped |
| 8 | Aesthetic and Minimalist Design | 2 | Nine black "FROZEN · TURBO" tiles; `#f_meta` duplicates the in-frame HUD |
| 9 | Error Recovery | 3 | Offline copy is exemplary; hardcoded `ws://127.0.0.1:8765` fails silently from another host |
| 10 | Help and Documentation | 2 | No visible key legend; sensor abbreviations explained only via `title` |
| **Total** | | **23/40** | **Acceptable (58 %)** — up from 20 |

## Design Specificity Verdict

**LLM assessment:** Domain-specific IA, template skin. The information architecture is authored — grouped raycast telemetry, level-progress bar with best-reach tick, first-win marker and level boundaries on the chart, persona, "ended by", "frozen · turbo", WASD takeover. The visual language is still stock dark-ops Tailwind (zinc-900, Inter + JetBrains Mono, tracked-caps eyebrows); the only colour on the page is Mario's sky, and the most game-native element — the level bar — is 8 px tall, smaller than the game's own HUD.

**Deterministic scan:** 2 findings, unchanged — `overused-font` (Inter; a taste call) and `gray-on-color` L15 (false positive: `selection:bg-red-500/30`). Manual counts: 0 `focus:outline-none`, 1 `:focus-visible` rule, every toggle has `aria-pressed`, the select has a `<label for>`, 3 `aria-live` regions, 8 `<kbd>` hints, JS parses. Residual: two `text-[9px]` strings (`.table-header`, the frozen overlay) and `.stale .sbar-out` at zinc-600 (≈ 2.6:1). Three external hosts (Tailwind CDN, Google Fonts ×2).

**Visual overlays:** skipped by user request.

## Overall Impression

The failure states are now designed — offline, turbo and manual all say what they are. What's left is rationing: the same number three times, the same state three times, nine identical black tiles, and a headline row that leads with fitness (a proxy) while reach and wins-this-gen sit under "Run configuration". One real bug slipped in with the new confirm strip: the level select reverts visually until the generation boundary, so people will re-issue the switch.

## What's Working

1. Offline and boot are designed states with honest copy and a live "seconds ago" counter.
2. Telemetry panel is real instrument design — inverted rays, centred VX/VY, grouped — and matches the ray overlay.
3. Chart carries level boundaries, the first-win dot and a good empty-state line.

## Priority Issues

1. **[P0] Level switch snaps back.** *Why:* `onUpdate` re-syncs `#k_levelsel` to `s.level` the moment `#levelconfirm` hides, so the control reverts until the gen boundary and users re-issue it. *Fix:* `pendingLevel`; show "→ Mario1-2 at gen N+1" in the strip until `s.level === pendingLevel`; skip the sync while pending. *Command:* /impeccable harden
2. **[P1] Manual mode leaks hotkeys.** *Why:* after the `manual && k` branch, other keys fall through to `HOTKEYS` — S flips sensors, T flips turbo while driving; Take control is enabled under turbo. *Fix:* in manual, return for everything but Escape; `setManual(true)` under turbo auto-disables turbo and says so in `#ribbon`. *Command:* /impeccable harden
3. **[P1] Turbo wall is a void.** *Why:* 9/10 tiles are black with a label; the ledger drops below the fold at 1440×1500 and adds ~1,100 px of black at 484. *Fix:* `body.turbo .spec` → single-line rows (dot · id · status · rail · fit), or one level-length track with 10 markers. *Command:* /impeccable distill
4. **[P2] Headline metrics are the proxy, not the goal.** *Why:* best ×3, fitness first; reach % and wins-this-gen buried. *Fix:* header = Gen · Level best · Δ vs last gen · Reach · Wins this gen · Win rate; move `#k_atb/#k_frames/#k_traintime` to a "Run so far" group; clamp Reach to 100 % on a win. *Command:* /impeccable clarify
5. **[P2] Chart is unreadable at the side-column scale.** *Why:* 13 px on a 680 px canvas rendered at ~400 px ≈ 7.6 px; non-round ticks. *Fix:* size the canvas by `devicePixelRatio` with a ResizeObserver, round ticks, ≥ 11 px labels, ←/→ scrub when focused, `#chartread` aria-live. *Command:* /impeccable typeset

## Persona Red Flags

**Alex (power user):** `#k_levelsel` snap-back; hotkeys fire during manual; no follow-best; `[` `]` undiscoverable; `ws://127.0.0.1:8765` hardcoded (no LAN viewing); `results` capped at 60 / `history` at 400 silently; `median`/`duration`/`kills` dropped.

**Sam (screen reader / keyboard):** `#census` is `aria-live` and rebuilt every stats frame (~8 Hz) — constant announcements; 14 `abbr[tabindex=0]` tab stops with `title`-only text; `.spec` `aria-label="Watch specimen 01"` hides status/fit/progress; `#chart` is `role=img` with a hover-only readout; `#levelconfirm` `role=alertdialog` without Escape, focus trap or `aria-describedby`; `#ribbon` instructions not live; residual 9 px type; `#p_best` 1 px tick imperceptible.

## Minor Observations

- Tailwind CDN + Google Fonts: a local-first tool whose skin dies with the internet.
- `title` tooltips on `#s_wr` / `#s_sps` / `#m_fit` invisible on touch and keyboard.
- "of level" hidden below `sm`.
- One 10 px eyebrow style serves section h2, metric labels, table headers and sensor groups — flat sidebar hierarchy.
- `.spec` statuses stay bright green while offline.
- `#f_firstwin` sits next to `#f_env`, attributing a level-wide event to the watched specimen.

## Questions to Consider

1. Why is "watched" a manual choice? Auto-follow-best with a pin would make the wall optional under turbo.
2. Is fitness the right headline at all? Balance researchers think in reach, wins and gens-since-improvement.
3. Why show the game twice (HUD + `#f_meta`) and the population zero times in turbo? One level-length track with 10 markers is denser, comparative and honest.
