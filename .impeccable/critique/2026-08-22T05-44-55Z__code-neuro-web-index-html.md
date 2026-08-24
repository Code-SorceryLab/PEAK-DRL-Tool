---
target: Live training dashboard (code/neuro/web/index.html)
total_score: 20
max_score: 40
na_heuristics: 
p0_count: 2
p1_count: 2
timestamp: 2026-08-22T05-44-55Z
slug: code-neuro-web-index-html
---
Method: dual-agent (A: design review sub-agent · B: detector sub-agent). No screenshots — Chrome extension not connected; A reviewed source + an 8 s live websocket sample (gen 19, Mario1-2, turbo on), B ran the CLI detector in regex-fallback mode.

## Design Health Score

| # | Heuristic | Score | Key Issue |
|---|-----------|-------|-----------|
| 1 | Visibility of System Status | 2 | In turbo, 9/10 wall tiles read "WAITING"/stale; only 10 px `#turbonote` below the wall explains |
| 2 | Match System / Real World | 3 | "ENV 01" / "Watched Specimen" / "env 1" — three names for one thing |
| 3 | User Control and Freedom | 2 | `#k_levelsel` switches level mid-run with no confirm; Turbo + Take control can both be on |
| 4 | Consistency and Standards | 3 | Header "Elapsed" vs config "Train Time" — same clock, two widgets |
| 5 | Error Prevention | 1 | Manual mode captures Space/arrows at document level, even on a focused `<select>` |
| 6 | Recognition Rather Than Recall | 2 | Toggle state is a faint `.on` background only; no `aria-pressed`, no label change except `#b_manual` |
| 7 | Flexibility and Efficiency | 2 | No hotkeys for Turbo / next specimen; no hover values on the chart |
| 8 | Aesthetic and Minimalist Design | 3 | Dense but disciplined; `#f_meta` is a 6-item run-on string |
| 9 | Error Recovery | 1 | On disconnect every number stays frozen and looks live; only a 9 px "Offline" label + boot overlay |
| 10 | Help and Documentation | 1 | No tooltip on FWD/U30/NMY/BLK sensor codes, no unit on fitness |
| **Total** | | **20/40** | **Acceptable (50 %)** |

## Design Specificity Verdict

**LLM assessment:** Authored, not generic — lab vocabulary ("Specimen", "Population", "Telemetry", "Results Ledger"), red-on-#09090b signal colour, pixelated canvases, the census line `4 RUNNING / 2 STUCK / 4 DEAD / 0 WON`. But the specificity is skin-deep: the trainer pushes `live.level_len` (5664) and `x` (5015), yet the wall rail is normalised to `genBest`, not the level — the single most Mario-specific fact (88 % of the way to the flag) is never drawn.

**Deterministic scan:** 2 findings in `code/neuro/web/index.html` — `overused-font` L11 (Google Fonts Inter — real, a taste call) and `gray-on-color` L15 (false positive: regex matched `selection:bg-red-500/30`, a selection-highlight variant, not the element background). Parser was degraded (htmlparser2 etc. missing) so contrast checks did not run; A's manual contrast estimates (zinc-700 on #09090b ≈ 1.9:1, zinc-600 ≈ 2.6:1) are the only contrast evidence.

**Visual overlays:** none — Chrome extension not connected. Fallback signal: source read + live websocket sample; narrow-width findings derive from Tailwind breakpoints in the markup, untested visually.

## Overall Impression

The page knows what it is — a lab bench, not a marketing dashboard — and the status-tinted wall (dead = greyscale, stuck = desaturated) is a genuinely good idea. What it lacks is the one number the whole product is about: how far along the level the best agent got. And its failure states (turbo freeze, disconnect, manual-mode key capture) all look like success states.

## What's Working

1. `.s-DEAD canvas { grayscale brightness(0.3) }` / `.s-STUCK` desaturation — specimen status readable at a glance across the wall.
2. Manual-control ribbon + red bezel ring — the mode change is unmistakable.
3. Server-authoritative toggle sync (`s.turbo !== turbo`) — two open tabs never disagree.

## Priority Issues

1. **[P0] Turbo wall state reads as broken.** *Why:* on a fresh load in turbo, 9 tiles say "WAITING" forever; the explanation is a 10 px zinc-600 note *below* the wall. *Fix:* overlay each non-watched tile with "frozen · turbo" and dim it; move `#turbonote` above the wall. *Command:* /impeccable clarify
2. **[P0] Keyboard focus is invisible.** *Why:* `.btn-sleek` and `.spec` set `focus:outline-none` with no `focus-visible` style — keyboard users cannot see where they are. *Fix:* `focus-visible:ring-2 ring-red-500` on both. *Command:* /impeccable harden
3. **[P1] Disconnect looks live.** *Why:* on `ws.onclose` every stat stays frozen at its last value; the user can't tell trainer-finished from crashed. *Fix:* add `body.offline` that dims header stats to 50 % and shows "last update 12 s ago". *Command:* /impeccable harden
4. **[P1] Level progress is missing.** *Why:* `level_len` and `x` are pushed but never drawn; the rail is normalised to the generation's best, so it never says "88 % of the level". *Fix:* rail = `x / live.level_len`; add "88 % of level" to `#f_meta`; mark first win on the chart. *Command:* /impeccable shape
5. **[P2] Manual keys hijack the document.** *Why:* `setKey` listens at document level so Space/arrows fire even on a focused `<select>`. *Fix:* ignore when `activeElement` is a form control; scope to `#bezel`. *Command:* /impeccable harden

## Persona Red Flags

**Alex (power user):** no hotkey for `#b_turbo`; clicking wall tiles is the only way to switch specimen (no `[`/`]`); chart has no hover readout; `#s_sps` hidden below the `sm` breakpoint.

**Sam (screen reader / keyboard):** 10 wall `<button>`s lack `aria-pressed` for "watched"; toggles lack `aria-pressed`; `<canvas id="watch">` has no `aria-label`/`role="img"`; `#f_status` changes aren't `aria-live`; sensor `<label>`/`<output>` not associated; `#k_levelsel` has no `<label for>`; zinc-700 headers on #09090b ≈ 1.9:1 and zinc-600 ≈ 2.6:1 both fail AA; 8–9 px eyebrow text.

## Minor Observations

- Tailwind CDN + Google Fonts: an offline lab machine gets an unstyled page (detector: overused-font).
- `#s_wr` shows "6 %" with no window context beyond the label.
- `results[].duration`, `best_x`, `avg_score` are pushed but unused.
- `#s_atb` red-400 vs `#s_gen` red-500 — two reds competing for "primary".
- `all_time_best` spans levels (11253 from Mario1-1) while the chart shows Mario1-2 — header and chart silently mix levels after a switch.
- Results table: 9 columns, `#resultsbox` is overflow-y only — at 390 px it squeezes or forces body scroll.
- First win: status turns blue and `#k_wins` ticks, nothing persists. Flat peak.

## Questions to Consider

1. Why show an all-time best that spans levels and cannot be beaten on the current one?
2. What if the watched feed were 30 % smaller and the level-progress bar were the hero?
3. Should "Take control" live on the bezel itself rather than in the toolbar, since it is a mode, not a view?
