---
target: Balance Command center (code/neuro/report.py)
total_score: 26
max_score: 40
na_heuristics: 
p0_count: 0
p1_count: 3
timestamp: 2026-08-22T05-44-55Z
slug: code-neuro-report-py
---
Method: dual-agent (A: design review sub-agent · B: detector sub-agent). Browser overlays unavailable (Chrome extension not connected); A used headless Chromium screenshots, B ran the CLI detector in regex-fallback mode.

## Design Health Score

| # | Heuristic | Score | Key Issue |
|---|-----------|-------|-----------|
| 1 | Visibility of System Status | 2 | ▶ Watch replay hands off to a bare page + 3 s meta-refresh; silently kills a running replay |
| 2 | Match System / Real World | 3 | "OOB", "σ", "xo 0.7", "h16", "k5" in .gamemeta/.recchip assume GA literacy |
| 3 | User Control and Freedom | 3 | No deep link to a level; "Collapse all" has no undo |
| 4 | Consistency and Standards | 3 | Three "select one" vocabularies: .cfgbtn, .tab, .lchip |
| 5 | Error Prevention | 2 | /watch terminates the live replay with no confirm |
| 6 | Recognition Rather Than Recall | 2 | 825 hover-only .tip tooltips carry the metric definitions |
| 7 | Flexibility and Efficiency | 3 | No keyboard shortcuts, no per-level URL, overview table not sortable/clickable |
| 8 | Aesthetic and Minimalist Design | 2 | Level dialog: 16 equal tiles + 3 verdicts + config + big red button, no priority |
| 9 | Error Recovery | 3 | /watch failure returns an HTTP 400 "bad replay request" page with no way back |
| 10 | Help and Documentation | 3 | Glossary + Instructions page are good; hover-only ⓘ is the weak link |
| **Total** | | **26/40** | **Acceptable (65 %)** |

## Design Specificity Verdict

**LLM assessment:** Authored for this product. The route canvas (level geometry under green win / red death traces), the "Where agents die (start → goal)" heat strip, B1/B2/B3 verdict pills tied to MarioThresholds.yaml, the rays-vs-grid sensor SVGs and the ▶ Watch → `trainer --replay` bridge could not exist on a generic dashboard. It slips back into generic BI in two places: the 16-tile `.stats` grid in the level dialog and the 16-column `_metric_table` — "dump every column" patterns. The voice on the GA-sweep page ("Bigger doesn't help. The bottleneck is the level, not the brain.") is the most product-specific element and is absent from the level dialog, where it matters most.

**Deterministic scan:** 23 findings across report/ablation/instructions (gasweep.html, 2.4 MB, never finished — detector hung in regex fallback). side-tab ×9, layout-transition ×6, em-dash-overuse ×3, marquee ×3, dark-glow ×2. Most are false positives: side-tab fires on left accent borders of `.tip` / `.cfgbtn` / `.gloss` (not nav); layout-transition on a chart marker dot and a fill bar; dark-glow on an 8 px verdict dot whose `var(--c)` the fallback parser mis-resolved. Real: the `.gametag::after` sheen (4.5 s infinite animation — the only non-data motion on the page) and ~300 em-dashes in generated prose. Note: `runs/balance/*.html` is gitignored so the detector's default config skips them — scan with `--no-config`.

**Visual overlays:** none — Chrome extension not connected. Fallback signal: headless-Chromium screenshots at 1440 and 484-cropped-to-390 (no horizontal overflow on any page).

## Overall Impression

A serious research instrument with real product identity — the route canvas and the honest statistics copy ("★ = clears its 95 % CI", "Bands are designer-set, not human-calibrated yet") earn trust. The single biggest opportunity: the level dialog, which is where a designer decides what to fix, opens on 16 equal tiles instead of a one-sentence verdict. The GA-sweep page already has the pattern (`.vsent`); apply it to levels.

## What's Working

1. **Route canvas (`canvas.routes`)** — geometry + entities + sampled traces in one frame, legend built only from glyphs present in that level. The product thesis made visible.
2. **GA-sweep verdict block (`.verdict` + `.vsent`)** — pill, one plain-English sentence, caveat in a caption. Decision-ready without reading the scatter.
3. **Honest statistics copy** — paired-cell language, CI stars, designer-set-band disclaimer. Researchers will trust this.

## Priority Issues

1. **[P1] Level dialog has no verdict sentence or priority order.** *Why:* designers open a card to decide what to fix; they get 16 equal tiles before the thing they came for (the routes). *Fix:* generate a `.vsent`-style line above the stats from `_verdicts` + dominant cause + death-hist peak ("Enemy kills 53 % in the 0–10 % bin: one learnable chokepoint, win time in band"); show 4 headline tiles (Win rate, First win, Dominant cause, Death spread), the other 12 behind a `<details>` "all 16 metrics". *Command:* /impeccable distill
2. **[P1] ▶ Watch replay lacks confirmation and status.** *Why:* `/watch` `terminate()`s a running replay silently, then hands off to an unstyled page with a 3 s meta-refresh that may 404 if the trainer isn't up. *Fix:* style the interstitial with the report CSS, poll :8000 and redirect only when it answers, and say "replacing the running replay of X" when `state["proc"]` is alive. *Command:* /impeccable harden
3. **[P1] Hover-only tooltips on 825 elements.** *Why:* `.tip:hover::after` has no focus or touch path; keyboard and screen-reader users never see a metric definition. *Fix:* make `.tip` a `<button type=button>` (or `tabindex=0`) with `:focus-visible::after`, and `aria-describedby` to a visually-hidden definition. *Command:* /impeccable adapt
4. **[P2] Persona is invisible at section level on ablation/gasweep.** *Why:* three consecutive `MARIO` gametags differ only by 0.78 rem faint `.gamemeta` text. *Fix:* render persona in the gametag (`MARIO · novice`) or as a coloured chip from the existing persona palette. *Command:* /impeccable clarify
5. **[P2] Meat Boy radar with dash-encoded personas.** *Why:* colour = config, dash = persona means one config yields three same-red polygons on 11 axes — reads as noise. *Fix:* when only one config exists, colour = persona; keep dash for config. *Command:* /impeccable colorize

## Persona Red Flags

**Alex (power user):** no URL for a level (dialog not reflected in the hash); "Collapse all" has no shortcut; overview table isn't sortable and rows don't open the card — must switch persona tab then hunt for the card; `.cfgbtn` rebuilds the chart with a 700 ms ease on every click; 16-column `_metric_table` scrolls inside `.ovwrap` with no sticky first column.

**Sam (screen reader / keyboard):** `.cardhead` buttons lack `aria-haspopup="dialog"`; all 41 `<dialog class="detail">` lack `aria-labelledby`; `.tab` buttons have no `role=tab`/`aria-selected` on report.html; 80 `<canvas>` on report.html and 435 on gasweep have no `aria-label`/fallback; zero h1–h3 on report.html and gasweep.html; no skip link; `<th>` without `scope`; `.chev` says "collapse section" and never flips to "expand"; `.tip:hover::after` content unreachable; `.lvlcard:has(dialog[open])` ring is border colour only.

## Minor Observations

- `.gametag::after` sheen loops forever on every game header — decoration on a data page (detector agrees: marquee ×3).
- `.stat .val` "gen 8.3 ± 11.7" wraps to two lines in a 160 px tile while neighbours stay single-line.
- Mario bar chart uses ~55 % of canvas width for 2 levels; `groupW` is uniform regardless of level count.
- `nav a.doc` colours (blue/yellow/green) are unexplained.
- Footer strings are file paths — developer notes shipped to designers.
- `.gloss` "Reading this report" sits at the bottom after 200 rows of data.
- Brain dialog persona chip: "senses every 1 frame".
- ~300 em-dashes in generated prose (detector em-dash-overuse ×3).

## Questions to Consider

1. If a designer can only read one sentence per level, what is it — and why isn't it the first thing in the dialog?
2. Should ▶ Watch be a first-class action on the card rather than line 3 of a modal?
3. Persona is encoded as dash pattern, config as colour. For a game designer, is persona really the secondary variable?
