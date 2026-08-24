---
target: Balance Command center (code/neuro/report.py)
total_score: 29
max_score: 40
na_heuristics: 
p0_count: 1
p1_count: 2
timestamp: 2026-08-22T06-38-54Z
slug: code-neuro-report-py
---
Method: dual-agent (A: design review sub-agent · B: detector sub-agent). Browser overlays skipped by user request; A used headless-Chromium screenshots at 1440 / 484, B ran the CLI detector (regex fallback, `--no-config`).

## Design Health Score

| # | Heuristic | Score | Key Issue |
|---|-----------|-------|-----------|
| 1 | Visibility of System Status | 3 | `section.game{opacity:0}` entrance relies on IntersectionObserver — print, hash-open and headless leave sections dimmed |
| 2 | Match System / Real World | 3 | "Stall is still the main killer (100 %)", "OOB", "B1 challenge: off band" — research jargon in designer copy |
| 3 | User Control and Freedom | 3 | ▶ Watch → new tab → `location.replace` to :8000 with no route back |
| 4 | Consistency and Standards | 2 | Three verdict vocabularies stacked in `.dverdict`: solved/partial/unsolved, learnable/…/too hard, in band/off band — Meat Boy 3 is green "learnable" under three red/yellow B-pills |
| 5 | Error Prevention | 2 | `/watch` reflects `npz`/`label` unescaped (localhost XSS); "43 % ± 78 %" shown as if meaningful |
| 6 | Recognition Rather Than Recall | 3 | "menu 12/13/14" references assume the CLI menu is memorised |
| 7 | Flexibility and Efficiency | 3 | Deep links, sort, copy, collapse-all; no prev/next level in the dialog, persona/config not in the URL |
| 8 | Aesthetic and Minimalist Design | 3 | Dialog is 7 stacked panels; 16-column table always rendered; "Pit 0 %" in the cause key |
| 9 | Error Recovery | 3 | Error page is plain English; "re-probe it (menu 13)" has no link |
| 10 | Help and Documentation | 3 | Glossary collapsed by default; Instructions is a TIPS dump |
| **Total** | | **29/40** | **Good (73 %)** — up from 26 |

## Design Specificity Verdict

**LLM assessment:** Authored, but the signature is buried. The routes-over-geometry canvas, the start→goal death strip, persona chips, pixel game icons and the verdict prose ("a precision demand, not a discovery problem") belong only to this product — yet report.html above the fold is still a generic dark dashboard (chart + table); everything distinctive lives one click deep in `dialog.detail`.

**Deterministic scan:** 3 findings per page (was 23): `layout-transition` ×2 (`.mk i` marker-dot resize — minor, `transform:scale()` would be cleaner; `.mini i` fill bar — intended) and `em-dash-overuse` (advisory; dominated by `<td>—</td>` placeholders and repeated templated tooltips). gasweep.html scanned this time. Manual counts found the real gap: **the a11y pass only landed on report.html** — ablation.html has 39/39 canvases and 97/97 `<th>` unlabelled, gasweep.html 435/435 canvases, 733/733 `<th>`, 7/7 dialogs; report.html has 3 brain dialogs without `aria-labelledby`. Zero infinite animations, zero left-accent borders, skip link + h1/h2 on every page.

**Visual overlays:** skipped by user request.

## Overall Impression

The level dialog now opens on the product's own voice and the page reads as built, not assembled. Two things hold it at "Good": the verdict is undercut by a wide-CI number right beneath it and by three competing verdict vocabularies, and the ablation/gasweep pages didn't get the accessibility pass report.html got. The biggest opportunity is the home page — it still leads with a chart when a ranked "fix these first" list of verdicts is one line of code away.

## What's Working

1. `_verdict_sentence` — plain-English, cause + location + a recommendation; genuinely decision-ready.
2. `canvas.routes` level map promoted into the first screen of the dialog.
3. Ablation overview with paired Δ ± CI and ★ — honest statistics, scannable.

## Priority Issues

1. **[P0] `/watch` reflects query params unescaped.** *Why:* `npz` and `label` are interpolated into HTML in `serve()._page` callers — reflected XSS, localhost-bound but a one-liner to close. *Fix:* `html.escape()` both in all three bodies. *Command:* /impeccable harden
2. **[P1] Three verdict vocabularies in one block.** *Why:* green "learnable" over three red/yellow B-pills reads as a contradiction. *Fix:* keep word + sentence; relabel the B-pills "Threshold checks" and move them under `details.more`; when CI ≥ mean render "43 % (3 seeds, wide CI)" in yellow without ±. *Command:* /impeccable clarify
3. **[P1] Ablation / gasweep pages missed the a11y pass.** *Why:* 474 unlabelled canvases, 830 `<th>` without scope, 8 dialogs without a label. *Fix:* route those pages' canvases/tables/dialogs through the same helpers report.html uses. *Command:* /impeccable harden
4. **[P2] Watch handoff has no way home.** *Why:* `location.replace` lands the user on :8000 with no link back. *Fix:* caption says "opens the live trainer in ~10 s, new tab"; dashboard gets a persistent "← Balance Command" link. *Command:* /impeccable onboard
5. **[P2] Entrance fade breaks print and hash-open.** *Why:* `section.game{opacity:0}` is decoration that fails outside scroll. *Fix:* delete the fade; keep the bar fill. *Command:* /impeccable layout

## Persona Red Flags

**Alex (power user):** no prev/next level inside `dialog.detail`; comparing personas means close → tab → reopen; config/persona absent from the URL; `code.cmd` clips the path at 1440.

**Sam (screen reader / keyboard):** `.tip::after` bubbles are CSS content, never announced; `_stat .lbl` is a focusable `div` with `aria-label` but no role; `<div>`s inside `button.cardhead` (invalid phrasing content); `table.metrics tr[tabindex]` has no role or instructions, Enter sorts but Space toggles the tip on `th.tip`; `.heat i` / `.causebar i` / `.mini i` are colour-only with `title`; dialog focus lands on the container, not the heading; ablation/gasweep canvases and tables unlabelled (see P1).

## Minor Observations

- Red focus ring on the red `.watchbtn`.
- Radar axis labels "0…10" aren't labelled as levels.
- 0 %-win levels still offer "▶ watch … best seed replay".
- "Pit 0 %" entries in `_causebar`; "Stall" described as a killer.
- ~326 em-dashes in generated prose (advisory).

## Questions to Consider

1. Why does the home page open on a win-rate chart instead of a ranked "fix these first" list of verdicts?
2. Should ▶ Watch leave the page at all — could the dashboard be iframed into the dialog?
3. With 3 seeds and ± 78 %, is colouring 43 % yellow more honest than saying "not enough data"?
