# Meat Boy in PEAK against the real Super Meat Boy

What the original Super Meat Boy (Team Meat, 2010, PC/Mac) does, measured or stated by a source, and the
value PEAK uses for it. Collected 2026-10-27 for the Meat Boy overhaul on `USER-STUDY`.

Tags: **measured** (someone measured it, or it is read from the game), **stated** (a wiki, developer or
speedrun-community statement), **estimate** (our own estimate; replace it when a better number turns up).

## Scale

PEAK keeps 32 px tiles. In the real game a metal block (one tile) is 48 px wide in a 1920 x 1080 capture of
1-3, and the visible area is about 40 x 22.5 tiles there (measured, AL's installed copy). Meat Boy is about
0.7 tile tall (estimate from wiki screenshots of 1-3 and 1-4, [SHOT]). So one Meat Boy height, **u**, is
0.7 tile = **22.4 px** in PEAK, and every value in u below is multiplied by 22.4 to get px.

The PEAK view is **1280 x 720** (40 x 22.5 tiles), the size of the game's own window, so 1-3 fits on one
screen as it does in the game. Larger levels scroll.

## Physics

| What | Super Meat Boy | Source | PEAK before | PEAK now |
|---|---|---|---|---|
| Body | about 0.7 tile tall, a cube | estimate [SHOT] | 18 x 26 px | 22 x 22 px |
| Walk top speed | 18 u/s | measured [DIGRA] | 300 px/s | 403 px/s |
| Walk acceleration | 35 u/s^2 (0.52 s to top speed) | measured [DIGRA] | 810 px/s^2 | 784 px/s^2 |
| Run top speed (Shift held) | about 27 u/s: the 1-3 record (1.37 s for 32 u) needs it | estimate [SRC] | 600 px/s | 600 px/s (kept) |
| Run acceleration | larger than walk; 0 to top speed in 0.4 s | stated [OLNEY], measured (Kevin, 1-13 walkthrough) | 1500 px/s^2 | 1500 px/s^2 (kept) |
| Turning (input against motion) | acceleration x 2 (70 u/s^2), skid 2.16 u | measured [DIGRA] | skid 7000 px/s^2 | x 2 |
| No direction on the ground | stops dead in one frame | measured [DIGRA], stated [OLNEY] | friction 3600 px/s^2 | instant stop |
| Air control | same acceleration and top speed as the ground, no air friction | measured [DIGRA] | air friction 300 px/s^2 | none |
| Run released in the air | horizontal speed drops to the walk limit at once | stated [CG1], [SHEET] | bled off by friction | clamped at once |
| Jump take-off | 21 u/s, the same at any speed | measured [DIGRA] | 480 px/s | 470 px/s |
| Gravity | 41 u/s^2 | measured [DIGRA] | 1060 px/s^2 | 918 px/s^2 |
| Full jump | 5.55 u high, 1.05 s, 18.6 u far at walk speed | measured [DIGRA] | about 3.4 tiles | 3.8 tiles, 1.02 s |
| Jump released early | upward speed set to 0; a tap counts as a 0.1 s hold (2.21 u) | measured [DIGRA] | cut to 150 px/s | 0 after 6 frames |
| Jump buffer, coyote time | none: a jump pressed early is ignored | measured [DIGRA] | 6 and 6 frames | none |
| Fall speed cap | none seen within a full jump; about 20 u/s in 1-13 | measured [DIGRA], measured (Kevin) | 450 px/s | 500 px/s (22 u/s) |
| Wall slide | slides while held into the wall; released, he falls | stated [WIKI 1-16] | slide on contact + push | slide while held |
| Wall slide speed | not published | estimate | 280 px/s | 180 px/s (8 u/s) |
| Wall stick when pulling away | 200 ms | stated [REFENES] | none | 12 frames |
| Wall jump | up and away; you can climb one wall by steering back | stated [WIKI 1-5], [SHEET] | 500 / 500 / 645 px/s, 6-frame lockout | 500 up, 500 away, no lockout |
| Physics step | fixed 60 Hz (the game breaks at other rates) | stated [OLNEY], [PCGW] | 60 Hz | 60 Hz |

The walk and the run are separate in PEAK as in the game: the run button (Shift, or X / a trigger on a pad)
raises the acceleration and the top speed. With the values above, 1-3 takes about 1.4 s at full run (the
record is 1.37 s), a running jump crosses 19 tiles, and a walking jump only 13, so the 15-tile gap needs the
run or the wall route, as the wiki says.

Measured in PEAK (`code/tests/test_meatboy.py`) against the game:

| What | Super Meat Boy | PEAK |
|---|---|---|
| Walk top speed after 0.52 s | 18 u/s | 18.0 u/s |
| Full jump | 5.55 u high, 1.05 s | 5.2 u high, 1.02 s |
| Tap jump | 2.21 u high, 0.47 s | 2.13 u high, 0.43 s |
| Jump distance at walk speed | 18.6 u | 18.3 u |
| 1-3 at full run, start to Bandage Girl | 1.37 s (record) | under 1.6 s (scripted run, not optimised) |
| One-wall climb by wall jumps | possible (1-5) | about 2 tiles per wall jump walking, 3 running |

## Death, respawn, lives

- One hit kills: saws, fire, a pit, leaving the level (stated [CG3], [WIKI]).
- Respawn is "almost instantaneous" (stated [HARD]); lives are unlimited (stated [WP]). PEAK: a short splat
  (about 0.3 s, estimate), then Meat Boy is back at the start. Blood stays on the walls across attempts, as
  in the game. Manual play and the User Study keep their `--lives` count (one row per life, like the other
  games; AL's choice, 2026-10-27).
- Timer: a hanging sign at the top left counts up from the start of the attempt (measured, AL's copy).

## World 1 elements

| Element | First level | PEAK glyph | PEAK behaviour |
|---|---|---|---|
| Saw blade (static) | 1-2 as decoration, kills from 1-3, first real hazard 1-6 | `*` or sidecar `saws:` | circle kill zone, any size, off the tile grid |
| Moving saw | 1-13 | sidecar `saws:` with `end`, `period` | ping-pongs between two points |
| Crumbling block | 1-9 | `%` | solid; breaks 0.75 s after first touch (estimate: "slightly longer than half a second", [WIKI]); back on respawn |
| Fire | 1-11 | `^` | kills on touch (drawn as flames) |
| Bandage Girl (goal) | every level | `G` | touching her ends the level |

No spikes, salt or conveyors in World 1 (stated [WIKI]). Spinny saws (1-17) and saw shooters (1-15) are
beyond the 11 levels PEAK plays and are not built.

## The levels

PEAK plays World 1 levels 1-1 to 1-11, in the game's order (level ids 0 to 10). The study's evaluation level
is id 2, **1-3 The Gap**. Every layout is our own ASCII drawing; no level data from the game is copied.
Redrawn on 27 Oct at the game's scale: 1-1, 1-2 and 1-3 from screenshots of AL's installed copy; 1-8,
1-10 and 1-11 from the playthrough video [VID] and the wiki. 1-4 to 1-7 and 1-9 are still Kevin's 13 Oct
drawings (the right names, not traced). The old 1-11 "Salt Factory" is not a World 1 level; it is now
1-11 Fired.

| Level | A+ par (s) | Record (s) | What it teaches (stated [WIKI]) |
|---|---|---|---|
| 1-1 Hello World | 3.00 | 0.85 | move and jump; a platform in an enclosed space |
| 1-2 Upward | 5.00 | 1.63 | wall jumps up two walls; a saw out of reach |
| 1-3 The Gap | 9.00 | 1.37 | run and jump a 15-tile gap over a pit; saws on the two pillars |
| 1-4 Nutshell | 9.00 | 3.83 | a spiral climb, no hazards |
| 1-5 Holy Mountain | 11.00 | 4.35 | climbing one wall by wall jumps |
| 1-6 Bladecatcher | 7.00 | 2.92 | saws that kill, jumped while climbing |
| 1-7 Diverge | 5.00 | 2.95 | a split path, saws on both |
| 1-8 The Bit | 4.50 | 1.60 | 1-3 with saws in the pit and a pillar in the middle |
| 1-9 Safety Third | 8.00 | 4.07 | crumbling walls |
| 1-10 The Levee | 7.50 | 4.30 | crumbling walls across the path, saws |
| 1-11 Fired | 8.00 | 2.80 | fire under crumbling blocks |

1-3 as measured in AL's copy (1920 x 1080, 48 px tiles): Meat Boy starts on the left ledge inside an alcove
6 tiles deep and 6 tiles high; Bandage Girl stands on the mirrored right ledge at the same height; the gap
between the ledge edges is 15 tiles over an open pit; a tall pillar rises over each alcove with a large saw
in a metal housing at its top, facing the gap.

## Local reference data (not in git)

`meatboy-data/` (git-ignored, like `megaman-data/`) holds the screenshots taken from AL's installed game
(`screens/`) and the playthrough video [VID] (`video/`). Nothing in PEAK needs the folder at run time.

## Sources

- [DIGRA] Fasterholdt, Pichlmair, Holmgard, "You Say Jump, I Say How High? Operationalising the Game Feel of
  Jumping", DiGRA/FDG 2016. https://dl.digra.org/index.php/dl/article/view/771 (Table 4; u = Meat Boy heights)
- [OLNEY] M. Olney, "Super Meat Boy Physics" (2012), reverse engineered update rules.
  http://web.archive.org/web/20121022180329/http://mpolney.galineer.com/index.php/Super_Meat_Boy_Physics
- [REFENES] T. Refenes, "My time with the Steam Controller", Gamasutra, 2013 (the 200 ms wall delay).
  http://web.archive.org/web/2014/http://www.gamasutra.com/blogs/TommyRefenes/20130928/201219/My_time_with_the_Steam_Controller.php
- [HARD] E. McMillen, Game Developer (respawn time).
  https://www.gamedeveloper.com/pc/-i-super-meat-boy-i-s-mcmillen-explains-why-so-hard-
- [CG1], [CG3] Critical-Gaming, Super Meat Boy parts 1 and 3.
  https://critical-gaming.squarespace.com/blog/2011/1/5/super-meat-boy-pt1.html
- [SHEET] Speedrun community "Any% per level strats" sheet, linked from https://www.speedrun.com/smb/guide/778tx
- [SRC] speedrun.com individual-level records (Light Forest, Meat Boy, no clips). https://www.speedrun.com/smb
- [WIKI] Super Meat Boy wiki, https://supermeatboy.fandom.com/wiki/The_Forest_(Super_Meat_Boy) and the level pages
- [WP] https://en.wikipedia.org/wiki/Super_Meat_Boy
- [PCGW] https://www.pcgamingwiki.com/wiki/Super_Meat_Boy (60 fps cap)
- [SHOT] Wiki screenshots, e.g. https://static.wikia.nocookie.net/supermeatboy/images/f/f9/SMB_1-3_LW.png
- [VID] Mastermaxify, "Super Meat Boy World 1 100% (Forest)", 2015. https://youtu.be/J4_yb6JX3rQ
- Kevin's measurements: commit `f177d6e39` (run acceleration, fall cap, wall jump, from a 1-13 walkthrough).

Full research notes with every quote: the session notes file `smb_research.md` (outside the repo).
