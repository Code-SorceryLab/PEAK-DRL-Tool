# Mega Man 1 — queued work

Things agreed on but deferred, in the order they depend on each other. The team's paper task
list is `paper-ast/TODO.md`; this file is only the queue for the `megaman` game and its pipeline.

## In progress

- [ ] **Robust section training.** Eight sections, each genome scored over three varied attempts
      (`training:` in `code/games/megaman_config.yaml`). When it finishes: tested pick
      (`python -m code.neuro.relay --pick`), twenty full-stage relay attempts, dashboard.

## Next

- [ ] **Human baseline as a head start (imitation).** Only the baseline level is ever played by a
      person; mutants only ever see agents, judged from the true start.
  - [x] Manual play `--demo` saves, per frame, the 26 sensor values and the buttons held, played
        through the agent's input caps: one `.npz` per attempt under
        `runs/demos/megaman/<player>/<level>/`.
  - [x] Fit a network of the agent's shape to those recordings, one per section
        (`python -m code.neuro.imitate --demos runs/demos/megaman/human`).
  - [x] Trainer option to seed the first generation from that network and mutated copies
        (`--seed-from runs/megaman/<section>/imitation`).
  - [ ] Compare tested win rates and relay results with and without the head start; show the
        human's own attempts beside the agents' on the dashboard.
  - Needs from Cristiano: a few clean playthroughs of the baseline Cut Man stage, some beating
    the boss.

- [ ] **Regression check on mutants** (paper section 3.6).
  - Replay first: baseline agents play the mutant; identical telemetry means nothing to retrain.
  - Otherwise retrain only the section that changed, warm-started from its baseline agents.
  - Comparison command: two run directories in, flagged levels and the signal that fired out.
  - Measure what a warm-start retrain costs, and how often replay fails on neutral mutants.
  - Open: section 3.6 says retrain "from the same seeds"; warm start changes that wording.
  - The mutants exist (see "Done"). Seen while testing the generator, replaying baseline agents:
    neutral mutants give telemetry identical to the baseline, row for row; a mutant on the
    route shifts every episode's `progress` slightly even when the agent plays the same way,
    because progress is measured along the mutant's own route. Compare on status, where the
    attempt ended and the route taken, not on raw progress.
  - A breaking mutant has no route from the start, so its `progress` is empty, not a number.
  - Spot-check a sample of mutant classes by hand: the route check is cautious and approximate.

## Later

- [ ] **Energy budget: reward energy left** (found 2026-10-12, see "Done" below).
  - The stage cannot be won as trained: a full run costs about 70–90 energy against 28, with no
    refill between sections. The fitness pays for progress, winning, speed and kills, not for
    energy left (`_attempt_fitness` in `code/neuro/adapters.py`).
  - Add a term for energy left at the end of a section (or a penalty per point of damage), then
    retrain all eight sections (about an hour at 5 runs x 80 generations each on this machine).
  - Lowering `min_start_energy` alone does not help: most sections cannot be cleared from 6.
  - Then: twenty full-stage relay attempts and the energy table again.

- [ ] **Ledges still has a sticking point.** Alone it wins 10 of 20; 7 of the 10 failures are
      stalls around tile column 90–91, rows 67–69. Two rounds of warm-started retraining did not
      remove it. Watch it live (command below) before trying more training.

- [ ] **Pipeline pages:** the Train page cannot show a run live (runs it starts are headless);
      the Mutate page lists mutants but does not play them (that is the regression check).

- [ ] **Telemetry page: crop the map to the section** when a section is selected (it draws the
      whole stage, so the routes sit in a corner).

- [ ] **Run training and play on a remote server.**
  - Turn the wave script (now in a scratch folder, hard-coded to this machine) into a launcher
    in the repo: sections, seeds, worker count; or a scheduler job array.
  - The stage file `code/games/levels/megaman/01_cutman.txt` is untracked: copy it to the
    server, or decide whether it goes in the repo (it is derived from the ROM's layout).
  - Pin Python and numpy versions on both sides; run a baseline and its mutants on the same
    machine so "identical telemetry" holds.
  - Needs from Cristiano: core count, scheduler or plain SSH, how to log in.

- [ ] **Telemetry gaps.**
  - Only `megaman` writes through the new writer (`code/stats/`); the other games still use the
    old one on this branch. The rest of `origin/new-stats` is not merged, and two of its files
    (`episode_stats.py`, `registry.py`) were edited here to add `megaman`.
  - Not yet on the dashboard from section 3.5: human first attempts shown separately, a replay
    of the best agent.
  - Decide whether "Stall" (ended after no progress) is reported as a cause of death.

## Housekeeping

- [ ] **Write a summary of everything done and the next TODOs, with the commands** (asked for
      2026-10-11, to do later). The 2026-10-12 session is summarized under "Done" below; the
      earlier work is not yet.

- [ ] Nothing from this work is committed yet (branch `ast-paper`).
- [ ] Credit the sprite and sound rippers if screenshots are published (the sheets ask for it).


## Done

### 2026-10-12: mutant generator (paper section 3.7)

`python -m code.games.mutants` writes 63 mutants of the Cut Man stage to
`runs/megaman/mutants/<id>/` (`mutant.json`, plus `level.txt` for level mutants) and an index,
`mutants.json` (`code/games/mutants.py`, tests in `code/tests/test_mutants.py`).

- **One change each.** Each level operator is applied once in every section where it has
  something to act on; otherwise the section is skipped and the reason recorded (10 skips).
- **Level operators (53 mutants):** `seal_route` (a wall across the route), `remove_ladder`,
  `widen_gap` (the ledge cut back two tiles), `add_hazard` (spikes on the floor),
  `add_enemy` (a Blader over the route), `remove_enemy` (the one nearest the route),
  `cosmetic_tile` (an air cell spelled the other way: the same game), `off_route` (a tile
  hollowed out deep inside a wall).
- **Mechanics operators (10 mutants):** one value of `megaman_config.yaml` moved by a fixed
  step: jump velocity (-15%, -30%, +10%), gravity, run, air and ladder speed, energy, shots on
  screen, time limit. Enemy speeds are constants in the code, not config, so they are not mutated.
- **Class, assigned before any agent plays:** from the route field searched from the stage's
  true start. 14 breaking (exit unreachable), 34 degrading, 15 neutral. Neutral is checked, not
  assumed: the parsed level is identical, or every reachable cell keeps its route cost.
- **Playing a mutant:** set `PEAK_MM1_MUTANT` to its folder; the game then loads the mutant's
  stage file or config values, and every command (relay, trainer, manual play) runs unchanged.
- **Checked by replaying baseline agents, 6 full-stage attempts each:** the two neutral mutants
  tried gave telemetry identical to the baseline; `s1-seal_route` and `m-jump_velocity_low`
  (both breaking) stopped every attempt in the first section.
- Not done: "move the goal" from the paper's operator list, and a neutral mechanics mutant
  (no config value was found that provably changes nothing).

### 2026-10-12: pipeline pages

`python -m code.pipeline --open` serves one local site (http://127.0.0.1:8003/) that walks the
pipeline in order: a hub with the state of each stage, then one page per stage
(`code/pipeline.py`, tests in `code/tests/test_pipeline.py`).

- **Design:** read-only map of the stage with its sections outlined, and the sections table.
- **Mutate:** the generated mutants by section: operator, class, what changed, a before/after
  map of the change, and the command that plays the mutant. A button generates them.
- **Train:** per section, the runs on disk and the one the relay uses. *Train* starts new
  warm-started runs (headless, new `seedNNN` folders); *Re-pick* tests the section's best runs
  and updates only that section's entry in `relay.json`
  (`python -m code.pipeline --pick-section CutMan-3-Ledges` does the same from a terminal).
- **Play:** for the full stage and each section: run a batch of agent attempts, watch agents
  live on the trainer dashboard, or play it yourself. Human attempts open the game window and
  are recorded under `runs/manual/megaman/<player>/`.
- **Measure:** the telemetry page over the agents' and the people's logs.

Everything a button starts is listed under "Jobs" on every page, with its last line of output
and a Stop button. Closing the server stops every job it started, training runs included.

### 2026-10-12: the old `megaman` game removed, `megaman1` renamed to `megaman`

- Removed: the old core and its levels, the `megaman:` block in `game_config.yaml`, the old
  adapter and stats class, and its entries in manual play, menu, editor, report, figures,
  registry, tests and docs. The shared level loader and object modules were left as they are.
- Renamed: game key, files (`megaman_core.py`, `megaman_enemies.py`, `megaman_route.py`,
  `megaman_assets.py`, `megaman_config.yaml`, `megaman_levels.yaml`, `levels/megaman/`,
  `test_megaman.py`, this file), classes (`MegamanCore`, `MegamanAdapter`, `MegamanStats`), and
  the run folders (`runs/megaman*`, `runs/demos/megaman`) with the game name inside their logs.
- Kept as they were: the tool scripts `mm1_extract_level.py` / `mm1_make_slices.py`, the
  environment variables `PEAK_MM1_ATTEMPTS` / `PEAK_MM1_ASSETS`, and the `megaman-data/` folder.

### 2026-10-12: section playtests, live attempts, the handoff fix

**Tools added**

- Replay warns when `--level` disagrees with the level the checkpoint was trained on
  (`code/neuro/trainer.py`).
- `relay --section NAME` plays one section alone with its picked network, from the section's
  start at full energy, one attempt per episode. Results go to
  `runs/megaman/relay/sections/<section>/`, so the relay's telemetry page lists every section
  played this way beside the full stage (`code/neuro/relay.py`).
- Telemetry page: `--serve` serves it locally with **Watch live** buttons, one for the selected
  level and one per episode. A click starts `relay --watch` on that level and seed and opens the
  trainer dashboard; a new click replaces the running attempt. Opened as a file the buttons are
  disabled (`code/stats/telemetry.py`).

**The handoff problem and its fix**

- The relay combines one network per section; no network is trained across a boundary. In the
  full stage 18 of 19 attempts ended in Climb (16 of them stalls just after the handoff), yet
  Climb alone won 18 of 20.
- Cause, found by changing one thing at the handoff over the same 20 seeds: Mega Man arrives
  21.5 px (0.67 tile) lower on the ladder than the section's training start, and with less
  energy (mean 16.5, as low as 10). Fixing both restored 18 of 20; facing and the screen-flip
  pause had minor effects.
- Fix: sections entered up a ladder (Climb, Ledges, Shaft, Rooftop) now start where the screen
  flip leaves the player (`code/games/megaman_core.py`). In training, later attempts arrive
  facing either way and the second attempt always starts in the lower half of the energy range
  (`code/neuro/adapters.py`, comment in `megaman_config.yaml`).
- Retrained, warm-started from the old picks, and picked by 12 varied test attempts. The manifest
  `runs/megaman/relay.json` was edited by hand, one section at a time (not `--pick`):

  | Section | Run in the relay | Alone, 20 attempts |
  |---|---|---|
  | CutMan-2-Climb | `seed701` | 15 |
  | CutMan-3-Ledges | `seed803` | 10 |
  | CutMan-4-Shaft | `seed806` | 19 |
  | CutMan-5-Rooftop | `seed704` | 20 |

  The other four sections keep their earlier runs (Street `seed501`, Descent `seed502`,
  Corridor `seed101`, Boss `seed501`).

**Result**

- Full stage, 20 attempts: Climb is now cleared by 18 of the 19 attempts that reach it (was 1).
  Still 0 wins: Ledges is cleared by 4 of 18 and Shaft by 0 of 4, 15 of those 18 failures being
  deaths to enemies. Mean distance along the stage 37%.
- Energy by section, each alone with its picked network (wins of 20 by entry energy):

  | Section | Damage in a winning run | 28 | 20 | 14 | 10 | 6 |
  |---|---|---|---|---|---|---|
  | CutMan-1-Street | 6–13 | 19 | 20 | 20 | 8 | 0 |
  | CutMan-2-Climb | 10–14 | 15 | 20 | 19 | 8 | 0 |
  | CutMan-3-Ledges | 9 | 10 | 11 | 11 | 7 | 0 |
  | CutMan-4-Shaft | 12–18 | 19 | 6 | 1 | 0 | 0 |
  | CutMan-5-Rooftop | 4–7 | 20 | 20 | 20 | 20 | 7 |
  | CutMan-6-Descent | 11–13 | 18 | 18 | 16 | 0 | 0 |
  | CutMan-7-Corridor | 2 | 20 | 20 | 20 | 20 | 20 |
  | CutMan-8-Boss | 16 | 20 | 19 | 0 | 0 | 0 |

  This table came from a throwaway script, not a command in the repo.

**Commands**

```bash
# Replay a checkpoint on the level it was trained on (omit --level; add --loop to stay on it)
python -m code.neuro.trainer --game megaman --replay runs/megaman/CutMan-3-Ledges/seed603/best.npz

# Full-stage relay: 20 attempts, results in runs/megaman/relay/
python -m code.neuro.relay --episodes 20

# One section alone: results in runs/megaman/relay/sections/<section>/
python -m code.neuro.relay --section CutMan-3-Ledges --episodes 20

# All eight sections alone
for s in CutMan-1-Street CutMan-2-Climb CutMan-3-Ledges CutMan-4-Shaft \
         CutMan-5-Rooftop CutMan-6-Descent CutMan-7-Corridor CutMan-8-Boss; do
  python -m code.neuro.relay --section $s --episodes 20
done

# Watch live on the trainer dashboard (http://127.0.0.1:8000/megaman/index.html)
python -m code.neuro.relay --watch --seed 1234                              # full stage
python -m code.neuro.relay --section CutMan-3-Ledges --watch --seed 1237    # a Ledges stall

# Pipeline pages: design, mutate, train, play (agents or yourself), measure (http://127.0.0.1:8003/)
python -m code.pipeline --open

# Mutants: generate (or --list), then play one with the trained agents
python -m code.games.mutants
PEAK_MM1_MUTANT=runs/megaman/mutants/s3-seal_route python -m code.neuro.relay --episodes 20 \
  --out runs/megaman/mutants/s3-seal_route/relay

# Telemetry page: static file, or served with working Watch live buttons (http://127.0.0.1:8002/)
python -m code.stats.telemetry runs/megaman/relay
python -m code.stats.telemetry runs/megaman/relay --serve --open

# Retrain one section, warm-started from an existing run (one command per seed; run in parallel)
python -m code.neuro.trainer --game megaman --level CutMan-2-Climb --seed 701 \
  --seed-from runs/megaman/CutMan-2-Climb/seed601 --gens 80 --no-serve \
  --run-dir runs/megaman/CutMan-2-Climb/seed701

# Pick the best run for every section and rewrite the manifest (tests each over 12 attempts).
# This re-picks all eight; to change one section, edit its entry in runs/megaman/relay.json.
python -m code.neuro.relay --pick

# Tests
python -m pytest code/tests -q
```

Retraining rounds run on 2026-10-12, all with the trainer command above:

| Section | Seeds | Generations | Warm-started from |
|---|---|---|---|
| CutMan-2-Climb | 701–703 | 80 | `seed601` |
| CutMan-3-Ledges | 701–705 | 80 | `seed304` |
| CutMan-4-Shaft | 701–705 | 80 | `seed501` |
| CutMan-5-Rooftop | 701–705 | 80 | `seed101` |
| CutMan-3-Ledges | 801–806 | 150 | `seed705` |
| CutMan-4-Shaft | 801–806 | 150 | `seed704` |
