"""Mega Man 1: level-file integrity, the route field, and the adapter/sensor/net contract."""
import os

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["PEAK_MM1_ASSETS"] = "/nonexistent"     # placeholder art: the tests never need megaman-data
os.environ["PEAK_MM1_ATTEMPTS"] = "1"              # one attempt per episode, except where a test says otherwise

from code.games import megaman_enemies as en
from code.games.megaman_core import MegamanCore
from code.neuro.adapters import N_INPUTS_BY_GAME, N_OUTPUTS_BY_GAME, list_levels, make_adapter
from code.neuro.evolution import GAConfig
from code.neuro.net import make_net
from code.neuro.sensors import read_sensors, sensor_dim

LEVELS = list_levels("megaman")
LEGAL = set(" .#H^DGXP") | set(en.ENEMY_GLYPHS) | set(en.ITEM_GLYPHS)


@pytest.mark.parametrize("level", LEVELS)
def test_level_file_is_well_formed(level):
    """Every level: known glyphs, whole screens, one start, a way to end, and a route to it."""
    core = MegamanCore(level=level)
    with open(core.level_file(), encoding="utf-8") as f:
        rows = f.read().split("\n")
    assert set("".join(rows)) <= LEGAL, f"unknown glyph in {core.level_file()}"
    assert "".join(rows).count("P") == 1
    assert len(rows) % core.screen_rows == 0 and max(map(len, rows)) % core.screen_cols == 0, "not whole screens"
    section = core.level_entry().get("section")
    if section:                                         # a section: its start is a place a body can be
        c, r = section["start"]
        assert rows[r][c] in ".HP" and (not section["climbing"] or rows[r][c] == "H")
        assert tuple(section["screens"][0]) == core.screen
    has_boss = any(g == en.BOSS_GLYPH for g, _, _ in core.level_data.spawns)
    assert has_boss or any("G" in r for r in rows), "no goal and no boss"
    adapter = make_adapter("megaman", level, 600, 5000.0)
    assert np.isfinite(adapter._start_cost), "the exit is unreachable from the start"
    for _ in range(30):                                 # standing still at the start is safe
        core.step((0, 0, 0, 0))
    assert core.alive and core.player.hp == core.player.hp_max


def test_levels_are_named_and_addressable_by_position():
    assert LEVELS[-1] == "CutMan" and LEVELS[0].startswith("CutMan-1")
    assert MegamanCore(level="CutMan").level_entry()["id"] == "CutMan"
    assert MegamanCore(level=str(len(LEVELS) - 1)).level_entry()["id"] == "CutMan"
    with pytest.raises(ValueError):
        MegamanCore(level="nope")


def test_sensor_and_net_shapes():
    adapter = make_adapter("megaman", LEVELS[0], 600, 5000.0)
    vec, rays, tiles = read_sensors(adapter, "rays")
    assert vec.shape == (N_INPUTS_BY_GAME["megaman"],) == (26,)
    assert np.all(np.abs(vec) <= 1.0 + 1e-6) and vec[25] == 1.0
    labelled = sorted(r["i"] for g in adapter.SENSOR_LABELS for r in g["rows"])
    assert labelled == list(range(25)), "every slot but the bias needs a dashboard label"
    grid, _, _ = read_sensors(adapter, "grid")
    assert grid.shape == (sensor_dim("grid"),)
    net = make_net(GAConfig(n_inputs=26, n_outputs=N_OUTPUTS_BY_GAME["megaman"]))
    action = net.act(vec)
    assert len(action) == 4 and isinstance(action[3], bool)          # move_x, jump, move_y, fire
    assert len(make_net(GAConfig()).act(np.zeros(14, dtype=np.float32))) == 3   # other games: unchanged


def test_held_outputs_repeat():
    """The console wants a fresh press for each jump and each shot; a net just holds the output."""
    adapter = make_adapter("megaman", "CutMan-7-Corridor", 600, 5000.0)
    core = adapter.core
    fired = jumps = 0
    was_grounded = True
    for _ in range(240):
        before = len(core.shots)
        adapter.step(0, True, 0, True)
        fired += len(core.shots) > before
        jumps += was_grounded and not core.player.grounded
        was_grounded = core.player.grounded
    assert fired >= 3 and jumps >= 3, (fired, jumps)


def test_progress_follows_the_route_not_x():
    """In the climb slice the exit is up a ladder: walking sideways earns nothing, climbing does."""
    adapter = make_adapter("megaman", "CutMan-2-Climb", 2000, 5000.0)
    assert adapter.progress() == 0.0
    gx, gy = adapter._compass()
    assert gy < 0 or gx != 0                           # the compass points somewhere
    best = adapter._best_cost
    for _ in range(200):                                # follow the compass, crudely
        gx, gy = adapter._compass()
        p = adapter.core.player
        up = -1 if (gy < 0 and (p.climbing or adapter.core.tile_at(adapter.x, adapter.y) == "H")) else 0
        adapter.step(int(np.sign(gx)), gy < 0 and not up, up, False)
        if not adapter.alive:
            break
    assert adapter._best_cost < best and adapter.fitness() > 0.0


def test_fitness_counts_boss_damage_and_runs_are_deterministic():
    def run(seed):
        a = make_adapter("megaman", "CutMan-8-Boss", 1500, 5000.0)
        a.core._seed = seed
        a.reset()
        rng = np.random.default_rng(3)
        for _ in range(1200):
            if not a.alive:
                break
            a.step(int(rng.integers(-1, 2)), bool(rng.integers(2)), 0, True)
        return round(a.fitness(), 3), a.core.boss_damage, a.status, a.episode_stats()["cause"]

    first = run(11)
    assert first == run(11)
    a = make_adapter("megaman", "CutMan-8-Boss", 1500, 5000.0)
    a.step(0, False, 0, False)
    assert a.busy                                       # the intro takes the pad away: not a stall
    base = a.fitness()
    a.core.boss_damage = en.BOSS_HP // 2
    assert a.fitness() > base + 200
    stats = a.episode_stats()
    assert stats["level_len"] == 1000.0 and {"cause", "kills", "coins"} <= set(stats)


def test_enemy_kill_reaches_the_event_counters():
    a = make_adapter("megaman", "CutMan-1-Street", 600, 5000.0)
    core = a.core
    e = en.make("b", 10, 5, -1)
    core.enemies.append(e)
    assert core._hit_enemy(e) and e not in core.enemies
    assert a.events()["kills"] == 1 and a.level_events()["kills"] == a._level_enemies > 0


def test_a_ladder_is_seen_before_it_is_touched():
    """Rays pass through ladders. Under the Street ladder, whose bottom rung hangs two tiles above
    the floor, the touch sensor reads nothing: the overhead sensor has to."""
    a = make_adapter("megaman", "CutMan-1-Street", 600, 5000.0)
    core, p, ts = a.core, a.core.player, a.core.tile_size

    def stand(col, row, facing=1):
        p.x, p.y = col * ts + (ts - p.width) / 2.0, (row + 1) * ts - p.height
        p.vx = p.vy = 0.0
        p.grounded, p.climbing, p.facing = True, False, facing
        return read_sensors(a, "rays")[0]

    under = stand(53, 131)                              # on the floor, ladder cells at rows 124-129 above
    assert under[16] == 0.0 and under[23] > 0.0, (under[16], under[23])
    near = stand(50, 131)                               # three tiles short of it, facing it
    assert near[23] == 0.0 and 0.0 < near[24] < 1.0
    assert stand(52, 131)[24] == 1.0                    # the next tile over
    assert stand(50, 131, facing=-1)[24] == 0.0         # facing away: nothing ahead
    assert stand(40, 127)[23] == 0.0 and stand(40, 127)[24] == 0.0   # the stairs: no ladder in sight


def test_a_section_is_the_stage_itself_and_ends_where_the_next_begins():
    """Sections share the stage file: same world, own start, won on entering the next one's screens."""
    street, climb = MegamanCore(level="CutMan-1-Street"), MegamanCore(level="CutMan-2-Climb")
    assert street.level_file() == climb.level_file() == MegamanCore(level="CutMan").level_file()
    assert climb.player.climbing and not climb.player.grounded          # entered up the ladder: starts on it
    a = make_adapter("megaman", "CutMan-1-Street", 2000, 5000.0)
    core, p, ts = a.core, a.core.player, a.core.tile_size
    assert 0.0 <= a.progress() < 0.01 and a._end_cost > 0
    p.x, p.y = 61 * ts + 2, 123 * ts - p.height                          # near the top of the exit ladder
    core.screen = (3, 8)
    core.camera_x, core.camera_y = core._camera_target()
    for _ in range(300):
        a.step(0, False, -1, False)
        if not a.alive:
            break
    assert a.won and core.screen == (3, 7) and a.progress() > 0.97, (a.status, core.screen, a.progress())
    full = make_adapter("megaman", "CutMan", 600, 5000.0)               # the whole stage only ends at the boss
    assert not full.core.goal_screens and full._has_boss and not a._has_boss


def test_a_section_is_scored_over_several_varied_attempts():
    """Training plays a section several times: only a genome that wins every attempt is a winner,
    later attempts start with other seeds and energy, and fitness is the mean."""
    a = make_adapter("megaman", "CutMan-7-Corridor", 3600, 5000.0)
    a.attempts = 3
    a.reset()
    energies, frames = [a.core.player.hp], 0
    while a.alive and frames < 6000:                    # walking right wins the corridor every time
        before = len(a._done)
        a.step(1, False, 0, False)
        frames += 1
        if a.alive and len(a._done) > before:
            energies.append(a.core.player.hp)
    assert a.won and a.status == "WON" and a.attempt_wins == 3 and len(energies) == 3
    assert energies[0] == 28 and all(10 <= e <= 28 for e in energies) and len(set(energies)) > 1
    assert a.fitness() >= 6000 and a.progress() == 1.0

    b = make_adapter("megaman", "CutMan-7-Corridor", 3600, 5000.0)   # one that stands still: three stalls
    b.attempts = 3
    b.reset()
    frames = 0
    while b.alive and frames < 3000:
        b.step(0, False, 0, False)
        frames += 1
    assert not b.won and b.status == "STUCK" and len(b._done) == 3 and frames < 800 and b.fitness() < 50
    full = make_adapter("megaman", "CutMan", 600, 5000.0)             # the whole stage: always one attempt
    full.attempts = 3
    full.reset()
    assert full._n_attempts == 1


def test_a_win_pays_for_energy_left():
    a = make_adapter("megaman", "CutMan-7-Corridor", 600, 5000.0)
    hp_max = a.core.player.hp_max
    a.core.player.hp = hp_max
    full = a._attempt_fitness(True)
    a.core.player.hp = hp_max // 4
    low = a._attempt_fitness(True)
    assert full - low == pytest.approx(a.energy_bonus * (hp_max - hp_max // 4) / hp_max)
    assert low > a._FIT_SCALE + a.win_bonus           # winning still beats any loss
    loss = a._attempt_fitness(False)
    a.core.player.hp = hp_max
    assert a._attempt_fitness(False) == loss          # a loss is not paid for energy: no reward for hiding


def test_practice_section_is_playable_but_not_a_stage_leg():
    from code.games.megaman_mutants import Stage
    from code.neuro.relay import stage_legs
    core = MegamanCore(level="CutMan-4-Shaft-Gap")
    ts = core.tile_size
    assert (int((core.player.x + core.player.width / 2) // ts), int((core.player.y + core.player.height - 1) // ts)) == (86, 42)
    assert "CutMan-4-Shaft-Gap" not in [leg.level for leg in stage_legs("CutMan")]
    assert "CutMan-4-Shaft-Gap" not in [name for name, _ in Stage("CutMan").sections]
    assert "CutMan-4-Shaft-Gap" not in list_levels("megaman")
    assert "CutMan-4-Shaft-Gap" in list_levels("megaman", include_disabled=True)
    core.won = True
    core.reset()                                     # a won practice section does not advance into the campaign
    assert core.level_entry()["id"] == "CutMan-4-Shaft-Gap"
