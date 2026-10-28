"""Per-episode balance stats: one object per agent per episode, one subclass per game.

The trainer (one per population slot) and manual play drive it explicitly — nothing is
patched or hooked:

    stats = make_stats(game, core, level=..., persona=..., source=...)  # right after reset
    stats.on_step(jump)        # after every core.step()
    stats.finish(status)       # once, when the episode ends
    stats.to_dict()            # the episode's row in episodes.csv

to_dict() is the schema. A field that doesn't apply to a game is absent from its rows, never
written as 0, so "Mario destroyed 0 bricks" can't happen. The per-level metrics computed from
these rows live in code/stats/registry.py.

Stats never crash the game: a failing hook logs one warning and switches itself off.
"""
from __future__ import annotations

import sys

FPS = 60
ROUTE_EVERY = 8  # frames between route samples (~7.5 per second)

_warned: set[tuple[str, str]] = set()


def _warn_once(obj, hook: str, exc: Exception) -> None:
    key = (type(obj).__name__, hook)
    if key not in _warned:
        _warned.add(key)
        print(f"[stats] warning: {key[0]}.{hook} failed ({type(exc).__name__}: {exc}); "
              f"that metric is disabled for this run", file=sys.stderr, flush=True)


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


class EpisodeStats:
    """Shared per-episode counters. Subclasses supply the game's body, progress and extras."""

    JUMP_INDEX = 1  # manual play: which action slot is the jump button

    def __init__(self, core, *, game: str, level: str | None, persona: str, source: str,
                 agent: int | str = 0, gen: int = 0, session: str = "") -> None:
        self.core = core
        self.meta = {"persona": persona, "game": game, "level": str(level) if level is not None else "auto",
                     "source": source, "session": session, "gen": gen, "agent": agent}
        self.frames = 0
        self.time_s = 0.0   # game time: the core's own dt summed (real frame time in manual play)
        self.kills = 0      # summed from core.kills_step, for cores without a kills_total
        self.jumps = 0
        self._prev_jump = False
        self.speed_sum = 0.0
        self.status = "RUNNING"
        self.cause = ""
        self.end_xy: tuple[float, float] | None = None
        self._dead_hooks: set[str] = set()
        self.route: list[tuple[float, float]] = []
        self._safe("begin", self._begin)
        # Snapshot now: a win can load the next level before finish() runs (Sonic).
        self._dims = self._safe("level_dims", self.level_dims) or {}
        x, y = self._safe("pos", self.pos) or (0.0, 0.0)
        self.route.append((round(float(x), 1), round(float(y), 1)))

    # ── per-game hooks ───────────────────────────────────────────────────
    def pos(self) -> tuple[float, float]:
        raise NotImplementedError

    def speed(self) -> float:
        return 0.0

    def _begin(self) -> None:
        """Snapshot what the level looks like at the start (goal, loot pool, enemies)."""

    def _frame(self) -> None:
        """Per-frame watchers for events the core doesn't count itself."""

    def progress(self) -> float | None:
        """0 = start, 1 = goal, measured where the episode ended."""
        return None

    def level_dims(self) -> dict:
        ld = getattr(self.core, "level_data", None)
        ts = getattr(self.core, "tile_size", None) or getattr(ld, "tile_size", None) or 32
        return {"tile": int(ts), "level_w": round(float(getattr(ld, "width", 0) or 0), 1),
                "level_h": round(float(getattr(ld, "height", 0) or 0), 1)}

    def extra(self) -> dict:
        return {}

    @staticmethod
    def end_status(core, terminated: bool, truncated: bool, info: dict) -> str:
        """WON / STUCK / DEAD for an episode the caller has decided is over."""
        raise NotImplementedError

    # ── driver API ───────────────────────────────────────────────────────
    def _safe(self, hook: str, fn, *args):
        if hook in self._dead_hooks:
            return None
        try:
            return fn(*args)
        except Exception as exc:  # noqa: BLE001 — stats must never take the game down
            self._dead_hooks.add(hook)
            _warn_once(self, hook, exc)
            return None

    def on_step(self, jump: bool = False) -> None:
        if self.status != "RUNNING":
            return
        self.frames += 1
        # Headless cores step a fixed 1/60 s; human-mode Mario/Sonic step the real frame time
        # (manual play runs at 30 FPS), so frames / 60 would halve a human's times.
        dt = getattr(self.core, "dt", None)
        self.time_s += dt if isinstance(dt, (int, float)) and 0 < dt <= 0.25 else 1.0 / FPS
        self.kills += int(getattr(self.core, "kills_step", 0) or 0)
        if jump and not self._prev_jump:
            self.jumps += 1
        self._prev_jump = bool(jump)
        self.speed_sum += abs(self._safe("speed", self.speed) or 0.0)
        if self.frames % ROUTE_EVERY == 0:
            p = self._safe("pos", self.pos)
            if p:
                self.route.append((round(float(p[0]), 1), round(float(p[1]), 1)))
        self._safe("frame", self._frame)

    def finish(self, status: str) -> None:
        if self.status != "RUNNING":
            return
        self.status = status
        if status == "WON":
            self.cause = ""
        elif status == "STUCK":
            self.cause = "Stall"
        else:
            core = self.core
            self.cause = str(getattr(core, "death_cause", "") or getattr(core, "last_cause", "") or "")
            if not self.cause:  # the frame budget ran out without a death
                self.cause = "Timeout" if self.frames >= (getattr(core, "max_steps", 0) or 10 ** 9) else "?"
        # A win can reload the level on the same frame (Sonic), so the body is back at the spawn:
        # keep the last sampled route point for wins.
        p = self.route[-1] if status == "WON" else self._safe("pos", self.pos)
        if p:
            self.end_xy = (float(p[0]), float(p[1]))
            if self.route[-1] != (round(float(p[0]), 1), round(float(p[1]), 1)):
                self.route.append((round(float(p[0]), 1), round(float(p[1]), 1)))
        self._final = {}
        prog = 1.0 if status == "WON" else self._safe("progress", self.progress)
        self._final["progress"] = None if prog is None else round(_clamp01(prog), 4)
        self._final.update(self._safe("extra", self.extra) or {})

    @property
    def finished(self) -> bool:
        return self.status != "RUNNING"

    def to_dict(self) -> dict:
        if not self.finished:
            raise RuntimeError("to_dict() before finish()")
        row = dict(self.meta)
        row.update({
            "status": self.status,
            "cause": self.cause,
            "frames": self.frames,
            "time_s": round(self.time_s, 2),
            "jumps": self.jumps,
            "avg_speed": round(self.speed_sum / max(self.frames, 1), 2),
            "end_x": round(self.end_xy[0], 1) if self.end_xy else None,
            "end_y": round(self.end_xy[1], 1) if self.end_xy else None,
        })
        row.update(self._dims)
        row.update(self._final)
        row["route"] = [list(p) for p in self.route]
        return row


# ── side-scrollers: Mario, Mega Man, Sonic ───────────────────────────────────

class _ScrollerStats(EpisodeStats):
    """Progress = death point interpolated horizontally between the start and the goal."""

    def pos(self) -> tuple[float, float]:
        g = self.core.player.gObj
        return g.x + g.width / 2.0, g.y + g.height / 2.0

    def speed(self) -> float:
        return float(self.core.player.vx)

    def _begin(self) -> None:
        ld = self.core.level_data
        self._start_x = self.pos()[0]
        goals = [g.gObj.x + g.gObj.width / 2.0 for g in (getattr(ld, "goals", None) or [])]
        ahead = [x for x in goals if x > self._start_x]
        if goals:
            self._goal_x = min(ahead) if ahead else goals[0]  # the nearest goal in front of the spawn
        else:
            self._goal_x = float(getattr(ld, "width", 0) or 0)
        # free coins plus the ones inside ? blocks (Mario 1-1 has only the latter)
        self._level_coins = len(getattr(ld, "coins", None) or []) + sum(
            1 for q in (getattr(ld, "qblocks", None) or []) if getattr(q, "contains", None) == "coin")

    def progress(self) -> float | None:
        if not self.end_xy:
            return None
        span = self._goal_x - self._start_x
        if span <= 0:
            width = float(getattr(self.core.level_data, "width", 0) or 0)
            return self.end_xy[0] / width if width else None
        return (self.end_xy[0] - self._start_x) / span

    def extra(self) -> dict:
        core = self.core
        kills = getattr(core, "kills_total", None)  # Mario / Sonic only count kills per step
        return {"coins": int(getattr(core, "coins_total", 0) or 0), "level_coins": self._level_coins,
                "kills": int(kills) if kills is not None else self.kills}


class MarioStats(_ScrollerStats):
    """Power-ups lost = tiers the power state drops between frames (FIRE -> BIG -> SMALL); a hit
    pops one tier and i-frames stop a second hit, so each drop is one lost power-up."""

    _TIER = {"SMALL": 0, "BIG": 1, "FIRE": 2}

    def _tier(self) -> int:
        pm = self.core.player.power_machine  # re-read: a level load can swap the player object
        return self._TIER.get(pm.state.name, 0)

    def _begin(self) -> None:
        self._powerups_lost = 0  # set first: if the tier read fails, only this metric goes quiet
        super()._begin()
        self._last_tier = self._tier()

    def _frame(self) -> None:
        super()._frame()
        tier = self._tier()
        self._powerups_lost += max(0, self._last_tier - tier)
        self._last_tier = tier

    def extra(self) -> dict:
        return {**super().extra(), "powerups_lost": self._powerups_lost}

    @staticmethod
    def end_status(core, terminated, truncated, info) -> str:
        if core.reached_goal:
            return "WON"
        return "STUCK" if core.death_cause == "Stall" else "DEAD"


class SonicStats(_ScrollerStats):
    @staticmethod
    def end_status(core, terminated, truncated, info) -> str:
        if info.get("won"):  # core.reached_goal is wiped by the mid-step level reload
            return "WON"
        if core.death_cause == "Stall" or (truncated and not terminated):
            return "STUCK"
        return "DEAD"


# ── Meat Boy: 2-D mazes, progress along the BFS shortest path ────────────────

class MeatboyStats(EpisodeStats):
    """Progress = share of the start's BFS path distance to the goal covered at the death point
    (exact shortest path through the maze, not a straight line)."""

    JUMP_INDEX = 2  # [move, run, jump]

    def pos(self) -> tuple[float, float]:
        p = self.core.player
        return p.x + p.width / 2.0, p.y + p.height / 2.0

    def speed(self) -> float:
        return float(self.core.player.vx)

    def _begin(self) -> None:
        self._wall_jumps = 0
        self._lockout = int(self.core.player.air_lockout)
        self._bfs_start = float(self.core._bfs_norm_dist())
        self._bfs_last = self._bfs_start

    def _frame(self) -> None:
        # A wall jump is the only thing that raises air_lockout (to control_lockout_frames, 6);
        # otherwise it only counts down. Its own flag is set and cleared inside one control()
        # call, so the rise is what's visible between frames. (Needs control_lockout_frames >= 2.)
        lockout = int(self.core.player.air_lockout)
        if lockout > self._lockout:
            self._wall_jumps += 1
        self._lockout = lockout
        d = float(self.core._bfs_norm_dist())
        if d >= 0.0:  # -1 = off-grid (fell out) — keep the last cell the player stood on
            self._bfs_last = d

    def progress(self) -> float | None:
        if self._bfs_start <= 0.0 or self._bfs_last < 0.0:
            return None
        return (self._bfs_start - self._bfs_last) / self._bfs_start

    def extra(self) -> dict:
        return {"wall_jumps": self._wall_jumps}

    @staticmethod
    def end_status(core, terminated, truncated, info) -> str:
        if core.won:
            return "WON"
        return "STUCK" if truncated else "DEAD"


# ── Bomberman: top-down grid ─────────────────────────────────────────────────

class BombermanStats(EpisodeStats):
    """Progress = share of the start's cost-to-exit removed (bricks cost extra), at the death point.
    The exit is "found" when a hidden exit is bombed open, or when the player reaches a visible one."""

    JUMP_INDEX = 2  # [dx, dy, bomb] — the "jump" button drops a bomb

    def pos(self) -> tuple[float, float]:
        p = self.core.player
        return p.x + p.width / 2.0, p.y + p.height / 2.0

    def speed(self) -> float:
        p = self.core.player
        return (float(p.vx) ** 2 + float(p.vy) ** 2) ** 0.5

    def _begin(self) -> None:
        core = self.core
        ld = core.level_data
        self._start_cost = max(float(core.start_cost()), 1.0)
        self._enemies = len(core.enemies)
        from code.games.bomberman_core import POWERUPS
        self._powerups = sum(ch in POWERUPS for row in ld.grid for ch in row)
        self._exit_hidden = bool(ld.exit_hidden)
        self._exit_found_frame: int | None = None
        self._bricks_at_exit: int | None = None
        self._first_kill_frame: int | None = None
        self._bomb_drops: list[tuple[int, int]] = []
        self._chain_bombs = 0
        self._bombs = {id(b): b for b in core.bombs}  # live bombs last frame (refs keep ids unique)

    def _watch_bombs(self) -> None:
        """A bomb new in core.bombs was dropped at its tile. A bomb gone from it went off: with
        fuse left, another blast set it off (a chain); at fuse 0, its own fuse did."""
        now = {id(b): b for b in self.core.bombs}
        self._bomb_drops += [(b.tx, b.ty) for k, b in now.items() if k not in self._bombs]
        self._chain_bombs += sum(1 for k, b in self._bombs.items() if k not in now and b.fuse > 0)
        self._bombs = now

    def _frame(self) -> None:
        core = self.core
        self._watch_bombs()
        if self._first_kill_frame is None and core.kills_total > 0:
            self._first_kill_frame = self.frames
        if self._exit_found_frame is None:
            ex, ey = core.level_data.exit
            if self._exit_hidden:
                from code.games.bomberman_core import EXIT
                found = core.tile(ex, ey) == EXIT
            else:
                p = core.player
                found = core._center_tile(p.x, p.y, p.width, p.height) == (ex, ey)
            if found:
                self._exit_found_frame = self.frames
                self._bricks_at_exit = int(core.bricks_destroyed)

    def progress(self) -> float | None:
        return 1.0 - float(self.core.goal_cost()) / self._start_cost

    def extra(self) -> dict:
        core = self.core
        found = self._exit_found_frame
        return {
            "kills": int(core.kills_total), "level_enemies": self._enemies,
            "powerups": int(core.coins_total), "level_powerups": self._powerups,
            "bricks": int(core.bricks_destroyed),
            "bombs_placed": len(self._bomb_drops),
            "bomb_drops": [list(t) for t in self._bomb_drops],
            "chain_bombs": self._chain_bombs,
            "exit_hidden": int(self._exit_hidden),
            "exit_found": int(found is not None),
            "exit_found_s": round(found / FPS, 2) if found is not None else None,
            "bricks_before_exit": self._bricks_at_exit,
            "first_kill_s": round(self._first_kill_frame / FPS, 2) if self._first_kill_frame is not None else None,
        }

    def level_dims(self) -> dict:
        ld = self.core.level_data
        ts = int(self.core.tile_size)
        return {"tile": ts, "level_w": float(ld.cols * ts), "level_h": float(ld.rows * ts)}

    @staticmethod
    def end_status(core, terminated, truncated, info) -> str:
        return "WON" if core.won else "DEAD"


# ── Mega Man (NES): a route through stacked screens, in sections ───────────

_mm1_fields: dict = {}   # (level file, mtime) -> route field; shared by every episode on that file


class MegamanStats(EpisodeStats):
    """Progress = share of the route covered at the furthest point reached (the route field of
    code/games/megaman_route.py: the stage climbs and doubles back, so x means nothing).
    Besides the shared counters it splits the attempt by stage section: time, damage and kills
    in each, and the energy Mega Man entered it with."""

    JUMP_INDEX = 2  # [dx, dy, jump, fire]

    def pos(self) -> tuple[float, float]:
        p = self.core.player
        return p.x + p.width / 2.0, p.y + p.height / 2.0

    def speed(self) -> float:
        return float(self.core.player.vx)

    def _begin(self) -> None:
        import os

        from code.games import megaman_enemies as en
        from code.games.megaman_route import cost_at, route_field, section_costs
        core = self.core
        self._cost_at = cost_at
        path = core.level_file()
        key = (path, os.path.getmtime(path))
        if key not in _mm1_fields:
            _mm1_fields.clear()
            _mm1_fields[key] = route_field(core.level_data)
        self._cost = _mm1_fields[key]
        entry = core.level_entry() if not core._level_file else {}
        self._start_cost, self._end_cost = section_costs(self._cost, entry, self.pos(), float(core.tile_size))
        self._best_cost = self._start_cost
        own = {tuple(s) for s in (entry.get("section") or {}).get("screens") or []}
        sc, sr = core.screen_cols, core.screen_rows
        spawns = core.level_data.spawns
        self._level_enemies = sum(1 for g, c, r in spawns if (g in en.ENEMY_GLYPHS and g != "c" or g == en.BOSS_GLYPH)
                                  and (not own or (c // sc, r // sr) in own))
        # the stage's sections, in order, by the screens each owns
        self._sections = [(name, {tuple(s) for s in e["section"]["screens"]}) for name, e in core.levels
                          if isinstance(e, dict) and e.get("file") == entry.get("file")
                          and (e.get("section") or {}).get("screens")]
        self._by_screen = {s: i for i, (_, screens) in enumerate(self._sections) for s in screens}
        self._split = [{"section": name, "entered": False, "frames": 0, "hp_in": None, "damage": 0, "kills": 0}
                       for name, _ in self._sections]
        self._cur = -1
        self._hp, self._kills_seen = core.player.hp, core.kills_total
        self._damage = 0
        self._shots = 0
        self._shots_out = len(core.shots)

    def _frame(self) -> None:
        core = self.core
        c = self._cost_at(self._cost, *self.pos(), float(core.tile_size))
        if c < self._best_cost:
            self._best_cost = c
        hurt = max(0, self._hp - core.player.hp)
        killed = core.kills_total - self._kills_seen
        self._hp, self._kills_seen = core.player.hp, core.kills_total
        self._damage += hurt
        self._shots += max(0, len(core.shots) - self._shots_out)
        self._shots_out = len(core.shots)
        self._cur = self._by_screen.get(core.screen, self._cur)
        if self._cur >= 0:
            s = self._split[self._cur]
            if not s["entered"]:
                s["entered"], s["hp_in"] = True, core.player.hp + hurt
            s["frames"] += 1
            s["damage"] += hurt
            s["kills"] += killed

    def progress(self) -> float | None:
        span = self._start_cost - self._end_cost
        if not span < float("inf"):          # no route from the start to the exit (a breaking mutant):
            return None                      # "how far along the route" has no answer
        return (self._start_cost - self._best_cost) / span if span > 0 else None

    def extra(self) -> dict:
        core = self.core
        out = {"kills": int(core.kills_total), "level_enemies": self._level_enemies,
               "coins": int(core.coins_total), "hp": int(core.player.hp), "damage_taken": self._damage,
               "shots": self._shots, "boss_damage": int(core.boss_damage)}
        if self._sections:
            out["section"] = self._sections[self._cur][0] if self._cur >= 0 else None
            out["sections_reached"] = sum(s["entered"] for s in self._split)
            out["sections"] = [s for s in self._split if s["entered"]]
        return out

    @staticmethod
    def end_status(core, terminated, truncated, info) -> str:
        if core.won:
            return "WON"
        return "STUCK" if truncated else "DEAD"


STATS_BY_GAME: dict[str, type[EpisodeStats]] = {
    "megaman": MegamanStats,
    "mario": MarioStats,
    "sonic": SonicStats,
    "meatboy": MeatboyStats,
    "bomberman": BombermanStats,
}
GAME_ALIASES = {"platformer": "mario"}


def stats_class(game: str) -> type[EpisodeStats]:
    return STATS_BY_GAME[GAME_ALIASES.get(game, game)]


def make_stats(game: str, core, **meta) -> EpisodeStats:
    game = GAME_ALIASES.get(game, game)
    return STATS_BY_GAME[game](core, game=game, **meta)
