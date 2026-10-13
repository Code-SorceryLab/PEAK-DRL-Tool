"""Thin game adapters exposing the minimal surface the neuroevolution loop needs.

Each adapter wraps one game-core instance. The trainer never touches gym APIs,
observations, or rewards — only this protocol.
"""
from __future__ import annotations

import os
from typing import Protocol

# SDL otherwise turns SIGTERM into a quit event nobody polls, so `kill` / Popen.terminate() did nothing
os.environ.setdefault("SDL_NO_SIGNAL_HANDLERS", "1")
import pygame  # noqa: E402


class GameAdapter(Protocol):
    alive: bool
    won: bool
    status: str            # RUNNING | STUCK | DEAD | WON
    tile_size: int

    @property
    def x(self) -> float: ...
    @property
    def y(self) -> float: ...
    @property
    def vx(self) -> float: ...
    @property
    def vy(self) -> float: ...
    @property
    def grounded(self) -> bool: ...
    @property
    def can_jump(self) -> bool: ...
    @property
    def camera(self) -> tuple[float, float]: ...

    def reset(self) -> None: ...
    def step(self, move_x: int, jump: bool, move_y: int = 0) -> None: ...
    def render(self, surface: pygame.Surface) -> None: ...
    def solid_at(self, wx: float, wy: float) -> bool: ...
    def enemy_positions(self) -> list[tuple[float, float]]: ...
    def qblock_count_near(self, r_tiles: int) -> int: ...
    def fitness(self) -> float: ...
    def episode_stats(self) -> dict: ...
    def set_level(self, level: str) -> None: ...


INDEXED_GAMES = {"meatboy", "bomberman"}     # level ids are list indices, not names
TOPDOWN_GAMES = {"bomberman"}                 # 2-D movement: the net grows up/down outputs
N_OUTPUTS_BY_GAME = {"bomberman": 5, "megaman": 6}   # megaman: left, right, jump, up, down, fire
N_INPUTS_BY_GAME = {"bomberman": 16, "megaman": 26}   # ray-mode games with their own sense(): see the adapter's SENSOR_LABELS
OWN_LEVEL_FILES = {"megaman": "megaman_levels.yaml"}  # named levels kept in the game's own, editor-owned file
# Archived: still fully playable and trainable by name (--game sonic), and old runs still replay,
# but kept out of the menus and off the default sweep roster so they do not cost probe budget.
ARCHIVED_GAMES = {"sonic"}


def _set_locked_level(core, level: str) -> None:
    """Repoint a gym-style core at a new level; applied by its next reset()."""
    core.locked_level = str(level)
    core.world = str(level)


def list_levels(game: str, include_disabled: bool = False) -> list[str]:
    """Enabled level ids for a game, in config order. Meatboy levels are indices."""
    import yaml
    games_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "games")
    if game in INDEXED_GAMES:  # meatboy / bomberman: levels are a list, ids are indices
        with open(os.path.join(games_dir, f"{game}_config.yaml"), encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        n = len(data.get("levels") or [])
        if include_disabled:      # disabled entries follow the enabled ones, so ids stay stable
            n += len(data.get("disabled_levels") or [])
        return [str(i) for i in range(n)]
    if game in OWN_LEVEL_FILES:
        with open(os.path.join(games_dir, OWN_LEVEL_FILES[game]), encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return list((data.get("levels") or {}).keys()) + \
            (list((data.get("disabled_levels") or {}).keys()) if include_disabled else [])
    with open(os.path.join(games_dir, "game_config.yaml"), encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    sections = {"mario": data, "sonic": data.get("sonic", {})}
    if game not in sections:
        raise SystemExit(f"unknown game '{game}' (available: {', '.join(_ADAPTERS)})")
    section = sections[game]
    names = list((section.get("levels") or {}).keys())
    if include_disabled:
        names += list((section.get("disabled_levels") or {}).keys())
    return names


def validate_level(game: str, level: str) -> None:
    """Fail fast on a level the cores can't load — a disabled or unknown level
    otherwise falls back to an empty world and produces garbage fitness."""
    enabled = list_levels(game)
    if level in enabled:
        return
    if level in list_levels(game, include_disabled=True):
        raise SystemExit(f"level '{level}' exists but is DISABLED in the config — "
                         f"enable it first (menu 10: Toggle Levels)")
    raise SystemExit(f"unknown level '{level}' for game '{game}' "
                     f"(enabled: {', '.join(enabled) or 'none'})")


def _move_md(move_x: int, sprint: bool) -> int:
    """move_x -1/0/+1 -> MultiDiscrete move value; sprint picks the run variant (walk + 1)."""
    base = {-1: 1, 0: 0, 1: 3}[move_x]
    return base + 1 if sprint and move_x else base


def _win_time_bonus(adapter) -> float:
    """Speedrunner fitness: time left on the clock pays out on a win."""
    return adapter.time_rate * max(float(getattr(adapter.core, "timer", 0.0) or 0.0), 0.0)


def _count_coins(core) -> int:
    ld = getattr(core, "level_data", None)  # some cores build level_data on the first reset
    return len(ld.coins) if ld is not None else 0


def _episode_stats(adapter, x: float) -> dict:
    """Game metrics shared by every core, read defensively (not every core has every field)."""
    core = adapter.core
    return {
        "x": round(float(x), 1),
        "score": int(getattr(core, "score", 0) or 0),
        "coins": int(getattr(core, "coins_total", 0) or 0),
        "kills": adapter.events()["kills"],
        "time_left": round(float(getattr(core, "timer", 0.0) or 0.0), 1),
        "cause": str(getattr(core, "death_cause", "") or getattr(core, "last_cause", "") or ""),
        # balance-metric fields (Amr's table): where the episode ended, level extents, loot pool
        "end_x": round(float(adapter._end_xy[0]), 1) if adapter._end_xy else None,
        "end_y": round(float(adapter._end_xy[1]), 1) if adapter._end_xy else None,
        "level_len": round(float(getattr(getattr(core, "level_data", None), "width", 0.0) or 0.0), 1),
        "level_coins": adapter._level_coins,
    }


class _EventWeights:
    """Persona event weights (personas.py). A weight is what doing ALL of an event in the level is
    worth, as a share of the progress-to-goal scale: kills 1.0 = clearing every enemy pays as much
    as reaching the goal. Per-level shares keep one weight meaningful across games and across
    crowded vs empty levels; an event the level does not contain pays nothing.
    make_adapter sets `weights`; empty = progress only."""
    weights: dict[str, float] = {}
    kills = 0           # side-scroller cores only count kills per step; their adapters sum them here
    _level_enemies = 0  # set on reset, beside _level_coins

    def _count_enemies(self) -> int:
        try:
            return len(self.enemy_positions())
        except Exception:  # a core that builds its level on the first reset
            return 0

    def events(self) -> dict[str, int]:
        core = self.core
        return {"kills": int(getattr(core, "kills_total", self.kills) or 0),
                "coins": int(getattr(core, "coins_total", 0) or 0)}

    def level_events(self) -> dict[str, int]:
        """How many of each event the current level holds (0 = a weight on it is dead here)."""
        return {"kills": self._level_enemies, "coins": self._level_coins}

    def progress_scale(self) -> float:
        return float(getattr(self, "_FIT_SCALE", 0.0) or self.core.level_data.width)

    def fitness(self) -> float:
        f = self._progress_fitness()
        if self.weights:
            ev, tot = self.events(), self.level_events()
            f += self.progress_scale() * sum(w * min(ev.get(e, 0) / tot[e], 1.0)
                                             for e, w in self.weights.items() if tot.get(e))
        return f


class MarioAdapter(_EventWeights):
    """Wraps PlatformerCore, pinned to one level, one life, no curriculum, no gym obs."""

    # move_x -1/0/+1 -> MultiDiscrete move component (1=left, 0=idle, 3=right; walk speed only)
    _MOVE_MD = {-1: 1, 0: 0, 1: 3}

    def __init__(self, level: str | None, max_frames: int, win_bonus: float,
                 sprint: bool = False, time_rate: float = 0.0) -> None:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        from code.games.platformer_core import PlatformerCore
        from code.games.modules.Parameters.Map_parameters import (
            TILE_GROUND, TILE_PLATFORM, TILE_QBLOCK, TILE_CRUMBLE, TILE_SIZE,
        )
        self._solid_codes = {TILE_GROUND, TILE_PLATFORM, TILE_QBLOCK, TILE_CRUMBLE}
        self.tile_size = TILE_SIZE
        self.win_bonus = win_bonus
        self.sprint = sprint        # persona capability: run/sprint action variants
        self.time_rate = time_rate  # persona objective: fitness per second left on a win
        level_kw = {"world": level} if level else {}  # None -> first level in game_config.yaml
        self.core = PlatformerCore(
            render_mode="none",
            **level_kw,
            lock_level=True,
            curriculum_enabled=False,
            terminate_on_goal=True,
            anti_stall=True,
            num_rays=0,
            max_steps=max_frames,
            skip_obs=True,
            dijkstra_enabled=False,  # grid sensors call _obs() directly; the oracle channel is unused
        )
        # Lives are hardcoded to 3; >1 makes _handle_death soft-reset mid-episode,
        # which restarts the level without ending the episode and corrupts fitness.
        self.core.max_lives = 1
        self.core.lives = 1
        self.alive = True
        self.won = False
        self.status = "RUNNING"
        self._end_xy: tuple[float, float] | None = None
        self._level_coins = _count_coins(self.core)
        self._level_enemies = self._count_enemies()

    # ── state ────────────────────────────────────────────────────────────

    @property
    def _p(self):  # player center
        return self.core.player

    @property
    def x(self) -> float:
        g = self._p.gObj
        return g.x + g.width / 2.0

    @property
    def y(self) -> float:
        g = self._p.gObj
        return g.y + g.height / 2.0

    @property
    def vx(self) -> float:
        return float(self._p.vx)

    @property
    def vy(self) -> float:
        return float(self._p.vy)

    @property
    def grounded(self) -> bool:
        return bool(self._p.on_ground)

    @property
    def can_jump(self) -> bool:
        return self._p.coyote > 0 and self._p.jump_hold == 0

    @property
    def camera(self) -> tuple[float, float]:
        return float(self.core.camera_x), float(self.core.camera_y)

    # ── control ──────────────────────────────────────────────────────────

    def reset(self) -> None:
        self.core.reset()
        self.core.lives = 1
        self.alive = True
        self.won = False
        self.status = "RUNNING"
        self._end_xy: tuple[float, float] | None = None
        self._level_coins = _count_coins(self.core)
        self._level_enemies = self._count_enemies()
        self.kills = 0

    def step(self, move_x: int, jump: bool, move_y: int = 0) -> None:
        if not self.alive:
            return
        _, _, terminated, truncated, _ = self.core.step([_move_md(move_x, self.sprint), int(jump), 0])
        self.kills += int(getattr(self.core, "kills_step", 0) or 0)
        if terminated or truncated:
            self.alive = False
            self._end_xy = (self.x, self.y)
            self.won = bool(self.core.reached_goal)
            if self.won:
                self.status = "WON"
            elif self.core.death_cause == "Stall":
                self.status = "STUCK"
            else:
                self.status = "DEAD"

    def render(self, surface: pygame.Surface) -> None:
        self.core.render(surface)

    # ── queries for sensors ──────────────────────────────────────────────

    def solid_at(self, wx: float, wy: float) -> bool:
        ld = self.core.level_data
        tx, ty = int(wx // self.tile_size), int(wy // self.tile_size)
        if tx < 0:
            return True  # level border acts as a wall
        if tx >= ld.cols or ty < 0 or ty >= ld.rows:
            return False
        return ld.grid[ty][tx] in self._solid_codes

    def enemy_positions(self) -> list[tuple[float, float]]:
        out = []
        for e in self.core.level_data.enemies:
            g = e.gObj
            if g.active:
                out.append((g.x + g.width / 2.0, g.y + g.height / 2.0))
        for bar in self.core.level_data.firebars:   # sensed like enemies, but cannot be killed
            out.extend(bar.balls())
        return out

    def _count_enemies(self) -> int:
        return sum(1 for e in self.core.level_data.enemies if e.gObj.active)

    def qblock_count_near(self, r_tiles: int) -> int:
        r = r_tiles * self.tile_size
        px, py = self.x, self.y
        n = 0
        for q in self.core.level_data.qblocks:
            g = q.gObj
            if abs(g.x + g.width / 2.0 - px) <= r and abs(g.y + g.height / 2.0 - py) <= r:
                n += 1
        return n

    def _progress_fitness(self) -> float:
        return float(self.core.max_x_seen) + ((self.win_bonus + _win_time_bonus(self)) if self.won else 0.0)

    def episode_stats(self) -> dict:
        return _episode_stats(self,self.core.max_x_seen)

    def set_level(self, level: str) -> None:
        _set_locked_level(self.core, level)


class SonicAdapter(_EventWeights):
    """Wraps SonicCore. Goal does NOT terminate the core's episode — we end it on info['won']."""

    _MOVE_MD = {-1: 1, 0: 0, 1: 3}

    def __init__(self, level: str | None, max_frames: int, win_bonus: float,
                 sprint: bool = False, time_rate: float = 0.0) -> None:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        from code.games.sonic_core import SonicCore
        from code.games.modules.Parameters.Map_parameters import (
            TILE_GROUND, TILE_PLATFORM, TILE_QBLOCK, TILE_SIZE,
        )
        self._solid_codes = {TILE_GROUND, TILE_PLATFORM, TILE_QBLOCK}
        self.tile_size = TILE_SIZE
        self.win_bonus = win_bonus
        self.sprint = sprint        # persona capability: run/sprint action variants
        self.time_rate = time_rate  # persona objective: fitness per second left on a win
        level_kw = {"world": level} if level else {}
        self.core = SonicCore(
            render_mode="none",
            **level_kw,
            lock_level=True,
            curriculum_enabled=False,
            anti_stall=True,  # required: max_x_seen only updates inside the stall tracker
            max_steps=max_frames,
        )
        self.core.max_lives = 1  # lives>1 soft-resets mid-episode and corrupts fitness
        self.core.lives = 1
        self.alive = True
        self.won = False
        self.status = "RUNNING"
        self._end_xy: tuple[float, float] | None = None
        self._level_coins = _count_coins(self.core)
        self._level_enemies = self._count_enemies()

    @property
    def x(self) -> float:
        g = self.core.player.gObj
        return g.x + g.width / 2.0

    @property
    def y(self) -> float:
        g = self.core.player.gObj
        return g.y + g.height / 2.0

    @property
    def vx(self) -> float:
        return float(self.core.player.vx)

    @property
    def vy(self) -> float:
        return float(self.core.player.vy)

    @property
    def grounded(self) -> bool:
        return bool(self.core.player.on_ground)

    @property
    def can_jump(self) -> bool:
        p = self.core.player
        return bool(p.on_ground or p.coyote > 0)

    @property
    def camera(self) -> tuple[float, float]:
        return float(self.core.camera_x), float(self.core.camera_y)

    def reset(self) -> None:
        self.core.reset()
        self.core.lives = 1
        self.alive = True
        self.won = False
        self.status = "RUNNING"
        self._end_xy: tuple[float, float] | None = None
        self._level_coins = _count_coins(self.core)
        self._level_enemies = self._count_enemies()
        self.kills = 0

    def step(self, move_x: int, jump: bool, move_y: int = 0) -> None:
        if not self.alive:
            return
        _, _, terminated, truncated, info = self.core.step([_move_md(move_x, self.sprint), int(jump), 0])
        self.kills += int(getattr(self.core, "kills_step", 0) or 0)
        won = bool(info.get("won"))  # core.reached_goal is wiped by the mid-step level reload
        if won or terminated or truncated:
            self.alive = False
            self._end_xy = (self.x, self.y)
            self.won = won
            if won:
                self.status = "WON"
            elif self.core.death_cause == "Stall" or (truncated and not terminated):
                self.status = "STUCK"
            else:
                self.status = "DEAD"

    def render(self, surface: pygame.Surface) -> None:
        # Headless SonicCore never updates its camera — mirror the human-mode logic here.
        core = self.core
        if core.player:
            core.camera_x = max(0.0, min(self.x - core.WIDTH / 2.0, core.level_data.width - core.WIDTH))
            core.camera_y = max(0.0, min(self.y - core.HEIGHT / 2.0, core.level_data.height - core.HEIGHT))
        core.render(surface, blit_only=True)

    def solid_at(self, wx: float, wy: float) -> bool:
        ld = self.core.level_data
        tx, ty = int(wx // self.tile_size), int(wy // self.tile_size)
        if tx < 0:
            return True
        if tx >= ld.cols or ty < 0 or ty >= ld.rows:
            return False
        return ld.grid[ty][tx] in self._solid_codes

    def enemy_positions(self) -> list[tuple[float, float]]:
        out = []
        for e in self.core.badniks:
            g = e.gObj
            if g.active:
                out.append((g.x + g.width / 2.0, g.y + g.height / 2.0))
        return out

    def qblock_count_near(self, r_tiles: int) -> int:
        return 0

    def _progress_fitness(self) -> float:
        return float(self.core.max_x_seen) + ((self.win_bonus + _win_time_bonus(self)) if self.won else 0.0)

    def episode_stats(self) -> dict:
        return _episode_stats(self,self.core.max_x_seen)

    def set_level(self, level: str) -> None:
        _set_locked_level(self.core, level)


class MeatboyAdapter(_EventWeights):
    """Wraps MeatboyCore (plain class, no gym). Levels are 2-D mazes, so fitness is
    BFS-distance-to-goal progress scaled to ~pixels (0..1000) instead of max_x."""

    _MOVE_MD = {-1: 1, 0: 0, 1: 2}
    _FIT_SCALE = 1000.0

    def __init__(self, level: str | None, max_frames: int, win_bonus: float,
                 sprint: bool = False, time_rate: float = 0.0) -> None:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()  # MeatboyCore does not init pygame itself
        from code.games.meatboy_core import MeatboyCore
        from code.games.modules.Parameters.Map_parameters import (
            TILE_GROUND, TILE_PLATFORM, TILE_QBLOCK, TILE_CRUMBLE,
        )
        self._solid_codes = {TILE_GROUND, TILE_PLATFORM, TILE_QBLOCK, TILE_CRUMBLE}
        self.win_bonus = win_bonus
        self.sprint = sprint        # persona capability: run/sprint action variants
        self.time_rate = time_rate  # persona objective: fitness per second left on a win
        self.core = MeatboyCore(render_mode="none", max_steps=max_frames)
        self.tile_size = int(self.core.tile_size)
        if level is not None:
            self.core._level_idx = int(level)  # meatboy levels are indexed, not named
        self.alive = True
        self.won = False
        self.status = "RUNNING"
        self._end_xy: tuple[float, float] | None = None
        self._level_coins = _count_coins(self.core)
        self._level_enemies = self._count_enemies()
        self._best_bfs = 1.0
        self.reset()

    @property
    def x(self) -> float:
        p = self.core.player
        return p.x + p.width / 2.0

    @property
    def y(self) -> float:
        p = self.core.player
        return p.y + p.height / 2.0

    @property
    def vx(self) -> float:
        return float(self.core.player.vx)

    @property
    def vy(self) -> float:
        return float(self.core.player.vy)

    @property
    def grounded(self) -> bool:
        return bool(self.core.player.on_ground)

    @property
    def can_jump(self) -> bool:
        p = self.core.player
        return bool(p.on_ground or p.coyote > 0 or (not p.on_ground and (p.contact_left or p.contact_right)))

    @property
    def camera(self) -> tuple[float, float]:
        cx, cy = self.core._camera(self.core.WIDTH, self.core.HEIGHT)
        return float(cx), float(cy)

    def reset(self) -> None:
        self.core.won = False  # reset() advances to the next level when won is left True
        self.core.reset()
        self.alive = True
        self.won = False
        self.status = "RUNNING"
        self._end_xy: tuple[float, float] | None = None
        self._level_coins = _count_coins(self.core)
        self._level_enemies = self._count_enemies()
        self._best_bfs = 1.0

    def step(self, move_x: int, jump: bool, move_y: int = 0) -> None:
        if not self.alive:
            return
        _, _, terminated, truncated, info = self.core.step(
            [self._MOVE_MD[move_x], int(self.sprint), int(jump)])
        bfs = float(info.get("bfs_dist", -1.0))
        if 0.0 <= bfs < self._best_bfs:
            self._best_bfs = bfs
        if terminated or truncated:
            self.alive = False
            self._end_xy = (self.x, self.y)
            self.won = bool(self.core.won)
            self.status = "WON" if self.won else ("STUCK" if truncated else "DEAD")

    def render(self, surface: pygame.Surface) -> None:
        self.core.render(surface)

    def solid_at(self, wx: float, wy: float) -> bool:
        ld = self.core.level_data
        tx, ty = int(wx // self.tile_size), int(wy // self.tile_size)
        if tx < 0:
            return True
        if tx >= ld.cols or ty < 0 or ty >= ld.rows:
            return False
        return ld.grid[ty][tx] in self._solid_codes

    def enemy_positions(self) -> list[tuple[float, float]]:
        return [(float(s.cx), float(s.cy)) for s in self.core.level_data.saws]

    def _count_enemies(self) -> int:
        return 0  # saws are hazards, not enemies: no kill events in Meat Boy

    def qblock_count_near(self, r_tiles: int) -> int:
        return 0

    def _progress_fitness(self) -> float:
        if self.won:
            return self._FIT_SCALE + self.win_bonus + _win_time_bonus(self)
        return (1.0 - self._best_bfs) * self._FIT_SCALE

    def episode_stats(self) -> dict:
        return _episode_stats(self,(1.0 - self._best_bfs) * self._FIT_SCALE)

    def set_level(self, level: str) -> None:
        self.core._level_idx = int(level)
        self.core.won = False


class BombermanAdapter(_EventWeights):
    """Wraps BombermanCore (top-down). jump = drop a bomb; move_y is the second axis.
    Fitness is cost-to-exit progress (Dijkstra, bricks cost extra) so bombing the right brick
    counts as progress, plus kills (the exit only opens on a clear arena)."""

    _FIT_SCALE = 1000.0
    BRICK_BONUS = 40.0       # exploration: bricks opened
    RETREAT_BONUS = 60.0     # surviving your own first bomb at all — the drop-and-retreat lesson,
                             # paid once so it cannot be farmed by littering the arena
    SAFE_BOMB_BONUS = 60.0   # per bomb that then achieved something: opened a brick, or landed on
    SAFE_BOMB_CAP = 5        # an enemy. An empty blast is litter and earns nothing.
    AIM_BONUS = 120.0        # how close a blast came to an enemy: the gradient from bombing to killing
    KILL_BONUS = 150.0       # flat, per kill — the share below shrinks per enemy, so a crowded level
                             # would otherwise pay less for each kill exactly where killing is harder
    SENSOR_LABELS = [  # dashboard telemetry layout for the 16-slot ray vector below
        {"g": "Rays (4 directions)", "color": "bg-zinc-300", "rows": [
            {"i": i, "l": n, "d": f"Ray {n}: distance to the nearest wall, brick or bomb", "inv": True}
            for i, n in enumerate(("N", "E", "S", "W"))]},
        {"g": "Blast danger", "color": "bg-red-500", "rows": [
            *({"i": 4 + i, "l": f"!{n}", "d": f"Step {n}: how soon that tile burns (1 = burning now, 0 = safe)",
               "inv": False} for i, n in enumerate(("N", "E", "S", "W"))),
            {"i": 8, "l": "BOOM", "d": "This tile is in a blast line: 1 = burning now, rises as the fuse runs down",
             "inv": False}]},
        {"g": "Enemies", "color": "bg-orange-400", "rows": [
            {"i": 9, "l": "NMY", "d": "Nearest living enemy, straight-line distance", "inv": True},
            {"i": 10, "l": "NX", "d": "Nearest enemy, horizontal bearing (left negative, right positive)",
             "center": True, "inv": False},
            {"i": 11, "l": "NY", "d": "Nearest enemy, vertical bearing (up negative, down positive)",
             "center": True, "inv": False}]},
        {"g": "Bombing", "color": "bg-amber-400", "rows": [
            {"i": 12, "l": "BMB", "d": "Bombs available to drop", "inv": False},
            {"i": 13, "l": "BRK", "d": "Bricks a bomb dropped here would destroy (of 4 arms)", "inv": False}]},
        {"g": "Exit", "color": "bg-sky-400", "rows": [
            {"i": 14, "l": "EX", "d": "Exit direction, horizontal (left negative, right positive)", "center": True, "inv": False},
            {"i": 15, "l": "EY", "d": "Exit direction, vertical (up negative, down positive)", "center": True, "inv": False}]},
    ]
    _RAY4 = [(0, -1), (1, 0), (0, 1), (-1, 0)]   # N, E, S, W — the directions the agent can actually move

    def __init__(self, level: str | None, max_frames: int, win_bonus: float,
                 sprint: bool = False, time_rate: float = 0.0) -> None:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()
        from code.games.bomberman_core import BombermanCore
        self.win_bonus = win_bonus
        self.sprint = sprint
        self.time_rate = time_rate
        self.core = BombermanCore(render_mode="none", max_steps=max_frames, level_idx=int(level or 0))
        self.tile_size = int(self.core.tile_size)
        self.alive = True
        self.won = False
        self.status = "RUNNING"
        self._end_xy: tuple[float, float] | None = None
        self.reset()

    # body: the platformer vocabulary, mapped honestly
    @property
    def x(self) -> float:
        p = self.core.player
        return p.x + p.width / 2.0

    @property
    def y(self) -> float:
        p = self.core.player
        return p.y + p.height / 2.0

    @property
    def vx(self) -> float:
        return float(self.core.player.vx)

    @property
    def vy(self) -> float:
        return float(self.core.player.vy)

    @property
    def grounded(self) -> bool:
        return True

    @property
    def can_jump(self) -> bool:  # "can act": a bomb is available
        return len(self.core.bombs) < self.core.player.bombs_max

    @property
    def camera(self) -> tuple[float, float]:
        return 0.0, 0.0

    def reset(self) -> None:
        self.core.won = False  # reset() advances to the next level when won is left True
        self.core.reset()
        if self.sprint:
            self.core.player.speed = float(self.core.cfg["player"].get("sprint_speed", 168))
        self.alive = True
        self.won = False
        self.status = "RUNNING"
        self._end_xy = None
        self._level_coins = sum(row.count("C") + row.count("F") + row.count("S") for row in self.core.level_data.grid)
        self._level_enemies = self._count_enemies()
        self._start_cost = max(self.core.start_cost(), 1.0)
        self._best_cost = self._start_cost

    def step(self, move_x: int, jump: bool, move_y: int = 0) -> None:
        if not self.alive:
            return
        _, _, terminated, truncated, _info = self.core.step((move_x, move_y, int(jump)))
        self._best_cost = min(self._best_cost, self.core.goal_cost())
        if terminated or truncated:
            self.alive = False
            self._end_xy = (self.x, self.y)
            self.won = bool(self.core.won)
            self.status = "WON" if self.won else "DEAD"

    def render(self, surface: pygame.Surface) -> None:
        self.core.render(surface)

    def solid_at(self, wx: float, wy: float) -> bool:
        tx, ty = int(wx // self.tile_size), int(wy // self.tile_size)
        b = self.core.bomb_at(tx, ty)
        return self.core.solid(tx, ty) or (b is not None and not b.passable)  # a fresh own bomb isn't a wall

    def enemy_positions(self) -> list[tuple[float, float]]:
        return [(e.x + e.width / 2.0, e.y + e.height / 2.0) for e in self.core.enemies if e.alive]

    def qblock_count_near(self, r_tiles: int) -> int:
        """Bricks a bomb on the player's tile would reach (what bombing here buys)."""
        from code.games.bomberman_core import Bomb
        p = self.core.player
        tx, ty = self.core._center_tile(p.x, p.y, p.width, p.height)
        cells = self.core.blast_cells(Bomb(tx, ty, 0, p.blast_range))
        return sum(1 for (x, y) in cells if self.core.tile(x, y) in ("?", "C", "F", "S", "@"))

    def sense(self, march, ray_max, hit_solid, hit_enemy, hit_none, tile_hit, tile_probe):
        """16-slot ray vector (see SENSOR_LABELS) + overlay rays/tiles for the dashboard.

        The four danger slots are what makes retreat learnable: a bomb covers a cross, so
        "step to the neighbour that burns latest" walks out of it one tile at a time."""
        import numpy as np
        core, p = self.core, self.core.player
        ox, oy = self.x, self.y
        ts = float(self.tile_size)
        fuse = float(getattr(core, "fuse_frames", core.cfg["bomb"]["fuse_frames"]))
        tx, ty = core._center_tile(p.x, p.y, p.width, p.height)
        vec = np.empty(16, dtype=np.float32)
        rays, tiles = [], []

        def burn(cx: int, cy: int) -> float:
            ttb = core.time_to_boom(cx, cy)
            return 0.0 if ttb is None else max(0.0, 1.0 - ttb / fuse)

        for i, (dx, dy) in enumerate(self._RAY4):
            d = march(self, ox, oy, dx, dy)
            vec[i] = d / ray_max
            vec[4 + i] = burn(tx + dx, ty + dy)
            hit = d < ray_max
            rays.append((ox, oy, ox + dx * d, oy + dy * d, hit_solid if hit else hit_none))
            if hit:
                tiles.append(((ox + dx * d) // ts * ts, (oy + dy * d) // ts * ts, ts, tile_hit))
        vec[8] = burn(tx, ty)
        for (cx, cy) in core.danger_cells():
            tiles.append((cx * ts, cy * ts, ts, tile_probe))
        best, bx, by = ray_max, None, None
        for ex, ey in self.enemy_positions():
            d = ((ex - ox) ** 2 + (ey - oy) ** 2) ** 0.5
            if d < best:
                best, bx, by = d, ex, ey
        vec[9] = best / ray_max
        vec[10] = 0.0 if bx is None else max(-1.0, min(1.0, (bx - ox) / ray_max))
        vec[11] = 0.0 if by is None else max(-1.0, min(1.0, (by - oy) / ray_max))
        if bx is not None:
            rays.append((ox, oy, bx, by, hit_enemy))
        vec[12] = (p.bombs_max - len(core.bombs)) / max(p.bombs_max, 1)
        vec[13] = min(self.qblock_count_near(p.blast_range), 4) / 4.0
        ex, ey = core.level_data.exit
        vec[14] = max(-1.0, min(1.0, ((ex + 0.5) * ts - ox) / core.level_data.width))
        vec[15] = max(-1.0, min(1.0, ((ey + 0.5) * ts - oy) / core.level_data.height))
        return vec, rays, tiles

    @property
    def busy(self) -> bool:
        """Standing still with a bomb ticking is the game, not a stall — the trainer's
        stuck rule comes from the platformers, where waiting is always wasted time."""
        return bool(self.core.bombs or self.core.blasts)

    @property
    def reach(self) -> float:
        """What the progress bar measures: the same 0..1000 exit-cost scale as episode_stats."""
        return self.progress() * self._FIT_SCALE

    def progress(self) -> float:
        return max(0.0, min(1.0, 1.0 - self._best_cost / self._start_cost))

    def _progress_fitness(self) -> float:
        """Two halves when the exit is gated on a clear arena: cost-to-exit progress and share of
        enemies killed — so a kill always beats camping the exit. Where the exit is not gated,
        killing is optional, so progress carries the full scale and kills only add their bonus.
        Small bonuses for bricks opened and bombs survived keep early, sparse behaviour learnable."""
        core = self.core
        bonus = (self.BRICK_BONUS * core.bricks_destroyed
                 + self.RETREAT_BONUS * min(core.safe_detonations, 1)
                 + self.SAFE_BOMB_BONUS * min(core.useful_detonations, self.SAFE_BOMB_CAP))
        if self.won:
            return self._FIT_SCALE + bonus + self.win_bonus + _win_time_bonus(self)
        n = len(core.enemies)
        kill_bonus = self.KILL_BONUS * core.kills_total + self.AIM_BONUS * core.best_aim
        if n and getattr(core, "requires_clear", True):
            return (0.5 * self._FIT_SCALE * (self.progress() + core.kills_total / n)
                    + kill_bonus + bonus)
        return self.progress() * self._FIT_SCALE + (kill_bonus if n else 0.0) + bonus

    def episode_stats(self) -> dict:
        st = _episode_stats(self, self.progress() * self._FIT_SCALE)
        st["level_len"] = self._FIT_SCALE            # reach % = exit-cost progress, not pixels
        st["end_x"] = self.progress() * self._FIT_SCALE if self._end_xy else None
        st["bricks"] = self.core.bricks_destroyed
        st["level_coins"] = self._level_coins
        return st

    def set_level(self, level: str) -> None:
        self.core._level_idx = int(level)
        self.core.won = False


class MegamanAdapter(_EventWeights):
    """Wraps MegamanCore (Mega Man, NES). A stage climbs, doubles back and drops, so progress is
    route cost to the exit, not x: a cost field over the tile grid that knows a body only goes up
    by ladder or by jumping. The net has six outputs: left, right, jump, up, down, fire.

    Jump and fire are edge-triggered on the console, so a net that simply holds an output would
    act once and never again; here a held output repeats. How fast it repeats, and how fast a
    direction may change, is capped at human speed by `agent_input` in megaman_config.yaml."""

    _FIT_SCALE = 1000.0
    KILL_BONUS = 15.0        # per enemy destroyed, capped at the level's count: a nudge toward using the buster
    JUMP_CELLS = 3           # how many cells a jump lifts the feet
    DRIFT_CELLS = 3          # how many cells a jump carries sideways over nothing
    COMPASS_STEPS = 4        # how far down the route the compass looks
    LADDER_UP_CELLS = 4      # how far above the body a ladder still counts as "within a jump"
    LADDER_AHEAD_CELLS = 7   # how far ahead a ladder is noticed (about the reach of a ray)
    SENSOR_LABELS = [  # dashboard telemetry layout for the 26-slot vector built in sense()
        {"g": "Rays (relative to facing)", "color": "bg-zinc-300", "rows": [
            {"i": i, "l": n, "d": f"Ray {d}: distance to the nearest solid tile", "inv": True}
            for i, (n, d) in enumerate((("FWD", "forward"), ("UP30", "forward-up 30"), ("UP60", "forward-up 60"),
                                        ("DN30", "forward-down 30"), ("DN60", "forward-down 60"), ("BACK", "back")))]},
        {"g": "Threats", "color": "bg-red-500", "rows": [
            {"i": 6, "l": "NMY", "d": "Enemy in the corridor straight ahead: distance", "inv": True},
            {"i": 7, "l": "NX", "d": "Nearest enemy, horizontal bearing (behind negative, ahead positive)",
             "center": True, "inv": False},
            {"i": 8, "l": "NY", "d": "Nearest enemy, vertical bearing (up negative, down positive)",
             "center": True, "inv": False},
            {"i": 9, "l": "SHOT", "d": "Nearest enemy shot or blade: distance", "inv": True},
            {"i": 10, "l": "PIT", "d": "No floor (or spikes) below a point 1.5 tiles ahead", "inv": False}]},
        {"g": "Body", "color": "bg-sky-400", "rows": [
            {"i": 11, "l": "GND", "d": "On the ground", "inv": False},
            {"i": 12, "l": "VX", "d": "Horizontal velocity", "center": True, "inv": False},
            {"i": 13, "l": "VY", "d": "Vertical velocity", "center": True, "inv": False},
            {"i": 14, "l": "JUMP", "d": "Jump available (on the ground or on a ladder)", "inv": False},
            {"i": 21, "l": "FACE", "d": "Facing (left negative, right positive)", "center": True, "inv": False},
            {"i": 22, "l": "HP", "d": "Energy left", "inv": False}]},
        {"g": "Ladders", "color": "bg-amber-400", "rows": [
            {"i": 15, "l": "CLMB", "d": "Holding a ladder", "inv": False},
            {"i": 16, "l": "LAD^", "d": "A ladder here: up grabs it", "inv": False},
            {"i": 17, "l": "LADv", "d": "A ladder under the feet: down climbs onto it", "inv": False},
            {"i": 23, "l": "LAD+", "d": "A ladder overhead within a jump (1 = just above, fades with height): "
                                        "jump and press up", "inv": False},
            {"i": 24, "l": "LAD>", "d": "The nearest ladder ahead at body height or within a jump of it "
                                        "(1 = next tile, fades with distance)", "inv": False}]},
        {"g": "Route", "color": "bg-emerald-400", "rows": [
            {"i": 18, "l": "GX", "d": "Which way the route to the exit goes next, horizontal (left negative, right positive)",
             "center": True, "inv": False},
            {"i": 19, "l": "GY", "d": "Which way the route goes next, vertical (up negative, down positive)",
             "center": True, "inv": False},
            {"i": 20, "l": "BOSS", "d": "Boss energy left (0 when there is no boss on screen)", "inv": False}]},
    ]

    def __init__(self, level: str | None, max_frames: int, win_bonus: float,
                 sprint: bool = False, time_rate: float = 0.0) -> None:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()
        from code.games.megaman_core import MegamanCore
        self.win_bonus = win_bonus
        self.time_rate = time_rate      # sprint has no meaning here: Mega Man has one running speed
        self.core = MegamanCore(render_mode="none", max_steps=max_frames, level=level)
        self.tile_size = int(self.core.tile_size)
        self.alive = True
        self.won = False
        self.status = "RUNNING"
        self._end_xy: tuple[float, float] | None = None
        self._fields: dict = {}
        pad = self.core.cfg.get("agent_input") or {}
        self.fire_period = max(1, int(pad.get("fire_period", 8)))
        self.jump_period = max(1, int(pad.get("jump_period", 8)))
        self.turn_period = max(1, int(pad.get("turn_period", 6)))
        tr = self.core.cfg.get("training") or {}
        self.attempts = max(1, int(os.environ.get("PEAK_MM1_ATTEMPTS") or tr.get("attempts", 1)))
        self.min_start_energy = int(tr.get("min_start_energy", 10))
        self.attempt_stall = int(tr.get("stall_frames", 240))
        self.sticky = float(tr.get("sticky_actions", 0.0))
        self._episode = 0               # how many times this adapter has been reset: varies the seeds
        self.reset()

    @property
    def x(self) -> float:
        p = self.core.player
        return p.x + p.width / 2.0

    @property
    def y(self) -> float:
        p = self.core.player
        return p.y + p.height / 2.0

    @property
    def vx(self) -> float:
        return float(self.core.player.vx)

    @property
    def vy(self) -> float:
        return float(self.core.player.vy)

    @property
    def grounded(self) -> bool:
        return bool(self.core.player.grounded)

    @property
    def can_jump(self) -> bool:
        return bool(self.core.player.grounded or self.core.player.climbing)

    @property
    def camera(self) -> tuple[float, float]:
        return self.core.camera

    def reset(self) -> None:
        """A new episode: on a stage section, several attempts in a row (see `training` in
        megaman_config.yaml); anywhere else, one."""
        self.alive = True
        self.won = False
        self.status = "RUNNING"
        self._end_xy = None
        self._episode += 1
        self._done: list[tuple[float, bool, float]] = []     # (fitness, won, progress) per finished attempt
        self._begin_attempt()

    def _begin_attempt(self) -> None:
        import random
        k = len(self._done)
        self.core.won = False  # reset() advances to the next level when won is left True
        section = self.core.level_entry().get("section") or {}
        self._n_attempts = self.attempts if section else 1
        if k:                                                 # later attempts: another seed ...
            self.core._seed = 1000 * self._episode + k
        self.core.reset()
        first = list(section.get("start") or []) == list(self.core.level_data.start)
        if k and not first:                                   # ... and whatever energy was left
            rng = random.Random(7919 * self._episode + k)
            lo, hi = min(self.min_start_energy, self.core.player.hp_max), self.core.player.hp_max
            # the second attempt always arrives worn down (the lower half of the range), as most
            # arrivals from the section before do; the rest draw from the whole range
            self.core.player.hp = rng.randint(lo, (lo + hi) // 2 if k == 1 else hi)
            self.core.player.facing = rng.choice((-1, 1))     # ... facing whichever way it arrived
        self.jitter(1299709 * self._episode + k if k else None)   # ... and a slightly unsteady hand
        self._attempt_best, self._attempt_idle = -1.0, 0
        self._jump_held = False
        self._since_fire = self._since_jump = 10 ** 6        # frames since the last press
        self._held = [0, 0]                                   # the direction held on each axis ...
        self._held_for = [10 ** 6, 10 ** 6]                   # ... and for how many frames
        from code.games import megaman_enemies as en
        spawns = self.core.level_data.spawns
        section = self.core.level_entry().get("section") or {}
        self._has_boss = any(g == en.BOSS_GLYPH for g, _, _ in spawns) and not section.get("goal_screens")
        owned = {tuple(sc) for sc in section.get("screens") or []}      # a section counts only its own enemies
        sc_, sr_ = self.core.screen_cols, self.core.screen_rows
        self._level_enemies = sum(1 for g, c, r in spawns if g in en.ENEMY_GLYPHS and g != "c"
                                  and (not owned or (c // sc_, r // sr_) in owned)) + int(self._has_boss)
        self._level_coins = 0
        key = (self.core.level_file(), os.path.getmtime(self.core.level_file()))
        if key not in self._fields:
            self._fields = {key: self._route_field()}
        self._cost = self._fields[key]
        # A section ends where the next one starts: its progress runs between those two costs.
        end = section.get("end")
        self._end_cost = float(self._cost[end[1], end[0]]) if end else 0.0
        if self._end_cost == float("inf"):
            self._end_cost = 0.0
        self._start_cost = max(self._cell_cost(), self._end_cost + 1.0)
        self._best_cost = self._start_cost

    def jitter(self, seed: int | None) -> None:
        """Make this attempt differ from the last. Most of the stage is not random, so the same
        network replays the same attempt frame for frame; with a seed, each frame has a chance
        (`sticky_actions`) of repeating the previous frame's buttons instead of taking new ones,
        the "sticky actions" used to test agents on deterministic games. None switches it off."""
        import random
        self._sticky_rng = random.Random(seed) if seed is not None and self.sticky > 0 else None
        self._last_action = None

    def step(self, move_x: int, jump: bool, move_y: int = 0, fire: bool = False) -> None:
        if not self.alive:
            return
        if self._sticky_rng is not None:
            if self._last_action is not None and self._sticky_rng.random() < self.sticky:
                move_x, jump, move_y, fire = self._last_action
            self._last_action = (move_x, jump, move_y, fire)
        # The pad, at human speed. Directions: a change waits until the last one was held long enough.
        want = [int(move_x), int(move_y)]
        for axis in (0, 1):
            if want[axis] != self._held[axis] and self._held_for[axis] >= self.turn_period:
                self._held[axis], self._held_for[axis] = want[axis], 0
            self._held_for[axis] += 1
        move_x, move_y = self._held
        # Buttons: a held output repeats, but each new press has to wait its turn. Jump stays down
        # while airborne (that is how a jump is held for height); a new one starts with a fresh press.
        self._since_fire += 1
        self._since_jump += 1
        fire_now = bool(fire) and self._since_fire >= self.fire_period
        if fire_now:
            self._since_fire = 0
        if not jump:
            jump_now = False
        elif self._jump_held and not self.can_jump:
            jump_now = True                                   # still rising on the same press
        else:
            jump_now = self._since_jump >= self.jump_period and not self._jump_held
            if jump_now:
                self._since_jump = 0
        self._jump_held = jump_now
        _, _, terminated, truncated, _info = self.core.step((move_x, move_y, int(jump_now), int(fire_now)))
        self._best_cost = min(self._best_cost, self._cell_cost())
        stalled = False
        if self._n_attempts > 1 and not (terminated or truncated):   # the adapter ends a stalled attempt itself
            fit = self._attempt_fitness(False)
            if fit > self._attempt_best + 1.0 or self.busy:
                self._attempt_best, self._attempt_idle = max(fit, self._attempt_best), 0
            else:
                self._attempt_idle += 1
                stalled = self._attempt_idle >= self.attempt_stall
        if terminated or truncated or stalled:
            won = bool(self.core.won)
            self._done.append((self._attempt_fitness(won), won, 1.0 if won else self._attempt_progress()))
            if len(self._done) < self._n_attempts:
                self._begin_attempt()
                return
            self.alive = False
            self._end_xy = (self.x, self.y)
            self.won = all(w for _, w, _ in self._done)       # a winner wins every attempt
            self.status = "WON" if self.won else "STUCK" if stalled else "DEAD"

    def render(self, surface: pygame.Surface) -> None:
        self.core.render(surface)
        self.core._hud(surface)         # the energy meter: the core only draws it for a person at the keyboard

    def solid_at(self, wx: float, wy: float) -> bool:
        return self.core.solid(int(wx // self.tile_size), int(wy // self.tile_size))

    def _threats(self) -> list[tuple[float, float]]:
        s = self.core.scale
        return [(e.x * s, e.y * s) for e in self.core.enemies if e.visible and e.hurts]

    def enemy_positions(self) -> list[tuple[float, float]]:
        return self._threats()

    def qblock_count_near(self, r_tiles: int) -> int:
        return 0

    # ── the route: how many cells of travel from each cell to the exit (megaman_route.py) ──
    def _route_field(self):
        from code.games.megaman_route import route_field
        return route_field(self.core.level_data, self.JUMP_CELLS, self.DRIFT_CELLS)

    def _cell(self) -> tuple[int, int]:
        return int(self.x // self.tile_size), int(self.y // self.tile_size)

    def _cell_cost(self) -> float:
        """Route cost at the player's exact position (continuous within a cell)."""
        from code.games.megaman_route import cost_at
        return cost_at(self._cost, self.x, self.y, float(self.tile_size))

    def _compass(self) -> tuple[float, float]:
        """Unit-ish vector toward where the route goes next: follow the cost field downhill."""
        c0, r0 = c, r = self._cell()
        rows, cols = self._cost.shape
        if not (0 <= r < rows and 0 <= c < cols) or self._cost[r, c] == float("inf"):
            return 0.0, 0.0
        for _ in range(self.COMPASS_STEPS):
            best = min(((self._cost[r + dr, c + dc], c + dc, r + dr)
                        for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                        if (dr or dc) and 0 <= r + dr < rows and 0 <= c + dc < cols), default=None)
            if best is None or best[0] >= self._cost[r, c]:
                break
            _, c, r = best
        n = float(self.COMPASS_STEPS)
        return (c - c0) / n, (r - r0) / n

    def sense(self, march, ray_max, hit_solid, hit_enemy, hit_none, tile_hit, tile_probe):
        """26-slot vector (see SENSOR_LABELS) + overlay rays/tiles for the dashboard."""
        import math

        import numpy as np
        from code.neuro.sensors import RAY_DIRS
        core, p = self.core, self.core.player
        ox, oy = self.x, self.y
        ts, s = float(self.tile_size), core.scale
        face = float(p.facing)
        vec = np.zeros(26, dtype=np.float32)
        rays, tiles = [], []
        for i, (dx, dy) in enumerate(RAY_DIRS):
            dx *= face
            d = march(self, ox, oy, dx, dy)
            vec[i] = d / ray_max
            hit = d < ray_max
            rays.append((ox, oy, ox + dx * d, oy + dy * d, hit_solid if hit else hit_none))
            if hit:
                tiles.append(((ox + dx * d) // ts * ts, (oy + dy * d) // ts * ts, ts, tile_hit))

        ahead, near, nx, ny = ray_max, ray_max, None, None
        for ex, ey in self._threats():
            fwd = (ex - ox) * face
            if 0 <= fwd < ahead and abs(ey - oy) <= ts:
                ahead = fwd
            d = math.hypot(ex - ox, ey - oy)
            if d < near:
                near, nx, ny = d, ex, ey
        vec[6] = ahead / ray_max
        if nx is not None:
            vec[7] = max(-1.0, min(1.0, (nx - ox) * face / ray_max))
            vec[8] = max(-1.0, min(1.0, (ny - oy) / ray_max))
            rays.append((ox, oy, nx, ny, hit_enemy))
        shots = [(b.x * s, b.y * s) for b in core.enemy_bullets]
        if core.boss_cutter:
            shots.append((core.boss_cutter.x * s, core.boss_cutter.y * s))
        vec[9] = min((math.hypot(bx - ox, by - oy) for bx, by in shots), default=ray_max) / ray_max if shots else 1.0
        vec[9] = min(float(vec[9]), 1.0)

        probe_x, feet = ox + face * 1.5 * ts, p.y + p.height
        pit = 1.0
        for k in range(4):
            t = core.tile_at(probe_x, feet + (k + 0.5) * ts)
            tiles.append((probe_x // ts * ts, (feet + k * ts) // ts * ts, ts, tile_probe))
            if t == "^":
                break
            if t in ("#", "H", "d"):
                pit = 0.0
                break
        vec[10] = pit
        vec[11] = 1.0 if p.grounded else 0.0
        vec[12] = max(-1.0, min(1.0, p.vx / (3.0 * s)))
        vec[13] = max(-1.0, min(1.0, p.vy / (12.0 * s)))
        vec[14] = 1.0 if self.can_jump else 0.0
        vec[15] = 1.0 if p.climbing else 0.0
        vec[16] = 1.0 if core.tile_at(ox, oy) == "H" else 0.0
        vec[17] = 1.0 if core.tile_at(ox, feet + 1) == "H" else 0.0
        gx, gy = self._compass()
        # In screen terms, like the left/right outputs. (Relative to facing, a net had to combine
        # this with FACE to know which button to hold, and it did not turn round when the route did.)
        vec[18], vec[19] = gx, gy
        from code.games import megaman_enemies as en
        vec[20] = core.boss.hp / en.BOSS_HP if core.boss else 0.0
        vec[21] = face
        vec[22] = p.hp / p.hp_max
        # Ladders are not solid, so no ray stops on one: these two see a ladder before it is touched.
        c, r = self._cell()
        for k in range(1, self.LADDER_UP_CELLS + 1):          # overhead, in this column
            t = core.tile(c, r - k)
            if t == "H":
                vec[23] = 1.0 - (k - 1) / self.LADDER_UP_CELLS
                tiles.append((c * ts, (r - k) * ts, ts, tile_probe))
                break
            if t in ("#", "d", " "):
                break
        step = int(face)
        for k in range(1, self.LADDER_AHEAD_CELLS + 1):       # ahead, from two tiles below to a jump above
            col = c + step * k
            hit = next((rr for rr in range(r + 2, r - self.LADDER_UP_CELLS - 1, -1) if core.tile(col, rr) == "H"), None)
            if hit is not None:
                vec[24] = 1.0 - (k - 1) / self.LADDER_AHEAD_CELLS
                tiles.append((col * ts, hit * ts, ts, tile_probe))
                break
        vec[25] = 1.0
        return vec, rays, tiles

    @property
    def busy(self) -> bool:
        """The boss-room intro and a screen flip take the pad away: not a stall."""
        return self.core.boss_intro > 0 or self.core.camera_y != self.core._camera_target()[1]

    @property
    def reach(self) -> float:
        return self.progress() * self._FIT_SCALE

    def _attempt_progress(self) -> float:
        """0..1 along the route; with a boss at the end, half is the route and half his energy."""
        route = max(0.0, min(1.0, (self._start_cost - self._best_cost) / (self._start_cost - self._end_cost)))
        if not self._has_boss:
            return route
        from code.games import megaman_enemies as en
        return 0.5 * route + 0.5 * min(self.core.boss_damage / en.BOSS_HP, 1.0)

    def _attempt_fitness(self, won: bool) -> float:
        kills = self.KILL_BONUS * min(self.core.kills_total, self._level_enemies)
        if won:
            return self._FIT_SCALE + kills + self.win_bonus + _win_time_bonus(self)
        return self._attempt_progress() * self._FIT_SCALE + kills

    @property
    def attempt_wins(self) -> int:
        return sum(1 for _, w, _ in self._done if w)

    def progress(self) -> float:
        """Mean progress over the episode's attempts (the one in play counts as far as it has got)."""
        live = [self._attempt_progress()] if self.alive else []
        return (sum(p for _, _, p in self._done) + sum(live)) / self._n_attempts

    def _progress_fitness(self) -> float:
        live = self._attempt_fitness(False) if self.alive else 0.0
        return (sum(f for f, _, _ in self._done) + live) / self._n_attempts

    def episode_stats(self) -> dict:
        st = _episode_stats(self, self.progress() * self._FIT_SCALE)
        st["level_len"] = self._FIT_SCALE            # reach % = route progress, not pixels
        st["energy"], st["energy_max"] = int(self.core.player.hp), int(self.core.player.hp_max)   # shown on the dashboard
        st["end_x"] = self.progress() * self._FIT_SCALE if self._end_xy else None
        st["level_coins"] = self._level_coins
        return st

    def set_level(self, level: str) -> None:
        self.core._level_idx = self.core.level_index(level)
        self.core.won = False


_ADAPTERS = {
    "mario": MarioAdapter,
    "sonic": SonicAdapter,
    "meatboy": MeatboyAdapter,
    "bomberman": BombermanAdapter,
    "megaman": MegamanAdapter,
}


def active_games() -> list[str]:
    """The games menus and sweeps offer by default — every adapter except the archived ones."""
    return [g for g in _ADAPTERS if g not in ARCHIVED_GAMES]


def make_adapter(game: str, level: str | None, max_frames: int, win_bonus: float,
                 sprint: bool = False, time_rate: float = 0.0,
                 weights: dict[str, float] | None = None) -> GameAdapter:
    try:
        cls = _ADAPTERS[game]
    except KeyError:
        raise ValueError(f"unknown game '{game}' (available: {', '.join(_ADAPTERS)})") from None
    adapter = cls(level, max_frames, win_bonus, sprint, time_rate)
    adapter.weights = dict(weights or {})
    return adapter
