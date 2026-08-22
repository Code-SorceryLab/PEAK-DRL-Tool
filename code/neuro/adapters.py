"""Thin game adapters exposing the minimal surface the neuroevolution loop needs.

Each adapter wraps one game-core instance. The trainer never touches gym APIs,
observations, or rewards — only this protocol.
"""
from __future__ import annotations

import os
from typing import Protocol

import pygame


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
N_OUTPUTS_BY_GAME = {"bomberman": 5}
N_INPUTS_BY_GAME = {"bomberman": 16}   # ray-mode games with their own sense(): see the adapter's SENSOR_LABELS


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
        return [str(i) for i in range(len(data.get("levels", [])))]
    with open(os.path.join(games_dir, "game_config.yaml"), encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    sections = {"mario": data, "megaman": data.get("megaman", {}), "sonic": data.get("sonic", {})}
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
    ld = getattr(core, "level_data", None)  # megaman builds level_data on first reset
    return len(ld.coins) if ld is not None else 0


def _episode_stats(adapter, x: float) -> dict:
    """Game metrics shared by every core, read defensively (not every core has every field)."""
    core = adapter.core
    return {
        "x": round(float(x), 1),
        "score": int(getattr(core, "score", 0) or 0),
        "coins": int(getattr(core, "coins_total", 0) or 0),
        "kills": int(getattr(core, "kills_total", 0) or 0),
        "time_left": round(float(getattr(core, "timer", 0.0) or 0.0), 1),
        "cause": str(getattr(core, "death_cause", "") or getattr(core, "last_cause", "") or ""),
        # balance-metric fields (Amr's table): where the episode ended, level extents, loot pool
        "end_x": round(float(adapter._end_xy[0]), 1) if adapter._end_xy else None,
        "end_y": round(float(adapter._end_xy[1]), 1) if adapter._end_xy else None,
        "level_len": round(float(getattr(getattr(core, "level_data", None), "width", 0.0) or 0.0), 1),
        "level_coins": adapter._level_coins,
    }


class MarioAdapter:
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

    def step(self, move_x: int, jump: bool, move_y: int = 0) -> None:
        if not self.alive:
            return
        _, _, terminated, truncated, _ = self.core.step([_move_md(move_x, self.sprint), int(jump), 0])
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
        return out

    def qblock_count_near(self, r_tiles: int) -> int:
        r = r_tiles * self.tile_size
        px, py = self.x, self.y
        n = 0
        for q in self.core.level_data.qblocks:
            g = q.gObj
            if abs(g.x + g.width / 2.0 - px) <= r and abs(g.y + g.height / 2.0 - py) <= r:
                n += 1
        return n

    def fitness(self) -> float:
        return float(self.core.max_x_seen) + ((self.win_bonus + _win_time_bonus(self)) if self.won else 0.0)

    def episode_stats(self) -> dict:
        return _episode_stats(self,self.core.max_x_seen)

    def set_level(self, level: str) -> None:
        _set_locked_level(self.core, level)


class MegamanAdapter:
    """Wraps MegamanCore. Horizontal progress fitness; enemies shoot but the net can't (fire=0)."""

    _MOVE_MD = {-1: 1, 0: 0, 1: 3}

    def __init__(self, level: str | None, max_frames: int, win_bonus: float,
                 sprint: bool = False, time_rate: float = 0.0) -> None:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        from code.games.megaman_core import MegamanCore
        from code.games.modules.Parameters.Map_parameters import (
            TILE_GROUND, TILE_PLATFORM, TILE_QBLOCK, TILE_SIZE,
        )
        self._solid_codes = {TILE_GROUND, TILE_PLATFORM, TILE_QBLOCK}
        self.tile_size = TILE_SIZE
        self.win_bonus = win_bonus
        self.sprint = sprint        # persona capability: run/sprint action variants
        self.time_rate = time_rate  # persona objective: fitness per second left on a win
        level_kw = {"world": level} if level else {}
        self.core = MegamanCore(
            render_mode="none",
            **level_kw,
            lock_level=True,
            curriculum_enabled=False,
            max_steps=max_frames,
        )
        self.alive = True
        self.won = False
        self.status = "RUNNING"
        self._end_xy: tuple[float, float] | None = None
        self._level_coins = _count_coins(self.core)

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
        return bool(p.on_ground and not p.on_ladder)

    @property
    def camera(self) -> tuple[float, float]:
        return float(self.core.camera_x), float(self.core.camera_y)

    def reset(self) -> None:
        self.core.reset()
        self.alive = True
        self.won = False
        self.status = "RUNNING"
        self._end_xy: tuple[float, float] | None = None
        self._level_coins = _count_coins(self.core)

    def step(self, move_x: int, jump: bool, move_y: int = 0) -> None:
        if not self.alive:
            return
        _, _, terminated, truncated, _ = self.core.step([_move_md(move_x, self.sprint), 0, int(jump), 0])
        if terminated or truncated:
            self.alive = False
            self._end_xy = (self.x, self.y)
            self.won = bool(self.core.reached_goal)
            self.status = "WON" if self.won else ("STUCK" if truncated else "DEAD")

    def render(self, surface: pygame.Surface) -> None:
        self.core.render(surface, blit_only=True)

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
        for e in self.core.enemies:
            g = e.gObj
            if g.active:
                out.append((g.x + g.width / 2.0, g.y + g.height / 2.0))
        return out

    def qblock_count_near(self, r_tiles: int) -> int:
        return 0

    def fitness(self) -> float:
        return float(self.core.max_x_seen) + ((self.win_bonus + _win_time_bonus(self)) if self.won else 0.0)

    def episode_stats(self) -> dict:
        return _episode_stats(self,self.core.max_x_seen)

    def set_level(self, level: str) -> None:
        _set_locked_level(self.core, level)


class SonicAdapter:
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

    def step(self, move_x: int, jump: bool, move_y: int = 0) -> None:
        if not self.alive:
            return
        _, _, terminated, truncated, info = self.core.step([_move_md(move_x, self.sprint), int(jump), 0])
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

    def fitness(self) -> float:
        return float(self.core.max_x_seen) + ((self.win_bonus + _win_time_bonus(self)) if self.won else 0.0)

    def episode_stats(self) -> dict:
        return _episode_stats(self,self.core.max_x_seen)

    def set_level(self, level: str) -> None:
        _set_locked_level(self.core, level)


class MeatboyAdapter:
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

    def qblock_count_near(self, r_tiles: int) -> int:
        return 0

    def fitness(self) -> float:
        if self.won:
            return self._FIT_SCALE + self.win_bonus + _win_time_bonus(self)
        return (1.0 - self._best_bfs) * self._FIT_SCALE

    def episode_stats(self) -> dict:
        return _episode_stats(self,(1.0 - self._best_bfs) * self._FIT_SCALE)

    def set_level(self, level: str) -> None:
        self.core._level_idx = int(level)
        self.core.won = False


class BombermanAdapter:
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
        fuse = float(core.cfg["bomb"]["fuse_frames"])
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

    def fitness(self) -> float:
        """Two halves when the level has enemies (the exit only opens on a clear arena): cost-to-exit
        progress and share of enemies killed — so a kill always beats camping the exit. Small
        bonuses for bricks opened and bombs survived keep early, sparse behaviour learnable."""
        core = self.core
        bonus = (self.BRICK_BONUS * core.bricks_destroyed
                 + self.RETREAT_BONUS * min(core.safe_detonations, 1)
                 + self.SAFE_BOMB_BONUS * min(core.useful_detonations, self.SAFE_BOMB_CAP))
        if self.won:
            return self._FIT_SCALE + bonus + self.win_bonus + _win_time_bonus(self)
        n = len(core.enemies)
        if n:
            return (0.5 * self._FIT_SCALE * (self.progress() + core.kills_total / n)
                    + self.KILL_BONUS * core.kills_total + self.AIM_BONUS * core.best_aim + bonus)
        return self.progress() * self._FIT_SCALE + bonus

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


_ADAPTERS = {
    "mario": MarioAdapter,
    "megaman": MegamanAdapter,
    "sonic": SonicAdapter,
    "meatboy": MeatboyAdapter,
    "bomberman": BombermanAdapter,
}


def make_adapter(game: str, level: str | None, max_frames: int, win_bonus: float,
                 sprint: bool = False, time_rate: float = 0.0) -> GameAdapter:
    try:
        cls = _ADAPTERS[game]
    except KeyError:
        raise ValueError(f"unknown game '{game}' (available: {', '.join(_ADAPTERS)})") from None
    return cls(level, max_frames, win_bonus, sprint, time_rate)
