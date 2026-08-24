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
    # The game's action space. The first three are the locomotion core every game
    # shares and map onto net outputs 0-2; anything after that is a game-specific
    # button delivered to step() as `extras`, in this order. Keeping the count
    # per-game means adding a button to one game does not change any other game's
    # weight vector, so existing runs and saved models stay comparable.
    BUTTONS: tuple[str, ...]

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
    def step(self, move_x: int, jump: bool, extras: tuple[bool, ...] = ()) -> None: ...
    def render(self, surface: pygame.Surface) -> None: ...
    def solid_at(self, wx: float, wy: float) -> bool: ...
    def enemy_positions(self) -> list[tuple[float, float]]: ...
    def qblock_count_near(self, r_tiles: int) -> int: ...
    def goal_frame(self) -> tuple[float, float, float, float]: ...
    max_frames: int
    frames_used: int
    def fitness(self) -> float: ...
    def episode_stats(self) -> dict: ...
    def set_level(self, level: str) -> None: ...


def _set_locked_level(core, level: str) -> None:
    """Repoint a gym-style core at a new level; applied by its next reset()."""
    core.locked_level = str(level)
    core.world = str(level)


def list_levels(game: str, include_disabled: bool = False) -> list[str]:
    """Enabled level ids for a game, in config order. Meatboy levels are indices."""
    import yaml
    games_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "games")
    if game == "meatboy":
        with open(os.path.join(games_dir, "meatboy_config.yaml"), encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return [str(i) for i in range(len(data.get("levels", [])))]
    with open(os.path.join(games_dir, "game_config.yaml"), encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    section = {"mario": data, "megaman": data.get("megaman", {}), "sonic": data.get("sonic", {})}[game]
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
    """Speedrunner fitness: the share of win_bonus earned by finishing early.

    Scored on unused *frames*, not on core.timer. Meatboy has no timer, so the old
    clock-based version silently paid 0 there and the speedrunner persona was inert
    on the very game the balance probes study. Every core takes the same frame
    budget, so frames work everywhere.

    Bounded by win_bonus: time_rate is a fraction in [0, 1], so the bonus can never
    outweigh the win itself. The old rate was an unbounded fitness-per-second and
    paid 7500 on mario against a win_bonus of 5000.
    """
    budget = max(int(getattr(adapter, "max_frames", 0)), 1)
    saved = max(budget - int(getattr(adapter, "frames_used", 0)), 0) / budget
    return adapter.time_rate * adapter.win_bonus * saved


def _goal_frame_from_goals(core, gx: float | None = None, gy: float | None = None
                          ) -> tuple[float, float, float, float]:
    """(agent_x, agent_y, goal_x, goal_y), each divided by the level size so every
    value sits in [0, 1] regardless of level dimensions.

    Absolute, NOT facing-relative: the first network layer is linear, so from these
    four it can form the goal offset (gx - px, gy - py) on its own, and it also gets
    "where am I in this level", which a per-level GA can use as a lookup key.
    Games without a goal object fall back to the right edge at mid-height, which is
    the implicit goal of the max_x fitness those games use.
    """
    ld = core.level_data
    lw = max(float(getattr(ld, "width", 0.0)), 1.0)
    lh = max(float(getattr(ld, "height", 0.0)), 1.0)
    if gx is None:
        goals = getattr(ld, "goals", None) or []
        if goals:
            gx, gy = float(goals[0].gObj.x), float(goals[0].gObj.y)
        else:
            gx, gy = lw, lh * 0.5
    px, py = _agent_xy(core)
    clip = lambda v: float(min(max(v, 0.0), 1.0))
    return clip(px / lw), clip(py / lh), clip(gx / lw), clip(gy / lh)


def _agent_xy(core) -> tuple[float, float]:
    p = core.player
    g = getattr(p, "gObj", None)
    return (float(g.x), float(g.y)) if g is not None else (float(p.x), float(p.y))


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
    BUTTONS = ("left", "right", "jump")
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
        self.max_frames = max_frames  # persona objective: the budget time_rate is scored against
        self.frames_used = 0          # steps taken this episode (reset in reset())
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
        self.frames_used = 0
        self.core.reset()
        self.core.lives = 1
        self.alive = True
        self.won = False
        self.status = "RUNNING"
        self._end_xy: tuple[float, float] | None = None
        self._level_coins = _count_coins(self.core)

    def step(self, move_x: int, jump: bool, extras: tuple[bool, ...] = ()) -> None:
        if not self.alive:
            return
        self.frames_used += 1
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

    def goal_frame(self) -> tuple[float, float, float, float]:
        return _goal_frame_from_goals(self.core)

    def fitness(self) -> float:
        return float(self.core.max_x_seen) + ((self.win_bonus + _win_time_bonus(self)) if self.won else 0.0)

    def episode_stats(self) -> dict:
        return _episode_stats(self,self.core.max_x_seen)

    def set_level(self, level: str) -> None:
        _set_locked_level(self.core, level)


class MegamanAdapter:
    """Wraps MegamanCore. Horizontal progress fitness; enemies shoot but the net can't (fire=0)."""

    BUTTONS = ("left", "right", "jump")
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
        self.max_frames = max_frames  # persona objective: the budget time_rate is scored against
        self.frames_used = 0          # steps taken this episode (reset in reset())
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
        self.frames_used = 0
        self.core.reset()
        self.alive = True
        self.won = False
        self.status = "RUNNING"
        self._end_xy: tuple[float, float] | None = None
        self._level_coins = _count_coins(self.core)

    def step(self, move_x: int, jump: bool, extras: tuple[bool, ...] = ()) -> None:
        if not self.alive:
            return
        self.frames_used += 1
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

    def goal_frame(self) -> tuple[float, float, float, float]:
        return _goal_frame_from_goals(self.core)

    def fitness(self) -> float:
        return float(self.core.max_x_seen) + ((self.win_bonus + _win_time_bonus(self)) if self.won else 0.0)

    def episode_stats(self) -> dict:
        return _episode_stats(self,self.core.max_x_seen)

    def set_level(self, level: str) -> None:
        _set_locked_level(self.core, level)


class SonicAdapter:
    """Wraps SonicCore. Goal does NOT terminate the core's episode — we end it on info['won']."""

    BUTTONS = ("left", "right", "jump")
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
        self.max_frames = max_frames  # persona objective: the budget time_rate is scored against
        self.frames_used = 0          # steps taken this episode (reset in reset())
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
        self.frames_used = 0
        self.core.reset()
        self.core.lives = 1
        self.alive = True
        self.won = False
        self.status = "RUNNING"
        self._end_xy: tuple[float, float] | None = None
        self._level_coins = _count_coins(self.core)

    def step(self, move_x: int, jump: bool, extras: tuple[bool, ...] = ()) -> None:
        if not self.alive:
            return
        self.frames_used += 1
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

    def goal_frame(self) -> tuple[float, float, float, float]:
        return _goal_frame_from_goals(self.core)

    def fitness(self) -> float:
        return float(self.core.max_x_seen) + ((self.win_bonus + _win_time_bonus(self)) if self.won else 0.0)

    def episode_stats(self) -> dict:
        return _episode_stats(self,self.core.max_x_seen)

    def set_level(self, level: str) -> None:
        _set_locked_level(self.core, level)


class MeatboyAdapter:
    """Wraps MeatboyCore (plain class, no gym). Levels are 2-D mazes, so fitness is
    BFS-distance-to-goal progress scaled to ~pixels (0..1000) instead of max_x."""

    BUTTONS = ("left", "right", "jump")
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
        self.max_frames = max_frames  # persona objective: the budget time_rate is scored against
        self.frames_used = 0          # steps taken this episode (reset in reset())
        self.sprint = sprint        # persona capability: run/sprint action variants
        self.time_rate = time_rate  # persona objective: fitness per second left on a win
        self.core = MeatboyCore(render_mode="none", max_steps=max_frames)
        # PEAK_MEATBOY_LEVELS points the core at a different level set (e.g. the
        # newMeat rebuild of World 1) without touching meatboy_config.yaml.
        _alt = os.environ.get("PEAK_MEATBOY_LEVELS")
        if _alt:
            import yaml as _yaml
            _p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "games", _alt)
            with open(_p, encoding="utf-8") as _f:
                self.core.levels = (_yaml.safe_load(_f) or {}).get("levels", self.core.levels)
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
        self.frames_used = 0
        self.core.won = False  # reset() advances to the next level when won is left True
        self.core.reset()
        self.alive = True
        self.won = False
        self.status = "RUNNING"
        self._end_xy: tuple[float, float] | None = None
        self._level_coins = _count_coins(self.core)
        self._best_bfs = 1.0

    def step(self, move_x: int, jump: bool, extras: tuple[bool, ...] = ()) -> None:
        if not self.alive:
            return
        self.frames_used += 1
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

    def goal_frame(self) -> tuple[float, float, float, float]:
        gx, gy = self.core._goal_xy   # world px, set at level load
        return _goal_frame_from_goals(self.core, gx, gy)

    def fitness(self) -> float:
        if self.won:
            return self._FIT_SCALE + self.win_bonus + _win_time_bonus(self)
        return (1.0 - self._best_bfs) * self._FIT_SCALE

    def episode_stats(self) -> dict:
        return _episode_stats(self,(1.0 - self._best_bfs) * self._FIT_SCALE)

    def set_level(self, level: str) -> None:
        self.core._level_idx = int(level)
        self.core.won = False


_ADAPTERS = {
    "mario": MarioAdapter,
    "megaman": MegamanAdapter,
    "sonic": SonicAdapter,
    "meatboy": MeatboyAdapter,
}


def buttons(game: str) -> tuple[str, ...]:
    """The game's action space, e.g. ("left", "right", "jump")."""
    try:
        return tuple(_ADAPTERS[game].BUTTONS)
    except KeyError:
        raise ValueError(f"unknown game '{game}' (available: {', '.join(_ADAPTERS)})") from None


def n_buttons(game: str) -> int:
    """How many output units the net needs for this game. The trainer sizes NeuralNet
    with it, so a game that gains a button only changes its own weight count."""
    return len(buttons(game))


def make_adapter(game: str, level: str | None, max_frames: int, win_bonus: float,
                 sprint: bool = False, time_rate: float = 0.0) -> GameAdapter:
    try:
        cls = _ADAPTERS[game]
    except KeyError:
        raise ValueError(f"unknown game '{game}' (available: {', '.join(_ADAPTERS)})") from None
    return cls(level, max_frames, win_bonus, sprint, time_rate)
