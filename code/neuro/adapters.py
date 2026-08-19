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
    def step(self, move_x: int, jump: bool) -> None: ...
    def render(self, surface: pygame.Surface) -> None: ...
    def solid_at(self, wx: float, wy: float) -> bool: ...
    def enemy_positions(self) -> list[tuple[float, float]]: ...
    def qblock_count_near(self, r_tiles: int) -> int: ...
    def fitness(self) -> float: ...


class MarioAdapter:
    """Wraps PlatformerCore, pinned to one level, one life, no curriculum, no gym obs."""

    # move_x -1/0/+1 -> MultiDiscrete move component (1=left, 0=idle, 3=right; walk speed only)
    _MOVE_MD = {-1: 1, 0: 0, 1: 3}

    def __init__(self, level: str | None, max_frames: int, win_bonus: float) -> None:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        from code.games.platformer_core import PlatformerCore
        from code.games.modules.Parameters.Map_parameters import (
            TILE_GROUND, TILE_PLATFORM, TILE_QBLOCK, TILE_CRUMBLE, TILE_SIZE,
        )
        self._solid_codes = {TILE_GROUND, TILE_PLATFORM, TILE_QBLOCK, TILE_CRUMBLE}
        self.tile_size = TILE_SIZE
        self.win_bonus = win_bonus
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
        )
        # Lives are hardcoded to 3; >1 makes _handle_death soft-reset mid-episode,
        # which restarts the level without ending the episode and corrupts fitness.
        self.core.max_lives = 1
        self.core.lives = 1
        self.alive = True
        self.won = False
        self.status = "RUNNING"

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

    def step(self, move_x: int, jump: bool) -> None:
        if not self.alive:
            return
        _, _, terminated, truncated, _ = self.core.step([self._MOVE_MD[move_x], int(jump), 0])
        if terminated or truncated:
            self.alive = False
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
        return float(self.core.max_x_seen) + (self.win_bonus if self.won else 0.0)


class MegamanAdapter:
    """Wraps MegamanCore. Horizontal progress fitness; enemies shoot but the net can't (fire=0)."""

    _MOVE_MD = {-1: 1, 0: 0, 1: 3}

    def __init__(self, level: str | None, max_frames: int, win_bonus: float) -> None:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        from code.games.megaman_core import MegamanCore
        from code.games.modules.Parameters.Map_parameters import (
            TILE_GROUND, TILE_PLATFORM, TILE_QBLOCK, TILE_SIZE,
        )
        self._solid_codes = {TILE_GROUND, TILE_PLATFORM, TILE_QBLOCK}
        self.tile_size = TILE_SIZE
        self.win_bonus = win_bonus
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

    def step(self, move_x: int, jump: bool) -> None:
        if not self.alive:
            return
        _, _, terminated, truncated, _ = self.core.step([self._MOVE_MD[move_x], 0, int(jump), 0])
        if terminated or truncated:
            self.alive = False
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
        return float(self.core.max_x_seen) + (self.win_bonus if self.won else 0.0)


class SonicAdapter:
    """Wraps SonicCore. Goal does NOT terminate the core's episode — we end it on info['won']."""

    _MOVE_MD = {-1: 1, 0: 0, 1: 3}

    def __init__(self, level: str | None, max_frames: int, win_bonus: float) -> None:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        from code.games.sonic_core import SonicCore
        from code.games.modules.Parameters.Map_parameters import (
            TILE_GROUND, TILE_PLATFORM, TILE_QBLOCK, TILE_SIZE,
        )
        self._solid_codes = {TILE_GROUND, TILE_PLATFORM, TILE_QBLOCK}
        self.tile_size = TILE_SIZE
        self.win_bonus = win_bonus
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

    def step(self, move_x: int, jump: bool) -> None:
        if not self.alive:
            return
        _, _, terminated, truncated, info = self.core.step([self._MOVE_MD[move_x], int(jump), 0])
        won = bool(info.get("won"))  # core.reached_goal is wiped by the mid-step level reload
        if won or terminated or truncated:
            self.alive = False
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
        return float(self.core.max_x_seen) + (self.win_bonus if self.won else 0.0)


class MeatboyAdapter:
    """Wraps MeatboyCore (plain class, no gym). Levels are 2-D mazes, so fitness is
    BFS-distance-to-goal progress scaled to ~pixels (0..1000) instead of max_x."""

    _MOVE_MD = {-1: 1, 0: 0, 1: 2}
    _FIT_SCALE = 1000.0

    def __init__(self, level: str | None, max_frames: int, win_bonus: float) -> None:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()  # MeatboyCore does not init pygame itself
        from code.games.meatboy_core import MeatboyCore
        from code.games.modules.Parameters.Map_parameters import (
            TILE_GROUND, TILE_PLATFORM, TILE_QBLOCK, TILE_CRUMBLE,
        )
        self._solid_codes = {TILE_GROUND, TILE_PLATFORM, TILE_QBLOCK, TILE_CRUMBLE}
        self.win_bonus = win_bonus
        self.core = MeatboyCore(render_mode="none", max_steps=max_frames)
        self.tile_size = int(self.core.tile_size)
        if level is not None:
            self.core._level_idx = int(level)  # meatboy levels are indexed, not named
        self.alive = True
        self.won = False
        self.status = "RUNNING"
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
        self._best_bfs = 1.0

    def step(self, move_x: int, jump: bool) -> None:
        if not self.alive:
            return
        _, _, terminated, truncated, info = self.core.step([self._MOVE_MD[move_x], 0, int(jump)])
        bfs = float(info.get("bfs_dist", -1.0))
        if 0.0 <= bfs < self._best_bfs:
            self._best_bfs = bfs
        if terminated or truncated:
            self.alive = False
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
            return self._FIT_SCALE + self.win_bonus
        return (1.0 - self._best_bfs) * self._FIT_SCALE


_ADAPTERS = {
    "mario": MarioAdapter,
    "megaman": MegamanAdapter,
    "sonic": SonicAdapter,
    "meatboy": MeatboyAdapter,
}


def make_adapter(game: str, level: str | None, max_frames: int, win_bonus: float) -> GameAdapter:
    try:
        cls = _ADAPTERS[game]
    except KeyError:
        raise ValueError(f"unknown game '{game}' (available: {', '.join(_ADAPTERS)})") from None
    return cls(level, max_frames, win_bonus)
