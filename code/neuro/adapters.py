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


def make_adapter(game: str, level: str | None, max_frames: int, win_bonus: float) -> GameAdapter:
    if game == "mario":
        return MarioAdapter(level, max_frames, win_bonus)
    raise ValueError(f"unknown game '{game}' (adapters implemented: mario)")
