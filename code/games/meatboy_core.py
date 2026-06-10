from __future__ import annotations
import os
import numpy as np
import pygame
import yaml
from gymnasium import spaces

from .modules.Objects.GameObject import GameObject
from .modules.System.LevelLoader import LevelLoader
from .modules.System.ModularPhysicsManager import ModularPhysicsManager
from .modules.Actor.MotorState import MotorState, MotorContext, Intent
from .modules.Actor.build import build_actor_from_config
from .modules.Actor.obs import ObsBuilder
from .modules.Parameters.Map_parameters import (
    TILE_SIZE, TILE_GOAL, TILE_SPIKE, TILE_PIT, TILE_GROUND, TILE_PLATFORM, TILE_QBLOCK,
    COLOR_SKY, COLOR_GROUND, COLOR_SPIKE, COLOR_GOAL,
)

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "meatboy_config.yaml")
_SOLID = {TILE_GROUND, TILE_PLATFORM, TILE_QBLOCK}


class MeatboyCore:
    """Host core for the modular Super Meat Boy character. Mirrors the Gym
    contract that GameEnv/train.py expect (reset/step/render + get_*_space)."""

    WIDTH = 672
    HEIGHT = 672

    def __init__(self, render_mode: str = "none", max_steps=None,
                 persona=None, arch_tag: str = "mlp", **kwargs):
        self.render_mode = render_mode
        with open(_CONFIG_PATH, "r") as f:
            self.cfg = yaml.safe_load(f)
        self.tile_size = int(self.cfg.get("tile_size", TILE_SIZE))
        phys = self.cfg.get("physics", {})
        self.ctx = MotorContext(
            gravity=float(phys.get("gravity", 2600.0)),
            fast_fall_grav=float(phys.get("fast_fall_grav", 2600.0)),
            max_fall_speed=float(phys.get("max_fall_speed", 1200.0)),
            tile_size=self.tile_size,
        )
        pdims = self.cfg.get("player", {})
        self._pw = int(pdims.get("width", 18))
        self._ph = int(pdims.get("height", 26))
        self.levels = list(self.cfg.get("levels", []))
        self.max_steps = int(max_steps) if max_steps else int(self.cfg.get("max_steps", 4000))

        human = render_mode in ("human", "random")
        self.state = MotorState()
        self.controller = build_actor_from_config(self.cfg, self.state, self.ctx, human_mode=human)

        self.action_space = self.controller.brain.action_space
        n_ability = len(self.controller.obs_field_names())
        self.obs_builder = ObsBuilder(window=21, n_ability_scalars=n_ability, tile_size=self.tile_size)
        self.observation_space = spaces.Dict({
            "grids": spaces.Box(low=-1.0, high=1.0, shape=(4, 21, 21), dtype=np.float32),
            "scalars": spaces.Box(low=-np.inf, high=np.inf,
                                  shape=(self.obs_builder.n_scalars,), dtype=np.float32),
        })

        self.physics = ModularPhysicsManager(tile_size=self.tile_size)
        self.loader = LevelLoader(tile_size=self.tile_size)

        self._level_idx = 0
        self._steps = 0
        self.score = 0
        self.lives = 1
        self.alive = True
        self.won = False
        self.player = None
        self.level_data = None
        self._screen = None
        self.reset()

    # ---- level handling ----
    def _load_level(self):
        path = self.levels[self._level_idx % len(self.levels)]
        self.level_data = self.loader.load_level(path)
        self._goal_tiles = []
        for r in range(self.level_data.rows):
            for c in range(self.level_data.cols):
                if self.level_data.grid[r][c] == TILE_GOAL:
                    self._goal_tiles.append((c, r))
        if self._goal_tiles:
            gx, gy = self._goal_tiles[0]
            self._goal_xy = (gx * self.tile_size + self.tile_size / 2,
                             gy * self.tile_size + self.tile_size / 2)
        else:
            self._goal_xy = (self.level_data.width, self.level_data.height / 2)
        self.obs_builder.prepare(self.level_data.grid, self._goal_tiles)

    def _spawn_player(self):
        sx, sy = self.level_data.player_start
        self.player = GameObject(float(sx), float(sy), self._pw, self._ph)
        self.state.vx = 0.0
        self.state.vy = 0.0
        self.state.on_ground = False
        self.state.facing_right = True
        self.state.gravity_scale = 1.0
        self.state.contact_left = False
        self.state.contact_right = False
        self.state.contact_ceiling = False
        self.state.air_lockout = 0
        self.state.intents = Intent()
        self.state.ext.clear()

    # ---- Gym API ----
    def reset(self, *, seed=None, options=None):
        self._steps = 0
        self.score = 0
        self.alive = True
        self.won = False
        self._load_level()
        self._spawn_player()
        return self._obs(), self._info()

    def step(self, action):
        self._steps += 1
        self.controller.brain.set_action(action)
        self.controller.update(1 / 60.0)             # brain + abilities set velocity
        self.physics.step(self.state, self.player, self.level_data.grid, self.ctx, 1 / 60.0)

        terminated = False
        truncated = False
        # hazards / pit / goal via tile overlap
        cause = self._lethal_overlap()
        if cause:
            self.alive = False
            terminated = True
        elif self._touching_goal():
            self.won = True
            self.score += 100
            terminated = True
        elif self.player.y > self.level_data.height + 64:
            self.alive = False
            terminated = True
        if self._steps >= self.max_steps:
            truncated = True

        return self._obs(), 0.0, terminated, truncated, self._info()

    # ---- helpers ----
    def _player_cells(self):
        ts = self.tile_size
        r = self.player.get_rect()
        cells = []
        for col in range(r.left // ts, r.right // ts + 1):
            for row in range(r.top // ts, r.bottom // ts + 1):
                if 0 <= row < self.level_data.rows and 0 <= col < self.level_data.cols:
                    cells.append((row, col))
        return cells

    def _lethal_overlap(self):
        for (row, col) in self._player_cells():
            t = self.level_data.grid[row][col]
            if t in (TILE_SPIKE, TILE_PIT):
                tr = pygame.Rect(col * self.tile_size, row * self.tile_size,
                                 self.tile_size, self.tile_size)
                if self.player.get_rect().colliderect(tr):
                    return "Spike" if t == TILE_SPIKE else "Pit"
        return None

    def _touching_goal(self):
        for (row, col) in self._player_cells():
            if self.level_data.grid[row][col] == TILE_GOAL:
                return True
        return False

    def _goal_dist(self):
        cx = self.player.x + self.player.width / 2
        cy = self.player.y + self.player.height / 2
        gx, gy = self._goal_xy
        return float(np.hypot(gx - cx, gy - cy))

    def _obs(self):
        ts = self.tile_size
        px = int((self.player.x + self.player.width / 2) // ts)
        py = int((self.player.y + self.player.height / 2) // ts)
        grids = self.obs_builder.build_grids(self.level_data.grid, (px, py), self._goal_tiles)
        scal = self.obs_builder.build_scalars(
            self.state, self.controller,
            player_xy=(self.player.x, self.player.y),
            goal_xy=self._goal_xy,
            level_wh=(self.level_data.width, self.level_data.height),
        )
        return {"grids": grids, "scalars": scal}

    def _info(self):
        return {
            "score": self.score,
            "lives": 1 if self.alive else 0,
            "won": self.won,
            "terminated": (not self.alive) or self.won,
            "goal_dist": self._goal_dist(),
            "x_position": float(self.player.x),
            "velocity_x": float(self.state.vx),
            "velocity_y": float(self.state.vy),
        }

    def action_to_str(self, action):
        return str(list(np.asarray(action).reshape(-1)))

    def get_action_space(self):
        return self.action_space

    def get_observation_space(self):
        return self.observation_space

    # ---- render ----
    def render(self, surface=None, blit_only=False):
        if surface is None:
            return
        ts = self.tile_size
        surface.fill(COLOR_SKY)
        for r in range(self.level_data.rows):
            for c in range(self.level_data.cols):
                t = self.level_data.grid[r][c]
                col = None
                if t in _SOLID: col = COLOR_GROUND
                elif t == TILE_SPIKE: col = COLOR_SPIKE
                elif t == TILE_GOAL: col = COLOR_GOAL
                if col:
                    pygame.draw.rect(surface, col, (c * ts, r * ts, ts, ts))
        pygame.draw.rect(surface, (210, 40, 60),
                         (int(self.player.x), int(self.player.y), self.player.width, self.player.height))
