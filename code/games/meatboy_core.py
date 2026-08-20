from __future__ import annotations
import os
from collections import deque
import numpy as np
import pygame
import yaml
from gymnasium import spaces

from .modules.Objects.GameObject import GameObject
from .modules.Objects.MeatboyPlayer import MeatboyPlayer
from .modules.System.LevelLoader import LevelLoader
from .modules.System.MeatboyPhysicsManager import MeatboyPhysicsManager, MeatboyContext
from .modules.Parameters.Map_parameters import (
    TILE_SIZE, TILE_GOAL, TILE_SPIKE, TILE_PIT, TILE_GROUND, TILE_PLATFORM, TILE_QBLOCK,
    TILE_AIR, TILE_CRUMBLE,
    COLOR_SKY, COLOR_GROUND, COLOR_SPIKE, COLOR_GOAL, COLOR_CRUMBLE,
)

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "meatboy_config.yaml")
_SOLID = {TILE_GROUND, TILE_PLATFORM, TILE_QBLOCK}   # crumble handled separately
_HAZARD = {TILE_SPIKE, TILE_PIT}
_CRUMBLE_DELAY = 0.5   # seconds after first touch before a crumble tile dissolves
_CORE_SCALARS = 11     # see _build_scalars layout


class MeatboyCore:
    """Super Meat Boy — monolithic character (the "old way", like PlatformerCore).

    A single MeatboyPlayer + a per-game MeatboyPhysicsManager, mirroring how
    Sonic/Megaman are structured. Mirrors the Gym contract GameEnv/train.py
    expect (reset/step/render + get_*_space)."""

    WIDTH = 672
    HEIGHT = 672

    def __init__(self, render_mode: str = "none", max_steps=None,
                 persona=None, arch_tag: str = "mlp", **kwargs):
        self.render_mode = render_mode
        with open(_CONFIG_PATH, "r") as f:
            self.cfg = yaml.safe_load(f)
        self.tile_size = int(self.cfg.get("tile_size", TILE_SIZE))
        phys = self.cfg.get("physics", {})
        self.ctx = MeatboyContext(
            gravity=float(phys.get("gravity", 1060.0)),
            fast_fall_grav=float(phys.get("fast_fall_grav", 1060.0)),
            max_fall_speed=float(phys.get("max_fall_speed", 1400.0)),
            tile_size=self.tile_size,
        )
        self._movement = dict(self.cfg.get("movement", {}))
        self._jump = dict(self.cfg.get("jump", {}))
        self._wall = dict(self.cfg.get("wall", {}))
        pdims = self.cfg.get("player", {})
        self._pw = int(pdims.get("width", 18))
        self._ph = int(pdims.get("height", 26))
        self.levels = list(self.cfg.get("levels", []))
        self.max_steps = int(max_steps) if max_steps else int(self.cfg.get("max_steps", 4000))
        self._human = render_mode in ("human", "random")

        # Fixed action space: [move_x(idle/left/right), run, jump]
        self.action_space = spaces.MultiDiscrete([3, 2, 2])
        self.window = 21
        self.n_scalars = _CORE_SCALARS + MeatboyPlayer.N_EXTRA
        self.observation_space = spaces.Dict({
            "grids": spaces.Box(low=-1.0, high=1.0, shape=(4, self.window, self.window),
                                dtype=np.float32),
            "scalars": spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_scalars,),
                                  dtype=np.float32),
        })

        self.physics = MeatboyPhysicsManager(tile_size=self.tile_size)
        self.loader = LevelLoader(tile_size=self.tile_size)

        self._level_idx = 0
        self._steps = 0
        self.score = 0
        self.lives = 1
        self.alive = True
        self.won = False
        self.player = None
        self.level_data = None
        self._dist = None
        self._dist_max = 1.0
        self._bfs_prev = None        # shaping anchor; owned by the core (see reset)
        self.reset()

    # ---- level handling ----
    def _load_level(self):
        path = self.levels[self._level_idx % len(self.levels)]
        self.current_level = path
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
        self._prepare_dist()

    def _spawn_player(self):
        sx, sy = self.level_data.player_start
        gObj = GameObject(float(sx), float(sy), self._pw, self._ph)
        self.player = MeatboyPlayer(gObj, self._movement, self._jump, self._wall,
                                    human_mode=self._human)

    # ---- Gym API ----
    def reset(self, *, seed=None, options=None):
        # Beating a level advances to the next (wrapping); dying/timeout replays.
        if self.won:
            self._level_idx = (self._level_idx + 1) % len(self.levels)
        self._steps = 0
        self.score = 0
        self.alive = True
        self.won = False
        self.death_cause = ""
        self._crumble_timers = {}
        # Clear the shaping anchor at the episode boundary. The shared
        # _ScoreTracker only resets on `terminated`, NOT on truncation, so
        # relying on it leaked a stale distance across timed-out episodes and
        # produced a large spurious potential spike on step 1 of the next one.
        # The core owns the anchor instead — it is the only thing that knows
        # exactly when an episode begins.
        self._bfs_prev = None
        self._load_level()
        self._spawn_player()
        obs, info = self._obs(), self._info()
        if info["bfs_dist"] >= 0.0:
            self._bfs_prev = info["bfs_dist"]
        return obs, info

    def step(self, action):
        self._steps += 1
        self.player.handle_input(action)
        self.player.control(1 / 60.0, self.ctx)          # velocity only
        self.physics.step(self.player, self.level_data.grid, self.ctx, 1 / 60.0)
        for saw in self.level_data.saws:
            saw.update(1 / 60.0)
        self._update_crumble(1 / 60.0)

        terminated = False
        truncated = False
        cause = self._lethal_overlap()
        if cause:
            self.alive = False
            self.death_cause = cause
            terminated = True
        elif self._touching_goal():
            self.won = True
            self.score += 100
            terminated = True
        elif self.player.y > self.level_data.height + 64:
            self.alive = False
            self.death_cause = "Pit"
            terminated = True
        if self._steps >= self.max_steps:
            truncated = True

        obs, info = self._obs(), self._info()
        if info["bfs_dist"] >= 0.0:        # advance anchor only on valid cells
            self._bfs_prev = info["bfs_dist"]
        return obs, 0.0, terminated, truncated, info

    # ---- hazards / crumble ----
    def _update_crumble(self, dt):
        """Crumble tiles dissolve _CRUMBLE_DELAY seconds after first contact."""
        ts = self.tile_size
        grid = self.level_data.grid
        touch = self.player.get_rect().inflate(4, 4)
        for col in range(max(0, touch.left // ts), touch.right // ts + 1):
            for row in range(max(0, touch.top // ts), touch.bottom // ts + 1):
                if (0 <= row < self.level_data.rows and 0 <= col < self.level_data.cols
                        and grid[row][col] == TILE_CRUMBLE):
                    self._crumble_timers.setdefault((row, col), _CRUMBLE_DELAY)
        for (row, col) in list(self._crumble_timers):
            self._crumble_timers[(row, col)] -= dt
            if self._crumble_timers[(row, col)] <= 0:
                if grid[row][col] == TILE_CRUMBLE:
                    grid[row][col] = TILE_AIR
                del self._crumble_timers[(row, col)]

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
        prect = self.player.get_rect()
        for saw in self.level_data.saws:
            if saw.hits_rect(prect):
                return "Saw"
        for (row, col) in self._player_cells():
            t = self.level_data.grid[row][col]
            if t in (TILE_SPIKE, TILE_PIT):
                tr = pygame.Rect(col * self.tile_size, row * self.tile_size,
                                 self.tile_size, self.tile_size)
                if prect.colliderect(tr):
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

    # ---- observation ----
    def _prepare_dist(self):
        """BFS distance-to-goal over non-solid cells (4-connected), once per level.
        Crumble tiles are passable for the gradient (they render solid but dissolve)."""
        grid = self.level_data.grid
        rows, cols = self.level_data.rows, self.level_data.cols
        dist = np.full((rows, cols), -1.0, dtype=np.float32)
        q = deque()
        for (gx, gy) in self._goal_tiles:
            if 0 <= gy < rows and 0 <= gx < cols:
                dist[gy, gx] = 0.0
                q.append((gx, gy))
        while q:
            x, y = q.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < cols and 0 <= ny < rows and dist[ny, nx] < 0:
                    if grid[ny][nx] in _SOLID:
                        continue
                    dist[ny, nx] = dist[y, x] + 1.0
                    q.append((nx, ny))
        self._dist = dist
        self._dist_max = max(1.0, float(dist.max()))

    def _saw_cells(self):
        ts = self.tile_size
        cells = []
        for saw in self.level_data.saws:
            cells.extend(saw.covered_cells(ts, self.level_data.rows, self.level_data.cols))
        return cells

    def _build_grids(self, px, py, hazard_cells):
        W = self.window
        half = W // 2
        grid = self.level_data.grid
        rows, cols = self.level_data.rows, self.level_data.cols
        out = np.zeros((4, W, W), dtype=np.float32)
        for j in range(W):
            gy = py - half + j
            if not (0 <= gy < rows):
                out[0, j, :] = 1.0                 # OOB = solid wall
                continue
            for i in range(W):
                gx = px - half + i
                if not (0 <= gx < cols):
                    out[0, j, i] = 1.0
                    continue
                t = grid[gy][gx]
                if t in _SOLID or t == TILE_CRUMBLE:
                    out[0, j, i] = 1.0
                if t == TILE_GOAL:
                    out[1, j, i] = 1.0
                if t in _HAZARD:
                    out[2, j, i] = -1.0
                d = self._dist[gy, gx]
                out[3, j, i] = (1.0 - d / self._dist_max) if d >= 0 else 0.0
        for (cx, cy) in hazard_cells:
            i = cx - (px - half)
            j = cy - (py - half)
            if 0 <= i < W and 0 <= j < W:
                out[2, j, i] = -1.0
        return out

    def _build_scalars(self):
        ts = float(self.tile_size)
        lw, lh = self.level_data.width, self.level_data.height
        gx, gy = self._goal_xy
        p = self.player
        px, py = p.x, p.y
        core = [
            px / ts, py / ts,
            float(np.clip(p.vx / 360.0, -1.0, 1.0)),
            float(np.clip(p.vy / 1200.0, -1.0, 1.0)),
            1.0 if p.on_ground else 0.0,
            1.0 if p.facing_right else 0.0,
            1.0 if p.contact_left else 0.0,
            1.0 if p.contact_right else 0.0,
            1.0 if p.contact_ceiling else 0.0,
            float(np.clip((gx - px) / max(lw, 1.0), -1.0, 1.0)),
            float(np.clip((gy - py) / max(lh, 1.0), -1.0, 1.0)),
        ]
        core.extend(p.obs_extra())
        return np.asarray(core, dtype=np.float32)

    def _obs(self):
        ts = self.tile_size
        px = int((self.player.x + self.player.width / 2) // ts)
        py = int((self.player.y + self.player.height / 2) // ts)
        grids = self._build_grids(px, py, self._saw_cells())
        return {"grids": grids, "scalars": self._build_scalars()}

    def _bfs_norm_dist(self):
        """Player-cell BFS distance to goal, normalised to [0,1] (0 = at goal,
        1 = farthest reachable). Returns -1.0 when the cell is unreachable or
        off-grid — the sentinel the shared tracker treats as "no information"
        (mirrors the platformer dijkstra_dist convention). This is a path
        distance (correct through walls), unlike euclidean goal_dist."""
        if self._dist is None:
            return -1.0
        ts = self.tile_size
        px = int((self.player.x + self.player.width / 2) // ts)
        py = int((self.player.y + self.player.height / 2) // ts)
        if not (0 <= py < self.level_data.rows and 0 <= px < self.level_data.cols):
            return -1.0
        raw = float(self._dist[py, px])
        if raw < 0:
            return -1.0
        return raw / self._dist_max

    def _info(self):
        bfs = self._bfs_norm_dist()
        return {
            "score": self.score,
            "lives": 1 if self.alive else 0,
            "won": self.won,
            "terminated": (not self.alive) or self.won,
            "goal_dist": self._goal_dist(),
            "bfs_dist": bfs,
            # Core-owned shaping anchor (-1.0 = none yet / episode start).
            # Reset in reset(), so it can never leak across a truncation.
            "bfs_dist_prev": (self._bfs_prev if self._bfs_prev is not None else -1.0),
            # meatboy has no Dijkstra solver; expose the BFS path distance under
            # this key so it flows through the shared tracker's potential-shaping
            # plumbing (_ScoreTracker computes dijkstra_dist_prev / dijkstra_valid).
            "dijkstra_dist": bfs,
            "x_position": float(self.player.x),
            "velocity_x": float(self.player.vx),
            "velocity_y": float(self.player.vy),
        }

    def action_to_str(self, action):
        return str(list(np.asarray(action).reshape(-1)))

    def get_action_space(self):
        return self.action_space

    def get_observation_space(self):
        return self.observation_space

    # ---- render ----
    def _camera(self, view_w, view_h):
        cx = self.player.x + self.player.width / 2 - view_w / 2
        cy = self.player.y + self.player.height / 2 - view_h / 2
        cx = max(0.0, min(cx, max(0.0, self.level_data.width - view_w)))
        cy = max(0.0, min(cy, max(0.0, self.level_data.height - view_h)))
        return cx, cy

    def render(self, surface=None, blit_only=False):
        if surface is None:
            return
        ts = self.tile_size
        vw, vh = surface.get_width(), surface.get_height()
        camx, camy = self._camera(vw, vh)
        surface.fill(COLOR_SKY)
        c0 = max(0, int(camx // ts)); c1 = min(self.level_data.cols, int((camx + vw) // ts) + 1)
        r0 = max(0, int(camy // ts)); r1 = min(self.level_data.rows, int((camy + vh) // ts) + 1)
        for r in range(r0, r1):
            for c in range(c0, c1):
                t = self.level_data.grid[r][c]
                col = None
                if t in _SOLID: col = COLOR_GROUND
                elif t == TILE_CRUMBLE: col = COLOR_CRUMBLE
                elif t == TILE_SPIKE: col = COLOR_SPIKE
                elif t == TILE_GOAL: col = COLOR_GOAL
                if col:
                    pygame.draw.rect(surface, col, (c * ts - camx, r * ts - camy, ts, ts))
        for saw in self.level_data.saws:
            saw.render(surface, saw.cx - camx, saw.cy - camy)
        pygame.draw.rect(surface, (210, 40, 60),
                         (int(self.player.x - camx), int(self.player.y - camy),
                          self.player.width, self.player.height))


if __name__ == "__main__":
    # Smoke test: load level 1, run a scripted episode headless, assert the
    # revert integrates, lands, and never crashes.
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    core = MeatboyCore(render_mode="none")
    obs, info = core.reset()
    assert set(obs.keys()) == {"grids", "scalars"}
    assert obs["grids"].shape == (4, 21, 21)
    assert obs["scalars"].shape == (core.n_scalars,)

    landed = False
    for t in range(600):
        act = [2, 1, 1] if t % 40 < 30 else [2, 1, 0]   # run right, tap-jump
        obs, r, term, trunc, info = core.step(act)
        if core.player.on_ground and t > 5:
            landed = True
        if term or trunc:
            obs, info = core.reset()
    assert landed, "player never touched ground in 600 steps"
    print(f"MeatboyCore smoke test OK — obs scalars={core.n_scalars}, "
          f"action={core.action_space.nvec.tolist()}")
