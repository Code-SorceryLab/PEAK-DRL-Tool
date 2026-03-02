from __future__ import annotations
import os
import math
import importlib
import heapq
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional, Set
import numpy as np
import pygame
import time
import gymnasium
from gymnasium import spaces
import random
import psutil

# FIX: Import the Class 'EntityType' explicitly so EntityType.GOAL works
from .modules.System.EntityType import EntityType

# Objects
from .modules.Objects.GameObject import GameObject
from .modules.Objects.Tile import Tile, create_tile
from .modules.Objects.Player import Player
from .modules.Objects.Enemy import Enemy
from .modules.Objects.Powerup import Powerup
from .modules.Objects.Coin import Coin
from .modules.Objects.QuestionBlock import QuestionBlock

# System
from .modules.System.LevelLoader import LevelLoader, LevelData
from .modules.System.PhysicsManager import PhysicsManager
from .modules.System.config_manager import ConfigManager
from .modules.System.debugging_mods.manager import DebugManager

# Parameters
from .modules.Parameters.Map_parameters import(TILE_AIR, TILE_GROUND, TILE_PLATFORM, TILE_GOAL, TILE_SPIKE, TILE_QBLOCK,
    COLOR_SKY, COLOR_GROUND, COLOR_PLATFORM, COLOR_GOAL, COLOR_SPIKE,
    COLOR_WHITE, COLOR_BLACK, COLOR_QBLOCK, COLOR_EMPTY, COLOR_ENEMY,
    COLOR_POWERUP_MUSH, COLOR_POWERUP_STAR, COLOR_COIN, COLOR_HITBOX,
    COLOR_SENSOR, COLOR_AGENT_PANEL, COLOR_STREAK, TILE_SIZE)

# =============================================================================
# Dijkstra Pathfinding Helper (Global Distance Map)
# =============================================================================
class DijkstraSolver:
    """
    Computes a 'heatmap' of distances from the Goal to every reachable tile.
    """
    def __init__(self, grid: List[List[int]], rows: int, cols: int):
        self.grid = grid
        self.rows = rows
        self.cols = cols
        # Initialize with Infinity
        self.dist_map = np.full((rows, cols), float('inf'), dtype=np.float32)

    def compute_map(self, goals: List[Tuple[int, int]], coins: Set[Tuple[int, int]] = None):
        """
        Physics-aware Dijkstra flood-fill from every goal tile.

        Cost model (tuned to ~64px tiles, 800px/s jump, 1200px/s² gravity):
          Horizontal movement   : 2.0   (free running)
          Downward movement     : 1.2   (gravity-assisted, cheap)
          Upward movement       : 3.5   (jump required, expensive)
          Upward with no ground : +3.0  (floating tile, nearly unreachable)
          On-platform tiles     : -0.6  (natural landing spots, strong discount)
          Near-platform tiles   : -0.25 / -0.1
          Spike tile            : +18.0
          Coin tile             : -0.8

        KEY FIX for vertical levels: upward steps cost 3.5 instead of 2.0.
        Without this penalty the solver routes straight up through open air at
        the same cost as running right, producing a gradient that leads the agent
        to stall mid-air chasing an impossible straight-up path. Heavier upward
        costs make paths naturally flow along platforms then jump.

        BUG FIX: ground-proximity checks were elif-chained so the 2-tile and
        3-tile bonuses were dead code (only fired at the bottom row of the map).
        Changed to independent if-checks so all three levels apply.
        """
        if coins is None:
            coins = set()

        MAX_JUMP_TILES = 6   # realistic max jump height: v0²/(2g)/TILE_SIZE ≈ 4.2, +hold ≈ 5.4
        SOLID = {TILE_GROUND, TILE_PLATFORM, TILE_QBLOCK}

        pq = []
        for gx, gy in goals:
            if 0 <= gx < self.cols and 0 <= gy < self.rows:
                self.dist_map[gy][gx] = 0.0
                heapq.heappush(pq, (0.0, gx, gy))

        # 8-way: (dx, dy) — dy<0 is UP in screen coordinates (y increases downward)
        directions = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1)]

        while pq:
            current_dist, cx, cy = heapq.heappop(pq)
            if current_dist > self.dist_map[cy][cx]:
                continue

            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < self.cols and 0 <= ny < self.rows):
                    continue

                tile = self.grid[ny][nx]
                if tile not in (TILE_AIR, TILE_GOAL, TILE_SPIKE):
                    continue   # wall — impassable

                # --- Directional base cost ---
                if dy < 0:          # moving UP — requires a jump
                    step_cost = 3.5
                elif dy > 0:        # moving DOWN — gravity-assisted
                    step_cost = 1.2
                else:               # horizontal
                    step_cost = 2.0

                if dx != 0 and dy != 0:
                    step_cost *= 1.1   # diagonal slightly more expensive

                # --- Spike penalty ---
                if tile == TILE_SPIKE:
                    step_cost += 18.0

                # --- Coin attraction ---
                if (nx, ny) in coins:
                    step_cost -= 0.8

                # --- Ground-proximity discount (independent if-checks, not elif) ---
                if ny + 1 < self.rows and self.grid[ny + 1][nx] in SOLID:
                    step_cost -= 0.6
                if ny + 2 < self.rows and self.grid[ny + 2][nx] in SOLID:
                    step_cost -= 0.25
                if ny + 3 < self.rows and self.grid[ny + 3][nx] in SOLID:
                    step_cost -= 0.1

                # --- Floating-tile penalty for upward steps ---
                # If moving up and there is no solid surface within jump height
                # below the target, the tile is likely only reachable mid-air
                # (e.g. inside a tall open shaft). Extra cost discourages routing
                # through tiles the agent can't easily reach or land on.
                if dy < 0:
                    has_ground_nearby = any(
                        ny + k < self.rows and self.grid[ny + k][nx] in SOLID
                        for k in range(1, MAX_JUMP_TILES + 1)
                    )
                    if not has_ground_nearby:
                        step_cost += 3.0

                step_cost = max(1.0, step_cost)
                new_dist  = current_dist + step_cost
                if new_dist < self.dist_map[ny][nx]:
                    self.dist_map[ny][nx] = new_dist
                    heapq.heappush(pq, (new_dist, nx, ny))

    def get_dist(self, x: int, y: int) -> float:
        if 0 <= x < self.cols and 0 <= y < self.rows:
            d = self.dist_map[y][x]
            # Return -1.0 if unreachable/infinite so the observation is clean
            return d if d != float('inf') else -1.0
        return -1.0

# =============================================================================
# Screen / Tile geometry
# =============================================================================
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
PLATFORMER_WIDTH, PLATFORMER_HEIGHT = 32, 32
DEBUG_PANEL_WIDTH = 350  # Width of the side debug panel (shown only in human mode)

# Action Map for Debug Display
ACTION_NAMES = {
    0: "IDLE", 1: "LEFT", 2: "RIGHT", 3: "JUMP",
    4: "RIGHT+JUMP", 5: "RUN+RIGHT", 6: "LEFT+JUMP", 7: "RUN+RIGHT+JUMP",
    8: "RUN+LEFT", 9: "RUN+LEFT+JUMP"
}

class PlatformerCore(gymnasium.Env):
    WIDTH, HEIGHT = SCREEN_WIDTH, SCREEN_HEIGHT

    @property
    def DEBUG_PANEL_X(self):
        """X offset where the debug panel begins."""
        return SCREEN_WIDTH if self.render_mode == "human" else 0

    @property
    def TOTAL_WIDTH(self):
        """Total window width including debug panel (human mode only)."""
        return SCREEN_WIDTH + DEBUG_PANEL_WIDTH if self.render_mode == "human" else SCREEN_WIDTH

    def __init__(self, render_mode: str = "none", **kwargs):
        self.render_mode = render_mode

        if self.render_mode == "human":
            pygame.init()
            pygame.display.set_caption("PEAK Platformer")
            self._surf = pygame.display.set_mode((SCREEN_WIDTH + DEBUG_PANEL_WIDTH, SCREEN_HEIGHT))
        else:
            # Headless / Training mode
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            pygame.init()
            self._surf = pygame.Surface((self.WIDTH, self.HEIGHT))

        # 1. Initialize Managers
        self.config_manager = ConfigManager("game_config.yaml")
        self.loader = LevelLoader()
        self.physics_manager = PhysicsManager()
        self.debug_manager = DebugManager(default_active=(render_mode=="human"), print_help=(render_mode=="human"))

        # 2. Config & State Containers
        self.level_data = LevelData() # Empty initial state
        # PERF: Initialise to None so load_level() knows to do full construction
        # on the first call, and reuse the object on subsequent soft resets.
        self.player: Player | None = None

        # Default world and speed multiplier
        self.level_order = self.config_manager.get_level_order()
        self.current_index_world = 0
        self.world = str(kwargs.pop("world", "1-1")).lower()
        self.speed_mult = float(kwargs.pop("speed_mult", 2.0))
        self.physics_manager.speed_mult = self.speed_mult

        self.max_steps = kwargs.pop("max_steps", None)

        self.persona = str(kwargs.pop("persona", "simple")).lower()
        if self.persona == "default":
            self.persona = "simple"
        # reward_fn is owned by generic_env (the wrapper), not the core game.
        # Kept as None here so the persona label is still accessible via self.persona.
        self.reward_fn = None
        self.ACTION_NAMES = ACTION_NAMES

        # Timer knobs
        self.use_timer = bool(kwargs.pop("use_timer", True))
        self.timer_seconds = int(kwargs.pop("timer_seconds", 400))
        self.timer_warn_threshold = int(kwargs.pop("timer_warn_threshold", 100))

        self.max_lives = 3
        self.lives = self.max_lives

        # Camera
        self.camera_x = 0.0
        self.camera_y = 0.0
        self.camera_smoothing = 0.15
        self.camera_lock = True

        # Anti-stall
        self.anti_stall = bool(kwargs.pop("anti_stall", True))
        self.stall_window = float(kwargs.pop("stall_window", 2))
        self.stall_kill_windows = int(kwargs.pop("stall_kill_windows", 10))

        # Observation sanity checker
        self._obs_check_interval = 5000   # Check every N steps
        self._obs_check_counter = 0
        self._obs_stats = {               # Latest stats, exposed via _info()
            "grid_player_mean": 0.0, "grid_player_std": 0.0,
            "grid_player_min": 0.0, "grid_player_max": 0.0,
            "grid_hazard_mean": 0.0, "grid_hazard_std": 0.0,
            "grid_hazard_min": 0.0, "grid_hazard_max": 0.0,
            "grid_collectible_mean": 0.0, "grid_collectible_std": 0.0,
            "grid_collectible_min": 0.0, "grid_collectible_max": 0.0,
            "grid_dijkstra_mean": 0.0, "grid_dijkstra_std": 0.0,
            "grid_dijkstra_min": 0.0, "grid_dijkstra_max": 0.0,
            "scalar_mean": 0.0, "scalar_std": 0.0,
            "scalar_min": 0.0, "scalar_max": 0.0,
            "dijkstra_val": 0.0, "obs_warnings": "",
        }

        self.reset_metrics()

        # --- GRID OBSERVATION SIZE ---
        self.obs_width = 11
        self.obs_height = 11
        self.obs_pad_x = self.obs_width // 2
        self.obs_pad_y = self.obs_height // 2

        # --- RAYCAST CONFIGURATION ---
        # Number of rays to cast around the player
        self.num_rays = int(kwargs.pop("num_rays", 48))
        self.ray_max_dist = 250.0
        # Create angles (0 to 2pi)
        self.ray_angles = np.linspace(0, 2 * math.pi, self.num_rays, endpoint=False)
        self.last_rays = [] # For debug drawing

        # NEW: Dijkstra Map Storage
        self.dijkstra = None
        self.dijkstra_current_tile= 0.0
        self._obs_space = spaces.Dict({
            # 5 Channels: Player, Solid, Hazard, Collectible, Dijkstra-Advantage
            # low=-1.0 because the Dijkstra channel is a relative advantage map
            # in [-1, 1] where positive = closer to goal than player's tile,
            # negative = further away. All other channels remain in [0, 1].
            "grids": spaces.Box(low=-1.0, high=1.0, shape=(4, self.obs_height, self.obs_width), dtype=np.float32),

            # Scalars: 18 (Player=5, Tracking=13)
            "scalars": spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32),

            # Raycasts: [dist, type, dist, type, ...] -> Size = num_rays * 2
            # "raycasts": spaces.Box(low=0.0, high=4.0, shape=(self.num_rays * 2,), dtype=np.float32)
        })

        # FIX: Action Space is 10 to match ACTION_NAMES (0 through 9)
        self._act_space = spaces.Discrete(10)

        self.ui_font = pygame.font.SysFont("arial", 20, bold=True)
        self.qblock_font = pygame.font.SysFont("arial", 26, bold=True)

        self.reset()

    def reset_metrics(self):
        """Helper to clear metrics on reset/death."""
        self.timer = self.timer_seconds
        self.time_last_step = time.time()
        self.dt = 0.0001
        self.score = 0; self.coins_total = 0
        self.alive = True; self.frame = 0
        self.game_over = False; self.reached_goal = False
        self.last_x = 0.0; self.last_score = 0; self.score_delta = 0
        self.kills_step = self.coins_step = self.powerups_step = 0
        self._last_action = 0
        self.max_x_seen = 0.0; self.stall_timer = 0
        self.stall_windows_count = 0; self.stalled_this_frame = False
        self.progress_x_best = 0.0; self.progress_y_best = 0.0
        self.death_cause = ""
        self.lives = self.max_lives  # restore lives on every full episode reset
        self.best_dist_to_goal = float('inf')  # stall tracker anchor
        self._needs_level_transition = False   # set True when goal reached mid-episode
        self._hash_dirty = True                # set True whenever an entity is removed
        # --- Velocity Alignment (cached from _tracking_obs each step) ---
        self._step_dx = 0.0   # unit vector toward cheapest reachable 8-direction tile, X
        self._step_dy = 0.0   # unit vector toward cheapest reachable 8-direction tile, Y

    def _load_reward_fn(self, persona_name):
        try:
            mod = importlib.import_module("code.rewards.train_platformer")
            return getattr(mod, persona_name, None)
        except ImportError:
            return None

    def get_action_space(self): return self._act_space
    def get_observation_space(self): return self._obs_space

    def step(self, action: int):
        if not self.alive:
            # Need to return raycasts in dead obs too
            # dead_rays = np.zeros(self.num_rays * 2, dtype=np.float32)
            dead_obs = self._obs()
            # dead_obs['raycasts'] = dead_rays
            return dead_obs, 0.0, True, False, {"episode_end": True, "won": self.reached_goal}

        # Time Calculation
        if self.render_mode != "human":
            self.dt = 1 / 60.0
        else:
            time_curr_step = time.time()
            raw_dt = time_curr_step - self.time_last_step
            self.time_last_step = time_curr_step
            self.dt = min(raw_dt, 0.05)

        if self.debug_manager.slow_motion:
            self.dt *= 0.5

        self.frame += 1

        if self.use_timer: self.timer -= self.dt
        if self.render_mode == "human": self.debug_manager.update_input()

        # Step Metrics Reset
        self._last_action = int(action)
        self.last_x = self.player.gObj.x if self.player else 0.0
        self.kills_step = self.coins_step = self.powerups_step = 0
        self.stalled_this_frame = False

        # PHYSICS & LOGIC
        # PERF: Split hash rebuild into two parts:
        #
        #   hazard_hash (enemies, powerups) — rebuilt EVERY frame.
        #     Enemies move each step, so their positions in the hash go stale
        #     immediately. resolve_collisions() queries this hash to detect
        #     player-enemy contact — if enemies are at wrong positions the
        #     collision check silently misses. Must always be current.
        #
        #   collectible_hash (coins, goals) — rebuilt only when dirty.
        #     Coins and goals are static mid-episode (coins teleport to
        #     gObj.active=False when collected, goals never move). Rebuilding
        #     them every frame is wasteful; only needed when the entity list
        #     actually changes (collection, new coin from QBlock spawn).
        #     _hash_dirty is set True in the cleanup block below.
        #
        # BUG (was): Dirty flag guarded the FULL rebuild, including hazard_hash.
        # Enemies moved every frame but hash positions were only updated on
        # entity removal — collision queries looked in stale buckets, missing hits.
        self.physics_manager.hazard_hash.clear()
        for enemy in self.level_data.enemies:
            if enemy.gObj.active:
                self.physics_manager.hazard_hash.insert(enemy)
        for pup in self.level_data.powerups:
            if pup.gObj.active:
                self.physics_manager.hazard_hash.insert(pup)

        if self._hash_dirty:
            self.physics_manager.collectible_hash.clear()
            for coin in self.level_data.coins:
                if coin.gObj.active and not coin.collected:
                    self.physics_manager.collectible_hash.insert(coin)
            for goal in self.level_data.goals:
                self.physics_manager.collectible_hash.insert(goal)
            self._hash_dirty = False

        if self.player:
            if not self.debug_manager.free_cam_active:
                self.player.handle_input(a = int(action))
            else:
                self.player.vx = 0; self.player.jump_hold = 0

        self.physics_manager.update_system(self.dt, self)
        self.physics_manager.resolve_collisions(self)

        # Cleanup Inactive Entities — set dirty flag if anything was actually removed
        # so the spatial hash is rebuilt next step.
        enemies_before   = len(self.level_data.enemies)
        coins_before     = len(self.level_data.coins)
        powerups_before  = len(self.level_data.powerups)

        self.level_data.enemies[:]  = [e for e in self.level_data.enemies  if e.gObj.active]
        self.level_data.coins[:]    = [c for c in self.level_data.coins    if c.gObj.active]
        self.level_data.powerups[:] = [p for p in self.level_data.powerups if p.gObj.active]

        if (len(self.level_data.enemies)  != enemies_before  or
            len(self.level_data.coins)    != coins_before    or
            len(self.level_data.powerups) != powerups_before):
            self._hash_dirty = True

        # PERF: Cache goal distance once per step.
        # _get_dist_to_goal() was called 3x per step (stall metrics, tracking obs,
        # _info). All three now read self._goal_dist_cache instead.
        self._goal_dist_cache = self._get_dist_to_goal()


        self._update_camera()
        if self.anti_stall: self._update_stall_metrics()

        # Check Truncation & Termination
        terminated = self._check_termination()

        # FIX: Truncation is exclusively tied to max_steps, not death.
        truncated = False
        if self.max_steps and self.frame >= self.max_steps:
            truncated = True

        self.score_delta = self.score - self.last_score
        self.last_score = self.score

        info = self._info()

        # Inline level transition on win.
        # complete_level() (called by PhysicsManager this frame) set
        # _needs_level_transition=True and pre-selected the next level index.
        # We do the actual load NOW, after _info() has captured the WIN event,
        # so the WIN is visible in this step's info dict.
        # lives and score are naturally preserved because load_level() never
        # touches them.
        if self._needs_level_transition:
            self._needs_level_transition = False
            self.load_level()   # loads self.world (already set by complete_level())

        if terminated:
            info["episode_end"] = True

        # Return raw score delta as base reward.
        # The GameEnv wrapper (generic_env.py) applies the actual persona reward fn.
        base_reward = float(self.score_delta)


        # BUG (was): Full Dijkstra recomputation was triggered on every coin
        # collection. Removed -- see comment in original for details.
        # if self.coins_step > 0:
        #     self._calculate_dijkstra_map()

        obs = self._obs()
        self._check_obs_sanity(obs)

        return obs, base_reward, bool(terminated), bool(truncated), info


    def reset(self, seed=None, options=None) -> np.ndarray:
        """
        Called by SB3 only when the episode truly ends:
          - lives = 0 (terminated=True from _handle_death)
          - max_steps hit (truncated=True)

        Level completion no longer calls reset() -- the episode continues
        with the next level loaded inline in step(). The was_win branch
        is therefore dead code and has been removed.
        """
        super().reset(seed=seed)

        # BUG (was): was_win branch tried to preserve lives/score across episodes
        # on level completion. This is now handled inline in step(), so reset()
        # is only ever called on death or truncation. Always do a full reset.
        # was_win = self.reached_goal
        # if not was_win: ...
        # else: ...  <- removed

        # Full reset: restore lives, score, return to level 0
        self.reset_metrics()
        self.current_index_world = 0
        self.world = self.level_order[self.current_index_world]
        self.load_level()
        return self._obs(), self._info()

    def load_level(self):
        self.alive = True
        self.frame = 0
        self.game_over = False
        self.reached_goal = False

        config = self.config_manager.get_level_config(self.world)
        self.level_data = self.loader.load_level(config)

        raw_grid = np.array(self.level_data.grid, dtype=np.int32)
        self.solid_grid_np = (raw_grid != TILE_AIR).astype(np.float32)

        pad_y = self.obs_height // 2
        pad_x = self.obs_width // 2

        self.padded_solid = np.pad(
            self.solid_grid_np,
            ((pad_y, pad_y), (pad_x, pad_x)),
            mode='constant',
            constant_values=0.0
        )

        px, py = self.level_data.player_start
        if 'spawn' in config:
            px = float(config['spawn'].get('x', px))
            py = float(config['spawn'].get('y', py))

        # PERF: Reuse the Player object across soft resets (deaths) rather than
        # destroying and recreating it. Player.__post_init__ calls
        # AnimationHandler.load_animations which does pygame.image.load() (disk I/O)
        # for every sprite frame. In early training the agent dies constantly,
        # making this the hottest path in load_level.
        #
        # Strategy:
        #   - First call (self.player is None): create and initialise normally.
        #   - Subsequent calls (soft reset): just teleport to spawn and reset physics state.
        #
        if self.player is None:
            # First episode — full construction including AnimationHandler / image I/O
            self.player = Player(gObj=GameObject(px, py, PLATFORMER_WIDTH, PLATFORMER_HEIGHT, True))
            self.player.__post_init__()
        else:
            # Reuse existing object — only reset mutable state
            self.player.gObj.x       = float(px)
            self.player.gObj.y       = float(py)
            self.player.gObj.active  = True
            self.player.vx           = 0.0
            self.player.vy           = 0.0
            self.player.on_ground    = False
            self.player.facing_right = True
            self.player.powered_up   = False
            self.player.invincible_timer = 0
            self.player.coyote       = 0
            self.player.jump_hold    = 0
            self.player.jump_buffer  = 0
            self.player.input_dir    = 0
            self.player.run_held     = False
            self.player.jump_pressed = False

        self.physics_manager.reset_to_defaults()
        self.physics_manager.apply_config_dict(config)
        # Force hash rebuild on the first step of the new level.
        self.physics_manager.rebuild_dynamic_hashes(self.level_data)
        self._hash_dirty = False  # already fresh, no need to rebuild again on step 1

        self.progress_x_best = self.player.gObj.x
        self.progress_y_best = self.level_data.height - self.player.gObj.y
        self.stall_timer = 0
        self.stall_windows_count = 0
        self.stalled_this_frame = False
        self.max_x_seen = px
        self.best_dist_to_goal = self._get_dist_to_goal()

        self.camera_x = 0.0
        self.camera_y = 0.0
        self.last_score = 0
        self.last_x = self.player.gObj.x

        self.timer = config.get('time_limit', self.timer_seconds) if self.use_timer else math.inf

        # --- CALCULATE DIJKSTRA MAP ---
        self._calculate_dijkstra_map()

    def _calculate_dijkstra_map(self):
        """Initializes and computes the global Dijkstra distance map."""
        if not self.level_data.goals:
            self.dijkstra = None
            return

        self.dijkstra = DijkstraSolver(self.level_data.grid, self.level_data.rows, self.level_data.cols)

        goal_positions = []
        for g in self.level_data.goals:
            gx = int(g.gObj.x // TILE_SIZE)
            gy = int(g.gObj.y // TILE_SIZE)
            goal_positions.append((gx, gy))

        # Collect coin positions for Dijkstra weighting
        coin_positions = set()
        for c in self.level_data.coins:
            cx = int(c.gObj.x // TILE_SIZE)
            cy = int(c.gObj.y // TILE_SIZE)
            coin_positions.add((cx, cy))

        self.dijkstra.compute_map(goal_positions, coin_positions)

    def complete_level(self):
        """
        Called by PhysicsManager the moment the player touches the goal tile.
        Selects the next level using a random curriculum walk but does NOT load
        it yet -- that happens inline in step() after _info() has captured the
        WIN event for this frame.

        The episode continues into the next level; only lives = 0 ends it.
        """
        # Random walk: +-1 from current index (curriculum learning)
        self.current_index_world = max(
            0, min(self.current_index_world + 1, random.randint(self.current_index_world - 1, self.current_index_world + 2) )
        )
        if self.current_index_world >= len(self.level_order):
            self.current_index_world = len(self.level_order) - 1
        self.world = self.level_order[self.current_index_world]

        # Signal step() to load the next level after _info() runs this frame.
        # BUG (was): game_over = True was set here, terminating the episode.
        # The episode should continue -- only lives = 0 should end it.
        # self.game_over = True  <- removed
        self._needs_level_transition = True

    def _handle_death(self, cause: str = "Unknown") -> bool:
        """
        Decrements lives and either soft-resets (lives > 0) or ends the episode.

        Returns True only when lives reach 0, signalling _check_termination
        to return True so SB3 calls reset() for a full episode restart.
        """
        self.death_cause = cause
        self.lives = max(0, self.lives - 1)
        if self.lives > 0:
            self._soft_reset()
            return False   # episode continues
        else:
            # Lives exhausted -- end the episode.
            # Do NOT call reset() here; SB3 does that after terminated=True.
            self.alive     = False
            self.game_over = True
            return True    # episode ends

    def _soft_reset(self):
        """
        Reloads the current level after a death, preserving lives and score.

        load_level() never modifies self.lives or self.score, so no explicit
        save/restore is needed. The old save/restore pattern was redundant.
        """
        # BUG (was): current_lives was saved and restored around load_level().
        # load_level() never modifies self.lives, making this dead code.
        # Removed for clarity.
        self.load_level()

    def _update_camera(self):
        # PERF: Camera state is only consumed by render(), which is never called
        # during headless training. Skip all camera math in non-human modes.
        if self.render_mode != "human":
            return

        if self.debug_manager.free_cam_active:
            movement_x, movement_y = self.debug_manager.current_cam_move
            self.camera_x += movement_x * self.dt
            self.camera_y += movement_y * self.dt
            return

        if not self.camera_lock or not self.player: return

        level_w = self.level_data.width
        level_h = self.level_data.height

        target_x = max(0, min(self.player.gObj.x - self.WIDTH // 3, level_w - self.WIDTH))
        self.camera_x += (target_x - self.camera_x) * self.camera_smoothing
        self.camera_x = max(0, min(self.camera_x, max(0, level_w - self.WIDTH)))

        target_y = 0.0
        if level_h > self.HEIGHT:
            target_y = max(0, min(self.player.gObj.y - self.HEIGHT // 2, level_h - self.HEIGHT))

        self.camera_y += (target_y - self.camera_y) * self.camera_smoothing
        self.camera_y = max(0, min(self.camera_y, max(0, level_h - self.HEIGHT)))

    def _get_dist_to_goal(self) -> float:
        if not self.player: return float('inf')
        if not self.level_data.goals:
            # Fallback: treat right edge as goal, ignore Y (no Y info available)
            return self.level_data.width - self.player.gObj.x

        px = self.player.gObj.x
        py = self.player.gObj.y
        min_d = float('inf')
        for g in self.level_data.goals:
            gx = g.gObj.x
            gy = g.gObj.y
            d = math.sqrt((gx - px) ** 2 + (gy - py) ** 2)
            if d < min_d:
                min_d = d

        return min_d

    def _update_stall_metrics(self):
        """FIX: Stall logic now uses X-distance to goal."""
        if not self.player: return

        if self.player.gObj.x > self.max_x_seen:
            self.max_x_seen = self.player.gObj.x

        # PERF: Read from cache set at the top of step() instead of recomputing.
        current_dist = getattr(self, '_goal_dist_cache', self._get_dist_to_goal())
        threshold = TILE_SIZE / 2.0

        if current_dist < (self.best_dist_to_goal - threshold):
            self.best_dist_to_goal = current_dist
            self.stall_timer = 0
            self.stalled_this_frame = False
        else:
            self.stall_timer += self.dt
            if self.stall_timer >= self.stall_window:
                self.stalled_this_frame = True
                self.stall_timer = 0
                self.stall_windows_count += 1

    def _check_termination(self) -> bool:
        """
        Returns True only when the episode should end (lives = 0, or no player).
        Goal completion is NOT a termination — it transitions to the next level
        inline inside step() so the episode continues uninterrupted.
        """
        player = self.player
        if not player:
            return True

        # 1. TIME LIMIT — counts as a death
        if self.use_timer and self.timer <= 0:
            return self._handle_death("Timeout")

        # 2. PIT DEATH
        if player.gObj.y > self.level_data.height:
            return self._handle_death("Pit")

        # 3. GOAL — handled by PhysicsManager which sets reached_goal=True and
        # calls complete_level() which sets _needs_level_transition=True.
        # We do NOT terminate here; step() loads the next level after _info()
        # has captured the WIN event for this frame.
        # BUG (was): returned True here, ending the episode on every level clear.
        # if self.reached_goal:
        #     return True

        # 4. STALL DEATH
        if self.anti_stall and self.stall_windows_count >= self.stall_kill_windows:
            return self._handle_death("Stall")

        return False

    def _check_obs_sanity(self, obs: Dict[str, np.ndarray]) -> None:
        """Compute observation statistics every N steps. Stored on self for _info() to expose."""
        self._obs_check_counter += 1
        if self._obs_check_counter % self._obs_check_interval != 0:
            return

        warnings_list = []
        # Stack order: Player(0), Hazard(1), Collectible(2), Dijkstra(3)
        grid_names = ["player", "hazard", "collectible", "dijkstra"]
        grids = obs.get("grids")
        if grids is not None:
            for i, name in enumerate(grid_names):
                ch = grids[i]
                self._obs_stats[f"grid_{name}_mean"] = float(ch.mean())
                self._obs_stats[f"grid_{name}_std"]  = float(ch.std())
                self._obs_stats[f"grid_{name}_min"]  = float(ch.min())
                self._obs_stats[f"grid_{name}_max"]  = float(ch.max())

                if float(ch.std()) < 1e-6 and i < 2:
                    warnings_list.append(f"Grid '{name}' DEAD")
                # Channels 0-2 (player, hazard, collectible) are in [0,1];
                # channel 3 (dijkstra) is in [-1,1], so skip the >1.0 check for it.
                if i < 3 and float(ch.max()) > 1.01:
                    warnings_list.append(f"Grid '{name}' >1.0")

        scalars = obs.get("scalars")
        if scalars is not None:
            self._obs_stats["scalar_mean"] = float(scalars.mean())
            self._obs_stats["scalar_std"]  = float(scalars.std())
            self._obs_stats["scalar_min"]  = float(scalars.min())
            self._obs_stats["scalar_max"]  = float(scalars.max())
            self._obs_stats["dijkstra_val"] = float(scalars[-1])

            if float(scalars.std()) < 1e-8:
                warnings_list.append("Scalars DEAD")
            if abs(float(scalars.max())) > 100:
                warnings_list.append("Scalars unnormalized")

            dijk_val = float(scalars[-1])
            if hasattr(self, '_last_dijk_nonzero'):
                if self._last_dijk_nonzero and dijk_val == 0.0:
                    self._dijk_zero_streak = getattr(self, '_dijk_zero_streak', 0) + 1
                    if self._dijk_zero_streak >= 3:
                        warnings_list.append(f"Dijkstra stuck@0 x{self._dijk_zero_streak}")
                else:
                    self._dijk_zero_streak = 0
            self._last_dijk_nonzero = dijk_val != 0.0

        self._obs_stats["obs_warnings"] = "|".join(warnings_list) if warnings_list else ""

    def _obs(self) -> Dict[str, np.ndarray]:
        # NULL Check Safety
        if not self.player:
            return {
                # CHANGE 1 FIX: was (5, ...) — must match declared obs_space shape (4, ...)
                # 4 channels: player, hazard, collectible, dijkstra-advantage.
                # solid_grid was already removed from the active stack; the null
                # path incorrectly still claimed 5 channels, causing SB3 shape errors.
                "grids": np.zeros((4, self.obs_height, self.obs_width), dtype=np.float32),
                "scalars": np.zeros(12, dtype=np.float32),  # 5 (player) + 7 (tracking)
                # "raycasts": np.zeros(self.num_rays * 2, dtype=np.float32)
            }

        p_obs    = self._player_obs()
        track_obs = self._tracking_obs()
        # rays = self._perform_raycasts()

        # _grid_obs_window returns the unpadded map tile coordinates of the window
        # so that _dijkstra_obs_window can slice dist_map at the identical region.
        # CHANGE 1: solid_grid removed from unpacking — it was already excluded from
        # the stacked observation channels and returned as dead code.
        hazard_grid, collect_grid, player_grid, map_row_start, map_col_start = \
            self._grid_obs_window()

        dijkstra_grid = self._dijkstra_obs_window(map_row_start, map_col_start)

        # Stack order: Player, Hazard, Collectible, Dijkstra-Advantage  (4 channels)
        # Channel 3 (Dijkstra) is in [-1, 1]; channels 0-2 are in [0, 1].
        stacked_grids = np.stack(
            [player_grid, hazard_grid, collect_grid, dijkstra_grid], axis=0
        ).astype(np.float32)

        scalars = np.concatenate([p_obs, track_obs]).astype(np.float32)

        return {
            "grids": stacked_grids,
            "scalars": scalars,
            # "raycasts": rays
        }

    def _player_obs(self) -> np.ndarray:
        p = self.player
        if not p: return np.zeros(5, dtype=np.float32)

        w = max(1.0, float(self.level_data.width))
        h = max(1.0, float(self.level_data.height))

        max_run = max(1.0, getattr(self.physics_manager.context, 'MAX_RUN_SPEED', 240.0))
        max_fall = max(1.0, getattr(self.physics_manager.context, 'MAX_FALL_SPEED', 400.0))

        return np.array([
            np.clip(p.gObj.x / w, 0.0, 1.0),
            np.clip(p.gObj.y / h, 0.0, 1.0),
            np.clip(p.vx / max_run, -1.0, 1.0),
            np.clip(p.vy / max_fall, -1.0, 1.0),
            1.0 if p.on_ground else 0.0,
        ], dtype=np.float32)

    def _grid_obs_window(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
        """
        Returns (hazard, collect, player, map_row_start, map_col_start).

        CHANGE 1: solid_grid removed from the return value. It was computed from
        padded_solid (used internally for window-position arithmetic) but was
        never included in the stacked observation — removing it avoids a NumPy
        slice allocation per step.

        padded_solid is still maintained in load_level() because the slice
        indices (slice_y_start, slice_x_start) are computed from it. Only the
        resulting solid_grid slice is no longer returned.

        map_row_start / map_col_start are the UNPADDED map tile coordinates of
        the top-left corner of the observation window. These are returned so
        _dijkstra_obs_window can slice dist_map at the same region without
        recomputing the window position independently.

        Coordinates can be negative when the player is near the top/left edge
        of the map (the window extends into the padded border). The Dijkstra
        method handles this by clamping and zero-padding accordingly.
        """
        p = self.player
        if not p:
            z = np.zeros((self.obs_height, self.obs_width), dtype=np.float32)
            # Return 0,0 as dummy map coords — _dijkstra_obs_window will see no
            # valid dijkstra and return zeros anyway.
            return z, z, z, 0, 0

        px = int(p.gObj.x // TILE_SIZE)
        py = int(p.gObj.y // TILE_SIZE)

        # Centre the window on the player in padded-solid space
        slice_y_start = py - self.obs_pad_y
        slice_x_start = px - self.obs_pad_x

        # Safety clamp to padded array bounds
        max_h, max_w = self.padded_solid.shape
        slice_y_start = max(0, min(slice_y_start, max_h - self.obs_height))
        slice_x_start = max(0, min(slice_x_start, max_w - self.obs_width))
        slice_y_end   = slice_y_start + self.obs_height
        slice_x_end   = slice_x_start + self.obs_width

        # CHANGE 1: solid_grid slice removed — padded_solid is still used above
        # to compute the window bounds (slice_y/x_start), but the resulting
        # channel is no longer needed in the observation.

        hazard_grid  = np.zeros((self.obs_height, self.obs_width), dtype=np.float32)
        collect_grid = np.zeros((self.obs_height, self.obs_width), dtype=np.float32)
        player_grid  = np.zeros((self.obs_height, self.obs_width), dtype=np.float32)

        # Convert padded indices → unpadded map tile coordinates
        map_row_start = slice_y_start - self.obs_pad_y
        map_col_start = slice_x_start - self.obs_pad_x

        # World-pixel origin of the window (used for entity spatial-hash queries)
        wx = slice_x_start * TILE_SIZE
        wy = slice_y_start * TILE_SIZE

        window_rect = pygame.Rect(wx, wy, self.obs_width * TILE_SIZE, self.obs_height * TILE_SIZE)

        nearby_hazards = self.physics_manager.hazard_hash.query_rect(
            window_rect.x, window_rect.y, window_rect.width, window_rect.height
        )
        for h in nearby_hazards:
            if not h.gObj.active: continue
            hx = int(h.gObj.x // TILE_SIZE)
            hy = int(h.gObj.y // TILE_SIZE)
            local_x = hx - slice_x_start
            local_y = hy - slice_y_start
            if 0 <= local_x < self.obs_width and 0 <= local_y < self.obs_height:
                hazard_grid[local_y, local_x] = 1.0

        nearby_items = self.physics_manager.collectible_hash.query_rect(
            window_rect.x, window_rect.y, window_rect.width, window_rect.height
        )
        for c in nearby_items:
            if not c.gObj.active: continue
            if hasattr(c, 'collected') and c.collected: continue
            cx = int(c.gObj.x // TILE_SIZE)
            cy = int(c.gObj.y // TILE_SIZE)
            local_x = cx - slice_x_start
            local_y = cy - slice_y_start
            if 0 <= local_x < self.obs_width and 0 <= local_y < self.obs_height:
                collect_grid[local_y, local_x] = 1.0

        # --- FILL PLAYER GRID ---
        p_rect      = p.gObj.get_rect()
        rel_x       = p_rect.x - wx
        rel_y       = p_rect.y - wy
        p_start_col = int(rel_x // TILE_SIZE)
        p_start_row = int(rel_y // TILE_SIZE)
        p_w_tiles   = math.ceil(p.gObj.width  / TILE_SIZE)
        p_h_tiles   = math.ceil(p.gObj.height / TILE_SIZE)

        for r in range(p_start_row, p_start_row + p_h_tiles):
            for c in range(p_start_col, p_start_col + p_w_tiles):
                if 0 <= r < self.obs_height and 0 <= c < self.obs_width:
                    player_grid[r, c] = 1.0

        return hazard_grid, collect_grid, player_grid, map_row_start, map_col_start

    def _dijkstra_obs_window(self, map_row_start: int, map_col_start: int) -> np.ndarray:
        """
        Returns a (obs_height, obs_width) float32 advantage map in [-1, 1].

        Each cell encodes how much closer (positive) or further (negative) that
        tile is from the goal compared to the player's current tile:

            delta[r, c] = player_dist - window_dist
                        = player_tile_cost - neighbour_tile_cost

        Interpretation for the CNN:
          +1.0  → tile is much closer to the goal than the player  (go here)
           0.0  → same distance as player, or unreachable tile     (neutral)
          -1.0  → tile is much further from the goal than player   (avoid)

        The map is normalised by (cols * 2) — the same heuristic max-cost used
        in _info() — and clipped to [-1, 1] for consistent scale with other
        channels.

        Tiles with inf cost (walls, unreachable air pockets) are set to 0 so
        they are neutral rather than misleadingly negative.
        """
        zero = np.zeros((self.obs_height, self.obs_width), dtype=np.float32)

        if self.dijkstra is None or not self.player:
            return zero

        # Player tile cost — if unreachable (inf), the whole window is meaningless
        px_tile = int(self.player.gObj.x // TILE_SIZE)
        py_tile = int(self.player.gObj.y // TILE_SIZE)
        player_dist = self.dijkstra.dist_map[py_tile, px_tile] \
            if (0 <= py_tile < self.level_data.rows and 0 <= px_tile < self.level_data.cols) \
            else np.inf

        if not np.isfinite(player_dist):
            # Player is on an unreachable tile (e.g. first frame before physics
            # settles). Return neutral zeros — don't emit misleading gradients.
            return zero

        # Clamp window to valid map bounds.
        # map_row_start can be negative when the player is near the top/left edge.
        r0 = max(0, map_row_start)
        c0 = max(0, map_col_start)
        r1 = min(self.level_data.rows, map_row_start + self.obs_height)
        c1 = min(self.level_data.cols, map_col_start + self.obs_width)

        # How many rows/cols of the output window are actually inside the map
        valid_rows = r1 - r0
        valid_cols = c1 - c0

        if valid_rows <= 0 or valid_cols <= 0:
            return zero

        # Direct numpy slice — no Python loop over tiles.
        # PERF: No .copy() needed here — the subtraction below (player_dist - dist_slice)
        # produces a new array without modifying dist_map in-place.
        dist_slice = self.dijkstra.dist_map[r0:r1, c0:c1]

        # Compute relative advantage: positive = closer to goal than player
        # player_dist - dist_slice:
        #   dist_slice small (near goal)  → large positive  ✓
        #   dist_slice large (far away)   → large negative  ✓
        delta = player_dist - dist_slice

        # Walls / unreachable tiles have cost=inf → delta = -inf → set to 0 (neutral)
        delta[~np.isfinite(delta)] = 0.0

        # Normalise to [-1, 1]
        max_cost = float(self.level_data.cols * 2)
        delta     = np.clip(delta / max_cost, -1.0, 1.0)

        # Place the valid slice into the full output window, leaving edge-padding
        # as zeros when the window extends outside the map.
        # PERF: zero.copy() is equivalent to np.zeros() but slower — avoid it.
        out = np.zeros((self.obs_height, self.obs_width), dtype=np.float32)
        out_r0 = r0 - map_row_start   # offset into output array
        out_c0 = c0 - map_col_start
        out[out_r0 : out_r0 + valid_rows, out_c0 : out_c0 + valid_cols] = delta

        return out.astype(np.float32)

    def _tracking_obs(self) -> np.ndarray:
        """
        Returns 13 scalar features used by the MLP branch of the extractor.

        Index  Feature
        -----  -------
          0    Nearest enemy distance (normalised)
          1    Nearest coin distance  (normalised)
          2    Goal distance          (normalised)
          3    Enemy count            (normalised)
          4    Coin count             (normalised)
          5    Score                  (normalised)
          6    Timer remaining        (normalised)
          7    Lives remaining        (normalised)
          8    Goal direction X       (-1, 0, +1)
          9    Goal direction Y       (normalised signed)
         10    Dijkstra distance      (0=at goal, 1=far, 1.0 if unreachable)
         11    Steepest descent dX    (-1 to +1, direction of cheapest neighbour)
         12    Steepest descent dY    (-1 to +1, direction of cheapest neighbour)

        Scalars 11-12 give the agent an explicit local step direction derived
        directly from the Dijkstra map. The CNN advantage channel (channel 4 of
        grids) provides the full spatial picture; these scalars provide an
        immediate, unambiguous "step this way" signal that the MLP can exploit
        from very early in training without needing to learn gradient extraction
        through convolution first.
        """
        p = self.player
        if not p: return np.zeros(7, dtype=np.float32)  # FIXED: 7 active scalars

        # PERF: Was iterating the full entity list and calling math.sqrt on every
        # enemy/coin regardless of distance. For a level with 60 coins that was
        # 60 sqrt calls per step. Instead, query the spatial hash with a search
        # radius and only compute sqrt on nearby candidates.
        #
        # Search radius = full obs window diagonal in pixels — anything outside
        # that window is irrelevant to the observation anyway.
        SEARCH_RADIUS = math.sqrt(
            (self.obs_width  * TILE_SIZE) ** 2 +
            (self.obs_height * TILE_SIZE) ** 2
        ) * 0.5

        def get_dist_hash(hash_obj, check_collected=False):
            """Query spatial hash within obs window, return (min_dist, count)."""
            min_d = 9999.0
            count = 0
            nearby = hash_obj.query_rect(
                p.gObj.x - SEARCH_RADIUS, p.gObj.y - SEARCH_RADIUS,
                SEARCH_RADIUS * 2, SEARCH_RADIUS * 2
            )
            for obj in nearby:
                if not obj.gObj.active: continue
                if check_collected and hasattr(obj, 'collected') and obj.collected: continue
                dx = p.gObj.x - obj.gObj.x
                dy = p.gObj.y - obj.gObj.y
                # Use squared distance for comparison, only sqrt the winner
                d_sq = dx*dx + dy*dy
                if d_sq < min_d:
                    min_d = d_sq
                count += 1
            return (math.sqrt(min_d) if min_d < 9999.0 else 9999.0), count

        e_dist, e_count = get_dist_hash(self.physics_manager.hazard_hash)
        c_dist, c_count = get_dist_hash(self.physics_manager.collectible_hash, check_collected=True)

        # PERF: Read from cache set at the top of step() — avoids a third
        # euclidean distance computation per step.
        raw_goal_dist = getattr(self, '_goal_dist_cache', self._get_dist_to_goal())
        norm_dist = max(self.level_data.width, self.level_data.height, 1.0)

        # Goal direction
        if self.level_data.goals:
            closest = min(self.level_data.goals, key=lambda g: abs(g.gObj.x - p.gObj.x))
            closest_goal_x = closest.gObj.x
            closest_goal_y = closest.gObj.y
        else:
            closest_goal_x = self.level_data.width
            closest_goal_y = p.gObj.y

        dx = closest_goal_x - p.gObj.x
        dy = closest_goal_y - p.gObj.y
        dir_x      = np.sign(dx)
        dist_y_norm = np.clip(dy / self.level_data.height, -1.0, 1.0)

        # Dijkstra distance scalar (for the MLP — same as _info convention)
        dijkstra_dist = 1.0  # default: treat as far if unavailable
        if self.dijkstra:
            px_tile = int(p.gObj.x // TILE_SIZE)
            py_tile = int(p.gObj.y // TILE_SIZE)
            d = self.dijkstra.get_dist(px_tile, py_tile)
            if d >= 0:
                dijkstra_dist = np.clip(d / (self.level_data.cols * 2), 0.0, 1.0)
            # else stays 1.0 — far/unknown (observation only, reward uses -1 sentinel)

        # --- Steepest descent direction ---
        # Check all 8 neighbours of the player's current tile and find the one
        # with the lowest Dijkstra cost. Emit a normalised (dx, dy) vector toward
        # that tile. When the player is mid-air (unreachable tile), emit (0, 0).
        #
        # This is the "which tile should I step onto right now" signal.
        # The CNN channel gives the full spatial gradient; this gives an explicit
        # local step direction that the MLP branch can use from the very first
        # rollout without having to learn gradient extraction first.
        step_dx, step_dy = 0.0, 0.0
        if self.dijkstra:
            px_tile = int(p.gObj.x // TILE_SIZE)
            py_tile = int(p.gObj.y // TILE_SIZE)

            # --- 8-Direction Best Tile Search ---
            # Checks all 8 neighbours (cardinal + diagonal), finds the one with
            # the lowest finite Dijkstra cost, and emits a unit direction vector
            # toward it. Diagonal directions are included because the Dijkstra
            # solver uses 8-way movement — the cheapest next step is often diagonal.
            #
            # Obstacle safety: tiles behind walls have inf cost and are excluded
            # automatically. The direction always points along a routable path.
            rows = self.level_data.rows
            cols = self.level_data.cols
            dm   = self.dijkstra.dist_map

            best_cost = (float(dm[py_tile, px_tile])
                         if 0 <= py_tile < rows and 0 <= px_tile < cols
                         else math.inf)
            best_ddx, best_ddy = 0, 0

            for ddx, ddy in [( 1, 0), (-1, 0), ( 0, 1), ( 0,-1),
                              ( 1,-1), (-1,-1), ( 1, 1), (-1, 1)]:
                nr, nc = py_tile + ddy, px_tile + ddx
                if 0 <= nr < rows and 0 <= nc < cols:
                    cost = float(dm[nr, nc])
                    if cost < best_cost:
                        best_cost = cost
                        best_ddx, best_ddy = ddx, ddy

            # Normalise: diagonal steps (1,1) have magnitude √2 so we normalise
            # to a proper unit vector rather than clamping per-axis.
            mag = math.sqrt(best_ddx * best_ddx + best_ddy * best_ddy)
            if mag > 0:
                step_dx = best_ddx / mag
                step_dy = best_ddy / mag

        # --- Velocity Alignment: cache direction so _info() can read it ---
        self._step_dx = step_dx
        self._step_dy = step_dy

        # 13 Elements Total
        return np.array([
            np.clip(e_dist       / norm_dist, 0.0, 1.0),
            #np.clip(c_dist       / norm_dist, 0.0, 1.0),
            np.clip(raw_goal_dist/ norm_dist, 0.0, 1.0),
            #np.clip(e_count      / 20.0,      0.0, 1.0),
            #np.clip(c_count      / 50.0,      0.0, 1.0),
            #np.clip(self.score   / 10000.0,   0.0, 1.0),
            np.clip(self.timer   / max(1.0, self.timer_seconds), 0.0, 1.0),
            #self.lives / float(max(1.0, self.max_lives)),
            #dir_x,
            dist_y_norm,
            dijkstra_dist,
            step_dx,    # direction toward cheapest reachable neighbour tile
            step_dy,
        ], dtype=np.float32)

    def _info(self) -> Dict:
        p = self.player
        ts = float(TILE_SIZE)

        event = ""
        cause = getattr(self, 'death_cause', "")
        if self.reached_goal:
            event = "WIN"
            cause = "Goal"
        elif not self.alive:
            event = "DIED"

        # NULL Check Safety for Info block
        if not p:
            return {
                "score": self.score, "score_delta": self.score_delta, "frame_count": self.frame,
                "x_position": 0.0, "y_position": 0.0, "velocity_x": 0.0, "velocity_y": 0.0,
                "coins_collected": self.coins_total, "enemies_killed_step": self.kills_step,
                "powered_up": False, "terminated": not self.alive, "won": False,
                "action": self._last_action, "time_left": math.ceil(self.timer),
                "max_x_seen": self.max_x_seen, "stall_windows": self.stall_windows_count,
                "stalled": self.stalled_this_frame, "persona": self.persona,
                "level": self.current_index_world, "goal_dist": 0.0, "lives": self.lives,
                "event": event, "cause": cause,

                # --- Velocity Alignment / Potential-Based Shaping ---
                "on_ground": False,
                "step_dx":   0.0,
                "step_dy":   0.0,

                **self._obs_stats

            }

        dijkstra_dist = 0.0
        if self.dijkstra:
            px_tile = int(p.gObj.x // TILE_SIZE)
            py_tile = int(p.gObj.y // TILE_SIZE)
            d = self.dijkstra.get_dist(px_tile, py_tile)
            self.dijkstra_current_tile = d
            if d >= 0:
                # Normalise: max meaningful cost ≈ cols * 2  (heuristic)
                dijkstra_dist = np.clip(d / (self.level_data.cols * 2), 0.0, 1.0)
            else:
                # BUG (was): set to 1.0 when unreachable.
                # The reward tracker interpreted this as "player is at maximum
                # distance from goal". If the player was near the goal
                # (last_dijkstra ≈ 0.3) and stepped onto an unreachable tile
                # (e.g. mid-air), the tracker computed:
                #     dijkstra_progress = 0.3 - 1.0 = -0.7  → reward = -70
                # This punished every jump, teaching the agent not to jump.
                #
                # FIX: Use the sentinel value -1.0 to signal "no valid reading".
                # The _ScoreTracker in train_platformer.py checks for this and
                # emits progress = 0.0 (neither reward nor penalty) instead of
                # computing a misleading delta.
                dijkstra_dist = -1.0   # sentinel: unreachable / off-grid

        return {
            "score": self.score,
            "score_delta": self.score_delta,
            "frame_count": self.frame,
            "x_position": p.gObj.x / ts,
            "y_position": p.gObj.y / ts,
            "velocity_x": p.vx / ts,
            "velocity_y": p.vy / ts,
            "coins_collected": self.coins_total,
            "enemies_killed_step": self.kills_step,
            "powered_up": p.powered_up,
            "terminated": not self.alive,
            "won": self.reached_goal,
            "action": self._last_action,
            "time_left": math.ceil(self.timer),
            "max_x_seen": self.max_x_seen,
            "stall_windows": self.stall_windows_count,
            "stalled": self.stalled_this_frame,
            "persona": self.persona,
            "level": self.current_index_world,
            # PERF: Read from cache set at top of step().
            "goal_dist": getattr(self, '_goal_dist_cache', self._get_dist_to_goal()) / ts,
            "lives" : self.lives,
            "event": event,
            "cause": cause,
            "dijkstra_dist": dijkstra_dist, # Used for delta_dijkstra reward

            # --- Velocity Alignment / Potential-Based Shaping ---
            "on_ground": p.on_ground,
            "step_dx":   self._step_dx,
            "step_dy":   self._step_dy,

            **self._obs_stats  # Observation sanity stats (populated every N steps)

        }

    def render(self, surface: pygame.Surface, blit_only: bool = True):
        # In human mode, draw game content into the left portion only
        if self.render_mode == "human":
            game_surf = surface.subsurface(pygame.Rect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT))
        else:
            game_surf = surface

        game_surf.fill(COLOR_SKY)
        self._draw_world(game_surf)
        self._draw_entities(game_surf)
        self._draw_player(game_surf)

        if self.debug_manager:
            self.debug_manager.render_overlays(surface, self)

            # Draw sensor rays Ã¢â‚¬â€ colour-coded, respects F1 toggle
            if self.debug_manager.show_sensors and hasattr(self, 'last_rays'):
                from .modules.System.debugging_mods.overlays import (
                    RAY_EMPTY, RAY_SOLID, RAY_HAZARD, RAY_COIN, RAY_GOAL)
                ray_surf = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
                for start, end, found, rtype in self.last_rays:
                    if   rtype == 0.0: color = RAY_EMPTY
                    elif rtype == 1.0: color = RAY_SOLID
                    elif rtype == 2.0: color = RAY_HAZARD
                    elif rtype == 3.0: color = RAY_COIN
                    elif rtype == 4.0: color = RAY_GOAL
                    else:              color = RAY_EMPTY
                    s_cam = (start[0] - self.camera_x, start[1] - self.camera_y)
                    e_cam = (end[0]   - self.camera_x, end[1]   - self.camera_y)
                    pygame.draw.line(ray_surf, color, s_cam, e_cam, 1)
                game_surf.blit(ray_surf, (0, 0))

        self._draw_ui(game_surf)

    def _draw_world(self, surface: pygame.Surface):
        visible_tiles = self.level_data.static_hash.query_rect(self.camera_x, self.camera_y, self.WIDTH, self.HEIGHT)
        for tile in visible_tiles:
            if tile.x + tile.width < self.camera_x or tile.x > self.camera_x + self.WIDTH: continue

            if isinstance(tile, Tile):
                if tile.color == COLOR_QBLOCK: continue
                tile.render(surface, self.camera_x, self.camera_y)
            elif isinstance(tile, QuestionBlock):
                tile.render(surface, tile.x - self.camera_x, tile.y - self.camera_y)
            elif hasattr(tile, 'render'):
                try:
                    tile.render(surface, self.camera_x, self.camera_y)
                except TypeError:
                    tile.render(surface, tile.x - self.camera_x, tile.y - self.camera_y)

    def _draw_entities(self, surface: pygame.Surface):
        cx, cy, cw, ch = self.camera_x, self.camera_y, self.WIDTH, self.HEIGHT

        visible_hazards = self.physics_manager.hazard_hash.query_rect(cx, cy, cw, ch)
        for entity in visible_hazards:
            if hasattr(entity, 'render'):
                entity.render(surface, entity.x - cx, entity.y - cy)

        visible_collectibles = self.physics_manager.collectible_hash.query_rect(cx, cy, cw, ch)
        for entity in visible_collectibles:
            if hasattr(entity, 'render'):
                entity.render(surface, entity.x - cx, entity.y - cy)

    def _draw_player(self, surface: pygame.Surface):
        p = self.player
        if not p: return
        sx = p.gObj.x - self.camera_x
        sy = p.gObj.y - self.camera_y

        colour = COLOR_POWERUP_STAR if (p.invincible_timer > 0 and (self.frame // 5) % 2) else \
            ((255, 100, 0) if p.powered_up else (255, 0, 0))
        p.color = colour
        p.render(surface, sx, sy, self.debug_manager.show_sensors)

    def _draw_ui(self, surface: pygame.Surface):
        status = "STAR" if (self.player and self.player.invincible_timer > 0) else ("SUPER" if (self.player and self.player.powered_up) else "SMALL")
        text = self.ui_font.render(
            f"Lives:{self.lives}  Score:{self.score}  Coins:{self.coins_total}  {status}  Time:{int(self.timer)}",
            True, COLOR_WHITE
        )
        bg = pygame.Surface((text.get_width() + 10, text.get_height() + 6), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 170))

        y = 5
        if self.debug_manager.show_obs_panel: y = self.HEIGHT - text.get_height() - 10
        surface.blit(bg, (5, y))
        surface.blit(text, (10, y + 3))

    def _world_to_screen(self, gObj:GameObject) -> Tuple[float, float, bool]:
        screen_x = gObj.x - self.camera_x
        screen_y = gObj.y - self.camera_y
        on_screen = (
                screen_x < SCREEN_WIDTH and
                screen_x + gObj.width > 0 and
                screen_y < SCREEN_HEIGHT and
                screen_y + gObj.height > 0)
        return screen_x, screen_y, on_screen

    def close(self):
        """Proper resource cleanup."""
        if self.render_mode == "human":
            pygame.quit()