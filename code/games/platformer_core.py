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
from .modules.Objects.FireFlowerProjectile import FireFlowerProjectile   # NEW

# System
from .modules.System.LevelLoader import LevelLoader, LevelData
from .modules.System.PhysicsManager import PhysicsManager
from .modules.System.config_manager import ConfigManager
from .modules.System.debugging_mods.manager import DebugManager
from .modules.Objects.Spike import Spike
from .modules.Objects.MovingPlatform import MovingPlatform

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
    # ── Original 10 (no fire) ─────────────────────────────────────────
    0:  "IDLE",
    1:  "LEFT",
    2:  "RIGHT",
    3:  "JUMP",
    4:  "RIGHT+JUMP",
    5:  "RUN+RIGHT",
    6:  "LEFT+JUMP",
    7:  "RUN+RIGHT+JUMP",
    8:  "RUN+LEFT",
    9:  "RUN+LEFT+JUMP",
    # ── Fire variants (offset +10) ────────────────────────────────────
    10: "FIRE",
    11: "LEFT+FIRE",
    12: "RIGHT+FIRE",
    13: "JUMP+FIRE",
    14: "RIGHT+JUMP+FIRE",
    15: "RUN+RIGHT+FIRE",
    16: "LEFT+JUMP+FIRE",
    17: "RUN+RIGHT+JUMP+FIRE",
    18: "RUN+LEFT+FIRE",
    19: "RUN+LEFT+JUMP+FIRE",
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
        _default_world = self.level_order[0] if self.level_order else "1-1"
        self.world = str(kwargs.pop("world", _default_world))
        # locked_level: when set, reset() always returns to this level (editor playtest)
        self.locked_level = str(self.world) if kwargs.pop("lock_level", False) else None
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

        # ── Batch Curriculum (Kevin's design) ─────────────────────────────────
        # Evaluates rolling windows of N episodes per level.
        #   Win rate >= advance_threshold  → progress to next level
        #   Win rate <= fallback_threshold → bump down one level
        #   Otherwise                      → stay and keep practicing
        #
        # Anti-oscillation: tracks consecutive fallbacks per level.
        #   After 2 consecutive fallbacks on the same level, the advance
        #   threshold is widened (made easier) to prevent ping-pong.
        #
        # Anti-stagnation: "long stay" counter.
        #   If the agent stays on the same level for `max_stay_windows`
        #   consecutive evaluation windows without advancing, force a
        #   fallback so it gets more practice on easier levels.
        #
        # _curriculum_position: index into level_order (which level we start on)
        # _batch_results: list of bools for current evaluation window
        # _episode_won_current_level: set True by complete_level() if the
        #   agent beats the level it STARTED the episode on.
        self._batch_window            = int(kwargs.pop("batch_window", 10))
        self._batch_advance_threshold = float(kwargs.pop("advance_threshold", 0.30))   # 3/10
        self._batch_fallback_threshold= float(kwargs.pop("fallback_threshold", 0.20))  # ≤2/10 → regress (Kev: 1/5)
        self._max_stay_windows        = int(kwargs.pop("max_stay_windows", 3))
        self._curriculum_position     = 0
        self._batch_results: list     = []
        self._episode_won_current     = False   # did this episode beat the starting level?
        self._windows_on_level        = 0       # consecutive eval windows on this level
        self._consecutive_fallbacks   = {}      # {level_idx: count} — oscillation detection

        # Legacy tracking (still used by dashboard, watch_all, etc.)
        self._level_visits     = {lvl: 0 for lvl in self.level_order}
        self._level_wins       = {lvl: 0 for lvl in self.level_order}

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
            "grid_solid_mean": 0.0, "grid_solid_std": 0.0,
            "grid_solid_min": 0.0, "grid_solid_max": 0.0,
            "grid_collectible_mean": 0.0, "grid_collectible_std": 0.0,
            "grid_collectible_min": 0.0, "grid_collectible_max": 0.0,
            "grid_hazard_mean": 0.0, "grid_hazard_std": 0.0,
            "grid_hazard_min": 0.0, "grid_hazard_max": 0.0,
            "grid_dijkstra_mean": 0.0, "grid_dijkstra_std": 0.0,
            "grid_dijkstra_min": 0.0, "grid_dijkstra_max": 0.0,
            "scalar_mean": 0.0, "scalar_std": 0.0,
            "scalar_min": 0.0, "scalar_max": 0.0,
            "dijkstra_val": 0.0, "obs_warnings": "",
        }

        self.reset_metrics()

        # --- GRID OBSERVATION SIZE ---
        self.obs_width = 21
        self.obs_height = 21
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
            # 5 Channels (in order): Player, Solids, Collectible, Hazard, Dijkstra-Advantage
            #   0 - Player       : tiles occupied by the player bounding box
            #   1 - Solids       : ground, static platforms, moving platforms, question blocks
            #   2 - Collectible  : coins, powerups, goals
            #   3 - Hazard       : enemies, spikes
            #   4 - Dijkstra     : relative advantage map in [-1, 1]
            # low=-1.0 because channel 4 spans [-1, 1]; channels 0-3 are binary {0, 1}.
            "grids": spaces.Box(low=-1.0, high=1.0, shape=(5, self.obs_height, self.obs_width), dtype=np.float32),

            # Scalars: 18 (Player=5, Tracking=13)
            "scalars": spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32),

            # Raycasts: [dist, type, dist, type, ...] -> Size = num_rays * 2
            # "raycasts": spaces.Box(low=0.0, high=4.0, shape=(self.num_rays * 2,), dtype=np.float32)
        })

        # FIX: Action Space is 10 to match ACTION_NAMES (0 through 9)
        self._act_space = spaces.Discrete(ACTION_NAMES.__len__())

        self.ui_font = pygame.font.SysFont("arial", 20, bold=True)
        self.qblock_font = pygame.font.SysFont("arial", 26, bold=True)

        self._dijkstra_window_cache = None
        
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
        # All three dynamic spatial hashes are rebuilt every frame.
        #
        # Why not use a dirty flag for collectible_hash?
        # Powerups move (mushrooms walk, stars bounce) so their hash
        # positions go stale every frame — the same problem that caused
        # the enemy-hash bug. The cost of rebuilding coins+goals alongside
        # powerups is ~20μs for 200 entities (O(1) hash inserts each),
        # which is negligible compared to the conv forward pass.

        # --- hazard_hash: enemies + spikes ---
        # Rebuilt every frame because enemies move. Spikes are static but live
        # here so the hazard_hash.query_rect in _grid_obs_window picks them up
        # in channel 1 (hazard) without a separate tile-grid scan. The cached
        # spike list avoids scanning the full tile grid each frame.
        # NOTE: _resolve_dynamic_interactions skips SPIKE targets to prevent
        # double-death (spikes are already lethal via _resolve_player_world).
        self.physics_manager.hazard_hash.clear()
        for enemy in self.level_data.enemies:
            if enemy.gObj.active:
                self.physics_manager.hazard_hash.insert(enemy)
        for spike in self._cached_spikes:
            self.physics_manager.hazard_hash.insert(spike)

        # --- platform_hash: moving platforms ---
        # Rebuilt every frame because platforms move.
        self.physics_manager.platform_hash.clear()
        for plat in self.level_data.moving_platforms:
            if plat.gObj.active:
                self.physics_manager.platform_hash.insert(plat)

        # --- collectible_hash: coins, powerups, goals ---
        # BUG FIX: Powerups were previously inserted into hazard_hash,
        # causing the CNN to see mushrooms/stars/flowers as threats in
        # channel 1 (hazard) instead of rewards in channel 2 (collectible).
        # Now correctly placed here alongside coins and goals.
        self.physics_manager.collectible_hash.clear()
        for coin in self.level_data.coins:
            if coin.gObj.active and not coin.collected:
                self.physics_manager.collectible_hash.insert(coin)
        for pup in self.level_data.powerups:
            if pup.gObj.active:
                self.physics_manager.collectible_hash.insert(pup)
        for goal in self.level_data.goals:
            self.physics_manager.collectible_hash.insert(goal)

        if self.player:
            if not self.debug_manager.free_cam_active:
                self.player.handle_input(a = int(action))
            else:
                self.player.vx = 0; self.player.jump_hold = 0

            # --- Fire Flower projectile spawn ---
            # Player.handle_input() sets fire_requested via try_fire() when the
            # Z key is pressed (human mode) or try_fire() is called externally
            # (RL mode). We consume the flag here — the core is the only place
            # that should read and clear it, keeping Player free of level/list refs.
            if self.player.fire_requested:
                self.player.fire_requested = False
                proj = FireFlowerProjectile.from_player(self.player)
                self.level_data.projectiles.append(proj)

        self.physics_manager.update_system(self.dt, self)
        self.physics_manager.resolve_collisions(self)

        # Cleanup Inactive Entities — remove dead objects so the per-frame
        # hash rebuild on the next step doesn't insert ghosts.
        self.level_data.enemies[:]  = [e for e in self.level_data.enemies  if e.gObj.active]
        self.level_data.coins[:]    = [c for c in self.level_data.coins    if c.gObj.active]
        self.level_data.powerups[:] = [p for p in self.level_data.powerups if p.gObj.active]

        # Prune dead projectiles so the list doesn't grow unbounded.
        # No dirty flag needed — projectiles are not in any spatial hash.
        self.level_data.projectiles[:] = [
            p for p in self.level_data.projectiles if p.gObj.active
        ]

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

        # Build observation BEFORE _info() so that _check_obs_sanity can
        # populate self._obs_stats in time for _info() to spread them.
        obs = self._obs()
        self._check_obs_sanity(obs)

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
            self.load_level(preserve_power=True)   # win — keep power state

        if terminated:
            info["episode_end"] = True

        # Return raw score delta as base reward.
        # The GameEnv wrapper (generic_env.py) applies the actual persona reward fn.
        base_reward = float(self.score_delta)

        return obs, base_reward, bool(terminated), bool(truncated), info


    def reset(self, seed=None, options=None) -> np.ndarray:
        """
        Called by SB3 only when the episode truly ends:
          - lives = 0 (terminated=True from _handle_death)
          - max_steps hit (truncated=True)

        Records the episode result into the batch curriculum and, when
        the evaluation window is full, decides whether to advance,
        regress, or stay on the current level.
        """
        super().reset(seed=seed)

        # ── Record episode result into the batch ──────────────────────────
        # _episode_won_current is set True by complete_level() if the agent
        # beat the level it STARTED the episode on.  It's False if the agent
        # died before ever reaching the goal on that starting level.
        if self.level_order and not self.locked_level:
            self._batch_results.append(self._episode_won_current)

        # ── Evaluate batch when window is full ────────────────────────────
        if len(self._batch_results) >= self._batch_window and self.level_order and not self.locked_level:
            self._evaluate_curriculum_batch()

        # ── Pick the level for the new episode ────────────────────────────
        self.reset_metrics()
        self._episode_won_current = False   # fresh episode, no win yet

        if self.locked_level:
            # Editor playtest — always stay on the locked level
            self.world = self.locked_level
            if self.locked_level in self.level_order:
                self.current_index_world = self.level_order.index(self.locked_level)
            else:
                self.current_index_world = 0
        elif self.level_order:
            # Curriculum picks the level
            self._curriculum_position = max(0, min(
                self._curriculum_position, len(self.level_order) - 1))
            self.current_index_world = self._curriculum_position
            self.world = self.level_order[self.current_index_world]

        self._level_visits[self.world] = self._level_visits.get(self.world, 0) + 1
        self.load_level()
        return self._obs(), self._info()

    def _evaluate_curriculum_batch(self):
        """
        Evaluate the current batch window and decide: advance / regress / stay.

        Kevin's design:
          - batch of N=10 episodes on the same level
          - win 3/10 (30%) → advance to next level
          - win ≤ 2/10 (20%) → bump down one level
          - otherwise → stay and try another window

        Anti-oscillation (from Claude/Kev discussion):
          - Track consecutive fallbacks per level position
          - After 2 consecutive fallbacks on the same level, widen the advance
            threshold by 10% (make it easier to stay / advance)

        Anti-stagnation:
          - If the agent stays on the same level for max_stay_windows consecutive
            evaluation windows without advancing, force a fallback so it gets
            more practice on the level below.
        """
        wins = sum(1 for r in self._batch_results if r)
        total = len(self._batch_results)
        win_rate = wins / total if total > 0 else 0.0

        pos = self._curriculum_position
        level_name = self.level_order[pos] if pos < len(self.level_order) else "?"
        consec_fb = self._consecutive_fallbacks.get(pos, 0)

        # Anti-oscillation: after 2 consecutive fallbacks on this level,
        # lower the advance threshold by 10% (easier to stay / advance)
        effective_advance = self._batch_advance_threshold
        if consec_fb >= 2:
            effective_advance = max(0.10, effective_advance - 0.10)

        action = "STAY"

        # ── Advance? ──
        if win_rate >= effective_advance and pos < len(self.level_order) - 1:
            action = "ADVANCE"
            self._curriculum_position += 1
            self._windows_on_level = 0
            self._consecutive_fallbacks[pos] = 0   # reset fallback streak
            print(f"  🎓 [Curriculum] ADVANCE → Level {self._curriculum_position} "
                  f"'{self.level_order[self._curriculum_position]}' "
                  f"(won {wins}/{total} = {win_rate:.0%}, "
                  f"threshold {effective_advance:.0%})")

        # ── Fallback? ──
        elif win_rate <= self._batch_fallback_threshold and pos > 0:
            action = "FALLBACK"
            self._curriculum_position -= 1
            self._windows_on_level = 0
            self._consecutive_fallbacks[pos] = consec_fb + 1
            print(f"  ⬇️  [Curriculum] FALLBACK → Level {self._curriculum_position} "
                  f"'{self.level_order[self._curriculum_position]}' "
                  f"(won {wins}/{total} = {win_rate:.0%}, "
                  f"threshold ≤{self._batch_fallback_threshold:.0%}, "
                  f"consec_fb={consec_fb + 1})")

        # ── Stay ──
        else:
            self._windows_on_level += 1
            # Anti-stagnation: force fallback if stuck too long
            if self._windows_on_level >= self._max_stay_windows and pos > 0:
                action = "FORCE_FALLBACK"
                self._curriculum_position -= 1
                self._windows_on_level = 0
                print(f"  ⏳ [Curriculum] FORCE FALLBACK → Level {self._curriculum_position} "
                      f"'{self.level_order[self._curriculum_position]}' "
                      f"(stuck for {self._max_stay_windows} windows, "
                      f"won {wins}/{total} = {win_rate:.0%})")
            else:
                print(f"  ➡️  [Curriculum] STAY on Level {pos} '{level_name}' "
                      f"(won {wins}/{total} = {win_rate:.0%}, "
                      f"window {self._windows_on_level}/{self._max_stay_windows})")

        # Clear the batch for the next window
        self._batch_results.clear()

    def load_level(self, preserve_power: bool = False):
        self.alive = True
        self.frame = 0
        self.game_over = False
        self.reached_goal = False

        config = self.config_manager.get_level_config(self.world)
        self.level_data = self.loader.load_level(config)

        # Clear any fireballs from the previous level — they belong to the old
        # world and should not carry over or linger across level transitions.
        self.level_data.projectiles = []

        # --- Cache spike objects for hazard_hash ---
        # Spikes are static tiles created by LevelLoader and stored only in
        # level_data.tiles / static_hash. We cache references here so step()
        # can insert them into hazard_hash every frame without scanning the
        # full tile grid. The list is rebuilt on every load_level() call
        # (new level or soft-reset) so it always matches the current map.
        self._cached_spikes = []
        for row in range(self.level_data.rows):
            for col in range(self.level_data.cols):
                if self.level_data.grid[row][col] == TILE_SPIKE:
                    tile = self.level_data.tiles[row][col]
                    if tile is not None:
                        self._cached_spikes.append(tile)

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
            self.player.coyote       = 0
            self.player.jump_hold    = 0
            self.player.jump_buffer  = 0
            self.player.input_dir    = 0
            self.player.run_held     = False
            self.player.jump_pressed = False
            # Reset fire state so a held key from last life can't instantly re-fire
            self.player.fire_requested  = False
            self.player._fire_cooldown  = 0.0

            if preserve_power:
                # Level completion — keep power state (stack + star timer).
                # Only clear the i-frame window since there's no pending hit
                # to carry over into the new level.
                self.player.power_machine._iframes_timer = 0.0
            else:
                # Death or full episode reset — wipe power state back to SMALL.
                self.player.power_machine.reset()
                self.player.powered_up       = False
                self.player.invincible_timer = 0

        self.physics_manager.reset_to_defaults()
        self.physics_manager.apply_config_dict(config)
        # Force hash rebuild on the first step of the new level.
        self.physics_manager.rebuild_dynamic_hashes(self.level_data, self._cached_spikes)

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
        Records the win for curriculum tracking, selects the next level for
        continued play, but does NOT load it yet — that happens inline in
        step() after _info() has captured the WIN event for this frame.

        The episode continues into the next level; only lives = 0 ends it.
        """
        # Record the win for the level just completed (legacy stats)
        self._level_wins[self.world] = self._level_wins.get(self.world, 0) + 1

        # ── Batch curriculum: mark win if this is the starting level ──────
        # The agent's "starting level" for this episode is level_order[_curriculum_position].
        # If it completes THAT level, the episode counts as a win for curriculum purposes.
        # Completing subsequent levels (after advancing mid-episode) doesn't count —
        # those are bonus practice, not the curriculum test.
        if self._curriculum_position < len(self.level_order):
            starting_level = self.level_order[self._curriculum_position]
            if self.world == starting_level:
                self._episode_won_current = True

        # Advance to the next level in order; wrap at the end so long episodes
        # cycle through the full curriculum without ending prematurely.
        self.current_index_world = (self.current_index_world + 1) % len(self.level_order)
        self.world = self.level_order[self.current_index_world]
        # Count the new level visit immediately so win-rates stay accurate mid-episode
        self._level_visits[self.world] = self._level_visits.get(self.world, 0) + 1

        # Signal step() to load the next level after _info() runs this frame.
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
        grid_names = ["player", "solid", "collectible", "hazard", "dijkstra"]
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
                if float(ch.max()) > 1.01:
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
                # 5 channels: player, solids, collectible, hazard, dijkstra-advantage.
                "grids": np.zeros((5, self.obs_height, self.obs_width), dtype=np.float32),
                "scalars": np.zeros(12, dtype=np.float32),  # 5 (player) + 7 (tracking)
            }

        p_obs    = self._player_obs()
        track_obs = self._tracking_obs()

        # _grid_obs_window returns the unpadded map tile coordinates of the window
        # so that _dijkstra_obs_window can slice dist_map at the identical region.
        hazard_grid, collect_grid, player_grid, solid_grid, map_row_start, map_col_start = \
            self._grid_obs_window()

        dijkstra_grid = self._dijkstra_obs_window(map_row_start, map_col_start)

        # Stack order: Player, Solids, Collectible, Hazard, Dijkstra-Advantage  (5 channels)
        # Channels 0-3 are binary {0, 1}. Channel 4 (Dijkstra) is in [-1, 1].
        stacked_grids = np.stack(
            [player_grid, solid_grid, collect_grid, hazard_grid, dijkstra_grid], axis=0
        ).astype(np.float32)

        scalars = np.concatenate([p_obs, track_obs]).astype(np.float32)

        return {
            "grids": stacked_grids,
            "scalars": scalars,
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

    def _grid_obs_window(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
        """
        Returns (hazard, collect, player, solid, map_row_start, map_col_start).

        All four binary grids share the same coordinate frame:
          player  — tiles occupied by the player bounding box
          solid   — ground, static platforms (incl. question blocks), moving platforms
          collect — coins + powerups + goals (from collectible_hash)
          hazard  — enemies + spikes (from hazard_hash)

        map_row_start / map_col_start are the unpadded map tile coordinates of
        the top-left corner of the observation window, centered on the player.
        Returned so _dijkstra_obs_window can slice dist_map at the same region.
        Coordinates can be negative when the player is near the top/left edge.
        """
        p = self.player
        if not p:
            z = np.zeros((self.obs_height, self.obs_width), dtype=np.float32)
            return z, z, z, z, 0, 0

        px = int(p.gObj.x // TILE_SIZE)
        py = int(p.gObj.y // TILE_SIZE)

        # The observation window is always centered exactly on the player.
        # We no longer clamp to map boundaries, so the agent always stays
        # perfectly centered in its own vision (translation invariance).
        map_row_start = py - self.obs_pad_y
        map_col_start = px - self.obs_pad_x

        hazard_grid  = np.zeros((self.obs_height, self.obs_width), dtype=np.float32)
        collect_grid = np.zeros((self.obs_height, self.obs_width), dtype=np.float32)
        player_grid  = np.zeros((self.obs_height, self.obs_width), dtype=np.float32)
        solid_grid   = np.zeros((self.obs_height, self.obs_width), dtype=np.float32)

        # World-pixel origin of the window (used for entity spatial-hash queries)
        wx = map_col_start * TILE_SIZE
        wy = map_row_start * TILE_SIZE

        window_rect = pygame.Rect(wx, wy, self.obs_width * TILE_SIZE, self.obs_height * TILE_SIZE)

        # --- Hazard channel: enemies + spikes (from hazard_hash) ---
        nearby_hazards = self.physics_manager.hazard_hash.query_rect(
            window_rect.x, window_rect.y, window_rect.width, window_rect.height
        )
        for h in nearby_hazards:
            if not h.gObj.active: continue
            hx = int(h.gObj.x // TILE_SIZE)
            hy = int(h.gObj.y // TILE_SIZE)
            local_x = hx - map_col_start
            local_y = hy - map_row_start
            if 0 <= local_x < self.obs_width and 0 <= local_y < self.obs_height:
                hazard_grid[local_y, local_x] = 1.0

        # --- Collectible channel: coins + powerups + goals (from collectible_hash) ---
        nearby_items = self.physics_manager.collectible_hash.query_rect(
            window_rect.x, window_rect.y, window_rect.width, window_rect.height
        )
        for c in nearby_items:
            if not c.gObj.active: continue
            if hasattr(c, 'collected') and c.collected: continue
            cx = int(c.gObj.x // TILE_SIZE)
            cy = int(c.gObj.y // TILE_SIZE)
            local_x = cx - map_col_start
            local_y = cy - map_row_start
            if 0 <= local_x < self.obs_width and 0 <= local_y < self.obs_height:
                collect_grid[local_y, local_x] = 1.0

        # --- Solid channel: static geometry (ground, platforms, qblocks) ---
        # Query static_hash — covers all immovable geometry inserted at load time.
        # Spikes live in static_hash too but belong in the hazard channel, so skip them.
        nearby_solids = self.level_data.static_hash.query_rect(
            window_rect.x, window_rect.y, window_rect.width, window_rect.height
        )
        for obj in nearby_solids:
            if isinstance(obj, Spike): continue   # spikes → hazard channel
            sx = int(obj.gObj.x // TILE_SIZE)
            sy = int(obj.gObj.y // TILE_SIZE)
            lx = sx - map_col_start
            ly = sy - map_row_start
            if 0 <= lx < self.obs_width and 0 <= ly < self.obs_height:
                solid_grid[ly, lx] = 1.0

        # Moving platforms (from platform_hash — positions update every frame).
        # platform_hash is rebuilt each step() so positions are always current.
        # Multi-tile platforms span several columns — iterate all covered tiles.
        nearby_plats = self.physics_manager.platform_hash.query_rect(
            window_rect.x, window_rect.y, window_rect.width, window_rect.height
        )
        for plat in nearby_plats:
            if not plat.gObj.active: continue
            pc0 = int(plat.gObj.x // TILE_SIZE)
            pc1 = int((plat.gObj.x + plat.gObj.width - 1) // TILE_SIZE) + 1
            pr  = int(plat.gObj.y // TILE_SIZE)
            for pc in range(pc0, pc1):
                lx = pc - map_col_start
                ly = pr - map_row_start
                if 0 <= lx < self.obs_width and 0 <= ly < self.obs_height:
                    solid_grid[ly, lx] = 1.0

        # --- Player channel ---
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

        return hazard_grid, collect_grid, player_grid, solid_grid, map_row_start, map_col_start

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

            All patching (moving-platform compensation) is done in raw cost space.
            Normalisation and clipping to [-1, 1] happen exactly once at the end.

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

            # Compute raw relative advantage (NOT normalised yet).
            # Positive = closer to goal than player, negative = further.
            delta = player_dist - dist_slice

            # Walls / unreachable tiles have cost=inf → delta = -inf → set to 0 (neutral).
            # Done BEFORE writing to out so inf values never enter the output array.
            delta[~np.isfinite(delta)] = 0.0

            # Place the valid slice into the full output window, leaving edge-padding
            # as zeros when the window extends outside the map.
            out = np.zeros((self.obs_height, self.obs_width), dtype=np.float32)
            out_r0 = r0 - map_row_start   # offset into output array
            out_c0 = c0 - map_col_start
            out[out_r0 : out_r0 + valid_rows, out_c0 : out_c0 + valid_cols] = delta

            # -----------------------------------------------------------------
            # Moving-platform compensation (raw cost space)
            # -----------------------------------------------------------------
            # Mirrors the DijkstraSolver's treatment of static solid tiles:
            #   - Platform surface tiles → 0.0 (impassable, same as walls)
            #   - 1 tile above → ground-proximity discount -0.6
            #   - 2 tiles above → -0.25
            #   - 3 tiles above → -0.1
            #
            # For tiles that are unreachable in the static map (inf cost),
            # we interpolate from the nearest reachable tile in the same row
            # so the gradient connects across the gap the platform bridges.
            # -----------------------------------------------------------------
            if self.level_data.moving_platforms:
                wx = map_col_start * TILE_SIZE
                wy = map_row_start * TILE_SIZE
                ww = self.obs_width  * TILE_SIZE
                wh = self.obs_height * TILE_SIZE

                nearby_plats = self.physics_manager.platform_hash.query_rect(
                    wx, wy, ww, wh
                )

                if nearby_plats:
                    HORIZ_COST = 2.0
                    MAX_SCAN   = self.obs_width * 2
                    dm         = self.dijkstra.dist_map
                    dm_rows, dm_cols = dm.shape

                    # (rows_above_surface, discount) — matches compute_map exactly
                    PROXIMITY = [(1, 0.6), (2, 0.25), (3, 0.1)]

                    for plat in nearby_plats:
                        if not plat.gObj.active:
                            continue

                        pc0       = int(plat.gObj.x // TILE_SIZE)
                        pc1       = int((plat.gObj.x + plat.gObj.width - 1) // TILE_SIZE) + 1
                        p_top_row = int(plat.gObj.y // TILE_SIZE)

                        # --- Platform surface: impassable (same as walls) ---
                        for pc in range(pc0, pc1):
                            ly = p_top_row - map_row_start
                            lx = pc - map_col_start
                            if 0 <= ly < self.obs_height and 0 <= lx < self.obs_width:
                                out[ly, lx] = 0.0

                        # --- Tiles above: ground-proximity discounts ---
                        for offset, discount in PROXIMITY:
                            patch_row = p_top_row - offset
                            if patch_row < 0 or patch_row >= dm_rows:
                                continue

                            for pc in range(pc0, pc1):
                                ly = patch_row - map_row_start
                                lx = pc - map_col_start
                                if not (0 <= ly < self.obs_height and 0 <= lx < self.obs_width):
                                    continue

                                if 0 <= pc < dm_cols:
                                    raw_cost = float(dm[patch_row, pc])
                                else:
                                    raw_cost = np.inf

                                if np.isfinite(raw_cost):
                                    # Tile reachable in static map but missing the
                                    # ground-proximity discount. Recalculate advantage.
                                    patched_cost = max(1.0, raw_cost - discount)
                                    out[ly, lx] = player_dist - patched_cost
                                else:
                                    # Tile unreachable — interpolate from nearest
                                    # reachable tile in the same row of full dist_map.
                                    best_est = np.inf
                                    for dc in range(1, MAX_SCAN + 1):
                                        for sign in (-1, 1):
                                            nc = pc + dc * sign
                                            if 0 <= nc < dm_cols:
                                                anchor = float(dm[patch_row, nc])
                                                if np.isfinite(anchor):
                                                    est = anchor + dc * HORIZ_COST - discount
                                                    if est < best_est:
                                                        best_est = est
                                        if np.isfinite(best_est):
                                            break

                                    if np.isfinite(best_est):
                                        best_est = max(1.0, best_est)
                                        out[ly, lx] = player_dist - best_est

            # --- Single normalisation pass (after ALL patches) ---
            max_cost = max(
                (self.obs_width  // 2) * 2.0,   # horizontal half-span × horiz cost
                (self.obs_height // 2) * 3.5,   # vertical half-span × upward cost
            )
            np.clip(out / max_cost, -1.0, 1.0, out=out)
            self._dijkstra_window_cache = out  # cache for debugging visualization
            return out.astype(np.float32)

    def _tracking_obs(self) -> np.ndarray:
        """
        Returns 7 scalar features used by the MLP branch of the extractor.
        Combined with 5 from _player_obs → 12 total scalars.
        """
        p = self.player
        if not p: return np.zeros(7, dtype=np.float32)  # FIXED: 7 active scalars

        SEARCH_RADIUS = math.sqrt(
            (self.obs_width  * TILE_SIZE) ** 2 +
            (self.obs_height * TILE_SIZE) ** 2
        ) * 0.5

        def get_dist_hash(hash_obj, check_collected=False, skip_spikes=False):
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
                if skip_spikes and getattr(obj.gObj, 'type_id', None) == EntityType.SPIKE: continue
                dx = p.gObj.x - obj.gObj.x
                dy = p.gObj.y - obj.gObj.y
                d_sq = dx*dx + dy*dy
                if d_sq < min_d:
                    min_d = d_sq
                count += 1
            return (math.sqrt(min_d) if min_d < 9999.0 else 9999.0), count

        e_dist, e_count = get_dist_hash(self.physics_manager.hazard_hash, skip_spikes=True)
        c_dist, c_count = get_dist_hash(self.physics_manager.collectible_hash, check_collected=True)

        raw_goal_dist = getattr(self, '_goal_dist_cache', self._get_dist_to_goal())
        norm_dist = max(self.level_data.width, self.level_data.height, 1.0)

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

        dijkstra_dist = 1.0  # default: treat as far if unavailable
        if self.dijkstra:
            px_tile = int(p.gObj.x // TILE_SIZE)
            py_tile = int(p.gObj.y // TILE_SIZE)
            d = self.dijkstra.get_dist(px_tile, py_tile)
            if d >= 0:
                dijkstra_dist = np.clip(d / (self.level_data.cols * 2), 0.0, 1.0)

        step_dx, step_dy = 0.0, 0.0
        if self.dijkstra:
            px_tile = int(p.gObj.x // TILE_SIZE)
            py_tile = int(p.gObj.y // TILE_SIZE)

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

            mag = math.sqrt(best_ddx * best_ddx + best_ddy * best_ddy)
            if mag > 0:
                step_dx = best_ddx / mag
                step_dy = best_ddy / mag

        self._step_dx = step_dx
        self._step_dy = step_dy

        return np.array([
            np.clip(e_dist       / norm_dist, 0.0, 1.0),
            np.clip(raw_goal_dist/ norm_dist, 0.0, 1.0),
            np.clip(self.timer   / max(1.0, self.timer_seconds), 0.0, 1.0),
            dist_y_norm,
            dijkstra_dist,
            step_dx,
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

        if not p:
            return {
                "score": self.score, "score_delta": self.score_delta, "frame_count": self.frame,
                "x_position": 0.0, "y_position": 0.0, "velocity_x": 0.0, "velocity_y": 0.0,
                "coins_collected": self.coins_total, "enemies_killed_step": self.kills_step,
                "powered_up": False, "terminated": not self.alive, "won": False,
                "action": self._last_action, "time_left": math.ceil(self.timer),
                "max_x_seen": self.max_x_seen, "stall_windows": self.stall_windows_count,
                "stalled": self.stalled_this_frame, "persona": self.persona,
                "level": self.world, "goal_dist": 0.0, "lives": self.lives,
                "event": event, "cause": cause,
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
                dijkstra_dist = np.clip(d / (self.level_data.cols * 2), 0.0, 1.0)
            else:
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
            "level": self.world,
            "goal_dist": getattr(self, '_goal_dist_cache', self._get_dist_to_goal()) / ts,
            "lives" : self.lives,
            "event": event,
            "cause": cause,
            "dijkstra_dist": dijkstra_dist,
            "on_ground": p.on_ground,
            "step_dx":   self._step_dx,
            "step_dy":   self._step_dy,
            # Batch curriculum state
            "curriculum_position": self._curriculum_position,
            "batch_progress": f"{len(self._batch_results)}/{self._batch_window}",
            "batch_wins": sum(1 for r in self._batch_results if r),
            "windows_on_level": self._windows_on_level,
            **self._obs_stats
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

            # Draw sensor rays — colour-coded, respects F1 toggle
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

            if isinstance(tile, Spike):
                tile.render(surface, tile.gObj.x - self.camera_x, tile.gObj.y - self.camera_y)
            elif isinstance(tile, Tile):
                if tile.color == COLOR_QBLOCK: continue
                tile.render(surface, self.camera_x, self.camera_y)
            elif isinstance(tile, QuestionBlock):
                tile.render(surface, tile.x - self.camera_x, tile.y - self.camera_y)
            elif hasattr(tile, 'render'):
                try:
                    tile.render(surface, self.camera_x, self.camera_y)
                except TypeError:
                    tile.render(surface, tile.x - self.camera_x, tile.y - self.camera_y)

        visible_platforms = self.physics_manager.platform_hash.query_rect(
            self.camera_x, self.camera_y, self.WIDTH, self.HEIGHT
        )
        for plat in visible_platforms:
            if plat.gObj.active:
                plat.render(surface, plat.gObj.x - self.camera_x, plat.gObj.y - self.camera_y)

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

        for proj in self.level_data.projectiles:
            if proj.gObj.active:
                proj.render(
                    surface,
                    proj.gObj.x - cx,
                    proj.gObj.y - cy,
                )

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