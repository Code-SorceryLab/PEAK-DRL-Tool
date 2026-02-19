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
        Flood fills from all goal positions to generate the distance map.
        Applies weighted costs to prefer walking on ground and collecting coins.
        """
        if coins is None:
            coins = set()

        # Priority Queue: (current_dist, x, y)
        pq = []

        # Initialize goals with distance 0
        for gx, gy in goals:
            if 0 <= gx < self.cols and 0 <= gy < self.rows:
                self.dist_map[gy][gx] = 0.0
                heapq.heappush(pq, (0.0, gx, gy))

        directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, -1), (-1, -1), (1, 1), (-1, 1)] # 8-way

        while pq:
            current_dist, cx, cy = heapq.heappop(pq)

            # If we found a shorter path already, skip
            if current_dist > self.dist_map[cy][cx]:
                continue

            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy

                if 0 <= nx < self.cols and 0 <= ny < self.rows:
                    # Check walkability of TARGET tile (nx, ny)
                    tile = self.grid[ny][nx]

                    # FIX: Strict Walkability. Only AIR and GOAL are walkable nodes.
                    # Spikes are technically walkable but dangerous.
                    # Solids (Ground, Platform, QBlock) are obstacles.
                    is_walkable = (tile == TILE_AIR or tile == TILE_GOAL or tile == TILE_SPIKE)

                    if is_walkable:
                        # Base Cost
                        step_cost = 2.0

                        # Spike Penalty (High Cost)
                        if tile == TILE_SPIKE:
                            step_cost = 20.0

                        # --- COIN ATTRACTION ---
                        # If the target tile contains a coin, make it cheaper to enter
                        if (nx, ny) in coins:
                            step_cost -= 1.0

                        # --- GROUND ATTRACTION LOGIC ---
                        # Check tile DIRECTLY BELOW the target node (ny+1, nx)
                        # If it is solid, make entering this node cheaper (gravity preference).
                        if ny + 1 < self.rows:
                            below_tile = self.grid[ny + 1][nx]
                            if below_tile in [TILE_GROUND, TILE_PLATFORM, TILE_QBLOCK]:
                                step_cost -= 0.5 # Huge bonus for being on ground

                        # Check 2 tiles below (ny+2)
                        elif ny + 2 < self.rows:
                            below_2 = self.grid[ny + 2][nx]
                            if below_2 in [TILE_GROUND, TILE_PLATFORM, TILE_QBLOCK]:
                                step_cost -= 0.2 # Small bonus for being near ground

                        # Check 3 tiles below (ny+3)
                        elif ny + 3 < self.rows:
                            below_3 = self.grid[ny + 3][nx]
                            if below_3 in [TILE_GROUND, TILE_PLATFORM, TILE_QBLOCK]:
                                step_cost -= 0.1 # Tiny bonus

                        # Ensure cost doesn't go negative or too low (min 1.0)
                        step_cost = max(1.0, step_cost)

                        new_dist = current_dist + step_cost

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
            "grid_solid_mean": 0.0, "grid_solid_std": 0.0,
            "grid_solid_min": 0.0, "grid_solid_max": 0.0,
            "grid_hazard_mean": 0.0, "grid_hazard_std": 0.0,
            "grid_hazard_min": 0.0, "grid_hazard_max": 0.0,
            "grid_collectible_mean": 0.0, "grid_collectible_std": 0.0,
            "grid_collectible_min": 0.0, "grid_collectible_max": 0.0,
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

        self._obs_space = spaces.Dict({
            # 4 Channels (Player, Solid, Hazard, Collectible)
            "grids": spaces.Box(low=0.0, high=1.0, shape=(4, self.obs_height, self.obs_width), dtype=np.float32),

            # Scalars: 16 (Player=5, Tracking=11)
            "scalars": spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32),

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
        self.physics_manager.rebuild_dynamic_hashes(self.level_data)

        if self.player:
            if not self.debug_manager.free_cam_active:
                self.player.handle_input(a = int(action))
            else:
                self.player.vx = 0; self.player.jump_hold = 0

        self.physics_manager.update_system(self.dt, self)
        self.physics_manager.resolve_collisions(self)

        # Cleanup Inactive Entities
        self.level_data.enemies[:] = [e for e in self.level_data.enemies if e.gObj.active]
        self.level_data.coins[:] = [c for c in self.level_data.coins if c.gObj.active]
        self.level_data.powerups[:] = [p for p in self.level_data.powerups if p.gObj.active]

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
        if terminated: info["episode_end"] = True

        # Return raw score delta as base reward.
        # The GameEnv wrapper (generic_env.py) applies the actual persona reward fn.
        base_reward = float(self.score_delta)

        if self.coins_step > 0:
            self._calculate_dijkstra_map()

        obs = self._obs()
        self._check_obs_sanity(obs)
        return obs, base_reward, bool(terminated), bool(truncated), info

    def reset(self, seed=None, options=None) -> np.ndarray:
        super().reset(seed=seed)

        was_win = self.reached_goal

        if not was_win:
            # Death/truncation — restart from level 0
            self.reset_metrics()
            self.current_index_world = 0
            self.world = self.level_order[self.current_index_world]
        else:
            # Level completed — world is already set by complete_level()
            # Just reset per-episode metrics, keep lives/score
            self.reached_goal = False
            self.game_over = False
            self.death_cause = ""

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

        # FIX: Stripped AnimationHandler entirely from Core instantiation
        self.player = Player(gObj=GameObject(px, py, PLATFORMER_WIDTH, PLATFORMER_HEIGHT, True))
        self.player.__post_init__()

        self.physics_manager.reset_to_defaults()
        self.physics_manager.apply_config_dict(config)
        self.physics_manager.rebuild_dynamic_hashes(self.level_data)

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
        # Pick next level but do NOT call load_level() yet.
        # reached_goal=True is already set by the caller (PhysicsManager).
        # _info() needs to read reached_goal to emit "WIN" this frame.
        # reset() will call load_level() when SB3 starts the next episode.
        # Random walk: move -1, 0, or +1 from current level (curriculum learning)
        self.current_index_world = max(0, random.randint(self.current_index_world - 1, self.current_index_world + 1))
        if self.current_index_world >= len(self.level_order):
            self.current_index_world = len(self.level_order) - 1
        self.world = self.level_order[self.current_index_world]
        # Mark episode as over — SB3 will call env.reset() after terminated=True
        self.game_over = True

    def _handle_death(self, cause: str = "Unknown") -> bool:
        self.death_cause = cause
        self.lives = max(0, self.lives - 1)
        if self.lives > 0:
            self._soft_reset()
            return False
        else:
            # Mark game over — do NOT call reset() here.
            # _info() reads these flags to emit the DIED event.
            # SB3 calls env.reset() after receiving terminated=True.
            self.alive = False
            self.game_over = True
            return True

    def _soft_reset(self):
        current_lives = self.lives
        self.load_level()
        self.lives = current_lives

    def _update_camera(self):
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

        current_dist = self._get_dist_to_goal()
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
        player = self.player
        if not player:
            return True

        # 1. TIME LIMIT
        if self.use_timer and self.timer <= 0:
            return self._handle_death("Timeout")

        # 2. PIT DEATH
        if player.gObj.y > self.level_data.height:
            return self._handle_death("Pit")

        # 3. Goal and spike collisions are handled by PhysicsManager.resolve_collisions
        # via the collectible_hash and hazard_hash — no need to duplicate here.
        # Check if PhysicsManager already triggered a win this frame.
        if self.reached_goal:
            return True

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
        grid_names = ["player", "solid", "hazard", "collectible"]
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
                "grids": np.zeros((4, self.obs_height, self.obs_width), dtype=np.float32),
                "scalars": np.zeros(16, dtype=np.float32), # 16 scalars
                # "raycasts": np.zeros(self.num_rays * 2, dtype=np.float32)
            }

        p_obs = self._player_obs()
        # Ensure _grid_obs_window returns 4 grids now
        solid_grid, hazard_grid, collect_grid, player_grid = self._grid_obs_window()
        track_obs = self._tracking_obs()
        # rays = self._perform_raycasts()

        # Stack order: Player (Top), Solid, Hazard, Collectible
        stacked_grids = np.stack([player_grid, solid_grid, hazard_grid, collect_grid], axis=0).astype(np.float32)
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

    def _grid_obs_window(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        p = self.player
        if not p:
            z = np.zeros((self.obs_height, self.obs_width), dtype=np.float32)
            return z, z, z, z # Return 4 zeros if no player

        px = int(p.gObj.x // TILE_SIZE)
        py = int(p.gObj.y // TILE_SIZE)

        # Centre the window on the player
        slice_y_start = py - self.obs_pad_y
        slice_x_start = px - self.obs_pad_x

        # Safety clamp
        max_h, max_w = self.padded_solid.shape
        slice_y_start = max(0, min(slice_y_start, max_h - self.obs_height))
        slice_x_start = max(0, min(slice_x_start, max_w - self.obs_width))
        slice_y_end = slice_y_start + self.obs_height
        slice_x_end = slice_x_start + self.obs_width

        solid_grid = self.padded_solid[slice_y_start:slice_y_end, slice_x_start:slice_x_end]

        hazard_grid = np.zeros((self.obs_height, self.obs_width), dtype=np.float32)
        collect_grid = np.zeros((self.obs_height, self.obs_width), dtype=np.float32)
        player_grid = np.zeros((self.obs_height, self.obs_width), dtype=np.float32) # Init player grid

        # Calculate window origin in world-space pixels
        wx = slice_x_start * TILE_SIZE
        wy = slice_y_start * TILE_SIZE

        window_rect = pygame.Rect(
            wx, wy,
            self.obs_width * TILE_SIZE,
            self.obs_height * TILE_SIZE
        )

        nearby_hazards = self.physics_manager.hazard_hash.query_rect(window_rect.x, window_rect.y, window_rect.width, window_rect.height)
        for h in nearby_hazards:
            if not h.gObj.active: continue
            hx, hy = int(h.gObj.x // TILE_SIZE), int(h.gObj.y // TILE_SIZE)
            local_x = hx - slice_x_start
            local_y = hy - slice_y_start
            if 0 <= local_x < self.obs_width and 0 <= local_y < self.obs_height:
                hazard_grid[local_y, local_x] = 1.0

        nearby_items = self.physics_manager.collectible_hash.query_rect(window_rect.x, window_rect.y, window_rect.width, window_rect.height)
        for c in nearby_items:
            if not c.gObj.active: continue
            if hasattr(c, 'collected') and c.collected: continue

            cx, cy = int(c.gObj.x // TILE_SIZE), int(c.gObj.y // TILE_SIZE)
            local_x = cx - slice_x_start
            local_y = cy - slice_y_start
            if 0 <= local_x < self.obs_width and 0 <= local_y < self.obs_height:
                collect_grid[local_y, local_x] = 1.0

        # --- FILL PLAYER GRID ---
        p_rect = p.gObj.get_rect()
        rel_x = p_rect.x - wx
        rel_y = p_rect.y - wy
        p_start_col = int(rel_x // TILE_SIZE)
        p_start_row = int(rel_y // TILE_SIZE)
        p_w_tiles = math.ceil(p.gObj.width / TILE_SIZE)
        p_h_tiles = math.ceil(p.gObj.height / TILE_SIZE)

        for r in range(p_start_row, p_start_row + p_h_tiles):
            for c in range(p_start_col, p_start_col + p_w_tiles):
                if 0 <= r < self.obs_height and 0 <= c < self.obs_width:
                    player_grid[r, c] = 1.0

        return solid_grid, hazard_grid, collect_grid, player_grid

    def _tracking_obs(self) -> np.ndarray:
        p = self.player
        if not p: return np.zeros(11, dtype=np.float32)

        def get_dist(obj_list):
            min_d = 9999.0
            count = 0
            for obj in obj_list:
                if not obj.gObj.active: continue
                if hasattr(obj, 'collected') and obj.collected: continue
                d = math.sqrt((p.gObj.x - obj.gObj.x)**2 + (p.gObj.y - obj.gObj.y)**2)
                if d < min_d: min_d = d
                count += 1
            return min_d, count

        e_dist, e_count = get_dist(self.level_data.enemies)
        c_dist, c_count = get_dist(self.level_data.coins)

        raw_goal_dist = self._get_dist_to_goal()
        norm_dist = max(self.level_data.width, self.level_data.height, 1.0)

        # --- NEW: DIRECTION CALCULATIONS ---
        closest_goal_x = p.gObj.x + raw_goal_dist # Approx
        if self.level_data.goals:
             # Find actual closest goal for accurate direction
             closest = min(self.level_data.goals, key=lambda g: abs(g.gObj.x - p.gObj.x))
             closest_goal_x = closest.gObj.x
             closest_goal_y = closest.gObj.y
        else:
             closest_goal_x = self.level_data.width
             closest_goal_y = p.gObj.y

        dx = closest_goal_x - p.gObj.x
        dy = closest_goal_y - p.gObj.y

        dir_x = np.sign(dx) # -1, 0, or 1
        dist_y_norm = np.clip(dy / self.level_data.height, -1.0, 1.0) # Y-distance normalized

        # --- Dijkstra Signal ---
        dijkstra_dist = 0.0
        if self.dijkstra:
            px = int(p.gObj.x // TILE_SIZE)
            py = int(p.gObj.y // TILE_SIZE)
            d = self.dijkstra.get_dist(px, py)
            if d >= 0:
                # Normalize based on map width (heuristic max cost)
                dijkstra_dist = np.clip(d / (self.level_data.cols * 2), 0.0, 1.0)
            else:
                dijkstra_dist = 1.0 # Unreachable or off-grid treated as max dist

        # 11 Elements Total
        return np.array([
            np.clip(e_dist / norm_dist, 0.0, 1.0),
            np.clip(c_dist / norm_dist, 0.0, 1.0),
            np.clip(raw_goal_dist / norm_dist, 0.0, 1.0),
            np.clip(e_count / 20.0, 0.0, 1.0),
            np.clip(c_count / 50.0, 0.0, 1.0),
            np.clip(self.score / 10000.0, 0.0, 1.0),
            np.clip(self.timer / max(1.0, self.timer_seconds), 0.0, 1.0),
            self.lives / float(max(1.0, self.max_lives)),
            dir_x,
            dist_y_norm,
            dijkstra_dist
        ], dtype=np.float32)

    def _perform_raycasts(self) -> np.ndarray:
        """
        Casts rays from player center.
        Returns: [dist1, type1, dist2, type2, ...]
        Type IDs: 0=Empty, 1=Solid, 2=Hazard, 3=Item
        """
        p = self.player
        if not p: return np.zeros(self.num_rays * 2, dtype=np.float32)

        cx, cy = p.gObj.x + p.gObj.width/2, p.gObj.y + p.gObj.height/2
        results = []
        self.last_rays = [] # Debug

        step_size = TILE_SIZE / 2.0

        for angle in self.ray_angles:
            sin_a = math.sin(angle)
            cos_a = math.cos(angle)

            hit_dist = 1.0 # Max Range default
            hit_type = 0.0 # Empty default

            curr_dist = 0.0
            found = False

            while curr_dist < self.ray_max_dist:
                curr_dist += step_size
                tx = cx + cos_a * curr_dist
                ty = cy + sin_a * curr_dist

                # Check Bounds
                if tx < 0 or tx >= self.level_data.width or ty < 0 or ty >= self.level_data.height:
                    hit_dist = curr_dist / self.ray_max_dist
                    hit_type = 0.0 # USER REQ: Count as Empty Space
                    found = True
                    break

                # 1. Check Static Grid
                col, row = int(tx // TILE_SIZE), int(ty // TILE_SIZE)
                if 0 <= row < len(self.level_data.grid) and 0 <= col < len(self.level_data.grid[0]):
                    tile_val = self.level_data.grid[row][col]
                    if tile_val != TILE_AIR:
                        if tile_val == TILE_SPIKE:
                            hit_dist = curr_dist / self.ray_max_dist
                            hit_type = 2.0 # Hazard
                            found = True; break
                        elif tile_val in [TILE_GROUND, TILE_PLATFORM, TILE_QBLOCK]:
                            hit_dist = curr_dist / self.ray_max_dist
                            hit_type = 1.0 # Solid
                            found = True; break

                # 2. Check Entities
                # Use a small rect for point query
                query_rect = pygame.Rect(tx-2, ty-2, 4, 4)

                hazards = self.physics_manager.hazard_hash.query_rect(tx-2, ty-2, 4, 4)
                for h in hazards:
                    if h.gObj.get_rect().collidepoint(tx, ty):
                        hit_dist = curr_dist / self.ray_max_dist
                        hit_type = 2.0 # Hazard
                        found = True; break
                if found: break

                items = self.physics_manager.collectible_hash.query_rect(tx-2, ty-2, 4, 4)
                for i in items:
                    if hasattr(i, 'collected') and i.collected: continue
                    if i.gObj.get_rect().collidepoint(tx, ty):
                        hit_dist = curr_dist / self.ray_max_dist
                        hit_type = 3.0 # Item
                        found = True; break
                if found: break

            hit_dist = min(1.0, hit_dist)
            results.extend([hit_dist, hit_type])

            # Debug Visuals
            end_x = cx + cos_a * (curr_dist if found else self.ray_max_dist)
            end_y = cy + sin_a * (curr_dist if found else self.ray_max_dist)
            self.last_rays.append(((cx, cy), (end_x, end_y), found, hit_type))

        return np.array(results, dtype=np.float32)

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
                **self._obs_stats
            }

        dijkstra_dist = 0.0
        if self.dijkstra:
            px = int(p.gObj.x // TILE_SIZE)
            py = int(p.gObj.y // TILE_SIZE)
            d = self.dijkstra.get_dist(px, py)
            if d >= 0:
                dijkstra_dist = np.clip(d / (self.level_data.cols * 2), 0.0, 1.0)
            else:
                dijkstra_dist = 1.0

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
            "goal_dist": self._get_dist_to_goal() / ts,
            "lives" : self.lives,
            "event": event,
            "cause": cause,
            "dijkstra_dist": dijkstra_dist, # Used for delta_dijkstra reward
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

            # Draw sensor rays â€” colour-coded, respects F1 toggle
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
