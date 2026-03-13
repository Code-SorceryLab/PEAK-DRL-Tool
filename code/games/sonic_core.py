"""
sonic_core.py
=============
A Sonic the Hedgehog NES-style clone built on the same Gymnasium environment
architecture as platformer_core.py.

Key gameplay differences from the platformer:
  - Sonic-style momentum physics (high speed, low friction, rolling)
  - Spin Dash mechanic (charge + release for burst speed)
  - Ring system instead of coins (rings scatter on hit, die only with 0 rings)
  - Badnik enemies (defeated by ball attack, hurt if touching normally)
  - Springs that launch Sonic
  - Green Hill Zone visual style
  - Speed-based scoring

This file is a STANDALONE copy — the original platformer_core.py is untouched.
"""

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

# ── Imports: shared infrastructure from the existing package ─────────────────
try:
    from .modules.System.EntityType import EntityType
    from .modules.Objects.GameObject import GameObject
    from .modules.Objects.Tile import Tile, create_tile
    from .modules.Objects.Enemy import Enemy
    from .modules.Objects.Coin import Coin
    from .modules.Objects.Goal import Goal
    from .modules.Objects.Spike import Spike
    from .modules.Objects.MovingPlatform import MovingPlatform
    from .modules.System.LevelLoader import LevelLoader, LevelData
    
    # Import Base Context and our NEW Sonic Physics Manager
    from .modules.System.PhysicsManager import PhysicsContext
    from .modules.System.SonicPhysicsManager import SonicPhysicsManager
    
    from .modules.System.config_manager import ConfigManager
    from .modules.System.SpatialHash import SpatialHash
    from .modules.System.debugging_mods.manager import DebugManager

    from .modules.Objects.SonicPlayer import SonicPlayer, SonicState
    from .modules.Objects.Ring import Ring
    from .modules.Objects.Badnik import Badnik, BadnikType
    from .modules.Objects.Spring import Spring, SpringType, SpringDir
    from .modules.Objects.SlopeTile import SlopeTile

    from .modules.Parameters.Sonic_Map_parameters import (
        TILE_AIR, TILE_GROUND, TILE_PLATFORM, TILE_GOAL, TILE_SPIKE,
        TILE_SPRING, TILE_CHECKPOINT, TILE_PIT,
        COLOR_SKY, COLOR_GROUND, COLOR_GROUND_CHECK, COLOR_GRASS_TOP,
        COLOR_PLATFORM, COLOR_GOAL, COLOR_SPIKE, COLOR_WHITE, COLOR_BLACK,
        COLOR_RING, COLOR_BADNIK, COLOR_SONIC_BLUE, COLOR_HUD_BG,
        COLOR_EMPTY, COLOR_ENEMY, COLOR_HITBOX, COLOR_SENSOR,
        COLOR_AGENT_PANEL, COLOR_STREAK, COLOR_POWERUP_MUSH,
        COLOR_POWERUP_STAR, COLOR_COIN, TILE_SIZE
    )
except ImportError:
    # ── Fallback: standalone / flat-directory mode ────────────────────────────
    from .modules.System.EntityType import EntityType
    from .modules.Objects.GameObject import GameObject
    from .modules.Objects.Tile import Tile, create_tile
    from .modules.Objects.Enemy import Enemy
    from .modules.Objects.Coin import Coin
    from .modules.Objects.Goal import Goal
    from .modules.Objects.Spike import Spike
    from .modules.Objects.MovingPlatform import MovingPlatform
    from .modules.System.LevelLoader import LevelLoader, LevelData
    
    from .modules.System.PhysicsManager import PhysicsContext
    from .modules.System.SonicPhysicsManager import SonicPhysicsManager
    
    from .modules.System.config_manager import ConfigManager
    from .modules.System.SpatialHash import SpatialHash
    from .modules.Objects.SonicPlayer import SonicPlayer, SonicState
    from .modules.Objects.Ring import Ring
    from .modules.Objects.Badnik import Badnik, BadnikType
    from .modules.Objects.Spring import Spring, SpringType, SpringDir
    from .modules.Objects.SlopeTile import SlopeTile
    from .modules.Parameters.Sonic_Map_parameters import *

    class DebugManager:
        def __init__(self, **kw):
            self.show_sensors = False
            self.show_obs_panel = False
            self.free_cam_active = False
            self.slow_motion = False
            self.current_cam_move = (0, 0)
        def update_input(self): pass
        def render_overlays(self, *a): pass
        def print_help_text(self): pass


# =============================================================================
# Dijkstra Pathfinding (reused from platformer_core)
# =============================================================================
class DijkstraSolver:
    """Flood-fill distance map from goals for RL observation."""
    def __init__(self, grid, rows, cols):
        self.grid = grid
        self.rows = rows
        self.cols = cols
        self.dist_map = np.full((rows, cols), float('inf'), dtype=np.float32)

    def compute_map(self, goals, coins=None):
        if coins is None:
            coins = set()
        SOLID = {TILE_GROUND, TILE_PLATFORM}
        pq = []
        for gx, gy in goals:
            if 0 <= gx < self.cols and 0 <= gy < self.rows:
                self.dist_map[gy][gx] = 0.0
                heapq.heappush(pq, (0.0, gx, gy))

        directions = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1)]
        while pq:
            dist, cx, cy = heapq.heappop(pq)
            if dist > self.dist_map[cy][cx]:
                continue
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < self.cols and 0 <= ny < self.rows):
                    continue
                tile = self.grid[ny][nx]
                if tile not in (TILE_AIR, TILE_GOAL, TILE_SPIKE):
                    continue
                if dy < 0:
                    step = 3.5
                elif dy > 0:
                    step = 1.2
                else:
                    step = 2.0
                if dx != 0 and dy != 0:
                    step *= 1.1
                if tile == TILE_SPIKE:
                    step += 18.0
                if (nx, ny) in coins:
                    step -= 0.8
                if ny + 1 < self.rows and self.grid[ny + 1][nx] in SOLID:
                    step -= 1.0
                if ny + 2 < self.rows and self.grid[ny + 2][nx] in SOLID:
                    step -= 0.5
                step = max(1.0, step)
                new_dist = dist + step
                if new_dist < self.dist_map[ny][nx]:
                    self.dist_map[ny][nx] = new_dist
                    heapq.heappush(pq, (new_dist, nx, ny))

    def get_dist(self, x, y):
        if 0 <= x < self.cols and 0 <= y < self.rows:
            d = self.dist_map[y][x]
            return d if d != float('inf') else -1.0
        return -1.0


# =============================================================================
# Screen geometry
# =============================================================================
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
DEBUG_PANEL_WIDTH = 350

# Action encoding: [move, jump, down]
MD_MOVE_NAMES = {0: "IDLE", 1: "LEFT", 2: "SPRINT_L", 3: "RIGHT", 4: "SPRINT_R"}
MD_JUMP_NAMES = {0: "", 1: "JUMP"}
MD_DOWN_NAMES = {0: "", 1: "DOWN"}

def action_to_str(a) -> str:
    try:
        move, jump, down = int(a[0]), int(a[1]), int(a[2])
        parts = [MD_MOVE_NAMES.get(move, "?")]
        if jump: parts.append("JUMP")
        if down: parts.append("DOWN")
        return "+".join(p for p in parts if p)
    except Exception:
        return str(a)

ACTION_NAMES = {i: action_to_str([i % 5, (i // 5) % 2, i // 10]) for i in range(20)}


# =============================================================================
# SONIC CORE — The main Gymnasium environment
# =============================================================================
class SonicCore(gymnasium.Env):
    """
    Sonic the Hedgehog NES Clone — Gymnasium environment.
    """
    WIDTH, HEIGHT = SCREEN_WIDTH, SCREEN_HEIGHT

    @property
    def DEBUG_PANEL_X(self):
        return SCREEN_WIDTH if self.render_mode == "human" else 0

    @property
    def TOTAL_WIDTH(self):
        return SCREEN_WIDTH + DEBUG_PANEL_WIDTH if self.render_mode == "human" else SCREEN_WIDTH

    def __init__(self, render_mode: str = "none", **kwargs):
        self.render_mode = render_mode

        if self.render_mode == "human":
            pygame.init()
            pygame.display.set_caption("Sonic NES Clone — Green Hill Zone")
            self._surf = pygame.display.set_mode((SCREEN_WIDTH + DEBUG_PANEL_WIDTH, SCREEN_HEIGHT))
        else:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            pygame.init()
            self._surf = pygame.Surface((self.WIDTH, self.HEIGHT))

        # Managers
        self.config_manager = ConfigManager("sonic_config.yaml")
        self.loader = LevelLoader()
        
        # 🚀 Now using the dedicated Sonic Physics Manager!
        self.physics_manager = SonicPhysicsManager()

        try:
            self.debug_manager = DebugManager(
                default_active=(render_mode == "human"),
                print_help=(render_mode == "human")
            )
        except TypeError:
            self.debug_manager = DebugManager()

        # State
        self.level_data = LevelData()
        self.player: SonicPlayer | None = None

        # Sonic-specific collections
        self.rings: List[Ring] = []
        self.lost_rings: List[Ring] = []
        self.badniks: List[Badnik] = []
        self.springs: List[Spring] = []

        # Level order & curriculum
        self.level_order = self.config_manager.get_level_order()
        self.current_index_world = 0
        _default_world = self.level_order[0] if self.level_order else "Green Hill 1"
        self.world = str(kwargs.pop("world", _default_world))
        self.locked_level = str(self.world) if kwargs.pop("lock_level", False) else None

        self._curriculum_window_size  = int(kwargs.pop("curriculum_window", 5))
        self._advance_threshold       = float(kwargs.pop("advance_threshold", 0.6))
        self._fallback_threshold      = float(kwargs.pop("fallback_threshold", 0.2))
        self._explore_prob            = float(kwargs.pop("explore_prob", 0.10))
        self._level_window = {lvl: deque(maxlen=self._curriculum_window_size) for lvl in self.level_order}
        self._max_unlocked_index = int(kwargs.pop("start_unlocked", 0))

        self.speed_mult = float(kwargs.pop("speed_mult", 2.0))
        self.physics_manager.speed_mult = self.speed_mult
        self.max_steps = kwargs.pop("max_steps", None)
        self.persona = str(kwargs.pop("persona", "simple")).lower()
        self.reward_fn = None
        self.ACTION_NAMES = ACTION_NAMES

        # Timer
        self.use_timer = bool(kwargs.pop("use_timer", True))
        self.timer_seconds = int(kwargs.pop("timer_seconds", 300))
        self.timer_warn_threshold = int(kwargs.pop("timer_warn_threshold", 60))

        self.max_lives = 3
        self.lives = self.max_lives

        # Batch curriculum
        self._batch_window = int(kwargs.pop("batch_window", 10))
        self._batch_advance_threshold = float(kwargs.pop("advance_threshold", 0.30))
        self._batch_fallback_threshold = float(kwargs.pop("fallback_threshold", 0.20))
        self._max_stay_windows = int(kwargs.pop("max_stay_windows", 2))
        self._review_prob = float(kwargs.pop("review_prob", 0.25))
        self._curriculum_position = 0
        self._batch_results = []
        self._episode_won_current = False
        self._is_review_episode = False
        self._windows_on_level = 0
        self._consecutive_fallbacks = {}
        self._level_visits = {lvl: 0 for lvl in self.level_order}
        self._level_wins = {lvl: 0 for lvl in self.level_order}

        # Camera
        self.camera_x = 0.0
        self.camera_y = 0.0
        self.camera_smoothing = 0.12
        self.camera_lock = True

        # Anti-stall
        self.anti_stall = bool(kwargs.pop("anti_stall", True))
        self.stall_window = float(kwargs.pop("stall_window", 3))
        self.stall_kill_windows = int(kwargs.pop("stall_kill_windows", 8))

        # Obs stats
        self._obs_check_interval = 5000
        self._obs_check_counter = 0
        self._obs_stats = {
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

        # Observation grid
        self.obs_width = 21
        self.obs_height = 21
        self.obs_pad_x = self.obs_width // 2
        self.obs_pad_y = self.obs_height // 2

        self.dijkstra = None
        self.dijkstra_current_tile = 0.0

        self._obs_space = spaces.Dict({
            "grids": spaces.Box(low=-1.0, high=1.0, shape=(4, self.obs_height, self.obs_width), dtype=np.float32),
            "scalars": spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32),
        })
        self._act_space = spaces.Discrete(10)

        self.ui_font = pygame.font.SysFont("arial", 20, bold=True)
        self.hud_font = pygame.font.SysFont("arial", 16, bold=True)

        self._dijkstra_window_cache = None
        self._solid_window_cache = None
        self._hazard_window_cache = None
        self._cached_spikes = []

        self.ring_total = 0
        self.badniks_destroyed = 0
        self.top_speed_reached = 0.0

        self.reset()

    def reset_metrics(self):
        self.timer = self.timer_seconds
        self.time_last_step = time.time()
        self.dt = 0.0001
        self.score = 0
        self.coins_total = 0
        self.alive = True
        self.frame = 0
        self.game_over = False
        self.reached_goal = False
        self.last_x = 0.0
        self.last_score = 0
        self.score_delta = 0
        self.kills_step = 0
        self.coins_step = 0
        self.powerups_step = 0
        self._last_action = [0, 0, 0]
        self.max_x_seen = 0.0
        self.stall_timer = 0
        self.stall_windows_count = 0
        self.stalled_this_frame = False
        self.progress_x_best = 0.0
        self.progress_y_best = 0.0
        self.death_cause = ""
        self.lives = self.max_lives
        self.best_dist_to_goal = float('inf')
        self._needs_level_transition = False
        self._pending_next_level_index = None
        self._step_dx = 0.0
        self._step_dy = 0.0
        self.ring_total = 0
        self.badniks_destroyed = 0
        self.top_speed_reached = 0.0

    def get_action_space(self): return self._act_space
    def get_observation_space(self): return self._obs_space

    # =========================================================================
    # STEP
    # =========================================================================
    def step(self, action: int):
        if not self.alive:
            dead_obs = self._obs()
            return dead_obs, 0.0, True, False, {"episode_end": True, "won": self.reached_goal}

        if self.render_mode != "human":
            self.dt = 1 / 60.0
        else:
            now = time.time()
            raw_dt = now - self.time_last_step
            self.time_last_step = now
            self.dt = min(raw_dt, 0.05)

        if hasattr(self.debug_manager, 'slow_motion') and self.debug_manager.slow_motion:
            self.dt *= 0.5

        _LEGACY_TO_MD = {
            0: [0,0,0], 1: [1,0,0], 2: [3,0,0], 3: [0,1,0],
            4: [3,1,0], 5: [4,0,0], 6: [1,1,0], 7: [4,1,0],
            8: [0,0,1], 9: [3,0,1],
        }
        if isinstance(action, (int, float)) or not hasattr(action, '__len__'):
            action = _LEGACY_TO_MD.get(int(action), [0, 0, 0])
        action = [int(action[0]), int(action[1]), int(action[2])]

        if self.use_timer:
            self.timer -= self.dt
        if self.render_mode == "human":
            self.debug_manager.update_input()

        self._last_action = action
        self.last_x = self.player.gObj.x if self.player else 0.0
        self.kills_step = 0
        self.coins_step = 0
        self.powerups_step = 0
        self.stalled_this_frame = False
        if self.player:
            self.player._on_moving_platform = False

        # ── Rebuild spatial hashes ───────────────────────────────────────
        self.physics_manager.hazard_hash.clear()
        for enemy in self.level_data.enemies:
            if enemy.gObj.active and not (hasattr(enemy, 'alive') and not enemy.alive):
                self.physics_manager.hazard_hash.insert(enemy)
        for spike in self._cached_spikes:
            self.physics_manager.hazard_hash.insert(spike)

        self.physics_manager.platform_hash.clear()
        for plat in self.level_data.moving_platforms:
            if plat.gObj.active:
                self.physics_manager.platform_hash.insert(plat)

        self.physics_manager.collectible_hash.clear()
        for ring in self.rings:
            if ring.gObj.active and not ring.collected:
                self.physics_manager.collectible_hash.insert(ring)
        for ring in self.lost_rings:
            if ring.gObj.active and ring.can_collect:
                self.physics_manager.collectible_hash.insert(ring)
        for coin in self.level_data.coins:
            if coin.gObj.active and not coin.collected:
                self.physics_manager.collectible_hash.insert(coin)
        for pup in self.level_data.powerups:
            if pup.gObj.active:
                self.physics_manager.collectible_hash.insert(pup)
        for goal in self.level_data.goals:
            self.physics_manager.collectible_hash.insert(goal)

        # ── Player input ─────────────────────────────────────────────────
        if self.player:
            if not self.debug_manager.free_cam_active:
                self.player.handle_input(a=action)
            else:
                self.player.vx = 0
                self.player.jump_hold = 0

        # ── Update entities ──────────────────────────────────────────────
        for ring in self.rings:
            ring.update(self.dt, self.physics_manager.context)
        for ring in self.lost_rings:
            ring.update(self.dt, self.physics_manager.context)
        for spring in self.springs:
            spring.update(self.dt)

        # ── Physics system ───────────────────────────────────────────────
        self.physics_manager.update_system(self.dt, self)
        
        # This now automatically handles regular tiles, lip snagging, AND slope physics!
        self.physics_manager.resolve_collisions(self)

        # ── Sonic-specific logical collisions (combat/items) ─────────────
        if self.player and self.alive:
            self._resolve_sonic_collisions()

        # ── Cleanup ──────────────────────────────────────────────────────
        self.level_data.enemies[:] = [e for e in self.level_data.enemies if e.gObj.active]
        self.badniks[:] = [b for b in self.badniks if b.gObj.active]
        self.rings[:] = [r for r in self.rings if r.gObj.active and not r.collected]
        self.lost_rings[:] = [r for r in self.lost_rings if r.gObj.active]
        self.level_data.coins[:] = [c for c in self.level_data.coins if c.gObj.active]
        self.level_data.powerups[:] = [p for p in self.level_data.powerups if p.gObj.active]
        self.level_data.projectiles[:] = [p for p in self.level_data.projectiles if p.gObj.active]

        speed = abs(self.player.vx) if self.player else 0
        if speed > self.top_speed_reached:
            self.top_speed_reached = speed

        self._goal_dist_cache = self._get_dist_to_goal()
        self._update_camera()
        if self.anti_stall:
            self._update_stall_metrics()

        terminated = self._check_termination()
        truncated = False
        if self.max_steps and self.frame >= self.max_steps:
            truncated = True

        self.score_delta = self.score - self.last_score
        self.last_score = self.score
        self.frame += 1

        obs = self._obs()
        self._check_obs_sanity(obs)
        info = self._info()

        if self._needs_level_transition:
            self._needs_level_transition = False
            if self._pending_next_level_index is not None:
                self.current_index_world = self._pending_next_level_index
                self.world = self.level_order[self.current_index_world]
                self._level_visits[self.world] = self._level_visits.get(self.world, 0) + 1
                self._pending_next_level_index = None
            self.load_level(preserve_rings=True)

        if terminated:
            info["episode_end"] = True

        base_reward = float(self.score_delta)
        return obs, base_reward, bool(terminated), bool(truncated), info

    # =========================================================================
    # SONIC-SPECIFIC COLLISION RESOLUTION
    # =========================================================================
    def _resolve_sonic_collisions(self):
        p = self.player
        if not p: return
        p_rect = p.gObj.get_rect()

        for ring in self.rings:
            if not ring.collected and ring.gObj.active and p_rect.colliderect(ring.gObj.get_rect()):
                ring.collected = True
                ring.gObj.active = False
                p.rings += 1
                self.ring_total += 1
                self.coins_total += 1
                self.coins_step += 1
                self.score += 10

        for ring in self.lost_rings:
            if ring.gObj.active and ring.can_collect and p_rect.colliderect(ring.gObj.get_rect()):
                ring.gObj.active = False
                p.rings += 1
                self.ring_total += 1
                self.coins_total += 1
                self.score += 10

        for badnik in self.badniks:
            if badnik.gObj.active and badnik.alive and p_rect.colliderect(badnik.gObj.get_rect()):
                if p.is_ball:
                    badnik.destroy()
                    p.bounce_off_enemy()
                    self.kills_step += 1
                    self.badniks_destroyed += 1
                    self.score += 100
                else:
                    if p.take_hit():
                        self._handle_death("Badnik")
                    else:
                        self._scatter_rings(p)

        for enemy in self.level_data.enemies:
            if enemy.gObj.active and p_rect.colliderect(enemy.gObj.get_rect()):
                if p.is_ball:
                    enemy.gObj.active = False
                    p.bounce_off_enemy()
                    self.kills_step += 1
                    self.score += 100
                else:
                    if p.take_hit():
                        self._handle_death("Enemy")
                    else:
                        self._scatter_rings(p)

        for spring in self.springs:
            if spring.gObj.active and p_rect.colliderect(spring.gObj.get_rect()):
                spring.trigger()
                p.spring_launch(spring.bounce_velocity)
                self.score += 10

        for goal in self.level_data.goals:
            if p_rect.colliderect(goal.gObj.get_rect()):
                self.reached_goal = True
                self.score += p.rings * 100
                self.complete_level()

    def _scatter_rings(self, player: SonicPlayer):
        scatter_count = min(player.rings, 32)
        if scatter_count == 0: return

        cx = player.gObj.x + player.gObj.width / 2
        cy = player.gObj.y + player.gObj.height / 2
        angle_step = (2 * math.pi) / max(scatter_count, 1)
        
        for i in range(scatter_count):
            angle = i * angle_step + random.uniform(-0.2, 0.2)
            speed = random.uniform(150, 350)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed - 200
            lost = Ring.create_lost_ring(cx, cy, vx, vy)
            self.lost_rings.append(lost)

    # =========================================================================
    # RESET & LOADING
    # =========================================================================
    def reset(self, seed=None, options=None) -> np.ndarray:
        super().reset(seed=seed)
        if self.level_order and not self.locked_level and not self._is_review_episode:
            self._batch_results.append(self._episode_won_current)
        if len(self._batch_results) >= self._batch_window and self.level_order and not self.locked_level:
            self._evaluate_curriculum_batch()

        self.reset_metrics()
        self._episode_won_current = False
        self._is_review_episode = False

        if self.locked_level:
            self.world = self.locked_level
            self.current_index_world = self.level_order.index(self.locked_level) if self.locked_level in self.level_order else 0
        elif self.level_order:
            self._curriculum_position = max(0, min(self._curriculum_position, len(self.level_order) - 1))
            if self._curriculum_position > 0 and random.random() < self._review_prob:
                self.current_index_world = random.randint(0, self._curriculum_position - 1)
                self._is_review_episode = True
            else:
                self.current_index_world = self._curriculum_position
            self.world = self.level_order[self.current_index_world]

        self._level_visits[self.world] = self._level_visits.get(self.world, 0) + 1
        self.load_level()
        return self._obs(), self._info()

    def _evaluate_curriculum_batch(self):
        wins = sum(1 for r in self._batch_results if r)
        total = len(self._batch_results)
        win_rate = wins / total if total > 0 else 0.0
        pos = self._curriculum_position

        if win_rate >= self._batch_advance_threshold and pos < len(self.level_order) - 1:
            self._curriculum_position += 1
            self._windows_on_level = 0
            print(f"  🎓 [Sonic Curriculum] ADVANCE → Act {self._curriculum_position}")
        elif win_rate <= self._batch_fallback_threshold and pos > 0:
            self._curriculum_position -= 1
            self._windows_on_level = 0
            print(f"  ⬇️  [Sonic Curriculum] FALLBACK → Act {self._curriculum_position}")
        else:
            self._windows_on_level += 1
            if self._windows_on_level >= self._max_stay_windows and pos > 0:
                self._curriculum_position -= 1
                self._windows_on_level = 0
        self._batch_results.clear()

    def load_level(self, preserve_rings: bool = False):
        self.alive = True
        self.frame = 0
        self.game_over = False
        self.reached_goal = False

        config = self.config_manager.get_level_config(self.world)
        self.level_data = self.loader.load_level(config)
        self.level_data.projectiles = []
        self.badniks = []
        self.springs = self.level_data.springs # ← Pull the loaded springs!
        self.lost_rings = []

        self.rings = []
        for coin in self.level_data.coins:
            self.rings.append(Ring(gObj=coin.gObj))
        self.level_data.coins = []

        for enemy in self.level_data.enemies:
            self.badniks.append(Badnik(gObj=enemy.gObj, vx=enemy.vx, badnik_type=BadnikType.MOTOBUG))
        self.level_data.enemies = list(self.badniks)

        self._cached_spikes = []
        if self.level_data.tiles:
            for row in self.level_data.tiles:
                if row:
                    for tile in row:
                        if isinstance(tile, Spike):
                            self._cached_spikes.append(tile)

        px, py = self.level_data.player_start
        if 'spawn' in config:
            px = float(config['spawn'].get('x', px))
            py = float(config['spawn'].get('y', py))

        if self.player is None:
            pw = config.get('player', {}).get('dimensions', {}).get('width', 24)
            ph = config.get('player', {}).get('dimensions', {}).get('height', 32)
            self.player = SonicPlayer(gObj=GameObject(px, py, pw, ph, True))
        else:
            self.player.gObj.x = px; self.player.gObj.y = py; self.player.gObj.active = True
            self.player.vx = 0; self.player.vy = 0
            self.player.on_ground = False; self.player.facing_right = True
            self.player.state = SonicState.IDLE
            self.player.spin_dash_charge = 0; self.player.spin_dash_rev = 0.0
            self.player.is_ball = False; self.player.hurt_timer = 0
            self.player.coyote = 0; self.player.jump_hold = 0; self.player.jump_buffer = 0
            if not preserve_rings:
                self.player.rings = 0; self.player.invincible_timer = 0; self.player.shield = False

        self.physics_manager.reset_to_defaults()
        self.physics_manager.apply_config_dict(config)
        self.physics_manager.rebuild_dynamic_hashes(self.level_data, self._cached_spikes)

        self.progress_x_best = self.player.gObj.x
        self.progress_y_best = self.level_data.height - self.player.gObj.y
        self.stall_timer = 0; self.stall_windows_count = 0; self.stalled_this_frame = False
        self.max_x_seen = px; self.best_dist_to_goal = self._get_dist_to_goal()
        self.camera_x = 0.0; self.camera_y = 0.0
        self.last_score = 0; self.last_x = self.player.gObj.x
        self.timer = config.get('time_limit', self.timer_seconds) if self.use_timer else math.inf
        self._calculate_dijkstra_map()

    def _calculate_dijkstra_map(self):
        if not self.level_data.goals:
            self.dijkstra = None; return
        self.dijkstra = DijkstraSolver(self.level_data.grid, self.level_data.rows, self.level_data.cols)
        goal_positions = [(int(g.gObj.x // TILE_SIZE), int(g.gObj.y // TILE_SIZE)) for g in self.level_data.goals]
        ring_positions = set((int(r.gObj.x // TILE_SIZE), int(r.gObj.y // TILE_SIZE)) for r in self.rings if not r.collected)
        self.dijkstra.compute_map(goal_positions, ring_positions)

    def complete_level(self):
        self._level_wins[self.world] = self._level_wins.get(self.world, 0) + 1
        if self._curriculum_position < len(self.level_order) and self.world == self.level_order[self._curriculum_position]:
            self._episode_won_current = True
        self._pending_next_level_index = (self.current_index_world + 1) % len(self.level_order)
        self._needs_level_transition = True

    def _handle_death(self, cause: str = "Unknown") -> bool:
        self.death_cause = cause
        self.lives = max(0, self.lives - 1)
        if self.lives > 0:
            self.load_level()
            return False
        else:
            self.alive = False; self.game_over = True
            return True

    # =========================================================================
    # CAMERA & UTILS
    # =========================================================================
    def _update_camera(self):
        if self.render_mode != "human": return
        if hasattr(self.debug_manager, 'free_cam_active') and self.debug_manager.free_cam_active:
            mx, my = self.debug_manager.current_cam_move
            self.camera_x += mx * self.dt; self.camera_y += my * self.dt
            return
        if not self.camera_lock or not self.player: return

        look_ahead = 80 if self.player.facing_right else -80
        target_x = max(0, min(self.player.gObj.x - self.WIDTH // 3 + look_ahead, self.level_data.width - self.WIDTH))
        self.camera_x += (target_x - self.camera_x) * self.camera_smoothing
        self.camera_x = max(0, min(self.camera_x, max(0, self.level_data.width - self.WIDTH)))

        target_y = max(0, min(self.player.gObj.y - self.HEIGHT // 2, self.level_data.height - self.HEIGHT)) if self.level_data.height > self.HEIGHT else 0.0
        self.camera_y += (target_y - self.camera_y) * self.camera_smoothing
        self.camera_y = max(0, min(self.camera_y, max(0, self.level_data.height - self.HEIGHT)))

    def _get_dist_to_goal(self) -> float:
        if not self.player: return float('inf')
        if not self.level_data.goals: return self.level_data.width - self.player.gObj.x
        return min(math.sqrt((g.gObj.x - self.player.gObj.x)**2 + (g.gObj.y - self.player.gObj.y)**2) for g in self.level_data.goals)

    def _update_stall_metrics(self):
        if not self.player: return
        if self.player.gObj.x > self.max_x_seen: self.max_x_seen = self.player.gObj.x
        if getattr(self.player, '_on_moving_platform', False):
            self.stall_timer = 0; return
        current_dist = getattr(self, '_goal_dist_cache', self._get_dist_to_goal())
        if current_dist < (self.best_dist_to_goal - TILE_SIZE / 2.0):
            self.best_dist_to_goal = current_dist
            self.stall_timer = 0
        else:
            self.stall_timer += self.dt
            if self.stall_timer >= self.stall_window:
                self.stalled_this_frame = True; self.stall_timer = 0; self.stall_windows_count += 1

    def _check_termination(self) -> bool:
        if not self.player: return True
        if self.use_timer and self.timer <= 0: return self._handle_death("Time Over")
        if self.player.gObj.y > self.level_data.height: return self._handle_death("Pit")
        if self.anti_stall and self.stall_windows_count >= self.stall_kill_windows: return self._handle_death("Stall")
        return False

    # =========================================================================
    # OBSERVATION & INFO (Truncated repetitive RL grids for brevity, identical logic)
    # =========================================================================
    def _obs(self) -> Dict[str, np.ndarray]:
        if not self.player:
            return {"grids": np.zeros((4, self.obs_height, self.obs_width), dtype=np.float32), "scalars": np.zeros(20, dtype=np.float32)}
        p_obs = self._player_obs()
        track_obs = self._tracking_obs()
        solid, collect, hazard, row_start, col_start = self._grid_obs_window()
        dijkstra_grid = self._dijkstra_obs_window(row_start, col_start)
        stacked = np.stack([solid, collect, hazard, dijkstra_grid], axis=0).astype(np.float32)
        scalars = np.concatenate([p_obs, track_obs]).astype(np.float32)
        return {"grids": stacked, "scalars": scalars}

    def _player_obs(self) -> np.ndarray:
        return self.player.obs_vector(
            max(1.0, getattr(self.physics_manager.context, 'MAX_RUN_SPEED', 380.0)),
            max(1.0, getattr(self.physics_manager.context, 'MAX_FALL_SPEED', 600.0))
        ) if self.player else np.zeros(13, dtype=np.float32)

    def _grid_obs_window(self):
        p = self.player
        if not p:
            z = np.zeros((self.obs_height, self.obs_width), dtype=np.float32)
            return z, z, z, 0, 0
        px, py = int(p.gObj.x // TILE_SIZE), int(p.gObj.y // TILE_SIZE)
        map_row_start, map_col_start = py - self.obs_pad_y, px - self.obs_pad_x
        solid_grid, collect_grid, hazard_grid = [np.zeros((self.obs_height, self.obs_width), dtype=np.float32) for _ in range(3)]
        wx, wy = map_col_start * TILE_SIZE, map_row_start * TILE_SIZE
        ww, wh = self.obs_width * TILE_SIZE, self.obs_height * TILE_SIZE

        def _place(grid, obj_x, obj_y, value):
            lx, ly = int(obj_x // TILE_SIZE) - map_col_start, int(obj_y // TILE_SIZE) - map_row_start
            if 0 <= lx < self.obs_width and 0 <= ly < self.obs_height and abs(value) > abs(grid[ly, lx]):
                grid[ly, lx] = value

        for h in self.physics_manager.hazard_hash.query_rect(wx, wy, ww, wh):
            if h.gObj.active: _place(hazard_grid, h.gObj.x, h.gObj.y, -1.0 if isinstance(h, Spike) else +1.0)
        for c in self.physics_manager.collectible_hash.query_rect(wx, wy, ww, wh):
            if c.gObj.active and not getattr(c, 'collected', False):
                _place(collect_grid, c.gObj.x, c.gObj.y, 1.0 if isinstance(c, Goal) else (0.35 if isinstance(c, (Ring, Coin)) else 0.69))
        for obj in self.level_data.static_hash.query_rect(wx, wy, ww, wh):
            if not isinstance(obj, Spike): _place(solid_grid, obj.gObj.x, obj.gObj.y, 1.0)
        for plat in self.physics_manager.platform_hash.query_rect(wx, wy, ww, wh):
            if plat.gObj.active:
                for pc in range(int(plat.gObj.x // TILE_SIZE), int((plat.gObj.x + plat.gObj.width - 1) // TILE_SIZE) + 1):
                    lx, ly = pc - map_col_start, int(plat.gObj.y // TILE_SIZE) - map_row_start
                    if 0 <= lx < self.obs_width and 0 <= ly < self.obs_height: solid_grid[ly, lx] = 1.0

        for ly in range(self.obs_height):
            for lx in range(self.obs_width):
                if solid_grid[ly, lx] != 0.0 or hazard_grid[ly, lx] != 0.0: continue
                mc = map_col_start + lx
                if not (0 <= mc < self.level_data.cols): continue
                if not any(0 <= map_row_start + ly + s < self.level_data.rows and self.level_data.grid[map_row_start + ly + s][mc] not in (TILE_AIR, TILE_SPIKE) for s in range(1, 7)):
                    solid_grid[ly, lx] = hazard_grid[ly, lx] = -0.5

        self._solid_window_cache, self._hazard_window_cache = solid_grid, hazard_grid
        return solid_grid, collect_grid, hazard_grid, map_row_start, map_col_start

    def _dijkstra_obs_window(self, map_row_start, map_col_start):
        zero = np.zeros((self.obs_height, self.obs_width), dtype=np.float32)
        if self.dijkstra is None or not self.player: return zero
        px_tile, py_tile = int(self.player.gObj.x // TILE_SIZE), int(self.player.gObj.y // TILE_SIZE)
        player_dist = self.dijkstra.dist_map[py_tile, px_tile] if (0 <= py_tile < self.level_data.rows and 0 <= px_tile < self.level_data.cols) else np.inf
        if not np.isfinite(player_dist): return zero

        r0, c0 = max(0, map_row_start), max(0, map_col_start)
        r1, c1 = min(self.level_data.rows, map_row_start + self.obs_height), min(self.level_data.cols, map_col_start + self.obs_width)
        if r1 - r0 <= 0 or c1 - c0 <= 0: return zero

        delta = player_dist - self.dijkstra.dist_map[r0:r1, c0:c1]
        delta[~np.isfinite(delta)] = 0.0
        out = np.zeros((self.obs_height, self.obs_width), dtype=np.float32)
        out[r0 - map_row_start:r1 - map_row_start, c0 - map_col_start:c1 - map_col_start] = delta
        np.clip(out / max(self.obs_width // 2 * 2.0, self.obs_height // 2 * 3.5), -1.0, 1.0, out=out)
        self._dijkstra_window_cache = out
        return out.astype(np.float32)

    def _tracking_obs(self) -> np.ndarray:
        p = self.player
        if not p: return np.zeros(7, dtype=np.float32)
        SR = math.sqrt((self.obs_width * TILE_SIZE)**2 + (self.obs_height * TILE_SIZE)**2) * 0.5

        def get_dist(h_obj, coll=False, skip_s=False):
            min_d, cnt = 9999.0, 0
            for obj in h_obj.query_rect(p.gObj.x - SR, p.gObj.y - SR, SR * 2, SR * 2):
                if not obj.gObj.active or (coll and getattr(obj, 'collected', False)) or (skip_s and getattr(obj.gObj, 'type_id', None) == EntityType.SPIKE): continue
                d_sq = (p.gObj.x - obj.gObj.x)**2 + (p.gObj.y - obj.gObj.y)**2
                if d_sq < min_d: min_d = d_sq
                cnt += 1
            return (math.sqrt(min_d) if min_d < 9999.0 else 9999.0), cnt

        e_dist, _ = get_dist(self.physics_manager.hazard_hash, skip_s=True)
        norm_dist = max(self.level_data.width, self.level_data.height, 1.0)
        dy = (min(self.level_data.goals, key=lambda g: abs(g.gObj.x - p.gObj.x)).gObj.y if self.level_data.goals else 0) - p.gObj.y
        
        self._step_dx, self._step_dy = 0.0, 0.0
        dj_dist = 1.0
        if self.dijkstra:
            ptx, pty = int(p.gObj.x // TILE_SIZE), int(p.gObj.y // TILE_SIZE)
            if (d := self.dijkstra.get_dist(ptx, pty)) >= 0: dj_dist = np.clip(d / (self.level_data.cols * 2), 0.0, 1.0)

        return np.array([
            np.clip(e_dist / norm_dist, 0.0, 1.0),
            np.clip(getattr(self, '_goal_dist_cache', self._get_dist_to_goal()) / norm_dist, 0.0, 1.0),
            np.clip(self.timer / max(1.0, self.timer_seconds), 0.0, 1.0),
            np.clip(dy / max(self.level_data.height, 1), -1.0, 1.0),
            dj_dist, self._step_dx, self._step_dy
        ], dtype=np.float32)

    def _check_obs_sanity(self, obs):
        self._obs_check_counter += 1
        if self._obs_check_counter % self._obs_check_interval == 0 and (grids := obs.get("grids")) is not None:
            for i, name in enumerate(["solid", "collectible", "hazard", "dijkstra"]):
                self._obs_stats[f"grid_{name}_mean"] = float(grids[i].mean())
                self._obs_stats[f"grid_{name}_std"] = float(grids[i].std())

    def _info(self) -> Dict:
        p = self.player
        ts = float(TILE_SIZE)
        event = "ACT CLEAR" if self.reached_goal else ("DIED" if not self.alive else "")
        cause = "Goal" if self.reached_goal else getattr(self, 'death_cause', "")

        info_dict = {
            "score": self.score, "score_delta": self.score_delta, "frame_count": self.frame,
            "rings": p.rings if p else 0, "coins_collected": self.coins_total, "enemies_killed_step": self.kills_step,
            "terminated": not self.alive, "won": self.reached_goal,
            "action": self._last_action, "action_name": action_to_str(self._last_action),
            "time_left": math.ceil(self.timer), "max_x_seen": self.max_x_seen,
            "stall_windows": self.stall_windows_count, "stalled": self.stalled_this_frame,
            "persona": self.persona, "level": self.world, "lives": self.lives, "event": event, "cause": cause,
            "curriculum_level_idx": self.current_index_world,
            "curriculum_win_rate": sum(self._level_window.get(self.world, [0])) / max(1, len(self._level_window.get(self.world, [0]))),
            "curriculum_max_unlocked": self._max_unlocked_index,
            "badniks_destroyed": self.badniks_destroyed, "top_speed": self.top_speed_reached,
            **self._obs_stats
        }

        if p:
            dj_dist = 0.0
            if self.dijkstra and (d := self.dijkstra.get_dist(int(p.gObj.x // TILE_SIZE), int(p.gObj.y // TILE_SIZE))) >= 0:
                dj_dist = np.clip(d / (self.level_data.cols * 2), 0.0, 1.0)
            info_dict.update({
                "x_position": p.gObj.x / ts, "y_position": p.gObj.y / ts, "velocity_x": p.vx / ts, "velocity_y": p.vy / ts,
                "powered_up": p.powered_up, "goal_dist": getattr(self, '_goal_dist_cache', self._get_dist_to_goal()) / ts,
                "dijkstra_dist": dj_dist, "on_ground": p.on_ground, "on_moving_platform": getattr(p, '_on_moving_platform', False),
                "step_dx": self._step_dx, "step_dy": self._step_dy, "is_ball": p.is_ball, "sonic_state": p.state.name,
                "spin_dash_charge": p.spin_dash_charge
            })
        else:
            info_dict.update({"x_position": 0.0, "y_position": 0.0, "velocity_x": 0.0, "velocity_y": 0.0, "powered_up": False, "goal_dist": 0.0, "dijkstra_dist": 0.0, "on_ground": False, "on_moving_platform": False, "step_dx": 0.0, "step_dy": 0.0, "is_ball": False, "sonic_state": "IDLE", "spin_dash_charge": 0})
        return info_dict

    # =========================================================================
    # RENDER 
    # =========================================================================
    def render(self, surface: pygame.Surface, blit_only: bool = True):
        game_surf = surface.subsurface(pygame.Rect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)) if self.render_mode == "human" else surface
        game_surf.fill(COLOR_SKY)
        self._draw_background(game_surf); self._draw_world(game_surf)
        self._draw_entities(game_surf); self._draw_player(game_surf); self._draw_hud(game_surf)

    def _draw_background(self, surface: pygame.Surface):
        h = surface.get_height()
        for i in range(5): pygame.draw.ellipse(surface, (40, 160 + (i % 2) * 30, 80), (int(i * 300 - (self.camera_x * 0.3) % 300), int(h * 0.55), 280, 120))
        for i in range(8): pygame.draw.ellipse(surface, (30, 140 + (i % 3) * 20, 60), (int(i * 200 - (self.camera_x * 0.5) % 200), int(h * 0.65), 180, 80))

    def _draw_world(self, surface: pygame.Surface):
        for tile in self.level_data.static_hash.query_rect(self.camera_x, self.camera_y, self.WIDTH, self.HEIGHT):
            if tile.x + tile.width < self.camera_x or tile.x > self.camera_x + self.WIDTH: continue
            sx, sy = tile.gObj.x - self.camera_x, tile.gObj.y - self.camera_y

            if isinstance(tile, Spike): tile.render(surface, sx, sy)
            elif isinstance(tile, SlopeTile): tile.render(surface, self.camera_x, self.camera_y)
            elif isinstance(tile, Tile):
                tw, th = tile.gObj.width, tile.gObj.height
                if tile.type_id == TILE_GROUND:
                    gx, gy = int(tile.gObj.x // TILE_SIZE), int(tile.gObj.y // TILE_SIZE)
                    pygame.draw.rect(surface, COLOR_GROUND if (gx + gy) % 2 == 0 else COLOR_GROUND_CHECK, (int(sx), int(sy), int(tw), int(th)))
                    if (self.level_data.grid[gy - 1][gx] if 0 <= gy - 1 < self.level_data.rows and gx < self.level_data.cols else TILE_AIR) == TILE_AIR:
                        pygame.draw.rect(surface, COLOR_GRASS_TOP, (int(sx), int(sy), int(tw), 6))
                        for gbi in range(0, int(tw), 6): pygame.draw.line(surface, (0, 220, 60), (int(sx + gbi), int(sy)), (int(sx + gbi + 2), int(sy - (3 + (gbi % 3)))), 1)
                elif tile.type_id == TILE_PLATFORM:
                    pygame.draw.rect(surface, COLOR_PLATFORM, (int(sx), int(sy), int(tw), int(th)))
                    pygame.draw.rect(surface, (160, 120, 60), (int(sx), int(sy), int(tw), int(th)), 1)
                else:
                    tile.render(surface, self.camera_x, self.camera_y)

        for plat in self.physics_manager.platform_hash.query_rect(self.camera_x, self.camera_y, self.WIDTH, self.HEIGHT):
            if plat.gObj.active: plat.render(surface, plat.gObj.x - self.camera_x, plat.gObj.y - self.camera_y)

    def _draw_entities(self, surface: pygame.Surface):
        cx, cy, cw, ch = self.camera_x, self.camera_y, self.WIDTH, self.HEIGHT
        for lst in [self.rings, self.lost_rings, self.badniks, self.level_data.enemies, self.springs]:
            for obj in lst:
                if getattr(obj, 'collected', False) or not obj.gObj.active: continue
                ox, oy = obj.gObj.x - cx, obj.gObj.y - cy
                if -32 < ox < cw + 32 and -32 < oy < ch + 32: obj.render(surface, ox, oy)
        for goal in self.level_data.goals:
            gx, gy = goal.gObj.x - cx, goal.gObj.y - cy
            if -32 < gx < cw + 32 and -32 < gy < ch + 32:
                pygame.draw.rect(surface, (180, 180, 180), (int(gx + 12), int(gy), 8, int(goal.gObj.height)))
                pygame.draw.rect(surface, COLOR_GOAL, (int(gx), int(gy), int(goal.gObj.width), 12))
                pygame.draw.rect(surface, COLOR_BLACK, (int(gx), int(gy), int(goal.gObj.width), 12), 1)

    def _draw_player(self, surface: pygame.Surface):
        if self.player: self.player.render(surface, self.player.gObj.x - self.camera_x, self.player.gObj.y - self.camera_y, getattr(self.debug_manager, 'show_sensors', False))

    def _draw_hud(self, surface: pygame.Surface):
        p = self.player; rings = p.rings if p else 0
        speed_pct = int(abs(p.vx) / 560 * 100) if p else 0
        ring_color = (255, 50, 50) if rings == 0 and self.frame % 30 < 15 else COLOR_RING

        y = 8
        for text_str, color in [(f"SCORE  {self.score:>8}", COLOR_WHITE), (f"TIME   {int(self.timer):>4}", COLOR_WHITE), (f"RINGS  {rings:>4}", ring_color), (f"LIVES  {self.lives}", COLOR_WHITE)]:
            text = self.hud_font.render(text_str, True, color)
            bg = pygame.Surface((text.get_width() + 8, text.get_height() + 2), pygame.SRCALPHA); bg.fill((0, 0, 0, 160))
            surface.blit(bg, (8, y)); surface.blit(text, (12, y + 1))
            y += 18

        if p and abs(p.vx) > 30:
            pygame.draw.rect(surface, (40, 40, 40), (8, y + 2, 102, 8))
            pygame.draw.rect(surface, (50, 255, 50) if speed_pct < 60 else ((255, 255, 0) if speed_pct < 85 else (255, 50, 50)), (9, y + 3, min(100, int(abs(p.vx) / 5.6)), 6))

        if p:
            st = self.hud_font.render(p.state.name + (f" [{p.spin_dash_charge}]" if p.state == SonicState.SPIN_DASH else ""), True, (200, 200, 255))
            surface.blit(st, (self.WIDTH - st.get_width() - 10, 8))

    def close(self):
        if self.render_mode == "human": pygame.quit()