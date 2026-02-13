from __future__ import annotations
import os
import math
import importlib
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pygame
import time
import gymnasium
from gymnasium import spaces
import psutil

from code.games.modules.System import EntityType
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
# Screen / Tile geometry
# =============================================================================
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
PLATFORMER_WIDTH, PLATFORMER_HEIGHT = 32, 32

# Action Map for Debug Display
ACTION_NAMES = {
    0: "IDLE", 1: "LEFT", 2: "RIGHT", 3: "JUMP",
    4: "RIGHT+JUMP", 5: "RUN+RIGHT", 6: "LEFT+JUMP", 7: "RUN+RIGHT+JUMP",
    8: "RUN+LEFT", 9: "RUN+LEFT+JUMP"
}

class PlatformerCore(gymnasium.Env):
    WIDTH, HEIGHT = SCREEN_WIDTH, SCREEN_HEIGHT

    def __init__(self, render_mode: str = "none", **kwargs):
        self.render_mode = render_mode

        if self.render_mode == "human":
            pygame.init()
            pygame.display.set_caption("PEAK Platformer")
            self._surf = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
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
        self.reward_fn = self._load_reward_fn(self.persona)
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

        self.reset_metrics()

        # --- NEW: VARIABLE GRID OBSERVATION SIZE ---
        self.obs_width = 31
        self.obs_height = 31

        self.obs_pad_x = self.obs_width // 2
        self.obs_pad_y = self.obs_height // 2

        self._obs_space = spaces.Dict({
            "grids": spaces.Box(low=0.0, high=1.0, shape=(3, self.obs_height, self.obs_width), dtype=np.float32),
            "scalars": spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)
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
            return self._obs(), 0.0, True, False, {"episode_end": True, "won": self.reached_goal}

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

        # URGENT FIX: Call the reward function here
        base_reward = float(self.score_delta)
        if self.reward_fn:
            reward = self.reward_fn(self._obs(), base_reward, terminated, info)
        else:
            reward = base_reward

        return self._obs(), reward, bool(terminated), bool(truncated), info

    def reset(self, seed=None, options=None) -> np.ndarray:
        super().reset(seed=seed)

        if not self.reached_goal:
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

    def complete_level(self):
        self.current_index_world += 1
        if self.current_index_world >= len(self.level_order):
            print("all levels done")
            self.current_index_world = 0
        self.world = self.level_order[self.current_index_world]
        self.load_level()

    def _handle_death(self, cause: str = "Unknown") -> bool:
        self.death_cause = cause
        self.lives -= 1
        if self.lives > 0:
            self._soft_reset()
            return False
        else:
            self.alive = False
            self.game_over = True
            self.reset()
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
            return self.level_data.width - self.player.gObj.x

        px = self.player.gObj.x
        min_d = float('inf')
        for g in self.level_data.goals:
            gx = g.gObj.x
            d = abs(gx - px)
            if d < min_d: min_d = d

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

        # 3. GOAL & SPIKES
        p_rect = player.gObj.get_rect()
        nearby = self.level_data.static_hash.query(player.gObj)

        for tile in nearby:
            if not hasattr(tile, 'gObj') or not hasattr(tile.gObj, 'type_id'):
                continue

            tid = tile.gObj.type_id

            if p_rect.colliderect(tile.gObj.get_rect()):
                if tid == EntityType.SPIKE:
                    self._handle_death("Spike")
                    return True

                elif tid == EntityType.GOAL:
                    # FIX: Prevent left-wall glitch while allowing actual right-side goals.
                    if player.gObj.x > 200:
                        self.score += 1000 + (int(self.timer) * 10)
                        self.reached_goal = True
                        self.complete_level()
                        return True

        # 4. STALL DEATH
        if self.anti_stall and self.stall_windows_count >= self.stall_kill_windows:
            return self._handle_death("Stall")

        return False

    def _obs(self) -> Dict[str, np.ndarray]:
        # NULL Check Safety
        if not self.player:
            return {
                "grids": np.zeros((3, self.obs_height, self.obs_width), dtype=np.float32),
                "scalars": np.zeros(13, dtype=np.float32)
            }

        p_obs = self._player_obs()
        solid_grid, hazard_grid, collect_grid = self._grid_obs_window()
        track_obs = self._tracking_obs()

        g1 = solid_grid
        g2 = hazard_grid
        g3 = collect_grid

        stacked_grids = np.stack([g1, g2, g3], axis=0).astype(np.float32)
        scalars = np.concatenate([p_obs, track_obs]).astype(np.float32)

        return {
            "grids": stacked_grids,
            "scalars": scalars
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

    def _grid_obs_window(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        p = self.player
        if not p:
            z = np.zeros((self.obs_height, self.obs_width), dtype=np.float32)
            return z, z, z

        px = int(p.gObj.x // TILE_SIZE)
        py = int(p.gObj.y // TILE_SIZE)

        r_start = py - self.obs_pad_y
        r_end = r_start + self.obs_height
        c_start = px - self.obs_pad_x
        c_end = c_start + self.obs_width

        r_start_clamped = max(0, r_start)
        c_start_clamped = max(0, c_start)

        raw_slice = self.padded_solid[r_start_clamped:r_end, c_start_clamped:c_end]

        if raw_slice.shape == (self.obs_height, self.obs_width):
            solid_grid = raw_slice
        else:
            solid_grid = np.zeros((self.obs_height, self.obs_width), dtype=np.float32)
            h, w = raw_slice.shape

            dest_y = 0 if r_start >= 0 else abs(r_start)
            dest_x = 0 if c_start >= 0 else abs(c_start)

            paste_h = min(h, self.obs_height - dest_y)
            paste_w = min(w, self.obs_width - dest_x)

            if paste_h > 0 and paste_w > 0:
                solid_grid[dest_y : dest_y + paste_h, dest_x : dest_x + paste_w] = raw_slice[:paste_h, :paste_w]

        hazard_grid = np.zeros((self.obs_height, self.obs_width), dtype=np.float32)
        collect_grid = np.zeros((self.obs_height, self.obs_width), dtype=np.float32)

        wx = max(0, (px - self.obs_pad_x) * TILE_SIZE)
        wy = max(0, (py - self.obs_pad_y) * TILE_SIZE)
        window_rect = pygame.Rect(
            wx, wy,
            self.obs_width * TILE_SIZE,
            self.obs_height * TILE_SIZE
        )

        nearby_hazards = self.physics_manager.hazard_hash.query_rect(window_rect.x, window_rect.y, window_rect.width, window_rect.height)
        for h in nearby_hazards:
            if not h.gObj.active: continue
            hx, hy = int(h.gObj.x // TILE_SIZE), int(h.gObj.y // TILE_SIZE)
            local_x = hx - (px - self.obs_pad_x)
            local_y = hy - (py - self.obs_pad_y)
            if 0 <= local_x < self.obs_width and 0 <= local_y < self.obs_height:
                hazard_grid[local_y, local_x] = 1.0

        nearby_items = self.physics_manager.collectible_hash.query_rect(window_rect.x, window_rect.y, window_rect.width, window_rect.height)
        for c in nearby_items:
            if not c.gObj.active: continue
            if hasattr(c, 'collected') and c.collected: continue

            cx, cy = int(c.gObj.x // TILE_SIZE), int(c.gObj.y // TILE_SIZE)
            local_x = cx - (px - self.obs_pad_x)
            local_y = cy - (py - self.obs_pad_y)
            if 0 <= local_x < self.obs_width and 0 <= local_y < self.obs_height:
                collect_grid[local_y, local_x] = 1.0

        return solid_grid, hazard_grid, collect_grid

    def _tracking_obs(self) -> np.ndarray:
        p = self.player
        if not p: return np.zeros(8, dtype=np.float32)

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

        return np.array([
            np.clip(e_dist / norm_dist, 0.0, 1.0),
            np.clip(c_dist / norm_dist, 0.0, 1.0),
            np.clip(raw_goal_dist / norm_dist, 0.0, 1.0),
            np.clip(e_count / 20.0, 0.0, 1.0),
            np.clip(c_count / 50.0, 0.0, 1.0),
            np.clip(self.score / 10000.0, 0.0, 1.0),
            np.clip(self.timer / max(1.0, self.timer_seconds), 0.0, 1.0),
            self.lives / float(max(1.0, self.max_lives)),
        ], dtype=np.float32)

    def _info(self) -> Dict:
        p = self.player
        ts = float(TILE_SIZE)

        event = ""
        cause = getattr(self, 'death_cause', "")
        if self.reached_goal and not self.game_over:
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
                "event": event, "cause": cause
            }

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
            "won": (self.reached_goal and not self.game_over),
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
            "cause": cause
        }

    def render(self, surface: pygame.Surface, blit_only: bool = True):
        surface.fill(COLOR_SKY)

        self._draw_world(surface)
        self._draw_entities(surface)
        self._draw_player(surface)

        if self.debug_manager:
            self.debug_manager.render_overlays(surface, self)

        self._draw_ui(surface)

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
