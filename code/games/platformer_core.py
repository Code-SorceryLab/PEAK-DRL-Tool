from __future__ import annotations
import os
import math
import importlib
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import numpy as np
import pygame
import time
import gymnasium
from gymnasium import spaces

# --- CORRECTED IMPORTS FOR NEW FOLDER STRUCTURE ---
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
# from .modules.System.SpriteManager import SpriteManager

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
PLATFORMER_WIDTH, PLATFORMER_HEIGHT = 20, 32

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
        self.stall_window = int(kwargs.pop("stall_window", 1.5))
        self.stall_kill_windows = int(kwargs.pop("stall_kill_windows", 6))
        
        self.timer = self.timer_seconds
        self.time_last_step = time.time()
        self.dt = 0.0001
        
        # Track previous state
        self.score = 0; self.coins_total = 0; self.alive = True; self.frame = 0
        self.game_over = False; self.reached_goal = False
        self.last_x = 0.0; self.last_score = 0; self.score_delta = 0
        self.kills_step = 0; self.coins_step = 0; self.powerups_step = 0; self._last_action = 0
        
        self.max_x_seen = 0.0
        self.stall_timer = 0
        self.stall_windows_count = 0
        self.stalled_this_frame = False
        self.progress_x_best = 0.0
        self.progress_y_best = 0.0

        # --- NEW OBSERVATION SPACE CALCULATION ---
        # 1. Player Data (5)
        # 2. Solid Array (11x9 = 99)
        # 3. Hazard Array (11x9 = 99)
        # 4. Collectable Array (11x9 = 99)
        # 5. Tracking Data (8)
        # Total: 5 + 99 + 99 + 99 + 8 = 310
        obs_len = 310
        self._obs_space = spaces.Box(low=0.0, high=1e9, shape=(obs_len,), dtype=np.float32)
        self._act_space = spaces.Discrete(8)


        self.ui_font = pygame.font.SysFont("arial", 20, bold=True)
        self.qblock_font = pygame.font.SysFont("arial", 26, bold=True)
        
        # SPRITE MANAGER
        core_dir = os.path.dirname(os.path.abspath(__file__))
        assets_dir = os.path.join(core_dir, "assets")
    
#        print(f"[DEBUG] Loading Assets from: {assets_dir}") 
#        self.sprite_manager = SpriteManager(assets_dir, sprite_width=32, sprite_height=32, scale=1.5)
        
        
        self.reset()

    def _load_reward_fn(self, persona_name):
        try:
            mod = importlib.import_module("code.rewards.platformer")
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

        # Frame Updates
        self.frame += 1
        if self.use_timer: self.timer -= self.dt
        if self.render_mode == "human": self.debug_manager.update_input()

        # Step Metrics Reset
        self._last_action = int(action)
        self.last_x = self.player.gObj.x
        self.kills_step = self.coins_step = self.powerups_step = 0
        self.stalled_this_frame = False

        # PHYSICS & LOGIC
        # 1. Rebuild Hash for dynamic interactions
        self.physics_manager.rebuild_dynamic_hashes(self.level_data)
        
        # 2. Player Input
        if not self.debug_manager.free_cam_active:
             self.player.handle_input(a = int(action))
        else:
            self.player.vx = 0; self.player.jump_hold = 0

        # 3. Physics System Update
        self.physics_manager.update_system(self.dt, self)
        
        # 4. Collision Resolution
        self.physics_manager.resolve_collisions(self)

        # 5. Logic Updates
        self._update_camera()
        if self.anti_stall: self._update_stall_metrics()

        terminated = self._check_termination()
        
        # Handle Truncation (Time limit logic moves here)
        truncated = False
        if self.use_timer and self.timer <= 0:
            truncated = True
        
        # but for death-by-time, your logic treats it as death.
        # If you want pure truncation (timeout isn't death):
        # terminated = False
        # truncated = True
        
        
        self.score_delta = self.score - self.last_score
        self.last_score = self.score
        
        info = self._info()
        if terminated: info["episode_end"] = True
        return self._obs(), 0.0, bool(terminated), bool(truncated), info

    def reset(self, seed=None, options=None) -> np.ndarray:
        
        super().reset(seed=seed)
        
        if not self.reached_goal:   
            self.lives = self.max_lives 
            self.score = 0
            self.coins_total = 0
        self.load_level()
        
        
        return self._obs(), self._info()

    def load_level(self):
        self.alive = True
        self.frame = 0
        self.game_over = False
        self.reached_goal = False
        # 1. LOAD CONFIG & LEVEL
        config = self.config_manager.get_level_config(self.world)
        self.level_data = self.loader.load_level(config)
        
        # --- OPTIMIZATION: Prepare Numpy Grid for Observation Slicing ---
        # Create a binary grid (1.0 for Solid, 0.0 for Air)
        # We assume TILE_AIR is 0 or distinct from solid blocks
        raw_grid = np.array(self.level_data.grid, dtype=np.int32)
        self.solid_grid_np = (raw_grid != TILE_AIR).astype(np.float32)
        
        # Pad the grid so we can slice the 11x9 window without boundary checks
        # Window is 11 wide (5 left, 5 right) and 9 high (4 up, 4 down)
        pad_y = 4 
        pad_x = 5
        self.padded_solid = np.pad(
            self.solid_grid_np, 
            ((pad_y, pad_y), (pad_x, pad_x)), 
            mode='constant', 
            constant_values=0.0
        )
        # ---------------------------------------------------------------
        
        # 2. CREATE PLAYER
        px, py = self.level_data.player_start
        if 'spawn' in config:
            px = float(config['spawn'].get('x', px))
            py = float(config['spawn'].get('y', py))
            
        # self.player = Player(gObj=GameObject(px, py, PLATFORMER_WIDTH, PLATFORMER_HEIGHT, True), sprite_manager=self.sprite_manager)
        self.player = Player(gObj=GameObject(px, py, PLATFORMER_WIDTH, PLATFORMER_HEIGHT, True))
        # 3. CONFIGURE PHYSICS
        self.physics_manager.reset_to_defaults()
        self.physics_manager.apply_config_dict(config)
        
        # 4. INITIALIZE HASHES
        self.physics_manager.rebuild_dynamic_hashes(self.level_data)

        # 5. RESET METRICS
        self.progress_x_best = self.player.gObj.x
        self.progress_y_best = self.level_data.height - self.player.gObj.y
        self.stall_timer = 0
        self.stall_windows_count = 0
        self.stalled_this_frame = False
        self.camera_x = 0.0
        self.camera_y = 0.0
        self.last_score = 0
        self.last_x = self.player.gObj.x

        self.timer = config.get('time_limit', self.timer_seconds) if self.use_timer else math.inf
        
    def complete_level(self):
        print(f"Level Complete! Current World: {self.world}")
        # --- Logic to Advance Level --
        self.current_index_world += 1
        if self.current_index_world >= len(self.level_order):
            print("all levels done") # As requested
            self.current_index_world = 0
        self.world = self.level_order[self.current_index_world]
        self.load_level()
    
    def _handle_death(self):
        self.lives -= 1
        if self.lives > 0:
            self._soft_reset()
        else:
            self.alive = False
            self.game_over = True

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

    def _update_stall_metrics(self):
        if not self.player: return
        prog_x = self.player.gObj.x
        prog_y = self.level_data.height - self.player.gObj.y
        progressed = False

        if prog_x > self.progress_x_best + (TILE_SIZE / 2):
            self.progress_x_best = prog_x; progressed = True
        if prog_y > self.progress_y_best + (TILE_SIZE / 2):
            self.progress_y_best = prog_y; progressed = True

        if progressed:
            self.stall_timer = 0; self.stalled_this_frame = False
        else:
            self.stall_timer += self.dt
            if self.stall_timer >= self.stall_window:
                self.stalled_this_frame = True; self.stall_timer = 0; self.stall_windows_count += 1

    def _check_termination(self) -> bool:
        player = self.player
        
        # 1. TIME
        if self.use_timer and self.timer <= 0:
            self._handle_death()
            return not self.alive 

        # 2. PIT
        if player.gObj.y > self.level_data.height:
            self._handle_death()
            return not self.alive
        
        # 3. GOAL & SPIKES
        cx, cy = player.gObj.x + player.gObj.width/2, player.gObj.y + player.gObj.height/2
        row, col = int(cy // TILE_SIZE), int(cx // TILE_SIZE)
        
        if 0 <= row < self.level_data.rows and 0 <= col < self.level_data.cols:
            tile_val = self.level_data.grid[row][col]
            if tile_val == TILE_GOAL:
                self.score += 1000 + (int(self.timer) * 10)
                self.alive = False
                self.reached_goal = True
                self.complete_level()
                return False
            if tile_val == TILE_SPIKE:
                self._handle_death()
                return not self.alive

        if self.anti_stall and self.stall_windows_count >= self.stall_kill_windows:
            self._handle_death()
            return not self.alive

        return False

    # =========================================================================
    # NEW OBSERVATION LOGIC
    # =========================================================================
    def _obs(self) -> np.ndarray:
        """
        Constructs the vector observation:
        [Player(5), Solid(99), Hazard(99), Collectable(99), Tracking(8)]
        """
        # 1. Player Data (5)
        p_obs = self._player_obs()
        
        # 2. Grids (99 * 3)
        # We compute these together to save tile coord calcs
        solid_grid, hazard_grid, collect_grid = self._grid_obs_window_11x9()
        
        # 3. Tracking Data (8)
        track_obs = self._tracking_obs()
        
        # Concatenate all
        return np.concatenate([
            p_obs,
            solid_grid, 
            hazard_grid,
            collect_grid,
            track_obs
        ]).astype(np.float32)

    def _player_obs(self) -> np.ndarray:
        p = self.player
        w = max(1.0, float(self.level_data.width))
        h = max(1.0, float(self.level_data.height))
        # 5 Values
        return np.array([
            p.gObj.x / w, 
            p.gObj.y / h,
            p.vx / max(1e-6, self.physics_manager.context.MAX_RUN_SPEED),
            p.vy / self.physics_manager.context.MAX_FALL_SPEED,
            1.0 if p.on_ground else 0.0,
        ], dtype=np.float32)
        
    def _grid_obs_window_11x9(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns flattened arrays for Solid, Hazard, and Collectable windows (11x9).
        Window: 5 left, 5 right, 4 up, 4 down relative to player tile.
        """
        p = self.player
        px = int(p.gObj.x // TILE_SIZE)
        py = int(p.gObj.y // TILE_SIZE)
        
        # --- A. SOLID ARRAY (Optimized Numpy Slice) ---
        # Map player (px, py) to padded grid coordinates
        # Padded has 4 top, 5 left.
        # Window Start in Padded: (py - 4) + 4 = py
        # Window End in Padded:   (py + 4) + 4 + 1 = py + 9
        # X logic similarly: px, px + 11
        
        # Clamp slicing just in case, though padding should cover typical map bounds
        # (Assuming player doesn't go waaaay out of bounds)
        try:
            solid_window = self.padded_solid[py : py + 9, px : px + 11].flatten()
        except IndexError:
            # Fallback for extreme out of bounds
            solid_window = np.zeros(99, dtype=np.float32)

        # --- B. DYNAMIC ARRAYS (Hazard/Collectable) ---
        # Optimization: Use Spatial Hash to find only relevant entities
        hazard_grid = np.zeros((9, 11), dtype=np.float32)
        collect_grid = np.zeros((9, 11), dtype=np.float32)
        
        # Define world rect for query (11 tiles wide, 9 tiles high)
        window_rect = pygame.Rect(
            (px - 5) * TILE_SIZE, 
            (py - 4) * TILE_SIZE, 
            11 * TILE_SIZE, 
            9 * TILE_SIZE
        )
        
        # Hazards
        nearby_hazards = self.physics_manager.hazard_hash.query_rect(window_rect.x, window_rect.y, window_rect.width, window_rect.height)
        for h in nearby_hazards:
            if not h.gObj.active: continue
            hx, hy = int(h.gObj.x // TILE_SIZE), int(h.gObj.y // TILE_SIZE)
            local_x = hx - (px - 5)
            local_y = hy - (py - 4)
            if 0 <= local_x < 11 and 0 <= local_y < 9:
                hazard_grid[local_y, local_x] = 1.0

        # Collectables
        nearby_items = self.physics_manager.collectible_hash.query_rect(window_rect.x, window_rect.y, window_rect.width, window_rect.height)
        for c in nearby_items:
            # Check active status. Coins have 'collected' flag too.
            if not c.gObj.active: continue
            if hasattr(c, 'collected') and c.collected: continue
            
            cx, cy = int(c.gObj.x // TILE_SIZE), int(c.gObj.y // TILE_SIZE)
            local_x = cx - (px - 5)
            local_y = cy - (py - 4)
            if 0 <= local_x < 11 and 0 <= local_y < 9:
                collect_grid[local_y, local_x] = 1.0

        return solid_window, hazard_grid.flatten(), collect_grid.flatten()

    def _tracking_obs(self) -> np.ndarray:
        p = self.player
        
        # Helpers for "Nearest" logic
        def get_dist(obj_list):
            min_d = 9999.0
            count = 0
            for obj in obj_list:
                if not obj.gObj.active: continue
                if hasattr(obj, 'collected') and obj.collected: continue
                
                # Euclidean distance
                d = math.sqrt((p.gObj.x - obj.gObj.x)**2 + (p.gObj.y - obj.gObj.y)**2)
                if d < min_d: min_d = d
                count += 1
            return min_d, count

        # 1. Enemies
        e_dist, e_count = get_dist(self.level_data.enemies)
        
        # 2. Coins
        c_dist, c_count = get_dist(self.level_data.coins)
        
        # 3. Goal Dist
        w = max(1.0, float(self.level_data.width))
        goal_dist = (w - p.gObj.x) / w

        # Normalization constants (approximate level diagonal or screen size)
        norm_dist = 1000.0 

        return np.array([
            min(1.0, e_dist / norm_dist), # Nearest Enemy Dist
            min(1.0, c_dist / norm_dist), # Nearest Coin Dist
            goal_dist,                    # Goal Distance
            min(1.0, e_count / 20.0),     # Active Enemies (Normalized cap 20)
            min(1.0, c_count / 50.0),     # Active Coins (Normalized cap 50)
            self.score / 5000.0,          # Score
            self.timer / 400.0,           # Time Left
            self.lives / 5.0              # Lives Left
        ], dtype=np.float32)

    def _info(self) -> Dict:
        p = self.player
        return {
            "score": self.score, 
            "score_delta": self.score_delta, 
            "frame_count": self.frame,
            "x_position": p.gObj.x, 
            "y_position": p.gObj.y, 
            "velocity_x": p.vx,
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
            "persona": self.persona
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
        
        # 1. Hazards (Enemies)
        visible_hazards = self.physics_manager.hazard_hash.query_rect(cx, cy, cw, ch)
        for entity in visible_hazards:
            if hasattr(entity, 'render'):
                entity.render(surface, entity.x - cx, entity.y - cy)

        # 2. Collectibles (Coins, Powerups)
        visible_collectibles = self.physics_manager.collectible_hash.query_rect(cx, cy, cw, ch)
        for entity in visible_collectibles:
             if hasattr(entity, 'render'):
                entity.render(surface, entity.x - cx, entity.y - cy)

    def _draw_player(self, surface: pygame.Surface):
        p = self.player
        sx = p.gObj.x - self.camera_x
        sy = p.gObj.y - self.camera_y
        
        colour = COLOR_POWERUP_STAR if (p.invincible_timer > 0 and (self.frame // 5) % 2) else \
              ((255, 100, 0) if p.powered_up else (255, 0, 0))
        p.color = colour
        p.render(surface, sx, sy, self.debug_manager.show_sensors)

    def _draw_ui(self, surface: pygame.Surface):
        p = self.player
        status = "STAR" if p.invincible_timer > 0 else ("SUPER" if p.powered_up else "SMALL")
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