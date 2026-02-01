from __future__ import annotations
import os
import math
import importlib
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import numpy as np
import pygame
import time
from gymnasium import spaces

# Modules 
from .modules.GameObject import GameObject
from .modules.Tile import Tile, create_tile
from .modules.Player import Player
from .modules.Enemy import Enemy
from .modules.Powerup import Powerup
from .modules.Coin import Coin
from .modules.QuestionBlock import QuestionBlock
from .modules.SpatialHash import SpatialHash

# Debug Module
from .modules.debugging_mods.manager import DebugManager

from .modules.Movement_parameters import(RUN_ACCEL, WALK_ACCEL, MAX_WALK_SPEED, MAX_RUN_SPEED,
    GROUND_FRICTION, AIR_FRICTION, AIR_CONTROL, SKID_DECEL,
    GRAVITY, FAST_FALL_GRAV, MAX_FALL_SPEED)

from .modules.Jump_parameters import(JUMP_VEL_MIN, JUMP_VEL_MAX, JUMP_HOLD_FRAMES,
    SPEED_JUMP_BONUS, COYOTE_FRAMES, JUMP_BUFFER_FRAMES)

from .modules.Map_parameters import(TILE_AIR, TILE_GROUND, TILE_PLATFORM, TILE_GOAL, TILE_SPIKE, TILE_QBLOCK,
    COLOR_SKY, COLOR_GROUND, COLOR_PLATFORM, COLOR_GOAL, COLOR_SPIKE,
    COLOR_WHITE, COLOR_BLACK, COLOR_QBLOCK, COLOR_EMPTY, COLOR_ENEMY,
    COLOR_POWERUP_MUSH, COLOR_POWERUP_STAR, COLOR_COIN, COLOR_HITBOX,
    COLOR_SENSOR, COLOR_AGENT_PANEL, COLOR_STREAK, TILE_SIZE)

# =============================================================================
# Screen / Tile geometry
# =============================================================================
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
LEVEL_ROWS, LEVEL_COLS = 0, 0 

# Character dimensions
PLATFORMER_WIDTH, PLATFORMER_HEIGHT = 20, 32

# Action Map for Debug Display
ACTION_NAMES = {
    0: "IDLE", 1: "LEFT", 2: "RIGHT", 3: "JUMP",
    4: "RIGHT+JUMP", 5: "RUN+RIGHT", 6: "LEFT+JUMP", 7: "RUN+RIGHT+JUMP"
}

class PlatformerCore:
    WIDTH, HEIGHT = SCREEN_WIDTH, SCREEN_HEIGHT

    def __init__(self, render_mode: str = "none", **kwargs):
        self.render_mode = render_mode
        self.level_rows = LEVEL_ROWS
        self.level_cols = LEVEL_COLS
        
        self.level_data: List[List[int]] = []
        self.level_tiles: List[List[Tile | None]] = []
        self.level_w = 0
        self.level_h = 0
        self.goal_x = 0.0
        
        # -------- Config knobs ----------------------------
        
        # Default world and speed multiplier
        self.world = str(kwargs.pop("world", "1-1")).lower()
        
        self.speed_mult = float(kwargs.pop("speed_mult", 2.0)) # URGENT - SPEED LINES 
        
        # Ensure debug defaults to False if not in human mode to prevent popups during training
        debug_def = bool(kwargs.pop("debug_default", True))
        self.debug_default = debug_def if render_mode == "human" else False
        
        self.max_steps = kwargs.pop("max_steps", None)
        
        # Reward Persona - Load from external file
        self.persona = str(kwargs.pop("persona", "Default")).lower()
        self.reward_fn = self._load_reward_fn(self.persona)

        # --- Timer knobs ------------------------------------
        self.use_timer = bool(kwargs.pop("use_timer", True))
        self.timer_seconds = int(kwargs.pop("timer_seconds", 400))
        self.timer_warn_threshold = int(kwargs.pop("timer_warn_threshold", 100))

        self.max_lives = 3
        self.lives = self.max_lives

        #--- Camera knobs -----------------------------------------
        self.camera_x = 0.0
        self.camera_y = 0.0
        self.camera_smoothing = 0.15
        self.camera_lock = True
        
        # Anti-stall knobs
        """
        This implements a simple anti-stall mechanism that tracks player progress and increments
        a stall timer when no progress is made. If the player fails to make progress within a 
        defined number of stall windows, a life is lost.
        """
        self.anti_stall = bool(kwargs.pop("anti_stall", True))
        self.stall_window = int(kwargs.pop("stall_window", 1.5))
        self.stall_kill_windows = int(kwargs.pop("stall_kill_windows", 6))
        self.stall_eps = float(kwargs.pop("stall_eps", 6.0))
        self.stall_penalty = float(kwargs.pop("stall_penalty", -0.02))
        self.backtrack_penalty = float(kwargs.pop("backtrack_penalty", 0.001))
        self.progress_best = 0.0

        self.timer = self.timer_seconds
        self.time_last_step = time.time()
        self.dt = 0.0001

        self.level_file = kwargs.pop("level_file", None)
        self.level_list = ["stage_1.txt", "stage_2.txt", "stage_3.txt", "stage_4.txt", "stage_5.txt", "stage_6.txt", "stage_7.txt", "stage_8.txt", "stage_9.txt", "stage_10.txt", "stage_11.txt"]
        self.current_level_idx = 0

        if self.level_file is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            candidate = os.path.join(base_dir, "levels", self.level_list[0])
            if os.path.exists(candidate):
                self.level_file = candidate

        self.player_start = (100.0, 350.0)

        # Apply speed multiplier - URGENT - SPEED LINES
        self.run_accel = RUN_ACCEL * self.speed_mult
        self.walk_accel = WALK_ACCEL * self.speed_mult
        self.max_walk = MAX_WALK_SPEED * self.speed_mult
        self.max_run = MAX_RUN_SPEED * self.speed_mult

        # -------- World / state ------------------------------------------
        self.player: Player | None = None
        self.enemies: List[Enemy] = []
        self.powerups: List[Powerup] = []
        self.coins: List[Coin] = []
        self.qblocks: List[QuestionBlock] = []
        
        # --- DUAL HASHING Setup ---
        """
        Spatial hashing for efficient collision detection. 
        Works by dividing the game world into a grid of cells and 
        assigning objects to cells based on their positions.
        """
        self.static_hash = SpatialHash(cell_size=64)
        self.dynamic_hash = SpatialHash(cell_size=64)
        
        # Game state
        self.score = 0; self.coins_total = 0; self.alive = True; self.frame = 0
        self.game_over = False; self.reached_goal = False
        self.last_x = 0.0; self.last_score = 0; self.score_delta = 0
        self.kills_step = 0; self.coins_step = 0; self.powerups_step = 0; self._last_action = 0
        
        # Tracking Stall metrics
        self.max_x_seen = 0.0
        self.stall_timer = 0
        self.stall_windows_count = 0
        self.stalled_this_frame = False
        self.progress_x_best = 0.0
        self.progress_y_best = 0.0

        # Gym spaces
        obs_len = 5 + (11 * 9 * 3) + 11
        self._obs_space = spaces.Box(low=0.0, high=1e9, shape=(obs_len,), dtype=np.float32)
        self._act_space = spaces.Discrete(8)

        # Video and Audio setup (for human rendering) | (Bug) ON LINUX 
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        #os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        pygame.init()
        
        # Set up the Canvas / Surface for rendering
        self._surf = pygame.Surface((self.WIDTH, self.HEIGHT))

        # --- Debug Manager ---
        self.debug_manager = DebugManager(default_active=self.debug_default, print_help=(self.render_mode == "human"))

        # --- Fonts ---
        self.ui_font = pygame.font.SysFont("arial", 20, bold=True)
        self.qblock_font = pygame.font.SysFont("arial", 26, bold=True)

        self.reset()

    def _load_reward_fn(self, persona_name):
        """Dynamically load reward function from code.rewards.platformer"""
        try:
            mod = importlib.import_module("code.rewards.platformer")
            return getattr(mod, persona_name, None)
        except ImportError:
            return None

    def get_action_space(self): 
        return self._act_space
    
    def get_observation_space(self): 
        return self._obs_space

    def _find_goal_x(self):
        self.goal_x = float(self.level_w)
        for row in range(self.level_rows):
            for col in range(self.level_cols):
                if self.level_data[row][col] == TILE_GOAL:
                    self.goal_x = float(col * TILE_SIZE)
                    return

    def reset(self) -> np.ndarray:

        self.lives = self.max_lives # Reset lives to 3

        self.enemies.clear(); self.powerups.clear(); self.coins.clear(); self.qblocks.clear()
        
        # Clear spatial hashes
        self.dynamic_hash.clear(); self.static_hash.clear()
        
        # Reset game specific var
        self.score = 0
        self.coins_total = 0
        self.alive = True
        self.frame = 0
        self.game_over = False
        self.reached_goal = False

        if self.level_file:
            self.level_data = self._build_level_from_txt(self.level_file)
        else:
            print("ERROR WITH LEVEL_FILE")
            exit
        
        # Post level load resize
        self._post_level_resize()
        self._find_goal_x()
        self._create_player()

        # Reset stall metrics
        self.progress_x_best = self.player.gObj.x
        self.progress_y_best = self.level_h - self.player.gObj.y
        self.stall_timer = 0
        self.stall_windows_count = 0
        self.stalled_this_frame = False
        
        # Reset camera
        self.camera_x = 0.0; self.camera_y = 0.0
        self.last_score = 0; self.last_x = self.player.gObj.x

        # Spawn default enemies and coins for world 1-1 if no level file is provided
        if not self.level_file and self.world == "1-1":
            self._spawn_static_actors_for_world()

        self.timer = self.timer_seconds if self.use_timer else math.inf
        return self._obs()

    def step(self, action: int):
        # CHECK ALIVE
        if not self.alive:
            return self._obs(), 0.0, True, {"episode_end": True, "won": self.reached_goal}

        # DELTA TIME CALCULATION
        if self.render_mode != "human":
            self.dt = 1 / 60.0
        else:
            time_curr_step = time.time()
            raw_dt = time_curr_step - self.time_last_step
            self.time_last_step = time_curr_step
            self.dt = min(raw_dt, 0.05)
            
        # SLOW MOTION DEBUG
        if self.debug_manager.slow_motion:
            self.dt *= 0.5

        # FRAME & TIMER UPDATE -> for stall logic
        self.frame += 1
        if self.use_timer: self.timer -= self.dt

        # Debug Input Update (only in human mode to avoid interfering with training)
        if self.render_mode == "human":
            self.debug_manager.update_input()

        # RESET STEP METRICS
        self._last_action = int(action)
        self.last_x = self.player.gObj.x
        self.kills_step = self.coins_step = self.powerups_step = 0
        self.stalled_this_frame = False

        # UPDATE DYNAMIC SPATIAL HASH
        self._rebuild_dynamic_hash()
        
        # GAME LOGIC UPDATES
        self._handle_action(int(action))
        self._update_physics(self.dt)
        self._update_objects(self.dt)
        self._handle_object_collisions()
        self._update_camera()
        
        # UPDATE STALL METRICS
        if self.anti_stall:
            self._update_stall_metrics()

        terminated = self._check_termination()
        self.score_delta = self.score - self.last_score
        self.last_score = self.score
        
        # CALCULATE REWARD
        #reward = float(self._reward())
        
        # LOG FOR DEBUG
        action_name = ACTION_NAMES.get(int(action), f"ACT_{action}")
        #self.debug_manager.log_step(reward, action_name)
        
        # Build dicts for reward and agent
        info = self._info()
        if terminated: info["episode_end"] = True
        return self._obs(), 0, bool(terminated), info

    def _rebuild_dynamic_hash(self):
        """"
        Rebuilds the dynamic spatial hash with active dynamic entities.
        """
        self.dynamic_hash.clear()
        for enemy in self.enemies:
            if enemy.gObj.active: self.dynamic_hash.insert(enemy)
        for coin in self.coins:
            if coin.gObj.active and not coin.collected: self.dynamic_hash.insert(coin)
        for power_up in self.powerups:
            if power_up.gObj.active: self.dynamic_hash.insert(power_up)

    def load_level(self, idx):
        """
        Check if level file exists and targets it
        """
        level_name = self.level_list[idx]
        base_dir = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(base_dir, "levels", level_name)
        
        if os.path.exists(candidate):
            self.level_file = candidate
        # else  <  error messare? or Assert

    def complete_level(self):
        """
        Advances to the next level if available.
        """
        if self.current_level_idx < len(self.level_list) - 1:
            self.current_level_idx += 1
            self.load_level(self.current_level_idx)
        else:
            print("Congratulations! All levels complete.")

    def _post_level_resize(self):
        """"
        Updates level dimensions after loading level data.
        """
        self.level_rows = len(self.level_data) if self.level_data else 0
        self.level_cols = len(self.level_data[0]) if self.level_rows > 0 else 0
        
        self.level_w = self.level_cols * TILE_SIZE
        self.level_h = self.level_rows * TILE_SIZE

    def _build_level_from_txt(self, path: str) -> List[List[int]]:
        """
        ASCII level loader.
        """
        if not os.path.exists(path): 
            return [] 

        with open(path, "r") as file:
            lines = [ln.rstrip("\n") for ln in file.readlines()]

        rows = len(lines)
        cols = max(len(ln) for ln in lines) if rows else 0
        
        # Initialize level data structures
        lvl = [[TILE_AIR for col in range(cols)] for row in range(rows)]
        self.level_tiles = [[None for col in range(cols)] for row in range(rows)]
        
        # Clear existing spatial hash entries
        self.static_hash.clear()
        self.qblocks.clear()
        self.coins.clear()
        self.enemies.clear()
        self.powerups.clear()
        
        # Default player start
        self.player_start = (100.0, 350.0)

        for row in range(rows):
            curr_row = lines[row]
            for col in range(len(curr_row)):
                ascii = curr_row[col]
                tile = TILE_AIR
                color = COLOR_SKY
                solid = False
                
                if ascii == '#': tile = TILE_GROUND; color = COLOR_GROUND; solid = True
                elif ascii == '=': tile = TILE_PLATFORM; color = COLOR_PLATFORM; solid = True
                elif ascii == 'G': tile = TILE_GOAL; color = COLOR_GOAL; solid = False
                elif ascii == '^': tile = TILE_SPIKE; color = COLOR_SPIKE; solid = False
                elif ascii == '?': 
                    tile = TILE_QBLOCK; color = COLOR_QBLOCK; solid = True
                    qblock = QuestionBlock(gObj=GameObject(col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE, True), contains="coin")
                    self.qblocks.append(qblock)
                    
                # Don't insert logic object into hash:
                elif ascii == 'C':
                    self.coins.append(Coin(gObj=GameObject(col * TILE_SIZE + 8, row * TILE_SIZE + 8, 16, 16, True)))
                elif ascii == 'E':
                    self.enemies.append(Enemy(GameObject(col * TILE_SIZE + 8, row * TILE_SIZE + 8, 25, 20, True), vx=-60.0))
                elif ascii == 'P':
                    self.player_start = (col * TILE_SIZE, row * TILE_SIZE)

                lvl[row][col] = tile
                if tile != TILE_AIR:
                    new_tile = create_tile(tile, col * TILE_SIZE, row * TILE_SIZE, solid, color)
                    self.level_tiles[row][col] = new_tile
                    
                    if solid or tile in (TILE_SPIKE, TILE_GOAL, TILE_QBLOCK):
                        self.static_hash.insert(new_tile)
        return lvl

    # MAY NOT NEED ? 
    # def _spawn_static_actors_for_world(self):
    #     """
    #     Spawns default enemies and coins for world 1-1.
    #     """
    #     # Spawn Enemies and Coins at hardcoded positions
    #     for col in (26, 32, 60, 110, 140):
    #         self.enemies.append(Enemy(GameObject(col * TILE_SIZE, (14-1)*TILE_SIZE, 20, 18, True), vx=-60.0))
        
    #     for (cx, ry) in [(21, 10), (22, 10)]:
    #         self.coins.append(Coin(gObj=GameObject(cx*TILE_SIZE+8, ry*TILE_SIZE+8, 16, 16, True)))

    def _create_player(self):
        """
        Initializes the player at the starting position.
        """
        x, y = self.player_start
        self.player = Player(gObj=GameObject(float(x), float(y), PLATFORMER_WIDTH, PLATFORMER_HEIGHT, True))

    def _handle_action(self, a: int):
        # Prevent player movement processing if free cam is active
        # This forcefully ignores any key presses the Player class might be reading internally
        if self.debug_manager.free_cam_active:
            self.player.vx = 0.0
            self.player.jump_hold = 0
            
            return 
        
        self.player.handle_input(a = a)

    #
    def _update_physics(self, dt: float):
        """
        Updates player physics and resolves tile collisions.
        """
        self.player.update(dt)
        self._resolve_player_tiles()

    # don't think nessessary remove in refactor
    def _is_solid(self, t: int) -> bool:
        return t in (TILE_GROUND, TILE_PLATFORM, TILE_QBLOCK)

    def _tile_rects_near(self, obj: GameObject):
        """"
        Returns list of solid tile rects near the given object.
        """
        # Optimized grid lookup for Player physics
        tx0 = max(0, int(obj.x // TILE_SIZE) - 1)
        tx1 = min(self.level_cols, int((obj.x + obj.width) // TILE_SIZE) + 2)
        ty0 = max(0, int(obj.y // TILE_SIZE) - 1)
        ty1 = min(self.level_rows, int((obj.y + obj.height) // TILE_SIZE) + 2)
        out = []
        for row in range(ty0, ty1):
            for col in range(tx0, tx1):
                tile = self.level_data[row][col]
                if self._is_solid(tile):
                    out.append((row, col, pygame.Rect(col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE), tile))
        return out

    def _resolve_player_tiles(self):
        """
        gets tiles near player and check and resolve collisions with all them
        """
        player = self.player  
        prect = player.gObj.get_rect()
        for (row, col, trect, type_tile) in self._tile_rects_near(player.gObj):
        #for (r, c, trect, tt) in self.static_hash.query(prect):  URGENT SWITCH TO DYNAMIC FOR QUERY
            if not prect.colliderect(trect): continue
            """
            Check where the player is compared to the collision tile
            """
            obj_x = min(prect.right - trect.left, trect.right - prect.left)
            obj_y = min(prect.bottom - trect.top, trect.bottom - prect.top)
            
            if obj_x < obj_y:
                player.gObj.x = (trect.left - player.gObj.width) if prect.centerx < trect.centerx else trect.right
                player.vx *= 0.5
            else:
                if prect.centery < trect.centery:
                    if abs(prect.bottom - trect.top)< max(2, player.vy + 1):
                        player.gObj.y = trect.top - player.gObj.height
                        player.vy = 0
                        player.on_ground = True
                        player.jump_hold = 0
                else:
                    player.gObj.y = trect.bottom
                    player.vy = max(0.0, player.vy)
                    if type_tile == TILE_QBLOCK and player.vy <=0: self._hit_qblock(col, row)
            prect = player.gObj.get_rect()

    def _hit_qblock(self, col: int, row: int):
        """
        Resolve if player hit question block hit to spawn coins or power up
        """
        for block in self.qblocks:
            block_col, block_row = b.tc()
            if block_col == col and block_row == row and not block.hit:
                block.hit = True
                spawn_x, spawn_y = col * TILE_SIZE, row * TILE_SIZE - 22
                if block.contains == "coin":
                    self.coins.append(Coin(gObj=GameObject(col*TILE_SIZE+8, row*TILE_SIZE+8, 16, 16, True), flyup=True, vy=-280.0, life=0.3, auto_collect=True))
                elif block.contains == "mushroom":
                    self.powerups.append(Powerup(gObj=GameObject(spawn_x, spawn_y, 20, 20, True), kind="mushroom"))
                else:
                    self.powerups.append(Powerup(gObj=GameObject(spawn_x, spawn_y, 20, 20, True), kind="star"))
                self.level_data[row][col] = TILE_PLATFORM
                break

    def _update_objects(self, dt: float):
        """
        Calls updates for all game objects
        """
        for enemy in self.enemies:
            if enemy.gObj.active:
                nearby = self.static_hash.query(enemy)
                enemy.update(dt, list(nearby))
                self._resolve_enemy_tiles(enemy=enemy)
                
        for powerup in self.powerups:
            if powerup.gObj.active:
                nearby = self.static_hash.query(powerup)
                powerup.update(dt, list(nearby)) 
                
        for col in self.coins:
            if col.gObj.active:
                col.update(dt)
                if col.auto_collect and col.flyup and col.life <= 0:
                    col.gObj.active = False
                    self.coins_total += 1; self.coins_step += 1; self.score += 10
    
    def _resolve_enemy_tiles(self, enemy: Enemy):
        """
        resolve collisions for enemies
        """
        rect = enemy.gObj.get_rect()
        nearby_objects = self.dynamic_hash.query(enemy)
        
        for other in nearby_objects:
            # Only process Enemy objects (skip coins, powerups, etc.)
            if not isinstance(other, Enemy):
                continue
            # Skip self-collision
            if other is enemy or not other.gObj.active:
                continue
            other_rect = other.gObj.get_rect()
            if not rect.colliderect(other_rect):
                continue
            # Calculate overlap amounts
            obj_x = min(rect.right - other_rect.left, other_rect.right - rect.left)
            obj_y = min(rect.bottom - other_rect.top, other_rect.bottom - rect.top)
            if obj_x < obj_y:
                # Horizontal collision - push apart and bounce
                if rect.centerx < other_rect.centerx:
                    enemy.gObj.x = other_rect.left - enemy.gObj.width
                else:
                    enemy.gObj.x = other_rect.right
                enemy.vx *= -1.0
            else:
                # Vertical collision - stack or separate vertically
                if rect.centery < other_rect.centery:
                    enemy.gObj.y = other_rect.top - enemy.gObj.height
                else:
                    enemy.gObj.y = other_rect.bottom
                enemy.vy = 0.0
            rect = enemy.gObj.get_rect()
            
    def _handle_object_collisions(self):
        """
        Resolve player & enemy collisions
        """
        player = self.player
        moving_down = player.vy > 0
        
        nearby_objects = self.dynamic_hash.query(player)
        
        for obj in nearby_objects:
            if isinstance(obj, Enemy):
                enemy = obj
                if not enemy.gObj.active: continue
                if player.gObj.collides_with(enemy.gObj):
                    player_bottom = player.gObj.y + player.gObj.height
                    enemy_center = enemy.gObj.y + enemy.gObj.height/2
                    
                    if player_bottom < enemy_center + 10 and moving_down:
                        # Platformer jumped on enemy
                        enemy.gObj.active = False
                        player.vy = JUMP_VEL_MIN * 0.6
                        self.score += 100; self.kills_step += 1
                    elif player.invincible_timer > 0:
                        # Star power
                        enemy.gObj.active = False
                        self.score += 100; self.kills_step += 1
                    else:
                        # Lost powerup
                        if player.powered_up:
                            player.powered_up = False; player.invincible_timer = 60
                        else:
                            # DIED TO ENEMY
                            self._handle_death() # If lives > 0, we just respawned. 
                            # We return immediately to prevent physics glitches this frame.
                            return

            elif isinstance(obj, Coin):
                coin = obj
                if coin.gObj.active and not coin.collected and player.gObj.collides_with(coin.gObj):
                    coin.gObj.active = False; coin.collected = True
                    self.coins_total += 1; self.coins_step += 1; self.score += 10

            elif isinstance(obj, Powerup):
                powerup = obj
                if powerup.gObj.active and player.gObj.collides_with(powerup.gObj):
                    powerup.gObj.active = False
                    self.powerups_step += 1
                    if powerup.kind == "mushroom":
                        player.powered_up = True; self.score += 50
                    else:
                        player.invincible_timer = 300; self.score += 100

    def _handle_death(self):
        """
        Called when player hits an enemy, pit, or time runs out.
        """
        self.lives -= 1
        
        
        if self.lives > 0:
            # We have lives left, reset the level state but keep score
            self._soft_reset()
        else:
            # No lives left, actual Game Over
            self.alive = False
            self.game_over = True

    def _soft_reset(self):
        """
        Resets player, enemies, and timer for a respawn.
        Keeps score and collected coin count.
        """
        # 1. Clear dynamic entities
        self.enemies.clear(); self.powerups.clear(); self.coins.clear(); self.qblocks.clear()
        self.dynamic_hash.clear(); self.static_hash.clear()
        
        # 2. Re-read level data (to respawn enemies and blocks)
        if self.level_file:
            self.level_data = self._build_level_from_txt(self.level_file)
        elif self.world == "1-1":
            self._spawn_static_actors_for_world()

        # 3. Reset Player Position
        self._create_player()
        self._post_level_resize() # Ensure grids are ready

        # 4. Reset Timer & Camera
        self.timer = self.timer_seconds
        self.camera_x = 0.0
        self.camera_y = 0.0
        
        # 5. Reset stalling logic
        self.stall_timer = 0
        self.stall_windows_count = 0
        self.stalled_this_frame = False
        self.progress_x_best = self.player.gObj.x

    def _update_camera(self):
        """
        Resolve Camera Movement Mode:
            - Default player follow
            - Free Cam Debug Mode
        """
        # 1. Check Free Cam Mode
        if self.debug_manager.free_cam_active:
             movement_x, movement_y = self.debug_manager.current_cam_move
             self.camera_x += movement_x * self.dt
             self.camera_y += movement_y * self.dt
             
             # Removed clamping in Free Cam mode so you can fly Up/Down 
             # outside the normal level boundaries.
             return

        # 2. Else Follow Player
        if not self.camera_lock or not self.player: 
            return
        
        target_x = max(0, min(self.player.gObj.x - self.WIDTH // 3, self.level_w - self.WIDTH))
        self.camera_x += (target_x - self.camera_x) * self.camera_smoothing
        self.camera_x = max(0, min(self.camera_x, max(0, self.level_w - self.WIDTH)))

        target_y = 0.0
        if self.level_h > self.HEIGHT:
            target_y = max(0, min(self.player.gObj.y - self.HEIGHT // 2, self.level_h - self.HEIGHT))
        
        self.camera_y += (target_y - self.camera_y) * self.camera_smoothing
        self.camera_y = max(0, min(self.camera_y, max(0, self.level_h - self.HEIGHT)))

    def _progress_components(self):
        """"
        Returns the player's progress in X and Y directions.
        """
        if not self.player: 
            return 0.0, 0.0
        return self.player.gObj.x, self.level_h - self.player.gObj.y

    def _update_stall_metrics(self):
        ''''
        Updates stall timer and progress metrics.
        1. Checks if player has made progress in X or Y direction.
        2. If progress made, reset stall timer.
        3. If no progress, increment stall timer.
        '''
        if not self.anti_stall or not self.player: 
            return
        prog_x, prog_y = self._progress_components()
        progressed = False

        if prog_x > self.progress_x_best + (TILE_SIZE / 2 ):
            self.progress_x_best = prog_x
            progressed = True
            
        if prog_y > self.progress_y_best + (TILE_SIZE / 2 ):
            self.progress_y_best = prog_y 
            progressed = True

        if progressed:
            self.stall_timer = 0  
            self.stalled_this_frame = False
            return

        self.stall_timer += self.dt
        
        if self.stall_timer >= self.stall_window:
            self.stalled_this_frame = True
            self.stall_timer = 0
            self.stall_windows_count += 1

    def _check_termination(self) -> bool:
        """
        Checks for various end states (Timer, Player death, Goal reached etc)
        """
        player = self.player
        
        # 1. TIME LIMIT CHECK
        if self.use_timer and self.timer <= 0:
            self._handle_death()
            # If we still have lives, we are NOT terminated yet
            return not self.alive 

        # 2. PIT CHECK
        if player.gObj.y > self.level_h:
            self._handle_death()
            return not self.alive
        
        # 3. GOAL CHECK
        tile = self._tile_at(player.gObj.x + player.gObj.width / 2, player.gObj.y + player.gObj.height / 2)
        if tile == TILE_GOAL:
            self.score += 1000 + (int(self.timer) * 10) # Time bonus!
            self.alive = False; self.reached_goal = True
            self.complete_level()
            return True

        # 4. SPIKE CHECK
        if tile == TILE_SPIKE:
            self._handle_death()
            return not self.alive

        # 5. ANTI-STALL (Optional: decide if stall kills a life or ends game)
        # now triggers death instead of immediate termination
        if self.anti_stall and self.stall_windows_count >= self.stall_kill_windows:
            self._handle_death()
            return not self.alive

        return False


    def _obs(self) -> np.ndarray:
        """"
        Constructs the observation vector for the current game state.
        Combines player state, tile window, and object proximities.
        1. Player State: position, velocity, on_ground
        2. Tile Window: 11x9 grid of tiles around player + coin/enemy maps
        3. Object Proximities: nearest enemy/coin distances + player status
        4. Additional Info: score, coins, frame ratio
        """
        out: List[float] = []
        out.extend(self._player_obs())
        out.extend(self._tile_window_obs())
        out.extend(self._object_obs())
        return np.array(out, dtype=np.float32)

    def _player_obs(self) -> List[float]:
        """
        Returns normalized player state observations.
        1. X Position / Level Width
        2. Y Position / Level Height
        3. X Velocity / Max Run Speed
        4. Y Velocity / Max Fall Speed
        5. On Ground (1.0 or 0.0)
        5 total values
        """
        player = self.player
        level_width = max(1.0, float(self.level_w))
        level_height = max(1.0, float(self.level_h))

        return [
            player.gObj.x / level_width,
            player.gObj.y / level_height,
            player.vx / max(1e-6, self.max_run),
            player.vy / MAX_FALL_SPEED,
            1.0 if player.on_ground else 0.0,
        ]
        
    def _tile_window_obs(self) -> List[float]:
        """
        Returns tile window observations around the player.
        1. 11x9 Tile Types (121 values)
        2. 11x9 Coin Presence Map (121 values)
        3. 11x9 Enemy Presence Map (121 values)
        Total: 363 values
        
        Normalized as floats:
        0.0 for empty/out-of-bounds, 1.0 for presence
        """
        player = self.player
        player_x = int(player.gObj.x // TILE_SIZE)
        player_y = int(player.gObj.y // TILE_SIZE)
        tiles: List[float] = []; coins_map: List[float] = []; enemies_map: List[float] = []

        coin_cells = {(int(c.gObj.x // TILE_SIZE), int(c.gObj.y // TILE_SIZE))
                      for c in self.coins if c.gObj.active and not c.collected}
        enemy_cells = {(int(e.gObj.x // TILE_SIZE), int(e.gObj.y // TILE_SIZE))
                       for e in self.enemies if e.gObj.active}

        for delta_y in range(-4, 5):
            for delta_x in range(-5, 6):
                tile_x, tile_y = player_x + delta_x, player_y + delta_y
                if 0 <= tile_y < self.level_rows and 0 <= tile_x < self.level_cols:
                    tiles.append(float(self.level_data[tile_y][tile_x]))
                    coins_map.append(1.0 if (tile_x, tile_y) in coin_cells else 0.0)
                    enemies_map.append(1.0 if (tile_x, tile_y) in enemy_cells else 0.0)
                else:
                    tiles.append(0.0); coins_map.append(0.0); enemies_map.append(0.0)
        return tiles + coins_map + enemies_map

    def _object_obs(self) -> List[float]:
        """
        Returns object proximity and player status observations.
        1. Nearest Enemy Distance (normalized)
        2. Nearest Coin Distance (normalized)
        3. Powered Up (1.0 or 0.0)
        4. Invincible Timer / Max Invincible Time
        5. Active Enemies Count / 10.0
        6. Active Coins Count / 10.0
        7. Active Powerups Count / 5.0
        8. Total Coins Collected / 10.0
        9. Score / 1000.0
        10. Frame / Max Steps (or large number if unlimited)
        """
        
        
        player = self.player
        
        def nearest(objs):
            """
            Returns nearest distance to any object in objs from player.
            Normalized by dividing by 1000.0
            """
            min_dis = 1000.0
            for obj in objs:
                if getattr(obj, "active", True):
                    distance = abs(player.gObj.x - obj.x) + abs(player.gObj.y - obj.y)
                    min_dis = min(min_dis, distance)
            return min_dis / 1000.0

        min_enemy = nearest([enemy.gObj for enemy in self.enemies if enemy.gObj.active])
        min_coin  = nearest([coin.gObj for coin in self.coins if coin.gObj.active and not coin.collected])
        
        # LOGIC: (Player X - Goal X) / Normalization Factor
        # We divide by level_width so the value is usually between -1.0 and 0.0
        level_w = max(1.0, float(self.level_w))
        goal_x = getattr(self, "goal_x", level_w)
        dist_to_goal = (self.level_w - player.gObj.x) / max(1.0, float(self.level_w))
        
        # This gives:
        # -0.9 = Far left
        # -0.1 = Close
        #  0.0 = On Goal
        # +0.1 = Oversho
        
        return [ 
            min_enemy, min_coin, dist_to_goal,
            1.0 if player.powered_up else 0.0,
            player.invincible_timer / 300.0,
            len([enemy for enemy in self.enemies if enemy.gObj.active]) / 10.0,
            len([coin for coin in self.coins if coin.gObj.active and not coin.collected]) / 10.0,
            len([powerup for powerup in self.powerups if powerup.gObj.active]) / 5.0,
            self.coins_total / 10.0,
            self.score / 1000.0,
            self.frame / float(self.max_steps if self.max_steps else 1e9), ]


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
            "enemies_killed": self.kills_step,
            "powered_up": p.powered_up, 
            "terminated": not self.alive,
            "won": (self.reached_goal and not self.game_over),
            "action": self._last_action,
            "time_left": math.ceil(self.timer),
            "max_x_seen": self.max_x_seen, 
            "stall_windows": self.stall_windows_count,
            "stalled": self.stalled_this_frame,
            "persona": self.persona # Expose current persona for debug tools
        }

    #depretiated?
    def _tile_at(self, x: float, y: float) -> int:
        col = int(x // TILE_SIZE)
        row = int(y // TILE_SIZE)
        if 0 <= row < self.level_rows and 0 <= col < self.level_cols:
            return self.level_data[row][col]
        return TILE_AIR

    # ---------------------------------------------------------------------
    # Rendering
    # ---------------------------------------------------------------------
    def render(self, surface: pygame.Surface, blit_only: bool = True):
        """
        Render all entities and overlays
        """
        surface.fill(COLOR_SKY)
        
        self._draw_world_from_hash(surface)
        self._draw_entities_from_hash(surface)
        self._draw_player(surface) 
        
        self._update_debug_key_toggles()
        # DELEGATE DEBUG RENDERING TO THE MANAGER
        # This fixes the AttributeError: 'PlatformerCore' object has no attribute 'db_sensors'
        if self.debug_manager.show_hitboxes or self.debug_manager.show_sensors or \
           self.debug_manager.show_agent_view or self.debug_manager.show_obs_panel or \
           self.debug_manager.show_grid:
            self.debug_manager.render_overlays(surface, self)
            
        self._draw_ui(surface)

    def _draw_world_from_hash(self, surface: pygame.Surface):
        """
        Helper function for render() draws static hash block within window rect
        """
        visible_tiles = self.static_hash.query_rect(self.camera_x, self.camera_y, self.WIDTH, self.HEIGHT)
        
        for tile in visible_tiles:
            if tile.x + tile.width < self.camera_x or tile.x > self.camera_x + self.WIDTH: 
                continue
            
            if not hasattr(tile, "type_id"): 
                continue
            
            if tile.type_id == TILE_QBLOCK:
                self._draw_qblock(surface, tile)
            else:
                tile.render(surface, self.camera_x, self.camera_y)

    def _draw_qblock(self, surface: pygame.Surface, tile: Tile):
        """
        Helper function for q block render << move to q block module
        """
        col, row = int(tile.x // TILE_SIZE), int(tile.y // TILE_SIZE)
        hit = False
        for qb in self.qblocks:
            qc, qr = qb.tc()
            if qc == col and qr == row:
                hit = qb.hit; break
        
        orig_col = tile.color
        if hit: tile.color = COLOR_EMPTY
        tile.render(surface, self.camera_x, self.camera_y)
        tile.color = orig_col
        
        if not hit:
            sx = tile.x - self.camera_x; sy = tile.y - self.camera_y
            q = self.qblock_font.render("?", True, COLOR_WHITE)
            surface.blit(q, q.get_rect(center=(sx + TILE_SIZE // 2, sy + TILE_SIZE // 2)))

    def _draw_entities_from_hash(self, surface: pygame.Surface):
        """
        Helper function for render() - Draws all entities within dynamic hash within camera rect 
        """
        visible_objs = self.dynamic_hash.query_rect(self.camera_x, self.camera_y, self.WIDTH, self.HEIGHT)
        for obj in visible_objs:
            if obj.x + obj.width < self.camera_x or obj.x > self.camera_x + self.WIDTH: continue
            
            sx = obj.x - self.camera_x
            sy = obj.y - self.camera_y
            
            if isinstance(obj, Enemy):
                obj.render(surface, sx, sy)
            elif isinstance(obj, Coin):
                obj.render(surface, sx, sy)
            elif isinstance(obj, Powerup):
                obj.render(surface, sx, sy)

    def _draw_player(self, surface: pygame.Surface):
        player = self.player
        screen_x, screen_y, on_screen = self._world_to_screen(player.gObj)
        colour = COLOR_POWERUP_STAR if (player.invincible_timer > 0 and (self.frame // 5) % 2) else \
              ((255, 100, 0) if player.powered_up else (255, 0, 0))
        player.color = colour
        # Fix: use debug_manager flag instead of self.db_sensors
        player.render(surface, screen_x, screen_y, self.debug_manager.show_sensors)

        # speed line - Urgent move to render in player
        # if p.run_pressed and abs(p.vx) > self.max_walk * 0.6:
        #     n = 3; spacing = 6; length = 10
        #     for i in range(n):
        #         offset = (i + 1) * spacing
        #         if p.facing_right: x1 = sx - offset; x2 = x1 - length
        #         else: x1 = sx + p.gObj.width + offset; x2 = x1 + length
        #         y = sy + 10 + (i % 2) * 4
        #         pygame.draw.line(surface, COLOR_STREAK, (int(x1), int(y)), (int(x2), int(y)), 2)
    
    # Removed _draw_debug since it's now handled by DebugManager

    def _draw_ui(self, surface: pygame.Surface):
        """"
        Helper function for Render() - Draws UI elements
        1. Lives, Score, Coins, Powerup Status, Time
        2. Timer with warning color if low    
        """
        # Helper function for Render() - Draws UI elements
        player = self.player
        font = self.ui_font
        
        status = "STAR" if player.invincible_timer > 0 else ("SUPER" if player.powered_up else "SMALL")
        text_stats = font.render(
            f"Lives:{self.lives}  Score:{self.score}  Coins:{self.coins_total}  {status}  Time:{int(self.timer)}",
            True, COLOR_WHITE
        )
        
        x = 5
        y = 5
        # Fix: use debug_manager flag
        if self.debug_manager.show_obs_panel: y = self.HEIGHT - text_stats.get_height() - 10
        
        ui_background = pygame.Surface((text_stats.get_width() + 10, text_stats.get_height() + 6), pygame.SRCALPHA)
        ui_background.fill((0, 0, 0, 170))
        surface.blit(ui_background, (x, y))
        surface.blit(text_stats, (x + 5, y + 3))

        if self.use_timer:
            timer_s = math.ceil(self.timer)
            text_color = (255, 80, 80) if timer_s <= self.timer_warn_threshold else COLOR_WHITE
            text_font = self.ui_font # Use the unified font
            timer_text = text_font.render(f"TIME {timer_s:03d}", True, text_color)
            text_pos_x = self.WIDTH - timer_text.get_width() - 5
            text_background = pygame.Surface((timer_text.get_width() + 10, timer_text.get_height() + 6), pygame.SRCALPHA)
            text_background.fill((0, 0, 0, 170))
            surface.blit(text_background, (text_pos_x - 5, 5 - 3))
            surface.blit(timer_text, (text_pos_x, 5))

    def _update_debug_key_toggles(self):
        if self.render_mode == "human":
             self.debug_manager.update_input()

    def _world_to_screen(self, gObj:GameObject) -> Tuple[float, float, bool]:
        screen_x = gObj.x - self.camera_x
        screen_y = gObj.y - self.camera_y
        on_screen = (
                screen_x < SCREEN_WIDTH and
                screen_x + gObj.width > 0 and
                screen_y < SCREEN_HEIGHT and
                screen_y + gObj.height > 0)
        
        return screen_x, screen_y, on_screen