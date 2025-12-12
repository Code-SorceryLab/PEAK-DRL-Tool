
from __future__ import annotations
import os
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import numpy as np
import pygame
import time
from gymnasium import spaces

##
# Modules 
from .modules.GameObject import GameObject
from .modules.Tile import Tile, create_tile
from .modules.Player import Player
from .modules.Enemy import Enemy
from .modules.Powerup import Powerup
from .modules.Coin import Coin
from .modules.QuestionBlock import QuestionBlock
from .modules.SpatialHash import SpatialHash

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

PLATFORMER_WIDTH, PLATFORMER_HEIGHT = 20, 32

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

        # -------- Config knobs ----------------------------
        self.world = str(kwargs.pop("world", "1-1")).lower()
        self.speed_mult = float(kwargs.pop("speed_mult", 2.0))
        self.debug_default = bool(kwargs.pop("debug_default", True))
        self.max_steps = kwargs.pop("max_steps", None)

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
        self.anti_stall = bool(kwargs.pop("anti_stall", True if self.render_mode != "human" else False))
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
        self.level_list = ["stage_1.txt", "stage_2.txt", "stage_3.txt", "stage_4.txt", "stage_5.txt"]
        self.current_level_idx = 0

        if self.level_file is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            candidate = os.path.join(base_dir, "levels", "stage_1.txt")
            if os.path.exists(candidate):
                self.level_file = candidate

        self.player_start = (100.0, 350.0)

        # Apply speed multiplier
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
        self.static_hash = SpatialHash(cell_size=64)
        self.dynamic_hash = SpatialHash(cell_size=64)
        
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

        # Gym spaces
        obs_len = 5 + (11 * 9 * 3) + 10
        self._obs_space = spaces.Box(low=0.0, high=1e9, shape=(obs_len,), dtype=np.float32)
        self._act_space = spaces.Discrete(8)

        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()
        self._surf = pygame.Surface((self.WIDTH, self.HEIGHT))

        self.db_hitboxes = self.debug_default
        self.db_agentview = self.debug_default
        self.db_sensors = self.debug_default
        self.db_obs_panel = self.debug_default
        self.db_tile_grid = False
        self._prev_keys = pygame.key.get_pressed()

        # --- Fonts ---
        self.ui_font = pygame.font.SysFont("arial", 20, bold=True)
        self.qblock_font = pygame.font.SysFont("arial", 26, bold=True)

        self.reset()

    def get_action_space(self): 
        return self._act_space
    
    def get_observation_space(self): 
        return self._obs_space

    def reset(self) -> np.ndarray:

        self.lives = self.max_lives # Reset lives to 3

        self.enemies.clear(); self.powerups.clear(); self.coins.clear(); self.qblocks.clear()
        self.dynamic_hash.clear(); self.static_hash.clear()
        
        self.score = 0; self.coins_total = 0; self.alive = True; self.frame = 0
        self.game_over = False; self.reached_goal = False

        if self.level_file:
            self.level_data = self._build_level_from_txt(self.level_file)
        else:
            print("ERROR WITH LEVEL_FILE")
            exit

        self._post_level_resize()
        self._create_player()

        self.progress_x_best = self.player.gObj.x
        self.progress_y_best = self.level_h - self.player.gObj.y
        self.stall_timer = 0; self.stall_windows_count = 0; self.stalled_this_frame = False
        self.camera_x = 0.0; self.camera_y = 0.0
        self.last_score = 0; self.last_x = self.player.gObj.x

        if not self.level_file and self.world == "1-1":
            self._spawn_static_actors_for_world()

        self.timer = self.timer_seconds if self.use_timer else math.inf
        return self._obs()

    def step(self, action: int):
        if not self.alive:
            return self._obs(), 0.0, True, {"episode_end": True, "won": self.reached_goal}

        if self.render_mode != "human":
            self.dt = 1 / 60.0
        else:
            time_curr_step = time.time()
            raw_dt = time_curr_step - self.time_last_step
            self.time_last_step = time_curr_step
            self.dt = min(raw_dt, 0.05)
            
        self.frame += 1
        if self.use_timer: self.timer -= self.dt

        self._last_action = int(action)
        self.last_x = self.player.gObj.x
        self.kills_step = self.coins_step = self.powerups_step = 0
        self.stalled_this_frame = False

        self._rebuild_dynamic_hash()
        
        self._handle_action(int(action))
        self._update_physics(self.dt)
        self._update_objects(self.dt)
        self._handle_object_collisions()
        self._update_camera()
        if self.anti_stall:
            self._update_stall_metrics()

        terminated = self._check_termination()
        self.score_delta = self.score - self.last_score
        self.last_score = self.score
        info = self._info()
        if terminated: info["episode_end"] = True
        return self._obs(), float(self._reward()), bool(terminated), info

    def _rebuild_dynamic_hash(self):
        self.dynamic_hash.clear()
        for e in self.enemies:
            if e.gObj.active: self.dynamic_hash.insert(e)
        for c in self.coins:
            if c.gObj.active and not c.collected: self.dynamic_hash.insert(c)
        for pu in self.powerups:
            if pu.gObj.active: self.dynamic_hash.insert(pu)

    def load_level(self, idx):
        level_name = self.level_list[idx]
        base_dir = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(base_dir, "levels", level_name)
        if os.path.exists(candidate):
            self.level_file = candidate

    def complete_level(self):
        print ("completed level\n")
        if self.current_level_idx < len(self.level_list) - 1:
            print("next level\n")
            self.current_level_idx += 1
            self.load_level(self.current_level_idx)
        else:
            print("Congratulations! All levels complete.")

    def _post_level_resize(self):
        self.level_rows = len(self.level_data) if self.level_data else 0
        self.level_cols = len(self.level_data[0]) if self.level_rows > 0 else 0
        self.level_w = self.level_cols * TILE_SIZE
        self.level_h = self.level_rows * TILE_SIZE

    def _build_level_from_txt(self, path: str) -> List[List[int]]:
        """
        ASCII level loader.
        """
        if not os.path.exists(path): return [] 

        with open(path, "r") as f:
            lines = [ln.rstrip("\n") for ln in f.readlines()]

        rows = len(lines)
        cols = max(len(ln) for ln in lines) if rows else 0

        lvl = [[TILE_AIR for _ in range(cols)] for _ in range(rows)]
        self.level_tiles = [[None for _ in range(cols)] for _ in range(rows)]
        
        self.static_hash.clear()
        self.qblocks.clear(); self.coins.clear(); self.enemies.clear(); self.powerups.clear()
        self.player_start = (100.0, 350.0)

        for r in range(rows):
            row = lines[r]
            for c in range(len(row)):
                ch = row[c]
                t = TILE_AIR
                color = COLOR_SKY
                solid = False
                
                if ch == '#': t = TILE_GROUND; color = COLOR_GROUND; solid = True
                elif ch == '=': t = TILE_PLATFORM; color = COLOR_PLATFORM; solid = True
                elif ch == 'G': t = TILE_GOAL; color = COLOR_GOAL; solid = False
                elif ch == '^': t = TILE_SPIKE; color = COLOR_SPIKE; solid = False
                elif ch == '?': 
                    t = TILE_QBLOCK; color = COLOR_QBLOCK; solid = True
                    qblock = QuestionBlock(gObj=GameObject(c * TILE_SIZE, r * TILE_SIZE, TILE_SIZE, TILE_SIZE, True), contains="coin")
                    self.qblocks.append(qblock)
                    # Don't insert logic object into hash:
                    # self.static_hash.insert(qblock)
                elif ch == 'C':
                    self.coins.append(Coin(gObj=GameObject(c * TILE_SIZE + 8, r * TILE_SIZE + 8, 16, 16, True)))
                elif ch == 'E':
                    self.enemies.append(Enemy(GameObject(c * TILE_SIZE + 8, r * TILE_SIZE + 8, 25, 20, True), vx=-60.0))
                elif ch == 'P':
                    self.player_start = (c * TILE_SIZE, r * TILE_SIZE)

                lvl[r][c] = t
                if t != TILE_AIR:
                    new_tile = create_tile(t, c * TILE_SIZE, r * TILE_SIZE, solid, color)
                    self.level_tiles[r][c] = new_tile
                    
                    if solid or t in (TILE_SPIKE, TILE_GOAL, TILE_QBLOCK):
                        self.static_hash.insert(new_tile)
        return lvl

    def _spawn_static_actors_for_world(self):
        for col in (26, 32, 60, 110, 140):
            self.enemies.append(Enemy(GameObject(col * TILE_SIZE, (14-1)*TILE_SIZE, 20, 18, True), vx=-60.0))
        for (cx, ry) in [(21, 10), (22, 10)]:
            self.coins.append(Coin(gObj=GameObject(cx*TILE_SIZE+8, ry*TILE_SIZE+8, 16, 16, True)))

    def _create_player(self):
        x, y = self.player_start
        self.player = Player(gObj=GameObject(float(x), float(y), PLATFORMER_WIDTH, PLATFORMER_HEIGHT, True))

    def _handle_action(self, a: int):
        self.player.handle_input(a = a)

    def _update_physics(self, dt: float):
        self.player.update(dt)
        self._resolve_player_tiles()

    def _is_solid(self, t: int) -> bool:
        return t in (TILE_GROUND, TILE_PLATFORM, TILE_QBLOCK)

    def _tile_rects_near(self, obj: GameObject):
        # Optimized grid lookup for Player physics
        tx0 = max(0, int(obj.x // TILE_SIZE) - 1)
        tx1 = min(self.level_cols, int((obj.x + obj.width) // TILE_SIZE) + 2)
        ty0 = max(0, int(obj.y // TILE_SIZE) - 1)
        ty1 = min(self.level_rows, int((obj.y + obj.height) // TILE_SIZE) + 2)
        out = []
        for r in range(ty0, ty1):
            for c in range(tx0, tx1):
                t = self.level_data[r][c]
                if self._is_solid(t):
                    out.append((r, c, pygame.Rect(c * TILE_SIZE, r * TILE_SIZE, TILE_SIZE, TILE_SIZE), t))
        return out

    def _resolve_player_tiles(self):
        p = self.player
        prect = p.gObj.get_rect()
        for (r, c, trect, tt) in self._tile_rects_near(p.gObj):
            if not prect.colliderect(trect): continue
            ox = min(prect.right - trect.left, trect.right - prect.left)
            oy = min(prect.bottom - trect.top, trect.bottom - prect.top)
            if ox < oy:
                p.gObj.x = (trect.left - p.gObj.width) if prect.centerx < trect.centerx else trect.right
                p.vx *= 0.5
            else:
                if prect.centery < trect.centery:
                    if abs(prect.bottom - trect.top)< max(2, p.vy + 1):
                        p.gObj.y = trect.top - p.gObj.height
                        p.vy = 0
                        p.on_ground = True
                        p.jump_hold = 0
                else:
                    p.gObj.y = trect.bottom
                    p.vy = max(0.0, p.vy)
                    if tt == TILE_QBLOCK and p.vy <=0: self._hit_qblock(c, r)
            prect = p.gObj.get_rect()

    def _hit_qblock(self, col: int, row: int):
        for b in self.qblocks:
            bc, br = b.tc()
            if bc == col and br == row and not b.hit:
                b.hit = True
                spawn_x, spawn_y = col * TILE_SIZE, row * TILE_SIZE - 22
                if b.contains == "coin":
                    self.coins.append(Coin(gObj=GameObject(col*TILE_SIZE+8, row*TILE_SIZE+8, 16, 16, True),
                                flyup=True, vy=-280.0, life=0.3, auto_collect=True))
                elif b.contains == "mushroom":
                    self.powerups.append(Powerup(gObj=GameObject(spawn_x, spawn_y, 20, 20, True), kind="mushroom"))
                else:
                    self.powerups.append(Powerup(gObj=GameObject(spawn_x, spawn_y, 20, 20, True), kind="star"))
                self.level_data[row][col] = TILE_PLATFORM
                break

    def _update_objects(self, dt: float):
        for e in self.enemies:
            if e.gObj.active:
                nearby = self.static_hash.query(e)
                e.update(dt, list(nearby))
                self._resolve_enemy_tiles(e=e)
                
        for pu in self.powerups:
            if pu.gObj.active:
                nearby = self.static_hash.query(pu)
                pu.update(dt, list(nearby)) 
                
        for c in self.coins:
            if c.gObj.active:
                c.update(dt)
                if c.auto_collect and c.flyup and c.life <= 0:
                    c.gObj.active = False
                    self.coins_total += 1; self.coins_step += 1; self.score += 10
    
    def _resolve_enemy_tiles(self, e: Enemy):
        r = e.gObj.get_rect()
        nearby_objects = self.dynamic_hash.query(e)
        for other in nearby_objects:
            # Only process Enemy objects (skip coins, powerups, etc.)
            if not isinstance(other, Enemy):
                continue
            # Skip self-collision
            if other is e or not other.gObj.active:
                continue
            other_rect = other.gObj.get_rect()
            if not r.colliderect(other_rect):
                continue
            # Calculate overlap amounts
            ox = min(r.right - other_rect.left, other_rect.right - r.left)
            oy = min(r.bottom - other_rect.top, other_rect.bottom - r.top)
            if ox < oy:
                # Horizontal collision - push apart and bounce
                if r.centerx < other_rect.centerx:
                    e.gObj.x = other_rect.left - e.gObj.width
                else:
                    e.gObj.x = other_rect.right
                e.vx *= -1.0
            else:
                # Vertical collision - stack or separate vertically
                if r.centery < other_rect.centery:
                    e.gObj.y = other_rect.top - e.gObj.height
                else:
                    e.gObj.y = other_rect.bottom
                e.vy = 0.0
            r = e.gObj.get_rect()
            
    def _handle_object_collisions(self):
        p = self.player
        moving_down = p.vy > 0
        
        nearby_objects = self.dynamic_hash.query(p)
        
        for obj in nearby_objects:
            if isinstance(obj, Enemy):
                e = obj
                if not e.gObj.active: continue
                if p.gObj.collides_with(e.gObj):
                    p_bottom = p.gObj.y + p.gObj.height
                    e_center = e.gObj.y + e.gObj.height/2
                    
                    if p_bottom < e_center + 10 and moving_down:
                        # Platformer jumped on enemy
                        e.gObj.active = False
                        p.vy = JUMP_VEL_MIN * 0.6
                        self.score += 100; self.kills_step += 1
                    elif p.invincible_timer > 0:
                        # Star power
                        e.gObj.active = False
                        self.score += 100; self.kills_step += 1
                    else:
                        # Lost powerup
                        if p.powered_up:
                            p.powered_up = False; p.invincible_timer = 60
                        else:
                            # DIED TO ENEMY
                            self._handle_death() # If lives > 0, we just respawned. 
                            # We return immediately to prevent physics glitches this frame.
                            return

            elif isinstance(obj, Coin):
                c = obj
                if c.gObj.active and not c.collected and p.gObj.collides_with(c.gObj):
                    c.gObj.active = False; c.collected = True
                    self.coins_total += 1; self.coins_step += 1; self.score += 10

            elif isinstance(obj, Powerup):
                pu = obj
                if pu.gObj.active and p.gObj.collides_with(pu.gObj):
                    pu.gObj.active = False
                    self.powerups_step += 1
                    if pu.kind == "mushroom":
                        p.powered_up = True; self.score += 50
                    else:
                        p.invincible_timer = 300; self.score += 100

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
        if not self.camera_lock or not self.player: return
        
        target_x = max(0, min(self.player.gObj.x - self.WIDTH // 3, self.level_w - self.WIDTH))
        self.camera_x += (target_x - self.camera_x) * self.camera_smoothing
        self.camera_x = max(0, min(self.camera_x, max(0, self.level_w - self.WIDTH)))

        target_y = 0.0
        if self.level_h > self.HEIGHT:
            target_y = max(0, min(self.player.gObj.y - self.HEIGHT // 2, self.level_h - self.HEIGHT))
        self.camera_y += (target_y - self.camera_y) * self.camera_smoothing
        self.camera_y = max(0, min(self.camera_y, max(0, self.level_h - self.HEIGHT)))

    def _progress_components(self):
        if not self.player: return 0.0, 0.0
        return self.player.gObj.x, self.level_h - self.player.gObj.y

    def _update_stall_metrics(self):
        if not self.anti_stall or not self.player: return
        prog_x, prog_y = self._progress_components()
        progressed = False

        if prog_x > self.progress_x_best + self.stall_eps:
            self.progress_x_best = prog_x; progressed = True
        if prog_y > self.progress_y_best + self.stall_eps:
            self.progress_y_best = prog_y; progressed = True

        if progressed:
            self.stall_timer = 0; self.stall_windows_count = 0; self.stalled_this_frame = False
            return

        self.stall_timer += self.dt
        if self.stall_timer >= self.stall_window:
            self.stalled_this_frame = True
            self.stall_timer -= self.stall_window
            self.stall_windows_count += 1

    def _check_termination(self) -> bool:
        p = self.player
        
        # 1. TIME LIMIT CHECK
        if self.use_timer and self.timer <= 0:
            self._handle_death()
            # If we still have lives, we are NOT terminated yet
            return not self.alive 

        # 2. PIT CHECK
        if p.gObj.y > self.level_h:
            self._handle_death()
            return not self.alive
        
        # 3. GOAL CHECK
        tile = self._tile_at(p.gObj.x + p.gObj.width / 2, p.gObj.y + p.gObj.height / 2)
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
        if self.anti_stall and self.stall_windows_count >= self.stall_kill_windows:
            self.alive = False; self.reached_goal = False
            return True

        return False

    def _reward(self) -> float:
        r = 0.0
        dx = self.player.gObj.x - self.last_x
        r += dx / 8.0
        r -= 0.005
        r += self.kills_step * 1.0 + self.coins_step * 0.5 + self.powerups_step * 1.0
        r += self.score_delta / 100.0
        if self.anti_stall:
            if self.stalled_this_frame:
                r += self.stall_penalty 
            
            prog_x, prog_y = self._progress_components()
            back_x = max(0.0, self.progress_x_best - prog_x)
            back_y = max(0.0, self.progress_y_best - prog_y)
            back = max(back_x, back_y)
                
            if back > 0:
                r -= self.backtrack_penalty * min(back, TILE_SIZE * 8)
        return r

    def _obs(self) -> np.ndarray:
        out: List[float] = []
        out.extend(self._player_obs())
        out.extend(self._tile_window_obs())
        out.extend(self._object_obs())
        return np.array(out, dtype=np.float32)

    def _player_obs(self) -> List[float]:
        p = self.player
        lw = max(1.0, float(self.level_w))
        lh = max(1.0, float(self.level_h))

        return [
            p.gObj.x / lw,
            p.gObj.y / lh,
            p.vx / max(1e-6, self.max_run),
            p.vy / MAX_FALL_SPEED,
            1.0 if p.on_ground else 0.0,
        ]
        
    def _tile_window_obs(self) -> List[float]:
        p = self.player
        px = int(p.gObj.x // TILE_SIZE); py = int(p.gObj.y // TILE_SIZE)
        tiles: List[float] = []; coins_map: List[float] = []; enemies_map: List[float] = []

        coin_cells = {(int(c.gObj.x // TILE_SIZE), int(c.gObj.y // TILE_SIZE))
                      for c in self.coins if c.gObj.active and not c.collected}
        enemy_cells = {(int(e.gObj.x // TILE_SIZE), int(e.gObj.y // TILE_SIZE))
                       for e in self.enemies if e.gObj.active}

        for dy in range(-4, 5):
            for dx in range(-5, 6):
                tx, ty = px + dx, py + dy
                if 0 <= ty < self.level_rows and 0 <= tx < self.level_cols:
                    tiles.append(float(self.level_data[ty][tx]))
                    coins_map.append(1.0 if (tx, ty) in coin_cells else 0.0)
                    enemies_map.append(1.0 if (tx, ty) in enemy_cells else 0.0)
                else:
                    tiles.append(0.0); coins_map.append(0.0); enemies_map.append(0.0)
        return tiles + coins_map + enemies_map

    def _object_obs(self) -> List[float]:
        p = self.player
        def nearest(objs):
            m = 1000.0
            for o in objs:
                if getattr(o, "active", True):
                    d = abs(p.gObj.x - o.x) + abs(p.gObj.y - o.y)
                    m = min(m, d)
            return m / 1000.0

        min_enemy = nearest([e.gObj for e in self.enemies if e.gObj.active])
        min_coin  = nearest([c.gObj for c in self.coins if c.gObj.active and not c.collected])
        return [
            min_enemy, min_coin,
            1.0 if p.powered_up else 0.0,
            p.invincible_timer / 300.0,
            len([e for e in self.enemies if e.gObj.active]) / 10.0,
            len([c for c in self.coins if c.gObj.active and not c.collected]) / 10.0,
            len([u for u in self.powerups if u.gObj.active]) / 5.0,
            self.coins_total / 10.0,
            self.score / 1000.0,
            self.frame / float(self.max_steps if self.max_steps else 1e9),
        ]

    def _info(self) -> Dict:
        p = self.player
        return {
            "score": self.score, "score_delta": self.score_delta, "frame_count": self.frame,
            "x_position": p.gObj.x, "y_position": p.gObj.y, "velocity_x": p.vx,
            "coins_collected": self.coins_total, "enemies_killed": self.kills_step,
            "powered_up": p.powered_up, "terminated": not self.alive,
            "won": (self.reached_goal and not self.game_over),
            "action": self._last_action,
            "time_left": math.ceil(self.timer),
            "max_x_seen": self.max_x_seen, "stall_windows": self.stall_windows_count,
            "stalled": self.stalled_this_frame,
        }

    def _tile_at(self, x: float, y: float) -> int:
        c = int(x // TILE_SIZE); r = int(y // TILE_SIZE)
        if 0 <= r < self.level_rows and 0 <= c < self.level_cols:
            return self.level_data[r][c]
        return TILE_AIR

    # ---------------------------------------------------------------------
    # Rendering
    # ---------------------------------------------------------------------
    def render(self, surface: pygame.Surface, blit_only: bool = True):
        surface.fill(COLOR_SKY)
        
        self._draw_world_from_hash(surface)
        self._draw_entities_from_hash(surface)
        self._draw_player(surface) 
        
        self._update_debug_key_toggles()
        if self.db_hitboxes or self.db_sensors or self.db_agentview or self.db_obs_panel or self.db_tile_grid:
            self._draw_debug(surface)
        self._draw_ui(surface)

    def _draw_world_from_hash(self, surface: pygame.Surface):
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
        c, r = int(tile.x // TILE_SIZE), int(tile.y // TILE_SIZE)
        hit = False
        for qb in self.qblocks:
            qc, qr = qb.tc()
            if qc == c and qr == r:
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
        p = self.player
        sx, sy, on_screen = self._world_to_screen(p.gObj)
        col = COLOR_POWERUP_STAR if (p.invincible_timer > 0 and (self.frame // 5) % 2) else \
              ((255, 100, 0) if p.powered_up else (255, 0, 0))
        p.color = col
        p.render(surface, sx, sy, self.db_sensors)

        if p.run_pressed and abs(p.vx) > self.max_walk * 0.6:
            n = 3; spacing = 6; length = 10
            for i in range(n):
                offset = (i + 1) * spacing
                if p.facing_right: x1 = sx - offset; x2 = x1 - length
                else: x1 = sx + p.gObj.width + offset; x2 = x1 + length
                y = sy + 10 + (i % 2) * 4
                pygame.draw.line(surface, COLOR_STREAK, (int(x1), int(y)), (int(x2), int(y)), 2)

    def _draw_debug(self, surface: pygame.Surface):
        p = self.player
        px, py, _ = self._world_to_screen(p.gObj)
        
        if self.db_hitboxes:
            pygame.draw.rect(surface, COLOR_HITBOX, (px, py, p.gObj.width, p.gObj.height), 2)
            visible = self.dynamic_hash.query_rect(self.camera_x, self.camera_y, self.WIDTH, self.HEIGHT)
            for o in visible:
                sx, sy, _ = self._world_to_screen(o.gObj)
                w = o.gObj.width
                h = o.gObj.height
                pygame.draw.rect(surface, (255, 255, 255), (sx, sy, w, h), 1)

        if self.db_obs_panel:
            obs = self._obs()
            lines = [
                f"obs[0:5]=[{obs[0]:.3f},{obs[1]:.3f},{obs[2]:.3f},{obs[3]:.3f},{obs[4]:.1f}]",
                f"max_x={int(self.max_x_seen)} stalled={self.stalled_this_frame} stallW={self.stall_windows_count}",
            ]
            line_height = self.ui_font.get_height()
            box = pygame.Surface((400, line_height * len(lines) + 10), pygame.SRCALPHA)
            box.fill((0, 0, 0, 160))
            surface.blit(box, (10, 10))
            for i, ln in enumerate(lines):
                surface.blit(self.ui_font.render(ln, True, COLOR_WHITE), (16, 16 + i * line_height))

    def _draw_ui(self, surface: pygame.Surface):
        p = self.player
        font = self.ui_font
        
        status = "STAR" if p.invincible_timer > 0 else ("SUPER" if p.powered_up else "SMALL")
        ts = font.render(
            f"Lives:{self.lives}  Score:{self.score}  Coins:{self.coins_total}  {status}  Time:{int(self.timer)}",
            True, COLOR_WHITE
        )
        
        x = 5; y = 5
        if self.db_obs_panel: y = self.HEIGHT - ts.get_height() - 10
        
        bg = pygame.Surface((ts.get_width() + 10, ts.get_height() + 6), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 170))
        surface.blit(bg, (x, y))
        surface.blit(ts, (x + 5, y + 3))

        if self.use_timer:
            timer_s = math.ceil(self.timer)
            tcol = (255, 80, 80) if timer_s <= self.timer_warn_threshold else COLOR_WHITE
            tfont = self.ui_font # Use the unified font
            ttext = tfont.render(f"TIME {timer_s:03d}", True, tcol)
            tx = self.WIDTH - ttext.get_width() - 5
            tbg = pygame.Surface((ttext.get_width() + 10, ttext.get_height() + 6), pygame.SRCALPHA)
            tbg.fill((0, 0, 0, 170))
            surface.blit(tbg, (tx - 5, 5 - 3))
            surface.blit(ttext, (tx, 5))

    def _update_debug_key_toggles(self):
        keys = pygame.key.get_pressed()
        toggles = [
            (pygame.K_1, pygame.K_F1, "db_hitboxes"),
            (pygame.K_2, pygame.K_F2, "db_agentview"),
            (pygame.K_3, pygame.K_F3, "db_sensors"),
            (pygame.K_4, pygame.K_F4, "db_obs_panel"),
            (pygame.K_5, pygame.K_F5, "camera_lock"),
            (pygame.K_6, pygame.K_F6, "db_tile_grid"),
        ]
        for k_num, k_fn, attr in toggles:
            if (not self._prev_keys[k_num] and keys[k_num]) or (not self._prev_keys[k_fn] and keys[k_fn]):
                if attr == "camera_lock":
                    self.camera_lock = not self.camera_lock
                else:
                    setattr(self, attr, not getattr(self, attr))
        self._prev_keys = keys

    def _world_to_screen(self, gObj:GameObject) -> Tuple[float, float, bool]:
        sx = gObj.x - self.camera_x
        sy = gObj.y - self.camera_y
        on_screen = (
                sx < SCREEN_WIDTH and
                sx + gObj.width > 0 and
                sy < SCREEN_HEIGHT and
                sy + gObj.height > 0)
        return sx, sy, on_screen