# code/games/mario_core.py
"""
SMB1-style core with deterministic 1-1 layout, faster feel, visualizer, and anti-stall shaping.

New knobs (via **game_kwargs or YAML):
  anti_stall: bool = True
  stall_window: int = 90         # frames with no forward progress before counting a "stall window"
  stall_kill_windows: int = 6    # after this many windows, truncate
  stall_eps: float = 6.0         # pixels of progress that resets the stall timer
  stall_penalty: float = -0.02   # penalty applied per counted stall window (sparse)
  backtrack_penalty: float = 0.001  # per-pixel penalty when moving left from max_x_seen
"""

from __future__ import annotations
import os
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np
import pygame
from gymnasium import spaces

# =============================================================================
# Screen / Tile geometry
# =============================================================================
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
TILE_SIZE = 32
LEVEL_ROWS, LEVEL_COLS = 18, 180  # long enough for 1-1 + flag/castle

# =============================================================================
# Mario movement tuning (some are scaled by speed_mult)
# =============================================================================
MARIO_WIDTH, MARIO_HEIGHT = 20, 32
RUN_ACCEL       = 0.55
WALK_ACCEL      = 0.45
MAX_WALK_SPEED  = 3.6
MAX_RUN_SPEED   = 6.8
GROUND_FRICTION = 0.80
AIR_FRICTION    = 0.97
AIR_CONTROL     = 0.75
SKID_DECEL      = 0.90
GRAVITY         = 0.80
FAST_FALL_GRAV  = 1.25
MAX_FALL_SPEED  = 12.5

# Jump parameters
JUMP_VEL_MIN       = -12
JUMP_VEL_MAX       = -15
JUMP_HOLD_FRAMES   = 14
SPEED_JUMP_BONUS   = 0.18
COYOTE_FRAMES      = 5
JUMP_BUFFER_FRAMES = 6

# Tiles
TILE_AIR      = 0
TILE_GROUND   = 1
TILE_PLATFORM = 2
TILE_GOAL     = 3
TILE_SPIKE    = 4
TILE_QBLOCK   = 5

# Colors (simple debug palette)
COLOR_SKY        = (107, 140, 255)
COLOR_GROUND     = (139, 69, 19)
COLOR_PLATFORM   = (205, 133, 63)
COLOR_GOAL       = (255, 215, 0)
COLOR_SPIKE      = (50, 50, 50)
COLOR_WHITE      = (255, 255, 255)
COLOR_BLACK      = (0, 0, 0)
COLOR_QBLOCK     = (255, 165, 0)
COLOR_EMPTY      = (165, 115, 50)
COLOR_ENEMY      = (139, 69, 19)
COLOR_POWERUP_MUSH = (255, 0, 0)
COLOR_POWERUP_STAR = (255, 215, 0)
COLOR_COIN       = (255, 215, 0)
COLOR_HITBOX     = (255, 64, 64)
COLOR_SENSOR     = (64, 255, 128)
COLOR_AGENT_PANEL= (30, 30, 30)
COLOR_STREAK     = (255, 255, 255)

# =============================================================================
# Basic object model
# =============================================================================
@dataclass
class GameObject:
    x: float; y: float; width: int; height: int; active: bool = True
    def get_rect(self) -> pygame.Rect:
        return pygame.Rect(int(self.x), int(self.y), self.width, self.height)
    def collides_with(self, other: "GameObject") -> bool:
        return self.active and other.active and self.get_rect().colliderect(other.get_rect())

@dataclass
class Player(GameObject):
    vx: float = 0.0; vy: float = 0.0
    on_ground: bool = False; facing_right: bool = True
    powered_up: bool = False; invincible_timer: int = 0
    coyote: int = 0; jump_hold: int = 0; jump_buffer: int = 0; last_jump_pressed: bool = False
    run_pressed: bool = False  # for streaks

@dataclass
class Enemy(GameObject):
    vx: float = -1.2; vy: float = 0.0
    bounce_clock: int = 0  # anti-trap counter
    def update(self, dt: float):
        if not self.active:
            return
        self.x += self.vx * 60 * dt
        self.vy += GRAVITY
        self.y += self.vy * 60 * dt

@dataclass
class Powerup(GameObject):
    vx: float = 1.2; vy: float = 0.0; kind: str = "mushroom"
    def update(self, dt: float):
        if not self.active:
            return
        self.x += self.vx * 60 * dt
        self.vy += GRAVITY
        self.y += self.vy * 60 * dt

@dataclass
class Coin(GameObject):
    collected: bool = False
    animation: int = 0
    flyup: bool = False
    vy: float = -4.8
    life: int = 18
    auto_collect: bool = False
    def update(self):
        if self.flyup:
            self.y += self.vy
            self.vy += 0.35
            self.life -= 1
        self.animation = (self.animation + 1) % 60

@dataclass
class QuestionBlock:
    x: int; y: int; contains: str = "coin"; hit: bool = False
    def tc(self) -> Tuple[int, int]:
        return (self.x // TILE_SIZE, self.y // TILE_SIZE)

# =============================================================================
# Core game
# =============================================================================
class MarioCore:
    WIDTH, HEIGHT = SCREEN_WIDTH, SCREEN_HEIGHT

    def __init__(self, render_mode: str = "none", **kwargs):
        # -------- Config knobs (YAML-friendly) ----------------------------
        self.world = str(kwargs.pop("world", "1-1")).lower()
        self.speed_mult = float(kwargs.pop("speed_mult", 2.0))
        self.debug_default = bool(kwargs.pop("debug_default", True))
        self.max_steps = kwargs.pop("max_steps", None)

        # Anti-stall knobs
        self.anti_stall = bool(kwargs.pop("anti_stall", True))
        self.stall_window = int(kwargs.pop("stall_window", 90))
        self.stall_kill_windows = int(kwargs.pop("stall_kill_windows", 6))
        self.stall_eps = float(kwargs.pop("stall_eps", 6.0))
        self.stall_penalty = float(kwargs.pop("stall_penalty", -0.02))
        self.backtrack_penalty = float(kwargs.pop("backtrack_penalty", 0.001))

        # Apply speed multiplier
        self.run_accel = RUN_ACCEL * self.speed_mult
        self.walk_accel = WALK_ACCEL * self.speed_mult
        self.max_walk = MAX_WALK_SPEED * self.speed_mult
        self.max_run = MAX_RUN_SPEED * self.speed_mult

        # -------- World / state ------------------------------------------
        self.level_data: List[List[int]] = [[TILE_AIR]*LEVEL_COLS for _ in range(LEVEL_ROWS)]
        self.level_w = LEVEL_COLS * TILE_SIZE
        self.level_h = LEVEL_ROWS * TILE_SIZE

        self.player: Player | None = None
        self.enemies: List[Enemy] = []
        self.powerups: List[Powerup] = []
        self.coins: List[Coin] = []
        self.qblocks: List[QuestionBlock] = []

        self.camera_x = 0.0; self.camera_smoothing = 0.15; self.camera_lock = True
        self.score = 0; self.coins_total = 0; self.alive = True; self.frame = 0
        self.game_over = False; self.reached_goal = False
        self.last_x = 0.0; self.last_score = 0; self.score_delta = 0
        self.kills_step = 0; self.coins_step = 0; self.powerups_step = 0; self._last_action = 0

        # Anti-stall state
        self.max_x_seen = 0.0
        self.stall_timer = 0
        self.stall_windows_count = 0
        self.stalled_this_frame = False

        # Gym spaces
        obs_len = 5 + (11 * 9 * 3) + 10  # player(5) + (tiles/coins/enemies 11x9 each) + object stats(10)
        self._obs_space = spaces.Box(low=0.0, high=1e9, shape=(obs_len,), dtype=np.float32)
        self._act_space = spaces.Discrete(8)

        # Pygame bootstrap
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()
        self._surf = pygame.Surface((self.WIDTH, self.HEIGHT))

        # Visualizer toggles
        self.db_hitboxes = self.debug_default
        self.db_agentview = self.debug_default
        self.db_sensors = self.debug_default
        self.db_obs_panel = self.debug_default
        self.db_tile_grid = False
        self._prev_keys = pygame.key.get_pressed()

        self.reset()

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def get_action_space(self):
        return self._act_space

    def get_observation_space(self):
        return self._obs_space

    def reset(self) -> np.ndarray:
        """Build world, (re)spawn actors, reset scoring and anti-stall telemetry."""
        self.enemies.clear(); self.powerups.clear(); self.coins.clear(); self.qblocks.clear()
        self.score = 0; self.coins_total = 0; self.alive = True; self.frame = 0
        self.game_over = False; self.reached_goal = False

        self.level_data = self._build_level_1_1() if self.world == "1-1" else self._generate_proc()
        self._create_player()
        self._spawn_static_actors_for_world()

        self.camera_x = 0.0; self.last_score = 0; self.last_x = self.player.x

        # Anti-stall reset
        self.max_x_seen = self.player.x
        self.stall_timer = 0
        self.stall_windows_count = 0
        self.stalled_this_frame = False

        return self._obs()

    def step(self, action: int):
        """Advance one frame with action, update physics, apply reward and termination."""
        if not self.alive:
            return self._obs(), 0.0, True, {"episode_end": True, "won": self.reached_goal}

        dt = 1 / 60.0
        self.frame += 1
        self._last_action = int(action)
        self.last_x = self.player.x
        self.kills_step = self.coins_step = self.powerups_step = 0
        self.stalled_this_frame = False

        self._handle_action(int(action))
        self._update_physics(dt)
        self._update_objects(dt)
        self._handle_object_collisions()
        self._update_camera()
        self._update_stall_metrics()  # anti-camping + telemetry

        terminated = self._check_termination()
        self.score_delta = self.score - self.last_score
        self.last_score = self.score
        info = self._info()
        if terminated:
            info["episode_end"] = True
        return self._obs(), float(self._reward()), bool(terminated), info

    # ---------------------------------------------------------------------
    # Level building
    # ---------------------------------------------------------------------
    def _build_level_1_1(self) -> List[List[int]]:
        """Deterministic 1-1 inspired layout with pipes, platforms, coins, and a vertical flag pole."""
        lvl = [[TILE_AIR for _ in range(LEVEL_COLS)] for _ in range(LEVEL_ROWS)]
        g = 14  # ground row index

        # Base ground
        for c in range(LEVEL_COLS):
            for r in range(g, LEVEL_ROWS):
                lvl[r][c] = TILE_GROUND

        # Pipes (simple 2-wide solid columns)
        pipes = [(35, 2), (45, 3), (55, 4), (65, 4)]
        for col, h in pipes:
            for r in range(g - h, g):
                lvl[r][col] = TILE_GROUND
                lvl[r][col + 1] = TILE_GROUND

        # Question blocks and bricks
        qb = [(20, g - 4), (21, g - 4), (22, g - 4)]
        bricks = [(23, g - 4)]
        for (cx, ry) in qb:
            lvl[ry][cx] = TILE_QBLOCK
            self.qblocks.append(QuestionBlock(cx * TILE_SIZE, ry * TILE_SIZE, contains="coin"))
        for (cx, ry) in bricks:
            lvl[ry][cx] = TILE_PLATFORM

        # Hidden safety block (lowered to avoid unreachable traps)
        lvl[g - 5][28] = TILE_PLATFORM

        # Floating platforms
        for cx in range(90, 95):
            lvl[g - 5][cx] = TILE_PLATFORM
        for cx in range(102, 106):
            lvl[g - 4][cx] = TILE_PLATFORM

        # Stairs
        base = 150
        for h in range(1, 6):
            for cx in range(base + h, base + h + 3):
                for rr in range(g - h, g):
                    lvl[rr][cx] = TILE_GROUND

        # ======== Vertical flag pole (FULL HEIGHT up to the sky) =========
        pole_col = LEVEL_COLS - 6
        for rr in range(0, g):       # from top down to just above ground
            lvl[rr][pole_col] = TILE_GOAL

        # Castle mound near the end
        for cx in range(LEVEL_COLS - 12, LEVEL_COLS - 8):
            for rr in range(g - 3, g):
                lvl[rr][cx] = TILE_GROUND

        # Coins
        for i in range(6):
            cx = 14 + i
            ry = g - 3 - (i // 3)
            self.coins.append(Coin(x=cx * TILE_SIZE + 8, y=ry * TILE_SIZE + 8, width=16, height=16))
        for col, h in pipes:
            self.coins.append(Coin(x=(col - 1) * TILE_SIZE + 8, y=(g - h - 2) * TILE_SIZE + 8, width=16, height=16))
        for cx in range(90, 95, 2):
            self.coins.append(Coin(x=cx * TILE_SIZE + 8, y=(g - 6) * TILE_SIZE + 8, width=16, height=16))
        for cx in range(102, 106, 2):
            self.coins.append(Coin(x=cx * TILE_SIZE + 8, y=(g - 5) * TILE_SIZE + 8, width=16, height=16))

        return lvl

    def _generate_proc(self) -> List[List[int]]:
        """Simple procedural layout for quick tests."""
        lvl = [[TILE_AIR for _ in range(LEVEL_COLS)] for _ in range(LEVEL_ROWS)]
        g = 14
        for c in range(LEVEL_COLS):
            for r in range(g, LEVEL_ROWS):
                lvl[r][c] = TILE_GROUND
        for cx in range(40, 47):
            lvl[g - 5][cx] = TILE_PLATFORM
        for cx in range(70, 74):
            lvl[g - 6][cx] = TILE_PLATFORM
        lvl[g - 6][100] = TILE_QBLOCK
        self.qblocks.append(QuestionBlock(100 * TILE_SIZE, (g - 6) * TILE_SIZE, contains="coin"))
        # keep a single bottom goal tile for proc world
        lvl[g][LEVEL_COLS - 2] = TILE_GOAL
        for cx in range(44, 47):
            self.coins.append(Coin(x=cx * TILE_SIZE + 8, y=(g - 6) * TILE_SIZE + 8, width=16, height=16))
        return lvl

    def _spawn_static_actors_for_world(self):
        """Place basic enemies and a couple of floating coins; avoid early micro-pockets."""
        for col in (26, 32, 60, 110, 140):
            self.enemies.append(Enemy(x=col * TILE_SIZE, y=(14 - 1) * TILE_SIZE, width=20, height=18, vx=-1.0))
        for (cx, ry) in [(21, 10), (22, 10)]:
            self.coins.append(Coin(x=cx * TILE_SIZE + 8, y=ry * TILE_SIZE + 8, width=16, height=16))

    def _create_player(self):
        self.player = Player(x=100.0, y=350.0, width=MARIO_WIDTH, height=MARIO_HEIGHT, active=True)

    # ---------------------------------------------------------------------
    # Input handling & physics
    # ---------------------------------------------------------------------
    def _handle_action(self, a: int):
        """Map discrete action to left/right/jump/run states and update desired accel."""
        p = self.player
        left  = (a in (1, 6))
        right = (a in (2, 4, 5, 7))
        jump_pressed = (a in (3, 4, 6, 7))

        kb = pygame.key.get_pressed() if pygame.get_init() else None
        kb_run = bool(kb and (kb[pygame.K_LSHIFT] or kb[pygame.K_RSHIFT]))
        run_from_action = (a in (5, 7))
        run = kb_run or run_from_action
        p.run_pressed = bool(run)

        if jump_pressed:
            p.jump_buffer = JUMP_BUFFER_FRAMES

        moving = left or right
        run_active = run and moving
        target = (self.max_run if run_active else self.max_walk) if moving else 0.0  # not used directly but informative
        accel = (self.run_accel if run_active else self.walk_accel) if moving else 0.0
        if not p.on_ground and moving:
            accel *= AIR_CONTROL

        # Horizontal accel / friction
        if left ^ right:
            ax = -accel if left else accel
            p.vx = max(-self.max_run, min(self.max_run, p.vx + ax))
            p.facing_right = not left
            if p.on_ground and ((p.vx > 0 and left) or (p.vx < 0 and right)):  # skid
                p.vx *= SKID_DECEL
        else:
            p.vx *= (GROUND_FRICTION if p.on_ground else AIR_FRICTION)
            if abs(p.vx) < 0.05:
                p.vx = 0.0

        # Coyote time & jump buffering
        p.coyote = COYOTE_FRAMES if p.on_ground else max(0, p.coyote - 1)
        if p.jump_buffer > 0:
            p.jump_buffer -= 1

        # Jump start
        if (p.coyote > 0) and (p.jump_hold == 0) and (p.jump_buffer >= 0):
            if jump_pressed:
                base = JUMP_VEL_MIN; long = JUMP_VEL_MAX
                bonus = min(2.2, abs(p.vx) * SPEED_JUMP_BONUS)
                p.vy = max(long, base - bonus)
                p.on_ground = False
                p.coyote = 0
                p.jump_hold = JUMP_HOLD_FRAMES

        # Variable jump height
        if p.jump_hold > 0:
            if jump_pressed:
                p.vy -= 0.30
            p.jump_hold -= 1

        p.last_jump_pressed = jump_pressed

    def _update_physics(self, dt: float):
        """Integrate velocity, apply gravity, resolve tile collisions."""
        p = self.player
        if p.invincible_timer > 0:
            p.invincible_timer -= 1
        grav = FAST_FALL_GRAV if p.vy > 0 else GRAVITY
        p.vy = min(p.vy + grav, MAX_FALL_SPEED)
        p.x += p.vx
        p.y += p.vy
        p.on_ground = False
        self._resolve_player_tiles()

    # ---------------------------------------------------------------------
    # Tile helpers & collision resolution
    # ---------------------------------------------------------------------
    def _is_solid(self, t: int) -> bool:
        return t in (TILE_GROUND, TILE_PLATFORM, TILE_QBLOCK)

    def _tile_rects_near(self, obj: GameObject):
        tx0 = max(0, int(obj.x // TILE_SIZE) - 1)
        tx1 = min(LEVEL_COLS, int((obj.x + obj.width) // TILE_SIZE) + 2)
        ty0 = max(0, int(obj.y // TILE_SIZE) - 1)
        ty1 = min(LEVEL_ROWS, int((obj.y + obj.height) // TILE_SIZE) + 2)
        out = []
        for r in range(ty0, ty1):
            for c in range(tx0, tx1):
                t = self.level_data[r][c]
                if self._is_solid(t):
                    out.append((r, c, pygame.Rect(c * TILE_SIZE, r * TILE_SIZE, TILE_SIZE, TILE_SIZE), t))
        return out

    def _resolve_player_tiles(self):
        p = self.player
        prect = p.get_rect()
        for (r, c, trect, tt) in self._tile_rects_near(p):
            if not prect.colliderect(trect):
                continue
            ox = min(prect.right - trect.left, trect.right - prect.left)
            oy = min(prect.bottom - trect.top, trect.bottom - prect.top)
            if ox < oy:
                p.x = (trect.left - p.width) if prect.centerx < trect.centerx else trect.right
                p.vx *= 0.5
            else:
                if prect.centery < trect.centery:
                    p.y = trect.top - p.height
                    p.vy = 0
                    p.on_ground = True
                    p.jump_hold = 0
                else:
                    p.y = trect.bottom
                    p.vy = max(0.0, p.vy)
                    if tt == TILE_QBLOCK:
                        self._hit_qblock(c, r)
            prect = p.get_rect()

    def _hit_qblock(self, col: int, row: int):
        """Spawn contents of a question block and convert to an empty platform."""
        for b in self.qblocks:
            bc, br = b.tc()
            if bc == col and br == row and not b.hit:
                b.hit = True
                spawn_x, spawn_y = col * TILE_SIZE, row * TILE_SIZE - 22
                if b.contains == "coin":
                    coin = Coin(x=col * TILE_SIZE + 8, y=row * TILE_SIZE + 8, width=16, height=16,
                                flyup=True, vy=-5.2, life=20, auto_collect=True)
                    self.coins.append(coin)
                elif b.contains == "mushroom":
                    self.powerups.append(Powerup(spawn_x, spawn_y, 20, 20, kind="mushroom"))
                else:
                    self.powerups.append(Powerup(spawn_x, spawn_y, 20, 20, kind="star"))
                self.level_data[row][col] = TILE_PLATFORM
                break

    # ---------------------------------------------------------------------
    # Object updates & interactions
    # ---------------------------------------------------------------------
    def _update_objects(self, dt: float):
        for e in self.enemies:
            if e.active:
                e.update(dt)
                self._resolve_enemy_tiles(e)
        for pu in self.powerups:
            if pu.active:
                pu.update(dt)
                self._resolve_powerup_tiles(pu)
        for c in self.coins:
            if not c.active:
                continue
            c.update()
            if c.auto_collect and c.flyup and c.life <= 0:
                c.active = False
                self.coins_total += 1
                self.coins_step += 1
                self.score += 10

    def _resolve_enemy_tiles(self, e: Enemy):
        r = e.get_rect()
        collided_horiz = False
        for (_r, _c, trect, _t) in self._tile_rects_near(e):
            if not r.colliderect(trect):
                continue
            ox = min(r.right - trect.left, trect.right - r.left)
            oy = min(r.bottom - trect.top, trect.bottom - r.top)
            if ox < oy:
                # Horizontal wall; place flush and flip
                collided_horiz = True
                e.x = (trect.left - e.width) if r.centerx < trect.centerx else trect.right
                e.vx = -e.vx
            else:
                # Vertical collision; land or bump head
                e.y = (trect.top - e.height) if r.centery < trect.centery else trect.bottom
                e.vy = 0.0
            r = e.get_rect()

        # Anti-ping-pong: after many horizontal hits, hop up or nudge out
        if collided_horiz:
            e.bounce_clock = min(e.bounce_clock + 1, 99)
        else:
            e.bounce_clock = max(e.bounce_clock - 1, 0)

        if e.bounce_clock >= 10:
            ahead = 1 if e.vx > 0 else -1
            tx = int((e.x + (ahead * e.width)) // TILE_SIZE) + (1 if ahead > 0 else 0)
            ty_bottom = int((e.y + e.height - 1) // TILE_SIZE)
            if 0 <= tx < LEVEL_COLS and 1 <= ty_bottom < LEVEL_ROWS:
                if self._is_solid(self.level_data[ty_bottom][tx]) and not self._is_solid(self.level_data[ty_bottom - 1][tx]):
                    # climb one tile
                    e.y -= TILE_SIZE
                    e.bounce_clock = 0
                else:
                    # nudge out of pocket
                    e.x += ahead * TILE_SIZE * 1.5
                    e.bounce_clock = 0

    def _resolve_powerup_tiles(self, pu: Powerup):
        r = pu.get_rect()
        for (_r, _c, trect, _t) in self._tile_rects_near(pu):
            if not r.colliderect(trect):
                continue
            ox = min(r.right - trect.left, trect.right - r.left)
            oy = min(r.bottom - trect.top, trect.bottom - r.top)
            if ox < oy:
                pu.x = (trect.left - pu.width) if r.centerx < trect.centerx else trect.right
                pu.vx = -pu.vx
            else:
                pu.y = (trect.top - pu.height) if r.centery < trect.centery else trect.bottom
                pu.vy = 0.0
            r = pu.get_rect()

    def _handle_object_collisions(self):
        p = self.player
        # Enemies
        for e in self.enemies:
            if not e.active:
                continue
            if p.collides_with(e):
                if p.vy > 0 and (p.y + p.height - 10) < e.y:
                    e.active = False
                    p.vy = JUMP_VEL_MIN * 0.6
                    self.score += 100
                    self.kills_step += 1
                elif p.invincible_timer > 0:
                    e.active = False
                    self.score += 100
                    self.kills_step += 1
                else:
                    if p.powered_up:
                        p.powered_up = False
                        p.invincible_timer = 60
                    else:
                        self.alive = False
                        self.game_over = True

        # Coins
        for c in self.coins:
            if c.active and not c.collected and p.collides_with(c):
                c.active = False
                c.collected = True
                self.coins_total += 1
                self.coins_step += 1
                self.score += 10

        # Powerups
        for pu in self.powerups:
            if pu.active and p.collides_with(pu):
                pu.active = False
                self.powerups_step += 1
                if pu.kind == "mushroom":
                    p.powered_up = True
                    self.score += 50
                else:
                    p.invincible_timer = 300
                    self.score += 100

    # ---------------------------------------------------------------------
    # Camera / reward / termination
    # ---------------------------------------------------------------------
    def _update_camera(self):
        if not self.camera_lock:
            return
        target = max(0, min(self.player.x - self.WIDTH // 3, self.level_w - self.WIDTH))
        self.camera_x += (target - self.camera_x) * self.camera_smoothing
        self.camera_x = max(0, min(self.camera_x, self.level_w - self.WIDTH))

    def _update_stall_metrics(self):
        """Track forward progress; penalize/terminate if camping."""
        if not self.anti_stall:
            return
        x = self.player.x
        # Progress resets timers
        if x > self.max_x_seen + self.stall_eps:
            self.max_x_seen = x
            self.stall_timer = 0
            self.stall_windows_count = 0
            self.stalled_this_frame = False
            return
        # Not beating best -> count time
        self.stall_timer += 1
        if self.stall_timer >= self.stall_window:
            self.stalled_this_frame = True
            self.stall_timer = 0
            self.stall_windows_count += 1

    def _check_termination(self) -> bool:
        p = self.player
        # Fell off world
        if p.y > self.level_h:
            self.alive = False; self.game_over = True; self.reached_goal = False
            return True
        # Goal / spike
        tile = self._tile_at(p.x + p.width / 2, p.y + p.height / 2)
        if tile == TILE_GOAL:
            self.score += 1000
            self.alive = False; self.reached_goal = True
            return True
        if tile == TILE_SPIKE:
            self.alive = False; self.game_over = True; self.reached_goal = False
            return True
        # Anti-stall truncate
        if self.anti_stall and self.stall_windows_count >= self.stall_kill_windows:
            self.alive = False; self.reached_goal = False
            return True
        # Max steps
        if self.max_steps and self.frame >= int(self.max_steps):
            self.alive = False; self.reached_goal = False
            return True
        return False

    def _reward(self) -> float:
        """Dense forward progress, small time pressure, shaping for collectibles and anti-stall."""
        r = 0.0
        # Forward progress
        dx = self.player.x - self.last_x
        r += dx / 8.0
        # Time pressure
        r -= 0.005
        # Collectibles / score deltas
        r += self.kills_step * 1.0 + self.coins_step * 0.5 + self.powerups_step * 1.0
        r += self.score_delta / 100.0
        # Anti-stall and anti-backtrack
        if self.anti_stall:
            if self.stalled_this_frame:
                r += self.stall_penalty  # applied once per counted stall window
            back = max(0.0, (self.max_x_seen - self.player.x))
            if back > 0:
                r -= self.backtrack_penalty * min(back, TILE_SIZE * 8)
        return r

    # ---------------------------------------------------------------------
    # Observation / info
    # ---------------------------------------------------------------------
    def _obs(self) -> np.ndarray:
        out: List[float] = []
        out.extend(self._player_obs())
        out.extend(self._tile_window_obs())
        out.extend(self._object_obs())
        return np.array(out, dtype=np.float32)

    def _player_obs(self) -> List[float]:
        p = self.player
        return [
            p.x / self.level_w,
            p.y / self.level_h,
            p.vx / max(1e-6, self.max_run),
            p.vy / MAX_FALL_SPEED,
            1.0 if p.on_ground else 0.0,
        ]

    def _tile_window_obs(self) -> List[float]:
        """11x9 window of tiles centered on player (plus coin/enemy masks)."""
        p = self.player
        px = int(p.x // TILE_SIZE); py = int(p.y // TILE_SIZE)
        tiles: List[float] = []; coins_map: List[float] = []; enemies_map: List[float] = []

        coin_cells = {(int(c.x // TILE_SIZE), int(c.y // TILE_SIZE))
                      for c in self.coins if c.active and not c.collected}
        enemy_cells = {(int(e.x // TILE_SIZE), int(e.y // TILE_SIZE))
                       for e in self.enemies if e.active}

        for dy in range(-4, 5):
            for dx in range(-5, 6):
                tx, ty = px + dx, py + dy
                if 0 <= ty < LEVEL_ROWS and 0 <= tx < LEVEL_COLS:
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
                    d = abs(p.x - o.x) + abs(p.y - o.y)
                    m = min(m, d)
            return m / 1000.0

        min_enemy = nearest([e for e in self.enemies if e.active])
        min_coin  = nearest([c for c in self.coins if c.active and not c.collected])
        return [
            min_enemy, min_coin,
            1.0 if p.powered_up else 0.0,
            p.invincible_timer / 300.0,
            len([e for e in self.enemies if e.active]) / 10.0,
            len([c for c in self.coins if c.active and not c.collected]) / 10.0,
            len([u for u in self.powerups if u.active]) / 5.0,
            self.coins_total / 10.0,
            self.score / 1000.0,
            self.frame / float(self.max_steps if self.max_steps else 1e9),
        ]

    def _info(self) -> Dict:
        p = self.player
        return {
            "score": self.score, "score_delta": self.score_delta, "frame_count": self.frame,
            "x_position": p.x, "y_position": p.y, "velocity_x": p.vx,
            "coins_collected": self.coins_total, "enemies_killed": self.kills_step,
            "powered_up": p.powered_up, "terminated": not self.alive,
            "won": (self.reached_goal and not self.game_over),
            "action": self._last_action,
            # Anti-stall telemetry
            "max_x_seen": self.max_x_seen, "stall_windows": self.stall_windows_count,
            "stalled": self.stalled_this_frame,
        }

    def _tile_at(self, x: float, y: float) -> int:
        c = int(x // TILE_SIZE); r = int(y // TILE_SIZE)
        if 0 <= r < LEVEL_ROWS and 0 <= c < LEVEL_COLS:
            return self.level_data[r][c]
        return TILE_AIR

    # ---------------------------------------------------------------------
    # Rendering
    # ---------------------------------------------------------------------
    def render(self, surface: pygame.Surface, blit_only: bool = True):
        surface.fill(COLOR_SKY)
        self._draw_tiles(surface)
        self._draw_coins(surface)
        self._draw_powerups(surface)
        self._draw_enemies(surface)
        self._draw_player(surface)
        self._update_debug_key_toggles()
        if self.db_hitboxes or self.db_sensors or self.db_agentview or self.db_obs_panel or self.db_tile_grid:
            self._draw_debug(surface)
        self._draw_ui(surface)

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

    def _world_to_screen(self, x: float) -> float:
        return x - self.camera_x

    def _draw_tiles(self, surface: pygame.Surface):
        tx0 = max(0, int(self.camera_x // TILE_SIZE))
        tx1 = min(LEVEL_COLS, int((self.camera_x + self.WIDTH) // TILE_SIZE) + 1)
        for r in range(LEVEL_ROWS):
            for c in range(tx0, tx1):
                t = self.level_data[r][c]
                if t == TILE_AIR:
                    continue
                sx = c * TILE_SIZE - self.camera_x
                sy = r * TILE_SIZE
                color = {
                    TILE_GROUND: COLOR_GROUND,
                    TILE_PLATFORM: COLOR_PLATFORM,
                    TILE_GOAL: COLOR_GOAL,
                    TILE_SPIKE: COLOR_SPIKE,
                    TILE_QBLOCK: COLOR_QBLOCK,
                }.get(t, COLOR_GROUND)
                # Question block draw (show '?' if not hit)
                if t == TILE_QBLOCK:
                    hit = False
                    for qb in self.qblocks:
                        qc, qr = qb.tc()
                        if qc == c and qr == r:
                            hit = qb.hit
                            break
                    if hit:
                        color = COLOR_EMPTY
                pygame.draw.rect(surface, color, (sx, sy, TILE_SIZE, TILE_SIZE))
                pygame.draw.rect(surface, COLOR_BLACK, (sx, sy, TILE_SIZE, TILE_SIZE), 1)
                if t == TILE_QBLOCK:
                    hit = False
                    for qb in self.qblocks:
                        qc, qr = qb.tc()
                        if qc == c and qr == r:
                            hit = qb.hit
                            break
                    if not hit:
                        font = pygame.font.Font(None, 26)
                        q = font.render("?", True, COLOR_WHITE)
                        surface.blit(q, q.get_rect(center=(sx + TILE_SIZE // 2, sy + TILE_SIZE // 2)))

        if self.db_tile_grid:
            for r in range(LEVEL_ROWS):
                y = r * TILE_SIZE
                pygame.draw.line(surface, (0, 0, 0), (0 - self.camera_x, y), (self.level_w - self.camera_x, y), 1)
            for c in range(LEVEL_COLS):
                x = c * TILE_SIZE - self.camera_x
                pygame.draw.line(surface, (0, 0, 0), (x, 0), (x, self.level_h), 1)

    def _draw_enemies(self, surface: pygame.Surface):
        for e in self.enemies:
            if e.active:
                sx, sy = self._world_to_screen(e.x), e.y
                pygame.draw.rect(surface, COLOR_ENEMY, (sx, sy, e.width, e.height))

    def _draw_coins(self, surface: pygame.Surface):
        for c in self.coins:
            if c.active and not c.collected:
                sx, sy = self._world_to_screen(c.x), c.y
                pygame.draw.circle(surface, COLOR_COIN, (int(sx), int(sy)), 8)
                pygame.draw.circle(surface, COLOR_BLACK, (int(sx), int(sy)), 8, 2)

    def _draw_powerups(self, surface: pygame.Surface):
        for pu in self.powerups:
            if pu.active:
                sx, sy = self._world_to_screen(pu.x), pu.y
                col = COLOR_POWERUP_MUSH if pu.kind == "mushroom" else COLOR_POWERUP_STAR
                pygame.draw.rect(surface, col, (sx, sy, pu.width, pu.height))

    def _draw_player(self, surface: pygame.Surface):
        p = self.player
        sx, sy = self._world_to_screen(p.x), p.y
        col = COLOR_POWERUP_STAR if (p.invincible_timer > 0 and (self.frame // 5) % 2) else \
              ((255, 100, 0) if p.powered_up else (255, 0, 0))
        pygame.draw.rect(surface, col, (sx, sy, p.width, p.height))
        pygame.draw.circle(surface, COLOR_WHITE, (int(sx + (14 if p.facing_right else 6)), int(sy + 8)), 3)

        # Acceleration streaks while RUN held and fast
        if p.run_pressed and abs(p.vx) > self.max_walk * 0.6:
            n = 3; spacing = 6; length = 10
            for i in range(n):
                offset = (i + 1) * spacing
                if p.facing_right:
                    x1 = sx - offset; x2 = x1 - length
                else:
                    x1 = sx + p.width + offset; x2 = x1 + length
                y = sy + 10 + (i % 2) * 4
                pygame.draw.line(surface, COLOR_STREAK, (int(x1), int(y)), (int(x2), int(y)), 2)

    def _draw_debug(self, surface: pygame.Surface):
        p = self.player
        px, py = self._world_to_screen(p.x), p.y
        font = pygame.font.Font(None, 20)

        if self.db_hitboxes:
            pygame.draw.rect(surface, COLOR_HITBOX, (px, py, p.width, p.height), 2)
            for e in self.enemies:
                if e.active:
                    ex, ey = self._world_to_screen(e.x), e.y
                    pygame.draw.rect(surface, (255, 128, 0), (ex, ey, e.width, e.height), 1)
            for pu in self.powerups:
                if pu.active:
                    ex, ey = self._world_to_screen(pu.x), pu.y
                    pygame.draw.rect(surface, (255, 255, 0), (ex, ey, pu.width, pu.height), 1)

        if self.db_sensors:
            rays = [((px + p.width // 2, py + p.height), (px + p.width // 2, py + p.height + 10)),
                    ((px + p.width // 2, py), (px + p.width // 2, py - 10)),
                    ((px, py + p.height // 2), (px - 10, py + p.height // 2)),
                    ((px + p.width, py + p.height // 2), (px + p.width + 10, py + p.height // 2))]
            for a, b in rays:
                pygame.draw.line(surface, COLOR_SENSOR, a, b, 2)
            v_end = (int(px + p.vx * 5), int(py + p.vy * 2))
            pygame.draw.line(surface, (100, 255, 255), (int(px + p.width / 2), int(py + p.height / 2)), v_end, 2)

        if self.db_agentview:
            panel_w, panel_h, cell = (11 * 12 + 4), (9 * 12 + 4), 12
            panel = pygame.Surface((panel_w, panel_h)); panel.fill(COLOR_AGENT_PANEL)
            px_t = int(p.x // TILE_SIZE); py_t = int(p.y // TILE_SIZE)
            coin_cells = {(int(c.x // TILE_SIZE), int(c.y // TILE_SIZE))
                          for c in self.coins if c.active and not c.collected}
            enemy_cells = {(int(e.x // TILE_SIZE), int(e.y // TILE_SIZE))
                           for e in self.enemies if e.active}
            for j, dy in enumerate(range(-4, 5)):
                for i, dx in enumerate(range(-5, 6)):
                    tx, ty = px_t + dx, py_t + dy
                    v = 0
                    if 0 <= ty < LEVEL_ROWS and 0 <= tx < LEVEL_COLS:
                        v = self.level_data[ty][tx]
                    cx, cy = 2 + i * cell, 2 + j * cell
                    color = (60, 60, 60) if v == 0 else \
                            (165, 115, 50) if v == TILE_QBLOCK else \
                            (200, 170, 80) if v == TILE_GOAL else \
                            (80, 80, 80)  if v == TILE_PLATFORM else \
                            (120, 90, 50)
                    pygame.draw.rect(panel, color, (cx, cy, cell - 1, cell - 1))
                    if (tx, ty) in coin_cells:
                        pygame.draw.circle(panel, COLOR_COIN, (cx + (cell // 2), cy + (cell // 2)), 3)
                    if (tx, ty) in enemy_cells:
                        pygame.draw.rect(panel, (200, 40, 40), (cx + 2, cy + 2, cell - 5, cell - 5), 2)
            surface.blit(panel, (10, 70))
            surface.blit(font.render("Agent 11x9 view  (● coin, ☐ enemy)", True, COLOR_WHITE), (12, 52))

        if self.db_obs_panel:
            obs = self._obs()
            lines = [
                f"obs[0:5]=[{obs[0]:.3f},{obs[1]:.3f},{obs[2]:.3f},{obs[3]:.3f},{obs[4]:.1f}]",
                f"max_x={int(self.max_x_seen)} stalled={self.stalled_this_frame} stallW={self.stall_windows_count}",
            ]
            box = pygame.Surface((360, 14 * len(lines) + 10), pygame.SRCALPHA)
            box.fill((0, 0, 0, 160))
            surface.blit(box, (10, 10))
            for i, ln in enumerate(lines):
                surface.blit(font.render(ln, True, COLOR_WHITE), (16, 16 + i * 14))

    def _draw_ui(self, surface: pygame.Surface):
        p = self.player
        font = pygame.font.Font(None, 24)
        status = "STAR" if p.invincible_timer > 0 else ("SUPER" if p.powered_up else "SMALL")
        ts = font.render(
            f"Step:{self.frame}  X:{int(p.x)}  Vx:{abs(p.vx):.2f}  Score:{self.score}  Coins:{self.coins_total}  {status}",
            True, COLOR_WHITE
        )
        bg = pygame.Surface((ts.get_width() + 10, ts.get_height() + 6), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 170))
        surface.blit(bg, (5, 5))
        surface.blit(ts, (10, 7))
