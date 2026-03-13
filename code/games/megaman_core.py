from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import List, Tuple

import gymnasium
from gymnasium import spaces
import numpy as np
import pygame

from .modules.Objects.GameObject import GameObject
from .modules.System.EntityType import EntityType
from .modules.System.LevelLoader import LevelLoader
from .modules.System.config_manager import ConfigManager
from .modules.System.debugging_mods.manager import DebugManager
from .modules.Parameters.Map_parameters import (
    TILE_AIR,
    TILE_GROUND,
    TILE_PLATFORM,
    TILE_GOAL,
    TILE_SPIKE,
    TILE_QBLOCK,
)


DEBUG_PANEL_WIDTH = 350
MEGAMAN_BG = (120, 160, 255)
MEGAMAN_SKY = (150, 190, 255)
MEGAMAN_GRID = (170, 205, 255)
METAL_LIGHT = (226, 234, 244)
METAL_MID = (170, 184, 204)
METAL_DARK = (104, 118, 146)
HAZARD_RED = (230, 76, 76)
GOAL_GREEN = (80, 214, 105)
MEGAMAN_BLUE = (40, 120, 255)
MEGAMAN_CYAN = (160, 236, 255)
MEGAMAN_DARK = (8, 40, 112)
MET_YELLOW = (255, 196, 72)
BAT_PURPLE = (148, 96, 204)
SHOT_ORANGE = (255, 182, 92)


@dataclass
class MegaManPlayer:
    gObj: GameObject
    vx: float = 0.0
    vy: float = 0.0
    on_ground: bool = False
    facing_right: bool = True
    hp: int = 28
    hp_max: int = 28
    i_frames: float = 0.0
    invincible_timer: float = 0.0
    star_timer: float = 0.0
    shot_cooldown: float = 0.0
    jump_cut: bool = False
    powered_up: bool = False
    fire_requested: bool = False
    on_ladder: bool = False
    ladder_x: float = 0.0


class _QueryBucket:
    def __init__(self, provider):
        self._provider = provider

    def query_rect(self, x, y, w, h):
        rect = pygame.Rect(int(x), int(y), int(w), int(h))
        out = []
        for obj in self._provider():
            gobj = obj.gObj if hasattr(obj, "gObj") else obj
            if getattr(gobj, "active", True) and rect.colliderect(gobj.get_rect()):
                out.append(obj)
        return out

    def query(self, obj):
        gobj = obj.gObj if hasattr(obj, "gObj") else obj
        return self.query_rect(gobj.x, gobj.y, gobj.width, gobj.height)


class _DebugPhysics:
    def __init__(self, core: "MegamanCore"):
        self.gravity = 980.0
        self.jump_force = 430.0
        self.hazard_hash = _QueryBucket(lambda: core.enemies)
        self.collectible_hash = _QueryBucket(
            lambda: list(getattr(core.level_data, "coins", [])) + list(getattr(core.level_data, "goals", []))
        )


class MegaBusterProjectile:
    WIDTH = 14
    HEIGHT = 8
    PLAYER_SPEED = 620.0
    ENEMY_SPEED = 300.0
    PLAYER_RANGE = 2.2
    ENEMY_RANGE = 1.8

    def __init__(self, x: float, y: float, vx: float, owner: str, damage: int, lifetime: float):
        self.gObj = GameObject(x, y, self.WIDTH, self.HEIGHT, True)
        self.gObj.type_id = EntityType.PROJECTILE
        self.vx = vx
        self.owner = owner
        self.damage = damage
        self.lifetime = lifetime
        self.age = 0.0

    @classmethod
    def from_player(cls, player: MegaManPlayer) -> "MegaBusterProjectile":
        x = player.gObj.x + (player.gObj.width - 4 if player.facing_right else -cls.WIDTH + 4)
        y = player.gObj.y + player.gObj.height * 0.45
        vx = cls.PLAYER_SPEED if player.facing_right else -cls.PLAYER_SPEED
        return cls(x, y, vx, "player", damage=1, lifetime=cls.PLAYER_RANGE)

    @classmethod
    def from_enemy(cls, x: float, y: float, facing_right: bool) -> "MegaBusterProjectile":
        vx = cls.ENEMY_SPEED if facing_right else -cls.ENEMY_SPEED
        return cls(x, y, vx, "enemy", damage=4, lifetime=cls.ENEMY_RANGE)

    def update(self, dt: float):
        self.age += dt
        if self.age >= self.lifetime:
            self.gObj.active = False
            return
        self.gObj.x += self.vx * dt

    def render(self, surface: pygame.Surface, sx: float, sy: float):
        if not self.gObj.active:
            return
        outer = MEGAMAN_CYAN if self.owner == "player" else SHOT_ORANGE
        pygame.draw.rect(surface, outer, (sx, sy, self.gObj.width, self.gObj.height), border_radius=4)
        pygame.draw.rect(surface, (255, 255, 255), (sx + 3, sy + 2, self.gObj.width - 6, self.gObj.height - 4), border_radius=3)


class BaseEnemy:
    def __init__(self, x: float, y: float, width: int, height: int, hp: int = 2):
        self.gObj = GameObject(x, y, width, height, True)
        self.gObj.type_id = EntityType.ENEMY
        self.vx = 0.0
        self.vy = 0.0
        self.on_ground = False
        self.hp = hp
        self.contact_damage = 4
        self.facing_right = True

    @property
    def x(self):
        return self.gObj.x

    @property
    def y(self):
        return self.gObj.y

    def take_damage(self, damage: int) -> bool:
        self.hp -= damage
        if self.hp <= 0:
            self.gObj.active = False
            return True
        return False


class MetEnemy(BaseEnemy):
    def __init__(self, x: float, y: float):
        super().__init__(x, y, 28, 24, hp=2)
        self.vx = 72.0
        self.shot_timer = 0.8

    def update(self, dt: float, core: "MegamanCore"):
        self.facing_right = self.vx >= 0
        self.shot_timer -= dt
        if self.shot_timer <= 0.0:
            shot_x = self.gObj.x + (self.gObj.width if self.facing_right else -MegaBusterProjectile.WIDTH)
            shot_y = self.gObj.y + self.gObj.height * 0.42
            core.projectiles.append(MegaBusterProjectile.from_enemy(shot_x, shot_y, self.facing_right))
            self.shot_timer = 1.3

    def render(self, surface: pygame.Surface, sx: float, sy: float):
        pygame.draw.rect(surface, MET_YELLOW, (sx, sy, self.gObj.width, 12), border_radius=6)
        pygame.draw.rect(surface, HAZARD_RED, (sx + 3, sy + 9, self.gObj.width - 6, self.gObj.height - 9), border_radius=5)
        eye_x = sx + (self.gObj.width - 10 if self.facing_right else 4)
        pygame.draw.rect(surface, (255, 255, 255), (eye_x, sy + 5, 5, 4), border_radius=2)


class BatEnemy(BaseEnemy):
    def __init__(self, x: float, y: float):
        super().__init__(x, y, 26, 20, hp=1)
        self.anchor_y = y
        self.vx = 96.0
        self.phase = 0.0

    def update(self, dt: float, core: "MegamanCore"):
        self.phase += dt * 5.0
        self.facing_right = self.vx >= 0
        self.gObj.y = self.anchor_y + np.sin(self.phase) * 10.0

    def render(self, surface: pygame.Surface, sx: float, sy: float):
        pygame.draw.polygon(surface, BAT_PURPLE, [(sx + 4, sy + 10), (sx - 7, sy + 3), (sx + 4, sy + 17)])
        pygame.draw.polygon(surface, BAT_PURPLE, [(sx + 22, sy + 10), (sx + 33, sy + 3), (sx + 22, sy + 17)])
        pygame.draw.rect(surface, BAT_PURPLE, (sx + 5, sy + 4, 16, 12), border_radius=6)
        eye_x = sx + (14 if self.facing_right else 9)
        pygame.draw.rect(surface, (255, 255, 255), (eye_x, sy + 8, 4, 3), border_radius=1)


class MegamanCore(gymnasium.Env):
    metadata = {"render_modes": ["none", "human", "rgb_array"]}

    def __init__(self, render_mode: str = "none", **kwargs):
        self.render_mode = render_mode
        self.config_manager = ConfigManager("game_config.yaml")
        self.raw_config = self.config_manager.yaml_data.get("megaman", {})

        self.WIDTH = int(self.raw_config.get("screen_width", 960))
        self.HEIGHT = int(self.raw_config.get("screen_height", 600))
        self.FPS = int(self.raw_config.get("fps", 60))
        self.TILE_SIZE = int(self.raw_config.get("tile_size", 32))
        self.TOTAL_WIDTH = self.WIDTH + DEBUG_PANEL_WIDTH
        self.DEBUG_PANEL_X = self.WIDTH

        self.action_space = spaces.MultiDiscrete([5, 2, 2])
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(40,), dtype=np.float32)

        self.loader = LevelLoader(tile_size=self.TILE_SIZE)
        pygame.init()
        self.debug_manager = DebugManager(default_active=(render_mode == "human"), print_help=(render_mode == "human"))
        self.debug_manager.show_grid = False

        if self.render_mode == "human":
            pygame.display.set_caption("PEAK Mega Man")
            self._surf = pygame.display.set_mode((self.TOTAL_WIDTH, self.HEIGHT))
        else:
            self._surf = pygame.Surface((self.TOTAL_WIDTH, self.HEIGHT))

        self.ui_font = pygame.font.SysFont("segoeui", 20, bold=True)
        self.small_font = pygame.font.SysFont("consolas", 16)

        self.player: MegaManPlayer | None = None
        self.level_data = None
        self.enemies: List[BaseEnemy] = []
        self.projectiles: List[MegaBusterProjectile] = []
        self.source_lines: List[str] = []
        self.physics_manager = _DebugPhysics(self)

        self.world = kwargs.get("world", "MM-Stage1")
        self.camera_x = 0.0
        self.camera_y = 0.0
        self.steps = 0
        self.max_steps = int(self.raw_config.get("episode", {}).get("max_steps", 3000))
        self.time_limit = 400.0
        self.lives = 3
        self.score = 0
        self.coins_total = 0
        self.reached_goal = False
        self.alive = True
        self.dt = 1.0 / float(self.FPS)
        self.last_action = [0, 0, 0]
        self.frame = 0
        self.timer = self.time_limit
        self.persona = "megaman"
        self.progress_x_best = 0.0
        self.stall_timer = 0.0
        self.stall_windows_count = 0
        self.obs_width = 21
        self.obs_height = 21
        self.obs_pad_x = self.obs_width // 2
        self.obs_pad_y = self.obs_height // 2
        self._visit_map = None
        self._dijkstra_window_cache = None
        self._hazard_window_cache = None
        self._step_dx = 0.0
        self._step_dy = 0.0

    def _level_config(self) -> dict:
        levels = self.raw_config.get("levels", {})
        return levels.get(self.world, {"file": "mm_stage1.txt"})

    def _level_file_path(self) -> str:
        level_file = self._level_config().get("file", "mm_stage1.txt")
        return level_file if os.path.isabs(level_file) else os.path.join(self.loader.level_path, level_file)

    def _load_source_lines(self) -> List[str]:
        path = self._level_file_path()
        if not os.path.isfile(path):
            return []
        with open(path, "r", encoding="utf-8") as fh:
            return [line.rstrip("\n") for line in fh]

    def _spawn_enemies(self):
        self.enemies = []
        for row, line in enumerate(self.source_lines):
            for col, ch in enumerate(line):
                x = float(col * self.TILE_SIZE)
                y = float(row * self.TILE_SIZE)
                if ch == "M":
                    self.enemies.append(MetEnemy(x, y))
                elif ch == "B":
                    self.enemies.append(BatEnemy(x, y))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.score = 0
        self.reached_goal = False
        self.alive = True
        self.time_limit = 400.0
        self.timer = self.time_limit
        self.camera_x = 0.0
        self.camera_y = 0.0
        self.last_action = [0, 0, 0]
        self.frame = 0
        self.progress_x_best = 0.0
        self.stall_timer = 0.0
        self.stall_windows_count = 0

        self.source_lines = self._load_source_lines()
        self.level_data = self.loader.load_level(self._level_config().get("file", "mm_stage1.txt"))
        px, py = self.level_data.player_start
        self.player = MegaManPlayer(GameObject(px, py, 20, 28, True))
        self.projectiles = []
        self._spawn_enemies()
        self.level_data.enemies = self.enemies
        self._visit_map = np.zeros((self.level_data.rows, self.level_data.cols), dtype=np.int32)
        self._update_debug_caches()

        return self._obs(), self._info()

    def _parse_action(self, action) -> Tuple[int, int, int]:
        try:
            move = int(action[0])
            jump = int(action[1])
            fire = int(action[2])
        except Exception:
            move = jump = fire = 0
        return move, jump, fire

    def _keyboard_state(self):
        if not pygame.get_init():
            return False, False, False, False, False, False
        keys = pygame.key.get_pressed()
        left = bool(keys[pygame.K_a] or keys[pygame.K_LEFT])
        right = bool(keys[pygame.K_d] or keys[pygame.K_RIGHT])
        up = bool(keys[pygame.K_w] or keys[pygame.K_UP])
        down = bool(keys[pygame.K_s] or keys[pygame.K_DOWN])
        jump = bool(keys[pygame.K_SPACE])
        fire = bool(keys[pygame.K_z])
        return left, right, up, down, jump, fire

    def _nearest_ladder(self, gObj: GameObject):
        if not self.level_data:
            return None
        probe = gObj.get_rect().inflate(-6, -2)
        ladders = []
        for ladder in getattr(self.level_data, "ladders", []):
            if probe.colliderect(ladder.gObj.get_rect().inflate(10, 8)):
                ladders.append(ladder)
        if not ladders:
            return None
        return min(
            ladders,
            key=lambda ladder: abs(
                (ladder.gObj.x + ladder.gObj.width * 0.5) - (gObj.x + gObj.width * 0.5)
            ),
        )

    def _handle_input(self, action):
        if not self.player:
            return
        move, jump, fire = self._parse_action(action)
        p = self.player
        kb_left, kb_right, kb_up, kb_down, kb_jump, kb_fire = self._keyboard_state()

        walk_speed = 190.0
        run_speed = 255.0
        accel = 1800.0
        friction = 2000.0
        jump_vel = -430.0
        climb_speed = 150.0

        if kb_left:
            move = 1
        elif kb_right:
            move = 3
        if kb_jump:
            jump = 1
        if kb_fire:
            fire = 1

        ladder = self._nearest_ladder(p.gObj)
        climb_up = kb_up
        climb_down = kb_down

        if ladder and (climb_up or climb_down):
            p.on_ladder = True
            p.ladder_x = ladder.gObj.x + ladder.gObj.width * 0.5 - p.gObj.width * 0.5
        elif p.on_ladder and ladder is None:
            p.on_ladder = False

        if p.on_ladder:
            p.gObj.x = p.ladder_x
            p.vx = 0.0
            if climb_up:
                p.vy = -climb_speed
            elif climb_down:
                p.vy = climb_speed
            else:
                p.vy = 0.0

            if jump:
                p.on_ladder = False
                p.vy = jump_vel
            elif move in (1, 2, 3, 4) and not (climb_up or climb_down):
                p.on_ladder = False
            else:
                return

        target_vx = 0.0
        if move == 1:
            target_vx = -walk_speed
            p.facing_right = False
        elif move == 2:
            target_vx = -run_speed
            p.facing_right = False
        elif move == 3:
            target_vx = walk_speed
            p.facing_right = True
        elif move == 4:
            target_vx = run_speed
            p.facing_right = True

        if target_vx != 0.0:
            if p.vx < target_vx:
                p.vx = min(target_vx, p.vx + accel * self.dt)
            elif p.vx > target_vx:
                p.vx = max(target_vx, p.vx - accel * self.dt)
        else:
            if p.vx > 0:
                p.vx = max(0.0, p.vx - friction * self.dt)
            elif p.vx < 0:
                p.vx = min(0.0, p.vx + friction * self.dt)

        if jump and p.on_ground:
            p.vy = jump_vel
            p.on_ground = False
            p.jump_cut = False
        elif not jump and p.vy < 0 and not p.jump_cut:
            p.vy *= 0.55
            p.jump_cut = True

        if fire and p.shot_cooldown <= 0.0:
            active_player_shots = sum(
                1 for proj in self.projectiles if proj.gObj.active and proj.owner == "player"
            )
            if active_player_shots < 3:
                self.projectiles.append(MegaBusterProjectile.from_player(p))
                p.shot_cooldown = 0.16

    def _nearby_tile_rects(self, gObj: GameObject):
        tx = int(gObj.x // self.TILE_SIZE)
        ty = int(gObj.y // self.TILE_SIZE)
        rects = []
        for row in range(ty - 2, ty + 4):
            for col in range(tx - 2, tx + 4):
                if 0 <= row < self.level_data.rows and 0 <= col < self.level_data.cols:
                    tile_type = self.level_data.grid[row][col]
                    if tile_type in (TILE_GROUND, TILE_PLATFORM, TILE_QBLOCK, TILE_SPIKE):
                        rect = pygame.Rect(col * self.TILE_SIZE, row * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
                        rects.append((rect, tile_type))
        return rects

    def _resolve_actor_y(self, actor, damage_spikes: bool = False):
        rect = actor.gObj.get_rect()
        actor.on_ground = False
        for tile_rect, tile_type in self._nearby_tile_rects(actor.gObj):
            if not rect.colliderect(tile_rect):
                continue
            if tile_type == TILE_SPIKE and damage_spikes and actor is self.player:
                self._damage_player(actor.hp_max, "Spike")
                continue
            if tile_type == TILE_PLATFORM:
                if actor.vy < 0:
                    continue
                if rect.bottom - actor.vy * self.dt > tile_rect.top + 4:
                    continue
            if actor.vy >= 0:
                actor.gObj.y = tile_rect.top - actor.gObj.height
                actor.vy = 0.0
                actor.on_ground = True
            else:
                actor.gObj.y = tile_rect.bottom
                actor.vy = 0.0
            rect = actor.gObj.get_rect()

    def _resolve_actor_x(self, actor, bounce: bool = False):
        rect = actor.gObj.get_rect()
        for tile_rect, tile_type in self._nearby_tile_rects(actor.gObj):
            if tile_type == TILE_SPIKE:
                continue
            if not rect.colliderect(tile_rect):
                continue
            if actor.vx > 0:
                actor.gObj.x = tile_rect.left - actor.gObj.width
                if bounce:
                    actor.vx = -abs(actor.vx)
                    actor.facing_right = False
                else:
                    actor.vx = 0.0
            elif actor.vx < 0:
                actor.gObj.x = tile_rect.right
                if bounce:
                    actor.vx = abs(actor.vx)
                    actor.facing_right = True
                else:
                    actor.vx = 0.0
            rect = actor.gObj.get_rect()

    def _damage_player(self, amount: int, cause: str):
        if not self.player or self.player.i_frames > 0.0:
            return
        self.player.hp = max(0, self.player.hp - amount)
        self.player.i_frames = 1.0
        self.player.invincible_timer = self.player.i_frames
        if self.player.hp <= 0:
            self.lives = max(0, self.lives - 1)
            self.alive = False

    def _update_debug_caches(self):
        if not self.player or not self.level_data:
            return

        p = self.player
        px = int(p.gObj.x // self.TILE_SIZE)
        py = int(p.gObj.y // self.TILE_SIZE)

        if self._visit_map is not None and 0 <= py < self.level_data.rows and 0 <= px < self.level_data.cols:
            self._visit_map[py, px] += 1

        goal_tiles = [
            (int(g.gObj.x // self.TILE_SIZE), int(g.gObj.y // self.TILE_SIZE))
            for g in self.level_data.goals
        ]
        player_goal_dist = min((abs(gx - px) + abs(gy - py) for gx, gy in goal_tiles), default=0)
        max_cost = max(1.0, self.obs_width + self.obs_height)

        dijkstra = np.zeros((self.obs_height, self.obs_width), dtype=np.float32)
        hazards = np.zeros((self.obs_height, self.obs_width), dtype=np.float32)

        best_cost = player_goal_dist
        best_step = (0, 0)
        for ddy in range(-self.obs_pad_y, self.obs_pad_y + 1):
            for ddx in range(-self.obs_pad_x, self.obs_pad_x + 1):
                tx = px + ddx
                ty = py + ddy
                ly = ddy + self.obs_pad_y
                lx = ddx + self.obs_pad_x

                if not (0 <= tx < self.level_data.cols and 0 <= ty < self.level_data.rows):
                    continue

                tile = self.level_data.grid[ty][tx]
                if tile == TILE_SPIKE:
                    hazards[ly, lx] = -1.0
                elif any(int(e.gObj.x // self.TILE_SIZE) == tx and int(e.gObj.y // self.TILE_SIZE) == ty for e in self.enemies if e.gObj.active):
                    hazards[ly, lx] = 1.0

                tile_dist = min((abs(gx - tx) + abs(gy - ty) for gx, gy in goal_tiles), default=player_goal_dist)
                dijkstra[ly, lx] = np.clip((player_goal_dist - tile_dist) / max_cost, -1.0, 1.0)

                if abs(ddx) + abs(ddy) == 1 and tile_dist < best_cost:
                    best_cost = tile_dist
                    best_step = (ddx, ddy)

        self._dijkstra_window_cache = dijkstra
        self._hazard_window_cache = hazards
        mag = max(1.0, float(np.hypot(best_step[0], best_step[1])))
        self._step_dx = best_step[0] / mag
        self._step_dy = best_step[1] / mag

    def _player_obs(self) -> np.ndarray:
        if not self.player:
            return np.zeros(13, dtype=np.float32)
        p = self.player
        return np.array([
            np.clip(p.gObj.x / max(1.0, self.level_data.width), 0.0, 1.0),
            np.clip(p.gObj.y / max(1.0, self.level_data.height), 0.0, 1.0),
            np.clip(p.vx / 320.0, -1.0, 1.0),
            np.clip(p.vy / 700.0, -1.0, 1.0),
            1.0 if p.on_ground else 0.0,
            0.0,
            1.0,
            1.0 if p.i_frames > 0 else 0.0,
            1.0 if p.facing_right else 0.0,
            np.clip(p.shot_cooldown / 0.16, 0.0, 1.0),
            np.clip(p.i_frames, 0.0, 1.0),
            1.0 if p.on_ladder else 0.0,
            1.0 if p.vy < 0 else 0.0,
        ], dtype=np.float32)

    def _tracking_obs(self) -> np.ndarray:
        if not self.player:
            return np.zeros(7, dtype=np.float32)

        p = self.player
        enemy_dist = 1.0
        if self.enemies:
            enemy_dist = min(
                np.hypot(e.gObj.x - p.gObj.x, e.gObj.y - p.gObj.y) for e in self.enemies if e.gObj.active
            ) / max(1.0, self.level_data.width)

        goal_dx = self.level_data.width - p.gObj.x
        goal_dy = 0.0
        if self.level_data.goals:
            goal = min(self.level_data.goals, key=lambda g: abs(g.gObj.x - p.gObj.x))
            goal_dx = goal.gObj.x - p.gObj.x
            goal_dy = goal.gObj.y - p.gObj.y

        return np.array([
            np.clip(enemy_dist, 0.0, 1.0),
            np.clip(abs(goal_dx) / max(1.0, self.level_data.width), 0.0, 1.0),
            np.clip(self.timer / 400.0, 0.0, 1.0),
            np.clip(goal_dy / max(1.0, self.level_data.height), -1.0, 1.0),
            np.clip(abs(goal_dx) / max(1.0, self.level_data.width), 0.0, 1.0),
            self._step_dx,
            self._step_dy,
        ], dtype=np.float32)

    def _world_to_screen(self, gObj: GameObject):
        sx = gObj.x - self.camera_x
        sy = gObj.y - self.camera_y
        on_screen = sx < self.WIDTH and sy < self.HEIGHT and sx + gObj.width > 0 and sy + gObj.height > 0
        return sx, sy, on_screen

    def _update_player(self):
        p = self.player
        if not p:
            return
        p.shot_cooldown = max(0.0, p.shot_cooldown - self.dt)
        p.i_frames = max(0.0, p.i_frames - self.dt)
        p.invincible_timer = p.i_frames
        p.grounded = p.on_ground
        p.can_jump = p.on_ground or p.on_ladder

        if p.on_ladder:
            p.gObj.x = p.ladder_x
            p.gObj.y += p.vy * self.dt
            ladder = self._nearest_ladder(p.gObj)
            if ladder is not None:
                p.ladder_x = ladder.gObj.x + ladder.gObj.width * 0.5 - p.gObj.width * 0.5
                p.gObj.x = p.ladder_x
            else:
                p.on_ladder = False
            self._resolve_actor_y(p, damage_spikes=True)
            self.progress_x_best = max(self.progress_x_best, p.gObj.x)
            return

        gravity = 980.0
        max_fall = 620.0
        p.vy = min(p.vy + gravity * self.dt, max_fall)
        p.gObj.y += p.vy * self.dt
        self._resolve_actor_y(p, damage_spikes=True)
        p.gObj.x += p.vx * self.dt
        self._resolve_actor_x(p)
        self.progress_x_best = max(self.progress_x_best, p.gObj.x)

    def _update_enemies(self):
        gravity = 1200.0
        max_fall = 660.0
        for enemy in self.enemies:
            if not enemy.gObj.active:
                continue
            enemy.update(self.dt, self)
            if isinstance(enemy, MetEnemy):
                enemy.vy = min(enemy.vy + gravity * self.dt, max_fall)
                enemy.gObj.y += enemy.vy * self.dt
                self._resolve_actor_y(enemy)
                enemy.gObj.x += enemy.vx * self.dt
                self._resolve_actor_x(enemy, bounce=True)
            else:
                self._resolve_actor_x(enemy, bounce=True)
            if enemy.gObj.y > self.level_data.height + 120:
                enemy.gObj.active = False

    def _update_projectiles(self):
        for proj in self.projectiles:
            if not proj.gObj.active:
                continue
            proj.update(self.dt)
            if not proj.gObj.active:
                continue
            for tile_rect, tile_type in self._nearby_tile_rects(proj.gObj):
                if tile_type == TILE_AIR:
                    continue
                if proj.gObj.get_rect().colliderect(tile_rect):
                    proj.gObj.active = False
                    break

    def _handle_combat(self):
        if not self.player:
            return

        player_rect = self.player.gObj.get_rect()
        for enemy in self.enemies:
            if enemy.gObj.active and player_rect.colliderect(enemy.gObj.get_rect()):
                self._damage_player(enemy.contact_damage, "Enemy")

        for proj in self.projectiles:
            if not proj.gObj.active:
                continue
            if proj.owner == "enemy":
                if player_rect.colliderect(proj.gObj.get_rect()):
                    proj.gObj.active = False
                    self._damage_player(proj.damage, "Projectile")
                continue

            for enemy in self.enemies:
                if enemy.gObj.active and proj.gObj.get_rect().colliderect(enemy.gObj.get_rect()):
                    proj.gObj.active = False
                    if enemy.take_damage(proj.damage):
                        self.score += 100
                    break

        self.projectiles = [proj for proj in self.projectiles if proj.gObj.active]
        self.enemies = [enemy for enemy in self.enemies if enemy.gObj.active]
        self.level_data.enemies = self.enemies

    def _check_goal_and_oob(self):
        if not self.player:
            return
        player_rect = self.player.gObj.get_rect()
        for goal in self.level_data.goals:
            if player_rect.colliderect(goal.gObj.get_rect()):
                self.reached_goal = True
                self.alive = False
                return
        if self.player.gObj.y > self.level_data.height + 80:
            self.alive = False

    def _update_camera(self):
        if not self.player:
            return

        if self.render_mode == "human" and self.debug_manager.free_cam_active:
            dx, dy = self.debug_manager.current_cam_move
            max_x = max(0.0, self.level_data.width - self.WIDTH)
            max_y = max(0.0, self.level_data.height - self.HEIGHT)
            self.camera_x = max(0.0, min(max_x, self.camera_x + dx * self.dt))
            self.camera_y = max(0.0, min(max_y, self.camera_y + dy * self.dt))
            return

        target_x = self.player.gObj.x - self.WIDTH * 0.35
        target_y = self.player.gObj.y - self.HEIGHT * 0.45
        max_x = max(0.0, self.level_data.width - self.WIDTH)
        max_y = max(0.0, self.level_data.height - self.HEIGHT)
        self.camera_x = max(0.0, min(max_x, target_x))
        self.camera_y = max(0.0, min(max_y, target_y))

    def step(self, action):
        if not self.alive and not self.reached_goal:
            return self._obs(), 0.0, True, False, self._info()

        self.last_action = list(self._parse_action(action))
        self.steps += 1
        self.frame += 1
        self.time_limit = max(0.0, self.time_limit - self.dt)
        self.timer = self.time_limit
        if self.render_mode == "human":
            self.debug_manager.update_input()

        self._handle_input(action)
        self._update_player()
        self._update_enemies()
        self._update_projectiles()
        self._handle_combat()
        self._check_goal_and_oob()
        self._update_camera()
        self._update_debug_caches()

        truncated = self.steps >= self.max_steps or self.time_limit <= 0.0
        terminated = self.reached_goal or not self.alive
        reward = 0.01
        reward += (self.player.gObj.x if self.player else 0.0) / max(1.0, self.level_data.width) * 0.02
        if self.reached_goal:
            reward += 10.0
        if terminated and not self.reached_goal:
            reward -= 5.0

        return self._obs(), reward, terminated, truncated, self._info()

    def _obs(self) -> np.ndarray:
        obs = np.zeros(40, dtype=np.float32)
        if not self.player:
            return obs

        p = self.player
        obs[0] = np.clip(p.gObj.x / max(1.0, self.level_data.width), 0.0, 1.0)
        obs[1] = np.clip(p.gObj.y / max(1.0, self.level_data.height), 0.0, 1.0)
        obs[2] = np.clip(p.vx / 400.0, -1.0, 1.0)
        obs[3] = np.clip(p.vy / 700.0, -1.0, 1.0)
        obs[4] = 1.0 if p.on_ground else 0.0
        obs[5] = p.hp / float(p.hp_max)
        obs[6] = np.clip(p.shot_cooldown / 0.16, 0.0, 1.0)

        if self.enemies:
            nearest = min(self.enemies, key=lambda e: abs(e.gObj.x - p.gObj.x) + abs(e.gObj.y - p.gObj.y))
            obs[7] = np.clip((nearest.gObj.x - p.gObj.x) / 500.0, -1.0, 1.0)
            obs[8] = np.clip((nearest.gObj.y - p.gObj.y) / 400.0, -1.0, 1.0)
            obs[9] = nearest.hp / 3.0

        if self.projectiles:
            shots = [proj for proj in self.projectiles if proj.owner == "player"]
            if shots:
                shot = shots[0]
                obs[10] = np.clip((shot.gObj.x - p.gObj.x) / 500.0, -1.0, 1.0)
                obs[11] = np.clip((shot.gObj.y - p.gObj.y) / 300.0, -1.0, 1.0)

        tx = int(p.gObj.x // self.TILE_SIZE)
        ty = int(p.gObj.y // self.TILE_SIZE)
        idx = 12
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                gx = tx + dx
                gy = ty + dy
                val = 0.0
                if 0 <= gx < self.level_data.cols and 0 <= gy < self.level_data.rows:
                    tile = self.level_data.grid[gy][gx]
                    if tile in (TILE_GROUND, TILE_PLATFORM, TILE_QBLOCK):
                        val = 1.0
                    elif tile == TILE_SPIKE:
                        val = -1.0
                    elif tile == TILE_GOAL:
                        val = 0.75
                obs[idx] = val
                idx += 1
        obs[37] = np.clip(self.time_limit / 400.0, 0.0, 1.0)
        obs[38] = np.clip((self.level_data.width - p.gObj.x) / max(1.0, self.level_data.width), 0.0, 1.0)
        obs[39] = 1.0 if self.reached_goal else 0.0
        return obs

    def _info(self):
        return {
            "hp": self.player.hp if self.player else 0,
            "score": self.score,
            "level": self.world,
            "won": self.reached_goal,
            "action_name": self.action_to_str(self.last_action),
        }

    def _draw_background(self):
        self._surf.fill(MEGAMAN_SKY)
        pygame.draw.rect(self._surf, MEGAMAN_BG, (0, self.HEIGHT * 0.55, self.WIDTH, self.HEIGHT * 0.45))
        for i in range(5):
            cx = 80 + i * 180 - (self.camera_x * 0.12) % 180
            cy = 70 + (i % 2) * 40
            pygame.draw.circle(self._surf, (230, 240, 255), (int(cx), cy), 22)
            pygame.draw.circle(self._surf, (230, 240, 255), (int(cx + 20), cy + 2), 18)
            pygame.draw.circle(self._surf, (230, 240, 255), (int(cx - 18), cy + 4), 16)

    def _draw_world(self):
        for ladder in getattr(self.level_data, "ladders", []):
            sx = ladder.gObj.x - self.camera_x
            sy = ladder.gObj.y - self.camera_y
            if sx + ladder.gObj.width < 0 or sy + ladder.gObj.height < 0 or sx > self.WIDTH or sy > self.HEIGHT:
                continue
            ladder.render(self._surf, sx, sy)

        for row in range(self.level_data.rows):
            for col in range(self.level_data.cols):
                tile = self.level_data.grid[row][col]
                if tile == TILE_AIR:
                    continue
                sx = col * self.TILE_SIZE - self.camera_x
                sy = row * self.TILE_SIZE - self.camera_y
                if sx + self.TILE_SIZE < 0 or sy + self.TILE_SIZE < 0 or sx > self.WIDTH or sy > self.HEIGHT:
                    continue
                if tile in (TILE_GROUND, TILE_PLATFORM, TILE_QBLOCK):
                    pygame.draw.rect(self._surf, METAL_MID, (sx, sy, self.TILE_SIZE, self.TILE_SIZE))
                    pygame.draw.rect(self._surf, METAL_LIGHT, (sx, sy, self.TILE_SIZE, 6))
                    pygame.draw.rect(self._surf, METAL_DARK, (sx, sy + self.TILE_SIZE - 6, self.TILE_SIZE, 6))
                elif tile == TILE_SPIKE:
                    pygame.draw.polygon(
                        self._surf,
                        HAZARD_RED,
                        [(sx, sy + self.TILE_SIZE), (sx + self.TILE_SIZE * 0.5, sy), (sx + self.TILE_SIZE, sy + self.TILE_SIZE)],
                    )
                elif tile == TILE_GOAL:
                    pygame.draw.rect(self._surf, GOAL_GREEN, (sx, sy, self.TILE_SIZE, self.TILE_SIZE * 2))

    def _draw_player(self):
        if not self.player:
            return
        if self.player.i_frames > 0.0 and int(self.player.i_frames * 20) % 2 == 0:
            return
        sx = int(self.player.gObj.x - self.camera_x)
        sy = int(self.player.gObj.y - self.camera_y)
        w = self.player.gObj.width
        h = self.player.gObj.height
        pygame.draw.rect(self._surf, MEGAMAN_BLUE, (sx + 4, sy + 6, w - 8, h - 6), border_radius=6)
        pygame.draw.rect(self._surf, MEGAMAN_CYAN, (sx + 3, sy, w - 6, 12), border_radius=6)
        pygame.draw.rect(self._surf, MEGAMAN_DARK, (sx + 5, sy + h - 10, 7, 10), border_radius=4)
        pygame.draw.rect(self._surf, MEGAMAN_DARK, (sx + w - 12, sy + h - 10, 7, 10), border_radius=4)
        arm_x = sx + (w - 8 if self.player.facing_right else 2)
        pygame.draw.rect(self._surf, MEGAMAN_CYAN, (arm_x, sy + 16, 6, 10), border_radius=3)
        eye_x = sx + (w - 10 if self.player.facing_right else 6)
        pygame.draw.rect(self._surf, (255, 255, 255), (eye_x, sy + 5, 4, 3))
        if self.player.on_ladder:
            pygame.draw.rect(self._surf, (255, 240, 150), (sx + 6, sy + 11, w - 12, 6), border_radius=3)

    def _draw_entities(self):
        for enemy in self.enemies:
            enemy.render(self._surf, enemy.gObj.x - self.camera_x, enemy.gObj.y - self.camera_y)
        for proj in self.projectiles:
            proj.render(self._surf, proj.gObj.x - self.camera_x, proj.gObj.y - self.camera_y)

    def _draw_hud(self):
        hud = pygame.Surface((270, 42), pygame.SRCALPHA)
        hud.fill((0, 0, 0, 170))
        self._surf.blit(hud, (10, 10))
        text = self.ui_font.render(
            f"Lives:{self.lives} HP:{self.player.hp}/{self.player.hp_max} Score:{self.score} Time:{int(self.time_limit)}",
            True,
            (255, 255, 255),
        )
        self._surf.blit(text, (18, 14))
        bar_w = 200
        bar_x = 18
        bar_y = 36
        pygame.draw.rect(self._surf, (30, 30, 30), (bar_x, bar_y, bar_w, 8), border_radius=4)
        pygame.draw.rect(
            self._surf,
            MEGAMAN_CYAN,
            (bar_x, bar_y, int(bar_w * (self.player.hp / float(self.player.hp_max))), 8),
            border_radius=4,
        )

    def render(self, surface: pygame.Surface | None = None, blit_only: bool = False):
        self._draw_background()
        self._draw_world()
        self._draw_entities()
        self._draw_player()
        self._draw_hud()
        if self.render_mode == "human":
            self.debug_manager.render_overlays(self._surf, self)

        if blit_only and surface is not None:
            surface.blit(self._surf, (0, 0))
            return

        if self.render_mode == "human":
            pygame.display.flip()

    def get_action_space(self):
        return self.action_space

    def get_observation_space(self):
        return self.observation_space

    def action_to_str(self, action):
        try:
            move, jump, fire = self._parse_action(action)
        except Exception:
            return "IDLE"
        names = {
            0: "IDLE",
            1: "LEFT",
            2: "RUN_LEFT",
            3: "RIGHT",
            4: "RUN_RIGHT",
        }
        out = names.get(move, "IDLE")
        if jump:
            out += "+JUMP"
        if fire:
            out += "+FIRE"
        return out

    def close(self):
        if self.render_mode == "human":
            pygame.quit()


MegaManCore = MegamanCore
