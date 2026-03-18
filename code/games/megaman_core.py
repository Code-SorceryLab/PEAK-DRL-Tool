from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import heapq
import math
import os
import random
import yaml
from typing import List, Set, Tuple

import gymnasium
from gymnasium import spaces
import numpy as np
import pygame

from .modules.Objects.Enemy import Enemy as SharedEnemy
from .modules.Objects.GameObject import GameObject
from .modules.Objects.Player import Player as SharedPlayer
from .modules.Objects.Projectile import Projectile as SharedProjectile
from .modules.System.EntityType import EntityType
from .modules.System.LevelLoader import LevelLoader
from .modules.System.PhysicsManager import PhysicsManager
from .modules.System.config_manager import ConfigManager
from .modules.System.debugging_mods.manager import DebugManager
from .modules.Parameters.Map_parameters import (
    TILE_AIR,
    TILE_GROUND,
    TILE_PLATFORM,
    TILE_GOAL,
    TILE_SPIKE,
    TILE_QBLOCK,
    TILE_PIT,
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
PIT_DARK = (26, 8, 16)
PIT_ALERT = (196, 56, 84)


@dataclass
class MegaManPlayer(SharedPlayer):
    hp: int = 28
    hp_max: int = 28
    shot_cooldown: float = 0.0
    jump_cut: bool = False
    on_ladder: bool = False
    ladder_x: float = 0.0
    color = MEGAMAN_BLUE

    def __post_init__(self):
        # Mega Man uses the shared Player container/fields, but not Mario's
        # animation and power-state machines.
        self.anim_handler = None
        self.power_machine = None
        self.fire_requested = False
        self.iframes_timer = 0.0
        self.invincible_timer = 0.0
        self.star_timer = 0.0
        self.powered_up = False
        self.coyote = 0
        self.jump_hold = 0
        self.jump_buffer = 0
        self.last_jump_pressed = False
        self.run_pressed = False
        self.dt = 1.0
        self.input_dir = 0
        self.run_held = False
        self.jump_pressed = False

    @property
    def i_frames(self) -> float:
        return float(self.iframes_timer)

    @i_frames.setter
    def i_frames(self, value: float):
        self.iframes_timer = float(value)


class MegaBusterProjectile(SharedProjectile):
    WIDTH = 14
    HEIGHT = 8
    PLAYER_SPEED = 620.0
    ENEMY_SPEED = 300.0
    PLAYER_RANGE = 2.2
    ENEMY_RANGE = 1.8

    def __init__(self, x: float, y: float, vx: float, owner: str, damage: int, lifetime: float):
        super().__init__(
            gObj=GameObject(x, y, self.WIDTH, self.HEIGHT, True),
            vx=vx,
            vy=0.0,
            owner=owner,
            damage=damage,
            lifetime=lifetime,
        )

    @classmethod
    def from_player(cls, player: MegaManPlayer) -> "MegaBusterProjectile":
        x = player.gObj.x + (player.gObj.width - 4 if player.facing_right else -cls.WIDTH + 4)
        y = player.gObj.y + player.gObj.height * 0.45
        vx = cls.PLAYER_SPEED if player.facing_right else -cls.PLAYER_SPEED
        return cls(x, y, vx, "player", damage=1, lifetime=cls.PLAYER_RANGE)

    @classmethod
    def from_enemy(cls, x: float, y: float, facing_right: bool, damage: int = 4) -> "MegaBusterProjectile":
        vx = cls.ENEMY_SPEED if facing_right else -cls.ENEMY_SPEED
        return cls(x, y, vx, "enemy", damage=damage, lifetime=cls.ENEMY_RANGE)

    def update(self, dt: float):
        self.begin_frame()
        if not self.tick_lifetime(dt):
            return
        self.gObj.x += self.vx * dt

    def render(self, surface: pygame.Surface, sx: float, sy: float):
        if not self.gObj.active:
            return
        outer = MEGAMAN_CYAN if self.owner == "player" else SHOT_ORANGE
        pygame.draw.rect(surface, outer, (sx, sy, self.gObj.width, self.gObj.height), border_radius=4)
        pygame.draw.rect(surface, (255, 255, 255), (sx + 3, sy + 2, self.gObj.width - 6, self.gObj.height - 4), border_radius=3)


class MegaDijkstraSolver:
    def __init__(self, grid: List[List[int]], rows: int, cols: int, ladders: Set[Tuple[int, int]]):
        self.grid = grid
        self.rows = rows
        self.cols = cols
        self.ladders = ladders
        self.dist_map = np.full((rows, cols), float("inf"), dtype=np.float32)

    def compute_map(self, goals: List[Tuple[int, int]]):
        self.dist_map.fill(float("inf"))
        if not goals:
            return

        solids = {TILE_GROUND, TILE_PLATFORM, TILE_QBLOCK}
        pq: list[tuple[float, int, int]] = []

        for gx, gy in goals:
            if 0 <= gx < self.cols and 0 <= gy < self.rows:
                self.dist_map[gy, gx] = 0.0
                heapq.heappush(pq, (0.0, gx, gy))

        directions = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (-1, 1), (1, -1), (-1, -1),
        ]

        def has_nearby_support(tx: int, ty: int) -> bool:
            if (tx, ty) in self.ladders:
                return True
            for depth in range(1, 5):
                ny = ty + depth
                if ny >= self.rows:
                    break
                if self.grid[ny][tx] in solids or (tx, ny) in self.ladders:
                    return True
            return False

        while pq:
            current_dist, cx, cy = heapq.heappop(pq)
            if current_dist > self.dist_map[cy, cx]:
                continue

            for dx, dy in directions:
                nx = cx + dx
                ny = cy + dy
                if not (0 <= nx < self.cols and 0 <= ny < self.rows):
                    continue

                tile = self.grid[ny][nx]
                if tile in solids or tile == TILE_PIT:
                    continue

                uses_ladder = (cx, cy) in self.ladders or (nx, ny) in self.ladders
                if dy < 0:
                    step_cost = 1.0 if uses_ladder else 3.5
                elif dy > 0:
                    step_cost = 1.0 if uses_ladder else 1.25
                else:
                    step_cost = 1.9

                if dx != 0 and dy != 0:
                    step_cost *= 1.12

                if tile == TILE_SPIKE:
                    step_cost += 14.0
                if (nx, ny) in self.ladders:
                    step_cost -= 0.45
                if not uses_ladder and dy < 0 and not has_nearby_support(nx, ny):
                    step_cost += 2.5
                if tile == TILE_AIR and ny + 1 < self.rows and self.grid[ny + 1][nx] in solids:
                    step_cost -= 0.35

                step_cost = max(0.75, step_cost)
                new_dist = current_dist + step_cost
                if new_dist < self.dist_map[ny, nx]:
                    self.dist_map[ny, nx] = new_dist
                    heapq.heappush(pq, (new_dist, nx, ny))

    def get_dist(self, x: int, y: int) -> float:
        if 0 <= x < self.cols and 0 <= y < self.rows:
            value = self.dist_map[y, x]
            if math.isfinite(float(value)):
                return float(value)
        return -1.0


class BaseEnemy(SharedEnemy):
    def __init__(self, x: float, y: float, width: int, height: int, hp: int = 2):
        super().__init__(
            gObj=GameObject(x, y, width, height, True),
            vx=0.0,
            vy=0.0,
            on_ground=False,
            facing_right=True,
            hp=hp,
            hp_max=hp,
            contact_damage=4,
        )
        self.gObj.type_id = EntityType.ENEMY

    def take_damage(self, damage: int) -> bool:
        self.hp -= damage
        if self.hp <= 0:
            self.gObj.active = False
            return True
        return False

    def render(self, surface: pygame.Surface, sx: float, sy: float):
        super().render(surface, sx, sy)


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


class BossEnemy(BaseEnemy):
    def __init__(self, x: float, y: float):
        super().__init__(x, y, 52, 58, hp=24)
        self.contact_damage = 6
        self.vx = -84.0
        self.shot_timer = 0.7
        self.jump_timer = 1.4
        self.name = "BOSS CORE"

    def update(self, dt: float, core: "MegamanCore"):
        player = core.player
        if player is not None:
            dx = player.gObj.x - self.gObj.x
            self.facing_right = dx >= 0.0
            if abs(dx) > 28.0:
                self.vx = 96.0 if dx > 0 else -96.0
            else:
                self.vx = 0.0

        self.shot_timer -= dt
        self.jump_timer -= dt
        if self.shot_timer <= 0.0 and player is not None:
            shot_x = self.gObj.x + (self.gObj.width if self.facing_right else -MegaBusterProjectile.WIDTH)
            # Boss shots should come from its own muzzle height, not snap to the
            # player's Y position.
            shot_y = self.gObj.y + self.gObj.height * 0.45
            core.projectiles.append(MegaBusterProjectile.from_enemy(shot_x, shot_y, self.facing_right, damage=5))
            self.shot_timer = 0.95

        if self.on_ground and self.jump_timer <= 0.0:
            self.vy = core.jump_velocity * 0.7
            self.on_ground = False
            self.jump_timer = 1.6

    def render(self, surface: pygame.Surface, sx: float, sy: float):
        pygame.draw.rect(surface, (36, 18, 22), (sx, sy, self.gObj.width, self.gObj.height), border_radius=10)
        pygame.draw.rect(surface, (236, 108, 90), (sx + 6, sy + 8, self.gObj.width - 12, self.gObj.height - 14), border_radius=10)
        pygame.draw.rect(surface, MET_YELLOW, (sx + 10, sy + 3, self.gObj.width - 20, 12), border_radius=6)
        pygame.draw.rect(surface, (88, 28, 32), (sx + 10, sy + self.gObj.height - 16, self.gObj.width - 20, 10), border_radius=5)
        eye_y = sy + 20
        if self.facing_right:
            pygame.draw.rect(surface, (255, 255, 255), (sx + 28, eye_y, 7, 5), border_radius=2)
            pygame.draw.rect(surface, (255, 255, 255), (sx + 38, eye_y, 7, 5), border_radius=2)
        else:
            pygame.draw.rect(surface, (255, 255, 255), (sx + 7, eye_y, 7, 5), border_radius=2)
            pygame.draw.rect(surface, (255, 255, 255), (sx + 17, eye_y, 7, 5), border_radius=2)


class MegaPhysicsManager(PhysicsManager):
    def __init__(self, core: "MegamanCore"):
        super().__init__()
        self.core = core
        self.gravity = 980.0
        self.jump_force = 430.0

    def rebuild_dynamic_hashes(self, core: "MegamanCore"):
        self.hazard_hash.clear()
        self.collectible_hash.clear()
        self.platform_hash.clear()

        for enemy in core.enemies:
            if enemy.gObj.active:
                self.hazard_hash.insert(enemy)
        for spike in core.spike_tiles:
            self.hazard_hash.insert(spike)
        for pit in core._exposed_pit_objects():
            self.hazard_hash.insert(pit)
        for proj in core.projectiles:
            if proj.gObj.active and proj.owner == "enemy":
                self.hazard_hash.insert(proj)

        if not core.level_data:
            return

        for coin in getattr(core.level_data, "coins", []):
            if coin.gObj.active and not getattr(coin, "collected", False):
                self.collectible_hash.insert(coin)

        if not core.is_boss_level or core._current_boss() is None:
            for goal in getattr(core.level_data, "goals", []):
                if goal.gObj.active:
                    self.collectible_hash.insert(goal)

    def _tile_rects_for_rect(self, core: "MegamanCore", rect: pygame.Rect):
        if not core.level_data:
            return []
        pad = core.TILE_SIZE * 1.5
        found = []
        nearby = core.level_data.static_hash.query_rect(rect.x - pad, rect.y - pad, rect.width + pad * 2, rect.height + pad * 2)
        for item in nearby:
            gobj = item.gObj if hasattr(item, "gObj") else item
            col = int(gobj.x // core.TILE_SIZE)
            row = int(gobj.y // core.TILE_SIZE)
            if not (0 <= row < core.level_data.rows and 0 <= col < core.level_data.cols):
                continue
            tile_type = core.level_data.grid[row][col]
            found.append((gobj.get_rect(), tile_type))
        return found

    def _support_tile_below(self, core: "MegamanCore", actor, epsilon: float = 3.0):
        rect = actor.gObj.get_rect()
        probe = pygame.Rect(rect.left + 2, rect.bottom, max(2, rect.width - 4), max(2, int(epsilon) + 1))
        best = None
        best_top = None
        for tile_rect, tile_type in self._tile_rects_for_rect(core, probe):
            if tile_type == TILE_SPIKE:
                continue
            if tile_type == TILE_PLATFORM and rect.bottom > tile_rect.top + epsilon:
                continue
            if not probe.colliderect(tile_rect):
                continue
            if best is None or tile_rect.top < best_top:
                best = (tile_rect, tile_type)
                best_top = tile_rect.top
        return best

    def _stabilize_ground_contact(self, core: "MegamanCore", actor, epsilon: float = 3.0) -> bool:
        support = self._support_tile_below(core, actor, epsilon=epsilon)
        if support is None:
            actor.on_ground = False
            return False
        tile_rect, _tile_type = support
        actor.gObj.y = tile_rect.top - actor.gObj.height
        if actor.vy > 0.0:
            actor.vy = 0.0
        actor.on_ground = True
        return True

    def _resolve_actor_y(self, core: "MegamanCore", actor, damage_spikes: bool = False):
        rect = actor.gObj.get_rect()
        actor.on_ground = False
        for tile_rect, tile_type in self._tile_rects_for_rect(core, rect):
            if not rect.colliderect(tile_rect):
                continue
            if tile_type == TILE_SPIKE and damage_spikes and actor is core.player:
                core._damage_player(actor.hp_max, "Spike")
                continue
            if tile_type == TILE_PLATFORM:
                if actor.vy < 0:
                    continue
                if rect.bottom - actor.vy * core.dt > tile_rect.top + 4:
                    continue
            if actor.vy >= 0:
                actor.gObj.y = tile_rect.top - actor.gObj.height
                actor.vy = 0.0
                actor.on_ground = True
            else:
                actor.gObj.y = tile_rect.bottom
                actor.vy = 0.0
            rect = actor.gObj.get_rect()

    def _resolve_actor_x(self, core: "MegamanCore", actor, bounce: bool = False):
        rect = actor.gObj.get_rect()
        for tile_rect, tile_type in self._tile_rects_for_rect(core, rect):
            if tile_type in (TILE_SPIKE, TILE_PLATFORM):
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

    def update_system(self, dt: float, core: "MegamanCore"):
        self._update_player(dt, core)
        self._update_enemies(dt, core)
        self._update_projectiles(dt, core)

    def _update_player(self, dt: float, core: "MegamanCore"):
        p = core.player
        if not p:
            return
        p.shot_cooldown = max(0.0, p.shot_cooldown - dt)
        p.i_frames = max(0.0, p.i_frames - dt)
        p.invincible_timer = p.i_frames
        supported = self._stabilize_ground_contact(core, p) if not p.on_ladder else False
        p.grounded = p.on_ground
        p.can_jump = p.on_ground or p.on_ladder

        if p.on_ladder:
            p.gObj.x = p.ladder_x
            p.gObj.y += p.vy * dt
            ladder = core._nearest_ladder(p.gObj)
            if ladder is not None:
                p.ladder_x = ladder.gObj.x + ladder.gObj.width * 0.5 - p.gObj.width * 0.5
                p.gObj.x = p.ladder_x
            else:
                p.on_ladder = False

            support = self._support_tile_below(core, p, epsilon=8.0)
            if p.vy < 0.0 and support is not None:
                tile_rect, _tile_type = support
                if p.gObj.y + p.gObj.height <= tile_rect.top + 8:
                    p.on_ladder = False
                    p.on_ground = True
                    p.can_jump = True
                    p.vy = 0.0
                    p.gObj.y = tile_rect.top - p.gObj.height
                    core.progress_x_best = max(core.progress_x_best, p.gObj.x)
                    return

            self._resolve_actor_y(core, p, damage_spikes=True)
            core.progress_x_best = max(core.progress_x_best, p.gObj.x)
            return

        if supported and p.vy >= 0.0:
            p.vy = 0.0
        else:
            p.vy = min(p.vy + core.gravity * dt, core.max_fall_speed)
        p.gObj.y += p.vy * dt
        self._resolve_actor_y(core, p, damage_spikes=True)
        p.gObj.x += p.vx * dt
        self._resolve_actor_x(core, p)
        self._stabilize_ground_contact(core, p)
        core.progress_x_best = max(core.progress_x_best, p.gObj.x)

    def _update_enemies(self, dt: float, core: "MegamanCore"):
        gravity = core.gravity * core.enemy_gravity_mult
        for enemy in core.enemies:
            if not enemy.gObj.active:
                continue
            enemy.update(dt, core)
            if isinstance(enemy, (MetEnemy, BossEnemy)):
                supported = self._stabilize_ground_contact(core, enemy)
                if supported and enemy.vy >= 0.0:
                    enemy.vy = 0.0
                else:
                    enemy.vy = min(enemy.vy + gravity * dt, core.enemy_max_fall_speed)
                enemy.gObj.y += enemy.vy * dt
                self._resolve_actor_y(core, enemy)
                enemy.gObj.x += enemy.vx * dt
                self._resolve_actor_x(core, enemy, bounce=True)
                self._stabilize_ground_contact(core, enemy)
            elif isinstance(enemy, BatEnemy):
                enemy.gObj.x += enemy.vx * dt
                self._resolve_actor_x(core, enemy, bounce=True)
            else:
                enemy.gObj.x += enemy.vx * dt
                self._resolve_actor_x(core, enemy, bounce=True)
            if enemy.gObj.y > core.level_data.height + 120:
                enemy.gObj.active = False

    def _update_projectiles(self, dt: float, core: "MegamanCore"):
        for proj in core.projectiles:
            if not proj.gObj.active:
                continue
            proj.update(dt)

    def _swept_rect(self, proj: MegaBusterProjectile) -> pygame.Rect:
        prev_rect = pygame.Rect(int(proj.prev_x), int(proj.prev_y), proj.gObj.width, proj.gObj.height)
        curr_rect = proj.gObj.get_rect()
        return prev_rect.union(curr_rect)

    def _hit_distance(self, start_x: float, rect: pygame.Rect, direction: float) -> float:
        hit_x = rect.left if direction >= 0.0 else rect.right
        return (float(hit_x) - start_x) * direction

    def _resolve_projectile_collisions(self, core: "MegamanCore"):
        player_rect = core.player.gObj.get_rect() if core.player else None

        for proj in core.projectiles:
            if not proj.gObj.active:
                continue

            swept = self._swept_rect(proj)
            direction = 1.0 if proj.vx >= 0.0 else -1.0
            start_x = proj.prev_x + proj.gObj.width * 0.5

            world_hit = None
            for tile_rect, tile_type in self._tile_rects_for_rect(core, swept):
                if tile_type == TILE_AIR:
                    continue
                if not swept.colliderect(tile_rect):
                    continue
                dist = self._hit_distance(start_x, tile_rect, direction)
                if dist < -4.0:
                    continue
                if world_hit is None or dist < world_hit[0]:
                    world_hit = (dist, tile_rect)

            if proj.owner == "player":
                enemy_hit = None
                for enemy in core.enemies:
                    if not enemy.gObj.active:
                        continue
                    enemy_rect = enemy.gObj.get_rect()
                    if not swept.colliderect(enemy_rect):
                        continue
                    dist = self._hit_distance(start_x, enemy_rect, direction)
                    if dist < -4.0:
                        continue
                    if enemy_hit is None or dist < enemy_hit[0]:
                        enemy_hit = (dist, enemy)

                if enemy_hit is not None and (world_hit is None or enemy_hit[0] <= world_hit[0]):
                    proj.gObj.active = False
                    enemy = enemy_hit[1]
                    if enemy.take_damage(proj.damage):
                        core.enemies_killed_step += 1
                        core.score += 100
                    else:
                        core.score += 20
                    if isinstance(enemy, BossEnemy):
                        core.boss_damage_step += proj.damage
                    continue
            elif player_rect is not None and swept.colliderect(player_rect):
                player_dist = self._hit_distance(start_x, player_rect, direction)
                if world_hit is None or player_dist <= world_hit[0]:
                    proj.gObj.active = False
                    core._damage_player(proj.damage, "Projectile")
                    continue

            if world_hit is not None:
                proj.gObj.active = False

    def _resolve_actor_contacts(self, core: "MegamanCore"):
        if not core.player:
            return

        player_rect = core.player.gObj.get_rect()
        for enemy in core.enemies:
            if enemy.gObj.active and player_rect.colliderect(enemy.gObj.get_rect()):
                core._damage_player(enemy.contact_damage, "Enemy")

    def _check_goal_and_oob(self, core: "MegamanCore"):
        if not core.player:
            return

        player_rect = core.player.gObj.get_rect()
        boss = core._current_boss()
        if core.is_boss_level:
            if boss is None:
                core.reached_goal = True
                core.alive = False
                core.last_event = "WIN"
                core.last_cause = ""
                core._episode_won_current = True
                return
        else:
            for goal in core.level_data.goals:
                if player_rect.colliderect(goal.gObj.get_rect()):
                    core.reached_goal = True
                    core.alive = False
                    core.last_event = "WIN"
                    core.last_cause = ""
                    core._episode_won_current = True
                    return

        for pit in core._exposed_pit_objects():
            if player_rect.colliderect(pit.get_rect()):
                core.alive = False
                core.last_cause = "Pit"
                return

        if core.player.gObj.y > core.level_data.height + 80:
            core.alive = False
            core.last_cause = "Pit"

    def resolve_collisions(self, core: "MegamanCore"):
        self._resolve_projectile_collisions(core)
        self.rebuild_dynamic_hashes(core)
        self._resolve_actor_contacts(core)
        self._check_goal_and_oob(core)
        core.projectiles = [proj for proj in core.projectiles if proj.gObj.active]
        core.enemies = [enemy for enemy in core.enemies if enemy.gObj.active]
        core.level_data.enemies = core.enemies
        core.level_data.projectiles = core.projectiles


class MegamanCore(gymnasium.Env):
    metadata = {"render_modes": ["none", "human", "rgb_array"]}

    def _apply_physics_config(self, physics_cfg: dict | None, enemy_cfg: dict | None = None):
        physics_cfg = physics_cfg or {}
        enemy_cfg = enemy_cfg or physics_cfg.get("enemies", {}) or {}
        friction_cfg = physics_cfg.get("friction", 2600.0)
        if isinstance(friction_cfg, dict):
            friction_value = float(friction_cfg.get("ground", 2600.0))
        else:
            friction_value = float(friction_cfg)
        self.walk_speed = float(physics_cfg.get("walk_speed", getattr(self, "walk_speed", 240.0)))
        self.run_speed = float(physics_cfg.get("run_speed", getattr(self, "run_speed", 320.0)))
        self.accel = float(physics_cfg.get("accel", getattr(self, "accel", 2800.0)))
        self.friction = friction_value
        self.jump_velocity = float(physics_cfg.get("jump_velocity", getattr(self, "jump_velocity", -560.0)))
        self.climb_speed = float(physics_cfg.get("climb_speed", getattr(self, "climb_speed", 190.0)))
        self.gravity = float(physics_cfg.get("gravity", getattr(self, "gravity", 1500.0)))
        self.max_fall_speed = float(physics_cfg.get("max_fall_speed", getattr(self, "max_fall_speed", 900.0)))
        self.i_frame_duration = float(physics_cfg.get("i_frame_duration", getattr(self, "i_frame_duration", 0.75)))
        self.hp_max = int(physics_cfg.get("hp_max", getattr(self, "hp_max", 28)))
        self.shot_cooldown_time = float(physics_cfg.get("shot_cooldown", getattr(self, "shot_cooldown_time", 0.16)))
        self.max_player_shots = int(physics_cfg.get("max_player_shots", getattr(self, "max_player_shots", 3)))
        self.fire_buffer_time = float(physics_cfg.get("fire_buffer", getattr(self, "fire_buffer_time", 0.12)))
        self.enemy_gravity_mult = float(enemy_cfg.get("gravity_mult", getattr(self, "enemy_gravity_mult", 1.0)))
        self.enemy_max_fall_speed = float(enemy_cfg.get("max_fall_speed", getattr(self, "enemy_max_fall_speed", 760.0)))
        self.physics_manager.gravity = self.gravity
        self.physics_manager.jump_force = abs(self.jump_velocity)
        MegaBusterProjectile.PLAYER_SPEED = float(physics_cfg.get("bullet_speed", getattr(MegaBusterProjectile, "PLAYER_SPEED", 760.0)))
        MegaBusterProjectile.ENEMY_SPEED = float(enemy_cfg.get("bullet_speed", getattr(MegaBusterProjectile, "ENEMY_SPEED", MegaBusterProjectile.PLAYER_SPEED * 0.58)))
        if getattr(self, "player", None) is not None:
            self.player.hp_max = self.hp_max
            self.player.hp = min(self.player.hp, self.player.hp_max)

    def __init__(self, render_mode: str = "none", **kwargs):
        self.render_mode = render_mode
        self.config_manager = ConfigManager("game_config.yaml")
        self.raw_config = self.config_manager.yaml_data.get("megaman", {})
        physics_cfg = self.raw_config.get("physics", {}) or {}

        self.WIDTH = int(self.raw_config.get("screen_width", 960))
        self.HEIGHT = int(self.raw_config.get("screen_height", 600))
        self.FPS = int(self.raw_config.get("fps", 60))
        self.TILE_SIZE = int(self.raw_config.get("tile_size", 32))
        self.TOTAL_WIDTH = self.WIDTH + DEBUG_PANEL_WIDTH
        self.DEBUG_PANEL_X = self.WIDTH
        self.obs_mode = "dict"

        self.action_space = spaces.MultiDiscrete([5, 3, 2, 2])
        self.observation_space = spaces.Dict({
            "grids": spaces.Box(low=-1.0, high=1.0, shape=(4, 21, 21), dtype=np.float32),
            "scalars": spaces.Box(low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32),
        })

        self.loader = LevelLoader(tile_size=self.TILE_SIZE)
        pygame.init()
        self.debug_manager = DebugManager(
            default_active=(render_mode == "human"),
            print_help=(render_mode == "human"),
            sensor_mode="shot",
        )
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
        self.spike_tiles: List[object] = []
        self.ladder_tiles: Set[Tuple[int, int]] = set()
        self.physics_manager = MegaPhysicsManager(self)
        self.level_meta: dict = {}
        self.is_boss_level = False
        self.shot_cooldown_time = 0.16
        self.max_player_shots = 3
        self.fire_buffer_time = 0.12
        self.fire_buffer = 0.0
        self._apply_physics_config(physics_cfg)

        config_levels = self.raw_config.get("levels", {}) or {}
        self.level_order = list(config_levels.keys())
        default_world = self.level_order[0] if self.level_order else "MM-Stage1"
        self.world = str(kwargs.get("world", default_world))
        self.current_index_world = 0
        self.locked_level = str(self.world) if kwargs.pop("lock_level", False) else None
        self._refresh_level_order()
        self._curriculum_window_size = int(kwargs.pop("curriculum_window", 5))
        self._batch_window = int(kwargs.pop("batch_window", 10))
        self._batch_advance_threshold = float(kwargs.pop("advance_threshold", 0.30))
        self._batch_fallback_threshold = float(kwargs.pop("fallback_threshold", 0.20))
        self._max_stay_windows = int(kwargs.pop("max_stay_windows", 3))
        self._review_prob = float(kwargs.pop("review_prob", 0.25))
        self.curriculum_enabled = bool(kwargs.pop("curriculum_enabled", True))
        self._curriculum_position = max(0, min(int(kwargs.pop("start_unlocked", 0)), max(0, len(self.level_order) - 1)))
        self._max_unlocked_index = self._curriculum_position
        self._batch_results: List[bool] = []
        self._episode_won_current = False
        self._is_review_episode = False
        self._windows_on_level = 0
        self._consecutive_fallbacks = {}
        self._level_visits = {lvl: 0 for lvl in self.level_order}
        self._level_wins = {lvl: 0 for lvl in self.level_order}
        self._level_window = {
            lvl: deque(maxlen=self._curriculum_window_size)
            for lvl in self.level_order
        }
        self.camera_x = 0.0
        self.camera_y = 0.0
        self.steps = 0
        max_steps_cfg = kwargs.pop("max_steps", self.raw_config.get("episode", {}).get("max_steps", 3000))
        self.max_steps = None if max_steps_cfg is None else int(max_steps_cfg)
        self.time_limit = 400.0
        self.lives = 3
        self.score = 0
        self.coins_total = 0
        self.reached_goal = False
        self.alive = True
        self.dt = 1.0 / float(self.FPS)
        self.last_action = [0, 0, 0, 0]
        self.frame = 0
        self.timer = self.time_limit
        self.persona = str(kwargs.pop("persona", "simple")).lower()
        if self.persona == "default":
            self.persona = "simple"
        self.arch_tag = str(kwargs.pop("arch_tag", "mlp")).lower()
        self.progress_x_best = 0.0
        self.stall_timer = 0.0
        self.stall_windows_count = 0
        self.stalled_this_frame = False
        self.obs_width = 21
        self.obs_height = 21
        self.obs_pad_x = self.obs_width // 2
        self.obs_pad_y = self.obs_height // 2
        self._visit_map = None
        self._dijkstra_window_cache = None
        self._solid_window_cache = None
        self._hazard_window_cache = None
        self._step_dx = 0.0
        self._step_dy = 0.0
        self.dijkstra = None
        self.dijkstra_current_tile = -1.0
        self.num_rays = 1
        self.ray_max_dist = 240.0
        self.ray_step = 8.0
        self.ray_angles = np.array([0.0], dtype=np.float32)
        self.last_rays = []
        self.last_event = ""
        self.last_cause = ""
        self.enemies_killed_step = 0
        self.boss_damage_step = 0
        self.damage_taken_step = 0
        self._prev_goal_dist = 0.0
        self.max_x_seen = 0.0
        self._obs_check_interval = 1
        self._obs_check_counter = 0
        self._obs_stats = {
            "grid_solid_mean": 0.0,       "grid_solid_std": 0.0,
            "grid_solid_min": 0.0,        "grid_solid_max": 0.0,
            "grid_collectible_mean": 0.0, "grid_collectible_std": 0.0,
            "grid_collectible_min": 0.0,  "grid_collectible_max": 0.0,
            "grid_hazard_mean": 0.0,      "grid_hazard_std": 0.0,
            "grid_hazard_min": 0.0,       "grid_hazard_max": 0.0,
            "grid_dijkstra_mean": 0.0,    "grid_dijkstra_std": 0.0,
            "grid_dijkstra_min": 0.0,     "grid_dijkstra_max": 0.0,
            "scalar_mean": 0.0, "scalar_std": 0.0,
            "scalar_min": 0.0,  "scalar_max": 0.0,
            "dijkstra_val": 0.0, "obs_warnings": "",
        }

    def _hazard_objects(self):
        hazards = []
        hazards.extend(self.enemies)
        hazards.extend(self.spike_tiles)
        if self.level_data:
            hazards.extend(self._exposed_pit_objects())
        hazards.extend(
            proj for proj in self.projectiles
            if getattr(proj.gObj, "active", True) and proj.owner == "enemy"
        )
        return hazards

    def _collectible_objects(self):
        if not self.level_data:
            return []
        items = list(getattr(self.level_data, "coins", []))
        if not self.is_boss_level or self._current_boss() is None:
            items.extend(list(getattr(self.level_data, "goals", [])))
        return items

    def _levels_config(self) -> dict:
        levels = {}
        for level_id, cfg in (self.raw_config.get("levels", {}) or {}).items():
            if isinstance(cfg, dict):
                levels[str(level_id)] = cfg

        injected_levels = self.config_manager.yaml_data.get("levels", {}) or {}
        for level_id, cfg in injected_levels.items():
            if not isinstance(cfg, dict):
                continue
            level_file = str(cfg.get("file", ""))
            if os.path.isabs(level_file) or str(level_id).startswith("__editor_"):
                levels[str(level_id)] = cfg
        return levels

    def _refresh_level_order(self):
        levels = self._levels_config()
        self.level_order = list(levels.keys())
        if self.locked_level:
            self.locked_level = self._resolve_level_key(self.locked_level)
        resolved_world = self._resolve_level_key(self.world)
        if self.level_order:
            if resolved_world not in self.level_order:
                resolved_world = self.level_order[0]
            self.world = resolved_world
            self.current_index_world = self.level_order.index(self.world)
        else:
            self.world = resolved_world
            self.current_index_world = 0

    def _select_level_for_reset(self, won_previous: bool):
        self._refresh_level_order()
        if self.locked_level:
            self.world = self.locked_level
            if self.locked_level in self.level_order:
                self.current_index_world = self.level_order.index(self.locked_level)
            else:
                self.current_index_world = 0
            self._episode_won_current = False
            self._is_review_episode = False
            return
        if not self.level_order:
            return

        if not self.curriculum_enabled:
            if self.world in self.level_order:
                self.current_index_world = self.level_order.index(self.world)
            else:
                self.current_index_world = 0
            if won_previous and len(self.level_order) > 1:
                self.current_index_world = (self.current_index_world + 1) % len(self.level_order)
            self.world = self.level_order[self.current_index_world]
            self._level_visits[self.world] = self._level_visits.get(self.world, 0) + 1
            self._episode_won_current = False
            self._is_review_episode = False
            return

        completed_level = self.world if self.world in self.level_order else self.level_order[self.current_index_world]
        self._level_window.setdefault(completed_level, deque(maxlen=self._curriculum_window_size)).append(1 if won_previous else 0)
        if won_previous:
            self._level_wins[completed_level] = self._level_wins.get(completed_level, 0) + 1

        if not self._is_review_episode:
            self._batch_results.append(bool(won_previous))
        if len(self._batch_results) >= self._batch_window:
            self._evaluate_curriculum_batch()

        self._episode_won_current = False
        self._is_review_episode = False
        self._curriculum_position = max(0, min(self._curriculum_position, len(self.level_order) - 1))
        self._max_unlocked_index = max(self._max_unlocked_index, self._curriculum_position)

        if self._curriculum_position > 0 and random.random() < self._review_prob:
            self.current_index_world = random.randint(0, self._curriculum_position - 1)
            self._is_review_episode = True
        else:
            self.current_index_world = self._curriculum_position

        self.world = self.level_order[self.current_index_world]
        self._level_visits[self.world] = self._level_visits.get(self.world, 0) + 1

    def _evaluate_curriculum_batch(self):
        wins = sum(1 for r in self._batch_results if r)
        total = len(self._batch_results)
        if total <= 0 or not self.level_order:
            self._batch_results.clear()
            return

        win_rate = wins / total
        pos = max(0, min(self._curriculum_position, len(self.level_order) - 1))
        consec_fb = self._consecutive_fallbacks.get(pos, 0)
        effective_advance = self._batch_advance_threshold
        if consec_fb >= 2:
            effective_advance = max(0.10, effective_advance - 0.10)

        if win_rate >= effective_advance and pos < len(self.level_order) - 1:
            self._curriculum_position += 1
            self._windows_on_level = 0
            self._consecutive_fallbacks[pos] = 0
            self._max_unlocked_index = max(self._max_unlocked_index, self._curriculum_position)
        elif win_rate <= self._batch_fallback_threshold and pos > 0:
            self._curriculum_position -= 1
            self._windows_on_level = 0
            self._consecutive_fallbacks[pos] = consec_fb + 1
        else:
            self._windows_on_level += 1
            if self._windows_on_level >= self._max_stay_windows and pos > 0:
                self._curriculum_position -= 1
                self._windows_on_level = 0

        self._batch_results.clear()

    def _curriculum_win_rate(self) -> float:
        if self.world not in self._level_window or not self._level_window[self.world]:
            return -1.0
        window = self._level_window[self.world]
        return float(sum(window) / len(window))

    def _resolve_level_key(self, level_id: str) -> str:
        levels = self._levels_config()
        if level_id in levels:
            return level_id
        level_id_lower = str(level_id).lower()
        for key in levels:
            if str(key).lower() == level_id_lower:
                return str(key)
        return str(level_id)

    def _level_config(self) -> dict:
        resolved = self._resolve_level_key(self.world)
        levels = self._levels_config()
        if resolved in levels:
            self.world = resolved
        return levels.get(self.world, {"file": "megaman/mm_stage1.txt"})

    def _level_file_path(self) -> str:
        level_file = self._level_config().get("file", "megaman/mm_stage1.txt")
        return self.loader.resolve_level_path(level_file)

    def _load_source_lines(self) -> List[str]:
        path = self._level_file_path()
        if not os.path.isfile(path):
            return []
        with open(path, "r", encoding="utf-8") as fh:
            return [line.rstrip("\n") for line in fh]

    def _goal_metrics(self) -> Tuple[float, float, float]:
        if not self.player or not self.level_data:
            return 0.0, 0.0, 0.0

        target = None
        if self.is_boss_level:
            target = self._current_boss()
        if target is None and self.level_data.goals:
            target = min(
                self.level_data.goals,
                key=lambda g: abs(g.gObj.x - self.player.gObj.x) + abs(g.gObj.y - self.player.gObj.y),
            )
        if target is None:
            return 0.0, 0.0, 0.0

        goal_dx = target.gObj.x - self.player.gObj.x
        goal_dy = target.gObj.y - self.player.gObj.y
        path_norm = self._normalized_dijkstra()
        if path_norm >= 0.0:
            return goal_dx, goal_dy, path_norm
        norm = (abs(goal_dx) + abs(goal_dy)) / max(1.0, self.level_data.width + self.level_data.height)
        return goal_dx, goal_dy, float(np.clip(norm, 0.0, 1.0))

    def _is_cover_tile(self, tile: int) -> bool:
        return tile in (TILE_GROUND, TILE_PLATFORM, TILE_QBLOCK)

    def _is_exposed_pit_tile(self, tile_x: int, tile_y: int) -> bool:
        if not self.level_data:
            return False
        if not (0 <= tile_x < self.level_data.cols and 0 <= tile_y < self.level_data.rows):
            return False
        if self.level_data.grid[tile_y][tile_x] != TILE_PIT:
            return False
        if tile_y == 0:
            return True
        return not self._is_cover_tile(self.level_data.grid[tile_y - 1][tile_x])

    def _exposed_pit_objects(self):
        if not self.level_data:
            return []
        out = []
        for pit in getattr(self.level_data, "pits", []):
            tx = int(pit.x // self.TILE_SIZE)
            ty = int(pit.y // self.TILE_SIZE)
            if self._is_exposed_pit_tile(tx, ty):
                out.append(pit)
        return out

    def _ladder_tiles(self) -> Set[Tuple[int, int]]:
        ladder_tiles: Set[Tuple[int, int]] = set()
        for ladder in getattr(self.level_data, "ladders", []):
            c0 = int(ladder.gObj.x // self.TILE_SIZE)
            c1 = int((ladder.gObj.x + ladder.gObj.width - 1) // self.TILE_SIZE)
            r0 = int(ladder.gObj.y // self.TILE_SIZE)
            r1 = int((ladder.gObj.y + ladder.gObj.height - 1) // self.TILE_SIZE)
            for row in range(r0, r1 + 1):
                for col in range(c0, c1 + 1):
                    ladder_tiles.add((col, row))
        return ladder_tiles

    def _build_dijkstra_map(self):
        self.dijkstra = None
        self.dijkstra_current_tile = -1.0
        if not self.level_data:
            return

        goal_positions = []
        if self.is_boss_level:
            boss = self._current_boss()
            if boss is not None:
                goal_positions.append(
                    (int(boss.gObj.x // self.TILE_SIZE), int(boss.gObj.y // self.TILE_SIZE))
                )
        if not goal_positions:
            goal_positions = [
                (int(goal.gObj.x // self.TILE_SIZE), int(goal.gObj.y // self.TILE_SIZE))
                for goal in self.level_data.goals
            ]
        if not goal_positions:
            return

        self.dijkstra = MegaDijkstraSolver(
            self.level_data.grid,
            self.level_data.rows,
            self.level_data.cols,
            self.ladder_tiles,
        )
        self.dijkstra.compute_map(goal_positions)

    def _dijkstra_norm_scale(self) -> float:
        return max(1.0, self.level_data.cols * 2.4, self.level_data.rows * 3.5)

    def _normalized_dijkstra(self, tile_x: int | None = None, tile_y: int | None = None) -> float:
        if not self.player or not self.level_data or self.dijkstra is None:
            return -1.0
        if tile_x is None:
            tile_x = int(self.player.gObj.x // self.TILE_SIZE)
        if tile_y is None:
            tile_y = int(self.player.gObj.y // self.TILE_SIZE)
        dist = self.dijkstra.get_dist(tile_x, tile_y)
        if dist < 0.0:
            return -1.0
        return float(np.clip(dist / self._dijkstra_norm_scale(), 0.0, 1.0))

    def _player_muzzle_world(self) -> Tuple[float, float]:
        if not self.player:
            return 0.0, 0.0
        p = self.player
        muzzle_x = p.gObj.x + (p.gObj.width + 2 if p.facing_right else -2)
        muzzle_y = p.gObj.y + p.gObj.height * 0.5
        return float(muzzle_x), float(muzzle_y)

    def get_jump_arc_debug_state(self) -> dict:
        if not self.player:
            return {}
        p = self.player
        origin_x = p.gObj.x + p.gObj.width * 0.5
        origin_y = p.gObj.y + p.gObj.height
        grounded = bool(p.on_ground)
        can_jump = bool(p.on_ground or p.on_ladder)
        preview_jump = grounded and can_jump
        if preview_jump:
            preview_vx = p.vx if abs(p.vx) > 10.0 else (self.run_speed * 0.55 if p.facing_right else -self.run_speed * 0.55)
            preview_vy = self.jump_velocity
        else:
            preview_vx = p.vx
            preview_vy = p.vy
        return {
            "x": float(origin_x),
            "y": float(origin_y),
            "vx": float(preview_vx),
            "vy": float(preview_vy),
            "grounded": grounded,
            "can_jump": can_jump,
            "preview_jump": preview_jump,
            "color": MEGAMAN_CYAN,
        }

    def _cast_debug_rays(self):
        self.last_rays = []
        if not self.player or not self.level_data:
            return

        start_x, start_y = self._player_muzzle_world()
        dx = 1.0 if self.player.facing_right else -1.0
        dy = 0.0
        end_x = start_x + dx * self.ray_max_dist
        end_y = start_y
        found = False

        dist = 0.0
        while dist <= self.ray_max_dist:
            wx = start_x + dx * dist
            wy = start_y
            tx = int(wx // self.TILE_SIZE)
            ty = int(wy // self.TILE_SIZE)

            if not (0 <= tx < self.level_data.cols and 0 <= ty < self.level_data.rows):
                end_x, end_y = wx, wy
                break

            tile = self.level_data.grid[ty][tx]
            enemy_hit = any(
                enemy.gObj.active and enemy.gObj.get_rect().collidepoint(wx, wy)
                for enemy in self.enemies
            )
            wall_hit = tile in (TILE_GROUND, TILE_PLATFORM, TILE_QBLOCK)
            spike_hit = tile == TILE_SPIKE
            pit_hit = tile == TILE_PIT and self._is_exposed_pit_tile(tx, ty)
            if enemy_hit or wall_hit or spike_hit or pit_hit:
                end_x, end_y = wx, wy
                found = True
                break

            dist += self.ray_step

        self.last_rays.append(((start_x, start_y), (end_x, end_y), found, 0.0))

    def _spawn_enemies(self):
        self.enemies = []
        placeholders = list(getattr(self.level_data, "enemies", []))
        for enemy in placeholders:
            tag = str(getattr(enemy, "spawn_tag", "enemy")).lower()
            x = float(enemy.gObj.x)
            y = float(enemy.gObj.y)
            if tag == "met":
                self.enemies.append(MetEnemy(x, y))
            elif tag == "bat":
                self.enemies.append(BatEnemy(x, y))
            elif tag == "boss":
                self.enemies.append(BossEnemy(x, y))
            else:
                self.enemies.append(MetEnemy(x, y))

    def _current_boss(self) -> BossEnemy | None:
        for enemy in self.enemies:
            if isinstance(enemy, BossEnemy) and enemy.gObj.active:
                return enemy
        return None

    def load_level(self):
        self.source_lines = self._load_source_lines()
        self.level_data = self.loader.load_level(self._level_config().get("file", "megaman/mm_stage1.txt"))
        level_yaml = os.path.splitext(self._level_file_path())[0] + ".yaml"
        self.level_meta = {}
        self.is_boss_level = False
        if os.path.isfile(level_yaml):
            try:
                with open(level_yaml, "r", encoding="utf-8") as fh:
                    sidecar = yaml.safe_load(fh) or {}
                self.level_meta = sidecar
                self.is_boss_level = bool(sidecar.get("boss_level", False))
                self._apply_physics_config(sidecar.get("physics", {}) or {})
            except Exception:
                self.level_meta = {}
                self.is_boss_level = False
                self._apply_physics_config(self.raw_config.get("physics", {}) or {})
        else:
            self._apply_physics_config(self.raw_config.get("physics", {}) or {})

        px, py = self.level_data.player_start
        if self.player is None:
            self.player = MegaManPlayer(GameObject(px, py, 20, 28, True))
            self.player.hp_max = self.hp_max
            self.player.hp = self.hp_max
        else:
            self.player.gObj.x = px
            self.player.gObj.y = py
            self.player.gObj.active = True
            self.player.gObj.width = 20
            self.player.gObj.height = 28
            self.player.vx = 0.0
            self.player.vy = 0.0
            self.player.on_ground = False
            self.player.facing_right = True
            self.player.hp_max = self.hp_max
            self.player.hp = self.player.hp_max
            self.player.i_frames = 0.0
            self.player.invincible_timer = 0.0
            self.player.star_timer = 0.0
            self.player.shot_cooldown = 0.0
            self.player.jump_cut = False
            self.player.powered_up = False
            self.player.fire_requested = False
            self.player.on_ladder = False
            self.player.ladder_x = 0.0
            self.player.can_jump = False
            self.player.grounded = False
            self.fire_buffer = 0.0

        self.projectiles = []
        self._spawn_enemies()
        self.is_boss_level = self.is_boss_level or self._current_boss() is not None
        self.level_data.enemies = self.enemies
        self.level_data.projectiles = self.projectiles
        self.spike_tiles = [
            tile for row in getattr(self.level_data, "tiles", [])
            for tile in row
            if tile is not None and getattr(getattr(tile, "gObj", tile), "type_id", None) == EntityType.SPIKE
        ]
        self.ladder_tiles = self._ladder_tiles()
        self._build_dijkstra_map()
        self._visit_map = np.zeros((self.level_data.rows, self.level_data.cols), dtype=np.int32)
        self.camera_x = 0.0
        self.camera_y = 0.0
        self.timer = self.time_limit
        self.alive = True
        self.reached_goal = False
        self.last_event = ""
        self.last_cause = ""
        self.enemies_killed_step = 0
        self.boss_damage_step = 0
        self.damage_taken_step = 0
        self.max_x_seen = max(0.0, self.player.gObj.x if self.player else 0.0)
        self.last_rays = []
        _goal_dx, _goal_dy, self._prev_goal_dist = self._goal_metrics()
        self.physics_manager.rebuild_dynamic_hashes(self)
        self._update_debug_caches()
        self._cast_debug_rays()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        won_previous = bool(self.reached_goal or self.last_event == "WIN")
        self.steps = 0
        self.score = 0
        self.reached_goal = False
        self.alive = True
        self.time_limit = 400.0
        self.timer = self.time_limit
        self.camera_x = 0.0
        self.camera_y = 0.0
        self.last_action = [0, 0, 0, 0]
        self.frame = 0
        self.progress_x_best = 0.0
        self.stall_timer = 0.0
        self.stall_windows_count = 0
        self.stalled_this_frame = False
        self.last_event = ""
        self.last_cause = ""
        self.enemies_killed_step = 0
        self.boss_damage_step = 0
        self.damage_taken_step = 0
        self.last_rays = []
        self.fire_buffer = 0.0

        self._select_level_for_reset(won_previous)
        self.load_level()

        return self._obs(), self._info()

    def _parse_action(self, action) -> Tuple[int, int, int, int]:
        try:
            move = int(action[0])
            if len(action) >= 4:
                climb = int(action[1])
                jump = int(action[2])
                fire = int(action[3])
            else:
                climb = 0
                jump = int(action[1])
                fire = int(action[2])
        except Exception:
            move = climb = jump = fire = 0
        return move, climb, jump, fire

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
        move, climb, jump, fire = self._parse_action(action)
        p = self.player
        kb_left, kb_right, kb_up, kb_down, kb_jump, kb_fire = self._keyboard_state()

        if kb_left:
            move = 1
        elif kb_right:
            move = 3
        if kb_jump:
            jump = 1
        if kb_fire:
            fire = 1

        ladder = self._nearest_ladder(p.gObj)
        climb_up = kb_up or climb == 1
        climb_down = kb_down or climb == 2
        climb_cmd = 1 if climb_up else (2 if climb_down else 0)
        self.last_action = [move, climb_cmd, jump, fire]
        if fire:
            self.fire_buffer = max(self.fire_buffer, self.fire_buffer_time)

        if ladder and (climb_up or climb_down):
            p.on_ladder = True
            p.ladder_x = ladder.gObj.x + ladder.gObj.width * 0.5 - p.gObj.width * 0.5
        elif p.on_ladder and ladder is None:
            p.on_ladder = False

        if p.on_ladder:
            p.gObj.x = p.ladder_x
            p.vx = 0.0
            if climb_up:
                p.vy = -self.climb_speed
            elif climb_down:
                p.vy = self.climb_speed
            else:
                p.vy = 0.0

            if jump:
                p.on_ladder = False
                p.vy = self.jump_velocity
            elif move in (1, 2, 3, 4) and not (climb_up or climb_down):
                p.on_ladder = False
            else:
                self._try_spawn_player_shot()
                return

        target_vx = 0.0
        if move == 1:
            target_vx = -self.walk_speed
            p.facing_right = False
        elif move == 2:
            target_vx = -self.run_speed
            p.facing_right = False
        elif move == 3:
            target_vx = self.walk_speed
            p.facing_right = True
        elif move == 4:
            target_vx = self.run_speed
            p.facing_right = True

        if target_vx != 0.0:
            if p.vx < target_vx:
                p.vx = min(target_vx, p.vx + self.accel * self.dt)
            elif p.vx > target_vx:
                p.vx = max(target_vx, p.vx - self.accel * self.dt)
        else:
            if p.vx > 0:
                p.vx = max(0.0, p.vx - self.friction * self.dt)
            elif p.vx < 0:
                p.vx = min(0.0, p.vx + self.friction * self.dt)

        if jump and p.on_ground:
            p.vy = self.jump_velocity
            p.on_ground = False
            p.jump_cut = False
        elif not jump and p.vy < 0 and not p.jump_cut:
            p.vy *= 0.55
            p.jump_cut = True

        self._try_spawn_player_shot()

    def _try_spawn_player_shot(self):
        p = self.player
        if not p or self.fire_buffer <= 0.0:
            return False

        self.projectiles = [proj for proj in self.projectiles if proj.gObj.active]
        if self.level_data is not None:
            self.level_data.projectiles = self.projectiles

        cooldown_ready = p.shot_cooldown <= 1e-4
        if not cooldown_ready:
            return False

        active_player_shots = sum(
            1 for proj in self.projectiles if proj.gObj.active and proj.owner == "player"
        )
        if active_player_shots >= self.max_player_shots:
            return False

        self.projectiles.append(MegaBusterProjectile.from_player(p))
        if self.level_data is not None:
            self.level_data.projectiles = self.projectiles
        p.shot_cooldown = self.shot_cooldown_time
        self.fire_buffer = 0.0
        return True

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

    def _support_tile_below(self, actor, epsilon: float = 3.0):
        rect = actor.gObj.get_rect()
        probe = pygame.Rect(rect.left + 2, rect.bottom, max(2, rect.width - 4), max(2, int(epsilon) + 1))
        best = None
        best_top = None
        for tile_rect, tile_type in self._nearby_tile_rects(actor.gObj):
            if tile_type == TILE_SPIKE:
                continue
            if tile_type == TILE_PLATFORM and rect.bottom > tile_rect.top + epsilon:
                continue
            if not probe.colliderect(tile_rect):
                continue
            if best is None or tile_rect.top < best_top:
                best = (tile_rect, tile_type)
                best_top = tile_rect.top
        return best

    def _stabilize_ground_contact(self, actor, epsilon: float = 3.0) -> bool:
        support = self._support_tile_below(actor, epsilon=epsilon)
        if support is None:
            actor.on_ground = False
            return False
        tile_rect, _tile_type = support
        actor.gObj.y = tile_rect.top - actor.gObj.height
        if actor.vy > 0.0:
            actor.vy = 0.0
        actor.on_ground = True
        return True

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
        self.damage_taken_step += max(0, int(amount))
        self.player.i_frames = self.i_frame_duration
        self.player.invincible_timer = self.player.i_frames
        if self.player.hp <= 0:
            self.lives = max(0, self.lives - 1)
            self.alive = False
            self.last_cause = cause

    def _update_debug_caches(self):
        if not self.player or not self.level_data:
            return

        p = self.player
        px = int(p.gObj.x // self.TILE_SIZE)
        py = int(p.gObj.y // self.TILE_SIZE)

        if self._visit_map is not None and 0 <= py < self.level_data.rows and 0 <= px < self.level_data.cols:
            self._visit_map[py, px] += 1

        dijkstra = np.zeros((self.obs_height, self.obs_width), dtype=np.float32)
        hazards = np.zeros((self.obs_height, self.obs_width), dtype=np.float32)
        player_cost = self.dijkstra.get_dist(px, py) if self.dijkstra else -1.0
        self.dijkstra_current_tile = player_cost
        max_cost = max(1.0, self.obs_width * 2.4, self.obs_height * 3.5)
        best_cost = player_cost if player_cost >= 0.0 else float("inf")
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
                elif tile == TILE_PIT and self._is_exposed_pit_tile(tx, ty):
                    hazards[ly, lx] = -0.5
                elif any(int(e.gObj.x // self.TILE_SIZE) == tx and int(e.gObj.y // self.TILE_SIZE) == ty for e in self.enemies if e.gObj.active):
                    hazards[ly, lx] = 1.0

                tile_cost = self.dijkstra.get_dist(tx, ty) if self.dijkstra else -1.0
                if player_cost >= 0.0 and tile_cost >= 0.0:
                    dijkstra[ly, lx] = np.clip((player_cost - tile_cost) / max_cost, -1.0, 1.0)
                else:
                    dijkstra[ly, lx] = 0.0

                if (abs(ddx) <= 1 and abs(ddy) <= 1 and (ddx != 0 or ddy != 0)
                        and tile_cost >= 0.0 and tile_cost < best_cost):
                    best_cost = tile_cost
                    best_step = (ddx, ddy)

        self._dijkstra_window_cache = dijkstra
        self._hazard_window_cache = hazards
        mag = max(1.0, float(np.hypot(best_step[0], best_step[1])))
        self._step_dx = best_step[0] / mag
        self._step_dy = best_step[1] / mag

    def _player_obs(self) -> np.ndarray:
        if not self.player:
            return np.zeros(15, dtype=np.float32)
        p = self.player
        active_player_shots = sum(
            1 for proj in self.projectiles if proj.gObj.active and proj.owner == "player"
        )
        climb_axis = 0.0
        if len(self.last_action) >= 2:
            if self.last_action[1] == 1:
                climb_axis = 1.0
            elif self.last_action[1] == 2:
                climb_axis = -1.0
        return np.array([
            np.clip(p.gObj.x / max(1.0, self.level_data.width), 0.0, 1.0),
            np.clip(p.gObj.y / max(1.0, self.level_data.height), 0.0, 1.0),
            np.clip(p.vx / max(1.0, self.run_speed), -1.0, 1.0),
            np.clip(p.vy / max(1.0, self.max_fall_speed), -1.0, 1.0),
            1.0 if p.on_ground else 0.0,
            np.clip(p.hp / float(max(1, p.hp_max)), 0.0, 1.0),
            1.0 if p.shot_cooldown <= 0.0 else 0.0,
            1.0 if p.i_frames > 0 else 0.0,
            1.0 if p.facing_right else 0.0,
            np.clip(p.shot_cooldown / max(0.001, self.shot_cooldown_time), 0.0, 1.0),
            np.clip(p.i_frames, 0.0, 1.0),
            1.0 if p.on_ladder else 0.0,
            climb_axis,
            1.0 if p.can_jump else 0.0,
            np.clip(active_player_shots / float(max(1, self.max_player_shots)), 0.0, 1.0),
        ], dtype=np.float32)

    def get_obs_value_labels(self):
        return (
            [
                "Px", "Py", "Vx", "Vy", "Grnd", "HP", "Fire",
                "Invinc", "FaceR", "FirCD", "InvTmr", "Laddr", "Climb",
                "CanJmp", "ShotQ",
            ],
            [
                "EnmDst", "GoalDst", "Timer", "GoalDY", "Dijkstra", "StepX", "StepY",
                "BossLvl", "BossHP",
            ],
        )

    def _tracking_obs(self) -> np.ndarray:
        if not self.player:
            return np.zeros(9, dtype=np.float32)

        p = self.player
        enemy_dist = 1.0
        active_enemies = [e for e in self.enemies if e.gObj.active]
        if active_enemies:
            enemy_dist = min(
                np.hypot(e.gObj.x - p.gObj.x, e.gObj.y - p.gObj.y) for e in active_enemies
            ) / max(1.0, np.hypot(self.level_data.width, self.level_data.height))

        goal_dx, goal_dy, goal_dist = self._goal_metrics()
        dijkstra_dist = self._normalized_dijkstra()
        boss = self._current_boss()
        boss_ratio = boss.hp / float(max(1, boss.hp_max)) if boss is not None else 0.0

        return np.array([
            np.clip(enemy_dist, 0.0, 1.0),
            goal_dist,
            np.clip(self.timer / 400.0, 0.0, 1.0),
            np.clip(goal_dy / max(1.0, self.level_data.height), -1.0, 1.0),
            max(0.0, dijkstra_dist),
            self._step_dx,
            self._step_dy,
            1.0 if self.is_boss_level else 0.0,
            np.clip(boss_ratio, 0.0, 1.0),
        ], dtype=np.float32)

    def _check_obs_sanity(self, obs: dict) -> None:
        self._obs_check_counter += 1
        if self._obs_check_counter % self._obs_check_interval != 0:
            return

        warnings_list = []
        grid_names = ["solid", "collectible", "hazard", "dijkstra"]
        grids = obs.get("grids")
        if grids is not None:
            for i, name in enumerate(grid_names):
                ch = grids[i]
                self._obs_stats[f"grid_{name}_mean"] = float(ch.mean())
                self._obs_stats[f"grid_{name}_std"] = float(ch.std())
                self._obs_stats[f"grid_{name}_min"] = float(ch.min())
                self._obs_stats[f"grid_{name}_max"] = float(ch.max())
                if i == 0 and float(ch.std()) < 1e-6:
                    warnings_list.append(f"Grid '{name}' DEAD")
                if i != 3 and float(ch.max()) > 1.01:
                    warnings_list.append(f"Grid '{name}' >1.0")

        scalars = obs.get("scalars")
        if scalars is not None:
            self._obs_stats["scalar_mean"] = float(scalars.mean())
            self._obs_stats["scalar_std"] = float(scalars.std())
            self._obs_stats["scalar_min"] = float(scalars.min())
            self._obs_stats["scalar_max"] = float(scalars.max())
            self._obs_stats["dijkstra_val"] = float(scalars[19])
            if float(scalars.std()) < 1e-8:
                warnings_list.append("Scalars DEAD")
            if abs(float(scalars.max())) > 100:
                warnings_list.append("Scalars unnormalized")

        self._obs_stats["obs_warnings"] = "|".join(warnings_list) if warnings_list else ""

    def _world_to_screen(self, gObj: GameObject):
        sx = gObj.x - self.camera_x
        sy = gObj.y - self.camera_y
        on_screen = sx < self.WIDTH and sy < self.HEIGHT and sx + gObj.width > 0 and sy + gObj.height > 0
        return sx, sy, on_screen

    def _update_player(self):
        self.physics_manager._update_player(self.dt, self)

    def _update_enemies(self):
        self.physics_manager._update_enemies(self.dt, self)

    def _update_projectiles(self):
        self.physics_manager._update_projectiles(self.dt, self)

    def _handle_combat(self):
        self.physics_manager.resolve_collisions(self)

    def _check_goal_and_oob(self):
        self.physics_manager._check_goal_and_oob(self)

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
        self.last_event = ""
        self.last_cause = ""
        self.enemies_killed_step = 0
        self.boss_damage_step = 0
        self.damage_taken_step = 0
        self.steps += 1
        self.frame += 1
        self.fire_buffer = max(0.0, self.fire_buffer - self.dt)
        self.time_limit = max(0.0, self.time_limit - self.dt)
        self.timer = self.time_limit
        if self.render_mode == "human":
            self.debug_manager.update_input()

        self._handle_input(action)
        self.physics_manager.update_system(self.dt, self)
        self.physics_manager.resolve_collisions(self)
        self._try_spawn_player_shot()
        self._update_camera()
        self._update_debug_caches()
        self._cast_debug_rays()

        if self.player:
            self.max_x_seen = max(self.max_x_seen, self.player.gObj.x)
            self.stalled_this_frame = False
            if abs(self.player.vx) < 8.0 and not self.player.on_ladder:
                prev_stall = self.stall_timer
                self.stall_timer += self.dt
                if prev_stall < 1.5 <= self.stall_timer:
                    self.stall_windows_count += 1
                    self.stalled_this_frame = True
            else:
                self.stall_timer = 0.0
        else:
            self.stalled_this_frame = False

        truncated = (self.max_steps is not None and self.steps >= self.max_steps) or self.time_limit <= 0.0
        terminated = self.reached_goal or not self.alive
        if terminated and not self.reached_goal:
            self.last_event = "DIED"
            if not self.last_cause:
                self.last_cause = "Enemy"
        reward = 0.01
        reward += (self.player.gObj.x if self.player else 0.0) / max(1.0, self.level_data.width) * 0.02
        if self.reached_goal:
            reward += 10.0
        if terminated and not self.reached_goal:
            reward -= 5.0

        obs = self._obs()
        self._check_obs_sanity(obs)
        return obs, reward, terminated, truncated, self._info()

    def _grid_obs_window(self):
        solid = np.zeros((self.obs_height, self.obs_width), dtype=np.float32)
        collect = np.zeros((self.obs_height, self.obs_width), dtype=np.float32)
        hazard = np.zeros((self.obs_height, self.obs_width), dtype=np.float32)
        dijkstra = np.zeros((self.obs_height, self.obs_width), dtype=np.float32)

        if not self.player or not self.level_data:
            return solid, collect, hazard, dijkstra

        if self._hazard_window_cache is not None:
            hazard = self._hazard_window_cache.copy()
        if self._dijkstra_window_cache is not None:
            dijkstra = self._dijkstra_window_cache.copy()

        px = int(self.player.gObj.x // self.TILE_SIZE)
        py = int(self.player.gObj.y // self.TILE_SIZE)
        active_coins = {
            (int(c.gObj.x // self.TILE_SIZE), int(c.gObj.y // self.TILE_SIZE))
            for c in self.level_data.coins
            if c.gObj.active and not getattr(c, "collected", False)
        }
        active_goals = {
            (int(g.gObj.x // self.TILE_SIZE), int(g.gObj.y // self.TILE_SIZE))
            for g in self.level_data.goals
            if g.gObj.active
        }

        for ddy in range(-self.obs_pad_y, self.obs_pad_y + 1):
            for ddx in range(-self.obs_pad_x, self.obs_pad_x + 1):
                tx = px + ddx
                ty = py + ddy
                ly = ddy + self.obs_pad_y
                lx = ddx + self.obs_pad_x

                if not (0 <= tx < self.level_data.cols and 0 <= ty < self.level_data.rows):
                    continue

                tile = self.level_data.grid[ty][tx]
                if tile in (TILE_GROUND, TILE_PLATFORM, TILE_QBLOCK) or (tx, ty) in self.ladder_tiles:
                    solid[ly, lx] = 1.0
                elif tile == TILE_PIT and self._is_exposed_pit_tile(tx, ty):
                    solid[ly, lx] = -0.5

                if (tx, ty) in active_goals or tile == TILE_GOAL:
                    collect[ly, lx] = max(collect[ly, lx], 1.0)
                if (tx, ty) in self.ladder_tiles:
                    collect[ly, lx] = max(collect[ly, lx], 0.69)
                if (tx, ty) in active_coins:
                    collect[ly, lx] = max(collect[ly, lx], 0.35)

        self._solid_window_cache = solid
        return solid, collect, hazard, dijkstra

    def _dict_obs(self) -> dict:
        solid, collect, hazard, dijkstra = self._grid_obs_window()
        return {
            "grids": np.stack([solid, collect, hazard, dijkstra], axis=0).astype(np.float32),
            "scalars": np.concatenate([self._player_obs(), self._tracking_obs()]).astype(np.float32),
        }

    def _obs(self):
        return self._dict_obs()

    def _info(self):
        goal_dx, goal_dy, goal_dist = self._goal_metrics()
        dijkstra_dist = self._normalized_dijkstra()
        progress = self._prev_goal_dist - goal_dist
        self._prev_goal_dist = goal_dist
        boss = self._current_boss()
        return {
            "hp": self.player.hp if self.player else 0,
            "lives": self.lives,
            "score": self.score,
            "level": self.world,
            "won": self.reached_goal,
            "event": self.last_event,
            "cause": self.last_cause,
            "terminated": self.reached_goal or not self.alive,
            "action_name": self.action_to_str(self.last_action),
            "x_position": self.player.gObj.x if self.player else 0.0,
            "y_position": self.player.gObj.y if self.player else 0.0,
            "velocity_x": self.player.vx if self.player else 0.0,
            "velocity_y": self.player.vy if self.player else 0.0,
            "goal_dx": goal_dx,
            "goal_dy": goal_dy,
            "goal_dist": goal_dist,
            "dijkstra_dist": max(0.0, dijkstra_dist),
            "progress": progress,
            "step_dx": self._step_dx,
            "step_dy": self._step_dy,
            "max_x_seen": self.max_x_seen,
            "coins_collected": self.coins_total,
            "enemies_killed_step": self.enemies_killed_step,
            "boss_damage_step": self.boss_damage_step,
            "damage_taken_step": self.damage_taken_step,
            "stalled": self.stall_timer >= 1.5,
            "stalled_this_frame": self.stalled_this_frame,
            "on_ladder": bool(self.player and self.player.on_ladder),
            "on_ground": bool(self.player and self.player.on_ground),
            "boss_level": self.is_boss_level,
            "boss_active": boss is not None,
            "boss_hp_ratio": boss.hp / float(max(1, boss.hp_max)) if boss is not None else 0.0,
            "curriculum_level_idx": self.current_index_world,
            "curriculum_win_rate": self._curriculum_win_rate(),
            "curriculum_max_unlocked": self._max_unlocked_index,
            **self._obs_stats,
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
                elif tile == TILE_PIT and self._is_exposed_pit_tile(col, row):
                    pygame.draw.line(self._surf, PIT_ALERT, (sx + 4, sy + 4), (sx + self.TILE_SIZE - 4, sy + self.TILE_SIZE - 4), 2)
                    pygame.draw.line(self._surf, PIT_ALERT, (sx + self.TILE_SIZE - 4, sy + 4), (sx + 4, sy + self.TILE_SIZE - 4), 2)
                    pygame.draw.line(self._surf, PIT_ALERT, (sx + 3, sy + self.TILE_SIZE - 3), (sx + self.TILE_SIZE - 3, sy + self.TILE_SIZE - 3), 1)
                elif tile == TILE_SPIKE:
                    pygame.draw.polygon(
                        self._surf,
                        HAZARD_RED,
                        [(sx, sy + self.TILE_SIZE), (sx + self.TILE_SIZE * 0.5, sy), (sx + self.TILE_SIZE, sy + self.TILE_SIZE)],
                    )
        for goal in self.level_data.goals:
            sx = goal.gObj.x - self.camera_x
            sy = goal.gObj.y - self.camera_y
            if sx + goal.gObj.width < 0 or sy + goal.gObj.height < 0 or sx > self.WIDTH or sy > self.HEIGHT:
                continue
            door_h = self.TILE_SIZE * 2
            pygame.draw.rect(self._surf, (32, 88, 44), (sx + 3, sy - self.TILE_SIZE, self.TILE_SIZE - 6, door_h), border_radius=6)
            pygame.draw.rect(self._surf, GOAL_GREEN, (sx + 5, sy - self.TILE_SIZE + 4, self.TILE_SIZE - 10, door_h - 8), border_radius=5)
            pygame.draw.circle(self._surf, (220, 255, 220), (int(sx + self.TILE_SIZE - 9), int(sy + 2)), 3)

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
        hud = pygame.Surface((338, 92), pygame.SRCALPHA)
        hud.fill((0, 0, 0, 170))
        self._surf.blit(hud, (10, 10))
        top_text = self.small_font.render(
            f"Lives {self.lives}   Score {self.score}   Time {int(self.time_limit)}",
            True,
            (255, 255, 255),
        )
        self._surf.blit(top_text, (18, 14))
        hp_text = self.small_font.render(f"HP {self.player.hp}/{self.player.hp_max}", True, MEGAMAN_CYAN)
        self._surf.blit(hp_text, (18, 32))
        bar_w = 170
        bar_x = 18
        bar_y = 48
        pygame.draw.rect(self._surf, (30, 30, 30), (bar_x, bar_y, bar_w, 8), border_radius=4)
        pygame.draw.rect(
            self._surf,
            MEGAMAN_CYAN,
            (bar_x, bar_y, int(bar_w * (self.player.hp / float(self.player.hp_max))), 8),
            border_radius=4,
        )
        shot_text = self.small_font.render("Buster", True, SHOT_ORANGE)
        self._surf.blit(shot_text, (205, 32))
        cd_w = 92
        cd_x = 205
        cd_y = 48
        cooldown_ratio = 1.0 - np.clip(self.player.shot_cooldown / max(0.001, self.shot_cooldown_time), 0.0, 1.0)
        pygame.draw.rect(self._surf, (30, 30, 30), (cd_x, cd_y, cd_w, 8), border_radius=4)
        pygame.draw.rect(self._surf, SHOT_ORANGE, (cd_x, cd_y, int(cd_w * cooldown_ratio), 8), border_radius=4)
        active_player_shots = sum(
            1 for proj in self.projectiles if proj.gObj.active and proj.owner == "player"
        )
        pip_x = 205
        pip_y = 64
        for i in range(self.max_player_shots):
            filled = i < active_player_shots
            color = SHOT_ORANGE if filled else (60, 52, 44)
            pygame.draw.circle(self._surf, color, (pip_x + i * 16, pip_y), 5)
            pygame.draw.circle(self._surf, (255, 228, 190), (pip_x + i * 16, pip_y), 5, 1)

        boss = self._current_boss()
        if self.is_boss_level and boss is not None:
            boss_w = 320
            boss_hud = pygame.Surface((boss_w, 42), pygame.SRCALPHA)
            boss_hud.fill((0, 0, 0, 170))
            hud_left = 10
            hud_w = 338
            safe_left = hud_left + hud_w + 24
            safe_right = self.WIDTH - 24
            open_w = safe_right - safe_left
            if open_w >= boss_w:
                boss_x = safe_left + (open_w - boss_w) // 2
            else:
                boss_x = max(10, (self.WIDTH - boss_w) // 2)
            boss_y = 12
            self._surf.blit(boss_hud, (boss_x, boss_y))
            title = self.small_font.render("BOSS", True, (255, 215, 180))
            self._surf.blit(title, (boss_x + 12, boss_y + 9))
            pygame.draw.rect(self._surf, (36, 24, 24), (boss_x + 74, boss_y + 15, 228, 10), border_radius=5)
            pygame.draw.rect(
                self._surf,
                HAZARD_RED,
                (boss_x + 74, boss_y + 15, int(228 * (boss.hp / float(max(1, boss.hp_max)))), 10),
                border_radius=5,
            )

    def render(self, surface: pygame.Surface | None = None, blit_only: bool = False):
        self._draw_background()
        self._draw_world()
        self._draw_entities()
        self._draw_player()
        self._draw_hud()
        if self.render_mode == "human" and getattr(self.debug_manager, "show_sensors", False):
            ray_surf = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            for start, end, found, rtype in self.last_rays:
                color = (*MEGAMAN_CYAN, 190)
                s_cam = (start[0] - self.camera_x, start[1] - self.camera_y)
                e_cam = (end[0] - self.camera_x, end[1] - self.camera_y)
                pygame.draw.line(ray_surf, color, s_cam, e_cam, 2)
                if found:
                    pygame.draw.circle(ray_surf, (255, 255, 255, 220), (int(e_cam[0]), int(e_cam[1])), 3)
            mx, my = self._player_muzzle_world()
            origin = (int(mx - self.camera_x), int(my - self.camera_y))
            pygame.draw.circle(ray_surf, MEGAMAN_CYAN, origin, 4)
            pygame.draw.circle(ray_surf, (255, 255, 255), origin, 4, 1)
            self._surf.blit(ray_surf, (0, 0))
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
            move, climb, jump, fire = self._parse_action(action)
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
        if climb == 1:
            out += "+UP"
        elif climb == 2:
            out += "+DOWN"
        if jump:
            out += "+JUMP"
        if fire:
            out += "+FIRE"
        return out

    def close(self):
        if self.render_mode == "human":
            pygame.quit()


MegaManCore = MegamanCore
