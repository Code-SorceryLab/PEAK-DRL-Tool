from dataclasses import dataclass
import importlib.util
import os
import sys
from typing import List, Any, Dict
import pygame
import math
from .EntityType import EntityType
from ..Parameters.Sonic_Movement_parameters import *
from ..Parameters.Sonic_Jump_parameters import *
from ..Parameters.Sonic_Map_parameters import TILE_SIZE
from ..Parameters.Map_parameters import TILE_QBLOCK
from .SpatialHash import SpatialHash
from ..Objects.Coin import Coin
from ..Objects.Mushroom import Mushroom
from ..Objects.LifeUp import LifeUp
from ..Objects.StarPowerUp import StarPowerUp
from ..Objects.FireFlower import FireFlower
from ..Objects.GameObject import GameObject
from ..Objects.Ring import Ring
from ..Objects.Spring import Spring

# ── Import Sonic Slope Functions ──
from ..System.SlopePhysics import resolve_slopes, apply_slope_speed

@dataclass
class PhysicsContext:
    """
    Carries all physics constants for a frame.
    """
    # Movement
    RUN_ACCEL: float = RUN_ACCEL
    WALK_ACCEL: float = WALK_ACCEL
    MAX_WALK_SPEED: float = MAX_WALK_SPEED
    MAX_RUN_SPEED: float = MAX_RUN_SPEED
    GROUND_FRICTION: float = GROUND_FRICTION
    AIR_FRICTION: float = AIR_FRICTION
    AIR_CONTROL: float = AIR_CONTROL
    SKID_DECEL: float = SKID_DECEL
    GRAVITY: float = GRAVITY
    FAST_FALL_GRAV: float = FAST_FALL_GRAV
    MAX_FALL_SPEED: float = MAX_FALL_SPEED

    # Jump
    JUMP_VEL_MIN: float = JUMP_VEL_MIN
    JUMP_VEL_MAX: float = JUMP_VEL_MAX
    JUMP_HOLD_FRAMES: int = JUMP_HOLD_FRAMES
    SPEED_JUMP_BONUS: float = SPEED_JUMP_BONUS
    COYOTE_FRAMES: int = COYOTE_FRAMES
    JUMP_BUFFER_FRAMES: int = JUMP_BUFFER_FRAMES

class SonicPhysicsManager:
    """
    Manages the physics simulation for the game, including movement updates,
    collision detection, and resolution for all entities.
    """
    def __init__(self, config_file: str = None, speed_mult: float = 1.0):
        self.context = PhysicsContext()
        self.speed_mult = speed_mult

        self.hazard_hash     = SpatialHash(64)
        self.collectible_hash = SpatialHash(64)
        self.platform_hash = SpatialHash(64)

        if config_file:
            self.load_config(config_file)

        self._apply_multiplier(speed_mult)

    def reset_to_defaults(self):
        self.context = PhysicsContext()
        self._apply_multiplier(self.speed_mult)

    def apply_config_dict(self, config: Dict[str, Any]):
        phys = config.get("physics") or {}
        if "gravity" in phys: self.context.GRAVITY = float(phys["gravity"])
        if "fast_fall_gravity" in phys: self.context.FAST_FALL_GRAV = float(phys["fast_fall_gravity"])

        fric = phys.get("friction") or {}
        if "ground" in fric: self.context.GROUND_FRICTION = float(fric["ground"])
        if "air" in fric: self.context.AIR_FRICTION = float(fric["air"])

        player = config.get("player") or {}
        move = player.get("movement") or {}
        if "max_run_speed" in move: self.context.MAX_RUN_SPEED = float(move["max_run_speed"])
        if "run_accel" in move: self.context.RUN_ACCEL = float(move["run_accel"])
        if "max_walk_speed" in move: self.context.MAX_WALK_SPEED = float(move["max_walk_speed"])
        if "walk_accel" in move: self.context.WALK_ACCEL = float(move["walk_accel"])
        if "air_control" in move: self.context.AIR_CONTROL = float(move["air_control"])

        jump = player.get("jump") or {}
        if "max_velocity" in jump: self.context.JUMP_VEL_MAX = float(jump["max_velocity"])
        if "min_velocity" in jump: self.context.JUMP_VEL_MIN = float(jump["min_velocity"])
        if "hold_frames" in jump: self.context.JUMP_HOLD_FRAMES = int(jump["hold_frames"])
        if "coyote_frames" in jump: self.context.COYOTE_FRAMES = int(jump["coyote_frames"])
        if "buffer_frames" in jump: self.context.JUMP_BUFFER_FRAMES = int(jump["buffer_frames"])

        self._apply_multiplier(self.speed_mult)

    def _apply_multiplier(self, mult: float):
        self.context.RUN_ACCEL *= mult
        self.context.WALK_ACCEL *= mult
        self.context.MAX_WALK_SPEED *= mult
        self.context.MAX_RUN_SPEED *= mult

    def rebuild_dynamic_hashes(
        self,
        level_data,
        cached_spikes=None,
        *,
        rings=None,
        lost_rings=None,
        springs=None,
    ):
        self.hazard_hash.clear()
        self.collectible_hash.clear()
        self.platform_hash.clear()
        for enemy in level_data.enemies:
            if enemy.gObj.active: self.hazard_hash.insert(enemy)
        if cached_spikes:
            for spike in cached_spikes:
                self.hazard_hash.insert(spike)
        for plat in level_data.moving_platforms:
            if plat.gObj.active:
                self.platform_hash.insert(plat)

        active_rings = rings if rings is not None else level_data.coins
        for ring in active_rings:
            if ring.gObj.active and not getattr(ring, 'collected', False):
                self.collectible_hash.insert(ring)

        for ring in (lost_rings or []):
            if ring.gObj.active and getattr(ring, "can_collect", True):
                self.collectible_hash.insert(ring)

        for spring in (springs if springs is not None else getattr(level_data, "springs", [])):
            if spring.gObj.active:
                self.collectible_hash.insert(spring)

        for pup in level_data.powerups:
            if pup.gObj.active: self.collectible_hash.insert(pup)
        for goal in level_data.goals:
            self.collectible_hash.insert(goal)

    def get_dynamic_hash(self) -> SpatialHash:
        return self.hazard_hash

    def update_system(self, dt: float, core):
        ctx = self.context
        player = core.player
        level_data = core.level_data

        if player:
            # Prevent Spin Dash from launching the player upwards
            if hasattr(player, 'state') and getattr(player.state, 'name', '') == "SPIN_DASH":
                if hasattr(player, 'jump_buffer'):
                    player.jump_buffer = 0
                if player.vy < 0:
                    player.vy = 0

            predicted_dist = max(abs(player.vx), abs(player.vy)) * dt
            step_count = 1
            if predicted_dist > TILE_SIZE * 0.5:
                step_count = int(math.ceil(predicted_dist / (TILE_SIZE * 0.5)))
            step_dt = dt / step_count

            for _ in range(step_count):
                pre_y = player.gObj.y          
                player.update(step_dt, ctx)    
                post_y = player.gObj.y         

                # Pass 1: X only 
                player.gObj.y = pre_y
                self._resolve_player_world_x(core, level_data)

                # Pass 2: Y only 
                player.gObj.y = post_y
                self._resolve_player_world_y(core, level_data)

        self.update_list(dt, level_data.enemies)
        self.update_list(dt, level_data.coins)
        self.update_list(dt, level_data.powerups)
        self.update_list(dt, level_data.moving_platforms)
        self.update_list(dt, level_data.projectiles)

        self._check_oob(core, level_data)

    def update_list(self, dt: float, objects: List[Any]):
        ctx = self.context
        for obj in objects:
            if hasattr(obj, 'gObj') and not obj.gObj.active: continue
            if hasattr(obj, "update"):
                try:
                    obj.update(dt, ctx)
                except TypeError:
                    obj.update(dt)

    def _check_oob(self, core, level_data):
        player = core.player
        if player and player.gObj.active and core.alive:
            PIT_GRACE_PX  = 48          
            OOB_MARGIN_PX = level_data.tile_size if hasattr(level_data, 'tile_size') else 32

            fell_out    = player.gObj.y > level_data.height + PIT_GRACE_PX
            left_out    = player.gObj.x + player.gObj.width < -OOB_MARGIN_PX
            right_out   = player.gObj.x > level_data.width  + OOB_MARGIN_PX

            if fell_out or left_out or right_out:
                cause = "Pit" if fell_out else "OOB"
                core._handle_death(cause)
                return   

        kill_y = level_data.height + 64
        for enemy in level_data.enemies:
            if enemy.gObj.active and enemy.gObj.y > kill_y:
                enemy.gObj.active = False

    def resolve_collisions(self, core):
        level_data = core.level_data
        player = core.player

        player_was_falling = False
        if player:
            player_was_falling = player.vy > 0

        self._resolve_player_world(core, level_data)
        self._resolve_player_moving_platforms(core, level_data)
        self._resolve_enemy_world(core, level_data)
        self._resolve_powerup_world(core, level_data)
        self._resolve_projectile_world(core, level_data)

        self._resolve_dynamic_interactions(core, player_was_falling)

        # ── RESOLVE SLOPES FOR PLAYER & ENEMIES ──
        if player and core.alive:
            resolve_slopes(player, level_data.slope_tiles, level_data)
            apply_slope_speed(player, level_data.slope_tiles, core.dt)
            
        for enemy in level_data.enemies:
            if enemy.gObj.active:
                resolve_slopes(enemy, level_data.slope_tiles, level_data)


    def _resolve_player_moving_platforms(self, core, level_data):
        player = core.player
        if not player or not level_data.moving_platforms:
            return

        prect = player.gObj.get_rect()

        for plat in level_data.moving_platforms:
            if not plat.gObj.active: continue
            trect = plat.gObj.get_rect()
            if not prect.colliderect(trect): continue

            ox = min(prect.right  - trect.left, trect.right  - prect.left)
            oy = min(prect.bottom - trect.top,  trect.bottom - prect.top)

            if ox < oy:
                if player.vx > 0:
                    player.gObj.x = trect.left - player.gObj.width
                elif player.vx < 0:
                    player.gObj.x = trect.right
                else:
                    if prect.centerx <= trect.centerx:
                        player.gObj.x = trect.left - player.gObj.width
                    else:
                        player.gObj.x = trect.right
                player.vx = 0

            elif player.vy >= 0:
                player.gObj.x  += plat.delta_x
                player.gObj.y   = trect.top - player.gObj.height
                player.vy       = 0
                player.on_ground = True
                player.jump_hold = 0
                player._on_moving_platform = True

            else:
                player.gObj.y = trect.bottom
                player.vy     = 0

            prect = player.gObj.get_rect()

    def _resolve_enemy_world(self, core, level_data):
        for enemy in level_data.enemies:
            if not enemy.gObj.active: continue
            nearby = self._get_tile_rects_near(level_data, enemy.gObj)
            self._solve_aabb_collision(enemy, nearby, bounce_x=True)

    def _resolve_powerup_world(self, core, level_data):
        for pup in level_data.powerups:
            if not pup.gObj.active: continue
            nearby = self._get_tile_rects_near(level_data, pup.gObj)
            self._solve_aabb_collision(pup, nearby, bounce_x=True)

    def _resolve_projectile_world(self, core, level_data):
        for proj in level_data.projectiles:
            if not proj.gObj.active: continue
            nearby = self._get_tile_rects_near(level_data, proj.gObj)
            self._solve_aabb_collision(proj, nearby, bounce_x=False)

    def _solve_aabb_collision(self, entity, nearby_tiles, bounce_x=False):
        # Resolve Y
        ent_rect = entity.gObj.get_rect()
        for (_, _, tile_rect, _) in nearby_tiles:
            if ent_rect.colliderect(tile_rect):
                ox = min(ent_rect.right - tile_rect.left, tile_rect.right - ent_rect.left)
                oy = min(ent_rect.bottom - tile_rect.top, tile_rect.bottom - ent_rect.top)

                if ox < oy: continue

                if entity.vy >= 0: 
                    if ent_rect.bottom >= tile_rect.top:
                        entity.gObj.y = tile_rect.top - entity.gObj.height
                        entity.vy = 0
                        if hasattr(entity, 'on_ground'): entity.on_ground = True
                elif entity.vy < 0: 
                     if ent_rect.top <= tile_rect.bottom:
                        entity.gObj.y = tile_rect.bottom
                        entity.vy = 0
                ent_rect = entity.gObj.get_rect() 

        # Resolve X (With Lip-Snag Fix)
        step_tolerance = min(8, entity.gObj.height - 4)
        entity.gObj.height -= step_tolerance
        ent_rect = entity.gObj.get_rect()
        
        for (_, _, tile_rect, _) in nearby_tiles:
             if ent_rect.colliderect(tile_rect):
                ox = min(ent_rect.right - tile_rect.left, tile_rect.right - ent_rect.left)
                oy = min(ent_rect.bottom - tile_rect.top, tile_rect.bottom - ent_rect.top)

                if ox < oy:
                    if entity.vx > 0: 
                        entity.gObj.x = tile_rect.left - entity.gObj.width
                        if bounce_x: entity.vx *= -1
                        else: entity.vx = 0
                    elif entity.vx < 0: 
                        entity.gObj.x = tile_rect.right
                        if bounce_x: entity.vx *= -1
                        else: entity.vx = 0
                    ent_rect = entity.gObj.get_rect()
                    
        entity.gObj.height += step_tolerance

    def _resolve_player_world(self, core, level_data):
        self._resolve_player_world_x(core, level_data)
        self._resolve_player_world_y(core, level_data)

    def _resolve_player_world_x(self, core, level_data):
        player = core.player
        if not player: return

        # ── Lip-Snag Fix ──
        # Shrink the hitbox bottom temporarily for wall detection (step-up tolerance)
        step_tolerance = min(16, player.gObj.height - 4)
        player.gObj.height -= step_tolerance

        nearby = self._get_tile_rects_near(level_data, player.gObj)
        rect   = player.gObj.get_rect()

        for (row, col, trect, tile_type) in nearby:
            if tile_type == EntityType.SPIKE:
                continue  
            if not rect.colliderect(trect):
                continue

            if player.vx > 0:
                player.gObj.x = trect.left - player.gObj.width
            elif player.vx < 0:
                player.gObj.x = trect.right
            else:
                if rect.centerx <= trect.centerx:
                    player.gObj.x = trect.left - player.gObj.width
                else:
                    player.gObj.x = trect.right
            player.vx = 0
            rect = player.gObj.get_rect()

        # Restore the actual height so Y-axis floor snapping works normally
        player.gObj.height += step_tolerance

    def _resolve_player_world_y(self, core, level_data):
        player = core.player
        if not player: return

        nearby = self._get_tile_rects_near(level_data, player.gObj)
        rect   = player.gObj.get_rect()

        for (row, col, trect, tile_type) in nearby:
            if not rect.colliderect(trect):
                continue

            if tile_type == EntityType.SPIKE:
                # SPIKE COLLISION
                if getattr(player, 'invincible_timer', 0) > 0 or getattr(player, 'star_timer', 0) > 0:
                    continue # Immune
                    
                rings_before = getattr(player, 'rings', 0)
                dies = player.take_hit() if hasattr(player, 'take_hit') else True
                
                if dies:
                    core._handle_death("Spike")
                elif rings_before > 0 and hasattr(core, '_scatter_rings'):
                    core._scatter_rings(player, rings_before)
                return

            if player.vy >= 0:
                player.gObj.y    = trect.top - player.gObj.height
                player.vy        = 0
                player.on_ground = True
                player.jump_hold = 0
            else:
                player.gObj.y = trect.bottom
                player.vy     = 0
                if tile_type == TILE_QBLOCK:
                    self._hit_qblock(core, col, row)

            rect = player.gObj.get_rect()

    def _resolve_dynamic_interactions(self, core, player_was_falling):
        player = core.player
        if not player: return

        nearby_hazards = self.hazard_hash.query(player)
        for obj in nearby_hazards:
            t_type = obj.gObj.type_id if hasattr(obj.gObj, 'type_id') else EntityType.NONE
            if t_type == EntityType.SPIKE: continue
            if player.gObj.collides_with(obj.gObj):
                self._dispatch_collision(core, player, obj, player_was_falling)

        nearby_items = self.collectible_hash.query(player)
        for obj in nearby_items:
            if player.gObj.collides_with(obj.gObj):
                self._dispatch_collision(core, player, obj, player_was_falling)
            else:
                obj_tid = obj.gObj.type_id if hasattr(obj.gObj, 'type_id') else EntityType.NONE
                if obj_tid == EntityType.GOAL:
                    probe = pygame.Rect(
                        player.gObj.x, player.gObj.y,
                        player.gObj.width, player.gObj.height + 2
                    )
                    if probe.colliderect(obj.gObj.get_rect()):
                        self._dispatch_collision(core, player, obj, player_was_falling)

        for enemy in core.level_data.enemies:
            if not enemy.gObj.active: continue
            nearby_enemies = self.hazard_hash.query(enemy)
            for other in nearby_enemies:
                if other is not enemy and other.gObj.active:
                    if isinstance(other, type(enemy)):
                        if enemy.gObj.collides_with(other.gObj):
                            self._dispatch_collision(core, enemy, other)

        for proj in core.level_data.projectiles:
            if not proj.gObj.active: continue
            nearby_enemies = self.hazard_hash.query_rect(
                proj.gObj.x, proj.gObj.y, proj.gObj.width, proj.gObj.height
            )
            for enemy in nearby_enemies:
                if not enemy.gObj.active: continue
                t_type = enemy.gObj.type_id if hasattr(enemy.gObj, 'type_id') else EntityType.NONE
                if t_type != EntityType.ENEMY: continue   
                if proj.gObj.collides_with(enemy.gObj):
                    self._handle_projectile_enemy(core, proj, enemy)
                    break      

    def _dispatch_collision(self, core, source, target, player_was_falling=False):
        s_type = source.gObj.type_id if hasattr(source.gObj, 'type_id') else EntityType.NONE
        t_type = target.gObj.type_id if hasattr(target.gObj, 'type_id') else EntityType.NONE

        if s_type == EntityType.NONE: s_type = self._infer_type(source)
        if t_type == EntityType.NONE: t_type = self._infer_type(target)

        match s_type:
            case EntityType.PLAYER:
                if t_type == EntityType.ENEMY:
                    self._handle_player_enemy(core, source, target, player_was_falling)
                elif t_type == EntityType.COIN:
                    self._handle_player_coin(core, source, target)
                elif t_type == EntityType.POWERUP:
                    self._handle_player_powerup(core, source, target)
                elif t_type == EntityType.GOAL:
                    goal_is_real = target.gObj.x > TILE_SIZE
                    if not core.reached_goal and goal_is_real:
                        core.score += int(core.timer)
                        core.reached_goal = True
                        core.complete_level()

            case EntityType.ENEMY:
                if t_type == EntityType.ENEMY:
                    self._handle_enemy_enemy(core, source, target)

    def _infer_type(self, obj):
        name = obj.__class__.__name__
        if name == "Player" or name == "SonicPlayer": return EntityType.PLAYER
        if name == "Enemy" or name == "Badnik": return EntityType.ENEMY
        if name == "Coin" or name == "Ring": return EntityType.COIN
        if name == "Powerup": return EntityType.POWERUP
        if name == "Goal": return EntityType.GOAL
        return EntityType.NONE

    def _handle_player_enemy(self, core, player, enemy, player_was_falling):
        if not enemy.gObj.active: return

        # 1. Star Power Invincibility
        if getattr(player, 'star_timer', 0.0) > 0:
            if hasattr(enemy, 'destroy'): enemy.destroy()
            else: enemy.gObj.active = False
            core.score += 100
            if hasattr(core, 'kills_step'): core.kills_step += 1
            if hasattr(core, 'badniks_destroyed'): core.badniks_destroyed += 1
            return

        # 2. Sonic Ball Attack (Jumping, Rolling, Spindash)
        if getattr(player, 'is_ball', False):
            if hasattr(enemy, 'destroy'): enemy.destroy()
            else: enemy.gObj.active = False
            
            if hasattr(player, 'bounce_off_enemy'):
                player.bounce_off_enemy()
                
            core.score += 100
            if hasattr(core, 'kills_step'): core.kills_step += 1
            if hasattr(core, 'badniks_destroyed'): core.badniks_destroyed += 1
            return

        # 3. Take Damage
        if getattr(player, 'invincible_timer', 0.0) > 0:
            return # Already invincible

        rings_before = getattr(player, 'rings', 0)
        dies = player.take_hit() if hasattr(player, 'take_hit') else True

        if dies:
            core._handle_death("Enemy")
        elif rings_before > 0 and hasattr(core, '_scatter_rings'):
            core._scatter_rings(player, rings_before)

    def _handle_player_coin(self, core, player, coin):
        # Ignore lost rings that are still in grace period
        if hasattr(coin, 'can_collect') and not coin.can_collect:
            return

        if not getattr(coin, 'collected', False):
            coin.gObj.active = False
            if hasattr(coin, 'collected'):
                coin.collected = True
            core.score += 10
            core.coins_step += 1
            core.coins_total += 1
            if hasattr(player, 'rings'):
                player.rings += 1
            if hasattr(core, 'ring_total'):
                core.ring_total += 1

    def _handle_projectile_enemy(self, core, proj, enemy):
        enemy.gObj.active = False
        proj.gObj.active  = False
        core.score += 100
        if hasattr(core, 'kills_step'): core.kills_step += 1

    def _handle_player_powerup(self, core, player, powerup):
        powerup.gObj.active = False
        core.powerups_step += 1

        kind = getattr(powerup, 'kind', 'star')
        
        if kind == "mushroom" or kind == "flower":
            if hasattr(player, 'shield'):
                player.shield = True
            core.score += 50
        elif kind == "life":
            core.lives += 1
            core.score += 200
        else: # Star
            if hasattr(player, 'star_timer'):
                player.star_timer = 10.0
            core.score += 100

    def _handle_enemy_enemy(self, core, e1, e2):
        enemy = e1
        other = e2
        rect = enemy.gObj.get_rect()
        other_rect = other.gObj.get_rect()

        obj_x = min(rect.right - other_rect.left, other_rect.right - rect.left)
        obj_y = min(rect.bottom - other_rect.top, other_rect.bottom - rect.top)

        if obj_x < obj_y:
            if rect.centerx < other_rect.centerx:
                enemy.gObj.x = other_rect.left - enemy.gObj.width
            else:
                enemy.gObj.x = other_rect.right
            enemy.vx *= -1.0
            other.vx *= -1.0
        else:
            if rect.centery < other_rect.centery:
                enemy.gObj.y = other_rect.top - enemy.gObj.height
                enemy.vy = 0 
            else:
                enemy.gObj.y = other_rect.bottom
                enemy.vy = 0

    def _hit_qblock(self, core, col: int, row: int):
        for block in core.level_data.qblocks:
            b_col = int(block.gObj.x // TILE_SIZE)
            b_row = int(block.gObj.y // TILE_SIZE)

            if b_col == col and b_row == row and not block.hit:
                block.hit = True
                spawn_x = col * TILE_SIZE
                spawn_y = row * TILE_SIZE - 22

                if block.contains == "coin":
                    c = Coin(gObj=GameObject(col*TILE_SIZE+8, row*TILE_SIZE+8, 16, 16, True),
                             flyup=True, vy=-280.0, life=0.3, auto_collect=True)
                    c.gObj.type_id = EntityType.COIN
                    core.level_data.coins.append(c)
                elif block.contains == "mushroom":
                    p = Mushroom(gObj=GameObject(spawn_x, spawn_y, 20, 20, True))
                    p.gObj.type_id = EntityType.POWERUP
                    core.level_data.powerups.append(p)
                elif block.contains == "star":
                    p = StarPowerUp(gObj=GameObject(spawn_x, spawn_y, 20, 20, True))
                    p.gObj.type_id = EntityType.POWERUP
                    core.level_data.powerups.append(p)
                elif block.contains == "flower":
                    p = FireFlower(gObj=GameObject(spawn_x, spawn_y, 20, 20, True))
                    p.gObj.type_id = EntityType.POWERUP
                    core.level_data.powerups.append(p)
                elif block.contains == "life":
                    p = LifeUp(gObj=GameObject(spawn_x, spawn_y, 20, 20, True))
                    p.gObj.type_id = EntityType.POWERUP
                    core.level_data.powerups.append(p)
                else:
                    p = StarPowerUp(gObj=GameObject(spawn_x, spawn_y, 20, 20, True))
                    p.gObj.type_id = EntityType.POWERUP
                    core.level_data.powerups.append(p)

                if core.level_data.tiles[row][col]:
                    core.level_data.tiles[row][col].type_id = EntityType.TILE
                break

    def _get_tile_rects_near(self, level_data, obj):
        nearby_objects = level_data.static_hash.query(obj)
        out = []
        for item in nearby_objects:
            is_solid  = getattr(item, 'solid', False)
            gobj_tid  = item.gObj.type_id if hasattr(item.gObj, 'type_id') else EntityType.TILE

            # Filter: pass solids, spikes, and qblocks through
            if not is_solid and gobj_tid not in (EntityType.SPIKE, EntityType.QBLOCK):
                continue

            rect = item.gObj.get_rect()
            col  = int(item.gObj.x // TILE_SIZE)
            row  = int(item.gObj.y // TILE_SIZE)

            if gobj_tid == EntityType.SPIKE:
                tid = EntityType.SPIKE
            elif gobj_tid == EntityType.QBLOCK:
                tid = TILE_QBLOCK
            elif hasattr(item, 'type_id') and isinstance(item.type_id, int):
                tid = item.type_id   
            else:
                tid = gobj_tid       

            out.append((row, col, rect, tid))
        return out
