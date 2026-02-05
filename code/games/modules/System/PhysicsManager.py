from dataclasses import dataclass
import importlib.util
import os
import sys
from typing import List, Any, Dict
import pygame

# Helper to ensure we don't crash on imports if structure varies
try:
    from .EntityType import EntityType
except ImportError:
    from ..EntityType import EntityType

from ..Parameters.Movement_parameters import *
from ..Parameters.Jump_parameters import *
from ..Parameters.Map_parameters import TILE_SIZE, TILE_QBLOCK
from .SpatialHash import SpatialHash
from ..Objects.Coin import Coin
from ..Objects.Powerup import Powerup
from ..Objects.GameObject import GameObject

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

class PhysicsManager:
    def __init__(self, config_file: str = None, speed_mult: float = 1.0):
        self.context = PhysicsContext()
        self.speed_mult = speed_mult
        
        # --- SPATIAL HASHES FOR AI OBSERVATION ---
        self.hazard_hash = SpatialHash(64)
        self.collectible_hash = SpatialHash(64)
        
        if config_file:
            self.load_config(config_file)
            
        self._apply_multiplier(speed_mult)

    # =========================================================================
    # CONTEXT MANAGEMENT
    # =========================================================================

    def reset_to_defaults(self):
        self.context = PhysicsContext()
        self._apply_multiplier(self.speed_mult)

    def apply_config_dict(self, config: Dict[str, Any]):
        """
        Safely applies configuration. Uses 'or {}' to handle NoneType from YAML.
        """
        # 1. Physics Section
        phys = config.get("physics") or {}
        if "gravity" in phys: self.context.GRAVITY = float(phys["gravity"])
        if "fast_fall_gravity" in phys: self.context.FAST_FALL_GRAV = float(phys["fast_fall_gravity"])
        
        fric = phys.get("friction") or {}
        if "ground" in fric: self.context.GROUND_FRICTION = float(fric["ground"])
        if "air" in fric: self.context.AIR_FRICTION = float(fric["air"])

        # 2. Player Section
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

    def rebuild_dynamic_hashes(self, level_data):
        self.hazard_hash.clear()
        self.collectible_hash.clear()
        for enemy in level_data.enemies:
            if enemy.gObj.active: self.hazard_hash.insert(enemy)
        for coin in level_data.coins:
            if coin.gObj.active and not coin.collected: self.collectible_hash.insert(coin)
        for pup in level_data.powerups:
            if pup.gObj.active: self.collectible_hash.insert(pup)
    
    def get_dynamic_hash(self) -> SpatialHash:
        return self.hazard_hash

    # =========================================================================
    # UPDATE LOOP
    # =========================================================================

    def update_system(self, dt: float, core):
        ctx = self.context
        player = core.player
        level_data = core.level_data

        if player:
            player.update(dt, ctx)

        self.update_list(dt, level_data.enemies)
        self.update_list(dt, level_data.coins)
        self.update_list(dt, level_data.powerups)

    def update_list(self, dt: float, objects: List[Any]):
        ctx = self.context
        for obj in objects:
            if hasattr(obj, 'gObj') and not obj.gObj.active: continue
            if hasattr(obj, "update"):
                try:
                    obj.update(dt, ctx) 
                except TypeError:
                    obj.update(dt) 

    # =========================================================================
    # COLLISION RESOLUTION LOOP
    # =========================================================================

    def resolve_collisions(self, core):
        level_data = core.level_data
        
        # 1. Resolve Static World Collisions (Walls/Floors)
        self._resolve_player_world(core, level_data)
        self._resolve_enemy_world(core, level_data)
        self._resolve_powerup_world(core, level_data)

        # 2. Resolve Dynamic Entity Interactions
        self._resolve_dynamic_interactions(core)

    # --- WORLD COLLISION IMPLEMENTATION ---

    def _resolve_enemy_world(self, core, level_data):
        for enemy in level_data.enemies:
            if not enemy.gObj.active: continue
            nearby = self._get_tile_rects_near(level_data, enemy.gObj)
            # Enemies bounce on walls (bounce_x=True)
            self._solve_aabb_collision(enemy, nearby, bounce_x=True)

    def _resolve_powerup_world(self, core, level_data):
        for pup in level_data.powerups:
            if not pup.gObj.active: continue
            nearby = self._get_tile_rects_near(level_data, pup.gObj)
            self._solve_aabb_collision(pup, nearby, bounce_x=True)

    def _solve_aabb_collision(self, entity, nearby_tiles, bounce_x=False):
        """
        Generic AABB resolution against static tiles.
        Resolves Y first (Floor/Ceiling), then X (Walls).
        """
        # Resolve Y
        ent_rect = entity.gObj.get_rect()
        for (_, _, tile_rect, _) in nearby_tiles:
            if ent_rect.colliderect(tile_rect):
                # Calculate Y overlap
                if entity.vy >= 0: # Falling/Ground
                    if ent_rect.bottom >= tile_rect.top:
                        entity.gObj.y = tile_rect.top - entity.gObj.height
                        entity.vy = 0
                elif entity.vy < 0: # Jumping up
                     if ent_rect.top <= tile_rect.bottom:
                        entity.gObj.y = tile_rect.bottom
                        entity.vy = 0
                ent_rect = entity.gObj.get_rect() # Update for next tile check

        # Resolve X
        ent_rect = entity.gObj.get_rect()
        for (_, _, tile_rect, _) in nearby_tiles:
             if ent_rect.colliderect(tile_rect):
                # Check overlaps
                ox = min(ent_rect.right - tile_rect.left, tile_rect.right - ent_rect.left)
                oy = min(ent_rect.bottom - tile_rect.top, tile_rect.bottom - ent_rect.top)
                
                # If X overlap is shallower than Y, it's a wall hit
                if ox < oy:
                    if entity.vx > 0: # Moving Right
                        entity.gObj.x = tile_rect.left - entity.gObj.width
                        if bounce_x: entity.vx *= -1
                        else: entity.vx = 0
                    elif entity.vx < 0: # Moving Left
                        entity.gObj.x = tile_rect.right
                        if bounce_x: entity.vx *= -1
                        else: entity.vx = 0
                    ent_rect = entity.gObj.get_rect()

    def _resolve_player_world(self, core, level_data):
        player = core.player
        if not player: return
        
        rect = player.gObj.get_rect()
        nearby = self._get_tile_rects_near(level_data, player.gObj)
        
        for (row, col, trect, tile_type) in nearby:
            if not rect.colliderect(trect): continue
            
            overlap_x = min(rect.right - trect.left, trect.right - rect.left)
            overlap_y = min(rect.bottom - trect.top, trect.bottom - rect.top)
            
            if overlap_x < overlap_y:
                if rect.centerx < trect.centerx: player.gObj.x = trect.left - player.gObj.width
                else: player.gObj.x = trect.right
                player.vx *= 0.5
            else:
                if rect.centery < trect.centery:
                    # NOTE: 'max(2, player.vy + 1)' logic handles fast falling tunnel prevention
                    if abs(rect.bottom - trect.top) < max(2, player.vy + 1):
                        player.gObj.y = trect.top - player.gObj.height
                        player.vy = 0
                        player.on_ground = True
                        player.jump_hold = 0
                else:
                    player.gObj.y = trect.bottom
                    player.vy = max(0.0, player.vy)
                    if tile_type == TILE_QBLOCK and player.vy <= 0:
                        self._hit_qblock(core, col, row)
            rect = player.gObj.get_rect()

    # --- DYNAMIC COLLISIONS ---

    def _resolve_dynamic_interactions(self, core):
        player = core.player
        if not player: return

        nearby_hazards = self.hazard_hash.query(player)
        for obj in nearby_hazards:
            if player.gObj.collides_with(obj.gObj):
                self._dispatch_collision(core, player, obj)
        
        nearby_items = self.collectible_hash.query(player)
        for obj in nearby_items:
            if player.gObj.collides_with(obj.gObj):
                self._dispatch_collision(core, player, obj)

        for enemy in core.level_data.enemies:
            if not enemy.gObj.active: continue
            nearby_enemies = self.hazard_hash.query(enemy)
            for other in nearby_enemies:
                if other is not enemy and other.gObj.active:
                    if isinstance(other, type(enemy)): 
                        if enemy.gObj.collides_with(other.gObj):
                            self._dispatch_collision(core, enemy, other)

    def _dispatch_collision(self, core, source, target):
        s_type = source.gObj.type_id if hasattr(source.gObj, 'type_id') else EntityType.NONE
        t_type = target.gObj.type_id if hasattr(target.gObj, 'type_id') else EntityType.NONE
        
        if s_type == EntityType.NONE: s_type = self._infer_type(source)
        if t_type == EntityType.NONE: t_type = self._infer_type(target)

        match s_type:
            case EntityType.PLAYER:
                if t_type == EntityType.ENEMY:
                    self._handle_player_enemy(core, source, target)
                elif t_type == EntityType.COIN:
                    self._handle_player_coin(core, source, target)
                elif t_type == EntityType.POWERUP:
                    self._handle_player_powerup(core, source, target)
            
            case EntityType.ENEMY:
                if t_type == EntityType.ENEMY:
                    self._handle_enemy_enemy(core, source, target)

    def _infer_type(self, obj):
        name = obj.__class__.__name__
        if name == "Player": return EntityType.PLAYER
        if name == "Enemy": return EntityType.ENEMY
        if name == "Coin": return EntityType.COIN
        if name == "Powerup": return EntityType.POWERUP
        return EntityType.NONE

    # --- SPECIFIC HANDLERS ---

    def _handle_player_enemy(self, core, player, enemy):
        player_bottom = player.gObj.y + player.gObj.height
        enemy_center = enemy.gObj.y + enemy.gObj.height/2
        moving_down = player.vy > 0
        
        jump_bounce = self.context.JUMP_VEL_MIN * 0.6

        if player_bottom < enemy_center + 10 and moving_down:
            enemy.gObj.active = False
            player.vy = jump_bounce
            core.score += 100
            core.kills_step += 1
        elif player.invincible_timer > 0:
            enemy.gObj.active = False
            core.score += 100
            core.kills_step += 1
        else:
            if player.powered_up:
                player.powered_up = False
                player.invincible_timer = 60
            else:
                core._handle_death()

    def _handle_player_coin(self, core, player, coin):
        if not coin.collected:
            coin.gObj.active = False
            coin.collected = True
            core.score += 10
            core.coins_step += 1
            core.coins_total += 1

    def _handle_player_powerup(self, core, player, powerup):
        powerup.gObj.active = False
        core.powerups_step += 1
        if powerup.kind == "mushroom":
            player.powered_up = True
            core.score += 50
        else:
            player.invincible_timer = 300
            core.score += 100

    def _handle_enemy_enemy(self, core, e1, e2):
        r1 = e1.gObj.get_rect()
        r2 = e2.gObj.get_rect()
        
        if r1.centerx < r2.centerx:
            e1.gObj.x = r2.left - e1.gObj.width
        else:
            e1.gObj.x = r2.right
        e1.vx *= -1.0

    # --- HELPERS ---

    def _hit_qblock(self, core, col: int, row: int):
        for block in core.level_data.qblocks:
            b_col = int(block.gObj.x // TILE_SIZE)
            b_row = int(block.gObj.y // TILE_SIZE)
            
            if b_col == col and b_row == row and not block.hit:
                block.hit = True
                spawn_x, spawn_y = col * TILE_SIZE, row * TILE_SIZE - 22
                
                if block.contains == "coin":
                    c = Coin(gObj=GameObject(col*TILE_SIZE+8, row*TILE_SIZE+8, 16, 16, True), flyup=True, vy=-280.0, life=0.3, auto_collect=True)
                    c.gObj.type_id = EntityType.COIN
                    core.level_data.coins.append(c)
                elif block.contains == "mushroom":
                    p = Powerup(gObj=GameObject(spawn_x, spawn_y, 20, 20, True), kind="mushroom")
                    p.gObj.type_id = EntityType.POWERUP
                    core.level_data.powerups.append(p)
                else:
                    p = Powerup(gObj=GameObject(spawn_x, spawn_y, 20, 20, True), kind="star")
                    p.gObj.type_id = EntityType.POWERUP
                    core.level_data.powerups.append(p)
                
                if core.level_data.tiles[row][col]:
                    core.level_data.tiles[row][col].type_id = EntityType.TILE 
                break

    def _get_tile_rects_near(self, level_data, obj):
        """
        Uses the LevelLoader Static Hash to find nearby tiles.
        """
        nearby_objects = level_data.static_hash.query(obj)
        out = []
        for item in nearby_objects:
            # FIX: Check 'solid' on the wrapper (Tile), NOT the inner gObj (GameObject)
            # GameObject does not have 'solid' attribute, so it always defaulted to False.
            if not getattr(item, 'solid', False): continue
            
            rect = item.gObj.get_rect()
            col = int(item.gObj.x // TILE_SIZE)
            row = int(item.gObj.y // TILE_SIZE)
            
            tid = item.gObj.type_id if hasattr(item.gObj, 'type_id') else EntityType.TILE
            if hasattr(item, 'type_id') and item.type_id == EntityType.QBLOCK:
                tid = TILE_QBLOCK 
                
            out.append((row, col, rect, tid))
        return out