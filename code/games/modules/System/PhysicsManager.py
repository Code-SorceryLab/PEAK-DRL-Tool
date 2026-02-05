from dataclasses import dataclass
import importlib.util
import os
import sys
from typing import List, Any, Dict
import pygame

from .EntityType import EntityType
from .Movement_parameters import *
from .Jump_parameters import *
from .Map_parameters import TILE_SIZE, TILE_GROUND, TILE_PLATFORM, TILE_QBLOCK, TILE_SPIKE
from .SpatialHash import SpatialHash

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
    # CONTEXT MANAGEMENT HELPERS
    # =========================================================================

    def reset_to_defaults(self):
        """
        Clears the current context and resets it to the default values 
        defined in Movement_parameters and Jump_parameters.
        """
        self.context = PhysicsContext()
        self._apply_multiplier(self.speed_mult)
        print("[PhysicsManager] Context reset to defaults.")

    def set_context(self, new_context: PhysicsContext):
        """
        Completely replaces the current physics context with a new one.
        """
        if isinstance(new_context, PhysicsContext):
            self.context = new_context
            self._apply_multiplier(self.speed_mult)
        else:
            print("[PhysicsManager] Error: set_context expected PhysicsContext object.")

    def apply_config_dict(self, config: Dict[str, Any]):
        """
        Parses the nested dictionary structure from the Config Manager
        and maps it to the flat PhysicsContext.
        """
        # 1. Physics Section
        phys = config.get("physics", {})
        if "gravity" in phys: self.context.GRAVITY = float(phys["gravity"])
        if "fast_fall_gravity" in phys: self.context.FAST_FALL_GRAV = float(phys["fast_fall_gravity"])
        
        fric = phys.get("friction", {})
        if "ground" in fric: self.context.GROUND_FRICTION = float(fric["ground"])
        if "air" in fric: self.context.AIR_FRICTION = float(fric["air"])

        # 2. Player Section
        player = config.get("player", {})
        
        # Movement Sub-section
        move = player.get("movement", {})
        if "max_run_speed" in move: self.context.MAX_RUN_SPEED = float(move["max_run_speed"])
        if "run_accel" in move: self.context.RUN_ACCEL = float(move["run_accel"])
        if "max_walk_speed" in move: self.context.MAX_WALK_SPEED = float(move["max_walk_speed"])
        if "walk_accel" in move: self.context.WALK_ACCEL = float(move["walk_accel"])
        if "air_control" in move: self.context.AIR_CONTROL = float(move["air_control"])
        
        # Jump Sub-section
        jump = player.get("jump", {})
        if "max_velocity" in jump: self.context.JUMP_VEL_MAX = float(jump["max_velocity"])
        if "min_velocity" in jump: self.context.JUMP_VEL_MIN = float(jump["min_velocity"])
        if "hold_frames" in jump: self.context.JUMP_HOLD_FRAMES = int(jump["hold_frames"])
        if "coyote_frames" in jump: self.context.COYOTE_FRAMES = int(jump["coyote_frames"])
        if "buffer_frames" in jump: self.context.JUMP_BUFFER_FRAMES = int(jump["buffer_frames"])

        # Re-apply speed multiplier to ensure scaled values are correct
        self._apply_multiplier(self.speed_mult)
        print("[PhysicsManager] Applied configuration from dictionary.")

    def override_parameters(self, params: Dict[str, Any]):
        """
        Overrides specific physics parameters using a flat dictionary.
        Example: manager.override_parameters({'GRAVITY': 500.0})
        """
        count = 0
        for key, value in params.items():
            if hasattr(self.context, key):
                setattr(self.context, key, float(value))
                count += 1
            else:
                print(f"[PhysicsManager] Warning: Parameter '{key}' not found in PhysicsContext.")
        if count > 0:
            print(f"[PhysicsManager] Overridden {count} parameters.")

    def load_config(self, file_path: str):
        """
        Loads a python file as configuration and updates the current context.
        This serves as a way to 'change' context via file.
        """
        if not os.path.exists(file_path):
            print(f"[PhysicsManager] Warning: Config file '{file_path}' not found. Using defaults.")
            return

        try:
            spec = importlib.util.spec_from_file_location("dynamic_phys_config", file_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules["dynamic_phys_config"] = module
                spec.loader.exec_module(module)
                
                overrides = {}
                for field in self.context.__dataclass_fields__:
                    if hasattr(module, field):
                        overrides[field] = getattr(module, field)
                
                self.override_parameters(overrides)
                self._apply_multiplier(self.speed_mult)
        except Exception as e:
            print(f"[PhysicsManager] Error loading config: {e}")

    def _apply_multiplier(self, mult: float):
        # We apply multiplier to the BASE values currently in context
        # Note: In a production system you might want to store base_values separate from active_values
        # but for this architecture, we assume set_context/apply_config provides base values.
        self.context.RUN_ACCEL *= mult
        self.context.WALK_ACCEL *= mult
        self.context.MAX_WALK_SPEED *= mult
        self.context.MAX_RUN_SPEED *= mult

    def rebuild_dynamic_hashes(self, level_data):
        """
        Rebuilds the Hazard and Collectible hashes for the current frame.
        """
        self.hazard_hash.clear()
        self.collectible_hash.clear()

        # 1. Fill Hazards (Enemies)
        for enemy in level_data.enemies:
            if enemy.gObj.active:
                self.hazard_hash.insert(enemy)
        
        # 2. Fill Collectibles (Coins, Powerups)
        for coin in level_data.coins:
            if coin.gObj.active and not coin.collected:
                self.collectible_hash.insert(coin)
        
        for pup in level_data.powerups:
            if pup.gObj.active:
                self.collectible_hash.insert(pup)

    # =========================================================================
    # PHYSICS UPDATE LOOP
    # =========================================================================

    def update_system(self, dt: float, player, level_data):
        """
        Updates the physics state (velocity/position) for all entities.
        Does NOT resolve collisions.
        """
        ctx = self.context

        if player:
            # Pass context so Player doesn't need hardcoded imports
            player.update(dt, ctx)

        self.update_list(dt, level_data.enemies)
        self.update_list(dt, level_data.coins)
        self.update_list(dt, level_data.powerups)

    def update_list(self, dt: float, objects: List[Any]):
        ctx = self.context
        for obj in objects:
            if hasattr(obj, 'gObj') and not obj.gObj.active: continue
            
            # Entities with physics logic
            if hasattr(obj, "update"):
                try:
                    obj.update(dt, ctx) # Try passing context
                except TypeError:
                    obj.update(dt) # Fallback

    # =========================================================================
    # COLLISION RESOLUTION LOOP
    # =========================================================================

    def resolve_collisions(self, core):
        """
        Main entry point for resolving all collisions in the frame.
        """
        level_data = core.level_data
        
        # 1. Resolve Static World Collisions (Walls/Floors)
        self._resolve_player_world(core, level_data)
        self._resolve_enemy_world(core, level_data)

        # 2. Resolve Dynamic Entity Interactions
        self._resolve_dynamic_interactions(core)

    def _resolve_dynamic_interactions(self, core):
        """
        Queries spatial hashes and dispatches pairs to the switch-case handler.
        """
        player = core.player
        if not player: return

        # A. Player vs Hazards (Enemies)
        nearby_hazards = self.hazard_hash.query(player)
        for obj in nearby_hazards:
            if player.gObj.collides_with(obj.gObj):
                self._dispatch_collision(core, player, obj)
        
        # B. Player vs Collectibles (Coins/Powerups)
        nearby_items = self.collectible_hash.query(player)
        for obj in nearby_items:
            if player.gObj.collides_with(obj.gObj):
                self._dispatch_collision(core, player, obj)

        # C. Enemy vs Enemy (using Hazard Hash)
        for enemy in core.level_data.enemies:
            if not enemy.gObj.active: continue
            nearby_enemies = self.hazard_hash.query(enemy)
            for other in nearby_enemies:
                if other is not enemy and other.gObj.active:
                    # Explicit Type Check for safety
                    if isinstance(other, type(enemy)): # Simple class check
                        if enemy.gObj.collides_with(other.gObj):
                            self._dispatch_collision(core, enemy, other)

    def _dispatch_collision(self, core, source, target):
        """
        SWITCH-CASE HANDLER FOR COLLISION PAIRS.
        Directs logic based on EntityType enums.
        """
        # Safe extraction for objects that might not have type_id (Player, Dynamic Entities)
        s_type = source.gObj.type_id if hasattr(source.gObj, 'type_id') else EntityType.NONE
        t_type = target.gObj.type_id if hasattr(target.gObj, 'type_id') else EntityType.NONE
        
        # Fallback inference if type_id is missing
        if s_type == EntityType.NONE: s_type = self._infer_type(source)
        if t_type == EntityType.NONE: t_type = self._infer_type(target)

        match s_type:
            case EntityType.PLAYER:
                match t_type:
                    case EntityType.ENEMY:
                        # [COLLISION] Player vs Enemy
                        self._handle_player_enemy(core, source, target)
                    
                    case EntityType.COIN:
                        # [COLLISION] Player vs Coin
                        self._handle_player_coin(core, source, target)
                    
                    case EntityType.POWERUP:
                        # [COLLISION] Player vs Powerup
                        self._handle_player_powerup(core, source, target)
            
            case EntityType.ENEMY:
                match t_type:
                    case EntityType.ENEMY:
                         # [COLLISION] Enemy vs Enemy
                         self._handle_enemy_enemy(core, source, target)

    def _infer_type(self, obj):
        name = obj.__class__.__name__
        if name == "Player": return EntityType.PLAYER
        if name == "Enemy": return EntityType.ENEMY
        if name == "Coin": return EntityType.COIN
        if name == "Powerup": return EntityType.POWERUP
        return EntityType.NONE

    # --- SPECIFIC COLLISION LOGIC ---

    def _handle_player_enemy(self, core, player, enemy):
        player_bottom = player.gObj.y + player.gObj.height
        enemy_center = enemy.gObj.y + enemy.gObj.height/2
        moving_down = player.vy > 0
        
        # Calculate dynamic bounce force
        jump_bounce = self.context.JUMP_VEL_MIN * 0.6

        if player_bottom < enemy_center + 10 and moving_down:
            # Stomp
            enemy.gObj.active = False
            player.vy = jump_bounce
            core.score += 100
        elif player.invincible_timer > 0:
            # Star Power Kill
            enemy.gObj.active = False
            core.score += 100
        else:
            # Player Hit / Damage
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

    def _handle_player_powerup(self, core, player, powerup):
        powerup.gObj.active = False
        if powerup.kind == "mushroom":
            player.powered_up = True
            core.score += 50
        else:
            player.invincible_timer = 300
            core.score += 100

    def _handle_enemy_enemy(self, core, e1, e2):
        # Simple bounce logic
        r1 = e1.gObj.get_rect()
        r2 = e2.gObj.get_rect()
        
        if r1.centerx < r2.centerx:
            e1.gObj.x = r2.left - e1.gObj.width
        else:
            e1.gObj.x = r2.right
        e1.vx *= -1.0

    # --- WORLD COLLISION LOGIC ---

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

    def _hit_qblock(self, core, col: int, row: int):
        """
        Logic for hitting a QBlock.
        """
        for block in core.level_data.qblocks:
            # Check coords
            b_col = int(block.gObj.x // TILE_SIZE)
            b_row = int(block.gObj.y // TILE_SIZE)
            
            if b_col == col and b_row == row and not block.hit:
                block.hit = True
                spawn_x, spawn_y = col * TILE_SIZE, row * TILE_SIZE - 22
                
                # Spawn item via Core Level Data modification
                from .Coin import Coin
                from .Powerup import Powerup
                from .GameObject import GameObject
                
                if block.contains == "coin":
                    core.level_data.coins.append(Coin(gObj=GameObject(col*TILE_SIZE+8, row*TILE_SIZE+8, 16, 16, True), flyup=True, vy=-280.0, life=0.3, auto_collect=True))
                elif block.contains == "mushroom":
                    core.level_data.powerups.append(Powerup(gObj=GameObject(spawn_x, spawn_y, 20, 20, True), kind="mushroom"))
                else:
                    core.level_data.powerups.append(Powerup(gObj=GameObject(spawn_x, spawn_y, 20, 20, True), kind="star"))
                
                # Update tile to solid platform visually/physically
                # We need to find the specific tile object and update it
                if core.level_data.tiles[row][col]:
                    core.level_data.tiles[row][col].type_id = TILE_PLATFORM
                    # Note: We don't change color here as the visual loop handles it based on qblock state
                break

    def _resolve_enemy_world(self, core, level_data):
        for enemy in level_data.enemies:
            if not enemy.gObj.active: continue
            
            # Use Static Hash for world queries
            nearby = level_data.static_hash.query(enemy)
            # Assuming Enemy class has resolve logic, if not, we would implement it here.
            # For now, using the existing method signature from your code.
            if hasattr(enemy, "resolve_world_collisions"):
                enemy.resolve_world_collisions(nearby)
            else:
                 # Fallback basic resolution if needed
                 pass

    def _get_tile_rects_near(self, level_data, obj):
        """
        URGENT REPLACE TO USE THE LEVEL_LOADER STATIC HASH
        Returns list of solid tile rects near the given object using array lookup (O(1)).
        """
        nearby_objects = level_data.static_hash.query(obj)
        
        out = []
        for item in nearby_objects:
            # Filter for Tile objects that are solid
            # (Hash may contain other logic objects like QuestionBlocks, we only want the physical Tiles)
            rect = item.gObj.get_rect()
            # Calculate grid coordinates for logic callbacks (like hitting a QBlock)
            col = int(item.gObj.x // TILE_SIZE)
            row = int(item.gObj.y // TILE_SIZE)
            out.append((row, col, rect, item.type_id))
        return out