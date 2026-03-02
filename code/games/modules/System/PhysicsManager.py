from dataclasses import dataclass
import importlib.util
import os
import sys
from typing import List, Any, Dict
import pygame
import math
from .EntityType import EntityType
from ..Parameters.Movement_parameters import *
from ..Parameters.Jump_parameters import *
from ..Parameters.Map_parameters import TILE_SIZE, TILE_QBLOCK, TILE_PLATFORM
from .SpatialHash import SpatialHash
from ..Objects.Coin import Coin
from ..Objects.Mushroom import Mushroom
from ..Objects.LifeUp import LifeUp
from ..Objects.StarPowerUp import StarPowerUp
from ..Objects.FireFlower import FireFlower
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
    """
    Manages the physics simulation for the game, including movement updates,
    collision detection, and resolution for all entities.
    """
    def __init__(self, config_file: str = None, speed_mult: float = 1.0):
        self.context = PhysicsContext()
        self.speed_mult = speed_mult

        # --- SPATIAL HASHES FOR AI OBSERVATION ---
        self.hazard_hash     = SpatialHash(64)
        self.collectible_hash = SpatialHash(64)

        # Moving platforms get their own hash — rebuilt every frame in step()
        # because their positions change. Kept separate from static_hash so
        # static geometry never needs to be rebuilt.
        self.platform_hash = SpatialHash(64)

        if config_file:
            self.load_config(config_file)

        self._apply_multiplier(speed_mult)

    # =========================================================================
    # CONTEXT MANAGEMENT
    # =========================================================================

    def reset_to_defaults(self):
        """
        Resets the physics context to the default hardcoded parameters
        and reapplies the current speed multiplier.
        """
        self.context = PhysicsContext()
        self._apply_multiplier(self.speed_mult)

    def apply_config_dict(self, config: Dict[str, Any]):
        """
        Safely applies configuration. Uses 'or {}' to handle NoneType from YAML.

        1. Reads 'physics' section for gravity and friction.
        2. Reads 'player' section for movement and jump parameters.
        3. Re-applies the speed multiplier to the new values.
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
        """
        Scales movement speeds and accelerations by a global multiplier.
        """
        self.context.RUN_ACCEL *= mult
        self.context.WALK_ACCEL *= mult
        self.context.MAX_WALK_SPEED *= mult
        self.context.MAX_RUN_SPEED *= mult

    def rebuild_dynamic_hashes(self, level_data):
        """
        Clears and repopulates the spatial hashes with active entities
        (enemies, coins, powerups) for optimized collision queries.
        """
        self.hazard_hash.clear()
        self.collectible_hash.clear()
        for enemy in level_data.enemies:
            if enemy.gObj.active: self.hazard_hash.insert(enemy)
        for coin in level_data.coins:
            if coin.gObj.active and not coin.collected: self.collectible_hash.insert(coin)
        for pup in level_data.powerups:
            if pup.gObj.active: self.collectible_hash.insert(pup)

        # Add goals to collectible hash (since they are 'collected' to win)
        for goal in level_data.goals:
            self.collectible_hash.insert(goal)

    def get_dynamic_hash(self) -> SpatialHash:
        return self.hazard_hash

    # =========================================================================
    # UPDATE LOOP
    # =========================================================================

    def update_system(self, dt: float, core):
        """
        Updates the physics state (position, velocity) for all entities.
        Includes Continuous Collision Detection (CCD) for the player to prevent tunneling.

        1. Checks how far the player intends to move this frame.
        2. If the distance > 0.5 tiles, splits the frame into multiple smaller sub-steps.
        3. Executes each sub-step for the player, resolving wall collisions immediately after each step.
        4. Updates enemies, coins, and powerups normally (single step).
        """
        ctx = self.context
        player = core.player
        level_data = core.level_data

        if player:
            # --- ANTI-TUNNELING (Continuous Collision Detection) ---
            # 1. Calculate how far the player WANTS to move this frame
            predicted_dist = max(abs(player.vx), abs(player.vy)) * dt

            # 2. If moving more than 1/2 a tile, break it into smaller steps
            #    (We use 0.5 tile size to be safe)
            step_count = 1
            if predicted_dist > TILE_SIZE * 0.5:
                step_count = int(math.ceil(predicted_dist / (TILE_SIZE * 0.5)))

            step_dt = dt / step_count

            # 3. Execute the steps
            for _ in range(step_count):
                player.update(step_dt, ctx)
                # Resolve ONLY player world collisions immediately
                # This stops them from entering a wall during a sub-step
                self._resolve_player_world(core, level_data)

        # Update everything else normally (Enemies usually don't move fast enough to tunnel)
        self.update_list(dt, level_data.enemies)
        self.update_list(dt, level_data.coins)
        self.update_list(dt, level_data.powerups)
        self.update_list(dt, level_data.moving_platforms)

    def update_list(self, dt: float, objects: List[Any]):
        """
        Helper to iterate through a list of objects and call their update method
        if they are active.
        """
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
        """
        Main entry point for resolving all collisions in the game world.

        1. Captures the player's falling state (vy > 0) before any velocity modifications.
           This is crucial for "stomp" logic, as hitting the ground later resets vy to 0.
        2. Resolves static world collisions (walls/floors) for the Player, Enemies, and Powerups.
        3. Resolves dynamic interactions (Player vs Enemy, Enemy vs Enemy, etc.) using the
           previously captured falling state.
        """
        level_data = core.level_data
        player = core.player

        # 1. FIX: Capture falling state BEFORE world collision resets vy to 0.
        # This ensures that even if we hit the ground in this frame, we know
        # we were coming down, allowing stomps to register.
        player_was_falling = False
        if player:
            player_was_falling = player.vy > 0

        # 2. Resolve Static World Collisions (Walls/Floors)
        self._resolve_player_world(core, level_data)
        self._resolve_player_moving_platforms(core, level_data)
        self._resolve_enemy_world(core, level_data)
        self._resolve_powerup_world(core, level_data)

        # 3. Resolve Dynamic Entity Interactions
        self._resolve_dynamic_interactions(core, player_was_falling)

    # --- WORLD COLLISION IMPLEMENTATION ---

    def _resolve_player_moving_platforms(self, core, level_data):
        """
        Handles player colliding with moving platforms — all four sides solid.

        Uses velocity to discriminate floor vs ceiling hit, NOT centery.
        centery fails for tall (BIG) players whose centre can be above the
        platform centre while still jumping into it from below.

        Resolution per overlapping platform:
          overlap_x < overlap_y  →  wall hit  (push out horizontally, zero vx)
          overlap_y <= overlap_x and vy >= 0  →  floor hit (snap to top, carry, land)
          overlap_y <= overlap_x and vy <  0  →  ceiling hit (snap to bottom, bonk)
        """
        player = core.player
        if not player or not level_data.moving_platforms:
            return

        prect = player.gObj.get_rect()

        for plat in level_data.moving_platforms:
            if not plat.gObj.active:
                continue

            trect = plat.gObj.get_rect()

            if not prect.colliderect(trect):
                continue

            overlap_x = min(prect.right  - trect.left, trect.right  - prect.left)
            overlap_y = min(prect.bottom - trect.top,  trect.bottom - prect.top)

            if overlap_x < overlap_y:
                # ── Wall hit ──────────────────────────────────────────────────
                if prect.centerx < trect.centerx:
                    player.gObj.x = trect.left - player.gObj.width
                else:
                    player.gObj.x = trect.right
                player.vx = 0

            elif player.vy >= 0 and prect.centery <= trect.top:
                # ── Floor hit (moving down, centre above platform surface) ────
                # The centery guard stops "ghost stepping": a player running into
                # the side of a thin platform can get overlap_x > overlap_y,
                # which would pass the axis test. But if the player's centre is
                # BELOW the platform top they came from the side, not the top.
                player.gObj.x += plat.delta_x
                player.gObj.y  = trect.top - player.gObj.height
                player.vy      = 0
                player.on_ground = True
                player.jump_hold = 0

            elif player.vy < 0:
                # ── Ceiling hit (moving up) ───────────────────────────────────
                player.gObj.y = trect.bottom
                player.vy     = 0

            else:
                # ── Side approach below platform — treat as wall ──────────────
                # Catches the ghost-step case where overlap_x > overlap_y but
                # the player's centre is below the platform top (ran into side).
                if prect.centerx < trect.centerx:
                    player.gObj.x = trect.left - player.gObj.width
                else:
                    player.gObj.x = trect.right
                player.vx = 0

            prect = player.gObj.get_rect()

    def _resolve_enemy_world(self, core, level_data):
        """
        Handles collisions between enemies and the static map.
        Sets 'bounce_x=True' so enemies reverse direction when hitting walls.
        """
        for enemy in level_data.enemies:
            if not enemy.gObj.active: continue
            nearby = self._get_tile_rects_near(level_data, enemy.gObj)
            # Enemies bounce on walls (bounce_x=True)
            self._solve_aabb_collision(enemy, nearby, bounce_x=True)

    def _resolve_powerup_world(self, core, level_data):
        """
        Handles collisions between powerups (mushroom/star) and the static map.
        """
        for pup in level_data.powerups:
            if not pup.gObj.active: continue
            nearby = self._get_tile_rects_near(level_data, pup.gObj)
            self._solve_aabb_collision(pup, nearby, bounce_x=True)

    def _solve_aabb_collision(self, entity, nearby_tiles, bounce_x=False):
        """
        Generic Axis-Aligned Bounding Box (AABB) resolution against static tiles.
        Used for non-player entities (Enemies, Powerups).

        1. Resolves Y-Axis (Vertical):
           - Checks overlaps.
           - If moving down, snaps to top of tile (Floor).
           - If moving up, snaps to bottom of tile (Ceiling).
        2. Resolves X-Axis (Horizontal):
           - Checks overlaps again (post-Y adjustment).
           - If X overlap is smaller than Y overlap, it treats it as a wall.
           - Snaps entity to the side of the tile.
           - If 'bounce_x' is True, inverts X velocity; otherwise zeroes it.
        """
        # Resolve Y
        ent_rect = entity.gObj.get_rect()
        for (_, _, tile_rect, _) in nearby_tiles:
            if ent_rect.colliderect(tile_rect):
                # Calculate overlaps to determine if this is actually a wall hit
                ox = min(ent_rect.right - tile_rect.left, tile_rect.right - ent_rect.left)
                oy = min(ent_rect.bottom - tile_rect.top, tile_rect.bottom - ent_rect.top)

                # If X overlap is smaller than Y, it's a wall.
                # Don't resolve Y here, let the X-pass handle it.
                if ox < oy:
                    continue

                # Calculate Y overlap
                if entity.vy >= 0: # Falling/Ground
                    if ent_rect.bottom >= tile_rect.top:
                        entity.gObj.y = tile_rect.top - entity.gObj.height
                        entity.vy = 0
                        if hasattr(entity, 'on_ground'):
                            entity.on_ground = True
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
                        if bounce_x:
                            entity.vx *= -1
                        else:
                            entity.vx = 0
                    elif entity.vx < 0: # Moving Left
                        entity.gObj.x = tile_rect.right
                        if bounce_x:
                            entity.vx *= -1
                        else:
                            entity.vx = 0
                    ent_rect = entity.gObj.get_rect()

    def _resolve_player_world(self, core, level_data):
        """
        Specialized collision resolution for the Player against static tiles.

        1. Identifies nearby tiles using spatial hashing.
        2. Iterates through tiles and calculates overlap depth in X and Y.
        3. Prioritizes the shallowest axis of penetration (Separating Axis Theorem principle).
        4. If X collision (Wall): Pushes player out and zeroes X velocity.
        5. If Y collision (Floor/Ceiling):
           - Floor: Pushes player up. Sets 'on_ground' = True ONLY if player was falling (vy >= 0).
           - Ceiling: Pushes player down. Zeroes upward velocity (Bonk).
           - Checks for QBlock interactions on ceiling hits.
        """
        player = core.player
        if not player: return

        rect = player.gObj.get_rect()
        nearby = self._get_tile_rects_near(level_data, player.gObj)

        for (row, col, trect, tile_type) in nearby:
            if not rect.colliderect(trect): continue

            # --- Handle Spikes (Static Tiles but Deadly) ---
            if tile_type == EntityType.SPIKE:
                # Star power makes the player invincible to spikes too
                if player.power_machine.is_invincible:
                    continue
                core._handle_death("Spike")
                return

            overlap_x = min(rect.right - trect.left, trect.right - rect.left)
            overlap_y = min(rect.bottom - trect.top, trect.bottom - rect.top)

            # Prioritize the shallowest axis
            if overlap_x < overlap_y:
                # --- Resolve X (Wall Hit) ---
                if rect.centerx < trect.centerx:
                    player.gObj.x = trect.left - player.gObj.width
                else:
                    player.gObj.x = trect.right

                # Zero X velocity on wall hit (stops sticking)
                player.vx = 0

            else:
                    # --- Resolve Y (Floor/Ceiling Hit) ---
                if rect.centery < trect.centery:
                    # Player is ABOVE the tile (Floor Hit)
                    player.gObj.y = trect.top - player.gObj.height

                    # Only trigger "landing" logic if we were actually falling
                    # This prevents "snapping" to the floor while jumping up
                    if player.vy >= 0:
                        player.vy = 0
                        player.on_ground = True
                        player.jump_hold = 0
                else:
                    # Player is BELOW the tile (Ceiling Hit)
                    # ONE-WAY PLATFORM: platforms are solid from the top only.
                    # If the player is jumping up into the underside of a platform,
                    # skip — let them pass through. Only ground tiles and qblocks
                    # act as true ceilings.
                    if tile_type == TILE_PLATFORM:
                        rect = player.gObj.get_rect()
                        continue

                    player.gObj.y = trect.bottom

                    # Bonk head: kill upward velocity
                    if player.vy < 0:
                        player.vy = 0

                    if tile_type == TILE_QBLOCK:
                        self._hit_qblock(core, col, row)

            # Update rect for the next tile check in the loop
            rect = player.gObj.get_rect()

    # --- DYNAMIC COLLISIONS ---

    def _resolve_dynamic_interactions(self, core, player_was_falling):
        """
        Handles interactions between moving entities (Player, Enemies, Coins).

        1. Queries the spatial hash for hazards (Enemies) near the player.
        2. Dispatches collision events if overlaps occur.
        3. Queries for collectibles (Coins/Powerups) and dispatches events.
        4. Checks Enemy-vs-Enemy collisions (to prevent stacking/overlap).
        """
        player = core.player
        if not player: return

        # Use the passed 'player_was_falling' boolean which represents
        # the physics state BEFORE wall/floor resolution.

        nearby_hazards = self.hazard_hash.query(player)
        for obj in nearby_hazards:
            if player.gObj.collides_with(obj.gObj):
                self._dispatch_collision(core, player, obj, player_was_falling)

        nearby_items = self.collectible_hash.query(player)
        for obj in nearby_items:
            if player.gObj.collides_with(obj.gObj):
                self._dispatch_collision(core, player, obj, player_was_falling)
            else:
                # Special case: player landed exactly ON TOP of a goal tile.
                # After AABB resolution the player sits flush on the tile's top edge
                # so collides_with returns False (zero overlap). Use a 2px downward
                # probe to catch this case.
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

    def _dispatch_collision(self, core, source, target, player_was_falling=False):
        """
        Routes the collision to the specific handler based on the entity types involved
        (e.g., Player vs Enemy, Player vs Coin, Enemy vs Enemy).
        """
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
                    # Guard: goal must not be the left-wall spawn tile (x < 1 tile)
                    goal_is_real = target.gObj.x > TILE_SIZE
                    if not core.reached_goal and goal_is_real:
                        # Time bonus only — no flat 1000pt base.
                        # The old formula (1000 + timer*10) gave up to 4000pts per
                        # level which dominated the score and made the metric
                        # misleading. Remaining time is a clean, bounded bonus.
                        core.score += int(core.timer)
                        core.reached_goal = True
                        core.complete_level()
                elif t_type == EntityType.SPIKE:
                    if not source.power_machine.is_invincible:
                        core._handle_death("Spike")

            case EntityType.ENEMY:
                if t_type == EntityType.ENEMY:
                    self._handle_enemy_enemy(core, source, target)

    def _infer_type(self, obj):
        name = obj.__class__.__name__
        if name == "Player": return EntityType.PLAYER
        if name == "Enemy": return EntityType.ENEMY
        if name == "Coin": return EntityType.COIN
        if name == "Powerup": return EntityType.POWERUP
        if name == "Goal": return EntityType.GOAL
        return EntityType.NONE

    # --- SPECIFIC HANDLERS ---

    def _handle_player_enemy(self, core, player, enemy, player_was_falling):
        """
        Handles logic when Player hits an Enemy.

        Resolution order:
          1. Stomp    — player was falling AND feet above enemy centre → kill enemy, bounce player.
          2. Star     — player.power_machine.is_invincible → kill enemy, no damage to player.
          3. Damage   — player.power_machine.take_hit():
                          True  → survived (downgraded or i-frames absorbed)
                          False → dead, core._handle_death() already called by machine
                                  (or we call it here if on_death was not set on machine)
        """
        if not enemy.gObj.active:
            return

        player_bottom = player.gObj.y + player.gObj.height
        enemy_center  = enemy.gObj.y  + enemy.gObj.height / 2

        # 1. Stomp — falling and feet above enemy midpoint
        if player_was_falling and player_bottom < enemy_center + 10:
            enemy.gObj.active = False
            player.vy = JUMP_VEL_MIN * 0.6   # bounce
            core.score += 100
            if hasattr(core, 'kills_step'):
                core.kills_step += 1
            return

        # 2. Star — kill enemy without taking damage
        if player.power_machine.is_invincible:
            enemy.gObj.active = False
            core.score += 100
            if hasattr(core, 'kills_step'):
                core.kills_step += 1
            return

        # 3. Take a hit through the state machine
        survived = player.power_machine.take_hit()
        if not survived:
            # on_death may not be wired on the machine — call core directly
            core._handle_death("Enemy")

    def _handle_player_coin(self, core, player, coin):
        """
        Collects coin, increments score and coin counters.
        """
        if not coin.collected:
            coin.gObj.active = False
            coin.collected = True
            core.score += 10
            core.coins_step += 1
            core.coins_total += 1

    def _handle_player_powerup(self, core, player, powerup):
        """
        Applies powerup effect and removes the item.

        Kind          Class         Effect
        ---------     ----------    ------------------------------------
        "mushroom"  → Mushroom    → collect_mushroom()  SMALL→BIG
        "flower"    → FireFlower  → collect_flower()    any→FIRE
        "star"      → StarPowerup → collect_star()      star timer
        "life"      → LifePowerup → lives += 1          no power change
        """
        powerup.gObj.active = False
        core.powerups_step += 1

        if powerup.kind == "mushroom":
            player.power_machine.collect_mushroom()
            core.score += 50
        elif powerup.kind == "flower":
            player.power_machine.collect_flower()
            core.score += 50
        elif powerup.kind == "life":
            core.lives += 1
            core.score += 200
        else:
            # "star" or unknown → star
            player.power_machine.collect_star()
            core.score += 100

    def _handle_enemy_enemy(self, core, e1, e2):
        """
        Resolves collisions between two enemies to prevent them from walking through each other.
        - Horizontal: Bounces both enemies away from each other.
        - Vertical: Stacks them (sets velocity to 0).
        """
        enemy = e1
        other = e2
        rect = enemy.gObj.get_rect()
        other_rect = other.gObj.get_rect()

        # Calculate overlap amounts
        obj_x = min(rect.right - other_rect.left, other_rect.right - rect.left)
        obj_y = min(rect.bottom - other_rect.top, other_rect.bottom - rect.top)

        if obj_x < obj_y:
            # Horizontal collision - push apart and bounce
            if rect.centerx < other_rect.centerx:
                enemy.gObj.x = other_rect.left - enemy.gObj.width
            else:
                enemy.gObj.x = other_rect.right

            # Bounce both to prevent sticking and ghosting
            enemy.vx *= -1.0
            other.vx *= -1.0

        else:
            # Vertical collision - STACKING
            if rect.centery < other_rect.centery:
                # 'enemy' is on top of 'other'
                enemy.gObj.y = other_rect.top - enemy.gObj.height
                enemy.vy = 0 # Land on top
            else:
                # 'enemy' hit 'other' from below
                enemy.gObj.y = other_rect.bottom
                enemy.vy = 0

    # --- HELPERS ---

    def _hit_qblock(self, core, col: int, row: int):
        """
        Triggered when a player hits a Question Block from below.
        Spawns the correct powerup class based on block.contains.

        contains      spawns
        ----------    ----------------
        "coin"      → fly-up Coin
        "mushroom"  → Mushroom  (walks, bounces)
        "star"      → StarPowerup (bounces and jumps)
        "flower"    → FireFlower  (stationary)
        "life"      → LifePowerup (walks, bounces)
        """
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
                    # Fallback — treat unknown as star
                    p = StarPowerUp(gObj=GameObject(spawn_x, spawn_y, 20, 20, True))
                    p.gObj.type_id = EntityType.POWERUP
                    core.level_data.powerups.append(p)

                if core.level_data.tiles[row][col]:
                    core.level_data.tiles[row][col].type_id = EntityType.TILE
                break

    def _get_tile_rects_near(self, level_data, obj):
        """
        Uses the LevelLoader Static Hash to find nearby tiles.

        Type resolution:
          Tile objects  → item.type_id  is an int (TILE_GROUND, TILE_PLATFORM …)
                          item.gObj.type_id is EntityType.TILE (used only for filter)
          Spike objects → item.gObj.type_id is EntityType.SPIKE
          QBlock objects→ item.gObj.type_id is EntityType.QBLOCK

        tid passed downstream must be:
          EntityType.SPIKE  — so _resolve_player_world death-check fires
          TILE_QBLOCK (int) — so _hit_qblock fires
          TILE_PLATFORM (int) — so one-way pass-through fires
          anything else     — treated as solid ground
        """
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

            # Resolve the tile type to pass to _resolve_player_world:
            #   Spikes  → keep EntityType.SPIKE  (death check uses this)
            #   QBlocks → TILE_QBLOCK int         (hit_qblock check uses this)
            #   Tiles   → item.type_id int        (TILE_GROUND or TILE_PLATFORM)
            #             This is the KEY fix: gObj.type_id is always EntityType.TILE
            #             for all ground/platform tiles. item.type_id (set by
            #             create_tile) is the actual int constant we need.
            if gobj_tid == EntityType.SPIKE:
                tid = EntityType.SPIKE
            elif gobj_tid == EntityType.QBLOCK:
                tid = TILE_QBLOCK
            elif hasattr(item, 'type_id') and isinstance(item.type_id, int):
                tid = item.type_id   # TILE_GROUND, TILE_PLATFORM etc.
            else:
                tid = gobj_tid       # fallback

            out.append((row, col, rect, tid))
        return out