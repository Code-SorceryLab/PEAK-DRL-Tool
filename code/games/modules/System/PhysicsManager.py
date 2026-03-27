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

    def rebuild_dynamic_hashes(self, level_data, cached_spikes=None):
        """
        Clears and repopulates the spatial hashes with active entities
        (enemies, spikes, coins, powerups, goals) for optimized collision queries.

        cached_spikes: pre-built list of Spike tile objects from load_level().
        Spikes are inserted into hazard_hash so the CNN observation grid
        picks them up in channel 1 (hazard) via a single query_rect call.
        """
        self.hazard_hash.clear()
        self.collectible_hash.clear()
        for enemy in level_data.enemies:
            if enemy.gObj.active: self.hazard_hash.insert(enemy)
        if cached_spikes:
            for spike in cached_spikes:
                self.hazard_hash.insert(spike)
        for coin in level_data.coins:
            if coin.gObj.active and not coin.collected: self.collectible_hash.insert(coin)
        for pup in level_data.powerups:
            if pup.gObj.active: self.collectible_hash.insert(pup)
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

        X/Y collision split:
          Each CCD sub-step applies the full player.update() (moves both axes),
          then resolves X and Y independently:
            1. Reset Y to pre-update value — only X has moved.
               Any collision is unambiguously a wall. Resolve X.
            2. Restore post-update Y — only Y has moved from the X-resolved position.
               Any collision is unambiguously floor or ceiling. Resolve Y.

          This eliminates the axis-ambiguity bug that fires when stacked tiles are
          processed in an order where a wall tile's oy < ox looks like a ceiling hit.
        """
        ctx = self.context
        player = core.player
        level_data = core.level_data

        if player:
            predicted_dist = max(abs(player.vx), abs(player.vy)) * dt
            step_count = 1
            if predicted_dist > TILE_SIZE * 0.5:
                step_count = int(math.ceil(predicted_dist / (TILE_SIZE * 0.5)))
            step_dt = dt / step_count

            for _ in range(step_count):
                pre_y      = player.gObj.y       # save Y before update
                pre_height = player.gObj.height  # save height before update

                player.update(step_dt, ctx)    # moves both X and Y

                post_y      = player.gObj.y       # save post-update Y (includes anchor correction)
                post_height = player.gObj.height  # save post-update height

                # ── Pass 1: X only ────────────────────────────────────────────
                # Restore Y to pre-update value, BUT account for any height change
                # that apply_to_player() made. Without this, the anchor correction
                # is lost and the player sits at the wrong Y for the X pass,
                # causing false wall overlaps that snap the X position.
                height_delta = post_height - pre_height
                player.gObj.y      = pre_y - height_delta  # keep feet at same spot
                player.gObj.height = post_height
                self._resolve_player_world_x(core, level_data)

                # ── Pass 2: Y only ────────────────────────────────────────────
                # X is now resolved. Restore post-update Y so only Y has moved.
                # Every remaining collision must be floor or ceiling.
                player.gObj.y = post_y
                self._resolve_player_world_y(core, level_data)

        self.update_list(dt, level_data.enemies)
        self.update_list(dt, level_data.coins)
        self.update_list(dt, level_data.powerups)
        self.update_list(dt, level_data.moving_platforms)
        self.update_list(dt, level_data.projectiles)

        # ── Pit / out-of-bounds death ─────────────────────────────────────────
        # Checked here (inside the physics step) rather than in
        # _check_termination so death fires on the same frame the player
        # crosses the kill plane, before the observation is built.
        # This also handles high-speed falls that could overshoot the boundary
        # check in _check_termination by an arbitrary amount.
        self._check_oob(core, level_data)

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

    def _check_oob(self, core, level_data):
        """
        Out-of-bounds / pit kill plane — owned by the physics manager.

        Kill plane geometry
        ───────────────────
        Bottom:  player bottom edge > level_data.height + PIT_GRACE_PX
                 Grace buffer absorbs sub-pixel overshoot at high fall speeds
                 without adding visible delay at normal speeds.

        Left/Right: player exits the map horizontally by more than one tile.
                 Platforms that extend slightly beyond the grid don't kill;
                 the extra-tile margin absorbs normal edge cases.

        Enemies that fall off the map are silently deactivated (no death event
        needed — they simply disappear, consistent with classic platformer feel).
        """
        # ── Player ───────────────────────────────────────────────────────────
        player = core.player
        if player and player.gObj.active and core.alive:
            PIT_GRACE_PX  = 48          # pixels below level bottom before kill fires
            OOB_MARGIN_PX = level_data.tile_size if hasattr(level_data, 'tile_size') else 32

            fell_out    = player.gObj.y > level_data.height + PIT_GRACE_PX
            left_out    = player.gObj.x + player.gObj.width < -OOB_MARGIN_PX
            right_out   = player.gObj.x > level_data.width  + OOB_MARGIN_PX

            if fell_out or left_out or right_out:
                cause = "Pit" if fell_out else "OOB"
                core._handle_death(cause)
                return   # no further physics this frame

        # ── Enemies ──────────────────────────────────────────────────────────
        # Enemies that walk off ledges or get knocked below the map are deactivated
        # so they don't accumulate as invisible, still-collidable phantoms.
        kill_y = level_data.height + 64
        for enemy in level_data.enemies:
            if enemy.gObj.active and enemy.gObj.y > kill_y:
                enemy.gObj.active = False

    # =========================================================================
    # COLLISION RESOLUTION LOOP
    # =========================================================================

    def resolve_collisions(self, core):
        """
        Main entry point for resolving all collisions in the game world.

        1. Captures the player's falling state (vy > 0) before any velocity modifications.
           This is crucial for "stomp" logic, as hitting the ground later resets vy to 0.
        2. Resolves static world collisions (walls/floors) for the Player, Enemies,
           Powerups, and Projectiles.
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
        self._resolve_projectile_world(core, level_data)

        # 3. Resolve Dynamic Entity Interactions
        self._resolve_dynamic_interactions(core, player_was_falling)

    # --- WORLD COLLISION IMPLEMENTATION ---

    def _resolve_player_moving_platforms(self, core, level_data):
        """
        Resolves player collisions against moving platforms.

        Moving platforms are fully solid on all four sides — unlike static
        TILE_PLATFORM tiles they have no one-way pass-through behaviour.

        Uses velocity direction (vx / vy) for axis discrimination, NOT center
        position comparisons:

          centerx / centery comparisons fail when the player clips more than
          halfway into a platform (deep penetration or fast platforms) and can
          push the player through to the wrong side.

          The old centery <= trect.top guard was added to prevent ghost-stepping
          (treating a side-clip as a floor hit) but broke for large (BIG) players
          whose centre sits below the platform top even when standing on it
          correctly. Removed in favour of the axis-dominance check (ox < oy).

        Resolution per overlapping platform:
          ox < oy  →  wall hit  : push horizontally (vx direction), zero vx
          vy >= 0  →  floor hit : snap to top, carry (delta_x), land
          vy <  0  →  ceiling   : snap to bottom, zero vy (bonk)
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

            ox = min(prect.right  - trect.left, trect.right  - prect.left)
            oy = min(prect.bottom - trect.top,  trect.bottom - prect.top)

            if ox < oy:
                # ── Wall hit ──────────────────────────────────────────────────
                if player.vx > 0:
                    player.gObj.x = trect.left - player.gObj.width
                elif player.vx < 0:
                    player.gObj.x = trect.right
                else:
                    # Stationary — push away from platform centre
                    if prect.centerx <= trect.centerx:
                        player.gObj.x = trect.left - player.gObj.width
                    else:
                        player.gObj.x = trect.right
                player.vx = 0

            elif player.vy >= 0:
                # ── Floor hit (falling or standing on platform) ───────────────
                # Carry the player with the platform's horizontal movement so
                # they don't slide off while riding it.
                player.gObj.x  += plat.delta_x
                player.gObj.y   = trect.top - player.gObj.height
                player.vy       = 0
                player.on_ground = True
                player.jump_hold = 0
                player._on_moving_platform = True

            else:
                # ── Ceiling hit (jumping into underside) ──────────────────────
                player.gObj.y = trect.bottom
                player.vy     = 0

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

    def _resolve_projectile_world(self, core, level_data):
        """
        Handles collisions between FireFlowerProjectiles and the static map.

        bounce_x=False: wall contact zeroes vx instead of reversing it.
        FireFlowerProjectile.update() detects vx==0 and deactivates the projectile,
        matching the classic behaviour where fireballs vanish on wall impact.

        Floor contact sets projectile.on_ground=True via _solve_aabb_collision's
        hasattr guard. FireFlowerProjectile.update() reads on_ground and applies
        PROJ_JUMP_VEL to produce the bouncing arc.
        """
        for proj in level_data.projectiles:
            if not proj.gObj.active: continue
            nearby = self._get_tile_rects_near(level_data, proj.gObj)
            self._solve_aabb_collision(proj, nearby, bounce_x=False)

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
        Full static-tile resolution — runs X pass then Y pass in sequence.
        Used by resolve_collisions after moving platforms may have shifted the player.
        """
        self._resolve_player_world_x(core, level_data)
        self._resolve_player_world_y(core, level_data)

    def _resolve_player_world_x(self, core, level_data):
        """
        Wall-only pass. Called after only X has moved so every overlap is
        unambiguously a wall. No ox/oy axis comparison needed or used —
        that comparison is the root cause of the stacked-tile ceiling-snap bug.
        Spikes deferred to the Y pass.
        """
        player = core.player
        if not player: return

        nearby = self._get_tile_rects_near(level_data, player.gObj)
        rect   = player.gObj.get_rect()

        for (row, col, trect, tile_type) in nearby:
            if tile_type == EntityType.SPIKE:
                continue  # handled in Y pass
            if not rect.colliderect(trect):
                continue

            if player.vx > 0:
                player.gObj.x = trect.left - player.gObj.width
            elif player.vx < 0:
                player.gObj.x = trect.right
            else:
                # Stationary (e.g. spawned inside geometry) — use center as tiebreaker
                if rect.centerx <= trect.centerx:
                    player.gObj.x = trect.left - player.gObj.width
                else:
                    player.gObj.x = trect.right
            player.vx = 0
            rect = player.gObj.get_rect()

    def _resolve_player_world_y(self, core, level_data):
        """
        Floor/ceiling-only pass. Called after X is resolved so every remaining
        overlap is unambiguously vertical. vy direction decides floor vs ceiling:
          vy >= 0  →  floor   : snap to top, zero vy, land
          vy <  0  →  ceiling : snap to bottom, zero vy (bonk / qblock)
        Spikes checked here so lethal contact fires on any approach direction.
        """
        player = core.player
        if not player: return

        nearby = self._get_tile_rects_near(level_data, player.gObj)
        rect   = player.gObj.get_rect()

        for (row, col, trect, tile_type) in nearby:
            if not rect.colliderect(trect):
                continue

            if tile_type == EntityType.SPIKE:
                if not player.power_machine.is_invincible:
                    if not player.power_machine.take_hit():
                        core._handle_death("Spike")
                    return
                continue

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

    # --- DYNAMIC COLLISIONS ---

    def _resolve_dynamic_interactions(self, core, player_was_falling):
        """
        Handles interactions between moving entities (Player, Enemies, Coins,
        and Projectiles).

        1. Queries the spatial hash for hazards (Enemies) near the player.
        2. Dispatches collision events if overlaps occur.
        3. Queries for collectibles (Coins/Powerups) and dispatches events.
        4. Checks Enemy-vs-Enemy collisions (to prevent stacking/overlap).
        5. Checks Projectile-vs-Enemy collisions — kills enemy and deactivates
           projectile on contact.
        """
        player = core.player
        if not player: return

        # Use the passed 'player_was_falling' boolean which represents
        # the physics state BEFORE wall/floor resolution.

        nearby_hazards = self.hazard_hash.query(player)
        for obj in nearby_hazards:
            # Skip spikes — their lethal collision is already handled by
            # _resolve_player_world via static_hash. Processing them here
            # would trigger a second _handle_death call, double-decrementing lives.
            t_type = obj.gObj.type_id if hasattr(obj.gObj, 'type_id') else EntityType.NONE
            if t_type == EntityType.SPIKE:
                continue
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

        # --- Projectile vs Enemy ---
        # Iterate active projectiles, query hazard_hash (which holds all active
        # enemies) at the projectile's position. On hit: kill enemy, kill projectile.
        # Guard against double-kills: check active flags before and after each hit
        # so a projectile that already died this frame can't kill a second enemy.
        for proj in core.level_data.projectiles:
            if not proj.gObj.active:
                continue
            nearby_enemies = self.hazard_hash.query_rect(
                proj.gObj.x, proj.gObj.y, proj.gObj.width, proj.gObj.height
            )
            for enemy in nearby_enemies:
                if not enemy.gObj.active:
                    continue
                t_type = enemy.gObj.type_id if hasattr(enemy.gObj, 'type_id') else EntityType.NONE
                if t_type != EntityType.ENEMY:
                    continue   # skip spikes or anything else in hazard_hash
                if proj.gObj.collides_with(enemy.gObj):
                    self._handle_projectile_enemy(core, proj, enemy)
                    break      # projectile is now dead — stop checking further enemies

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
          0. State-machine enemies (e.g. Koopa) -- duck-typed via on_stomp / on_touch.
             These handle their own scoring, bouncing, and damage internally.
          1. Stomp    -- player was falling AND feet above enemy centre -> kill enemy, bounce player.
          2. Star     -- player.power_machine.is_star_active -> kill enemy, no damage to player.
          3. Damage   -- player.power_machine.take_hit():
                          True  -> survived (downgraded or i-frames absorbed)
                          False -> dead, core._handle_death() called by PhysicsManager
        """
        if not enemy.gObj.active:
            return

        # 0. Duck-type dispatch for state-machine enemies (Koopa etc.)
        #    on_stomp / on_touch own all logic for these enemies -- do not fall through.
        if hasattr(enemy, 'on_stomp'):
            player_bottom = player.gObj.y + player.gObj.height
            enemy_center  = enemy.gObj.y  + enemy.gObj.height / 2
            # A flying enemy can rise INTO the player rather than the player falling
            # onto it. In that case player_was_falling is False even though the player
            # is clearly above the enemy. Treat upward enemy movement as equivalent to
            # the player falling for the purposes of stomp detection.
            enemy_rising  = hasattr(enemy, 'vy') and enemy.vy < 0
            if (player_was_falling or enemy_rising) and player_bottom < enemy_center + 10:
                enemy.on_stomp(player, core)
            else:
                enemy.on_touch(player, core)
            return

        player_bottom = player.gObj.y + player.gObj.height
        enemy_center  = enemy.gObj.y  + enemy.gObj.height / 2

        # 1. Stomp -- falling and feet above enemy midpoint
        if player_was_falling and player_bottom < enemy_center + 10:
            player.vy = JUMP_VEL_MIN * 0.6   # bounce
            self._handle_player_kill_enemy(core, enemy)
            return

        # 2. Star -- kill enemy without taking damage
        if player.power_machine.is_star_active:
            self._handle_player_kill_enemy(core, enemy)
            return

        # 3. Take a hit through the state machine
        survived = player.power_machine.take_hit()
        if not survived:
            core._handle_death("Enemy")

    def _handle_player_kill_enemy(self, core, enemy):
        enemy.gObj.active = False
        core.score += 100     
        if hasattr(core, 'kills_step'): core.kills_step += 1

    def _handle_player_coin(self, core, player, coin):
        """
        Collects coin, increments score and coin counters.
        """
        if not coin.collected:
            self._collect_coin(core, coin)

    def _collect_coin(self, core, coin):
        coin.gObj.active = False
        coin.collected = True
        core.score += 10
        core.coins_step += 1
        core.coins_total += 1

    def _handle_projectile_enemy(self, core, proj, enemy):
        """
        Resolves a FireFlowerProjectile hitting an Enemy.

        - Enemy is killed (active = False) and awards 100pts + kills_step counter.
        - Projectile is deactivated so it cannot hit a second enemy the same frame.
        """
        proj.gObj.active  = False
        self._handle_player_kill_enemy(core,enemy)

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
          TILE_PLATFORM (int) — treated as fully solid on all sides (same as TILE_GROUND)
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