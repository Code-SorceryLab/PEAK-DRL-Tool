from __future__ import annotations
import os
import math
import random
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import pygame
from gymnasium import spaces

# ============================================================================
# GAME CONSTANTS
# ============================================================================

# Screen
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# Ship
SHIP_SIZE = 10
SHIP_MAX_SPEED = 20
SHIP_ACCELERATION = 0.2
SHIP_TURN_SPEED = 20

# Asteroids
ASTEROID_MIN_SIZE = 20
ASTEROID_MAX_SIZE = 60
ASTEROID_MIN_SPEED = 1.5
ASTEROID_MAX_SPEED = 4.5
MAX_ASTEROIDS = 50

# Bullets
BULLET_SPEED = 13.0
BULLET_LIFETIME = 180.0
MAX_BULLETS = 20
SHOOT_COOLDOWN = 200

# Debris
DEBRIS_TTL = 50

# Spawning
RESPAWN_DURATION = 120
MIN_SPAWN_DISTANCE = 150

# Danger field
RADAR_DIRECTIONS = 32
RADAR_RINGS = 3
RADAR_TIER_SPACING = 60

# Rendering
MAX_DANGER_FOR_COLOR = 10.0
DANGER_FIELD_ALPHA = 64
DANGER_FIELD_GRID_SIZE = 20

# Physics
UP = np.array([0, -1], dtype=np.float32)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def wrap_position(pos: np.ndarray, width: int, height: int) -> np.ndarray:
    """Wrap position to screen boundaries"""
    return np.array([pos[0] % width, pos[1] % height], dtype=np.float32)

def get_random_velocity(min_speed: float, max_speed: float) -> np.ndarray:
    """Generate random velocity vector"""
    speed = random.uniform(min_speed, max_speed)
    angle = random.uniform(0, 2 * math.pi)
    return np.array([math.cos(angle) * speed, math.sin(angle) * speed], dtype=np.float32)

def wrap_delta(delta: float, screen_dimension: float) -> float:
    """Handle screen wrapping for distance calculations"""
    if abs(delta) > screen_dimension / 2:
        delta = screen_dimension - abs(delta)
        if delta < 0:
            delta = -delta
    return delta

def calculate_wrapped_distance(pos1: np.ndarray, pos2: np.ndarray) -> Tuple[float, np.ndarray]:
    """Calculate distance and delta between two points with screen wrapping"""
    dx = pos1[0] - pos2[0]
    dy = pos1[1] - pos2[1]
    
    # Handle wrapping
    if abs(dx) > SCREEN_WIDTH / 2:
        dx = SCREEN_WIDTH - abs(dx)
        if pos1[0] < pos2[0]:
            dx = -dx
    
    if abs(dy) > SCREEN_HEIGHT / 2:
        dy = SCREEN_HEIGHT - abs(dy)
        if pos1[1] < pos2[1]:
            dy = -dy
    
    distance = math.sqrt(dx * dx + dy * dy)
    delta = np.array([dx, dy], dtype=np.float32)
    
    return distance, delta

# ============================================================================
# GAME OBJECTS
# ============================================================================

@dataclass
class GameObject:
    position: np.ndarray
    velocity: np.ndarray
    radius: float
    active: bool = True

    def move(self, dt: float):
        """Move object and wrap position"""
        if self.active:
            self.position += self.velocity * dt * 60
            self.position = wrap_position(self.position, SCREEN_WIDTH, SCREEN_HEIGHT)

    def collides_with(self, other: 'GameObject') -> bool:
        """Check collision with another game object using wrapped distance"""
        if not (self.active and other.active):
            return False
        distance, _ = calculate_wrapped_distance(self.position, other.position)
        return distance < (self.radius + other.radius)

@dataclass
class Ship(GameObject):
    angle: float = 0.0
    thrust: bool = False
    direction: np.ndarray = None

    def __post_init__(self):
        if self.direction is None:
            self.direction = UP.copy()

    def rotate(self, clockwise: bool = True):
        """Rotate ship"""
        multiplier = 1 if clockwise else -1
        self.angle += SHIP_TURN_SPEED * multiplier
        angle_rad = math.radians(self.angle)
        self.direction = np.array([math.sin(angle_rad), -math.cos(angle_rad)], dtype=np.float32)

    def accelerate(self):
        """Apply thrust"""
        self.thrust = True
        thrust_vector = self.direction * SHIP_ACCELERATION
        self.velocity += thrust_vector
        
        # Cap speed
        speed = np.linalg.norm(self.velocity)
        if speed > SHIP_MAX_SPEED:
            self.velocity = (self.velocity / speed) * SHIP_MAX_SPEED

    def update(self, dt: float):
        """Update ship state"""
        if not self.active:
            return
        
        # Apply drag
        if not self.thrust:
            self.velocity *= 0.995
        
        self.thrust = False

@dataclass  
class Asteroid(GameObject):
    size: int = 3
    rotation: float = 0.0
    rotation_speed: float = 0.0

    def __post_init__(self):
        self.rotation_speed = random.uniform(-2, 2)

    def update(self, dt: float):
        """Update asteroid rotation"""
        if self.active:
            self.rotation += self.rotation_speed

    def split(self) -> List['Asteroid']:
        """Split asteroid into smaller pieces"""
        if self.size <= 1:
            return []
        
        new_size = self.size - 1
        new_radius = ASTEROID_MIN_SIZE + (new_size - 1) * 15
        new_asteroids = []
        
        for _ in range(2):
            speed_multiplier = 1.2
            new_velocity = get_random_velocity(
                ASTEROID_MIN_SPEED * speed_multiplier,
                ASTEROID_MAX_SPEED * speed_multiplier
            )
            new_asteroid = Asteroid(
                position=self.position.copy(),
                velocity=new_velocity,
                radius=new_radius,
                size=new_size
            )
            new_asteroids.append(new_asteroid)
        
        return new_asteroids

@dataclass
class Bullet(GameObject):
    creation_time: int = 0

    def update(self, current_time: int):
        """Deactivate bullet after lifetime expires"""
        if self.active and current_time - self.creation_time > BULLET_LIFETIME:
            self.active = False

@dataclass
class Debris(GameObject):
    creation_time: int = 0

    def update(self, current_time: int):
        """Deactivate debris after TTL expires"""
        if self.active and current_time - self.creation_time > DEBRIS_TTL:
            self.active = False

# ============================================================================
# MAIN GAME CLASS
# ============================================================================

class AsteroidsCore:
    WIDTH = SCREEN_WIDTH
    HEIGHT = SCREEN_HEIGHT

    def __init__(self, render_mode: str = "none", **kwargs):
        # Game configuration
        self.initial_asteroids = int(kwargs.pop("initial_asteroids", 4))
        self.max_lives = int(kwargs.pop("max_lives", 3))
        self.shoot_cooldown_time = int(kwargs.pop("shoot_cooldown", SHOOT_COOLDOWN))
        
        # Game state
        self.score = 0
        self.lives = self.max_lives
        self.level = 1
        self.alive = True
        self.game_over = False
        self.frame_count = 0
        
        # Game objects
        self.ship: Ship = None
        self.asteroids: List[Asteroid] = []
        self.bullets: List[Bullet] = []
        self.debris: List[Debris] = []
        
        # Timers
        self.shoot_timer = 0
        self.respawn_timer = 0
        
        # Per-step tracking
        self.asteroids_destroyed_this_step = 0
        self.bullets_fired_this_step = 0
        self.collision_this_step = False
        self.shots_hit_this_step = 0
        self.ship_destroyed_this_step = False
        self.score_delta = 0
        self.last_score = 0
        
        # Danger field state
        self.current_danger = 0.0
        self.current_gradient = np.zeros(2, dtype=np.float32)
        self.radar_samples = [0.0] * (RADAR_DIRECTIONS * RADAR_RINGS)
        self.last_targeting_bonus = 0.0
        
        # Observation configuration
        self.obs_asteroid = 3
        obs_len = (
            6                                  # Ship: pos(2) + vel(2) + angle(1) + alive(1)
            + (self.obs_asteroid * 5)          # Asteroids: pos(2) + vel(2) + size(1) each
            + 2                                # Score + targeting bonus
            + 1                                # Danger field value
            + 2                                # Danger gradient (x, y)
            + (RADAR_DIRECTIONS * RADAR_RINGS) # Radar samples
        )
        self._obs_space = spaces.Box(-np.inf, np.inf, shape=(obs_len,), dtype=np.float32)
        self._act_space = spaces.Discrete(5)
        
        # RNG
        self.rng = np.random.RandomState(1337)
        
        # Initialize pygame
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()
        self._surf = pygame.Surface((self.WIDTH, self.HEIGHT))
        
        self.reset()

    # ========================================================================
    # PUBLIC API
    # ========================================================================

    def get_action_space(self):
        return self._act_space

    def get_observation_space(self):
        return self._obs_space

    def reset(self):
        """Reset game to initial state"""
        # Game state
        self.score = 0
        self.lives = self.max_lives
        self.level = 1
        self.alive = True
        self.game_over = False
        self.frame_count = 0
        
        # Create initial objects
        self._create_ship()
        self._spawn_asteroids(self.initial_asteroids)
        
        # Clear collections
        self.bullets.clear()
        self.debris.clear()
        
        # Reset timers
        self.shoot_timer = 0
        self.respawn_timer = 0
        
        # Reset per-step tracking
        self.asteroids_destroyed_this_step = 0
        self.bullets_fired_this_step = 0
        self.collision_this_step = False
        self.shots_hit_this_step = 0
        self.ship_destroyed_this_step = False
        self.score_delta = 0
        self.last_score = 0
        
        # Reset danger field
        self.current_danger = 0.0
        self.current_gradient = np.zeros(2, dtype=np.float32)
        self.radar_samples = [0.0] * (RADAR_DIRECTIONS * RADAR_RINGS)
        self.last_targeting_bonus = 0.0
        
        return self._obs()

    def step(self, action: int):
        """Execute one game step"""
        if not self.alive:
            return self._obs(), 0.0, True, {"episode_end": True, "won": False}

        dt = 1.0 / 60.0
        self.frame_count += 1

        # Reset per-step tracking
        self._reset_step_tracking()
        
        # Update timers
        self._update_timers()

        # Process action
        self._process_action(action)

        # Update all objects
        self._update_all_objects(dt)

        # Clean up inactive objects
        self._cleanup_inactive_objects()
        
        # Check collisions
        self._check_collisions()
        
        # Check level completion
        self._check_level_completion()
        
        # Check game over
        if self.lives <= 0:
            self.alive = False
            self.game_over = True
        
        # Update score tracking
        self.score_delta = self.score - self.last_score
        self.last_score = self.score
        
        # Calculate danger field and metrics
        self._update_danger_field()
        
        # Build info dict
        info = self._build_info_dict()
        
        base_reward = 1.0
        terminated = not self.alive

        return self._obs(), float(base_reward), bool(terminated), info

    # ========================================================================
    # PRIVATE: INITIALIZATION
    # ========================================================================

    def _create_ship(self):
        """Create ship at center of screen"""
        position = np.array([SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2], dtype=np.float32)
        velocity = np.array([0, 0], dtype=np.float32)
        self.ship = Ship(position=position, velocity=velocity, radius=SHIP_SIZE)
        self.respawn_timer = RESPAWN_DURATION

    def _spawn_asteroids(self, count: int):
        """Spawn asteroids away from ship"""
        self.asteroids.clear()
        
        for _ in range(count):
            position = self._find_safe_spawn_position()
            velocity = get_random_velocity(ASTEROID_MIN_SPEED, ASTEROID_MAX_SPEED)
            radius = ASTEROID_MIN_SIZE + (3 - 1) * 15
            
            asteroid = Asteroid(
                position=position,
                velocity=velocity,
                radius=radius,
                size=3
            )
            self.asteroids.append(asteroid)

    def _find_safe_spawn_position(self) -> np.ndarray:
        """Find spawn position away from ship"""
        for _ in range(50):
            x = self.rng.uniform(0, SCREEN_WIDTH)
            y = self.rng.uniform(0, SCREEN_HEIGHT)
            
            if self.ship is None:
                break
            
            distance, _ = calculate_wrapped_distance(
                np.array([x, y]), 
                self.ship.position
            )
            
            if distance > MIN_SPAWN_DISTANCE:
                break
        
        return np.array([x, y], dtype=np.float32)

    # ========================================================================
    # PRIVATE: STEP PROCESSING
    # ========================================================================

    def _reset_step_tracking(self):
        """Reset per-step counters"""
        self.asteroids_destroyed_this_step = 0
        self.bullets_fired_this_step = 0
        self.collision_this_step = False
        self.shots_hit_this_step = 0
        self.ship_destroyed_this_step = False

    def _update_timers(self):
        """Decrement all timers"""
        self.shoot_timer = max(0, self.shoot_timer - 1)
        self.respawn_timer = max(0, self.respawn_timer - 1)

    def _process_action(self, action: int):
        """Process player action"""
        if not self.ship or not self.ship.active:
            return
        
        a = int(action)
        
        if a == 1:  # Rotate left
            self.ship.rotate(clockwise=False)
        elif a == 2:  # Rotate right
            self.ship.rotate(clockwise=True)
        elif a == 3:  # Thrust
            self.ship.accelerate()
        elif a == 4:  # Shoot
            self._try_shoot()

    def _try_shoot(self):
        """Attempt to fire bullet"""
        if self.shoot_timer > 0:
            return
        
        if len([b for b in self.bullets if b.active]) >= MAX_BULLETS:
            return
        
        bullet_pos = self.ship.position + self.ship.direction * (SHIP_SIZE + 5)
        bullet_vel = self.ship.direction * BULLET_SPEED + self.ship.velocity
        
        bullet = Bullet(
            position=bullet_pos,
            velocity=bullet_vel,
            radius=2,
            creation_time=self.frame_count
        )
        self.bullets.append(bullet)
        self.shoot_timer = self.shoot_cooldown_time // 17
        self.bullets_fired_this_step = 1

    def _update_all_objects(self, dt: float):
        """Update all game objects"""
        # Update ship
        if self.ship:
            self.ship.update(dt)
            self.ship.move(dt)
        
        # Update asteroids
        for asteroid in self.asteroids:
            asteroid.update(dt)
            asteroid.move(dt)
        
        # Update bullets
        for bullet in self.bullets:
            bullet.update(self.frame_count)
            if bullet.active:
                bullet.move(dt)
        
        # Update debris
        for debris_piece in self.debris:
            debris_piece.update(self.frame_count)
            if debris_piece.active:
                debris_piece.move(dt)

    def _cleanup_inactive_objects(self):
        """Remove inactive bullets and debris"""
        self.bullets = [b for b in self.bullets if b.active]
        self.debris = [d for d in self.debris if d.active]

    def _check_collisions(self):
        """Check and handle all collisions"""
        # Ship-asteroid collisions
        if self.ship and self.ship.active and self.respawn_timer <= 0:
            for asteroid in self.asteroids:
                if asteroid.active and self.ship.collides_with(asteroid):
                    self._destroy_ship()
                    break
        
        # Bullet-asteroid collisions
        for bullet in self.bullets[:]:
            if not bullet.active:
                continue
            
            for asteroid in self.asteroids[:]:
                if not asteroid.active:
                    continue
                
                if bullet.collides_with(asteroid):
                    bullet.active = False
                    self._destroy_asteroid(asteroid)
                    self.shots_hit_this_step += 1
                    break

    def _check_level_completion(self):
        """Check if level is complete and spawn new asteroids"""
        if not self.asteroids and self.alive:
            self.level += 1
            self.score += 100
            self._spawn_asteroids(self.initial_asteroids + self.level - 1)

    def _update_danger_field(self):
        """Update danger field and radar samples"""
        self.current_danger, self.current_gradient = self._calculate_danger_gradient()
        self.radar_samples = self._get_radar_samples()

    # ========================================================================
    # PRIVATE: DESTRUCTION
    # ========================================================================

    def _destroy_ship(self):
        """Destroy ship and handle respawn/game over"""
        if not self.ship:
            return
        
        self.ship_destroyed_this_step = True
        self.collision_this_step = True
        self.lives -= 1
        
        # Create debris
        self._create_debris(self.ship.position, count=8, spread=10)
        
        # Respawn or end game
        if self.lives > 0:
            self._create_ship()
        else:
            self.ship = None

    def _destroy_asteroid(self, asteroid: Asteroid):
        """Destroy asteroid and handle splitting"""
        self.asteroids_destroyed_this_step += 1
        
        # Award points
        score_values = {3: 20, 2: 50, 1: 100}
        self.score += score_values.get(asteroid.size, 20)
        
        # Create debris
        self._create_debris(asteroid.position, count=4, spread=5)
        
        # Split asteroid
        new_asteroids = asteroid.split()
        self.asteroids.extend(new_asteroids)
        
        # Deactivate
        asteroid.active = False
        self.asteroids = [a for a in self.asteroids if a.active]

    def _create_debris(self, position: np.ndarray, count: int, spread: float):
        """Create debris particles"""
        for _ in range(count):
            debris_piece = Debris(
                position=position.copy() + np.random.uniform(-spread, spread, 2),
                velocity=np.random.uniform(-3, 3, 2),
                radius=1,
                creation_time=self.frame_count
            )
            self.debris.append(debris_piece)

    # ========================================================================
    # PRIVATE: OBSERVATION & METRICS
    # ========================================================================

    def _obs(self) -> np.ndarray:
        """Build observation vector"""
        obs = []
        
        # Ship observation
        obs.extend(self._get_ship_obs())
        
        # Asteroid observations
        obs.extend(self._get_asteroids_obs())
        
        # Score and targeting
        obs.append(self.score / 10000.0)
        targeting_bonus, _ = self._calculate_targeting_info()
        obs.append(targeting_bonus)
        
        # Danger field
        obs.append(self.current_danger / 10.0)
        obs.extend(self.current_gradient)
        
        # Radar
        obs.extend(self.radar_samples)
        
        return np.array(obs, dtype=np.float32)

    def _get_ship_obs(self) -> List[float]:
        """Get ship observation"""
        if self.ship and self.ship.active:
            return [
                self.ship.position[0] / SCREEN_WIDTH,
                self.ship.position[1] / SCREEN_HEIGHT,
                self.ship.velocity[0] / SHIP_MAX_SPEED,
                self.ship.velocity[1] / SHIP_MAX_SPEED,
                (self.ship.angle % 360) / 360.0,
                1.0
            ]
        else:
            return [0.5, 0.5, 0.0, 0.0, 0.0, 0.0]

    def _get_asteroids_obs(self) -> List[float]:
        """Get asteroid observations sorted by distance"""
        sorted_asteroids = self._get_sorted_asteroids()
        obs = []
        
        for i in range(self.obs_asteroid):
            if i < len(sorted_asteroids):
                ast = sorted_asteroids[i]
                obs.extend([
                    ast.position[0] / SCREEN_WIDTH,
                    ast.position[1] / SCREEN_HEIGHT,
                    ast.velocity[0] / ASTEROID_MAX_SPEED,
                    ast.velocity[1] / ASTEROID_MAX_SPEED,
                    ast.size / 3.0
                ])
            else:
                obs.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        
        return obs

    def _get_sorted_asteroids(self) -> List[Asteroid]:
        """Get asteroids sorted by distance to ship"""
        if not self.ship or not self.ship.active:
            return [a for a in self.asteroids if a.active][:MAX_ASTEROIDS]
        
        distances_and_asteroids = []
        for asteroid in self.asteroids:
            if not asteroid.active:
                continue
            distance, _ = calculate_wrapped_distance(self.ship.position, asteroid.position)
            distances_and_asteroids.append((distance, asteroid))
        
        distances_and_asteroids.sort(key=lambda x: x[0])
        return [ast for _, ast in distances_and_asteroids[:MAX_ASTEROIDS]]

    def _get_closest_distances(self, n: int = 3) -> List[float]:
        """Get distances to N closest asteroids"""
        sorted_asteroids = self._get_sorted_asteroids()
        
        if not self.ship or not self.ship.active:
            return [SCREEN_WIDTH] * n
        
        distances = []
        for asteroid in sorted_asteroids[:n]:
            distance, _ = calculate_wrapped_distance(self.ship.position, asteroid.position)
            distances.append(distance)
        
        while len(distances) < n:
            distances.append(SCREEN_WIDTH)
        
        return distances

    def _calculate_targeting_info(self) -> Tuple[float, float]:
        """Calculate targeting bonus and angle to nearest asteroid"""
        sorted_asteroids = self._get_sorted_asteroids()
        
        if not sorted_asteroids or not self.ship or not self.ship.active:
            return 0.0, 180.0
        
        closest_ast = sorted_asteroids[0]
        
        # Calculate ship's forward direction
        ship_angle_rad = math.radians(self.ship.angle % 360)
        ship_front = np.array([math.sin(ship_angle_rad), -math.cos(ship_angle_rad)])
        
        # Calculate direction to asteroid
        _, delta = calculate_wrapped_distance(closest_ast.position, self.ship.position)
        
        if np.linalg.norm(delta) < 1e-6:
            return 0.0, 180.0
        
        to_asteroid = delta / (np.linalg.norm(delta) + 1e-8)
        
        # Calculate angle difference
        dot = np.clip(np.dot(ship_front, to_asteroid), -1.0, 1.0)
        angle_diff_rad = math.acos(dot)
        angle_diff = math.degrees(angle_diff_rad)
        
        # Calculate bonus
        targeting_bonus = 4.0 * max(math.cos(angle_diff_rad), -0.125)
        
        return targeting_bonus, angle_diff

    def _build_info_dict(self) -> dict:
        """Build info dictionary with essential metrics only"""
        closest_3_distances = self._get_closest_distances(3)
        targeting_bonus, angle_diff = self._calculate_targeting_info()
        targeting_bonus_delta = targeting_bonus - self.last_targeting_bonus
        self.last_targeting_bonus = targeting_bonus
        
        return {
            # Core game state
            "score": self.score,
            "score_delta": self.score_delta,
            "lives": self.lives,
            "level": self.level,
            
            # Per-step events
            "collision": self.collision_this_step,
            "ship_destroyed": self.ship_destroyed_this_step,
            "asteroids_destroyed": self.asteroids_destroyed_this_step,
            "bullets_fired": self.bullets_fired_this_step,
            "shots_hit": self.shots_hit_this_step,
            
            # Spatial awareness
            "distances_to_closest_3": closest_3_distances,
            "targeting_bonus": targeting_bonus,
            "targeting_bonus_delta": targeting_bonus_delta,
            "angle_to_nearest": angle_diff,
            
            # Object counts
            "asteroids_remaining": len([a for a in self.asteroids if a.active]),
            "bullets_active": len([b for b in self.bullets if b.active]),
            
            # Ship state (if alive)
            "ship_speed": np.linalg.norm(self.ship.velocity) if self.ship and self.ship.active else 0.0,
            "ship_velocity": self.ship.velocity.copy() if self.ship and self.ship.active else np.array([0.0, 0.0]),
            "ship_angle": self.ship.angle if self.ship else 0.0,
            
            # Danger field (already in observation, but useful for logging)
            "danger_field": self.current_danger,
            "danger_gradient_magnitude": np.linalg.norm(self.current_gradient),
            
            # Episode status
            "terminated": not self.alive,
            "level_completed": len(self.asteroids) == 0 and self.alive,
        }


    # ========================================================================
    # PRIVATE: DANGER FIELD
    # ========================================================================

    def _calculate_danger_at_point(self, point: np.ndarray) -> Tuple[float, np.ndarray]:
        """Calculate danger field and gradient at any point"""
        total_danger = 0.0
        gradient = np.zeros(2, dtype=np.float32)
        
        for asteroid in self.asteroids:
            if not asteroid.active:
                continue
            
            ast_pos = wrap_position(asteroid.position, SCREEN_WIDTH, SCREEN_HEIGHT)
            _, delta = calculate_wrapped_distance(point, ast_pos)
            dist = np.linalg.norm(delta) + 1e-6
            
            danger_radius = asteroid.radius * 3
            
            if dist < danger_radius:
                # Potential field: danger = k / dist²
                danger = (asteroid.size / 3.0) * (danger_radius / dist) ** 2
                total_danger += danger
                
                # Gradient points away from asteroid
                grad_magnitude = 2 * danger / dist
                gradient += grad_magnitude * (delta / dist)
        
        # Normalize gradient
        grad_magnitude = np.linalg.norm(gradient)
        if grad_magnitude > 1e-6:
            gradient_normalized = gradient / grad_magnitude
        else:
            gradient_normalized = np.zeros(2, dtype=np.float32)
        
        return total_danger, gradient_normalized

    def _calculate_danger_gradient(self) -> Tuple[float, np.ndarray]:
        """Calculate danger at ship's current position"""
        if not self.ship or not self.ship.active:
            return 0.0, np.zeros(2, dtype=np.float32)
        
        return self._calculate_danger_at_point(self.ship.position)

    def _get_radar_samples(self) -> List[float]:
        """Sample danger at multiple rings × directions"""
        if not self.ship or not self.ship.active:
            return [0.0] * (RADAR_DIRECTIONS * RADAR_RINGS)
        
        samples = []
        sample_distances = np.arange(
            RADAR_TIER_SPACING, 
            (RADAR_RINGS + 1) * RADAR_TIER_SPACING, 
            RADAR_TIER_SPACING
        )
        
        for distance in sample_distances:
            for angle_deg in np.linspace(0, 360, RADAR_DIRECTIONS, endpoint=False):
                angle_rad = math.radians(angle_deg + self.ship.angle)
                
                test_pos = self.ship.position + np.array([
                    math.sin(angle_rad) * distance,
                    -math.cos(angle_rad) * distance
                ])
                test_pos = wrap_position(test_pos, SCREEN_WIDTH, SCREEN_HEIGHT)
                
                danger, _ = self._calculate_danger_at_point(test_pos)
                samples.append(danger / 10.0)
        
        return samples

    # ========================================================================
    # RENDERING
    # ========================================================================

    def render(self, surface: pygame.Surface, blit_only: bool = True, 
            show_danger_field: bool = False, show_radar: bool = True):
        """Render game state"""
        surface.fill((0, 0, 0))
        
        # Render danger field first (background layer)
        if show_danger_field:
            self._render_danger_field(surface)
        
        # Render radar (separate overlay)
        if show_radar:
            self._render_radar(surface)
        
        # Render game objects
        self._render_asteroids(surface)
        self._render_bullets(surface)
        self._render_debris(surface)
        self._render_ship(surface)
        
        # Render UI
        self._render_ui(surface, show_danger_field, show_radar)

    def _render_danger_field(self, surface: pygame.Surface):
        """Render danger field visualization"""
        if not self.ship or not self.ship.active:
            return
        
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        
        # Sample danger at grid points
        for x in range(0, SCREEN_WIDTH, DANGER_FIELD_GRID_SIZE):
            for y in range(0, SCREEN_HEIGHT, DANGER_FIELD_GRID_SIZE):
                test_pos = np.array([x, y], dtype=np.float32)
                danger, _ = self._calculate_danger_at_point(test_pos)
                danger_ratio = min(danger / MAX_DANGER_FOR_COLOR, 1.0)
                color = self._danger_to_color(danger_ratio)
                color_with_alpha = (*color, DANGER_FIELD_ALPHA)
                
                # Draw transparent dot
                dot_radius = int(3 + danger_ratio * 5)
                pygame.draw.circle(overlay, color_with_alpha, (x, y), dot_radius)
        
        surface.blit(overlay, (0, 0))

    def _render_radar(self, surface: pygame.Surface):
        """Render radar visualization"""
        if not self.ship or not self.ship.active:
            return
        
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        
        ship_center = (int(self.ship.position[0]), int(self.ship.position[1]))
        sample_distances = np.arange(
            RADAR_TIER_SPACING,
            (RADAR_RINGS + 1) * RADAR_TIER_SPACING,
            RADAR_TIER_SPACING
        )
        
        for dir_idx, angle_deg in enumerate(np.linspace(0, 360, RADAR_DIRECTIONS, endpoint=False)):
            for ring_idx, distance in enumerate(sample_distances):
                angle_rad = math.radians(angle_deg + self.ship.angle)
                end_x = ship_center[0] + math.sin(angle_rad) * distance
                end_y = ship_center[1] - math.cos(angle_rad) * distance
                
                # Get danger sample
                sample_idx = ring_idx * RADAR_DIRECTIONS + dir_idx
                danger_at_direction = self.radar_samples[sample_idx] * 10.0
                danger_ratio = min(danger_at_direction / MAX_DANGER_FOR_COLOR, 1.0)
                color = self._danger_to_color(danger_ratio)
                
                # Vary appearance by ring
                line_width = max(1, 4 - ring_idx)
                alpha_value = max(80, 200 - ring_idx * 40)
                color_with_alpha = (*color, alpha_value)
                
                # Draw radar line
                pygame.draw.line(overlay, color_with_alpha, ship_center, 
                            (int(end_x), int(end_y)), line_width)
                
                # Draw circle at sample point
                circle_radius = max(2, 7 - ring_idx)
                pygame.draw.circle(overlay, color_with_alpha, 
                                (int(end_x), int(end_y)), circle_radius)
        
        surface.blit(overlay, (0, 0))


    def _danger_to_color(self, danger_ratio: float) -> Tuple[int, int, int]:
        """Convert danger ratio (0-1) to color gradient"""
        danger_ratio = np.clip(danger_ratio, 0.0, 1.0)
        
        if danger_ratio < 0.5:
            # Green to Yellow
            t = danger_ratio * 2
            r = int(255 * t)
            g = 255
            b = 0
        else:
            # Yellow to Red
            t = (danger_ratio - 0.5) * 2
            r = 255
            g = int(255 * (1 - t))
            b = 0
        
        return (r, g, b)

    def _render_asteroids(self, surface: pygame.Surface):
        """Render asteroids"""
        for asteroid in self.asteroids:
            if not asteroid.active:
                continue
            
            center = (int(asteroid.position[0]), int(asteroid.position[1]))
            radius = int(asteroid.radius)
            
            # Draw circle
            pygame.draw.circle(surface, (200, 200, 200), center, radius, 2)
            
            # Draw rotation indicator
            angle_rad = math.radians(asteroid.rotation)
            end_x = center[0] + math.cos(angle_rad) * radius * 0.8
            end_y = center[1] + math.sin(angle_rad) * radius * 0.8
            pygame.draw.line(surface, (200, 200, 200), center, (int(end_x), int(end_y)), 1)

    def _render_bullets(self, surface: pygame.Surface):
        """Render bullets"""
        for bullet in self.bullets:
            if bullet.active:
                center = (int(bullet.position[0]), int(bullet.position[1]))
                pygame.draw.circle(surface, (255, 255, 255), center, 2)

    def _render_debris(self, surface: pygame.Surface):
        """Render debris"""
        for debris_piece in self.debris:
            if debris_piece.active:
                center = (int(debris_piece.position[0]), int(debris_piece.position[1]))
                pygame.draw.circle(surface, (150, 150, 150), center, 1)

    def _render_ship(self, surface: pygame.Surface):
        """Render ship"""
        if not self.ship or not self.ship.active:
            return
        
        # Flicker during respawn invulnerability
        if self.respawn_timer > 0 and (self.frame_count // 5) % 2 != 0:
            return
        
        center = self.ship.position
        angle_rad = math.radians(self.ship.angle)
        
        # Calculate ship triangle points
        nose = center + np.array([
            math.sin(angle_rad) * SHIP_SIZE,
            -math.cos(angle_rad) * SHIP_SIZE
        ])
        left_rear = center + np.array([
            math.sin(angle_rad + 2.5) * SHIP_SIZE * 0.7,
            -math.cos(angle_rad + 2.5) * SHIP_SIZE * 0.7
        ])
        right_rear = center + np.array([
            math.sin(angle_rad - 2.5) * SHIP_SIZE * 0.7,
            -math.cos(angle_rad - 2.5) * SHIP_SIZE * 0.7
        ])
        
        points = [
            (int(nose[0]), int(nose[1])),
            (int(left_rear[0]), int(left_rear[1])),
            (int(right_rear[0]), int(right_rear[1]))
        ]
        pygame.draw.polygon(surface, (255, 255, 255), points, 2)
        
        # Draw thrust flame
        if self.ship.thrust:
            thrust_end = center - np.array([
                math.sin(angle_rad) * SHIP_SIZE * 1.5,
                -math.cos(angle_rad) * SHIP_SIZE * 1.5
            ])
            thrust_points = [
                (int(left_rear[0]), int(left_rear[1])),
                (int(thrust_end[0]), int(thrust_end[1])),
                (int(right_rear[0]), int(right_rear[1]))
            ]
            pygame.draw.polygon(surface, (255, 100, 100), thrust_points)

    def _render_ui(self, surface: pygame.Surface, show_danger_field: bool, show_radar: bool):
        """Render UI text"""
        font = pygame.font.Font(None, 36)
        small_font = pygame.font.Font(None, 24)
        
        # Score, lives, level
        score_text = font.render(f"SCORE: {self.score:06d}", True, (255, 255, 255))
        surface.blit(score_text, (10, 10))
        
        lives_text = font.render(f"LIVES: {max(0, self.lives)}", True, (255, 255, 255))
        surface.blit(lives_text, (10, 50))
        
        level_text = font.render(f"LEVEL: {self.level}", True, (255, 255, 255))
        surface.blit(level_text, (10, 90))
        
        # Danger field info
        if show_danger_field:
            danger_text = font.render(f"DANGER: {self.current_danger:.2f}", True, (255, 255, 0))
            surface.blit(danger_text, (10, 130))
            
            grad_text = small_font.render(
                f"GRAD: ({self.current_gradient[0]:+.2f}, {self.current_gradient[1]:+.2f})", 
                True, (255, 255, 0)
            )
            surface.blit(grad_text, (10, 170))
            
            if show_radar:
                radar_text = small_font.render(
                    f"RADAR: F:{self.radar_samples[0]:.1f} R:{self.radar_samples[2]:.1f} "
                    f"B:{self.radar_samples[4]:.1f} L:{self.radar_samples[6]:.1f}", 
                    True, (255, 255, 0)
                )
                surface.blit(radar_text, (10, 195))
        
        # Game over
        if self.game_over:
            game_over_text = font.render("GAME OVER", True, (255, 0, 0))
            text_rect = game_over_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2))
            surface.blit(game_over_text, text_rect)
