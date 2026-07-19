from __future__ import annotations
import os
import math
import importlib
import heapq
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional, Set
import numpy as np
import pygame
import time
import gymnasium
from gymnasium import spaces
import random
import psutil

# FIX: Import the Class 'EntityType' explicitly so EntityType.GOAL works
from .modules.System.EntityType import EntityType

# Objects
from .modules.Objects.GameObject import GameObject
from .modules.Objects.Tile import Tile, create_tile
from .modules.Objects.Player import Player
from .modules.Objects.Enemy import Enemy
from .modules.Objects.Powerup import Powerup
from .modules.Objects.Coin import Coin
from .modules.Objects.QuestionBlock import QuestionBlock
from .modules.Objects.FireFlowerProjectile import FireFlowerProjectile

# System
from .modules.System.LevelLoader import LevelLoader, LevelData
from .modules.System.PhysicsManager import PhysicsManager
from .modules.System.config_manager import ConfigManager
from .modules.System.debugging_mods.manager import DebugManager
from .modules.Objects.Spike import Spike
from .modules.Objects.MovingPlatform import MovingPlatform

# Parameters
from .modules.Parameters.Map_parameters import(TILE_AIR, TILE_GROUND, TILE_PLATFORM, TILE_GOAL, TILE_SPIKE, TILE_QBLOCK,
    COLOR_SKY, COLOR_GROUND, COLOR_PLATFORM, COLOR_GOAL, COLOR_SPIKE,
    COLOR_WHITE, COLOR_BLACK, COLOR_QBLOCK, COLOR_EMPTY, COLOR_ENEMY,
    COLOR_POWERUP_MUSH, COLOR_POWERUP_STAR, COLOR_COIN, COLOR_HITBOX,
    COLOR_SENSOR, COLOR_AGENT_PANEL, COLOR_STREAK, TILE_SIZE)

# =============================================================================
# Dijkstra Pathfinding Helper (Global Distance Map)
# =============================================================================
class DijkstraSolver:
    """
    Computes a 'heatmap' of distances from the Goal to every reachable tile.
    """
    def __init__(self, grid: List[List[int]], rows: int, cols: int):
        self.grid = grid
        self.rows = rows
        self.cols = cols
        # Initialize with Infinity
        self.dist_map = np.full((rows, cols), float('inf'), dtype=np.float32)

    def compute_map(self, goals: List[Tuple[int, int]], coins: Set[Tuple[int, int]] = None):
        """
        Physics-aware Dijkstra flood-fill from every goal tile.

        Cost model (tuned to ~64px tiles, 800px/s jump, 1200px/s² gravity):
          Horizontal movement   : 2.0   (free running)
          Downward movement     : 1.2   (gravity-assisted, cheap)
          Upward movement       : 3.5   (jump required, expensive)
          Upward with no ground : +3.0  (floating tile, nearly unreachable)
          On-platform tiles     : -0.6  (natural landing spots, strong discount)
          Near-platform tiles   : -0.25 / -0.1
          Spike tile            : +18.0
          Coin tile             : -0.8

        KEY FIX for vertical levels: upward steps cost 3.5 instead of 2.0.
        Without this penalty the solver routes straight up through open air at
        the same cost as running right, producing a gradient that leads the agent
        to stall mid-air chasing an impossible straight-up path. Heavier upward
        costs make paths naturally flow along platforms then jump.

        BUG FIX: ground-proximity checks were elif-chained so the 2-tile and
        3-tile bonuses were dead code (only fired at the bottom row of the map).
        Changed to independent if-checks so all three levels apply.
        """
        if coins is None:
            coins = set()

        MAX_JUMP_TILES = 6   # realistic max jump height: v0²/(2g)/TILE_SIZE ≈ 4.2, +hold ≈ 5.4
        SOLID = {TILE_GROUND, TILE_PLATFORM, TILE_QBLOCK}

        pq = []
        for gx, gy in goals:
            if 0 <= gx < self.cols and 0 <= gy < self.rows:
                self.dist_map[gy][gx] = 0.0
                heapq.heappush(pq, (0.0, gx, gy))

        # 8-way: (dx, dy) — dy<0 is UP in screen coordinates (y increases downward)
        directions = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1)]

        while pq:
            current_dist, cx, cy = heapq.heappop(pq)
            if current_dist > self.dist_map[cy][cx]:
                continue

            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < self.cols and 0 <= ny < self.rows):
                    continue

                tile = self.grid[ny][nx]
                if tile not in (TILE_AIR, TILE_GOAL, TILE_SPIKE):
                    continue   # wall — impassable

                # --- Directional base cost ---
                if dy < 0:          # moving UP — requires a jump
                    step_cost = 3.5
                elif dy > 0:        # moving DOWN — gravity-assisted
                    step_cost = 1.2
                else:               # horizontal
                    step_cost = 2.0

                if dx != 0 and dy != 0:
                    step_cost *= 1.1   # diagonal slightly more expensive

                # --- Spike penalty ---
                if tile == TILE_SPIKE:
                    step_cost += 18.0

                # --- Coin attraction ---
                if (nx, ny) in coins:
                    step_cost -= 0.8

                # --- Ground-proximity discount (independent if-checks, not elif) ---
                if ny + 1 < self.rows and self.grid[ny + 1][nx] in SOLID:
                    step_cost -= 1.0
                if ny + 2 < self.rows and self.grid[ny + 2][nx] in SOLID:
                    step_cost -= 0.5
                if ny + 3 < self.rows and self.grid[ny + 3][nx] in SOLID:
                    step_cost -= 0.2

                # --- Floating-tile penalty for upward steps ---
                # If moving up and there is no solid surface within jump height
                # below the target, the tile is likely only reachable mid-air
                # (e.g. inside a tall open shaft). Extra cost discourages routing
                # through tiles the agent can't easily reach or land on.
                if dy < 0:
                    has_ground_nearby = any(
                        ny + k < self.rows and self.grid[ny + k][nx] in SOLID
                        for k in range(1, MAX_JUMP_TILES + 1)
                    )
                    if not has_ground_nearby:
                        step_cost += 3.0

                step_cost = max(1.0, step_cost)
                new_dist  = current_dist + step_cost
                if new_dist < self.dist_map[ny][nx]:
                    self.dist_map[ny][nx] = new_dist
                    heapq.heappush(pq, (new_dist, nx, ny))

    def get_dist(self, x: int, y: int) -> float:
        if 0 <= x < self.cols and 0 <= y < self.rows:
            d = self.dist_map[y][x]
            # Return -1.0 if unreachable/infinite so the observation is clean
            return d if d != float('inf') else -1.0
        return -1.0

# =============================================================================
# Jump Arc Precomputation (Physics-Aware Dijkstra Boost)
# =============================================================================
class JumpArcComputer:
    """
    Precomputes physically-accurate jump parabolas at multiple speed tiers
    and directions, then at runtime applies an ADDITIVE boost to the
    Dijkstra advantage channel on tiles the player can reach via jump.

    This nudges the gradient toward physically-traversable paths without
    destroying the existing signal. Unlike multiplicative masking, additive
    boosting works correctly on the signed [-1, 1] Dijkstra range:
      - Negative tiles become slightly less negative (less repulsive)
      - Positive tiles become slightly more positive (more attractive)
      - Tiles not on any arc are completely untouched

    Arc encoding (internal):
        0.0  — not reachable by any jump arc
        0.5  — in-flight tile (arc passes through, no landing here)
        1.0  — valid landing tile (arc descending + solid ground below)
    """

    SPEED_FRACTIONS = [0.0, 0.5, 1.0]   # standing, walking, sprinting
    MAX_SIM_FRAMES  = 200                # ~3.3s at 60fps
    DT              = 1.0 / 60.0

    def __init__(self, physics_context, tile_size: int, obs_half: int):
        self.tile_size = tile_size
        self.obs_half = obs_half
        self.arcs: dict = {}
        self._precompute(physics_context)

    def recompute(self, physics_context):
        """Re-precompute arcs after physics constants change (e.g. config reload)."""
        self.arcs.clear()
        self._precompute(physics_context)

    def _precompute(self, ctx):
        for direction in (-1, 1):
            for frac in self.SPEED_FRACTIONS:
                speed = ctx.MAX_RUN_SPEED * frac
                arc = self._simulate_arc(direction, speed, ctx)
                self.arcs[(direction, frac)] = arc

    def _simulate_arc(self, direction, speed, ctx):
        """
        Simulate one max-hold jump arc. Returns list of (dx, dy, is_descending)
        tile offsets visited, in traversal order.

        Physics replicated from Player.update() + Player.handle_jump():
          1. hold phase:  vy -= GRAVITY * 0.12 * dt   (for JUMP_HOLD_FRAMES)
          2. gravity:     grav = FAST_FALL if vy>0 else GRAVITY
                          vy = min(vy + grav*dt, MAX_FALL_SPEED)
          3. position:    x += vx*dt,  y += vy*dt
        """
        ts = self.tile_size
        half = self.obs_half

        vx = speed * direction
        bonus = min(2.2, abs(speed) * ctx.SPEED_JUMP_BONUS)
        vy = ctx.JUMP_VEL_MIN - bonus

        x, y = 0.0, 0.0
        offsets = []
        visited = {(0, 0)}

        for frame in range(self.MAX_SIM_FRAMES):
            if frame < ctx.JUMP_HOLD_FRAMES:
                vy -= ctx.GRAVITY * 0.12 * self.DT

            grav = ctx.FAST_FALL_GRAV if vy > 0 else ctx.GRAVITY
            vy = min(vy + grav * self.DT, ctx.MAX_FALL_SPEED)

            x += vx * self.DT
            y += vy * self.DT

            tx = int(x // ts) if x >= 0 else -int((-x - 1) // ts) - 1
            ty = int(y // ts) if y >= 0 else -int((-y - 1) // ts) - 1

            if abs(tx) > half or abs(ty) > half:
                break

            tile = (tx, ty)
            if tile not in visited:
                visited.add(tile)
                offsets.append((tx, ty, vy > 0))

        return offsets

    def compute_arc_grid(self, solid_grid, on_ground, obs_pad_x, obs_pad_y):
        """
        Compute raw arc reachability grid (obs_height x obs_width).
        Returns: 0.0 = unreachable, 0.5 = in-flight, 1.0 = valid landing.
        """
        h, w = solid_grid.shape
        arc_grid = np.zeros((h, w), dtype=np.float32)

        if not on_ground:
            return arc_grid

        cx, cy = obs_pad_x, obs_pad_y

        for arc_offsets in self.arcs.values():
            for dx, dy, is_descending in arc_offsets:
                lx = cx + dx
                ly = cy + dy

                if not (0 <= lx < w and 0 <= ly < h):
                    continue

                if solid_grid[ly, lx] > 0.5:
                    break

                below_ly = ly + 1
                is_landing = False
                if is_descending and below_ly < h:
                    if solid_grid[below_ly, lx] > 0.5:
                        is_landing = True

                if is_landing:
                    arc_grid[ly, lx] = max(arc_grid[ly, lx], 1.0)
                else:
                    arc_grid[ly, lx] = max(arc_grid[ly, lx], 0.5)

        return arc_grid

    def boost_dijkstra(self, dijkstra_grid, solid_grid, on_ground,
                       obs_pad_x, obs_pad_y,
                       landing_boost=0.4, flight_boost=0.2):
        """
        Apply an additive boost to Dijkstra tiles on the jump arc.

        Unlike multiplicative masking, this preserves the entire existing
        gradient and only nudges arc-reachable tiles to be slightly more
        attractive. Works correctly on signed [-1, 1] values:
          - A tile at -0.3 with landing boost becomes -0.15 (less repulsive)
          - A tile at +0.2 with landing boost becomes +0.35 (more attractive)
          - Tiles not on any arc: completely untouched

        Args:
            dijkstra_grid: (H, W) advantage map, modified IN-PLACE
            solid_grid:    (H, W) solid channel
            on_ground:     player grounded flag
            obs_pad_x/y:   player's local position in the grid
            landing_boost:  added to landing tiles (default 0.15)
            flight_boost:   added to in-flight tiles (default 0.05)

        Returns:
            (dijkstra_grid, arc_grid)
        """
        h, w = dijkstra_grid.shape

        if not on_ground:
            return dijkstra_grid, np.zeros((h, w), dtype=np.float32)

        arc_grid = self.compute_arc_grid(
            solid_grid, on_ground, obs_pad_x, obs_pad_y
        )

        # Only boost air tiles — solid tiles already have correct gradient
        is_air = solid_grid <= 0.0

        landing_mask = is_air & (arc_grid >= 1.0)
        flight_mask  = is_air & (arc_grid >= 0.5) & (arc_grid < 1.0)

        dijkstra_grid[landing_mask] += landing_boost
        dijkstra_grid[flight_mask]  += flight_boost

        # np.clip(dijkstra_grid, -1.0, 1.0, out=dijkstra_grid)

        return dijkstra_grid, arc_grid

# =============================================================================
# Screen / Tile geometry
# =============================================================================
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
PLATFORMER_WIDTH, PLATFORMER_HEIGHT = 32, 32
DEBUG_PANEL_WIDTH = 350  # Width of the side debug panel (shown only in human mode)

# MultiDiscrete action axes — [move, jump, fire]
#   move : 0=IDLE  1=LEFT  2=SPRINT_LEFT  3=RIGHT  4=SPRINT_RIGHT
#   jump : 0=IDLE  1=JUMP
#   fire : 0=IDLE  1=FIRE
MD_MOVE_NAMES = {0: "IDLE", 1: "LEFT", 2: "SPRINT_L", 3: "RIGHT", 4: "SPRINT_R"}
MD_JUMP_NAMES = {0: "",     1: "JUMP"}
MD_FIRE_NAMES = {0: "",     1: "FIRE"}

def action_to_str(a) -> str:
    """Convert a MultiDiscrete action [move, jump, fire] to a readable string."""
    try:
        move, jump, fire = int(a[0]), int(a[1]), int(a[2])
        parts = [MD_MOVE_NAMES.get(move, "?")]
        if jump: parts.append(MD_JUMP_NAMES[1])
        if fire: parts.append(MD_FIRE_NAMES[1])
        return "+".join(p for p in parts if p)
    except Exception:
        return str(a)

# Keep for legacy / logging compat
ACTION_NAMES = {i: action_to_str([i % 5, (i // 5) % 2, i // 10]) for i in range(20)}

class PlatformerCore(gymnasium.Env):
    WIDTH, HEIGHT = SCREEN_WIDTH, SCREEN_HEIGHT

    @property
    def DEBUG_PANEL_X(self):
        """X offset where the debug panel begins."""
        return SCREEN_WIDTH if self.render_mode == "human" else 0

    @property
    def TOTAL_WIDTH(self):
        """Total window width including debug panel (human mode only)."""
        return SCREEN_WIDTH + DEBUG_PANEL_WIDTH if self.render_mode == "human" else SCREEN_WIDTH

    def __init__(self, render_mode: str = "none", **kwargs):
        self.render_mode = render_mode

        if self.render_mode == "human":
            pygame.init()
            pygame.display.set_caption("PEAK Platformer")
            self._surf = pygame.display.set_mode((SCREEN_WIDTH + DEBUG_PANEL_WIDTH, SCREEN_HEIGHT))
        else:
            # Headless / Training mode
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            pygame.init()
            self._surf = pygame.Surface((self.WIDTH, self.HEIGHT))

        # 1. Initialize Managers
        self.config_manager = ConfigManager("game_config.yaml")
        self.loader = LevelLoader()
        self.physics_manager = PhysicsManager()
        self.debug_manager = DebugManager(
            default_active=(render_mode=="human"),
            print_help=(render_mode=="human"),
            sensor_mode="rays",
        )

        # 2. Config & State Containers
        self.level_data = LevelData() # Empty initial state
        # PERF: Initialise to None so load_level() knows to do full construction
        # on the first call, and reuse the object on subsequent soft resets.
        self.player: Player | None = None

        # Default world and speed multiplier
        self.level_order = self.config_manager.get_level_order()
        self.current_index_world = 0
        _default_world = self.level_order[0] if self.level_order else "1-1"
        self.world = str(kwargs.pop("world", _default_world))
        # locked_level: when set, reset() always returns to this level (editor playtest)
        self.locked_level = str(self.world) if kwargs.pop("lock_level", False) else None

        # ── Mastery-gated curriculum ──────────────────────────────────────────
        # Each level keeps a sliding window of the last N episode outcomes (1=win,
        # 0=loss). At every episode boundary the win rate over that window drives
        # the level selection decision:
        #
        #   win_rate >= ADVANCE_THRESHOLD  → move forward one level
        #   win_rate <= FALLBACK_THRESHOLD → step back one level (if not at 0)
        #   otherwise                      → stay and keep practising
        #
        # EXPLORE_PROB: with this probability the curriculum is ignored entirely
        # and a uniformly random level is picked. Prevents the agent from
        # catastrophically forgetting earlier levels once they are "mastered",
        # and ensures all levels see at least some training traffic.
        #
        # Knobs exposed as kwargs so they can be tuned from the config YAML
        # without touching this file.
        self._curriculum_window_size  =   int(kwargs.pop("curriculum_window" ,   5))
        # NOTE: advance_threshold / fallback_threshold are intentionally NOT
        # popped here. They are consumed by the ACTIVE batch curriculum below
        # (self._batch_advance_threshold / self._batch_fallback_threshold).
        # Popping them here was a dead "mastery-gated" path that silently
        # starved the batch curriculum of its YAML-configured thresholds.
        self._explore_prob            = float(kwargs.pop("explore_prob",       0.10))

        # Per-level outcome windows — diagnostic sliding window feeding
        # curriculum_win_rate() in _info(). Persist across episodes, never reset.
        self._level_window = {
            lvl: deque(maxlen=self._curriculum_window_size)
            for lvl in self.level_order
        }

        # ── Progressive level unlocking ───────────────────────────────────────
        # _max_unlocked_index is the highest level index the agent is currently
        # allowed to visit. Starts at 0 (only the first level available) and
        # advances when the agent masters the current frontier.
        #
        # Exploration is capped to [0, _max_unlocked_index] so random picks
        # never send the agent to levels it has no chance of completing.
        # A new level is unlocked only when the agent masters the current
        # hardest available level (current_index == _max_unlocked_index and
        # win_rate >= advance_threshold).
        #
        # start_unlocked: pre-unlock N levels at init (useful for resuming a
        # run or skipping trivial early levels). Default 0 = start from level 1.
        self._max_unlocked_index = int(kwargs.pop("start_unlocked", 0))

        # ── Eval-trust flags (Task 1) ─────────────────────────────────────────
        # curriculum_enabled=False: reset() pins to self.world and never touches
        #   the shared curriculum/batch state. Mirrors sonic_core/megaman_core.
        # terminate_on_goal=True: reaching the goal ENDS the episode (eval only),
        #   instead of the inline next-level transition done during training.
        self.curriculum_enabled = bool(kwargs.pop("curriculum_enabled", True))
        self.terminate_on_goal  = bool(kwargs.pop("terminate_on_goal", False))
        # dijkstra_enabled=False: ABLATION — zero the Dijkstra obs channel (channel 3),
        # keeping the (4,H,W) shape so the extractor is unchanged. Lets developers test
        # generalization without the solver-derived navigational prior (paper Req. note).
        self.dijkstra_enabled = bool(kwargs.pop("dijkstra_enabled", True))

        self.speed_mult = float(kwargs.pop("speed_mult", 2.0))
        self.physics_manager.speed_mult = self.speed_mult

        self.max_steps = kwargs.pop("max_steps", None)

        self.persona = str(kwargs.pop("persona", "simple")).lower()
        if self.persona == "default":
            self.persona = "simple"
        # Architecture tag — passed from training config for debug overlay
        self.arch_tag = str(kwargs.pop("arch_tag", "slim")).lower()
        # reward_fn is owned by generic_env (the wrapper), not the core game.
        # Kept as None here so the persona label is still accessible via self.persona.
        self.reward_fn = None
        self.ACTION_NAMES = ACTION_NAMES

        # Timer knobs
        self.use_timer = bool(kwargs.pop("use_timer", True))
        self.timer_seconds = int(kwargs.pop("timer_seconds", 400))
        # Episode time-horizon override (game-clock units). When set, it wins
        # over each level's time_limit from game_config.yaml — makes the
        # in-game episode horizon a run parameter (+time_limit=150) instead
        # of a per-level constant. Also drives the obs timer normalisation.
        self._time_limit_override = kwargs.pop("time_limit", None)
        if self._time_limit_override is not None:
            self.timer_seconds = int(self._time_limit_override)
        self.timer_warn_threshold = int(kwargs.pop("timer_warn_threshold", 100))

        self.max_lives = 3
        self.lives = self.max_lives

        # ── Batch Curriculum ───────────────────────────────────────────────
        self._batch_window            = int(kwargs.pop("batch_window", 10))
        # SOLE consumer of advance_threshold / fallback_threshold kwargs (see
        # the curriculum block above). Defaults apply only when YAML omits them.
        self._batch_advance_threshold = float(kwargs.pop("advance_threshold", 0.30))
        self._batch_fallback_threshold= float(kwargs.pop("fallback_threshold", 0.20))
        self._max_stay_windows        = int(kwargs.pop("max_stay_windows", 2))  # reduced from 3
        self._review_prob             = float(kwargs.pop("review_prob", 0.25))
        self._curriculum_position     = 0
        self._batch_results: list     = []
        self._episode_won_current     = False
        self._is_review_episode       = False
        self._windows_on_level        = 0
        self._consecutive_fallbacks   = {}
        self._level_visits     = {lvl: 0 for lvl in self.level_order}
        self._level_wins       = {lvl: 0 for lvl in self.level_order}

        # Camera
        self.camera_x = 0.0
        self.camera_y = 0.0
        self.camera_smoothing = 0.15
        self.camera_lock = True

        # Anti-stall
        self.anti_stall = bool(kwargs.pop("anti_stall", True))
        self.stall_window = float(kwargs.pop("stall_window", 2))
        self.stall_kill_windows = int(kwargs.pop("stall_kill_windows", 10))
        # Stall progress metric: "euclid" (legacy straight-line px) or "path"
        # (Dijkstra path distance). Euclidean falsely kills correct VERTICAL
        # play — riding a platform or climbing a shaft on Mario1-2 does not
        # shrink straight-line distance to the goal, so the watchdog executes
        # the agent for progressing (measured: 14/28 eval failures on 1-2).
        # Path distance counts any movement along the solver's route as
        # progress; falls back to Euclidean while the tile reading is invalid
        # (mid-air / unreachable), so genuine stalls still die.
        self.stall_metric = str(kwargs.pop("stall_metric", "euclid")).lower()

        # Observation sanity checker
        self._obs_check_interval = 5000   # Check every N steps
        self._obs_check_counter = 0
        self._obs_stats = {               # Latest stats, exposed via _info()
            # 4 grid channels: solid, collectible, hazard, dijkstra
            "grid_solid_mean": 0.0,       "grid_solid_std": 0.0,
            "grid_solid_min": 0.0,        "grid_solid_max": 0.0,
            "grid_collectible_mean": 0.0, "grid_collectible_std": 0.0,
            "grid_collectible_min": 0.0,  "grid_collectible_max": 0.0,
            "grid_hazard_mean": 0.0,      "grid_hazard_std": 0.0,
            "grid_hazard_min": 0.0,       "grid_hazard_max": 0.0,
            "grid_dijkstra_mean": 0.0,    "grid_dijkstra_std": 0.0,
            "grid_dijkstra_min": 0.0,     "grid_dijkstra_max": 0.0,
            # scalars
            "scalar_mean": 0.0, "scalar_std": 0.0,
            "scalar_min": 0.0,  "scalar_max": 0.0,
            "dijkstra_val": 0.0, "obs_warnings": "",
        }

        self.reset_metrics()

        # --- GRID OBSERVATION SIZE ---
        self.obs_width = 21
        self.obs_height = 21
        self.obs_pad_x = self.obs_width // 2
        self.obs_pad_y = self.obs_height // 2

        # --- RAYCAST CONFIGURATION ---
        # Number of rays to cast around the player
        self.num_rays = int(kwargs.pop("num_rays", 48))
        self.ray_max_dist = 250.0
        # Create angles (0 to 2pi)
        self.ray_angles = np.linspace(0, 2 * math.pi, self.num_rays, endpoint=False)
        self.last_rays = [] # For debug drawing

        # Dijkstra Map Storage
        self.dijkstra = None
        self.dijkstra_current_tile = 0.0

        # Jump arc precomputation — boosts Dijkstra on reachable jump tiles.
        # Recomputed in load_level after config changes.
        self.jump_arc_computer = JumpArcComputer(
            self.physics_manager.context,
            TILE_SIZE,
            self.obs_pad_x
        )

        self._obs_space = spaces.Dict({
            # 4 Channels (in order): Solids, Collectibles, Hazards, Dijkstra
            #
            #   0 - Solids       : ground, static platforms, moving platforms,
            #                      question blocks  → binary {0.0, 1.0}
            #
            #   1 - Collectibles : importance-weighted, single channel
            #                      goal    = 1.0   (level exit — highest priority)
            #                      powerup = 0.69  (changes player capability)
            #                      coin    = 0.35  (bonus reward)
            #                      empty   = 0.0
            #
            #   2 - Hazards      : sign encodes agent's viable response
            #                      enemy  = +1.0   (defeatable by stomp / fireball)
            #                      spike  = -1.0   (always lethal, never avoidable)
            #                      empty  =  0.0
            #
            #   3 - Dijkstra     : relative advantage map in [-1, 1]
            #                      +1.0 = tile much closer to goal than player
            #                       0.0 = same distance / unreachable (neutral)
            #                      -1.0 = tile much further from goal than player
            #
            # All channels span [-1, 1] so low=-1.0 covers spikes and Dijkstra.
            "grids": spaces.Box(low=-1.0, high=1.0, shape=(4, self.obs_height, self.obs_width), dtype=np.float32),

            # Scalars: 20  (Player=13: obs_vector  +  Tracking=7: e_dist,goal_dist,timer,dir_y,dijkstra,step_dx,step_dy)
            "scalars": spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32),

            # Raycasts: [dist, type, dist, type, ...] -> Size = num_rays * 2
            # "raycasts": spaces.Box(low=0.0, high=4.0, shape=(self.num_rays * 2,), dtype=np.float32)
        })

        # Action Space: MultiDiscrete [move, jump, fire]
        #   move : 0=idle 1=left 2=run_left 3=right 4=run_right
        #   jump : 0=idle 1=jump
        #   fire : 0=idle 1=fire
        self._act_space = spaces.MultiDiscrete([5, 2, 2])

        self.ui_font = pygame.font.SysFont("arial", 20, bold=True)
        self.qblock_font = pygame.font.SysFont("arial", 26, bold=True)

        self._dijkstra_window_cache = None
        self._solid_window_cache    = None   # set by _grid_obs_window each step
        self._hazard_window_cache   = None   # set by _grid_obs_window each step
        self._jump_arc_cache        = None   # set by _obs each step

        self.reset()

    def reset_metrics(self):
        """Helper to clear metrics on reset/death."""
        self.timer = self.timer_seconds
        self.time_last_step = time.time()
        self.dt = 0.0001
        self.score = 0; self.coins_total = 0
        self.alive = True; self.frame = 0
        self.game_over = False; self.reached_goal = False
        self.last_x = 0.0; self.last_score = 0; self.score_delta = 0
        self.kills_step = self.coins_step = self.powerups_step = 0
        self._last_action = [0, 0, 0]
        self.max_x_seen = 0.0; self.stall_timer = 0
        self.stall_windows_count = 0; self.stalled_this_frame = False
        self.progress_x_best = 0.0; self.progress_y_best = 0.0
        self.death_cause = ""
        self.lives = self.max_lives  # restore lives on every full episode reset
        self.best_dist_to_goal = float('inf')  # stall tracker anchor (Euclidean px)
        self.best_path_dist = float('inf')     # stall tracker anchor (Dijkstra cost, path mode)
        self._needs_level_transition = False   # set True when goal reached mid-episode
        self._pending_next_level_index = None  # set by complete_level, applied after _info()
        # --- Velocity Alignment (cached from _tracking_obs each step) ---
        self._step_dx = 0.0   # unit vector toward cheapest reachable 8-direction tile, X
        self._step_dy = 0.0   # unit vector toward cheapest reachable 8-direction tile, Y

    def _load_reward_fn(self, persona_name):
        try:
            mod = importlib.import_module("code.rewards.train_platformer")
            return getattr(mod, persona_name, None)
        except ImportError:
            return None

    def get_action_space(self): return self._act_space
    def get_observation_space(self): return self._obs_space

    def step(self, action: int):
        if not self.alive:
            dead_obs = self._obs()
            return dead_obs, 0.0, True, False, {"episode_end": True, "won": self.reached_goal}

        # Time Calculation
        if self.render_mode != "human":
            self.dt = 1 / 60.0
        else:
            time_curr_step = time.time()
            raw_dt = time_curr_step - self.time_last_step
            self.time_last_step = time_curr_step
            self.dt = min(raw_dt, 0.05)

        if self.debug_manager.slow_motion:
            self.dt *= 0.5

        # Normalise action — accept both MD arrays ([move, jump, fire]) and
        # legacy flat ints (e.g. keyboard shortcuts, old test code).
        _LEGACY_INT_TO_MD = {
            0: [0,0,0], 1: [1,0,0], 2: [3,0,0], 3: [0,1,0],
            4: [3,1,0], 5: [4,0,0], 6: [1,1,0], 7: [4,1,0],
            8: [2,0,0], 9: [2,1,0],
        }
        if isinstance(action, (int, float)) or (
                hasattr(action, '__len__') is False):
            action = _LEGACY_INT_TO_MD.get(int(action), [0, 0, 0])
        action = [int(action[0]), int(action[1]), int(action[2])]



        if self.use_timer: self.timer -= self.dt
        # Physics-frame counter. Was never incremented before (latent bug):
        # the frame-based max_steps truncation below never fired (the GameEnv
        # wrapper's decision counter masked it) and info["frame_count"] logged
        # 0 forever. With frame-skip, max_steps must count FRAMES so the
        # in-game episode budget is invariant to the skip value.
        self.frame += 1
        if self.render_mode == "human": self.debug_manager.update_input()

        # Step Metrics Reset
        self._last_action = action  # already normalised to [move, jump, fire] above
        self.last_x = self.player.gObj.x if self.player else 0.0
        self.kills_step = self.coins_step = self.powerups_step = 0
        self.stalled_this_frame = False
        # Reset per-frame platform flag — PhysicsManager sets it True if riding one
        if self.player:
            self.player._on_moving_platform = False

        # PHYSICS & LOGIC
        # All three dynamic spatial hashes are rebuilt every frame.
        #
        # Why not use a dirty flag for collectible_hash?
        # Powerups move (mushrooms walk, stars bounce) so their hash
        # positions go stale every frame. The cost of rebuilding coins+goals
        # alongside powerups is negligible (~20μs for 200 entities).

        # --- hazard_hash: enemies + spikes ---
        # Rebuilt every frame because enemies move. Spikes are static but live
        # here so hazard_hash.query_rect in _grid_obs_window picks them up
        # without a separate tile-grid scan each frame.
        self.physics_manager.hazard_hash.clear()
        for enemy in self.level_data.enemies:
            if enemy.gObj.active:
                self.physics_manager.hazard_hash.insert(enemy)
        for spike in self._cached_spikes:
            self.physics_manager.hazard_hash.insert(spike)

        # --- platform_hash: moving platforms ---
        # Rebuilt every frame because platforms move.
        self.physics_manager.platform_hash.clear()
        for plat in self.level_data.moving_platforms:
            if plat.gObj.active:
                self.physics_manager.platform_hash.insert(plat)

        # --- collectible_hash: coins, powerups, goals ---
        self.physics_manager.collectible_hash.clear()
        for coin in self.level_data.coins:
            if coin.gObj.active and not coin.collected:
                self.physics_manager.collectible_hash.insert(coin)
        for pup in self.level_data.powerups:
            if pup.gObj.active:
                self.physics_manager.collectible_hash.insert(pup)
        for goal in self.level_data.goals:
            self.physics_manager.collectible_hash.insert(goal)

        if self.player:
            if not self.debug_manager.free_cam_active:
                self.player.handle_input(a=action)
            else:
                self.player.vx = 0; self.player.jump_hold = 0

            # --- Fire Flower projectile spawn ---
            if self.player.fire_requested:
                self.player.fire_requested = False
                proj = FireFlowerProjectile.from_player(self.player)
                self.level_data.projectiles.append(proj)

        self.physics_manager.update_system(self.dt, self)
        self.physics_manager.resolve_collisions(self)

        # Cleanup Inactive Entities
        self.level_data.enemies[:]  = [e for e in self.level_data.enemies  if e.gObj.active]
        self.level_data.coins[:]    = [c for c in self.level_data.coins    if c.gObj.active]
        self.level_data.powerups[:] = [p for p in self.level_data.powerups if p.gObj.active]
        self.level_data.projectiles[:] = [
            p for p in self.level_data.projectiles if p.gObj.active
        ]

        # PERF: Cache goal distance once per step — read by stall metrics,
        # tracking obs, and _info() so it only gets computed once.
        self._goal_dist_cache = self._get_dist_to_goal()

        self._update_camera()
        if self.anti_stall: self._update_stall_metrics()

        # Check Truncation & Termination
        terminated = self._check_termination()

        truncated = False
        if self.max_steps and self.frame >= self.max_steps:
            truncated = True

        self.score_delta = self.score - self.last_score
        self.last_score = self.score

        # Build observation BEFORE _info() so _check_obs_sanity can
        # populate self._obs_stats in time for _info() to spread them.
        obs = self._obs()
        self._check_obs_sanity(obs)

        info = self._info()

        # Inline level transition on win.
        # Advance self.world AFTER _info() so WIN is logged on the right level.
        if self._needs_level_transition and not self.terminate_on_goal:
            self._needs_level_transition = False
            if self._pending_next_level_index is not None:
                self.current_index_world = self._pending_next_level_index
                self.world = self.level_order[self.current_index_world]
                self._level_visits[self.world] = self._level_visits.get(self.world, 0) + 1
                self._pending_next_level_index = None
            self.load_level(preserve_power=True)

        if terminated:
            info["episode_end"] = True

        # Return raw score delta as base reward.
        # The GameEnv wrapper (generic_env.py) applies the actual persona reward fn.
        base_reward = float(self.score_delta)

        return obs, base_reward, bool(terminated), bool(truncated), info

    def reset(self, seed=None, options=None) -> np.ndarray:
        super().reset(seed=seed)

        # Record episode result into batch — only if it was a curriculum episode
        # (NOT a review episode, which played a different level)
        if self.curriculum_enabled and self.level_order and not self.locked_level and not self._is_review_episode:
            self._batch_results.append(self._episode_won_current)

        # Record into the per-level diagnostic window (feeds curriculum_win_rate
        # in _info()). self.world still names the level that was just played —
        # it is reassigned to the next level later in this method. Use setdefault
        # so levels not present in level_order (edge cases) still get a window.
        if self.level_order and not self.locked_level:
            win_dq = self._level_window.setdefault(
                self.world, deque(maxlen=self._curriculum_window_size))
            win_dq.append(1 if self._episode_won_current else 0)

        # Evaluate batch when window is full
        if self.curriculum_enabled and len(self._batch_results) >= self._batch_window and self.level_order and not self.locked_level:
            self._evaluate_curriculum_batch()

        self.reset_metrics()
        self._episode_won_current = False
        self._is_review_episode = False

        if self.locked_level:
            self.world = self.locked_level
            if self.locked_level in self.level_order:
                self.current_index_world = self.level_order.index(self.locked_level)
            else:
                self.current_index_world = 0
        elif not self.curriculum_enabled and self.level_order:
            # Eval / playback: stay on the level we were constructed with.
            if self.world in self.level_order:
                self.current_index_world = self.level_order.index(self.world)
            else:
                self.current_index_world = 0
                self.world = self.level_order[self.current_index_world]
        elif self.level_order:
            self._curriculum_position = max(0, min(
                self._curriculum_position, len(self.level_order) - 1))

            # ── Review rotation: 25% chance to play a random earlier level ──
            # Keeps skills sharp on mastered levels, distributes visits,
            # prevents catastrophic forgetting. Results don't count for batch.
            if (self._curriculum_position > 0
                    and random.random() < self._review_prob):
                review_idx = random.randint(0, self._curriculum_position - 1)
                self.current_index_world = review_idx
                self._is_review_episode = True
            else:
                self.current_index_world = self._curriculum_position

            self.world = self.level_order[self.current_index_world]

        self._level_visits[self.world] = self._level_visits.get(self.world, 0) + 1
        self.load_level()
        return self._obs(), self._info()

    def _evaluate_curriculum_batch(self):
        wins = sum(1 for r in self._batch_results if r)
        total = len(self._batch_results)
        win_rate = wins / total if total > 0 else 0.0
        pos = self._curriculum_position
        level_name = self.level_order[pos] if pos < len(self.level_order) else "?"
        consec_fb = self._consecutive_fallbacks.get(pos, 0)

        effective_advance = self._batch_advance_threshold
        if consec_fb >= 2:
            effective_advance = max(0.10, effective_advance - 0.10)

        if win_rate >= effective_advance and pos < len(self.level_order) - 1:
            self._curriculum_position += 1
            self._windows_on_level = 0
            self._consecutive_fallbacks[pos] = 0
            print(f"  🎓 [Curriculum] ADVANCE → Lvl {self._curriculum_position} "
                  f"'{self.level_order[self._curriculum_position]}' "
                  f"({wins}/{total} = {win_rate:.0%})")
        elif win_rate <= self._batch_fallback_threshold and pos > 0:
            self._curriculum_position -= 1
            self._windows_on_level = 0
            self._consecutive_fallbacks[pos] = consec_fb + 1
            print(f"  ⬇️  [Curriculum] FALLBACK → Lvl {self._curriculum_position} "
                  f"'{self.level_order[self._curriculum_position]}' "
                  f"({wins}/{total} = {win_rate:.0%}, fb={consec_fb + 1})")
        else:
            self._windows_on_level += 1
            if self._windows_on_level >= self._max_stay_windows and pos > 0:
                self._curriculum_position -= 1
                self._windows_on_level = 0
                print(f"  ⏳ [Curriculum] FORCE FALLBACK → Lvl {self._curriculum_position} "
                      f"(stuck {self._max_stay_windows} windows)")

        self._batch_results.clear()

    def load_level(self, preserve_power: bool = False):
        self.alive = True
        self.frame = 0
        self.game_over = False
        self.reached_goal = False

        config = self.config_manager.get_level_config(self.world)
        self.level_data = self.loader.load_level(config)

        # Clear any fireballs from the previous level.
        self.level_data.projectiles = []

        # --- Cache spike objects for hazard_hash ---
        # Spikes are static tiles created by LevelLoader and stored only in
        # level_data.tiles / static_hash. We cache references here so step()
        # can insert them into hazard_hash every frame without scanning the
        # full tile grid. The list is rebuilt on every load_level() call
        # (new level or soft-reset) so it always matches the current map.
        self._cached_spikes = []
        for row in range(self.level_data.rows):
            for col in range(self.level_data.cols):
                if self.level_data.grid[row][col] == TILE_SPIKE:
                    tile = self.level_data.tiles[row][col]
                    if tile is not None:
                        self._cached_spikes.append(tile)

        px, py = self.level_data.player_start
        if 'spawn' in config:
            px = float(config['spawn'].get('x', px))
            py = float(config['spawn'].get('y', py))

        # PERF: Reuse the Player object across soft resets (deaths).
        if self.player is None:
            self.player = Player(gObj=GameObject(px, py, PLATFORMER_WIDTH, PLATFORMER_HEIGHT, True))
            self.player.__post_init__()
        else:
            self.player.gObj.x       = float(px)
            self.player.gObj.y       = float(py)
            self.player.gObj.active  = True
            self.player.vx           = 0.0
            self.player.vy           = 0.0
            self.player.on_ground    = False
            self.player.facing_right = True
            self.player.coyote       = 0
            self.player.jump_hold    = 0
            self.player.jump_buffer  = 0
            self.player.input_dir    = 0
            self.player.run_held     = False
            self.player.jump_pressed = False
            self.player.fire_requested  = False
            self.player._fire_cooldown  = 0.0

            if preserve_power:
                self.player.power_machine._iframes_timer = 0.0
            else:
                self.player.power_machine.reset()
                self.player.powered_up       = False
                self.player.invincible_timer = 0

        self.physics_manager.reset_to_defaults()
        self.physics_manager.apply_config_dict(config)
        self.physics_manager.rebuild_dynamic_hashes(self.level_data, self._cached_spikes)

        # Recompute jump arcs for the (possibly updated) physics context
        self.jump_arc_computer.recompute(self.physics_manager.context)

        self.progress_x_best = self.player.gObj.x
        self.progress_y_best = self.level_data.height - self.player.gObj.y
        self.stall_timer = 0
        self.stall_windows_count = 0
        self.stalled_this_frame = False
        self.max_x_seen = px
        self.best_dist_to_goal = self._get_dist_to_goal()
        self.best_path_dist = float('inf')   # re-anchor path-stall on level (re)load

        self.camera_x = 0.0
        self.camera_y = 0.0
        self.last_score = 0
        self.last_x = self.player.gObj.x

        if not self.use_timer:
            self.timer = math.inf
        elif self._time_limit_override is not None:
            # Run-level horizon parameter beats the per-level config value.
            self.timer = float(self._time_limit_override)
        else:
            self.timer = config.get('time_limit', self.timer_seconds)

        # --- CALCULATE DIJKSTRA MAP ---
        self._calculate_dijkstra_map()

    def _calculate_dijkstra_map(self):
        """Initializes and computes the global Dijkstra distance map."""
        if not self.level_data.goals:
            self.dijkstra = None
            return

        self.dijkstra = DijkstraSolver(self.level_data.grid, self.level_data.rows, self.level_data.cols)

        goal_positions = []
        for g in self.level_data.goals:
            gx = int(g.gObj.x // TILE_SIZE)
            gy = int(g.gObj.y // TILE_SIZE)
            goal_positions.append((gx, gy))

        coin_positions = set()
        for c in self.level_data.coins:
            cx = int(c.gObj.x // TILE_SIZE)
            cy = int(c.gObj.y // TILE_SIZE)
            coin_positions.add((cx, cy))

        self.dijkstra.compute_map(goal_positions, coin_positions)

    def complete_level(self):
        self._level_wins[self.world] = self._level_wins.get(self.world, 0) + 1

        # Batch curriculum: mark win if this is the current curriculum level
        if self._curriculum_position < len(self.level_order):
            if self.world == self.level_order[self._curriculum_position]:
                self._episode_won_current = True

        # Store next level index but do NOT advance self.world yet.
        # self.world must stay as the completed level until _info() is called
        # so the WIN event is logged against the correct level.
        #
        # When locked_level is set (editor playtest), loop back to the same
        # level instead of advancing — the player should restart the level
        # they just completed, not be pushed into the next registered level.
        if self.locked_level:
            next_idx = self.current_index_world  # stay on the same level
        else:
            next_idx = (self.current_index_world + 1) % len(self.level_order)
        self._pending_next_level_index = next_idx
        self._needs_level_transition = True

    def _handle_death(self, cause: str = "Unknown") -> bool:
        """
        Decrements lives and either soft-resets (lives > 0) or ends the episode.
        Returns True only when lives reach 0.
        """
        self.death_cause = cause
        self.lives = max(0, self.lives - 1)
        if self.lives > 0:
            self._soft_reset()
            return False
        else:
            self.alive     = False
            self.game_over = True
            return True

    def _soft_reset(self):
        """Reloads the current level after a death, preserving lives and score."""
        self.load_level()

    def _update_camera(self):
        # Camera math is only needed in human mode.
        if self.render_mode != "human":
            return

        if self.debug_manager.free_cam_active:
            movement_x, movement_y = self.debug_manager.current_cam_move
            self.camera_x += movement_x * self.dt
            self.camera_y += movement_y * self.dt
            return

        if not self.camera_lock or not self.player: return

        level_w = self.level_data.width
        level_h = self.level_data.height

        target_x = max(0, min(self.player.gObj.x - self.WIDTH // 3, level_w - self.WIDTH))
        self.camera_x += (target_x - self.camera_x) * self.camera_smoothing
        self.camera_x = max(0, min(self.camera_x, max(0, level_w - self.WIDTH)))

        target_y = 0.0
        if level_h > self.HEIGHT:
            target_y = max(0, min(self.player.gObj.y - self.HEIGHT // 2, level_h - self.HEIGHT))

        self.camera_y += (target_y - self.camera_y) * self.camera_smoothing
        self.camera_y = max(0, min(self.camera_y, max(0, level_h - self.HEIGHT)))

    def _get_dist_to_goal(self) -> float:
        if not self.player: return float('inf')
        if not self.level_data.goals:
            return self.level_data.width - self.player.gObj.x

        px = self.player.gObj.x
        py = self.player.gObj.y
        min_d = float('inf')
        for g in self.level_data.goals:
            gx = g.gObj.x
            gy = g.gObj.y
            d = math.sqrt((gx - px) ** 2 + (gy - py) ** 2)
            if d < min_d:
                min_d = d
        return min_d

    def _update_stall_metrics(self):
        """Stall detection — suppressed when riding a moving platform."""
        if not self.player: return

        if self.player.gObj.x > self.max_x_seen:
            self.max_x_seen = self.player.gObj.x

        # FIX: If riding a moving platform, reset stall timer — the player
        # IS making progress, just not under their own velocity. Without this,
        # the agent gets stall-killed for being patient on platforms.
        if getattr(self.player, '_on_moving_platform', False):
            self.stall_timer = 0
            self.stalled_this_frame = False
            return

        # Progress test. "path" mode measures along the Dijkstra route (in
        # tile-cost units; 0.5 ≈ the 16px Euclidean threshold) so correct
        # vertical play counts as progress; it falls back to the Euclidean
        # test whenever the tile reading is invalid (mid-air / unreachable),
        # keeping the watchdog live against genuine stalls.
        made_progress = False
        used_path = False
        if self.stall_metric == "path" and self.dijkstra:
            d = self.dijkstra.get_dist(int(self.player.gObj.x // TILE_SIZE),
                                       int(self.player.gObj.y // TILE_SIZE))
            if d >= 0:
                used_path = True
                if d < (self.best_path_dist - 0.5):
                    self.best_path_dist = d
                    made_progress = True

        if not used_path:
            # PERF: Read from cache set at the top of step() instead of recomputing.
            current_dist = getattr(self, '_goal_dist_cache', self._get_dist_to_goal())
            threshold = TILE_SIZE / 2.0
            if current_dist < (self.best_dist_to_goal - threshold):
                self.best_dist_to_goal = current_dist
                made_progress = True

        if made_progress:
            self.stall_timer = 0
            self.stalled_this_frame = False
        else:
            self.stall_timer += self.dt
            if self.stall_timer >= self.stall_window:
                self.stalled_this_frame = True
                self.stall_timer = 0
                self.stall_windows_count += 1

    def _check_termination(self) -> bool:
        """
        Returns True only when the episode should end (lives = 0, or no player).
        Goal completion is NOT a termination during training — it transitions
        inline in step(). In eval (terminate_on_goal=True) the goal ENDS the
        episode instead.
        """
        player = self.player
        if not player:
            return True

        # Eval-only: reaching the goal ends the episode.
        if self.terminate_on_goal and self.reached_goal:
            return True

        if self.use_timer and self.timer <= 0:
            return self._handle_death("Timeout")

        if player.gObj.y > self.level_data.height:
            # Fallback: PhysicsManager._check_oob() should have caught this
            # first, but this guard ensures no frame is missed if the physics
            # manager is bypassed (e.g. render-only mode, test harness).
            return self._handle_death("Pit")

        # 4. STALL DEATH
        if self.anti_stall and self.stall_windows_count >= self.stall_kill_windows:
            return self._handle_death("Stall")

        return False

    def _check_obs_sanity(self, obs: Dict[str, np.ndarray]) -> None:
        """Compute observation statistics every N steps. Stored on self for _info() to expose."""
        self._obs_check_counter += 1
        if self._obs_check_counter % self._obs_check_interval != 0:
            return

        warnings_list = []

        # Channel names match the 4-channel layout declared in _obs_space
        grid_names = ["solid", "collectible", "hazard", "dijkstra"]
        grids = obs.get("grids")
        if grids is not None:
            for i, name in enumerate(grid_names):
                ch = grids[i]
                self._obs_stats[f"grid_{name}_mean"] = float(ch.mean())
                self._obs_stats[f"grid_{name}_std"]  = float(ch.std())
                self._obs_stats[f"grid_{name}_min"]  = float(ch.min())
                self._obs_stats[f"grid_{name}_max"]  = float(ch.max())

                # Only flag solid (ch 0) as DEAD — other channels are
                # legitimately sparse or zero when nothing is nearby.
                if float(ch.std()) < 1e-6 and i == 0:
                    warnings_list.append(f"Grid '{name}' DEAD")
                if float(ch.max()) > 1.01:
                    warnings_list.append(f"Grid '{name}' >1.0")

        scalars = obs.get("scalars")
        if scalars is not None:
            self._obs_stats["scalar_mean"] = float(scalars.mean())
            self._obs_stats["scalar_std"]  = float(scalars.std())
            self._obs_stats["scalar_min"]  = float(scalars.min())
            self._obs_stats["scalar_max"]  = float(scalars.max())
            # dijkstra_dist is at index 17 (13 player scalars + 4th tracking value)
            self._obs_stats["dijkstra_val"] = float(scalars[17])

            if float(scalars.std()) < 1e-8:
                warnings_list.append("Scalars DEAD")
            if abs(float(scalars.max())) > 100:
                warnings_list.append("Scalars unnormalized")

            dijk_val = float(scalars[17])
            if hasattr(self, '_last_dijk_nonzero'):
                if self._last_dijk_nonzero and dijk_val == 0.0:
                    self._dijk_zero_streak = getattr(self, '_dijk_zero_streak', 0) + 1
                    if self._dijk_zero_streak >= 3:
                        warnings_list.append(f"Dijkstra stuck@0 x{self._dijk_zero_streak}")
                else:
                    self._dijk_zero_streak = 0
            self._last_dijk_nonzero = dijk_val != 0.0

        self._obs_stats["obs_warnings"] = "|".join(warnings_list) if warnings_list else ""

    def _obs(self) -> Dict[str, np.ndarray]:
        if not self.player:
            return {
                # 4 channels: solid, collectible, hazard, dijkstra
                "grids":   np.zeros((4, self.obs_height, self.obs_width), dtype=np.float32),
                # 20 scalars: 13 (player obs_vector) + 7 (tracking)
                "scalars": np.zeros(20, dtype=np.float32),
            }

        p_obs     = self._player_obs()
        track_obs = self._tracking_obs()

        solid_grid, collect_grid, hazard_grid, map_row_start, map_col_start = \
            self._grid_obs_window()

        if self.dijkstra_enabled:
            dijkstra_grid = self._dijkstra_obs_window(map_row_start, map_col_start)

            # ── Physics-aware Dijkstra boost ──────────────────────────────────
            # Add a small positive value to air tiles the player can physically
            # reach via jump. Landing tiles get +0.15, in-flight tiles get +0.05.
            # This nudges the gradient toward jumpable paths without destroying
            # the existing signal. Solid tiles and unreachable air are untouched.
            # When airborne, no boost is applied (arc only valid from ground).
            on_ground = self.player.on_ground if self.player else False
            dijkstra_grid, arc_grid = self.jump_arc_computer.boost_dijkstra(
                dijkstra_grid, solid_grid, on_ground,
                self.obs_pad_x, self.obs_pad_y
            )
            self._dijkstra_window_cache = dijkstra_grid
            self._jump_arc_cache = arc_grid
        else:
            # ABLATION: Dijkstra channel off — zero it (shape preserved).
            dijkstra_grid = np.zeros((self.obs_height, self.obs_width), dtype=np.float32)
            self._dijkstra_window_cache = dijkstra_grid
            self._jump_arc_cache = None

        # Stack order: Solids, Collectibles, Hazards, Dijkstra  (4 channels)
        #   Ch 0 solid       : {-0.5, 0.0, 1.0}  — pit / air / wall+platform
        #   Ch 1 collectible : {0.0, 0.35, 0.69, 1.0}  — coin / powerup / goal
        #   Ch 2 hazard      : {-1.0, -0.5, 0.0, +1.0} — spike / pit / empty / enemy
        #   Ch 3 dijkstra    : continuous [-1.0, 1.0]
        #
        # Pit encoding (-0.5) sits between safe air (0.0) and lethal spike (-1.0)
        # on both solid and hazard channels so the CNN learns a clear danger gradient.
        stacked_grids = np.stack([
            solid_grid,
            collect_grid,
            hazard_grid,
            dijkstra_grid,
        ], axis=0).astype(np.float32)

        scalars = np.concatenate([p_obs, track_obs]).astype(np.float32)

        return {
            "grids":   stacked_grids,
            "scalars": scalars,
        }

    def _player_obs(self) -> np.ndarray:
        p = self.player
        if not p:
            return np.zeros(13, dtype=np.float32)

        max_run  = max(1.0, getattr(self.physics_manager.context, 'MAX_RUN_SPEED',  240.0))
        max_fall = max(1.0, getattr(self.physics_manager.context, 'MAX_FALL_SPEED', 400.0))

        return p.obs_vector(max_run, max_fall)

    def _grid_obs_window(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
        """
        Returns (solid_grid, collect_grid, hazard_grid, map_row_start, map_col_start).

        solid_grid   — binary {0.0, 1.0}
                    ground, static platforms, question blocks, moving platforms

        collect_grid — importance-weighted float, single channel:
                    goal    = 1.0   (level exit — highest priority)
                    powerup = 0.69  (changes player capability)
                    coin    = 0.35  (bonus reward)
                    empty   = 0.0
                    If two entities share a tile, the higher value wins.

        hazard_grid  — sign encodes the agent's viable response:
                    enemy  = +1.0  (defeatable by stomp or fireball)
                    spike  = -1.0  (always lethal, never safe to touch)
                    empty  =  0.0
                    If spike and enemy share a tile, spike wins (more dangerous).

        map_row_start / map_col_start: unpadded map tile coordinates of the
        top-left corner of the 21×21 window centered on the player. Passed
        to _dijkstra_obs_window so both functions slice the same region.
        Coordinates can be negative when the player is near the top/left edge.

        All entity lookups use spatial hashes — only objects within the window
        are visited, keeping per-step overhead minimal.
        """
        p = self.player
        if not p:
            z = np.zeros((self.obs_height, self.obs_width), dtype=np.float32)
            return z, z, z, 0, 0

        px = int(p.gObj.x // TILE_SIZE)
        py = int(p.gObj.y // TILE_SIZE)

        # Window always centered on player — no edge clamping (translation invariance)
        map_row_start = py - self.obs_pad_y
        map_col_start = px - self.obs_pad_x

        solid_grid   = np.zeros((self.obs_height, self.obs_width), dtype=np.float32)
        collect_grid = np.zeros((self.obs_height, self.obs_width), dtype=np.float32)
        hazard_grid  = np.zeros((self.obs_height, self.obs_width), dtype=np.float32)

        # World-pixel origin and dimensions of the window
        wx = map_col_start * TILE_SIZE
        wy = map_row_start * TILE_SIZE
        ww = self.obs_width  * TILE_SIZE
        wh = self.obs_height * TILE_SIZE

        # ── Helper: world pixel coords → local grid cell, write value ────────
        # Keeps the highest-magnitude value when two entities share a tile.
        # For hazards this means spike (-1.0) beats enemy (+1.0) on equal
        # magnitude via the strict > check, so spike always wins conflicts.
        def _place(grid, obj_x, obj_y, value):
            lx = int(obj_x // TILE_SIZE) - map_col_start
            ly = int(obj_y // TILE_SIZE) - map_row_start
            if 0 <= lx < self.obs_width and 0 <= ly < self.obs_height:
                if abs(value) > abs(grid[ly, lx]):
                    grid[ly, lx] = value

        # ── Hazard hash: enemies (+1.0) and spikes (-1.0) ────────────────────
        # Single query, dispatched by type. Spike beats enemy on tile conflicts
        # because |-1.0| is not > |+1.0|, so spike value written first is kept.
        # Insert spikes after enemies so spike overwrites on equal magnitude.
        for h in self.physics_manager.hazard_hash.query_rect(wx, wy, ww, wh):
            if not h.gObj.active:
                continue
            if isinstance(h, Enemy):
                _place(hazard_grid, h.gObj.x, h.gObj.y, +1.0)
            elif isinstance(h, Spike):
                _place(hazard_grid, h.gObj.x, h.gObj.y, -1.0)

        # ── Collectible hash: goal (1.0), powerup (0.69), coin (0.35) ────────
        # Single query, dispatched by type. Goal wins all tile conflicts.
        for c in self.physics_manager.collectible_hash.query_rect(wx, wy, ww, wh):
            if not c.gObj.active:
                continue
            if hasattr(c, 'collected') and c.collected:
                continue
            if isinstance(c, Coin):
                _place(collect_grid, c.gObj.x, c.gObj.y, 0.35)
            elif isinstance(c, Powerup):
                _place(collect_grid, c.gObj.x, c.gObj.y, 0.69)
            else:
                # Goal or any unknown collectible — treat as highest importance
                _place(collect_grid, c.gObj.x, c.gObj.y, 1.0)

        # ── Static hash: ground, platforms, qblocks (skip spikes) ────────────
        # Spikes live in static_hash but are already handled via hazard_hash.
        for obj in self.level_data.static_hash.query_rect(wx, wy, ww, wh):
            if isinstance(obj, Spike):
                continue
            _place(solid_grid, obj.gObj.x, obj.gObj.y, 1.0)

        # ── Moving platforms: mark all covered tile columns ───────────────────
        # Multi-tile platforms span several columns — iterate all covered tiles.
        for plat in self.physics_manager.platform_hash.query_rect(wx, wy, ww, wh):
            if not plat.gObj.active:
                continue
            pc0 = int(plat.gObj.x // TILE_SIZE)
            pc1 = int((plat.gObj.x + plat.gObj.width - 1) // TILE_SIZE) + 1
            pr  = int(plat.gObj.y // TILE_SIZE)
            for pc in range(pc0, pc1):
                lx = pc - map_col_start
                ly = pr - map_row_start
                if 0 <= lx < self.obs_width and 0 <= ly < self.obs_height:
                    solid_grid[ly, lx] = 1.0

        # ── Pit detection ─────────────────────────────────────────────────────
        # A "pit" cell is an air tile that has no solid ground within
        # PIT_SCAN_DEPTH tiles directly below it (i.e. the agent would fall
        # to its death or off-screen if it stepped there).
        #
        # We write -0.5 into the hazard channel for these tiles so the agent
        # can distinguish "safe air" (hazard=0) from "fatal gap" (hazard=-0.5),
        # complementing the existing spike signal (-1.0).
        #
        # The value -0.5 sits between:
        #   0.0  = empty air   (safe)
        #  -0.5  = pit/gap     (lethal fall — this addition)
        #  -1.0  = spike tile  (always lethal)
        #
        # Both the solid channel AND the hazard channel are updated so the
        # agent gets the signal on two separate input planes.
        #   solid[ly, lx]  = -0.5  (was 0.0 = air, visually identical to safe air)
        #   hazard[ly, lx] = -0.5  (new — now agent can see the danger explicitly)
        #
        # PIT_SCAN_DEPTH: how many rows to scan downward per column.
        # 6 tiles ≈ 192px ≈ 3 player heights — enough to detect any
        # meaningful drop without scanning the entire level height.
        PIT_SCAN_DEPTH = 6

        grid     = self.level_data.grid
        map_rows = self.level_data.rows
        map_cols = self.level_data.cols

        # Precompute: a tile is solid if it is NOT air (TILE_AIR = 0) AND not spike.
        # We treat spikes as non-solid for the pit test because landing on a spike
        # IS a hazard (it's already in hazard_grid) — we don't want to suppress
        # the pit signal just because a spike sits at the bottom of a gap.
        def _is_solid_floor(map_row, map_col):
            if map_row < 0 or map_row >= map_rows:
                return False
            if map_col < 0 or map_col >= map_cols:
                return False
            t = grid[map_row][map_col]
            return t not in (TILE_AIR, TILE_SPIKE)

        for ly in range(self.obs_height):
            for lx in range(self.obs_width):
                # Only check cells that are currently air (solid=0) and
                # not already flagged as a hazard (enemy/spike)
                if solid_grid[ly, lx] != 0.0:
                    continue
                if hazard_grid[ly, lx] != 0.0:
                    continue

                map_row = map_row_start + ly
                map_col = map_col_start + lx

                # Skip tiles outside map bounds entirely (already OOB)
                if map_col < 0 or map_col >= map_cols:
                    continue

                # Scan downward for solid ground
                found_floor = False
                for scan in range(1, PIT_SCAN_DEPTH + 1):
                    scan_row = map_row + scan
                    if _is_solid_floor(scan_row, map_col):
                        found_floor = True
                        break

                if not found_floor:
                    # No floor within PIT_SCAN_DEPTH — this is a pit.
                    # Write pit marker into BOTH channels:
                    #   solid:  -0.5  (distinguishes from safe air 0.0)
                    #   hazard: -0.5  (distinguishes from spike -1.0 and enemy +1.0)
                    solid_grid[ly,  lx] = -0.5
                    hazard_grid[ly, lx] = -0.5

        # Cache for debug overlay visualization (AgentViewOverlay reads these)
        self._solid_window_cache  = solid_grid
        self._hazard_window_cache = hazard_grid

        return solid_grid, collect_grid, hazard_grid, map_row_start, map_col_start

    def _dijkstra_obs_window(self, map_row_start: int, map_col_start: int) -> np.ndarray:
        """
        Returns a (obs_height, obs_width) float32 advantage map in [-1, 1].

        Each cell encodes how much closer (positive) or further (negative) that
        tile is from the goal compared to the player's current tile:

            delta[r, c] = player_dist - window_dist

        Interpretation for the CNN:
        +1.0  → tile is much closer to the goal than the player  (go here)
         0.0  → same distance as player, or unreachable tile     (neutral)
        -1.0  → tile is much further from the goal than player   (avoid)

        All patching (moving-platform compensation) is done in raw cost space.
        Normalisation and clipping to [-1, 1] happen exactly once at the end.

        Tiles with inf cost (walls, unreachable air pockets) are set to 0 so
        they are neutral rather than misleadingly negative.
        """
        zero = np.zeros((self.obs_height, self.obs_width), dtype=np.float32)

        if self.dijkstra is None or not self.player:
            return zero

        px_tile = int(self.player.gObj.x // TILE_SIZE)
        py_tile = int(self.player.gObj.y // TILE_SIZE)
        player_dist = self.dijkstra.dist_map[py_tile, px_tile] \
            if (0 <= py_tile < self.level_data.rows and 0 <= px_tile < self.level_data.cols) \
            else np.inf

        if not np.isfinite(player_dist):
            return zero

        r0 = max(0, map_row_start)
        c0 = max(0, map_col_start)
        r1 = min(self.level_data.rows, map_row_start + self.obs_height)
        c1 = min(self.level_data.cols, map_col_start + self.obs_width)

        valid_rows = r1 - r0
        valid_cols = c1 - c0

        if valid_rows <= 0 or valid_cols <= 0:
            return zero

        dist_slice = self.dijkstra.dist_map[r0:r1, c0:c1]
        delta = player_dist - dist_slice
        delta[~np.isfinite(delta)] = 0.0

        out = np.zeros((self.obs_height, self.obs_width), dtype=np.float32)
        out_r0 = r0 - map_row_start
        out_c0 = c0 - map_col_start
        out[out_r0 : out_r0 + valid_rows, out_c0 : out_c0 + valid_cols] = delta

        # -----------------------------------------------------------------
        # Moving-platform compensation (raw cost space)
        # Mirrors DijkstraSolver's treatment of static solid tiles so the
        # gradient connects across gaps that moving platforms bridge.
        # -----------------------------------------------------------------
        if self.level_data.moving_platforms:
            wx = map_col_start * TILE_SIZE
            wy = map_row_start * TILE_SIZE
            ww = self.obs_width  * TILE_SIZE
            wh = self.obs_height * TILE_SIZE

            nearby_plats = self.physics_manager.platform_hash.query_rect(wx, wy, ww, wh)

            if nearby_plats:
                HORIZ_COST = 2.0
                MAX_SCAN   = self.obs_width * 2
                dm         = self.dijkstra.dist_map
                dm_rows, dm_cols = dm.shape

                PROXIMITY = [(1, 0.6), (2, 0.25), (3, 0.1)]

                for plat in nearby_plats:
                    if not plat.gObj.active:
                        continue

                    pc0       = int(plat.gObj.x // TILE_SIZE)
                    pc1       = int((plat.gObj.x + plat.gObj.width - 1) // TILE_SIZE) + 1
                    p_top_row = int(plat.gObj.y // TILE_SIZE)

                    # Platform surface: impassable (same as walls)
                    for pc in range(pc0, pc1):
                        ly = p_top_row - map_row_start
                        lx = pc - map_col_start
                        if 0 <= ly < self.obs_height and 0 <= lx < self.obs_width:
                            out[ly, lx] = 0.0

                    # Tiles above: ground-proximity discounts
                    for offset, discount in PROXIMITY:
                        patch_row = p_top_row - offset
                        if patch_row < 0 or patch_row >= dm_rows:
                            continue

                        for pc in range(pc0, pc1):
                            ly = patch_row - map_row_start
                            lx = pc - map_col_start
                            if not (0 <= ly < self.obs_height and 0 <= lx < self.obs_width):
                                continue

                            if 0 <= pc < dm_cols:
                                raw_cost = float(dm[patch_row, pc])
                            else:
                                raw_cost = np.inf

                            if np.isfinite(raw_cost):
                                patched_cost = max(1.0, raw_cost - discount)
                                out[ly, lx] = player_dist - patched_cost
                            else:
                                best_est = np.inf
                                for dc in range(1, MAX_SCAN + 1):
                                    for sign in (-1, 1):
                                        nc = pc + dc * sign
                                        if 0 <= nc < dm_cols:
                                            anchor = float(dm[patch_row, nc])
                                            if np.isfinite(anchor):
                                                est = anchor + dc * HORIZ_COST - discount
                                                if est < best_est:
                                                    best_est = est
                                    if np.isfinite(best_est):
                                        break

                                if np.isfinite(best_est):
                                    best_est = max(1.0, best_est)
                                    out[ly, lx] = player_dist - best_est

                    # Secondary proximity boosts (2 and 3 tiles above)
                    for offset, discount in [(2, 0.25), (3, 0.1)]:
                        boost_row = p_top_row - offset
                        if boost_row < 0:
                            continue
                        for pc in range(pc0, pc1):
                            ly = boost_row - map_row_start
                            lx = pc - map_col_start
                            if 0 <= ly < self.obs_height and 0 <= lx < self.obs_width:
                                out[ly, lx] += discount

                        # --- Tiles above: ground-proximity discounts ---
                        for offset, discount in PROXIMITY:
                            patch_row = p_top_row - offset
                            if patch_row < 0 or patch_row >= dm_rows:
                                continue

                            for pc in range(pc0, pc1):
                                ly = patch_row - map_row_start
                                lx = pc - map_col_start
                                if not (0 <= ly < self.obs_height and 0 <= lx < self.obs_width):
                                    continue

                                if 0 <= pc < dm_cols:
                                    raw_cost = float(dm[patch_row, pc])
                                else:
                                    raw_cost = np.inf

                                if np.isfinite(raw_cost):
                                    # Tile reachable in static map but missing the
                                    # ground-proximity discount. Recalculate advantage.
                                    patched_cost = max(1.0, raw_cost - discount)
                                    out[ly, lx] = player_dist - patched_cost
                                else:
                                    # Tile unreachable — interpolate from nearest
                                    # reachable tile in the same row of full dist_map.
                                    best_est = np.inf
                                    for dc in range(1, MAX_SCAN + 1):
                                        for sign in (-1, 1):
                                            nc = pc + dc * sign
                                            if 0 <= nc < dm_cols:
                                                anchor = float(dm[patch_row, nc])
                                                if np.isfinite(anchor):
                                                    est = anchor + dc * HORIZ_COST - discount
                                                    if est < best_est:
                                                        best_est = est
                                        if np.isfinite(best_est):
                                            break

                                    if np.isfinite(best_est):
                                        best_est = max(1.0, best_est)
                                        out[ly, lx] = player_dist - best_est

        # --- Single normalisation pass (after ALL patches) ---
        max_cost = max(
            (self.obs_width  // 2) * 2.0,   # horizontal half-span × horiz cost
            (self.obs_height // 2) * 3.5,   # vertical half-span × upward cost
        )
        np.clip(out / max_cost, -1.0, 1.0, out=out)
        self._dijkstra_window_cache = out  # cache for debugging visualization
        return out.astype(np.float32)

    def _tracking_obs(self) -> np.ndarray:
        """Returns 7 scalar features (+ 5 from _player_obs = 12 total)."""
        p = self.player
        if not p: return np.zeros(7, dtype=np.float32)

        SEARCH_RADIUS = math.sqrt(
            (self.obs_width  * TILE_SIZE) ** 2 +
            (self.obs_height * TILE_SIZE) ** 2
        ) * 0.5

        def get_dist_hash(hash_obj, check_collected=False, skip_spikes=False):
            """Query spatial hash within obs window, return (min_dist, count)."""
            min_d = 9999.0
            count = 0
            nearby = hash_obj.query_rect(
                p.gObj.x - SEARCH_RADIUS, p.gObj.y - SEARCH_RADIUS,
                SEARCH_RADIUS * 2, SEARCH_RADIUS * 2
            )
            for obj in nearby:
                if not obj.gObj.active: continue
                if check_collected and hasattr(obj, 'collected') and obj.collected: continue
                if skip_spikes and getattr(obj.gObj, 'type_id', None) == EntityType.SPIKE: continue
                dx = p.gObj.x - obj.gObj.x
                dy = p.gObj.y - obj.gObj.y
                d_sq = dx*dx + dy*dy
                if d_sq < min_d:
                    min_d = d_sq
                count += 1
            return (math.sqrt(min_d) if min_d < 9999.0 else 9999.0), count

        e_dist, e_count = get_dist_hash(self.physics_manager.hazard_hash, skip_spikes=True)
        c_dist, c_count = get_dist_hash(self.physics_manager.collectible_hash, check_collected=True)

        raw_goal_dist = getattr(self, '_goal_dist_cache', self._get_dist_to_goal())
        norm_dist = max(self.level_data.width, self.level_data.height, 1.0)

        if self.level_data.goals:
            closest = min(self.level_data.goals, key=lambda g: abs(g.gObj.x - p.gObj.x))
            closest_goal_x = closest.gObj.x
            closest_goal_y = closest.gObj.y
        else:
            closest_goal_x = self.level_data.width
            closest_goal_y = p.gObj.y

        dx = closest_goal_x - p.gObj.x
        dy = closest_goal_y - p.gObj.y
        dir_x       = np.sign(dx)
        dist_y_norm = np.clip(dy / self.level_data.height, -1.0, 1.0) if self.level_data.height > 0 else 0.0

        dijkstra_dist = 1.0
        if self.dijkstra:
            px_tile = int(p.gObj.x // TILE_SIZE)
            py_tile = int(p.gObj.y // TILE_SIZE)
            d = self.dijkstra.get_dist(px_tile, py_tile)
            if d >= 0:
                dijkstra_dist = np.clip(d / (self.level_data.cols * 2), 0.0, 1.0)

        step_dx, step_dy = 0.0, 0.0
        if self.dijkstra:
            px_tile = int(p.gObj.x // TILE_SIZE)
            py_tile = int(p.gObj.y // TILE_SIZE)

            rows = self.level_data.rows
            cols = self.level_data.cols
            dm   = self.dijkstra.dist_map

            best_cost = (float(dm[py_tile, px_tile])
                         if 0 <= py_tile < rows and 0 <= px_tile < cols
                         else math.inf)
            best_ddx, best_ddy = 0, 0

            for ddx, ddy in [( 1, 0), (-1, 0), ( 0, 1), ( 0,-1),
                              ( 1,-1), (-1,-1), ( 1, 1), (-1, 1)]:
                nr, nc = py_tile + ddy, px_tile + ddx
                if 0 <= nr < rows and 0 <= nc < cols:
                    cost = float(dm[nr, nc])
                    if cost < best_cost:
                        best_cost = cost
                        best_ddx, best_ddy = ddx, ddy

            mag = math.sqrt(best_ddx * best_ddx + best_ddy * best_ddy)
            if mag > 0:
                step_dx = best_ddx / mag
                step_dy = best_ddy / mag

        self._step_dx = step_dx
        self._step_dy = step_dy

        # Dijkstra ablation (dijkstra_enabled=False) must hide the solver
        # prior from the POLICY's scalars too — _obs() already zeroes grid
        # channel 3, but without this gate the exact distance + unit
        # direction-to-goal would still leak through scalars [4]/[5]/[6],
        # silently invalidating any "without Dijkstra" ablation run.
        # Only the OBSERVED values are masked: self._step_dx/_step_dy and
        # the solver itself stay live because _info() -> the adept persona's
        # PBRS/alignment reward legitimately consume them even when the
        # observation is ablated (reward ablation is a separate axis).
        if not self.dijkstra_enabled:
            dijkstra_dist = 1.0            # neutral default (same as no-solver)
            step_dx, step_dy = 0.0, 0.0

        return np.array([
            np.clip(e_dist        / norm_dist, 0.0, 1.0),  # [0] enemy dist
            np.clip(raw_goal_dist / norm_dist, 0.0, 1.0),  # [1] goal dist
            np.clip(self.timer    / max(1.0, self.timer_seconds), 0.0, 1.0),  # [2] timer
            dist_y_norm,                                    # [3] goal delta-Y
            dijkstra_dist,                                  # [4] dijkstra dist
            step_dx,                                        # [5] best step X
            step_dy,                                        # [6] best step Y
        ], dtype=np.float32)

    def _curriculum_win_rate(self) -> float:
        """
        Returns the win rate over the current level's sliding window [0.0, 1.0].
        Returns -1.0 if the window is empty (no data yet for this level).
        Useful for TensorBoard and CSV logging via _info().
        """
        window = self._level_window.get(self.world, deque())
        if not window:
            return -1.0
        return sum(window) / len(window)

    def _info(self) -> Dict:
        p = self.player
        ts = float(TILE_SIZE)

        event = ""
        cause = getattr(self, 'death_cause', "")
        if self.reached_goal:
            event = "WIN"
            cause = "Goal"
        elif not self.alive:
            event = "DIED"

        if not p:
            return {
                "score": self.score, "score_delta": self.score_delta, "frame_count": self.frame,
                "x_position": 0.0, "y_position": 0.0, "velocity_x": 0.0, "velocity_y": 0.0,
                "coins_collected": self.coins_total, "enemies_killed_step": self.kills_step,
                "powered_up": False, "terminated": not self.alive, "won": False,
                "action": self._last_action, "action_name": action_to_str(self._last_action), "time_left": math.ceil(self.timer),
                "max_x_seen": self.max_x_seen, "stall_windows": self.stall_windows_count,
                "stalled": self.stalled_this_frame, "persona": self.persona,
                "level": self.world, "goal_dist": 0.0, "lives": self.lives,
                "event": event, "cause": cause,
                "on_ground": False,
                "step_dx":   0.0,
                "step_dy":   0.0,
                "curriculum_level_idx": self.current_index_world,
                "curriculum_win_rate":  self._curriculum_win_rate(),
                "curriculum_max_unlocked": self._max_unlocked_index,
                **self._obs_stats
            }

        dijkstra_dist = 0.0
        if self.dijkstra:
            px_tile = int(p.gObj.x // TILE_SIZE)
            py_tile = int(p.gObj.y // TILE_SIZE)
            d = self.dijkstra.get_dist(px_tile, py_tile)
            self.dijkstra_current_tile = d
            if d >= 0:
                dijkstra_dist = np.clip(d / (self.level_data.cols * 2), 0.0, 1.0)
            else:
                dijkstra_dist = -1.0

        return {
            "score": self.score,
            "score_delta": self.score_delta,
            "frame_count": self.frame,
            "x_position": p.gObj.x / ts,
            "y_position": p.gObj.y / ts,
            "velocity_x": p.vx / ts,
            "velocity_y": p.vy / ts,
            "coins_collected": self.coins_total,
            "enemies_killed_step": self.kills_step,
            "powered_up": p.powered_up,
            "terminated": not self.alive,
            "won": self.reached_goal,
            "action": self._last_action,
            "action_name": action_to_str(self._last_action),
            "time_left": math.ceil(self.timer),
            "max_x_seen": self.max_x_seen,
            "stall_windows": self.stall_windows_count,
            "stalled": self.stalled_this_frame,
            "persona": self.persona,
            "level": self.world,
            "goal_dist": getattr(self, '_goal_dist_cache', self._get_dist_to_goal()) / ts,
            "lives": self.lives,
            "event": event,
            "cause": cause,
            "dijkstra_dist": dijkstra_dist,
            "on_ground": p.on_ground,
            "on_moving_platform": getattr(p, '_on_moving_platform', False),
            "step_dx":   self._step_dx,
            "step_dy":   self._step_dy,
            # Curriculum diagnostics — useful for TensorBoard / CSV logging
            "curriculum_level_idx": self.current_index_world,
            "curriculum_win_rate":  self._curriculum_win_rate(),
            "curriculum_max_unlocked": self._max_unlocked_index,
            **self._obs_stats
        }

    def get_jump_arc_debug_state(self) -> dict:
        if not self.player:
            return {}

        p = self.player
        ctx = self.physics_manager.context
        origin_x = p.gObj.x + p.gObj.width * 0.5
        origin_y = p.gObj.y + p.gObj.height
        grounded = bool(p.on_ground)
        can_jump = bool(p.on_ground or p.coyote > 0)
        preview_jump = grounded and can_jump

        if preview_jump:
            preview_speed = max(180.0, float(getattr(ctx, "MAX_RUN_SPEED", 240.0)) * 0.6)
            preview_vx = p.vx if abs(p.vx) > 10.0 else (preview_speed if p.facing_right else -preview_speed)
            preview_vy = float(getattr(ctx, "JUMP_VEL_MIN", -620.0))
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
            "color": (80, 220, 80) if preview_jump else (80, 190, 255),
        }

    def render(self, surface: pygame.Surface, blit_only: bool = True):
        if self.render_mode == "human":
            game_surf = surface.subsurface(pygame.Rect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT))
        else:
            game_surf = surface

        game_surf.fill(COLOR_SKY)
        self._draw_world(game_surf)
        self._draw_entities(game_surf)
        self._draw_player(game_surf)

        if self.debug_manager:
            self.debug_manager.render_overlays(surface, self)

            if self.debug_manager.show_sensors and hasattr(self, 'last_rays'):
                from .modules.System.debugging_mods.overlays import (
                    RAY_EMPTY, RAY_SOLID, RAY_HAZARD, RAY_COIN, RAY_GOAL)
                ray_surf = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
                for start, end, found, rtype in self.last_rays:
                    if   rtype == 0.0: color = RAY_EMPTY
                    elif rtype == 1.0: color = RAY_SOLID
                    elif rtype == 2.0: color = RAY_HAZARD
                    elif rtype == 3.0: color = RAY_COIN
                    elif rtype == 4.0: color = RAY_GOAL
                    else:              color = RAY_EMPTY
                    s_cam = (start[0] - self.camera_x, start[1] - self.camera_y)
                    e_cam = (end[0]   - self.camera_x, end[1]   - self.camera_y)
                    pygame.draw.line(ray_surf, color, s_cam, e_cam, 1)
                game_surf.blit(ray_surf, (0, 0))

        self._draw_ui(game_surf)

    def _draw_world(self, surface: pygame.Surface):
        visible_tiles = self.level_data.static_hash.query_rect(self.camera_x, self.camera_y, self.WIDTH, self.HEIGHT)
        for tile in visible_tiles:
            if tile.x + tile.width < self.camera_x or tile.x > self.camera_x + self.WIDTH: continue

            if isinstance(tile, Spike):
                tile.render(surface, tile.gObj.x - self.camera_x, tile.gObj.y - self.camera_y)
            elif isinstance(tile, Tile):
                if tile.color == COLOR_QBLOCK: continue
                tile.render(surface, self.camera_x, self.camera_y)
            elif isinstance(tile, QuestionBlock):
                tile.render(surface, tile.x - self.camera_x, tile.y - self.camera_y)
            elif hasattr(tile, 'render'):
                try:
                    tile.render(surface, self.camera_x, self.camera_y)
                except TypeError:
                    tile.render(surface, tile.x - self.camera_x, tile.y - self.camera_y)

        visible_platforms = self.physics_manager.platform_hash.query_rect(
            self.camera_x, self.camera_y, self.WIDTH, self.HEIGHT
        )
        for plat in visible_platforms:
            if plat.gObj.active:
                plat.render(surface, plat.gObj.x - self.camera_x, plat.gObj.y - self.camera_y)

    def _draw_entities(self, surface: pygame.Surface):
        cx, cy, cw, ch = self.camera_x, self.camera_y, self.WIDTH, self.HEIGHT

        visible_hazards = self.physics_manager.hazard_hash.query_rect(cx, cy, cw, ch)
        for entity in visible_hazards:
            if hasattr(entity, 'render'):
                entity.render(surface, entity.x - cx, entity.y - cy)

        visible_collectibles = self.physics_manager.collectible_hash.query_rect(cx, cy, cw, ch)
        for entity in visible_collectibles:
            if hasattr(entity, 'render'):
                entity.render(surface, entity.x - cx, entity.y - cy)

        for proj in self.level_data.projectiles:
            if proj.gObj.active:
                proj.render(surface, proj.gObj.x - cx, proj.gObj.y - cy)

    def _draw_player(self, surface: pygame.Surface):
        p = self.player
        if not p: return
        sx = p.gObj.x - self.camera_x
        sy = p.gObj.y - self.camera_y

        colour = COLOR_POWERUP_STAR if (p.invincible_timer > 0 and (self.frame // 5) % 2) else \
            ((255, 100, 0) if p.powered_up else (255, 0, 0))
        p.color = colour
        p.render(surface, sx, sy, self.debug_manager.show_sensors)

    def _draw_ui(self, surface: pygame.Surface):
        status = "STAR" if (self.player and self.player.star_timer > 0) else \
                ("SUPER" if (self.player and self.player.powered_up) else "SMALL")
        text = self.ui_font.render(
            f"Lives:{self.lives}  Score:{self.score}  Coins:{self.coins_total}  {status}  Time:{int(self.timer)}",
            True, COLOR_WHITE
        )
        bg = pygame.Surface((text.get_width() + 10, text.get_height() + 6), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 170))

        y = 5
        if self.debug_manager.show_obs_panel: y = self.HEIGHT - text.get_height() - 10
        surface.blit(bg, (5, y))
        surface.blit(text, (10, y + 3))

    def _world_to_screen(self, gObj: GameObject) -> Tuple[float, float, bool]:
        screen_x = gObj.x - self.camera_x
        screen_y = gObj.y - self.camera_y
        on_screen = (
            screen_x < SCREEN_WIDTH and
            screen_x + gObj.width > 0 and
            screen_y < SCREEN_HEIGHT and
            screen_y + gObj.height > 0)
        return screen_x, screen_y, on_screen

    def close(self):
        """Proper resource cleanup."""
        if self.render_mode == "human":
            pygame.quit()
