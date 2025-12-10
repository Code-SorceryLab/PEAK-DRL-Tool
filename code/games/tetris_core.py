from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np
import pygame
from gymnasium import spaces

# ================================================================================
# DISPLAY & BOARD CONFIGURATION
# ================================================================================

SCREEN_SCALE = 24                                    # Pixels per cell
WIDTH_CELLS, HEIGHT_CELLS = 6, 20                    # Board dimensions (columns x rows)
PLAYFIELD_W = WIDTH_CELLS * SCREEN_SCALE             # Playfield width in pixels
PLAYFIELD_H = HEIGHT_CELLS * SCREEN_SCALE            # Playfield height in pixels
SIDEBAR_W = 160                                      # Sidebar width for UI elements
WINDOW_W = PLAYFIELD_W + SIDEBAR_W                   # Total window width
WINDOW_H = PLAYFIELD_H                               # Total window height

# ================================================================================
# OBSERVATION SPACE LAYOUT (FLAT VECTOR)
# ================================================================================
# Observation vector breakdown (total: 297 elements):
# [0:120]     - Current board state (20x6 grid, flattened, binary 0/1)
# [120:127]   - Current piece ID (one-hot encoded, 7 piece types)
# [127:131]   - Current piece rotation (one-hot encoded, 4 rotations)
# [131:133]   - Current piece position (normalized x, y)
# [133:168]   - Next queue (5 pieces x 7 one-hot = 35 elements)
# [168:175]   - Hold piece (one-hot encoded, 7 piece types)
# [175:176]   - Hold available flag (1 if can use hold, 0 if not)
# [176:296]   - Predicted board after ghost drop (20x6 grid, flattened, binary)
# [296:297]   - Predicted lines cleared (normalized 0-1)

QUEUE_K = 5                                          # Number of next pieces to track
OBS_LEN = 120 + 7 + 4 + 2 + 35 + 7 + 1 + 120 + 1   # Total observation length = 297

# ================================================================================
# ACTION SPACE
# ================================================================================
# Action IDs:
# 0: No-op (do nothing)
# 1: Move left
# 2: Move right
# 3: Rotate clockwise
# 4: Soft drop (move down one row)
# 5: Hard drop (instant drop to bottom)

N_ACTIONS = 6

# ================================================================================
# SCORING
# ================================================================================

LINE_SCORES = {1: 100, 2: 300, 3: 500, 4: 800}      # Points for clearing 1-4 lines

# ================================================================================
# TETROMINO PIECE DEFINITIONS
# ================================================================================
# Each piece has 4 rotations, each rotation is a list of (dy, dx) offsets
# from the piece's anchor point

PIECES = {
    0: [  # I-piece (straight line)
        [(0, -1), (0, 0), (0, 1), (0, 2)],           # Horizontal
        [(-1, 1), (0, 1), (1, 1), (2, 1)],           # Vertical
        [(1, -1), (1, 0), (1, 1), (1, 2)],           # Horizontal (shifted)
        [(-1, 0), (0, 0), (1, 0), (2, 0)],           # Vertical (shifted)
    ],
    1: [  # O-piece (square) - all rotations are identical
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
    ],
    2: [  # T-piece
        [(0, -1), (0, 0), (0, 1), (1, 0)],
        [(-1, 0), (0, 0), (1, 0), (0, 1)],
        [(0, -1), (0, 0), (0, 1), (-1, 0)],
        [(-1, 0), (0, 0), (1, 0), (0, -1)],
    ],
    3: [  # S-piece
        [(0, 0), (0, 1), (1, -1), (1, 0)],
        [(-1, 0), (0, 0), (0, 1), (1, 1)],
        [(0, 0), (0, 1), (1, -1), (1, 0)],
        [(-1, 0), (0, 0), (0, 1), (1, 1)],
    ],
    4: [  # Z-piece
        [(0, -1), (0, 0), (1, 0), (1, 1)],
        [(-1, 1), (0, 0), (0, 1), (1, 0)],
        [(0, -1), (0, 0), (1, 0), (1, 1)],
        [(-1, 1), (0, 0), (0, 1), (1, 0)],
    ],
    5: [  # J-piece
        [(0, -1), (0, 0), (0, 1), (1, -1)],
        [(-1, 0), (0, 0), (1, 0), (1, -1)],
        [(0, -1), (0, 0), (0, 1), (-1, 1)],
        [(-1, 0), (0, 0), (1, 0), (1, 1)],
    ],
    6: [  # L-piece
        [(0, -1), (0, 0), (0, 1), (1, 1)],
        [(-1, 0), (0, 0), (1, 0), (1, 1)],
        [(0, -1), (0, 0), (0, 1), (-1, -1)],
        [(-1, 0), (0, 0), (1, 0), (1, -1)],
    ],
}

# ================================================================================
# PIECE DATACLASS
# ================================================================================

@dataclass
class Piece:
    """Represents a Tetris piece with position and rotation."""
    pid: int                                          # Piece ID (0-6)
    y: int                                            # Row position (0 = top)
    x: int                                            # Column position
    rot: int = 0                                      # Rotation state (0-3)
    
    def cells(self) -> List[Tuple[int, int]]:
        """Returns absolute (y, x) positions of all cells in this piece."""
        return [(self.y + dy, self.x + dx) for (dy, dx) in PIECES[self.pid][self.rot % 4]]

# ================================================================================
# BOARD ANALYSIS HELPER FUNCTIONS
# ================================================================================

def _col_heights(board: np.ndarray) -> np.ndarray:
    """
    Calculate the height of each column on the board.
    Height = number of rows from bottom to the topmost filled cell.
    Returns array of shape (WIDTH_CELLS,) with heights for each column.
    """
    h = np.zeros(WIDTH_CELLS, dtype=np.int32)
    for x in range(WIDTH_CELLS):
        col = board[:, x]
        nz = np.where(col != 0)[0]                    # Find filled cells
        h[x] = (HEIGHT_CELLS - nz[0]) if nz.size else 0
    return h


def _holes(board: np.ndarray) -> int:
    """
    Count total holes on the board.
    A hole = an empty cell with at least one filled cell above it in the same column.
    """
    holes = 0
    for x in range(WIDTH_CELLS):
        seen_filled = False
        for y in range(HEIGHT_CELLS):
            if board[y, x]:                           # Filled cell
                seen_filled = True
            elif seen_filled:                         # Empty cell below filled cell = hole
                holes += 1
    return int(holes)


def _bumpiness(board: np.ndarray) -> int:
    """
    Calculate bumpiness (surface roughness).
    Bumpiness = sum of absolute height differences between adjacent columns.
    Lower bumpiness = flatter, more stable board.
    """
    h = _col_heights(board)
    return int(np.abs(np.diff(h)).sum())


def _max_height(board: np.ndarray) -> int:
    """Return the maximum column height on the board."""
    return int(_col_heights(board).max())

# ================================================================================
# TETRIS CORE GAME CLASS WITH ANTI-EXPLOIT MECHANICS
# ================================================================================

class TetrisCore:
    """
    Main Tetris game logic with RL-compatible interface.
    
    Features:
    - Predictive grid generation for ghost piece simulation
    - Anti-exploit mechanics (rotation limits, movement limits)
    - Lock delay system with reset limits
    - 7-bag randomization for fair piece distribution
    """
    
    WIDTH = WINDOW_W
    HEIGHT = WINDOW_H

    def __init__(self, render_mode: str = "none", **kwargs):
        """
        Initialize Tetris game.
        
        Args:
            render_mode: Rendering mode ("human", "rgb_array", or "none")
            **kwargs: Game configuration parameters
        """
        
        # -------------------- Game Configuration --------------------
        self.level = 1
        self.lines_per_level = int(kwargs.pop("lines_per_level", 10))
        self.gravity_start = int(kwargs.pop("gravity_start", 16))     # Initial gravity delay (frames)
        self.gravity_min = int(kwargs.pop("gravity_min", 2))          # Minimum gravity delay
        self.gravity_decay = float(kwargs.pop("gravity_decay", 0.90)) # Gravity acceleration per level
        self.gravity_every = self.gravity_start
        self._lines_into_level = 0
        self.score = 0

        # -------------------- Lock Delay Mechanics --------------------
        self.spawn_grace_frames = int(kwargs.pop("spawn_grace_frames", 6))
        self._spawn_cooldown = 0
        self.lock_delay_frames = int(kwargs.pop("lock_delay_frames", 6))
        self.lock_reset_limit = int(kwargs.pop("lock_reset_limit", 4))
        self._lock_counter = 0
        self._lock_resets_remaining = self.lock_reset_limit
        self.resting_frames = 0
        self.rotations_while_resting = 0

        # -------------------- Anti-Exploit: Rotation Limiting --------------------
        self.rotation_cooldown_frames = int(kwargs.pop("rotation_cooldown_frames", 3))
        self._rotation_cooldown = 0                                   # Frames until next rotation allowed
        self.max_rotations_per_piece = int(kwargs.pop("max_rotations_per_piece", 15))
        self._rotations_this_piece = 0                                # Total rotations for current piece

        # -------------------- Anti-Exploit: Movement Limiting --------------------
        self.max_moves_while_resting = int(kwargs.pop("max_moves_while_resting", 8))
        self._moves_while_resting = 0                                 # Horizontal moves while piece is resting

        # -------------------- Win Condition --------------------
        self.max_level = int(kwargs.pop("max_level", 15))
        self.win_on_max_level = bool(kwargs.pop("win_on_max_level", True))
        self._won = False

        # -------------------- Observation & Action Spaces --------------------
        self._obs_space = spaces.Box(0.0, 1.0, shape=(OBS_LEN,), dtype=np.float32)
        self._act_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = self._obs_space
        self.action_space = self._act_space

        # -------------------- Game State --------------------
        self.board = np.zeros((HEIGHT_CELLS, WIDTH_CELLS), dtype=np.int8)
        self.curr: Optional[Piece] = None                              # Current falling piece
        self.hold_id: Optional[int] = None                             # Held piece ID
        self.hold_used: bool = False                                   # Hold used this turn?
        self.lines_cleared: int = 0                                    # Total lines cleared
        self.next_queue: List[int] = []                                # Upcoming pieces
        self.next_id: int = 0                                          # Next piece to spawn
        self.gravity_timer = 0
        self.alive = True
        
        # -------------------- Random Number Generator (7-bag system) --------------------
        self.rng = np.random.RandomState(1337)
        self._bag: List[int] = list(range(7))                          # Bag for fair piece distribution
        self.rng.shuffle(self._bag)

        # -------------------- Rendering --------------------
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()
        self._surf = pygame.Surface((self.WIDTH, self.HEIGHT))
        self._just_locked = False

    # ================================================================================
    # PREDICTIVE SIMULATION (Ghost Piece)
    # ================================================================================

    def _simulate_drop_grid_and_lines(self, piece: Piece) -> Tuple[np.ndarray, int]:
        """
        Simulate dropping the current piece to its ghost position, locking it,
        and clearing any completed lines.
        
        Returns:
            pred_grid: Flattened binary board state after drop (120 elements)
            num_clears: Number of lines that would be cleared
        """
        temp_board = self.board.copy()
        ghost = Piece(piece.pid, piece.y, piece.x, piece.rot)
        
        # Drop ghost piece to lowest valid position
        while self._valid(Piece(ghost.pid, ghost.y + 1, ghost.x, ghost.rot)):
            ghost = Piece(ghost.pid, ghost.y + 1, ghost.x, ghost.rot)
        
        # Place ghost piece on board
        for y, x in ghost.cells():
            if 0 <= y < HEIGHT_CELLS and 0 <= x < WIDTH_CELLS:
                temp_board[y, x] = ghost.pid + 1
        
        # Clear completed lines
        full_rows = [y for y in range(HEIGHT_CELLS) if np.all(temp_board[y, :] != 0)]
        if full_rows:
            keep = [y for y in range(HEIGHT_CELLS) if y not in full_rows]
            newb = np.zeros_like(temp_board)
            newb[-len(keep):] = temp_board[keep, :]
            temp_board = newb
        
        return (temp_board > 0).astype(np.float32).reshape(-1), len(full_rows)

    # ================================================================================
    # GYMNASIUM COMPATIBILITY
    # ================================================================================

    def get_action_space(self):
        """Return action space for wrapper compatibility."""
        return self.action_space

    def get_observation_space(self):
        """Return observation space for wrapper compatibility."""
        return self.observation_space

    # ================================================================================
    # PIECE GENERATION (7-BAG SYSTEM)
    # ================================================================================

    def _draw_from_bag(self) -> int:
        """
        Draw next piece from 7-bag system.
        Ensures all 7 pieces appear once before any repeats.
        """
        if not self._bag:
            self._bag = list(range(7))
            self.rng.shuffle(self._bag)
        return int(self._bag.pop())

    def _spawn(self, pid: Optional[int] = None):
        """
        Spawn a new piece at the top of the board.
        Resets all per-piece counters (rotation, movement, lock delay).
        
        Args:
            pid: Optional piece ID to spawn (if None, draws from queue)
        """
        if pid is None:
            if not self.next_queue:
                self.next_queue = [self._draw_from_bag() for _ in range(QUEUE_K)]
            pid = int(self.next_queue.pop(0))
            self.next_queue.append(self._draw_from_bag())
        
        self.curr = Piece(pid=int(pid), y=0, x=WIDTH_CELLS // 2, rot=0)
        self.hold_used = False
        
        # Reset lock delay mechanics
        self._lock_resets_remaining = self.lock_reset_limit
        self._lock_counter = 0
        self.resting_frames = 0
        self.rotations_while_resting = 0
        
        # Reset anti-exploit counters
        self._rotation_cooldown = 0
        self._rotations_this_piece = 0
        self._moves_while_resting = 0
        
        self.next_id = self.next_queue[0]
        
        # Check for instant game over (piece can't spawn)
        if not self._valid(self.curr):
            self.alive = False

    # ================================================================================
    # COLLISION DETECTION & PIECE STATE
    # ================================================================================

    def _valid(self, piece: Piece) -> bool:
        """Check if piece position is valid (no collisions)."""
        for y, x in piece.cells():
            if not (0 <= x < WIDTH_CELLS and 0 <= y < HEIGHT_CELLS):
                return False                          # Out of bounds
            if self.board[y, x] != 0:
                return False                          # Collision with existing blocks
        return True

    def _is_resting(self, piece: Piece) -> bool:
        """Check if piece is resting on the ground or another piece."""
        below = Piece(piece.pid, piece.y + 1, piece.x, piece.rot)
        return not self._valid(below)

    def _consume_reset_if_lifted(self, was_resting: bool, now_resting: bool):
        """Reset lock timer if piece was lifted off the ground (allows player adjustment)."""
        if was_resting and not now_resting and self._lock_resets_remaining > 0:
            self._lock_counter = 0
            self._lock_resets_remaining -= 1

    # ================================================================================
    # PIECE MOVEMENT (WITH ANTI-EXPLOIT LIMITS)
    # ================================================================================

    def _move(self, dx: int):
        """
        Move piece horizontally with anti-wiggle protection.
        Forces lock after too many horizontal moves while resting.
        
        Args:
            dx: Direction (-1 for left, +1 for right)
        """
        if not self.curr:
            return
        
        was_resting = self._is_resting(self.curr)
        
        # Anti-exploit: Force lock if too many moves while resting (wiggling)
        if was_resting and self._moves_while_resting >= self.max_moves_while_resting:
            self._lock()
            self._spawn()
            self._lock_counter = 0
            self._spawn_cooldown = self.spawn_grace_frames
            return
        
        p = Piece(self.curr.pid, self.curr.y, self.curr.x + dx, self.curr.rot)
        if self._valid(p):
            self.curr = p
            now_resting = self._is_resting(self.curr)
            
            # Track movements while resting
            if now_resting:
                self._moves_while_resting += 1
            
            self._consume_reset_if_lifted(was_resting, now_resting)

    def _rotate(self):
        """
        Rotate piece clockwise with anti-spin protection.
        Includes cooldown between rotations and max rotations per piece.
        Wall kicks try positions: center, left-1, right+1
        """
        if not self.curr:
            return
        
        # Anti-exploit: Enforce rotation cooldown
        if self._rotation_cooldown > 0:
            return
        
        # Anti-exploit: Enforce max rotations per piece (prevent infinite spinning)
        if self._rotations_this_piece >= self.max_rotations_per_piece:
            return
        
        was_resting = self._is_resting(self.curr)
        p = Piece(self.curr.pid, self.curr.y, self.curr.x, (self.curr.rot + 1) % 4)
        
        # Try rotation with wall kicks
        rotation_succeeded = False
        if self._valid(p):
            self.curr = p
            rotation_succeeded = True
        elif self._valid(Piece(p.pid, p.y, p.x - 1, p.rot)):
            self.curr = Piece(p.pid, p.y, p.x - 1, p.rot)
            rotation_succeeded = True
        elif self._valid(Piece(p.pid, p.y, p.x + 1, p.rot)):
            self.curr = Piece(p.pid, p.y, p.x + 1, p.rot)
            rotation_succeeded = True
        
        if not rotation_succeeded:
            return
        
        # Successful rotation - apply cooldown and increment counter
        self._rotation_cooldown = self.rotation_cooldown_frames
        self._rotations_this_piece += 1
        
        now_resting = self._is_resting(self.curr)
        self._consume_reset_if_lifted(was_resting, now_resting)
        
        if now_resting:
            self.rotations_while_resting += 1

    def _soft_drop(self):
        """Move piece down one row (soft drop)."""
        if not self.curr:
            return
        p = Piece(self.curr.pid, self.curr.y + 1, self.curr.x, self.curr.rot)
        if self._valid(p):
            self.curr = p
            self._lock_counter = 0
        else:
            # Piece can't move down - increment lock counter
            self._lock_counter += 1
            if self._lock_counter >= self.lock_delay_frames:
                self._lock()
                self._spawn()
                self._lock_counter = 0
                self._spawn_cooldown = self.spawn_grace_frames

    def _hard_drop(self):
        """Instantly drop piece to lowest valid position (hard drop)."""
        if not self.curr:
            return 0
        steps = 0
        while True:
            p = Piece(self.curr.pid, self.curr.y + 1, self.curr.x, self.curr.rot)
            if self._valid(p):
                self.curr = p
                steps += 1
            else:
                break
        self._lock()
        self._spawn()
        self._lock_counter = 0
        self._spawn_cooldown = self.spawn_grace_frames
        return steps

    # ================================================================================
    # PIECE LOCKING & LINE CLEARING
    # ================================================================================

    def _lock(self):
        """Lock current piece into the board."""
        if not self.curr:
            return
        for y, x in self.curr.cells():
            if y >= 0:                                # Only place visible cells
                self.board[y, x] = self.curr.pid + 1
        self.curr = None
        self._just_locked = True

    def _clear_lines(self) -> int:
        """
        Clear all completed lines and move remaining lines down.
        
        Returns:
            Number of lines cleared
        """
        full_rows = [y for y in range(HEIGHT_CELLS) if np.all(self.board[y, :] != 0)]
        if not full_rows:
            return 0
        keep = [y for y in range(HEIGHT_CELLS) if y not in full_rows]
        newb = np.zeros_like(self.board)
        newb[-len(keep):] = self.board[keep, :]
        self.board = newb
        return len(full_rows)

    # ================================================================================
    # ENVIRONMENT INTERFACE (reset & step)
    # ================================================================================

    def reset(self):
        """Reset game to initial state."""
        self.board[:, :] = 0
        self.level = 1
        self.gravity_every = self.gravity_start
        self._lines_into_level = 0
        self.score = 0
        self._spawn_cooldown = 0
        self._won = False
        self.hold_id = None
        self.hold_used = False
        self.lines_cleared = 0
        self.gravity_timer = 0
        self.alive = True
        self._just_locked = False
        
        # Reset anti-exploit counters
        self._rotation_cooldown = 0
        self._rotations_this_piece = 0
        self._moves_while_resting = 0
        
        self.next_queue = [self._draw_from_bag() for _ in range(QUEUE_K)]
        self.next_id = self.next_queue[0]
        self._spawn()
        return self._obs()

    def step(self, action: int):
        """
        Execute one game step with anti-exploit mechanics.
        
        Args:
            action: Action ID (0-5)
            
        Returns:
            obs: Observation vector
            reward: Base reward (0.0, reward shaping done externally)
            terminated: Whether episode has ended
            info: Dict with game stats
        """
        if not self.alive:
            return self._obs(), 0.0, True, {"episode_end": True, "won": bool(self._won)}

        self._just_locked = False
        a = int(action)

        # Decrement rotation cooldown each frame
        if self._rotation_cooldown > 0:
            self._rotation_cooldown -= 1
        
        # Execute action
        if a == 1:
            self._move(-1)
        elif a == 2:
            self._move(+1)
        elif a == 3:
            self._rotate()
        elif a == 4:
            self._soft_drop()
        elif a == 5:
            self._hard_drop()

        # Apply gravity (auto drop) if not manually dropping
        if self.curr is not None and a not in (4, 5):
            self.gravity_timer += 1
            if self.gravity_timer >= self.gravity_every:
                self.gravity_timer = 0
                p = Piece(self.curr.pid, self.curr.y + 1, self.curr.x, self.curr.rot)
                if self._valid(p):
                    self.curr = p
                    self._lock_counter = 0

        # Lock delay system (piece resting on ground)
        if self.curr is not None:
            if self._is_resting(self.curr):
                self.resting_frames += 1
                self._lock_counter += 1
                if self._lock_resets_remaining <= 0:
                    self._lock_counter = max(self._lock_counter, self.lock_delay_frames - 2)
                if self._lock_counter >= self.lock_delay_frames:
                    self._lock()
                    self._spawn()
                    self._lock_counter = 0
                    self._spawn_cooldown = self.spawn_grace_frames
            else:
                self.resting_frames = 0
                self.rotations_while_resting = 0
                self._lock_counter = 0

        # Clear lines and update score
        lines = 0
        if self._just_locked:
            lines = self._clear_lines()
            self.score += LINE_SCORES.get(lines, 0)
            self.lines_cleared += lines
            self._lines_into_level += lines
            
            # Level up
            while self._lines_into_level >= self.lines_per_level and self.level < self.max_level:
                self._lines_into_level -= self.lines_per_level
                self.level += 1
                self.gravity_every = max(self.gravity_min, int(self.gravity_every * self.gravity_decay))
                if self.level >= self.max_level and self.win_on_max_level:
                    self._won = True
        
        done = (not self.alive) or self._won
        
        obs = self._obs()

        base_reward = 0.0 
        # Info dict for RL/logging
        info = {
            "score": int(self.score),
            "level": int(self.level),
            "lines": int(self.lines_cleared),
            "episode_end": bool(done),
            "won": bool(self._won),
            "lines_delta": int(lines),
            "just_locked": bool(self._just_locked),
            "resting_frames": int(self.resting_frames),
            "rot_rest": int(self.rotations_while_resting),
        }
        
                                    # Reward shaping done externally
        return obs, float(base_reward), bool(done), info

    # ================================================================================
    # OBSERVATION BUILDER
    # ================================================================================

    def _one_hot7(self, i: int) -> np.ndarray:
        """Create one-hot encoding for piece ID (7 types)."""
        v = np.zeros(7, dtype=np.float32)
        if 0 <= i < 7:
            v[i] = 1.0
        return v

    def _one_hot4(self, i: int) -> np.ndarray:
        """Create one-hot encoding for rotation (4 states)."""
        v = np.zeros(4, dtype=np.float32)
        v[i % 4] = 1.0
        return v

    def _obs(self) -> np.ndarray:
        """
        Build flat observation vector (297 elements).
        
        Layout:
        - Current board (120)
        - Current piece info (7 + 4 + 2 = 13)
        - Next queue (35)
        - Hold info (7 + 1 = 8)
        - Predicted board after drop (120)
        - Predicted lines cleared (1)
        """
        # Current board state (flattened binary grid)
        grid = (self.board > 0).astype(np.float32).reshape(-1)
        
        # Current piece information
        if self.curr is None:
            cur_id, rot, y, x = 0, 0, 0, WIDTH_CELLS // 2
            pred_grid = np.zeros_like(grid)
            clears_norm = np.array([0.0], dtype=np.float32)
        else:
            cur_id, rot, y, x = self.curr.pid, self.curr.rot % 4, self.curr.y, self.curr.x
            pred_grid, pred_clears = self._simulate_drop_grid_and_lines(self.curr)
            clears_norm = np.array([pred_clears / 4.0], dtype=np.float32)
        
        cur_onehot = self._one_hot7(cur_id)
        rot_onehot = self._one_hot4(rot)
        pos_norm = np.array([x / (WIDTH_CELLS - 1), y / (HEIGHT_CELLS - 1)], dtype=np.float32)
        
        # Next queue (K upcoming pieces)
        nxt = []
        for i in range(QUEUE_K):
            nid = self.next_queue[i] if i < len(self.next_queue) else 0
            nxt.append(self._one_hot7(nid))
        next_vec = np.concatenate(nxt, axis=0).astype(np.float32)
        
        # Hold piece information
        hold_onehot = self._one_hot7(self.hold_id if self.hold_id is not None else 0)
        hold_avail = np.array([0.0 if self.hold_used else 1.0], dtype=np.float32)

        # Compose final observation vector
        obs = np.concatenate([
            grid,                                      # [0:120] Current board
            cur_onehot,                                # [120:127] Current piece ID
            rot_onehot,                                # [127:131] Current rotation
            pos_norm,                                  # [131:133] Current position
            next_vec,                                  # [133:168] Next queue
            hold_onehot,                               # [168:175] Hold piece
            hold_avail,                                # [175:176] Hold available?
            pred_grid,                                 # [176:296] Predicted board
            clears_norm                                # [296:297] Predicted clears
        ], axis=0)
        
        if obs.shape[0] != OBS_LEN:
            raise RuntimeError(f"Obs size {obs.shape[0]} != {OBS_LEN}")
        
        return obs.astype(np.float32)

    # ================================================================================
    # RENDERING
    # ================================================================================

    def render(self, screen=None, blit_only=False, mode="human"):
        """Render the game state visually."""
        surf = self._surf if screen is None else screen
        surf.fill((18, 18, 18))                        # Dark background
        cell = SCREEN_SCALE
        pf_w, pf_h = PLAYFIELD_W, PLAYFIELD_H
        sidebar_x = pf_w

        # Draw playfield background
        pygame.draw.rect(surf, (12, 12, 14), pygame.Rect(0, 0, pf_w, pf_h))
        pygame.draw.rect(surf, (40, 40, 50), pygame.Rect(0, 0, pf_w, pf_h), width=2)

        # Draw grid lines
        grid_major, grid_minor = (66, 66, 76), (42, 42, 50)
        for y in range(HEIGHT_CELLS + 1):
            color = grid_major if y % 5 == 0 else grid_minor
            width = 2 if y % 5 == 0 else 1
            pygame.draw.line(surf, color, (0, y * cell), (pf_w, y * cell), width)
        for x in range(WIDTH_CELLS + 1):
            color = grid_major if x % 5 == 0 else grid_minor
            width = 2 if x % 5 == 0 else 1
            pygame.draw.line(surf, color, (x * cell, 0), (x * cell, pf_h), width)

        # Piece colors (one for each of 7 piece types)
        piece_colors = [
            (180, 228, 255),  # I - Cyan
            (255, 250, 120),  # O - Yellow
            (179, 130, 255),  # T - Purple
            (140, 255, 130),  # S - Green
            (255, 110, 110),  # Z - Red
            (130, 160, 255),  # J - Blue
            (255, 165, 60),   # L - Orange
        ]

        def draw_cell(cx, cy, color, alpha=255):
            """Draw a single cell with optional transparency."""
            r = pygame.Rect(cx * cell, cy * cell, cell - 2, cell - 2)
            if alpha < 255:
                tmp = pygame.Surface((cell - 2, cell - 2), pygame.SRCALPHA)
                tmp.fill((*color, alpha))
                surf.blit(tmp, r.topleft)
            else:
                pygame.draw.rect(surf, color, r, border_radius=3)
            pygame.draw.rect(surf, (25, 25, 30), r, width=1, border_radius=3)

        # Draw locked pieces
        for y in range(HEIGHT_CELLS):
            for x in range(WIDTH_CELLS):
                if val := self.board[y, x]:
                    draw_cell(x, y, piece_colors[val - 1])

        # Draw ghost piece and current piece
        if self.curr:
            # Ghost piece (semi-transparent preview of landing position)
            ghost = Piece(self.curr.pid, self.curr.y, self.curr.x, self.curr.rot)
            while self._valid(below := Piece(ghost.pid, ghost.y + 1, ghost.x, ghost.rot)):
                ghost = below
            for (gy, gx) in ghost.cells():
                if 0 <= gy < HEIGHT_CELLS and 0 <= gx < WIDTH_CELLS:
                    draw_cell(gx, gy, piece_colors[self.curr.pid], alpha=70)
            
            # Current piece (solid)
            for (cy, cx) in self.curr.cells():
                if 0 <= cy < HEIGHT_CELLS and 0 <= cx < WIDTH_CELLS:
                    draw_cell(cx, cy, piece_colors[self.curr.pid])

        # Draw sidebar
        sidebar_bg = pygame.Rect(sidebar_x, 0, SIDEBAR_W, pf_h)
        pygame.draw.rect(surf, (16, 16, 18), sidebar_bg)
        pygame.draw.line(surf, (40, 40, 50), (sidebar_x, 0), (sidebar_x, pf_h), 2)
        
        big = pygame.font.Font(None, 32)
        surf.blit(big.render("Next", True, (255, 255, 255)), (sidebar_x + 18, 18))
        
        # Draw next piece preview
        px0, py0 = sidebar_x + 18, 56
        for (py_, px_) in PIECES[self.next_id][0]:
            r = pygame.Rect(px0 + px_ * cell, py0 + py_ * cell, cell - 2, cell - 2)
            pygame.draw.rect(surf, piece_colors[self.next_id], r, border_radius=3)
            pygame.draw.rect(surf, (25, 25, 30), r, width=1, border_radius=3)

        # Draw score and level
        y_hud = py0 + 5 * cell + 14
        surf.blit(big.render(f"Score: {self.score}", True, (255, 255, 255)), (sidebar_x + 18, y_hud))
        surf.blit(big.render(f"Lvl: {self.level}", True, (255, 255, 255)), (sidebar_x + 18, y_hud + 32))

        # Display update
        if screen is not None:
            if not blit_only:
                pygame.display.flip()
        else:
            if mode == "human":
                from pygame import display
                display.set_mode((self.WIDTH, self.HEIGHT))
                display.get_surface().blit(surf, (0, 0))
                pygame.display.flip()
            else:
                return surf
