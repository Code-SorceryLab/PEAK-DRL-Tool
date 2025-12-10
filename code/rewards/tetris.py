from __future__ import annotations
import random
from typing import Callable, Tuple, Dict
import numpy as np

# ================================================================================
# SAFE FLOAT CONVERSION
# ================================================================================
def _safe_float(x) -> float:
    """
    Convert x to float safely, returning 0.0 if x is non-finite or on error.
    Prevents NaN/Inf from breaking reward calculations.
    """
    try:
        x = float(x)
        if not np.isfinite(x):
            return 0.0
        return x
    except Exception:
        return 0.0

# ================================================================================
# OBSERVATION PARSING / GRID EXTRACT
# ================================================================================
def grids_from_obs(obs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract current and predicted grids from flat observation vector.
    Observation format:
      - [0:120] Current board (20x6 grid, flattened)
      - [176:296] Predicted board after ghost drop (20x6 grid, flattened)
    """
    arr = np.array(obs, dtype=np.float32)
    if arr.ndim == 1 and arr.size >= 297:
        cur_grid = arr[:120].reshape(20, 6)
        pred_grid = arr[176:296].reshape(20, 6)
        return cur_grid, pred_grid
    return np.zeros((20, 6), dtype=np.float32), np.zeros((20, 6), dtype=np.float32)

# ================================================================================
# BOARD FEATURE EXTRACTION
# ================================================================================
def _col_heights(grid: np.ndarray) -> np.ndarray:
    H, W = grid.shape
    h = np.zeros(W, dtype=np.int32)
    for x in range(W):
        col = grid[:, x]
        nz = np.where(col > 0)[0]
        h[x] = (H - nz[0]) if nz.size else 0
    return h

def _holes(grid: np.ndarray) -> int:
    H, W = grid.shape
    holes = 0
    for x in range(W):
        filled_above = False
        for y in range(H):
            if grid[y, x] > 0:
                filled_above = True
            elif filled_above:
                holes += 1
    return int(holes)

def _bumpiness(heights: np.ndarray) -> int:
    return int(np.sum(np.abs(np.diff(heights))))

def _aggregate_height(heights: np.ndarray) -> int:
    return int(np.sum(heights))

def _max_height(heights: np.ndarray) -> int:
    return int(np.max(heights))

def _deepest_well(heights: np.ndarray) -> Tuple[int, int]:
    W = heights.size
    best_depth = 0
    best_pos = 0
    for i in range(W):
        left = heights[i-1] if i-1 >= 0 else heights[i]
        right = heights[i+1] if i+1 < W else heights[i]
        neigh_min = min(left, right)
        depth = max(0, int(neigh_min - heights[i]))
        if depth > best_depth:
            best_depth = depth
            best_pos = i
    return best_depth, best_pos

# ================================================================================
# STRUCTURE EXTRACTION: ALL FEATURES
# ================================================================================
def structure_from_obs_both(obs) -> Dict[str, float]:
    cur_grid, pred_grid = grids_from_obs(obs)
    heights_cur = _col_heights(cur_grid)
    heights_pred = _col_heights(pred_grid)

    well_cur, well_cur_pos = _deepest_well(heights_cur)
    well_pred, well_pred_pos = _deepest_well(heights_pred)

    max_h_cur = int(np.max(heights_cur))
    is_floor_cur = bool(np.all(heights_cur == max_h_cur))
    max_h_pred = int(np.max(heights_pred))
    is_floor_pred = bool(np.all(heights_pred == max_h_pred))

    return {
        # Current features
        "holes_cur": _holes(cur_grid),
        "bumpiness_cur": _bumpiness(heights_cur),
        "avg_height_cur": float(np.mean(heights_cur)),
        "aggregate_height_cur": _aggregate_height(heights_cur),
        "max_height_cur": _max_height(heights_cur),
        "deepest_well_cur": well_cur,
        "deepest_well_cur_pos": well_cur_pos,
        "is_floor_cur": is_floor_cur,
        # Predicted features
        "holes_pred": _holes(pred_grid),
        "bumpiness_pred": _bumpiness(heights_pred),
        "avg_height_pred": float(np.mean(heights_pred)),
        "aggregate_height_pred": _aggregate_height(heights_pred),
        "max_height_pred": _max_height(heights_pred),
        "deepest_well_pred": well_pred,
        "deepest_well_pred_pos": well_pred_pos,
        "is_floor_pred": is_floor_pred,
    }

def structure_from_obs(obs) -> Dict[str, float]:
    cur_grid, _ = grids_from_obs(obs)
    h = _col_heights(cur_grid)
    well_depth, well_pos = _deepest_well(h)
    return {
        "holes": float(_holes(cur_grid)),
        "bumpiness": float(_bumpiness(h)),
        "avg_height": float(np.mean(h)) if h.size else 0.0,
        "aggregate_height": float(_aggregate_height(h)),
        "max_height": float(_max_height(h)),
        "deepest_well": float(well_depth),
        "deepest_well_pos": float(well_pos),
    }

# ================================================================================
# REWARD WRAPPER / TRACKER
# ================================================================================
class _LineTracker:
    def __init__(self):
        self.prev_lines = 0
        self.prev_score = 0
    def step(self, info: dict) -> Tuple[int, int, bool]:
        lines = int(info.get("lines", 0))
        score = int(info.get("score", 0))
        inc = lines > self.prev_lines
        self.prev_lines = lines
        self.prev_score = score
        return lines, score, inc
    def reset(self):
        self.prev_lines = 0
        self.prev_score = 0

def _wrap_with_tracker(core_fn) -> Callable:
    tracker = _LineTracker()
    def reward(obs, base, terminated: bool, info: dict) -> float:
        # Extract info fields
        lines_delta = int(info.get("lines_delta", 0))
        just_locked = bool(info.get("just_locked", False))
        rot_rest = int(info.get("rot_rest", 0))
        resting_frames = int(info.get("resting_frames", 0))
        total_lines, score, inc = tracker.step(info)
        level = int(info.get("level", 1))
        # Call core reward function (note: pass all info as arguments!)
        r = core_fn(
            lines_delta, inc, terminated, obs, level, score, just_locked, rot_rest, resting_frames
        )
        # Reset tracker at episode end
        if terminated or info.get("episode_end", False):
            tracker.reset()
        return _safe_float(r)
    return reward

# ================================================================================
# REWARD FUNCTIONS (main formulas)
# ================================================================================

@_wrap_with_tracker
def tetris_hunter(lines_delta, inc, terminated, obs, level, score, just_locked, rot_rest, resting_frames):
    """
    Main reward function for RL Tetris agent.
    - Discourages excessive horizontal moves while resting (resting_frames)
    - Strong line clear and edge-well bonuses
    - Penalizes holes, bumpiness, stack height, and center wells
    """
    features = structure_from_obs_both(obs)
    well_pos = int(features["deepest_well_pred_pos"])
    board_width = 6
    r = -0.02

    # Extract next piece IDs
    next_pieces = np.array(obs[133:168]).reshape(5, 7)  # One-hot, next queue
    next_piece_ids = np.argmax(next_pieces, axis=1)

    # --- Next-piece lookahead bonus ---
    next_pieces = np.array(obs[133:168]).reshape(5, 7)
    next_piece_ids = np.argmax(next_pieces, axis=1)
    if 0 in next_piece_ids:  # 'I' piece coming
        if features["deepest_well_pred"] >= 4 and (well_pos == 0 or well_pos == board_width-1):
            r += 2.5

    if rot_rest > 0:
        r -= 0.02 * float(rot_rest)
    if just_locked:
        if lines_delta == 4: r += 18.0
        elif lines_delta == 3: r += 4.0
        elif lines_delta == 2: r += 1.5
        elif lines_delta == 1: r += 0.5
        if features["deepest_well_pred"] >= 4 and (well_pos == 0 or well_pos == board_width-1):
            r += 2.0
        if features["deepest_well_pred"] >= 3 and (well_pos != 0 and well_pos != board_width-1):
            r -= 1.0
        r -= features["holes_pred"] * 1.1
        r -= features["bumpiness_pred"] * 0.13
        r -= 0.28 * max(0, features["max_height_pred"] - 10)
        if features["is_floor_pred"]:
            r += 1.5
    # Penalize excessive horizontal moves while resting
    if resting_frames > 4:
        r -= 0.25 * (resting_frames - 4)
    if terminated:
        r -= 12.0
    return r

@_wrap_with_tracker
def shaped(lines_delta, inc, terminated, obs, level, score, just_locked, rot_rest, resting_frames):
    """
    Alternate advanced reward using predictive grid deltas.
    (Keep for ablation comparisons)
    """
    features = structure_from_obs_both(obs)
    # Deltas
    holes_pred = features["holes_pred"]
    bumpiness_pred = features["bumpiness_pred"]
    avg_height_pred = features["avg_height_pred"]
    aggregate_height_pred = features["aggregate_height_pred"]
    max_height_pred = features["max_height_pred"]
    well_depth = features["deepest_well_pred"]
    well_pos = features["deepest_well_pred_pos"]
    is_floor_pred = features["is_floor_pred"]
    holes_delta = features["holes_cur"] - holes_pred
    bumpiness_delta = features["bumpiness_cur"] - bumpiness_pred
    height_delta = features["avg_height_cur"] - avg_height_pred

    r = -0.01
    if rot_rest > 0:
        r -= 0.10 * float(rot_rest)
    if just_locked:
        r += 0.5
        dl = int(lines_delta)
        if dl == 1: r += 1.2
        elif dl == 2: r += 2.5
        elif dl == 3: r += 5.0
        elif dl == 4: r += 15.0
        # Reward improvement deltas
        r += 1.0 * holes_delta
        r += 0.3 * bumpiness_delta
        r += 0.2 * height_delta
        # Penalize bad predicted features
        r -= 1.2 * holes_pred
        r -= 0.4 * bumpiness_pred
        r -= 0.3 * avg_height_pred
        r -= 0.5 * aggregate_height_pred
        if max_height_pred > 15: r -= 2.0
        elif max_height_pred > 12: r -= 0.5
        if is_floor_pred:
            r += 1.5
        elif well_depth >= 4:
            board_width = 6
            edge_well = (well_pos == 0 or well_pos == (board_width - 1))
            if edge_well: r += 0.5
            else: r -= 1.0
    if terminated:
        r -= 10.0
    return r

@_wrap_with_tracker
def baseline(lines_delta, inc, terminated, obs, level, score, just_locked, rot_rest, resting_frames):
    """Random baseline for debugging. DO NOT use for training."""
    return random.random() - 0.5

@_wrap_with_tracker
def simple(lines_delta, inc, terminated, obs, level, score, just_locked, rot_rest, resting_frames):
    """Simple baseline reward."""
    r = 0.1
    if inc:
        r += 1.0
    if terminated:
        r -= 2.0
    return r
