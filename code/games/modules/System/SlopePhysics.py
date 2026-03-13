"""
SlopePhysics.py
===============
Slope-aware collision detection and physics for the Sonic engine.

This module provides functions that sonic_core.py calls AFTER the standard
AABB physics pass. It handles:

  1. **Floor snapping** — When Sonic walks over a slope tile, his Y position
     is set to the height-map surface rather than the tile's flat top edge.

  2. **Slope sticking** — At high speeds, Sonic should stay glued to slopes
     rather than launching off the top of a hill. A downward probe keeps him
     attached when on_ground was True last frame.

  3. **Slope speed modifiers** — Running uphill is slower, downhill is faster.
     The angle from SlopeTile.get_surface_angle() drives a sine-based modifier.

  4. **Slope landing** — When Sonic lands on a slope from the air, the standard
     AABB floor snap (tile top) is corrected to the actual height-map surface.

Integration
───────────
In sonic_core.py step(), AFTER physics_manager.update_system:

    from .modules.Objects.SlopePhysics import resolve_slopes, apply_slope_speed

    resolve_slopes(self.player, self.slope_tiles, self.level_data)
    apply_slope_speed(self.player, self.slope_tiles, self.dt)
"""

from __future__ import annotations
import math
from typing import List, Optional

try:
    from .SlopeTile import SlopeTile, SlopeType, TILE_SIZE
except ImportError:
    from SlopeTile import SlopeTile, SlopeType
    TILE_SIZE = 32


# ── Constants ────────────────────────────────────────────────────────────────
SNAP_PROBE_DEPTH   = 16     # Pixels below feet to probe for slope surface
STICK_PROBE_DEPTH  = 24     # Deeper probe when already on ground (slope sticking)
SLOPE_SPEED_FACTOR = 0.08   # How strongly slopes affect speed (tune this)
MIN_STICK_SPEED    = 40.0   # Minimum horizontal speed to engage slope sticking


def find_slope_at(world_x: float, world_y: float,
                  slope_tiles: List[SlopeTile]) -> Optional[SlopeTile]:
    """
    Find the slope tile whose column contains world_x and whose vertical
    range includes world_y (with a small probe tolerance below).

    Returns None if no slope tile is at that position.
    """
    for st in slope_tiles:
        if st.gObj.x <= world_x < st.gObj.x + st.gObj.width:
            # Check if the point is within the tile's vertical range
            # (with some tolerance for probing below)
            if st.gObj.y - 4 <= world_y < st.gObj.y + st.gObj.height + STICK_PROBE_DEPTH:
                return st
    return None


def find_slope_at_feet(player, slope_tiles: List[SlopeTile]) -> Optional[SlopeTile]:
    """
    Find the slope tile under Sonic's feet (center-bottom of hitbox).
    """
    foot_x = player.gObj.x + player.gObj.width / 2
    foot_y = player.gObj.y + player.gObj.height
    return find_slope_at(foot_x, foot_y, slope_tiles)


def resolve_slopes(player, slope_tiles: List[SlopeTile], level_data=None):
    """
    Main slope collision resolution. Call once per frame AFTER AABB physics.

    1. Find the slope tile under Sonic's feet.
    2. Read the height-map surface Y at Sonic's horizontal center.
    3. Snap Sonic's Y to the surface if he's within probe range.
    4. Set on_ground if snapped.

    Parameters
    ----------
    player : SonicPlayer
        The player object.
    slope_tiles : list[SlopeTile]
        All slope tiles in the current level.
    level_data : LevelData, optional
        Not currently used but available for future extensions.
    """
    if not player or not slope_tiles:
        return

    foot_x = player.gObj.x + player.gObj.width / 2
    foot_y = player.gObj.y + player.gObj.height

    # Determine probe depth: deeper when already on ground (for sticking)
    was_on_ground = player.on_ground
    probe = STICK_PROBE_DEPTH if was_on_ground else SNAP_PROBE_DEPTH

    # ── Find slope tile at feet ──────────────────────────────────────────
    slope = None
    for st in slope_tiles:
        # Horizontal check: foot_x within tile column
        if not (st.gObj.x <= foot_x < st.gObj.x + st.gObj.width):
            continue

        surface_y = st.get_surface_y(foot_x)

        # Is Sonic close enough to snap? (within probe distance below surface)
        # foot_y should be near or slightly below the surface
        if surface_y - 2 <= foot_y <= surface_y + probe:
            slope = st
            break

        # Also check if Sonic is INSIDE the slope (embedded due to fast movement)
        if foot_y > surface_y and foot_y < st.gObj.y + st.gObj.height:
            slope = st
            break

    if slope is None:
        # ── Also check slopes AHEAD for approaching ramps ────────────────
        # When running toward a slope, check slightly ahead so Sonic
        # transitions smoothly onto the ramp
        if was_on_ground and abs(player.vx) > MIN_STICK_SPEED:
            ahead_x = foot_x + math.copysign(8, player.vx)
            for st in slope_tiles:
                if not (st.gObj.x <= ahead_x < st.gObj.x + st.gObj.width):
                    continue
                surface_y = st.get_surface_y(ahead_x)
                if surface_y - 4 <= foot_y <= surface_y + probe:
                    slope = st
                    foot_x = ahead_x  # Use the ahead position for snapping
                    break

    if slope is None:
        return

    # ── Snap to surface ──────────────────────────────────────────────────
    surface_y = slope.get_surface_y(foot_x)

    # Distance from Sonic's feet to the slope surface
    delta = foot_y - surface_y

    if -2 <= delta <= probe:
        # Snap Sonic so his feet sit exactly on the surface
        player.gObj.y = surface_y - player.gObj.height

        if player.vy >= 0:
            # Landing on or walking along the slope
            player.vy = 0
            player.on_ground = True
            player.jump_hold = 0


def apply_slope_speed(player, slope_tiles: List[SlopeTile], dt: float):
    """
    Apply slope-based speed modifiers.

    When Sonic is on the ground and on a slope tile:
      - Ascending (positive angle): decelerate
      - Descending (negative angle): accelerate
      - The effect scales with the sine of the slope angle

    This makes running uphill harder and downhill easier, which is
    the core feel of Sonic's momentum gameplay.
    """
    if not player or not player.on_ground or not slope_tiles:
        return

    slope = find_slope_at_feet(player, slope_tiles)
    if slope is None or slope.slope_type == SlopeType.FLAT:
        return

    foot_x = player.gObj.x + player.gObj.width / 2
    angle_deg = slope.get_surface_angle(foot_x)

    if abs(angle_deg) < 2.0:
        return  # Too shallow to matter

    angle_rad = math.radians(angle_deg)
    sin_angle = math.sin(angle_rad)

    # Gravity component along the slope surface
    # Positive angle = ascending right, sin > 0 → decelerate when moving right
    # Negative angle = descending right, sin < 0 → accelerate when moving right
    gravity_component = 1100.0 * sin_angle * SLOPE_SPEED_FACTOR

    if player.vx > 0:
        # Moving right
        player.vx -= gravity_component * dt
    elif player.vx < 0:
        # Moving left (slope effect is reversed)
        player.vx += gravity_component * dt

    # Rolling gets extra slope effect (Sonic rolls faster downhill)
    try:
        from ..Objects.SonicPlayer import SonicState
    except ImportError:
        from SonicPlayer import SonicState
    if hasattr(player, 'state') and player.state == SonicState.ROLLING:
        roll_bonus = 1100.0 * sin_angle * SLOPE_SPEED_FACTOR * 0.5
        if player.vx > 0:
            player.vx -= roll_bonus * dt
        elif player.vx < 0:
            player.vx += roll_bonus * dt


def get_slope_angle_at(player, slope_tiles: List[SlopeTile]) -> float:
    """
    Returns the slope angle in degrees at Sonic's current position.
    0 = flat, positive = ascending right, negative = descending right.
    """
    slope = find_slope_at_feet(player, slope_tiles)
    if slope is None:
        return 0.0
    foot_x = player.gObj.x + player.gObj.width / 2
    return slope.get_surface_angle(foot_x)