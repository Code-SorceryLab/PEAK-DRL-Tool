"""
MovingPlatform.py
-----------------
A solid platform that travels back and forth between two world-space points
at a fixed speed, and carries any entity standing on it.

Design
------
- Two waypoints: ``start`` and ``end`` (world-pixel coordinates).
- Moves linearly from start → end, then end → start, and so on (ping-pong).
- Exposes ``delta_x`` / ``delta_y`` — the displacement this frame — so the
  PhysicsManager (or game core) can push the player along with it.

PhysicsManager integration
---------------------------
Add a ``moving_platforms`` list to your level data and update the physics loop:

    # In PhysicsManager.update_system, after player update:
    for plat in level_data.moving_platforms:
        plat.update(dt)

    # In resolve_collisions, after _resolve_player_world:
    _resolve_player_moving_platforms(core, level_data.moving_platforms)

Then add this helper (or incorporate into PhysicsManager):

    def _resolve_player_moving_platforms(core, platforms):
        player = core.player
        if not player: return
        prect = player.gObj.get_rect()

        for plat in platforms:
            prect_exp = prect.inflate(0, 4)  # 4px downward probe
            if prect_exp.colliderect(plat.gObj.get_rect()):
                # Check player is above platform centre (standing on top)
                if player.gObj.y + player.gObj.height <= plat.gObj.y + plat.gObj.height / 2 + 6:
                    # Carry: push player by the platform's displacement this frame
                    player.gObj.x += plat.delta_x
                    player.gObj.y += plat.delta_y
                    # Snap to top surface
                    player.gObj.y = plat.gObj.y - player.gObj.height
                    player.vy = 0
                    player.on_ground = True

LevelLoader integration
-----------------------
Platforms are not encoded in the ASCII grid (they're dynamic). Load them from
a sidecar JSON / YAML section in the level file, e.g.:

    moving_platforms:
      - start: [320, 256]
        end:   [640, 256]
        speed: 80
        width: 96
        height: 16
"""

from __future__ import annotations

import math
import pygame
from dataclasses import dataclass, field
from typing import Tuple

from .GameObject import GameObject
from ..System.EntityType import EntityType
from ..Parameters.Map_parameters import TILE_SIZE, COLOR_PLATFORM


# Visual colours (match the static platform palette)
_COL_BODY: tuple = (205, 133,  63)   # Tan — same as static platform
_COL_TOP:  tuple = (230, 165,  90)   # Lighter highlight strip on the top edge
_COL_EDGE: tuple = (160,  90,  30)   # Darker left/right edge accent


@dataclass
class MovingPlatform:
    """
    A platform that travels linearly between two world-space waypoints.

    Parameters
    ----------
    gObj : GameObject
        Bounding box (position = start position initially).
    start : Tuple[float, float]
        World-pixel coordinate (x, y) of waypoint A.
    end : Tuple[float, float]
        World-pixel coordinate (x, y) of waypoint B.
    speed : float
        Movement speed in pixels per second.
    solid : bool
        True — players and enemies are pushed out of it like a normal tile.
    """

    gObj:  GameObject
    start: Tuple[float, float]
    end:   Tuple[float, float]
    speed: float               = 80.0
    solid: bool                = True

    # ── Internal state (not part of public API) ───────────────────────────────
    _direction: int   = field(default=1,   init=False, repr=False)  # 1 = toward end, -1 = toward start
    _delta_x:   float = field(default=0.0, init=False, repr=False)
    _delta_y:   float = field(default=0.0, init=False, repr=False)

    def __post_init__(self):
        self.gObj.type_id = EntityType.TILE  # treated as a solid tile for collision
        # Place at start position
        self.gObj.x = float(self.start[0])
        self.gObj.y = float(self.start[1])

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def x(self) -> float:
        return self.gObj.x

    @property
    def y(self) -> float:
        return self.gObj.y

    @property
    def width(self) -> int:
        return self.gObj.width

    @property
    def height(self) -> int:
        return self.gObj.height

    @property
    def active(self) -> bool:
        return self.gObj.active

    @property
    def delta_x(self) -> float:
        """Horizontal displacement applied this frame (pixels). Used to carry the player."""
        return self._delta_x

    @property
    def delta_y(self) -> float:
        """Vertical displacement applied this frame (pixels). Used to carry the player."""
        return self._delta_y

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_points(
        cls,
        start:  Tuple[float, float],
        end:    Tuple[float, float],
        speed:  float = 80.0,
        width:  int   = TILE_SIZE * 3,   # 96 px — three tiles wide
        height: int   = TILE_SIZE // 2,  # 16 px — half a tile tall
    ) -> "MovingPlatform":
        """
        Convenience constructor.

        Parameters
        ----------
        start, end : (x, y) in world pixels
        speed : pixels per second
        width, height : platform dimensions in pixels
        """
        gobj = GameObject(
            x=float(start[0]),
            y=float(start[1]),
            width=width,
            height=height,
            active=True,
        )
        return cls(gObj=gobj, start=start, end=end, speed=speed)

    # ── Update ────────────────────────────────────────────────────────────────

    def update(self, dt: float, context=None) -> None:
        """
        Move the platform toward the current target waypoint.

        When the platform reaches (or passes) a waypoint it snaps to it and
        reverses direction, producing smooth ping-pong motion.

        Parameters
        ----------
        dt : float
            Delta-time in seconds.
        context : PhysicsContext, optional
            Not used — present for interface compatibility with update_list.
        """
        target = self.end if self._direction == 1 else self.start

        # Vector from current position to target
        dx = target[0] - self.gObj.x
        dy = target[1] - self.gObj.y
        dist = math.hypot(dx, dy)

        # How far can we travel this frame?
        move = self.speed * dt

        if dist <= move:
            # Snap to waypoint and reverse
            self._delta_x = target[0] - self.gObj.x
            self._delta_y = target[1] - self.gObj.y
            self.gObj.x = target[0]
            self.gObj.y = target[1]
            self._direction *= -1
        else:
            # Move toward target
            ratio = move / dist
            self._delta_x = dx * ratio
            self._delta_y = dy * ratio
            self.gObj.x += self._delta_x
            self.gObj.y += self._delta_y

    # ── Rendering ─────────────────────────────────────────────────────────────

    def render(
        self,
        surface: pygame.Surface,
        sx: float,
        sy: float,
        debug: bool = False,
    ) -> None:
        """
        Draw the platform at screen position (sx, sy).

        Visual layers
        -------------
        1. Main body rectangle.
        2. Light highlight strip along the top edge (same style as static platforms).
        3. Dark accent on the left and right edges.
        4. Optional debug: velocity arrow and bounding box.
        """
        w = self.gObj.width
        h = self.gObj.height
        isx, isy = int(sx), int(sy)

        # 1. Body
        pygame.draw.rect(surface, _COL_BODY, (isx, isy, w, h))

        # 2. Top highlight strip
        hi_h = max(2, h // 4)
        pygame.draw.rect(surface, _COL_TOP, (isx, isy, w, hi_h))

        # 3. Edge accents (1 px wide on left/right)
        pygame.draw.line(surface, _COL_EDGE, (isx, isy), (isx, isy + h - 1), 2)
        pygame.draw.line(surface, _COL_EDGE, (isx + w - 1, isy), (isx + w - 1, isy + h - 1), 2)

        # 4. Debug overlays
        if debug:
            # Bounding box
            pygame.draw.rect(surface, (64, 255, 128), (isx, isy, w, h), 1)

            # Velocity arrow showing direction of travel
            cx = isx + w // 2
            cy = isy + h // 2
            target = self.end if self._direction == 1 else self.start
            dx = target[0] - self.gObj.x
            dy = target[1] - self.gObj.y
            dist = math.hypot(dx, dy)
            if dist > 0:
                ax = int(cx + (dx / dist) * 12)
                ay = int(cy + (dy / dist) * 12)
                pygame.draw.line(surface, (255, 220, 0), (cx, cy), (ax, ay), 2)
                pygame.draw.circle(surface, (255, 220, 0), (ax, ay), 3)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @property
    def travel_length(self) -> float:
        """Total pixel distance between the two waypoints."""
        return math.hypot(
            self.end[0] - self.start[0],
            self.end[1] - self.start[1],
        )

    @property
    def cycle_duration(self) -> float:
        """Time in seconds to complete one full A→B→A cycle."""
        return (2.0 * self.travel_length) / max(self.speed, 1e-6)

    def __repr__(self) -> str:
        return (
            f"<MovingPlatform "
            f"pos=({self.gObj.x:.0f},{self.gObj.y:.0f}) "
            f"start={self.start} end={self.end} "
            f"speed={self.speed} dir={'→end' if self._direction == 1 else '→start'}>"
        )