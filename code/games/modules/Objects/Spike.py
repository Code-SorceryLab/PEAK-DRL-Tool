"""
Spike.py
--------
A static hazard tile that kills the player on contact.

Rendering: dark grey base rectangle with a lighter grey triangle pointing up,
matching the editor's `^` preview.

Collision: The gObj bounding box is used for physics. EntityType.SPIKE is
already handled in PhysicsManager._resolve_player_world — it calls
core._handle_death() on contact, so no extra code is needed there.

LevelLoader integration
-----------------------
Wherever `^` is read from the ASCII map, create a Spike instead of a plain Tile:

    if ch == '^':
        spike = Spike.from_tile(col * TILE_SIZE, row * TILE_SIZE)
        level_data.tiles[row][col] = spike
        level_data.static_hash.insert(spike)

The Spike object is ``solid = False`` so the player is not pushed out of it —
the physics manager kills the player immediately instead.
"""

from __future__ import annotations

import pygame
from dataclasses import dataclass

from .GameObject import GameObject
from ..System.EntityType import EntityType
from ..Parameters.Map_parameters import TILE_SIZE, COLOR_SPIKE


# Base colours (match editor rendering in level_editor.py)
_COL_BASE:     tuple = (50,  50,  50)   # Dark grey rect
_COL_TRIANGLE: tuple = (120, 120, 120)  # Lighter grey triangle


@dataclass
class Spike:
    """
    Static spike hazard.

    Parameters
    ----------
    gObj : GameObject
        Position and size.  Typically a TILE_SIZE × TILE_SIZE square at
        the grid cell coordinates.
    solid : bool
        False — the player is NOT pushed out; instead they die instantly.
    """

    gObj:  GameObject
    solid: bool = False   # Deadly but not physically blocking

    def __post_init__(self):
        self.gObj.type_id = EntityType.SPIKE

    # ── Convenience properties (mirrors other entities) ───────────────────────

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

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def from_tile(
        cls,
        pixel_x: float,
        pixel_y: float,
        size: int = TILE_SIZE,
    ) -> "Spike":
        """
        Convenience constructor for placing a spike at a grid position.

        Parameters
        ----------
        pixel_x, pixel_y : float
            Top-left corner in world pixels (``col * TILE_SIZE``, ``row * TILE_SIZE``).
        size : int
            Width and height in pixels.  Defaults to ``TILE_SIZE`` (32).
        """
        gobj = GameObject(float(pixel_x), float(pixel_y), size, size, active=True)
        return cls(gObj=gobj)

    # ── Rendering ─────────────────────────────────────────────────────────────

    def render(
        self,
        surface: pygame.Surface,
        sx: float,
        sy: float,
        debug: bool = False,
    ) -> None:
        """
        Draw the spike at screen position (sx, sy).

        Layers
        ------
        1. Dark base rectangle (fills the tile).
        2. Lighter isoceles triangle pointing upward — the actual spike.
        3. Optional debug: red bounding box overlay.
        """
        w = self.gObj.width
        h = self.gObj.height

        #    Apex at horizontal centre, top edge.
        #    Base corners at bottom-left and bottom-right.
        apex  = (int(sx + w / 2), int(sy + 1))
        bl    = (int(sx + 1),     int(sy + h - 1))
        br    = (int(sx + w - 1), int(sy + h - 1))
        pygame.draw.polygon(surface, _COL_BASE, [apex, bl, br])

        # 3. Debug: red hitbox
        if debug:
            pygame.draw.rect(
                surface,
                (255, 64, 64),
                (int(sx), int(sy), w, h),
                1,
            )

    # ── Stub update (required by PhysicsManager.update_list interface) ────────

    def update(self, dt: float = 0.0, context=None) -> None:
        """Spikes are static — nothing to update."""
        pass

    def __repr__(self) -> str:
        return (
            f"<Spike x={self.gObj.x:.0f} y={self.gObj.y:.0f} "
            f"w={self.gObj.width} h={self.gObj.height}>"
        )