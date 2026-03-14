"""
Ring.py
-------
The iconic Sonic ring collectible.

Behaviour:
  - Collected on contact with player → +1 ring, score bonus
  - When Sonic is hit and has rings > 0: rings scatter outward as
    bouncing "lost rings" that can be re-collected briefly
  - Rings have a gentle bobbing animation
  - Lost rings expire after a few seconds
"""

from __future__ import annotations
from dataclasses import dataclass
import math
import pygame

# These imports assume the file lives in the Objects folder of the package.
# Adjust if placed elsewhere.
try:
    from .GameObject import GameObject
    from ..Parameters.Sonic_Map_parameters import (
        COLOR_RING, COLOR_RING_INNER, COLOR_BLACK, TILE_SIZE
    )
    from ..Parameters.Sonic_Movement_parameters import GRAVITY
except ImportError:
    # Fallback for standalone usage / testing
    from GameObject import GameObject
    TILE_SIZE = 32
    GRAVITY = 1100.0
    COLOR_RING = (255, 215, 0)
    COLOR_RING_INNER = (200, 170, 0)
    COLOR_BLACK = (0, 0, 0)


@dataclass
class Ring:
    """A collectible ring."""
    gObj: GameObject
    collected: bool = False
    animation_tick: int = 0

    # ── Lost ring fields ─────────────────────────────────────────────────
    is_lost: bool = False       # True = scattered ring after being hit
    vx: float = 0.0
    vy: float = 0.0
    lifetime: float = 0.0       # Seconds remaining before despawn
    collect_delay: float = 0.0  # Seconds before it can be picked up again
    can_collect: bool = True    # True if ready to be collected

    # Convenience properties matching other entity interfaces
    @property
    def x(self): return self.gObj.x
    @property
    def y(self): return self.gObj.y
    @property
    def width(self): return self.gObj.width
    @property
    def height(self): return self.gObj.height
    @property
    def active(self): return self.gObj.active

    # ── Factory for scattered rings ──────────────────────────────────────
    @classmethod
    def create_lost_ring(cls, x: float, y: float, vx: float, vy: float) -> "Ring":
        """Create a ring that was scattered when Sonic got hit."""
        gobj = GameObject(x, y, 16, 16, active=True)
        ring = cls(
            gObj=gobj,
            is_lost=True,
            vx=vx,
            vy=vy,
            lifetime=4.0,       # 4 seconds to re-collect
            collect_delay=0.5,  # 0.5s grace period before it can be grabbed
            can_collect=False,  
        )
        return ring

    # ── Update ───────────────────────────────────────────────────────────
    def update(self, dt: float = 0.016, context=None):
        self.animation_tick = (self.animation_tick + 1) % 120

        if self.is_lost:
            # Apply base physics (gravity and velocity)
            grav = context.GRAVITY if context else GRAVITY
            self.vy += grav * dt
            self.gObj.x += self.vx * dt
            self.gObj.y += self.vy * dt

            # NOTE: Bouncing off walls/floors, horizontal friction, and 
            # lifetime/delay countdowns are now explicitly handled by 
            # sonic_core.py in the step() loop so they interact perfectly 
            # with the level environment.

    # ── Render ───────────────────────────────────────────────────────────
    def render(self, surface: pygame.Surface, sx: float, sy: float, debug: bool = False):
        if not self.gObj.active or self.collected:
            return

        # Bobbing animation for static rings (lost rings don't bob, they bounce)
        bob = 0
        if not self.is_lost:
            bob = int(2 * math.sin(self.animation_tick * 0.1))

        cx = int(sx + self.gObj.width // 2)
        cy = int(sy + self.gObj.height // 2) + bob
        radius = 7

        # Ring shape: outer gold circle, inner darker circle, center hole
        pygame.draw.circle(surface, COLOR_RING, (cx, cy), radius)
        pygame.draw.circle(surface, COLOR_RING_INNER, (cx, cy), radius - 2)
        pygame.draw.circle(surface, COLOR_BLACK, (cx, cy), radius, 1)
        
        # Inner hole
        pygame.draw.circle(surface, (0, 100, 200), (cx, cy), 3)

        # Shine highlight
        pygame.draw.circle(surface, (255, 255, 220), (cx - 2, cy - 2), 2)

    def __repr__(self) -> str:
        state = "lost" if self.is_lost else ("collected" if self.collected else "active")
        return f"<Ring x={self.gObj.x:.0f} y={self.gObj.y:.0f} {state}>"