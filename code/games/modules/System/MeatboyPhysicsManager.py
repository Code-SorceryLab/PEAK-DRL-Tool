from __future__ import annotations
import math
from dataclasses import dataclass
import pygame
from ..Parameters.Map_parameters import TILE_GROUND, TILE_PLATFORM, TILE_QBLOCK, TILE_CRUMBLE

_SOLID = {TILE_GROUND, TILE_PLATFORM, TILE_QBLOCK, TILE_CRUMBLE}


@dataclass
class MeatboyContext:
    """Physics constants the manager applies. Per-actor tuning (speeds, jump)
    lives on the player, not here."""
    gravity: float = 1060.0
    fast_fall_grav: float = 1060.0
    max_fall_speed: float = 1400.0
    tile_size: int = 32


class MeatboyPhysicsManager:
    """Meat Boy's own physics manager (like SonicPhysicsManager / MegaPhysicsManager).

    Applies gravity, integrates position with continuous (sub-stepped) collision,
    resolves X then Y, and reports contact flags. It does NOT decide velocity
    (the player's control() does) and does NOT handle hazards/goal (the core does).

    Operates directly on the monolithic player: reads/writes player.vx, player.vy,
    player.on_ground, player.gravity_scale, player.contact_* and player.gObj.
    """

    def __init__(self, tile_size: int = 32):
        self.tile_size = tile_size

    def step(self, player, grid, ctx, dt: float) -> None:
        ts = self.tile_size
        gObj = player.gObj

        # 1. gravity (player.control resets gravity_scale before this each frame)
        grav = ctx.fast_fall_grav if player.vy > 0 else ctx.gravity
        player.vy = min(player.vy + grav * player.gravity_scale * dt, ctx.max_fall_speed)

        # 2. CCD sub-steps so fast movers can't tunnel
        dist = max(abs(player.vx), abs(player.vy)) * dt
        steps = 1
        if dist > ts * 0.5:
            steps = int(math.ceil(dist / (ts * 0.5)))
        sdt = dt / steps

        player.on_ground = False
        for _ in range(steps):
            gObj.x += player.vx * sdt
            self._resolve_x(player, gObj, grid)
            gObj.y += player.vy * sdt
            self._resolve_y(player, gObj, grid)

        # 3. contact flags via 1px probes (read by control() next frame)
        self._probe_contacts(player, gObj, grid)

    # --- helpers ---
    def _solid_rects_near(self, gObj, grid):
        ts = self.tile_size
        rect = gObj.get_rect()
        c0 = max(0, rect.left // ts - 1)
        c1 = rect.right // ts + 1
        r0 = max(0, rect.top // ts - 1)
        r1 = rect.bottom // ts + 1
        rects = []
        for row in range(r0, min(r1 + 1, len(grid))):
            line = grid[row]
            for col in range(c0, min(c1 + 1, len(line))):
                if line[col] in _SOLID:
                    rects.append(pygame.Rect(col * ts, row * ts, ts, ts))
        return rects

    def _resolve_x(self, player, gObj, grid):
        rect = gObj.get_rect()
        for tr in self._solid_rects_near(gObj, grid):
            if not rect.colliderect(tr):
                continue
            if player.vx > 0:
                gObj.x = tr.left - gObj.width
            elif player.vx < 0:
                gObj.x = tr.right
            else:
                if rect.centerx <= tr.centerx:
                    gObj.x = tr.left - gObj.width
                else:
                    gObj.x = tr.right
            player.vx = 0.0
            rect = gObj.get_rect()

    def _resolve_y(self, player, gObj, grid):
        rect = gObj.get_rect()
        for tr in self._solid_rects_near(gObj, grid):
            if not rect.colliderect(tr):
                continue
            if player.vy >= 0:
                gObj.y = tr.top - gObj.height
                player.on_ground = True
            else:
                gObj.y = tr.bottom
            player.vy = 0.0
            rect = gObj.get_rect()

    def _overlaps_solid(self, gObj, grid, dx, dy):
        probe = gObj.get_rect().move(dx, dy)
        for tr in self._solid_rects_near(gObj, grid):
            if probe.colliderect(tr):
                return True
        return False

    def _probe_contacts(self, player, gObj, grid):
        player.contact_left = self._overlaps_solid(gObj, grid, -1, 0)
        player.contact_right = self._overlaps_solid(gObj, grid, 1, 0)
        player.contact_ceiling = self._overlaps_solid(gObj, grid, 0, -1)
        if not player.on_ground:
            player.on_ground = self._overlaps_solid(gObj, grid, 0, 1)
