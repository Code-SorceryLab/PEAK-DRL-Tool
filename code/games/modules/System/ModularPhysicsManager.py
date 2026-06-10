from __future__ import annotations
import math
import pygame
from ..Parameters.Map_parameters import TILE_GROUND, TILE_PLATFORM, TILE_QBLOCK, TILE_CRUMBLE

_SOLID = {TILE_GROUND, TILE_PLATFORM, TILE_QBLOCK, TILE_CRUMBLE}


class ModularPhysicsManager:
    """Integrates an actor's position and resolves it against the solid grid.

    Responsibility (same boundary as the legacy PhysicsManager): apply gravity,
    move position with continuous (sub-stepped) collision, resolve X then Y,
    and report contact flags. It does NOT decide velocity (abilities do) and
    does NOT handle hazards/goal (the core does)."""

    def __init__(self, tile_size: int = 32):
        self.tile_size = tile_size

    def step(self, state, gObj, grid, ctx, dt: float) -> None:
        ts = self.tile_size
        # 1. gravity (host resets gravity_scale before abilities each frame)
        grav = ctx.fast_fall_grav if state.vy > 0 else ctx.gravity
        state.vy = min(state.vy + grav * state.gravity_scale * dt, ctx.max_fall_speed)

        # 2. CCD sub-steps so fast movers can't tunnel
        dist = max(abs(state.vx), abs(state.vy)) * dt
        steps = 1
        if dist > ts * 0.5:
            steps = int(math.ceil(dist / (ts * 0.5)))
        sdt = dt / steps

        state.on_ground = False
        for _ in range(steps):
            gObj.x += state.vx * sdt
            self._resolve_x(state, gObj, grid)
            gObj.y += state.vy * sdt
            self._resolve_y(state, gObj, grid)

        # 3. contact flags via 1px probes (read by abilities next frame)
        self._probe_contacts(state, gObj, grid)

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

    def _resolve_x(self, state, gObj, grid):
        rect = gObj.get_rect()
        for tr in self._solid_rects_near(gObj, grid):
            if not rect.colliderect(tr):
                continue
            if state.vx > 0:
                gObj.x = tr.left - gObj.width
            elif state.vx < 0:
                gObj.x = tr.right
            else:
                if rect.centerx <= tr.centerx:
                    gObj.x = tr.left - gObj.width
                else:
                    gObj.x = tr.right
            state.vx = 0.0
            rect = gObj.get_rect()

    def _resolve_y(self, state, gObj, grid):
        rect = gObj.get_rect()
        for tr in self._solid_rects_near(gObj, grid):
            if not rect.colliderect(tr):
                continue
            if state.vy >= 0:
                gObj.y = tr.top - gObj.height
                state.on_ground = True
            else:
                gObj.y = tr.bottom
            state.vy = 0.0
            rect = gObj.get_rect()

    def _overlaps_solid(self, gObj, grid, dx, dy):
        probe = gObj.get_rect().move(dx, dy)
        for tr in self._solid_rects_near(gObj, grid):
            if probe.colliderect(tr):
                return True
        return False

    def _probe_contacts(self, state, gObj, grid):
        state.contact_left = self._overlaps_solid(gObj, grid, -1, 0)
        state.contact_right = self._overlaps_solid(gObj, grid, 1, 0)
        state.contact_ceiling = self._overlaps_solid(gObj, grid, 0, -1)
        if not state.on_ground:
            state.on_ground = self._overlaps_solid(gObj, grid, 0, 1)
