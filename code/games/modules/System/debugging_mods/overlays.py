import pygame
import traceback
from ...Parameters.Map_parameters import (TILE_SIZE, TILE_AIR, TILE_SPIKE, TILE_GOAL, COLOR_HITBOX)

class DebugOverlay:
    def render(self, surface: pygame.Surface, core):
        raise NotImplementedError

class HitboxOverlay(DebugOverlay):
    def render(self, surface: pygame.Surface, core):
        # 1. Player
        player = core.player
        if player:
            px, py, _ = core._world_to_screen(player.gObj)
            pygame.draw.rect(surface, COLOR_HITBOX, (px, py, player.gObj.width, player.gObj.height), 2)
        
        # 2. Dynamics (Hazards/Items)
        pm = core.physics_manager
        cx, cy, cw, ch = core.camera_x, core.camera_y, core.WIDTH, core.HEIGHT

        # Query hashes
        targets = pm.hazard_hash.query_rect(cx, cy, cw, ch) + \
                  pm.collectible_hash.query_rect(cx, cy, cw, ch)

        for entity in targets:
            gObj = entity.gObj if hasattr(entity, 'gObj') else entity
            sx, sy, _ = core._world_to_screen(gObj)
            color = (255, 255, 0) if hasattr(entity, 'kind') else (255, 255, 255)
            pygame.draw.rect(surface, color, (sx, sy, gObj.width, gObj.height), 1)

class GridOverlay(DebugOverlay):
    def render(self, surface: pygame.Surface, core):
        start_col = int(core.camera_x // TILE_SIZE)
        end_col = int((core.camera_x + core.WIDTH) // TILE_SIZE) + 1
        start_row = int(core.camera_y // TILE_SIZE)
        end_row = int((core.camera_y + core.HEIGHT) // TILE_SIZE) + 1

        for col in range(start_col, end_col):
            x = col * TILE_SIZE - core.camera_x
            pygame.draw.line(surface, (50, 50, 50), (x, 0), (x, core.HEIGHT))
        for row in range(start_row, end_row):
            y = row * TILE_SIZE - core.camera_y
            pygame.draw.line(surface, (50, 50, 50), (0, y), (core.WIDTH, y))

class AgentViewOverlay(DebugOverlay):
    def render(self, surface: pygame.Surface, core):
        cell_w, cell_h = 16, 16
        panel_x, panel_y = 10, 50
        
        # Draw BG
        pygame.draw.rect(surface, (20, 20, 30), (panel_x - 5, panel_y - 5, 11*cell_w + 10, 9*cell_h + 10))
        pygame.draw.rect(surface, (255, 255, 255), (panel_x - 5, panel_y - 5, 11*cell_w + 10, 9*cell_h + 10), 2)

        player = core.player
        if not player: return

        p_cx = int((player.gObj.x + player.gObj.width / 2) // TILE_SIZE)
        p_cy = int((player.gObj.y + player.gObj.height / 2) // TILE_SIZE)

        # Optimization: Pre-fetch sets
        enemy_locs = {(int(e.gObj.x//TILE_SIZE), int(e.gObj.y//TILE_SIZE)) for e in core.level_data.enemies if e.gObj.active}
        coin_locs = {(int(c.gObj.x//TILE_SIZE), int(c.gObj.y//TILE_SIZE)) for c in core.level_data.coins if c.gObj.active and not getattr(c, 'collected', False)}

        for dy_i, dy in enumerate(range(-4, 5)):
            for dx_i, dx in enumerate(range(-5, 6)):
                tx, ty = p_cx + dx, p_cy + dy
                draw_x, draw_y = panel_x + dx_i * cell_w, panel_y + dy_i * cell_h
                color = (50, 50, 50) # Air

                # World
                if 0 <= ty < core.level_data.rows and 0 <= tx < core.level_data.cols:
                    tile = core.level_data.grid[ty][tx]
                    if tile != TILE_AIR:
                        color = (180, 180, 180)
                        if tile == TILE_SPIKE: color = (255, 0, 255)
                        elif tile == TILE_GOAL: color = (0, 255, 0)
                else: color = (0, 0, 0)

                # Entities
                if (tx, ty) in enemy_locs: color = (255, 50, 50)
                elif (tx, ty) in coin_locs: color = (255, 215, 0)
                if dx == 0 and dy == 0: color = (50, 150, 255) # Player

                pygame.draw.rect(surface, color, (draw_x, draw_y, cell_w - 1, cell_h - 1))

        if hasattr(core, "ui_font"):
            surface.blit(core.ui_font.render("Agent View", True, (255, 255, 255)), (panel_x, panel_y - 25))

class InfoPanelOverlay(DebugOverlay):
    def render(self, surface: pygame.Surface, core):
        if not core.player: return
        p = core.player
        lines = [
            f"Pos: {int(p.gObj.x)}, {int(p.gObj.y)}",
            f"Vel: {p.vx:.1f}, {p.vy:.1f}",
            f"Stall: {core.stall_timer:.2f}s (x{core.stall_windows_count})",
            f"Best X: {int(core.progress_x_best)}"
        ]
        y_start = 220 if core.debug_manager.show_agent_view else 50
        for i, ln in enumerate(lines):
            surface.blit(core.ui_font.render(ln, True, (200, 200, 200)), (10, y_start + i * 20))

class ObsValuesOverlay(DebugOverlay):
    def render(self, surface: pygame.Surface, core):
        # 1. Safe Fetch
        try:
            p_vals = core._player_obs()
            t_vals = core._tracking_obs()
        except Exception as e:
            # FIX: Print error once so we know why the panel is missing!
            print(f"[DEBUG ERROR] ObsValuesOverlay crashed: {e}")
            traceback.print_exc()
            return

        # 2. Labels
        p_labels = ["Px (n)", "Py (n)", "Vx (n)", "Vy (n)", "Grounded"]
        t_labels = ["Enm Dist", "Coin Dist", "Goal Dist", "Act Enms", "Act Coins", "Score", "Time", "Lives"]
        
        data = list(zip(p_labels, p_vals)) + list(zip(t_labels, t_vals))

        # 3. Draw
        panel_x, panel_y = 220, 50
        line_h = 16
        w, h = 180, len(data) * line_h + 10
        
        bg = pygame.Surface((w, h))
        bg.set_alpha(200); bg.fill((10, 10, 10))
        surface.blit(bg, (panel_x, panel_y))
        pygame.draw.rect(surface, (100, 100, 100), (panel_x, panel_y, w, h), 1)

        # FIX: Use cached font from DebugManager
        font = core.debug_manager.small_font
        
        for i, (label, val) in enumerate(data):
            color = (150, 150, 150)
            if val > 0: color = (255, 255, 255)
            if "Dist" in label and val < 0.1: color = (255, 100, 100) # Alert red for close objects
            
            txt = font.render(f"{label}: {val:.3f}", True, color)
            surface.blit(txt, (panel_x + 5, panel_y + 5 + i * line_h))

        head = core.ui_font.render("Vector OBS", True, (0, 255, 255))
        surface.blit(head, (panel_x, panel_y - 20))