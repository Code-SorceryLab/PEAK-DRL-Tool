import pygame
from ..Map_parameters import (TILE_SIZE, TILE_AIR, TILE_SPIKE, TILE_GOAL,COLOR_HITBOX, COLOR_STREAK)
# Assuming these imports exist based on your provided code structure
# If not, ensure TILE_SIZE etc are available.

class DebugOverlay:
    def render(self, surface: pygame.Surface, core):
        raise NotImplementedError

class HitboxOverlay(DebugOverlay):
    def render(self, surface: pygame.Surface, core):
        # Draw Player Hitbox
        player = core.player
        player_x, player_y, _ = core._world_to_screen(player.gObj)
        pygame.draw.rect(surface, COLOR_HITBOX, (player_x, player_y, player.gObj.width, player.gObj.height), 2)
        
        # Draw Entity Hitboxes (only visible ones)
        visible = core.dynamic_hash.query_rect(core.camera_x, core.camera_y, core.WIDTH, core.HEIGHT)
        for object in visible:
            screen_x, screen_y, _ = core._world_to_screen(object.gObj)
            pygame.draw.rect(surface, (255, 255, 255), (screen_x, screen_y, object.gObj.width, object.gObj.height), 1)

class GridOverlay(DebugOverlay):
    def render(self, surface: pygame.Surface, core):
        # Draw Tile Grid
        # Calculate start/end col/row based on camera
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
        # Settings for the mini-window
        cell_w, cell_h = 16, 16
        grid_w, grid_h = 11, 9
        panel_x, panel_y = 10, 50
        
        # Draw Background
        pygame.draw.rect(surface, (20, 20, 30), 
                        (panel_x - 5, panel_y - 5, grid_w * cell_w + 10, grid_h * cell_h + 10))
        pygame.draw.rect(surface, (255, 255, 255), 
                        (panel_x - 5, panel_y - 5, grid_w * cell_w + 10, grid_h * cell_h + 10), 2)

        player = core.player
        p_cx = int((player.gObj.x + player.gObj.width / 2) // TILE_SIZE)
        p_cy = int((player.gObj.y + player.gObj.height / 2) // TILE_SIZE)

        enemy_locs = {(int(e.gObj.x // TILE_SIZE), int(e.gObj.y // TILE_SIZE)) 
                      for e in core.enemies if e.gObj.active}
        coin_locs = {(int(c.gObj.x // TILE_SIZE), int(c.gObj.y // TILE_SIZE)) 
                     for c in core.coins if c.gObj.active and not c.collected}

        for dy_i, dy in enumerate(range(-4, 5)):
            for dx_i, dx in enumerate(range(-5, 6)):
                tx, ty = p_cx + dx, p_cy + dy
                draw_x = panel_x + dx_i * cell_w
                draw_y = panel_y + dy_i * cell_h
                
                color = (50, 50, 50) # Air

                # 1. Level Geometry
                if 0 <= ty < core.level_rows and 0 <= tx < core.level_cols:
                    tile_type = core.level_data[ty][tx]
                    if tile_type != TILE_AIR:
                        color = (180, 180, 180)
                        if tile_type == TILE_SPIKE: color = (255, 0, 255)
                        elif tile_type == TILE_GOAL: color = (0, 255, 0)
                else:
                    color = (0, 0, 0) # OOB

                # 2. Entities
                if (tx, ty) in enemy_locs: color = (255, 50, 50)
                elif (tx, ty) in coin_locs: color = (255, 215, 0)
                
                # 3. Player
                if dx == 0 and dy == 0: color = (50, 150, 255)

                pygame.draw.rect(surface, color, (draw_x, draw_y, cell_w - 1, cell_h - 1))

        if hasattr(core, "ui_font"):
            surface.blit(core.ui_font.render("Agent View", True, (255, 255, 255)), (panel_x, panel_y - 25))

class InfoPanelOverlay(DebugOverlay):
    def render(self, surface: pygame.Surface, core):
        if not hasattr(core, "ui_font"): return

        # Recalculate obs only if needed, or grab last obs? 
        # For debug, recalculating is fine.
        # Note: We aren't displaying the full vector, just specific stats.
        
        player = core.player
        lines = [
            f"Pos: {int(player.gObj.x)}, {int(player.gObj.y)}",
            f"Vel: {player.vx:.1f}, {player.vy:.1f}",
            f"Stall Timer: {core.stall_timer:.2f}s",
            f"Stall Count: {core.stall_windows_count}",
            f"Best X: {int(core.progress_x_best)}"
        ]
        
        # Offset if Agent View is also on
        panel_y_start = 220 if core.debug_manager.show_agent_view else 50
        
        for i, ln in enumerate(lines):
            surface.blit(core.ui_font.render(ln, True, (200, 200, 200)), (10, panel_y_start + i * 20))
            

class ObsValuesOverlay(DebugOverlay):
    def render(self, surface: pygame.Surface, core):
        # 1. Fetch specific semantic vectors (skipping tile window)
        # We access protected methods because this is a debug tool coupled to the core
        p_vals = core._player_obs()
        o_vals = core._object_obs()
        
        # 2. Define labels matching platformer_core.py logic
        p_labels = [
            "Plyr X (norm)", "Plyr Y (norm)", 
            "Vel X (norm)", "Vel Y (norm)", 
            "On Ground"
        ]
        o_labels = [
            "Enm Dist (rel)", "Coin Dist (rel)", 
            "Goal Dist (sgn)", 
            "Powered Up", "Invinc Tmr", 
            "Act Enemies", "Act Coins", 
            "Act P-Ups", "Tot Coins", 
            "Score (scl)", "Frame (scl)"
        ]
        
        # Combine
        data = list(zip(p_labels, p_vals)) + list(zip(o_labels, o_vals))

        # 3. Draw Panel
        # Position to the right of Agent View (which is ~190px wide)
        panel_x = 220 
        panel_y = 50
        line_h = 16
        w, h = 200, len(data) * line_h + 10
        
        # Background
        bg = pygame.Surface((w, h))
        bg.set_alpha(200)
        bg.fill((10, 10, 10))
        surface.blit(bg, (panel_x, panel_y))
        pygame.draw.rect(surface, (100, 100, 100), (panel_x, panel_y, w, h), 1)

        # Text
        font = getattr(core, "small_font", None) or pygame.font.SysFont("arial", 12)
        
        for i, (label, val) in enumerate(data):
            # Color code: Zero/Low = Gray, High/Active = White/Green
            color = (150, 150, 150)
            if val > 0: color = (255, 255, 255)
            if "Dist" in label and val < 0.1: color = (255, 100, 100) # Close danger/coin
            
            txt_str = f"{label}: {val:.3f}"
            txt = font.render(txt_str, True, color)
            surface.blit(txt, (panel_x + 8, panel_y + 5 + i * line_h))

        # Header
        head_font = getattr(core, "ui_font", None) or pygame.font.SysFont("arial", 14, bold=True)
        surface.blit(head_font.render("Semantic OBS", True, (0, 255, 255)), (panel_x, panel_y - 20))
