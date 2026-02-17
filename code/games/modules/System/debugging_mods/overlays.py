import pygame
import traceback
from ...Parameters.Map_parameters import (TILE_SIZE, TILE_AIR, TILE_SPIKE, TILE_GOAL, COLOR_HITBOX)
from ...Objects.Goal import Goal

class DebugOverlay:
    def render(self, surface: pygame.Surface, core):
        raise NotImplementedError

class HitboxOverlay(DebugOverlay):
    def render(self, surface: pygame.Surface, core):
        # 1. Player Hitbox (The "Selection Box" Style)
        player = core.player
        if player:
            px, py, _ = core._world_to_screen(player.gObj)
            # FIX: Inflate by 4px so it draws AROUND the sprite, not inside it
            # This makes it pop visually as a distinct collider box
            rect = pygame.Rect(px, py, player.gObj.width, player.gObj.height)
            inflated = rect.inflate(4, 4) 
            pygame.draw.rect(surface, (0, 255, 0), inflated, 2) # Bright Green
        
        # 2. Dynamics (Hazards/Items)
        pm = core.physics_manager
        cx, cy, cw, ch = core.camera_x, core.camera_y, core.WIDTH, core.HEIGHT

        # Query hashes
        targets = pm.hazard_hash.query_rect(cx, cy, cw, ch) + \
                  pm.collectible_hash.query_rect(cx, cy, cw, ch)

        for entity in targets:
            gObj = entity.gObj if hasattr(entity, 'gObj') else entity
            sx, sy, _ = core._world_to_screen(gObj)
            
            # Color coding: Yellow for items, Red for hazards
            color = (255, 255, 0) if hasattr(entity, 'kind') else (255, 50, 50)
            
            # Draw standard hitbox
            pygame.draw.rect(surface, color, (sx, sy, gObj.width, gObj.height), 1)

class GridOverlay(DebugOverlay):
    def render(self, surface: pygame.Surface, core):
        start_col = int(core.camera_x // TILE_SIZE)
        end_col = int((core.camera_x + core.WIDTH) // TILE_SIZE) + 1
        start_row = int(core.camera_y // TILE_SIZE)
        end_row = int((core.camera_y + core.HEIGHT) // TILE_SIZE) + 1

        # Draw simpler, cleaner grid lines with alpha
        grid_surf = pygame.Surface((core.WIDTH, core.HEIGHT), pygame.SRCALPHA)
        
        for col in range(start_col, end_col):
            x = col * TILE_SIZE - core.camera_x
            pygame.draw.line(grid_surf, (255, 255, 255, 50), (x, 0), (x, core.HEIGHT))
        for row in range(start_row, end_row):
            y = row * TILE_SIZE - core.camera_y
            pygame.draw.line(grid_surf, (255, 255, 255, 50), (0, y), (core.WIDTH, y))
            
        surface.blit(grid_surf, (0,0))

class AgentViewOverlay(DebugOverlay):
    def render(self, surface: pygame.Surface, core):
        # Configuration matches platformer_core.obs_width/height (11x11)
        grid_cols = getattr(core, 'obs_width', 21)
        grid_rows = getattr(core, 'obs_height', 21)
        cell_w, cell_h = 24, 24 # Slightly larger for text visibility
        
        # Dimensions
        content_w = grid_cols * cell_w
        content_h = grid_rows * cell_h
        padding = 5
        panel_w = content_w + (padding * 2)
        panel_h = content_h + (padding * 2) + 20 # +20 for Header
        
        panel_x, panel_y = 10, 50
        
        # 1. Background Panel (Dark Grey Container)
        bg = pygame.Surface((panel_w, panel_h))
        bg.fill((20, 20, 25))
        bg.set_alpha(240)
        surface.blit(bg, (panel_x, panel_y))
        
        # Border
        pygame.draw.rect(surface, (100, 100, 100), (panel_x, panel_y, panel_w, panel_h), 1)
        
        # 2. Header
        header_font = core.debug_manager.font
        header = header_font.render("Agent Vision (Dijkstra)", True, (200, 200, 200))
        surface.blit(header, (panel_x + padding, panel_y + 2))
        
        # Separator Line
        line_y = panel_y + 22
        pygame.draw.line(surface, (80, 80, 80), (panel_x, line_y), (panel_x + panel_w, line_y))

        # 3. The Grid Content
        start_x = panel_x + padding
        start_y = line_y + padding

        player = core.player
        if not player: return

        # Calculate grid origin (top-left of observation window)
        p_cx = int(player.gObj.x // TILE_SIZE)
        p_cy = int(player.gObj.y // TILE_SIZE)
        
        pad_x = core.obs_pad_x
        pad_y = core.obs_pad_y
        
        origin_x = p_cx - pad_x
        origin_y = p_cy - pad_y

        # Optimization: Pre-fetch sets for entity lookup
        enemy_locs = {(int(e.gObj.x//TILE_SIZE), int(e.gObj.y//TILE_SIZE)) for e in core.level_data.enemies if e.gObj.active}
        coin_locs = {(int(c.gObj.x//TILE_SIZE), int(c.gObj.y//TILE_SIZE)) for c in core.level_data.coins if c.gObj.active and not getattr(c, 'collected', False)}
        goal_locs = {(int(g.gObj.x//TILE_SIZE), int(g.gObj.y//TILE_SIZE)) for g in core.level_data.goals if g.gObj.active}

        # Font for cost numbers
        cost_font = core.debug_manager.small_font

        for r in range(grid_rows):
            for c in range(grid_cols):
                # World Coordinates
                tx = origin_x + c
                ty = origin_y + r
                
                draw_x = start_x + c * cell_w
                draw_y = start_y + r * cell_h
                
                # Default: Empty Space (Black/Dark Grey)
                color = (10, 10, 10) 
                border_color = (40, 40, 40)

                # World Geometry (Solid/Hazard)
                if 0 <= ty < core.level_data.rows and 0 <= tx < core.level_data.cols:
                    tile = core.level_data.grid[ty][tx]
                    if tile != TILE_AIR:
                        color = (100, 100, 100) # Walls are Grey
                        if tile == TILE_SPIKE: color = (150, 0, 0) # Spikes Red
                        elif tile == TILE_GOAL: color = (0, 150, 0) # Goal Green
                else:
                    # Out of bounds
                    color = (5, 5, 5)
                
                # Entities Overlay
                if (tx, ty) in enemy_locs: 
                    color = (255, 50, 50)   # Enemy Red
                    border_color = (255, 100, 100)
                elif (tx, ty) in coin_locs: 
                    color = (255, 215, 0)   # Coin Gold
                    border_color = (255, 255, 200)
                elif (tx, ty) in goal_locs:
                    color = (0, 200, 0)     # Goal Green
                    border_color = (100, 255, 100)
                
                # Player (Center relative to grid)
                if tx == p_cx and ty == p_cy: 
                    color = (0, 150, 255) # Player Blue
                    border_color = (100, 200, 255)

                # Draw Cell
                pygame.draw.rect(surface, color, (draw_x, draw_y, cell_w - 1, cell_h - 1))
                # Draw Subtle Border
                pygame.draw.rect(surface, border_color, (draw_x, draw_y, cell_w - 1, cell_h - 1), 1)
                
                # --- DRAW DIJKSTRA COST ---
                if core.dijkstra:
                    cost = core.dijkstra.get_dist(tx, ty)
                    if cost >= 0 and cost != float('inf'):
                        # Render number
                        val_str = f"{int(cost)}"
                        # Scale color based on cost? (Lower = Green, Higher = Red)
                        # Just white for visibility
                        txt = cost_font.render(val_str, True, (255, 255, 255))
                        # Center text
                        tx_x = draw_x + (cell_w - txt.get_width()) // 2
                        tx_y = draw_y + (cell_h - txt.get_height()) // 2
                        surface.blit(txt, (tx_x, tx_y))


class InfoPanelOverlay(DebugOverlay):
    def render(self, surface: pygame.Surface, core):
        if not core.player: return
        
        # Layout relative to Agent View
        # Agent View Height approx: 20 (Header) + 11*24 (Grid) + 10 (Pad) = ~300
        # We start just below it.
        panel_x = 10
        panel_y = 350 
        panel_w = 190 
        
        p = core.player
        lines = [
            f"Pos:  {int(p.gObj.x):>4}, {int(p.gObj.y):>4}",
            f"Vel:  {p.vx:>5.1f}, {p.vy:>5.1f}",
            f"Stall: {core.stall_timer:>4.2f}s (x{core.stall_windows_count})",
            f"Best X:{int(core.progress_x_best):>5}"
        ]
        
        # 1. Background for Text (The "Little Better" Fix)
        # Makes text readable against any background
        line_h = 20
        bg_h = len(lines) * line_h + 10
        
        bg = pygame.Surface((panel_w, bg_h))
        bg.fill((0, 0, 0))
        bg.set_alpha(180) # Semi-transparent black
        surface.blit(bg, (panel_x, panel_y))
        
        # Border
        pygame.draw.rect(surface, (100, 100, 100), (panel_x, panel_y, panel_w, bg_h), 1)

        # 2. Render Text
        font = core.ui_font # Using the bold font
        for i, ln in enumerate(lines):
            # Render White text
            txt = font.render(ln, True, (230, 230, 230))
            surface.blit(txt, (panel_x + 10, panel_y + 5 + i * line_h))

class ObsValuesOverlay(DebugOverlay):
    def render(self, surface: pygame.Surface, core):
        # Copy the code from the previous "UI Fix" response here
        try:
            p_vals = core._player_obs()
            t_vals = core._tracking_obs()
        except Exception as e:
            # print(f"[DEBUG ERROR] ObsValuesOverlay crashed: {e}") 
            # traceback.print_exc()
            return

        p_labels = ["Px (n)", "Py (n)", "Vx (n)", "Vy (n)", "Grounded"]
        t_labels = ["Enm Dist", "Coin Dist", "Goal Dist", "Act Enms", "Act Coins", "Score", "Time", "Lives", "Dir X", "Goal Y", "Dijkstra"]
        
        # Pad t_labels if necessary to match t_vals length from core
        if len(t_vals) > len(t_labels):
             t_labels.extend([f"Val {i}" for i in range(len(t_labels), len(t_vals))])

        data = list(zip(p_labels, p_vals)) + list(zip(t_labels, t_vals))

        panel_w = 180 
        line_h = 16
        panel_h = len(data) * line_h + 25
        panel_x = core.WIDTH - panel_w - 10 
        panel_y = 200 
        
        bg = pygame.Surface((panel_w, panel_h))
        bg.set_alpha(220); bg.fill((20, 20, 20)) 
        surface.blit(bg, (panel_x, panel_y))
        pygame.draw.rect(surface, (100, 100, 100), (panel_x, panel_y, panel_w, panel_h), 1)

        head_font = core.debug_manager.font 
        head = head_font.render("Vector Values", True, (220, 220, 220))
        surface.blit(head, (panel_x + 5, panel_y + 5))
        pygame.draw.line(surface, (100, 100, 100), (panel_x, panel_y + 22), (panel_x + panel_w, panel_y + 22))

        font = core.debug_manager.small_font
        content_y = panel_y + 25
        
        for i, (label, val) in enumerate(data):
            color = (150, 150, 150) 
            if val > 0: color = (255, 255, 255)
            if "Dist" in label and val < 0.15: color = (255, 100, 100) 
            if "Grounded" in label and val > 0.5: color = (100, 255, 100) 
            
            txt_label = font.render(f"{label}:", True, (120, 120, 120))
            surface.blit(txt_label, (panel_x + 5, content_y + i * line_h))
            
            txt_val = font.render(f"{val:.3f}", True, color)
            val_x = panel_x + panel_w - txt_val.get_width() - 5
            surface.blit(txt_val, (val_x, content_y + i * line_h))