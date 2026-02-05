import pygame
from ...Parameters.Map_parameters import (TILE_SIZE, TILE_AIR, TILE_SPIKE, TILE_GOAL, COLOR_HITBOX, COLOR_STREAK)

class DebugOverlay:
    def render(self, surface: pygame.Surface, core):
        raise NotImplementedError

class HitboxOverlay(DebugOverlay):
    def render(self, surface: pygame.Surface, core):
        # 1. Draw Player Hitbox
        player = core.player
        if player:
            player_x, player_y, _ = core._world_to_screen(player.gObj)
            pygame.draw.rect(surface, COLOR_HITBOX, (player_x, player_y, player.gObj.width, player.gObj.height), 2)
        
        # 2. Draw Dynamic Entity Hitboxes
        # Refactored: Access hashes via PhysicsManager, not Core
        pm = core.physics_manager
        cx, cy = core.camera_x, core.camera_y
        cw, ch = core.WIDTH, core.HEIGHT

        # Query both hashes (Hazards and Collectibles)
        visible_hazards = pm.hazard_hash.query_rect(cx, cy, cw, ch)
        visible_items = pm.collectible_hash.query_rect(cx, cy, cw, ch)
        
        # Combine lists for drawing
        all_visible = visible_hazards + visible_items

        for entity in all_visible:
            # entities in hash might be wrappers or GameObjects. 
            # The SpatialHash.insert handles this, but we need to be sure we get the gObj for rect.
            gObj = entity.gObj if hasattr(entity, 'gObj') else entity
            
            screen_x, screen_y, _ = core._world_to_screen(gObj)
            
            # Color differentiation
            color = (255, 255, 255) # Default White
            if hasattr(entity, 'kind'): color = (255, 255, 0) # Items Yellow
            
            pygame.draw.rect(surface, color, (screen_x, screen_y, gObj.width, gObj.height), 1)

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
        grid_w, grid_h = 11, 9
        panel_x, panel_y = 10, 50
        
        # Background
        pygame.draw.rect(surface, (20, 20, 30), 
                        (panel_x - 5, panel_y - 5, grid_w * cell_w + 10, grid_h * cell_h + 10))
        pygame.draw.rect(surface, (255, 255, 255), 
                        (panel_x - 5, panel_y - 5, grid_w * cell_w + 10, grid_h * cell_h + 10), 2)

        player = core.player
        if not player: return

        p_cx = int((player.gObj.x + player.gObj.width / 2) // TILE_SIZE)
        p_cy = int((player.gObj.y + player.gObj.height / 2) // TILE_SIZE)

        # Refactored: Access lists via LevelData
        enemy_locs = {(int(e.gObj.x // TILE_SIZE), int(e.gObj.y // TILE_SIZE)) 
                      for e in core.level_data.enemies if e.gObj.active}
        
        coin_locs = {(int(c.gObj.x // TILE_SIZE), int(c.gObj.y // TILE_SIZE)) 
                     for c in core.level_data.coins if c.gObj.active and not getattr(c, 'collected', False)}

        for dy_i, dy in enumerate(range(-4, 5)):
            for dx_i, dx in enumerate(range(-5, 6)):
                tx, ty = p_cx + dx, p_cy + dy
                draw_x = panel_x + dx_i * cell_w
                draw_y = panel_y + dy_i * cell_h
                
                color = (50, 50, 50) # Air

                # 1. Level Geometry (Refactored: use core.level_data.rows/cols/grid)
                if 0 <= ty < core.level_data.rows and 0 <= tx < core.level_data.cols:
                    # Fix: Access .grid, not the object itself
                    tile_type = core.level_data.grid[ty][tx]
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

        player = core.player
        if not player: return

        lines = [
            f"Pos: {int(player.gObj.x)}, {int(player.gObj.y)}",
            f"Vel: {player.vx:.1f}, {player.vy:.1f}",
            f"Stall Timer: {core.stall_timer:.2f}s",
            f"Stall Count: {core.stall_windows_count}",
            f"Best X: {int(core.progress_x_best)}"
        ]
        
        panel_y_start = 220 if core.debug_manager.show_agent_view else 50
        
        for i, ln in enumerate(lines):
            surface.blit(core.ui_font.render(ln, True, (200, 200, 200)), (10, panel_y_start + i * 20))

class ObsValuesOverlay(DebugOverlay):
    def render(self, surface: pygame.Surface, core):
        # 1. Fetch values (Refactored: use _tracking_obs instead of _object_obs)
        # We also grab player obs for completeness
        try:
            p_vals = core._player_obs()
            # This matches the 8 values in core._tracking_obs()
            t_vals = core._tracking_obs()
        except AttributeError:
            return 

        # 2. Define labels matching platformer_core.py logic
        p_labels = [
            "Plyr X (norm)", "Plyr Y (norm)", 
            "Vel X (norm)", "Vel Y (norm)", 
            "On Ground"
        ]
        
        # UPDATED LABELS to match _tracking_obs (8 items)
        t_labels = [
            "Enm Dist (rel)",  # 1
            "Coin Dist (rel)", # 2
            "Goal Dist",       # 3
            "Act Enemies",     # 4
            "Act Coins",       # 5
            "Score (scl)",     # 6
            "Time (scl)",      # 7
            "Lives (scl)"      # 8
        ]
        
        # Combine
        data = list(zip(p_labels, p_vals)) + list(zip(t_labels, t_vals))

        # 3. Draw Panel
        panel_x = 220 
        panel_y = 50
        line_h = 16
        w, h = 200, len(data) * line_h + 10
        
        bg = pygame.Surface((w, h))
        bg.set_alpha(200)
        bg.fill((10, 10, 10))
        surface.blit(bg, (panel_x, panel_y))
        pygame.draw.rect(surface, (100, 100, 100), (panel_x, panel_y, w, h), 1)

        font = getattr(core, "small_font", None) or pygame.font.SysFont("arial", 12)
        
        for i, (label, val) in enumerate(data):
            color = (150, 150, 150)
            if val > 0: color = (255, 255, 255)
            if "Dist" in label and val < 0.1: color = (255, 100, 100) 
            
            txt_str = f"{label}: {val:.3f}"
            txt = font.render(txt_str, True, color)
            surface.blit(txt, (panel_x + 8, panel_y + 5 + i * line_h))

        head_font = getattr(core, "ui_font", None) or pygame.font.SysFont("arial", 14, bold=True)
        surface.blit(head_font.render("Semantic OBS", True, (0, 255, 255)), (panel_x, panel_y - 20))