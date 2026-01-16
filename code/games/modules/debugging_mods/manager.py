import pygame
from .overlays import HitboxOverlay, GridOverlay, AgentViewOverlay, InfoPanelOverlay

class DebugManager:
    def __init__(self, default_active=True):
        self.active = True
        
        # Toggles
        self.show_hitboxes = default_active
        self.show_agent_view = default_active
        self.show_sensors = default_active
        self.show_obs_panel = default_active
        self.show_grid = False
        
        # Free Cam Tools
        self.free_cam_active = False # Default to locked (following player)
        self.cam_move_speed = 600.0  # Pixels per second
        self.current_cam_move = [0.0, 0.0]

        # Input tracking
        self._prev_keys = pygame.key.get_pressed()
        self.font = pygame.font.SysFont("arial", 16, bold=True)

        # Composited Visualizers
        self.hitbox_overlay = HitboxOverlay()
        self.grid_overlay = GridOverlay()
        self.agent_view_overlay = AgentViewOverlay()
        self.info_overlay = InfoPanelOverlay()

        self._print_help_to_terminal()

    def _print_help_to_terminal(self):
        print("\n" + "="*40)
        print("   PEAK ENGINE - DEBUG TOOLS ENABLED")
        print("="*40)
        print(" [1] / [F1] : Toggle Hitboxes")
        print(" [2] / [F2] : Toggle Agent View (Matrix)")
        print(" [3] / [F3] : Toggle Sensor Rays")
        print(" [4] / [F4] : Toggle Info Panel")
        print(" [5] / [F5] : Toggle Free Camera Mode")
        print("              (Use I-J-K-L to Pan)")
        print(" [6] / [F6] : Toggle Tile Grid")
        print("="*40 + "\n")

    def update_input(self):
        """
        Polls keys and updates toggle states. 
        Must be called once per frame.
        """
        # CRITICAL: Pump events to ensure key states are fresh
        pygame.event.pump()
        
        keys = pygame.key.get_pressed()
        
        # Helper for rising-edge detection
        def check_toggle(k1, k2):
            return (keys[k1] and not self._prev_keys[k1]) or \
                   (keys[k2] and not self._prev_keys[k2])

        if check_toggle(pygame.K_1, pygame.K_F1): 
            self.show_hitboxes = not self.show_hitboxes
            print(f"[Debug] Hitboxes: {self.show_hitboxes}")
            
        if check_toggle(pygame.K_2, pygame.K_F2): 
            self.show_agent_view = not self.show_agent_view
            print(f"[Debug] Agent View: {self.show_agent_view}")

        if check_toggle(pygame.K_3, pygame.K_F3): 
            self.show_sensors = not self.show_sensors
            print(f"[Debug] Sensors: {self.show_sensors}")

        if check_toggle(pygame.K_4, pygame.K_F4): 
            self.show_obs_panel = not self.show_obs_panel
            print(f"[Debug] Info Panel: {self.show_obs_panel}")

        if check_toggle(pygame.K_5, pygame.K_F5): 
            self.free_cam_active = not self.free_cam_active
            mode = "FREE CAM" if self.free_cam_active else "PLAYER LOCKED"
            print(f"[Debug] Camera Mode: {mode}")

        if check_toggle(pygame.K_6, pygame.K_F6): 
            self.show_grid = not self.show_grid
            print(f"[Debug] Grid: {self.show_grid}")

        # --- Free Cam Movement ---
        # Changed to I-J-K-L to avoid ANY conflict with WASD or Arrows
        self.current_cam_move = [0.0, 0.0]
        if self.free_cam_active:
            if keys[pygame.K_j]:  self.current_cam_move[0] = -self.cam_move_speed
            if keys[pygame.K_l]:  self.current_cam_move[0] =  self.cam_move_speed
            if keys[pygame.K_i]:  self.current_cam_move[1] = -self.cam_move_speed
            if keys[pygame.K_k]:  self.current_cam_move[1] =  self.cam_move_speed

        self._prev_keys = keys

    def render_overlays(self, surface: pygame.Surface, core):
        """
        Draws active overlays onto the surface.
        Requires the core instance to access game data.
        """
        if self.show_grid:
            self.grid_overlay.render(surface, core)
            
        if self.show_hitboxes:
            self.hitbox_overlay.render(surface, core)
            
        if self.show_agent_view:
            self.agent_view_overlay.render(surface, core)
            
        if self.show_obs_panel:
            self.info_overlay.render(surface, core)
        
        # Visual Indicator for Free Cam
        if self.free_cam_active:
            txt = self.font.render("FREE CAM ACTIVE - PLAYER LOCKED", True, (255, 50, 50))
            sub = self.font.render("Use I-J-K-L to Move Camera", True, (255, 200, 200))
            
            bg_h = txt.get_height() + sub.get_height() + 15
            bg_w = max(txt.get_width(), sub.get_width()) + 20
            bg = pygame.Surface((bg_w, bg_h))
            bg.fill((0, 0, 0))
            bg.set_alpha(200)
            
            x = (core.WIDTH // 2) - (bg_w // 2)
            y = 10
            surface.blit(bg, (x, y))
            surface.blit(txt, (x + 10, y + 5))
            surface.blit(sub, (x + 10, y + 5 + txt.get_height()))