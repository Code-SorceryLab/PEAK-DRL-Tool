import pygame
import collections
from code.wrappers.RewardHub import RewardHub
from .overlays import HitboxOverlay, GridOverlay, AgentViewOverlay, InfoPanelOverlay

class DebugManager:
    def __init__(self, default_active=True, print_help=True):
        self.active = True
        # Toggles
        self.show_hitboxes = default_active
        self.show_agent_view = default_active
        self.show_sensors = default_active
        self.show_obs_panel = default_active
        self.show_grid = False
        
        # New DRL specific tools
        self.show_reward_log = False
        self.slow_motion = False
        
        # Free Cam Tools
        self.free_cam_active = False 
        self.cam_move_speed = 600.0  
        self.current_cam_move = [0.0, 0.0]

        # Metric tracking
        #self.reward_history = collections.deque(maxlen=60) # Store last 60 frames of rewards
        self.last_action_name = "None"

        # Input tracking
        self._prev_keys = pygame.key.get_pressed()
        self.font = pygame.font.SysFont("arial", 16, bold=True)
        self.small_font = pygame.font.SysFont("arial", 12)

        # Composited Visualizers
        self.hitbox_overlay = HitboxOverlay()
        self.grid_overlay = GridOverlay()
        self.agent_view_overlay = AgentViewOverlay()
        self.info_overlay = InfoPanelOverlay()

        if print_help:
            self._print_help_to_terminal()

    def _print_help_to_terminal(self):
        print("\n" + "="*40)
        print("   PEAK ENGINE - DRL INSPECTOR ENABLED")
        print("="*40)
        print(" [1] / [F1] : Toggle Hitboxes")
        print(" [2] / [F2] : Toggle Agent View (The Matrix)")
        print(" [3] / [F3] : Toggle Sensor Rays")
        print(" [4] / [F4] : Toggle Info Panel")
        print(" [5] / [F5] : Toggle Free Camera (I-J-K-L)")
        print(" [6] / [F6] : Toggle Tile Grid")
        print(" [7] / [F7] : Toggle Reward Trace")
        print(" [8] / [F8] : Toggle Slow Motion (0.5x Speed)")
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
        def check_toggle(k1, k2=None):
            if k2:
                return (keys[k1] and not self._prev_keys[k1]) or \
                       (keys[k2] and not self._prev_keys[k2])
            return keys[k1] and not self._prev_keys[k1]

        if check_toggle(pygame.K_1, pygame.K_F1): self.show_hitboxes = not self.show_hitboxes
        if check_toggle(pygame.K_2, pygame.K_F2): self.show_agent_view = not self.show_agent_view
        if check_toggle(pygame.K_3, pygame.K_F3): self.show_sensors = not self.show_sensors
        if check_toggle(pygame.K_4, pygame.K_F4): self.show_obs_panel = not self.show_obs_panel
        if check_toggle(pygame.K_5, pygame.K_F5): self.free_cam_active = not self.free_cam_active
        if check_toggle(pygame.K_6, pygame.K_F6): self.show_grid = not self.show_grid
        
        # New Toggles
        if check_toggle(pygame.K_7, pygame.K_F7): 
            self.show_reward_log = not self.show_reward_log
            print(f"[Debug] Reward Log: {self.show_reward_log}")
            
        if check_toggle(pygame.K_8, pygame.K_F8): 
            self.slow_motion = not self.slow_motion
            print(f"[Debug] Slow Motion: {self.slow_motion}")

        # --- Free Cam Movement ---
        # Changed to I-J-K-L to avoid ANY conflict with WASD or Arrows
        self.current_cam_move = [0.0, 0.0]
        if self.free_cam_active:
            if keys[pygame.K_j]:  self.current_cam_move[0] = -self.cam_move_speed
            if keys[pygame.K_l]:  self.current_cam_move[0] =  self.cam_move_speed
            if keys[pygame.K_i]:  self.current_cam_move[1] = -self.cam_move_speed
            if keys[pygame.K_k]:  self.current_cam_move[1] =  self.cam_move_speed

        self._prev_keys = keys

    def log_step(self, reward, action_name):
        """Called by core every step to log metrics"""
        #self.reward_history.append(reward)
        self.last_action_name = action_name

    def render_overlays(self, surface: pygame.Surface, core):
        """
        Draws active overlays onto the surface.
        """
        if self.show_grid:
            self.grid_overlay.render(surface, core)
            
        if self.show_hitboxes:
            self.hitbox_overlay.render(surface, core)
            
        if self.show_agent_view:
            self.agent_view_overlay.render(surface, core)
            
        if self.show_obs_panel:
            self.info_overlay.render(surface, core)
        
        if self.show_reward_log:
            self._render_reward_graph(surface, core)

        # Visual Indicator for Free Cam
        if self.free_cam_active:
            self._draw_badge(surface, core, "FREE CAM - IJKL TO MOVE", (50, 100, 255))
        
        if self.slow_motion:
            self._draw_badge(surface, core, "SLOW MOTION ACTIVE", (255, 165, 0), y_offset=40)

    def _render_reward_graph(self, surface, core):
        # Draw a mini graph of recent rewards on UPPER RIGHT
        panel_w, panel_h = 220, 140
        x = core.WIDTH - panel_w - 10
        y = 50  # Moved to top right, below Time display
        
        # bg
        s = pygame.Surface((panel_w, panel_h))
        s.set_alpha(230) # Increased visibility
        s.fill((20, 20, 20))
        surface.blit(s, (x, y))
        
        # Border
        pygame.draw.rect(surface, (100, 100, 100), (x, y, panel_w, panel_h), 1)
        
        # --- Info Text ---
        # 1. Action
        t_action = self.font.render(f"Last Action: {self.last_action_name}", True, (255, 255, 255))
        surface.blit(t_action, (x + 5, y + 5))
        
        # 2. Persona
        persona_name = getattr(core, "persona", "Default")
        # Format "platformer_dense" to "Platformer Dense"
        clean_persona = persona_name.replace("_", " ").title()
        t_persona = self.small_font.render(f"Persona: {clean_persona}", True, (200, 200, 200))
        surface.blit(t_persona, (x + 5, y + 25))
        
        # 3. Reward Function Name
        rf_name = "Internal Default"
        if hasattr(core, "reward_fn") and core.reward_fn:
            # Unwrap if it's wrapped (the uploaded files wrap functions)
            if hasattr(core.reward_fn, "__name__"):
                rf_name = core.reward_fn.__name__
            # If it's a functools.partial or similar
            elif hasattr(core.reward_fn, "func"):
                rf_name = core.reward_fn.func.__name__
                
        # Format function name for display
        clean_rf_name = rf_name.replace("_", " ").title()
        t_rf = self.small_font.render(f"{clean_rf_name}", True, (200, 200, 200))
        surface.blit(t_rf, (x + 5, y + 40))

        # --- Graph ---
        graph_rect = pygame.Rect(x + 5, y + 60, panel_w - 10, panel_h - 65)
        #pygame.draw.rect(surface, (50,50,50), graph_rect, 1) # inner border

        reward_history = RewardHub.get_instance().reward_history
        if len(reward_history) < 2: return

        # Normalize metrics for graph
        max_r = max(max(reward_history), 0.1)
        min_r = min(min(reward_history), -0.1)
        r_range = max_r - min_r
        
        if r_range == 0: r_range = 1.0
        
        points = []
        for i, r in enumerate(reward_history):
            px = graph_rect.left + (i / (len(reward_history) - 1)) * graph_rect.width
            # Map r to 0..h (inverted because y is down)
            # relative height in graph area
            rel_h = (r - min_r) / r_range
            py = graph_rect.bottom - (rel_h * graph_rect.height)
            points.append((px, py))
            
        if len(points) > 1:
            pygame.draw.lines(surface, (0, 255, 0), False, points, 2)
            
        # Current Value
        curr = reward_history[-1]
        col = (0, 255, 0) if curr > 0 else ((255, 0, 0) if curr < 0 else (200, 200, 200))
        val_t = self.small_font.render(f"R: {curr:.4f}", True, col)
        surface.blit(val_t, (x + panel_w - val_t.get_width() - 5, y + 60))

    def _draw_badge(self, surface, core, text, color, y_offset=10):
        txt = self.font.render(text, True, (255, 255, 255))
        bg = pygame.Surface((txt.get_width() + 20, txt.get_height() + 10))
        bg.fill(color)
        bg.set_alpha(230) # Increased visibility
        
        x = (core.WIDTH // 2) - (bg.get_width() // 2)
        y = y_offset
        
        surface.blit(bg, (x, y))
        surface.blit(txt, (x + 10, y + 5))