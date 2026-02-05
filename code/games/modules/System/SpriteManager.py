import pygame
import os

class SpriteManager:
    def __init__(self, assets_dir, sprite_width=32, sprite_height=32, scale=1.0):
        self.assets_dir = assets_dir
        self.sprite_width = sprite_width
        self.sprite_height = sprite_height
        self.scale = scale
        
        # --- THE ANIMATION DICTIONARY ---
        # Key: Action Name (str), Value: List of pygame.Surfaces
        self.animations = {}
        
        print(f"[SpriteManager] Pre-slicing assets from: {assets_dir}")
        
        # Map logical names to your actual filenames
        self.file_map = {
            "idle": "Idle.png",
            "run": "Run.png",
            "jump": "Jump.png",
            "fall": "Jump.png",       # Reuse Jump if Fall is missing
            "attack": "Attack.png",
            "hurt": "Hurt.png",
            "walk": "Walk.png"
        }
        
        # Load and slice every file found
        for action, filename in self.file_map.items():
            self._load_and_slice(action, filename)

    def _load_and_slice(self, action_key, filename):
        path = os.path.join(self.assets_dir, filename)
        
        if not os.path.exists(path):
            if action_key in ["idle", "run"]:
                print(f"[SpriteManager] WARNING: Critical asset '{filename}' missing.")
            return

        try:
            # 1. Load the full strip
            sheet = pygame.image.load(path).convert()
            
            # Auto-detect background color (top-left pixel) for transparency
            colorkey = sheet.get_at((0, 0))
            sheet.set_colorkey(colorkey)
            
            sheet_w, sheet_h = sheet.get_size()
            
            # 2. Calculate frame count
            n_frames = sheet_w // self.sprite_width
            
            frames_list = []
            for i in range(n_frames):
                # Calculate the slice rectangle
                x = i * self.sprite_width
                rect = pygame.Rect(x, 0, self.sprite_width, self.sprite_height)
                
                try:
                    # Create a new Surface for this single frame
                    frame_surf = pygame.Surface((self.sprite_width, self.sprite_height)).convert()
                    frame_surf.fill(colorkey) # Fill with transparent key
                    frame_surf.blit(sheet, (0, 0), rect) # Copy the slice
                    frame_surf.set_colorkey(colorkey) # Re-apply transparency
                    
                    # Scale if needed (e.g. 1.5x zoom)
                    if self.scale != 1.0:
                        new_w = int(self.sprite_width * self.scale)
                        new_h = int(self.sprite_height * self.scale)
                        frame_surf = pygame.transform.scale(frame_surf, (new_w, new_h))
                    
                    frames_list.append(frame_surf)
                except ValueError:
                    continue
            
            # 3. Store in Dictionary
            if frames_list:
                self.animations[action_key] = frames_list
                print(f"[SpriteManager] Cached '{action_key}': {len(frames_list)} frames.")
                
        except pygame.error as e:
            print(f"[SpriteManager] Error processing {filename}: {e}")

    def get_frame(self, action_key, frame_idx, facing_right=True):
        # 1. Retrieve the frame list
        frames = self.animations.get(action_key)
        
        # Fallback if specific action is missing (e.g. "attack" not loaded)
        if not frames: 
            frames = self.animations.get("idle")
        
        if not frames: return None # No assets at all

        # 2. Loop safely using Modulo
        # This prevents "index out of range" even if frame_idx is 1000
        clean_idx = int(frame_idx) % len(frames)
        image = frames[clean_idx]
        
        # 3. Flip direction
        if not facing_right:
            return pygame.transform.flip(image, True, False)
        return image