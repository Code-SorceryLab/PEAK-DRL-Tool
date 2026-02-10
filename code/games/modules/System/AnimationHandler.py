from typing import Dict, List, Optional
import pygame

class AnimationHandler:
    """
    Generic Animation Handler.
    Keys are integers (derived from Enums in the specific object classes).
    """
    def __init__(self, animations: Dict[int, List[pygame.Surface]], default_state: int = 0, duration: float = 0.1):
        self.animations = animations
        self.current_anim = default_state
        self.current_frames = self.animations.get(default_state, [])
        self.frame_index = 0
        self.timer = 0.0
        self.duration = duration

    def update(self, dt: float):
        """Advances the animation timer and frame index."""
        if not self.current_frames:
            return

        self.timer += dt
        if self.timer >= self.duration:
            self.timer -= self.duration
            self.frame_index = (self.frame_index + 1) % len(self.current_frames)

    def set_state(self, state: int):
        """
        Switch animation if different from current.
        Resets frame index to 0 when switching states.
        """
        if state != self.current_anim:
            if state in self.animations:
                self.current_anim = state
                self.current_frames = self.animations[state]
                self.frame_index = 0
                self.timer = 0.0
            else:
                # Silent failure allows asking for states that might not have images loaded yet
                pass

    def get_sprite(self, facing_right: bool = True) -> Optional[pygame.Surface]:
        """Returns the current frame, flipped if necessary."""
        if not self.current_frames:
            return None
        
        # Calculate index safely
        if len(self.current_frames) == 0:
            return None
            
        idx = self.frame_index % len(self.current_frames)
        image = self.current_frames[idx]

        # Handle flipping logic
        if not facing_right:
            return pygame.transform.flip(image, True, False)
        
        return image

    @staticmethod
    def load_animations(
        anim_dict: Dict[int, List[str]], 
        base_size: tuple[int, int] = None
    ) -> Dict[int, List[pygame.Surface]]:
        """
        Static utility to load images from disk.
        Takes: { 1: ['path/to/img.png'] }  (Where 1 is an Enum.value)
        Returns: { 1: [Surface, ...] }
        """
        loaded_anims = {}
        
        if pygame.display.get_surface() is None:
            return loaded_anims

        for anim_id, file_paths in anim_dict.items():
            frames = []
            for path in file_paths:
                try:
                    img = pygame.image.load(path).convert_alpha()
                    if base_size:
                        img = pygame.transform.scale(img, base_size)
                    frames.append(img)
                except Exception as e:
                    print(f"Failed to load {path} for animation ID {anim_id}: {e}")
            
            if frames:
                loaded_anims[anim_id] = frames
                
        return loaded_anims