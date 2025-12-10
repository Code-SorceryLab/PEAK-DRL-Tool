from typing import Tuple

def _world_to_screen(camera_x: float, camera_y: float, x: float, y: float) -> Tuple[float, float]:
    return x - camera_x, y - camera_y
