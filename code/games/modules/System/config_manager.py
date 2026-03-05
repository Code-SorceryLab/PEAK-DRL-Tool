import yaml
import os
import copy

# Importing from sibling package 'Parameters' relative to 'System'
from ..Parameters import Movement_parameters as MP
from ..Parameters import Jump_parameters as JP

class ConfigManager:
    def __init__(self, config_filename="game_config.yaml"):
        # Go up 3 levels: System -> modules -> games -> [Root]
        base_dir = os.path.dirname(os.path.abspath(__file__)) # System
        modules_dir = os.path.dirname(base_dir) # modules
        games_dir = os.path.dirname(modules_dir) # games
        full_path = os.path.join(games_dir, config_filename)
        
        self.yaml_data = self._load_yaml(full_path)
        
        # Base Configuration (Layer 3: The Python Constants)
        self.base_config = {
            "physics": {
                "gravity": MP.GRAVITY,
                "fast_fall_gravity": MP.FAST_FALL_GRAV,
                "friction": {
                    "ground": MP.GROUND_FRICTION,
                    "air": MP.AIR_FRICTION
                }
            },
            "player": {
                "movement": {
                    "max_run_speed": MP.MAX_RUN_SPEED,
                    "run_accel": MP.RUN_ACCEL,
                    "max_walk_speed": MP.MAX_WALK_SPEED,
                    "walk_accel": MP.WALK_ACCEL,
                    "air_control": MP.AIR_CONTROL
                },
                "jump": {
                    "max_velocity": JP.JUMP_VEL_MAX,
                    "min_velocity": JP.JUMP_VEL_MIN,
                    "hold_frames": JP.JUMP_HOLD_FRAMES,
                    "coyote_frames": JP.COYOTE_FRAMES,
                    "buffer_frames": JP.JUMP_BUFFER_FRAMES
                }
            }
        }

    def _load_yaml(self, path):
        if not os.path.exists(path):
             # Try current working directory as fallback
             if os.path.exists("game_config.yaml"):
                 path = "game_config.yaml"
             else:
                 print(f"Warning: Config file {path} not found! Using code defaults.")
                 return {}
        with open(path, 'r') as f:
            try:
                return yaml.safe_load(f) or {}
            except yaml.YAMLError as exc:
                print(f"Error parsing YAML: {exc}")
                return {}

    def get_level_config(self, level_id):
        final_config = copy.deepcopy(self.base_config)
        if 'defaults' in self.yaml_data:
            self._deep_update(final_config, self.yaml_data['defaults'])
        if 'levels' in self.yaml_data and (self.yaml_data['levels'] or {}).get(level_id):
            self._deep_update(final_config, self.yaml_data['levels'][level_id])
        return final_config

    def _deep_update(self, d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = self._deep_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d
    
    def get_level_order(self):
        """Returns a list of level IDs in the order they appear in YAML."""
        levels = self.yaml_data.get('levels') or {}
        return list(levels.keys())