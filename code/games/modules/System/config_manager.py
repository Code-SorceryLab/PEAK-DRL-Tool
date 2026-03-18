import yaml
import os
import copy

# Importing from sibling package 'Parameters' relative to 'System'
from ..Parameters import Movement_parameters as MP
from ..Parameters import Jump_parameters as JP

class ConfigManager:
    def __init__(self, config_filename="game_config.yaml"):
        # Go up 3 levels: System -> modules -> games -> [Root]
        base_dir  = os.path.dirname(os.path.abspath(__file__)) # System
        modules_dir = os.path.dirname(base_dir)                # modules
        games_dir   = os.path.dirname(modules_dir)             # games
        repo_root   = os.path.dirname(os.path.dirname(games_dir))  # repo root

        # Search order: games/, repo root, cwd, cwd/code/games/
        # Robust against subprocess spawn (Windows) where cwd may differ.
        search_paths = [
            os.path.join(games_dir,  config_filename),
            os.path.join(repo_root,  config_filename),
            os.path.join(os.getcwd(), config_filename),
            os.path.join(os.getcwd(), "code", "games", config_filename),
            config_filename,   # bare name → cwd
        ]
        full_path = next((p for p in search_paths if os.path.isfile(p)), None)
        if full_path is None:
            print(f"[ConfigManager] WARNING: '{config_filename}' not found in any search path. "
                  f"Checked: {search_paths}")

        self.yaml_data = self._load_yaml(full_path) if full_path else {}
        
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
        if not path or not os.path.isfile(path):
            print(f"[ConfigManager] Warning: Config file not found: {path!r}. Using code defaults.")
            return {}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
            if not data.get('levels'):
                print(f"[ConfigManager] Warning: No 'levels' found in {path}. Check game_config.yaml.")
            return data
        except yaml.YAMLError as exc:
            print(f"[ConfigManager] YAML parse error in {path}: {exc}")
            return {}
        except Exception as exc:
            print(f"[ConfigManager] Failed to read {path}: {exc}")
            return {}

    def _resolve_level_id(self, level_id):
        """Case-insensitive lookup — returns actual YAML key or original if not found."""
        levels = self.yaml_data.get('levels') or {}
        if level_id in levels:
            return level_id
        lower = level_id.lower()
        for key in levels:
            if str(key).lower() == lower:
                return key
        return level_id

    def get_level_config(self, level_id):
        resolved_id = self._resolve_level_id(level_id)
        final_config = copy.deepcopy(self.base_config)
        if 'defaults' in self.yaml_data:
            self._deep_update(final_config, self.yaml_data['defaults'])
        levels = self.yaml_data.get('levels') or {}
        if resolved_id in levels and levels[resolved_id]:
            self._deep_update(final_config, levels[resolved_id])
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