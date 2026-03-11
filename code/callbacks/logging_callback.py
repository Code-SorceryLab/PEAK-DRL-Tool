import csv
import os
from stable_baselines3.common.callbacks import BaseCallback

class CsvLoggerCallback(BaseCallback):
    def __init__(self, log_dir, file_name="training_log.csv", verbose=0):
        super().__init__(verbose)
        self.log_path = os.path.join(log_dir, file_name)
        self.file_handle = None
        self.writer = None
        self.headers_written = False
        self.levels_completed = 0  # running count of WIN events seen
        print(f"[DEBUG] Logger target: {self.log_path}")

    def _on_training_start(self):
        if os.path.exists(self.log_path):
            try:
                os.remove(self.log_path)
                print(f"[INFO] Cleared old log file: {self.log_path}")
            except Exception as e:
                print(f"[WARN] Could not delete old log: {e}")

        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self.file_handle = open(self.log_path, 'w', newline='')
        self.writer = csv.writer(self.file_handle)

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [{}])
        info = infos[0] # Log the first environment's info

        main_reward = self.locals['rewards'][0]
        reward_breakdown = info.get('reward_breakdown', {})

        # --- EXTRACT TELEMETRY ---
        # These are now populated by the refactored PlatformerCore
        action_name = info.get("action_name", info.get("action", "N/A"))

        # Telemetry
        level = info.get("level", "")
        x_pos = info.get("x_position", 0.0)
        y_pos = info.get("y_position", 0.0)
        vx = info.get("velocity_x", 0.0)
        vy = info.get("velocity_y", 0.0)
        goal_dist = info.get("goal_dist", 0.0)

        # Events (Now explicitly passed from Core)
        event = info.get("event", "")
        cause = info.get("cause", "")

        # Track level completions across all envs
        if event == "WIN":
            self.levels_completed += 1

        # Observation sanity stats (populated every N steps by _check_obs_sanity in core)
        # 4-channel order: Solid(0), Collectible(1), Hazard(2), Dijkstra(3)

        _OBS_SANITY_KEYS = [
            'grid_solid_mean',       'grid_solid_std',       'grid_solid_min',       'grid_solid_max',
            'grid_hazard_mean',      'grid_hazard_std',      'grid_hazard_min',      'grid_hazard_max',
            'grid_collectible_mean', 'grid_collectible_std', 'grid_collectible_min', 'grid_collectible_max',
            'grid_dijkstra_mean',    'grid_dijkstra_std',    'grid_dijkstra_min',    'grid_dijkstra_max',
            'scalar_mean', 'scalar_std', 'scalar_min', 'scalar_max',
            'dijkstra_val', 'obs_warnings',
        ]
        obs_sanity = {k: info.get(k, 0.0) for k in _OBS_SANITY_KEYS}

        # --- BUILD ROW ---
        row_data = {
            'step': self.num_timesteps,
            'total_reward': main_reward,
            'action': action_name,
            'level': level,
            'levels_completed': self.levels_completed,
            'x': x_pos,
            'y': y_pos,
            'vx': vx,
            'vy': vy,
            'goal_dist': goal_dist,
            'event': event,
            'cause': cause,
            **reward_breakdown,
            **obs_sanity,
        }

        # Write Headers (Once)
        if not self.headers_written:
            self.headers = list(row_data.keys())
            self.writer.writerow(self.headers)
            self.headers_written = True

        # Write Data
        self.writer.writerow([row_data.get(k, 0) for k in self.headers])
        self.file_handle.flush()

        return True

    def _on_training_end(self):
        if self.file_handle:
            self.file_handle.close()