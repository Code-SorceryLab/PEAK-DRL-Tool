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
        print(f"[DEBUG] Logger target: {self.log_path}")

    def _on_training_start(self):
        if os.path.exists(self.log_path):
            try:
                os.remove(self.log_path)
                print(f"[INFO] 🗑️  Cleared old log file: {self.log_path}")
            except Exception as e:
                print(f"[WARN] Could not delete old log: {e}")

        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self.file_handle = open(self.log_path, 'w', newline='')
        self.writer = csv.writer(self.file_handle)

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [{}])
        info = infos[0] # Assume single env for logging
        
        main_reward = self.locals['rewards'][0]
        reward_breakdown = info.get('reward_breakdown', {})

        # --- EXTRACT CONTEXT METRICS ---
        # We grab these from the 'info' dict provided by the game core
        
        # 1. Action Name (Added in generic_env.py)
        action_name = info.get("action_name", "N/A")
        
        # 2. Physics & Position
        level = info.get("level", 0)
        x_pos = info.get("x_position", 0.0)
        y_pos = info.get("y_position", 0.0)
        vx = info.get("velocity_x", 0.0)
        vy = info.get("velocity_y", 0.0)
        
        # 3. Goal Distance
        goal_dist = info.get("goal_dist", 0.0)

        # 4. Simple Event Detection (Replaces old dashboard logic)
        event = ""
        cause = ""
        if info.get("won", False):
            event = "WIN"
            cause = "Goal"
        elif info.get("terminated", False):
            event = "DIED"
            cause = "Hazard"

        # --- BUILD ROW ---
        row_data = {
            'step': self.num_timesteps,
            'total_reward': main_reward,
            'action': action_name,
            'level': level,
            'x': x_pos,
            'y': y_pos,
            'vx': vx,
            'vy': vy,
            'goal_dist': goal_dist,
            'event': event,
            'cause': cause,
            **reward_breakdown # Add dynamic reward keys
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