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
        self.levels_completed = 0
        self.beaten_levels = set()
        print(f"[DEBUG] Logger target: {self.log_path}")

    def _on_training_start(self):
        import datetime as _dt
        if os.path.exists(self.log_path):
            try:
                ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                backup = self.log_path.replace(".csv", f"_backup_{ts}.csv")
                os.rename(self.log_path, backup)
                print(f"[INFO] Previous log preserved → {backup}")
            except Exception as e:
                print(f"[WARN] Could not rename old log: {e}")

        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self.file_handle = open(self.log_path, 'w', newline='')
        self.writer = csv.writer(self.file_handle)

    def _on_step(self) -> bool:
        infos = list(self.locals.get('infos', [{}]) or [{}])
        rewards = self.locals.get('rewards', [0.0])

        _OBS_SANITY_KEYS = [
            'grid_solid_mean',       'grid_solid_std',       'grid_solid_min',       'grid_solid_max',
            'grid_hazard_mean',      'grid_hazard_std',      'grid_hazard_min',      'grid_hazard_max',
            'grid_collectible_mean', 'grid_collectible_std', 'grid_collectible_min', 'grid_collectible_max',
            'grid_dijkstra_mean',    'grid_dijkstra_std',    'grid_dijkstra_min',    'grid_dijkstra_max',
            'scalar_mean', 'scalar_std', 'scalar_min', 'scalar_max',
            'dijkstra_val', 'obs_warnings',
        ]

        if not hasattr(rewards, "__len__"):
            rewards = [rewards]

        for env_idx, info in enumerate(infos):
            info = info or {}
            main_reward = rewards[env_idx] if env_idx < len(rewards) else rewards[0]
            reward_breakdown = info.get('reward_breakdown', {})

            action_name = info.get("action_name", info.get("action", "N/A"))
            level = info.get("level", "")
            x_pos = info.get("x_position", 0.0)
            y_pos = info.get("y_position", 0.0)
            vx = info.get("velocity_x", 0.0)
            vy = info.get("velocity_y", 0.0)
            goal_dist = info.get("goal_dist", 0.0)
            progress = info.get("progress", 0.0)
            max_x_seen = info.get("max_x_seen", x_pos)
            dijkstra_dist = info.get("dijkstra_dist", 0.0)
            step_dx = info.get("step_dx", 0.0)
            step_dy = info.get("step_dy", 0.0)
            boss_level = info.get("boss_level", False)
            boss_active = info.get("boss_active", False)
            boss_hp_ratio = info.get("boss_hp_ratio", 0.0)
            first_completion_step = info.get("first_completion_step", -1)
            event = info.get("event", "")
            cause = info.get("cause", "")

            if event == "WIN":
                self.levels_completed += 1
                if str(level).strip():
                    self.beaten_levels.add(str(level).strip())

            obs_sanity = {k: info.get(k, 0.0) for k in _OBS_SANITY_KEYS}

            row_data = {
                'step': self.num_timesteps,
                'env_idx': env_idx,
                'total_reward': main_reward,
                'action': action_name,
                'level': level,
                'levels_completed': len(self.beaten_levels),
                'x': x_pos,
                'y': y_pos,
                'vx': vx,
                'vy': vy,
                'goal_dist': goal_dist,
                'progress': progress,
                'max_x_seen': max_x_seen,
                'dijkstra_dist': dijkstra_dist,
                'step_dx': step_dx,
                'step_dy': step_dy,
                'boss_level': boss_level,
                'boss_active': boss_active,
                'boss_hp_ratio': boss_hp_ratio,
                'first_completion_step': first_completion_step,
                'event': event,
                'cause': cause,
                **reward_breakdown,
                **obs_sanity,
            }

            if not self.headers_written:
                self.headers = list(row_data.keys())
                self.writer.writerow(self.headers)
                self.headers_written = True
                self.file_handle.flush()

            self.writer.writerow([row_data.get(k, 0) for k in self.headers])

        self.file_handle.flush()

        return True

    def _on_training_end(self):
        if self.file_handle:
            self.file_handle.close()
