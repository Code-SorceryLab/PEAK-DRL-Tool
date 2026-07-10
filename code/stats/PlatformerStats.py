import time
import numpy as np

class PlatformerStats:
    def __init__(self, world, goal_pos):
        self.world = world
        self.goal_pos = goal_pos
        self.cause_of_death = "Success"
        self.jumps = 0
        self.coins_collected = 0
        self.sum_vx = 0
        self.count_vx = 0
        self.max_x_seen = 0
        self.last_pos_record_time = 0
        self.pos_record_update_rate = 0.1
        self.route = []

    def record_jump(self):
        self.jumps += 1

    def record_death(self, cause):
        self.cause_of_death = cause

    def record_coins_collected(self):
        self.coins_collected += 1

    def record_horizontal_velocity(self, vx):
        self.sum_vx += abs(vx)
        self.count_vx += 1

    def record_max_x_seen(self, value):
        self.max_x_seen = value

    def get_avg_vx(self):
        return self.sum_vx / self.count_vx if self.count_vx > 0 else 0

    def get_level_progress(self):
        return min(self.max_x_seen / self.goal_pos,1)

    def record_position_update(self, posX, posY):        
        if time.time() - self.last_pos_record_time < self.pos_record_update_rate:
            return

        self.route.append((round(posX,1), round(posY,1)))
        self.last_pos_record_time = time.time()

    def to_dict(self):
        return {
            "world": self.world,
            "cause_of_death": self.cause_of_death,
            "jump_count": self.jumps,
            "coins_collected": self.coins_collected,
            "avg_vx": round(self.get_avg_vx(), 2),
            "progress_ratio": round(self.get_level_progress(), 2),
            "route": self.route
        }
    
    @staticmethod
    def resample_arclength(traj, n=64):
        traj = np.asarray(traj, dtype=np.float64)
        seg = np.linalg.norm(np.diff(traj, axis=0), axis=1)
        dist = np.concatenate([[0], np.cumsum(seg)])
        total = dist[-1]
        if total == 0:                      # stuck-at-spawn run
            return np.repeat(traj[:1], n, axis=0)
        targets = np.linspace(0, total, n)
        x = np.interp(targets, dist, traj[:, 0])
        y = np.interp(targets, dist, traj[:, 1])
        return np.stack([x, y], axis=1)

    def save_route(self):
        import os
        import pandas as pd
        
        os.makedirs("code/stats/results/", exist_ok=True)

        df = pd.DataFrame(
            [(run_id, i, x, y) for i, (x, y) in enumerate(self.route)],
            columns=["run_id", "point_index", "x", "y"],
        )
        df.to_parquet(f"{out_dir}/run_{run_id}.parquet", index=False)
