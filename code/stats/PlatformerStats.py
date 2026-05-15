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

    def to_dict(self):
        return {
            "world": self.world,
            "cause_of_death": self.cause_of_death,
            "jump_count": self.jumps,
            "coins_collected": self.coins_collected,
            "avg_vx": round(self.get_avg_vx(), 2),
            "progress_ratio": round(self.get_level_progress(), 2),
        }