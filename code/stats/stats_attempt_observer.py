class stats_attempt_observer:
    def __init__(self, world):
        self.world = world
        self.cause_of_death = "Success"
        self.jumps = 0
        self.coins_collected = 0
        self.enemies_killed = 0
        self.sum_vx = 0
        self.count_vx = 0

    def record_jump(self):
        self.jumps += 1

    def record_death(self, cause):
        self.cause_of_death = cause

    def record_coins_collected(self):
        self.coins_collected += 1

    def record_enemies_killed(self):
        self.enemies_killed += 1

    def record_horizontal_velocity(self, vx):
        self.sum_vx += abs(vx)
        self.count_vx += 1

    def get_avg_vx(self):
        return self.sum_vx / self.count_vx if self.count_vx > 0 else 0
    
    def to_dict(self):
        return {
            "world": self.world,
            "cause_of_death": self.cause_of_death,
            "jump_count": self.jumps,
            "enemies_killed": self.enemies_killed,
            "coins_collected": self.coins_collected,
            "avg_vx": round(self.get_avg_vx(),2)
        }