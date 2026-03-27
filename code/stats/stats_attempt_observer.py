class stats_attempt_observer:
    def __init__(self, world):
        self.world = world
        self.cause_of_death = "Success"
        self.sum_vx = 0
        self.count_vx = 0

    def record_death(self, cause):
        self.cause_of_death = cause

    def record_horizontal_velocity(self, vx):
        self.sum_vx += abs(vx)
        self.count_vx += 1

    def get_avg_vx(self):
        return self.sum_vx / self.count_vx if self.count_vx > 0 else 0
    
    def to_dict(self):
        return {
            "world": self.world,
            "cause_of_death": self.cause_of_death,
            "avg_vx": round(self.get_avg_vx(),2)
        }