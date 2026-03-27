class stats_attempt_observer:
    def __init__(self, world):
        self.world = world
        self.sum_vx = 0
        self.count_vx = 0

    def record_horizontal_velocity(self, vx):
        self.sum_vx += abs(vx)
        self.count_vx += 1

    def record_action(self, action):
        print(f"Action tracked: {action}")

    def to_dict(self):
        avg_vx = self.sum_vx / self.count_vx if self.count_vx > 0 else 0

        return {
            "world": self.world,
            "avg_vx": avg_vx
        }