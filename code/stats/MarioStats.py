from .PlatformerStats import PlatformerStats

class MarioStats(PlatformerStats):
    def __init__(self, world, goal_pos):
        super().__init__(world, goal_pos)
        self.enemies_killed = 0

    def record_enemies_killed(self):
        self.enemies_killed += 1

    def to_dict(self):
        data = super().to_dict()
        data["enemies_killed"] = self.enemies_killed
        return data