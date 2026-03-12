from ...action_map import ACTION_NAMES
import csv
import os

class LevelStatsObserver:
    def __init__(self, levelName):
        self.deathCause = "Success" 
        self.jumps = 0
        self.coins_collected = 0
        self.enemies_killed = 0
        self.actions = {i: 0 for i in ACTION_NAMES}
        self.sum_vx = 0.0
        self.count_vx = 0
        self.levelName = levelName

    def record(self, event_type, **data):
        handler_name = f"record_{event_type}"
        handler = getattr(self, handler_name, None)

        if handler:
            handler(**data)

    def record_jump(self, **data):
        self.jumps += 1

    def record_death(self, cause, **data):
        self.deathCause = cause

    def record_coins_collected(self, **data):
        self.coins_collected += 1

    def record_enemies_killed(self, **data):
        self.enemies_killed += 1

    def record_horizontal_velocity(self, vx, **data):
        self.sum_vx += abs(vx)
        self.count_vx += 1


    def record_input(self, **data):
        # Extract the keyboard/agent booleans from data
        kb_left  = data.get("kb_left", False)
        kb_right = data.get("kb_right", False)
        kb_jump  = data.get("kb_jump", False)
        kb_run   = data.get("kb_run", False)

        # Compute action index (0–9) based on the original ACTION_NAMES mapping
        if kb_run:
            if kb_left and kb_jump:
                action_id = 9  # RUN+LEFT+JUMP
            elif kb_left:
                action_id = 8  # RUN+LEFT
            elif kb_right and kb_jump:
                action_id = 7  # RUN+RIGHT+JUMP
            elif kb_right:
                action_id = 5  # RUN+RIGHT
            else:
                action_id = 0  # IDLE / run alone not mapped
        else:
            if kb_left and kb_jump:
                action_id = 6  # LEFT+JUMP
            elif kb_left:
                action_id = 1  # LEFT
            elif kb_right and kb_jump:
                action_id = 4  # RIGHT+JUMP
            elif kb_right:
                action_id = 2  # RIGHT
            elif kb_jump:
                action_id = 3  # JUMP
            else:
                action_id = 0  # IDLE
        # Increment the counter
        self.actions[action_id] += 1

    def set_elapsed_time (self, elapsed_time):
        self.elapsed_time = elapsed_time

    def get_average_vx(self):
        if (self.count_vx == 0):
            return 0
        else:
            return self.sum_vx / self.count_vx

    def write_to_csv(self, filename):
        file_exists = os.path.isfile(filename)

        row = {
            "level": self.levelName,
            "jumps": self.jumps,
            "coins_collected": self.coins_collected,
            "enemies_killed": self.enemies_killed,
            "elapsed_time": round(self.elapsed_time, 2),
            "avg_horizontal_velocity": round(self.get_average_vx(), 2),
            "cause_of_death": self.deathCause
        }

        for action_id, count in self.actions.items():
            row[f"action_{ACTION_NAMES[action_id]}"] = count

        with open(filename, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())

            # Write header only if file doesn't exist yet
            if not file_exists:
                writer.writeheader()

            writer.writerow(row)
        
    def reset(self):
        exclude = {"deaths"}  # keep elapsed time
        for attr, value in self.__dict__.items():
            if attr in exclude:
                continue
            if isinstance(value, int) or isinstance(value, float):
                setattr(self, attr, 0)
            elif isinstance(value, dict):
                setattr(self, attr, {})
            elif isinstance(value, type(self.actions)):
                setattr(self, attr, {i: 0 for i in ACTION_NAMES})
        
        self.actions = {i: 0 for i in ACTION_NAMES}