import inspect
from functools import wraps
from.LevelStatsObserver import LevelStatsObserver
import time
import os
import csv
import atexit

class StatsObserver:    
    def __init__(self):
        self.last_reset_time = time.time()
        self.attempts = []
        atexit.register(self.at_exit)

    def record(self, event_type, **data):
         self.currentAttempt.record(event_type, **data)

    def get_elapsed_time(self):
        return time.time() - self.last_reset_time
    
    def get_level_clear_rate(self):
        result = {}

        for attempt in self.attempts:
            level = attempt["level"]
            cause = attempt.get("cause_of_death", "Unknown")

            if cause == "Success":
                continue

            if level not in result:
                result[level] = {"total_deaths": 0}

            result[level]["total_deaths"] += 1
            if cause not in result[level]:
                result[level][cause] = 0
            result[level][cause] += 1

        return result

    def get_idle_ratio(self):
        total_actions = sum(self.currentAttempt.actions.values())
        idle_count = self.currentAttempt.actions.get(0, 0)
        return idle_count / total_actions if total_actions > 0 else 0

    def get_actions_per_second(self):
        total_actions = sum(self.currentAttempt.actions.values())
        idle_count = self.currentAttempt.actions.get(0, 0)
        return (total_actions - idle_count) / self.get_elapsed_time()

    def write_level_clear_rate_csv(self, filename="level_clear_rate.csv"):
        level_data = self.get_level_clear_rate()

        all_causes = set()
        for stats in level_data.values():
            for key in stats:
                if key != "total_deaths":
                    all_causes.add(key)

        fieldnames = ["level", "total_deaths"] + sorted(all_causes)

        with open(filename, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for level, stats in level_data.items():
                row = {"level": level, "total_deaths": stats["total_deaths"]}
                for cause in all_causes:
                    row[cause] = stats.get(cause, 0)
                writer.writerow(row)

    def write_to_csv(self, deathReason = "Success", filename="stats.csv"):
        self.currentAttempt.deathCause = deathReason
        self.currentAttempt.set_elapsed_time(self.get_elapsed_time())
        row = self.currentAttempt.to_dict()

        row["Idle_Ratio"] =  round(self.get_idle_ratio(),2)
        row["Actions_Per_Second"] = round(self.get_actions_per_second(),2)

        self.attempts.append(row)

        file_exists = os.path.isfile(filename)

        with open(filename, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())

            if not file_exists:
                writer.writeheader()

            writer.writerow(row)
        
    def reset(self, world):
        self.currentAttempt = LevelStatsObserver(world)
        self.last_reset_time = time.time()

    def at_exit(self):
        self.write_level_clear_rate_csv()

statsObserver = StatsObserver()

def track(event_type):
    def decorator(func):
        sig = inspect.signature(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)

            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            # Remove "self"
            arguments = dict(bound.arguments)
            arguments.pop("self", None)

            statsObserver.record(event_type, **arguments)

            return result
        return wrapper
    return decorator