import inspect
from functools import wraps
from.LevelStatsObserver import LevelStatsObserver
import time

class StatsObserver:    
    def __init__(self):
        self.last_reset_time = time.time()

    def record(self, event_type, **data):
         self.currentAttempt.record(event_type, **data)

    def get_elapsed_time(self):
        return time.time() - self.last_reset_time

    def write_to_csv(self, deathReason = "Success", filename="stats.csv"):
        self.currentAttempt.deathCause = deathReason
        self.currentAttempt.set_elapsed_time(self.get_elapsed_time())
        self.currentAttempt.write_to_csv(filename)
        
    def reset(self, world):
        self.currentAttempt = LevelStatsObserver(world)
        self.last_reset_time = time.time()

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