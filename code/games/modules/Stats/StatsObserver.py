import inspect
from functools import wraps
from.LevelStatsObserver import LevelStatsObserver
import time

class StatsObserver:    
    def __init__(self):
        self.last_reset_time = time.time()
    
    def init_level_observers(self, level_order):
        self.levelObservers = {level: LevelStatsObserver() for level in level_order}

    def set_current_level(self, world):
        self.currentLevel = world

    def get_current_level_observer(self):
        return self.levelObservers[self.currentLevel]
    
    def record(self, event_type, **data):
         self.get_current_level_observer().record(event_type, **data)

    def get_elapsed_time(self):
        return time.time() - self.last_reset_time

    def print_all(self):
        print("\n" + "=" * 30)
        print("        GAME STATISTICS")
        print("=" * 30)
        
        self.get_current_level_observer().set_elapsed_time(self.get_elapsed_time())
        self.get_current_level_observer().print()
        
    def reset(self):
        self.get_current_level_observer().reset()
        self.last_reset_time = time.time()

statisticsObserver = StatsObserver()

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

            statisticsObserver.record(event_type, **arguments)

            return result
        return wrapper
    return decorator