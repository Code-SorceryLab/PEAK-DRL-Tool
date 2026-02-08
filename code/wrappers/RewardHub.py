from collections import deque
import threading

class RewardHub:
    # REMOVED: _shared_state = {} 
    # REMOVED: __new__ method (No more Singleton)

    def __init__(self):
        """
        Initialize the RewardHub with a rolling window for rewards.
        Now allows multiple independent instances (one per Environment).
        """
        self._lock = threading.Lock() # Lock is now per-instance
        self.reward_history = deque(maxlen=240)
        self.last_action_name = "None"
        self.episodic_reward = 0.0

    def update_reward(self, reward, action_name, is_episode_end=False):
        """
        Update the reward history and track the last action name.
        """
        with self._lock:
            self.reward_history.append(reward)
            self.last_action_name = action_name
            self.episodic_reward += reward
            if is_episode_end:
                self.episodic_reward = 0.0  # Reset

    def compute_default_reward(self, info: dict) -> float:
        """
        Fallback reward if no Persona is active.
        """
        if info.get("episode_end", False) and not info.get("won", False):
            return -1.0
            
        dx = float(info.get("dx", 0.0)) 
        vx = float(info.get("velocity_x", 0.0))
        
        return (vx / 100.0) + (dx * 0.1)
    
    @classmethod
    def get_instance(cls):
        """
        Factory method to create a new instance.
        (Renamed from get_instance to create_new to be clear it's not a global singleton)
        """
        return cls()