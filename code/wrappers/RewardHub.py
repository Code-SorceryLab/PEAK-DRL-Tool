from collections import deque
import threading

class RewardHub:
    _shared_state = {}
    _lock = threading.Lock()  # Thread-safe for PPO VecEnv/Pygame

    def __new__(cls):
        """
        Ensure singleton (Only one instance gets made) behavior by sharing state across instances.
        """
        obj = super().__new__(cls)
        obj.__dict__ = cls._shared_state
        return obj

    def __init__(self):
        """
        Initialize the RewardHub with a rolling window for rewards.
        """
        if not hasattr(self, 'reward_history'):
            self.reward_history = deque(maxlen=240)  # Rolling window
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

    @classmethod
    def get_instance(cls):
        """
        Get the singleton instance of RewardHub.
        """
        return cls()