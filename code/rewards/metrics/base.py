from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict

class MetricsCollector(ABC):
    """Base class for per‑episode statistics."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.data = defaultdict(float)

    @abstractmethod
    def on_step(self, obs: Any, action: int, reward: float, done: bool, info: Dict):
        ...

    @abstractmethod
    def summary(self) -> Dict:
        ...