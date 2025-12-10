from __future__ import annotations
import numpy as np
import json

class TetrisBalanceStats:
    """
    Collector compatible with evaluate.py (expects on_step(), summary()).
    Aggregates per-episode stats for radar/skill-matrix later.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.rewards = []
        self.lines = []
        self.holes = []
        self.bumpiness = []
        self.height = []
        self.tetrises = []
        self.levels = []       # <--
        self.scores = []       # <--
        self.current_ep_reward = 0.0
        self.episodes = 0
        self.max_level = 1     # <--
        self.final_score = 0   # <--
        self.wins = 0

    def on_step(self, obs, act, rew, done, info):
        self.current_ep_reward += float(rew)
        lvl = int(info.get("level", 1))
        sc  = int(info.get("score", 0))
        self.lines.append(int(info.get("lines_delta", 0)))

        # ----- THIS IS THE FIX -----
        if info.get("just_locked"): 
        # -------------------------
            self.holes.append(float(info.get("avg_holes", 0.0)))
            self.bumpiness.append(float(info.get("avg_bumpiness", 0.0)))
            self.height.append(float(info.get("avg_height", 0.0)))
            self.tetrises.append(int(info.get("tetrises", 0)))
            self.levels.append(lvl)
            self.scores.append(sc)
            
        self.max_level = max(self.max_level, lvl)
        self.final_score = sc
        
        if done:
            self.rewards.append(self.current_ep_reward)
            self.current_ep_reward = 0.0
            self.episodes += 1
        if info.get("won", False):
            self.wins += 1

    def summary(self) -> dict:
        if not self.episodes:
            return {"episodes": 0}
        import numpy as np
        return {
            "episodes": self.episodes,
            #"mean_reward": float(np.sum(self.rewards)),
            "total_lines": float(np.sum(self.lines)),
            "mean_holes": float(np.mean(self.holes)),
            "mean_bumpiness": float(np.mean(self.bumpiness)),
            "mean_height": float(np.mean(self.height)),
            "total_tetrises": float(np.sum(self.tetrises)),
            "mean_level": float(np.mean(self.levels)),  # avg level over steps
            "max_level": int(self.max_level),           # peak level reached in ep
            "final_score": int(self.final_score),      # score at episode end
            "win_rate": float(self.wins) / max(1, self.episodes),

        }

    def to_json(self, path: str):
        with open(path, "w") as f:
            json.dump(self.summary(), f, indent=2)
