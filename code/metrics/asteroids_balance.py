import numpy as np
import json
from collections import defaultdict
from typing import Any, Dict

class AsteroidsBalanceStats:
    """
    Collects per-episode and aggregate metrics for Asteroids RL evaluation.
    Compatible with collector/evaluate.py.

    Metrics:
      - episode_accuracy: asteroids destroyed / bullets fired (per episode, calculated at termination)
      - episode_score_per_min: score / episode duration (min)
      - episode_reward_per_min: reward / episode duration (min)
      - episode_score_reward_ratio: score / reward (per episode)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.episode_steps = []
        self.scores = []
        self.levels = []
        self.bullets_fired = []
        self.asteroids_destroyed = []
        self.hyperspace_used = []
        self.lives_lost = []
        self.collisions = []
        self.mean_speeds = []
        self.mean_targeting = []
        self.mean_distances = []
        self.deaths_by = []
        self.episodes = 0

        self.episode_rewards = []
        self.episode_max_reward = []
        self.episode_min_reward = []
        self.episode_mean_reward = []
        self.reward_rates = []

        # New metrics
        self.episode_accuracy = []
        self.episode_score_per_min = []
        self.episode_reward_per_min = []
        self.episode_score_reward_ratio = []

        # temp step storage
        self._cur_steps = 0
        self._cur_score = 0
        self._cur_level = 0
        self._cur_bullets = 0
        self._cur_asteroids = 0
        self._cur_hyperspace = 0
        self._cur_lives = 0
        self._cur_collisions = 0
        self._cur_speeds = []
        self._cur_targeting = []
        self._cur_distances = []
        self._cur_rewards = []
        self._cur_death_by = ""

    def on_step(self, obs: Any, action: int, reward: float, done: bool, info: Dict):
        self._cur_steps += 1
        self._cur_score = info.get("score", 0)
        self._cur_level = info.get("level", 1)
        self._cur_bullets += info.get("bullets_fired", 0)
        self._cur_asteroids += info.get("asteroids_destroyed", 0)
        self._cur_hyperspace += int(info.get("hyperspace_used", False))
        self._cur_lives = info.get("lives", 0)
        self._cur_collisions += int(info.get("collision", False))
        self._cur_speeds.append(info.get("ship_speed", 0.0))
        self._cur_targeting.append(info.get("targeting_bonus", 0.0))
        self._cur_distances.append(info.get("distance_to_nearest", 0.0))
        self._cur_rewards.append(float(reward))
        #print(f"[STEP] reward={reward}, info={info}")

        if done:
            if info.get("collision", False) or self._cur_lives < 1:
                self._cur_death_by = "collision"
            elif info.get("level_completed", False):
                self._cur_death_by = "completed"
            else:
                self._cur_death_by = "other"

            self.episode_steps.append(self._cur_steps)
            self.scores.append(self._cur_score)
            self.levels.append(self._cur_level)
            self.bullets_fired.append(self._cur_bullets)
            self.asteroids_destroyed.append(self._cur_asteroids)
            self.hyperspace_used.append(self._cur_hyperspace)
            self.lives_lost.append(3 - self._cur_lives)
            self.collisions.append(self._cur_collisions)
            self.mean_speeds.append(float(np.mean(self._cur_speeds)) if self._cur_speeds else 0.0)
            self.mean_targeting.append(float(np.mean(self._cur_targeting)) if self._cur_targeting else 0.0)
            self.mean_distances.append(float(np.mean(self._cur_distances)) if self._cur_distances else 0.0)
            self.deaths_by.append(self._cur_death_by)

            # Reward stats
            total_reward = float(np.sum(self._cur_rewards)) if self._cur_rewards else 0.0
            max_reward = float(np.max(self._cur_rewards)) if self._cur_rewards else 0.0
            min_reward = float(np.min(self._cur_rewards)) if self._cur_rewards else 0.0
            mean_reward = float(np.mean(self._cur_rewards)) if self._cur_rewards else 0.0
            self.episode_rewards.append(total_reward)
            self.episode_max_reward.append(max_reward)
            self.episode_min_reward.append(min_reward)
            self.episode_mean_reward.append(mean_reward)
            rate = total_reward / self._cur_steps if self._cur_steps > 0 else 0.0
            self.reward_rates.append(rate)

            # --- Metrics calculated at episode end only ---
            acc = self._cur_asteroids / max(1, self._cur_bullets)
            duration_min = self._cur_steps / 60.0  # Assume env runs at 60 steps/sec
            score_min = self._cur_score / duration_min if duration_min > 0 else 0.0
            reward_min = total_reward / duration_min if duration_min > 0 else 0.0
            score_reward_ratio = self._cur_score / max(1.0, total_reward)
            self.episode_accuracy.append(acc)
            self.episode_score_per_min.append(score_min)
            self.episode_reward_per_min.append(reward_min)
            self.episode_score_reward_ratio.append(score_reward_ratio)

            self.episodes += 1
            self._cur_steps = 0
            self._cur_bullets = 0
            self._cur_asteroids = 0
            self._cur_hyperspace = 0
            self._cur_lives = 0
            self._cur_collisions = 0
            self._cur_speeds = []
            self._cur_targeting = []
            self._cur_distances = []
            self._cur_rewards = []
            self._cur_death_by = ""

    def summary(self):
        return dict(
            episodes=self.episodes,
            episode_steps=self.episode_steps,
            scores=self.scores,
            levels=self.levels,
            bullets_fired=self.bullets_fired,
            asteroids_destroyed=self.asteroids_destroyed,
            episode_accuracy=self.episode_accuracy,
            episode_score_per_min=self.episode_score_per_min,
            episode_reward_per_min=self.episode_reward_per_min,
            episode_score_reward_ratio=self.episode_score_reward_ratio,
            hyperspace_used=self.hyperspace_used,
            lives_lost=self.lives_lost,
            collisions=self.collisions,
            mean_speeds=self.mean_speeds,
            mean_targeting=self.mean_targeting,
            mean_distances=self.mean_distances,
            deaths_by=self.deaths_by,
            episode_rewards=self.episode_rewards,
            episode_max_reward=self.episode_max_reward,
            episode_min_reward=self.episode_min_reward,
            episode_mean_reward=self.episode_mean_reward,
            reward_rates=self.reward_rates,
        )

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump(self.summary(), f, indent=2)
