import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EventCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv, DummyVecEnv
from sb3_contrib import RecurrentPPO
from typing import Any, Dict, Optional, Union
import os

class RecurrentEvalCallback(EventCallback):
    """
    Custom evaluation callback that properly handles RecurrentPPO LSTM states.
    Based on EvalCallback but with LSTM state management.
    """
    
    def __init__(
        # RPPO requires VecEnv for eval_env
        self,
        eval_env: Union[VecEnv, object],
        callback_on_new_best: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
    ):
        super().__init__(callback_on_new_best, verbose=verbose)
        
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        
        # Convert to VecEnv if needed
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])
        
        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        self.log_path = log_path
        
        # For logging results
        if log_path is not None:
            os.makedirs(log_path, exist_ok=True)
    
    def _init_callback(self) -> None:
        # Calling the super class to insure callback is initialized
        super()._init_callback()
        
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
    
    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Evaluate the model
            episode_rewards = []
            episode_lengths = []
            
            # Check if model is RecurrentPPO
            is_recurrent = isinstance(self.model, RecurrentPPO)
            
            for episode_idx in range(self.n_eval_episodes):
                obs = self.eval_env.reset()
                done = False
                episode_reward = 0.0
                episode_length = 0
                
                # Initialize LSTM (Long Short Term Memory) states for RecurrentPPO
                if is_recurrent:
                    lstm_states = None
                    episode_starts = np.ones((self.eval_env.num_envs,), dtype=bool)
                
                while not done:
                    if is_recurrent:
                        # RecurrentPPO prediction with LSTM states
                        action, lstm_states = self.model.predict(
                            obs,
                            state=lstm_states,
                            episode_start=episode_starts,
                            deterministic=self.deterministic,
                        )
                        episode_starts = np.zeros((self.eval_env.num_envs,), dtype=bool)
                    else:
                        # Standard prediction
                        action, _ = self.model.predict(obs, deterministic=self.deterministic)
                    
                    obs, reward, done, info = self.eval_env.step(action)
                    
                    # Handle both scalar and array rewards
                    if isinstance(reward, (list, np.ndarray)):
                        reward = reward[0]
                    episode_reward += reward
                    episode_length += 1
                    
                    if self.render:
                        self.eval_env.render()
                    
                    # Handle vectorized env done signal
                    if isinstance(done, (list, np.ndarray)):
                        done = done[0]
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
            
            mean_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
            mean_length = np.mean(episode_lengths)
            
            self.last_mean_reward = mean_reward
            
            if self.verbose > 0:
                print(f"Eval num_timesteps={self.num_timesteps}, "
                      f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_length:.2f}")
            
            # Log to TensorBoard
            self.logger.record("eval/mean_reward", mean_reward)
            self.logger.record("eval/mean_ep_length", mean_length)
            
            # Save best model
            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print(f"New best mean reward: {mean_reward:.2f} > {self.best_mean_reward:.2f}")
                
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                
                self.best_mean_reward = mean_reward
                
                # EventCallback stores the callback as self.callback, not self.callback_on_new_best
                if self.callback is not None:
                    continue_training = self.callback.on_step()
                    if not continue_training:
                        return False
        
        return True
