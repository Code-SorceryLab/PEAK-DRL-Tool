# code/scripts/train.py
import os
from code.callbacks.dashboard import DashboardCallback


try:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
except Exception:
    os.environ["SDL_VIDEODRIVER"] = ""

import sys
import importlib
import inspect
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EventCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv, DummyVecEnv
from sb3_contrib import RecurrentPPO

from code.wrappers.generic_env import GameEnv
from code.algos import get_algo


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

class AnnealCallback(BaseCallback):
    """
    Callback to anneal entropy coefficient and gradient clipping during training.
    Only works with algorithms that have these attributes (PPO, A2C).
    """
    def __init__(self, total_timesteps, start_ent=0.1, end_ent=0.01, start_grad_clip=1.0, end_grad_clip=0.3, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.start_ent = start_ent
        self.end_ent = end_ent
        self.start_grad_clip = start_grad_clip
        self.end_grad_clip = end_grad_clip

    def _on_step(self) -> bool:
        # Calculate current fraction of progress
        frac = self.num_timesteps / self.total_timesteps
        # Linearly interpolate
        ent_coef = self.start_ent * (1 - frac) + self.end_ent * frac
        max_grad_norm = self.start_grad_clip * (1 - frac) + self.end_grad_clip * frac
        
        # Update parameters if they exist (PPO, A2C, RecurrentPPO)
        if hasattr(self.model, 'ent_coef'):
            self.model.ent_coef = ent_coef
        if hasattr(self.model, 'max_grad_norm'):
            self.model.max_grad_norm = max_grad_norm
        
        return True


def _pretty_steps(n: int) -> str:
    """Convert large step counts to human-readable format."""
    if n >= 1_000_000:
        return f"{n // 1_000_000}M"
    return f"{n // 1_000}k"


def _load_yaml(conf_root: Path, group: str, name: str) -> Dict:
    """Load a YAML file like conf/<group>/<name>.yaml into a dict."""
    path = conf_root / group / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def _import_attr(dotted: str) -> Any:
    """Import and return the attribute given a dotted path 'pkg.mod.Attr'."""
    mod_path, attr = dotted.rsplit(".", 1)
    mod = importlib.import_module(mod_path)
    return getattr(mod, attr)


def _resolve_callable_or_instance(node: Dict[str, Any]) -> Any:
    """
    Resolve a Hydra node for rewards/callbacks:
    {'_target_': 'pkg.mod.Attr', ...kwargs}
    - If Attr is a class, instantiate with kwargs.
    - If Attr is a function/callable, return it as-is (kwargs ignored).
    """
    if not isinstance(node, dict) or "_target_" not in node:
        raise ValueError(f"Bad hydra target node: {node}")
    obj = _import_attr(node["_target_"])
    if inspect.isclass(obj):
        kwargs = {k: v for k, v in node.items() if k != "_target_"}
        return obj(**kwargs)
    return obj


@hydra.main(version_base=None, config_path="../conf", config_name="grid")
def main(cfg: DictConfig):
    """
    Trains across models × personas × skills for a game.

    Usage:
        python -m code.scripts.train game=flappy
        # or edit code/conf/grid.yaml and run without overrides
    """
    # Stable paths regardless of Hydra's run dir
    repo_root = Path(get_original_cwd())
    conf_root = repo_root / "code" / "conf"
    models_dir = repo_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Device configuration for training (CPU/GPU)
    device = cfg.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA is not available on this system, falling back to CPU.")
        device = "cpu"
    
    print(f"[INFO] Training device: {device}")

    # Logging/callback configuration
    # Default frequencies if not specified in grid.yaml
    tb_root  = str(cfg.get("tb_root", "runs"))
    eval_freq = int(cfg.get("eval_freq", 20_000))
    save_freq = int(cfg.get("save_freq", 50_000))

    # Build game CLASS (not instance)
    # Checks if the game has a game config dict
    if isinstance(cfg.game, (DictConfig, dict)):
        game_conf = OmegaConf.to_container(cfg.game, resolve=True)
        if "_target_" in game_conf:
            dotted = game_conf["_target_"]
            game_name = dotted.split(".")[-2].replace("_core", "")
        else:
            game_name = "game"
    else:
        game_conf = _load_yaml(conf_root, "game", str(cfg.game))
        game_name = str(cfg.game)

    if "_target_" not in game_conf:
        raise ValueError(f"Game config must have _target_: {game_conf}")
    game_cls = _import_attr(game_conf["_target_"])

    # Shared env params (reward set per persona)
    base_env_kwargs = dict(
        render_mode=cfg.render_mode,
        fps=None if str(cfg.fps).lower() == "none" else int(cfg.fps),
        max_steps=None if str(cfg.max_steps).lower() == "none" else int(cfg.max_steps),
    )

    # Ensure deterministic folders for artifacts
    os.makedirs(models_dir / "best", exist_ok=True)
    os.makedirs(models_dir / "checkpoints", exist_ok=True)
    os.makedirs(models_dir / "eval_logs", exist_ok=True)

    # Optional single-value shortcuts (CLI-friendly)
    selected_models = list(cfg.models)
    if "model" in cfg and cfg.model:
        selected_models = [str(cfg.model)]

    selected_personas = list(cfg.personas)
    if "persona" in cfg and cfg.persona:
        selected_personas = [str(cfg.persona)]

    selected_skills = dict(cfg.skills)
    if "skill" in cfg and cfg.skill:
        key = str(cfg.skill)
        if key not in selected_skills:
            raise ValueError(f"skill='{key}' not in cfg.skills {list(cfg.skills.keys())}")
        selected_skills = {key: selected_skills[key]}

    run_count = 0
    for model_name in selected_models:
        # Algo params from conf/algo/<model>.yaml
        algo_conf = _load_yaml(conf_root, "algo", model_name)
        Algo = get_algo(algo_conf.get("name", model_name))
        policy = algo_conf.get("policy", "MlpPolicy")

        policy_kwargs = algo_conf.get("policy_kwargs", None)
        algo_kwargs = {k: v for k, v in algo_conf.items() if k not in {"_target_", "name", "policy", "policy_kwargs"}}

        # Add policy_kwargs if present
        if policy_kwargs:
            # Convert activation_fn string to callable for SB3 compatibility
            activation_fn_map = {
                "ReLU": torch.nn.ReLU,
                "Tanh": torch.nn.Tanh,
                "LeakyReLU": torch.nn.LeakyReLU,
                "ELU": torch.nn.ELU,
                "GELU": torch.nn.GELU,
            }
            
            if "activation_fn" in policy_kwargs:
                act_fn = policy_kwargs["activation_fn"]
                if isinstance(act_fn, str):
                    if act_fn in activation_fn_map:
                        policy_kwargs["activation_fn"] = activation_fn_map[act_fn]
                    else:
                        raise ValueError(
                            f"Unknown activation_fn: '{act_fn}'. "
                            f"Available: {list(activation_fn_map.keys())}"
                        )
            
            algo_kwargs["policy_kwargs"] = policy_kwargs


        for persona in selected_personas:
            # Reward function from conf/reward/<persona>.yaml
            reward_conf = _load_yaml(conf_root, "reward", persona)
            reward_fn = _resolve_callable_or_instance(reward_conf)

            def make_env():
                def _init():
                    return GameEnv(game_cls, reward_fn=reward_fn, **base_env_kwargs)
                return _init

            n_envs = int(cfg.get("n_envs", 1))
            
            if n_envs > 1:
                # subroutine for multiple envs
                env = SubprocVecEnv([make_env() for env in range(n_envs)])
            else:
                env = GameEnv(game_cls, reward_fn=reward_fn, **base_env_kwargs)
            
            # Dedicated eval env (no training noise)
            eval_env = GameEnv(game_cls, reward_fn=reward_fn, **base_env_kwargs)

            for skill, total_timesteps in selected_skills.items():
                run_count += 1

                # TB directory for this (game × algo × persona)
                tb_dir = os.path.join(tb_root, f"{game_name}_{model_name}_{persona}")
                os.makedirs(tb_dir, exist_ok=True)
                
                # Check if this is a recurrent model
                is_recurrent_model = (model_name.lower() in ['rppo', 'recurrent_ppo'])
                eval_cb = None
                
                if is_recurrent_model:
                    # Use custom RecurrentEvalCallback for LSTM models
                    eval_cb = RecurrentEvalCallback(
                        eval_env,
                        best_model_save_path=str(models_dir / "best" / f"{game_name}_{model_name}_{persona}_{str(skill).lower()}"),
                        log_path=str(models_dir / "eval_logs" / f"{game_name}_{model_name}_{persona}_{str(skill).lower()}"),
                        eval_freq=eval_freq,
                        n_eval_episodes=5,
                        deterministic=True,
                        render=False,
                        verbose=1,
                    )
                else:
                    # Use standard EvalCallback for non-recurrent models
                    eval_cb = EvalCallback(
                        eval_env,
                        best_model_save_path=str(models_dir / "best" / f"{game_name}_{model_name}_{persona}_{str(skill).lower()}"),
                        log_path=str(models_dir / "eval_logs" / f"{game_name}_{model_name}_{persona}_{str(skill).lower()}"),
                        eval_freq=eval_freq,
                        deterministic=True,
                        render=False,
                    )
                
                # CheckpointCallback for model saving
                ckpt_cb = CheckpointCallback(
                    save_freq=save_freq,
                    save_path=str(models_dir / "checkpoints"),
                    name_prefix=f"{game_name}_{model_name}_{persona}"
                )

                # Inject TensorBoard path into algo kwargs
                train_kwargs = dict(algo_kwargs)
                train_kwargs["tensorboard_log"] = tb_dir
                train_kwargs["device"] = device
                
                # Default to True unless user passes 'dashboard=False'
                use_dashboard = bool(cfg.get("dashboard", True))
                current_callbacks = [eval_cb, ckpt_cb]
                
                if use_dashboard:
                # If we have many parallel envs, updating every step is chaotic.
                # Update every 5 steps for smoothness.
                    dash_cb = DashboardCallback(update_freq=1)
                    current_callbacks.append(dash_cb)
                    
                # Build model (SB3-compatible Algo expected)
                model = Algo(policy, env, **train_kwargs)

                # Label inside TB so runs are grouped by persona/skill
                tb_run_name = f"{model_name}_{persona}_{str(skill).lower()}"

                # Learn with callbacks + TB run label
                model.learn(
                    total_timesteps=int(total_timesteps),
                    callback=current_callbacks,
                    tb_log_name=tb_run_name,
                    progress_bar=True
                )

                # Save each trained variant (SB3 appends .zip)
                filename = f"{game_name}_{model_name}_{persona}_{str(skill).lower()}.zip"
                save_path = models_dir / filename
                
                print(f"[{run_count}] saved --> {save_path}  ({_pretty_steps(int(total_timesteps))} steps)")

            # Close envs between personas
            try:
                env.close()
                eval_env.close()
            except Exception:
                pass

    print(f"Done. Trained {run_count} models for game='{game_name}'. Models at: {models_dir}")


if __name__ == "__main__":
    main()
