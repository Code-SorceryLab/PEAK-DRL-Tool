# code/scripts/train.py
import os
from code.callbacks.logging_callback import CsvLoggerCallback

try:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
except Exception:
    os.environ["SDL_VIDEODRIVER"] = ""

import sys
import subprocess
import importlib
import inspect
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EventCallback, EvalCallback, CheckpointCallback
# --- FIX: Import VecNormalize ---
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib import RecurrentPPO

# Import your Reward Modules to pass to the environment
import code.rewards.train_platformer as reward_module
from code.wrappers.generic_env import GameEnv
from code.algos import get_algo


#### Helper functions START ####

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
#### Helper functions END ####

# =========================================================================
# Custom Multimodal Extractor
# =========================================================================
class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, features_dim=1)

        extractors = {}
        total_concat_size = 0

        for key, subspace in observation_space.spaces.items():
            if key == "grids":
                n_input_channels = subspace.shape[0]
                cnn = nn.Sequential(
                    nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((7, 7)),
                    nn.Flatten(),
                )
                n_flatten = 7 * 7 * 64
                linear = nn.Sequential(nn.Linear(n_flatten, 256), nn.ReLU())
                extractors[key] = nn.Sequential(cnn, linear)
                total_concat_size += 256

            elif key == "scalars":
                # Standard MLP for the 1D scalars
                extractors[key] = nn.Sequential(
                    nn.Linear(subspace.shape[0], 64),
                    nn.BatchNorm1d(64),
                    nn.ReLU()
                )
                total_concat_size += 64

        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return torch.cat(encoded_tensor_list, dim=1)

@hydra.main(version_base=None, config_path="../conf", config_name="grid")
def main(cfg: DictConfig):
    # Stable paths regardless of Hydra's run dir
    repo_root = Path(get_original_cwd())
    conf_root = repo_root / "code" / "conf"
    models_dir = repo_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Device configuration
    device = cfg.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA is not available on this system, falling back to CPU.")
        device = "cpu"
    print(f"[INFO] Training device: {device}")

    # Logging/callback configuration
    tb_root  = str(cfg.get("tb_root", "runs"))
    eval_freq = int(cfg.get("eval_freq", 20_000))
    save_freq = int(cfg.get("save_freq", 50_000))

    # Build game CLASS
    game_name = str(cfg.game)
    if game_name == 'none':
        print("ERROR: No game specified. Please run with 'game=<game_name>'")
        sys.exit(1)

    try:
        game_module_name = f"code.games.{game_name}_core"
        game_class_name = f"{game_name.capitalize()}Core"
        game_module = importlib.import_module(game_module_name)
        game_cls = getattr(game_module, game_class_name)
    except (ImportError, AttributeError) as e:
        print(f"ERROR: Could not load game class '{game_class_name}' from module '{game_module_name}'.")
        sys.exit(1)

    # Shared env params
    base_env_kwargs = dict(
        render_mode=cfg.render_mode,
        fps=None if str(cfg.fps).lower() == "none" else int(cfg.fps),
        max_steps=None if str(cfg.max_steps).lower() == "none" else int(cfg.max_steps),
    )

    os.makedirs(models_dir / "best", exist_ok=True)
    os.makedirs(models_dir / "checkpoints", exist_ok=True)
    os.makedirs(models_dir / "eval_logs", exist_ok=True)

    # 🚀 AUTO-LAUNCH DASHBOARD
    if cfg.get("dashboard", True):
        dash_script = repo_root / "dashboard_viewer.py"
        if dash_script.exists():
            print(f"[INFO] 🚀 Launching Flight Recorder...")
            subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(dash_script)])

    # Load configuration lists
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
        algo_conf = _load_yaml(conf_root, "algo", model_name)
        Algo = get_algo(algo_conf.get("name", model_name))
        policy = algo_conf.get("policy", "MlpPolicy")

        policy_kwargs = algo_conf.get("policy_kwargs", None)
        algo_kwargs = {k: v for k, v in algo_conf.items() if k not in {"_target_", "name", "policy", "policy_kwargs"}}

        if policy == "MultiInputPolicy":
            if policy_kwargs is None: policy_kwargs = {}
            policy_kwargs["features_extractor_class"] = CustomCombinedExtractor
        
        if policy_kwargs and "activation_fn" in policy_kwargs:
            act_fn = policy_kwargs["activation_fn"]
            if isinstance(act_fn, str):
                activation_fn_map = {
                    "ReLU": torch.nn.ReLU, "Tanh": torch.nn.Tanh,
                    "LeakyReLU": torch.nn.LeakyReLU, "ELU": torch.nn.ELU, "GELU": torch.nn.GELU,
                }
                policy_kwargs["activation_fn"] = activation_fn_map.get(act_fn, torch.nn.ReLU)
            algo_kwargs["policy_kwargs"] = policy_kwargs


        for persona in selected_personas:
            env_kwargs = base_env_kwargs.copy()
            env_kwargs['persona'] = persona
            
            # --- LOAD REWARD FUNCTION ---
            active_reward_fn = None
            if hasattr(reward_module, persona):
                active_reward_fn = getattr(reward_module, persona)
                print(f"[INFO] Loaded reward persona: {persona}")
            else:
                print(f"[WARNING] Persona '{persona}' not found! Using default.")
                active_reward_fn = reward_module.default

            def make_env():
                def _init():
                    return GameEnv(game_cls, reward_fn=active_reward_fn, **env_kwargs)
                return _init

            n_envs = int(cfg.get("n_envs", 1))
            
            if n_envs > 1:
                raw_env = SubprocVecEnv([make_env() for _ in range(n_envs)])
            else:
                raw_env = DummyVecEnv([make_env()])
            
            # --- FIX: WRAP IN VECNORMALIZE ---
            # This automatically normalizes observations (800 -> 1.0) and rewards
            env = VecNormalize(raw_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

            # Dedicated eval env (Also wrapped, but without reward normalization training)
            # Eval envs must be VecEnvs to use VecNormalize correctly
            eval_raw_env = DummyVecEnv([make_env()])
            eval_env = VecNormalize(eval_raw_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
            
            # Sync normalization stats from train env to eval env
            eval_env.training = False 
            eval_env.norm_reward = False

            for skill, total_timesteps in selected_skills.items():
                run_count += 1
                tb_dir = os.path.join(tb_root, f"{game_name}_{model_name}_{persona}")
                os.makedirs(tb_dir, exist_ok=True)
                
                # --- CSV Logger Save to ROOT ---
                log_name = f"training_log_{game_name}_{persona}.csv"
                csv_logger = CsvLoggerCallback(log_dir=str(repo_root), file_name=log_name)
                
                # Callbacks
                eval_cb = EvalCallback(
                    eval_env,
                    best_model_save_path=str(models_dir / "best" / f"{game_name}_{model_name}_{persona}_{str(skill).lower()}"),
                    log_path=str(models_dir / "eval_logs" / f"{game_name}_{model_name}_{persona}_{str(skill).lower()}"),
                    eval_freq=eval_freq,
                    deterministic=True,
                    render=False,
                )
                
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

                current_callbacks = [eval_cb, ckpt_cb, csv_logger]

                train_kwargs = dict(algo_kwargs)
                train_kwargs["tensorboard_log"] = tb_dir
                train_kwargs["device"] = device

                model = Algo(policy, env, **train_kwargs)
                tb_run_name = f"{model_name}_{persona}_{str(skill).lower()}"

                model.learn(
                    total_timesteps=int(total_timesteps),
                    callback=current_callbacks,
                    tb_log_name=tb_run_name,
                    progress_bar=True
                )

                # Save model + normalization stats
                filename = f"{game_name}_{model_name}_{persona}_{str(skill).lower()}.zip"
                save_path = models_dir / filename
                model.save(save_path)
                
                # Save the Normalization Stats too (Critical for loading later!)
                norm_path = models_dir / f"{game_name}_{model_name}_{persona}_{str(skill).lower()}_vecnorm.pkl"
                env.save(str(norm_path))
                
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