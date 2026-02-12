# code/scripts/train.py
import os
import subprocess
from code.callbacks.AnnealCallback import AnnealCallback
from code.callbacks.RecurrentEvalCallback import RecurrentEvalCallback
from code.callbacks.logging_callback import CsvLoggerCallback
import code.rewards.train_platformer as reward_module

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
import torch.nn as nn
from gymnasium import spaces
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EventCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv, DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib import RecurrentPPO

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
# Custom Multimodal Extractor (Fixes PyTorch Crash on small grids)
# =========================================================================
class CustomCombinedExtractor(BaseFeaturesExtractor):
    """
    ===== FIXED VERSION WITH ADAPTIVE POOLING =====
    
    REASON: Old CNN had fixed architecture that broke with different grid sizes.
            For 21x21 input: Conv->Pool->Conv->Flatten
            - After Conv2d(3x3, pad=1): still 21x21
            - After MaxPool2d(2): becomes 10x10 (floor division!)
            - After Conv2d(3x3, pad=1): still 10x10
            - Flatten: 10*10*64 = 6400 features
            
            But forward pass tried to use shape from sample, which could mismatch.
            AdaptiveAvgPool2d ensures consistent output regardless of input size.
    """
    def __init__(self, observation_space: spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires int!
        super().__init__(observation_space, features_dim=1)

        extractors = {}
        total_concat_size = 0

        for key, subspace in observation_space.spaces.items():
            if key == "grids":
                # We will just use a simpler CNN appropriate for 21x21 or 11x9
                n_input_channels = subspace.shape[0]
                
                # ===== FIX #12: ADD ADAPTIVE POOLING FOR VARIABLE GRID SIZES =====
                # OLD CODE:
                # cnn = nn.Sequential(
                #     nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
                #     nn.ReLU(),
                #     nn.MaxPool2d(2),
                #     nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                #     nn.ReLU(),
                #     nn.Flatten(),  # Output size depends on input size!
                # )
                # # Compute shape by doing one forward pass
                # with torch.no_grad():
                #     sample_tensor = torch.as_tensor(subspace.sample()[None]).float()
                #     n_flatten = cnn(sample_tensor).shape[1]  # Could mismatch at runtime
                
                # NEW CODE: Use adaptive pooling for consistent output
                cnn = nn.Sequential(
                    nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    # ADDED: AdaptiveAvgPool2d ensures output is always 7x7
                    # This works for ANY input size (11x9, 21x21, etc.)
                    nn.AdaptiveAvgPool2d((7, 7)),
                    nn.Flatten(),
                )

                # Fixed output size: 7 * 7 * 64 = 3136
                n_flatten = 7 * 7 * 64
                # =================================================================

                # ===== FIX #13: INCREASE FEATURE CAPACITY =====
                # OLD CODE:
                # linear = nn.Sequential(nn.Linear(n_flatten, 128), nn.ReLU())
                # total_concat_size += 128
                
                # NEW CODE: Increase to 256 for better capacity with larger grids
                linear = nn.Sequential(nn.Linear(n_flatten, 256), nn.ReLU())
                extractors[key] = nn.Sequential(cnn, linear)
                total_concat_size += 256
                # ==============================================

            elif key == "scalars":
                # Standard MLP for the 1D scalars
                extractors[key] = nn.Sequential(
                    nn.Linear(subspace.shape[0], 64),
                    nn.BatchNorm1d(64),
                    nn.ReLU()
                )
                total_concat_size += 64

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
            
        # Return a (B, features_dim) PyTorch tensor
        return torch.cat(encoded_tensor_list, dim=1)

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
        print(f"Please ensure that the file 'code/games/{game_name}_core.py' exists and contains a class named '{game_class_name}'.")
        print(f"Original error: {e}")
        sys.exit(1)

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

    # This checks if dashboard=True (default) and launches the viewer
    if cfg.get("dashboard", True):
        dash_script = repo_root / "dashboard_viewer.py"
        if dash_script.exists():
            print(f"[INFO] 🚀 Launching Flight Recorder...")
            # Popen launches it in the background so training continues!
            subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(dash_script)])
        else:
            print(f"[WARNING] Dashboard script not found at {dash_script}")
    
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
        if policy == "MultiInputPolicy":
            if policy_kwargs is None:
                policy_kwargs = {}
            policy_kwargs["features_extractor_class"] = CustomCombinedExtractor
        
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
            env_kwargs = base_env_kwargs.copy()
            env_kwargs['persona'] = persona

            # --- FIX: LOAD THE ACTUAL REWARD FUNCTION ---
            active_reward_fn = None
            if hasattr(reward_module, persona):
                active_reward_fn = getattr(reward_module, persona)
                print(f"[INFO] Loaded reward persona: {persona}")
            else:
                print(f"[WARNING] Persona '{persona}' not found in train_platformer.py! Using default.")
                active_reward_fn = reward_module.default

            def make_env():
                def _init():
                    # --- FIX: PASS THE FUNCTION TO THE WRAPPER ---
                    return GameEnv(game_cls, reward_fn=active_reward_fn, **env_kwargs)
                return _init

            n_envs = int(cfg.get("n_envs", 1))
            
            if n_envs > 1:
                # subroutine for multiple envs
                env = SubprocVecEnv([make_env() for _ in range(n_envs)])
            else:
                # Wrap in DummyVecEnv for consistency.
                # This ensures that the environment always behaves like a vectorized environment.
                env = DummyVecEnv([make_env()])
            
            # Dedicated eval env (no training noise)
            # IMPORTANT: Do not wrap eval_env in a VecEnv. Callbacks expect a single environment.
            eval_env = GameEnv(game_cls, **env_kwargs)

            for skill, total_timesteps in selected_skills.items():
                run_count += 1

                # TB directory for this (game × algo × persona)
                tb_dir = os.path.join(tb_root, f"{game_name}_{model_name}_{persona}")
                os.makedirs(tb_dir, exist_ok=True)
                
                log_name = f"training_log_{game_name}_{persona}.csv"
                csv_logger = CsvLoggerCallback(log_dir=str(repo_root), file_name=log_name)
                
                
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

                
                use_dashboard = bool(cfg.get("dashboard", True))
                

                # Start with standard callbacks
                current_callbacks = [eval_cb, ckpt_cb, csv_logger]
            
                
                # Inject TensorBoard path into algo kwargs
                train_kwargs = dict(algo_kwargs)
                train_kwargs["tensorboard_log"] = tb_dir
                train_kwargs["device"] = device


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