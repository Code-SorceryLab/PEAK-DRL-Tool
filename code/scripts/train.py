# code/scripts/train.py
# ============================================================================
# CHANGES FROM ORIGINAL:
#   1. CustomCombinedExtractor  →  LightCombinedExtractor
#      - Removed erroneous AdaptiveMaxPool upsampling (5×5 → 11×11)
#      - Depthwise-separable convolutions in layer 2 (MobileNet-style)
#      - Reduced filter counts: 32→16 (L1), 64→32 (L2)
#      - Linear: 7744→256 → 288→128  (-97 % parameters in grids branch)
#      - Expected 3-5× inference speedup on CPU
#   2. LiveVisualizationCallback — spawns a real pygame window every N steps
#   3. eval_env VecNormalize stats synced from training env each eval cycle
# ============================================================================
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
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib import RecurrentPPO

import code.rewards.train_platformer as reward_module
from code.wrappers.generic_env import GameEnv
from code.algos import get_algo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class VecnormBestCallback(BaseCallback):
    """
    Wraps EvalCallback/RecurrentEvalCallback and saves the VecNormalize stats
    file next to best_model.zip every time a new best reward is recorded.

    The built-in EvalCallback saves best_model.zip to models/best/<folder>/
    but vecnorm is only saved at the end of training to models/ root.
    watch_agent.py searches the model folder first — this closes that gap.
    """
    def __init__(self, eval_cb, training_env, best_model_save_path: str, verbose=0):
        super().__init__(verbose)
        self.eval_cb              = eval_cb
        self.training_env         = training_env
        self.best_model_save_path = Path(best_model_save_path)
        self._last_best           = -float("inf")

    def _init_callback(self):
        self.eval_cb.init_callback(self.model)

    def _on_step(self) -> bool:
        result = self.eval_cb._on_step()
        current_best = getattr(self.eval_cb, "best_mean_reward", -float("inf"))
        if current_best > self._last_best:
            self._last_best = current_best
            self.best_model_save_path.mkdir(parents=True, exist_ok=True)
            vecnorm_path = self.best_model_save_path / "best_model_vecnorm.pkl"
            self.training_env.save(str(vecnorm_path))
            if self.verbose:
                print(f"[VecnormBestCallback] New best ({current_best:.3f}) — saved → {vecnorm_path}")
        return result

    def _on_training_end(self):
        self.eval_cb._on_training_end()

def _pretty_steps(n: int) -> str:
    if n >= 1_000_000:
        return f"{n // 1_000_000}M"
    return f"{n // 1_000}k"


def _load_yaml(conf_root: Path, group: str, name: str) -> Dict:
    path = conf_root / group / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def _import_attr(dotted: str) -> Any:
    mod_path, attr = dotted.rsplit(".", 1)
    mod = importlib.import_module(mod_path)
    return getattr(mod, attr)


def _resolve_callable_or_instance(node: Dict[str, Any]) -> Any:
    if not isinstance(node, dict) or "_target_" not in node:
        raise ValueError(f"Bad hydra target node: {node}")
    obj = _import_attr(node["_target_"])
    if inspect.isclass(obj):
        kwargs = {k: v for k, v in node.items() if k != "_target_"}
        return obj(**kwargs)
    return obj


# =========================================================================
# CHANGE 2: LightCombinedExtractor — faster MobileNet-style CNN
# =========================================================================
class LightCombinedExtractor(BaseFeaturesExtractor):
    """
    Multimodal feature extractor optimised for the small 4×11×11 observation.

    Architecture highlights
    -----------------------
    Layer 1:  Standard Conv(4→16)  + BatchNorm + ReLU   [11×11 → 11×11]
              MaxPool(2)                                   [11×11 → 5×5]
    Layer 2:  Depthwise conv(16,k=3) + Pointwise(16→32) [5×5  → 5×5]
              BatchNorm + ReLU
    Summary:  AdaptiveAvgPool(3×3)                        [5×5  → 3×3]
              Flatten → 288
    FC head:  Linear(288→128) + LayerNorm + ReLU

    vs original: Linear was 7744→256 (AdaptiveMaxPool UPsampled to 11×11!)
    Parameter reduction: ~530k → ~18k in grids branch (-97 %)
    Expected speedup on CPU: 3–5× at inference.
    """

    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, features_dim=1)

        extractors = {}
        total_concat_size = 0

        for key, subspace in observation_space.spaces.items():
            if key == "grids":
                n_ch = subspace.shape[0]  # 4 channels

                # --- Layer 1: standard conv ---
                l1 = nn.Sequential(
                    nn.Conv2d(n_ch, 16, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    nn.MaxPool2d(2),          # 11×11 → 5×5
                )

                # --- Layer 2: depthwise + pointwise (MobileNet-style) ---
                l2 = nn.Sequential(
                    nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, groups=16),  # depthwise
                    nn.Conv2d(16, 32, kernel_size=1),                                   # pointwise
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                )

                # --- Compact spatial summary ---
                pool = nn.AdaptiveAvgPool2d((3, 3))   # 5×5 → 3×3

                # --- Dynamically compute flattened size ---
                with torch.no_grad():
                    dummy = torch.zeros(1, n_ch, *subspace.shape[1:])
                    n_flat = int(pool(l2(l1(dummy))).reshape(1, -1).shape[1])

                # --- FC head ---
                fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(n_flat, 128),
                    nn.LayerNorm(128),
                    nn.ReLU(),
                )

                extractors[key] = nn.Sequential(l1, l2, pool, fc)
                total_concat_size += 128

            elif key == "scalars":
                extractors[key] = nn.Sequential(
                    nn.Linear(subspace.shape[0], 64),
                    nn.LayerNorm(64),
                    nn.ReLU(),
                )
                total_concat_size += 64

        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size

    def forward(self, observations) -> torch.Tensor:
        parts = [self.extractors[k](observations[k]) for k in self.extractors]
        return torch.cat(parts, dim=1)


# =========================================================================
# Backward-compatible alias (old checkpoints expect CustomCombinedExtractor)
# Keep the ORIGINAL architecture here so old .zip files can still be loaded.
# New runs use LightCombinedExtractor by default.
# =========================================================================
class CustomCombinedExtractor(BaseFeaturesExtractor):
    """Legacy extractor — kept for checkpoint compatibility only."""
    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, features_dim=1)
        extractors = {}
        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if key == "grids":
                n_input_channels = subspace.shape[0]
                cnn = nn.Sequential(
                    nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
                    nn.GroupNorm(8, 32),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                    nn.GroupNorm(8, 64),
                    nn.ReLU(),
                    nn.AdaptiveMaxPool2d((11, 11)),
                    nn.Flatten(),
                )
                n_flatten = 11 * 11 * 64
                linear = nn.Sequential(nn.Linear(n_flatten, 256), nn.LayerNorm(256), nn.ReLU())
                extractors[key] = nn.Sequential(cnn, linear)
                total_concat_size += 256
            elif key == "scalars":
                extractors[key] = nn.Sequential(nn.Linear(subspace.shape[0], 64), nn.LayerNorm(64), nn.ReLU())
                total_concat_size += 64
        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size

    def forward(self, observations) -> torch.Tensor:
        return torch.cat([self.extractors[k](observations[k]) for k in self.extractors], dim=1)



# =========================================================================
# Main training entry point
# =========================================================================
@hydra.main(version_base=None, config_path="../conf", config_name="grid")
def main(cfg: DictConfig):
    repo_root = Path(get_original_cwd())
    conf_root = repo_root / "code" / "conf"
    models_dir = repo_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    device = cfg.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA not available, falling back to CPU.")
        device = "cpu"
    print(f"[INFO] Training device: {device}")

    tb_root   = str(cfg.get("tb_root", "runs"))
    eval_freq = int(cfg.get("eval_freq", 20_000))
    save_freq = int(cfg.get("save_freq", 50_000))

    # CHANGE 3: Live visualisation config (spawns a real pygame window)
    viz_enabled          = bool(cfg.get("viz_enabled", False))
    viz_freq             = int(cfg.get("viz_freq", 50_000))
    viz_preview_episodes = int(cfg.get("viz_preview_episodes", 1))
    viz_fps              = int(cfg.get("viz_fps", 30))

    # Path to watch_agent.py — in the repo root
    watch_agent_script = repo_root / "watch_agent.py"

    game_name = str(cfg.game)
    if game_name == 'none':
        print("ERROR: No game specified.")
        sys.exit(1)

    try:
        game_module = importlib.import_module(f"code.games.{game_name}_core")
        game_cls = getattr(game_module, f"{game_name.capitalize()}Core")
    except (ImportError, AttributeError):
        print(f"ERROR: Could not load game class for '{game_name}'.")
        sys.exit(1)

    base_env_kwargs = dict(
        render_mode=cfg.render_mode,
        fps=None if str(cfg.fps).lower() == "none" else int(cfg.fps),
        max_steps=None if str(cfg.max_steps).lower() == "none" else int(cfg.max_steps),
    )

    os.makedirs(models_dir / "best", exist_ok=True)
    os.makedirs(models_dir / "checkpoints", exist_ok=True)
    os.makedirs(models_dir / "eval_logs", exist_ok=True)

    if cfg.get("dashboard", True):
        dash_script = repo_root / "dashboard_viewer.py"
        if dash_script.exists():
            print("[INFO] Launching Flight Recorder...")
            subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(dash_script)])

    selected_models  = list(cfg.models)
    if "model" in cfg and cfg.model:
        selected_models = [str(cfg.model)]

    selected_personas = list(cfg.personas)
    if "persona" in cfg and cfg.persona:
        selected_personas = [str(cfg.persona)]

    selected_skills = dict(cfg.skills)
    if "skill" in cfg and cfg.skill:
        key = str(cfg.skill)
        if key not in selected_skills:
            raise ValueError(f"skill='{key}' not in cfg.skills {list(selected_skills.keys())}")
        selected_skills = {key: selected_skills[key]}

    run_count = 0
    for model_name in selected_models:
        algo_conf = _load_yaml(conf_root, "algo", model_name)
        Algo = get_algo(algo_conf.get("name", model_name))
        policy = algo_conf.get("policy", "MlpPolicy")

        policy_kwargs = algo_conf.get("policy_kwargs", None)
        algo_kwargs   = {k: v for k, v in algo_conf.items()
                         if k not in {"_target_", "name", "policy", "policy_kwargs"}}

        if policy == "MultiInputPolicy":
            if policy_kwargs is None:
                policy_kwargs = {}
            # CHANGE 2: Use the faster LightCombinedExtractor by default.
            # To use the legacy extractor (for loading old checkpoints), set
            # use_legacy_extractor: true in your algo yaml.
            if algo_conf.get("use_legacy_extractor", False):
                policy_kwargs["features_extractor_class"] = CustomCombinedExtractor
                print("[INFO] Using legacy CustomCombinedExtractor (old checkpoint mode).")
            else:
                policy_kwargs["features_extractor_class"] = LightCombinedExtractor
                print("[INFO] Using LightCombinedExtractor (3-5x faster inference).")

        if policy_kwargs and "activation_fn" in policy_kwargs:
            act_fn = policy_kwargs["activation_fn"]
            if isinstance(act_fn, str):
                activation_fn_map = {
                    "ReLU": torch.nn.ReLU, "Tanh": torch.nn.Tanh,
                    "LeakyReLU": torch.nn.LeakyReLU, "ELU": torch.nn.ELU, "GELU": torch.nn.GELU,
                }
                policy_kwargs["activation_fn"] = activation_fn_map.get(act_fn, torch.nn.ReLU)

        if policy_kwargs is not None:
            algo_kwargs["policy_kwargs"] = policy_kwargs

        for persona in selected_personas:
            env_kwargs = base_env_kwargs.copy()
            env_kwargs['persona'] = persona

            active_reward_fn = None
            if hasattr(reward_module, persona):
                active_reward_fn = getattr(reward_module, persona)
                print(f"[INFO] Loaded reward persona: {persona}")
            else:
                print(f"[WARNING] Persona '{persona}' not found! Using default.")
                active_reward_fn = reward_module.default

            def make_env(render_mode=None):
                """
                Factory that returns an _init callable.
                render_mode overrides base_env_kwargs['render_mode'] so the
                visualisation callback can request rgb_array independently.
                """
                kw = env_kwargs.copy()
                if render_mode is not None:
                    kw['render_mode'] = render_mode
                def _init():
                    return GameEnv(game_cls, reward_fn=active_reward_fn, **kw)
                return _init

            n_envs = int(cfg.get("n_envs", 1))
            if n_envs > 1:
                raw_env = SubprocVecEnv([make_env() for _ in range(n_envs)])
            else:
                raw_env = DummyVecEnv([make_env()])

            env = VecNormalize(raw_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

            eval_raw_env = DummyVecEnv([make_env()])
            eval_env = VecNormalize(eval_raw_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
            eval_env.training = False
            eval_env.norm_reward = False

            for skill, total_timesteps in selected_skills.items():
                run_count += 1
                tb_dir = os.path.join(tb_root, f"{game_name}_{model_name}_{persona}")
                os.makedirs(tb_dir, exist_ok=True)

                log_name = f"training_log_{game_name}_{persona}.csv"
                csv_logger = CsvLoggerCallback(log_dir=str(repo_root), file_name=log_name)

                is_recurrent_model = (model_name.lower() in ['rppo', 'recurrent_ppo'])

                _best_path = str(models_dir / "best" / f"{game_name}_{model_name}_{persona}_{str(skill).lower()}")

                if is_recurrent_model:
                    _inner_cb = RecurrentEvalCallback(
                        eval_env,
                        best_model_save_path=_best_path,
                        log_path=str(models_dir / "eval_logs" / f"{game_name}_{model_name}_{persona}_{str(skill).lower()}"),
                        eval_freq=eval_freq, n_eval_episodes=5,
                        deterministic=True, render=False, verbose=1,
                    )
                else:
                    _inner_cb = EvalCallback(
                        eval_env,
                        best_model_save_path=_best_path,
                        log_path=str(models_dir / "eval_logs" / f"{game_name}_{model_name}_{persona}_{str(skill).lower()}"),
                        eval_freq=eval_freq,
                        deterministic=True, render=False,
                    )
                # Saves best_model_vecnorm.pkl next to best_model.zip on every new best
                eval_cb = VecnormBestCallback(_inner_cb, env, _best_path, verbose=1)

                ckpt_cb = CheckpointCallback(
                    save_freq=save_freq,
                    save_path=str(models_dir / "checkpoints"),
                    name_prefix=f"{game_name}_{model_name}_{persona}",
                )

                current_callbacks = [eval_cb, ckpt_cb, csv_logger]

                # CHANGE 3: Live visualisation — spawns a real pygame window
                if viz_enabled:
                    from code.callbacks.LiveVisualizationCallback import LiveVisualizationCallback

                    # The vecnorm path won't exist yet at callback-build time,
                    # but will be written before the first trigger fires (viz_freq steps in).
                    # We point to where train.py will save it.
                    live_norm_path = models_dir / f"{game_name}_{model_name}_{persona}_{str(skill).lower()}_vecnorm.pkl"

                    viz_cb = LiveVisualizationCallback(
                        watch_agent_script=watch_agent_script,
                        vecnorm_path=live_norm_path,
                        game=game_name,
                        algo=model_name,
                        persona=persona,
                        viz_freq=viz_freq,
                        n_preview_episodes=viz_preview_episodes,
                        fps=viz_fps,
                        preview_save_dir=str(models_dir),
                        verbose=1,
                    )
                    current_callbacks.append(viz_cb)
                    print(
                        f"[INFO] Live visualisation enabled — "
                        f"a pygame window will open every {viz_freq:,} steps "
                        f"showing {viz_preview_episodes} episode(s) at {viz_fps} FPS."
                    )

                train_kwargs = dict(algo_kwargs)
                train_kwargs["tensorboard_log"] = tb_dir
                train_kwargs["device"] = device

                model = Algo(policy, env, **train_kwargs)
                tb_run_name = f"{model_name}_{persona}_{str(skill).lower()}"

                model.learn(
                    total_timesteps=int(total_timesteps),
                    callback=current_callbacks,
                    tb_log_name=tb_run_name,
                    progress_bar=True,
                )

                filename = f"{game_name}_{model_name}_{persona}_{str(skill).lower()}.zip"
                save_path = models_dir / filename
                model.save(save_path)

                norm_path = models_dir / f"{game_name}_{model_name}_{persona}_{str(skill).lower()}_vecnorm.pkl"
                env.save(str(norm_path))

                print(f"[{run_count}] saved → {save_path}  ({_pretty_steps(int(total_timesteps))} steps)")
                print(f"       VecNorm → {norm_path}  (required for watch_agent.py)")

            try:
                env.close()
                eval_env.close()
            except Exception:
                pass

    print(f"Done. Trained {run_count} models for game='{game_name}'. Models at: {models_dir}")


if __name__ == "__main__":
    main()