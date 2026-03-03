# code/scripts/train.py
# ============================================================================
# CHANGES FROM ORIGINAL:
#   1. CustomCombinedExtractor  →  PEAKExtractor  (replaces old buggy extractor)
#      Old extractor bug: AdaptiveMaxPool2d upsampled 5×5 → 11×11 (wrong direction)
#      PEAKExtractor improvements:
#        - Splits grids into semantic branch (ch 0-2) + Dijkstra branch (ch 3)
#          so the pre-computed gradient field doesn't pollute binary feature learning
#        - SEBlock channel attention in semantic branch
#        - Dedicated shallow Dijkstra CNN (gradient field needs less capacity)
#        - Scalar MLP deepened to 2 layers (12→64→64)
#        - Fusion layer (384→256) before SB3 MlpExtractor
#        - ~922K params total — lightweight enough for CPU + 16 parallel envs
#      Obs shape verified: grids (4,11,11) + scalars (12,) — matches env exactly
#   2. LightCombinedExtractor — kept as fast sweep option (use_light_extractor: true)
#   3. LiveVisualizationCallback — spawns a real pygame window every N steps
#   4. eval_env VecNormalize stats synced from training env each eval cycle
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
        self._vecnorm_env         = training_env
        self.best_model_save_path = Path(best_model_save_path)
        self._last_best           = -float("inf")

    def _init_callback(self):
        self.eval_cb.init_callback(self.model)

    def _on_step(self) -> bool:
        # BUG WAS: self.eval_cb._on_step() — private method skips n_calls increment,
        # so n_calls stays 0 forever → 0 % eval_freq == 0 always True → eval every step.
        # FIX: call on_step() (public) which increments n_calls before checking frequency.
        result = self.eval_cb.on_step()
        current_best = getattr(self.eval_cb, "best_mean_reward", -float("inf"))
        if current_best > self._last_best:
            self._last_best = current_best
            self.best_model_save_path.mkdir(parents=True, exist_ok=True)
            vecnorm_path = self.best_model_save_path / "best_model_vecnorm.pkl"
            self._vecnorm_env.save(str(vecnorm_path))
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
# SEBlock — Squeeze-and-Excitation channel attention
# =========================================================================
class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation channel attention.
    Learns a per-channel weight via global avg pool → FC → sigmoid.
    Adds only 2*(C^2 / reduction) parameters.

    Used in PEAKExtractor's semantic branch to let the network suppress
    the hazard channel when no enemies are on screen, amplify the
    collectible channel near coins, etc.
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = x.mean(dim=[2, 3])                           # (B, C) — global avg pool
        w = self.fc(w).unsqueeze(-1).unsqueeze(-1)        # (B, C, 1, 1)
        return x * w                                      # channel-wise scale


# =========================================================================
# PEAKExtractor — primary multi-branch extractor (default)
# =========================================================================
class PEAKExtractor(BaseFeaturesExtractor):
    """
    Multi-branch feature extractor for the PEAK platformer environment.

    Observation space
    -----------------
      grids  : Box(-1, 1, shape=(4, 11, 11))
                 ch 0  — Player        binary {0,1}
                 ch 1  — Hazards       binary {0,1}
                 ch 2  — Collectibles  binary {0,1}
                 ch 3  — Dijkstra      continuous [-1, 1]  ← separated branch
      scalars: Box(-inf, inf, shape=(12,))
                 [0-4]  player obs  (x, y, vx, vy, on_ground)
                 [5]    enemy dist  normalised
                 [6]    goal dist   normalised
                 [7]    timer       normalised
                 [8]    goal dir Y  signed normalised
                 [9]    dijkstra dist scalar
                 [10-11] steepest descent (dX, dY)

    Architecture
    ------------
    Branch A  Semantic CNN  (ch 0-2, 3×11×11)
              Conv(3→32,k=3,pad=1) → ReLU → SEBlock(32)
              Conv(32→64,k=3,pad=1) → ReLU
              Conv(64→64,k=3,stride=2,pad=1) → ReLU  [→ 64×6×6]
              Flatten → Linear(2304→256) → ReLU

    Branch B  Dijkstra CNN  (ch 3, 1×11×11)
              Conv(1→16,k=3,pad=1) → ReLU
              Conv(16→16,k=3,stride=2,pad=1) → ReLU  [→ 16×6×6]
              Flatten → Linear(576→64) → ReLU

    Branch C  Scalar MLP   (12,)
              Linear(12→64) → ReLU → Linear(64→64) → ReLU

    Fusion    Cat(256+64+64=384) → Linear(384→256) → ReLU
              → features_dim = 256

    Rationale for channel split:
      The Dijkstra channel is a pre-computed continuous gradient field.
      Mixing it with binary semantic channels in Branch A lets its
      high-magnitude signal dominate early conv gradients, slowing
      semantic feature learning. A dedicated shallow branch avoids this.

    ~922K parameters — lightweight for CPU training with 16+ parallel envs.
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim=features_dim)

        grid_shape   = observation_space["grids"].shape    # (4, H, W)
        scalar_shape = observation_space["scalars"].shape  # (12,)

        n_channels = grid_shape[0]    # 4
        n_scalars  = scalar_shape[0]  # 12
        H, W       = grid_shape[1], grid_shape[2]  # 11, 11

        # ── Branch A: Semantic CNN (channels 0-2) ───────────────────────────
        self.semantic_cnn = nn.Sequential(
            nn.Conv2d(n_channels - 1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            SEBlock(32, reduction=4),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # 11×11 → 6×6
            nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            sem_flat = self.semantic_cnn(torch.zeros(1, n_channels - 1, H, W)).shape[1]

        self.semantic_fc = nn.Sequential(
            nn.Linear(sem_flat, 256),
            nn.ReLU()
        )

        # ── Branch B: Dijkstra CNN (channel 3 only) ─────────────────────────
        self.dijkstra_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),  # 11×11 → 6×6
            nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            dij_flat = self.dijkstra_cnn(torch.zeros(1, 1, H, W)).shape[1]

        self.dijkstra_fc = nn.Sequential(
            nn.Linear(dij_flat, 64),
            nn.ReLU()
        )

        # ── Branch C: Scalar MLP ─────────────────────────────────────────────
        self.scalar_mlp = nn.Sequential(
            nn.Linear(n_scalars, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # ── Fusion ────────────────────────────────────────────────────────────
        self.fusion = nn.Sequential(
            nn.Linear(256 + 64 + 64, features_dim),  # 384 → 256
            nn.ReLU()
        )

        self._features_dim = features_dim

    def forward(self, observations: dict) -> torch.Tensor:
        grids   = observations["grids"]    # (B, 4, H, W)
        scalars = observations["scalars"]  # (B, 12)

        sem  = self.semantic_fc(self.semantic_cnn(grids[:, :3, :, :]))  # (B, 256)
        dij  = self.dijkstra_fc(self.dijkstra_cnn(grids[:, 3:, :, :]))  # (B, 64)
        scal = self.scalar_mlp(scalars)                                  # (B, 64)

        return self.fusion(torch.cat([sem, dij, scal], dim=1))           # (B, 256)


# =========================================================================
# SlimPEAKExtractor — middle ground (default for CPU+CUDA mixed setups)
# =========================================================================
class SlimPEAKExtractor(BaseFeaturesExtractor):
    """
    Trimmed version of PEAKExtractor that keeps the architecturally
    important channel split but eliminates the two biggest cost drivers:

      Removed:  SEBlock (global pool + 2× FC per forward pass)
      Removed:  Large Linear(2304→256) — replaced with AdaptiveAvgPool
                before flatten, cutting sem_flat from 2304 → 288

    Branch A  Semantic CNN  (ch 0-2, 3×11×11)
              Conv(3→16, k=3, pad=1) → ReLU
              MaxPool(2)                          [11×11 → 5×5]
              Conv(16→32, k=3, pad=1) → ReLU
              AdaptiveAvgPool(3×3)                [5×5  → 3×3]
              Flatten(288) → Linear(288→128) → ReLU

    Branch B  Dijkstra CNN  (ch 3, 1×11×11)
              Conv(1→16, k=3, pad=1) → ReLU
              AdaptiveAvgPool(3×3)                [11×11 → 3×3]
              Flatten(144) → Linear(144→32) → ReLU

    Branch C  Scalar MLP   (12,)
              Linear(12→64) → ReLU

    Fusion    Cat(128+32+64=224) → Linear(224→128) → ReLU
              → features_dim = 128

    Parameter count:
      Branch A CNN:  ~5K   (vs 646K in full PEAK)
      Branch A FC:   ~37K
      Branch B:      ~5K   (vs 39K in full PEAK)
      Scalar MLP:    ~1K
      Fusion:        ~29K
      Total:         ~77K  (vs 922K PEAK, vs 18K Light)

    Keeps what matters:
      ✓ Dijkstra channel isolated from binary semantic channels
      ✓ Dedicated scalar MLP branch
      ✓ Fusion layer before policy head
    Drops what's expensive:
      ✗ SEBlock
      ✗ Deep semantic conv stack (3 conv layers → 2)
      ✗ Oversized linear after flatten
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 128):
        super().__init__(observation_space, features_dim=features_dim)

        grid_shape   = observation_space["grids"].shape    # (4, H, W)
        scalar_shape = observation_space["scalars"].shape  # (12,)

        n_channels = grid_shape[0]    # 4
        n_scalars  = scalar_shape[0]  # 12
        H, W       = grid_shape[1], grid_shape[2]  # 11, 11

        # ── Branch A: Semantic CNN (channels 0-2) ───────────────────────────
        self.semantic_cnn = nn.Sequential(
            nn.Conv2d(n_channels - 1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 11×11 → 5×5
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3)),                 # 5×5 → 3×3
            nn.Flatten()                                  # 32×3×3 = 288
        )
        with torch.no_grad():
            sem_flat = self.semantic_cnn(torch.zeros(1, n_channels - 1, H, W)).shape[1]

        self.semantic_fc = nn.Sequential(
            nn.Linear(sem_flat, 128),
            nn.ReLU()
        )

        # ── Branch B: Dijkstra CNN (channel 3 only) ─────────────────────────
        self.dijkstra_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3)),                 # 11×11 → 3×3
            nn.Flatten()                                  # 16×3×3 = 144
        )
        with torch.no_grad():
            dij_flat = self.dijkstra_cnn(torch.zeros(1, 1, H, W)).shape[1]

        self.dijkstra_fc = nn.Sequential(
            nn.Linear(dij_flat, 32),
            nn.ReLU()
        )

        # ── Branch C: Scalar MLP ─────────────────────────────────────────────
        self.scalar_mlp = nn.Sequential(
            nn.Linear(n_scalars, 64),
            nn.ReLU()
        )

        # ── Fusion ────────────────────────────────────────────────────────────
        self.fusion = nn.Sequential(
            nn.Linear(128 + 32 + 64, features_dim),      # 224 → 128
            nn.ReLU()
        )

        self._features_dim = features_dim

    def forward(self, observations: dict) -> torch.Tensor:
        grids   = observations["grids"]    # (B, 4, H, W)
        scalars = observations["scalars"]  # (B, 12)

        sem  = self.semantic_fc(self.semantic_cnn(grids[:, :3, :, :]))  # (B, 128)
        dij  = self.dijkstra_fc(self.dijkstra_cnn(grids[:, 3:, :, :]))  # (B,  32)
        scal = self.scalar_mlp(scalars)                                  # (B,  64)

        return self.fusion(torch.cat([sem, dij, scal], dim=1))           # (B, 128)


# =========================================================================
# LightCombinedExtractor — fast sweep option (use_light_extractor: true)
# =========================================================================
class LightCombinedExtractor(BaseFeaturesExtractor):
    """
    Lightweight MobileNet-style extractor for fast hyperparameter sweeps.
    Enable with use_light_extractor: true in your algo yaml.

    Architecture
    ------------
    Layer 1:  Conv(4→16) + BatchNorm + ReLU → MaxPool(2)     [11×11 → 5×5]
    Layer 2:  Depthwise(16) + Pointwise(16→32) + BatchNorm   [5×5  → 5×5]
    Summary:  AdaptiveAvgPool(3×3) → Flatten(288)
    FC:       Linear(288→128) + LayerNorm + ReLU
    Scalars:  Linear(12→64) + LayerNorm + ReLU
    features_dim = 192  (128 grids + 64 scalars)

    NOTE: Does NOT split the Dijkstra channel — all 4 channels share the
    same conv stack. Use PEAKExtractor (default) for full training runs.
    ~18K params in grids branch, 3-5× faster CPU inference than PEAKExtractor.
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
# CustomCombinedExtractor — legacy alias for old checkpoint compatibility
# Points to PEAKExtractor so old .zip files load without errors while
# still using the correct architecture on any new forward pass.
# The old implementation (buggy AdaptiveMaxPool upsample) is removed.
# =========================================================================



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
            # Extractor selection (set in algo yaml):
            #   Default                → SlimPEAKExtractor   ~77K params, features_dim=128
            #                            Dijkstra channel split preserved, no SEBlock.
            #                            Best balance of quality vs CPU/GPU speed.
            #   use_full_peak: true    → PEAKExtractor       ~922K params, features_dim=256
            #                            Full SEBlock + deep semantic branch.
            #                            Use only if training on GPU with plenty of time.
            #   use_light_extractor: true → LightCombinedExtractor  ~18K params
            #                            No channel split. Use for rapid sweeps only.
            # obs shape: grids (4,11,11) + scalars (12,) — verified against platformer_core.py
            if algo_conf.get("use_light_extractor", False):
                policy_kwargs["features_extractor_class"] = LightCombinedExtractor
                print("[INFO] Using LightCombinedExtractor (~18K params, fast sweep mode).")
            elif algo_conf.get("use_full_peak", True):
                policy_kwargs["features_extractor_class"] = PEAKExtractor
                policy_kwargs.setdefault("features_extractor_kwargs", {"features_dim": 256})
                print("[INFO] Using PEAKExtractor (~922K params, full architecture).")
            else:
                policy_kwargs["features_extractor_class"] = SlimPEAKExtractor
                policy_kwargs.setdefault("features_extractor_kwargs", {"features_dim": 128})
                print("[INFO] Using SlimPEAKExtractor (~77K params, channel split, no SEBlock).")

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