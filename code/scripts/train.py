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
from stable_baselines3.common.monitor import Monitor
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
      grids  : Box(-1, 1, shape=(5, H, W))
                 ch 0  — Player        binary {0,1}
                 ch 1  — Solids        binary {0,1}
                 ch 2  — Collectibles  binary {0,1}
                 ch 3  — Hazards       binary {0,1}
                 ch 4  — Dijkstra      continuous [-1, 1]
      scalars: Box(-inf, inf, shape=(12,))

    Architecture
    ------------
    Branch A  Semantic CNN  (ch 0-3, 4×H×W)
              Conv(4→32, k=3) → GroupNorm(8,32) → ReLU → SEBlock(32)
              Conv(32→64, k=3) → GroupNorm(16,64) → ReLU
              Conv(64→64, k=3) → GroupNorm(16,64) → ReLU
              AdaptiveAvgPool(4×4) → Flatten(1024) → Linear(1024→256) → LayerNorm → ReLU

    Branch B  Dijkstra CNN  (ch 4, 1×H×W)
              Conv(1→16, k=3) → GroupNorm(4,16) → ReLU
              Conv(16→32, k=3) → GroupNorm(8,32) → ReLU
              AdaptiveAvgPool(3×3) → Flatten(288) → Linear(288→64) → LayerNorm → ReLU

    Branch C  Scalar MLP   (12,)
              Linear(12→64) → LayerNorm → ReLU
              Linear(64→64) → LayerNorm → ReLU

    Fusion    Cat(256+64+64=384) → Linear(384→256) → LayerNorm → ReLU
              → features_dim = 256

    Grid-size invariant: AdaptiveAvgPool absorbs any H×W before flatten.
    No stride-2 convs — spatial reduction handled entirely by pooling.

    GroupNorm instead of BatchNorm: correct for RL (no batch-dimension
    dependency, works identically in train and eval, stable under
    non-stationary observation distributions).

    SEBlock retained in Branch A: channel attention helps the network
    suppress irrelevant semantic channels per-frame (e.g. collectible
    channel when no coins are visible).
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim=features_dim)

        grid_shape   = observation_space["grids"].shape    # (5, H, W)
        scalar_shape = observation_space["scalars"].shape  # (12,)

        n_channels = grid_shape[0]    # 5
        n_scalars  = scalar_shape[0]  # 12
        H, W       = grid_shape[1], grid_shape[2]

        n_semantic = n_channels - 1   # 4 (all except Dijkstra)

        # ── Branch A: Semantic CNN (channels 0 to n_semantic-1) ─────────────
        self.semantic_cnn = nn.Sequential(
            nn.Conv2d(n_semantic, 32, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            SEBlock(32, reduction=4),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(16, 64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(16, 64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),   # any H×W → 4×4
            nn.Flatten()                    # 64×4×4 = 1024
        )
        with torch.no_grad():
            sem_flat = self.semantic_cnn(torch.zeros(1, n_semantic, H, W)).shape[1]

        self.semantic_fc = nn.Sequential(
            nn.Linear(sem_flat, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )

        # ── Branch B: Dijkstra CNN (last channel only) ───────────────────────
        self.dijkstra_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3)),   # any H×W → 3×3
            nn.Flatten()                    # 32×3×3 = 288
        )
        with torch.no_grad():
            dij_flat = self.dijkstra_cnn(torch.zeros(1, 1, H, W)).shape[1]

        self.dijkstra_fc = nn.Sequential(
            nn.Linear(dij_flat, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )

        # ── Branch C: Scalar MLP ─────────────────────────────────────────────
        self.scalar_mlp = nn.Sequential(
            nn.Linear(n_scalars, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )

        # ── Fusion ────────────────────────────────────────────────────────────
        self.fusion = nn.Sequential(
            nn.Linear(256 + 64 + 64, features_dim),   # 384 → 256
            nn.LayerNorm(features_dim),
            nn.ReLU()
        )

        self._features_dim = features_dim

    def forward(self, observations: dict) -> torch.Tensor:
        grids   = observations["grids"]    # (B, 5, H, W)
        scalars = observations["scalars"]  # (B, 12)

        # ch 0-3: semantic binary channels  |  ch 4: Dijkstra continuous
        sem  = self.semantic_fc(self.semantic_cnn(grids[:, :-1, :, :]))   # (B, 256)
        dij  = self.dijkstra_fc(self.dijkstra_cnn(grids[:, -1:, :, :]))   # (B,  64)
        scal = self.scalar_mlp(scalars)                                    # (B,  64)

        return self.fusion(torch.cat([sem, dij, scal], dim=1))             # (B, 256)


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

    Observation space
    -----------------
      grids  : Box(-1, 1, shape=(5, 11, 11))
                 ch 0  — Player        binary {0,1}
                 ch 1  — Solids        binary {0,1}  (ground, platforms, qblocks)
                 ch 2  — Collectibles  binary {0,1}
                 ch 3  — Hazards       binary {0,1}
                 ch 4  — Dijkstra      continuous [-1, 1]  ← separated branch

    Branch A  Semantic CNN  (ch 0-3, 4×11×11)
              Conv(4→16, k=3, pad=1) → ReLU
              MaxPool(2)                          [11×11 → 5×5]
              Conv(16→32, k=3, pad=1) → ReLU
              AdaptiveAvgPool(3×3)                [5×5  → 3×3]
              Flatten(288) → Linear(288→128) → ReLU

    Branch B  Dijkstra CNN  (ch 4, 1×11×11)
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

        grid_shape   = observation_space["grids"].shape    # (5, H, W)
        scalar_shape = observation_space["scalars"].shape  # (12,)

        n_channels = grid_shape[0]    # 5
        n_scalars  = scalar_shape[0]  # 12
        H, W       = grid_shape[1], grid_shape[2]  # 21,21

        # Number of semantic channels = all except the last (Dijkstra)
        n_semantic = n_channels - 1   # 4

        # ── Branch A: Semantic CNN (channels 0 to n_semantic-1) ─────────────
        self.semantic_cnn = nn.Sequential(
            nn.Conv2d(n_semantic, 16, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, 16),  # GroupNorm with num_groups=4 (1 group per channel) is a lightweight alternative to BatchNorm/SEBlock
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 21×21 → 10×10
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 32),  # num_groups=8 for 32 channels
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3)),                 # 10×10 → 3×3
            nn.Flatten()                                  # 32×3×3 = 288
        )
        with torch.no_grad():
            sem_flat = self.semantic_cnn(torch.zeros(1, n_semantic, H, W)).shape[1]

        self.semantic_fc = nn.Sequential(
            nn.Linear(sem_flat, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )

        # ── Branch B: Dijkstra CNN (last channel only) ──────────────────────
        self.dijkstra_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, 16),  # num_groups=4 for 16 channels
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3)),                 # 21×21 → 3×3
            nn.Flatten()                                  # 16×3×3 = 144
        )
        with torch.no_grad():
            dij_flat = self.dijkstra_cnn(torch.zeros(1, 1, H, W)).shape[1]

        self.dijkstra_fc = nn.Sequential(
            nn.Linear(dij_flat, 32),
            nn.LayerNorm(32),
            nn.ReLU()
        )

        # ── Branch C: Scalar MLP ─────────────────────────────────────────────
        self.scalar_mlp = nn.Sequential(
            nn.Linear(n_scalars, 64),
            nn.LayerNorm(64),  # num_groups=4 for 64 channels
            nn.ReLU()
        )

        # ── Fusion ────────────────────────────────────────────────────────────
        self.fusion = nn.Sequential(
            nn.Linear(128 + 32 + 64, features_dim),      # 224 → 128
            nn.LayerNorm(features_dim),
            nn.ReLU()
        )

        self._features_dim = features_dim

    def forward(self, observations: dict) -> torch.Tensor:
        grids   = observations["grids"]    # (B, 5, H, W)
        scalars = observations["scalars"]  # (B, 12)

        # ch 0-3: semantic binary channels  |  ch 4: Dijkstra continuous
        sem  = self.semantic_fc(self.semantic_cnn(grids[:, :-1, :, :]))  # (B, 128)
        dij  = self.dijkstra_fc(self.dijkstra_cnn(grids[:, -1:, :, :]))  # (B,  32)
        scal = self.scalar_mlp(scalars)                                   # (B,  64)

        return self.fusion(torch.cat([sem, dij, scal], dim=1))            # (B, 128)

# =========================================================================
# BalancedPEAKExtractor — sweet spot (~230K params)
# =========================================================================
class BalancedPEAKExtractor(BaseFeaturesExtractor):
    """
    Middle ground between SlimPEAK (77K) and full PEAK (922K).

    Keeps what matters from PEAK:
      ✓ Channel split (semantic vs dijkstra)
      ✓ SEBlock channel attention (cheap: ~500 params, big impact)
      ✓ 2-layer scalar MLP
      ✓ GroupNorm everywhere

    Cuts what's expensive:
      ✗ Third conv layer in semantic branch (redundant on 21×21)
      ✗ 64-channel convs (32→48 is plenty for 4 binary channels)
      ✗ 256-dim fusion (192 captures enough information)

    Branch A  Semantic CNN  (ch 0-3, 4×H×W)
              Conv(4→32, k=3) → GroupNorm(8,32) → ReLU → SEBlock(32)
              Conv(32→48, k=3) → GroupNorm(12,48) → ReLU
              AdaptiveAvgPool(4×4) → Flatten(768) → Linear(768→192) → LayerNorm → ReLU

    Branch B  Dijkstra CNN  (ch 4, 1×H×W)
              Conv(1→16, k=3) → GroupNorm(4,16) → ReLU
              Conv(16→32, k=3) → GroupNorm(8,32) → ReLU
              AdaptiveAvgPool(3×3) → Flatten(288) → Linear(288→48) → LayerNorm → ReLU

    Branch C  Scalar MLP   (12,)
              Linear(12→64) → LayerNorm → ReLU
              Linear(64→64) → LayerNorm → ReLU

    Fusion    Cat(192+48+64=304) → Linear(304→192) → LayerNorm → ReLU
              → features_dim = 192

    ~230K params — 3× slim, ~4× cheaper than peak.
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 192):
        super().__init__(observation_space, features_dim=features_dim)

        grid_shape   = observation_space["grids"].shape
        scalar_shape = observation_space["scalars"].shape

        n_channels = grid_shape[0]
        n_scalars  = scalar_shape[0]
        H, W       = grid_shape[1], grid_shape[2]
        n_semantic = n_channels - 1

        # ── Branch A: Semantic CNN ────────────────────────────────────────
        self.semantic_cnn = nn.Sequential(
            nn.Conv2d(n_semantic, 32, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            SEBlock(32, reduction=4),          # channel attention — cheap, impactful
            nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(12, 48),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),      # → 4×4
            nn.Flatten()                       # 48×4×4 = 768
        )
        with torch.no_grad():
            sem_flat = self.semantic_cnn(torch.zeros(1, n_semantic, H, W)).shape[1]

        self.semantic_fc = nn.Sequential(
            nn.Linear(sem_flat, 192),
            nn.LayerNorm(192),
            nn.ReLU()
        )

        # ── Branch B: Dijkstra CNN ────────────────────────────────────────
        self.dijkstra_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Flatten()
        )
        with torch.no_grad():
            dij_flat = self.dijkstra_cnn(torch.zeros(1, 1, H, W)).shape[1]

        self.dijkstra_fc = nn.Sequential(
            nn.Linear(dij_flat, 48),
            nn.LayerNorm(48),
            nn.ReLU()
        )

        # ── Branch C: Scalar MLP ──────────────────────────────────────────
        self.scalar_mlp = nn.Sequential(
            nn.Linear(n_scalars, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )

        # ── Fusion ────────────────────────────────────────────────────────
        self.fusion = nn.Sequential(
            nn.Linear(192 + 48 + 64, features_dim),   # 304 → 192
            nn.LayerNorm(features_dim),
            nn.ReLU()
        )

        self._features_dim = features_dim

    def forward(self, observations: dict) -> torch.Tensor:
        grids   = observations["grids"]
        scalars = observations["scalars"]

        sem  = self.semantic_fc(self.semantic_cnn(grids[:, :-1, :, :]))
        dij  = self.dijkstra_fc(self.dijkstra_cnn(grids[:, -1:, :, :]))
        scal = self.scalar_mlp(scalars)

        return self.fusion(torch.cat([sem, dij, scal], dim=1))


# =========================================================================
# LightCombinedExtractor — fast sweep option (use_light_extractor: true)
# =========================================================================
class LightCombinedExtractor(BaseFeaturesExtractor):
    """
    Lightweight MobileNet-style extractor for fast hyperparameter sweeps.
    Enable with use_light_extractor: true in your algo yaml.

    Architecture
    ------------
    Layer 1:  Conv(5→16) + BatchNorm + ReLU → MaxPool(2)     [11×11 → 5×5]
    Layer 2:  Depthwise(16) + Pointwise(16→32) + BatchNorm   [5×5  → 5×5]
    Summary:  AdaptiveAvgPool(3×3) → Flatten(288)
    FC:       Linear(288→128) + LayerNorm + ReLU
    Scalars:  Linear(12→64) + LayerNorm + ReLU
    features_dim = 192  (128 grids + 64 scalars)

    NOTE: Does NOT split the Dijkstra channel — all 5 channels share the
    same conv stack (n_ch is read dynamically from the observation space).
    Use PEAKExtractor (default) for full training runs where the Dijkstra
    channel split matters.
    ~18K params in grids branch, 3-5× faster CPU inference than PEAKExtractor.
    """

    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, features_dim=1)

        extractors = {}
        total_concat_size = 0

        for key, subspace in observation_space.spaces.items():
            if key == "grids":
                n_ch = subspace.shape[0]  # 5 channels

                # --- Layer 1: standard conv ---
                l1 = nn.Sequential(
                    nn.Conv2d(n_ch, 16, kernel_size=3, stride=1, padding=1),
                    nn.GroupNorm(4, 16),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                )

                # --- Layer 2: depthwise + pointwise (MobileNet-style) ---
                l2 = nn.Sequential(
                    nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, groups=16),
                    nn.Conv2d(16, 32, kernel_size=1),
                    nn.GroupNorm(8, 32),
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


# ─────────────────────────────────────────────────────────────────────────────
# EvalPreviewCallback
# ─────────────────────────────────────────────────────────────────────────────
# Wraps EvalCallback (or RecurrentEvalCallback).
# Every time a new best model is saved it:
#   1. Saves the VecNormalize stats alongside best_model.zip
#   2. Opens a live pygame window, plays 2 episodes, then closes it
#
# Why synchronous / in-process?
#   Training is already PAUSED while EvalCallback runs its eval episodes.
#   The preview just uses that idle window. No subprocess, no thread,
#   no Windows flash-and-close, no SDL conflicts.
# ─────────────────────────────────────────────────────────────────────────────
class EvalPreviewCallback(BaseCallback):
    def __init__(self, eval_cb, vecnorm_env, best_model_save_path,
                 make_env_fn, repo_root, fps=30, n_preview_episodes=2, verbose=1):
        super().__init__(verbose)
        self.eval_cb              = eval_cb
        self._vecnorm_env         = vecnorm_env          # training VecNormalize
        self.best_model_save_path = Path(best_model_save_path)
        self.make_env_fn          = make_env_fn           # kept for API compat
        self.repo_root            = Path(repo_root)
        self.fps                  = fps
        self.n_preview_episodes   = n_preview_episodes
        self._last_best           = float("-inf")

    # Wire the inner eval_cb so it has model / logger access
    def _init_callback(self):
        self.eval_cb.parent = self
        self.eval_cb.init_callback(self.model)

    def _on_step(self):
        result = self.eval_cb.on_step()
        current_best = getattr(self.eval_cb, "best_mean_reward", float("-inf"))
        if current_best > self._last_best:
            self._last_best = current_best
            if self.verbose:
                print(f"\n[EvalPreview] New best: {current_best:.3f}")
            self._save_vecnorm()
            if self.n_preview_episodes > 0:
                self._run_preview()
        return result

    def _on_training_end(self):
        self.eval_cb.on_training_end()

    # ── Save vecnorm next to best_model.zip ───────────────────────────────
    def _save_vecnorm(self):
        out = self.best_model_save_path / "best_model_vecnorm.pkl"
        try:
            self.best_model_save_path.mkdir(parents=True, exist_ok=True)
            self._vecnorm_env.save(str(out))
            if self.verbose:
                print(f"[EvalPreview] Saved vecnorm -> {out}")
        except Exception as e:
            print(f"[EvalPreview] vecnorm save failed: {e}")

    # ── Launch watch_agent.py as a SUBPROCESS ─────────────────────────────
    # Why subprocess?
    #   In-process pygame init/quit cycles leak Windows GDI handles.
    #   After enough eval cycles the OS limit (10,000) is hit and
    #   "Couldn't create window" errors appear. A subprocess gets its
    #   own handle space; when it exits the OS reclaims everything.
    def _run_preview(self):
        model_zip   = self.best_model_save_path / "best_model.zip"
        vecnorm_pkl = self.best_model_save_path / "best_model_vecnorm.pkl"

        if not model_zip.exists():
            print(f"[EvalPreview] Skipping preview — {model_zip} not found")
            return

        cmd = [
            sys.executable, "-m", "code.scripts.watch_agent",
            str(model_zip),
            "--episodes", str(self.n_preview_episodes),
            "--fps",      str(self.fps),
        ]
        if vecnorm_pkl.exists():
            cmd += ["--vecnorm", str(vecnorm_pkl)]

        if self.verbose:
            print(f"[EvalPreview] Launching preview subprocess "
                  f"({self.n_preview_episodes} ep @ {self.fps} FPS)...")

        try:
            # Run from repo root so "code.*" imports resolve.
            # Timeout = 120s safety net (preview should finish well before).
            proc = subprocess.Popen(
                cmd,
                cwd=str(self.repo_root),
                # Inherit stdout/stderr so preview logs are visible
                stdout=None, stderr=None,
            )
            proc.wait(timeout=120)
            if self.verbose:
                rc = proc.returncode
                print(f"[EvalPreview] Preview subprocess exited (rc={rc}) "
                      f"— training resumed.\n")
        except subprocess.TimeoutExpired:
            print("[EvalPreview] Preview timed out — killing subprocess.")
            proc.kill()
            proc.wait()
        except Exception as exc:
            print(f"[EvalPreview] Preview error: {type(exc).__name__}: {exc}")


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
        batch_window=10, advance_threshold=0.30, fallback_threshold=0.20,
        max_stay_windows=2, review_prob=0.25,
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

        extractor_tag = "mlp"  # default for non-MultiInputPolicy
        if policy == "MultiInputPolicy":
            if policy_kwargs is None:
                policy_kwargs = {}

            # ── Architecture selection ────────────────────────────────────────
            # Priority order:
            #   1. +architecture=<tag>  CLI/menu override
            #   2. use_light_extractor / use_full_peak flags in the algo YAML
            #   3. Hardcoded default: slim
            #
            # Architectures:
            #   light    → LightCombinedExtractor   ~18K params   no channel split, fast
            #   slim     → SlimPEAKExtractor         ~77K params   channel split, no SEBlock
            #   balanced → BalancedPEAKExtractor     ~230K params  channel split + SEBlock
            #   peak     → PEAKExtractor             ~922K params  full deep architecture
            arch_override = str(cfg.get("architecture", "") or "").strip().lower()

            if arch_override == "light":
                use_light = True
                use_peak  = False
                use_balanced = False
            elif arch_override == "balanced":
                use_light = False
                use_peak  = False
                use_balanced = True
            elif arch_override == "peak":
                use_light = False
                use_peak  = True
                use_balanced = False
            elif arch_override == "slim":
                use_light = False
                use_peak  = False
                use_balanced = False
            else:
                # Fall back to YAML flags; default to slim if neither flag is set
                use_light = bool(algo_conf.get("use_light_extractor", False))
                use_peak  = bool(algo_conf.get("use_full_peak",       False))
                use_balanced = False

            if use_light:
                policy_kwargs["features_extractor_class"] = LightCombinedExtractor
                extractor_tag = "light"
                print("[INFO] Using LightCombinedExtractor (~18K params, fast sweep mode).")
            elif use_balanced:
                policy_kwargs["features_extractor_class"] = BalancedPEAKExtractor
                policy_kwargs.setdefault("features_extractor_kwargs", {"features_dim": 192})
                extractor_tag = "balanced"
                print("[INFO] Using BalancedPEAKExtractor (~230K params, SEBlock + channel split).")
            elif use_peak:
                policy_kwargs["features_extractor_class"] = PEAKExtractor
                policy_kwargs.setdefault("features_extractor_kwargs", {"features_dim": 256})
                extractor_tag = "peak"
                print("[INFO] Using PEAKExtractor (~922K params, full architecture).")
            else:
                policy_kwargs["features_extractor_class"] = SlimPEAKExtractor
                policy_kwargs.setdefault("features_extractor_kwargs", {"features_dim": 128})
                extractor_tag = "slim"
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

            env = VecNormalize(raw_env, norm_obs=True, norm_reward=False, clip_obs=10.0,
                               norm_obs_keys=["scalars"])

            def make_monitored_env(render_mode=None):
                """Factory that wraps the env with Monitor for proper eval logging."""
                kw = env_kwargs.copy()
                if render_mode is not None:
                    kw['render_mode'] = render_mode
                def _init():
                    return Monitor(GameEnv(game_cls, reward_fn=active_reward_fn, **kw))
                return _init

            eval_raw_env = DummyVecEnv([make_monitored_env()])
            eval_env = VecNormalize(eval_raw_env, norm_obs=True, norm_reward=False, clip_obs=10.0,
                                    norm_obs_keys=["scalars"])
            eval_env.training = False
            eval_env.norm_reward = False

            for skill, total_timesteps in selected_skills.items():
                run_count += 1
                tb_dir = os.path.join(tb_root, f"{game_name}_{model_name}_{persona}")
                os.makedirs(tb_dir, exist_ok=True)

                log_name = f"training_log_{run_id}.csv"
                csv_dir = repo_root / "csv"
                csv_dir.mkdir(parents=True, exist_ok=True)
                csv_logger = CsvLoggerCallback(log_dir=str(csv_dir), file_name=log_name)

                is_recurrent_model = (model_name.lower() in ['rppo', 'recurrent_ppo'])

                # Build a unique run ID that includes the extractor tag
                # Format: {game}_{algo}_{persona}_{skill}_{extractor}
                run_id = f"{game_name}_{model_name}_{persona}_{str(skill).lower()}_{extractor_tag}"

                if is_recurrent_model:
                    eval_cb = RecurrentEvalCallback(
                        eval_env,
                        best_model_save_path=str(models_dir / "best" / run_id),
                        log_path=str(models_dir / "eval_logs" / run_id),
                        eval_freq=eval_freq, n_eval_episodes=5,
                        deterministic=True, render=False, verbose=1,
                    )
                else:
                    eval_cb = EvalCallback(
                        eval_env,
                        best_model_save_path=str(models_dir / "best" / run_id),
                        log_path=str(models_dir / "eval_logs" / run_id),
                        eval_freq=eval_freq,
                        deterministic=True, render=False,
                    )

                ckpt_cb = CheckpointCallback(
                    save_freq=save_freq,
                    save_path=str(models_dir / "checkpoints"),
                    name_prefix=run_id,
                )

                # Wrap eval_cb: saves vecnorm + opens pygame preview on each new best
                _best_path = models_dir / "best" / run_id
                eval_cb = EvalPreviewCallback(
                    eval_cb              = eval_cb,
                    vecnorm_env          = env,
                    best_model_save_path = _best_path,
                    make_env_fn          = make_env,
                    repo_root            = repo_root,
                    fps                  = viz_fps,
                    n_preview_episodes   = viz_preview_episodes if viz_enabled else 0,
                    verbose              = 1,
                )
                if viz_enabled:
                    print("[INFO] Preview ON — window opens on each new best.")

                current_callbacks = [eval_cb, ckpt_cb, csv_logger]

                train_kwargs = dict(algo_kwargs)
                train_kwargs["tensorboard_log"] = tb_dir
                train_kwargs["device"] = device

                model = Algo(policy, env, **train_kwargs)
                tb_run_name = f"{model_name}_{persona}_{str(skill).lower()}_{extractor_tag}"

                model.learn(
                    total_timesteps=int(total_timesteps),
                    callback=current_callbacks,
                    tb_log_name=tb_run_name,
                    progress_bar=True,
                )

                filename = f"{run_id}.zip"
                save_path = models_dir / filename
                model.save(save_path)

                norm_path = models_dir / f"{run_id}_vecnorm.pkl"
                env.save(str(norm_path))

                # Write model_info.json alongside best_model.zip so metadata
                # is readable without parsing the folder name
                import json as _json, datetime as _dt
                _best_path.mkdir(parents=True, exist_ok=True)
                (_best_path / "model_info.json").write_text(_json.dumps({
                    "game":      game_name,
                    "algo":      model_name,
                    "persona":   persona,
                    "skill":     str(skill).lower(),
                    "extractor": extractor_tag,
                    "trained":   _dt.datetime.now().isoformat(timespec="seconds"),
                    "timesteps": int(total_timesteps),
                }, indent=2))

                print(f"[{run_count}] saved → {save_path}  ({_pretty_steps(int(total_timesteps))} steps)")
                print(f"       VecNorm → {norm_path}  (required for watch_agent.py)")
                print(f"       Extractor tag: [{extractor_tag}]")

            try:
                env.close()
                eval_env.close()
            except Exception:
                pass

    print(f"Done. Trained {run_count} models for game='{game_name}'. Models at: {models_dir}")


if __name__ == "__main__":
    main()