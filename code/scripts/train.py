import os
from code.callbacks.logging_callback import CsvLoggerCallback
from code.callbacks.profiling_callback import ProfilingCallback, EvalTimerCallback
from code.callbacks.RecurrentEvalCallback import RecurrentEvalCallback

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


def _resolve_device(spec, *, verbose: bool = True) -> str:
    """Resolve a device spec string to a concrete torch device.

    Accepts: "auto", "cpu", "cuda", "mps".  Anything else (incl. None)
    falls back to "cpu" with a warning.

    When the resolved device is "mps", sets PYTORCH_ENABLE_MPS_FALLBACK=1
    so ops without MPS kernels fall back to CPU instead of crashing.
    """
    spec_l = str(spec or "").strip().lower() or "cpu"
    cuda_ok = torch.cuda.is_available()
    mps_ok = (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )

    if spec_l == "auto":
        if cuda_ok:
            resolved, why = "cuda", "CUDA detected"
        elif mps_ok:
            resolved, why = "mps", "Apple Silicon detected"
        else:
            resolved, why = "cpu", "no GPU detected"
    elif spec_l == "cuda":
        resolved, why = ("cuda", "") if cuda_ok else ("cpu", "CUDA not available")
    elif spec_l == "mps":
        resolved, why = ("mps", "Apple Silicon detected") if mps_ok else ("cpu", "MPS not available")
    elif spec_l == "cpu":
        resolved, why = "cpu", ""
    else:
        resolved, why = "cpu", f"unknown device spec '{spec}'"

    if resolved == "mps":
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        suffix = f" ({why}, MPS fallback enabled)" if why else " (MPS fallback enabled)"
    else:
        suffix = f" ({why})" if why else ""

    if verbose:
        print(f"[INFO] Device requested: '{spec}' → resolved: '{resolved}'{suffix}")

    return resolved


ARCH_ALIASES = {
    "lightmobile": "lightmobile",
    "spatialattention": "spatialattention",
    "channelattention": "channelattention",
    "deepchannelattention": "deepchannelattention",
    "mlp": "mlp",
}


def _canonical_arch_tag(raw: str) -> str:
    return ARCH_ALIASES.get(str(raw or "").strip().lower(), "")


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

    Used in DeepChannelAttentionExtractor's semantic branch to let the network suppress
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
# DeepChannelAttentionExtractor — primary multi-branch extractor (default)
# =========================================================================
class DeepChannelAttentionExtractor(BaseFeaturesExtractor):
    """
    Multi-branch feature extractor for the PEAK platformer environment.

    Observation space
    -----------------
      grids  : Box(-1, 1, shape=(4, H, W))
                 ch 0  — Solids        binary {0, 1}  (ground, platforms, qblocks)
                 ch 1  — Collectibles  value-encoded  {0.0, 0.35, 0.69, 1.0}
                                    coin=0.35 / powerup=0.69 / goal=1.0
                 ch 2  — Hazards    sign-encoded   {-1.0, 0.0, +1.0}
                                    spike=-1.0 (always lethal) / enemy=+1.0 (defeatable)
            ch 3  — Dijkstra      continuous [-1, 1]  (relative advantage map)
      scalars: Box(-inf, inf, shape=(20,))
            Player=13 (from obs_vector) + Tracking=7 (from _tracking_obs)

    Architecture
    ------------
    Branch A  Semantic CNN  (ch 0-2, 3×H×W)
            Conv(3→32, k=3) → GroupNorm(8,32) → ReLU → SEBlock(32)
            Conv(32→64, k=3) → GroupNorm(16,64) → ReLU
            Conv(64→64, k=3) → GroupNorm(16,64) → ReLU
            AdaptiveAvgPool(4×4) → Flatten(1024) → Linear(1024→256) → LayerNorm → ReLU

    Branch B  Dijkstra CNN  (ch 3, 1×H×W)
            Conv(1→8,  k=3) → GroupNorm(4,8)  → ReLU
            AvgPool2d(3)                          [21×21 → 7×7, preserves gradient]
            Conv(8→16, k=3) → GroupNorm(4,16) → ReLU
            AdaptiveAvgPool(3×3) → Flatten(144) → Linear(144→64) → LayerNorm → ReLU

            Why AvgPool over MaxPool for Dijkstra:
            The Dijkstra channel is a smooth continuous gradient [-1, 1].
            MaxPool would select the peak value in each 2×2 window, artificially
            dilating the path and destroying the slope. AvgPool acts as a gentle
            blur, preserving the physical shape of the distance gradient so the
            second conv can learn directional patterns ("gradient pointing right")
            before the final adaptive pool collapses to a compact summary.

    Branch C  Scalar MLP   (20,)
            Linear(20→64) → LayerNorm → ReLU
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
    channel when no coins are visible, hazard channel when no enemies nearby).
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim=features_dim)

        grid_shape   = observation_space["grids"].shape    # (4, H, W)
        scalar_shape = observation_space["scalars"].shape  # (20,)

        n_scalars = scalar_shape[0]  # 20 — read dynamically, not hardcoded
        H, W      = grid_shape[1], grid_shape[2]

        # Channel split:
        #   ch 0-1 → Semantic (Solids + Collectibles)  — navigation
        #   ch 2-3 → Jump     (Hazards + Dijkstra)     — timing + path
        N_SEM = 2   # ch 0-1
        N_JMP = 2   # ch 2-3

        # ── Branch A: Semantic CNN (ch 0-1, navigation) ─────────────────────
        self.semantic_cnn = nn.Sequential(
            nn.Conv2d(N_SEM, 32, kernel_size=3, stride=1, padding=1),
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
            sem_flat = self.semantic_cnn(torch.zeros(1, N_SEM, H, W)).shape[1]

        self.semantic_fc = nn.Sequential(
            nn.Linear(sem_flat, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )

        # ── Branch B: Jump CNN (ch 2-3, Hazards + Dijkstra) ─────────────────
        # Asymmetric kernels encode jump-timing geometry:
        #   5×1 vertical  — spans full jump height, fires on hazard columns
        #   1×5 horizontal — sweeps timing window (safe landing strips)
        # AvgPool(3) on Dijkstra preserves smooth gradient slope.
        self.jump_cnn = nn.Sequential(
            nn.Conv2d(N_JMP, 16, kernel_size=(5, 1), stride=1, padding=(2, 0)),  # vertical scan
            nn.GroupNorm(4, 16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=(1, 5), stride=1, padding=(0, 2)),     # horizontal timing
            nn.GroupNorm(4, 16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten()
        )
        with torch.no_grad():
            jmp_flat = self.jump_cnn(torch.zeros(1, N_JMP, H, W)).shape[1]

        self.jump_fc = nn.Sequential(
            nn.Linear(jmp_flat, 64),
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
        grids   = observations["grids"]    # (B, 4, H, W)
        scalars = observations["scalars"]  # (B, 20)

        # ch 0-1: navigation (solids + collectibles)
        sem  = self.semantic_fc(self.semantic_cnn(grids[:, :2, :, :]))    # (B, 256)
        # ch 2-3: jump timing (hazards + dijkstra) — asymmetric kernels
        jmp  = self.jump_fc(self.jump_cnn(grids[:, 2:, :, :]))            # (B,  64)
        scal = self.scalar_mlp(scalars)                                    # (B,  64)

        return self.fusion(torch.cat([sem, jmp, scal], dim=1))             # (B, 256)


# =========================================================================
# SpatialSelfAttention — lightweight spatial attention block
# =========================================================================
class SpatialSelfAttention(nn.Module):
    """
    Applies multi-head self-attention across spatial positions of a feature map.

    Input:  (B, C, H, W)
    Output: (B, C, H, W)  — same shape, residual connection included

    Steps:
      1. Flatten spatial dims → (B, H*W, C)  [each pixel = one token]
      2. LayerNorm → MultiheadAttention       [tokens attend to each other]
      3. Residual add                         [preserves CNN features]
      4. Reshape back → (B, C, H, W)

    Kept lightweight intentionally:
      - num_heads=2, operates on C=16 after first conv+pool
      - No FF sublayer (that's the job of the next Conv2d)
      - batch_first=True requires PyTorch >= 1.9
    """
    def __init__(self, channels: int, num_heads: int = 2):
        super().__init__()
        self.norm  = nn.LayerNorm(channels)
        self.attn  = nn.MultiheadAttention(
            embed_dim   = channels,
            num_heads   = num_heads,
            batch_first = True,
            bias        = True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # (B, C, H, W) → (B, H*W, C)
        tokens = x.permute(0, 2, 3, 1).reshape(B, H * W, C)

        normed = self.norm(tokens)
        attended, _ = self.attn(normed, normed, normed)

        # Residual — keeps the CNN's local features intact
        tokens = tokens + attended

        # (B, H*W, C) → (B, C, H, W)
        return tokens.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()


# =========================================================================
# SpatialAttentionExtractor — middle ground (default for CPU+CUDA mixed setups)
# =========================================================================
class SpatialAttentionExtractor(BaseFeaturesExtractor):
    """
    Trimmed version of DeepChannelAttentionExtractor that keeps the architecturally
    important channel split but eliminates the two biggest cost drivers:

      Removed:  SEBlock (global pool + 2× FC per forward pass)
      Removed:  Large Linear(2304→256) — replaced with AdaptiveAvgPool
                before flatten, cutting sem_flat from 2304 → 288

    Observation space
    -----------------
      grids  : Box(-1, 1, shape=(4, H, W))
                 ch 0  — Solids        binary {0, 1}  (ground, platforms, qblocks)
                 ch 1  — Collectibles  value-encoded  {0.0, 0.35, 0.69, 1.0}
                                       coin=0.35 / powerup=0.69 / goal=1.0
                 ch 2  — Hazards       sign-encoded   {-1.0, 0.0, +1.0}
                                       spike=-1.0 (always lethal) / enemy=+1.0 (defeatable)
                 ch 3  — Dijkstra      continuous [-1, 1]  ← separated branch
      scalars: Box(-inf, inf, shape=(20,))
               Player=13 (from obs_vector) + Tracking=7 (from _tracking_obs)

    Architecture
    ------------
    Branch A  Semantic CNN + Spatial Attention  (ch 0-2, 3×H×W)
              Conv(3→16, k=3, pad=1) → GroupNorm(4,16) → ReLU
              MaxPool(2)                          [21×21 → 10×10]
              SpatialSelfAttention(16, heads=2)  [tokens attend across 10×10]
              Conv(16→32, k=3, pad=1) → GroupNorm(8,32) → ReLU
              AdaptiveAvgPool(3×3)                [10×10 → 3×3]
              Flatten(288) → Linear(288→128) → LayerNorm → ReLU

    Branch B  Dijkstra CNN  (ch 3, 1×H×W)
              Conv(1→8,  k=3, pad=1) → GroupNorm(4,8)  → ReLU
              AvgPool2d(3)                          [21×21 → 7×7]
              Conv(8→16, k=3, pad=1) → GroupNorm(4,16) → ReLU
              AdaptiveAvgPool(3×3)                  [7×7 → 3×3]
              Flatten(144) → Linear(144→32) → LayerNorm → ReLU

              Why AvgPool2d (not MaxPool2d) for Dijkstra:
              The Dijkstra channel is a smooth continuous gradient [-1, 1].
              MaxPool selects the peak value in each patch, which artificially
              dilates the path and destroys the slope of the distance gradient.
              AvgPool blurs gently, preserving the spatial structure so the second
              conv can learn directional patterns ("gradient flows right") before
              AdaptiveAvgPool compresses to a compact 3×3 summary.

              The staged reduction (21→7→3) also avoids the 49× one-shot
              compression of the original single-pool design, giving both conv
              layers real spatial extent to learn from.

    Branch C  Scalar MLP   (20,)
              Linear(20→64) → LayerNorm → ReLU

    Fusion    Cat(128+32+64=224) → Linear(224→128) → LayerNorm → ReLU
              → features_dim = 128

    Parameter count:
      Branch A CNN:   ~5K
      Branch A Attn:  ~1K   (16*16*3 weights + proj ≈ 1.1K)
      Branch A FC:    ~37K
      Branch B:       ~5K   (two conv layers + fc)
      Scalar MLP:     ~1K
      Fusion:         ~29K
      Total:          ~78K

    Why attention after first conv+pool (not before, not after second pool):
      MaxPool2d(2) reduces 21×21 → 10×10 — still 100 spatial tokens with
      enough resolution to reason about relative positions. Attention at this
      stage lets distant tiles influence each other before the second conv and
      final pool destroy spatial extent. After AdaptiveAvgPool(3×3) it's too
      late — spatial info is already collapsed to 9 cells.

    Keeps what matters:
      ✓ Dijkstra channel isolated from binary semantic channels
      ✓ Dedicated scalar MLP branch
      ✓ Fusion layer before policy head
      ✓ Spatial self-attention on semantic branch
      ✓ Staged gradient-preserving reduction for Dijkstra (AvgPool → second conv)
      ✓ GroupNorm + LayerNorm throughout for RL stability
    Drops what's expensive:
      ✗ SEBlock
      ✗ Deep semantic conv stack (3 conv layers → 2)
      ✗ Oversized linear after flatten
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 128):
        super().__init__(observation_space, features_dim=features_dim)

        grid_shape   = observation_space["grids"].shape    # (4, H, W)
        scalar_shape = observation_space["scalars"].shape  # (20,)

        n_scalars = scalar_shape[0]  # 20 — read dynamically, not hardcoded
        H, W      = grid_shape[1], grid_shape[2]  # 21, 21

        # Channel split:
        #   ch 0-1 → Semantic (Solids + Collectibles) — navigation
        #   ch 2-3 → Jump     (Hazards + Dijkstra)    — timing + path
        N_SEM = 2
        N_JMP = 2

        # ── Branch A: Semantic CNN + Spatial Attention (ch 0-1) ─────────────
        self.sem_conv1 = nn.Sequential(
            nn.Conv2d(N_SEM, 16, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 21×21 → 10×10
        )

        # Attention operates on the 10×10 feature map (100 tokens of dim 16)
        self.sem_attn = SpatialSelfAttention(channels=16, num_heads=2)

        self.sem_conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3)),   # 10×10 → 3×3
            nn.Flatten()                    # 32×3×3 = 288
        )

        with torch.no_grad():
            _x       = torch.zeros(1, N_SEM, H, W)
            _x       = self.sem_conv1(_x)
            _x       = self.sem_attn(_x)
            sem_flat = self.sem_conv2(_x).shape[1]

        self.semantic_fc = nn.Sequential(
            nn.Linear(sem_flat, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )

        # ── Branch B: Jump CNN (ch 2-3, Hazards + Dijkstra) ─────────────────
        # Asymmetric kernels:
        #   5×1 vertical  — spans full jump height, detects hazard columns
        #   1×5 horizontal — sweeps timing window for safe landings
        self.jump_cnn = nn.Sequential(
            nn.Conv2d(N_JMP, 8, kernel_size=(5, 1), stride=1, padding=(2, 0)),
            nn.GroupNorm(4, 8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=(1, 5), stride=1, padding=(0, 2)),
            nn.GroupNorm(4, 16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Flatten()
        )

        with torch.no_grad():
            jmp_flat = self.jump_cnn(torch.zeros(1, N_JMP, H, W)).shape[1]

        self.jump_fc = nn.Sequential(
            nn.Linear(jmp_flat, 32),
            nn.LayerNorm(32),
            nn.ReLU()
        )

        # ── Branch C: Scalar MLP ─────────────────────────────────────────────
        self.scalar_mlp = nn.Sequential(
            nn.Linear(n_scalars, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )

        # ── Fusion ────────────────────────────────────────────────────────────
        self.fusion = nn.Sequential(
            nn.Linear(128 + 32 + 64, features_dim),   # 224 → 128
            nn.LayerNorm(features_dim),
            nn.ReLU()
        )

        self._features_dim = features_dim

    def forward(self, observations: dict) -> torch.Tensor:
        grids   = observations["grids"]    # (B, 4, H, W)
        scalars = observations["scalars"]  # (B, 20)

        # Branch A: ch 0-1 (navigation) — conv1 → attention → conv2 → fc
        x   = self.sem_conv1(grids[:, :2, :, :])    # (B, 16, 10, 10)
        x   = self.sem_attn(x)                       # (B, 16, 10, 10)
        sem = self.semantic_fc(self.sem_conv2(x))    # (B, 128)

        # Branch B: ch 2-3 (jump timing) — asymmetric 5×1 → 1×5 kernels
        jmp  = self.jump_fc(self.jump_cnn(grids[:, 2:, :, :]))    # (B, 32)

        # Branch C: scalars
        scal = self.scalar_mlp(scalars)                            # (B, 64)

        return self.fusion(torch.cat([sem, jmp, scal], dim=1))     # (B, 128)


# =========================================================================
# ChannelAttentionExtractor — sweet spot (~230K params)
# =========================================================================
class ChannelAttentionExtractor(BaseFeaturesExtractor):
    """
    Middle ground between SpatialAttention (77K) and full DeepChannelAttention (922K).

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

        n_scalars = scalar_shape[0]
        H, W      = grid_shape[1], grid_shape[2]

        # Channel split: ch 0-1 navigation | ch 2-3 jump timing
        N_SEM = 2
        N_JMP = 2

        # ── Branch A: Semantic CNN (ch 0-1) + SEBlock ────────────────────
        self.semantic_cnn = nn.Sequential(
            nn.Conv2d(N_SEM, 32, kernel_size=3, stride=1, padding=1),
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
            sem_flat = self.semantic_cnn(torch.zeros(1, N_SEM, H, W)).shape[1]

        self.semantic_fc = nn.Sequential(
            nn.Linear(sem_flat, 192),
            nn.LayerNorm(192),
            nn.ReLU()
        )

        # ── Branch B: Jump CNN (ch 2-3, Hazards + Dijkstra) ──────────────
        # Asymmetric kernels: 5×1 vertical hazard scan → 1×5 horizontal timing
        self.jump_cnn = nn.Sequential(
            nn.Conv2d(N_JMP, 16, kernel_size=(5, 1), stride=1, padding=(2, 0)),
            nn.GroupNorm(4, 16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(1, 5), stride=1, padding=(0, 2)),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Flatten()
        )
        with torch.no_grad():
            jmp_flat = self.jump_cnn(torch.zeros(1, N_JMP, H, W)).shape[1]

        self.jump_fc = nn.Sequential(
            nn.Linear(jmp_flat, 48),
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

        sem  = self.semantic_fc(self.semantic_cnn(grids[:, :2, :, :]))    # ch 0-1
        jmp  = self.jump_fc(self.jump_cnn(grids[:, 2:, :, :]))            # ch 2-3
        scal = self.scalar_mlp(scalars)

        return self.fusion(torch.cat([sem, jmp, scal], dim=1))


# =========================================================================
# LightMobileExtractor — fast sweep option (use_light_extractor: true)
# =========================================================================
class LightMobileExtractor(BaseFeaturesExtractor):
    """
    Lightweight MobileNet-style extractor for fast hyperparameter sweeps.
    Enable with use_light_extractor: true in your algo yaml.

    Architecture
    ------------
    Layer 1:  Conv(n_ch→16) + BatchNorm + ReLU → MaxPool(2)  [21×21 → 10×10]
    Layer 2:  Depthwise(16) + Pointwise(16→32) + BatchNorm   [10×10 → 10×10]
    Summary:  AdaptiveAvgPool(3×3) → Flatten(288)
    FC:       Linear(288→128) + LayerNorm + ReLU
    Scalars:  Linear(20→64) + LayerNorm + ReLU
    features_dim = 192  (128 grids + 64 scalars)

    NOTE: Does NOT split the Dijkstra channel — all 4 channels share the same
    conv stack (n_ch is read dynamically from the observation space).
    Use SpatialAttentionExtractor (default) or DeepChannelAttentionExtractor for full training runs
    where the Dijkstra channel split matters.
    ~18K params in grids branch, 3-5× faster CPU inference than DeepChannelAttentionExtractor.
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
                    nn.GroupNorm(4, 16),
                    nn.ReLU(),
                    nn.MaxPool2d(2),          # 21×21 → 10×10
                )

                # --- Layer 2: depthwise + pointwise (MobileNet-style) ---
                l2 = nn.Sequential(
                    nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, groups=16),
                    nn.Conv2d(16, 32, kernel_size=1),
                    nn.GroupNorm(8, 32),
                    nn.ReLU(),
                )

                # --- Compact spatial summary ---
                pool = nn.AdaptiveAvgPool2d((3, 3))   # 10×10 → 3×3

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
class FlatVectorExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim=features_dim)
        in_dim = int(np.prod(observation_space.shape))
        hidden = max(features_dim, 96)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
        )
        self._features_dim = features_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


class EvalPreviewCallback(BaseCallback):
    def __init__(self, eval_cb, vecnorm_env, best_model_save_path,
                 make_env_fn, repo_root, fps=30, n_preview_episodes=2, verbose=1):
        super().__init__(verbose)
        self.eval_cb              = eval_cb
        self._vecnorm_env         = vecnorm_env
        self.best_model_save_path = Path(best_model_save_path)
        self.make_env_fn          = make_env_fn
        self.repo_root            = Path(repo_root)
        self.fps                  = fps
        self.n_preview_episodes   = n_preview_episodes
        self._last_best           = float("-inf")

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

    def _save_vecnorm(self):
        out = self.best_model_save_path / "best_model_vecnorm.pkl"
        try:
            self.best_model_save_path.mkdir(parents=True, exist_ok=True)
            self._vecnorm_env.save(str(out))
            if self.verbose:
                print(f"[EvalPreview] Saved vecnorm -> {out}")
        except Exception as e:
            print(f"[EvalPreview] vecnorm save failed: {e}")

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
            proc = subprocess.Popen(
                cmd,
                cwd=str(self.repo_root),
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


# TRAIN clip_reward must exceed the win bonus (5.0) so the terminal win
# signal survives normalization-time clipping. 10.0 is the SB3 default and
# clears every persona win bonus (5.0 default, 7.0 megaman, 8.0 balanced_win).
_VECNORM_TRAIN_CLIP_REWARD = 10.0


def _build_vecnorm_kwargs(uses_dict_obs, obs_space, *, training):
    """Construct VecNormalize kwargs.

    training=True  -> norm_reward=True  (+ clip_reward=10.0 > win bonus 5.0)
    training=False -> norm_reward=False (interpretable eval rewards)
    """
    kwargs = dict(
        norm_obs=True,
        clip_obs=10.0,
        norm_reward=bool(training),
        clip_reward=_VECNORM_TRAIN_CLIP_REWARD,
    )
    if uses_dict_obs:
        norm_keys = [k for k, v in obs_space.spaces.items()
                     if isinstance(v, spaces.Box)]
        if norm_keys:
            kwargs["norm_obs_keys"] = norm_keys
    return kwargs


@hydra.main(version_base=None, config_path="../conf", config_name="grid")
def main(cfg: DictConfig):
    repo_root = Path(get_original_cwd())
    conf_root = repo_root / "code" / "conf"
    # out_root: optional override of the model output root so concurrent runs
    # (e.g. a multi-seed experiment matrix) can write to isolated directories.
    # Defaults to repo_root/"models" (unchanged behaviour when unset).
    _out_root = cfg.get("out_root", None)
    models_dir = Path(_out_root) if _out_root else repo_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(cfg.get("device", "auto"))

    profile_enabled = bool(cfg.get("profile", False))
    if profile_enabled:
        print("[INFO] Profiling ON — per-rollout breakdown will be written to profiling_log.csv")

    tb_root   = str(cfg.get("tb_root", "runs"))
    eval_freq = int(cfg.get("eval_freq", 20_000))
    save_freq = int(cfg.get("save_freq", 50_000))

    viz_enabled          = bool(cfg.get("viz_enabled", False))
    viz_freq             = int(cfg.get("viz_freq", 50_000))
    viz_preview_episodes = int(cfg.get("viz_preview_episodes", 1))
    viz_fps              = int(cfg.get("viz_fps", 30))

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

    try:
        reward_module = importlib.import_module(f"code.rewards.train_{game_name}")
        print(f"[INFO] Loaded reward module: code.rewards.train_{game_name}")
    except Exception as e:
        print(f"[WARNING] Could not load code.rewards.train_{game_name} ({e}). Falling back to train_platformer.")
        reward_module = importlib.import_module("code.rewards.train_platformer")

    base_env_kwargs = dict(
        render_mode=cfg.render_mode,
        fps=None if str(cfg.fps).lower() == "none" else int(cfg.fps),
        max_steps=None if str(cfg.max_steps).lower() == "none" else int(cfg.max_steps),
        batch_window=10,
        advance_threshold=0.30,
        fallback_threshold=0.20,
        max_stay_windows=3,
        curriculum_advance_step=2,   # levels to skip forward on mastery
        curriculum_fallback_step=2,  # levels to drop back on failure
        # Dijkstra ablation: +dijkstra_enabled=false zeros the Dijkstra obs channel
        # for both train and eval (flows through env_kwargs.copy()).
        dijkstra_enabled=bool(cfg.get("dijkstra_enabled", True)),
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

    probe_persona = selected_personas[0] if selected_personas else "simple"
    probe_env = GameEnv(game_cls, reward_fn=None, persona=probe_persona, arch_tag="mlp", **base_env_kwargs)
    obs_space = probe_env.observation_space
    probe_env.close()
    uses_dict_obs = isinstance(obs_space, spaces.Dict)

    train_vecnorm_kwargs = _build_vecnorm_kwargs(uses_dict_obs, obs_space, training=True)
    eval_vecnorm_kwargs  = _build_vecnorm_kwargs(uses_dict_obs, obs_space, training=False)

    run_count = 0
    for model_name in selected_models:
        algo_conf = _load_yaml(conf_root, "algo", model_name)
        Algo = get_algo(algo_conf.get("name", model_name))
        policy = algo_conf.get("policy", "MlpPolicy")

        policy_kwargs = algo_conf.get("policy_kwargs", None)
        algo_kwargs   = {k: v for k, v in algo_conf.items()
                         if k not in {"_target_", "name", "policy", "policy_kwargs"}}

        extractor_tag = "mlp"  # default for non-MultiInputPolicy
        arch_override = _canonical_arch_tag(cfg.get("architecture", ""))
        if policy == "MultiInputPolicy" and not uses_dict_obs:
            policy = "MlpPolicy"
            if policy_kwargs is None:
                policy_kwargs = {}

            flat_dim = 128
            if arch_override == "lightmobile":
                flat_dim = 64
            elif arch_override == "channelattention":
                flat_dim = 192
            elif arch_override == "deepchannelattention":
                flat_dim = 256

            policy_kwargs["features_extractor_class"] = FlatVectorExtractor
            policy_kwargs.setdefault("features_extractor_kwargs", {"features_dim": flat_dim})
            print(f"[INFO] {game_name} uses flat observations â€” switching {model_name.upper()} to MlpPolicy.")
        elif policy == "MultiInputPolicy":
            if policy_kwargs is None:
                policy_kwargs = {}

            # ── Architecture selection ────────────────────────────────────────
            # ALL architectures use asymmetric jump kernels (5×1 → 1×5) on
            # ch 2-3 (Hazards + Dijkstra). The tag selects capacity level only.
            #
            # Priority:
            #   1. +architecture=<tag>  CLI/menu override
            #   2. use_light_extractor / use_full_peak flags in the algo YAML
            #   3. Default: spatialattention
            #
            # Tags:
            #   lightmobile         → LightMobileExtractor         ~18K params
            #   spatialattention    → SpatialAttentionExtractor    ~77K params
            #   channelattention    → ChannelAttentionExtractor   ~230K params
            #   deepchannelattention→ DeepChannelAttentionExtractor ~922K params
            if arch_override == "lightmobile":
                use_light = True
                use_peak  = False
                use_balanced = False
            elif arch_override == "channelattention":
                use_light = False
                use_peak  = False
                use_balanced = True
            elif arch_override == "deepchannelattention":
                use_light = False
                use_peak  = True
                use_balanced = False
            elif arch_override == "spatialattention":
                use_light = False
                use_peak  = False
                use_balanced = False
            else:
                # Fall back to YAML flags; default to SpatialAttention if neither flag is set
                use_light = bool(algo_conf.get("use_light_extractor", False))
                use_peak  = bool(algo_conf.get("use_full_peak",       False))
                use_balanced = False

            if use_light:
                policy_kwargs["features_extractor_class"] = LightMobileExtractor
                extractor_tag = "lightmobile"
                print("[INFO] Using LightMobileExtractor (~18K params, fast sweep mode).")
            elif use_balanced:
                policy_kwargs["features_extractor_class"] = ChannelAttentionExtractor
                policy_kwargs.setdefault("features_extractor_kwargs", {"features_dim": 192})
                extractor_tag = "channelattention"
                print("[INFO] Using ChannelAttentionExtractor (~230K params, SEBlock + jump CNN ch2-3).")
            elif use_peak:
                policy_kwargs["features_extractor_class"] = DeepChannelAttentionExtractor
                policy_kwargs.setdefault("features_extractor_kwargs", {"features_dim": 256})
                extractor_tag = "deepchannelattention"
                print("[INFO] Using DeepChannelAttentionExtractor (~922K params, deep + jump CNN ch2-3).")
            else:
                policy_kwargs["features_extractor_class"] = SpatialAttentionExtractor
                policy_kwargs.setdefault("features_extractor_kwargs", {"features_dim": 128})
                extractor_tag = "spatialattention"
                print("[INFO] Using SpatialAttentionExtractor (~77K params, Spatial Attn + jump CNN ch2-3).")

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
            env_kwargs['persona']  = persona
            env_kwargs['arch_tag'] = extractor_tag   # for debug overlay display

            active_reward_fn = None
            if hasattr(reward_module, persona):
                active_reward_fn = getattr(reward_module, persona)
                print(f"[INFO] Loaded reward persona: {persona}")
            else:
                print(f"[WARNING] Persona '{persona}' not found! Using default.")
                active_reward_fn = reward_module.default

            def make_env(render_mode=None, _fn=active_reward_fn, _kw=None):
                """
                Factory that returns an _init callable.
                render_mode overrides base_env_kwargs['render_mode'] so the
                visualisation callback can request rgb_array independently.
                FIX: _fn and _kw captured as default args to avoid late-binding
                closure bug — if the persona loop ever iterates, all envs share
                the last persona's reward_fn without this guard.
                """
                kw = (_kw if _kw is not None else env_kwargs).copy()
                if render_mode is not None:
                    kw['render_mode'] = render_mode
                def _init():
                    return GameEnv(game_cls, reward_fn=_fn, **kw)
                return _init

            n_envs = int(cfg.get("n_envs", 1))
            if n_envs > 1:
                raw_env = SubprocVecEnv([make_env() for _ in range(n_envs)])
            else:
                raw_env = DummyVecEnv([make_env()])

            env = VecNormalize(raw_env, **train_vecnorm_kwargs)

            # Eval env is pinned to ONE level with the curriculum OFF and goal
            # made terminal, so best_model scores reflect single-level skill and
            # cannot drift via curriculum state. See PlatformerCore eval flags.
            _eval_level = cfg.get("eval_level") or (
                env_kwargs.get("world") or None
            )
            def make_monitored_env(render_mode=None, _level=_eval_level):
                """Factory that wraps the env with Monitor for proper eval logging."""
                kw = env_kwargs.copy()
                if render_mode is not None:
                    kw['render_mode'] = render_mode
                kw["curriculum_enabled"] = False
                kw["terminate_on_goal"] = True
                if _level is not None:
                    kw["world"] = _level
                def _init():
                    return Monitor(GameEnv(game_cls, reward_fn=active_reward_fn, **kw))
                return _init

            for skill, total_timesteps in selected_skills.items():
                run_count += 1

                # Fresh eval env per run — no state leaks across runs.
                eval_raw_env = DummyVecEnv([make_monitored_env()])
                eval_env = VecNormalize(eval_raw_env, **eval_vecnorm_kwargs)
                # FIX: sync running obs statistics from the training wrapper so the
                # eval agent sees identical normalised observations. Without this,
                # eval_env accumulates its own obs_rms from scratch and best_model
                # scores are measured against a different normalisation than training.
                eval_env.obs_rms     = env.obs_rms   # share TRAIN obs normalisation
                eval_env.ret_rms     = env.ret_rms
                eval_env.training    = False
                eval_env.norm_reward = False         # keep eval rewards un-normalised
                tb_dir = os.path.join(tb_root, f"{game_name}_{model_name}_{persona}")
                os.makedirs(tb_dir, exist_ok=True)

                is_recurrent_model = (model_name.lower() in ['rppo', 'recurrent_ppo'])

                # Build a unique run ID that includes the extractor tag
                # Format: {game}_{algo}_{persona}_{skill}_{extractor}
                run_id = f"{game_name}_{model_name}_{persona}_{str(skill).lower()}_{extractor_tag}"

                log_name = f"training_log_{run_id}.csv"
                # Isolate per-step CSV under out_root when set (so concurrent
                # multi-seed runs don't race on the same csv); default unchanged.
                csv_dir = (Path(_out_root) / "csv") if _out_root else (repo_root / "csv")
                csv_dir.mkdir(parents=True, exist_ok=True)
                csv_logger = CsvLoggerCallback(log_dir=str(csv_dir), file_name=log_name)

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

                if profile_enabled:
                    # Profile CSV + TB go alongside the existing CSV logger output.
                    profiler = ProfilingCallback(
                        log_dir=str(csv_dir / run_id),
                        device=device,
                        verbose=1,
                    )
                    # Wrap eval_cb so its wall time is reported back to profiler.
                    eval_cb_for_run = EvalTimerCallback(inner_cb=eval_cb, profiler=profiler)
                    current_callbacks = [eval_cb_for_run, ckpt_cb, csv_logger, profiler]
                else:
                    current_callbacks = [eval_cb, ckpt_cb, csv_logger]

                train_kwargs = dict(algo_kwargs)
                train_kwargs["tensorboard_log"] = tb_dir
                train_kwargs["device"] = device

                model = Algo(policy, env, **train_kwargs)
                tb_run_name = f"{model_name}_{persona}_{str(skill).lower()}_{extractor_tag}"

                if bool(cfg.get("compile", False)):
                    try:
                        model.policy = torch.compile(model.policy)
                        print("[INFO] torch.compile enabled on policy (first rollout will warm up)")
                    except Exception as exc:
                        print(f"[WARN] torch.compile failed, continuing uncompiled: {type(exc).__name__}: {exc}")

                model.learn(
                    total_timesteps=int(total_timesteps),
                    callback=current_callbacks,
                    tb_log_name=tb_run_name,
                    progress_bar=True,
                )

                # Unwrap compiled policy before save so checkpoints load on any torch version.
                if hasattr(model.policy, "_orig_mod"):
                    model.policy = model.policy._orig_mod

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
