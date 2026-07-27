"""Feature extractors for PEAK — moved out of train.py (behaviour unchanged).

Original multi-branch extractors (DeepChannelAttention / ChannelAttention /
SpatialAttention / LightMobile / FlatVector) + helper blocks, plus a modern
residual extractor (ImpalaSimbaExtractor, appended below). Architecture
selection still lives in train.py.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor




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
# FlatVectorExtractor — MLP for non-Dict (flat) observation spaces
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


# =========================================================================
# ImpalaSimbaExtractor — modern residual extractor (recommended)
# =========================================================================
# Design (grounded in 2024-2025 DRL literature):
#   • Grids  → IMPALA-style residual conv tower (Espeholt et al. 2018;
#     residual blocks beat the flat Nature-CNN on Atari/Procgen), finished
#     with Impoola GLOBAL AVERAGE POOLING (Trumpp et al. 2025) instead of a
#     large flatten+Linear — better generalization with far fewer params.
#     NO hardcoded channel split → works for any channel count (frame-stack
#     safe) and doesn't bake in a Dijkstra-specific prior.
#   • Scalars → SimBa pre-LayerNorm residual MLP (Lee et al., ICLR 2025):
#     LayerNorm + a linear skip pathway is the single biggest lever the paper
#     found for scaling parameters in RL without overfitting. Observation
#     normalization (SimBa's 3rd component) is already handled upstream by
#     VecNormalize(norm_obs_keys=['scalars']), so it's not duplicated here.
#   • GroupNorm in conv (RL-correct: no batch-dim dependency), LayerNorm in
#     the MLP/fusion. AdaptiveAvgPool → grid-size invariant.
def _gn_groups(c: int) -> int:
    """Largest group count <= 8 that divides c (GroupNorm requires divisibility)."""
    for g in (8, 4, 2, 1):
        if c % g == 0:
            return g
    return 1


class _ImpalaResBlock(nn.Module):
    """Pre-activation residual block: x + conv(relu(gn(conv(relu(gn(x))))))."""
    def __init__(self, c: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.GroupNorm(_gn_groups(c), c), nn.ReLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(_gn_groups(c), c), nn.ReLU(),
            nn.Conv2d(c, c, 3, padding=1),
        )

    def forward(self, x):
        return x + self.body(x)


class _ImpalaStage(nn.Module):
    """Conv → downsample (maxpool 3x3 s2) → one residual block. (IMPALA sequence.)"""
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res = _ImpalaResBlock(cout)

    def forward(self, x):
        return self.res(self.pool(self.conv(x)))


class _SimBaBlock(nn.Module):
    """SimBa residual MLP block: x + Linear(relu(Linear(LayerNorm(x))))."""
    def __init__(self, dim: int, hidden_mult: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * hidden_mult), nn.ReLU(),
            nn.Linear(dim * hidden_mult, dim),
        )

    def forward(self, x):
        return x + self.mlp(self.norm(x))


class ImpalaSimbaExtractor(BaseFeaturesExtractor):
    """IMPALA conv tower (+ Impoola global avg pool) for grids, SimBa residual
    MLP for scalars, fused with a post-LayerNorm head.

    Channel-count-agnostic (no ch0-1 / ch2-3 split) and grid-size invariant.
    Defaults are modest (~fits this 4x21x21 obs); widen via the ctor args.
    """
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256,
                 grid_channels=(32, 64), scalar_dim: int = 128,
                 scalar_blocks: int = 2, scalar_hidden_mult: int = 2):
        super().__init__(observation_space, features_dim=features_dim)
        gshape = observation_space["grids"].shape          # (C, H, W)
        n_ch = gshape[0]
        n_scalars = observation_space["scalars"].shape[0]

        # --- grids: IMPALA tower + Impoola global average pool ---
        stages = []
        cin = n_ch
        for cout in grid_channels:
            stages.append(_ImpalaStage(cin, cout))
            cin = cout
        self.grid_tower = nn.Sequential(*stages, nn.ReLU())
        self.grid_pool = nn.AdaptiveAvgPool2d((1, 1))       # -> (B, cin, 1, 1)
        grid_feat = cin

        # --- scalars: SimBa embedding + residual blocks + final LayerNorm ---
        self.scalar_embed = nn.Linear(n_scalars, scalar_dim)
        self.scalar_body = nn.Sequential(
            *[_SimBaBlock(scalar_dim, scalar_hidden_mult) for _ in range(scalar_blocks)]
        )
        self.scalar_norm = nn.LayerNorm(scalar_dim)

        # --- fusion (post-LayerNorm head) ---
        self.fusion = nn.Sequential(
            nn.Linear(grid_feat + scalar_dim, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
        )
        self._features_dim = features_dim

    def forward(self, observations: dict) -> torch.Tensor:
        g = self.grid_pool(self.grid_tower(observations["grids"])).flatten(1)   # (B, grid_feat)
        s = self.scalar_norm(self.scalar_body(self.scalar_embed(observations["scalars"])))
        return self.fusion(torch.cat([g, s], dim=1))
