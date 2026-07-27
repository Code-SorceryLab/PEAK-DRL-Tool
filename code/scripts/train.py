import os
from code.callbacks.logging_callback import CsvLoggerCallback
from code.callbacks.AnnealCallback import AnnealCallback
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
from stable_baselines3.common.utils import set_random_seed
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
    "impala": "impala",
    "impalasimba": "impala",
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


from code.algos.extractors import (
    DeepChannelAttentionExtractor,
    SpatialAttentionExtractor,
    ChannelAttentionExtractor,
    LightMobileExtractor,
    FlatVectorExtractor,
    ImpalaSimbaExtractor,
)


class WinRateEvalCallback(EvalCallback):
    """EvalCallback that selects best_model by WIN RATE, not mean reward.

    The stock EvalCallback ranks checkpoints by mean episode reward. Under
    these personas a long stalling episode (alive + potential income over
    thousands of steps) can out-earn a fast WIN — the historically shipped
    best_model.zip was exactly such an artifact: a 480k checkpoint selected
    on the strength of a 4,486-step STALL episode. Win rate is the metric we
    actually optimize for, so it must be the selection key; mean reward only
    breaks ties within an equal win rate.

    Win rate comes from SB3's own success-rate convention: GameEnv emits
    info["is_success"] on terminal steps, evaluate_policy fills
    self._is_success_buffer via _log_success_callback. If the buffer is ever
    empty (env without is_success), we fall back to stock reward ranking
    rather than silently never saving a best model.

    `best_marker` increments on every save — EvalPreviewCallback watches it
    (instead of best_mean_reward) so the vecnorm snapshot + preview stay in
    sync with win-based saves even when the new best has a LOWER reward.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_win_rate  = -np.inf
        self.best_win_reward = -np.inf   # tie-break within equal win rate
        self.best_marker    = 0          # bumps on every best-model save

    def _on_step(self) -> bool:
        is_eval_step = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0
        if not is_eval_step:
            return super()._on_step()

        # Run the stock eval (evaluate_policy, npz logging, success buffer)
        # with the parent's reward-based save suppressed: the comparison
        # `mean_reward > best_mean_reward` is made unwinnable for the call.
        saved_best = self.best_mean_reward
        self.best_mean_reward = float("inf")
        try:
            continue_training = super()._on_step()
        finally:
            self.best_mean_reward = saved_best

        mean_reward = float(self.last_mean_reward)
        win_rate = (float(np.mean(self._is_success_buffer))
                    if len(self._is_success_buffer) > 0 else None)

        if win_rate is not None:
            new_best = (win_rate, mean_reward) > (self.best_win_rate, self.best_win_reward)
        else:
            new_best = mean_reward > saved_best   # fallback: stock behaviour

        if new_best:
            if win_rate is not None:
                self.best_win_rate, self.best_win_reward = win_rate, mean_reward
                if self.verbose >= 1:
                    print(f"New best by WIN RATE: {win_rate:.0%} "
                          f"(mean reward {mean_reward:.2f})")
            self.best_mean_reward = max(saved_best, mean_reward)
            if self.best_model_save_path is not None:
                self.model.save(os.path.join(self.best_model_save_path, "best_model"))
            self.best_marker += 1
            if self.callback_on_new_best is not None:
                continue_training = self.callback_on_new_best.on_step() and continue_training

        return continue_training


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
        # WinRateEvalCallback bumps best_marker on every win-based save (a
        # new best can have a LOWER mean reward, so best_mean_reward alone
        # would miss it and the vecnorm snapshot would go stale). Recurrent /
        # stock callbacks lack the marker — fall back to best_mean_reward.
        current_best = getattr(self.eval_cb, "best_marker", None)
        if current_best is not None:
            # Marker baseline is 0 ("no saves yet"), not -inf — without this
            # the wrapper fires once spuriously on the very first step.
            if self._last_best == float("-inf"):
                self._last_best = 0
        else:
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


def _resolve_run_seed(cfg):
    """Read the run seed from the Hydra config (`seed=` override; grid.yaml
    default). Returns int, or None when unset/none — unseeded runs stay
    possible by passing seed=none.

    HISTORICAL BUG: run_paper_matrix.py passed `seed={N}` for its 3-seed
    matrix, but train.py never consumed it — no set_random_seed, no seed
    kwarg to the algo — so all "seeds" ran identical configs and any
    per-seed CI computed over them was fabricated variance. The seed must
    be plumbed (a) globally via set_random_seed and (b) into the SB3 algo
    constructor, which re-seeds torch/numpy/random, the action space, AND
    the VecEnv at _setup_learn time.
    """
    try:
        raw = cfg.get("seed", None)
    except AttributeError:
        raw = None
    if raw is None or str(raw).strip().lower() in ("none", "null", ""):
        return None
    return int(raw)


def _build_vecnorm_kwargs(uses_dict_obs, obs_space, *, training):
    """Construct VecNormalize kwargs.

    Reward is NOT normalized (norm_reward=False) for train OR eval — reverted to
    main. Normalizing the training reward buried the dense movement signal (which
    is the agent's only rightward-navigation cue) and degraded the policy.
    The `training` arg is kept for call-site compatibility but no longer changes
    norm_reward.
    """
    kwargs = dict(
        norm_obs=True,
        clip_obs=10.0,
        norm_reward=False,
        clip_reward=_VECNORM_TRAIN_CLIP_REWARD,
    )
    if uses_dict_obs:
        # Normalise only vector sub-spaces (e.g. "scalars"). Image-like grids
        # (3D Box) are already bounded/semantically encoded in [-1,1]; running a
        # per-element running mean/std over them distorts the encoding and
        # couples channel stats to the level distribution — leave grids raw.
        # (Plan Item 5.)
        norm_keys = [k for k, v in obs_space.spaces.items()
                     if isinstance(v, spaces.Box) and len(v.shape) <= 2]
        if norm_keys:
            kwargs["norm_obs_keys"] = norm_keys
    return kwargs


def _linear_schedule(initial_value: float):
    """SB3 learning-rate schedule: linear decay from initial_value to 0 over
    training (progress_remaining goes 1.0 -> 0.0). (Plan Item 4.)"""
    def f(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return f


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

    # Seed the run BEFORE any env/model construction so network init,
    # action sampling, and env resets are all reproducible per seed.
    run_seed = _resolve_run_seed(cfg)
    if run_seed is not None:
        set_random_seed(run_seed)
        print(f"[INFO] Run seed = {run_seed} (python/numpy/torch + SB3 algo + VecEnv)")
    else:
        print("[WARN] No seed set — run is not reproducible (pass seed=<int>).")

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
        # Action repeat (frame_skip=4 = classic Mario recipe). Consumed by
        # GameEnv itself, NOT forwarded to the game core. Train and eval
        # envs share base_env_kwargs, so both always use the same skip.
        frame_skip=int(cfg.get("frame_skip", 1)),
        # Anti-stall progress metric (euclid legacy | path). Forwarded to the
        # game core; train and in-train-eval envs share it.
        stall_metric=str(cfg.get("stall_metric", "euclid")),
        # Episode time-horizon override in game-clock units (+time_limit=150).
        # None = keep each level's own time_limit from game_config.yaml.
        time_limit=cfg.get("time_limit", None),
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

        # ── Horizon / optimizer CLI overrides ────────────────────────────
        # gamma, n_steps, etc. live in conf/algo/<model>.yaml, which Hydra
        # CLI overrides can't reach. Allow per-run appends like:
        #   +gamma=0.997 +n_steps=4096 +batch_size=512
        # gamma is the TIME HORIZON knob (effective horizon ≈ 1/(1-gamma)
        # decisions); n_steps × n_envs is the EXPERIENCE BUFFER per update.
        _ALGO_OVERRIDABLE = ("gamma", "gae_lambda", "n_steps", "batch_size",
                             "n_epochs", "ent_coef", "vf_coef", "clip_range",
                             "learning_rate")
        for _k in _ALGO_OVERRIDABLE:
            _v = cfg.get(_k, None)
            if _v is not None:
                algo_kwargs[_k] = _v
                print(f"[INFO] algo override: {_k} = {_v}")

        # Keep the PBRS shaping discount equal to the agent's discount
        # (Ng et al. policy-invariance needs Φ discounted by the SAME γ).
        # Personas read this lazily per step, and SubprocVecEnv workers
        # inherit environ — so both in-process eval envs and spawned
        # workers stay in sync with any +gamma= override.
        os.environ["PEAK_POTENTIAL_GAMMA"] = str(algo_kwargs.get("gamma", 0.99))

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

            if arch_override == "impala":
                policy_kwargs["features_extractor_class"] = ImpalaSimbaExtractor
                policy_kwargs.setdefault("features_extractor_kwargs", {"features_dim": 256})
                extractor_tag = "impala"
                print("[INFO] Using ImpalaSimbaExtractor (IMPALA residual tower + SimBa scalar MLP).")
            elif use_light:
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
                # Force 'fork' for the env workers on Linux. The default
                # (forkserver/spawn) RE-IMPORTS this module in every worker, which
                # on Python 3.10 triggers a cv2/typing circular-import crash
                # (cv2.typing shadows stdlib typing during the re-import). 'fork'
                # inherits the already-imported parent and sidesteps it entirely.
                # We are on CPU with no live CUDA context, so fork is safe here.
                import multiprocessing as _mp, platform as _plat
                _start = ("fork" if (_plat.system() == "Linux"
                                     and "fork" in _mp.get_all_start_methods())
                          else None)
                raw_env = SubprocVecEnv(
                    [make_env() for _ in range(n_envs)], start_method=_start)
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
                    # Win-rate-based best-model selection (see class docstring:
                    # reward ranking shipped a stall artifact as "best").
                    eval_cb = WinRateEvalCallback(
                        eval_env,
                        best_model_save_path=str(models_dir / "best" / run_id),
                        log_path=str(models_dir / "eval_logs" / run_id),
                        eval_freq=eval_freq,
                        deterministic=True, render=False, verbose=1,
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

                # Schedules (Plan Item 4): linear LR decay + entropy/grad-clip
                # anneal (high->low exploration). Gated by grid.yaml `anneal`.
                # PPO/A2C/RecurrentPPO only (they expose ent_coef/max_grad_norm).
                if bool(cfg.get("anneal", True)) and not is_recurrent_model:
                    base_lr = float(train_kwargs.get("learning_rate", 3e-4))
                    train_kwargs["learning_rate"] = _linear_schedule(base_lr)
                    base_ent = float(train_kwargs.get("ent_coef", 0.01))
                    current_callbacks = current_callbacks + [AnnealCallback(
                        total_timesteps=int(total_timesteps),
                        start_ent=max(0.03, base_ent * 3.0), end_ent=base_ent,
                        start_grad_clip=1.0, end_grad_clip=0.5,
                    )]
                if run_seed is not None:
                    # SB3 re-seeds global RNGs, the action space, and the
                    # VecEnv (env.seed → per-worker reset(seed=seed+idx))
                    # in _setup_learn when this kwarg is set.
                    train_kwargs["seed"] = run_seed

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
                    "seed":      run_seed,   # None = legacy unseeded run
                    "frame_skip": int(cfg.get("frame_skip", 1)),  # eval MUST match
                    "stall_metric": str(cfg.get("stall_metric", "euclid")),
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
