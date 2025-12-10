# code/scripts/play.py
"""
Play trained models, a random agent, or as a human.

Examples
--------
# Play every trained model for 'flappy' with rendering
python -m code.scripts.play --game flappy --render

# Play a random agent (no models needed)
python -m code.scripts.play --game flappy --random --episodes 3 --render

# Play as a human at 60 FPS
python -m code.scripts.play --game flappy --human --fps 60

# Filter to only PPO models and a specific persona
python -m code.scripts.play --game flappy --model PPO --persona flappy_default --render
"""
import argparse
import glob
import importlib
import inspect
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
from omegaconf import OmegaConf

from code.wrappers.generic_env import GameEnv
from code.algos import get_algo  # your registry of SB3 algos


# ---------- Small Hydra helpers (mirrors train.py) ----------

def _load_yaml(path: Path) -> Dict:
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)

def _import_attr(dotted: str) -> Any:
    mod_path, attr = dotted.rsplit(".", 1)
    return getattr(importlib.import_module(mod_path), attr)

def _resolve_callable_or_instance(node: Dict[str, Any]) -> Any:
    if not isinstance(node, dict) or "_target_" not in node:
        raise ValueError(f"Bad hydra node: {node}")
    obj = _import_attr(node["_target_"])
    if inspect.isclass(obj):
        kwargs = {k: v for k, v in node.items() if k != "_target_"}
        return obj(**kwargs)
    return obj


# ---------- Filename parsing ----------

def parse_model_filename(path: Path) -> Tuple[str, str, str, str]:
    """
    Expecting: GAME_MODEL_PERSONA_SKILL.zip
    Returns: (game, model, persona, skill)
    """
    name = path.stem
    parts = name.split("_", 3)  # split into max 4
    if len(parts) != 4:
        raise ValueError(f"Bad model filename (expect game_model_persona_skill.zip): {path.name}")
    game, model, persona, skill = parts
    return game, model, persona, skill


# ---------- Human controller ----------

class HumanPolicy:
    """
    Generic keyboard controller for pygame-based envs.
    Tries reasonable defaults:
      - Discrete(n=2): SPACE => 1, otherwise 0 (e.g., Flappy)
      - Discrete(n=3+): arrows map to 1..4 if present, else 0
      - MultiBinary: SPACE toggles first action bit
      - Box: always zeros (no-op) – override here if needed for your game
    """
    def __init__(self, env):
        self.env = env
        self._last_action = None
        try:
            import pygame
            self.pygame = pygame
        except Exception:
            self.pygame = None

    def _poll_discrete(self, n: int) -> int:
        action = 0  # idle
        if not self.pygame:
            return action
        # If pygame or display not ready yet, do nothing this frame
        if not self.pygame.get_init() or not self.pygame.display.get_init():
            return action
        try:
            for event in self.pygame.event.get():
                if event.type == self.pygame.QUIT:
                    raise KeyboardInterrupt
                if event.type == self.pygame.KEYDOWN:
                    if event.key in (self.pygame.K_ESCAPE, self.pygame.K_q):
                        raise KeyboardInterrupt
                    if n == 2:
                        if event.key in (self.pygame.K_SPACE, self.pygame.K_UP, self.pygame.K_w):
                            action = 1
                    else:
                        if event.key in (self.pygame.K_UP, self.pygame.K_w):   action = min(1, n-1)
                        if event.key in (self.pygame.K_RIGHT, self.pygame.K_d): action = min(2, n-1)
                        if event.key in (self.pygame.K_LEFT, self.pygame.K_a):  action = min(3, n-1)
                        if event.key in (self.pygame.K_DOWN, self.pygame.K_s):  action = min(4, n-1)
        except Exception:
            # If pygame throws (e.g., not initialized yet), ignore this frame
            return 0
        return action

    def _poll_multibinary(self, shape: Tuple[int, ...]) -> np.ndarray:
        k = int(np.prod(shape))
        action = np.zeros(k, dtype=np.int8)
        if not self.pygame or not self.pygame.get_init() or not self.pygame.display.get_init():
            return np.zeros(int(np.prod(shape)), dtype=np.int8).reshape(shape)
        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                raise KeyboardInterrupt
            if event.type == self.pygame.KEYDOWN:
                if event.key in (self.pygame.K_ESCAPE, self.pygame.K_q):
                    raise KeyboardInterrupt
                if event.key in (self.pygame.K_SPACE, self.pygame.K_UP, self.pygame.K_w):
                    action[0] = 1
        return action.reshape(shape)

    def _poll_box(self, shape: Tuple[int, ...]) -> np.ndarray:
        # No-op by default; customize per game if needed
        return np.zeros(shape, dtype=np.float32)

    def __call__(self, obs, deterministic: bool = True):
        space = self.env.action_space
        if hasattr(space, "n"):  # Discrete
            return self._poll_discrete(space.n)
        if space.__class__.__name__ == "MultiBinary":
            return self._poll_multibinary(space.shape)
        if hasattr(space, "shape"):  # Box
            return self._poll_box(space.shape)
        # Fallback
        return 0


# ---------- Core runners ----------

def make_env(game_name: str, persona: Optional[str], render: bool, fps: Optional[int], conf_root: Path):
    """Create a GameEnv with the appropriate reward (if persona provided)."""
    # Load game class from conf/game/<game>.yaml
    game_cfg = _load_yaml(conf_root / "game" / f"{game_name}.yaml")
    game_cls = _import_attr(game_cfg["_target_"])

    # Reward (if persona is given). For random/human we still allow reward to keep env consistent.
    reward_fn = None
    if persona is not None:
        reward_yaml = conf_root / "reward" / f"{persona}.yaml"
        if reward_yaml.exists():
            reward_cfg = _load_yaml(reward_yaml)
            reward_fn = _resolve_callable_or_instance(reward_cfg)

    env = GameEnv(
        game_cls,
        reward_fn=reward_fn,
        render_mode="human" if render else "none",
        fps=None if not render else int(fps) if fps else 60,
        max_steps=None,
    )
    return env

import time

def run_episode(env, policy_fn, deterministic: bool = True, render: bool = False, fps: int = 60) -> float:
    obs, _ = env.reset()

    # Ensure a window exists before polling keyboard
    if render:
        try:
            env.render()
        except Exception:
            pass
        if fps and fps > 0:
            time.sleep(1.0 / float(fps))

    done = False
    total = 0.0
    delay = (1.0 / float(fps)) if (render and fps and fps > 0) else 0.0

    while not done:
        act = policy_fn(obs, deterministic=deterministic)
        if isinstance(act, tuple):
            act = act[0]
        obs, reward, terminated, truncated, _info = env.step(act)
        total += float(reward)
        done = terminated or truncated

        if render:
            try:
                env.render()
            except Exception:
                pass
            if delay:
                time.sleep(delay)
    return total

def sb3_policy_from_path(model_path: Path, algo_name_hint: Optional[str], env):
    """Load an SB3 model given a file path and optional algo hint."""
    algo_name = algo_name_hint or "ppo"
    Algo = get_algo(algo_name)
    model = Algo.load(str(model_path), env=env, print_system_info=False)
    def _predict(obs, deterministic=True):
        action, _ = model.predict(obs, deterministic=deterministic)
        return action
    return _predict

def random_policy_from_env(env):
    def _rand(obs, deterministic: bool = True):
        return env.action_space.sample()
    return _rand


# ---------- Main ----------

def find_models(models_dir: Path, game: str, model_filter: Optional[str], persona_filter: Optional[str], skill_filter: Optional[str]) -> Iterable[Path]:
    pattern = f"{game}_*.zip"
    for p in sorted(models_dir.glob(pattern)):
        try:
            g, m, persona, skill = parse_model_filename(p)
        except ValueError:
            continue
        if model_filter and m.lower() != model_filter.lower():
            continue
        if persona_filter and persona != persona_filter:
            continue
        if skill_filter and skill != skill_filter:
            continue
        yield p

def main():
    parser = argparse.ArgumentParser(description="Play trained models / random / human.")
    parser.add_argument("--game", required=True, help="Game name (must match conf/game/<game>.yaml)")
    parser.add_argument("--models_dir", default="models", help="Directory with trained models")
    parser.add_argument("--episodes", type=int, default=1, help="Episodes per run")
    parser.add_argument("--render", action="store_true", help="Render the game window")
    parser.add_argument("--fps", type=int, default=60, help="FPS when rendering")
    parser.add_argument("--deterministic", action="store_true", help="Deterministic SB3 policy")
    parser.add_argument("--random", action="store_true", help="Use a random agent instead of loading models")
    parser.add_argument("--human", action="store_true", help="Play as a human (keyboard)")
    parser.add_argument("--model", default=None, help="Filter by algo name in filename (e.g., ppo, a2c)")
    parser.add_argument("--persona", default=None, help="Filter by persona name in filename")
    parser.add_argument("--skill", default=None, help="Filter by skill name in filename")
    args = parser.parse_args()

    repo_root = Path(os.getcwd()).resolve()
    conf_root = repo_root / "code" / "conf"
    models_dir = (repo_root / args.models_dir).resolve()

    if args.human and not args.render:
        print("Note: --human implies --render. Enabling rendering.")
        args.render = True

    if args.random and args.human:
        raise SystemExit("Choose either --random or --human, not both.")

    if args.random:
        # Random agent: no need to find models, but keep persona for reward if specified
        env = make_env(args.game, args.persona, render=args.render, fps=args.fps, conf_root=conf_root)
        policy = random_policy_from_env(env)
        for ep in range(args.episodes):
            try:
                total = run_episode(env, policy, render=args.render, fps=args.fps)
                print(f"[random] ep {ep+1}/{args.episodes}  return={total:.3f}")
            except KeyboardInterrupt:
                break
        try:
            env.close()
        except Exception:
            pass
        return

    if args.human:
        # Human controller
        env = make_env(args.game, args.persona, render=True, fps=args.fps, conf_root=conf_root)
        human = HumanPolicy(env)
        for ep in range(args.episodes):
            try:
                total = run_episode(env, human, render=args.render, fps=args.fps)
                print(f"[human] ep {ep+1}/{args.episodes}  return={total:.3f}")
            except KeyboardInterrupt:
                break
        try:
            env.close()
        except Exception:
            pass
        return

    # Otherwise: play all matching trained models
    model_paths = list(find_models(models_dir, args.game, args.model, args.persona, args.skill))
    if not model_paths:
        raise SystemExit(f"No models found in {models_dir} for pattern {args.game}_*.zip (with filters model={args.model}, persona={args.persona}, skill={args.skill})")

    print(f"Found {len(model_paths)} model(s). Playing each for {args.episodes} episode(s).")
    for path in model_paths:
        game, algo_name, persona, skill = parse_model_filename(path)

        # Create env with reward from persona
        env = make_env(game, persona, render=args.render, fps=args.fps, conf_root=conf_root)

        # Load algo from filename (ppo, a2c, etc.)
        Algo = get_algo(algo_name)
        model = Algo.load(str(path), env=env, print_system_info=False)

        def policy(obs, deterministic=True):
            action, _ = model.predict(obs, deterministic=deterministic)
            return action

        for ep in range(args.episodes):
            try:
                total = run_episode(env, policy, deterministic=args.deterministic, render=args.render, fps=args.fps)
                print(f"[{path.name}] ep {ep+1}/{args.episodes}  return={total:.3f}")
            except KeyboardInterrupt:
                break

        try:
            env.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
