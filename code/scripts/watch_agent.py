#!/usr/bin/env python3
"""
watch_agent.py — Watch a trained AI agent play visually.

Must be run as a module from the repo root:
    python -m code.scripts.watch_agent <model.zip> [options]

Key fixes vs original:
  - Loads _vecnorm.pkl so observations are normalised identically to training.
  - Loads the correct reward persona so the env matches training exactly.
  - --persona CLI arg (required when called from menu.py).
  - RecurrentPPO / LSTM state support.
"""

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pygame
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

try:
    from sb3_contrib import RecurrentPPO
    _HAS_RPPO = True
except ImportError:
    _HAS_RPPO = False

from code.wrappers.generic_env import GameEnv

# Remove headless SDL so pygame can open a real window
os.environ.pop("SDL_VIDEODRIVER", None)


# ---------------------------------------------------------------------------
# Parse (game, algo, persona, skill) from the model folder/filename
# ---------------------------------------------------------------------------
# Known extractor tags (the last segment added to run_id by train.py)
_EXTRACTOR_TAGS = {"light", "slim", "peak", "mlp"}
# Known skill levels (the second-to-last segment)
_SKILL_LEVELS   = {"novice", "expert", "custom"}

def parse_model_info(model_path: str):
    """
    Parse (game, algo, persona, skill) from the model folder or filename.

    run_id format produced by train.py:
        {game}_{algo}_{persona}_{skill}_{extractor_tag}
    e.g. platformer_ppo_platformer_coin_hunter_novice_slim

    The extractor_tag segment was added later, so we detect it by checking
    the last part against known tags before falling back to old 4-part logic.
    """
    path = Path(model_path)
    folder = path.parent.name
    if folder not in {".", "models", "best"}:
        parts = folder.split("_")
    else:
        parts = path.stem.replace("_model", "").replace("best", "").split("_")

    # Drop trailing extractor tag if present (5-part format)
    if len(parts) >= 5 and parts[-1].lower() in _EXTRACTOR_TAGS:
        parts = parts[:-1]

    if len(parts) >= 4:
        game        = parts[0]
        algo        = parts[1]
        skill       = parts[-1]
        raw_persona = "_".join(parts[2:-1])
        # Strip leading game prefix that train.py prepends to persona names
        persona = (
            raw_persona[len(game) + 1:] if raw_persona.startswith(f"{game}_") else
            raw_persona[len(game):]     if raw_persona.startswith(game) else
            raw_persona
        )
        return game, algo, persona, skill
    elif parts:
        return parts[0], "ppo", "default", "unknown"
    raise ValueError(f"Cannot parse model info from: {model_path}")


# ---------------------------------------------------------------------------
# Find the _vecnorm.pkl sidecar
# ---------------------------------------------------------------------------
def find_vecnorm_path(model_path: str) -> Optional[Path]:
    """
    Search for the companion VecNormalize .pkl file.

    Training saves it as:  <model_folder>/<folder_name>_vecnorm.pkl
    e.g. models/best/platformer_ppo_platformer_dijkstra_novice/
                      platformer_ppo_platformer_dijkstra_novice_vecnorm.pkl

    The model inside that folder is named best_model.zip, so the stem-based
    guess (best_model_vecnorm.pkl) is always wrong. We therefore search the
    parent folder for ANY *vecnorm*.pkl before falling back to the grandparent.
    """
    p = Path(model_path)

    # 1. Exact stem match (correct when model and vecnorm share a stem)
    candidate = p.parent / (p.stem + "_vecnorm.pkl")
    if candidate.exists():
        print(f"[INFO] Vecnorm found (stem match): {candidate}")
        return candidate

    # 2. Glob: any *vecnorm*.pkl in the same directory as the .zip
    for pkl in sorted(p.parent.glob("*vecnorm*.pkl")):
        print(f"[INFO] Vecnorm found (folder glob): {pkl}")
        return pkl

    # 3. Glob: search one level up (models/best/ etc.)
    for pkl in sorted(p.parent.parent.glob("*vecnorm*.pkl")):
        print(f"[INFO] Vecnorm found (parent glob): {pkl}")
        return pkl

    print(f"[DEBUG] Vecnorm search locations checked:")
    print(f"        {p.parent} (model folder)")
    print(f"        {p.parent.parent} (parent folder)")
    return None


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
def load_model(model_path: str, algo: str = "ppo"):
    classes = {"ppo": PPO, "a2c": A2C, "dqn": DQN}
    if _HAS_RPPO:
        classes["rppo"] = RecurrentPPO
        classes["recurrent_ppo"] = RecurrentPPO
    cls = classes.get(algo.lower(), PPO)
    try:
        return cls.load(model_path)
    except Exception as e:
        print(f"[WARN] Failed to load as {algo.upper()}: {e} — trying PPO")
        return PPO.load(model_path)


# ---------------------------------------------------------------------------
# Build env (matches training: same game class, reward fn, VecNormalize)
# ---------------------------------------------------------------------------
def build_env(game: str, persona: str, fps: int, vecnorm_path: Optional[Path], random_start_world: bool = False):
    game_module   = importlib.import_module(f"code.games.{game}_core")
    GameCoreClass = getattr(game_module, f"{game.capitalize()}Core")

    # Load reward fn to match training exactly.
    # Try several name variants because the CLI receives the short name (e.g.
    # "dijkstra") while the actual function may be prefixed ("delta_dijkstra").
    reward_fn = None
    try:
        reward_module = importlib.import_module(f"code.rewards.train_{game}")
        # Candidate names in priority order
        # Strip any trailing skill suffix that may have leaked into the name
        # e.g. "coin_hunter_novice" → "coin_hunter"
        _base_persona = persona
        for _sfx in ("_novice", "_expert", "_custom"):
            if _base_persona.endswith(_sfx):
                _base_persona = _base_persona[: -len(_sfx)]
                break

        candidates = [
            _base_persona,                    # exact: "coin_hunter"
            persona,                          # original (in case suffix is intentional)
            f"delta_{_base_persona}",         # prefixed: "delta_dijkstra"
            f"{game}_{_base_persona}",        # game-prefixed: "platformer_coin_hunter"
            f"{game}_{persona}",              # game-prefixed with skill: last resort
        ]
        found = None
        for name in candidates:
            if hasattr(reward_module, name):
                found = name
                break
        if found:
            reward_fn = getattr(reward_module, found)
            print(f"[INFO] Loaded reward persona: {found}" +
                  (f"  (resolved from '{persona}')" if found != persona else ""))
        else:
            # Last resort: find any function whose name contains the persona string
            all_fns = [n for n in dir(reward_module)
                       if not n.startswith('_') and persona in n
                       and callable(getattr(reward_module, n))]
            if all_fns:
                reward_fn = getattr(reward_module, all_fns[0])
                print(f"[INFO] Loaded reward persona: {all_fns[0]}  (fuzzy match for '{persona}')")
            else:
                print(f"[WARN] Persona '{persona}' not found in {game}_rewards — using default env reward")
                print(f"       Available: {[n for n in dir(reward_module) if not n.startswith('_')]}")
    except Exception as e:
        print(f"[WARN] Could not load reward module: {e}")

    raw_env = GameEnv(
        GameCoreClass,
        render_mode="human",
        fps=fps,
        persona=persona,
        random_start_world=random_start_world,
        **({"reward_fn": reward_fn} if reward_fn else {}),
    )

    if vecnorm_path is None:
        print("\n[WARN] No _vecnorm.pkl found — obs will NOT be normalised.")
        print("       Place the _vecnorm.pkl next to the .zip for correct behaviour.\n")
        return raw_env, False

    print(f"[INFO] Loading VecNormalize from: {vecnorm_path}")
    vec_env  = DummyVecEnv([lambda: raw_env])
    norm_env = VecNormalize.load(str(vecnorm_path), vec_env)
    norm_env.training    = False
    norm_env.norm_reward = False
    print("[INFO] VecNormalize loaded — obs normalised correctly.\n")
    return norm_env, True


# ---------------------------------------------------------------------------
# Main play loop
# ---------------------------------------------------------------------------
def watch_agent_play(
    model_path:         str,
    episodes:           int  = 5,
    fps:                int  = 30,
    deterministic:      bool = True,
    game:    Optional[str] = None,
    algo:    Optional[str] = None,
    persona: Optional[str] = None,
    vecnorm: Optional[str] = None,
    random_start_world: bool = False,
):
    os.environ.pop("SDL_VIDEODRIVER", None)

    # Parse metadata from path, but CLI args always win
    try:
        p_game, p_algo, p_persona, skill = parse_model_info(model_path)
    except Exception as e:
        print(f"[WARN] Could not parse model info: {e}")
        p_game, p_algo, p_persona, skill = "unknown", "ppo", "default", "unknown"

    if game    is None: game    = p_game
    if algo    is None: algo    = p_algo
    if persona is None: persona = p_persona

    print(f"[INFO] game={game}  algo={algo.upper()}  persona={persona}  skill={skill}")

    model = load_model(model_path, algo)

    vn_path = Path(vecnorm) if vecnorm else find_vecnorm_path(model_path)

    pygame.init()
    pygame.display.set_caption(f"PEAK — {game} / {persona}")

    env, is_vec = build_env(game, persona, fps, vn_path, random_start_world=random_start_world)

    is_recurrent = _HAS_RPPO and isinstance(model, RecurrentPPO)
    print(f"[INFO] Watching {episodes} episode(s) at {fps} FPS — ESC or close to stop.\n")

    total_score, completed = 0, 0

    try:
        for ep in range(episodes):
            print(f"--- Episode {ep + 1}/{episodes} ---")

            obs       = env.reset() if is_vec else env.reset()[0]
            done      = truncated = False
            lstm_states = None
            ep_start  = np.ones((1,), dtype=bool)
            score, steps = 0, 0

            while not (done or truncated):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt("window closed")
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        raise KeyboardInterrupt("ESC")
                pygame.event.pump()

                # Debug overlay input (F1 / free-cam)
                core = getattr(getattr(env, "env", env), "game", None)
                if core and hasattr(core, "debug_manager"):
                    core.debug_manager.update_input()

                if is_recurrent:
                    action, lstm_states = model.predict(
                        obs, state=lstm_states,
                        episode_start=ep_start, deterministic=deterministic,
                    )
                    ep_start = np.zeros((1,), dtype=bool)
                else:
                    action, _ = model.predict(obs, deterministic=deterministic)

                if is_vec:
                    obs, reward, dones, info = env.step(action)
                    done = bool(dones[0])
                    info_dict = info[0] if isinstance(info, (list, tuple)) else info
                    try:
                        env.env_method("render", indices=[0])
                    except Exception:
                        pass
                else:
                    obs, reward, done, truncated, info_dict = env.step(action)
                    env.render()

                pygame.time.wait(max(1, 1000 // fps))
                score = info_dict.get("score", score)
                steps += 1

            completed += 1
            total_score += score
            print(f"Episode {ep + 1} done — score={score}  steps={steps}")

    except KeyboardInterrupt as e:
        print(f"\nStopped: {e}")
    finally:
        try: env.close()
        except Exception: pass
        try: pygame.quit()
        except Exception: pass

    if completed:
        print(f"\nDone — {completed} ep(s)  avg score={total_score / completed:.2f}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Watch a trained PEAK agent play.")
    ap.add_argument("model_path",             help="Path to the .zip model file")
    ap.add_argument("--episodes",  type=int,  default=5)
    ap.add_argument("--fps",       type=int,  default=30)
    ap.add_argument("--game",      type=str,  default=None)
    ap.add_argument("--algo",      type=str,  default=None)
    ap.add_argument("--persona",   type=str,  default=None)
    ap.add_argument("--vecnorm",   type=str,  default=None,
                    help="Explicit path to _vecnorm.pkl (auto-detected if omitted)")
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--random-level", action="store_true",
                    help="Start each episode on a random level (mirrors curriculum training)")
    args = ap.parse_args()

    if not Path(args.model_path).exists():
        print(f"Error: model not found: {args.model_path}")
        sys.exit(1)

    try:
        watch_agent_play(
            model_path         = args.model_path,
            episodes           = args.episodes,
            fps                = args.fps,
            deterministic      = not args.stochastic,
            game               = args.game,
            algo               = args.algo,
            persona            = args.persona,
            vecnorm            = args.vecnorm,
            random_start_world = args.random_level,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()