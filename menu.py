# menu.py — unified CLI for training, eval, TensorBoard, and manual play
# Compatible with: Hydra overrides, TB logs under mylogs/, flat model files in models/

# Imports 
import subprocess 
import webbrowser
import os
import sys
import platform
from pathlib import Path
import shutil
import time
from omegaconf import OmegaConf
import importlib
import random
import pygame
import numpy as np


# Stable-Baselines3 Algo Imports
try:
    from stable_baselines3 import PPO, A2C, DQN, SAC, TD3
    from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
    
    # Import Monitor for recording purposes
    from stable_baselines3.common.monitor import Monitor

    # Algo mapping
    HAS_SB3 = True
    ALGO_CLASS_MAP = {
        "ppo": PPO,
        "a2c": A2C,
        "dqn": DQN,
        "sac": SAC,
        "td3": TD3,
    }
    
except ImportError:
    HAS_SB3 = False
    ALGO_CLASS_MAP = {}


# Add winsound for Windows only
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


# root folder setup
DEFAULT_TB_ROOT = "mylogs"
MODELS_DIR = Path("models/")
CONF_ROOT = Path("code/conf")
GRID_CONFIG_PATH = CONF_ROOT / "grid.yaml"
CONF_GAME_DIR = CONF_ROOT / "game"
CONF_REWARD_DIR = CONF_ROOT / "reward"
CONF_ALGO_DIR = CONF_ROOT / "algo"

CURRENT_ALGO = None

REQUIRED_PACKAGES = [
    'torch>=1.9.0',
    'stable-baselines3>=1.6.0',
    'sb3-contrib>=1.6.0', 
    'gymnasium>=0.26.0',
    'pygame>=2.1.0',
    'numpy>=1.21.0',
    'tensorboard>=2.8.0',
    'hydra-core>=1.1.0',
    'pyyaml>=6.0',
    'omegaconf>=2.1.0',
    'imageio',
    'moviepy',
    'streamlit',
    'plotly'
]

# ============================================================================
# HELPER FUNCTIONS - Configuration & Setup
# ============================================================================

# video deps (lazy-loaded to avoid import issues)
def get_moviepy_editor():
    """
    Try to obtain a MoviePy editor-like module in a version-agnostic way.

    Returns:
        mpy module (with ImageSequenceClip / VideoFileClip) or None.
    """
    # Newer MoviePy: recommended pattern is `from moviepy import editor`
    try:
        from moviepy import editor as mpy
        return mpy
    except Exception:
        pass

    # Fallback: classic style `import moviepy.editor as mpy`
    try:
        import moviepy.editor as mpy
        return mpy
    except Exception:
        pass

    # Last resort: raw moviepy; some versions expose classes at top-level
    try:
        import moviepy as mpy
        return mpy
    except Exception:
        pass

    return None

# MOVE START UP 
MPY = get_moviepy_editor()
HAS_MOVIEPY = MPY is not None

def check_and_install_dependencies():
    """Check if required packages are installed and install missing ones"""
    # check dependencies if missing install requirements
    print("Checking dependencies...")
    
    missing_packages = []
    
    for package in REQUIRED_PACKAGES:
        package_name = package.split('>=')[0].split('==')[0]
        try:
            __import__(package_name.replace('-', '_'))
        except ImportError:
            try:
                if package_name == 'stable-baselines3':
                    import stable_baselines3
                elif package_name == 'sb3-contrib':
                    import sb3_contrib
                elif package_name == 'hydra-core':
                    import hydra
                elif package_name == 'pyyaml':
                    import yaml
                elif package_name == 'omegaconf':
                    import omegaconf
                else:
                    raise ImportError()
            except ImportError:
                missing_packages.append(package)
    
    if not missing_packages:
        print("All dependencies are installed!")
        return True
    
    print(f"Missing packages: {', '.join(missing_packages)}")
    
    response = input("\nWould you like to install missing dependencies? [y/N]: ").strip().lower()
    if response not in ('y', 'yes'):
        print("Cannot proceed without required dependencies.")
        return False
    
    print("Installing missing packages...")
    try:
        cmd = [sys.executable, '-m', 'pip', 'install'] + missing_packages
        subprocess.check_call(cmd)
        print("Successfully installed all dependencies!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install dependencies: {e}")
        print("Please install manually using: pip install -r requirements.txt")
        return False

def setup_project():
    """Initial project setup"""
    # if requirements don't exists makes requirements.txt
    
    # Check and create requirements.txt if missing
    requirements_path = Path("requirements.txt")
    if not requirements_path.exists():
        requirements_content = "\n".join(REQUIRED_PACKAGES) + "\n"
        requirements_path.write_text(requirements_content)

# checking if grid yaml 
def load_grid_config():
    """Load grid.yaml configuration"""
    if not GRID_CONFIG_PATH.exists():
        return None
    try:
        return OmegaConf.load(GRID_CONFIG_PATH)
    except Exception as e:
        print(f"Error loading grid.yaml: {e}")
        return None

def get_available_games():
    """Get games from grid.yaml."""
    cfg = load_grid_config()
    if cfg is not None and 'games' in cfg and cfg.games:
        return sorted(list(cfg.games))
    
    print("Warning: No 'games' section found or it is empty in grid.yaml.")
    return []

def get_available_algos_from_grid():
    """Get algorithms from grid.yaml that have corresponding YAML files"""
    cfg = load_grid_config()
    if cfg is None or 'models' not in cfg:
        return []
    
    grid_algos = list(cfg.models) if cfg.models else []
    available = []
    for algo in grid_algos:
        algo_file = CONF_ALGO_DIR / f"{algo}.yaml"
        if algo_file.exists():
            available.append(algo)
        else:
            print(f"Warning: Algorithm '{algo}' in grid.yaml but no file at {algo_file}")
    
    return sorted(available)

def get_available_personas_from_grid():
    """Get personas from grid.yaml."""
    cfg = load_grid_config()
    if cfg is not None and 'personas' in cfg and cfg.personas:
        return sorted(list(cfg.personas))
        
    print("Warning: No 'personas' section found or it is empty in grid.yaml.")
    return []

def get_personas_for_game(game: str):
    """Return only personas from grid.yaml whose YAML stem starts with '<game>_'"""
    all_personas = get_available_personas_from_grid()
    filtered = [p for p in all_personas if p.startswith(f"{game}_")]
    return filtered if filtered else all_personas

def get_trained_models_count():
    """Count total number of trained models in models/best/"""
    BEST_DIR = MODELS_DIR / "best"
    if not BEST_DIR.exists():
        return 0
    model_folders = [f for f in BEST_DIR.iterdir() if f.is_dir() and (f / "best_model.zip").exists()]
    return len(model_folders)

def get_trained_games_from_models_flat():
    """Infer trained games from model folders in models/best/"""
    BEST_DIR = MODELS_DIR / "best"
    if not BEST_DIR.exists():
        return []
    
    model_folders = [f for f in BEST_DIR.iterdir() if f.is_dir() and (f / "best_model.zip").exists()]
    if not model_folders:
        return []
    
    games = set()
    for folder in model_folders:
        parts = folder.name.split("_")
        if len(parts) >= 5:
            games.add(parts[0])
    
    return sorted(games)

def get_model_folders():
    """Get all model folders containing best_model.zip"""
    BEST_DIR = MODELS_DIR / "best"
    if not BEST_DIR.exists():
        return []
    return [f for f in BEST_DIR.iterdir() if f.is_dir() and (f / "best_model.zip").exists()]

# ============================================================================
# HELPER FUNCTIONS - User Interface
# ============================================================================

def open_browser(url):
    """Open URL in default browser, cross-platform"""
    try:
        webbrowser.open(url)
    except Exception:
        try:
            if platform.system() == "Linux":
                os.system(f"xdg-open {url}")
            elif platform.system() == "Windows":
                os.system(f"start {url}")
            elif platform.system() == "Darwin":
                os.system(f"open {url}")
        except Exception:
            pass
    print(f"If your browser did not open automatically, go to: {url}")

def ask_index(prompt, options, add_back=True, default=None):
    """Print numbered options and return the selected item (or None if back)"""
    if not options:
        print("No options available.")
        return None
    
    print(prompt)
    for i, opt in enumerate(options, 1):
        default_flag = " (default)" if opt == default else ""
        print(f"  {i}. {opt}{default_flag}")
    
    back_idx = len(options) + 1
    if add_back:
        print(f"  {back_idx}. Back")
    
    prompt_text = f"Select (1-{back_idx if add_back else len(options)})"
    if default:
        prompt_text += f" or Enter for [{default}]"
    prompt_text += ": "
    
    choice = input(prompt_text).strip()
    
    if choice == "" and default:
        return default
    
    try:
        num = int(choice)
        if add_back and num == back_idx:
            return None
        if 1 <= num <= len(options):
            return options[num - 1]
    except ValueError:
        pass
    
    print("Invalid selection.")
    return None

# def ensure_current_algo():
#     """Ensure CURRENT_ALGO is set, defaulting to PPO if available"""
#     global CURRENT_ALGO
    
#     if CURRENT_ALGO is None:
#         algos = get_available_algos_from_grid()
#         if algos:
#             CURRENT_ALGO = "ppo" if "ppo" in algos else algos[0]

# ============================================================================
# TRAINING EXECUTION - Core Logic (DRY refactor)
# ============================================================================

def execute_training_run(game, algo, persona, skill, tb_root=DEFAULT_TB_ROOT):
    """Execute a single training run with error handling"""
    cmd = [
        sys.executable, "-m", "code.scripts.train",
        f"+game={game}",
        f"+model={algo}",
        f"+persona={persona}",
        f"+skill={skill}",
        f"tb_root={tb_root}",
    ]
    
    print(">>> " + " ".join(cmd) + "\n")
    
    # Runs the hydra CMD training script with specified parameters
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {game} | {algo} | {persona} | {skill} (exit code {e.returncode})")
        return False
    except KeyboardInterrupt:
        raise  # Re-raise to be caught by caller

def print_training_summary(total, successful, failed):
    """Print final training summary with sound notification"""
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"✓ Successful: {successful}/{total}")
    if failed > 0:
        print(f"❌ Failed: {failed}/{total}")
    print(f"Logs saved to: {DEFAULT_TB_ROOT}/")
    print(f"Models saved to: {MODELS_DIR}/best/")
    
    if HAS_WINSOUND and failed == 0:
        winsound.PlaySound("chime.wav", winsound.SND_FILENAME)
    
    print()

# ============================================================================
# MAIN ACTIONS
# ============================================================================

def run_training():
    """Single training run with user-selected parameters"""
    global CURRENT_ALGO
    
    print("\n=== Training ===")
    games = get_available_games()
    if not games:
        print("No game configurations found in grid.yaml or code/conf/game/")
        return

    game = ask_index("Available games:", games)
    if game is None:
        return


    algos = get_available_algos_from_grid()
    if not algos:
        print("No algorithm configurations found in grid.yaml with matching YAML files")
        return
    
    #ensure_current_algo() - URGENT
    default_algo = CURRENT_ALGO if CURRENT_ALGO in algos else ("ppo" if "ppo" in algos else algos[0])
    
    algo_choice = ask_index("Available algorithms:", algos, default=default_algo)
    if algo_choice is None:
        return
    
    CURRENT_ALGO = algo_choice

    personas = get_personas_for_game(game)
    if not personas:
        print(f"No personas found for game='{game}' in grid.yaml.")
        return

    persona_choice = ask_index("Available personas:", personas)
    if persona_choice is None:
        return

    skills = ["Novice", "Expert", "Novice & Expert", "Custom steps"]
    skill_choice = ask_index("Skill / steps:", skills)
    if skill_choice is None:
        return

    tb_root = input(f"TensorBoard log root [{DEFAULT_TB_ROOT}]: ").strip() or DEFAULT_TB_ROOT

    # Custom steps
    if skill_choice == "Custom steps":
        steps_str = input("Enter total steps (e.g., 300000): ").strip()
        try:
            steps = int(steps_str)
        except ValueError:
            print("Invalid number.")
            return

        cmd = [
            sys.executable, "-m", "code.scripts.train",
            f"+game={game}", f"+model={algo_choice}", f"+persona={persona_choice}",
            "skill=Custom", f"+skills.Custom={steps}", f"tb_root={tb_root}",
        ]
        print("\n>>>", " ".join(cmd), "\n")
        subprocess.run(cmd)
        if HAS_WINSOUND:
            winsound.PlaySound("chime.wav", winsound.SND_FILENAME)
        print("\nTraining completed.\n")
        return

    # Train both Novice & Expert
    if skill_choice == "Novice & Expert":
        for sk in ("Novice", "Expert"):
            print(f"\n>>> Training {game} | Algo: {algo_choice} | Persona: {persona_choice} | Skill: {sk}")
            execute_training_run(game, algo_choice, persona_choice, sk, tb_root)
        print("\n✓ Completed Novice & Expert runs.\n")
        if HAS_WINSOUND:
            winsound.PlaySound("chime.wav", winsound.SND_FILENAME)
        return

    # Single skill
    execute_training_run(game, algo_choice, persona_choice, skill_choice, tb_root)
    if HAS_WINSOUND:
        winsound.PlaySound("chime.wav", winsound.SND_FILENAME)
    print("\nTraining completed.\n")

def train_all_models_for_game():
    """Train all (algo x persona x skill) models for ONE user-selected game"""
    global CURRENT_ALGO
    
    print("\n" + "=" * 60)
    print("TRAIN ALL MODELS FOR ONE GAME")
    print("=" * 60)
    
    games = get_available_games()
    if not games:
        print("No game configurations found in grid.yaml or code/conf/game/")
        return

    game = ask_index("Available games:", games)
    if game is None:
        return

    algos = get_available_algos_from_grid()
    if not algos:
        print("No algorithm configurations found in grid.yaml")
        return
    
    ensure_current_algo()
    
    # Algorithm selection
    print("\nSelect algorithms to train:")
    print("  1. All algorithms")
    
    for i, algo in enumerate(algos, 2):
        default_flag = " (current)" if algo == CURRENT_ALGO else ""
        print(f"  {i}. {algo} only{default_flag}")
    print(f"  {len(algos) + 2}. Back")
    
    choice = input(f"Select (1-{len(algos) + 2}): ").strip()
    try:
        num = int(choice)
        if num == len(algos) + 2:
            return
        elif num == 1:
            selected_algos = algos
        elif 2 <= num <= len(algos) + 1:
            selected_algos = [algos[num - 2]]
            CURRENT_ALGO = selected_algos[0]
        else:
            print("Invalid selection.")
            return
    except ValueError:
        print("Invalid selection.")
        return

    personas = get_personas_for_game(game)
    if not personas:
        print(f"No personas for game '{game}' in grid.yaml")
        return

    skills = ["Novice", "Expert"]
    total_runs = len(selected_algos) * len(personas) * len(skills)
    
    # Show summary
    print(f"\nTraining {total_runs} model(s) for '{game}':")
    print(f"  Algorithms: {len(selected_algos)}")
    print(f"  Personas: {len(personas)}")
    print(f"  Skills: {len(skills)}")
    
    confirm = input(f"\nProceed with {total_runs} training runs? [y/N]: ").strip().lower()
    if confirm not in ('y', 'yes'):
        print("Aborted.")
        return
    
    # Execute training
    completed = 0
    failed = 0
    
    try:
        for algo in selected_algos:
            for persona in personas:
                for skill in skills:
                    completed += 1
                    print("\n" + "=" * 60)
                    print(f"Progress: {completed}/{total_runs}")
                    print(f"Training: {game} | {algo} | {persona} | {skill}")
                    print("=" * 60)
                    
                    success = execute_training_run(game, algo, persona, skill)
                    if not success:
                        failed += 1
    except KeyboardInterrupt:
        print("\n\n Training interrupted by user.")
        print(f"Completed: {completed - 1}/{total_runs}")
        print(f"Failed: {failed}")
        return
    
    print_training_summary(total_runs, completed - failed, failed)

def train_complete_grid():
    """Train ALL (game x algo x persona x skill) combinations"""
    print("\n" + "=" * 60)
    print("TRAIN COMPLETE GRID (All Games × Algos × Personas)")
    print("=" * 60)
    
    games = get_available_games()
    algos = get_available_algos_from_grid()
    
    if not games:
        print("No game configurations found in grid.yaml or code/conf/game/")
        return
    
    if not algos:
        print("No algorithm configurations found in grid.yaml")
        return
    
    # Calculate total runs
    total_runs = 0
    breakdown = []
    
    for game in games:
        personas = get_personas_for_game(game)
        if not personas:
            continue
        
        runs_for_game = len(algos) * len(personas) * 2
        total_runs += runs_for_game
        breakdown.append(f"  • {game}: {len(personas)} persona(s) × {len(algos)} algo(s) × 2 skills = {runs_for_game} runs")
    
    if total_runs == 0:
        print("\n❌ No valid training configurations found.")
        return
    
    # Show summary
    print(f"\nThis will train {total_runs} total model(s):")
    print(f"\nGames: {len(games)}")
    print(f"Algorithms: {len(algos)}")
    print(f"Skills: 2 (Novice, Expert)")
    print("\nBreakdown by game:")
    for line in breakdown:
        print(line)
    
    print(f"Logs will be saved to: {DEFAULT_TB_ROOT}/")
    
    confirm = input(f"\nProceed with {total_runs} training runs? [y/N]: ").strip().lower()
    if confirm not in ('y', 'yes'):
        print("Aborted.")
        return
    
    # Execute training grid
    skills = ["Novice", "Expert"]
    completed = 0
    failed = 0
    
    try:
        for game in games:
            personas = get_personas_for_game(game)
            if not personas:
                continue
            
            for algo in algos:
                for persona in personas:
                    for skill in skills:
                        completed += 1
                        print("\n" + "=" * 60)
                        print(f"Progress: {completed}/{total_runs}")
                        print(f"Training: {game} | {algo} | {persona} | {skill}")
                        print("=" * 60)
                        
                        success = execute_training_run(game, algo, persona, skill)
                        if not success:
                            failed += 1
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print(f"Completed: {completed - 1}/{total_runs}")
        print(f"Failed: {failed}")
        return
    
    print_training_summary(total_runs, completed - failed, failed)


# Currently NOT IN USE - needs adaptation
def run_evaluation():
    """Evaluate all trained models for a selected game"""
    print("\n===== Evaluation =====")

    BEST_DIR = MODELS_DIR / "best"
    if not BEST_DIR.exists():
        print("[!] models/best/ does not exist — please train some models first.")
        return

    model_folders = get_model_folders()
    if not model_folders:
        print("No best_model.zip files found in models/best/.")
        return

    games = sorted(set(f.name.split("_")[0] for f in model_folders))
    if not games:
        print("No recognized games found in models/best/.")
        return

    game = ask_index("Games with trained models", games)
    if game is None:
        return

    model_dirs = [f for f in model_folders if f.name.startswith(game)]
    if not model_dirs:
        print(f"No best_model.zip folders found for game='{game}' in {BEST_DIR}")
        return

    print(f"\nFound {len(model_dirs)} model(s) for '{game}'. Running quick eval (5 eps each)...\n")

    for model_dir in model_dirs:
        model_zip = model_dir / "best_model.zip"
        model_name = model_dir.name

        parts = model_name.split("_")
        algo = parts[1] if len(parts) > 1 else "ppo"

        out_json = MODELS_DIR / f"{model_name}_eval.json"
        metrics_class = f"code.metrics.{game}_balance.{game.capitalize()}BalanceStats"

        cmd = [
            sys.executable, "-m", "code.scripts.evaluate",
            "--game", game, "--algo", algo, "--model", str(model_zip),
            "--episodes", "5", "--render", "none",
            "--out", str(out_json), "--metrics", metrics_class,
        ]
        print(">>>", " ".join(cmd))
        subprocess.run(cmd)

    print("\n✓ Evaluation completed for all best models.\n")


def parse_model_metadata(model_path: Path):
    """
    Take folder name and parse out game, algo, persona, skill.
    Returns a dict with keys: game, algo, persona, skill (values or None).
    """
    folder = model_path.parent.name
    parts = folder.split("_")

    meta = {
        "game": parts[0] if len(parts) >= 1 else None,
        "algo": parts[1] if len(parts) >= 2 else None,
        "persona": parts[3] if len(parts) >= 4 else None,
        "skill": parts[4] if len(parts) >= 5 else None,
    }
    return meta

def record_agent_video(model_path: Path, episodes: int, fps: int = 30):
    """
    Run the trained agent and save both MP4 and GIF as:

      videos/<game>/<game>_<persona>[_<skill>].mp4
      videos/<game>/<game>_<persona>[_<skill>].gif
    """
    mpy = get_moviepy_editor()
    if mpy is None:
        print("Recording requires 'moviepy'. Run inside the venv and: pip install moviepy imageio[ffmpeg]")
        return


    meta = parse_model_metadata(model_path)
    
    game = meta["game"]
    algo_name = (meta["algo"] or "").lower()
    persona = (meta["persona"] or "default").lower()
    skill = (meta["skill"] or "").lower()

    if not game or not algo_name:
        print("Could not infer game/algo from model path; skipping recording.")
        return

    algo_cls = ALGO_CLASS_MAP.get(algo_name)
    if algo_cls is None:
        print(f"Unsupported/unknown algo '{algo_name}' for recording.")
        return

    # Error handling for game env + model load
    try:
        from code.wrappers.generic_env import GameEnv
    except ImportError as e:
        print(f"Could not import GameEnv for recording: {e}")
        return

    # Check if game exists
    try:
        game_mod = importlib.import_module(f"code.games.{game}_core")
    except ImportError as e:
        print(f"Could not import game core for '{game}': {e}")
        return

    # Find the *Core - URGENT
    GameCls = None
    for attr in dir(game_mod):
        if attr.endswith("Core"):
            GameCls = getattr(game_mod, attr)
            break
    if GameCls is None:
        print(f"No *Core class found in code.games.{game}_core.")
        return

    # Use rgb_array for capture
    env = GameEnv(GameCls, render_mode="rgb_array", fps=fps)

    # Load model path
    try:
        model = algo_cls.load(str(model_path), env=env)
    except Exception as e:
        env.close()
        print(f"Failed to load model for recording: {e}")
        return

    frames = []
    max_steps_per_ep = 2000

    for ep in range(episodes):
        reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

        done = False
        steps = 0

        while not done and steps < max_steps_per_ep:
            action, ep = model.predict(obs, deterministic=True)
            step_result = env.step(action)

            # Ensure's 5 variables are handled
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result

            frame = env.render(mode="rgb_array")
            
            if frame is not None:
                frames.append(np.asarray(frame))

            if info.get("episode_end", False):
                done = True

            steps += 1

    env.close()

    if not frames:
        print("No frames captured; nothing to save.")
        return

    # Save MP4 + GIF
    videos_root = Path("videos")
    game_dir = videos_root / game
    game_dir.mkdir(parents=True, exist_ok=True)

    parts = [game, persona if persona else None, skill if skill else None]
    base_name = "_".join(p for p in parts if p)

    mp4_path = game_dir / f"{base_name}.mp4"
    gif_path = game_dir / f"{base_name}.gif"

    clip = mpy.ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(str(mp4_path), codec="libx264")
    clip.write_gif(str(gif_path), fps=fps)
    clip.close()

    print(f"\nSaved MP4: {mp4_path}")
    print(f"Saved GIF: {gif_path}\n")

def record_random_agent_video(game: str, episodes: int = 5, fps: int = 30):
    """
    Run a random policy in the given game and save:

      videos/<game>/<game>_random.mp4
      videos/<game>/<game>_random.gif
    """
    mpy = get_moviepy_editor()
    if mpy is None:
        print("Recording random agent requires 'moviepy'. Run inside the venv and: pip install moviepy imageio[ffmpeg]")
        return

    try:
        from code.wrappers.generic_env import GameEnv
    except ImportError as e:
        print(f"Could not import GameEnv for random recording: {e}")
        return

    try:
        game_mod = importlib.import_module(f"code.games.{game}_core")
    except ImportError as e:
        print(f"Could not import game core for '{game}': {e}")
        return

    GameCls = None
    for attr in dir(game_mod):
        if attr.endswith("Core"):
            GameCls = getattr(game_mod, attr)
            break
    if GameCls is None:
        print(f"No *Core class found in code.games.{game}_core.")
        return

    env = GameEnv(GameCls, render_mode="rgb_array", fps=fps)

    frames = []
    max_steps_per_ep = 2000

    for _ in range(episodes):
        reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

        done = False
        steps = 0

        while not done and steps < max_steps_per_ep:
            action = env.action_space.sample()
            step_result = env.step(action)

            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result

            frame = env.render(mode="rgb_array")
            if frame is not None:
                frames.append(np.asarray(frame))

            if info.get("episode_end", False):
                done = True

            steps += 1

    env.close()

    if not frames:
        print("No frames captured for random agent recording.")
        return

    videos_root = Path("videos")
    game_dir = videos_root / game
    game_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"{game}_random"
    mp4_path = game_dir / f"{base_name}.mp4"
    gif_path = game_dir / f"{base_name}.gif"

    clip = mpy.ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(str(mp4_path), codec="libx264")
    clip.write_gif(str(gif_path), fps=fps)
    clip.close()

    print(f"\nSaved MP4: {mp4_path}")
    print(f"Saved GIF: {gif_path}\n")

def watch_trained_agent():
    """Select a trained model, watch it play in a pygame window, optionally record that exact session."""
    print("\n=== Watch Trained Agent Play ===")

    model_folders = get_model_folders()
    if not model_folders:
        print("No best_model.zip files found in models/best/.")
        return

    # Build display list
    display_options = []
    paths = []
    
    for folder in model_folders:
        parts = folder.name.split("_")
        
        if len(parts) >= 5:
            game, algo, _, persona, skill = parts[:5]
            display = f"{game:<8} | {algo:<4} | {persona:<12} | {skill:<8}"
        else:
            display = folder.name
        
        display_options.append(display)
        paths.append(folder / "best_model.zip")

    selected = ask_index("Select a trained model to visualize:", display_options)
    
    if selected is None:
        return

    # Error handling for selection
    try:
        model_idx = display_options.index(selected)
        model_path = paths[model_idx]
    except (ValueError, IndexError):
        print("Invalid selection.")
        return

    # Infer metadata
    meta = parse_model_metadata(model_path)
    game = meta["game"]
    algo_name = (meta["algo"] or "").lower()
    persona = (meta["persona"] or "default").lower()
    skill = (meta["skill"] or "").lower()

    if not game or not algo_name:
        print("Could not infer game/algo from model path.")
        return

    algo_cls = ALGO_CLASS_MAP.get(algo_name)
    
    if algo_cls is None:
        print(f"Unsupported/unknown algo '{algo_name}'.")
        return

    # FPS + episode cap
    fps_input = input("Rendering FPS? [30]: ").strip()
    fps = int(fps_input) if fps_input.isdigit() and int(fps_input) > 0 else 30

    max_ep_input = input("Max episodes to play before auto-exit? [10]: ").strip()
    max_episodes = int(max_ep_input) if max_ep_input.isdigit() and int(max_ep_input) >= 0 else 10
    # Note: 0 = no auto-stop (ESC/close only)

    # Prompt for recording
    rec_choice = input("Record this session to videos/ as MP4 + GIF? [y/N]: ").strip().lower()
    
    if rec_choice in ("y", "yes"):
        # INLINE MODE (Visual + Record)
        # Keeps the user request "see the agent play while its recording"
        # Warning: Might freeze if run repeatedly due to Pygame process limitations.
        mpy = get_moviepy_editor()
        if mpy is None:
            print("Recording requires 'moviepy'. Run inside the venv and: pip install moviepy imageio[ffmpeg]")
            return

        
        # --- Env + model setup
        try:
            from code.wrappers.generic_env import GameEnv
        except ImportError as e:
            print(f"Could not import GameEnv: {e}")
            return

        try:
            game_mod = importlib.import_module(f"code.games.{game}_core")
        except ImportError as e:
            print(f"Could not import game core for '{game}': {e}")
            return

        GameCls = None
        for attr in dir(game_mod):
            if attr.endswith("Core"):
                GameCls = getattr(game_mod, attr)
                break
        if GameCls is None:
            print(f"No *Core class found in code.games.{game}_core.")
            return

        pygame.init()

        # IMPORTANT: use human mode for visible window
        # Pass persona to core for correct reward calculation
        env = GameEnv(GameCls, render_mode="human", fps=fps, persona=persona)

        try:
            model = algo_cls.load(str(model_path))
        except Exception as e:
            env.close()
            pygame.quit()
            print(f"Failed to load model: {e}")
            return

        reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

        clock = pygame.time.Clock()
        running = True
        episodes = 0
        frames = []

        print("Press ESC or close the window to end the session.")

        while running and (max_episodes == 0 or episodes < max_episodes):
            clock.tick(fps)

            # Handle quit
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    running = False

            if not running:
                break

            # Ensure debug manager processes inputs (like Free Cam)
            if hasattr(env, 'game') and hasattr(env.game, 'debug_manager'):
                 env.game.debug_manager.update_input()

            # Model action
            action, _ = model.predict(obs, deterministic=True)
            
            # --- FREE CAM OVERRIDE ---
            if hasattr(env, 'game') and hasattr(env.game, 'debug_manager'):
                if env.game.debug_manager.free_cam_active:
                    action = 0
            
            step_result = env.step(action)

            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result

            # Render to window
            env.render()

            # Record what is actually on screen
            surface = pygame.display.get_surface()
            
            if surface:
                frame = pygame.surfarray.array3d(surface).swapaxes(0, 1)
                frames.append(frame)

            if done or info.get("episode_end", False):
                episodes += 1
                reset_out = env.reset()
                obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

        env.close()
        pygame.quit()
        print("Session ended.")

        # --- Finalize recording
        if frames:
            videos_root = Path("videos")
            game_dir = videos_root / game
            game_dir.mkdir(parents=True, exist_ok=True)

            name_parts = [game, algo_name, persona if persona else None, skill if skill else None]
            base_name = "_".join(p for p in name_parts if p)

            # Unique filename generator to avoid permission errors or overwrites
            counter = 0
            while True:
                suffix = f"_{counter:02d}" if counter > 0 else ""
                mp4_path = game_dir / f"{base_name}{suffix}.mp4"
                gif_path = game_dir / f"{base_name}{suffix}.gif"
                if not mp4_path.exists() and not gif_path.exists():
                    break
                counter += 1

            print(f"Saving video ({len(frames)} frames) to {mp4_path}...")
            try:
                clip = mpy.ImageSequenceClip(frames, fps=fps)
                clip.write_videofile(str(mp4_path), codec="libx264")
                clip.write_gif(str(gif_path), fps=fps)
                clip.close()

                print(f"\nSaved MP4: {mp4_path}")
                print(f"Saved GIF: {gif_path}\n")
            except Exception as e:
                print(f"Failed to save video: {e}")
                print("Tip: Ensure the file is not open in another player.")

    else:
        # SUBPROCESS MODE (Visual Only - Fixes Freeze)
        # Use this for standard viewing when recording is not required.
        
        env_vars = os.environ.copy()
        if "SDL_VIDEODRIVER" in env_vars:
            del env_vars["SDL_VIDEODRIVER"]

        cmd = [
            sys.executable, "-m", "code.scripts.watch_agent",
            str(model_path),
            "--episodes", str(max_episodes),
            "--fps", str(fps),
            "--game", game,
            "--algo", algo_name
        ]
        
        print("\nLaunching viewer in separate process...")
        print(">>>", " ".join(cmd), "\n")
        
        try:
            subprocess.run(cmd, check=True, env=env_vars)
        except subprocess.CalledProcessError as e:
            print(f"\nViewer exited with error code {e.returncode}")
        except KeyboardInterrupt:
            print("\nViewer stopped by user.")

    print("\nVisualization completed.\n")

def run_tensorboard():
    """Launch TensorBoard with auto-open browser"""
    print("\n=== TensorBoard (auto-open, blocking) ===")
    root = Path(DEFAULT_TB_ROOT)
    if not root.exists():
        print(f"No '{DEFAULT_TB_ROOT}/' folder found yet. Train first or change tb_root in train.")
        return

    games = get_available_games()
    
    # Create a special "Show All" option
    filter_options = ["Show All (no filter)"] + games
    filter_choice = ask_index("Choose TensorBoard filter:", filter_options)
    
    # If user selected Back, return immediately
    if filter_choice is None:
        return
    
    # Determine filter game (None means "Show All")
    if filter_choice == "Show All (no filter)":
        filter_game = None
    else:
        filter_game = filter_choice

    for port in range(6006, 6011):
        cmd = [sys.executable, "-m", "tensorboard.main", "--logdir", str(root), "--port", str(port)]

        print("\nLaunching TensorBoard…")
        print(">>>", " ".join(cmd))
        print(f"\nTensorBoard will run on: http://localhost:{port}/")
        print("Opening browser in a few seconds… (Press Ctrl+C here to stop TensorBoard)\n")

        try:
            tb_proc = subprocess.Popen(cmd)
            time.sleep(3)
            open_browser(f"http://localhost:{port}/")

            if filter_game:
                print(f"\nTip: In TensorBoard's left panel, use the run filter: .*{filter_game}.*")

            tb_proc.wait()
            return

        except KeyboardInterrupt:
            print("\nTensorBoard stopped by user.")
            return
        except FileNotFoundError:
            print("TensorBoard not found. Install with: pip install tensorboard")
            return
        except Exception as e:
            print(f"Error launching TB on port {port}: {e}")

    print("Unable to start TensorBoard on ports 6006–6010.")

def delete_logs_and_models():
    """Permanently delete TensorBoard logs and all trained models"""
    print("\n=== Delete TensorBoard Logs and Models ===")
    confirm = input(
        f"This will permanently delete '{DEFAULT_TB_ROOT}/' and '{MODELS_DIR}/'.\n"
        "Are you sure you want to continue? [y/N]: "
    ).strip().lower()

    if confirm not in ("y", "yes"):
        print("Aborted. Nothing was deleted.")
        return
    
    def safe_clear_dir(path: Path):
        if not path.exists():
            return
        for attempt in range(3):
            try:
                shutil.rmtree(path)
                break
            except Exception as e:
                if attempt == 2:
                    print(f"Failed to delete {path}: {e}")
                    return
                print(f"{path} might be in use. Retrying in 1s…")
                time.sleep(1)
        try:
            path.mkdir(parents=True, exist_ok=True)
            print(f"✓ Cleared and recreated {path}")
        except Exception as e:
            print(f"Deleted {path} but failed to recreate it: {e}")

    safe_clear_dir(Path(DEFAULT_TB_ROOT))
    safe_clear_dir(MODELS_DIR)

    print("\n🧹 All logs and models deleted successfully.\n")

def run_manual_play():
    """Manual gameplay interface"""
    print("\n=== Manual Play Options ===")

    available_games = get_available_games()
    if not available_games:
        print("No game configurations found in grid.yaml or code/conf/game/")
        return

    print("Available games:")
    for i, game in enumerate(available_games, 1):
        print(f"  {i}. Play {game}")
    print(f"  {len(available_games) + 1}. Back to main menu")

    choice = input(f"Select game to play (1-{len(available_games) + 1}): ").strip()
    try:
        idx = int(choice)
    except ValueError:
        print("Invalid selection.")
        return

    if idx == len(available_games) + 1:
        return
    if not (1 <= idx <= len(available_games)):
        print("Invalid selection.")
        return

    selected_game = available_games[idx - 1]

    env = os.environ.copy()
    
    # Remove SDL_VIDEODRIVER to ensure proper window display
    if "SDL_VIDEODRIVER" in env:
        env.pop("SDL_VIDEODRIVER")

    print(f"\n=== Playing {selected_game} manually ===")
    print("Use game controls (e.g., spacebar for Flappy Bird). Press ESC to quit.")

    script_path = Path("code/scripts/manual_play.py")
    if not script_path.exists():
        print("Manual play script not found at code/scripts/manual_play.py")
        return
    
    subprocess.run([sys.executable, "-m", "code.scripts.manual_play", "--game", selected_game, "--fps", "30"], env=env)


def show_project_status():
    """
    Display comprehensive project status
        - Shows how many models of each combination of games_algo_persona
    """
    print("\n=== Project Status ===")
    games = get_available_games()
    algos = get_available_algos_from_grid()
    trained = get_trained_games_from_models_flat()
    
    print(f"Available game configurations (from grid.yaml): {len(games)}")
    for g in games:
        flag = "✓ Trained" if g in trained else "○ Not trained"
        print(f"   {g}: {flag}")

    print(f"\nAvailable algorithms (from grid.yaml): {len(algos)}")
    for algo in algos:
        current_flag = " (current)" if algo == CURRENT_ALGO else ""
        print(f"   {algo}{current_flag}")

    if MODELS_DIR.exists():
        model_count = get_trained_models_count()
        print(f"\nTotal trained models: {model_count}")
        
        model_folders = get_model_folders()
        algo_counts = {}
        for folder in model_folders:
            parts = folder.name.split("_")
            if len(parts) >= 2:
                algo = parts[1]
                algo_counts[algo] = algo_counts.get(algo, 0) + 1
        
        if algo_counts:
            print("\nModels by algorithm:")
            for algo, count in sorted(algo_counts.items()):
                print(f"   {algo}: {count}")
    else:
        print("\n✗ No models directory found")

    if CONF_ROOT.exists():
        print("\nConfiguration status:")
        for sub in ["algo"]:
            p = CONF_ROOT / sub
            n = len(list(p.glob("*.yaml"))) if p.exists() else 0
            print(f"   {sub}: {n} configs")
    else:
        print("\n✗ Configuration directory not found")
    print()

# # depretiatied
# def change_algorithm():
#     """Manually change the currently selected algorithm"""
#     global CURRENT_ALGO
    
#     print("\n=== Change Current Algorithm ===")
#     algos = get_available_algos_from_grid()
#     if not algos:
#         print("No algorithm configurations found in grid.yaml with matching YAML files")
#         return
    
#     ensure_current_algo()
    
#     print(f"\nCurrent algorithm: {CURRENT_ALGO}")
#     print("\nAvailable algorithms:")
#     for i, algo in enumerate(algos, 1):
#         current_flag = " (current)" if algo == CURRENT_ALGO else ""
#         print(f"  {i}. {algo}{current_flag}")
#     print(f"  {len(algos) + 1}. Back")
    
#     choice = input(f"Select new algorithm (1-{len(algos) + 1}): ").strip()
    
#     try:
#         num = int(choice)
#         if num == len(algos) + 1:
#             return
#         if 1 <= num <= len(algos):
#             new_algo = algos[num - 1]
#             CURRENT_ALGO = new_algo
#             print(f"\n✓ Algorithm changed to: {CURRENT_ALGO}\n")
#         else:
#             print("Invalid selection.")
#     except ValueError:
#         print("Invalid selection.")

def watch_random_agent():
    """Watch a random agent; optionally record the actual interactive session."""
    print("\n=== Watch Random Agent Play ===")

    available_games = get_available_games()
    if not available_games:
        print("No game configurations found in grid.yaml or code/conf/game/")
        return

    print("Available games:")
    for i, game in enumerate(available_games, 1):
        print(f"  {i}. {game}")
    print(f"  {len(available_games) + 1}. Back to main menu")

    choice = input(f"Select game to watch (1-{len(available_games) + 1}): ").strip()
    try:
        idx = int(choice)
    except ValueError:
        print("Invalid selection.")
        return

    if idx == len(available_games) + 1:
        return
    if not (1 <= idx <= len(available_games)):
        print("Invalid selection.")
        return

    selected_game = available_games[idx - 1]
    print(f"\n=== Watching random agent in {selected_game} ===")

    fps_input = input("Rendering FPS? [30]: ").strip()
    fps = int(fps_input) if fps_input.isdigit() and int(fps_input) > 0 else 30

    max_ep_input = input("Max episodes to play before auto-exit? [0 = only when you exit manually]: ").strip()
    max_episodes = int(max_ep_input) if max_ep_input.isdigit() and int(max_ep_input) >= 0 else 0

    # Recording?
    record = False
    frames = []
    mpy = None

    rec_choice = input("Record this session to videos/ as MP4 + GIF? [y/N]: ").strip().lower()
    if rec_choice in ("y", "yes"):
        mpy = get_moviepy_editor()
        if mpy is None:
            print("Recording random agent requires 'moviepy'. Run inside the venv and: pip install moviepy imageio[ffmpeg]")
        else:
            record = True

    # --- Setup env
    try:
        from code.wrappers.generic_env import GameEnv
    except ImportError as e:
        print(f"Could not import GameEnv: {e}")
        return

    try:
        game_mod = importlib.import_module(f"code.games.{selected_game}_core")
    except ImportError as e:
        print(f"Could not import game core for '{selected_game}': {e}")
        return

    GameCls = None
    for attr in dir(game_mod):
        if attr.endswith("Core"):
            GameCls = getattr(game_mod, attr)
            break
    if GameCls is None:
        print(f"No *Core class found in code.games.{selected_game}_core.")
        return
    
    pygame.init()

    # IMPORTANT: human mode so you see it
    env = GameEnv(GameCls, render_mode="human", fps=fps)
    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    clock = pygame.time.Clock()
    running = True
    episodes = 0

    print("Press ESC or close the window to end the session.")

    while running and (max_episodes == 0 or episodes < max_episodes):
        clock.tick(fps)

        # Handle quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                running = False

        if not running:
            break

        # Proper random action from env's action space
        action = env.action_space.sample()
        step_result = env.step(action)

        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            obs, reward, done, info = step_result

        env.render()

        if record:
            surface = pygame.display.get_surface()
            if surface:
                frame = pygame.surfarray.array3d(surface).swapaxes(0, 1)
                frames.append(frame)

        if info.get("episode_end", False) or (done if "done" in locals() else False):
            episodes += 1
            reset_out = env.reset()
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    env.close()
    pygame.quit()
    print("Session ended.")

    if record and frames:
        videos_root = Path("videos")
        game_dir = videos_root / selected_game
        game_dir.mkdir(parents=True, exist_ok=True)

        base_name = f"{selected_game}_random"
        mp4_path = game_dir / f"{base_name}.mp4"
        gif_path = game_dir / f"{base_name}.gif"

        clip = mpy.ImageSequenceClip(frames, fps=fps)
        clip.write_videofile(str(mp4_path), codec="libx264")
        clip.write_gif(str(gif_path), fps=fps)
        clip.close()

        print(f"\nSaved MP4: {mp4_path}")
        print(f"Saved GIF: {gif_path}\n")

# ============================================================================
# MAIN MENU
# ============================================================================

def main():
    import warnings
    warnings.filterwarnings(
        "ignore",
        message=r".*pkg_resources is deprecated as an API.*"
    )

    global CURRENT_ALGO
    
    while True:
        setup_project()
        
        print("=" * 60)
        print("MULTI-GAME RL TRAINING & EVALUATION MENU")
        print("=" * 60)
        
        games = get_available_games()
        algos = get_available_algos_from_grid()
        trained_games = get_trained_games_from_models_flat()
        trained_models = get_trained_models_count()
        

        
        print(f"Games: {len(games)} | Algorithms: {len(algos)} | Trained games: {len(trained_games)} | Trained models: {trained_models}")

        print("\nOptions:")
        print("1. Show Detailed Project Status")
        print("2. Run Training (pick Game, Algorithm, Persona, Skill)")
        #print("2. Run Evaluation (per-game, scans models/*.zip)") - Not Adapted Yet
        print("3. Train All Models for One Game")
        print("4. Train Complete Grid (all games × algos × personas)")
        print("5. Play Game Manually (keyboard)")
        print("6. Watch Trained Agent Play (visualize AI performance)")
        print("7. Watch Random Agent Play (random actions)")
        print("8. View TensorBoard Logs (mylogs/)")
        print("9. Delete TensorBoard Logs & Models")
        print("10. Exit")
        print("=" * 60)

        choice = input("Select option (1-10): ").strip()
        
        if choice == "1":
            show_project_status()
        # elif choice == "2":
        #     run_evaluation() -> Not Adapted Yet
        elif choice == "2":
            run_training()
        elif choice == "3":
            train_all_models_for_game()
        elif choice == "4":
            train_complete_grid()
        elif choice == "5":
            run_manual_play()
        elif choice == "6":
            watch_trained_agent()
        elif choice == "7":
            watch_random_agent()
        elif choice == "8":
            run_tensorboard()
        elif choice == "9":
            delete_logs_and_models()
        elif choice == "10":
            print("Exiting. Happy training!")
            break
        else:
            print("Invalid selection. Please choose 1-10.\n")

if __name__ == "__main__":
    main()