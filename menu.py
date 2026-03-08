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

# Enable ANSI escape codes on Windows 10+ terminals
if sys.platform == "win32":
    try:
        os.system("")  # triggers VT100 mode in cmd.exe / powershell
    except Exception:
        pass


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

# Architecture metadata used by the menu
_ARCH_INFO = {
    "light": ("LightCombinedExtractor", "~18K params  — no channel split, fast sweeps"),
    "slim":  ("SlimPEAKExtractor",       "~77K params  — channel split, no SEBlock  [recommended]"),
    "peak":  ("PEAKExtractor",           "~922K params — full SEBlock + deep semantic branch"),
}

def get_available_architectures_from_grid():
    """Return architectures list from grid.yaml, falling back to all three if absent."""
    cfg = load_grid_config()
    if cfg is not None and "architectures" in cfg and cfg.architectures:
        return list(cfg.architectures)
    return ["light", "slim", "peak"]


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


# ── Shared theme helpers (defined here so they are available before LOGO) ──────
_SUPPORTS_COLOR = (
    hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    and os.environ.get("NO_COLOR") is None
)
def _c(code, text):
    return f"\033[{code}m{text}\033[0m" if _SUPPORTS_COLOR else text

_DIM  = lambda t: _c("2",    t)
_BOLD = lambda t: _c("1",    t)
_CYAN = lambda t: _c("96",   t)
_MAG  = lambda t: _c("95",   t)
_YEL  = lambda t: _c("93",   t)
_GRN  = lambda t: _c("92",   t)
_RED  = lambda t: _c("91",   t)
_WHT  = lambda t: _c("97",   t)


def _sub_header(title: str):
    """Themed section header for sub-menus."""
    bar = "─" * 54
    print()
    print(f"    {_RED('▌')} {_BOLD(_WHT(title.upper()))}")
    print(f"    {_DIM(bar)}")


def _row(key: str, label: str, hint: str = "") -> str:
    k = _YEL(f"  [{key:>2}]")
    h = _DIM(f"  {hint}") if hint else ""
    return f"{k}  {label}{h}"


def _err(msg: str):
    print(_RED(f"\n    ✖  {msg}"))


def _ok(msg: str):
    print(_GRN(f"\n    ✔  {msg}"))


def ask_index(prompt, options, add_back=True, default=None):
    """Themed numbered picker — returns chosen item or None for back."""
    if not options:
        _err("No options available.")
        return None

    print(f"\n    {_BOLD(_CYAN('▸'))} {_BOLD(prompt)}")
    for i, opt in enumerate(options, 1):
        flag = _DIM("  (default)") if opt == default else ""
        print(f"    {_YEL(f'[{i:>2}]')}  {opt}{flag}")
    if add_back:
        print(f"    {_YEL(f'[{len(options)+1:>2}]')}  {_DIM('Back')}")

    hint = f" or Enter for [{default}]" if default else ""
    raw = input(_BOLD(f"    ⟫{hint} ")).strip()

    if raw == "" and default:
        return default
    try:
        n = int(raw)
        if add_back and n == len(options) + 1:
            return None
        if 1 <= n <= len(options):
            return options[n - 1]
    except ValueError:
        pass
    _err(f"Invalid selection: '{raw}'")
    return None


def _toggle_picker(prompt: str, options: list, defaults: list = None,
                   labels: dict = None) -> list:
    """
    Interactive multi-select toggle picker.
    Type numbers to toggle items on/off. Enter with no input confirms.
    Returns list of selected items (may be empty).
    """
    selected = set(defaults or options)  # default: all on
    labels = labels or {}

    while True:
        print(f"\n    {_BOLD(_CYAN('▸'))} {_BOLD(prompt)}")
        print(f"    {_DIM('Type numbers to toggle · Enter to confirm')}")
        for i, opt in enumerate(options, 1):
            tick  = _GRN("✓") if opt in selected else _DIM("○")
            extra = _DIM(f"  {labels[opt]}") if opt in labels else ""
            print(f"    {_YEL(f'[{i}]')}  {tick}  {opt}{extra}")

        raw = input(_BOLD("    ⟫ ")).strip()
        if raw == "":
            break
        if raw.lower() == "a":
            selected = set(options)
            continue
        if raw.lower() == "n":
            selected = set()
            continue
        for tok in raw.replace(",", " ").split():
            try:
                idx = int(tok) - 1
                if 0 <= idx < len(options):
                    opt = options[idx]
                    if opt in selected:
                        selected.discard(opt)
                    else:
                        selected.add(opt)
            except ValueError:
                pass

    return [o for o in options if o in selected]


def ensure_current_algo():
    """Ensure CURRENT_ALGO is set, defaulting to PPO if available"""
    global CURRENT_ALGO
    if CURRENT_ALGO is None:
        algos = get_available_algos_from_grid()
        if algos:
            CURRENT_ALGO = "ppo" if "ppo" in algos else algos[0]


# ============================================================================
# TRAINING EXECUTION - Core Logic (DRY refactor)
# ============================================================================

def execute_training_run(game, algo, persona, skill, tb_root=DEFAULT_TB_ROOT, architecture=None):
    """Execute a single training run with error handling"""
    cmd = [
        sys.executable, "-m", "code.scripts.train",
        f"+game={game}",
        f"+model={algo}",
        f"+persona={persona}",
        f"+skill={skill}",
        f"tb_root={tb_root}",
    ]
    if architecture:
        cmd.append(f"+architecture={architecture}")

    print("\n    " + _DIM(">>> " + " ".join(cmd)))
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        _err(f"Failed: {game} | {algo} | {persona} | {skill}  (exit {e.returncode})")
        return False
    except KeyboardInterrupt:
        raise


def print_training_summary(total, successful, failed):
    """Themed training summary."""
    print()
    print(f"    {_DIM('─' * 54)}")
    print(f"    {_BOLD(_WHT('TRAINING COMPLETE'))}")
    print(f"    {_GRN(f'✔  Successful: {successful}/{total}')}")
    if failed:
        print(f"    {_RED(f'✖  Failed:     {failed}/{total}')}")
    print(f"    {_DIM(f'Logs   → {DEFAULT_TB_ROOT}/')}")
    print(f"    {_DIM(f'Models → {MODELS_DIR}/best/')}")
    print(f"    {_DIM('─' * 54)}")
    if HAS_WINSOUND and failed == 0:
        winsound.PlaySound("chime.wav", winsound.SND_FILENAME)
    print()

# ============================================================================
# MAIN ACTIONS
# ============================================================================

def run_training():
    """Single training run — toggle personas, skills, architectures."""
    global CURRENT_ALGO
    _sub_header("Train Single Run")

    games = get_available_games()
    if not games:
        _err("No games found in grid.yaml"); return
    game = ask_index("Game", games)
    if game is None: return

    algos = get_available_algos_from_grid()
    if not algos:
        _err("No algorithms found in grid.yaml"); return
    default_algo = CURRENT_ALGO if CURRENT_ALGO in algos else ("ppo" if "ppo" in algos else algos[0])
    algo_choice = ask_index("Algorithm", algos, default=default_algo)
    if algo_choice is None: return
    CURRENT_ALGO = algo_choice

    personas = get_personas_for_game(game)
    if not personas:
        _err(f"No personas found for game='{game}'"); return

    selected_personas = _toggle_picker(
        "Personas  (toggle · Enter to confirm)",
        personas,
        defaults=personas[:1],          # first one on by default
    )
    if not selected_personas:
        _err("No personas selected."); return

    selected_skills = _toggle_picker(
        "Skills  (toggle · Enter to confirm)",
        ["Novice", "Expert"],
        defaults=["Novice"],
    )
    if not selected_skills:
        _err("No skills selected."); return

    arch_labels = {a: _ARCH_INFO[a][1] if a in _ARCH_INFO else "" for a in get_available_architectures_from_grid()}
    selected_archs = _toggle_picker(
        "Architectures  (toggle · Enter to confirm)",
        get_available_architectures_from_grid(),
        defaults=["slim"],
        labels=arch_labels,
    )
    if not selected_archs:
        _err("No architectures selected."); return

    total = len(selected_personas) * len(selected_skills) * len(selected_archs)
    print()
    print(f"    {_DIM('─' * 54)}")
    print(f"    {_BOLD(_WHT('Run Summary'))}")
    print(f"    Game        : {_YEL(game)}  |  Algo: {_YEL(algo_choice)}")
    print(f"    Personas    : {_GRN(', '.join(selected_personas))}")
    print(f"    Skills      : {_GRN(', '.join(selected_skills))}")
    print(f"    Archs       : {_GRN(', '.join(selected_archs))}")
    print(f"    Total runs  : {_WHT(str(total))}")
    print(f"    {_DIM('─' * 54)}")

    confirm = input(_BOLD("    Proceed? [y/N] ⟫ ")).strip().lower()
    if confirm not in ('y', 'yes'):
        print(_DIM("    Aborted.")); return

    completed, failed = 0, 0
    try:
        for persona in selected_personas:
            for skill in selected_skills:
                for arch in selected_archs:
                    completed += 1
                    print(f"\n    {_CYAN(f'[{completed}/{total}]')}  {game} | {algo_choice} | {persona} | {skill} | {arch}")
                    ok = execute_training_run(game, algo_choice, persona, skill, architecture=arch)
                    if not ok: failed += 1
    except KeyboardInterrupt:
        print(_RED("\n    Interrupted."))

    print_training_summary(total, completed - failed, failed)
    if HAS_WINSOUND and failed == 0:
        winsound.PlaySound("chime.wav", winsound.SND_FILENAME)


def train_all_models_for_game():
    """Train all architectures × algos × personas × skills for ONE game."""
    global CURRENT_ALGO
    _sub_header("Train All — One Game")

    games = get_available_games()
    if not games:
        _err("No games found in grid.yaml"); return
    game = ask_index("Game", games)
    if game is None: return

    algos = get_available_algos_from_grid()
    if not algos:
        _err("No algorithms found"); return
    ensure_current_algo()

    # Algorithm toggle
    selected_algos = _toggle_picker("Algorithms", algos, defaults=algos)
    if not selected_algos:
        _err("No algorithms selected."); return

    personas = get_personas_for_game(game)
    if not personas:
        _err(f"No personas for game='{game}'"); return

    skills = ["Novice", "Expert"]
    archs  = get_available_architectures_from_grid()
    total  = len(selected_algos) * len(archs) * len(personas) * len(skills)

    print()
    print(f"    {_DIM('─' * 54)}")
    print(f"    {_BOLD(_WHT('Sweep Summary'))}")
    print(f"    Game          : {_YEL(game)}")
    print(f"    Algorithms    : {_GRN(', '.join(selected_algos))}")
    print(f"    Architectures : {_GRN(', '.join(archs))}  {_DIM('(all — from grid.yaml)')}")
    print(f"    Personas      : {_GRN(str(len(personas)))} persona(s)")
    print(f"    Skills        : {_GRN('Novice + Expert')}")
    print(f"    Total runs    : {_WHT(str(total))}")
    print(f"    {_DIM('─' * 54)}")

    confirm = input(_BOLD("    Proceed? [y/N] ⟫ ")).strip().lower()
    if confirm not in ('y', 'yes'):
        print(_DIM("    Aborted.")); return

    outer = len(selected_algos) * len(personas) * len(skills)
    completed, failed = 0, 0
    try:
        for algo in selected_algos:
            for persona in personas:
                for skill in skills:
                    completed += 1
                    print(f"\n    {_CYAN(f'[{completed}/{outer}]')} (×{len(archs)} archs)  {game} | {algo} | {persona} | {skill}")
                    ok = execute_training_run(game, algo, persona, skill, architecture=None)
                    if not ok: failed += 1
    except KeyboardInterrupt:
        print(_RED("\n    Interrupted."))

    print_training_summary(total, completed - failed, failed)


def train_complete_grid():
    """Train ALL games × algos × architectures × personas × skills."""
    _sub_header("Train Complete Grid")

    games = get_available_games()
    algos = get_available_algos_from_grid()
    archs = get_available_architectures_from_grid()

    if not games: _err("No games found."); return
    if not algos: _err("No algorithms found."); return

    breakdown = []
    total = 0
    for game in games:
        personas = get_personas_for_game(game)
        if not personas: continue
        n = len(algos) * len(archs) * len(personas) * 2
        total += n
        breakdown.append((game, personas, n))

    if total == 0:
        _err("No valid training configurations found."); return

    print()
    print(f"    {_DIM('─' * 54)}")
    print(f"    {_BOLD(_WHT('Full Grid Summary'))}")
    print(f"    Algorithms    : {_GRN(', '.join(algos))}")
    print(f"    Architectures : {_GRN(', '.join(archs))}  {_DIM('(all)')}")
    print(f"    Skills        : {_GRN('Novice + Expert')}")
    print()
    for game, personas, n in breakdown:
        print(f"    {_YEL(f'  {game:<16}')} {len(personas)} persona(s) × {len(algos)} algo(s) × {len(archs)} arch(s) × 2 skills = {_WHT(str(n))}")
    print(f"    {_DIM('─' * 54)}")
    print(f"    Total runs    : {_WHT(str(total))}")
    print(f"    {_DIM('─' * 54)}")

    confirm = input(_BOLD("    Proceed? [y/N] ⟫ ")).strip().lower()
    if confirm not in ('y', 'yes'):
        print(_DIM("    Aborted.")); return

    outer = sum(len(algos) * len(get_personas_for_game(g) or []) * 2 for g in games)
    completed, failed = 0, 0
    skills = ["Novice", "Expert"]
    try:
        for game in games:
            personas = get_personas_for_game(game)
            if not personas: continue
            for algo in algos:
                for persona in personas:
                    for skill in skills:
                        completed += 1
                        print(f"\n    {_CYAN(f'[{completed}/{outer}]')} (×{len(archs)} archs)  {game} | {algo} | {persona} | {skill}")
                        ok = execute_training_run(game, algo, persona, skill, architecture=None)
                        if not ok: failed += 1
    except KeyboardInterrupt:
        print(_RED("\n    Interrupted."))

    print_training_summary(total, completed - failed, failed)


# Currently NOT IN USE - needs adaptation
def run_evaluation():
    """Evaluate all trained models for a selected game"""
    _sub_header("Evaluation")

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

def run_agent_analyzer():
    """Run the CSV log analyzer script"""
    _sub_header("Agent Performance Analyzer")
    script_path = Path("code/scripts/agent_analyzer.py")

    # Check if it exists in code/scripts or root
    if script_path.exists():
        cmd = [sys.executable, str(script_path)]
    elif Path("agent_analyzer.py").exists():
        cmd = [sys.executable, "agent_analyzer.py"]
    else:
        _err("Cannot find agent_analyzer.py — check root or code/scripts/")
        return

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        pass
    _ok("Analysis complete.")

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
    _sub_header("Watch Trained Agent Play")

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
        _err("Invalid selection.")
        return

    # Infer metadata
    meta = parse_model_metadata(model_path)
    game = meta["game"]
    algo_name = (meta["algo"] or "").lower()
    persona = (meta["persona"] or "default").lower()
    skill = (meta["skill"] or "").lower()

    if not game or not algo_name:
        _err("Could not infer game/algo from model path.")
        return

    algo_cls = ALGO_CLASS_MAP.get(algo_name)

    if algo_cls is None:
        _err(f"Unsupported/unknown algo '{algo_name}'.")
        return

    # FPS + episode cap
    fps_input = input(_BOLD("    FPS [30] ⟫ ")).strip()
    fps = int(fps_input) if fps_input.isdigit() and int(fps_input) > 0 else 30

    max_ep_input = input(_BOLD("    Max episodes (0 = run until ESC) [10] ⟫ ")).strip()
    max_episodes = int(max_ep_input) if max_ep_input.isdigit() and int(max_ep_input) >= 0 else 10

    rec_choice = input(_BOLD("    Record to videos/ as MP4+GIF? [y/N] ⟫ ")).strip().lower()

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
            "--algo", algo_name,
            "--persona", persona,
        ]

        print("\nLaunching viewer in separate process...")
        print(">>>", " ".join(cmd), "\n")

        try:
            subprocess.run(cmd, check=True, env=env_vars)
        except subprocess.CalledProcessError as e:
            print(f"\nViewer exited with error code {e.returncode}")
        except KeyboardInterrupt:
            print(_DIM("\n    Viewer stopped."))

    _ok("Visualization completed.")

def watch_all_models():
    """Launch the grid viewer to watch ALL trained models simultaneously."""
    _sub_header("Watch All Models (Grid View)")

    # Quick check for models
    from pathlib import Path as _P
    best_dir = _P("models/best")
    if not best_dir.exists():
        _err("No models/best/ directory found. Train some models first.")
        return

    folders = [f for f in best_dir.iterdir()
               if f.is_dir() and (f / "best_model.zip").exists()]
    if not folders:
        _err("No trained models found in models/best/.")
        return

    print(f"    {_GRN(str(len(folders)))} trained model(s) found.")

    fps_input = input(_BOLD("    FPS [20] ⟫ ")).strip()
    fps = int(fps_input) if fps_input.isdigit() and int(fps_input) > 0 else 20

    ep_input = input(_BOLD("    Episodes per agent [5] ⟫ ")).strip()
    episodes = int(ep_input) if ep_input.isdigit() and int(ep_input) >= 0 else 5

    env_vars = os.environ.copy()
    env_vars.pop("SDL_VIDEODRIVER", None)

    cmd = [
        sys.executable, "watch_all.py",
        "--fps", str(fps),
        "--episodes", str(episodes),
    ]

    cmd_str = " ".join(cmd)
    print(f"\n    {_DIM('>>>')} {_DIM(cmd_str)}")

    try:
        subprocess.run(cmd, check=True, env=env_vars)
    except subprocess.CalledProcessError as e:
        _err(f"Grid viewer exited with code {e.returncode}")
    except KeyboardInterrupt:
        print(_DIM("\n    Grid viewer stopped."))

    _ok("Grid view completed.")

def run_tensorboard():
    """Launch TensorBoard with auto-open browser"""
    _sub_header("TensorBoard")
    _ = None  # (auto-open, blocking) ===")
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
    _sub_header("⚠  Delete All Logs & Models")
    print(f"    {_RED('WARNING: This permanently deletes all training data!')}")
    print(f"    {_DIM(f'Targets: {DEFAULT_TB_ROOT}/  ·  {MODELS_DIR}/')}")
    confirm = input(_BOLD("    Type DELETE to confirm, or Enter to cancel ⟫ ")).strip()

    if confirm != "DELETE":
        print(_DIM("    Aborted. Nothing was deleted."))
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
                    _err(f"Failed to delete {path}: {e}")
                    return
                print(f"    {_YEL(f'{path} in use, retrying...')}")
                time.sleep(1)
        try:
            path.mkdir(parents=True, exist_ok=True)
            print(f"    {_GRN('✔')}  Cleared → {path}")
        except Exception as e:
            print(f"    {_DIM(f'Deleted {path} but failed to recreate: {e}')}")

    safe_clear_dir(Path(DEFAULT_TB_ROOT))
    safe_clear_dir(MODELS_DIR)
    safe_clear_dir(Path("csv"))

    _ok("All logs and models deleted.")

def run_manual_play():
    """Manual gameplay interface"""
    _sub_header("Play Manually")

    available_games = get_available_games()
    if not available_games:
        _err("No game configurations found in grid.yaml"); return

    selected_game = ask_index("Game", available_games)
    if selected_game is None: return

    env = os.environ.copy()

    # Remove SDL_VIDEODRIVER to ensure proper window display
    if "SDL_VIDEODRIVER" in env:
        env.pop("SDL_VIDEODRIVER")

    print(f"\n    {_DIM('Playing')} {_YEL(selected_game)} {_DIM('— ESC to quit')}") 

    script_path = Path("code/scripts/manual_play.py")
    if not script_path.exists():
        _err("Manual play script not found at code/scripts/manual_play.py")
        return

    subprocess.run([sys.executable, "-m", "code.scripts.manual_play", "--game", selected_game, "--fps", "30"], env=env)


def show_project_status():
    """Themed project status display."""
    _sub_header("Project Status")

    games         = get_available_games()
    algos         = get_available_algos_from_grid()
    archs         = get_available_architectures_from_grid()
    trained       = get_trained_games_from_models_flat()
    model_folders = get_model_folders()
    model_count   = get_trained_models_count()

    print(f"    {_BOLD(_CYAN('▸ Games'))}  ({len(games)} configured)")
    for g in games:
        flag = _GRN("✔  trained") if g in trained else _DIM("○  not trained")
        print(f"    {_YEL('   ·')}  {g:<20}  {flag}")

    print(f"\n    {_BOLD(_CYAN('▸ Algorithms'))}  ({len(algos)} available)")
    for a in algos:
        cur = _YEL("  ← current") if a == CURRENT_ALGO else ""
        print(f"    {_YEL('   ·')}  {a}{cur}")

    print(f"\n    {_BOLD(_CYAN('▸ Architectures'))}  ({len(archs)} available)")
    for a in archs:
        name, desc = _ARCH_INFO.get(a, (a, ""))
        print(f"    {_YEL('   ·')}  {a:<6}  {_DIM(name)}  {_DIM(desc)}")

    print(f"\n    {_BOLD(_CYAN('▸ Trained Models'))}  ({model_count} total)")
    if model_folders:
        algo_counts = {}
        for f in model_folders:
            parts = f.name.split("_")
            k = parts[1] if len(parts) >= 2 else "?"
            algo_counts[k] = algo_counts.get(k, 0) + 1
        for algo, cnt in sorted(algo_counts.items()):
            bar = _GRN("█" * min(cnt, 20)) + _DIM("░" * (20 - min(cnt, 20)))
            print(f"    {_YEL('   ·')}  {algo:<8}  {bar}  {cnt}")
    else:
        print(f"    {_DIM('   No models trained yet.')}")

    if CONF_ROOT.exists():
        n_algo = len(list((CONF_ROOT / "algo").glob("*.yaml"))) if (CONF_ROOT / "algo").exists() else 0
        print(f"\n    {_BOLD(_CYAN('▸ Config'))}  algo: {n_algo} YAML files")
    print()


def watch_random_agent():
    """Watch a random agent; optionally record the actual interactive session."""
    _sub_header("Watch Random Agent")

    available_games = get_available_games()
    if not available_games:
        _err("No game configurations found in grid.yaml")
        return

    selected_game = ask_index("Game", available_games)
    if selected_game is None: return

    print(f"\n    {_DIM('Watching random agent in')} {_YEL(selected_game)}")

    fps_input = input(_BOLD("    FPS [30] ⟫ ")).strip()
    fps = int(fps_input) if fps_input.isdigit() and int(fps_input) > 0 else 30

    max_ep_input = input(_BOLD("    Max episodes (0 = run until ESC) [0] ⟫ ")).strip()
    max_episodes = int(max_ep_input) if max_ep_input.isdigit() and int(max_ep_input) >= 0 else 0

    record = False
    frames = []
    mpy = None

    rec_choice = input(_BOLD("    Record to videos/ as MP4+GIF? [y/N] ⟫ ")).strip().lower()
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

def clear_cli():
    """Clear the terminal screen cross-platform."""
    os.system("cls" if sys.platform == "win32" else "clear")

def run_level_editor():
    """Launch the PEAK level editor (pygame-based tile painter)."""
    _sub_header("Level Editor")
    print(f"    {_DIM('Launching PEAK Level Editor...')}") 

    # Locate the script — check both code/scripts and project root
    candidates = [
        Path("code/scripts/level_editor.py"),
        Path("code/games/platformer/level_editor.py"),
        Path("level_editor.py"),
    ]

    editor_path = None
    for p in candidates:
        if p.exists():
            editor_path = p
            break

    if editor_path is None:
        print("  Could not find level_editor.py.")
        print("  Checked: " + ", ".join(str(c) for c in candidates))
        return

    # Optionally open an existing level file
    levels_dir = Path("code/games/platformer/levels")
    level_files = sorted(levels_dir.glob("*.txt")) if levels_dir.exists() else []

    if level_files:
        print(f"\n    {_BOLD(_CYAN('▸'))} {_BOLD('Existing levels')}")
        print(f"    {_YEL('[  0]')}  New blank level")
        for i, lf in enumerate(level_files, 1):
            print(f"    {_YEL(f'[{i:>3}]')}  {lf.name}")
        print(f"    {_YEL(f'[{len(level_files)+1:>3}]')}  {_DIM('Back')}")

        pick = input(_BOLD(f"\n    Open level (0–{len(level_files)+1}) ⟫ ")).strip()
        try:
            n = int(pick)
            if n == len(level_files) + 1:
                return
            if n == 0:
                # new blank
                cmd = [sys.executable, str(editor_path)]
            elif 1 <= n <= len(level_files):
                cmd = [sys.executable, str(editor_path), str(level_files[n - 1])]
            else:
                print("  Invalid selection.")
                return
        except ValueError:
            print("  Invalid selection.")
            return
    else:
        cmd = [sys.executable, str(editor_path)]

    env_vars = os.environ.copy()
    env_vars.pop("SDL_VIDEODRIVER", None)

    print("  >>>", " ".join(cmd), "\n")
    try:
        subprocess.run(cmd, check=True, env=env_vars)
    except subprocess.CalledProcessError as e:
        print(f"\n  Editor exited with error code {e.returncode}")
    except KeyboardInterrupt:
        print("\n  Editor closed.")


# ============================================================================
# MAIN MENU — PEAK ENGINE
# ============================================================================

# (ANSI helpers defined earlier — shared across all menus)

LOGO = ("""
    ██████╗ ███████╗ █████╗ ██╗  ██╗
    ██╔══██╗██╔════╝██╔══██╗██║ ██╔╝
    ██████╔╝█████╗  ███████║█████╔╝
    ██╔═══╝ ██╔══╝  ██╔══██║██╔═██╗
    ██║     ███████╗██║  ██║██║  ██╗
    ╚═╝     ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝
""")

SUB_LOGO = "         E  N  G  I  N  E   By AL and Kevin"


def _print_header(games, algos, trained_games, trained_models):
    """Print the PEAK ENGINE banner + live stats bar."""
    W = 58

    print()
    for line in LOGO.splitlines():
        if line.strip():
            print(_RED(line))

    print(_DIM("    ─" * 11))
    print(_MAG(SUB_LOGO))
    print(_DIM("    ─" * 11))

    # Stats bar
    g_str = _WHT(f"{len(games)}")
    a_str = _WHT(f"{len(algos)}")
    tg_str = _GRN(f"{len(trained_games)}") if trained_games else _DIM("0")
    tm_str = _GRN(f"{trained_models}") if trained_models else _DIM("0")

    print()
    print(f"    Games {g_str}  │  Algos {a_str}  │  Trained {tg_str}  │  Models {tm_str}")
    print(_DIM("    " + "─" * (W - 4)))


def _menu_item(key: str, label: str, hint: str = "") -> str:
    """Format a single menu row."""
    k = _YEL(f"  [{key:>2}]")
    h = _DIM(f"  {hint}") if hint else ""
    return f"{k}  {label}{h}"


def _section(title: str):
    print(f"\n    {_BOLD(_CYAN('▸'))} {_BOLD(title)}")


def main():
    import warnings
    warnings.filterwarnings(
        "ignore",
        message=r".*pkg_resources is deprecated as an API.*"
    )

    global CURRENT_ALGO

    DISPATCH = {
        "1":  ("show_project_status",  show_project_status),
        "2":  ("run_training",         run_training),
        "3":  ("train_all_game",       train_all_models_for_game),
        "4":  ("train_grid",           train_complete_grid),
        "5":  ("manual_play",          run_manual_play),
        "6":  ("watch_agent",          watch_trained_agent),
        "7":  ("watch_all",            watch_all_models),
        "8":  ("watch_random",         watch_random_agent),
        "9":  ("level_editor",         run_level_editor),
        "10": ("tensorboard",          run_tensorboard),
        "11": ("analyzer",             run_agent_analyzer),
        "12": ("delete_all",           delete_logs_and_models),
        "c":  ("clear_cli",            clear_cli),
        "0":  ("exit",                 None),
    }

    while True:
        setup_project()
        clear_cli()

        games          = get_available_games()
        algos          = get_available_algos_from_grid()
        trained_games  = get_trained_games_from_models_flat()
        trained_models = get_trained_models_count()

        _print_header(games, algos, trained_games, trained_models)

        _section("TRAIN")
        print(_menu_item("1",  "Project Status"))
        print(_menu_item("2",  "Train Single",        "game / algo / persona / skill"))
        print(_menu_item("3",  "Train All (1 game)",  "every combo for one game"))
        print(_menu_item("4",  "Train Full Grid",     "all games × algos × personas"))

        _section("PLAY")
        print(_menu_item("5",  "Play Manually",       "keyboard controls"))
        print(_menu_item("6",  "Watch Agent",         "visualize a trained model"))
        print(_menu_item("7",  "Watch All Models",    "side-by-side grid"))
        print(_menu_item("8",  "Watch Random Agent",  "random actions"))

        _section("TOOLS")
        print(_menu_item("9",  "Level Editor",         "paint tiles, place entities"))
        print(_menu_item("10", "TensorBoard",          "mylogs/"))
        print(_menu_item("11", "Analyze Performance",  "CSV log deep-dive"))
        print(_menu_item("12", "Delete Logs & Models", "nuclear option"))
        print(_menu_item(" C", "Clear Screen",         "clear terminal output"))

        print()
        print(f"    {_DIM('───────────────────────────────────')}")
        print(_menu_item("0",  _RED("Exit")))
        print()

        choice = input(_BOLD("    ⟫ ")).strip().lower()

        if choice == "0":
            print()
            print(_DIM("    Shutting down PEAK ENGINE. Happy training!"))
            print()
            break

        entry = DISPATCH.get(choice)
        if entry is None:
            print(_RED(f"\n    Invalid selection: '{choice}'"))
            time.sleep(1.2)
            continue

        _, fn = entry
        if fn is not None:
            try:
                fn()
            except Exception as e:
                print(_RED(f"\n    ✖  Error: {e}"))

            # Pause so the user can read output before the screen is cleared,
            # unless the action was "clear screen" itself.
            if fn is not clear_cli:
                try:
                    input(_DIM("\n    Press Enter to return to menu..."))
                except (EOFError, KeyboardInterrupt):
                    pass

if __name__ == "__main__":
    main()