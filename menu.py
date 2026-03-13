# menu.py — unified CLI for training, evaluation, TensorBoard, and manual play
# Compatible with: Hydra overrides, TB logs under mylogs/, flat model files in models/
#
# Structure overview:
#   1. Imports & constants          — packages, paths, required deps list
#   2. Config helpers               — read grid.yaml, discover games/algos/personas
#   3. UI helpers                   — ANSI palette, toggle_select, ask_index
#   4. Training execution           — execute_training_run, print_training_summary
#   5. Main actions                 — run_training, train_all, train_complete_grid,
#                                     run_manual_play, watch_*, level editor, etc.
#   6. Main menu loop               — DISPATCH table + main()

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
    print(_DIM("  Checking dependencies..."))

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
        print(_GRN("  ✓ All dependencies are installed!"))
        return True

    print(_YEL(f"  Missing packages: {', '.join(missing_packages)}"))

    response = input(_BOLD("\n  ⟫ Install missing dependencies? [y/N]: ")).strip().lower()
    if response not in ('y', 'yes'):
        print(_RED("  Cannot proceed without required dependencies."))
        return False

    print(_DIM("  Installing missing packages..."))
    try:
        cmd = [sys.executable, '-m', 'pip', 'install'] + missing_packages
        subprocess.check_call(cmd)
        print(_GRN("  ✓ Successfully installed all dependencies!"))
        return True
    except subprocess.CalledProcessError as e:
        print(_RED(f"  ✖ Failed to install dependencies: {e}"))
        print(_DIM("  Please install manually using: pip install -r requirements.txt"))
        return False

def setup_project():
    """Initial project setup"""
    # if requirements don't exists makes requirements.txt

    # Check and create requirements.txt if missing
    requirements_path = Path("requirements.txt")
    if not requirements_path.exists():
        requirements_content = "\n".join(REQUIRED_PACKAGES) + "\n"
        requirements_path.write_text(requirements_content)

# Grid config is the single source of truth for which games, algos, and
# personas are active. All get_available_* helpers call this first.
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
    """
    Read the 'games' key from grid.yaml and return a sorted list of game names.
    Falls back to an empty list with a warning if the key is absent or the
    file cannot be parsed. Game names here must match code/games/*_core.py files.
    """
    cfg = load_grid_config()
    if cfg is not None and 'games' in cfg and cfg.games:
        return sorted(list(cfg.games))

    print(_YEL("  Warning: No 'games' section found or it is empty in grid.yaml."))
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

    print(_YEL("  Warning: No 'personas' section found or it is empty in grid.yaml."))
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
    return ["light", "slim", "balanced", "peak"]


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
    print(_DIM(f"  If your browser did not open automatically, go to: {url}"))

def ask_index(prompt, options, add_back=True, default=None):
    """
    Simple numbered-list prompt — used for linear (non-toggle) selections.
    Prints the option list, reads an integer, and returns the chosen item.
    Returns None if the user selects the auto-appended "Back" entry.
    Supports an optional default value selected by pressing Enter.
    """
    if not options:
        print(_DIM("  No options available."))
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

    print(_RED("  ✖  Invalid selection."))
    return None

def ensure_current_algo():
    """Ensure CURRENT_ALGO is set, defaulting to PPO if available"""
    global CURRENT_ALGO

    if CURRENT_ALGO is None:
        algos = get_available_algos_from_grid()
        if algos:
            CURRENT_ALGO = "ppo" if "ppo" in algos else algos[0]

def toggle_select(title, options, default_indices=None, min_select=1, show_desc=None):
    """
    Interactive toggle-style multi-select used throughout the training menus.

    Renders a numbered checklist where each item can be toggled on/off.
    Supports comma-separated input ("1,3") and ranges ("1-3") in one go.
    Returns the confirmed list of selected items, or None if user typed "0" (back).

    Parameters
    ----------
    title         : str         — section header shown above the list
    options       : list[str]   — the items to display
    default_indices: list[int]  — 0-based indices that start toggled ON
    min_select    : int         — minimum items required before confirming
    show_desc     : dict | None — {option: hint_string} for extra detail per row
    
    Type a number to flip that item on/off.  Press Enter to confirm.
    Returns a list of selected items, or None if the user backs out.
    
    show_desc: optional dict {option: description_string} for extra info per row.
    """
    selected = set(default_indices or [0])   # 0-based indices

    while True:
        print(f"\n    {_BOLD(_CYAN('▸'))} {_BOLD(title)}  {_DIM('(toggle · Enter to confirm)')}")
        print(_DIM("    Type numbers to toggle  ·  Enter to confirm  ·  0 = Back"))
        print()

        for i, opt in enumerate(options):
            tick = _GRN("✓") if i in selected else _DIM("o")
            desc = ""
            if show_desc:
                d = show_desc.get(opt, "")
                if d:
                    desc = _DIM(f"  {d}")
            print(f"    {_YEL(f'[{i+1}]')}  {tick}  {opt}{desc}")

        print()
        raw = input(_BOLD("    ⟫ ")).strip()

        if raw == "":
            if len(selected) >= min_select:
                return [options[i] for i in sorted(selected)]
            print(_RED(f"    Please select at least {min_select} item(s)."))
            continue

        if raw == "0":
            return None

        # Support comma-separated and range input: "1,3" or "1-3"
        tokens = []
        for part in raw.replace(" ", "").split(","):
            if "-" in part:
                try:
                    a, b = part.split("-", 1)
                    tokens.extend(range(int(a), int(b) + 1))
                except ValueError:
                    pass
            else:
                try:
                    tokens.append(int(part))
                except ValueError:
                    pass

        for n in tokens:
            idx = n - 1
            if 0 <= idx < len(options):
                if idx in selected:
                    selected.discard(idx)
                else:
                    selected.add(idx)
            else:
                print(_RED(f"    Invalid: {n}"))

# ============================================================================
# TRAINING EXECUTION - Core Logic (DRY refactor)
# ============================================================================

def execute_training_run(game, algo, persona, skill, tb_root=DEFAULT_TB_ROOT, architecture=None):
    """
    Build and run a single Hydra training command via subprocess.

    Constructs the `python -m code.scripts.train` invocation with the
    appropriate +game, +model, +persona, +skill, tb_root, and optionally
    +architecture overrides. Returns True on success, False on non-zero exit.
    KeyboardInterrupt is re-raised so the caller's loop can catch and abort.
    """
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

    print(_DIM("  >>> ") + _WHT(" ".join(cmd)) + "\n")

    # Runs the hydra CMD training script with specified parameters
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(_RED(f"  ✖ Failed: {game} | {algo} | {persona} | {skill} (exit {e.returncode})"))
        return False
    except KeyboardInterrupt:
        raise  # Re-raise to be caught by caller

def print_training_summary(total, successful, failed):
    """Print final training summary with sound notification"""
    print()
    print(_DIM("    ─" * 11))
    print(f"    {_BOLD(_GRN('✓'))} {_BOLD('TRAINING COMPLETE')}")
    print(_DIM("    ─" * 11))
    print(f"    {_DIM('Successful:')}  {_GRN(f'{successful}/{total}')}")
    if failed > 0:
        print(f"    {_DIM('Failed    :')}  {_RED(f'{failed}/{total}')}")
    print(f"    {_DIM('Logs  →')}  {_WHT(DEFAULT_TB_ROOT + '/')}")
    print(f"    {_DIM('Models→')}  {_WHT(str(MODELS_DIR) + '/best/')}")

    if HAS_WINSOUND and failed == 0:
        winsound.PlaySound("chime.wav", winsound.SND_FILENAME)

    print()

# ============================================================================
# MAIN ACTIONS
# ============================================================================

def run_training():
    """Training run — multi-toggle selectors for algo, persona, skill, architecture."""
    global CURRENT_ALGO

    clear_cli()
    _print_header(
        get_available_games(), get_available_algos_from_grid(),
        get_trained_games_from_models_flat(), get_trained_models_count()
    )
    _section("TRAIN  ›  Configure Run")

    # ── Game (single select via toggle) ───────────────────────────
    games = get_available_games()
    if not games:
        print(_RED("  No game configurations found in grid.yaml"))
        return
    game_sel = toggle_select("Game", games, default_indices=[0], min_select=1)
    if game_sel is None:
        return
    game = game_sel[0]      # single game for now

    # ── Algorithm ─────────────────────────────────────────────────
    algos = get_available_algos_from_grid()
    if not algos:
        print(_RED("  No algorithm configurations found"))
        return
    def_algo_idx = next((i for i, a in enumerate(algos) if a == CURRENT_ALGO), 0)
    algo_sel = toggle_select("Algorithm", algos, default_indices=[def_algo_idx], min_select=1)
    if algo_sel is None:
        return
    CURRENT_ALGO = algo_sel[0]

    # ── Personas ──────────────────────────────────────────────────
    personas = get_personas_for_game(game)
    if not personas:
        print(_RED(f"  No personas found for game='{game}'"))
        return
    persona_sel = toggle_select("Personas", personas, default_indices=[0])
    if persona_sel is None:
        return

    # ── Skills ────────────────────────────────────────────────────
    skills_opts = ["Novice", "Expert", "Custom steps"]
    skill_sel = toggle_select("Skills", skills_opts, default_indices=[0])
    if skill_sel is None:
        return

    # Custom steps shortcut — if selected, prompt once
    custom_steps = None
    if "Custom steps" in skill_sel:
        steps_str = input(_BOLD("\n    ⟫ Custom total steps (e.g. 300000): ")).strip()
        try:
            custom_steps = int(steps_str)
        except ValueError:
            print(_RED("  Invalid number."))
            return
        skill_sel = [s for s in skill_sel if s != "Custom steps"]

    # ── Architectures ─────────────────────────────────────────────
    archs = get_available_architectures_from_grid()
    def_arch_idx = next((i for i, a in enumerate(archs) if a == "slim"), 0)
    arch_desc = {a: f"{_ARCH_INFO[a][0]:<28}  {_ARCH_INFO[a][1]}" for a in archs if a in _ARCH_INFO}
    arch_sel = toggle_select("Architecture", archs, default_indices=[def_arch_idx], min_select=1,
                             show_desc=arch_desc)
    if arch_sel is None:
        return

    # ── TensorBoard root ──────────────────────────────────────────
    tb_root = input(_DIM(f"\n    TensorBoard root [{DEFAULT_TB_ROOT}] (Enter to keep): ")).strip() or DEFAULT_TB_ROOT

    # ── Summary ───────────────────────────────────────────────────
    run_skills = list(skill_sel)
    if custom_steps is not None:
        run_skills.append(f"Custom({custom_steps})")

    total = len(arch_sel) * len(algo_sel) * len(persona_sel) * max(len(run_skills), 1)

    print()
    print(f"    {_BOLD(_CYAN('▸'))} {_BOLD('Run Summary')}")
    print(f"    {_DIM('─' * 50)}")
    print(f"    {_DIM('Game      :')}  {_WHT(game)}")
    print(f"    {_DIM('Algos     :')}  {_GRN(', '.join(algo_sel))}")
    personas_short = ', '.join(p.replace(f'{game}_', '') for p in persona_sel)
    print(f"    {_DIM('Personas  :')}  {_GRN(personas_short)}  {_DIM(f'({len(persona_sel)})')}")
    print(f"    {_DIM('Skills    :')}  {_GRN(', '.join(run_skills))}")
    print(f"    {_DIM('Archs     :')}  {_GRN(', '.join(arch_sel))}")
    print(f"    {_DIM('Total runs:')}  {_WHT(str(total))}")
    print(f"    {_DIM('─' * 50)}")
    print()
    confirm = input(_BOLD("    ⟫ Proceed? [Y/n]: ")).strip().lower()
    if confirm in ("n", "no"):
        print(_DIM("  Aborted."))
        return

    # ── Execute ───────────────────────────────────────────────────
    completed = 0; failed = 0
    try:
        for arch in arch_sel:
            for algo in algo_sel:
                for persona in persona_sel:
                    if custom_steps is not None:
                        completed += 1
                        print(_DIM(f"\n  [{completed}/{total}]") + f"  {game} | {algo} | {persona} | Custom {custom_steps} | {arch}")
                        cmd = [
                            sys.executable, "-m", "code.scripts.train",
                            f"+game={game}", f"+model={algo}", f"+persona={persona}",
                            "skill=Custom", f"+skills.Custom={custom_steps}",
                            f"tb_root={tb_root}", f"+architecture={arch}",
                        ]
                        print(">>> " + " ".join(cmd))
                        try:
                            subprocess.run(cmd, check=True)
                        except subprocess.CalledProcessError:
                            failed += 1
                    for skill in skill_sel:
                        completed += 1
                        print(_DIM(f"\n  [{completed}/{total}]") + f"  {game} | {algo} | {persona} | {skill} | {arch}")
                        ok = execute_training_run(game, algo, persona, skill, tb_root, arch)
                        if not ok:
                            failed += 1
    except KeyboardInterrupt:
        print(_RED(f"\n  Interrupted.  Completed {completed-1}/{total}"))
        return

    print_training_summary(total, completed - failed, failed)

def train_all_models_for_game():
    """Train all selected (algo x persona x skill) for ONE game — toggle selectors."""
    global CURRENT_ALGO

    clear_cli()
    _print_header(
        get_available_games(), get_available_algos_from_grid(),
        get_trained_games_from_models_flat(), get_trained_models_count()
    )
    _section("TRAIN ALL  ›  One Game")

    # ── Game ──────────────────────────────────────────────────────
    games = get_available_games()
    if not games:
        print(_RED("  No game configurations found in grid.yaml"))
        return
    game_sel = toggle_select("Game", games, default_indices=[0], min_select=1)
    if game_sel is None:
        return
    game = game_sel[0]

    # ── Algorithms ────────────────────────────────────────────────
    algos = get_available_algos_from_grid()
    if not algos:
        print(_RED("  No algorithm configurations found"))
        return
    ensure_current_algo()
    def_algo_idx = next((i for i, a in enumerate(algos) if a == CURRENT_ALGO), 0)
    algo_sel = toggle_select("Algorithms", algos,
                             default_indices=list(range(len(algos))))  # all ON by default
    if algo_sel is None:
        return
    CURRENT_ALGO = algo_sel[0]

    # ── Personas ──────────────────────────────────────────────────
    personas = get_personas_for_game(game)
    if not personas:
        print(_RED(f"  No personas for game '{game}'"))
        return
    persona_sel = toggle_select("Personas", personas,
                                default_indices=list(range(len(personas))))
    if persona_sel is None:
        return

    # ── Skills ────────────────────────────────────────────────────
    skills_opts = ["Novice", "Expert"]
    skill_sel = toggle_select("Skills", skills_opts,
                              default_indices=[0, 1])   # both ON by default
    if skill_sel is None:
        return

    # ── Architecture ──────────────────────────────────────────────
    archs = get_available_architectures_from_grid()
    def_arch_idx = next((i for i, a in enumerate(archs) if a == "slim"), 0)
    arch_desc = {a: f"{_ARCH_INFO[a][0]:<28}  {_ARCH_INFO[a][1]}" for a in archs if a in _ARCH_INFO}
    arch_sel = toggle_select("Architecture", archs, default_indices=[def_arch_idx], min_select=1,
                             show_desc=arch_desc)
    if arch_sel is None:
        return

    # ── Summary + confirm ─────────────────────────────────────────
    total_runs = len(arch_sel) * len(algo_sel) * len(persona_sel) * len(skill_sel)
    print()
    print(f"    {_BOLD(_CYAN('▸'))} {_BOLD('Run Summary')}")
    print(f"    {_DIM('─' * 50)}")
    print(f"    {_DIM('Game      :')}  {_WHT(game)}")
    print(f"    {_DIM('Algos     :')}  {_GRN(', '.join(algo_sel))}")
    personas_short = ', '.join(p.replace(f'{game}_', '') for p in persona_sel)
    print(f"    {_DIM('Personas  :')}  {_GRN(personas_short)}  {_DIM(f'({len(persona_sel)})')}")
    print(f"    {_DIM('Skills    :')}  {_GRN(', '.join(skill_sel))}")
    print(f"    {_DIM('Archs     :')}  {_GRN(', '.join(arch_sel))}")
    print(f"    {_DIM('Total runs:')}  {_WHT(str(total_runs))}")
    print(f"    {_DIM('─' * 50)}")
    confirm = input(_BOLD("\n    ⟫ Proceed? [Y/n]: ")).strip().lower()
    if confirm in ("n", "no"):
        print(_DIM("  Aborted."))
        return

    # ── Execute ───────────────────────────────────────────────────
    completed = 0; failed = 0
    try:
        for arch in arch_sel:
            for algo in algo_sel:
                for persona in persona_sel:
                    for skill in skill_sel:
                        completed += 1
                        print(_DIM(f"\n  [{completed}/{total_runs}]") + f"  {game} | {algo} | {persona} | {skill} | {arch}")
                        ok = execute_training_run(game, algo, persona, skill, architecture=arch)
                        if not ok:
                            failed += 1
    except KeyboardInterrupt:
        print(_RED(f"\n  Interrupted.  Completed {completed-1}/{total_runs}"))
        return

    print_training_summary(total_runs, completed - failed, failed)

def train_complete_grid():
    """Train ALL (game x algo x persona x skill) combinations"""
    clear_cli()
    _print_header(
        get_available_games(), get_available_algos_from_grid(),
        get_trained_games_from_models_flat(), get_trained_models_count()
    )
    _section("TRAIN FULL GRID  ›  All Games × Algos × Personas")

    games = get_available_games()
    algos = get_available_algos_from_grid()

    if not games:
        print(_RED("  No game configurations found in grid.yaml"))
        return

    if not algos:
        print(_RED("  No algorithm configurations found in grid.yaml"))
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
        breakdown.append(f"    {_WHT(game)}: {len(personas)} persona(s) × {len(algos)} algo(s) × 2 skills = {_GRN(str(runs_for_game))} runs")

    if total_runs == 0:
        print(_RED("\n  No valid training configurations found."))
        return

    print(f"\n    Total runs: {_WHT(str(total_runs))}  across {_WHT(str(len(games)))} game(s)")
    print()
    for line in breakdown:
        print(line)

    # Architecture toggle
    archs = get_available_architectures_from_grid()
    def_arch_idx = next((i for i, a in enumerate(archs) if a == "slim"), 0)
    arch_desc = {a: f"{_ARCH_INFO[a][0]:<28}  {_ARCH_INFO[a][1]}" for a in archs if a in _ARCH_INFO}
    arch_sel = toggle_select("Architecture", archs, default_indices=[def_arch_idx], min_select=1,
                             show_desc=arch_desc)
    if arch_sel is None:
        return

    total_runs *= len(arch_sel)
    confirm = input(_BOLD(f"\n    ⟫ Proceed with {total_runs} runs using [{', '.join(arch_sel)}]? [Y/n]: ")).strip().lower()
    if confirm in ("n", "no"):
        print(_DIM("  Aborted."))
        return

    # Execute training grid
    skills = ["Novice", "Expert"]
    completed = 0
    failed = 0

    try:
        for arch in arch_sel:
            for game in games:
                personas = get_personas_for_game(game)
                if not personas:
                    continue

                for algo in algos:
                    for persona in personas:
                        for skill in skills:
                            completed += 1
                            print(_DIM(f"\n  [{completed}/{total_runs}]") + f"  {game} | {algo} | {persona} | {skill} | {arch}")
                            success = execute_training_run(game, algo, persona, skill, architecture=arch)
                            if not success:
                                failed += 1
    except KeyboardInterrupt:
        print(_RED(f"\n  Interrupted.  Completed {completed-1}/{total_runs}"))
        return

    print_training_summary(total_runs, completed - failed, failed)


# Currently NOT IN USE - needs adaptation
def run_evaluation():
    """Evaluate all trained models for a selected game"""
    _section("EVALUATE  ›  Quick Eval")

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

    _hint = _DIM("— 5 eps each")
    print(f"    {_DIM(chr(8250))}  {_WHT(str(len(model_dirs)))} model(s) for {_GRN(game)}  {_hint}")

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

    print(_GRN("    ✓ Evaluation complete."))


def parse_model_metadata(model_path: Path):
    """
    Reverse-engineers metadata from a best_model folder name.

    Folder names follow the convention:
        {game}_{algo}_{persona}_{skill}_{arch}
    e.g. "platformer_ppo_platformer_simple_novice_slim"

    Returns a dict with keys: game, algo, persona, skill, arch.
    Architecture tag is stripped from the end if it matches a known set.
    """
    folder = model_path.parent.name
    parts = folder.split("_")
    _ARCH_TAGS = {"light", "slim", "balanced", "peak", "mlp"}
    arch = None
    if len(parts) >= 2 and parts[-1].lower() in _ARCH_TAGS:
        arch = parts[-1].lower()
        parts = parts[:-1]
    meta = {
        "game":    parts[0] if len(parts) >= 1 else None,
        "algo":    parts[1] if len(parts) >= 2 else None,
        "persona": "_".join(parts[2:-1]) if len(parts) >= 4 else (parts[2] if len(parts) >= 3 else None),
        "skill":   parts[-1] if len(parts) >= 4 else None,
        "arch":    arch,
    }
    if meta["persona"] and meta["game"] and meta["persona"].startswith(f"{meta['game']}_"):
        meta["persona"] = meta["persona"][len(meta["game"])+1:]
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
    _section("ANALYZE  ›  Agent Performance")
    script_path = Path("code/scripts/agent_analyzer.py")

    # Check if it exists in code/scripts or root
    if script_path.exists():
        cmd = [sys.executable, str(script_path)]
    elif Path("agent_analyzer.py").exists():
        cmd = [sys.executable, "agent_analyzer.py"]
    else:
        print("❌ Cannot find 'agent_analyzer.py'. Make sure it's in the root or code/scripts/ folder.")
        return

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        pass
    print("\nAnalysis complete.\n")

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
    _section("WATCH  ›  Trained Agent")

    model_folders = get_model_folders()
    if not model_folders:
        print("No best_model.zip files found in models/best/.")
        return

    # Build display list
    display_options = []
    paths = []

    for folder in model_folders:
        parts = folder.name.split("_")
        _ARCH_TAGS = {"light", "slim", "balanced", "peak", "mlp"}

        if len(parts) >= 6 and parts[-1].lower() in _ARCH_TAGS:
            game, algo = parts[0], parts[1]
            arch = parts[-1]
            skill = parts[-2]
            persona = "_".join(parts[2:-2])
            if persona.startswith(f"{game}_"):
                persona = persona[len(game)+1:]
            display = f"{game:<12} | {algo:<4} | {persona:<16} | {skill:<8} | {arch}"
        elif len(parts) >= 5:
            game, algo, _, persona, skill = parts[:5]
            display = f"{game:<12} | {algo:<4} | {persona:<16} | {skill:<8}"
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
        print(_RED("  ✖  Invalid selection."))
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
            print("\nViewer stopped by user.")

    print("\nVisualization completed.\n")

def watch_all_models():
    """Launch the grid viewer to watch ALL trained models simultaneously."""
    _section("WATCH  ›  All Models Grid")

    # Quick check for models
    from pathlib import Path as _P
    best_dir = _P("models/best")
    if not best_dir.exists():
        print(_RED("  ✖  No models/best/ directory found.  Train some models first."))
        return

    folders = [f for f in best_dir.iterdir()
               if f.is_dir() and (f / "best_model.zip").exists()]
    if not folders:
        print("No trained models found in models/best/.")
        return

    print(f"Found {len(folders)} trained model(s).")

    fps_input = input("Rendering FPS? [20]: ").strip()
    fps = int(fps_input) if fps_input.isdigit() and int(fps_input) > 0 else 20

    ep_input = input("Episodes per agent? [5]: ").strip()
    episodes = int(ep_input) if ep_input.isdigit() and int(ep_input) >= 0 else 5

    env_vars = os.environ.copy()
    env_vars.pop("SDL_VIDEODRIVER", None)

    cmd = [
        sys.executable, "watch_all.py",
        "--fps", str(fps),
        "--episodes", str(episodes),
    ]

    print(f"\nLaunching grid viewer ({len(folders)} models, {fps} FPS)...")
    print(">>>", " ".join(cmd), "\n")

    try:
        subprocess.run(cmd, check=True, env=env_vars)
    except subprocess.CalledProcessError as e:
        print(f"\nGrid viewer exited with error code {e.returncode}")
    except KeyboardInterrupt:
        print("\nGrid viewer stopped by user.")

    print("\nGrid view completed.\n")

def run_tensorboard():
    """Launch TensorBoard with auto-open browser"""
    _section("TENSORBOARD  ›  Launch")
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
    _section("DANGER  ›  Delete Logs & Models")
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
    safe_clear_dir(Path("csv"))
    safe_clear_dir(Path("sessions"))

    print("\n🧹 All logs and models deleted successfully.\n")

def run_manual_play():
    """Manual gameplay interface — lets the user play any configured game with keyboard controls."""

    # ── Panel width matches the main header (58 chars) ────────────────────────
    W = 58

    def _box_top():
        print(_DIM("    ╔" + "═" * W + "╗"))

    def _box_mid():
        print(_DIM("    ╠" + "═" * W + "╣"))

    def _box_bot():
        print(_DIM("    ╚" + "═" * W + "╝"))

    def _box_row(text="", color_fn=None):
        """Print a single ║-bordered row, centered if no color_fn, left-padded otherwise."""
        if color_fn:
            # Left-aligned content with 2-space indent inside box
            inner = f"  {text}"
            pad   = W - len(inner)
            print(_DIM("    ║") + color_fn(inner) + _DIM(" " * max(pad, 0) + "║"))
        else:
            # Centered plain-dim text
            centered = text.center(W)
            print(_DIM("    ║" + centered + "║"))

    def _box_kv(key: str, val: str, key_w: int = 16):
        """Print a key-value row inside the box."""
        k_part  = _YEL(f"  {key:<{key_w}}")
        v_part  = _WHT(val)
        inner   = f"  {key:<{key_w}}{val}"   # plain for length calc
        pad     = W - 2 - key_w - len(val)
        print(_DIM("    ║") + k_part + v_part + _DIM(" " * max(pad, 0) + "║"))

    def _box_key2(lbl1: str, key1: str, lbl2: str, key2: str):
        """Print two F-key hints side-by-side on one box row."""
        left  = f"  {_YEL(lbl1):<6}  {_DIM(key1)}"
        right = f"  {_YEL(lbl2):<6}  {_DIM(key2)}"
        # plain lengths for padding
        left_len  = 2 + len(lbl1) + 2 + len(key1)
        right_len = 2 + len(lbl2) + 2 + len(key2)
        pad = W - left_len - right_len
        print(_DIM("    ║") + left + " " * max(pad, 2) + right + _DIM("║"))

    # ── Resolve available games ───────────────────────────────────────────────
    clear_cli()
    _print_header(
        get_available_games(), get_available_algos_from_grid(),
        get_trained_games_from_models_flat(), get_trained_models_count()
    )
    _section("MANUAL PLAY  ›  Select Game")

    available_games = get_available_games()
    if not available_games:
        print(_RED("  ✖  No game configurations found in grid.yaml"))
        return

    # ── Themed game list ──────────────────────────────────────────────────────
    print()
    for i, game in enumerate(available_games, 1):
        print(f"    {_YEL(f'[{i}]')}  {_WHT(game)}")
    back_idx = len(available_games) + 1
    print(f"    {_YEL(f'[{back_idx}]')}  {_DIM('Back')}")
    print()

    raw = input(_BOLD("    ⟫ ")).strip()
    try:
        idx = int(raw)
    except ValueError:
        print(_RED("  ✖  Invalid selection."))
        return

    if idx == back_idx:
        return
    if not (1 <= idx <= len(available_games)):
        print(_RED("  ✖  Invalid selection."))
        return

    selected_game = available_games[idx - 1]

    # ── Strip dummy SDL driver so the real window opens ───────────────────────
    proc_env = os.environ.copy()
    proc_env.pop("SDL_VIDEODRIVER", None)

    # ── Verify the manual_play script exists before printing the banner ───────
    script_path = Path("code/scripts/manual_play.py")
    if not script_path.exists():
        print(_RED("  ✖  Manual play script not found at code/scripts/manual_play.py"))
        return

    # ── Pre-launch banner ─────────────────────────────────────────────────────
    print()
    _box_top()
    _box_row()
    _box_row("PEAK ENGINE  ·  MANUAL PLAY", color_fn=lambda t: _BOLD(_RED("    " + t.center(W - 4))))
    _box_row(_WHT(f"  {selected_game.upper()}").center(W), color_fn=lambda t: _BOLD(_WHT("    " + selected_game.upper().center(W - 4))))
    _box_row()
    _box_mid()
    # Controls section
    _box_row("  CONTROLS", color_fn=_BOLD)
    _box_row()
    _box_kv("A / D",         "Move left / right")
    _box_kv("SPACE",         "Jump")
    _box_kv("SHIFT",         "Run")
    _box_kv("ESC",           "Quit session")
    _box_row()
    _box_mid()
    # Debug keys section
    _box_row("  DEBUG OVERLAY  (F-keys)", color_fn=_BOLD)
    _box_row()
    _box_kv("F1",  "Sensor rays   (toggle)",    key_w=5)
    _box_kv("F2",  "Free camera   (IJKL)",       key_w=5)
    _box_kv("F3",  "Slow motion   (0.5×)",       key_w=5)
    _box_kv("F4",  "Hitboxes      (toggle)",     key_w=5)
    _box_kv("F5",  "Agent vision  (max view)",   key_w=5)
    _box_row()
    _box_bot()
    print()

    # ── Launch ────────────────────────────────────────────────────────────────
    print(_DIM(f"    Launching {selected_game}..."))
    print()

    subprocess.run(
        [sys.executable, "-m", "code.scripts.manual_play",
         "--game", selected_game, "--fps", "30"],
        env=proc_env
    )

    # ── Session-end banner ────────────────────────────────────────────────────
    print()
    _box_top()
    _box_row()
    _box_row("SESSION ENDED", color_fn=lambda t: _BOLD(_GRN("    " + t.center(W - 4))))
    _box_row()
    _box_row(_DIM("  Thanks for playing  ·  PEAK ENGINE"), color_fn=lambda t: _DIM("    " + "Thanks for playing  ·  PEAK ENGINE".center(W - 4)))
    _box_row()
    _box_bot()
    print()


def show_project_status():
    """
    Display comprehensive project status
        - Shows how many models of each combination of games_algo_persona
    """
    _section("STATUS  ›  Project Overview")
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
#             print(_RED("  ✖  Invalid selection."))
#     except ValueError:
#         print(_RED("  ✖  Invalid selection."))

def watch_random_agent():
    """Watch a random agent; optionally record the actual interactive session."""
    _section("WATCH  ›  Random Agent")

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
        print(_RED("  ✖  Invalid selection."))
        return

    if idx == len(available_games) + 1:
        return
    if not (1 <= idx <= len(available_games)):
        print(_RED("  ✖  Invalid selection."))
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

def clear_cli():
    """
    Clear the terminal screen using the platform-appropriate command.
    'cls' on Windows, 'clear' on Unix/macOS. Called before every major
    screen to keep the menu feeling like a full-screen TUI.
    """
    os.system("cls" if sys.platform == "win32" else "clear")

def _toggle_level_in_config(config_path, lid, enable):
    """Comment out or uncomment a level block in game_config.yaml."""
    import re
    try:
        text = config_path.read_text(encoding="utf-8")
    except Exception:
        return False

    lines = text.split('\n')
    nl = []
    in_block = False
    base_indent = 0

    if enable:
        # Uncomment the block
        for line in lines:
            s = line.lstrip()
            if not in_block:
                # Match:  # "key":  OR  # 'key':  OR  # key:
                if s.startswith('#'):
                    inner = s.lstrip('#').lstrip()
                    for pat in [f'"{lid}":', f"'{lid}':", f'{lid}:']:
                        if inner.startswith(pat):
                            in_block = True
                            nl.append(line.replace('# ', '', 1) if '# ' in line else line.replace('#', '', 1))
                            break
                    else:
                        nl.append(line)
                else:
                    nl.append(line)
            else:
                if s.startswith('#') and (len(s) > 1 and (s[1:].startswith('  ') or s[1:].startswith(' '))):
                    nl.append(line.replace('# ', '', 1) if '# ' in line else line.replace('#', '', 1))
                else:
                    in_block = False
                    nl.append(line)
    else:
        # Comment out the block
        for line in lines:
            s = line.lstrip()
            ind = len(line) - len(s)
            if not in_block:
                for pat in [f'"{lid}":', f"'{lid}':", f'{lid}:']:
                    if s.startswith(pat):
                        in_block = True
                        base_indent = ind
                        nl.append(' ' * ind + '# ' + s)
                        break
                else:
                    nl.append(line)
            else:
                if s and ind > base_indent:
                    nl.append(' ' * ind + '# ' + s)
                else:
                    in_block = False
                    nl.append(line)
    try:
        config_path.write_text('\n'.join(nl), encoding="utf-8")
        return True
    except Exception:
        return False


def run_toggle_levels():
    """Enable or disable levels in game_config.yaml from the CLI."""
    import re
    try:
        import yaml
    except ImportError:
        print("\n  PyYAML is not installed — cannot read game_config.yaml")
        return

    # Locate game_config.yaml
    candidates = [
        Path("game_config.yaml"),
        Path("code/games/platformer/game_config.yaml"),
        Path("code/games/game_config.yaml"),
    ]
    config_path = None
    for p in candidates:
        if p.exists():
            config_path = p
            break

    if config_path is None:
        print("\n  Could not find game_config.yaml")
        print("  Checked: " + ", ".join(str(c) for c in candidates))
        return

    _LEVEL_SUBKEYS = {
        'file', 'time_limit', 'background_color', 'time', 'physics',
        'gravity', 'friction', 'max_fall_speed', 'max_run_speed',
        'dynamics', 'enemies', 'coins', 'powerups', 'moving_platforms',
        'player', 'render', 'reward', 'observation', 'seed',
    }

    while True:
        # Re-read on every iteration so changes are reflected immediately
        try:
            text = config_path.read_text(encoding="utf-8")
            data = yaml.safe_load(text) or {}
        except Exception as e:
            print(f"\n  Error reading config: {e}")
            return

        active = data.get('levels', {}) or {}
        active_ids = list(active.keys())

        # Find commented-out (disabled) levels
        disabled_ids = []
        for m in re.finditer(
            r'^\s*#\s*(?:"([^"]+)"|\'([^\']+)\'|([^\s:][^:]*?)):\s*$',
            text, re.MULTILINE
        ):
            lid = (m.group(1) or m.group(2) or m.group(3) or "").strip()
            if lid and lid not in active and lid not in _LEVEL_SUBKEYS:
                if lid not in disabled_ids:
                    disabled_ids.append(lid)

        print(_BOLD("\n  ═══ Level Toggle ═══"))
        print(f"  Config: {_DIM(str(config_path))}\n")

        if active_ids:
            print(f"  {_GRN('ENABLED')}  ({len(active_ids)})")
            for i, lid in enumerate(active_ids, 1):
                fn = active[lid].get('file', '???') if isinstance(active[lid], dict) else '???'
                print(f"    {_YEL(f'{i:>3}.')}  {_GRN('[ON ]')}  {_WHT(lid):<20}  {_DIM(fn)}")
        else:
            print(f"  {_DIM('(no active levels)')}")

        start = len(active_ids)
        if disabled_ids:
            print(f"\n  {_RED('DISABLED')}  ({len(disabled_ids)})")
            for j, lid in enumerate(disabled_ids, start + 1):
                print(f"    {_YEL(f'{j:>3}.')}  {_RED('[OFF]')}  {_DIM(lid)}")

        all_ids = active_ids + disabled_ids
        back_n = len(all_ids) + 1
        print(f"\n    {_YEL(f'{back_n:>3}.')}  {_RED('Back')}")
        print()

        pick = input(_BOLD("    ⟫ Toggle # ")).strip()
        try:
            n = int(pick)
            if n == back_n:
                return
            if 1 <= n <= len(all_ids):
                lid = all_ids[n - 1]
                is_active = n <= len(active_ids)
                ok = _toggle_level_in_config(config_path, lid, enable=not is_active)
                if ok:
                    state = _GRN("enabled") if not is_active else _RED("disabled")
                    print(f"\n  ✓  '{lid}' {state}\n")
                else:
                    print(_RED(f"\n  ✖  Failed to toggle '{lid}'\n"))
            else:
                print(_RED("  Invalid selection."))
        except ValueError:
            print(_RED("  Invalid input."))


def run_level_editor():
    """Launch the PEAK level editor (pygame-based tile painter)."""
    print("\n  Launching PEAK Level Editor...")

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
        print("\n  Existing levels:")
        print("    0. New blank level")
        for i, lf in enumerate(level_files, 1):
            print(f"    {i}. {lf.name}")
        print(f"    {len(level_files) + 1}. Back")

        pick = input(f"\n  Open level (0-{len(level_files) + 1}): ").strip()
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

# ── ANSI helpers ──────────────────────────────────────────────────────────────
# All color/style output goes through these lambda wrappers.
# When stdout is not a TTY (e.g., piped to a file), _SUPPORTS_COLOR is False
# and all wrappers return plain text, keeping log files readable.
_SUPPORTS_COLOR = (
    hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    and os.environ.get("NO_COLOR") is None
)

def _c(code: str, text: str) -> str:
    """Wrap *text* in an ANSI escape if the terminal supports it."""
    if not _SUPPORTS_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"

# Palette — each lambda wraps text in the matching ANSI escape sequence.
# _DIM:  muted/secondary text (hints, dividers, labels)
# _BOLD: emphasis (headers, prompts, important values)
# _CYAN: accent color for section arrows and highlights
# _MAG:  magenta — used for the sub-logo tagline
# _YEL:  yellow  — menu key numbers and toggle brackets
# _GRN:  green   — success states, trained counts, selected items
# _RED:  red     — PEAK logo, errors, warnings, destructive actions
# _WHT:  bright white — primary values and game names
_DIM     = lambda t: _c("2",    t)
_BOLD    = lambda t: _c("1",    t)
_CYAN    = lambda t: _c("96",   t)
_MAG     = lambda t: _c("95",   t)
_YEL     = lambda t: _c("93",   t)
_GRN     = lambda t: _c("92",   t)
_RED     = lambda t: _c("91",   t)
_WHT     = lambda t: _c("97",   t)
_BGDARK  = lambda t: _c("48;5;233", t)

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
    """
    Render the top section of every PEAK ENGINE screen:
      - ASCII art PEAK logo (red)
      - Sub-logo tagline (magenta)
      - Live stats bar: # games / algos available, # trained games / models
    Called at the start of every action that clears the screen.
    """
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
    """
    Format a single menu row as:  [key]  Label  dim-hint
    Key is right-aligned to 2 chars and yellow.  Hint is optional and dimmed.
    """
    k = _YEL(f"  [{key:>2}]")
    h = _DIM(f"  {hint}") if hint else ""
    return f"{k}  {label}{h}"


def _section(title: str):
    """Print a cyan-accented section header — used to divide the menu into labelled groups."""
    print(f"\n    {_BOLD(_CYAN('▸'))} {_BOLD(title)}")


def main():
    import warnings
    warnings.filterwarnings(
        "ignore",
        message=r".*pkg_resources is deprecated as an API.*"
    )

    global CURRENT_ALGO

    # DISPATCH table maps menu key → (debug_label, callable).
    # None as the callable means "handled inline" (currently only exit/0).
    # Adding a new menu item: add the entry here AND a matching _menu_item() print below.
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
        "10": ("toggle_levels",        run_toggle_levels),
        "11": ("tensorboard",          run_tensorboard),
        "12": ("analyzer",             run_agent_analyzer),
        "13": ("delete_all",           delete_logs_and_models),
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
        print(_menu_item("2",  "Train Single",        "toggle: game / algo / personas / skills / arch"))
        print(_menu_item("3",  "Train All (1 game)",  "toggle: algos / personas / skills"))
        print(_menu_item("4",  "Train Full Grid",     "all games × algos × personas"))

        _section("PLAY")
        print(_menu_item("5",  "Play Manually",       "keyboard controls"))
        print(_menu_item("6",  "Watch Agent",         "visualize a trained model"))
        print(_menu_item("7",  "Watch All Models",    "side-by-side grid"))
        print(_menu_item("8",  "Watch Random Agent",  "random actions"))

        _section("TOOLS")
        print(_menu_item("9",  "Level Editor",         "paint tiles, place entities"))
        print(_menu_item("10", "Toggle Levels",        "enable / disable levels in config"))
        print(_menu_item("11", "TensorBoard",          "mylogs/"))
        print(_menu_item("12", "Analyze Performance",  "CSV log deep-dive"))
        print(_menu_item("13", "Delete Logs & Models", "nuclear option"))
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