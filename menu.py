"""PEAK ENGINE — main menu.

The classic PEAK ENGINE hub (logo, sections, toggle pickers, chime and all),
rewired to the neuroevolution backend in code/neuro/.
"""
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

# Enable VT100 escape processing on Windows terminals.
os.system("")

# The logo and box glyphs need UTF-8; legacy consoles default to cp1252.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

RUNS_DIR = Path("runs")
DASHBOARD_URL = "http://127.0.0.1:8000/mario/index.html"
DISABLED_LEVELS_KEY = "disabled_levels"
CHIME_PATH = Path("chime.wav")

# Adapter keys → game_config.yaml section key (platformer sits at the root).
GAMES = ["mario", "megaman", "sonic", "meatboy"]
CONFIG_KEY = {"mario": "platformer", "megaman": "megaman", "sonic": "sonic"}

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


def clear_cli():
    """Clear the terminal window (cls on Windows, clear elsewhere)."""
    os.system("cls" if platform.system() == "Windows" else "clear")


def play_chime():
    """Victory chime after a fully successful training batch."""
    if HAS_WINSOUND and CHIME_PATH.exists():
        try:
            winsound.PlaySound(str(CHIME_PATH), winsound.SND_FILENAME)
        except Exception:
            pass


# ============================================================================
# CONFIG / STATE READERS
# ============================================================================

def _load_game_config_yaml():
    try:
        import yaml
    except ImportError:
        return None, None, None

    candidates = [Path("game_config.yaml"), Path("code/games/game_config.yaml")]
    config_path = next((p for p in candidates if p.exists()), None)
    if config_path is None:
        return None, None, yaml

    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        data = {}
    return config_path, data, yaml


def _game_config_root(data, game, create=False):
    if game == "platformer":
        return data
    root = data.get(game)
    if isinstance(root, dict):
        return root
    if create:
        data[game] = {}
        return data[game]
    return {}


def _game_level_maps(data, game, create=False):
    root = _game_config_root(data, game, create=create)
    levels = root.get("levels")
    if not isinstance(levels, dict):
        if create:
            root["levels"] = {}
            levels = root["levels"]
        else:
            levels = {}

    disabled = root.get(DISABLED_LEVELS_KEY)
    if not isinstance(disabled, dict):
        if create:
            root[DISABLED_LEVELS_KEY] = {}
            disabled = root[DISABLED_LEVELS_KEY]
        else:
            disabled = {}
    return levels, disabled


def _toggle_game_level_in_config(game, lid, enable):
    config_path, data, yaml = _load_game_config_yaml()
    if not config_path or yaml is None:
        return False

    levels, disabled = _game_level_maps(data, game, create=True)
    if enable:
        if lid in disabled:
            levels[lid] = disabled.pop(lid)
    else:
        if lid in levels:
            disabled[lid] = levels.pop(lid)

    root = _game_config_root(data, game, create=True)
    if not disabled:
        root.pop(DISABLED_LEVELS_KEY, None)

    try:
        config_path.write_text(
            yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        return True
    except Exception:
        return False


def get_available_games():
    return list(GAMES)


def get_levels_for_game(game: str) -> list:
    """Enabled level ids for an adapter game. Meatboy levels are indices."""
    if game == "meatboy":
        try:
            import yaml
            data = yaml.safe_load(Path("code/games/meatboy_config.yaml").read_text(encoding="utf-8")) or {}
            return [str(i) for i in range(len(data.get("levels", [])))]
        except Exception:
            return []
    _, data, _ = _load_game_config_yaml()
    if not data:
        return []
    levels, _ = _game_level_maps(data, CONFIG_KEY[game])
    return list(levels.keys())


def get_enabled_level_count() -> int:
    return sum(len(get_levels_for_game(g)) for g in GAMES)


def get_run_dirs():
    """Run dirs under runs/ that hold a trained population (state.json)."""
    if not RUNS_DIR.exists():
        return []
    return sorted(p for p in RUNS_DIR.iterdir() if (p / "state.json").exists())


def get_trained_games():
    trained = set()
    for p in get_run_dirs():
        for g in GAMES:
            if p.name.startswith(g):
                trained.add(g)
    return trained


def get_trained_models_count() -> int:
    return len(list(RUNS_DIR.glob("*/best.npz"))) if RUNS_DIR.exists() else 0


def guess_game_for_run(run_dir: Path) -> str:
    for g in GAMES:
        if run_dir.name.startswith(g):
            return g
    return GAMES[0]


def setup_project():
    RUNS_DIR.mkdir(exist_ok=True)


# ============================================================================
# UI PRIMITIVES
# ============================================================================

def _print_header(games, levels, trained_games, trained_models):
    """
    Render the top section of every PEAK ENGINE screen:
      - ASCII art PEAK logo (red)
      - Sub-logo tagline (magenta)
      - Live stats bar: # games / levels available, # trained games / genomes
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
    l_str = _WHT(f"{levels}")
    tg_str = _GRN(f"{len(trained_games)}") if trained_games else _DIM("0")
    tm_str = _GRN(f"{trained_models}") if trained_models else _DIM("0")

    print()
    print(f"    Games {g_str}  │  Levels {l_str}  │  Trained {tg_str}  │  Models {tm_str}")
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


def toggle_select(title, options, default_indices=None, min_select=1, show_desc=None):
    """
    Interactive toggle-style multi-select used throughout the training menus.

    Renders a numbered checklist where each item can be toggled on/off.
    Supports comma-separated input ("1,3") and ranges ("1-3") in one go.
    Returns the confirmed list of selected items, or None if user typed "0" (back).

    Type a number to flip that item on/off.  Press Enter to confirm.
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


def _refresh_screen():
    clear_cli()
    _print_header(get_available_games(), get_enabled_level_count(),
                  get_trained_games(), get_trained_models_count())


# ============================================================================
# TRAINING EXECUTION
# ============================================================================

def _trainer_cmd(game, level=None, gens=None, turbo=False, serve=True, run_dir=None, extra=()):
    cmd = [sys.executable, "-m", "code.neuro.trainer", "--game", game]
    if level:
        cmd += ["--level", str(level)]
    if gens:
        cmd += ["--gens", str(gens)]
    if turbo:
        cmd += ["--turbo"]
    if not serve:
        cmd += ["--no-serve"]
    if run_dir:
        cmd += ["--run-dir", str(run_dir)]
    cmd += list(extra)
    return cmd


def execute_training_run(cmd) -> bool:
    """Run one trainer subprocess; echo the command like classic PEAK did."""
    print(_DIM("  >>> " + " ".join(cmd)))
    try:
        return subprocess.run(cmd).returncode == 0
    except KeyboardInterrupt:
        print(_DIM("\n    Training interrupted."))
        return False


def print_training_summary(total, successful, failed):
    W = 50
    print()
    print(_DIM("    " + "─" * W))
    if failed == 0:
        print(f"    {_GRN('✓ TRAINING COMPLETE')}")
    else:
        print(f"    {_YEL('TRAINING FINISHED WITH ERRORS')}")
    print(f"    Successful: {_WHT(f'{successful}/{total}')}")
    print(f"    Logs  →  {_DIM('runs/<name>/state.json')}")
    print(f"    Models→  {_DIM('runs/<name>/best.npz')}")
    print(_DIM("    " + "─" * W))


def _prompt_gens(default_hint="Enter = train until Ctrl+C"):
    raw = input(_DIM(f"\n    Generations ({default_hint}): ")).strip()
    if not raw:
        return None
    try:
        n = int(raw)
        if n < 1:
            raise ValueError
        return n
    except ValueError:
        print(_RED("  Invalid generation count. Use a positive integer."))
        return "invalid"


def run_training():
    """Train Single — one game, one level, live dashboard."""
    _refresh_screen()
    _section("TRAIN  ›  Single Run")

    game = ask_index("\n  Choose a game:", get_available_games(), default="mario")
    if not game:
        return

    levels = get_levels_for_game(game)
    level = None
    if levels:
        level = ask_index("\n  Choose a level:", ["auto (first level)"] + levels,
                          default="auto (first level)")
        if level is None:
            return
        if level.startswith("auto"):
            level = None

    persona = ask_index("\n  Player persona (who should the agents play like?):",
                        ["experienced", "novice", "speedrunner"], default="experienced")
    if not persona:
        return
    gens = _prompt_gens()
    if gens == "invalid":
        return
    turbo = input(_DIM("    Start in turbo? [y/N]: ")).strip().lower().startswith("y")

    run_dir = RUNS_DIR / (game if not level else f"{game}_{level}")
    if persona != "experienced":
        run_dir = Path(str(run_dir) + f"_{persona}")

    W = 50
    print()
    print(_DIM("    " + "─" * W))
    print(f"    {_BOLD('Run Summary')}")
    print(f"    Game:        {_WHT(game)}")
    print(f"    Level:       {_WHT(level or 'auto')}")
    print(f"    Persona:     {_WHT(persona)}")
    print(f"    Generations: {_WHT(str(gens) if gens else 'until stopped')}")
    print(f"    Mode:        {_WHT('turbo' if turbo else 'real-time')}")
    print(f"    Run dir:     {_WHT(str(run_dir))}")
    print(_DIM("    " + "─" * W))

    proceed = input(_BOLD("    ⟫ Proceed? [Y/n]: ")).strip().lower()
    if proceed in ("n", "no"):
        return

    print()
    print(f"    Dashboard  →  {_CYAN(DASHBOARD_URL)}")
    print()
    ok = execute_training_run(_trainer_cmd(game, level, gens, turbo, run_dir=run_dir,
                                           extra=("--persona", persona)))
    print_training_summary(1, int(ok), int(not ok))
    if ok:
        play_chime()


def train_all_models_for_game():
    """Train All (1 game) — every selected level, back to back, headless turbo."""
    _refresh_screen()
    _section("TRAIN  ›  All Levels (1 game)")

    game = ask_index("\n  Choose a game:", get_available_games(), default="mario")
    if not game:
        return

    levels = get_levels_for_game(game)
    if not levels:
        print(_RED("  ✖  No levels found for this game."))
        return

    chosen = toggle_select(f"LEVELS  ·  {game}", levels, default_indices=list(range(len(levels))))
    if not chosen:
        return

    gens = _prompt_gens("e.g. 60")
    if gens in ("invalid", None):
        if gens is None:
            print(_RED("  A generation count is required for batch runs."))
        return

    W = 50
    print()
    print(_DIM("    " + "─" * W))
    print(f"    {_BOLD('Run Summary')}")
    print(f"    Game:        {_WHT(game)}")
    print(f"    Levels:      {_WHT(', '.join(chosen))}")
    print(f"    Generations: {_WHT(str(gens))} per level")
    print(f"    Total runs:  {_WHT(str(len(chosen)))}")
    print(_DIM("    " + "─" * W))

    proceed = input(_BOLD("    ⟫ Proceed? [Y/n]: ")).strip().lower()
    if proceed in ("n", "no"):
        return

    print(f"\n    Dashboard  →  {_CYAN(DASHBOARD_URL)}  {_DIM('(follows each run; reconnects between them)')}")
    successful = 0
    for i, level in enumerate(chosen, 1):
        print(f"\n  {_YEL(f'[{i}/{len(chosen)}]')}  {_WHT(game)} | {_WHT(level)}")
        run_dir = RUNS_DIR / f"{game}_{level}"
        ok = execute_training_run(
            _trainer_cmd(game, level, gens, turbo=True, run_dir=run_dir))
        successful += int(ok)

    print_training_summary(len(chosen), successful, len(chosen) - successful)
    if successful == len(chosen):
        play_chime()


def train_complete_grid():
    """Train Full Grid — every game × its enabled levels, headless turbo."""
    _refresh_screen()
    _section("TRAIN  ›  Full Grid")

    jobs = [(g, lvl) for g in get_available_games() for lvl in (get_levels_for_game(g) or [None])]

    gens = _prompt_gens("e.g. 60")
    if gens in ("invalid", None):
        if gens is None:
            print(_RED("  A generation count is required for batch runs."))
        return

    W = 50
    print()
    print(_DIM("    " + "─" * W))
    print(f"    {_BOLD('Run Summary')}")
    print(f"    Games:       {_WHT(', '.join(get_available_games()))}")
    print(f"    Generations: {_WHT(str(gens))} per run")
    print(f"    Total runs:  {_WHT(str(len(jobs)))}")
    print(_DIM("    " + "─" * W))

    proceed = input(_BOLD("    ⟫ Proceed? [Y/n]: ")).strip().lower()
    if proceed in ("n", "no"):
        return

    print(f"\n    Dashboard  →  {_CYAN(DASHBOARD_URL)}  {_DIM('(follows each run; reconnects between them)')}")
    successful = 0
    for i, (game, level) in enumerate(jobs, 1):
        print(f"\n  {_YEL(f'[{i}/{len(jobs)}]')}  {_WHT(game)} | {_WHT(level or 'auto')}")
        run_dir = RUNS_DIR / (game if not level else f"{game}_{level}")
        ok = execute_training_run(
            _trainer_cmd(game, level, gens, turbo=True, run_dir=run_dir))
        successful += int(ok)

    print_training_summary(len(jobs), successful, len(jobs) - successful)
    if successful == len(jobs):
        play_chime()


# ============================================================================
# PLAY / WATCH
# ============================================================================

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
            inner = f"  {text}"
            pad   = W - len(inner)
            print(_DIM("    ║") + color_fn(inner) + _DIM(" " * max(pad, 0) + "║"))
        else:
            centered = text.center(W)
            print(_DIM("    ║" + centered + "║"))

    def _box_kv(key: str, val: str, key_w: int = 16):
        """Print a key-value row inside the box."""
        k_part  = _YEL(f"  {key:<{key_w}}")
        v_part  = _WHT(val)
        pad     = W - 2 - key_w - len(val)
        print(_DIM("    ║") + k_part + v_part + _DIM(" " * max(pad, 0) + "║"))

    # ── Resolve available games (meatboy has no manual key mapping) ───────────
    _refresh_screen()
    _section("MANUAL PLAY  ›  Select Game")

    available_games = [g for g in get_available_games() if g != "meatboy"]
    if not available_games:
        print(_RED("  ✖  No game configurations found"))
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
    config_game = CONFIG_KEY.get(selected_game, selected_game)

    # ── Strip dummy SDL driver so the real window opens ───────────────────────
    proc_env = os.environ.copy()
    proc_env.pop("SDL_VIDEODRIVER", None)

    script_path = Path("code/games/tools/manual_play.py")
    if not script_path.exists():
        print(_RED("  ✖  Manual play script not found at code/games/tools/manual_play.py"))
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
    if selected_game.lower() == "megaman":
        _box_kv("W / S",     "Climb ladder")
        _box_kv("Z",         "Fire")
    elif selected_game.lower() == "sonic":
        _box_kv("SHIFT",     "Run")
        _box_kv("S / DOWN",  "Crouch / spin dash")
    else:
        _box_kv("SHIFT",     "Run")
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
        [sys.executable, "-m", "code.games.tools.manual_play",
         "--game", config_game, "--fps", "30"],
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


def _pick_run(prompt="\n  Choose a run:"):
    runs = get_run_dirs()
    if not runs:
        print(_RED("  ✖  No trained runs found in runs/. Train something first."))
        return None
    names = [p.name for p in runs]
    name = ask_index(prompt, names, default=names[0])
    return RUNS_DIR / name if name else None


def watch_trained_agent():
    """Watch Agent — replay a run's all-time best genome in real time."""
    _refresh_screen()
    _section("WATCH  ›  Trained Agent")

    run_dir = _pick_run()
    if not run_dir:
        return
    best = run_dir / "best.npz"
    if not best.exists():
        print(_RED(f"  ✖  {best} not found."))
        return

    game = guess_game_for_run(run_dir)
    print(f"\n    Dashboard  →  {_CYAN(DASHBOARD_URL)}")
    print(_DIM("    Ctrl+C to stop the replay.\n"))
    execute_training_run(
        [sys.executable, "-m", "code.neuro.trainer", "--game", game,
         "--replay", str(best)])


def watch_all_models():
    """Watch All Models — resume a run with the live dashboard grid (all 10 envs side by side)."""
    _refresh_screen()
    _section("WATCH  ›  All Envs (dashboard grid)")

    run_dir = _pick_run()
    if not run_dir:
        return

    game = guess_game_for_run(run_dir)
    print(f"\n    Dashboard  →  {_CYAN(DASHBOARD_URL)}")
    print(_DIM("    Training continues while you watch. Ctrl+C to stop.\n"))
    execute_training_run(
        [sys.executable, "-m", "code.neuro.trainer", "--game", game,
         "--resume", str(run_dir)])


def watch_random_agent():
    """Watch Random Agent — random actions, human window."""
    _refresh_screen()
    _section("WATCH  ›  Random Agent")

    available = [g for g in get_available_games() if g != "meatboy"]
    game = ask_index("\n  Choose a game:", available, default=available[0])
    if not game:
        return

    proc_env = os.environ.copy()
    proc_env.pop("SDL_VIDEODRIVER", None)
    print(_DIM(f"\n    Launching {game} with random actions... (ESC to quit)\n"))
    subprocess.run(
        [sys.executable, "-m", "code.games.tools.manual_play",
         "--game", CONFIG_KEY.get(game, game), "--fps", "30", "--random"],
        env=proc_env
    )


# ============================================================================
# TOOLS
# ============================================================================

def run_level_editor():
    """Launch the PEAK level editor (pygame-based tile painter)."""
    print("\n  Launching PEAK Level Editor...")

    # Locate the script — check both code/games/tools and project root
    candidates = [
        Path("code/games/tools/level_editor.py"),
        Path("code/scripts/level_editor.py"),
        Path("level_editor.py"),
    ]

    editor_path = next((p for p in candidates if p.exists()), None)
    if editor_path is None:
        print("  Could not find level_editor.py.")
        print("  Checked: " + ", ".join(str(c) for c in candidates))
        return

    game_choices = ["platformer", "megaman", "sonic"]
    game = ask_index(
        "\n  Choose which game to edit:",
        game_choices,
        add_back=True,
        default=game_choices[0],
    )
    if not game:
        return

    env_vars = os.environ.copy()
    env_vars.pop("SDL_VIDEODRIVER", None)
    cmd = [sys.executable, str(editor_path), "--game", game]

    print("  >>>", " ".join(cmd), "\n")
    try:
        subprocess.run(cmd, check=True, env=env_vars)
    except subprocess.CalledProcessError as e:
        print(f"\n  Editor exited with error code {e.returncode}")
    except KeyboardInterrupt:
        print("\n  Editor closed.")


def run_toggle_levels():
    """Enable or disable levels in game_config.yaml from the CLI."""
    config_path, _, yaml = _load_game_config_yaml()
    if not config_path or yaml is None:
        print("\n  Could not find a readable game_config.yaml")
        return

    game_choices = ["platformer", "megaman", "sonic"]
    game = ask_index(
        "\n  Choose which game's levels to manage:",
        game_choices,
        add_back=True,
        default=game_choices[0],
    )
    if not game:
        return

    while True:
        try:
            data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except Exception as e:
            print(f"\n  Error reading config: {e}")
            return

        active, disabled = _game_level_maps(data, game, create=False)
        active_ids = list(active.keys())
        disabled_ids = list(disabled.keys())

        print(_BOLD(f"\n  Level Toggle [{game}]"))
        print(f"  Config: {_DIM(str(config_path))}\n")

        if active_ids:
            print(f"  {_GRN('ENABLED')}  ({len(active_ids)})")
            for i, lid in enumerate(active_ids, 1):
                fn = active[lid].get("file", "???") if isinstance(active[lid], dict) else "???"
                print(f"    {_YEL(f'{i:>3}.')}  {_GRN('[ON ]')}  {_WHT(lid):<20}  {_DIM(fn)}")
        else:
            print(f"  {_DIM('(no active levels)')}")

        start = len(active_ids)
        if disabled_ids:
            print(f"\n  {_RED('DISABLED')}  ({len(disabled_ids)})")
            for j, lid in enumerate(disabled_ids, start + 1):
                fn = disabled[lid].get("file", "???") if isinstance(disabled[lid], dict) else "???"
                print(f"    {_YEL(f'{j:>3}.')}  {_RED('[OFF]')}  {_WHT(lid):<20}  {_DIM(fn)}")

        all_ids = active_ids + disabled_ids
        back_n = len(all_ids) + 1
        print(f"\n    {_YEL(f'{back_n:>3}.')}  {_RED('Back')}")
        print()

        pick = input(_BOLD("    Toggle # ")).strip()
        try:
            n = int(pick)
            if n == back_n:
                return
            if 1 <= n <= len(all_ids):
                lid = all_ids[n - 1]
                is_active = n <= len(active_ids)
                ok = _toggle_game_level_in_config(game, lid, enable=not is_active)
                if ok:
                    state = _GRN("enabled") if not is_active else _RED("disabled")
                    print(f"\n  '{lid}' {state}\n")
                else:
                    print(_RED(f"\n  Failed to toggle '{lid}'\n"))
            else:
                print(_RED("  Invalid selection."))
        except ValueError:
            print(_RED("  Invalid input."))


def run_dashboard():
    """Open the live training dashboard in the default browser."""
    _section("TOOLS  ›  Dashboard")
    print(f"\n    Dashboard  →  {_CYAN(DASHBOARD_URL)}")
    print(_DIM("    (a trainer must be running for live data — start one via TRAIN)"))
    try:
        webbrowser.open(DASHBOARD_URL)
    except Exception as e:
        print(_RED(f"  ✖  Could not open browser: {e}"))


def open_balance_command():
    """Balance Command — regenerate the command center from runs/ and open it."""
    _refresh_screen()
    _section("BALANCE COMMAND")
    print(_DIM("    Regenerating from runs/balance, runs/probes, and every training run..."))
    subprocess.run([sys.executable, "-m", "code.neuro.report", "--open"])


def run_balance_report():
    """Balance Report — multi-seed neuroevolution probes per level (the paper-matrix successor)."""
    _refresh_screen()
    _section("BALANCE  ›  Multi-seed Level Report")

    game = ask_index("\n  Choose a game:", get_available_games(), default="mario")
    if not game:
        return

    levels = get_levels_for_game(game)
    if not levels:
        print(_RED("  ✖  No levels found for this game."))
        return
    chosen = toggle_select(f"LEVELS  ·  {game}", levels, default_indices=list(range(len(levels))))
    if not chosen:
        return

    persona = ask_index("\n  Probe persona:", ["experienced", "novice", "speedrunner"],
                        default="experienced") or "experienced"
    gens_raw = input(_DIM("\n    Generation budget per probe [25]: ")).strip()
    gens = gens_raw if gens_raw.isdigit() else "25"
    seeds_raw = input(_DIM("    Seeds [1234 2025 31337]: ")).strip()
    seeds = seeds_raw.split() if seeds_raw else ["1234", "2025", "31337"]

    n_jobs = len(chosen) * len(seeds)
    print(f"\n    {_WHT(str(n_jobs))} probes ({len(chosen)} levels × {len(seeds)} seeds), "
          f"{_DIM('roughly ' + str(n_jobs * 2) + '-' + str(n_jobs * 4) + ' minutes')}")
    proceed = input(_BOLD("    ⟫ Proceed? [Y/n]: ")).strip().lower()
    if proceed in ("n", "no"):
        return

    cmd = [sys.executable, "-m", "code.neuro.balance", "--game", game, "--persona", persona,
           "--gens", gens, "--seeds", *seeds, "--levels", *chosen]
    print(_DIM("  >>> " + " ".join(cmd) + "\n"))
    try:
        ok = subprocess.run(cmd).returncode == 0
    except KeyboardInterrupt:
        return
    if ok:
        subprocess.run([sys.executable, "-m", "code.neuro.report", "--open"])
        play_chime()


def run_full_sweep():
    """Full Sweep — balance probes across games × personas × seeds, all parallel."""
    _refresh_screen()
    _section("BALANCE  ›  Full Sweep")

    games = toggle_select("GAMES", get_available_games(),
                          default_indices=[i for i, g in enumerate(get_available_games())
                                           if g in ("mario", "meatboy")])
    if not games:
        return
    personas = toggle_select("PERSONAS", ["experienced", "novice", "speedrunner"],
                             default_indices=[0, 1, 2])
    if not personas:
        return
    gens_raw = input(_DIM("\n    Generation budget per probe [40]: ")).strip()
    gens = gens_raw if gens_raw.isdigit() else "40"
    seeds = ["1234", "2025", "31337"]

    n_levels = sum(len(get_levels_for_game(g)) for g in games)
    n_jobs = n_levels * len(seeds) * len(personas)
    workers = max(1, (os.cpu_count() or 2) - 1)
    print(f"\n    {_WHT(str(n_jobs))} probes ({n_levels} levels × {len(seeds)} seeds × "
          f"{len(personas)} personas), {workers} parallel workers")
    print(_DIM("    Only ENABLED levels are probed — use Toggle Levels [10] first if needed."))
    if input(_BOLD("    ⟫ Proceed? [Y/n]: ")).strip().lower() in ("n", "no"):
        return

    for game in games:
        for persona in personas:
            print(_BOLD(f"\n  ── {game} · {persona} " + "─" * 30))
            cmd = [sys.executable, "-m", "code.neuro.balance", "--game", game,
                   "--persona", persona, "--gens", gens, "--seeds", *seeds]
            try:
                subprocess.run(cmd)
            except KeyboardInterrupt:
                print(_RED("\n  Sweep interrupted — finished probes are already merged into the report."))
                return

    play_chime()
    if input(_DIM("\n    Open the Balance Command center? [Y/n]: ")).strip().lower() not in ("n", "no"):
        subprocess.run([sys.executable, "-m", "code.neuro.report", "--open"])


def delete_logs_and_models():
    """Permanently delete all training runs (populations, checkpoints, results)."""
    _section("DANGER  ›  Delete Logs & Models")
    confirm = input(
        f"This will permanently delete '{RUNS_DIR}/' (all populations, checkpoints, results).\n"
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

    safe_clear_dir(RUNS_DIR)

    print("\n🧹 All logs and models deleted successfully.\n")


def show_project_status():
    """
    Display comprehensive project status
        - Games, their enabled levels, and which have trained runs
    """
    _section("STATUS  ›  Project Overview")
    games = get_available_games()
    trained = get_trained_games()

    print(f"Available games: {len(games)}")
    for g in games:
        flag = _GRN("✓ Trained") if g in trained else _DIM("○ Not trained")
        levels = get_levels_for_game(g)
        print(f"   {g}: {flag}  {_DIM(f'({len(levels)} levels enabled)')}")

    runs = get_run_dirs()
    print(f"\nTotal trained runs: {len(runs)}")
    for p in runs:
        try:
            state = json.loads((p / "state.json").read_text(encoding="utf-8"))
            gens = state.get("generation", 0)
            best = state.get("best_fitness", 0.0)
            wins = sum(r.get("wins", 0) for r in state.get("history", []))
            win_str = _GRN(f"{wins} wins") if wins else _DIM("0 wins")
            print(f"   {p.name}: gen {gens}, best {best:.1f}, {win_str}")
        except Exception:
            print(f"   {p.name}: {_DIM('(unreadable state.json)')}")

    print(f"\nAlgorithm: {_WHT('fixed-topology GA')}  {_DIM('(elitism + tournament + crossover + mutation)')}")
    print(f"Network:   {_WHT('14 sensors → 16 tanh → 3 (left/right/jump)')}")
    print()


# ============================================================================
# MAIN LOOP
# ============================================================================

def main():
    import warnings
    warnings.filterwarnings(
        "ignore",
        message=r".*pkg_resources is deprecated as an API.*"
    )

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
        "11": ("dashboard",            run_dashboard),
        "12": ("balance_command",      open_balance_command),
        "14": ("balance",              run_balance_report),
        "15": ("full_sweep",           run_full_sweep),
        "13": ("delete_all",           delete_logs_and_models),
        "c":  ("clear_cli",            clear_cli),
        "0":  ("exit",                 None),
    }

    while True:
        setup_project()
        clear_cli()

        _print_header(get_available_games(), get_enabled_level_count(),
                      get_trained_games(), get_trained_models_count())

        _section("TRAIN")
        print(_menu_item("1",  "Project Status"))
        print(_menu_item("2",  "Train Single",        "pick game / level / generations"))
        print(_menu_item("3",  "Train All (1 game)",  "toggle levels, back-to-back runs"))
        print(_menu_item("4",  "Train Full Grid",     "all games × enabled levels"))

        _section("PLAY")
        print(_menu_item("5",  "Play Manually",       "keyboard controls"))
        print(_menu_item("6",  "Watch Agent",         "visualize a trained model"))
        print(_menu_item("7",  "Watch All Models",    "side-by-side grid"))
        print(_menu_item("8",  "Watch Random Agent",  "random actions"))

        _section("TOOLS")
        print(_menu_item("9",  "Level Editor",         "paint tiles, place entities"))
        print(_menu_item("10", "Toggle Levels",        "enable / disable levels in config"))
        print(_menu_item("11", "Dashboard",            "live training UI in browser"))
        print(_menu_item("12", "Balance Command",      "open the command center (all runs + probes)"))
        print(_menu_item("14", "Balance Report",       "multi-seed level difficulty ± CI"))
        print(_menu_item("15", "Full Sweep",           "games × personas × seeds, parallel"))
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
