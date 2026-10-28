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
import threading
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
# macOS: after any Python crash, AppKit asks "reopen windows?" before the next pygame window opens, and that
# hidden dialog looks like a frozen game. The game select pop-up and its games skip it (they ignore unknown args).
NO_RESTORE = ["-ApplePersistenceIgnoreState", "YES"] if sys.platform == "darwin" else []
DASHBOARD_URL = "http://127.0.0.1:8000/mario/index.html"
DISABLED_LEVELS_KEY = "disabled_levels"
CHIME_PATH = Path("chime.wav")

# Adapter keys → game_config.yaml section key (platformer sits at the root).
def _active_games() -> list[str]:
    """Roster minus archived games (code/neuro/adapters.py). Read without importing numpy/pygame."""
    try:
        import re
        src = (Path(__file__).parent / "code" / "neuro" / "adapters.py").read_text(encoding="utf-8")
        archived = set(re.findall(r'"(\w+)"', re.search(r"ARCHIVED_GAMES = \{([^}]*)\}", src).group(1)))
        order = re.findall(r'^    "(\w+)": \w+Adapter,', src, re.M)
        return [g for g in order if g not in archived] or ["mario", "meatboy", "bomberman"]
    except Exception:
        return ["mario", "meatboy", "bomberman"]


GAMES = _active_games()
INDEXED_GAMES = {"meatboy", "bomberman"}  # levels are list indices (their own <game>_config.yaml)
CONFIG_KEY = {"mario": "platformer"}
OWN_LEVEL_FILES = {"megaman": "megaman_levels.yaml"}  # named levels in the game's own, editor-owned file

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
    """Victory chime after a fully successful training batch (winsound / afplay / aplay)."""
    if not CHIME_PATH.exists():
        return
    try:
        if HAS_WINSOUND:
            winsound.PlaySound(str(CHIME_PATH), winsound.SND_FILENAME)
            return
        player = shutil.which("afplay") or shutil.which("aplay")
        if player:
            subprocess.Popen([player, str(CHIME_PATH)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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
    if game in INDEXED_GAMES:
        try:
            import yaml
            data = yaml.safe_load(Path(f"code/games/{game}_config.yaml").read_text(encoding="utf-8")) or {}
            active = data.get("active_levels")  # the AST evaluation level; the others stay loadable by index
            return [str(i) for i in range(len(data.get("levels", []))) if active is None or i in active]
        except Exception:
            return []
    if game in OWN_LEVEL_FILES:
        try:
            import yaml
            data = yaml.safe_load(Path(f"code/games/{OWN_LEVEL_FILES[game]}").read_text(encoding="utf-8")) or {}
            return list((data.get("levels") or {}).keys())
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
    """Every dir under runs/ holding a trained population (state.json + best.npz),
    including balance probes (runs/probes/<game>/<persona>/<tag>/<level>_<seed>)."""
    if not RUNS_DIR.exists():
        return []
    found = []
    for root, dirs, files in os.walk(RUNS_DIR):
        if Path(root) == RUNS_DIR:    # hundreds of ablation cells: watch those from the command center, and
            dirs[:] = [d for d in dirs if d != "gasweep"]   # not walking their 200,000 files saves the menu 15 s
        if "state.json" in files and "best.npz" in files and os.path.basename(root) != "_replay":
            found.append(Path(root))
    return sorted(found)


def run_meta(run_dir: Path) -> dict:
    """Identity tags embedded in best.npz (game/level/persona/fitness/generation)."""
    try:
        import numpy as np
        return json.loads(str(np.load(run_dir / "best.npz")["meta"]))
    except Exception:
        return {}


def get_trained_games():
    return {guess_game_for_run(p) for p in get_run_dirs()}


def get_trained_models_count() -> int:
    return len(get_run_dirs())


def guess_game_for_run(run_dir: Path) -> str:
    game = run_meta(run_dir).get("game")
    if game in GAMES:
        return game
    for part in run_dir.relative_to(RUNS_DIR).parts:
        for g in GAMES:
            if part.startswith(g):
                return g
    return GAMES[0]


def tty_sane() -> None:
    """Enter ends a line again. A game or a decoder it starts can leave the terminal in raw input
    (Enter shows ^M and input() never returns): newlines, line editing and echo back on."""
    try:
        import termios
    except ImportError:                          # Windows has no termios
        return
    try:
        attrs = termios.tcgetattr(sys.stdin)
        attrs[0] |= termios.ICRNL
        attrs[3] |= termios.ICANON | termios.ECHO
        termios.tcsetattr(sys.stdin, termios.TCSANOW, attrs)
    except (termios.error, OSError, ValueError):   # stdin is a pipe or closed, not a terminal
        pass


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


def toggle_select(title, options, default_indices=None, min_select=1, show_desc=None, single=False):
    """
    The checklist every menu choice goes through (Train All, sweeps, pickers).

    Renders a numbered checklist where each item can be toggled on/off.
    Supports comma-separated input ("1,3") and ranges ("1-3") in one go.
    Returns the confirmed list of selected items, or None if user typed "0" (back).

    Type a number to flip that item on/off.  Press Enter to confirm.
    show_desc: optional dict {option: description_string} for extra info per row.
    single: a choice of one (pick_one): typing a number moves the one tick.
    """
    selected = set([0] if default_indices is None else default_indices)   # 0-based indices; [] = none ticked
    if not options:
        print(_DIM("    Nothing to choose from."))
        return None

    while True:
        mode = "pick one" if single else "toggle"
        print(f"\n    {_BOLD(_CYAN('▸'))} {_BOLD(title)}  {_DIM(f'({mode} · Enter to confirm)')}")
        print(_DIM("    Type a number to choose" if single else "    Type numbers to toggle")
              + _DIM("  ·  Enter to confirm  ·  0 = Back"))
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
                if single:
                    selected = {idx}
                elif idx in selected:
                    selected.discard(idx)
                else:
                    selected.add(idx)
            else:
                print(_RED(f"    Invalid: {n}"))


def pick_one(title, options, default=None, show_desc=None):
    """A choice of one on the same checklist screen: the default starts ticked, a number moves the tick."""
    start = options.index(default) if default in options else 0
    picked = toggle_select(title, options, default_indices=[start], show_desc=show_desc, single=True)
    return picked[0] if picked else None


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


def run_with_loading(cmd, env=None, label="Loading", on_ready=None) -> int:
    """Run a PEAK program that opens a window or a dashboard, with a loading bar in the terminal until it
    is up (the program touches PEAK_READY_FILE then) or it ends; on_ready runs once it is up."""
    import tempfile
    ready = Path(tempfile.gettempdir()) / f"peak_ready_menu_{os.getpid()}"
    ready.unlink(missing_ok=True)
    proc = subprocess.Popen(cmd, env={**(env or os.environ), "PEAK_READY_FILE": str(ready)})
    start, width = time.time(), 24
    try:
        while proc.poll() is None and not ready.exists():
            done = int(width * min(0.95, (time.time() - start) / 5))   # about 5 s for the slowest to show
            print(f"\r    {_DIM(label)}  {_CYAN('█' * done + '·' * (width - done))}", end="", flush=True)
            time.sleep(0.1)
        print("\r" + " " * (width + len(label) + 8) + "\r", end="", flush=True)
        if ready.exists() and on_ready:
            on_ready()
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        return proc.wait()
    finally:
        ready.unlink(missing_ok=True)


def execute_training_run(cmd) -> bool:
    """Run one trainer subprocess; echo the command like classic PEAK did."""
    print(_DIM("  >>> " + " ".join(cmd)))
    try:
        return subprocess.run(cmd).returncode == 0
    except KeyboardInterrupt:
        print(_DIM("\n    Training interrupted."))
        return False


def execute_parallel_runs(runs, workers) -> int:
    """Run (label, cmd, run_dir) trainer subprocesses, at most `workers` at once, started in list order.
    Each run's output goes to <run_dir>/train.log; one line per finished run. Returns how many succeeded."""
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")  # 1 core per run
    pending, running, successful, done = list(runs), [], 0, 0
    try:
        while pending or running:
            while pending and len(running) < workers:
                label, cmd, run_dir = pending.pop(0)
                Path(run_dir).mkdir(parents=True, exist_ok=True)
                log = open(Path(run_dir) / "train.log", "w", encoding="utf-8")
                proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
                running.append((label, proc, log, time.time(), run_dir))
                print(_DIM(f"    started   {label}"))
            time.sleep(1)
            for item in [r for r in running if r[1].poll() is not None]:
                label, proc, log, t0, run_dir = item
                log.close()
                running.remove(item)
                done += 1
                ok = proc.returncode == 0
                successful += int(ok)
                mins = (time.time() - t0) / 60
                status = _GRN("done  ") if ok else _RED(f"FAILED (exit {proc.returncode}, see {run_dir}\\train.log)")
                print(f"  {_YEL(f'[{done}/{len(runs)}]')}  {status}  {label}  {_DIM(f'{mins:.1f} min')}")
    except KeyboardInterrupt:
        print(_DIM("\n    Training interrupted — stopping the running jobs."))
        for _, proc, log, _, _ in running:
            proc.terminate()
            proc.wait()
            log.close()
    return successful


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


def _prompt_persona_sensors_tag():
    """Shared by every TRAIN entry: (persona, sensors, tag) or None on back."""
    persona = pick_one("PERSONA  ·  who should the agents play like?", PERSONA_CHOICES,
                       default="experienced", show_desc=PERSONA_DESC)
    if not persona:
        return None
    sensors = pick_one("SENSORS  ·  what the agents see", SENSOR_CHOICES, default="rays", show_desc=SENSOR_DESC)
    if not sensors:
        return None
    raw = input(_DIM("\n    Run tag (optional, appended to the run dir name): ")).strip()
    tag = "".join(c if c.isalnum() or c in "-_" else "_" for c in raw)
    return persona, sensors, tag


def _run_dir(game, level, persona, sensors, tag) -> Path:
    """runs/<game>[_<level>][_<persona>][_<sensors>][_<tag>] — every config gets its own population."""
    name = game if not level else f"{game}_{level}"
    if persona != "experienced":
        name += f"_{persona}"
    if sensors != "rays":  # a grid population has a different genome size — never share a run dir
        name += f"_{sensors}"
    if tag:
        name += f"_{tag}"
    return RUNS_DIR / name


def _print_run_summary(**fields):
    W = 50
    print()
    print(_DIM("    " + "─" * W))
    print(f"    {_BOLD('Run Summary')}")
    for k, v in fields.items():
        print(f"    {k + ':':<13}{_WHT(str(v))}")
    print(_DIM("    " + "─" * W))
    return input(_BOLD("    ⟫ Proceed? [Y/n]: ")).strip().lower() not in ("n", "no")


def run_training():
    """Train Single — one game, one level, live dashboard."""
    _refresh_screen()
    _section("TRAIN  ›  Single Run")

    game = pick_one("GAME", get_available_games(), default="mario")
    if not game:
        return

    levels = get_levels_for_game(game)
    level = None
    if levels:
        level = pick_one(f"LEVEL  ·  {game}", ["auto (first level)"] + levels, default="auto (first level)")
        if level is None:
            return
        if level.startswith("auto"):
            level = None

    pst = _prompt_persona_sensors_tag()
    if not pst:
        return
    persona, sensors, tag = pst
    gens = _prompt_gens()
    if gens == "invalid":
        return
    mode = pick_one("SPEED", ["real-time", "turbo"], default="real-time",
                    show_desc={"real-time": "game speed, every feed live", "turbo": "max speed, only the watched agent streams"})
    if not mode:
        return
    turbo = mode == "turbo"

    run_dir = _run_dir(game, level, persona, sensors, tag)
    if not _print_run_summary(Game=game, Level=level or "auto", Persona=persona, Sensors=sensors,
                              Tag=tag or "-", Generations=gens or "until stopped",
                              Mode="turbo" if turbo else "real-time", **{"Run dir": run_dir}):
        return

    print()
    print(f"    Dashboard  →  {_CYAN(DASHBOARD_URL)}")
    print()
    ok = execute_training_run(_trainer_cmd(game, level, gens, turbo, run_dir=run_dir,
                                           extra=("--persona", persona, "--sensors", sensors)))
    print_training_summary(1, int(ok), int(not ok))
    if ok:
        play_chime()


def train_all_models_for_game():
    """Train All (1 game) — every selected level, back to back, headless turbo."""
    _refresh_screen()
    _section("TRAIN  ›  All Levels (1 game)")

    game = pick_one("GAME", get_available_games(), default="mario")
    if not game:
        return

    levels = get_levels_for_game(game)
    if not levels:
        print(_RED("  ✖  No levels found for this game."))
        return

    chosen = toggle_select(f"LEVELS  ·  {game}", levels, default_indices=list(range(len(levels))))
    if not chosen:
        return
    _train_batch([(game, lvl) for lvl in chosen], Game=game, Levels=", ".join(chosen))


def train_complete_grid():
    """Train Full Grid — every game × its enabled levels, headless turbo."""
    _refresh_screen()
    _section("TRAIN  ›  Full Grid")
    jobs = [(g, lvl) for g in get_available_games() for lvl in (get_levels_for_game(g) or [None])]
    _train_batch(jobs, Games=", ".join(get_available_games()))


def _train_batch(jobs, **summary):
    """Back-to-back headless turbo runs for every (game, level) job — shared by Train All / Full Grid."""
    pst = _prompt_persona_sensors_tag()
    if not pst:
        return
    persona, sensors, tag = pst
    gens = _prompt_gens("e.g. 60")
    if gens in ("invalid", None):
        if gens is None:
            print(_RED("  A generation count is required for batch runs."))
        return
    if not _print_run_summary(**summary, Persona=persona, Sensors=sensors, Tag=tag or "-",
                              Generations=f"{gens} per run", **{"Total runs": len(jobs)}):
        return

    print(f"\n    Dashboard  →  {_CYAN(DASHBOARD_URL)}  {_DIM('(follows each run; reconnects between them)')}")
    successful = 0
    for i, (game, level) in enumerate(jobs, 1):
        print(f"\n  {_YEL(f'[{i}/{len(jobs)}]')}  {_WHT(game)} | {_WHT(level or 'auto')}")
        ok = execute_training_run(
            _trainer_cmd(game, level, gens, turbo=True, run_dir=_run_dir(game, level, persona, sensors, tag),
                         extra=("--persona", persona, "--sensors", sensors)))
        successful += int(ok)

    print_training_summary(len(jobs), successful, len(jobs) - successful)
    if successful == len(jobs):
        play_chime()


def train_all_personas():
    """Train All Personas — per game, one population per persona walks every enabled level in order (the
    trainer's curriculum: a generation with 3 winners moves it to the next level, nets carry over). Several
    seeds run seed by seed: every game × persona on one seed finishes before the next seed starts."""
    _refresh_screen()
    _section("TRAIN  ›  All Personas (curriculum)")
    games = toggle_select("GAMES", get_available_games(),
                          default_indices=[i for i, g in enumerate(get_available_games())
                                           if g in ("mario", "meatboy", "bomberman")])
    if not games:
        return
    personas = toggle_select("PERSONAS", PERSONA_CHOICES, default_indices=list(range(len(PERSONA_CHOICES))))
    if not personas:
        return
    gens = _prompt_gens("total for all levels, e.g. 800 for Mario's 4")
    if gens in ("invalid", None):
        if gens is None:
            print(_RED("  A generation count is required: a curriculum run stops at its budget."))
        return
    seeds_raw = input(_DIM("    Seeds, space-separated [1234]: ")).strip()
    seeds = [s for s in seeds_raw.split() if s.lstrip("-").isdigit()] or ["1234"]
    jobs = [(g, p, s) for s in seeds for g in games for p in personas]  # seed outermost: one seed at a time
    default_workers = min(len(jobs), max(1, (os.cpu_count() or 2) - 1))
    workers_raw = input(_DIM(f"    Runs at once, 1 core each (1 = live dashboard) [{default_workers}]: ")).strip()
    workers = max(1, min(len(jobs), int(workers_raw))) if workers_raw.isdigit() else default_workers
    if not _print_run_summary(Games=", ".join(games), Personas=", ".join(personas), Seeds=" ".join(seeds),
                              Generations=f"{gens} per persona, across all levels", **{"Total runs": len(jobs)},
                              Parallel=f"{workers} at once" + (" (logs → <run dir>/train.log)" if workers > 1 else "")):
        return

    def run_args(game, persona, seed):
        run_dir = _run_dir(game, None, persona, "rays", f"curriculum_seed{seed}")
        cmd = _trainer_cmd(game, None, gens, turbo=True, serve=workers == 1, run_dir=run_dir,
                           extra=("--persona", persona, "--seed", seed, "--best"))  # GA-sweep config per game
        return f"{game} | {persona} | seed {seed} | all levels", cmd, run_dir

    if workers > 1:
        print(f"\n    Running {len(jobs)} runs, {workers} at once  {_DIM('(Ctrl+C stops them all)')}")
        successful = execute_parallel_runs([run_args(*job) for job in jobs], workers)
        print_training_summary(len(jobs), successful, len(jobs) - successful)
        if successful == len(jobs):
            play_chime()
        return

    print(f"\n    Dashboard  →  {_CYAN(DASHBOARD_URL)}  {_DIM('(follows each run; reconnects between them)')}")
    successful = 0
    for i, job in enumerate(jobs, 1):
        label, cmd, _ = run_args(*job)
        print(f"\n  {_YEL(f'[{i}/{len(jobs)}]')}  {_WHT(label)}")
        ok = execute_training_run(cmd)
        successful += int(ok)

    print_training_summary(len(jobs), successful, len(jobs) - successful)
    if successful == len(jobs):
        play_chime()


# ============================================================================
# PLAY / WATCH
# ============================================================================

# Keys exactly as code/games/tools/manual_play.py (and MeatboyPlayer) read them.
_PLAY_CONTROLS = {
    "mario":   [("A / D", "Move"), ("SHIFT / J", "Run"), ("SPACE / W", "Jump"),
                ("Z", "Fire (with fire flower)")],
    "meatboy": [("A / D  ← / →", "Move"), ("SHIFT", "Run"), ("SPACE / W / ↑", "Jump (wall-jump on contact)")],
    "bomberman": [("W A S D  ↑ ← ↓ →", "Move"), ("SPACE / Z", "Drop a bomb"), ("", "Exit opens once every enemy is dead")],
    "megaman": [("A / D  ← / →", "Move"), ("W / S  ↑ / ↓", "Climb ladder"), ("SPACE", "Jump"), ("Z / J", "Fire"),
                 ("I", "Toggle invincibility (debug)"), ("", "A controller works too: stick / d-pad, Cross, Square")],
}
_DEBUG_KEYS = [("F1", "Sensor rays"), ("F2", "Free camera (I J K L to pan)"), ("F3", "Slow motion"),
               ("F4", "Hitboxes"), ("F5", "Agent max view"), ("F6", "Background grid"),
               ("F7", "Close / open the debug panel")]


def _print_controls(game, level):
    W = 50
    print()
    print(_DIM("    " + "─" * W))
    print(f"    {_BOLD('Controls')}   {_WHT(game)}  ·  {_WHT(level or 'auto')}")
    for key, what in _PLAY_CONTROLS[game] + [("ESC", "Quit")]:
        print(f"    {_YEL(f'{key:<16}')}{what}")
    if game != "bomberman" and game not in OWN_LEVEL_FILES:  # those cores have no debug manager
        print()
        print(f"    {_BOLD('Debug overlays')}")
        for key, what in _DEBUG_KEYS:
            print(f"    {_YEL(f'{key:<16}')}{what}")
    print(_DIM("    " + "─" * W))
    print()


# ── User study ────────────────────────────────────────────────────────────────
WARMUP_MINUTES = 5          # warm-up: a normal playthrough of the game, stopped after this long, not recorded
BASELINE_RUNS = 3           # recorded attempts on the evaluation level
STUDY_LIVES = 3             # lives per recorded run; fewer rows than this and no win means it was quit
STUDY_DIR = RUNS_DIR / "user_study"


def _study_row(old: dict, game: str, participant: str, phase: str, run: int) -> dict:
    """One manual-play row (episode_log.read already turned 2D manual play's older CSV into its schema)
    with the study's labels."""
    return {**old, "game": game, "persona": participant, "source": "human", "session": f"{participant}/{phase}",
            "gen": 0, "agent": run}


def _study_env(mutant: dict | None = None) -> dict:
    """A real window, playing the given mutant (or the unchanged game)."""
    env = {k: v for k, v in os.environ.items() if k not in ("SDL_VIDEODRIVER", "PEAK_MUTANT", "PEAK_MM1_MUTANT")}
    if mutant:
        env["PEAK_MUTANT"] = str(mutant["dir"])
    return env


def _study_attempt(ui, game: str, participant: str, phase: str, run: int, mutant: dict | None = None) -> bool:
    """Play one recorded run on the evaluation level (or its mutant) with the game's usual 3 lives,
    a row per life; store it as
    runs/user_study/<participant>/<game>/<phase>/run<run>/episodes.csv. False if none was recorded, or if the
    run was quit before its last life (its rows are kept, labelled quit, so it can be played again)."""
    import tempfile
    from code.games.eval_mutants import EVAL_LEVELS
    from code.stats import episode_log
    tmp = Path(tempfile.mkdtemp(prefix="study_"))
    cmd = [sys.executable, "-m", "code.games.tools.manual_play", "--game", CONFIG_KEY.get(game, game),
           "--fps", "60", "--level", EVAL_LEVELS[game][0], "--record", str(tmp), "--player", participant,
           "--episodes", "1", "--lives", str(STUDY_LIVES)]
    ui.run_game(cmd, _study_env(mutant))
    found = list(tmp.glob("*/*/episodes.csv"))
    rows = episode_log.read(str(found[0])) if found else []
    inputs = found[0].with_name("inputs.csv") if found else None
    inputs_text = inputs.read_text(encoding="utf-8") if inputs and inputs.exists() else None
    shutil.rmtree(tmp, ignore_errors=True)
    if not rows:
        return False
    rows = [_study_row(r, game, participant, phase, run) for r in rows]
    result = ("cleared" if any(r["status"] == "WON" for r in rows)
              else "game_over" if len(rows) >= STUDY_LIVES else "quit")   # quit: the window closed with lives left
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")      # a session played again with ADD TO lands in the same folders
    rows = [{**r, "run_result": result, "lives_used": len(rows), "life": i, "recorded_at": stamp}
            for i, r in enumerate(rows, 1)]
    if mutant:
        rows = [{**r, "mutant": mutant["id"], "mutant_class": mutant["class"]} for r in rows]
    out = STUDY_DIR / participant / game / phase / f"run{run}" / "episodes.csv"
    episode_log.append(str(out), rows)
    if inputs_text:                                 # the buttons held each frame, beside the rows
        log = out.with_name("inputs.csv")
        with open(log, "a", encoding="utf-8", newline="") as f:
            f.write(inputs_text if not log.exists() or not log.stat().st_size else inputs_text.split("\n", 1)[1])
    said = {"cleared": _GRN("Cleared the level"), "game_over": _RED("Game over"), "quit": _YEL("Quit early")}
    print(f"    {said[result]} ({len(rows)} {'life' if len(rows) == 1 else 'lives'})   Saved  →  {_CYAN(str(out))}")
    return result != "quit"


def _study_ask(ui, *a, **k):
    """ui.ask on a study screen. B runs and fires in the games, so it never backs out here; ESC (also
    the games' quit key) ends the session (None) only when the experimenter confirms it."""
    while True:
        got = ui.ask(*a, b_backs=False, **k)
        if got is not None or ui.ask("END SESSION?", ["NO, GO ON", "YES, END IT"], 0, "A  OK", b_backs=False) == 1:
            return got


def _study_attempts(ui, game: str, participant: str, phase: str, plays: list) -> bool:
    """Each play (None, or a mutant) once, as run1, run2, ...; a run that ended early (window closed, or quit
    with lives left) asks to try that one again. False if the experimenter ended the session."""
    for run, mutant in enumerate(plays, 1):
        if _study_ask(ui, phase.upper(), [f"ATTEMPT {run} OF {len(plays)}"], None, "A  START") is None:
            return False
        while not _study_attempt(ui, game, participant, phase, run, mutant):
            got = _study_ask(ui, "NOT FINISHED", ["TRY AGAIN", "SKIP"], 0, "THE RUN ENDED EARLY")
            if got is None:
                return False
            if got == 1:
                break
    return True


def study_session(ui) -> bool:
    """One participant's User Study, run in the game select pop-up (ui: code/games/tools/game_select.py's
    ask, ask_text, show_game and run_game): participant ID, game, a timed warm-up playthrough, the evaluation
    level BASELINE_RUNS times, then its five mutants once each in a random order; each attempt in its own
    run folder. False when the ID screen was left with ESC (close the pop-up), else True (next participant)."""
    import random
    import re
    from code.games.eval_mutants import EVAL_LEVELS, eval_set
    participant = "".join(c if c.isalnum() or c in "-_" else "_" for c in ui.ask_text("PARTICIPANT ID") or "")
    if not participant:
        return False
    games = [g for g in get_available_games() if g in EVAL_LEVELS]
    pick = ui.ask("SELECT GAME", [ui.NAMES.get(g, g) for g in games], 0, "A  START      ESC  EXIT", b_backs=False,
                  games=games)
    if pick is None:
        return True
    game = games[pick]
    ui.show_game(game)
    mutants = eval_set(game)
    if len(mutants) < 5:
        print(_RED(f"\n    Only {len(mutants)} of {game}'s 5 mutants are on disk. Make them first: "
                   + ("python -m code.games.megaman_mutants" if game == "megaman" else "python -m code.games.eval_mutants")))
        ui.ask("NOT READY", [f"ONLY {len(mutants)} OF 5 MUTANTS", "SEE THE TERMINAL"], None, "A  BACK", b_backs=False)
        return True
    where = STUDY_DIR / participant / game
    if where.exists() and ui.ask("ALREADY RECORDED", [f"ADD TO {participant}", "CANCEL"], 1, b_backs=False) != 0:
        return True
    # The pixel font has no arrows or brackets: drop them, as the pad overlay in the game shows the buttons.
    plain = lambda t: re.sub(r"\s+", " ", re.sub(r"[^A-Za-z0-9 ./:-]", " ", t)).strip(" /")
    if _study_ask(ui, "WARM-UP", [f"{plain(key):<10}{plain(what)}" for key, what in _PLAY_CONTROLS[game]], None,
                  f"{WARMUP_MINUTES:g} MIN, NOT RECORDED      A  START", size=2) is None:
        return True
    ui.run_game([sys.executable, "-m", "code.games.tools.manual_play", "--game", CONFIG_KEY.get(game, game),
                 "--fps", "60", "--quit-after", str(WARMUP_MINUTES),   # counts down on screen, pauses excluded
                 "--lives", "0"], _study_env())                        # practice: lives never run out
    if _study_attempts(ui, game, participant, "baseline", [None] * BASELINE_RUNS):
        random.shuffle(mutants)             # the participant isn't told which mutant is which
        if _study_attempts(ui, game, participant, "mutation", mutants):
            ui.ask("SESSION DONE", ["THANK YOU"], None, "A  NEXT PARTICIPANT", b_backs=False)
    print(f"\n    {_GRN('Session over.')}  Data  →  {_CYAN(str(where))}")
    return True


def run_user_study():
    """User Study: the whole session runs in the game select pop-up (study_session), so the participant
    only ever looks at the game windows."""
    _refresh_screen()
    _section("USER STUDY")
    print(_DIM("    The study runs in the pop-up window: type the participant ID there, then pick the game.\n"
               "    After each session it asks for the next ID; ESC there closes it."))
    env = {k: v for k, v in os.environ.items() if k != "SDL_VIDEODRIVER"}   # a real window
    run_with_loading([sys.executable, "-m", "code.games.tools.game_select", "--study"] + NO_RESTORE, env=env,
                     label="Loading the study")


def run_manual_play():
    """Play Manually — a game selection screen; quitting a game comes back to it, ESC there exits."""
    _refresh_screen()
    _section("PLAY  ›  Manual")
    # Strip the dummy SDL driver so a real window opens.
    proc_env = os.environ.copy()
    proc_env.pop("SDL_VIDEODRIVER", None)
    cmd = [sys.executable, "-m", "code.games.tools.game_select"]
    player = None
    if input(_DIM("    Record this session like an agent run? [y/N]: ")).strip().lower().startswith("y"):
        raw = input(_DIM("    Player name [human]: ")).strip()
        player = "".join(c if c.isalnum() or c in "-_" else "_" for c in raw) or "human"  # it becomes a folder name
        cmd += ["--record", "--player", player]
    cmd += NO_RESTORE
    print(_DIM("    Pick a game on screen: d-pad / arrows, A / Enter to play, B to go back, ESC to exit."))
    print()
    run_with_loading(cmd, env=proc_env, label="Loading the games")
    if player:
        print(f"    Recorded attempts  →  {_CYAN(str(Path('runs') / 'manual' / '<game>' / player / 'episodes.csv'))}")


def _pick_run(title="RUN  ·  strongest first"):
    runs = get_run_dirs()
    if not runs:
        print(_RED("  ✖  No trained populations under runs/. Train (2) or probe (13/14) first."))
        return None
    metas = [run_meta(p) for p in runs]
    order = sorted(range(len(runs)), key=lambda i: -(metas[i].get("fitness") or 0))  # strongest first
    labels = []
    for i in order:
        m = metas[i]
        info = " · ".join(str(m[k]) for k in ("level", "persona") if m.get(k))
        fit = f"fit {m['fitness']:,.0f}" if m.get("fitness") is not None else ""
        labels.append(f"{runs[i].relative_to(RUNS_DIR)}  {_DIM('[' + ' · '.join(x for x in (info, fit) if x) + ']')}")
    name = pick_one(title, labels, default=labels[0])
    return runs[order[labels.index(name)]] if name else None


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
    run_with_loading([sys.executable, "-m", "code.neuro.trainer", "--game", game, "--replay", str(best)],
                     label="Loading the agent", on_ready=lambda: webbrowser.open(DASHBOARD_URL))


def watch_all_models():
    """Watch All Models — resume a run with the live dashboard grid (all 10 envs side by side)."""
    _refresh_screen()
    _section("WATCH  ›  All Envs (dashboard grid)")

    run_dir = _pick_run()
    if not run_dir:
        return
    if any(d in run_dir.parents for d in (RUNS_DIR / "probes", RUNS_DIR / "ast")):
        print(_RED("\n    That is a finished probe: training it on would overwrite a published agent. "
                   "Watch it with Watch Trained Agent (6)."))
        return

    game = guess_game_for_run(run_dir)
    print(f"\n    Dashboard  →  {_CYAN(DASHBOARD_URL)}")
    print(_DIM("    Training continues while you watch. Ctrl+C to stop.\n"))
    run_with_loading([sys.executable, "-m", "code.neuro.trainer", "--game", game, "--resume", str(run_dir)],
                     label="Loading the run")


def watch_random_agent():
    """Watch Random Agent — random actions, human window."""
    _refresh_screen()
    _section("WATCH  ›  Random Agent")

    game = pick_one("GAME", get_available_games(), default="mario")
    if not game:
        return

    proc_env = os.environ.copy()
    proc_env.pop("SDL_VIDEODRIVER", None)
    print(_DIM(f"\n    Launching {game} with random actions... (ESC to quit)\n"))
    run_with_loading(
        [sys.executable, "-m", "code.games.tools.manual_play",
         "--game", CONFIG_KEY.get(game, game), "--fps", "60", "--random"],
        env=proc_env, label="Loading the game"
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
        Path("level_editor.py"),
    ]

    editor_path = next((p for p in candidates if p.exists()), None)
    if editor_path is None:
        print("  Could not find level_editor.py.")
        print("  Checked: " + ", ".join(str(c) for c in candidates))
        return

    game_choices = ["platformer", "megaman"]
    game = pick_one("GAME  ·  which game to edit", game_choices, default=game_choices[0])
    if not game:
        return

    env_vars = os.environ.copy()
    env_vars.pop("SDL_VIDEODRIVER", None)
    cmd = [sys.executable, str(editor_path), "--game", game]

    print("  >>>", " ".join(cmd), "\n")
    try:
        code = run_with_loading(cmd, env=env_vars, label="Loading the editor")
        if code:
            print(f"\n  Editor exited with error code {code}")
    except KeyboardInterrupt:
        print("\n  Editor closed.")


def run_toggle_levels():
    """Enable or disable levels in game_config.yaml from the CLI."""
    config_path, _, yaml = _load_game_config_yaml()
    if not config_path or yaml is None:
        print("\n  Could not find a readable game_config.yaml")
        return

    game = "platformer"   # the one game whose levels live in game_config.yaml
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception as e:
        print(f"\n  Error reading config: {e}")
        return
    active, disabled = _game_level_maps(data, game, create=False)
    ids = list(active) + list(disabled)
    files = {lid: (e.get("file", "???") if isinstance(e, dict) else "???") for lid, e in {**active, **disabled}.items()}
    chosen = toggle_select(f"LEVELS  ·  {game}  (ticked = enabled)", ids,
                           default_indices=list(range(len(active))), show_desc=files)
    if chosen is None:
        return
    for lid in ids:
        on = lid in chosen
        if on != (lid in active):
            ok = _toggle_game_level_in_config(game, lid, enable=on)
            state = _GRN("enabled") if on else _RED("disabled")
            print(f"    '{lid}' {state}" if ok else _RED(f"    Failed to toggle '{lid}'"))


def run_dashboard():
    """Open the live training dashboard in the default browser."""
    _section("TOOLS  ›  Dashboard")
    print(f"\n    Dashboard  →  {_CYAN(DASHBOARD_URL)}")
    print(_DIM("    (a trainer must be running for live data — start one via TRAIN)"))
    try:
        webbrowser.open(DASHBOARD_URL)
    except Exception as e:
        print(_RED(f"  ✖  Could not open browser: {e}"))


def open_command_center():
    """Command Center: regenerate the pages from runs/ and open them."""
    _refresh_screen()
    _section("COMMAND CENTER")
    print(_DIM("    Regenerating from runs/balance, runs/probes, and every training run...\n"))
    try:
        subprocess.run([sys.executable, "-m", "code.neuro.command_center", "--serve", "--open"])
    except KeyboardInterrupt:
        pass


def run_readme_figures():
    """README Figures — regenerate docs/img/fig_*.png from runs/balance and refresh the README stamp."""
    _refresh_screen()
    _section("TOOLS  ›  README Figures")
    print(_DIM("    fig_difficulty · fig_capacity · fig_sensors · fig_knobs  ←  runs/balance/*.json"))
    what = pick_one("WRITE", ["figures and README", "figures only"], default="figures and README",
                    show_desc={"figures and README": "also the stamp between the <!-- figures --> markers"})
    if not what:
        return
    readme = what == "figures and README"
    cmd = [sys.executable, "-m", "code.neuro.figures"] + (["--readme"] if readme else [])
    print(_DIM("  >>> " + " ".join(cmd) + "\n"))
    subprocess.run(cmd)


SENSOR_CHOICES = ["rays", "grid"]
SENSOR_DESC = {"rays": "6 raycasts + probes, 14 inputs", "grid": "3×11×11 tiles around the agent, 368 inputs"}


def _personas() -> dict[str, str]:
    """Persona registry (code/neuro/personas.py): {name: description}, experienced first."""
    try:
        from code.neuro.personas import PERSONAS
        found = {n: p.description for n, p in PERSONAS.items()}
    except Exception:
        found = {n: "" for n in ("experienced", "novice", "speedrunner")}
    return {n: found[n] for n in ["experienced"] + sorted(n for n in found if n != "experienced")}


PERSONA_DESC = _personas()
PERSONA_CHOICES = list(PERSONA_DESC)


def _sweep_prompts(n_modes: int = 1, default_gens: str = "40", modes_label: str = "sensor modes",
                   cost_note: str = ""):
    """Shared sweep selection: games, personas, generation budget, seeds (None = back).
    Every ENABLED level of each chosen game is probed."""
    games = toggle_select("GAMES", get_available_games(),
                          default_indices=[i for i, g in enumerate(get_available_games())
                                           if g in ("mario", "meatboy", "bomberman")])
    if not games:
        return None
    personas = toggle_select("PERSONAS", PERSONA_CHOICES,  # new personas opt-in: each one multiplies sweep cost
                             default_indices=[i for i, n in enumerate(PERSONA_CHOICES)
                                              if n in ("experienced", "novice", "speedrunner")])
    if not personas:
        return None
    gens_raw = input(_DIM(f"\n    Generation budget per probe [{default_gens}]: ")).strip()
    gens = gens_raw if gens_raw.isdigit() else default_gens
    seeds_raw = input(_DIM("    Seeds [1234 2025 31337]: ")).strip()
    seeds = [s for s in seeds_raw.split() if s.lstrip("-").isdigit()] or ["1234", "2025", "31337"]
    default_workers = max(1, (os.cpu_count() or 2) - 1)
    workers_raw = input(_DIM(f"    Runs at once (1 core each) [{default_workers}]: ")).strip()
    workers = max(1, int(workers_raw)) if workers_raw.isdigit() else default_workers

    n_levels = sum(len(get_levels_for_game(g)) for g in games)
    n_jobs = n_levels * len(seeds) * len(personas) * n_modes
    modes = f" × {n_modes} {modes_label}" if n_modes > 1 else ""
    print(f"\n    {_WHT(str(n_jobs))} probes ({n_levels} levels × {len(seeds)} seeds × "
          f"{len(personas)} personas{modes}), {workers} parallel workers")
    if cost_note:
        print(_DIM(f"    {cost_note}"))
    print(_DIM("    Only ENABLED levels are probed — use Toggle Levels [10] first if needed."))
    if input(_BOLD("    ⟫ Proceed? [Y/n]: ")).strip().lower() in ("n", "no"):
        return None
    return games, personas, gens, seeds, workers


def _sweep(games, personas, gens, seeds, workers, sensors_list, interrupted_msg,
           module: str = "code.neuro.balance", extra: tuple = ()) -> bool:
    """games × personas × sensor modes probe runs; False if interrupted."""
    for game in games:
        for persona in personas:
            for sensors in sensors_list:
                tag = f" · {sensors}" if len(sensors_list) > 1 else ""
                print(_BOLD(f"\n  ── {game} · {persona}{tag} " + "─" * 30))
                cmd = [sys.executable, "-m", module, "--game", game, "--persona", persona,
                       "--gens", gens, "--seeds", *seeds, "--sensors", sensors, "--workers", str(workers), *extra]
                try:
                    subprocess.run(cmd)
                except KeyboardInterrupt:
                    print(_RED(interrupted_msg))
                    return False
    return True


def _open_command_center(game: str = "mario", tab: str = "balance") -> None:
    if input(_DIM("\n    Open the Command Center? [Y/n]: ")).strip().lower() in ("n", "no"):
        return
    subprocess.run([sys.executable, "-m", "code.neuro.command_center"])  # rebuilds runs/site
    webbrowser.open((RUNS_DIR / "site" / "index.html").resolve().as_uri() + f"#{game}/{tab}")


def record_wall_videos():
    """Record Replay Wall: one looping video per game and persona for the Command Center (runs/videos/)."""
    _refresh_screen()
    _section("TOOLS  ›  Record Replay Wall")
    print(_DIM("    About 100 s per video, 24 videos; the Mega Man art folder must not be present.\n"))
    try:
        subprocess.run([sys.executable, "-m", "code.neuro.wall_video"])
    except KeyboardInterrupt:
        print(_RED("\n  Stopped: the videos written so far are kept."))


def publish_site():
    """Publish: build the Command Center and, after a yes, push it to GitHub Pages."""
    _refresh_screen()
    _section("TOOLS  ›  Publish to GitHub Pages")
    print(_DIM("    Builds runs/site, puts it on the gh-pages branch as one commit (the old landing page stays"))
    print(_DIM("    as tag landing-page-2026-03) and refuses while runs/balance or runs/ast have uncommitted changes.\n"))
    if input(_BOLD("    ⟫ Publish to https://code-sorcerylab.github.io/PEAK-DRL-Tool/? [y/N]: ")).strip().lower() != "y":
        return
    subprocess.run([sys.executable, "-m", "code.neuro.command_center", "--publish", "--push"])


def run_sensor_ablation():
    """Sensor Ablation — the Full Sweep selection, run once per sensor mode (rays, grid), then compared."""
    _refresh_screen()
    _section("BALANCE  ›  Sensor Ablation  (rays vs grid)")
    print(_DIM("    Same games × personas × seeds as Full Sweep, evolved once with raycasts and once with\n"
               "    the 3×11×11 tile grid; prints generations-to-first-win ± CI and win rate side by side."))
    picked = _sweep_prompts(n_modes=len(SENSOR_CHOICES))
    if not picked:
        return
    games, personas, gens, seeds, workers = picked
    if not _sweep(games, personas, gens, seeds, workers, SENSOR_CHOICES,
                  "\n  Ablation interrupted — finished probes are kept; rerun to fill the gaps.",
                  extra=("--ablation",)):  # runs/ablation + ablation_*.json — never touches the real sweeps
        return
    print()
    for game in games:
        for persona in personas:
            subprocess.run([sys.executable, "-m", "code.neuro.balance", "--game", game,
                            "--persona", persona, "--gens", gens, "--compare", "--ablation"])
    play_chime()
    _open_command_center(games[0], "ablation")


def run_ga_sweep():
    """GA Sweep — one GAConfig knob at a time against literature bounds; best config per game."""
    _refresh_screen()
    _section("BALANCE  ›  GA Sweep  (hyperparameter ablation)")
    print(_DIM("    Baseline GAConfig + one knob moved to its literature low / high bound per config\n"
               "    (hidden size, population, elite, tournament, crossover, mutation, anneal, init,\n"
               "    action feedback, memory units). Same coverage as Full Sweep; 23× the probes, ≈33× the episodes."))
    from code.neuro.gasweep import AXES, AXIS_DOC, cost_multiplier, sweep_configs  # lazy: numpy import
    sensors = pick_one("SENSORS  ·  for every config", SENSOR_CHOICES, default="rays", show_desc=SENSOR_DESC)
    if not sensors:
        return
    axes = toggle_select("GA AXES", list(AXES), default_indices=list(range(len(AXES))),
                         show_desc={a: f"{' · '.join(map(str, vs))}  —  {AXIS_DOC[a]}" for a, vs in AXES.items()})
    if not axes:
        return
    after = pick_one("AFTERWARDS", ["probe the composite", "stop"], default="probe the composite",
                     show_desc={"probe the composite": "every axis at its winner, probed together (--confirm)"})
    if not after:
        return
    confirm = after == "probe the composite"
    configs = sweep_configs(axes)
    picked = _sweep_prompts(n_modes=len(configs), modes_label="GA configs",
                            cost_note=f"≈{cost_multiplier(configs):.0f}× the episodes of one Full Sweep "
                                      f"(population axis dominates) — run overnight for all four games.")
    if not picked:
        return
    games, personas, gens, seeds, workers = picked
    extra = ("--axes", *axes) + (("--confirm",) if confirm else ())
    if not _sweep(games, personas, gens, seeds, workers, [sensors],
                  "\n  GA sweep interrupted — finished probes are kept; rerun to fill the gaps.",
                  module="code.neuro.gasweep", extra=extra):
        return
    play_chime()
    _open_command_center(games[0], "gasweep")


def run_full_sweep():
    """Full Sweep — balance probes across games × personas × seeds, all parallel."""
    _refresh_screen()
    _section("BALANCE  ›  Full Sweep")
    sensors = pick_one("SENSORS  ·  what the probe agents see", SENSOR_CHOICES, default="rays",
                       show_desc=SENSOR_DESC)
    if not sensors:
        return
    picked = _sweep_prompts()
    if not picked:
        return
    games, personas, gens, seeds, workers = picked
    if not _sweep(games, personas, gens, seeds, workers, [sensors],
                  "\n  Sweep interrupted — finished probes are already in the report."):
        return
    play_chime()
    _open_command_center(games[0], "balance")


# ============================================================================
# MUTATE — the AST evaluation (code/games/eval_mutants.py, code/neuro/ast_eval.py, code/stats/ast_metrics.py)
# ============================================================================

AST_GAMES = ["mario", "meatboy", "bomberman", "megaman"]


def _ast_games():
    games = [g for g in AST_GAMES if g in get_available_games()]
    return toggle_select("GAMES", games, default_indices=list(range(len(games))))


def _eval_set(game):
    from code.games.eval_mutants import eval_set  # lazy: the menu starts without numpy/pygame
    return eval_set(game)


def _variant_desc(game, retrained_only=False):
    """{variant: what it is} for one game: the original level, then its mutants."""
    out = {} if retrained_only else {"base": "the original level"}
    for m in _eval_set(game):
        if m["retrain"] or not retrained_only:
            out[m["id"]] = f"{m['class']:<10} {m['description']}"
    return out


def run_mutants():
    """Mutants — every game's evaluation set; write the ticked games' mutants again."""
    _refresh_screen()
    _section("MUTATE  ›  Mutants")
    from code.games import eval_mutants
    for g in AST_GAMES:
        print("\n" + eval_mutants.table(g))
    print(_DIM("\n    Writing checks every cell against the spec and loads the mutant in the game first."))
    games = toggle_select("WRITE", AST_GAMES, min_select=0,
                          default_indices=[i for i, g in enumerate(AST_GAMES) if not _eval_set(g)])
    for g in games or []:
        cmd = ([sys.executable, "-m", "code.games.megaman_mutants"] if g == "megaman" else
               [sys.executable, "-m", "code.games.eval_mutants", "--game", g])
        print(_DIM("  >>> " + " ".join(cmd)))
        subprocess.run(cmd)


def play_mutant():
    """Play a Mutant — you at the keyboard on the evaluation level or one of its mutants; every
    attempt is logged under your name (runs/ast/<game>/<variant>/human/)."""
    _refresh_screen()
    _section("MUTATE  ›  Play a Mutant  (human playtest)")
    games = [g for g in AST_GAMES if g in get_available_games()]
    game = pick_one("GAME", games, default=games[0])
    if not game:
        return
    desc = _variant_desc(game)
    variant = pick_one(f"VARIANT  ·  {game}", list(desc), default="base", show_desc=desc)
    if not variant:
        return
    raw = input(_DIM("\n    Player name (attempts are logged under it) [human]: ")).strip()
    player = "".join(c for c in raw if c.isalnum() or c in "-_") or "human"
    from code.games.eval_mutants import EVAL_LEVELS
    level = EVAL_LEVELS[game][0]
    _print_controls(game, level)
    env = os.environ.copy()
    env.pop("SDL_VIDEODRIVER", None)       # a real window
    env.pop("PEAK_MM1_MUTANT", None)
    env.pop("PEAK_MUTANT", None)
    if variant != "base":
        env["PEAK_MUTANT"] = str(RUNS_DIR / game / "mutants" / variant)
    record = RUNS_DIR / "ast" / game / variant / "human"
    cmd = [sys.executable, "-m", "code.games.tools.manual_play", "--game", CONFIG_KEY.get(game, game),
           "--fps", "60", "--level", level, "--record", str(record), "--player", player]
    print(_DIM(f"    Launching {game} · {variant}... (ESC to quit)\n"))
    run_with_loading(cmd, env=env, label="Loading the game")


def retrain_mutants():
    """Retrain on Mutants — each persona's three seeds trained again on the ticked mutants, with the
    baseline's GA setting and budget (the cosmetic mutant is replayed, never retrained)."""
    _refresh_screen()
    _section("MUTATE  ›  Retrain on Mutants")
    print(_DIM("    3 seeds × the generation budget per persona and mutant: hours of CPU per game.\n"
               "    On this PC keep the workers low; the full batch belongs on the cluster."))
    games = _ast_games()
    if not games:
        return
    jobs = []
    for g in games:
        desc = _variant_desc(g, retrained_only=True)
        if not desc:
            print(_RED(f"    {g}: no mutants yet — write them with [17]"))
            continue
        picked = toggle_select(f"MUTANTS  ·  {g}", list(desc), default_indices=list(range(len(desc))), show_desc=desc)
        if picked is None:
            return
        jobs += [(g, mid) for mid in picked]
    if not jobs:
        return
    personas = toggle_select("PERSONAS  ·  Mega Man retrains each section's own", PERSONA_CHOICES,
                             default_indices=list(range(len(PERSONA_CHOICES))), show_desc=PERSONA_DESC)
    if not personas:
        return
    gens = _prompt_gens("200 = the baseline's budget")
    if gens in ("invalid", None):
        if gens is None:
            print(_RED("  A generation count is required."))
        return
    raw = input(_DIM("    Parallel workers [3]: ")).strip()
    workers = raw if raw.isdigit() and int(raw) > 0 else "3"
    if not _print_run_summary(Mutants=", ".join(f"{g}/{m}" for g, m in jobs), Personas=", ".join(personas),
                              Generations=f"{gens} × 3 seeds", Workers=workers):
        return
    ok_runs = 0
    for i, (g, mid) in enumerate(jobs, 1):
        print(f"\n  {_YEL(f'[{i}/{len(jobs)}]')}  {_WHT(g)} | {_WHT(mid)}")
        cmd = [sys.executable, "-m", "code.neuro.ast_eval", "retrain", "--game", g, "--mutant", mid,
               "--gens", str(gens), "--workers", workers] + ([] if g == "megaman" else ["--persona", *personas])
        ok_runs += execute_training_run(cmd)
    print_training_summary(len(jobs), ok_runs, len(jobs) - ok_runs)
    if ok_runs == len(jobs):
        play_chime()


def playtest_agents():
    """Agent Playtest — one attempt per agent with learning off, on the original level and the ticked
    mutants; then the metric tables."""
    _refresh_screen()
    _section("MUTATE  ›  Agent Playtest")
    games = _ast_games()
    if not games:
        return
    plan = []
    for g in games:
        desc = _variant_desc(g)
        picked = toggle_select(f"VARIANTS  ·  {g}", list(desc), default_indices=list(range(len(desc))), show_desc=desc)
        if picked is None:
            return
        plan.append((g, picked))
    personas = PERSONA_CHOICES
    if any(g != "megaman" for g, _ in plan):
        personas = toggle_select("PERSONAS", PERSONA_CHOICES, default_indices=list(range(len(PERSONA_CHOICES))),
                                 show_desc=PERSONA_DESC)
        if not personas:
            return
    for g, picked in plan:
        if picked:
            execute_training_run([sys.executable, "-m", "code.neuro.ast_eval", "play", "--game", g,
                                  "--variant", *picked] + ([] if g == "megaman" else ["--persona", *personas]))
    subprocess.run([sys.executable, "-m", "code.stats.ast_metrics"] + [a for g, _ in plan for a in ("--game", g)])
    play_chime()


def mutation_metrics():
    """Mutation Metrics: the metric tables here, and the Command Center's Mutations tab in the browser."""
    _refresh_screen()
    _section("MUTATE  ›  Mutation Metrics")
    subprocess.run([sys.executable, "-m", "code.stats.ast_metrics", "--page"])   # also rebuilds runs/site
    site = RUNS_DIR / "site" / "index.html"
    if site.exists():
        webbrowser.open(site.resolve().as_uri() + f"#{AST_GAMES[0]}/mutations")


_RUN_GROUPS = [  # label, dir — whole probe/sweep trees are deleted as one unit
    ("Full Sweep probes", RUNS_DIR / "probes"),
    ("Sensor Ablation probes", RUNS_DIR / "ablation"),
    ("GA Sweep cells", RUNS_DIR / "gasweep"),
    ("Balance reports + command center", RUNS_DIR / "balance"),
]


def delete_logs_and_models():
    """Delete selected training runs / probe trees — or everything under runs/."""
    _refresh_screen()
    _section("DANGER  ›  Delete Runs")
    groups = [(lbl, d) for lbl, d in _RUN_GROUPS if d.exists()]
    singles = [p for p in get_run_dirs() if not any(d in p.parents for _, d in groups)]
    targets = [(str(p.relative_to(RUNS_DIR)), p) for p in singles] + \
              [(f"{lbl}  ({sum(1 for _ in d.rglob('state.json'))} populations)", d) for lbl, d in groups] + \
              [("EVERYTHING under runs/", RUNS_DIR)]
    chosen = toggle_select("DELETE", [t for t, _ in targets], default_indices=[])
    if not chosen:
        print("Aborted. Nothing was deleted.")
        return
    paths = [p for t, p in targets if t in chosen]
    if RUNS_DIR in paths:              # all of it but the user study: people's play cannot be run again
        paths = sorted(p for p in RUNS_DIR.iterdir() if p != STUDY_DIR)
    print()
    for p in paths:
        print(f"    {_RED('✖')}  {p}")
    if input(_BOLD(f"\n    ⟫ Permanently delete {len(paths)} item(s)? [y/N]: ")).strip().lower() not in ("y", "yes"):
        print("Aborted. Nothing was deleted.")
        return

    for path in paths:
        for attempt in range(3):
            try:
                shutil.rmtree(path) if path.is_dir() else path.unlink()
                print(f"    {_GRN('✓')}  deleted {path}")
                break
            except Exception as e:
                if attempt == 2:
                    print(_RED(f"    Failed to delete {path}: {e}"))
                else:
                    print(_DIM(f"    {path} might be in use. Retrying in 1s…"))
                    time.sleep(1)
    RUNS_DIR.mkdir(exist_ok=True)


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
    print(f"Network:   {_WHT('14 sensors (or 368 grid cells) → H tanh (default 16) → 3 (left/right/jump)')}")
    print()


# ============================================================================
# MAIN LOOP
# ============================================================================

# DISPATCH table maps menu key → (debug_label, callable).
# None as the callable means "handled inline" (currently only exit/0).
# Adding a new menu item: add the entry here AND a matching _menu_item() print below.
DISPATCH = {
    "1":  ("show_project_status",  show_project_status),
    "2":  ("run_training",         run_training),
    "3":  ("train_all_game",       train_all_models_for_game),
    "4":  ("train_grid",           train_complete_grid),
    "p":  ("train_all_personas",   train_all_personas),
    "5":  ("manual_play",          run_manual_play),
    "u":  ("user_study",           run_user_study),
    "6":  ("watch_agent",          watch_trained_agent),
    "7":  ("watch_all",            watch_all_models),
    "8":  ("watch_random",         watch_random_agent),
    "9":  ("level_editor",         run_level_editor),
    "10": ("toggle_levels",        run_toggle_levels),
    "11": ("dashboard",            run_dashboard),
    "12": ("command_center",       open_command_center),
    "13": ("full_sweep",           run_full_sweep),
    "14": ("sensor_ablation",      run_sensor_ablation),
    "15": ("ga_sweep",             run_ga_sweep),
    "16": ("readme_figures",       run_readme_figures),
    "17": ("mutants",              run_mutants),
    "18": ("play_mutant",          play_mutant),
    "19": ("retrain_mutants",      retrain_mutants),
    "20": ("playtest_agents",      playtest_agents),
    "21": ("mutation_metrics",     mutation_metrics),
    "22": ("record_wall",          record_wall_videos),
    "23": ("publish_site",         publish_site),
    "d":  ("delete_all",           delete_logs_and_models),
    "c":  ("clear_cli",            clear_cli),
    "0":  ("exit",                 None),
}


def main():
    import warnings
    warnings.filterwarnings(
        "ignore",
        message=r".*pkg_resources is deprecated as an API.*"
    )

    while True:
        tty_sane()
        setup_project()
        header = []
        for step, fn in enumerate((get_available_games, get_enabled_level_count, get_trained_games,
                                   get_trained_models_count)):
            bar = "█" * (6 * step) + "·" * (6 * (4 - step))   # the run scan below reads every run folder
            print(f"\r    {_DIM('Loading PEAK ENGINE')}  {_CYAN(bar)}", end="", flush=True)
            header.append(fn())
        clear_cli()

        _print_header(*header)

        _section("TRAIN")
        print(_menu_item("1",  "Project Status"))
        print(_menu_item("2",  "Train Single",        "pick game / level / generations"))
        print(_menu_item("3",  "Train All (1 game)",  "toggle levels, back-to-back runs"))
        print(_menu_item("4",  "Train Full Grid",     "all games × enabled levels"))
        print(_menu_item(" P", "Train All Personas",  "one agent per persona carries on level after level"))

        _section("PLAY")
        print(_menu_item("5",  "Play Manually",       "keyboard controls"))
        print(_menu_item("6",  "Watch Agent",         "visualize a trained model"))
        print(_menu_item("7",  "Watch All Models",    "resume a run with the dashboard grid"))
        print(_menu_item("8",  "Watch Random Agent",  "random actions"))
        print(_menu_item(" U", "User Study",          "participant ID, warm-up, eval level ×3, its 5 mutants"))

        _section("TOOLS")
        print(_menu_item("9",  "Level Editor",         "paint tiles, place entities"))
        print(_menu_item("10", "Toggle Levels",        "enable / disable levels in config"))
        print(_menu_item("11", "Dashboard",            "live training UI in browser"))
        print(_menu_item("12", "Command Center",       "every result, one game at a time"))
        print(_menu_item("22", "Record Replay Wall",   "one looping video per game and persona"))
        print(_menu_item("23", "Publish Command Center", "push it to GitHub Pages (asks first)"))

        _section("BALANCE")
        print(_menu_item("13", "Full Sweep",           "games × personas × seeds, parallel"))
        print(_menu_item("14", "Sensor Ablation",      "same sweep with rays and tile grid, compared"))
        print(_menu_item("15", "GA Sweep",             "one GA knob at a time vs literature bounds; best config per game"))
        print(_menu_item("16", "README Figures",       "regenerate docs/img/fig_*.png from the JSONs, update README"))

        _section("MUTATE  ·  AST evaluation")
        print(_menu_item("17", "Mutants",              "5 per game, checked against the spec cells"))
        print(_menu_item("18", "Play a Mutant",        "you at the keyboard, logged per player"))
        print(_menu_item("19", "Retrain on Mutants",   "every persona's 3 seeds trained again on each mutant"))
        print(_menu_item("20", "Agent Playtest",       "one attempt per agent, learning off: original vs mutants"))
        print(_menu_item("21", "Mutation Metrics",     "completion, win rate, death cluster ... + the dashboard"))

        _section("SYSTEM")
        print(_menu_item(" D", "Delete Runs",          "pick runs / probe trees, or everything"))
        print(_menu_item(" C", "Clear Screen",         "clear terminal output"))

        print()
        print(f"    {_DIM('───────────────────────────────────')}")
        print(_menu_item("0",  _RED("Exit")))
        print()

        try:
            choice = input(_BOLD("    ⟫ ")).strip().lower()
        except (EOFError, KeyboardInterrupt):
            choice = "0"

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
                tty_sane()                    # the action may have run a game that left the terminal raw
                try:
                    input(_DIM("\n    Press Enter to return to menu..."))
                except (EOFError, KeyboardInterrupt):
                    pass

if __name__ == "__main__":
    main()
