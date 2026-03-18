# menu_gui.py — PyQt5 GUI for RL training/eval/manual play/TensorBoard
# Run: python menu_gui.py

import os
import sys
import platform
import subprocess
import time
import math
from pathlib import Path
from threading import Thread

try:
    import winsound
    HAS_WINSOUND = True
except Exception:
    HAS_WINSOUND = False

from omegaconf import OmegaConf

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QRectF, QPointF

from PyQt5.QtGui import (
    QFont, QPalette, QColor, QPixmap, QIcon, QPainter, 
    QBrush, QPen, QPolygonF, QLinearGradient
)
from PyQt5.QtWidgets import (
    QApplication, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QLineEdit, QTextEdit, QSpinBox, QListWidget,
    QListWidgetItem, QMessageBox, QCheckBox, QGroupBox, QFormLayout, QProgressBar,
    QTableWidget, QTableWidgetItem, QHeaderView, QSplitter, QAbstractItemView, QFrame,
    QGridLayout, QStyle, QScrollArea
)

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

DEFAULT_TB_ROOT = "mylogs"
MODELS_DIR = Path("models/")
CONF_ROOT = Path("code/conf")
GRID_CONFIG_PATH = CONF_ROOT / "grid.yaml"
GAME_CONFIG_PATH = Path("code/games/game_config.yaml")
REWARD_MODULE_DIR = Path("code/rewards")
CONF_ALGO_DIR = CONF_ROOT / "algo"
LOGO_PATH = Path("Screenshots/PEAK_LOGO.png")

REQUIRED_PACKAGES = [
    'torch>=1.9.0',
    'stable-baselines3[extra]>=1.6.0',
    'sb3-contrib>=1.6.0',
    'gymnasium>=0.26.0',
    'pygame>=2.1.0',
    'numpy>=1.21.0',
    'tensorboard>=2.8.0',
    'hydra-core>=1.1.0',
    'pyyaml>=6.0',
    'omegaconf>=2.1.0',
    'PyQt5>=5.15.0',
]

# ============================================================================
# STYLING (Modern Ember + Ice Theme)
# ============================================================================

CONTROL_CENTER_STYLESHEET = """
QWidget {
    background-color: #0d1117;
    color: #ecf2f8;
    font-family: 'Segoe UI', 'Inter', sans-serif;
    font-size: 14px;
}

QWidget#rootWindow {
    background-color: #091018;
}

QLabel {
    background: transparent;
}

QFrame#heroCard, QFrame#actionDeck, QFrame#statCard {
    background-color: #111924;
    border: 1px solid #243246;
    border-radius: 16px;
}

QFrame#heroCard {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #111827, stop:0.55 #191724, stop:1 #2a1320);
    border: 1px solid #34455f;
}

QFrame#actionDeck {
    background-color: #0f1722;
}

QFrame#statCard {
    min-height: 96px;
}

QLabel#heroEyebrow {
    color: #8fb9ff;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}

QLabel#heroTitle {
    color: #f7fbff;
    font-size: 28px;
    font-weight: 900;
    letter-spacing: 1px;
}

QLabel#heroSubtitle {
    color: #9eb0c4;
    font-size: 13px;
}

QLabel#statusPill {
    background-color: #132233;
    border: 1px solid #2d4f75;
    border-radius: 14px;
    color: #8bd3ff;
    padding: 6px 12px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
}

QLabel#statusPill[mode="busy"] {
    background-color: #2c1f14;
    border: 1px solid #8a5b23;
    color: #ffbe6b;
}

QLabel#statusPill[mode="error"] {
    background-color: #32161b;
    border: 1px solid #8f3643;
    color: #ff9aa8;
}

QLabel#statusPill[mode="ready"] {
    background-color: #132233;
    border: 1px solid #2d4f75;
    color: #8bd3ff;
}

QLabel#chipLabel {
    color: #8ea2b7;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
}

QLabel#chipValue {
    color: #f6fbff;
    font-size: 22px;
    font-weight: 900;
}

QLabel#chipMeta {
    color: #ff8d76;
    font-size: 11px;
    font-weight: 600;
}

QGroupBox {
    background-color: #111924;
    border: 1px solid #243246;
    border-radius: 14px;
    margin-top: 22px;
    padding: 20px 16px 16px 16px;
    font-weight: 600;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 14px;
    padding: 0 8px;
    color: #ff8d76;
    font-weight: 800;
    font-size: 13px;
    letter-spacing: 1px;
}

QTabWidget::pane {
    border: 1px solid #223043;
    background-color: #0f1722;
    border-radius: 16px;
    top: -1px;
}

QTabBar::tab {
    background: #111924;
    color: #97a7b9;
    padding: 12px 20px;
    min-width: 110px;
    border: 1px solid #243246;
    border-bottom: none;
    border-top-left-radius: 12px;
    border-top-right-radius: 12px;
    margin-right: 6px;
    font-weight: 700;
}

QTabBar::tab:selected {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #1e2e45, stop:1 #4a1f2d);
    color: #ffffff;
    border: 1px solid #48658c;
    border-bottom: 2px solid #7dd3fc;
}

QTabBar::tab:hover:!selected {
    background: #172231;
    color: #d6e4f2;
}

QPushButton {
    background-color: #1b2635;
    color: #f4f8fc;
    border: 1px solid #34475f;
    padding: 10px 16px;
    border-radius: 10px;
    font-weight: 800;
    min-height: 42px;
}

QPushButton:hover {
    background-color: #243246;
    border: 1px solid #5f87b3;
}

QPushButton:pressed {
    background-color: #121c29;
}

QPushButton:disabled {
    background-color: #11161d;
    color: #556170;
    border: 1px solid #202833;
}

QPushButton#accentButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #b9493e, stop:1 #e06b52);
    border: 1px solid #f28c6a;
    color: white;
}

QPushButton#accentButton:hover {
    background: #e5735a;
    border: 1px solid #ffaf90;
}

QPushButton#dangerButton {
    background-color: #38171d;
    border: 1px solid #8f3643;
    color: #ffb8c2;
}

QPushButton#dangerButton:hover {
    background-color: #492028;
    border: 1px solid #c54f63;
}

QLineEdit, QComboBox, QSpinBox {
    background-color: #0d141d;
    border: 1px solid #2b3b4f;
    border-radius: 10px;
    padding: 8px 10px;
    color: #f5f9fd;
    selection-background-color: #7dd3fc;
    min-height: 38px;
}

QLineEdit:focus, QComboBox:focus, QSpinBox:focus {
    border: 1px solid #7dd3fc;
    background-color: #101a25;
}

QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 28px;
    border-left: 1px solid #2b3b4f;
    border-top-right-radius: 10px;
    border-bottom-right-radius: 10px;
    background-color: #132233;
}

QComboBox QAbstractItemView {
    background-color: #101824;
    color: #f5f9fd;
    border: 1px solid #34475f;
    selection-background-color: #23364c;
    selection-color: white;
    padding: 4px;
    outline: 0;
}

QCheckBox {
    spacing: 8px;
    color: #dbe5ee;
}

QCheckBox::indicator {
    width: 18px;
    height: 18px;
    background: #0d141d;
    border: 2px solid #486178;
    border-radius: 5px;
}

QCheckBox::indicator:hover {
    border: 2px solid #7dd3fc;
}

QCheckBox::indicator:checked {
    background: #7dd3fc;
    border: 2px solid #7dd3fc;
}

QTextEdit {
    background-color: #081019;
    border: 1px solid #223043;
    color: #cfe0ef;
    font-family: 'Consolas', 'JetBrains Mono', 'Courier New', monospace;
    font-size: 12px;
    border-radius: 14px;
    padding: 6px;
}

QTableWidget {
    background-color: #0d141d;
    gridline-color: #233141;
    border: 1px solid #243246;
    border-radius: 10px;
}

QTableWidget::item {
    padding: 6px;
}

QTableWidget::item:selected {
    background-color: #23364c;
    color: white;
}

QHeaderView::section {
    background-color: #111924;
    color: #8bd3ff;
    padding: 8px;
    border: none;
    border-bottom: 1px solid #223043;
    font-weight: 800;
}

QSplitter::handle {
    background-color: #162131;
    width: 8px;
}

QScrollBar:vertical {
    border: none;
    background: #091018;
    width: 12px;
    margin: 2px 0 2px 0;
}

QScrollBar::handle:vertical {
    background: #32485e;
    min-height: 24px;
    border-radius: 6px;
    margin: 2px;
}

QScrollBar::handle:vertical:hover {
    background: #7dd3fc;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollArea {
    border: none;
    background: transparent;
}
"""

# ============================================================================
# HELPER FUNCTIONS & LOGIC
# ============================================================================

def open_browser(url: str):
    try:
        import webbrowser
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

def write_requirements(requirements_log: QTextEdit | None = None, overwrite: bool = True):
    requirements_path = Path("requirements.txt")
    requirements_content = "\n".join(REQUIRED_PACKAGES) + "\n"
    mode_desc = "overwritten" if overwrite or not requirements_path.exists() else "written"
    if overwrite or not requirements_path.exists():
        requirements_path.write_text(requirements_content)
        if requirements_log is not None:
            requirements_log.append(f"✓ requirements.txt {mode_desc} at {requirements_path.resolve()}")
    else:
        if requirements_log is not None:
            requirements_log.append("requirements.txt already exists — not modified")

def load_grid_config():
    if not GRID_CONFIG_PATH.exists():
        return None
    try:
        return OmegaConf.load(GRID_CONFIG_PATH)
    except Exception:
        return None

def load_game_config():
    if not GAME_CONFIG_PATH.exists():
        return None
    try:
        return OmegaConf.load(GAME_CONFIG_PATH)
    except Exception:
        return None

def _game_defined(game_cfg, game: str) -> bool:
    if game_cfg is None:
        return False
    try:
        if game == "platformer":
            return bool(game_cfg.get("levels"))
        return bool(game_cfg.get(game))
    except Exception:
        return False

def get_available_algos_from_grid():
    cfg = load_grid_config()
    if cfg is None or 'models' not in cfg:
        return []
    grid_algos = list(cfg.models) if cfg.models else []
    return sorted([a for a in grid_algos if (CONF_ALGO_DIR / f"{a}.yaml").exists()])

def get_available_personas_from_grid():
    cfg = load_grid_config()
    if cfg is None or 'personas' not in cfg:
        return []
    grid_personas = list(cfg.personas) if cfg.personas else []
    personas = []
    for persona in grid_personas:
        persona_name = str(persona)
        game_prefix = persona_name.split("_", 1)[0]
        if (REWARD_MODULE_DIR / f"train_{game_prefix}.py").exists():
            personas.append(persona_name)
    return personas

def get_personas_for_game(game: str):
    all_personas = get_available_personas_from_grid()
    filtered = [p for p in all_personas if p.startswith(f"{game}_")]
    return filtered if filtered else all_personas

def get_available_games():
    grid_cfg = load_grid_config()
    game_cfg = load_game_config()

    candidates = list(grid_cfg.games) if grid_cfg is not None and grid_cfg.get("games") else [
        "platformer", "megaman", "sonic"
    ]
    available = [str(game) for game in candidates if _game_defined(game_cfg, str(game))]
    if available:
        return list(dict.fromkeys(available))

    fallback = []
    for game in ("platformer", "megaman", "sonic"):
        if _game_defined(game_cfg, game):
            fallback.append(game)
    return fallback

def get_trained_models_count():
    best_dir = MODELS_DIR / "best"
    if not best_dir.exists():
        return 0
    return sum(1 for f in best_dir.iterdir() if f.is_dir() and (f / "best_model.zip").exists())

def get_trained_games_from_models_flat():
    best_dir = MODELS_DIR / "best"
    if not best_dir.exists():
        return []
    model_folders = [f for f in best_dir.iterdir() if f.is_dir() and (f / "best_model.zip").exists()]
    games = set()
    for folder in model_folders:
        parts = folder.name.split("_")
        if len(parts) >= 5:
            games.add(parts[0])
    return sorted(games)

# ============================================================================
# CUSTOM WIDGET: Animated Mountain Climb
# ============================================================================
class AnimatedMountainWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(140)
        
        self._progress = 0.0
        self._step_counter = 0
        
        # Victory Animation State
        self._victory_active = False
        self._victory_frame = 0
        
        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._advance_frame)
        self._anim_timer.start(100) # 10 FPS

    def setProgress(self, value_0_to_100):
        prev = self._progress
        self._progress = max(0.0, min(100.0, float(value_0_to_100))) / 100.0
        
        # Trigger victory if we just hit 100%
        if self._progress >= 0.99 and prev < 0.99:
            self._victory_active = True
            self._victory_frame = 0
        # Reset victory if we reset progress
        elif self._progress < 0.1:
            self._victory_active = False
            self._victory_frame = 0
            
        self.update()

    def _advance_frame(self):
        # Animate running legs if climbing
        if 0 < self._progress < 1.0:
            self._step_counter += 1
            self.update()
        # Animate victory jump if finished (runs for ~3 seconds)
        elif self._victory_active:
            self._victory_frame += 1
            if self._victory_frame > 30: # Stop jumping after 3s
                self._victory_active = False
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w = self.width()
        h = self.height()

        # 1. Background
        gradient = QLinearGradient(0, 0, 0, h)
        gradient.setColorAt(0.0, QColor("#0a1320"))
        gradient.setColorAt(1.0, QColor("#131b26"))
        painter.fillRect(0, 0, w, h, QBrush(gradient))

        # 2. Mountain
        peak_x = w * 0.85
        peak_y = h * 0.15
        base_y = h * 0.9
        
        path = QPolygonF()
        path.append(QPointF(0, base_y))
        path.append(QPointF(peak_x, peak_y))
        path.append(QPointF(w, base_y))
        
        painter.setBrush(QBrush(QColor("#1d2a3a")))
        painter.setPen(QPen(QColor("#34506f"), 3))
        painter.drawPolygon(path)

        # 3. Snow
        painter.setBrush(QBrush(QColor("#eef7ff")))
        painter.setPen(Qt.NoPen)
        snow_path = QPolygonF()
        snow_path.append(QPointF(peak_x, peak_y))
        snow_path.append(QPointF(peak_x - (peak_x * 0.15), peak_y + (base_y - peak_y) * 0.15))
        snow_path.append(QPointF(peak_x + ((w - peak_x) * 0.15), peak_y + (base_y - peak_y) * 0.15))
        painter.drawPolygon(snow_path)

        # 4. Man Position
        start_x = 20.0
        start_y = base_y
        current_x = start_x + (peak_x - start_x) * self._progress
        current_y = start_y + (peak_y - start_y) * self._progress
        
        # 5. Victory Jump Calculation
        jump_offset = 0
        if self._victory_active:
            # Physics: Bounce 3 times
            cycle = self._victory_frame % 8
            # Simple parabola: y = 4 - (x-2)^2
            hop = max(0, 4 - (cycle - 2)**2) * 4
            jump_offset = hop

        man_h = 24
        man_w = 12
        man_x = current_x - man_w/2
        man_y = current_y - man_h - jump_offset # Apply jump

        # Animation frame
        leg_offset = 0
        if 0 < self._progress < 1.0:
            leg_offset = 2 if (self._step_counter % 2) == 0 else -2
        elif self._victory_active:
            # Spread legs during jump
            leg_offset = 3 if jump_offset > 2 else 0

        # --- DRAW MAN ---
        painter.setPen(Qt.NoPen)
        
        # Body (Blue)
        painter.setBrush(QColor("#e06b52"))
        painter.drawRect(int(man_x), int(man_y + 8), int(man_w), 10)
        
        # Head (Flesh)
        painter.setBrush(QColor("#ffd0b2"))
        painter.drawRect(int(man_x + 2), int(man_y), 8, 8)
        
        # Hair (Retro Brown)
        painter.setBrush(QColor("#5b3d35")) 
        painter.drawRect(int(man_x + 1), int(man_y - 2), 10, 4) # Top hair
        painter.drawRect(int(man_x + 1), int(man_y), 2, 6)      # Sideburn L
        painter.drawRect(int(man_x + 9), int(man_y), 2, 6)      # Sideburn R

        # Legs (Dark Grey)
        painter.setBrush(QColor("#8ea2b7"))
        painter.drawRect(int(man_x + 2 + leg_offset), int(man_y + 18), 3, 6)
        painter.drawRect(int(man_x + 7 - leg_offset), int(man_y + 18), 3, 6)

        # 6. Flag
        if self._progress >= 0.99:
            painter.setBrush(QColor("#8ea2b7"))
            painter.drawRect(int(peak_x + 2), int(peak_y - 30), 2, 30)
            painter.setBrush(QColor("#7dd3fc"))
            flag_poly = QPolygonF()
            flag_poly.append(QPointF(peak_x + 4, peak_y - 30))
            flag_poly.append(QPointF(peak_x + 24, peak_y - 22))
            flag_poly.append(QPointF(peak_x + 4, peak_y - 14))
            painter.drawPolygon(flag_poly)

        # 7. Text
        pct_text = f"{int(self._progress * 100)}%"
        painter.setPen(QColor("white"))
        painter.setFont(QFont("Consolas", 10, QFont.Bold))
        
        # Position text above head (accounting for jump height)
        text_w, text_h = 40, 20
        text_x = man_x + (man_w / 2) - (text_w / 2)
        text_y = man_y - text_h - 5
        
        painter.drawText(QRectF(text_x, text_y, text_w, text_h), Qt.AlignCenter, pct_text)

# ============================================================================
# WORKER THREAD
# ============================================================================

class ProcWorker(QThread):
    line = pyqtSignal(str)
    finished = pyqtSignal(int)

    def __init__(self, cmd, env=None, cwd=None):
        super().__init__()
        self.cmd = cmd
        self.env = env or os.environ.copy()
        self.cwd = cwd or None
        self._proc = None

    def run(self):
        try:
            # Set creation flags to suppress console window on Windows
            creation_flags = 0
            if platform.system() == "Windows":
                creation_flags = subprocess.CREATE_NO_WINDOW

            self._proc = subprocess.Popen(
                self.cmd,
                cwd=self.cwd,
                env=self.env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                creationflags=creation_flags
            )
            for line in self._proc.stdout:
                self.line.emit(line.rstrip())
            self._proc.wait()
            self.finished.emit(self._proc.returncode)
        except FileNotFoundError:
            self.line.emit("[!] Command not found. Check your Python/paths.")
            self.finished.emit(127)
        except Exception as e:
            self.line.emit(f"[!] Error: {e}")
            self.finished.emit(1)

    def stop(self):
        try:
            if self._proc and self._proc.poll() is None:
                # Use os.kill to terminate all child processes/threads on Windows
                if platform.system() == "Windows":
                    subprocess.call(['taskkill', '/F', '/T', '/PID', str(self._proc.pid)])
                else:
                    self._proc.terminate()
        except Exception:
            pass
        
        
# ============================================================================
# MAIN GUI CLASS
# ============================================================================
class RLManagerGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("rootWindow")
        self.setWindowTitle("PEAK Agents — Control Center")
        self.resize(1600, 900)  # Made window wider for side-by-side layout
        
        # Apply the updated control-center theme
        self.setStyleSheet(CONTROL_CENTER_STYLESHEET)

        # Set Window Icon if exists
        if LOGO_PATH.exists():
            self.setWindowIcon(QIcon(str(LOGO_PATH)))

        self.proc = None
        self._progress_mode = None
        self._progress_total = None
        self._progress_value = 0
        self._train_start_time = None
        
        # --- Timer for Decoupled Log/Progress Updates ---
        self._log_buffer = []
        self._latest_progress_value = 0
        self._timer = QTimer(self)
        self._timer.setInterval(50) # Update GUI max 20 times per second
        self._timer.timeout.connect(self._update_log_and_progress_from_buffer)
        
        # --- Main Tabs ---
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setMovable(False)
        self.tabs.setUsesScrollButtons(True)
        
        # Initialize tabs
        self.tab_status = self._build_status_tab()
        self.tab_train = self._build_train_tab()
        self.tab_eval = self._build_eval_tab()
        self.tab_tensorboard = self._build_tb_tab()
        self.tab_manual = self._build_manual_tab()
        self.tab_watch = self._build_watch_tab()
        self.tab_bulk = self._build_bulk_tab()
        self.tab_maint = self._build_maint_tab()

        self.tabs.addTab(self._wrap_tab(self.tab_status), "STATUS")
        self.tabs.addTab(self._wrap_tab(self.tab_train), "TRAIN")
        self.tabs.addTab(self._wrap_tab(self.tab_eval), "EVALUATE")
        self.tabs.addTab(self._wrap_tab(self.tab_tensorboard), "TENSORBOARD")
        self.tabs.addTab(self._wrap_tab(self.tab_manual), "MANUAL PLAY")
        self.tabs.addTab(self._wrap_tab(self.tab_watch), "WATCH AGENT")
        self.tabs.addTab(self._wrap_tab(self.tab_bulk), "TRAIN ALL")
        self.tabs.addTab(self._wrap_tab(self.tab_maint), "MAINTENANCE")
        self._apply_tab_icons()

        # --- Top Shell ---
        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(16)

        hero_card = QFrame()
        hero_card.setObjectName("heroCard")
        hero_layout = QHBoxLayout(hero_card)
        hero_layout.setContentsMargins(22, 18, 22, 18)
        hero_layout.setSpacing(18)

        if LOGO_PATH.exists():
            logo_lbl = QLabel()
            pixmap = QPixmap(str(LOGO_PATH))
            scaled = pixmap.scaledToHeight(58, Qt.SmoothTransformation)
            logo_lbl.setPixmap(scaled)
            hero_layout.addWidget(logo_lbl, 0, Qt.AlignTop)

        hero_text = QVBoxLayout()
        hero_text.setSpacing(4)
        eyebrow = QLabel("PEAK ENGINE")
        eyebrow.setObjectName("heroEyebrow")
        title_lbl = QLabel("Agent Control Center")
        title_lbl.setObjectName("heroTitle")
        subtitle_lbl = QLabel("Train, watch, debug, and maintain every game pipeline from one desktop cockpit.")
        subtitle_lbl.setObjectName("heroSubtitle")
        subtitle_lbl.setWordWrap(True)
        hero_text.addWidget(eyebrow)
        hero_text.addWidget(title_lbl)
        hero_text.addWidget(subtitle_lbl)
        hero_layout.addLayout(hero_text, 1)

        self.job_status_pill = QLabel("READY")
        self.job_status_pill.setObjectName("statusPill")
        self.job_status_pill.setProperty("mode", "ready")
        self.job_status_pill.setAlignment(Qt.AlignCenter)
        self.job_status_pill.setMinimumWidth(120)
        hero_layout.addWidget(self.job_status_pill, 0, Qt.AlignTop | Qt.AlignRight)

        action_deck = QFrame()
        action_deck.setObjectName("actionDeck")
        action_deck.setMinimumWidth(300)
        action_layout = QVBoxLayout(action_deck)
        action_layout.setContentsMargins(16, 16, 16, 16)
        action_layout.setSpacing(10)

        deck_title = QLabel("Quick Actions")
        deck_title.setObjectName("chipLabel")
        action_layout.addWidget(deck_title)

        btn_row_1 = QHBoxLayout()
        btn_row_1.setSpacing(8)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        refresh_btn.clicked.connect(self._refresh_all)
        refresh_btn.setObjectName("accentButton")
        req_btn = QPushButton("Requirements")
        req_btn.setIcon(self.style().standardIcon(QStyle.SP_FileDialogDetailedView))
        req_btn.clicked.connect(lambda: write_requirements(self.log, overwrite=True))
        btn_row_1.addWidget(refresh_btn)
        btn_row_1.addWidget(req_btn)
        action_layout.addLayout(btn_row_1)

        btn_row_2 = QHBoxLayout()
        btn_row_2.setSpacing(8)
        req_install_btn = QPushButton("Install Deps")
        req_install_btn.setIcon(self.style().standardIcon(QStyle.SP_ArrowDown))
        req_install_btn.clicked.connect(lambda: self._install_requirements_safe())
        self.clear_log_btn = QPushButton("Clear Log")
        self.clear_log_btn.setIcon(self.style().standardIcon(QStyle.SP_DialogResetButton))
        self.clear_log_btn.clicked.connect(self.log_clear)
        btn_row_2.addWidget(req_install_btn)
        btn_row_2.addWidget(self.clear_log_btn)
        action_layout.addLayout(btn_row_2)

        self.stop_btn = QPushButton("Stop Job")
        self.stop_btn.setIcon(self.style().standardIcon(QStyle.SP_BrowserStop))
        self.stop_btn.clicked.connect(self._stop_proc)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setObjectName("dangerButton")
        action_layout.addWidget(self.stop_btn)

        top_row.addWidget(hero_card, 1)
        top_row.addWidget(action_deck, 0)

        stats_row = QHBoxLayout()
        stats_row.setSpacing(12)
        self.stat_games = self._create_stat_card("Games", "0", "configured")
        self.stat_algos = self._create_stat_card("Algorithms", "0", "available")
        self.stat_models = self._create_stat_card("Models", "0", "best checkpoints")
        self.stat_runtime = self._create_stat_card("Runtime", "--", "default")
        stats_row.addWidget(self.stat_games, 1)
        stats_row.addWidget(self.stat_algos, 1)
        stats_row.addWidget(self.stat_models, 1)
        stats_row.addWidget(self.stat_runtime, 1)

        # --- Log Window ---
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("System logs will appear here...")

        # --- Layout Assembly (Side-by-Side) ---
        splitter = QSplitter(Qt.Horizontal) # CHANGED: Horizontal split
        splitter.setChildrenCollapsible(False)
        
        # Left section (Tabs)
        tabs_wrap = QWidget()
        tl = QVBoxLayout(tabs_wrap)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.addWidget(self.tabs)
        
        # Right section (Logs)
        log_wrap = QWidget()
        ll = QVBoxLayout(log_wrap)
        ll.setContentsMargins(10, 0, 0, 0) # Left margin for spacing
        
        log_header = QLabel("SYSTEM OUTPUT LOG:")
        log_header.setObjectName("chipLabel")
        ll.addWidget(log_header)
        ll.addWidget(self.log)

        splitter.addWidget(tabs_wrap)
        splitter.addWidget(log_wrap)
        tabs_wrap.setMinimumWidth(920)
        log_wrap.setMinimumWidth(320)

        # Set initial sizes (70% Tabs, 30% Log)
        splitter.setStretchFactor(0, 7)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([1260, 360])

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(16)
        root.addLayout(top_row)
        root.addLayout(stats_row)
        root.addWidget(splitter)

        self._refresh_all()
        self.tabs.setCurrentIndex(0)

    # ---------------- helpers ----------------
    def log_clear(self):
        self.log.clear()

    def _install_requirements_safe(self):
        cmd = [sys.executable, "-m", "pip", "install"] + REQUIRED_PACKAGES
        self._run_cmd(cmd)

    def _lbl(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setProperty('role', 'form-left')
        lbl.setMinimumWidth(110)
        return lbl

    def _wrap_tab(self, widget: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(widget)
        return scroll

    def _create_stat_card(self, title: str, value: str, meta: str) -> QFrame:
        card = QFrame()
        card.setObjectName("statCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(2)

        title_lbl = QLabel(title.upper())
        title_lbl.setObjectName("chipLabel")
        value_lbl = QLabel(value)
        value_lbl.setObjectName("chipValue")
        meta_lbl = QLabel(meta)
        meta_lbl.setObjectName("chipMeta")

        layout.addWidget(title_lbl)
        layout.addWidget(value_lbl)
        layout.addWidget(meta_lbl)
        layout.addStretch(1)

        card.value_label = value_lbl
        card.meta_label = meta_lbl
        return card

    def _set_job_status(self, text: str, mode: str = "ready"):
        self.job_status_pill.setText(text.upper())
        self.job_status_pill.setProperty("mode", mode)
        self.job_status_pill.style().unpolish(self.job_status_pill)
        self.job_status_pill.style().polish(self.job_status_pill)
        self.job_status_pill.update()

    def _apply_tab_icons(self):
        icons = [
            QStyle.SP_ComputerIcon,
            QStyle.SP_MediaPlay,
            QStyle.SP_DialogApplyButton,
            QStyle.SP_BrowserReload,
            QStyle.SP_ArrowForward,
            QStyle.SP_FileDialogContentsView,
            QStyle.SP_DialogYesButton,
            QStyle.SP_MessageBoxWarning,
        ]
        for idx, icon_type in enumerate(icons):
            self.tabs.setTabIcon(idx, self.style().standardIcon(icon_type))

    def _append_cmd(self, cmd_list):
        pretty = " ".join(str(x) for x in cmd_list)
        self.log.append(f"<span style='color:#ff7f7f;'>>>></span> {pretty}")

    @staticmethod
    def _fmt_hms(sec: float) -> str:
        sec = max(0, int(sec))
        m, s = divmod(sec, 60)
        h, m = divmod(m, 60)
        return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

    def _selected_total_steps(self) -> int | None:
        s = self.train_step_choice.currentText().strip()
        if not s: return None
        if s.lower() == "custom":
            return int(self.train_steps.value())
        try:
            return int(s.replace(",", ""))
        except Exception:
            return None

    def _on_proc_line_received(self, s: str):
        self._log_buffer.append(s)
        if self._progress_mode != 'train': return

        line = s.strip()
        cur = None

        if 'PROGRESS:' in line and '/' in line:
            try:
                after = line.split('PROGRESS:')[1].strip()
                left, right = after.split('/', 1)
                cur = int(''.join(ch for ch in left if ch.isdigit()))
                tot = int(''.join(ch for ch in right if ch.isdigit()))
                if self._progress_total is None:
                    self._progress_total = tot
            except Exception:
                cur = None
        elif any(k in line for k in ['total_timesteps', 'num_timesteps', 'timesteps', 'steps']):
            ints, token = [], ''
            for ch in line:
                if ch.isdigit():
                    token += ch
                else:
                    if token:
                        ints.append(int(token)); token = ''
            if token: ints.append(int(token))
            if ints: cur = max(ints)

        if cur is not None:
            self._latest_progress_value = cur

    def _update_log_and_progress_from_buffer(self):
        # 1. Update Log
        if self._log_buffer:
            temp_buffer = self._log_buffer.copy()
            self._log_buffer.clear()
            for s in temp_buffer:
                formatted = s
                if "ERROR" in s or "Error" in s or "Exception" in s:
                    formatted = f"<span style='color:#ff5555;'>{s}</span>"
                elif "SUCCESS" in s or "Completed" in s:
                    formatted = f"<span style='color:#55ff55;'>{s}</span>"
                self.log.append(formatted)
        
        # 2. Update Progress Bar/Mountain
        if self._progress_mode == 'train' and hasattr(self, 'train_mountain'):
            cur = self._latest_progress_value
            if self._progress_total:
                if cur == self._progress_value: return
                self._progress_value = cur
                
                # Calculate percentage 0-100
                val = max(0, min(cur, int(self._progress_total)))
                pct = (val / float(self._progress_total)) * 100.0
                
                # Update mountain widget
                self.train_mountain.setProgress(pct)

                done_pct = int(pct)
                if self._train_start_time:
                    elapsed = time.time() - self._train_start_time
                else:
                    elapsed = 0.0
                speed = (val / elapsed) if elapsed > 0 else 0.0
                remain = (self._progress_total - val) / speed if speed > 0 else 0.0

                self.train_prog_label.setText(
                    f"Training… {val:,}/{self._progress_total:,} "
                    f"({done_pct}% done)  "
                    f"[ {self._fmt_hms(elapsed)} < {self._fmt_hms(remain)} ]"
                )
            elif cur > 0:
                # Indeterminate
                self.train_prog_label.setText(f"Training… ~{cur:,} steps")
            
            self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def _run_cmd(self, cmd, env=None, purpose=None, progress_total=None):
        if self.proc and self.proc.isRunning():
            QMessageBox.warning(self, "Busy", "Another job is running.")
            return
        self._append_cmd(cmd)
        self._set_job_status("Running", "busy")

        self._progress_mode = purpose
        self._progress_total = progress_total
        self._progress_value = 0
        self._latest_progress_value = 0
        self._log_buffer.clear()
        
        if purpose == 'train':
            self._train_start_time = time.time()
            if hasattr(self, 'train_mountain'):
                self.train_mountain.setProgress(0)
        else:
            self._train_start_time = None
        
        self._timer.start() 

        # --- FIX: Sanitize Environment for Windows ---
        # 1. Start with provided env or copy of system env
        run_env = env if env is not None else os.environ.copy()
        
        # 2. Force UTF-8 to fix the arrow character crash
        run_env["PYTHONIOENCODING"] = "utf-8"

        # 3. Clean "Illegal" keys (Windows hidden vars like '=C:' or empty keys)
        # These keys exist in os.environ but crash subprocess.Popen if passed explicitly
        safe_env = {k: v for k, v in run_env.items() if k and "=" not in k}
        # ---------------------------------------------

        self.proc = ProcWorker(cmd, env=safe_env)
        self.proc.line.connect(self._on_proc_line_received)
        self.proc.finished.connect(self._on_proc_finished)
        self.stop_btn.setEnabled(True)
        self.proc.start()

    def _stop_proc(self):
        if self.proc:
            self._timer.stop() 
            self.proc.stop()
            self.log.append("[!] Termination requested.")
            self._set_job_status("Stopping", "error")
            
            # --- RESET ANIMATION HERE ---
            if hasattr(self, 'train_mountain'):
                self.train_mountain.setProgress(0)
            if hasattr(self, 'train_prog_label'):
                self.train_prog_label.setText("Training stopped (Reset)")

    def _on_proc_finished(self, code: int):
        self._timer.stop() 
        self._update_log_and_progress_from_buffer() 
        self.stop_btn.setEnabled(False)
        self._set_job_status("Ready" if code == 0 else "Attention", "ready" if code == 0 else "error")
        
        # --- SOUND LOGIC ---
        if HAS_WINSOUND and code == 0:
            try:
                if Path("chime.wav").exists():
                    winsound.PlaySound("chime.wav", winsound.SND_FILENAME | winsound.SND_ASYNC)
                else:
                    winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS)
            except Exception:
                pass
        
        if self._progress_mode == 'train' and hasattr(self, 'train_mountain'):
            if code == 0:
                self.train_mountain.setProgress(100) # Finish visuals
                self.train_prog_label.setText('Training complete')
                
                # --- NEW: Refresh lists so the new model shows up immediately ---
                self._refresh_status()
                self._populate_eval_games()
                self._populate_watch_list()
                self.log.append("[✓] Model lists refreshed.")
                # ---------------------------------------------------------------
            else:
                self.train_prog_label.setText('Training stopped')
            
            self._progress_mode = None
            self._progress_total = None
            self._progress_value = 0
            self._train_start_time = None
        
        self.log.append(f"[✓] Process finished with code {code}.")

    def _refresh_all(self):
        self._populate_games(self.train_game)
        self._populate_algos(self.train_algo)
        self._on_train_game_changed()
        self._populate_eval_games()
        self._populate_games(self.manual_game)
        self._populate_watch_list()
        self._populate_games(self.bulk_game)
        self._populate_algos(self.bulk_algo)
        self._populate_personas_for_game(self.bulk_personas, self.bulk_game.currentText())
        self._refresh_status()

    def _populate_games(self, combo: QComboBox):
        combo.clear()
        for g in get_available_games():
            combo.addItem(g)
        if combo.count() > 0:
            combo.setCurrentIndex(0)

    def _populate_algos(self, combo: QComboBox):
        combo.clear()
        algos = get_available_algos_from_grid()
        pref = "ppo" if "ppo" in algos else (algos[0] if algos else None)
        for a in algos:
            combo.addItem(a)
        if pref:
            idx = combo.findText(pref)
            if idx >= 0:
                combo.setCurrentIndex(idx)

    def _populate_personas_for_game(self, combo: QComboBox, game: str):
        combo.clear()
        for p in get_personas_for_game(game):
            combo.addItem(p)
        if combo.count() > 0:
            combo.setCurrentIndex(0)

    # ---------------- Train Tab ----------------
    def _build_train_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(20)

        # Config Group
        grp = QGroupBox("Configuration")
        form = QFormLayout()
        form.setSpacing(15)
        
        self.train_game = QComboBox(); self.train_game.currentTextChanged.connect(self._on_train_game_changed)
        self.train_algo = QComboBox()
        self.train_persona = QComboBox()
        self.train_skill = QComboBox(); self.train_skill.addItems(["Novice", "Expert", "Novice & Expert"])
        self.train_tbroot = QLineEdit(DEFAULT_TB_ROOT)

        form.addRow(self._lbl("Game"), self.train_game)
        form.addRow(self._lbl("Algorithm"), self.train_algo)
        form.addRow(self._lbl("Persona"), self.train_persona)
        form.addRow(self._lbl("Skill Level"), self.train_skill)
        grp.setLayout(form)

        # Steps Group
        grp2 = QGroupBox("Duration & Logging")
        form2 = QFormLayout()
        form2.setSpacing(15)
        
        self.train_step_choice = QComboBox()
        self.train_step_choice.addItems(["10,000", "50,000", "100,000", "500,000", "1,000,000", "Custom"])
        self.train_steps = QSpinBox(); self.train_steps.setRange(1_000, 100_000_000); self.train_steps.setValue(300_000)
        self.train_steps.setEnabled(False)

        def on_step_choice_changed(s):
            self.train_steps.setEnabled(s == "Custom")
        self.train_step_choice.currentTextChanged.connect(on_step_choice_changed)

        form2.addRow(self._lbl("Steps Preset"), self.train_step_choice)
        form2.addRow(self._lbl("Custom Steps"), self.train_steps)
        form2.addRow(self._lbl("TensorBoard Root"), self.train_tbroot)
        grp2.setLayout(form2)

        # Execution Group
        grp3 = QGroupBox("Execution")
        v = QVBoxLayout()
        self.train_prog_label = QLabel("Ready to train")
        self.train_prog_label.setStyleSheet("color: #888; font-style: italic;")
        
        # --- REPLACED PROGRESS BAR WITH MOUNTAIN WIDGET ---
        self.train_mountain = AnimatedMountainWidget()
        # --------------------------------------------------
        
        btn_train = QPushButton("INITIATE TRAINING SEQUENCE"); 
        btn_train.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        btn_train.setObjectName("accentButton")
        btn_train.setFixedHeight(48)
        btn_train.clicked.connect(self._on_train_clicked)

        v.addWidget(self.train_prog_label)
        v.addWidget(self.train_mountain) # Added mountain
        v.addSpacing(10)
        v.addWidget(btn_train)
        grp3.setLayout(v)

        lay.addWidget(grp)
        lay.addWidget(grp2)
        lay.addWidget(grp3)
        lay.addStretch(1)
        return w

    def _on_train_game_changed(self):
        game = self.train_game.currentText()
        if game:
            self._populate_personas_for_game(self.train_persona, game)

    def _on_train_clicked(self):
        game = self.train_game.currentText()
        algo = self.train_algo.currentText()
        persona = self.train_persona.currentText()
        skill = self.train_skill.currentText()
        tb_root = self.train_tbroot.text().strip() or DEFAULT_TB_ROOT

        if not all([game, algo, persona, skill]):
            QMessageBox.warning(self, "Missing Data", "Please select Game, Algo, Persona, and Skill.")
            return

        self.train_mountain.setProgress(0)
        self.train_prog_label.setText('Initializing...')

        total_steps = self._selected_total_steps()

        if total_steps:
            self.train_prog_label.setText(f"Training... 0/{total_steps:,}")
            cmd = [
                sys.executable, "-m", "code.scripts.train",
                f"game={game}",
                f"model={algo}",
                f"persona={persona}",
                "skill=Custom",
                f"+skills.Custom={total_steps}",
                f"tb_root={tb_root}",
            ]
            self._run_cmd(cmd, purpose='train', progress_total=total_steps)
            return

        if skill == "Novice & Expert":
            def run_both():
                self._timer.stop() 
                self.train_mountain.setProgress(0)
                self.train_prog_label.setText("Training sequence: Novice -> Expert")
                
                for sk in ("Novice", "Expert"):
                    cmd2 = [
                        sys.executable, "-m", "code.scripts.train",
                        f"game={game}", f"model={algo}", f"persona={persona}",
                        f"skill={sk}", f"tb_root={tb_root}",
                    ]
                    self.log.append(f"<span style='color:#ff7f7f;'>>>></span> {' '.join(cmd2)}")
                    creation_flags = 0
                    if platform.system() == "Windows":
                        creation_flags = subprocess.CREATE_NO_WINDOW
                    p = subprocess.run(cmd2, creationflags=creation_flags)
                    self.log.append(f"[run {sk}] exit {p.returncode}")
                    self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

                if HAS_WINSOUND:
                    try: winsound.MessageBeep()
                    except Exception: pass
                
                self.log.append("[✓] Completed Novice & Expert runs.")
                self.train_mountain.setProgress(100)
                self.train_prog_label.setText("Sequence Complete")
                
            Thread(target=run_both, daemon=True).start()
            return

        cmd = [
            sys.executable, "-m", "code.scripts.train",
            f"game={game}", f"model={algo}", f"persona={persona}",
            f"skill={skill}", f"tb_root={tb_root}",
        ]
        self.train_mountain.setProgress(0)
        self.train_prog_label.setText(f"Training... {skill}")
        self._run_cmd(cmd, purpose='train')

    # ---------------- Evaluate Tab ----------------
    def _build_eval_tab(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)
        grp = QGroupBox("Evaluation Parameters")
        form = QFormLayout()
        
        self.eval_game = QComboBox()
        self.eval_eps = QSpinBox(); self.eval_eps.setRange(1, 500); self.eval_eps.setValue(5)
        
        form.addRow(self._lbl("Target Game"), self.eval_game)
        form.addRow(self._lbl("Episodes per Model"), self.eval_eps)
        grp.setLayout(form)
        
        run_btn = QPushButton("RUN EVALUATION (Best Models)"); 
        run_btn.setIcon(self.style().standardIcon(QStyle.SP_DialogApplyButton))
        run_btn.setObjectName("accentButton")
        run_btn.setFixedHeight(50)
        run_btn.clicked.connect(self._on_eval_clicked)
        
        lay.addWidget(grp)
        lay.addSpacing(20)
        lay.addWidget(run_btn)
        lay.addStretch(1)
        return w

    def _populate_eval_games(self):
        self.eval_game.clear()
        best_dir = MODELS_DIR / "best"
        games = []
        if best_dir.exists():
            subfolders = [p for p in best_dir.iterdir() if p.is_dir() and (p / "best_model.zip").exists()]
            games = sorted(set(f.name.split("_")[0] for f in subfolders))
        for g in games:
            self.eval_game.addItem(g)

    def _on_eval_clicked(self):
        game = self.eval_game.currentText()
        if not game:
            QMessageBox.information(self, "No models", "No best models found for any game.")
            return
        best_dir = MODELS_DIR / "best"
        subfolders = [p for p in best_dir.iterdir() if p.is_dir() and (p / "best_model.zip").exists() and p.name.startswith(game)]
        if not subfolders:
            QMessageBox.information(self, "Missing", f"No best models for {game}.")
            return
        self.log.append(f"Found {len(subfolders)} model(s) for '{game}'. Running eval...")
        
        def run_eval_sequence():
            self._timer.stop() 
            for model_dir in subfolders:
                model_zip = model_dir / "best_model.zip"
                model_name = model_dir.name
                parts = model_name.split("_")
                algo = parts[1] if len(parts) > 1 else "ppo"
                out_json = MODELS_DIR / f"{model_name}_eval.json"
                metrics_class = f"code.metrics.{game}_balance.{game.capitalize()}BalanceStats"
                cmd = [
                    sys.executable, "-m", "code.scripts.evaluate",
                    "--game", game, "--algo", algo, "--model", str(model_zip),
                    "--episodes", str(self.eval_eps.value()), "--render", "none",
                    "--out", str(out_json), "--metrics", metrics_class,
                ]
                self.log.append(f"<span style='color:#ff7f7f;'>>>></span> {' '.join(cmd)}")
                creation_flags = 0
                if platform.system() == "Windows":
                    creation_flags = subprocess.CREATE_NO_WINDOW
                p = subprocess.run(cmd, creationflags=creation_flags)
                self.log.append(f"[eval {model_name}] exit {p.returncode}")
                self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())
            self.log.append(f"[✓] Evaluation sequence complete for {game}.")

        Thread(target=run_eval_sequence, daemon=True).start()

    # ---------------- TensorBoard Tab ----------------
    def _build_tb_tab(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)
        grp = QGroupBox("Server Settings")
        form = QFormLayout()
        self.tb_root_edit = QLineEdit(DEFAULT_TB_ROOT)
        self.tb_port = QSpinBox(); self.tb_port.setRange(1024, 65535); self.tb_port.setValue(6006)
        form.addRow(self._lbl("Log Directory"), self.tb_root_edit)
        form.addRow(self._lbl("Port"), self.tb_port)
        grp.setLayout(form)
        
        run_btn = QPushButton("LAUNCH TENSORBOARD"); 
        run_btn.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        run_btn.setObjectName("accentButton")
        run_btn.setFixedHeight(50)
        run_btn.clicked.connect(self._on_tb_clicked)
        
        lay.addWidget(grp)
        lay.addSpacing(20)
        lay.addWidget(run_btn)
        lay.addStretch(1)
        return w

    def _on_tb_clicked(self):
        root = Path(self.tb_root_edit.text().strip() or DEFAULT_TB_ROOT)
        if not root.exists():
            QMessageBox.information(self, "Missing", f"No '{root}/' folder. Train first.")
            return
        port = int(self.tb_port.value())
        cmd = [sys.executable, "-m", "tensorboard.main", "--logdir", str(root), "--port", str(port)]
        self._run_cmd(cmd)
        def later():
            time.sleep(3); open_browser(f"http://localhost:{port}/")
        Thread(target=later, daemon=True).start()

    # ---------------- Manual Play Tab ----------------
    def _build_manual_tab(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)
        grp = QGroupBox("Game Settings")
        form = QFormLayout()
        self.manual_game = QComboBox()
        self.manual_fps = QSpinBox(); self.manual_fps.setRange(5, 240); self.manual_fps.setValue(30)
        form.addRow(self._lbl("Select Game"), self.manual_game)
        form.addRow(self._lbl("Target FPS"), self.manual_fps)
        grp.setLayout(form)
        
        run_btn = QPushButton("LAUNCH MANUAL PLAY"); 
        run_btn.setIcon(self.style().standardIcon(QStyle.SP_ArrowForward))
        run_btn.setObjectName("accentButton")
        run_btn.setFixedHeight(50)
        run_btn.clicked.connect(self._on_manual_clicked)
        
        lay.addWidget(grp)
        lay.addSpacing(20)
        lay.addWidget(run_btn)
        lay.addStretch(1)
        return w

    def _on_manual_clicked(self):
        game = self.manual_game.currentText()
        if not game:
            return
        env = os.environ.copy()
        if "SDL_VIDEODRIVER" in env:
            env.pop("SDL_VIDEODRIVER")
        cmd = [sys.executable, "-m", "code.scripts.manual_play", "--game", game, "--fps", str(self.manual_fps.value())]
        self._run_cmd(cmd, env=env)

    # ---------------- Watch Agent Tab ----------------
    def _build_watch_tab(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)
        grp = QGroupBox("Visualization Settings")
        form = QFormLayout()
        
        self.watch_list = QComboBox()
        self.watch_eps = QSpinBox(); self.watch_eps.setRange(1, 200); self.watch_eps.setValue(10)
        self.watch_fps = QSpinBox(); self.watch_fps.setRange(5, 240); self.watch_fps.setValue(30)
        
        form.addRow(self._lbl("Trained Model"), self.watch_list)
        form.addRow(self._lbl("Episodes"), self.watch_eps)
        form.addRow(self._lbl("Playback FPS"), self.watch_fps)
        grp.setLayout(form)
        
        run_btn = QPushButton("VISUALIZE AGENT"); 
        run_btn.setIcon(self.style().standardIcon(QStyle.SP_FileDialogContentsView))
        run_btn.setObjectName("accentButton")
        run_btn.setFixedHeight(50)
        run_btn.clicked.connect(self._on_watch_clicked)
        
        lay.addWidget(grp)
        lay.addSpacing(20)
        lay.addWidget(run_btn)
        lay.addStretch(1)
        return w

    def _populate_watch_list(self):
        self.watch_list.clear()
        best_dir = MODELS_DIR / "best"
        if not best_dir.exists():
            return
        for folder in best_dir.iterdir():
            if folder.is_dir() and (folder / "best_model.zip").exists():
                parts = folder.name.split("_")
                if len(parts) >= 5:
                    game, algo, _g2, persona, skill = parts[0], parts[1], parts[2], parts[3], parts[4]
                    display = f"{game.upper()} | {algo} | {persona} | {skill}"
                else:
                    display = folder.name
                self.watch_list.addItem(display, userData=str((folder / "best_model.zip").resolve()))

    def _on_watch_clicked(self):
        data = self.watch_list.currentData()
        if not data:
            QMessageBox.information(self, "Missing", "No trained best models found.")
            return
        model_path = data
        cmd = [
            sys.executable, "-m", "code.scripts.watch_agent",
            model_path, "--episodes", str(self.watch_eps.value()), "--fps", str(self.watch_fps.value())
        ]
        self._run_cmd(cmd)

    # ---------------- Train ALL Tab ----------------
    def _build_bulk_tab(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)
        grp = QGroupBox("Bulk Training Configuration")
        form = QFormLayout()
        
        self.bulk_game = QComboBox(); self.bulk_game.currentTextChanged.connect(self._on_bulk_game_changed)
        self.bulk_algo = QComboBox()
        self.bulk_all_algos = QCheckBox("Train ALL algorithms (Grid)")
        self.bulk_personas = QComboBox()
        
        form.addRow(self._lbl("Game"), self.bulk_game)
        form.addRow(self._lbl("Single Algo"), self.bulk_algo)
        form.addRow(self._lbl(" "), self.bulk_all_algos)
        form.addRow(self._lbl("Persona"), self.bulk_personas)
        grp.setLayout(form)
        
        run_btn = QPushButton("EXECUTE BULK TRAINING (NOVICE + EXPERT)"); 
        run_btn.setIcon(self.style().standardIcon(QStyle.SP_DialogYesButton))
        run_btn.setObjectName("accentButton")
        run_btn.setFixedHeight(50)
        run_btn.clicked.connect(self._on_bulk_clicked)
        
        lay.addWidget(grp)
        lay.addSpacing(20)
        lay.addWidget(run_btn)
        lay.addStretch(1)
        return w

    def _on_bulk_game_changed(self):
        self._populate_personas_for_game(self.bulk_personas, self.bulk_game.currentText())

    def _on_bulk_clicked(self):
        game = self.bulk_game.currentText()
        if not game: return
        algos = get_available_algos_from_grid()
        if not algos:
            QMessageBox.information(self, "Missing", "No algorithms found from grid.yaml")
            return
        selected_algos = algos if self.bulk_all_algos.isChecked() else [self.bulk_algo.currentText()]
        persona = self.bulk_personas.currentText()
        skills = ["Novice", "Expert"]

        def run_all():
            total = 0
            self._timer.stop()
            for algo in selected_algos:
                for sk in skills:
                    cmd = [
                        sys.executable, "-m", "code.scripts.train",
                        f"game={game}", f"model={algo}", f"persona={persona}", f"skill={sk}", f"tb_root={DEFAULT_TB_ROOT}",
                    ]
                    self.log.append(f"<span style='color:#ff7f7f;'>>>></span> {' '.join(cmd)}")
                    creation_flags = 0
                    if platform.system() == "Windows":
                        creation_flags = subprocess.CREATE_NO_WINDOW
                    p = subprocess.run(cmd, creationflags=creation_flags)
                    self.log.append(f"[{algo}/{sk}] exit {p.returncode}")
                    self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())
                    total += 1
            if HAS_WINSOUND:
                try: winsound.MessageBeep()
                except Exception: pass
            self.log.append(f"[✓] Completed training for {total} run(s) in game '{game}'.")
        Thread(target=run_all, daemon=True).start()

    # ---------------- Status Tab ----------------
    def _build_status_tab(self) -> QWidget:
        w = QWidget(); root = QVBoxLayout(w)

        overview = QGroupBox("Project Overview")
        overview_grid = QGridLayout()
        overview_grid.setHorizontalSpacing(12)
        overview_grid.setVerticalSpacing(12)

        overview_games = self._create_stat_card("Games", "0", "configured")
        overview_algos = self._create_stat_card("Algorithms", "0", "ready")
        overview_models = self._create_stat_card("Models", "0", "trained")
        overview_trained_games = self._create_stat_card("Trained Games", "0", "with best model")

        self.ov_games = overview_games.value_label
        self.ov_algos = overview_algos.value_label
        self.ov_trained = overview_models.value_label
        self.ov_trained_games = overview_trained_games.value_label

        overview_grid.addWidget(overview_games, 0, 0)
        overview_grid.addWidget(overview_algos, 0, 1)
        overview_grid.addWidget(overview_models, 1, 0)
        overview_grid.addWidget(overview_trained_games, 1, 1)
        overview.setLayout(overview_grid)

        games_box = QGroupBox("Games Status")
        self.games_table = QTableWidget(0, 2)
        self.games_table.setHorizontalHeaderLabels(["Game", "Status"])
        self.games_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.games_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.games_table.verticalHeader().setVisible(False)
        self.games_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        v1 = QVBoxLayout(); v1.addWidget(self.games_table); games_box.setLayout(v1)

        alg_box = QGroupBox("Algorithms")
        self.alg_table = QTableWidget(0, 1)
        self.alg_table.setHorizontalHeaderLabels(["Algorithm"])
        self.alg_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.alg_table.verticalHeader().setVisible(False)
        self.alg_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        v2 = QVBoxLayout(); v2.addWidget(self.alg_table); alg_box.setLayout(v2)

        refresh = QPushButton("REFRESH STATUS DATA"); 
        refresh.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        refresh.setObjectName("accentButton")
        refresh.setFixedHeight(40)
        refresh.clicked.connect(self._refresh_status)

        root.addWidget(overview)
        root.addWidget(games_box)
        root.addWidget(alg_box)
        root.addWidget(refresh)
        root.addStretch(1)
        return w

    def _bold_label_left(self, grid_like, text: str, value_label: QLabel):
        lbl = QLabel(text)
        value_label.setStyleSheet("color: #ff7f7f; font-weight: bold;")
        grid_like.add_row(lbl, value_label)

    def _refresh_status(self):
        games = get_available_games()
        algos = get_available_algos_from_grid()
        trained = get_trained_games_from_models_flat()
        trained_models = get_trained_models_count()
        default_device = "cpu"
        default_n_envs = 1
        cfg = load_grid_config()
        if cfg is not None:
            try:
                default_n_envs = max(1, int(cfg.get("n_envs", 1)))
            except Exception:
                default_n_envs = 1
            default_device = str(cfg.get("device", "cpu") or "cpu").lower()

        self.ov_games.setText(str(len(games)))
        self.ov_algos.setText(str(len(algos)))
        self.ov_trained.setText(str(trained_models))
        if hasattr(self, "ov_trained_games"):
            self.ov_trained_games.setText(str(len(trained)))
        self.stat_games.value_label.setText(str(len(games)))
        self.stat_games.meta_label.setText("configured")
        self.stat_algos.value_label.setText(str(len(algos)))
        self.stat_algos.meta_label.setText("available")
        self.stat_models.value_label.setText(str(trained_models))
        self.stat_models.meta_label.setText(f"{len(trained)} games with checkpoints")
        self.stat_runtime.value_label.setText(f"{default_n_envs} · {default_device.upper()}")
        self.stat_runtime.meta_label.setText("grid default")

        self.games_table.setRowCount(0)
        for g in games:
            row = self.games_table.rowCount(); self.games_table.insertRow(row)
            self.games_table.setItem(row, 0, QTableWidgetItem(g))
            status = "✓ Trained" if g in trained else "○ Not trained"
            item = QTableWidgetItem(status)
            if g in trained:
                item.setForeground(QColor("#55ff55"))
            else:
                item.setForeground(QColor("#888888"))
            self.games_table.setItem(row, 1, item)

        self.alg_table.setRowCount(0)
        for a in algos:
            row = self.alg_table.rowCount(); self.alg_table.insertRow(row)
            self.alg_table.setItem(row, 0, QTableWidgetItem(a))

    # ---------------- Maintenance Tab ----------------
    def _build_maint_tab(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)
        grp = QGroupBox("Danger Zone")
        v = QVBoxLayout()
        
        warn = QLabel("Select items to permanently delete. This cannot be undone.")
        warn.setStyleSheet("color: #ff7f7f; font-weight: bold;")
        
        self.chk_del_logs = QCheckBox("Delete TensorBoard Logs (mylogs/)")
        self.chk_del_models = QCheckBox("Delete Trained Models (models/)")
        
        del_btn = QPushButton("PERFORM DELETION"); 
        del_btn.setObjectName("dangerButton")
        del_btn.clicked.connect(self._on_delete_clicked)
        
        v.addWidget(warn)
        v.addSpacing(10)
        v.addWidget(self.chk_del_logs)
        v.addWidget(self.chk_del_models)
        v.addSpacing(20)
        v.addWidget(del_btn)
        grp.setLayout(v)
        
        lay.addWidget(grp)
        lay.addStretch(1)
        return w

    def _on_delete_clicked(self):
        from shutil import rmtree
        to_delete = []
        if self.chk_del_logs.isChecked():
            to_delete.append(Path(DEFAULT_TB_ROOT))
        if self.chk_del_models.isChecked():
            to_delete.append(MODELS_DIR)
        if not to_delete:
            QMessageBox.information(self, "Nothing selected", "Check at least one item to delete.")
            return
        text = "\n".join(str(p.resolve()) for p in to_delete)
        reply = QMessageBox.question(self, "CONFIRM DELETION", f"Permanently delete the following?\n{text}", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        for path in to_delete:
            if path.exists():
                try:
                    rmtree(path); path.mkdir(parents=True, exist_ok=True)
                    self.log.append(f"✓ Cleared and recreated {path}")
                except Exception as e:
                    self.log.append(f"[!] Failed to clear {path}: {e}")
        self._refresh_status()

class QGridLayoutLike:
    def __init__(self):
        from PyQt5.QtWidgets import QGridLayout
        self.layout = QGridLayout()
        self._row = 0
        self.layout.setSpacing(10)
    def add_row(self, left: QWidget, right: QWidget):
        self.layout.addWidget(left, self._row, 0, alignment=Qt.AlignLeft)
        self.layout.addWidget(right, self._row, 1, alignment=Qt.AlignLeft)
        self._row += 1

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("Fusion") 
    gui = RLManagerGUI()
    gui.show()
    sys.exit(app.exec_())


