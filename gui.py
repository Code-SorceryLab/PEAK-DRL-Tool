# menu_gui.py — PyQt5 GUI for RL training/eval/manual play/TensorBoard
# Run:  python menu_gui.py

import os
import sys
import platform
import subprocess
import time
from pathlib import Path
from threading import Thread

try:
    import winsound
    HAS_WINSOUND = True
except Exception:
    HAS_WINSOUND = False

from omegaconf import OmegaConf

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QLineEdit, QTextEdit, QSpinBox, QListWidget,
    QListWidgetItem, QMessageBox, QCheckBox, QGroupBox, QFormLayout, QProgressBar,
    QTableWidget, QTableWidgetItem, QHeaderView, QSplitter, QAbstractItemView
)

DEFAULT_TB_ROOT = "mylogs"
MODELS_DIR = Path("models/")
CONF_ROOT = Path("code/conf")
GRID_CONFIG_PATH = CONF_ROOT / "grid.yaml"
CONF_GAME_DIR = CONF_ROOT / "game"
CONF_REWARD_DIR = CONF_ROOT / "reward"
CONF_ALGO_DIR = CONF_ROOT / "algo"

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
    'PyQt5>=5.15.0',
]

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
    return sorted([p for p in grid_personas if (CONF_REWARD_DIR / f"{p}.yaml").exists()])

def get_personas_for_game(game: str):
    all_personas = get_available_personas_from_grid()
    filtered = [p for p in all_personas if p.startswith(f"{game}_")]
    return filtered if filtered else all_personas

def get_available_games():
    if not CONF_GAME_DIR.exists():
        return []
    return sorted([f.stem for f in CONF_GAME_DIR.glob("*.yaml")])

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
            self._proc = subprocess.Popen(
                self.cmd,
                cwd=self.cwd,
                env=self.env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
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
                self._proc.terminate()
        except Exception:
            pass

class RLManagerGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RL Manager — Training / Eval / TensorBoard / Manual Play")
        self.resize(1200, 1000)

        self.setStyleSheet(
            """
            QWidget { font-size: 18px; }
            QGroupBox { font-weight: 600; border: 1px solid #d0d0d0; border-radius: 8px; margin-top: 12px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
            QTabWidget::pane { border: 1px solid #d0d0d0; }
            QTabBar::tab { padding: 6px 12px; }
            QPushButton { padding: 6px 10px; }
            QTableWidget { gridline-color: #e0e0e0; }
            QLabel[role="form-left"] { font-weight: 600; }
            """
        )

        self.proc = None
        self._progress_mode = None
        self._progress_total = None
        self._progress_value = 0
        self._train_start_time = None

        self.tabs = QTabWidget()
        tb = self.tabs.tabBar()
        tf = tb.font(); tf.setBold(True); tb.setFont(tf)
        tb.setElideMode(Qt.ElideRight); tb.setUsesScrollButtons(True)

        self.log = QTextEdit(); self.log.setReadOnly(True)
        lf = self.log.font(); lf.setPointSize(max(11, lf.pointSize() + 2)); self.log.setFont(lf)

        self.tab_status = self._build_status_tab()
        self.tab_train = self._build_train_tab()
        self.tab_eval = self._build_eval_tab()
        self.tab_tensorboard = self._build_tb_tab()
        self.tab_manual = self._build_manual_tab()
        self.tab_watch = self._build_watch_tab()
        self.tab_bulk = self._build_bulk_tab()
        self.tab_maint = self._build_maint_tab()

        self.tabs.addTab(self.tab_status, "Status")
        self.tabs.addTab(self.tab_train, "Train")
        self.tabs.addTab(self.tab_eval, "Evaluate")
        self.tabs.addTab(self.tab_tensorboard, "TensorBoard")
        self.tabs.addTab(self.tab_manual, "Manual Play")
        self.tabs.addTab(self.tab_watch, "Watch Agent")
        self.tabs.addTab(self.tab_bulk, "Train ALL")
        self.tabs.addTab(self.tab_maint, "Maintenance")

        refresh_btn = QPushButton("Refresh Configs"); refresh_btn.clicked.connect(self._refresh_all)
        req_btn = QPushButton("Write requirements.txt"); req_btn.clicked.connect(lambda: write_requirements(self.log, overwrite=True))
        req_install_btn = QPushButton("Install requirements"); req_install_btn.clicked.connect(lambda: self._install_requirements_safe())
        self.stop_btn = QPushButton("Stop Running Job"); self.stop_btn.clicked.connect(self._stop_proc); self.stop_btn.setEnabled(False)
        self.clear_log_btn = QPushButton("Clear Output"); self.clear_log_btn.clicked.connect(self.log.clear)

        top_row = QHBoxLayout()
        top_row.addWidget(refresh_btn); top_row.addWidget(req_btn); top_row.addWidget(req_install_btn)
        top_row.addStretch(1); top_row.addWidget(self.clear_log_btn); top_row.addWidget(self.stop_btn)

        splitter = QSplitter(Qt.Vertical)
        tabs_wrap = QWidget(); tl = QVBoxLayout(tabs_wrap); tl.setContentsMargins(0,0,0,0); tl.addWidget(self.tabs)
        log_wrap = QWidget(); ll = QVBoxLayout(log_wrap); ll.setContentsMargins(0,0,0,0)
        out_lbl = QLabel("Output Log:"); f = out_lbl.font(); f.setBold(True); out_lbl.setFont(f)
        ll.addWidget(out_lbl); ll.addWidget(self.log)
        splitter.addWidget(tabs_wrap); splitter.addWidget(log_wrap)
        splitter.setStretchFactor(0, 1); splitter.setStretchFactor(1, 2); splitter.setSizes([380, 600])

        root = QVBoxLayout(self)
        root.addLayout(top_row); root.addWidget(splitter)

        self._refresh_all()
        self.tabs.setCurrentIndex(0)

    # ---------------- helpers ----------------
    def _install_requirements_safe(self):
        # run pip in a thread so UI stays responsive
        Thread(target=lambda: install_requirements(self.log), daemon=True).start()

    def _lbl(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setProperty('role', 'form-left')
        f = lbl.font(); f.setBold(True); lbl.setFont(f)
        return lbl

    def _append_cmd(self, cmd_list):
        pretty = " ".join(str(x) for x in cmd_list)
        self.log.append(f">>> {pretty}")

    @staticmethod
    def _fmt_hms(sec: float) -> str:
        sec = max(0, int(sec))
        m, s = divmod(sec, 60)
        h, m = divmod(m, 60)
        return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

    def _selected_total_steps(self) -> int | None:
        """Return total steps if the Steps selector is used; else None."""
        s = self.train_step_choice.currentText().strip()
        if not s:
            return None
        if s.lower() == "custom":
            return int(self.train_steps.value())
        try:
            return int(s.replace(",", ""))
        except Exception:
            return None

    def _on_proc_line_received(self, s: str):
        self.log.append(s)

        if self._progress_mode != 'train' or not hasattr(self, 'train_prog'):
            return

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
                    self.train_prog.setRange(0, tot)
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
            if token:
                ints.append(int(token))
            if ints:
                cur = max(ints)

        if cur is None:
            return

        self._progress_value = cur
        if self._progress_total:
            val = max(0, min(cur, int(self._progress_total)))
            self.train_prog.setValue(val)

            done_pct = int(100.0 * val / float(self._progress_total))
            left_pct = 100 - done_pct

            # ETA / speed like tqdm
            if self._train_start_time:
                elapsed = time.time() - self._train_start_time
            else:
                elapsed = 0.0
            speed = (val / elapsed) if elapsed > 0 else 0.0
            remain = (self._progress_total - val) / speed if speed > 0 else 0.0

            self.train_prog.setFormat("%p%")               # show percent overlay
            self.train_prog.setTextVisible(True)

            self.train_prog_label.setText(
                f"Training… {val:,}/{self._progress_total:,} "
                f"({done_pct}% done, {left_pct}% left)  "
                f"[ {self._fmt_hms(elapsed)} < {self._fmt_hms(remain)} , "
                f"{speed:,.0f} it/s ]"
            )
        else:
            self.train_prog_label.setText(f"Training… ~{cur:,} steps")

    def _run_cmd(self, cmd, env=None, purpose=None, progress_total=None):
        if self.proc and self.proc.isRunning():
            QMessageBox.warning(self, "Busy", "Another job is running.")
            return
        self._append_cmd(cmd)

        self._progress_mode = purpose
        self._progress_total = progress_total
        self._progress_value = 0
        if purpose == 'train':
            self._train_start_time = time.time()
        else:
            self._train_start_time = None

        self.proc = ProcWorker(cmd, env=env)
        self.proc.line.connect(self._on_proc_line_received)
        self.proc.finished.connect(self._on_proc_finished)
        self.stop_btn.setEnabled(True)
        self.proc.start()

    def _stop_proc(self):
        if self.proc:
            self.proc.stop()
            self.log.append("[!] Termination requested.")

    def _on_proc_finished(self, code: int):
        self.stop_btn.setEnabled(False)
        if HAS_WINSOUND and code == 0:
            try:
                winsound.MessageBeep()
            except Exception:
                pass
        if self._progress_mode == 'train' and hasattr(self, 'train_prog'):
            self.train_prog.setRange(0, 100)
            self.train_prog.setValue(100 if code == 0 else 0)
            self.train_prog_label.setText('Training complete' if code == 0 else 'Training stopped')
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

    # ---------------- Train Tab ----------------
    def _build_train_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)

        form = QFormLayout()
        self.train_game = QComboBox(); self.train_game.currentTextChanged.connect(self._on_train_game_changed)
        self.train_algo = QComboBox()
        self.train_persona = QComboBox()
        self.train_skill = QComboBox(); self.train_skill.addItems(["Novice", "Expert", "Novice & Expert"])
        self.train_tbroot = QLineEdit(DEFAULT_TB_ROOT)

        form.addRow(self._lbl("Game"), self.train_game)
        form.addRow(self._lbl("Algorithm"), self.train_algo)
        form.addRow(self._lbl("Persona"), self.train_persona)
        form.addRow(self._lbl("Skill"), self.train_skill)

        # Steps controller (drives total & progress)
        self.train_step_choice = QComboBox()
        self.train_step_choice.addItems(["10,000", "50,000", "100,000", "500,000", "1,000,000", "Custom"])
        self.train_steps = QSpinBox(); self.train_steps.setRange(1_000, 100_000_000); self.train_steps.setValue(300_000)
        self.train_steps.setEnabled(False)

        def on_step_choice_changed(s):
            self.train_steps.setEnabled(s == "Custom")
        self.train_step_choice.currentTextChanged.connect(on_step_choice_changed)

        form.addRow(self._lbl("Steps"), self.train_step_choice)
        form.addRow(self._lbl("Custom steps"), self.train_steps)
        form.addRow(self._lbl("TensorBoard root"), self.train_tbroot)

        # Progress widgets
        self.train_prog_label = QLabel("Idle")
        self.train_prog = QProgressBar(); self.train_prog.setValue(0); self.train_prog.setTextVisible(True); self.train_prog.setFormat("%p%")
        form.addRow(self._lbl("Status"), self.train_prog_label)
        form.addRow(self._lbl("Progress"), self.train_prog)

        btn_row = QHBoxLayout()
        btn_train = QPushButton("Run Training"); btn_train.clicked.connect(self._on_train_clicked)
        btn_row.addWidget(btn_train); btn_row.addStretch(1)

        lay.addLayout(form); lay.addLayout(btn_row); lay.addStretch(1)
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
            QMessageBox.warning(self, "Missing", "Pick game, algo, persona, and skill.")
            return

        self.train_prog.setValue(0)
        self.train_prog_label.setText('Preparing…')

        # If Steps is specified, we ALWAYS run a determinate Custom-steps train
        total_steps = self._selected_total_steps()

        if total_steps:
            self.train_prog.setRange(0, total_steps)
            self.train_prog_label.setText(f"Training… 0/{total_steps:,}")
            cmd = [
                sys.executable, "-m", "code.scripts.train",
                f"game={game}", f"model={algo}", f"persona={persona}",
                "skill=Custom", f"+skills.Custom={total_steps}", f"tb_root={tb_root}",
            ]
            self._run_cmd(cmd, purpose='train', progress_total=total_steps)
            return

        # Fallback: skill-guided (unknown total)
        if skill == "Novice & Expert":
            def run_both():
                self.train_prog.setRange(0, 0)
                self.train_prog_label.setText("Training… Novice & Expert")
                for sk in ("Novice", "Expert"):
                    cmd2 = [
                        sys.executable, "-m", "code.scripts.train",
                        f"game={game}", f"model={algo}", f"persona={persona}",
                        f"skill={sk}", f"tb_root={tb_root}",
                    ]
                    self._append_cmd(cmd2)
                    p = subprocess.run(cmd2)
                    self.log.append(f"[run {sk}] exit {p.returncode}")
                if HAS_WINSOUND:
                    try: winsound.MessageBeep()
                    except Exception: pass
                self.train_prog.setRange(0, 100)
                self.train_prog.setValue(100)
                self.train_prog_label.setText("Training complete")
                self.log.append("[✓] Completed Novice & Expert runs.")
            Thread(target=run_both, daemon=True).start()
            return

        cmd = [
            sys.executable, "-m", "code.scripts.train",
            f"game={game}", f"model={algo}", f"persona={persona}",
            f"skill={skill}", f"tb_root={tb_root}",
        ]
        self.train_prog.setRange(0, 0)
        self.train_prog_label.setText(f"Training… {skill}")
        self._run_cmd(cmd, purpose='train')

    # ---------------- Evaluate Tab ----------------
    def _build_eval_tab(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)
        form = QFormLayout()
        self.eval_game = QComboBox()
        self.eval_eps = QSpinBox(); self.eval_eps.setRange(1, 500); self.eval_eps.setValue(5)
        form.addRow(self._lbl("Game (from models/best)"), self.eval_game)
        form.addRow(self._lbl("Episodes per model"), self.eval_eps)
        run_btn = QPushButton("Run Evaluation For All Best Models"); run_btn.clicked.connect(self._on_eval_clicked)
        lay.addLayout(form); lay.addWidget(run_btn); lay.addStretch(1)
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
        self.log.append(f"Found {len(subfolders)} model(s) for '{game}'. Running eval…")
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
            self._run_cmd(cmd)

    # ---------------- TensorBoard Tab ----------------
    def _build_tb_tab(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)
        form = QFormLayout()
        self.tb_root_edit = QLineEdit(DEFAULT_TB_ROOT)
        self.tb_port = QSpinBox(); self.tb_port.setRange(1024, 65535); self.tb_port.setValue(6006)
        form.addRow(self._lbl("Log root"), self.tb_root_edit)
        form.addRow(self._lbl("Port"), self.tb_port)
        run_btn = QPushButton("Launch TensorBoard and Open Browser"); run_btn.clicked.connect(self._on_tb_clicked)
        lay.addLayout(form); lay.addWidget(run_btn); lay.addStretch(1)
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
        form = QFormLayout()
        self.manual_game = QComboBox()
        self.manual_fps = QSpinBox(); self.manual_fps.setRange(5, 240); self.manual_fps.setValue(30)
        form.addRow(self._lbl("Game"), self.manual_game)
        form.addRow(self._lbl("FPS"), self.manual_fps)
        run_btn = QPushButton("Start Manual Play"); run_btn.clicked.connect(self._on_manual_clicked)
        lay.addLayout(form); lay.addWidget(run_btn); lay.addStretch(1)
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
        form = QFormLayout()
        self.watch_list = QComboBox()
        self.watch_eps = QSpinBox(); self.watch_eps.setRange(1, 200); self.watch_eps.setValue(10)
        self.watch_fps = QSpinBox(); self.watch_fps.setRange(5, 240); self.watch_fps.setValue(30)
        form.addRow(self._lbl("Model (from models/best)"), self.watch_list)
        form.addRow(self._lbl("Episodes"), self.watch_eps)
        form.addRow(self._lbl("FPS"), self.watch_fps)
        run_btn = QPushButton("Watch Selected Agent"); run_btn.clicked.connect(self._on_watch_clicked)
        lay.addLayout(form); lay.addWidget(run_btn); lay.addStretch(1)
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
                    display = f"{game:<8} | {algo:<4} | {persona:<12} | {skill:<8}"
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
        form = QFormLayout()
        self.bulk_game = QComboBox(); self.bulk_game.currentTextChanged.connect(self._on_bulk_game_changed)
        self.bulk_algo = QComboBox()
        self.bulk_all_algos = QCheckBox("Train ALL algorithms from grid.yaml")
        self.bulk_personas = QComboBox()
        form.addRow(self._lbl("Game"), self.bulk_game)
        form.addRow(self._lbl("Algorithm"), self.bulk_algo)
        form.addRow(self._lbl(" "), self.bulk_all_algos)
        form.addRow(self._lbl("Persona subset (from grid.yaml)"), self.bulk_personas)
        run_btn = QPushButton("Start Bulk Training (Novice & Expert)"); run_btn.clicked.connect(self._on_bulk_clicked)
        lay.addLayout(form); lay.addWidget(run_btn); lay.addStretch(1)
        return w

    def _on_bulk_game_changed(self):
        self._populate_personas_for_game(self.bulk_personas, self.bulk_game.currentText())

    def _on_bulk_clicked(self):
        game = self.bulk_game.currentText()
        if not game:
            return
        algos = get_available_algos_from_grid()
        if not algos:
            QMessageBox.information(self, "Missing", "No algorithms found from grid.yaml")
            return
        selected_algos = algos if self.bulk_all_algos.isChecked() else [self.bulk_algo.currentText()]
        persona = self.bulk_personas.currentText()
        skills = ["Novice", "Expert"]

        def run_all():
            total = 0
            for algo in selected_algos:
                for sk in skills:
                    cmd = [
                        sys.executable, "-m", "code.scripts.train",
                        f"game={game}", f"model={algo}", f"persona={persona}", f"skill={sk}", f"tb_root={DEFAULT_TB_ROOT}",
                    ]
                    self._append_cmd(cmd)
                    p = subprocess.run(cmd)
                    self.log.append(f"[{algo}/{sk}] exit {p.returncode}")
                    total += 1
            if HAS_WINSOUND:
                try: winsound.MessageBeep()
                except Exception: pass
            self.log.append(f"[✓] Completed training for {total} run(s) in game '{game}'.")
        Thread(target=run_all, daemon=True).start()

    # ---------------- Status Tab ----------------
    def _build_status_tab(self) -> QWidget:
        w = QWidget(); root = QVBoxLayout(w)
        overview = QGroupBox("Overview")
        og = QGridLayoutLike()
        self.ov_games = QLabel(""); self._bold_label_left(og, "Games", self.ov_games)
        self.ov_algos = QLabel(""); self._bold_label_left(og, "Algorithms", self.ov_algos)
        self.ov_trained = QLabel(""); self._bold_label_left(og, "Trained 'best' folders", self.ov_trained)
        overview.setLayout(og.layout)

        games_box = QGroupBox("Games status")
        self.games_table = QTableWidget(0, 2)
        self.games_table.setHorizontalHeaderLabels(["Game", "Status"])
        self.games_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.games_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.games_table.verticalHeader().setVisible(False)
        self.games_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        v1 = QVBoxLayout(); v1.addWidget(self.games_table); games_box.setLayout(v1)

        alg_box = QGroupBox("Algorithms (from grid.yaml)")
        self.alg_table = QTableWidget(0, 1)
        self.alg_table.setHorizontalHeaderLabels(["Algorithm"])
        self.alg_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.alg_table.verticalHeader().setVisible(False)
        self.alg_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        v2 = QVBoxLayout(); v2.addWidget(self.alg_table); alg_box.setLayout(v2)

        refresh = QPushButton("Refresh Status"); refresh.clicked.connect(self._refresh_status)

        root.addWidget(overview); root.addWidget(games_box); root.addWidget(alg_box); root.addWidget(refresh); root.addStretch(1)
        return w

    def _bold_label_left(self, grid_like, text: str, value_label: QLabel):
        lbl = QLabel(text)
        f = QFont(); f.setBold(True); lbl.setFont(f)
        grid_like.add_row(lbl, value_label)

    def _refresh_status(self):
        games = get_available_games()
        algos = get_available_algos_from_grid()
        trained = get_trained_games_from_models_flat()
        trained_models = get_trained_models_count()
        self.ov_games.setText(str(len(games)))
        self.ov_algos.setText(str(len(algos)))
        self.ov_trained.setText(str(trained_models))

        self.games_table.setRowCount(0)
        for g in games:
            row = self.games_table.rowCount(); self.games_table.insertRow(row)
            self.games_table.setItem(row, 0, QTableWidgetItem(g))
            status = "✓ Trained" if g in trained else "○ Not trained"
            self.games_table.setItem(row, 1, QTableWidgetItem(status))

        self.alg_table.setRowCount(0)
        for a in algos:
            row = self.alg_table.rowCount(); self.alg_table.insertRow(row)
            self.alg_table.setItem(row, 0, QTableWidgetItem(a))

    # ---------------- Maintenance Tab ----------------
    def _build_maint_tab(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)
        warn = QLabel("Select what to delete. This is permanent.")
        self.chk_del_logs = QCheckBox("Delete logs (mylogs/)")
        self.chk_del_models = QCheckBox("Delete models (models/)")
        del_btn = QPushButton("Delete Selected"); del_btn.clicked.connect(self._on_delete_clicked)
        lay.addWidget(warn); lay.addWidget(self.chk_del_logs); lay.addWidget(self.chk_del_models)
        lay.addStretch(1); lay.addWidget(del_btn)
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
        reply = QMessageBox.question(self, "Confirm", f"Permanently delete the following?\n{text}", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
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
    def add_row(self, left: QWidget, right: QWidget):
        self.layout.addWidget(left, self._row, 0, alignment=Qt.AlignLeft)
        self.layout.addWidget(right, self._row, 1, alignment=Qt.AlignLeft)
        self._row += 1

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = RLManagerGUI()
    gui.show()
    sys.exit(app.exec_())
