# drl-PEAK-agents-balance: Gemini CLI Instructional Context

This `GEMINI.md` file provides an instructional context for the `drl-PEAK-agents-balance` project, generated to help the Gemini CLI agent understand and interact with the codebase effectively.

## Project Overview

`drl-PEAK-agents-balance` (PEAK: Platformer Engine by Al & Kevin) is a research-grade, high-performance reinforcement learning benchmarking engine. Developed for Ontario Tech University's Master's Program, its primary purpose is to evaluate deep RL agent adaptability, learning, and generalization in deterministic 2D platformer environments.

The project emphasizes:
- **Deterministic Physics Simulation:** Consistent and reproducible platformer mechanics.
- **Multi-Stage Level Progression:** Progressively challenging levels for skill development.
- **Gymnasium-Compatible Interface:** Seamless integration with modern RL frameworks like Stable-Baselines3.
- **Configurable Reward Functions (Personas):** Allows for diverse training objectives (e.g., Simple, Speedrunner, Coin Collector, Master).
- **High-Speed Training:** Achieves 1000+ environment steps per second through optimized collision detection and spatial hashing.
- **Hydra-based Configuration:** Manages experiments, parameters, and automatic experiment tracking via YAML files and command-line overrides.
- **User Interfaces:** Includes both a PyQt5-based Graphical User Interface (GUI) and a terminal-based menu system.

### Architecture Highlights:

The project's architecture is structured in five layers:
1.  **Physics Engine:** Implements SMB1-style platformer physics.
2.  **Spatial Hashing:** Optimizes collision detection for static and dynamic entities.
3.  **GameObject System:** Unified data structure for all game entities.
4.  **Observation & Reward:** Extracts game state into a 308-element vector and applies persona-based reward shaping.
5.  **Training Integration:** Gymnasium-compatible interface for RL training and evaluation.

## Building and Running

### Installation

To set up the project, clone the repository and install the Python dependencies:

```bash
# Clone the repository
git clone https://github.com/Code-SorceryLab/drl-PEAK-agents-balance.git
cd drl-PEAK-agents-balance

# Install dependencies
pip install -r requirements.txt
```

### Key Commands

-   **Run GUI:**
    ```bash
    python gui.py
    ```
    (Select game parameters, persona, and algorithm, then click Train.)

-   **Run Terminal Menu:**
    ```bash
    python menu.py
    ```
    (Follow interactive prompts for training, watching agents, etc.)

-   **Direct Training via Command Line:**
    ```bash
    python code/scripts/train.py game=platformer persona=simple algo=ppo total_timesteps=1000000
    ```

-   **Evaluate a Trained Model:**
    ```bash
    python code/scripts/evaluate.py --model models/platformer_simple_20250129_143022.zip --render human
    ```

-   **Manual Gameplay (for understanding the environment):**
    ```bash
    python code/scripts/manual_play.py
    ```
    (Use arrow keys to move, spacebar to jump. Press F1-F8 for debug overlays.)

-   **View TensorBoard Logs:**
    ```bash
    python -m tensorboard.main --logdir mylogs/
    ```

## Development Conventions

-   **Configuration:** The project uses Hydra for configuration management. All experiment settings are defined in YAML files located under `code/conf/`. Parameters can be overridden directly from the command line using the `key=value` syntax. For adding new top-level keys not present in the base config, use the `+key=value` syntax.
-   **Reward Personas:** New reward functions (personas) can be added by creating a function in `code/rewards/platformer.py`, decorating it with `@_wrap_with_tracker`, and returning a float reward value. A corresponding YAML configuration should be added in `code/conf/reward/`.
-   **Levels:** Game levels are defined using ASCII text files (`.txt`) located in `code/games/levels/`. New levels can be easily created by defining character-based layouts (e.g., `#` for ground, `^` for spikes, `G` for goal).
-   **Debugging:** The project includes a comprehensive debugging suite with real-time visualizations (hitbox overlays, agent observation views, reward trace graphs) accessible during manual play (`manual_play.py`) via keyboard toggles (F1-F8).

## Key Files and Directories

-   `README.md`: Comprehensive project documentation and quick start guide.
-   `requirements.txt`: Lists all Python package dependencies.
-   `menu.py`: The main interactive terminal-based menu for operating the project.
-   `gui.py`: The PyQt5-based graphical user interface.
-   `code/`: Contains the core source code.
    -   `code/scripts/train.py`: The main script for training RL agents.
    -   `code/scripts/evaluate.py`: Script for evaluating trained models.
    -   `code/scripts/manual_play.py`: Script for manual interaction with the game environment.
    -   `code/games/platformer_core.py`: Implements the core logic and physics of the platformer game.
    -   `code/conf/grid.yaml`: The primary Hydra configuration file defining experiment grids.
    -   `code/conf/algo/`: Directory containing YAML configurations for different RL algorithms (e.g., `ppo.yaml`).
    -   `code/conf/callback/`: Directory containing YAML configurations for callbacks during training.
    -   `code/rewards/train_platformer.py`: Contains the definitions for different reward personas.
-   `models/`: Stores trained models (`.zip` files), checkpoints, and evaluation logs.
-   `mylogs/`: Default directory for TensorBoard logs.
-   `videos/`: Directory where recorded gameplay sessions (MP4, GIF) are saved.
