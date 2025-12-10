A comprehensive Deep Reinforcement Learning framework for training and evaluating agents in FlappyBird with various reward shaping strategies. This project focuses on analyzing agent behavior balance using configurable personas, skill levels, and metrics.

## 🎯 Key Features

- **FlappyBird Environment**: Focused implementation with multiple reward shaping strategies
- **Configurable Personas**: Different reward shaping strategies (baseline, speedrunner, master, shaped, simple)
- **Skill Levels**: Novice (100k steps) to Expert (500k steps)
- **Interactive Menu System**: Easy-to-use console interface for all operations
- **TensorBoard Integration**: Real-time training monitoring and visualization
- **Hydra Configuration**: Flexible YAML-based configuration management
- **Model Management**: Automatic saving, loading, and organization with best model tracking
- **Comprehensive Metrics**: Built-in evaluation and analysis tools

## 📁 Repository Structure

```
drl-agents-balance/
├── .vscode/                    # VS Code settings and configurations
├── code/                       # Main Python package
│   ├── __init__.py            # Package initialization
│   ├── conf/                  # Hydra configuration files
│   │   ├── algo/              # Algorithm configs (PPO, DQN, etc.)
│   │   │   └── ppo.yaml
│   │   ├── callback/          # Training callback configs
│   │   │   ├── none.yaml
│   │   │   └── render.yaml
│   │   ├── game/              # Game environment configs
│   │   │   └── flappy.yaml
│   │   ├── reward/            # Persona/reward shaping configs
│   │   │   ├── flappy_baseline.yaml
│   │   │   ├── flappy_master.yaml
│   │   │   ├── flappy_shaped.yaml
│   │   │   ├── flappy_simple.yaml
│   │   │   └── flappy_speedrunner.yaml
│   │   ├── space/             # Observation/action space configs
│   │   │   └── flappy.yaml
│   │   └── grid.yaml          # Main grid configuration
│   ├── games/                 # Game implementations
│   │   ├── __init__.py
│   │   └── flappy_core.py     # FlappyBird game logic
│   ├── metrics/               # Evaluation and analysis metrics
│   │   ├── __init__.py
│   │   └── flappy_balance.py  # FlappyBird-specific metrics
│   ├── rewards/               # Reward shaping functions
│   │   ├── __init__.py
│   │   └── flappy_rewards.py  # FlappyBird reward functions
│   ├── scripts/               # Command-line entry points
│   │   ├── __init__.py
│   │   ├── analyze_metrics.py # Metrics analysis and plotting
│   │   ├── callbacks.py       # Training callbacks
│   │   ├── evaluate.py        # Model evaluation
│   │   └── train.py          # Training script with grid support
│   └── wrappers/              # Environment wrappers
│       ├── __init__.py
│       └── generic_env.py     # Generic environment wrapper
├── models/                    # Trained model storage
│   └── [game]/[algorithm]/    # Organized by game and algorithm
│       └── best/              # Best models per persona and skill
├── mylogs/                    # Training logs and TensorBoard data
│   └── [persona]_[skill]/     # Organized by persona and skill level
├── runs/                      # Alternative TensorBoard log directory
├── .gitignore                 # Git ignore patterns
├── README.md                  # This file
├── games.md                   # Game-specific documentation
├── menu.py                    # Interactive menu system
├── pyproject.toml            # Project configuration
└── tensorboard.md            # TensorBoard usage guide
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -U pip
pip install "gymnasium[all]" stable-baselines3 hydra-core pygame matplotlib
```

### 2. Using the Interactive Menu

The easiest way to use this framework is through the interactive menu system:

```bash
python menu.py
```

#### Menu Options Explained

When you run `menu.py`, you'll see a main menu with the following options:

**1. Train a New Agent**
- Starts training a new RL agent from scratch
- You'll be prompted to select:
  - **Game**: Currently FlappyBird
  - **Algorithm**: PPO (Proximal Policy Optimization) or other supported algorithms
  - **Persona**: Choose from baseline, speedrunner, master, shaped, or simple reward strategies
  - **Skill Level**: Select training duration (novice=100k, intermediate=250k, advanced=400k, expert=500k steps)
- The agent will train and save checkpoints to `models/[game]/[algorithm]/` with best models in the `best/` subfolder
- Training logs are saved to `mylogs/[persona]_[skill]/` for TensorBoard visualization

**2. Evaluate Existing Agent**
- Load a previously trained model and watch it play
- You'll be prompted to:
  - Select the game (FlappyBird)
  - Choose the algorithm used for training
  - Pick the specific model file from the saved models
  - Specify number of evaluation episodes
- The agent will play the game with rendering enabled so you can watch its performance
- Evaluation metrics (average score, success rate, etc.) are displayed at the end

**3. Continue Training Existing Agent**
- Resume training from a saved checkpoint
- Useful for:
  - Extending training duration (e.g., taking a novice model to intermediate level)
  - Fine-tuning an existing agent
  - Recovering from interrupted training
- You'll select the model file and specify additional training steps
- Progress is saved incrementally with best models preserved

**4. Analyze Metrics**
- Generate visualizations and statistics from training logs
- Creates plots showing:
  - Learning curves (reward over time)
  - Performance comparisons across personas
  - Balance metrics specific to each persona's objectives
- Outputs are saved as images in the metrics directory
- Useful for comparing different reward shaping strategies

**5. View TensorBoard Logs**
- Launches TensorBoard to visualize real-time training progress
- Shows detailed metrics including:
  - Episode rewards and lengths
  - Value function estimates
  - Policy loss and entropy
  - Learning rate schedules
- Access at http://localhost:6006 in your browser
- Logs are organized by persona and skill level for easy comparison

**6. Exit**
- Safely closes the menu system

#### Menu Navigation Tips

- Simply enter the number corresponding to your choice and press Enter
- Follow the prompts for each option—the menu will guide you through all required selections
- You can cancel most operations by pressing Ctrl+C
- All file paths and model names are displayed clearly so you know what you're working with
- The menu validates your inputs and provides helpful error messages if something goes wrong

### 3. Direct Command-Line Usage (Advanced)

If you prefer not to use the menu, you can run scripts directly:

```bash
# Train an agent
python -m code.scripts.train --game flappy --algo ppo --persona baseline --skill novice

# Evaluate an agent
python -m code.scripts.evaluate --model-path models/flappy/ppo/best/baseline_novice.zip

# Analyze metrics
python -m code.scripts.analyze_metrics --persona baseline --skill novice
```

## 🎮 Personas & Reward Shaping

Each persona uses a different reward shaping strategy to encourage specific behaviors:

### Baseline
- Minimal reward shaping
- Focuses on core game objectives (staying alive, avoiding obstacles)
- Best for understanding fundamental game mechanics

### Speedrunner
- Rewards fast progression through the game
- Optimized for high-speed gameplay
- May take more risks for speed

### Master
- Balanced reward function optimizing multiple objectives
- Aims for consistent, high-quality performance
- Considers both safety and efficiency

### Shaped
- Heavily shaped rewards to guide learning
- Provides more frequent feedback signals
- Can accelerate early learning but may learn suboptimal policies

### Simple
- Minimalist reward structure
- Only essential rewards for survival
- Tests agent's ability to learn from sparse feedback

## 📊 Skill Levels

- **Novice**: 100,000 training steps - Basic competency
- **Intermediate**: 250,000 training steps - Improved performance
- **Advanced**: 400,000 training steps - Strong performance
- **Expert**: 500,000 training steps - Near-optimal play

## 🔧 Configuration

All configurations are managed through YAML files in `code/conf/`:

- **algo/**: Algorithm hyperparameters (learning rate, batch size, etc.)
- **game/**: Environment settings (frame skip, rendering options)
- **reward/**: Persona-specific reward functions and weights
- **space/**: Observation and action space definitions
- **callback/**: Training callbacks (checkpointing, rendering)

## 📈 Monitoring Training

Use TensorBoard to monitor training in real-time:

```bash
# Option 1: Through the menu
python menu.py  # Select option 5

# Option 2: Direct command
tensorboard --logdir=mylogs
```

Then open http://localhost:6006 in your browser.

## 🧪 Evaluation & Analysis

After training, evaluate your agents:

```bash
# Through the menu
python menu.py  # Select option 2 for evaluation or option 4 for analysis

# Direct evaluation
python -m code.scripts.evaluate --model-path models/flappy/ppo/best/master_expert.zip --episodes 100

# Generate analysis plots
python -m code.scripts.analyze_metrics --persona master --skill expert
```

## 📦 Model Organization

Models are automatically organized as follows:

```
models/
└── flappy/
    └── ppo/
        ├── baseline_novice.zip
        ├── speedrunner_intermediate.zip
        ├── master_expert.zip
        └── best/              # Best performing models
            ├── baseline_novice.zip
            ├── speedrunner_intermediate.zip
            └── master_expert.zip
```

The `best/` folder contains the top-performing models for each persona-skill combination, automatically saved during training when performance improves.

## 🛠️ Development

### Adding New Personas

1. Create a new YAML config in `code/conf/reward/`
2. Define reward function in `code/rewards/flappy_rewards.py`
3. Update the menu options if needed

### Extending to New Games

1. Implement game logic in `code/games/[game]_core.py`
2. Create environment wrapper in `code/wrappers/`
3. Add configuration files in `code/conf/game/` and `code/conf/space/`
4. Define reward functions in `code/rewards/[game]_rewards.py`

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📚 Additional Documentation

- See `games.md` for detailed game-specific information
- See `tensorboard.md` for TensorBoard usage and tips
