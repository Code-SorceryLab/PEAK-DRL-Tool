import os
import numpy as np
import warnings
from stable_baselines3.common.callbacks import BaseCallback

class DashboardCallback(BaseCallback):
    """
    A live 'Pop-Up' dashboard to visualize what the agent actually sees.
    Robust to window closing and headless environments.
    """
    def __init__(self, update_freq: int = 1, verbose: int = 0):
        super().__init__(verbose)
        self.update_freq = update_freq
        self.active = True
        self.fig = None
        self.axs = None
        self.bar_container = None
        self.line_reward = None
        self.reward_buffer = []
        
        # --- CONFIG: CHART LABELS ---
        # 1. Input State (Indices 0-7 from obs vector)
        self.obs_labels = ["Enm Dist", "Coin Dist", "Goal Dist", "Enm Count", 
                           "Coin Count", "Score", "Time", "Lives"]
        
        # 2. Reward Breakdown Configuration
        # Keys match the dictionary in platformer.py
        self.rew_map = {
            "movement": "R:Move",
            "coins":    "R:Coin",
            "kills":    "R:Kill", 
            "win":      "R:Win",
            "time":     "R:Time",
            "death":    "R:Death"
        }
        self.rew_keys = list(self.rew_map.keys())
        self.rew_labels = list(self.rew_map.values())
        
        self.all_labels = self.obs_labels + self.rew_labels

    def _init_callback(self) -> None:
        # 1. Safety Check for Headless Mode
        if os.name == 'posix' and "DISPLAY" not in os.environ:
            if self.verbose > 0: 
                print("[Dashboard] ⚠️ Headless environment (No DISPLAY). UI disabled.")
            self.active = False
            return

        # 2. Try to initialize Matplotlib
        try:
            import matplotlib.pyplot as plt
            plt.ion() # Interactive mode ON
            
            # Setup Window
            self.fig, self.axs = plt.subplots(2, 1, figsize=(8, 9))
            self.fig.canvas.manager.set_window_title("PEAK Training Inspector")
            
            # --- Plot 1: Reward History (The Pulse) ---
            self.axs[0].set_title("Total Reward History")
            self.axs[0].set_ylabel("Reward")
            self.axs[0].grid(True, linestyle='--', alpha=0.3)
            self.axs[0].set_facecolor('#111111') 
            self.line_reward, = self.axs[0].plot([], [], color='#00ff00', linewidth=1.5)

            # --- Plot 2: State & Reward Breakdown (The Details) ---
            self.axs[1].set_title("Live Tracking: Inputs (Cyan) vs Rewards (Color)")
            # No fixed Y-limit because 'Win' reward is +10.0
            self.axs[1].grid(axis='y', linestyle='--', alpha=0.3)
            
            # Create Bars
            x_pos = range(len(self.all_labels))
            self.bar_container = self.axs[1].bar(x_pos, [0]*len(self.all_labels), color='cyan')
            
            # Set labels
            self.axs[1].set_xticks(x_pos)
            self.axs[1].set_xticklabels(self.all_labels, rotation=45, ha='right')
            
            # --- COLOR CODING ---
            # Inputs = Cyan
            # Rewards = Custom
            colors = ['cyan'] * len(self.obs_labels)
            # Order matches self.rew_keys: movement, coins, kills, win, time, death
            colors += [
                '#3498db', # Blue (Move)
                '#f1c40f', # Gold (Coin)
                '#e74c3c', # Red (Kill)
                '#2ecc71', # Green (Win)
                '#9b59b6', # Purple (Time)
                '#34495e'  # Dark Blue (Death)
            ]
            
            for bar, c in zip(self.bar_container, colors):
                bar.set_color(c)

            plt.tight_layout()
            
        except Exception as e:
            print(f"[Dashboard] ⚠️ Init failed: {e}. Training continues.")
            self.active = False

    def _on_step(self) -> bool:
        if not self.active: return True
        if self.n_calls % self.update_freq != 0: return True

        # Check for user close
        import matplotlib.pyplot as plt
        if self.fig is None or not plt.fignum_exists(self.fig.number):
            if self.verbose > 0: print("[Dashboard] Closed by user.")
            self.active = False
            return True

        # 1. Fetch Data
        obs = self.locals.get("new_obs")      
        rewards = self.locals.get("rewards") 
        infos = self.locals.get("infos")
        
        if obs is None or rewards is None: return True

        # Use first env data
        current_obs = obs[0] 
        current_reward = float(rewards[0])
        current_info = infos[0] if infos else {}

        # 2. Prepare Bar Data
        # A. Obs Data (Last 8)
        if current_obs.shape[0] >= 310:
            obs_vals = list(current_obs[-8:])
        else:
            obs_vals = [0]*8
            
        # B. Reward Components (From our dictionary)
        rew_dict = current_info.get("reward_components", {})
        # Extract based on keys order
        rew_vals = [float(rew_dict.get(k, 0.0)) for k in self.rew_keys]
        
        # Combine
        all_vals = obs_vals + rew_vals

        # 3. Update Plots
        try:
            # A. Update Line Graph
            self.reward_buffer.append(current_reward)
            if len(self.reward_buffer) > 200: self.reward_buffer.pop(0)
            
            self.line_reward.set_data(range(len(self.reward_buffer)), self.reward_buffer)
            self.axs[0].set_xlim(0, max(200, len(self.reward_buffer)))
            
            # Smart Scale
            r_min, r_max = min(self.reward_buffer), max(self.reward_buffer)
            margin = 0.5 if r_max == r_min else (r_max - r_min) * 0.1
            self.axs[0].set_ylim(r_min - margin, r_max + margin)

            # B. Update Bars
            max_val = 1.0
            min_val = 0.0
            for rect, val in zip(self.bar_container, all_vals):
                rect.set_height(val)
                if val > max_val: max_val = val
                if val < min_val: min_val = val
            
            # Auto-scale Y axis (handle negative death penalties)
            self.axs[1].set_ylim(min_val * 1.1, max_val * 1.1)

            # C. Render
            self.fig.canvas.flush_events()
            
        except Exception:
            self.active = False

        return True