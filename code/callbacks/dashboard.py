import os
import csv
import numpy as np
import collections
from stable_baselines3.common.callbacks import BaseCallback

class DashboardCallback(BaseCallback):
    """
    The 'Black Box' Dashboard (Telemetry Edition).
    - Logs: Step, Event, Reward, Cause, Action, Level, Pos(X,Y), Vel(X,Y), GoalDist.
    - Features: Stall Detection, Trend Line, Event Log.
    - Performance: Lazy Rendering (high fps) + Event-driven Updates.
    """
    def __init__(self, update_freq: int = 1000, log_file: str = "training_events.csv", 
                 show_event_log: bool = True, show_detailed_stats: bool = True, verbose: int = 0):
        super().__init__(verbose)
        self.update_freq = update_freq
        self.log_file = log_file
        self.show_event_log = show_event_log
        self.show_detailed_stats = show_detailed_stats
        self.active = True
        
        # Data Buffers
        self.reward_buffer = collections.deque(maxlen=1000)
        self.trend_buffer = collections.deque(maxlen=1000)
        self.event_log = collections.deque(maxlen=8)
        
        # UI Objects
        self.fig = None
        self.ax_graph = None
        self.ax_bars = None
        self.ax_text = None
        self.bar_container = None
        self.line_reward = None
        self.line_trend = None
        self.text_display = None
        
        # Action Map
        self.action_map = {
            0: "IDLE", 1: "LEFT", 2: "RIGHT", 3: "JUMP",
            4: "RIGHT+JUMP", 5: "RUN+RIGHT", 6: "LEFT+JUMP", 
            7: "RUN+RIGHT+JUMP", 8: "RUN+LEFT", 9: "RUN+LEFT+JUMP"
        }
        
        # Labels
        self.phys_labels = ["Px", "Py", "Vx", "Vy", "Ground"]
        self.obs_labels = ["E-Dist", "C-Dist", "G-Dist", "E-Cnt", "C-Cnt", "Score", "Time", "Lives"]
        self.rew_map = {
            "movement": "Move", "coins": "Coin", "kills": "Kill", 
            "win": "WIN", "time": "Time", "death": "Death"
        }
        self.rew_keys = list(self.rew_map.keys())
        self.rew_labels = list(self.rew_map.values())
        self.bar_labels = self.obs_labels + self.rew_labels

    def _init_callback(self) -> None:
        # 1. Initialize CSV (Clean Format, No Timestamp)
        try:
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                # Exact columns requested
                writer.writerow([
                    "Step", "Event", "Reward", "Cause", 
                    "Action", "Level", "X", "Y", "Vx", "Vy", "Goal_Dist"
                ])
            if self.verbose > 0: print(f"[Dashboard] Logging telemetry to {self.log_file}")
        except Exception as e:
            print(f"[Dashboard] CSV Error: {e}")

        # 2. Check Headless
        if os.name == 'posix' and "DISPLAY" not in os.environ:
            self.active = False
            return

        # 3. Setup UI
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
            try: plt.style.use('dark_background')
            except: pass 

            plt.ion()
            self.fig = plt.figure(figsize=(12, 7), constrained_layout=True)
            self.fig.canvas.manager.set_window_title("PEAK Training Telemetry")
            
            gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1])
            
            # Graph
            self.ax_graph = self.fig.add_subplot(gs[0, 0])
            self.ax_graph.set_title("Reward Trend (White=Avg)")
            self.ax_graph.set_ylabel("Reward")
            self.ax_graph.grid(True, linestyle=':', alpha=0.4, color='#444444')
            self.ax_graph.set_facecolor('#111111')
            self.line_reward, = self.ax_graph.plot([], [], color='#00ff00', linewidth=0.8, alpha=0.6)
            self.line_trend, = self.ax_graph.plot([], [], color='#ffffff', linewidth=1.5)

            # Bars
            self.ax_bars = self.fig.add_subplot(gs[1, 0])
            self.ax_bars.set_title("Live Inputs vs Rewards")
            self.ax_bars.grid(axis='y', linestyle=':', alpha=0.4, color='#444444')
            x_pos = range(len(self.bar_labels))
            self.bar_container = self.ax_bars.bar(x_pos, [0]*len(self.bar_labels), color='cyan')
            self.ax_bars.set_xticks(x_pos)
            self.ax_bars.set_xticklabels(self.bar_labels, rotation=45, ha='right', fontsize=9)
            
            colors = ['#00ffff'] * len(self.obs_labels)
            colors += ['#3498db', '#f1c40f', '#e74c3c', '#2ecc71', '#9b59b6', '#c0392b']
            for bar, c in zip(self.bar_container, colors):
                bar.set_color(c)

            # Sidebar
            self.ax_text = self.fig.add_subplot(gs[:, 1])
            self.ax_text.axis('off')
            self.text_display = self.ax_text.text(0.05, 0.98, "SYSTEM ONLINE...", 
                                                  transform=self.ax_text.transAxes,
                                                  verticalalignment='top',
                                                  fontfamily='monospace', fontsize=9, color='#eeeeee')
            
        except Exception as e:
            print(f"[Dashboard] UI Init failed: {e}. Running Log-Only.")
            self.fig = None

    def _identify_cause(self, rew_dict, total_reward):
        if not rew_dict: return "Unknown"
        max_key = max(rew_dict, key=lambda k: abs(rew_dict[k]))
        if abs(rew_dict[max_key]) < 0.1: return "Noise"
        return self.rew_map.get(max_key, max_key)

    def _log_to_csv(self, event_type, reward, cause, info):
        """
        Writes expanded telemetry to CSV.
        """
        try:
            # Extract extended info
            act_idx = info.get("action", 0)
            action_name = self.action_map.get(act_idx, str(act_idx))
            
            # Fetch new metrics from Core
            level = info.get("level", 0)
            x_pos = info.get("x_position", 0.0)
            y_pos = info.get("y_position", 0.0)
            vx = info.get("velocity_x", 0.0)
            vy = info.get("velocity_y", 0.0)
            goal_dist = info.get("goal_dist", 0.0)

            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.n_calls,
                    event_type,
                    f"{reward:.4f}",
                    cause,
                    action_name,
                    level,
                    f"{x_pos:.1f}",
                    f"{y_pos:.1f}",
                    f"{vx:.1f}",
                    f"{vy:.1f}",
                    f"{goal_dist:.1f}"
                ])
        except:
            pass

    def _on_step(self) -> bool:
        if not self.active: return True

        obs = self.locals.get("new_obs")
        rewards = self.locals.get("rewards")
        infos = self.locals.get("infos")
        
        if obs is None or rewards is None: return True

        current_obs = obs[0] 
        current_reward = float(rewards[0])
        current_info = infos[0] if infos else {}
        rew_dict = current_info.get("reward_components", {})
        
        # Buffer
        self.reward_buffer.append(current_reward)
        recent = list(self.reward_buffer)[-50:]
        trend_val = sum(recent) / len(recent) if recent else 0
        self.trend_buffer.append(trend_val)

        # --- EVENT DETECTION ---
        event_str = None
        cause_str = ""
        
        is_win = current_info.get("won", False)
        is_death = current_info.get("terminated", False) and not is_win
        is_stalled = current_info.get("stalled", False) # <--- NEW CHECK
        
        is_big_gain = (current_reward > 2.0) and not is_win
        is_big_loss = (current_reward < -2.0) and not is_death and not is_stalled

        if is_win:
            event_str = "🏆 WIN"
            cause_str = "Goal"
        elif is_death:
            event_str = "💀 DIED"
            cause_str = "Hazard"
        elif is_stalled:
            event_str = "⚠️ STALL" # <--- Explicit Event for Stall
            cause_str = "Time/Camp"
        elif is_big_gain:
            event_str = "🚀 GAIN"
            cause_str = self._identify_cause(rew_dict, current_reward)
        elif is_big_loss:
            event_str = "📉 LOSS"
            cause_str = self._identify_cause(rew_dict, current_reward)

        # Log to CSV
        if event_str:
            self._log_to_csv(event_str.split()[-1], current_reward, cause_str, current_info)
            if self.show_event_log:
                log_msg = f"{self.n_calls}: {event_str} ({current_reward:+.1f}) [{cause_str}]"
                self.event_log.append(log_msg)

        # --- RENDER LOGIC ---
        is_critical = is_win or is_death # Stalls don't force render, just log
        is_periodic = (self.n_calls % self.update_freq == 0)
        should_render = is_periodic or is_critical
        
        if self.fig is None or not should_render:
            return True
            
        import matplotlib.pyplot as plt
        if not plt.fignum_exists(self.fig.number):
            self.fig = None 
            return True

        try:
            # Prepare Data
            if current_obs.shape[0] >= 310:
                phys_vals = list(current_obs[:5])
                obs_vals = list(current_obs[-8:])
            else:
                phys_vals = [0]*5
                obs_vals = [0]*8
            rew_vals = [float(rew_dict.get(k, 0.0)) for k in self.rew_keys]

            # Update Text
            lines = []
            if self.show_event_log:
                lines.append("== EVENT LOG ==")
                if not self.event_log: lines.append(" (Waiting...)")
                lines.extend(list(self.event_log))
            
            if self.show_detailed_stats:
                lines.append("\n== TRENDS ==")
                lines.append(f"Reward  : {current_reward:+.3f}")
                lines.append(f"Trend   : {trend_val:+.3f}")
                
                act_idx = current_info.get("action", 0)
                act_name = self.action_map.get(act_idx, "UNK")
                lines.append(f"Action  : {act_name}")
                lines.append(f"Level   : {current_info.get('level', '?')}") # Show Level

                lines.append("\n== REWARDS ==")
                for lbl, val in zip(self.rew_labels, rew_vals):
                    mark = " <<" if abs(val) > 0.001 else ""
                    lines.append(f"{lbl:<6}: {val:+.2f}{mark}")

            self.text_display.set_text("\n".join(lines))

            # Update Graphs
            self.line_reward.set_data(range(len(self.reward_buffer)), self.reward_buffer)
            self.line_trend.set_data(range(len(self.trend_buffer)), self.trend_buffer)
            self.ax_graph.set_xlim(0, max(300, len(self.reward_buffer)))
            
            r_data = list(self.reward_buffer)
            if r_data:
                y_min, y_max = min(r_data), max(r_data)
                pad = (y_max - y_min) * 0.1 if y_max != y_min else 1.0
                self.ax_graph.set_ylim(y_min - pad, y_max + pad)

            # Update Bars
            all_vals = obs_vals + rew_vals
            for rect, val in zip(self.bar_container, all_vals):
                rect.set_height(val)
            
            bar_data = [v for v in all_vals if abs(v) > 0.001]
            if bar_data:
                b_min, b_max = min(0, min(bar_data)), max(0, max(bar_data))
                self.ax_bars.set_ylim(b_min * 1.1, b_max * 1.1)

            self.fig.canvas.flush_events()
            
        except Exception:
            pass

        return True