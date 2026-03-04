import pandas as pd
import glob
import os
import numpy as np

# ── Columns that are NOT reward components ─────────────────────────────────
STANDARD_COLS = {
    'step', 'total_reward', 'action', 'level', 'levels_completed',
    'x', 'y', 'vx', 'vy', 'goal_dist', 'event', 'cause'
}

# ── Obs-sanity columns (also not reward components) ────────────────────────
OBS_SANITY_COLS = {
    'grid_player_mean','grid_player_std','grid_player_min','grid_player_max',
    'grid_hazard_mean','grid_hazard_std','grid_hazard_min','grid_hazard_max',
    'grid_collectible_mean','grid_collectible_std','grid_collectible_min','grid_collectible_max',
    'grid_dijkstra_mean','grid_dijkstra_std','grid_dijkstra_min','grid_dijkstra_max',
    'grid_solid_mean','grid_solid_std','grid_solid_min','grid_solid_max',
    'scalar_mean','scalar_std','scalar_min','scalar_max','dijkstra_val','obs_warnings',
}


def get_all_log_files():
    """Return all CSV log files found in csv/ (plus legacy locations)."""
    files = glob.glob(os.path.join("csv", "training_log*.csv"))
    if not files:
        for root_dir in [".", "mylogs", "runs"]:
            files.extend(glob.glob(os.path.join(root_dir, "**", "training_log*.csv"), recursive=True))
    return sorted(files, key=os.path.getmtime)


def _bar(value, max_val, width=20, char="█"):
    filled = int((value / max(max_val, 1)) * width)
    return char * filled + "░" * (width - filled)


def _load_csv_safe(filepath, max_rows=500_000):
    """Load a CSV with chunked reading and tail-sampling to cap memory."""
    chunks = []
    total_rows = 0
    for chunk in pd.read_csv(filepath, low_memory=False, chunksize=200_000):
        chunks.append(chunk)
        total_rows += len(chunk)
    if not chunks:
        return None, 0
    full = pd.concat(chunks, ignore_index=True)
    if total_rows > max_rows:
        early_n = max_rows // 5
        late_n  = max_rows - early_n
        sampled = pd.concat([full.iloc[:early_n], full.iloc[-late_n:]], ignore_index=True)
        return sampled, total_rows
    return full, total_rows


# ══════════════════════════════════════════════════════════════════════════════
# Per-persona analysis
# ══════════════════════════════════════════════════════════════════════════════
def _analyze_single(df, persona_name):
    """Analyze a single persona's data and print a compact report."""
    W = 58
    print(f"\n{'━'*W}")
    print(f"  📋  PERSONA: {persona_name.upper()}")
    print(f"{'━'*W}")

    if df.empty:
        print("  ⚠  No data.")
        return {"persona": persona_name, "win_rate": 0, "steps": 0}

    reward_cols = [c for c in df.columns
                   if c.lower() not in STANDARD_COLS
                   and c.lower() not in OBS_SANITY_COLS
                   and "unnamed" not in c.lower()]

    total_steps    = len(df)
    events         = df[df['event'].notna() & (df['event'] != "")]
    total_episodes = len(events)
    wins           = int((events['event'] == 'WIN').sum())
    deaths         = int((events['event'] == 'DIED').sum())
    win_rate       = (wins / total_episodes * 100) if total_episodes > 0 else 0.0

    q = max(1, total_steps // 4)
    quarters = [df.iloc[i*q:(i+1)*q] for i in range(4)]
    q_rewards = [float(qt['total_reward'].mean()) if 'total_reward' in df.columns and len(qt) > 0 else 0.0
                 for qt in quarters]

    eq = max(1, total_episodes // 4)
    event_quarters = [events.iloc[i*eq:(i+1)*eq] for i in range(4)] if total_episodes >= 4 else []
    q_wins = [int((eq_['event'] == 'WIN').sum()) for eq_ in event_quarters] if event_quarters else [0]*4

    avg_ep_len = 0.0
    if 'step' in df.columns and total_episodes > 0:
        event_steps = events['step'].values
        episode_lengths = np.diff(np.concatenate([[0], event_steps]))
        avg_ep_len = float(episode_lengths.mean())

    death_rows = events[events['event'] == 'DIED']
    causes = death_rows['cause'].value_counts() if ('cause' in death_rows.columns and deaths > 0) else pd.Series()

    levels_completed = int(df['levels_completed'].max()) if 'levels_completed' in df.columns else 0

    actions = df['action'].value_counts(normalize=True) * 100 if 'action' in df.columns else pd.Series()
    idle_pct  = float(actions.get('IDLE', 0.0))
    left_pct  = sum(float(actions.get(a, 0.0)) for a in ['LEFT', 'LEFT+JUMP', 'RUN+LEFT', 'RUN+LEFT+JUMP'])
    right_pct = sum(float(actions.get(a, 0.0)) for a in ['RIGHT', 'RIGHT+JUMP', 'RUN+RIGHT', 'RUN+RIGHT+JUMP'])
    jump_pct  = float(actions.get('JUMP', 0.0))

    max_x      = float(df['x'].max()) if 'x' in df.columns else 0.0
    avg_goal_d = float(df['goal_dist'].mean()) if 'goal_dist' in df.columns else 0.0

    component_sums = {}
    component_means = {}
    if reward_cols:
        for col in reward_cols:
            vals = pd.to_numeric(df[col], errors='coerce').fillna(0)
            component_sums[col]  = float(vals.sum())
            component_means[col] = float(vals.mean())
        total_component_sum = sum(abs(v) for v in component_sums.values()) or 1.0

    # ── Print ─────────────────────────────────────────────────────────────
    print(f"  ⏱  Steps: {total_steps:,}  |  Episodes: {total_episodes:,}  |  Avg Ep: {avg_ep_len:,.0f}")
    print(f"  🏆 Win Rate: {win_rate:.1f}%  ({wins} W / {deaths} D)  |  Levels: {levels_completed}")
    print(f"  📏 Max X: {max_x:.1f}  |  Avg Goal Dist: {avg_goal_d:.1f}")

    if avg_ep_len < 100 and total_episodes > 10:
        print("  ⚠  Very short episodes — dying almost immediately.")

    if len(causes) > 0:
        cause_strs = [f"{c} {n}x ({n/deaths*100:.0f}%)" for c, n in causes.items()]
        print(f"  💀  {' | '.join(cause_strs)}")

    print(f"  🎮  R={right_pct:.0f}% L={left_pct:.0f}% J={jump_pct:.0f}% Idle={idle_pct:.0f}%")
    if idle_pct > 15:
        print(f"  ⚠  High IDLE ({idle_pct:.1f}%)")

    if reward_cols:
        print(f"  🧬  Reward breakdown:")
        for col in reward_cols:
            pct = abs(component_sums[col]) / total_component_sum * 100
            bar = _bar(pct, 100, 12)
            print(f"     {col:<14} {bar} {pct:>5.1f}%  (mean: {component_means[col]:+.5f})")
        for col in reward_cols:
            pct = abs(component_sums[col]) / total_component_sum * 100
            if pct > 60:
                print(f"  ⚠  '{col}' dominates at {pct:.0f}% — other signals drowned out.")

    print(f"  📈  Q1→Q4 reward: {' → '.join(f'{r:+.4f}' for r in q_rewards)}")
    print(f"  📈  Q1→Q4 wins:   {' → '.join(f'{w:>4}' for w in q_wins)}")

    return {
        "persona": persona_name, "steps": total_steps, "episodes": total_episodes,
        "win_rate": win_rate, "wins": wins, "deaths": deaths,
        "levels_completed": levels_completed, "avg_ep_len": avg_ep_len,
        "max_x": max_x, "idle_pct": idle_pct, "right_pct": right_pct,
        "q_rewards": q_rewards, "q_wins": q_wins, "component_means": component_means,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Cross-persona comparison
# ══════════════════════════════════════════════════════════════════════════════
def _print_comparison(summaries):
    W = 58
    print(f"\n{'═'*W}")
    print("  📊  CROSS-PERSONA COMPARISON")
    print(f"{'═'*W}")

    header = f"  {'Persona':<20} {'Win%':>6} {'Lvls':>5} {'AvgEp':>7} {'MaxX':>7} {'Idle%':>6}"
    print(header)
    print(f"  {'─'*54}")

    for name, s in sorted(summaries.items(), key=lambda x: -x[1].get('win_rate', 0)):
        print(f"  {name:<20} {s.get('win_rate',0):>5.1f}% {s.get('levels_completed',0):>5}"
              f" {s.get('avg_ep_len',0):>7.0f} {s.get('max_x',0):>7.1f} {s.get('idle_pct',0):>5.1f}%")

    best  = max(summaries.items(), key=lambda x: x[1].get('win_rate', 0))
    worst = min(summaries.items(), key=lambda x: x[1].get('win_rate', 0))
    if best[0] != worst[0]:
        print(f"\n  🏆 Best:  {best[0]} ({best[1].get('win_rate',0):.1f}%)")
        print(f"  💀 Worst: {worst[0]} ({worst[1].get('win_rate',0):.1f}%)")

    print(f"\n{'═'*W}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def analyze_agent():
    print("\n=== Agent Performance Analyzer ===")
    all_files = get_all_log_files()
    if not all_files:
        print("❌ No training logs found. Run training first.")
        return

    print(f"📂 Found {len(all_files)} CSV log(s):")

    per_persona = {}
    for f in all_files:
        basename = os.path.basename(f)
        parts = basename.replace("training_log_", "").replace(".csv", "").split("_", 1)
        persona_name = parts[1] if len(parts) > 1 else basename
        print(f"   • {f}", end="")
        try:
            df, total_rows = _load_csv_safe(f)
            if df is None or df.empty:
                print("  ⚠  Empty")
                continue
            sampled_note = f" (sampled to {len(df):,})" if len(df) < total_rows else ""
            print(f"  {total_rows:,} rows{sampled_note}")
            per_persona[persona_name] = df
        except Exception as e:
            print(f"  ⚠  Skipped: {e}")

    if not per_persona:
        print("❌ All files failed to load.")
        return

    print(f"\n{'═'*58}")
    print(f"  🤖  PEAK AGENT PERFORMANCE REPORT  ({len(per_persona)} persona(s))")
    print(f"{'═'*58}")

    summaries = {}
    for name, df in per_persona.items():
        summaries[name] = _analyze_single(df, name)

    if len(summaries) > 1:
        _print_comparison(summaries)

    print("Analysis complete.")


if __name__ == "__main__":
    analyze_agent()