import pandas as pd
import glob
import os
import numpy as np

# ── Columns that are NOT reward components ─────────────────────────────────
STANDARD_COLS = {
    'step', 'total_reward', 'action', 'level', 'levels_completed',
    'x', 'y', 'vx', 'vy', 'goal_dist', 'event', 'cause'
}

def get_latest_log_file():
    search_paths = [".", "mylogs", "runs"]
    files = []
    for root_dir in search_paths:
        found = glob.glob(os.path.join(root_dir, "**", "training_log*.csv"), recursive=True)
        files.extend(found)
    if not files: return None
    return max(files, key=os.path.getmtime)


def _trend(early_val, late_val, higher_is_better=True):
    if higher_is_better:
        return "✅ Improved" if late_val > early_val else ("➡  Flat" if abs(late_val - early_val) < 0.001 else "⚠  Declined")
    else:
        return "✅ Improved" if late_val < early_val else ("➡  Flat" if abs(late_val - early_val) < 0.001 else "⚠  Declined")


def _bar(value, max_val, width=20, char="█"):
    filled = int((value / max(max_val, 1)) * width)
    return char * filled + "░" * (width - filled)


def analyze_agent():
    log_file = get_latest_log_file()
    if not log_file:
        print("❌ No training logs found. Run training first.")
        return

    print(f"📂 Loading: {log_file}")
    try:
        df = pd.read_csv(log_file, low_memory=False)
    except Exception as e:
        print(f"❌ Failed to read CSV: {e}")
        return

    if df.empty:
        print("⚠  Log file is empty.")
        return

    # ── Detect reward component columns ──────────────────────────────────
    reward_cols = [c for c in df.columns
                   if c.lower() not in STANDARD_COLS and "unnamed" not in c.lower()]

    # ── Core counts ───────────────────────────────────────────────────────
    total_steps   = len(df)
    events        = df[df['event'].notna() & (df['event'] != "")]
    total_episodes = len(events)
    wins           = int((events['event'] == 'WIN').sum())
    deaths         = int((events['event'] == 'DIED').sum())
    win_rate       = (wins / total_episodes * 100) if total_episodes > 0 else 0.0

    # ── Quarter splits for richer trend analysis ─────────────────────────
    q = total_steps // 4
    quarters = [df.iloc[i*q:(i+1)*q] for i in range(4)]
    q_rewards = [qt['total_reward'].mean() if 'total_reward' in df.columns else 0
                 for qt in quarters]

    eq = total_episodes // 4
    event_quarters = [events.iloc[i*eq:(i+1)*eq] for i in range(4)] if eq > 0 else []
    q_wins = [int((eq_['event'] == 'WIN').sum()) for eq_ in event_quarters] if event_quarters else [0]*4

    # ── Episode length ────────────────────────────────────────────────────
    if 'step' in df.columns and total_episodes > 0:
        event_steps  = events['step'].values
        episode_lengths = np.diff(np.concatenate([[0], event_steps]))
        avg_ep_len   = float(episode_lengths.mean())
        min_ep_len   = float(episode_lengths.min())
        max_ep_len   = float(episode_lengths.max())
    else:
        avg_ep_len = min_ep_len = max_ep_len = 0.0

    # ── Death causes ──────────────────────────────────────────────────────
    death_rows = events[events['event'] == 'DIED']
    causes     = death_rows['cause'].value_counts() if ('cause' in death_rows.columns and deaths > 0) else pd.Series()

    # ── Level analysis ────────────────────────────────────────────────────
    levels_completed = int(df['levels_completed'].max()) if 'levels_completed' in df.columns else 0
    level_counts     = events['level'].value_counts().sort_index() if 'level' in events.columns else pd.Series()
    level_win_counts = events[events['event'] == 'WIN']['level'].value_counts().sort_index() if wins > 0 and 'level' in events.columns else pd.Series()

    # ── Stall detection ───────────────────────────────────────────────────
    stall_deaths     = int(causes.get('Stall', 0)) if len(causes) > 0 else 0
    stall_pct        = (stall_deaths / deaths * 100) if deaths > 0 else 0.0

    # ── Idle / action bias ────────────────────────────────────────────────
    actions = df['action'].value_counts(normalize=True) * 100 if 'action' in df.columns else pd.Series()
    idle_pct = float(actions.get('IDLE', 0.0))
    left_pct = float(actions.get('LEFT', 0.0)) + float(actions.get('LEFT+JUMP', 0.0)) + float(actions.get('RUN+LEFT', 0.0)) + float(actions.get('RUN+LEFT+JUMP', 0.0))
    right_pct = float(actions.get('RIGHT', 0.0)) + float(actions.get('RIGHT+JUMP', 0.0)) + float(actions.get('RUN+RIGHT', 0.0)) + float(actions.get('RUN+RIGHT+JUMP', 0.0))
    jump_pct  = float(actions.get('JUMP', 0.0))

    # ── Movement ─────────────────────────────────────────────────────────
    max_x      = float(df['x'].max())     if 'x'  in df.columns else 0.0
    avg_vx     = float(df['vx'].mean())   if 'vx' in df.columns else 0.0
    avg_goal_d = float(df['goal_dist'].mean()) if 'goal_dist' in df.columns else 0.0

    # ── Reward component totals ────────────────────────────────────────────
    component_sums = {}
    component_means = {}
    if reward_cols:
        for col in reward_cols:
            vals = pd.to_numeric(df[col], errors='coerce').fillna(0)
            component_sums[col]  = float(vals.sum())
            component_means[col] = float(vals.mean())
        total_component_sum = sum(abs(v) for v in component_sums.values()) or 1.0

    # ══════════════════════════════════════════════════════════════════════
    # PRINT REPORT
    # ══════════════════════════════════════════════════════════════════════
    W = 58
    print("\n" + "═"*W)
    print("  🤖  PEAK AGENT PERFORMANCE REPORT")
    print("═"*W)

    # ── Overview ──────────────────────────────────────────────────────────
    print(f"\n  ⏱  Steps:          {total_steps:>10,}")
    print(f"  🎬 Episodes:        {total_episodes:>10,}")
    print(f"  🏆 Win Rate:        {win_rate:>9.1f}%  ({wins} W / {deaths} D)")
    print(f"  🗺  Levels Beaten:   {levels_completed:>10,}")
    print(f"  📏 Furthest X:      {max_x:>10.1f} tiles")
    print(f"  🎯 Avg Goal Dist:   {avg_goal_d:>10.1f} tiles")

    # ── Episode length ─────────────────────────────────────────────────
    print(f"\n{'─'*W}")
    print("  ⏳  EPISODE LENGTH")
    print(f"  Avg: {avg_ep_len:,.0f} steps  |  Min: {min_ep_len:,.0f}  |  Max: {max_ep_len:,.0f}")
    if avg_ep_len < 100:
        print("  ⚠  Very short episodes — agent is dying almost immediately.")
    elif avg_ep_len > 8000:
        print("  ⚠  Very long episodes — agent may be stalling rather than dying.")

    # ── Death causes ───────────────────────────────────────────────────
    print(f"\n{'─'*W}")
    print("  💀  DEATH CAUSES")
    if len(causes) > 0:
        for cause, count in causes.items():
            pct = count / deaths * 100
            print(f"  {_bar(pct, 100, 16)} {cause:<12} {count:>5}x  ({pct:.1f}%)")
        if stall_pct > 30:
            print(f"  ⚠  {stall_pct:.0f}% of deaths are stalls — agent is getting stuck frequently.")
    else:
        print("  No deaths recorded yet.")

    # ── Level breakdown ───────────────────────────────────────────────
    print(f"\n{'─'*W}")
    print("  🗺  LEVEL BREAKDOWN")
    if len(level_counts) > 0:
        for lvl in level_counts.index:
            visits = int(level_counts[lvl])
            wins_here = int(level_win_counts.get(lvl, 0))
            lvl_win_rate = (wins_here / visits * 100) if visits > 0 else 0
            bar = _bar(lvl_win_rate, 100, 16)
            print(f"  Level {lvl:<3}  {bar} {lvl_win_rate:>5.1f}% win  ({wins_here}/{visits})")
        # Identify the bottleneck level
        if len(level_win_counts) > 0:
            bottleneck = level_counts[~level_counts.index.isin(level_win_counts.index)]
            if not bottleneck.empty:
                worst = bottleneck.idxmax()
                print(f"  ⚠  Level {worst} is a bottleneck — visited {int(level_counts[worst])}x with 0 wins.")
    else:
        print("  No level data yet.")

    # ── Action bias ────────────────────────────────────────────────────
    print(f"\n{'─'*W}")
    print("  🎮  ACTION BIAS")
    for act, pct in actions.items():
        bar = _bar(pct, 100, 16)
        flag = " ⚠ " if (act == 'IDLE' and pct > 15) else "    "
        print(f"  {flag}{act:<22} {bar} {pct:>5.1f}%")
    if idle_pct > 15:
        print(f"  ⚠  High IDLE ({idle_pct:.1f}%) — agent may be freezing under uncertainty.")
    if left_pct > right_pct * 1.5:
        print(f"  ⚠  Agent moves LEFT more than RIGHT — check reward shaping.")
    if jump_pct < 2 and right_pct > 10:
        print(f"  ⚠  Barely jumping ({jump_pct:.1f}%) — may struggle with obstacles.")

    # ── Reward components ──────────────────────────────────────────────
    if reward_cols:
        print(f"\n{'─'*W}")
        print("  🧬  REWARD COMPONENT BREAKDOWN (contribution %)")
        for col in reward_cols:
            pct = abs(component_sums[col]) / total_component_sum * 100
            mean_val = component_means[col]
            bar = _bar(pct, 100, 16)
            sign = "+" if component_sums[col] >= 0 else "-"
            print(f"  {col:<14} {bar} {pct:>5.1f}%  (mean/step: {mean_val:+.5f})")
        # Flag dominant components
        for col in reward_cols:
            pct = abs(component_sums[col]) / total_component_sum * 100
            if pct > 60:
                print(f"  ⚠  '{col}' dominates at {pct:.0f}% — other signals may be drowned out.")

    # ── Quarter trend analysis ─────────────────────────────────────────
    print(f"\n{'─'*W}")
    print("  📈  LEARNING CURVE  (Q1 → Q2 → Q3 → Q4)")
    reward_str = "  Reward:  " + "  →  ".join(f"{r:+.4f}" for r in q_rewards)
    wins_str   = "  Wins:    " + "  →  ".join(f"{w:>4}" for w in q_wins)
    print(reward_str)
    print(wins_str)

    # Classify learning shape
    if q_rewards[-1] > q_rewards[0] and q_rewards[-1] > q_rewards[1]:
        shape = "📈 Still improving"
    elif q_rewards[2] > q_rewards[3] and q_rewards[1] < q_rewards[2]:
        shape = "📉 Plateaued and regressing — possible overfit or reward exhaustion"
    elif max(q_rewards) - min(q_rewards) < 0.01:
        shape = "➡  Flat — agent may not be learning at all"
    else:
        shape = "〰 Unstable / noisy"
    print(f"  Shape: {shape}")

    print("\n" + "═"*W)

    # ── LLM Export ────────────────────────────────────────────────────
    print("\n  💡 AI ADVICE PAYLOAD  (paste to Claude/GPT for reward tuning)")
    print("─"*W)
    print(f"Game: platformer | Steps: {total_steps:,} | Episodes: {total_episodes:,}")
    print(f"Win rate: {win_rate:.1f}% | Levels beaten: {levels_completed}")
    print(f"Avg episode length: {avg_ep_len:.0f} steps | Furthest X: {max_x:.1f} tiles")
    print(f"Learning curve (quarterly avg reward): {[round(float(r),4) for r in q_rewards]}")
    print(f"Quarterly wins: {q_wins}")
    if len(causes):
        print(f"Death causes: {dict(causes)}")
    if level_counts is not None and len(level_counts) > 0:
        print(f"Level visits: {dict(level_counts)}")
        print(f"Level wins:   {dict(level_win_counts)}")
    if reward_cols:
        print(f"Reward components (mean/step): { {c: round(component_means[c], 5) for c in reward_cols} }")
    print(f"Action distribution (%): { {a: round(p, 1) for a, p in actions.items()} }")
    print(f"Idle: {idle_pct:.1f}% | Left bias: {left_pct:.1f}% | Right bias: {right_pct:.1f}%")
    print("─"*W + "\n")


if __name__ == "__main__":
    analyze_agent()
