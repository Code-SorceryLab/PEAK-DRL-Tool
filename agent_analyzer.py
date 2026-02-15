import pandas as pd
import glob
import os
import numpy as np

def get_latest_log_file():
    """Finds the most recent CSV log file."""
    search_paths = [".", "mylogs", "runs"]
    files = []
    for root_dir in search_paths:
        found = glob.glob(os.path.join(root_dir, "**", "training_log*.csv"), recursive=True)
        files.extend(found)
    if not files: return None
    return max(files, key=os.path.getmtime)

def analyze_agent():
    log_file = get_latest_log_file()
    if not log_file:
        print("❌ No training logs found. Run training first.")
        return

    print(f"📂 Loading data from: {log_file}")
    try:
        df = pd.read_csv(log_file)
    except Exception as e:
        print(f"❌ Failed to read CSV: {e}")
        return

    if df.empty:
        print("⚠️ Log file is empty. Agent hasn't taken enough steps yet.")
        return

    # --- 1. BASIC METRICS ---
    total_steps = len(df)

    # Filter for actual events (ignoring NaN or empty strings)
    events = df[df['event'].notna() & (df['event'] != "")]
    total_episodes = len(events)

    wins = len(events[events['event'] == 'WIN'])
    deaths = len(events[events['event'] == 'DIED'])
    win_rate = (wins / total_episodes * 100) if total_episodes > 0 else 0.0

    # --- 2. MOVEMENT & PHYSICS ---
    max_x = df['x'].max() if 'x' in df else 0
    avg_vx = df['vx'].mean() if 'vx' in df else 0

    # --- 3. ACTION DISTRIBUTION ---
    if 'action' in df:
        actions = df['action'].value_counts(normalize=True) * 100
    else:
        actions = pd.Series()

    # --- 4. LEARNING TRENDS (First Half vs Second Half) ---
    half = total_steps // 2
    first_half = df.iloc[:half]
    second_half = df.iloc[half:]

    fh_reward = first_half['total_reward'].mean() if 'total_reward' in df else 0
    sh_reward = second_half['total_reward'].mean() if 'total_reward' in df else 0

    # Episode splitting for trend analysis
    fh_events = events.iloc[:len(events)//2]
    sh_events = events.iloc[len(events)//2:]

    fh_wins = len(fh_events[fh_events['event'] == 'WIN'])
    sh_wins = len(sh_events[sh_events['event'] == 'WIN'])

    # --- 5. PRINT THE REPORT ---
    print("\n" + "="*50)
    print(" 🤖 PEAK AGENT PERFORMANCE REPORT")
    print("="*50)
    print(f"⏱️  Total Steps: {total_steps:,}")
    print(f"🎬 Total Episodes: {total_episodes:,}")
    print(f"🏆 Win Rate: {win_rate:.2f}% ({wins} Wins / {deaths} Deaths)")

    print("\n💀 DEATH CAUSES:")
    if deaths > 0 and 'cause' in events:
        causes = events[events['event'] == 'DIED']['cause'].value_counts()
        for cause, count in causes.items():
            print(f"   - {cause}: {count} times ({(count/deaths)*100:.1f}%)")
    else:
        print("   - N/A")

    print("\n📐 PHYSICS & TRAVERSAL:")
    print(f"   - Furthest Distance Reached: {max_x:.2f} tiles")
    print(f"   - Average X Velocity: {avg_vx:.2f} tiles/sec")

    print("\n🎮 ACTION PREFERENCES:")
    for action, pct in actions.items():
        print(f"   - {action}: {pct:.1f}%")

    print("\n📈 LEARNING TRENDS (Early vs Late Training):")
    print(f"   - Avg Reward: {fh_reward:.2f} ➔ {sh_reward:.2f} " + ("✅ Improved" if sh_reward > fh_reward else "⚠️ Declined"))
    print(f"   - Wins:       {fh_wins} ➔ {sh_wins} " + ("✅ Improved" if sh_wins > fh_wins else "⚠️ Declined"))
    print("="*50 + "\n")

    # --- 6. LLM EXPORT PAYLOAD ---
    print("💡 TIP: Copy the block below and paste it to your AI to ask for advice on tweaking rewards:\n")
    print("--- DATA FOR AI ---")
    print(f"Steps: {total_steps}, Win Rate: {win_rate:.1f}%")
    print(f"Reward Trend: {fh_reward:.2f} -> {sh_reward:.2f}")
    if deaths > 0 and 'cause' in events:
        print("Death Causes:", dict(causes))
    print("Actions:", dict(actions.apply(lambda x: round(x, 1))))
    print("-------------------\n")

if __name__ == "__main__":
    analyze_agent()
