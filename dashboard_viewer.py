import streamlit as st
import pandas as pd
import time
import os
import glob
import plotly.express as px

# --- THEME CONFIGURATION ---
THEME = {
    "bg": "#220817",        # Dark Purple
    "accent": "#E7575A",    # Salmon
    "text": "#B3C8EF",      # Light Blue
    "secondary": "#801830", # Dark Red
    "grid": "#442233"
}

st.set_page_config(page_title="PEAK Analysis", layout="wide", page_icon="ðŸ“ˆ")

# --- CUSTOM CSS ---
st.markdown(f"""
    <style>
        .stApp {{
            background-color: {THEME['bg']};
            color: {THEME['text']};
        }}
        div[data-testid="stStatusWidget"] {{ visibility: hidden; }}

        /* Typography */
        h1, h2, h3 {{
            color: {THEME['text']} !important;
            font-family: 'Courier New', monospace;
        }}

        /* Metrics */
        div[data-testid="stMetricValue"] {{
            color: {THEME['accent']} !important;
            font-family: 'Courier New', monospace;
            font-weight: bold;
            font-size: 1.2rem !important;
        }}
        div[data-testid="stMetricLabel"] {{
            color: {THEME['text']} !important;
            opacity: 0.8;
            font-family: 'Arial', sans-serif;
            font-size: 0.8rem !important;
        }}

        /* Checkboxes */
        label[data-testid="stCheckbox"] {{ color: {THEME['text']} !important; }}
        div[data-testid="stCheckbox"] div[role="checkbox"][aria-checked="true"] {{
            background-color: {THEME['accent']} !important;
            border-color: {THEME['accent']} !important;
        }}
    </style>
""", unsafe_allow_html=True)

# --- HELPER: Find Logs ---
def get_latest_log_file():
    search_paths = [".", "mylogs", "runs"]
    files = []
    for root_dir in search_paths:
        found = glob.glob(os.path.join(root_dir, "**", "training_log*.csv"), recursive=True)
        files.extend(found)
    if not files: return None
    return max(files, key=os.path.getmtime)

# --- SIDEBAR ---
with st.sidebar:
    st.header("MISSION CONTROL")
    do_refresh = st.checkbox("LIVE DATA FEED", value=True)

    st.divider()
    # --- FIX: VIEW MODE TOGGLE ---
    st.caption("Graph View Mode")
    show_full_history = st.checkbox("Show Full History (Start -> Cursor)", value=True)

    st.divider()
    if st.button("WIPE LOGS"):
        log_file = get_latest_log_file()
        if log_file and os.path.exists(log_file):
            try:
                os.remove(log_file)
                st.toast("Logs cleared.")
                time.sleep(1)
                st.rerun()
            except: pass

# --- HEADER ---
st.title("PEAK AGENT TRAINING ANALYSIS DASHBOARD")

log_file = get_latest_log_file()
if not log_file:
    st.warning("WAITING FOR SIGNAL...")
    time.sleep(2)
    st.rerun()

try:
    df = pd.read_csv(log_file,low_memory=False)
except:
    st.stop()

if df.empty:
    st.info("BUFFERING...")
    time.sleep(1)
    st.rerun()

# --- PREPROCESSING ---
standard_cols = ['step', 'total_reward', 'action', 'level', 'levels_completed', 'x', 'y', 'vx', 'vy', 'goal_dist', 'event', 'cause']
reward_cols = [c for c in df.columns if c.lower() not in standard_cols and "unnamed" not in c.lower()]

# --- SCRUBBER ---
total_steps = len(df)
if do_refresh:
    selected_idx = total_steps - 1
else:
    selected_idx = st.slider("TIMELINE", 0, total_steps - 1, total_steps - 1)

if selected_idx >= len(df): st.stop()
row = df.iloc[selected_idx]

# =========================================================
# 1. MISSION STATUS
# =========================================================
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("STEP", int(row.get('step', 0)))
c2.metric("TOTAL REWARD", f"{row.get('total_reward', 0):.4f}")
c3.metric("CURRENT LEVEL", str(row.get('level', 0)))
c4.metric("LEVELS BEATEN", int(row.get('levels_completed', 0)))

# Compute wins and deaths up to the scrubber position
events_up_to_now = df.iloc[:selected_idx + 1]
wins_so_far  = (events_up_to_now.get('event', pd.Series(dtype=str)) == 'WIN').sum()  if 'event' in df.columns else 0
deaths_so_far = (events_up_to_now.get('event', pd.Series(dtype=str)) == 'DIED').sum() if 'event' in df.columns else 0
c5.metric("WINS", int(wins_so_far))
c6.metric("DEATHS", int(deaths_so_far))

st.divider()

# =========================================================
# 2. LEVEL PROGRESS
# =========================================================
st.subheader("🗺️ LEVEL PROGRESS")

lp1, lp2, lp3 = st.columns([2, 2, 1])

with lp1:
    # Level visited over time — shows which levels the agent is reaching
    if 'level' in df.columns and 'step' in df.columns:
        subset_level = df.iloc[:selected_idx + 1][['step', 'level']].copy()
        fig_level = px.line(
            subset_level, x='step', y='level',
            color_discrete_sequence=["#00FFFF"],
            height=200,
            labels={"level": "Level Index", "step": "Step"},
        )
        fig_level.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=THEME['text']),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor=THEME['grid'], dtick=1),
            margin=dict(l=0, r=0, t=24, b=0),
            title=dict(text="Level Index Over Time", font=dict(size=12), x=0),
        )
        st.plotly_chart(fig_level, width='stretch')
    else:
        st.caption("No level data yet.")

with lp2:
    # Levels completed over time — the cumulative wins curve
    if 'levels_completed' in df.columns and 'step' in df.columns:
        subset_wins = df.iloc[:selected_idx + 1][['step', 'levels_completed']].copy()
        fig_wins = px.area(
            subset_wins, x='step', y='levels_completed',
            color_discrete_sequence=["#00FF88"],
            height=200,
            labels={"levels_completed": "Levels Beaten", "step": "Step"},
        )
        fig_wins.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=THEME['text']),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor=THEME['grid']),
            margin=dict(l=0, r=0, t=24, b=0),
            title=dict(text="Cumulative Levels Beaten", font=dict(size=12), x=0),
        )
        st.plotly_chart(fig_wins, width='stretch')
    else:
        st.caption("No completion data yet.")

with lp3:
    st.caption("Level Breakdown")
    if 'level' in df.columns and 'event' in df.columns:
        wins_df = df.iloc[:selected_idx + 1]
        win_rows = wins_df[wins_df['event'] == 'WIN']
        if not win_rows.empty:
            level_counts = win_rows['level'].value_counts().sort_index()
            for lvl, count in level_counts.items():
                st.metric(f"Level {lvl}", f"✅ {count}x")
        else:
            st.caption("No wins yet.")
    # Always show current level action
    st.metric("ACTION", str(row.get('action', 'N/A')))

st.divider()

# =========================================================
# 3. REWARD DNA
# =========================================================
st.subheader("🧬 REWARD COMPOSITION")
if reward_cols:
    cols = st.columns(len(reward_cols))
    for i, col_name in enumerate(reward_cols):
        val = row[col_name]
        label = col_name.upper()
        if abs(val) > 0.0001: label = f"   {label}"
        cols[i].metric(label, f"{val:.4f}")
else:
    st.caption("No custom components found.")

st.divider()

# =========================================================
# 3. SPLIT VIEW: GRAPH (Left) | PHYSICS (Right)
# =========================================================
col_graph, col_phys = st.columns([3, 1]) # 3:1 Ratio

# --- LEFT COLUMN: GRAPH ---
with col_graph:
    st.subheader("SIGNAL VISUALIZER")

    available_signals = ['total_reward'] + reward_cols
    selected_lines = []

    # Toggles Row
    check_cols = st.columns(len(available_signals))
    for i, signal_name in enumerate(available_signals):
        short_name = signal_name.replace("total_", "").replace("_", " ").title()
        is_checked = check_cols[i].checkbox(short_name, value=True, key=f"chk_{i}")
        if is_checked:
            selected_lines.append(signal_name)

    # --- FIX: GRAPH LOGIC ---
    if show_full_history:
        start_viz = 0  # Always start from Step 0
    else:
        # Default "Zoom" window of 1000 steps
        start_viz = max(0, selected_idx - 1000)

    subset = df.iloc[start_viz : selected_idx + 1]

    if not subset.empty and selected_lines:
        # Cyberpunk Palette
        custom_palette = [
            "#FFFFFF", # White
            "#00FFFF", # Cyan
            "#FFD700", # Gold
            "#FF0055", # Neon Red
            "#00FF00", # Lime
            "#BD93F9", # Purple
            "#555555"  # Grey
        ]

        fig = px.line(subset, x='step', y=selected_lines,
                      color_discrete_sequence=custom_palette,
                      height=500)

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=THEME['text']),
            xaxis=dict(showgrid=False, title="Training Steps"),
            yaxis=dict(showgrid=True, gridcolor=THEME['grid']),
            legend=dict(orientation="h", y=1.1, title=None)
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.caption("No signals selected.")

# --- RIGHT COLUMN: PHYSICS ---
with col_phys:
    st.subheader("PHYSICS")
    st.caption("Live Telemetry (Tiles)")

    # Using nested columns to create a tight 2x2 Grid
    pc1, pc2 = st.columns(2)

    with pc1:
        st.metric("X POS", f"{row.get('x', 0):.1f}")
        st.metric("VEL X", f"{row.get('vx', 0):.2f}")

    with pc2:
        st.metric("Y POS", f"{row.get('y', 0):.1f}")
        st.metric("VEL Y", f"{row.get('vy', 0):.2f}")

    st.metric("GOAL DIST", f"{row.get('goal_dist', 0):.1f}")

    st.divider()

    # Show Event if one happened
    evt = row.get('event', "")
    cause = row.get('cause', "")
    if evt and isinstance(evt, str) and len(evt) > 0:
        st.error(f"{evt} ({cause})")

if do_refresh:
    time.sleep(1)
    st.rerun()
