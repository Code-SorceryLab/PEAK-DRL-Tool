import streamlit as st
import pandas as pd
import time
import os
import glob
import plotly.express as px
import logging
import warnings
import sys

# Suppress WebSocket closure warnings (harmless when browser tab closes/refreshes)
logging.getLogger('tornado.access').setLevel(logging.ERROR)
logging.getLogger('tornado.application').setLevel(logging.ERROR)
logging.getLogger('tornado.general').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Suppress asyncio task exception warnings for closed websockets
if sys.version_info >= (3, 8):
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy() if os.name == 'nt' else None)

# --- THEME CONFIGURATION ---
THEME = {
    "bg": "#220817",        # Dark Purple
    "accent": "#E7575A",    # Salmon
    "text": "#B3C8EF",      # Light Blue
    "secondary": "#801830", # Dark Red
    "grid": "#442233"
}

st.set_page_config(page_title="PEAK Analysis", layout="wide", page_icon="ÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã‹â€ ")

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
st.subheader("\U0001f5fa\ufe0f LEVEL PROGRESS")

if 'level' in df.columns and 'event' in df.columns:
    data_so_far = df.iloc[:selected_idx + 1]
    all_levels  = sorted(data_so_far['level'].dropna().unique().astype(int))

    rows = []
    for lvl in all_levels:
        lvl_rows = data_so_far[data_so_far['level'] == lvl]
        wins     = int((lvl_rows['event'] == 'WIN').sum())
        deaths   = int((lvl_rows['event'] == 'DIED').sum())
        visits   = wins + deaths  # Count completed attempts only
        win_rate = (wins / visits * 100) if visits > 0 else 0.0
        filled   = int(win_rate / 10)
        bar      = "\u2588" * filled + "\u2591" * (10 - filled)
        rows.append({
            "LEVEL":     f"Level {lvl}",
            "VISITS":    visits,
            "WINS":      wins,
            "DEATHS":    deaths,
            "WIN RATE":  f"{win_rate:.1f}%",
            "PROGRESS":  bar,
        })

    if rows:
        level_table = pd.DataFrame(rows)

        def _style_row(r):
            try:
                wr = float(r["WIN RATE"].replace("%", ""))
            except Exception:
                wr = 0.0
            color = "#27ae60" if wr >= 90 else ("#e67e22" if wr >= 50 else "#e74c3c")
            return [f"color:{color}" if c == "WIN RATE" else "" for c in r.index]

        styled = (
            level_table.style
            .apply(_style_row, axis=1)
            .set_properties(**{
                "font-family": "Courier New, monospace",
                "font-size": "14px",
                "text-align": "center",
            })
            .set_table_styles([{
                "selector": "th",
                "props": [
                    ("background-color", "#2a0a1a"),
                    ("color", "#B3C8EF"),
                    ("font-family", "Courier New, monospace"),
                    ("font-size", "12px"),
                    ("letter-spacing", "1px"),
                    ("text-align", "center"),
                    ("padding", "8px 16px"),
                    ("border-bottom", "1px solid #801830"),
                ]
            }, {
                "selector": "td",
                "props": [
                    ("background-color", "#1a0510"),
                    ("color", "#B3C8EF"),
                    ("padding", "7px 16px"),
                    ("border-bottom", "1px solid #330020"),
                ]
            }, {
                "selector": "tr:hover td",
                "props": [("background-color", "#2d0a1e")]
            }])
            .hide(axis="index")
        )
        st.markdown(styled.to_html(), unsafe_allow_html=True)

        st.caption("")
        total_visits = sum(r["VISITS"] for r in rows)
        total_wins   = sum(r["WINS"]   for r in rows)
        total_deaths = sum(r["DEATHS"] for r in rows)
        overall_wr   = (total_wins / total_visits * 100) if total_visits > 0 else 0.0
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("TOTAL VISITS",  total_visits)
        s2.metric("TOTAL WINS",    total_wins)
        s3.metric("TOTAL DEATHS",  total_deaths)
        s4.metric("OVERALL WIN %", f"{overall_wr:.1f}%")
    else:
        st.caption("No level data yet.")

    st.caption(f"\u21b3 Current action: **{str(row.get('action', 'N/A'))}**")
else:
    st.caption("No level / event data available.")
st.divider()

# =========================================================
# 3. REWARD DNA
# =========================================================
st.subheader("REWARD COMPOSITION")
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
        try:
            st.plotly_chart(fig, width='stretch')
        except Exception:
            # Gracefully handle rendering errors
            pass
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
    try:
        st.rerun()
    except Exception:
        # Gracefully handle websocket closure (when browser tab is closed)
        pass
