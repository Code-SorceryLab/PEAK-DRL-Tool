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

    # Custom exception handler to suppress WebSocketClosedError
    def _suppress_websocket_errors(loop, context):
        """Suppress harmless WebSocket closure exceptions"""
        exception = context.get('exception')
        if exception:
            exc_type = type(exception).__name__
            # Suppress WebSocketClosedError and StreamClosedError
            if exc_type in ('WebSocketClosedError', 'StreamClosedError'):
                return
        # For other exceptions, use default handler
        loop.default_exception_handler(context)

    # Set the custom exception handler
    try:
        loop = asyncio.get_event_loop()
        loop.set_exception_handler(_suppress_websocket_errors)
    except RuntimeError:
        pass  # No event loop running yet

    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# --- THEME CONFIGURATION ---
THEME = {
    "bg": "#220817",        # Dark Purple
    "accent": "#E7575A",    # Salmon
    "text": "#B3C8EF",      # Light Blue
    "secondary": "#801830", # Dark Red
    "grid": "#442233"
}

st.set_page_config(page_title="PEAK Analysis", layout="wide", page_icon="Kev & AL ")

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

# Obs sanity columns — kept for the OBSERVATION SANITY table only, NOT the graph or reward composition
# Channel order: player(0), hazard(1), collectible(2), dijkstra(3) — solid removed
_OBS_SANITY_COLS = {
    'grid_player_mean',      'grid_player_std',      'grid_player_min',      'grid_player_max',
    'grid_solid_mean',       'grid_solid_std',       'grid_solid_min',       'grid_solid_max',
    'grid_hazard_mean',      'grid_hazard_std',      'grid_hazard_min',      'grid_hazard_max',
    'grid_collectible_mean', 'grid_collectible_std', 'grid_collectible_min', 'grid_collectible_max',
    'grid_dijkstra_mean',    'grid_dijkstra_std',    'grid_dijkstra_min',    'grid_dijkstra_max',
    'scalar_mean', 'scalar_std', 'scalar_min', 'scalar_max',
    'dijkstra_val', 'obs_warnings',
}

reward_cols = [
    c for c in df.columns
    if c.lower() not in standard_cols
    and c.lower() not in _OBS_SANITY_COLS
    and "unnamed" not in c.lower()
]

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
        died_rows = lvl_rows[lvl_rows['event'] == 'DIED']
        wins     = int((lvl_rows['event'] == 'WIN').sum())
        deaths   = int(len(died_rows))
        visits   = wins + deaths  # Count completed attempts only
        win_rate = (wins / visits * 100) if visits > 0 else 0.0
        filled   = int(win_rate / 10)
        bar      = "\u2588" * filled + "\u2591" * (10 - filled)

        # Count deaths by cause
        d_stall   = int((died_rows['cause'] == 'Stall').sum())   if 'cause' in df.columns else 0
        d_enemy   = int((died_rows['cause'] == 'Enemy').sum())   if 'cause' in df.columns else 0
        d_pit     = int((died_rows['cause'] == 'Pit').sum())     if 'cause' in df.columns else 0
        d_spike   = int((died_rows['cause'] == 'Spike').sum())   if 'cause' in df.columns else 0
        d_timeout = int((died_rows['cause'] == 'Timeout').sum()) if 'cause' in df.columns else 0

        rows.append({
            "LEVEL":     f"Level {lvl}",
            "VISITS":    visits,
            "WINS":      wins,
            "DEATHS":    deaths,
            "STALL":     d_stall,
            "ENEMY":     d_enemy,
            "PIT":       d_pit,
            "SPIKE":     d_spike,
            "TIMEOUT":   d_timeout,
            "WIN RATE":  f"{win_rate:.1f}%",
            "PROGRESS":  bar,
        })

    if rows:
        level_table = pd.DataFrame(rows)

        death_cause_cols = {"STALL", "ENEMY", "PIT", "SPIKE", "TIMEOUT"}

        def _style_row(r):
            try:
                wr = float(r["WIN RATE"].replace("%", ""))
            except Exception:
                wr = 0.0
            wr_color = "#27ae60" if wr >= 90 else ("#e67e22" if wr >= 50 else "#e74c3c")
            styles = []
            for c in r.index:
                if c == "WIN RATE":
                    styles.append(f"color:{wr_color}")
                elif c in death_cause_cols:
                    val = int(r[c]) if r[c] else 0
                    styles.append("color:#e74c3c" if val > 0 else "color:#555555")
                else:
                    styles.append("")
            return styles

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

# =========================================================
# 5. OBSERVATION SANITY
# =========================================================
obs_cols = [c for c in df.columns if c.startswith("grid_") or c.startswith("scalar_") or c == "dijkstra_val" or c == "obs_warnings"]
if obs_cols:
    st.divider()
    st.subheader("OBSERVATION SANITY")

    # Show warnings if any
    warn_val = row.get("obs_warnings", "")
    if warn_val and isinstance(warn_val, str) and len(warn_val) > 0:
        for w in warn_val.split("|"):
            st.error(f"[WARN] {w}")
    else:
        st.success("All channels OK")

    # Grid channel stats table
    # Channel order matches Change 1: solid removed, dijkstra added as ch3
    # 5 channels: Player(0), Solid(1), Hazard(2), Collectible(3), Dijkstra(4)
    grid_channels = ["player", "solid", "hazard", "collectible", "dijkstra"]
    has_grid_data = any(f"grid_{ch}_mean" in df.columns for ch in grid_channels)

    if has_grid_data:
        obs_rows = []
        for ch in grid_channels:
            mean_val = row.get(f"grid_{ch}_mean", 0.0)
            std_val  = row.get(f"grid_{ch}_std", 0.0)
            min_val  = row.get(f"grid_{ch}_min", 0.0)
            max_val  = row.get(f"grid_{ch}_max", 0.0)

            # Determine status
            status = "OK"
            if isinstance(std_val, (int, float)) and std_val < 1e-6 and ch in ("player", "solid"):  # these channels should be non-sparse
                status = "DEAD"
            elif isinstance(max_val, (int, float)) and max_val > 1.01:
                status = "OVERFLOW"

            obs_rows.append({
                "CHANNEL": ch.upper(),
                "MEAN": f"{float(mean_val):.4f}" if isinstance(mean_val, (int, float)) else "NULL",
                "STD":  f"{float(std_val):.4f}" if isinstance(std_val, (int, float)) else "NULL",
                "MIN":  f"{float(min_val):.4f}" if isinstance(min_val, (int, float)) else "NULL",
                "MAX":  f"{float(max_val):.4f}" if isinstance(max_val, (int, float)) else "NULL",
                "STATUS": status,
            })

        # Scalars row
        s_mean = row.get("scalar_mean", 0.0)
        s_std  = row.get("scalar_std", 0.0)
        s_min  = row.get("scalar_min", 0.0)
        s_max  = row.get("scalar_max", 0.0)
        s_status = "OK"
        if isinstance(s_std, (int, float)) and s_std < 1e-8:
            s_status = "DEAD"
        elif isinstance(s_max, (int, float)) and abs(s_max) > 100:
            s_status = "UNNORM"
        obs_rows.append({
            "CHANNEL": "SCALARS",
            "MEAN": f"{float(s_mean):.4f}" if isinstance(s_mean, (int, float)) else "NULL",
            "STD":  f"{float(s_std):.4f}" if isinstance(s_std, (int, float)) else "NULL",
            "MIN":  f"{float(s_min):.4f}" if isinstance(s_min, (int, float)) else "NULL",
            "MAX":  f"{float(s_max):.4f}" if isinstance(s_max, (int, float)) else "NULL",
            "STATUS": s_status,
        })

        # Dijkstra stats come from grid_dijkstra_* via the grid_channels loop above.
        # dijkstra_val scalar is included in the SCALARS row (it is scalars[-1]).

        obs_table = pd.DataFrame(obs_rows)

        def _style_obs(r):
            status = r.get("STATUS", "OK")
            styles = []
            for c in r.index:
                if c == "STATUS":
                    if status == "OK":
                        styles.append("color:#27ae60")
                    elif status == "ZERO":
                        styles.append("color:#e67e22")
                    else:
                        styles.append("color:#e74c3c; font-weight:bold")
                else:
                    styles.append("")
            return styles

        styled_obs = (
            obs_table.style
            .apply(_style_obs, axis=1)
            .set_properties(**{
                "font-family": "Courier New, monospace",
                "font-size": "13px",
                "text-align": "center",
            })
            .set_table_styles([{
                "selector": "th",
                "props": [
                    ("background-color", "#2a0a1a"),
                    ("color", "#B3C8EF"),
                    ("font-family", "Courier New, monospace"),
                    ("font-size", "11px"),
                    ("text-align", "center"),
                    ("padding", "6px 12px"),
                    ("border-bottom", "1px solid #801830"),
                ]
            }, {
                "selector": "td",
                "props": [
                    ("background-color", "#1a0510"),
                    ("color", "#B3C8EF"),
                    ("padding", "5px 12px"),
                    ("border-bottom", "1px solid #330020"),
                ]
            }])
            .hide(axis="index")
        )
        st.markdown(styled_obs.to_html(), unsafe_allow_html=True)
        st.caption(f"Updated every {5000} steps")

if do_refresh:
    time.sleep(1)
    try:
        st.rerun()
    except Exception:
        # Gracefully handle websocket closure (when browser tab is closed)
        pass