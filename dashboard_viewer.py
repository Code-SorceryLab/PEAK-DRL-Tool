import streamlit as st
import pandas as pd
import time
import os
import glob
import plotly.express as px
import logging
import warnings
import sys

logging.getLogger('tornado.access').setLevel(logging.ERROR)
logging.getLogger('tornado.application').setLevel(logging.ERROR)
logging.getLogger('tornado.general').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=DeprecationWarning)

if sys.version_info >= (3, 8):
    import asyncio
    def _suppress_ws(loop, ctx):
        exc = ctx.get('exception')
        if exc and type(exc).__name__ in ('WebSocketClosedError', 'StreamClosedError'):
            return
        loop.default_exception_handler(ctx)
    try:
        loop = asyncio.get_event_loop()
        loop.set_exception_handler(_suppress_ws)
    except RuntimeError:
        pass
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

THEME = {"bg":"#220817","accent":"#E7575A","text":"#B3C8EF","secondary":"#801830","grid":"#442233"}

st.set_page_config(page_title="PEAK Analysis", layout="wide", page_icon="PEAK")

st.markdown(f"""
    <style>
        .stApp {{background-color:{THEME['bg']};color:{THEME['text']};}}
        div[data-testid="stStatusWidget"]{{visibility:hidden;}}
        h1,h2,h3{{color:{THEME['text']} !important;font-family:'Courier New',monospace;}}
        div[data-testid="stMetricValue"]{{color:{THEME['accent']} !important;font-family:'Courier New',monospace;font-weight:bold;font-size:1.2rem !important;}}
        div[data-testid="stMetricLabel"]{{color:{THEME['text']} !important;opacity:0.8;font-size:0.8rem !important;}}
        label[data-testid="stCheckbox"]{{color:{THEME['text']} !important;}}
    </style>
""", unsafe_allow_html=True)

def get_all_log_files():
    """Return all CSV log files, newest first."""
    files = []
    for root_dir in [".", "csv", "mylogs", "runs"]:
        files.extend(glob.glob(os.path.join(root_dir, "**", "training_log*.csv"), recursive=True))
    # Deduplicate and sort by modification time (newest first)
    seen = set()
    unique = []
    for f in files:
        fp = os.path.abspath(f)
        if fp not in seen:
            seen.add(fp)
            unique.append(f)
    return sorted(unique, key=os.path.getmtime, reverse=True)

def get_latest_log_file():
    files = get_all_log_files()
    return files[0] if files else None

def parse_run_info_from_filename(filepath):
    """Extract persona, skill, architecture from CSV filename."""
    basename = os.path.basename(filepath).replace("training_log_", "").replace(".csv", "")
    parts = basename.split("_")
    _ARCH_TAGS = {"light", "slim", "peak", "mlp"}
    arch = None
    if parts and parts[-1].lower() in _ARCH_TAGS:
        arch = parts[-1]
        parts = parts[:-1]
    # Format: game_algo_persona_skill  or  game_algo_game_persona_skill
    info = {"persona": "unknown", "skill": "unknown", "arch": arch or "unknown"}
    if len(parts) >= 4:
        info["skill"] = parts[-1]
        # persona is everything between algo and skill
        info["persona"] = "_".join(parts[2:-1])
        game = parts[0]
        if info["persona"].startswith(f"{game}_"):
            info["persona"] = info["persona"][len(game)+1:]
    return info

def downsample(df, max_pts=2000):
    """Reduce to at most max_pts rows, keeping first+last. Main crash fix."""
    if len(df) <= max_pts:
        return df
    step = max(1, len(df) // max_pts)
    idx  = list(range(0, len(df), step))
    if idx[-1] != len(df) - 1:
        idx.append(len(df) - 1)
    return df.iloc[idx].reset_index(drop=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("MISSION CONTROL")
    do_refresh = st.checkbox("LIVE DATA FEED", value=True)
    st.divider()

    st.caption("Graph Settings")
    show_graph = st.checkbox("Show Signal Graph", value=True,
        help="Turn OFF if dashboard crashes/lags. Graph is the heaviest section.")
    if show_graph:
        show_full_history = st.checkbox("Full History (slow on large runs)", value=False)
        max_plot_pts = st.select_slider("Max graph points",
            options=[500, 1000, 2000, 4000, 8000], value=2000,
            help="Lower = faster. Raise only if you need fine detail.")
    else:
        show_full_history = False
        max_plot_pts = 2000

    st.divider()
    if st.button("WIPE LOGS"):
        log_file = get_latest_log_file()
        if log_file and os.path.exists(log_file):
            try:
                os.remove(log_file)
                st.toast("Logs cleared.")
                time.sleep(1)
                st.rerun()
            except Exception:
                pass

    st.divider()
    st.caption("Session")
    if st.button("STOP DASHBOARD", type="primary",
            help="Freezes auto-refresh. Scrubber still works for post-training analysis."):
        st.session_state["force_stopped"] = True
        st.toast("Dashboard stopped. You can still scrub through history.")
        st.rerun()

    if st.session_state.get("force_stopped", False):
        do_refresh = False
        st.warning("Stopped. Refresh page to restart live feed.")

# ── Load data ──────────────────────────────────────────────────────────────────
st.title("PEAK AGENT TRAINING ANALYSIS DASHBOARD")

all_logs = get_all_log_files()
if not all_logs:
    st.warning("WAITING FOR SIGNAL...  (no training_log*.csv found yet)")
    if do_refresh:
        time.sleep(2)
        st.rerun()
    st.stop()

# Let user pick which log to view
if len(all_logs) > 1:
    log_labels = []
    for f in all_logs:
        info = parse_run_info_from_filename(f)
        label = f"{info['persona']} | {info['skill']} | {info['arch']}  ({os.path.basename(f)})"
        log_labels.append(label)
    selected_log_idx = st.selectbox("Select training run:", range(len(all_logs)),
                                     format_func=lambda i: log_labels[i])
    log_file = all_logs[selected_log_idx]
else:
    log_file = all_logs[0]

try:
    df = pd.read_csv(log_file, low_memory=False)
except Exception as e:
    st.error(f"Could not read log: {e}")
    st.stop()

# Parse run info from filename
run_info = parse_run_info_from_filename(log_file)

if df.empty:
    st.info("BUFFERING...")
    if do_refresh:
        time.sleep(1)
        st.rerun()
    st.stop()

standard_cols = {'step','total_reward','action','level','levels_completed','x','y','vx','vy','goal_dist','event','cause'}
_OBS_SANITY_COLS = {
    'grid_solid_mean','grid_solid_std','grid_solid_min','grid_solid_max',
    'grid_hazard_mean','grid_hazard_std','grid_hazard_min','grid_hazard_max',
    'grid_collectible_mean','grid_collectible_std','grid_collectible_min','grid_collectible_max',
    'grid_dijkstra_mean','grid_dijkstra_std','grid_dijkstra_min','grid_dijkstra_max',
    'scalar_mean','scalar_std','scalar_min','scalar_max','dijkstra_val','obs_warnings',
}
reward_cols = [c for c in df.columns
    if c.lower() not in standard_cols and c.lower() not in _OBS_SANITY_COLS
    and "unnamed" not in c.lower()]

total_steps = len(df)
col_i1, col_i2, col_i3, col_i4 = st.columns(4)
col_i1.caption(f"Log: `{os.path.basename(log_file)}`")
col_i2.caption(f"Rows: **{total_steps:,}**")
col_i3.caption(f"Arch: **{run_info.get('arch', '?')}** | Persona: **{run_info.get('persona', '?')}**")
col_i4.caption(f"{'LIVE' if do_refresh else 'PAUSED'}")

if do_refresh:
    selected_idx = total_steps - 1
else:
    selected_idx = st.slider("TIMELINE", 0, max(0, total_steps - 1), total_steps - 1)

if selected_idx >= len(df):
    st.stop()
row = df.iloc[selected_idx]

# ── 1. Status ──────────────────────────────────────────────────────────────────
c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric("STEP",          int(row.get('step', 0)))
c2.metric("TOTAL REWARD",  f"{row.get('total_reward', 0):.4f}")
c3.metric("CURRENT LEVEL", str(row.get('level', 0)))
c4.metric("LEVELS BEATEN", int(row.get('levels_completed', 0)))
events_up = df.iloc[:selected_idx+1]
wins_so_far   = int((events_up['event']=='WIN').sum())  if 'event' in df.columns else 0
deaths_so_far = int((events_up['event']=='DIED').sum()) if 'event' in df.columns else 0
c5.metric("WINS",   wins_so_far)
c6.metric("DEATHS", deaths_so_far)
st.divider()

# ── 2. Level Progress ──────────────────────────────────────────────────────────
st.subheader("LEVEL PROGRESS")
if 'level' in df.columns and 'event' in df.columns:
    data_so_far = df.iloc[:selected_idx+1]
    # Sort levels by their order of first appearance in the log, not alphabetically
    seen_order = data_so_far['level'].dropna().unique().tolist()
    all_levels  = list(dict.fromkeys(seen_order))  # deduplicated, insertion-ordered
    rows_list   = []
    for lvl in all_levels:
        lvl_rows  = data_so_far[data_so_far['level']==lvl]
        died_rows = lvl_rows[lvl_rows['event']=='DIED']
        wins      = int((lvl_rows['event']=='WIN').sum())
        deaths    = int(len(died_rows))
        visits    = wins + deaths
        win_rate  = (wins/visits*100) if visits > 0 else 0.0
        filled    = int(win_rate/10)
        bar       = "\u2588"*filled + "\u2591"*(10-filled)
        d_stall   = int((died_rows['cause']=='Stall').sum())   if 'cause' in df.columns else 0
        d_enemy   = int((died_rows['cause']=='Enemy').sum())   if 'cause' in df.columns else 0
        d_pit     = int((died_rows['cause']=='Pit').sum())     if 'cause' in df.columns else 0
        d_spike   = int((died_rows['cause']=='Spike').sum())   if 'cause' in df.columns else 0
        d_timeout = int((died_rows['cause']=='Timeout').sum()) if 'cause' in df.columns else 0
        rows_list.append({"LEVEL": str(lvl),"VISITS":visits,"WINS":wins,"DEATHS":deaths,
            "STALL":d_stall,"ENEMY":d_enemy,"PIT":d_pit,"SPIKE":d_spike,"TIMEOUT":d_timeout,
            "WIN RATE":f"{win_rate:.1f}%","PROGRESS":bar})

    if rows_list:
        level_table = pd.DataFrame(rows_list)
        death_cause_cols = {"STALL","ENEMY","PIT","SPIKE","TIMEOUT"}
        def _style_row(r):
            try: wr = float(r["WIN RATE"].replace("%",""))
            except: wr = 0.0
            wr_color = "#27ae60" if wr>=90 else ("#e67e22" if wr>=50 else "#e74c3c")
            styles=[]
            for c in r.index:
                if c=="WIN RATE": styles.append(f"color:{wr_color}")
                elif c in death_cause_cols:
                    val=int(r[c]) if r[c] else 0
                    styles.append("color:#e74c3c" if val>0 else "color:#555555")
                else: styles.append("")
            return styles
        styled=(level_table.style.apply(_style_row,axis=1)
            .set_properties(**{"font-family":"Courier New, monospace","font-size":"14px","text-align":"center"})
            .set_table_styles([
                {"selector":"th","props":[("background-color","#2a0a1a"),("color","#B3C8EF"),("font-family","Courier New, monospace"),("font-size","12px"),("letter-spacing","1px"),("text-align","center"),("padding","8px 16px"),("border-bottom","1px solid #801830")]},
                {"selector":"td","props":[("background-color","#1a0510"),("color","#B3C8EF"),("padding","7px 16px"),("border-bottom","1px solid #330020")]},
                {"selector":"tr:hover td","props":[("background-color","#2d0a1e")]}])
            .hide(axis="index"))
        st.markdown(styled.to_html(), unsafe_allow_html=True)
        st.caption("")
        total_visits=sum(r["VISITS"] for r in rows_list)
        total_wins=sum(r["WINS"] for r in rows_list)
        total_deaths=sum(r["DEATHS"] for r in rows_list)
        overall_wr=(total_wins/total_visits*100) if total_visits>0 else 0.0
        s1,s2,s3,s4=st.columns(4)
        s1.metric("TOTAL VISITS",total_visits)
        s2.metric("TOTAL WINS",total_wins)
        s3.metric("TOTAL DEATHS",total_deaths)
        s4.metric("OVERALL WIN %",f"{overall_wr:.1f}%")
    else:
        st.caption("No level data yet.")
    st.caption(f"\u21b3 Current action: **{str(row.get('action','N/A'))}**")
else:
    st.caption("No level/event data available.")
st.divider()

# ── 3. Reward Composition ──────────────────────────────────────────────────────
st.subheader("REWARD COMPOSITION")
if reward_cols:
    cols=st.columns(len(reward_cols))
    for i,col_name in enumerate(reward_cols):
        val=row[col_name]
        label=col_name.upper()
        if abs(val)>0.0001: label=f"   {label}"
        cols[i].metric(label, f"{val:.4f}")
else:
    st.caption("No custom components found.")
st.divider()

# ── 4. Graph + Physics ─────────────────────────────────────────────────────────
col_graph, col_phys = st.columns([3,1])

with col_graph:
    st.subheader("SIGNAL VISUALISER")
    if not show_graph:
        st.info("Graph is OFF — toggle 'Show Signal Graph' in the sidebar to enable it. "
                "Turn off if the dashboard crashes or lags on large runs.")
    else:
        available_signals = ['total_reward'] + reward_cols
        check_cols = st.columns(min(len(available_signals), 8))
        selected_lines = []
        for i, sig_name in enumerate(available_signals):
            short = sig_name.replace("total_","").replace("_"," ").title()
            if check_cols[i%8].checkbox(short, value=True, key=f"chk_{i}"):
                selected_lines.append(sig_name)

        if selected_lines:
            start_viz = 0 if show_full_history else max(0, selected_idx-5000)
            subset = df.iloc[start_viz:selected_idx+1][['step']+selected_lines]

            # ── CRASH FIX: Downsample before Plotly renders ───────────────────
            n_raw = len(subset)
            if n_raw > max_plot_pts:
                step_ds = max(1, n_raw // max_plot_pts)
                idx_ds  = list(range(0, n_raw, step_ds))
                if idx_ds[-1] != n_raw-1:
                    idx_ds.append(n_raw-1)
                subset = subset.iloc[idx_ds]

            st.caption(
                f"Showing {len(subset):,} points"
                + (f" (downsampled from {n_raw:,})" if len(subset)<n_raw else "")
            )

            palette=["#FFFFFF","#00FFFF","#FFD700","#FF0055","#00FF00","#BD93F9","#FF8C00","#55FFFF"]
            try:
                fig = px.line(subset, x='step', y=selected_lines,
                    color_discrete_sequence=palette, height=480)
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color=THEME['text']),
                    xaxis=dict(showgrid=False, title="Training Steps"),
                    yaxis=dict(showgrid=True, gridcolor=THEME['grid']),
                    legend=dict(orientation="h", y=1.1, title=None),
                    margin=dict(l=40,r=20,t=40,b=40),
                )
                st.plotly_chart(fig, width='stretch')
            except Exception as e:
                st.warning(f"Graph render failed: {e}. Try reducing 'Max graph points'.")
        else:
            st.caption("No signals selected.")

with col_phys:
    st.subheader("PHYSICS")
    st.caption("Live Telemetry")
    pc1,pc2=st.columns(2)
    with pc1:
        st.metric("X POS",f"{row.get('x',0):.1f}")
        st.metric("VEL X",f"{row.get('vx',0):.2f}")
    with pc2:
        st.metric("Y POS",f"{row.get('y',0):.1f}")
        st.metric("VEL Y",f"{row.get('vy',0):.2f}")
    st.metric("GOAL DIST",f"{row.get('goal_dist',0):.1f}")
    st.divider()
    evt=row.get('event',""); cause=row.get('cause',"")
    if evt and isinstance(evt,str) and evt:
        if evt=="WIN": st.success(f"WIN")
        else: st.error(f"{evt} ({cause})")

# ── 5. Observation Sanity ─────────────────────────────────────────────────────
obs_cols=[c for c in df.columns if c.startswith("grid_") or c.startswith("scalar_") or c in ("dijkstra_val","obs_warnings")]
if obs_cols:
    st.divider()
    st.subheader("OBSERVATION SANITY")
    warn_val=row.get("obs_warnings","")
    if warn_val and isinstance(warn_val,str) and warn_val:
        for w in warn_val.split("|"): st.error(f"[WARN] {w}")
    else:
        st.success("All channels OK")

    grid_channels=["solid","hazard","collectible","dijkstra"]
    has_grid_data=any(f"grid_{ch}_mean" in df.columns for ch in grid_channels)
    if has_grid_data:
        obs_rows=[]
        for ch in grid_channels:
            mean_v=row.get(f"grid_{ch}_mean",0.0); std_v=row.get(f"grid_{ch}_std",0.0)
            min_v=row.get(f"grid_{ch}_min",0.0);  max_v=row.get(f"grid_{ch}_max",0.0)
            status="OK"
            if isinstance(std_v,(int,float)) and std_v<1e-6 and ch == "solid": status="DEAD"
            elif ch != "dijkstra" and isinstance(max_v,(int,float)) and max_v>1.01: status="OVERFLOW"
            obs_rows.append({"CHANNEL":ch.upper(),
                "MEAN":f"{float(mean_v):.4f}" if isinstance(mean_v,(int,float)) else "NULL",
                "STD":f"{float(std_v):.4f}"   if isinstance(std_v,(int,float))  else "NULL",
                "MIN":f"{float(min_v):.4f}"   if isinstance(min_v,(int,float))  else "NULL",
                "MAX":f"{float(max_v):.4f}"   if isinstance(max_v,(int,float))  else "NULL",
                "STATUS":status})
        s_mean=row.get("scalar_mean",0.0); s_std=row.get("scalar_std",0.0)
        s_min=row.get("scalar_min",0.0);   s_max=row.get("scalar_max",0.0)
        s_status="OK"
        if isinstance(s_std,(int,float)) and s_std<1e-8: s_status="DEAD"
        elif isinstance(s_max,(int,float)) and abs(s_max)>100: s_status="UNNORM"
        obs_rows.append({"CHANNEL":"SCALARS",
            "MEAN":f"{float(s_mean):.4f}" if isinstance(s_mean,(int,float)) else "NULL",
            "STD":f"{float(s_std):.4f}"   if isinstance(s_std,(int,float))  else "NULL",
            "MIN":f"{float(s_min):.4f}"   if isinstance(s_min,(int,float))  else "NULL",
            "MAX":f"{float(s_max):.4f}"   if isinstance(s_max,(int,float))  else "NULL",
            "STATUS":s_status})
        dijk=row.get("dijkstra_val",0.0)
        obs_rows.append({"CHANNEL":"DIJK (SCALAR)",
            "MEAN":f"{float(dijk):.4f}" if isinstance(dijk,(int,float)) else "NULL",
            "STD":"-","MIN":"-","MAX":"-",
            "STATUS":"OK" if (isinstance(dijk,(int,float)) and dijk>0) else "ZERO"})
        obs_table=pd.DataFrame(obs_rows)
        def _style_obs(r):
            status=r.get("STATUS","OK"); styles=[]
            for c in r.index:
                if c=="STATUS":
                    if status=="OK": styles.append("color:#27ae60")
                    elif status=="ZERO": styles.append("color:#e67e22")
                    else: styles.append("color:#e74c3c;font-weight:bold")
                else: styles.append("")
            return styles
        styled_obs=(obs_table.style.apply(_style_obs,axis=1)
            .set_properties(**{"font-family":"Courier New, monospace","font-size":"13px","text-align":"center"})
            .set_table_styles([
                {"selector":"th","props":[("background-color","#2a0a1a"),("color","#B3C8EF"),("font-family","Courier New, monospace"),("font-size","11px"),("text-align","center"),("padding","6px 12px"),("border-bottom","1px solid #801830")]},
                {"selector":"td","props":[("background-color","#1a0510"),("color","#B3C8EF"),("padding","5px 12px"),("border-bottom","1px solid #330020")]}])
            .hide(axis="index"))
        st.markdown(styled_obs.to_html(), unsafe_allow_html=True)

# ── Auto-refresh — bottom of page, 2s interval ─────────────────────────────────
if do_refresh and not st.session_state.get("force_stopped", False):
    time.sleep(2)
    try:
        st.rerun()
    except Exception:
        pass