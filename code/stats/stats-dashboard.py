import glob
import os
import yaml
import streamlit as st
import pandas as pd

# =========================================================
# CONFIG
# =========================================================

RESULTS_DIR = "code/stats/results"
THRESHOLDS_PATH = "code/stats/MarioThresholds.yaml"

# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="Balance Dashboard",
    layout="wide"
)

st.title("Balance Dashboard")

# =========================================================
# LOAD CSV FILES
# =========================================================

csv_files = glob.glob(os.path.join(RESULTS_DIR, "*.csv"))

if not csv_files:
    st.error(f"No CSV files found in {RESULTS_DIR}")
    st.stop()

df = pd.concat(
    [
        pd.read_csv(path).assign(
            source_file=os.path.basename(path)
        )
        for path in csv_files
    ],
    ignore_index=True
)

# =========================================================
# LOAD THRESHOLDS
# =========================================================

with open(THRESHOLDS_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

metrics_config = config["metrics"]

b1_t = metrics_config["B1_fairness"]["thresholds"]
b2_t = metrics_config["B2_challenge_success"]["thresholds"]

# =========================================================
# PREP DATA
# =========================================================

df["won"] = df["cause_of_death"] == "Success"

# =========================================================
# SIDEBAR FILTERS
# =========================================================

st.sidebar.header("Filters")

games = sorted(df["game"].dropna().unique())
selected_games = st.sidebar.multiselect(
    "Games",
    games,
    default=games
)

worlds = sorted(df["world"].dropna().unique())
selected_worlds = st.sidebar.multiselect(
    "Worlds",
    worlds,
    default=worlds
)

players = sorted(df["player"].dropna().unique())
selected_players = st.sidebar.multiselect(
    "Players",
    players,
    default=players
)

filtered_df = df[
    df["game"].isin(selected_games)
    & df["world"].isin(selected_worlds)
    & df["player"].isin(selected_players)
].copy()

if filtered_df.empty:
    st.warning("No data after filtering.")
    st.stop()

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def percent(v):
    return f"{v * 100:.0f}%"

def pp(v):
    return f"{v * 100:.0f} pp"

def metric_card(title, value, subtitle="", status=None):
    colors = {
        "balanced": "#1f8f3a",
        "warning": "#a87b00",
        "imbalance": "#992f2f",
    }

    status_html = ""
    if status:
        status_html = (
            f'<span style="display:inline-block;background-color:{colors.get(status, "#555")};'
            f'color:white;padding:6px 10px;border-radius:8px;font-size:0.8rem;margin-top:10px;">'
            f'{status}</span>'
        )

    html = (
        f'<div style="background-color:#262626;border-radius:12px;padding:18px;'
        f'border:1px solid #333;min-height:150px;">'
        f'<div style="font-size:0.9rem;color:#aaaaaa;margin-bottom:8px;">{title}</div>'
        f'<div style="font-size:2rem;font-weight:700;color:white;">{value}</div>'
        f'<div style="color:#aaaaaa;margin-top:6px;margin-bottom:8px;">{subtitle}</div>'
        f'{status_html}'
        f'</div>'
    )

    st.markdown(html, unsafe_allow_html=True)
# =========================================================
# OVERALL STATS
# =========================================================

total_attempts = len(filtered_df)

overall_win_rate = filtered_df["won"].mean()

overall_avg_progress = filtered_df["progress_ratio"].mean()

overall_avg_time = filtered_df["elapsed_time"].mean()

st.caption(
    f"{total_attempts} attempts • "
    f"{len(selected_worlds)} worlds • "
    f"{len(selected_players)} personas"
)

top1, top2, top3, top4 = st.columns(4)

top1.metric("Attempts", total_attempts)

top2.metric(
    "Win Rate",
    percent(overall_win_rate)
)

top3.metric(
    "Average Progress",
    percent(overall_avg_progress)
)

top4.metric(
    "Average Time",
    f"{overall_avg_time:.2f}s"
)

st.divider()

# =========================================================
# WIN RATES
# =========================================================

win_rates = filtered_df.groupby("player")["won"].mean()

best_player = win_rates.idxmax()
worst_player = win_rates.idxmin()

best_win_rate = win_rates[best_player]
worst_win_rate = win_rates[worst_player]

fairness_gap = best_win_rate - worst_win_rate

# =========================================================
# PROGRESS
# =========================================================

progress_by_player = (
    filtered_df.groupby("player")["progress_ratio"]
    .mean()
)

best_progress_player = progress_by_player.idxmax()

worst_progress_player = progress_by_player.idxmin()

best_progress = progress_by_player[best_progress_player]

worst_progress = progress_by_player[worst_progress_player]

average_progress = filtered_df["progress_ratio"].mean()

# =========================================================
# B1 STATUS
# =========================================================

if fairness_gap < b1_t["warning_gap_low"]:
    b1_status = "imbalance"
    b1_message = (
        "Skill gap is too small. "
        "The level may not distinguish skill."
    )

elif fairness_gap > b1_t["warning_gap_high"]:
    b1_status = "imbalance"
    b1_message = (
        "Skill gap is extremely large. "
        "The level may be unfair to weaker players."
    )

elif (
    b1_t["target_gap_min"]
    <= fairness_gap
    <= b1_t["target_gap_max"]
):
    b1_status = "balanced"
    b1_message = (
        "The level discriminates skill within "
        "the target range."
    )

else:
    b1_status = "warning"
    b1_message = (
        "The skill gap is outside the preferred range."
    )

if worst_win_rate < b1_t["novice_min_win_rate"]:
    b1_status = "warning"
    b1_message += (
        " Worst persona win rate is below the target minimum."
    )

if best_win_rate > b1_t["expert_max_win_rate"]:
    b1_status = "warning"
    b1_message += (
        " Best persona win rate is above the target maximum."
    )

# =========================================================
# B2 STATUS
# =========================================================

if (
    average_progress
    < b2_t["warning_average_progress_low"]
):
    b2_status = "imbalance"
    b2_message = (
        "Average progress is too low. "
        "The level may be too difficult."
    )

elif (
    average_progress
    > b2_t["warning_average_progress_high"]
):
    b2_status = "warning"
    b2_message = (
        "Average progress is very high. "
        "The level may be too easy."
    )

elif (
    best_progress
    >= b2_t["target_best_progress_min"]
    and worst_progress
    >= b2_t["target_worst_progress_min"]
):
    b2_status = "balanced"
    b2_message = (
        "Progress spread across personas is healthy."
    )

else:
    b2_status = "warning"
    b2_message = (
        "Progress spread is outside the target range."
    )

# =========================================================
# PLAYER SUMMARY
# =========================================================

st.subheader("Win Rate by Persona")

player_columns = st.columns(max(1, len(selected_players)))

for col, player in zip(player_columns, selected_players):

    player_df = filtered_df[
        filtered_df["player"] == player
    ]

    player_win_rate = player_df["won"].mean()

    player_progress = (
        player_df["progress_ratio"].mean()
    )

    with col:
        metric_card(
            title=player,
            value=percent(player_win_rate),
            subtitle=(
                f"avg progress "
                f"{percent(player_progress)}"
            ),
            status="balanced"
        )

st.divider()

# =========================================================
# BALANCE METRICS
# =========================================================

st.subheader("Balance Metrics")

b1_col, b2_col = st.columns(2)

with b1_col:
    metric_card(
        title="B1 — Fairness",
        value=pp(fairness_gap),
        subtitle=(
            f"{worst_player}: {percent(worst_win_rate)} → "
            f"{best_player}: {percent(best_win_rate)}"
        ),
        status=b1_status
    )

with b2_col:
    metric_card(
        title="B2 — Challenge vs. Success",
        value=percent(best_progress),
        subtitle=(
            f"best persona: {best_progress_player}"
        ),
        status=b2_status
    )

st.divider()

# =========================================================
# DETAILS
# =========================================================

left, right = st.columns(2)

# ---------------------------------------------------------
# B1 DETAILS
# ---------------------------------------------------------

with left:

    st.subheader("B1 — Fairness")

    st.write(
        f"**Current gap:** "
        f"{pp(fairness_gap)}"
    )

    st.write(
        f"**Target gap:** "
        f"{pp(b1_t['target_gap_min'])}–"
        f"{pp(b1_t['target_gap_max'])}"
    )

    st.write(b1_message)

    fairness_table = (
        filtered_df.groupby("player")
        .agg(
            attempts=("player", "count"),
            win_rate=("won", "mean"),
            avg_progress=("progress_ratio", "mean"),
            avg_time=("elapsed_time", "mean"),
        )
        .reset_index()
    )

    fairness_table["win_rate"] = (
        fairness_table["win_rate"] * 100
    ).round(2)

    fairness_table["avg_progress"] = (
        fairness_table["avg_progress"] * 100
    ).round(2)

    fairness_table["avg_time"] = (
        fairness_table["avg_time"]
    ).round(2)

    st.dataframe(
        fairness_table,
        width="stretch"
    )

# ---------------------------------------------------------
# B2 DETAILS
# ---------------------------------------------------------

with right:

    st.subheader("B2 — Challenge vs. Success")

    st.write(
        f"**Best persona:** "
        f"{best_progress_player}"
    )

    st.write(
        f"**Best progress:** "
        f"{percent(best_progress)}"
    )

    st.write(
        f"**Worst progress:** "
        f"{percent(worst_progress)}"
    )

    st.write(
        f"**Average progress:** "
        f"{percent(average_progress)}"
    )

    st.write(
        f"**Target best progress:** "
        f"{percent(b2_t['target_best_progress_min'])}"
    )

    st.write(b2_message)

    progress_table = (
        filtered_df.groupby("player")
        .agg(
            avg_progress=("progress_ratio", "mean"),
            avg_time=("elapsed_time", "mean"),
            win_rate=("won", "mean"),
        )
        .reset_index()
    )

    progress_table["avg_progress"] = (
        progress_table["avg_progress"] * 100
    ).round(2)

    progress_table["win_rate"] = (
        progress_table["win_rate"] * 100
    ).round(2)

    progress_table["avg_time"] = (
        progress_table["avg_time"]
    ).round(2)

    st.dataframe(
        progress_table,
        width="stretch"
    )

st.divider()

# =========================================================
# CHARTS
# =========================================================

chart1, chart2 = st.columns(2)

with chart1:
    st.subheader("Progress by Persona")
    st.bar_chart(progress_by_player)

with chart2:
    st.subheader("Win Rate by Persona")
    st.bar_chart(win_rates)

# =========================================================
# RAW DATA
# =========================================================

st.subheader("Detailed Attempts")

st.dataframe(
    filtered_df,
    width="stretch"
)