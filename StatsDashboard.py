import streamlit as st
import pandas as pd

# titles and initializations
st.set_page_config(page_title="Game Stats Dashboard", layout="wide")

st.title("Game Stats Dashboard")
st.caption("Interactive dashboard for level outcomes, death causes, and player behavior.")

deaths_df = pd.read_csv("level_clear_rate.csv")
attempts_df = pd.read_csv("stats.csv")

# Filters
st.sidebar.header("Filters")

levels = sorted(attempts_df["level"].unique(), key=lambda x: tuple(map(int, x.split("-"))))
selected_levels = st.sidebar.multiselect("Select levels", levels, default=levels)

filtered_df = attempts_df[
    attempts_df["level"].isin(selected_levels)
].copy()


# metrics visualization
total_attempts = len(filtered_df)
success_count = (filtered_df["cause_of_death"] == "Success").sum()
death_count = total_attempts - success_count
success_rate = (success_count / total_attempts * 100) if total_attempts else 0
avg_time = filtered_df["elapsed_time"].mean() if total_attempts else 0
avg_velocity = filtered_df["avg_horizontal_velocity"].mean() if total_attempts else 0

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Attempts", total_attempts)
col2.metric("Successes", success_count)
col3.metric("Deaths", death_count)
col4.metric("Success Rate", f"{success_rate:.1f}%")
col5.metric("Avg Time", f"{avg_time:.2f}s")
col6.metric("Avg Horizontal Velocity", f"{avg_velocity:.2f}")

# Level table
st.subheader("Level Table")

level_table = filtered_df.groupby("level").agg(
    attempts=("level", "count"),
    successes=("cause_of_death", lambda x: (x == "Success").sum()),
    avg_jumps=("jumps", "mean"),
    avg_coins=("coins_collected", "mean"),
    avg_enemies_killed=("enemies_killed", "mean"),
    avg_time=("elapsed_time", "mean"),
    avg_velocity=("avg_horizontal_velocity", "mean"),
    avg_idle_ratio=("Idle_Ratio", "mean"),
    avg_actions_per_second=("Actions_Per_Second", "mean"),
).reset_index()

level_table["success_rate"] = (
    level_table["successes"] / level_table["attempts"] * 100
).round(2)

st.dataframe(level_table, use_container_width=True)

# Attempts and Death Distribution Charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("Attempts by Level")
    attempts_per_level = filtered_df["level"].value_counts().sort_index()
    st.bar_chart(attempts_per_level)

with col2:
    st.subheader("Death Distribution")
    death_counts = (
        filtered_df[filtered_df["cause_of_death"] != "Success"]
        ["cause_of_death"]
        .value_counts())
    st.bar_chart(death_counts)

# Average Time and Velocity Charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("Average Elapsed Time by Level")
    avg_time_by_level = filtered_df.groupby("level")["elapsed_time"].mean().sort_index()
    st.bar_chart(avg_time_by_level)

with col2:
    st.subheader("Average Horizontal Velocity by Level")
    avg_velocity_by_level = filtered_df.groupby("level")["avg_horizontal_velocity"].mean().sort_index()
    st.bar_chart(avg_velocity_by_level)

# Death Summary
st.subheader("Death Summary from Aggregated CSV")
st.dataframe(deaths_df, use_container_width=True)

death_breakdown = deaths_df.set_index("level")[["Pit", "enemy", "stall"]]
st.bar_chart(death_breakdown)

# Action Usage Table
st.subheader("Action Analysis")

action_cols = [col for col in filtered_df.columns if col.startswith("action_")]

if not filtered_df.empty:
    action_totals = filtered_df[action_cols].sum().sort_values(ascending=False)

    action_by_level = filtered_df.groupby("level")[action_cols].sum()
    st.subheader("Action Usage by Level")
    st.dataframe(action_by_level, use_container_width=True)

    st.bar_chart(action_totals)

# Detailed attempts Table

st.subheader("Detailed Attempts")
st.dataframe(filtered_df, use_container_width=True)

# Quick Insights
st.subheader("Quick Insights")

if total_attempts > 0:
    most_common_outcome = filtered_df["cause_of_death"].value_counts().idxmax()
    hardest_level = (
        filtered_df[filtered_df["cause_of_death"] != "Success"]["level"]
        .value_counts()
        .idxmax()
        if not filtered_df[filtered_df["cause_of_death"] != "Success"].empty
        else "N/A"
    )

    st.write(f"**Most common outcome:** {most_common_outcome}")
    st.write(f"**Level with most failures:** {hardest_level}")

    if "Success" in filtered_df["cause_of_death"].values:
        success_df = filtered_df[filtered_df["cause_of_death"] == "Success"]
        st.write(f"**Average time on successful runs:** {success_df['elapsed_time'].mean():.2f}s")
        st.write(f"**Average idle ratio on successful runs:** {success_df['Idle_Ratio'].mean():.2f}")
