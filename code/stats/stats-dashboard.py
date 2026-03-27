import streamlit as st
import pandas as pd

st.set_page_config(page_title="Game Stats Dashboard", layout="wide")
st.title("Game Stats Dashboard")
st.caption("Interactive dashboard for run outcomes and player behavior.")

df = pd.read_csv("stats.csv")

st.sidebar.header("Filters")

worlds = sorted(df["world"].dropna().unique())
selected_worlds = st.sidebar.multiselect("Select worlds", worlds, default=worlds)

outcomes = sorted(df["cause_of_death"].dropna().unique())
selected_outcomes = st.sidebar.multiselect("Select outcomes", outcomes, default=outcomes)

filtered_df = df[
    df["world"].isin(selected_worlds) &
    df["cause_of_death"].isin(selected_outcomes)
].copy()

total_attempts = len(filtered_df)
success_count = (filtered_df["cause_of_death"] == "Success").sum()
death_count = total_attempts - success_count
success_rate = (success_count / total_attempts * 100) if total_attempts else 0
avg_time = filtered_df["elapsed_time"].mean() if total_attempts else 0
avg_velocity = filtered_df["avg_vx"].mean() if total_attempts else 0

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Attempts", total_attempts)
col2.metric("Successes", success_count)
col3.metric("Deaths", death_count)
col4.metric("Success Rate", f"{success_rate:.1f}%")
col5.metric("Avg Time", f"{avg_time:.2f}s")
col6.metric("Avg Velocity", f"{avg_velocity:.2f}")

st.subheader("World Summary")

world_table = filtered_df.groupby("world").agg(
    attempts=("world", "count"),
    successes=("cause_of_death", lambda x: (x == "Success").sum()),
    avg_jumps=("jump_count", "mean"),
    avg_coins=("coins_collected", "mean"),
    avg_enemies_killed=("enemies_killed", "mean"),
    avg_time=("elapsed_time", "mean"),
    avg_vx=("avg_vx", "mean"),
).reset_index()

world_table["success_rate"] = (
    world_table["successes"] / world_table["attempts"] * 100
).round(2)

st.dataframe(world_table, use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Attempts by World")
    attempts_per_world = filtered_df["world"].value_counts().sort_index()
    st.bar_chart(attempts_per_world)

with col2:
    st.subheader("Outcome Distribution")
    outcome_counts = filtered_df["cause_of_death"].value_counts()
    st.bar_chart(outcome_counts)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Average Elapsed Time by World")
    avg_time_by_world = filtered_df.groupby("world")["elapsed_time"].mean().sort_index()
    st.bar_chart(avg_time_by_world)

with col2:
    st.subheader("Average Velocity by World")
    avg_vx_by_world = filtered_df.groupby("world")["avg_vx"].mean().sort_index()
    st.bar_chart(avg_vx_by_world)

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Average Jumps by World")
    jumps_by_world = filtered_df.groupby("world")["jump_count"].mean().sort_index()
    st.bar_chart(jumps_by_world)

with col2:
    st.subheader("Average Coins Collected by World")
    coins_by_world = filtered_df.groupby("world")["coins_collected"].mean().sort_index()
    st.bar_chart(coins_by_world)

with col3:
    st.subheader("Average Enemies Killed by World")
    enemies_by_world = filtered_df.groupby("world")["enemies_killed"].mean().sort_index()
    st.bar_chart(enemies_by_world)

st.subheader("Outcome Breakdown by World")
outcome_breakdown = (
    filtered_df.groupby(["world", "cause_of_death"])
    .size()
    .unstack(fill_value=0)
)
st.dataframe(outcome_breakdown, use_container_width=True)
st.bar_chart(outcome_breakdown)

st.subheader("Detailed Attempts")
st.dataframe(filtered_df, use_container_width=True)

st.subheader("Quick Insights")

if total_attempts > 0:
    most_common_outcome = filtered_df["cause_of_death"].value_counts().idxmax()

    failed_attempts = filtered_df[filtered_df["cause_of_death"] != "Success"]
    hardest_world = (
        failed_attempts["world"].value_counts().idxmax()
        if not failed_attempts.empty
        else "N/A"
    )

    st.write(f"**Most common outcome:** {most_common_outcome}")
    st.write(f"**World with most failures:** {hardest_world}")

    success_df = filtered_df[filtered_df["cause_of_death"] == "Success"]
    if not success_df.empty:
        st.write(f"**Average time on successful runs:** {success_df['elapsed_time'].mean():.2f}s")
        st.write(f"**Average jumps on successful runs:** {success_df['jump_count'].mean():.2f}")
        st.write(f"**Average velocity on successful runs:** {success_df['avg_vx'].mean():.2f}")