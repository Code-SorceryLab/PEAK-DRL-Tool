# -*- coding: utf-8 -*-
"""PEAK Level Analyzer -- Streamlit entry point.

Run with:  streamlit run code/stats/dashboard/app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

from data import CONFIG_PATH, load_config, load_all_csvs, win_rate
from metrics import compute_world_metrics
from components import PERSONA_COLORS, card_html
from panels import (
    render_b1_detail, render_b2_detail, render_b3_detail,
    render_m4_detail, render_m5_detail, render_m6_detail,
    render_route_viz,
)

st.set_page_config(page_title="Peak Level Analyzer", layout="wide")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

  .stApp, .stApp > div, section[data-testid="stAppViewContainer"] {
    background-color: #1a1a1a !important;
    color: #e0e0e0 !important;
    font-family: 'IBM Plex Sans', sans-serif;
  }
  .block-container { padding: 2rem 2.5rem 2rem 2.5rem !important; max-width: 1100px !important; }

  #MainMenu, footer, header { visibility: hidden; }

  .stSelectbox > div > div {
    background-color: #232323 !important;
    border: 1px solid #3a3a3a !important;
    color: #e0e0e0 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.85rem !important;
  }

  .header-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #e5e5e5;
    margin-right: 4px;
  }
  .header-meta {
    font-size: 0.78rem;
    color: #666;
    font-family: 'IBM Plex Mono', monospace;
  }
  .score-badge {
    background: #2a1a1a;
    border: 1px solid #7f1d1d;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.82rem;
    color: #fca5a5;
    font-family: 'IBM Plex Mono', monospace;
    white-space: nowrap;
  }
  .score-badge span { color: #ef4444; font-weight: 600; }

  .persona-row { display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap; }
  .persona-card {
    background: #232323;
    border: 1px solid #2e2e2e;
    border-radius: 6px;
    padding: 12px 16px;
    flex: 1;
    min-width: 120px;
  }
  .persona-label { font-size: 0.72rem; color: #777; margin-bottom: 2px; }
  .persona-sublabel { font-size: 0.78rem; color: #aaa; margin-bottom: 6px; font-weight: 500; }
  .persona-pct { font-size: 1.6rem; font-weight: 600; font-family: 'IBM Plex Mono', monospace; }
  .persona-bar {
    height: 3px;
    border-radius: 2px;
    margin-top: 6px;
    max-width: 100%;
  }

  .section-label {
    font-size: 0.78rem;
    color: #666;
    margin-bottom: 12px;
    margin-top: 4px;
  }

  hr { border-color: #2e2e2e !important; }
  .sep { border-top: 1px solid #2e2e2e; margin: 20px 0 16px 0; }

  .analysis-box {
    background: #232323;
    border: 1px solid #333333;
    border-radius: 6px;
    padding: 16px 20px;
    margin-top: 16px;
  }
  .analysis-tag {
    display: inline-block;
    background: #3d3300;
    border: 1px solid #a16207;
    color: #facc15;
    font-size: 0.72rem;
    font-weight: 600;
    border-radius: 3px;
    padding: 2px 8px;
    margin-right: 8px;
    vertical-align: middle;
    font-family: 'IBM Plex Mono', monospace;
  }
  .analysis-title { font-size: 0.92rem; font-weight: 600; color: #e5e5e5; display: inline; vertical-align: middle; }
  .analysis-body { font-size: 0.83rem; color: #b0b0b0; margin-top: 10px; line-height: 1.6; }
  .analysis-italic { font-size: 0.80rem; color: #777777; font-style: italic; margin-top: 10px; line-height: 1.6; }

  .no-data-box {
    background: #232323;
    border: 1px solid #3a3a3a;
    border-radius: 6px;
    padding: 20px;
    color: #666;
    font-size: 0.85rem;
    font-family: 'IBM Plex Mono', monospace;
    text-align: center;
    margin: 10px 0;
  }
</style>
""", unsafe_allow_html=True)

try:
    config = load_config(CONFIG_PATH)
except FileNotFoundError:
    st.error("Config file not found: %s" % CONFIG_PATH)
    st.stop()

data_path = config.get("path", "code/stats/results/")
metrics_cfg = config.get("metrics", {})
df_all = load_all_csvs(data_path)

if df_all.empty:
    st.warning("No CSV data found in `%s`." % data_path)
    st.stop()

worlds = sorted(df_all["world"].unique().tolist())

if "selected_metric" not in st.session_state:
    st.session_state.selected_metric = list(metrics_cfg.keys())[0] if metrics_cfg else "B1_challenge_calibration"
if "selected_world" not in st.session_state:
    st.session_state.selected_world = worlds[0] if worlds else None

world = st.session_state.selected_world
world_metrics = compute_world_metrics(world, metrics_cfg, df_all)

total_score = sum(
    m.get("score_pt", 0) for m in world_metrics.values() if "error" not in m
)
max_score = len([m for m in world_metrics.values() if "error" not in m])

st.markdown(
    '<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:18px;">'
    '<div>'
    '<div class="header-title">Balance dashboard -- Peak</div>'
    '<div class="header-meta">Level: %s -- data path: %s</div>'
    '</div>'
    '<div class="score-badge">balance score <span>%d</span> / %d</div>'
    '</div>' % (world, data_path, total_score, max_score),
    unsafe_allow_html=True,
)

st.markdown('<div class="sep"></div>', unsafe_allow_html=True)

st.markdown('<div class="section-label">select level</div>', unsafe_allow_html=True)
selected_world = st.selectbox(
    label="level",
    options=worlds,
    index=worlds.index(st.session_state.selected_world) if st.session_state.selected_world in worlds else 0,
    label_visibility="collapsed",
)
if selected_world != st.session_state.selected_world:
    st.session_state.selected_world = selected_world
    st.rerun()

st.markdown('<div style="height:14px;"></div>', unsafe_allow_html=True)

st.markdown('<div class="section-label">win rate by persona</div>', unsafe_allow_html=True)
personas_in_world = sorted(df_all[df_all["world"] == world]["persona"].unique().tolist())
cards_html = '<div class="persona-row">'
for i, persona in enumerate(personas_in_world):
    color = PERSONA_COLORS[i % len(PERSONA_COLORS)]
    wr = win_rate(df_all, world, persona)
    if wr is None:
        pct_display, bar_w = "N/A", 0
    else:
        pct_val = round(wr * 100, 1)
        pct_display, bar_w = "%s%%" % pct_val, pct_val
    cards_html += (
        '<div class="persona-card">'
        '<div class="persona-sublabel">%s</div>'
        '<div class="persona-pct" style="color:%s;">%s</div>'
        '<div class="persona-bar" style="background:%s;width:%s%%;"></div>'
        '</div>' % (persona, color, pct_display, color, bar_w)
    )
cards_html += '</div>'
st.markdown(cards_html, unsafe_allow_html=True)

st.markdown(
    '<div class="section-label">%d balance metric(s) -- click a card to inspect</div>'
    % len(world_metrics),
    unsafe_allow_html=True,
)

selected_metric = st.session_state.selected_metric
metric_keys = list(world_metrics.keys())
sel_idx = metric_keys.index(selected_metric) if selected_metric in metric_keys else 0

st.markdown("""
<style>
  div[data-testid="stColumn"] div[data-testid="stButton"] {
    margin-top: 0 !important;
  }
  div[data-testid="stColumn"] div[data-testid="stButton"] > button {
    width: 100%% !important;
    border-radius: 0 0 8px 8px !important;
    border-top: 1px solid #3a3a3a !important;
    background: #1e1e1e !important;
    font-size: 0.75rem !important;
    color: #555 !important;
    padding: 7px !important;
    letter-spacing: 0.04em !important;
  }
  div[data-testid="stColumn"] div[data-testid="stButton"] > button:hover {
    background: #2a2a2a !important;
    color: #aaa !important;
    border-color: #555 !important;
  }
  div[data-testid="stColumn"]:nth-child(%d) div[data-testid="stButton"] > button {
    border-left: 1px solid #4a9eff !important;
    border-right: 1px solid #4a9eff !important;
    border-bottom: 1px solid #4a9eff !important;
  }
</style>
""" % (sel_idx + 1), unsafe_allow_html=True)

col_weights = [1] * len(metric_keys) + [max(1, 4 - len(metric_keys))]
cols = st.columns(col_weights)

for i, metric_key in enumerate(metric_keys):
    m = world_metrics[metric_key]
    with cols[i]:
        if "error" in m:
            st.markdown(
                '<div style="background:#232323;border:1px solid #2e2e2e;border-radius:8px;'
                'padding:14px 16px;min-height:130px;color:#666;font-size:0.75rem;'
                "font-family:'IBM Plex Mono',monospace;\">"
                '%s<br><br>%s</div>' % (metric_key, m["error"]),
                unsafe_allow_html=True,
            )
        elif m["type"] == "B1":
            cr_pct = round(m["completion_rate"] * 100, 1)
            mct = m["mean_completion_time"]
            mct_str = "%.1fs" % mct if mct is not None else "N/A"
            st.markdown(card_html(
                metric_key, "Challenge calibration",
                "%.1f%%" % cr_pct, m["zone_color"],
                "completion rate -- avg time %s" % mct_str,
                m["zone_pill"], m["zone_key"].replace("-", " ").title(),
                selected_metric == metric_key,
            ), unsafe_allow_html=True)
            if st.button("Analyze %s" % metric_key, key="btn_%s" % metric_key, use_container_width=True):
                st.session_state.selected_metric = metric_key
                st.rerun()
        elif m["type"] == "B2":
            dpr = round(m["deaths_per_run"], 2)
            ent = round(m["death_cluster_entropy"], 2)
            st.markdown(card_html(
                metric_key, "Punishment severity",
                str(dpr), m["b2_color"],
                "deaths/run -- entropy %s" % ent,
                m["b2_pill"], m["b2_key"].replace("-", " ").title(),
                selected_metric == metric_key,
            ), unsafe_allow_html=True)
            if st.button("Analyze %s" % metric_key, key="btn_%s" % metric_key, use_container_width=True):
                st.session_state.selected_metric = metric_key
                st.rerun()
        elif m["type"] == "B3":
            sc = m["strategy_count"]
            ds = round(m["dominant_path_share"] * 100, 1)
            st.markdown(card_html(
                metric_key, "Strategy diversity",
                str(sc), m["b3_color"],
                "strategies -- %s%% dominant" % ds,
                m["b3_pill"], m["b3_key"].replace("-", " ").title(),
                selected_metric == metric_key,
            ), unsafe_allow_html=True)
            if st.button("Analyze %s" % metric_key, key="btn_%s" % metric_key, use_container_width=True):
                st.session_state.selected_metric = metric_key
                st.rerun()
        elif m["type"] == "M4":
            tc = m["time_cost"]
            tc_str = "%.2fx" % tc if tc is not None else "N/A"
            st.markdown(card_html(
                metric_key, "Risk-reward",
                tc_str, m["m4_color"],
                "time cost -- %.2fx death premium" % m["death_premium"],
                m["m4_pill"], m["m4_key"].replace("-", " ").title(),
                selected_metric == metric_key,
            ), unsafe_allow_html=True)
            if st.button("Analyze %s" % metric_key, key="btn_%s" % metric_key, use_container_width=True):
                st.session_state.selected_metric = metric_key
                st.rerun()
        elif m["type"] == "M5":
            ne = m["novice_expert_gap"]
            st.markdown(card_html(
                metric_key, "Skill expression",
                "%d%%" % round(ne * 100), m["m5_color"],
                "novice-expert gap",
                m["m5_pill"], m["m5_key"].replace("-", " ").title(),
                selected_metric == metric_key,
            ), unsafe_allow_html=True)
            if st.button("Analyze %s" % metric_key, key="btn_%s" % metric_key, use_container_width=True):
                st.session_state.selected_metric = metric_key
                st.rerun()
        elif m["type"] == "M6":
            n_ok = len(m["rates"])
            n_total = n_ok + len(m["missing_skills"])
            st.markdown(card_html(
                metric_key, "Progression fit",
                "%d/%d" % (n_ok, n_total), m["m6_color"],
                "skill tiers tracked",
                m["m6_pill"], m["m6_key"].replace("-", " ").title(),
                selected_metric == metric_key,
            ), unsafe_allow_html=True)
            if st.button("Analyze %s" % metric_key, key="btn_%s" % metric_key, use_container_width=True):
                st.session_state.selected_metric = metric_key
                st.rerun()

st.markdown('<div class="sep"></div>', unsafe_allow_html=True)

if selected_metric not in world_metrics:
    st.markdown(
        '<div class="no-data-box">Select a metric above to inspect.</div>',
        unsafe_allow_html=True,
    )
else:
    m = world_metrics[selected_metric]
    if "error" in m:
        st.markdown(
            '<div class="no-data-box">%s</div>' % m["error"],
            unsafe_allow_html=True,
        )
    elif m["type"] == "B1":
        render_b1_detail(m)
    elif m["type"] == "B2":
        render_b2_detail(m)
    elif m["type"] == "B3":
        render_b3_detail(m)
    elif m["type"] == "M4":
        render_m4_detail(m)
    elif m["type"] == "M5":
        render_m5_detail(m)
    elif m["type"] == "M6":
        render_m6_detail(m)

render_route_viz(world, df_all)
