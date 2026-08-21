# -*- coding: utf-8 -*-
"""Detail panel renderers and route visualization for the PEAK dashboard."""

import os

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from data import (
    GAME_CONFIG_PATH, LEVELS_ROOT, TILE_SIZE,
    load_game_config, load_level_grid, parse_route,
)
from metrics import (B1_ANALYSIS, B2_ANALYSIS, B3_ANALYSIS,
                      M4_ANALYSIS, M5_ANALYSIS, M6_ANALYSIS)

BAR_W, BAR_H = 800, 36


def _zone_bar_svg(value, value_label, segments, ticks, color, bar_max):
    def to_x(v):
        return round(min(max(v, 0), bar_max) / bar_max * BAR_W)

    seg_rects = ""
    for s, e, col, lbl in segments:
        x1, x2 = to_x(s), to_x(e)
        w = max(x2 - x1, 1)
        seg_rects += '<rect x="%d" y="0" width="%d" height="%d" fill="%s"/>' % (x1, w, BAR_H, col)
        mid = (x1 + x2) // 2
        seg_rects += (
            '<text x="%d" y="%d" text-anchor="middle" font-size="10" '
            'fill="#cccccc" font-family="IBM Plex Sans,sans-serif">%s</text>'
            % (mid, BAR_H // 2 + 5, lbl)
        )

    ticks_svg = ""
    for t_val, t_lbl in ticks:
        tx = to_x(t_val)
        ticks_svg += (
            '<text x="%d" y="52" text-anchor="middle" font-size="10" '
            'fill="#777777" font-family="IBM Plex Mono,monospace">%s</text>'
            % (tx, t_lbl)
        )
        ticks_svg += '<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#555" stroke-width="1"/>' % (
            tx, BAR_H, tx, BAR_H + 6
        )

    mx = to_x(value)
    marker_svg = (
        '<line x1="%d" y1="-4" x2="%d" y2="%d" stroke="%s" stroke-width="2"/>'
        '<circle cx="%d" cy="%d" r="7" fill="%s" opacity="0.9"/>'
        '<rect x="%d" y="-26" width="80" height="20" fill="#1a1a1a" rx="3"/>'
        '<text x="%d" y="-11" text-anchor="middle" font-size="10" fill="#e5e5e5" '
        'font-family="IBM Plex Mono,monospace">%s</text>'
    ) % (mx, mx, BAR_H + 4, color, mx, BAR_H // 2, color, mx - 40, mx, value_label)

    return (
        '<svg viewBox="-10 -35 %d 90" xmlns="http://www.w3.org/2000/svg" '
        'style="width:100%%;max-width:860px;display:block;margin:0 auto;">'
        '<g>%s%s%s</g></svg>'
    ) % (BAR_W + 20, seg_rects, ticks_svg, marker_svg)


def _stat_card(label, value_html, subtitle):
    return (
        '<div style="background:#232323;border:1px solid #2e2e2e;border-radius:6px;padding:14px 16px;">'
        '<div style="font-size:0.78rem;color:#888;margin-bottom:4px;">%s</div>'
        "<div style=\"font-size:2rem;font-weight:600;font-family:'IBM Plex Mono',monospace;\">%s</div>"
        '<div style="font-size:0.72rem;color:#555;">%s</div>'
        '</div>'
    ) % (label, value_html, subtitle)


def _analysis_box(key, analysis_dict):
    data = analysis_dict.get(key, analysis_dict.get("warning"))
    st.markdown(
        '<div class="analysis-box">'
        '<span class="analysis-tag">%s</span>'
        '<span class="analysis-title">%s</span>'
        '<div class="analysis-body">%s</div>'
        '<div class="analysis-italic">%s</div>'
        '</div>' % (key.replace("-", " ").title(), data[0], data[1], data[2]),
        unsafe_allow_html=True,
    )


def render_b1_detail(m):
    cr = m["completion_rate"]
    cr_pct = round(cr * 100, 1)
    mct = m["mean_completion_time"]
    ct_std = m["completion_time_stddev"]
    pad = m["progress_at_death"]
    t = m["thresholds"]
    target_cr      = t.get("target_completion_rate", 0.7)
    warn_cr_diff   = t.get("warning_completion_rate_difference", 0.2)
    target_time    = t.get("target_mean_completion_time", 15)
    warn_time_diff = t.get("warning_mean_completion_time_difference", 4)

    cr_low  = max(target_cr - warn_cr_diff, 0.0)
    cr_high = min(target_cr + warn_cr_diff, 1.0)
    segments = [
        (0.0,     cr_low,  "#7f1d1d", "too hard"),
        (cr_low,  cr_high, "#14532d", "target"),
        (cr_high, 1.0,     "#78350f", "too easy"),
    ]
    ticks = [(v, "%d%%" % round(v * 100)) for v in [0.0, cr_low, target_cr, cr_high, 1.0]]
    bar_svg = _zone_bar_svg(cr, "%.1f%%" % cr_pct, segments, ticks, m["zone_color"], 1.0)
    st.markdown(
        '<div style="font-size:0.75rem;color:#777;margin-bottom:2px;">Completion rate -- target zone</div>',
        unsafe_allow_html=True,
    )
    st.markdown(bar_svg, unsafe_allow_html=True)

    if mct is not None:
        time_lo = max(target_time - warn_time_diff, 0)
        time_hi = target_time + warn_time_diff
        time_max = max(mct * 1.3, time_hi * 1.3, 1)
        t_segments = [
            (0,       time_lo,  "#78350f", "fast"),
            (time_lo, time_hi,  "#14532d", "target"),
            (time_hi, time_max, "#7f1d1d", "slow"),
        ]
        t_ticks = [(v, "%ss" % v) for v in [0, time_lo, target_time, time_hi]]
        t_bar = _zone_bar_svg(mct, "%.1fs" % mct, t_segments, t_ticks, m["zone_color"], time_max)
        st.markdown(
            '<div style="font-size:0.75rem;color:#777;margin-bottom:2px;margin-top:16px;">'
            'Mean completion time -- target zone</div>',
            unsafe_allow_html=True,
        )
        st.markdown(t_bar, unsafe_allow_html=True)

    st.markdown('<div class="sep"></div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(_stat_card(
            "Completion rate",
            '<span style="color:%s;">%.1f%%</span>' % (m["zone_color"], cr_pct),
            "%d of %d runs" % (m["successful_runs"], m["total_runs"]),
        ), unsafe_allow_html=True)
    with c2:
        mct_display = "%.2fs" % mct if mct is not None else "N/A"
        st.markdown(_stat_card(
            "Mean completion time",
            '<span style="color:#3b82f6;">%s</span>' % mct_display,
            "successful runs only",
        ), unsafe_allow_html=True)
    with c3:
        pad_display = "%.1f%%" % (pad * 100) if pad is not None else "N/A"
        st.markdown(_stat_card(
            "Progress at death",
            '<span style="color:#f59e0b;">%s</span>' % pad_display,
            "avg of failed runs",
        ), unsafe_allow_html=True)
    with c4:
        std_display = "%.2fs" % ct_std if ct_std is not None else "N/A"
        st.markdown(_stat_card(
            "Time std dev",
            '<span style="color:#a855f7;">%s</span>' % std_display,
            "completion time spread",
        ), unsafe_allow_html=True)

    _analysis_box(m["zone_key"], B1_ANALYSIS)


def render_b2_detail(m):
    dpr = m["deaths_per_run"]
    entropy = m["death_cluster_entropy"]
    b2_color = m["b2_color"]
    t = m["thresholds"]
    target_dpr     = t.get("target_deaths_per_run", 2)
    warn_dpr       = t.get("warning_deaths_per_run", 1)
    target_entropy = t.get("target_death_cluster_entropy", 5)
    warn_entropy   = t.get("warning_death_cluster_entropy", 2)

    dpr_lo = max(target_dpr - warn_dpr, 0)
    dpr_hi = target_dpr + warn_dpr
    dpr_bar_max = max(dpr * 1.3, dpr_hi * 1.3, 1.0)
    segments = [
        (0,      dpr_lo,      "#78350f", "low"),
        (dpr_lo, dpr_hi,      "#14532d", "target"),
        (dpr_hi, dpr_bar_max, "#7f1d1d", "high"),
    ]
    ticks = [(v, str(v)) for v in [0, dpr_lo, target_dpr, dpr_hi]]
    bar_svg = _zone_bar_svg(dpr, "%.2f" % dpr, segments, ticks, b2_color, dpr_bar_max)
    st.markdown(
        '<div style="font-size:0.75rem;color:#777;margin-bottom:2px;">Deaths per run -- target zone</div>',
        unsafe_allow_html=True,
    )
    st.markdown(bar_svg, unsafe_allow_html=True)

    ent_lo = round(max(target_entropy - warn_entropy, 0), 10)
    ent_hi = round(target_entropy + warn_entropy, 10)
    ent_bar_max = min(max(entropy * 1.3, ent_hi * 1.3, 0.1), 1.0)
    e_segments = [
        (0,      ent_lo,      "#7f1d1d", "clustered"),
        (ent_lo, ent_hi,      "#14532d", "target"),
        (ent_hi, ent_bar_max, "#78350f", "dispersed"),
    ]
    e_ticks = [(v, str(v)) for v in [0, ent_lo, target_entropy, ent_hi]]
    e_bar = _zone_bar_svg(entropy, "%.2f" % entropy, e_segments, e_ticks, b2_color, ent_bar_max)
    st.markdown(
        '<div style="font-size:0.75rem;color:#777;margin-bottom:2px;margin-top:16px;">'
        'Death cluster entropy -- spread of death locations (0 = clustered, 1 = uniform)</div>',
        unsafe_allow_html=True,
    )
    st.markdown(e_bar, unsafe_allow_html=True)
    st.markdown('<div class="sep"></div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(_stat_card(
            "Deaths per run",
            '<span style="color:%s;">%.2f</span>' % (b2_color, dpr),
            "%d deaths / %d successes" % (m["total_deaths"], m["total_successes"]),
        ), unsafe_allow_html=True)
    with c2:
        st.markdown(_stat_card(
            "Death cluster entropy",
            '<span style="color:#3b82f6;">%.2f</span>' % entropy,
            "0 = clustered, 1 = uniform",
        ), unsafe_allow_html=True)
    with c3:
        st.markdown(_stat_card(
            "Total runs",
            '<span style="color:#a855f7;">%d</span>' % m["total_runs"],
            "%d wins, %d deaths" % (m["total_successes"], m["total_deaths"]),
        ), unsafe_allow_html=True)

    _analysis_box(m["b2_key"], B2_ANALYSIS)


def render_b3_detail(m):
    sc = m["strategy_count"]
    ds = m["dominant_path_share"]
    svf = m["safe_vs_fast_ratio"]
    b3_color = m["b3_color"]
    t = m["thresholds"]
    tgt_sc   = t.get("target_strategy_count", 3)
    warn_sc  = t.get("warning_strategy_count", 1)
    tgt_ds   = t.get("target_dominant_path_share", 0.5)
    warn_ds  = t.get("warning_dominant_path_share", 0.15)

    sc_lo = max(tgt_sc - warn_sc, 0)
    sc_hi = tgt_sc + warn_sc
    sc_bar_max = max(sc * 1.3, sc_hi * 1.3, 1)
    segments = [
        (0,     sc_lo,      "#7f1d1d", "few"),
        (sc_lo, sc_hi,      "#14532d", "target"),
        (sc_hi, sc_bar_max, "#78350f", "many"),
    ]
    ticks = [(v, str(v)) for v in [0, sc_lo, tgt_sc, sc_hi]]
    bar_svg = _zone_bar_svg(sc, str(sc), segments, ticks, b3_color, sc_bar_max)
    st.markdown(
        '<div style="font-size:0.75rem;color:#777;margin-bottom:2px;">'
        'Strategy count -- number of distinct routes</div>',
        unsafe_allow_html=True,
    )
    st.markdown(bar_svg, unsafe_allow_html=True)

    ds_lo = round(max(tgt_ds - warn_ds, 0), 10)
    ds_hi = round(min(tgt_ds + warn_ds, 1.0), 10)
    ds_bar_max = min(max(ds * 1.3, ds_hi * 1.3, 0.1), 1.0)
    ds_segments = [
        (0,     ds_lo,      "#14532d", "diverse"),
        (ds_lo, ds_hi,      "#78350f", "warning"),
        (ds_hi, ds_bar_max, "#7f1d1d", "dominant"),
    ]
    ds_ticks = [(v, "%d%%" % round(v * 100)) for v in [0, ds_lo, tgt_ds, ds_hi]]
    ds_pct = round(ds * 100, 1)
    ds_bar = _zone_bar_svg(ds, "%.1f%%" % ds_pct, ds_segments, ds_ticks, b3_color, ds_bar_max)
    st.markdown(
        '<div style="font-size:0.75rem;color:#777;margin-bottom:2px;margin-top:16px;">'
        'Dominant path share -- fraction of runs on the most common route (lower = more diverse)</div>',
        unsafe_allow_html=True,
    )
    st.markdown(ds_bar, unsafe_allow_html=True)
    st.markdown('<div class="sep"></div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(_stat_card(
            "Strategy count",
            '<span style="color:%s;">%d</span>' % (b3_color, sc),
            "distinct route clusters",
        ), unsafe_allow_html=True)
    with c2:
        st.markdown(_stat_card(
            "Dominant path share",
            '<span style="color:#3b82f6;">%.1f%%</span>' % ds_pct,
            "%d of %d runs" % (max(m["cluster_sizes"]), m["total_successful"]),
        ), unsafe_allow_html=True)
    with c3:
        st.markdown(_stat_card(
            "Safe vs fast ratio",
            '<span style="color:#a855f7;">%.2fx</span>' % svf,
            "slowest cluster / fastest cluster",
        ), unsafe_allow_html=True)

    _analysis_box(m["b3_key"], B3_ANALYSIS)


def render_m4_detail(m):
    tc = m["time_cost"]
    dp = m["death_premium"]
    m4_color = m["m4_color"]
    t = m["thresholds"]
    tgt_tc  = t.get("target_time_cost", 1.3)
    warn_tc = t.get("warning_time_cost", 0.2)
    tgt_dp  = t.get("target_death_premium", 1.5)
    warn_dp = t.get("warning_death_premium", 0.3)

    # Time cost bar
    if tc is not None:
        tc_lo = max(tgt_tc - warn_tc, 0)
        tc_hi = tgt_tc + warn_tc
        tc_bar_max = max(tc * 1.3, tc_hi * 1.3, 1)
        segments = [
            (0,     tc_lo,      "#14532d", "free"),
            (tc_lo, tc_hi,      "#78350f", "target"),
            (tc_hi, tc_bar_max, "#7f1d1d", "costly"),
        ]
        ticks = [(v, "%.1fx" % v) for v in [0, tc_lo, tgt_tc, tc_hi]]
        bar_svg = _zone_bar_svg(tc, "%.2fx" % tc, segments, ticks, m4_color, tc_bar_max)
        st.markdown(
            '<div style="font-size:0.75rem;color:#777;margin-bottom:2px;">'
            'Time cost -- %s_* avg time / %s_* avg time</div>'
            % (m["collector_prefix"], m["baseline_prefix"]),
            unsafe_allow_html=True,
        )
        st.markdown(bar_svg, unsafe_allow_html=True)

    # Death premium bar
    dp_lo = max(tgt_dp - warn_dp, 0)
    dp_hi = tgt_dp + warn_dp
    dp_bar_max = max(dp * 1.3, dp_hi * 1.3, 1)
    dp_segments = [
        (0,     dp_lo,      "#14532d", "safe"),
        (dp_lo, dp_hi,      "#78350f", "target"),
        (dp_hi, dp_bar_max, "#7f1d1d", "deadly"),
    ]
    dp_ticks = [(v, "%.1fx" % v) for v in [0, dp_lo, tgt_dp, dp_hi]]
    dp_bar = _zone_bar_svg(dp, "%.2fx" % dp, dp_segments, dp_ticks, m4_color, dp_bar_max)
    st.markdown(
        '<div style="font-size:0.75rem;color:#777;margin-bottom:2px;margin-top:16px;">'
        'Death premium -- %s_* deaths/run / %s_* deaths/run</div>'
        % (m["collector_prefix"], m["baseline_prefix"]),
        unsafe_allow_html=True,
    )
    st.markdown(dp_bar, unsafe_allow_html=True)
    st.markdown('<div class="sep"></div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        tc_str = "%.2fx" % tc if tc is not None else "N/A"
        st.markdown(_stat_card(
            "Time cost",
            '<span style="color:%s;">%s</span>' % (m4_color, tc_str),
            "%s_* / %s_*" % (m["collector_prefix"], m["baseline_prefix"]),
        ), unsafe_allow_html=True)
    with c2:
        st.markdown(_stat_card(
            "Death premium",
            '<span style="color:#3b82f6;">%.2fx</span>' % dp,
            "%.2f / %.2f dpr" % (m["collector_dpr"], m["baseline_dpr"]),
        ), unsafe_allow_html=True)
    with c3:
        ct = m["collector_avg_time"]
        ct_str = "%.1fs" % ct if ct is not None else "N/A"
        st.markdown(_stat_card(
            "%s_* avg time" % m["collector_prefix"],
            '<span style="color:#f59e0b;">%s</span>' % ct_str,
            "successful runs",
        ), unsafe_allow_html=True)
    with c4:
        bt = m["baseline_avg_time"]
        bt_str = "%.1fs" % bt if bt is not None else "N/A"
        st.markdown(_stat_card(
            "%s_* avg time" % m["baseline_prefix"],
            '<span style="color:#a855f7;">%s</span>' % bt_str,
            "successful runs",
        ), unsafe_allow_html=True)

    _analysis_box(m["m4_key"], M4_ANALYSIS)


def render_m5_detail(m):
    ne_gap = m["novice_expert_gap"]
    me_gap = m["mid_expert_gap"]
    m5_color = m["m5_color"]
    t = m["thresholds"]
    tgt_ne  = t.get("target_novice_expert_gap", 0.4)
    warn_ne = t.get("warning_novice_expert_gap", 0.15)
    tgt_me  = t.get("target_mid_expert_gap", 0.15)
    warn_me = t.get("warning_mid_expert_gap", 0.1)

    # Novice-expert gap bar (0-1 range since it is a CR difference)
    ne_lo = max(tgt_ne - warn_ne, 0)
    ne_hi = min(tgt_ne + warn_ne, 1.0)
    ne_bar_max = min(max(abs(ne_gap) * 1.3, ne_hi * 1.3, 0.1), 1.0)
    segments = [
        (0,     ne_lo,      "#7f1d1d", "flat"),
        (ne_lo, ne_hi,      "#14532d", "target"),
        (ne_hi, ne_bar_max, "#78350f", "steep"),
    ]
    ticks = [(v, "%d%%" % round(v * 100)) for v in [0, ne_lo, tgt_ne, ne_hi]]
    bar_svg = _zone_bar_svg(
        abs(ne_gap), "%d%%" % round(abs(ne_gap) * 100),
        segments, ticks, m5_color, ne_bar_max,
    )
    st.markdown(
        '<div style="font-size:0.75rem;color:#777;margin-bottom:2px;">'
        'Novice-expert gap -- *_%s CR - *_%s CR</div>'
        % (m["expert_suffix"], m["novice_suffix"]),
        unsafe_allow_html=True,
    )
    st.markdown(bar_svg, unsafe_allow_html=True)

    # Mid-expert gap bar
    if me_gap is not None:
        me_lo = max(tgt_me - warn_me, 0)
        me_hi = min(tgt_me + warn_me, 1.0)
        me_bar_max = min(max(abs(me_gap) * 1.3, me_hi * 1.3, 0.1), 1.0)
        me_segments = [
            (0,     me_lo,      "#7f1d1d", "flat"),
            (me_lo, me_hi,      "#14532d", "target"),
            (me_hi, me_bar_max, "#78350f", "steep"),
        ]
        me_ticks = [(v, "%d%%" % round(v * 100)) for v in [0, me_lo, tgt_me, me_hi]]
        me_bar = _zone_bar_svg(
            abs(me_gap), "%d%%" % round(abs(me_gap) * 100),
            me_segments, me_ticks, m5_color, me_bar_max,
        )
        st.markdown(
            '<div style="font-size:0.75rem;color:#777;margin-bottom:2px;margin-top:16px;">'
            'Mid-expert gap -- *_%s CR - *_%s CR</div>'
            % (m["expert_suffix"], m["mid_suffix"]),
            unsafe_allow_html=True,
        )
        st.markdown(me_bar, unsafe_allow_html=True)

    st.markdown('<div class="sep"></div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(_stat_card(
            "*_%s completion" % m["expert_suffix"],
            '<span style="color:#22c55e;">%.1f%%</span>' % (m["expert_cr"] * 100),
            "expert baseline",
        ), unsafe_allow_html=True)
    with c2:
        if m["mid_cr"] is not None:
            mid_str = "%.1f%%" % (m["mid_cr"] * 100)
        else:
            mid_str = "N/A"
        st.markdown(_stat_card(
            "*_%s completion" % m["mid_suffix"],
            '<span style="color:#3b82f6;">%s</span>' % mid_str,
            "mid tier",
        ), unsafe_allow_html=True)
    with c3:
        st.markdown(_stat_card(
            "*_%s completion" % m["novice_suffix"],
            '<span style="color:#f59e0b;">%.1f%%</span>' % (m["novice_cr"] * 100),
            "novice tier",
        ), unsafe_allow_html=True)

    _analysis_box(m["m5_key"], M5_ANALYSIS)


def render_m6_detail(m):
    rates = m["rates"]
    missing = m["missing_skills"]
    skill_suffixes = m["skill_suffixes"]
    m6_color = m["m6_color"]
    t = m["thresholds"]

    # One bar per skill tier
    for role in ("novice", "mid", "expert"):
        tgt = t.get("target_%s_completion" % role)
        warn = t.get("warning_%s_completion" % role)
        if tgt is None or warn is None:
            continue
        rate = rates.get(role)
        suffix = skill_suffixes.get(role, role)
        if rate is None:
            st.markdown(
                '<div style="font-size:0.75rem;color:#777;margin-bottom:2px;margin-top:16px;">'
                '%s (*_%s) -- no data</div>' % (role.title(), suffix),
                unsafe_allow_html=True,
            )
            continue
        lo = max(tgt - warn, 0)
        hi = min(tgt + warn, 1.0)
        segments = [
            (0,  lo, "#7f1d1d", "too hard"),
            (lo, hi, "#14532d", "target"),
            (hi, 1.0, "#78350f", "too easy"),
        ]
        ticks = [(v, "%d%%" % round(v * 100)) for v in [0, lo, tgt, hi, 1.0]]
        bar = _zone_bar_svg(rate, "%d%%" % round(rate * 100), segments, ticks, m6_color, 1.0)
        st.markdown(
            '<div style="font-size:0.75rem;color:#777;margin-bottom:2px;margin-top:16px;">'
            '%s (*_%s) completion rate</div>' % (role.title(), suffix),
            unsafe_allow_html=True,
        )
        st.markdown(bar, unsafe_allow_html=True)

    st.markdown('<div class="sep"></div>', unsafe_allow_html=True)

    # Stat cards for each tier
    cols = st.columns(max(len(rates), 1))
    colors = {"novice": "#f59e0b", "mid": "#3b82f6", "expert": "#22c55e"}
    for i, (role, rate) in enumerate(sorted(rates.items(), key=lambda x: ["novice","mid","expert"].index(x[0]) if x[0] in ["novice","mid","expert"] else 99)):
        with cols[i % len(cols)]:
            suffix = skill_suffixes.get(role, role)
            c = colors.get(role, "#a855f7")
            st.markdown(_stat_card(
                "%s (*_%s)" % (role.title(), suffix),
                '<span style="color:%s;">%.1f%%</span>' % (c, rate * 100),
                "completion rate",
            ), unsafe_allow_html=True)

    if missing:
        st.markdown(
            '<div style="font-size:0.75rem;color:#777;margin-top:8px;">Missing skill tiers: %s</div>'
            % ", ".join(missing),
            unsafe_allow_html=True,
        )

    _analysis_box(m["m6_key"], M6_ANALYSIS)


def render_route_viz(world, df_all):
    st.markdown('<div class="sep"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-label">agent route on level</div>',
        unsafe_allow_html=True,
    )

    world_to_file = load_game_config(GAME_CONFIG_PATH)
    level_rel_path = world_to_file.get(world)

    if level_rel_path is None:
        st.markdown(
            '<div class="no-data-box">No level file mapping found for this world.</div>',
            unsafe_allow_html=True,
        )
        return

    level_full_path = os.path.join(LEVELS_ROOT, level_rel_path)
    grid, grid_rows, grid_cols = load_level_grid(level_full_path)

    if grid is None:
        st.markdown(
            '<div class="no-data-box">Could not load level file: %s</div>' % level_full_path,
            unsafe_allow_html=True,
        )
        return

    world_df = df_all[df_all["world"] == world].copy()
    world_df = world_df[world_df["route"].notna() & (world_df["route"].str.strip() != "")]
    world_df = world_df.reset_index(drop=True)

    if world_df.empty:
        st.markdown(
            '<div class="no-data-box">No route data for this world.</div>',
            unsafe_allow_html=True,
        )
        return

    run_labels = []
    for idx, row in world_df.iterrows():
        death = row.get("cause_of_death", "?")
        try:
            prog = float(row.get("progress_ratio", 0))
        except (ValueError, TypeError):
            prog = 0.0
        player_name = row.get("player", row.get("persona", "?"))
        label = "Run %d: %s -- %s (progress %.0f%%)" % (idx, player_name, death, prog * 100)
        run_labels.append(label)

    sel_col1, sel_col2 = st.columns([3, 1])
    with sel_col1:
        selected_run_label = st.selectbox(
            "select run",
            options=run_labels,
            label_visibility="collapsed",
            key="route_run_selector",
        )
    selected_run_idx = run_labels.index(selected_run_label)

    with sel_col2:
        show_all = st.checkbox("overlay all runs", value=False, key="route_show_all")

    route_colors = [
        "#ef4444", "#3b82f6", "#22c55e", "#f59e0b", "#a855f7",
        "#ec4899", "#06b6d4", "#f97316", "#84cc16", "#e879f9",
    ]

    fig, ax = plt.subplots(
        1, 1,
        figsize=(min(grid_cols * 0.15, 28), min(grid_rows * 0.15, 6)),
        dpi=100,
    )
    fig.patch.set_facecolor("#1a1a1a")
    ax.set_facecolor("#1a1a1a")

    level_cmap = mcolors.ListedColormap(["#0d0d0d", "#1a1a1a", "#3a3a3a"])
    display_grid = grid.copy().astype(np.int8) + 1
    ax.imshow(
        display_grid, cmap=level_cmap, aspect="equal",
        interpolation="nearest", extent=[0, grid_cols, grid_rows, 0],
    )

    if show_all:
        for i, (_, row) in enumerate(world_df.iterrows()):
            pts = parse_route(row["route"])
            if len(pts) < 2:
                continue
            xs = [p[0] / TILE_SIZE for p in pts]
            ys = [p[1] / TILE_SIZE for p in pts]
            color = route_colors[i % len(route_colors)]
            ax.plot(xs, ys, color=color, linewidth=0.6, alpha=0.4)
        sel_pts = parse_route(world_df.iloc[selected_run_idx]["route"])
        if len(sel_pts) >= 2:
            xs = [p[0] / TILE_SIZE for p in sel_pts]
            ys = [p[1] / TILE_SIZE for p in sel_pts]
            ax.plot(xs, ys, color="#ffffff", linewidth=1.5, alpha=0.9)
            ax.plot(xs[0], ys[0], "o", color="#22c55e", markersize=5, zorder=5)
            ax.plot(xs[-1], ys[-1], "x", color="#ef4444", markersize=6, markeredgewidth=2, zorder=5)
    else:
        sel_pts = parse_route(world_df.iloc[selected_run_idx]["route"])
        if len(sel_pts) >= 2:
            xs = [p[0] / TILE_SIZE for p in sel_pts]
            ys = [p[1] / TILE_SIZE for p in sel_pts]
            ax.plot(xs, ys, color="#ef4444", linewidth=1.2, alpha=0.85)
            ax.plot(xs[0], ys[0], "o", color="#22c55e", markersize=5, zorder=5)
            ax.plot(xs[-1], ys[-1], "x", color="#ef4444", markersize=6, markeredgewidth=2, zorder=5)

    ax.set_xlim(0, grid_cols)
    ax.set_ylim(grid_rows, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown(
        '<div style="font-size:0.72rem;color:#555;margin-top:4px;'
        "font-family:'IBM Plex Mono',monospace;\">"
        '<span style="color:#22c55e;">&#9679;</span> start &nbsp;'
        '<span style="color:#ef4444;">&#10005;</span> end &nbsp;'
        '<span style="color:#3a3a3a;">&#9608;</span> solid &nbsp;'
        '<span style="color:#0d0d0d;">&#9608;</span> pit'
        '</div>',
        unsafe_allow_html=True,
    )
