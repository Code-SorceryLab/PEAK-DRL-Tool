import streamlit as st
import pandas as pd
import yaml
import os
import glob

st.set_page_config(page_title="Peak Level Analyzer", layout="wide")

# ── CSS ─────────────────────────────────────────────────────────────────────
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


# ── Load config ──────────────────────────────────────────────────────────────
CONFIG_PATH = "code/stats/MarioThresholds.yaml"

@st.cache_data
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

@st.cache_data
def load_all_csvs(data_path):
    """Read all CSVs from the data path, concatenate into one DataFrame."""
    pattern = os.path.join(data_path, "*.csv")
    files = glob.glob(pattern)
    if not files:
        return pd.DataFrame()
    dfs = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
            dfs.append(df)
        except Exception:
            continue
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


# ── Data helpers ─────────────────────────────────────────────────────────────

def win_rate(df, world, persona):
    """Win rate = fraction of runs where cause_of_death == 'Success'."""
    sub = df[(df["world"] == world) & (df["persona"] == persona)]
    if len(sub) == 0:
        return None
    return (sub["cause_of_death"].str.lower() == "success").sum() / len(sub)


def flow_score(df, world, persona):
    """Flow score = mean(progress_ratio) for a given world/persona."""
    sub = df[(df["world"] == world) & (df["persona"] == persona)]
    if len(sub) == 0:
        return None
    return sub["progress_ratio"].mean()


# ── B1 classification ────────────────────────────────────────────────────────

def classify_b1(gap, novice_wr, thresholds):
    t = thresholds
    target_min  = t.get("target_gap_min", 0.15)
    target_max  = t.get("target_gap_max", 0.40)
    warn_low    = t.get("warning_gap_low", 0.05)
    warn_high   = t.get("warning_gap_high", 0.55)
    novice_min  = t.get("novice_min_win_rate", 0.03)

    if novice_wr is None or novice_wr < novice_min:
        return "gate", "#ef4444", "pill-imbalance", 0
    if gap < warn_low:
        return "thin", "#f59e0b", "pill-warning", 0
    if target_min <= gap <= target_max:
        return "healthy", "#22c55e", "pill-balanced", 1
    if warn_low <= gap < target_min:
        return "skill-insensitive", "#eab308", "pill-warning", 0
    if gap > warn_high:
        return "unfair", "#dc2626", "pill-imbalance", 0
    if gap > target_max:
        return "steep", "#f97316", "pill-warning", 0
    return "steep", "#f97316", "pill-warning", 0


# ── B2 classification ────────────────────────────────────────────────────────

def classify_b2(score, thresholds):
    t = thresholds
    balanced_min = t.get("balanced_min", 0.60)
    warning_min  = t.get("warning_min", 0.35)

    if score >= balanced_min:
        return "balanced", "#22c55e", "pill-balanced", 1
    if score >= warning_min:
        return "warning", "#eab308", "pill-warning", 0
    return "imbalance", "#ef4444", "pill-imbalance", 0


# ── Analysis text ────────────────────────────────────────────────────────────

ZONE_ANALYSIS = {
    "gate": (
        "Gate — impossible for novices",
        "The novice win rate is near zero, meaning the level is a hard blocker regardless of skill. "
        "This is not a skill-test — it's a wall. Consider lowering the entry difficulty.",
        "A gate with near-zero novice wins signals the room is either miscalibrated or intentionally placed "
        "as a final boss section."
    ),
    "thin": (
        "Gap too small — nearly skill-insensitive",
        "Both agents perform very similarly. The level offers almost no discrimination between skill tiers.",
        "Thin gaps near the low end suggest a gate. Thin gaps at the high end suggest boredom."
    ),
    "healthy": (
        "Gap in target range — good skill signal",
        "This is the sweet spot. The expert wins meaningfully more often than the novice, but the novice still "
        "has a fair chance.",
        "Maintain this balance. Watch for drift if you shorten or lengthen the room."
    ),
    "skill-insensitive": (
        "Gap below target — weak skill signal",
        "The expert does a bit better but not by much. The level has some skill sensitivity but not enough "
        "to meaningfully separate player tiers.",
        "Consider tightening the main hazard corridor or removing an accidental shortcut."
    ),
    "steep": (
        "Gap above target — steep",
        "The gap is large but tilted toward extreme difficulty for novices.",
        "Large gaps with very low novice rates risk drop-off in player retention."
    ),
    "unfair": (
        "Unfair — too steep",
        "The gap between expert and novice is extreme. Novices almost never clear the level.",
        "Extreme gaps almost always indicate a missing difficulty layer."
    ),
}

B2_ANALYSIS = {
    "balanced": (
        "Challenge vs. success — well balanced",
        "The flow score indicates a healthy ratio of challenge to achievable progress. Players are "
        "stretched but not overwhelmed.",
        "Monitor this metric across playtester sessions. High variance often signals a hidden shortcut."
    ),
    "warning": (
        "Challenge vs. success — mild imbalance",
        "The flow score sits in a borderline range. Some player segments will find the level satisfying, "
        "but others may feel the challenge ramps too quickly or slowly.",
        "Check the reward distribution across sections. Uneven pacing often produces mid-range flow scores."
    ),
    "imbalance": (
        "Challenge vs. success — significant imbalance",
        "The flow score is low, indicating the challenge curve is misaligned with player success feedback. "
        "Players are likely hitting a frustration wall.",
        "Redesign the pacing curve. Introduce recovery zones after high-difficulty sections."
    ),
}


# ── Pill helper ──────────────────────────────────────────────────────────────

def pill_span(pill_class, text):
    styles = {
        "pill-balanced":  "background:#14532d;color:#4ade80",
        "pill-warning":   "background:#3d2e00;color:#fbbf24",
        "pill-imbalance": "background:#3d1a1a;color:#f87171",
    }
    s = styles.get(pill_class, "background:#333;color:#ccc")
    return f'<span style="display:inline-block;font-size:0.7rem;font-weight:600;border-radius:4px;padding:2px 9px;margin-top:8px;{s}">{text}</span>'


# ── Load data ────────────────────────────────────────────────────────────────

try:
    config = load_config(CONFIG_PATH)
except FileNotFoundError:
    st.error(f"Config file not found: {CONFIG_PATH}")
    st.stop()

data_path = config.get("path", "code/stats/results/")
metrics_cfg = config.get("metrics", {})

df_all = load_all_csvs(data_path)

if df_all.empty:
    st.warning(f"No CSV data found in `{data_path}`. Make sure the path is correct and CSVs exist.")
    st.stop()

worlds = sorted(df_all["world"].unique().tolist())

# ── Session state ────────────────────────────────────────────────────────────
if "selected_metric" not in st.session_state:
    st.session_state.selected_metric = list(metrics_cfg.keys())[0] if metrics_cfg else "B1_fairness"
if "selected_world" not in st.session_state:
    st.session_state.selected_world = worlds[0] if worlds else None


# ── Per-world metric computation ─────────────────────────────────────────────

def compute_world_metrics(world, cfg):
    results = {}

    for metric_key, metric_cfg in cfg.items():
        thresholds = metric_cfg.get("thresholds", {})

        if "B1" in metric_key.upper() or "fairness" in metric_key.lower():
            personas_raw = metric_cfg.get("personas", "")
            personas = [p.strip() for p in str(personas_raw).split(",") if p.strip()]
            if len(personas) < 2:
                results[metric_key] = {"error": "Need exactly 2 personas for B1"}
                continue
            novice_persona, expert_persona = personas[0], personas[1]
            novice_wr = win_rate(df_all, world, novice_persona)
            expert_wr = win_rate(df_all, world, expert_persona)

            if novice_wr is None and expert_wr is None:
                results[metric_key] = {"error": f"No data for personas '{novice_persona}' or '{expert_persona}' in world '{world}'"}
                continue

            nwr = novice_wr if novice_wr is not None else 0.0
            ewr = expert_wr if expert_wr is not None else 0.0
            gap = ewr - nwr
            zone_key, zone_color, zone_pill, score_pt = classify_b1(gap, nwr, thresholds)

            results[metric_key] = {
                "type": "B1",
                "novice_persona": novice_persona,
                "expert_persona": expert_persona,
                "novice_wr": novice_wr,
                "expert_wr": expert_wr,
                "gap": gap,
                "zone_key": zone_key,
                "zone_color": zone_color,
                "zone_pill": zone_pill,
                "score_pt": score_pt,
                "thresholds": thresholds,
            }

        elif "B2" in metric_key.upper() or "challenge" in metric_key.lower():
            # Best persona = highest mean progress_ratio
            personas_in_world = df_all[df_all["world"] == world]["persona"].unique()
            best_persona = None
            best_score = -1.0
            for p in personas_in_world:
                fs = flow_score(df_all, world, p)
                if fs is not None and fs > best_score:
                    best_score = fs
                    best_persona = p

            if best_persona is None:
                results[metric_key] = {"error": f"No data for world '{world}'"}
                continue

            b2_key, b2_color, b2_pill, score_pt = classify_b2(best_score, thresholds)
            results[metric_key] = {
                "type": "B2",
                "best_persona": best_persona,
                "flow_score": best_score,
                "b2_key": b2_key,
                "b2_color": b2_color,
                "b2_pill": b2_pill,
                "score_pt": score_pt,
                "thresholds": thresholds,
            }

    return results


# ── Render ───────────────────────────────────────────────────────────────────

world = st.session_state.selected_world
world_metrics = compute_world_metrics(world, metrics_cfg)

total_score = sum(
    m.get("score_pt", 0) for m in world_metrics.values() if "error" not in m
)
max_score = len([m for m in world_metrics.values() if "error" not in m])

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:18px;">
  <div>
    <div class="header-title">Balance dashboard — Peak</div>
    <div class="header-meta">Level: {world} · data path: {data_path}</div>
  </div>
  <div class="score-badge">balance score <span>{total_score}</span> / {max_score}</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="sep"></div>', unsafe_allow_html=True)

# ── Level selector ────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">select level</div>', unsafe_allow_html=True)
selected_world = st.selectbox(
    label="level",
    options=worlds,
    index=worlds.index(st.session_state.selected_world) if st.session_state.selected_world in worlds else 0,
    label_visibility="collapsed"
)
if selected_world != st.session_state.selected_world:
    st.session_state.selected_world = selected_world
    st.rerun()

st.markdown('<div style="height:14px;"></div>', unsafe_allow_html=True)

# ── Persona win rate cards (all personas in this world) ───────────────────────
st.markdown('<div class="section-label">win rate by persona</div>', unsafe_allow_html=True)
PERSONA_COLORS = ["#ef4444", "#f59e0b", "#22c55e", "#3b82f6", "#a855f7", "#ec4899"]
personas_in_world = sorted(df_all[df_all["world"] == world]["persona"].unique().tolist())
cards_html = '<div class="persona-row">'
for i, persona in enumerate(personas_in_world):
    color = PERSONA_COLORS[i % len(PERSONA_COLORS)]
    wr = win_rate(df_all, world, persona)
    if wr is None:
        pct_display, bar_w = "N/A", 0
    else:
        pct_val = round(wr * 100, 1)
        pct_display, bar_w = f"{pct_val}%", pct_val
    cards_html += f"""
    <div class="persona-card">
      <div class="persona-sublabel">{persona}</div>
      <div class="persona-pct" style="color:{color};">{pct_display}</div>
      <div class="persona-bar" style="background:{color};width:{bar_w}%;"></div>
    </div>"""
cards_html += '</div>'
st.markdown(cards_html, unsafe_allow_html=True)

# ── Metric cards ──────────────────────────────────────────────────────────────
st.markdown(f'<div class="section-label">{len(world_metrics)} balance metric(s) — click a card to inspect</div>', unsafe_allow_html=True)

selected_metric = st.session_state.selected_metric

def card_html(metric_key, name, value, value_color, sub, pill_class, status, selected):
    border = "#4a9eff" if selected else "#2e2e2e"
    bg     = "#1e2a3a" if selected else "#232323"
    return f"""<div style="background:{bg};border:1px solid {border};border-radius:8px 8px 0 0;
        padding:14px 16px 12px 16px;min-height:130px;">
  <span style="font-size:0.72rem;color:#666;font-family:'IBM Plex Mono',monospace;">{metric_key}</span>
  <div style="font-size:0.85rem;color:#b0b0b0;font-weight:500;margin:6px 0 4px 0;">{name}</div>
  <div style="font-size:1.9rem;font-weight:600;line-height:1.1;font-family:'IBM Plex Mono',monospace;color:{value_color};">{value}</div>
  <div style="font-size:0.72rem;color:#666;margin-top:4px;">{sub}</div>
  <div>{pill_span(pill_class, status)}</div>
</div>"""

# Determine selected column index for border styling
metric_keys = list(world_metrics.keys())
sel_idx = metric_keys.index(selected_metric) if selected_metric in metric_keys else 0

# Dynamic CSS for selected card button border
st.markdown(f"""
<style>
  div[data-testid="stColumn"] div[data-testid="stButton"] {{
    margin-top: 0 !important;
  }}
  div[data-testid="stColumn"] div[data-testid="stButton"] > button {{
    width: 100% !important;
    border-radius: 0 0 8px 8px !important;
    border-top: 1px solid #3a3a3a !important;
    background: #1e1e1e !important;
    font-size: 0.75rem !important;
    color: #555 !important;
    padding: 7px !important;
    letter-spacing: 0.04em !important;
  }}
  div[data-testid="stColumn"] div[data-testid="stButton"] > button:hover {{
    background: #2a2a2a !important;
    color: #aaa !important;
    border-color: #555 !important;
  }}
  div[data-testid="stColumn"]:nth-child({sel_idx + 1}) div[data-testid="stButton"] > button {{
    border-left: 1px solid #4a9eff !important;
    border-right: 1px solid #4a9eff !important;
    border-bottom: 1px solid #4a9eff !important;
  }}
</style>
""", unsafe_allow_html=True)

# Render one column per metric + a spacer
col_weights = [1] * len(metric_keys) + [max(1, 4 - len(metric_keys))]
cols = st.columns(col_weights)

for i, metric_key in enumerate(metric_keys):
    m = world_metrics[metric_key]
    with cols[i]:
        if "error" in m:
            st.markdown(f"""<div style="background:#232323;border:1px solid #2e2e2e;border-radius:8px;
                padding:14px 16px;min-height:130px;color:#666;font-size:0.75rem;font-family:'IBM Plex Mono',monospace;">
                {metric_key}<br><br>{m['error']}</div>""", unsafe_allow_html=True)
        elif m["type"] == "B1":
            gap_pp = round(m["gap"] * 100, 1)
            nwr_pct = round(m["novice_wr"] * 100, 1) if m["novice_wr"] is not None else "N/A"
            ewr_pct = round(m["expert_wr"] * 100, 1) if m["expert_wr"] is not None else "N/A"
            st.markdown(card_html(
                metric_key, "Fairness",
                f"{gap_pp} pp", m["zone_color"],
                f"novice {nwr_pct}% → expert {ewr_pct}%",
                m["zone_pill"], m["zone_key"].replace("-", " ").title(),
                selected_metric == metric_key
            ), unsafe_allow_html=True)
            if st.button(f"Analyze {metric_key}", key=f"btn_{metric_key}", use_container_width=True):
                st.session_state.selected_metric = metric_key
                st.rerun()
        elif m["type"] == "B2":
            fs = round(m["flow_score"], 3)
            st.markdown(card_html(
                metric_key, "Challenge vs. success",
                str(fs), m["b2_color"],
                f"best persona: {m['best_persona']}",
                m["b2_pill"], m["b2_key"].replace("-", " ").title(),
                selected_metric == metric_key
            ), unsafe_allow_html=True)
            if st.button(f"Analyze {metric_key}", key=f"btn_{metric_key}", use_container_width=True):
                st.session_state.selected_metric = metric_key
                st.rerun()

st.markdown('<div class="sep"></div>', unsafe_allow_html=True)

# ── Detail panel ──────────────────────────────────────────────────────────────
if selected_metric not in world_metrics:
    st.markdown('<div class="no-data-box">Select a metric above to inspect.</div>', unsafe_allow_html=True)
else:
    m = world_metrics[selected_metric]

    if "error" in m:
        st.markdown(f'<div class="no-data-box">{m["error"]}</div>', unsafe_allow_html=True)

    elif m["type"] == "B1":
        novice_pct = round(m["novice_wr"] * 100, 1) if m["novice_wr"] is not None else 0
        expert_pct = round(m["expert_wr"] * 100, 1) if m["expert_wr"] is not None else 0
        gap_pct    = round(m["gap"] * 100, 1)

        def render_readonly_slider(label, value, color="#ef4444"):
            st.markdown(f"""
            <div style="margin-bottom: 22px;">
              <div style="font-size:0.82rem; color:#aaaaaa; font-family:'IBM Plex Sans',sans-serif; margin-bottom:10px;">{label}</div>
              <div style="position:relative; height:20px; display:flex; align-items:center;">
                <div style="position:absolute; left:0; right:0; height:4px; background:#3a3a3a; border-radius:2px;"></div>
                <div style="position:absolute; left:0; width:{value}%; height:4px; background:{color}; border-radius:2px;"></div>
                <div style="position:absolute; left:calc({value}% - 8px); width:16px; height:16px; border-radius:50%; background:{color}; box-shadow:0 0 0 3px #2a2a2a;"></div>
                <div style="position:absolute; left:calc({value}% - 10px); top:-20px; font-size:0.75rem; color:#e0e0e0; font-family:'IBM Plex Mono',monospace;">{value}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        render_readonly_slider(f"Novice win rate — {m['novice_persona']}", novice_pct, "#ef4444")
        render_readonly_slider(f"Expert win rate — {m['expert_persona']}", expert_pct, "#3b82f6")

        # Gap bar
        t = m["thresholds"]
        BAR_W, BAR_H = 800, 36
        SCALE_MAX = 100

        def pct_to_x(p):
            return round(min(p, SCALE_MAX) / SCALE_MAX * BAR_W)

        warn_low    = round(t.get("warning_gap_low", 0.05) * 100)
        target_min  = round(t.get("target_gap_min", 0.15) * 100)
        target_max  = round(t.get("target_gap_max", 0.40) * 100)
        warn_high   = round(t.get("warning_gap_high", 0.55) * 100)

        segments = [
            (0,          warn_low,   "#7f1d1d", "gate/thin"),
            (warn_low,   target_min, "#78350f", f"weak ({warn_low}–{target_min}pp)"),
            (target_min, target_max, "#14532d", f"healthy ({target_min}–{target_max}pp)"),
            (target_max, warn_high,  "#713f12", f"steep ({target_max}–{warn_high}pp)"),
            (warn_high,  100,        "#7f1d1d", "unfair"),
        ]
        seg_rects = ""
        for s, e_seg, col, lbl in segments:
            x1, x2 = pct_to_x(s), pct_to_x(e_seg)
            w = max(x2 - x1, 1)
            seg_rects += f'<rect x="{x1}" y="0" width="{w}" height="{BAR_H}" fill="{col}"/>'
            mid = (x1 + x2) // 2
            seg_rects += f'<text x="{mid}" y="{BAR_H//2+5}" text-anchor="middle" font-size="9" fill="#cccccc" font-family="IBM Plex Sans,sans-serif">{lbl}</text>'

        ticks_svg = ""
        for t_val in [0, warn_low, target_min, target_max, warn_high, 100]:
            tx = pct_to_x(t_val)
            ticks_svg += f'<text x="{tx}" y="52" text-anchor="middle" font-size="10" fill="#777777" font-family="IBM Plex Mono,monospace">{t_val}%</text>'
            ticks_svg += f'<line x1="{tx}" y1="{BAR_H}" x2="{tx}" y2="{BAR_H+6}" stroke="#555" stroke-width="1"/>'

        nx   = pct_to_x(min(novice_pct, SCALE_MAX))
        ex_x = pct_to_x(min(expert_pct, SCALE_MAX))
        box_x = min(nx, ex_x)
        box_w = max(abs(ex_x - nx), 2)
        tip_x = (nx + ex_x) // 2

        marker_svg = f"""
          <rect x="{box_x}" y="0" width="{box_w}" height="{BAR_H}" fill="rgba(255,255,255,0.08)" rx="2"/>
          <rect x="{tip_x - 70}" y="-28" width="140" height="22" fill="#1a1a1a" rx="3"/>
          <text x="{tip_x}" y="-12" text-anchor="middle" font-size="10" fill="#e5e5e5" font-family="IBM Plex Sans,sans-serif">
            <tspan fill="#ef4444">Novice {novice_pct}%</tspan>
            <tspan fill="#888888">  </tspan>
            <tspan fill="#3b82f6">Expert {expert_pct}%</tspan>
          </text>
          <line x1="{nx}" y1="-4" x2="{nx}" y2="{BAR_H+4}" stroke="#ef4444" stroke-width="2"/>
          <line x1="{ex_x}" y1="-4" x2="{ex_x}" y2="{BAR_H+4}" stroke="#3b82f6" stroke-width="2"/>
        """
        gap_bar_svg = f"""
        <svg viewBox="-10 -35 {BAR_W+20} 90" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:860px;display:block;margin:0 auto;">
          <g>{seg_rects}{ticks_svg}{marker_svg}</g>
        </svg>
        """
        st.markdown('<div style="font-size:0.75rem;color:#777;margin-bottom:2px;">Win rate scale — gap zones</div>', unsafe_allow_html=True)
        st.markdown(gap_bar_svg, unsafe_allow_html=True)
        st.markdown('<div class="sep"></div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div style="background:#232323;border:1px solid #2e2e2e;border-radius:6px;padding:14px 16px;">
              <div style="font-size:0.78rem;color:#888;margin-bottom:4px;">Novice win rate</div>
              <div style="font-size:2rem;font-weight:600;color:#ef4444;font-family:'IBM Plex Mono',monospace;">{novice_pct}%</div>
              <div style="font-size:0.72rem;color:#555;">{m['novice_persona']}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div style="background:#232323;border:1px solid #2e2e2e;border-radius:6px;padding:14px 16px;">
              <div style="font-size:0.78rem;color:#888;margin-bottom:4px;">Expert win rate</div>
              <div style="font-size:2rem;font-weight:600;color:#3b82f6;font-family:'IBM Plex Mono',monospace;">{expert_pct}%</div>
              <div style="font-size:0.72rem;color:#555;">{m['expert_persona']}</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div style="background:#232323;border:1px solid #2e2e2e;border-radius:6px;padding:14px 16px;">
              <div style="font-size:0.78rem;color:#888;margin-bottom:4px;">B1 fairness gap</div>
              <div style="font-size:2rem;font-weight:600;color:#eab308;font-family:'IBM Plex Mono',monospace;">{gap_pct} pp</div>
              <div style="font-size:0.72rem;color:#555;">{m['zone_key'].replace('-', ' ').title()}</div>
            </div>""", unsafe_allow_html=True)

        zone_data = ZONE_ANALYSIS.get(m["zone_key"], ZONE_ANALYSIS["skill-insensitive"])
        st.markdown(f"""
        <div class="analysis-box">
          <span class="analysis-tag">{m['zone_key'].replace('-', ' ').title()}</span>
          <span class="analysis-title">{zone_data[0]}</span>
          <div class="analysis-body">{zone_data[1]}</div>
          <div class="analysis-italic">{zone_data[2]}</div>
        </div>
        """, unsafe_allow_html=True)

    elif m["type"] == "B2":
        fs = m["flow_score"]
        b2_color = m["b2_color"]
        t = m["thresholds"]
        balanced_min = t.get("balanced_min", 0.60)
        warning_min  = t.get("warning_min", 0.35)

        BAR_W, BAR_H = 800, 36

        def flow_to_x(v):
            return round(min(max(v, 0.0), 1.0) * BAR_W)

        b2_segments = [
            (0.0,         warning_min,  "#7f1d1d", "imbalance"),
            (warning_min, balanced_min, "#78350f", "warning"),
            (balanced_min, 1.0,         "#14532d", "balanced"),
        ]
        seg_rects = ""
        for s, e_seg, col, lbl in b2_segments:
            x1, x2 = flow_to_x(s), flow_to_x(e_seg)
            w = max(x2 - x1, 1)
            seg_rects += f'<rect x="{x1}" y="0" width="{w}" height="{BAR_H}" fill="{col}"/>'
            mid = (x1 + x2) // 2
            seg_rects += f'<text x="{mid}" y="{BAR_H//2+5}" text-anchor="middle" font-size="10" fill="#cccccc" font-family="IBM Plex Sans,sans-serif">{lbl}</text>'

        ticks_svg = ""
        for t_val in [0.0, warning_min, balanced_min, 1.0]:
            tx = flow_to_x(t_val)
            ticks_svg += f'<text x="{tx}" y="52" text-anchor="middle" font-size="10" fill="#777777" font-family="IBM Plex Mono,monospace">{t_val}</text>'
            ticks_svg += f'<line x1="{tx}" y1="{BAR_H}" x2="{tx}" y2="{BAR_H+6}" stroke="#555" stroke-width="1"/>'

        fx = flow_to_x(min(fs, 1.0))
        marker_svg = f"""
          <line x1="{fx}" y1="-4" x2="{fx}" y2="{BAR_H+4}" stroke="{b2_color}" stroke-width="2"/>
          <circle cx="{fx}" cy="{BAR_H//2}" r="7" fill="{b2_color}" opacity="0.9"/>
          <rect x="{fx-40}" y="-26" width="80" height="20" fill="#1a1a1a" rx="3"/>
          <text x="{fx}" y="-11" text-anchor="middle" font-size="10" fill="#e5e5e5" font-family="IBM Plex Mono,monospace">flow {round(fs,3)}</text>
        """
        flow_bar_svg = f"""
        <svg viewBox="-10 -35 {BAR_W+20} 90" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:860px;display:block;margin:0 auto;">
          <g>{seg_rects}{ticks_svg}{marker_svg}</g>
        </svg>
        """
        st.markdown('<div style="font-size:0.75rem;color:#777;margin-bottom:2px;">Flow score scale — challenge vs. success zones</div>', unsafe_allow_html=True)
        st.markdown(flow_bar_svg, unsafe_allow_html=True)
        st.markdown('<div class="sep"></div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div style="background:#232323;border:1px solid #2e2e2e;border-radius:6px;padding:14px 16px;">
              <div style="font-size:0.78rem;color:#888;margin-bottom:4px;">Flow score</div>
              <div style="font-size:2rem;font-weight:600;color:{b2_color};font-family:'IBM Plex Mono',monospace;">{round(fs,3)}</div>
              <div style="font-size:0.72rem;color:#555;">best persona</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div style="background:#232323;border:1px solid #2e2e2e;border-radius:6px;padding:14px 16px;">
              <div style="font-size:0.78rem;color:#888;margin-bottom:4px;">Best persona</div>
              <div style="font-size:1.1rem;font-weight:600;color:{b2_color};font-family:'IBM Plex Mono',monospace;padding-top:6px;">{m['best_persona']}</div>
              <div style="font-size:0.72rem;color:#555;">highest mean progress</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div style="background:#232323;border:1px solid #2e2e2e;border-radius:6px;padding:14px 16px;">
              <div style="font-size:0.78rem;color:#888;margin-bottom:4px;">Thresholds</div>
              <div style="font-size:0.72rem;color:#777;margin-top:8px;line-height:1.7;font-family:'IBM Plex Mono',monospace;">
                ≥ {balanced_min} balanced<br>≥ {warning_min} warning<br>&lt; {warning_min} imbalance
              </div>
            </div>""", unsafe_allow_html=True)

        b2_data = B2_ANALYSIS.get(m["b2_key"], B2_ANALYSIS["warning"])
        st.markdown(f"""
        <div class="analysis-box">
          <span class="analysis-tag">{m['b2_key'].replace('-', ' ').title()}</span>
          <span class="analysis-title">{b2_data[0]}</span>
          <div class="analysis-body">{b2_data[1]}</div>
          <div class="analysis-italic">{b2_data[2]}</div>
        </div>
        """, unsafe_allow_html=True)