"""Reusable UI component helpers for the PEAK dashboard."""


PERSONA_COLORS = ["#ef4444", "#f59e0b", "#22c55e", "#3b82f6", "#a855f7", "#ec4899"]


def pill_span(pill_class, text):
    """Return an HTML pill badge string."""
    styles = {
        "pill-balanced":  "background:#14532d;color:#4ade80",
        "pill-warning":   "background:#3d2e00;color:#fbbf24",
        "pill-imbalance": "background:#3d1a1a;color:#f87171",
    }
    s = styles.get(pill_class, "background:#333;color:#ccc")
    return (
        f'<span style="display:inline-block;font-size:0.7rem;font-weight:600;'
        f'border-radius:4px;padding:2px 9px;margin-top:8px;{s}">{text}</span>'
    )


def card_html(metric_key, name, value, value_color, sub, pill_class, status, selected):
    """Return the HTML for a metric summary card."""
    border = "#4a9eff" if selected else "#2e2e2e"
    bg     = "#1e2a3a" if selected else "#232323"
    return (
        f'<div style="background:{bg};border:1px solid {border};border-radius:8px 8px 0 0;'
        f'padding:14px 16px 12px 16px;min-height:130px;">'
        f'<span style="font-size:0.72rem;color:#666;font-family:\'IBM Plex Mono\',monospace;">{metric_key}</span>'
        f'<div style="font-size:0.85rem;color:#b0b0b0;font-weight:500;margin:6px 0 4px 0;">{name}</div>'
        f'<div style="font-size:1.9rem;font-weight:600;line-height:1.1;font-family:\'IBM Plex Mono\',monospace;'
        f'color:{value_color};">{value}</div>'
        f'<div style="font-size:0.72rem;color:#666;margin-top:4px;">{sub}</div>'
        f'<div>{pill_span(pill_class, status)}</div>'
        f'</div>'
    )
