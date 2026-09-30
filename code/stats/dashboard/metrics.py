# -*- coding: utf-8 -*-
"""Balance cards for the PEAK dashboard. Every value comes from code/stats/registry.py — the
same functions the Balance Command report uses — so the two can't disagree."""

from code.stats import registry

from data import episodes

ZONE = {  # registry verdict -> (zone key per card type, colour, pill, score point)
    "B1": {"ok": ("calibrated", "#22c55e", "pill-balanced", 1), "partial": ("warning", "#eab308", "pill-warning", 0),
           "bad": ("miscalibrated", "#ef4444", "pill-imbalance", 0)},
    "B2": {"ok": ("calibrated", "#22c55e", "pill-balanced", 1), "partial": ("warning", "#eab308", "pill-warning", 0),
           "bad": ("severe", "#ef4444", "pill-imbalance", 0)},
    "B3": {"ok": ("diverse", "#22c55e", "pill-balanced", 1), "partial": ("warning", "#eab308", "pill-warning", 0),
           "bad": ("homogeneous", "#ef4444", "pill-imbalance", 0)},
}
CARD_DIM = {"B1": 1, "B2": 2, "B3": 4}


def _legacy_thresholds(bands):
    """The panels' band keys, read from the registry's bands."""
    def b(key, part, default):
        return (bands.get(key) or {}).get(part, default)
    return {
        "target_completion_rate": b("completion_rate", "target", 0.7),
        "warning_completion_rate_difference": b("completion_rate", "warning", 0.2),
        "target_mean_completion_time": b("mean_completion_time", "target", 20),
        "warning_mean_completion_time_difference": b("mean_completion_time", "warning", 4),
        "target_deaths_per_run": b("deaths_per_run", "target", 2),
        "warning_deaths_per_run": b("deaths_per_run", "warning", 1),
        "target_death_cluster_entropy": b("death_cluster_entropy", "target", 0.4),
        "warning_death_cluster_entropy": b("death_cluster_entropy", "warning", 0.1),
        "target_strategy_count": b("strategy_count", "target", 2),
        "warning_strategy_count": b("strategy_count", "warning", 1),
        "target_dominant_path_share": b("dominant_path_share", "target", 0.5),
        "warning_dominant_path_share": b("dominant_path_share", "warning", 0.1),
    }


B1_ANALYSIS = {
    "calibrated": (
        "Challenge calibration -- well tuned",
        "Completion rate and completion time are both within the target range. "
        "The level provides a fair challenge without being frustrating or trivial.",
        "Continue monitoring as level design evolves. Small layout changes can shift these numbers.",
    ),
    "warning": (
        "Challenge calibration -- partially off-target",
        "One of the two signals (completion rate or completion time) sits outside the target range. "
        "The level may be slightly too easy, too hard, or paced unevenly.",
        "Check which signal is off-target and investigate the section(s) that cause most failures or slowdowns.",
    ),
    "miscalibrated": (
        "Challenge calibration -- significant misalignment",
        "Both completion rate and completion time are outside the target range. "
        "The level difficulty is substantially off from the intended calibration.",
        "A full difficulty pass is recommended. Look at progress-at-death to identify the blocking section.",
    ),
}

B2_ANALYSIS = {
    "calibrated": (
        "Punishment severity -- well calibrated",
        "Deaths per run and death spread are both within the target range. "
        "The level punishes mistakes at a reasonable rate and deaths occur across varied sections, "
        "indicating no single chokepoint dominates.",
        "Keep monitoring after layout changes. Adding or removing a hazard can shift both metrics.",
    ),
    "warning": (
        "Punishment severity -- partially off-target",
        "One of the two signals (death rate or death spread) sits outside the target range. "
        "The level may kill too often / too rarely, or deaths may cluster in a single section.",
        "If entropy is low, look for the dominant death spot and consider softening it. "
        "If deaths-per-run is off, adjust the overall hazard density.",
    ),
    "severe": (
        "Punishment severity -- significant misalignment",
        "Both death rate and death spread are outside target. "
        "The level is either a meat-grinder with a single chokepoint, or so forgiving that the few "
        "deaths that happen are random outliers.",
        "A full hazard audit is recommended. Cross-reference with the route view to locate the problem area.",
    ),
}

B3_ANALYSIS = {
    "diverse": (
        "Strategy diversity -- healthy variety",
        "Multiple distinct routes are being used to complete the level, and no single path dominates. "
        "This indicates the level supports genuine strategic choice.",
        "Good variety. Watch for a dominant strategy emerging if you add or remove shortcuts.",
    ),
    "warning": (
        "Strategy diversity -- partially limited",
        "Either the number of distinct strategies or the dominance of the top path is outside the "
        "target range. The level may funnel players toward one route more than intended.",
        "Check whether a seemingly optional path is actually unreachable or too punishing to be viable.",
    ),
    "homogeneous": (
        "Strategy diversity -- low",
        "Almost all successful runs follow the same route. The level effectively has a single viable "
        "strategy, reducing replay value and skill expression.",
        "Consider opening alternative paths, adding optional shortcuts with risk/reward trade-offs, "
        "or making existing side routes more rewarding.",
    ),
}


def compute_world_metrics(game, world, df_all, persona=None):
    """(B-cards, every registry metric) for one level; persona=None pools every persona."""
    eps = episodes(df_all, game, world, persona)
    if not eps:
        return {}, {}
    th = registry.load_thresholds(game)
    m = registry.compute_level(eps, game, th.get("route_cluster_threshold", 96))
    zones = dict(registry.verdicts(m, game, th))
    legacy = _legacy_thresholds(th["bands"])
    wins = [e for e in eps if e.get("status") == "WON"]
    fails = [e for e in eps if e.get("status") in registry.FAILED]

    def zone(card):
        state = zones.get(CARD_DIM[card], "bad")
        key, color, pill, pt = ZONE[card][state]
        return {"zone_key": key, "zone_color": color, "zone_pill": pill, "score_pt": pt,
                f"{card.lower()}_key": key, f"{card.lower()}_color": color, f"{card.lower()}_pill": pill}

    cards = {
        "B1_challenge_calibration": {
            "type": "B1", "completion_rate": m["completion_rate"],
            "mean_completion_time": m["mean_completion_time"],
            "completion_time_stddev": m["completion_time_stddev"],
            "progress_at_death": m.get("progress_at_death"),
            "total_runs": len(eps), "successful_runs": len(wins), "thresholds": legacy, **zone("B1")},
        "B2_punishment_severity": {
            "type": "B2", "deaths_per_run": m["deaths_per_run"],
            "death_cluster_entropy": m["death_cluster_entropy"] or 0.0,
            "total_runs": len(eps), "total_deaths": len(fails), "total_successes": len(wins),
            "thresholds": legacy, **zone("B2")},
    }
    if "strategy_count" in m:  # route clustering applies to this game
        if m["strategy_count"] is None:
            cards["B3_strategy_diversity"] = {
                "error": "Need 2+ winning runs with logged routes for strategy diversity (have %d)" % len(wins)}
        else:
            cards["B3_strategy_diversity"] = {
                "type": "B3", "strategy_count": m["strategy_count"],
                "dominant_path_share": m["dominant_path_share"],
                "safe_vs_fast_ratio": m["safe_vs_fast_path_ratio"] or 1.0,
                "dominant_runs": round(m["dominant_path_share"] * len(wins)),
                "total_successful": len(wins), "thresholds": legacy, **zone("B3")}
    return cards, m
