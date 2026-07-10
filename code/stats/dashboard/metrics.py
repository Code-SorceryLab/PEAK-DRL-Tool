# -*- coding: utf-8 -*-
"""Metric computation, classification, and analysis text for the PEAK dashboard."""

import numpy as np
import pandas as pd

from data import parse_route

ENTROPY_BINS = 64
ROUTE_PROFILE_BINS = 50


def classify_b1_calibration(completion_rate, mean_time, thresholds):
    t = thresholds
    target_cr      = t.get("target_completion_rate", 0.7)
    warn_cr_diff   = t.get("warning_completion_rate_difference", 0.2)
    target_time    = t.get("target_mean_completion_time", 15)
    warn_time_diff = t.get("warning_mean_completion_time_difference", 4)
    cr_in_range   = abs(completion_rate - target_cr) <= warn_cr_diff
    time_in_range = mean_time is None or abs(mean_time - target_time) <= warn_time_diff
    if cr_in_range and time_in_range:
        return "calibrated", "#22c55e", "pill-balanced", 1
    if cr_in_range or time_in_range:
        return "warning", "#eab308", "pill-warning", 0
    return "miscalibrated", "#ef4444", "pill-imbalance", 0


def compute_death_cluster_entropy(progress_ratios):
    if len(progress_ratios) == 0:
        return 0.0
    counts, _ = np.histogram(progress_ratios, bins=ENTROPY_BINS, range=(0.0, 1.0))
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts[counts > 0] / total
    raw = float(-np.sum(probs * np.log2(probs)))
    return raw / np.log2(ENTROPY_BINS)


def classify_b2_punishment(deaths_per_run, entropy, thresholds):
    t = thresholds
    target_dpr     = t.get("target_deaths_per_run", 2)
    warn_dpr       = t.get("warning_deaths_per_run", 1)
    target_entropy = t.get("target_death_cluster_entropy", 0.4)
    warn_entropy   = t.get("warning_death_cluster_entropy", 0.1)
    dpr_in_range     = abs(deaths_per_run - target_dpr) <= warn_dpr
    entropy_in_range = abs(entropy - target_entropy) <= warn_entropy
    if dpr_in_range and entropy_in_range:
        return "calibrated", "#22c55e", "pill-balanced", 1
    if dpr_in_range or entropy_in_range:
        return "warning", "#eab308", "pill-warning", 0
    return "severe", "#ef4444", "pill-imbalance", 0


def route_to_profile(route, n_bins=ROUTE_PROFILE_BINS, x_range=None):
    if len(route) < 2:
        return None
    xs = np.array([p[0] for p in route], dtype=np.float64)
    ys = np.array([p[1] for p in route], dtype=np.float64)
    if x_range is None:
        x_min, x_max = xs.min(), xs.max()
    else:
        x_min, x_max = x_range
    if x_max - x_min < 1:
        return None
    edges = np.linspace(x_min, x_max, n_bins + 1)
    profile = np.full(n_bins, np.nan)
    for i in range(n_bins):
        mask = (xs >= edges[i]) & (xs < edges[i + 1])
        if mask.any():
            profile[i] = ys[mask].mean()
    valid = ~np.isnan(profile)
    if valid.sum() < 2:
        return None
    profile = np.interp(np.arange(n_bins), np.where(valid)[0], profile[valid])
    return profile


def route_distance(a, b):
    return float(np.mean(np.abs(a - b)))


def cluster_routes(profiles, threshold):
    if len(profiles) == 0:
        return [], []
    clusters = [[0]]
    centroids = [profiles[0].copy()]
    for i in range(1, len(profiles)):
        best_c, best_d = -1, float("inf")
        for ci, cen in enumerate(centroids):
            d = route_distance(profiles[i], cen)
            if d < best_d:
                best_d = d
                best_c = ci
        if best_d <= threshold:
            clusters[best_c].append(i)
            centroids[best_c] = np.mean(
                [profiles[j] for j in clusters[best_c]], axis=0
            )
        else:
            clusters.append([i])
            centroids.append(profiles[i].copy())
    return clusters, centroids


def classify_b3_diversity(strategy_count, dominant_share, thresholds):
    t = thresholds
    tgt_sc  = t.get("target_strategy_count", 3)
    warn_sc = t.get("warning_strategy_count", 1)
    tgt_ds  = t.get("target_dominant_path_share", 0.5)
    warn_ds = t.get("warning_dominant_path_share", 0.15)
    sc_ok = abs(strategy_count - tgt_sc) <= warn_sc
    ds_ok = abs(dominant_share - tgt_ds) <= warn_ds
    if sc_ok and ds_ok:
        return "diverse", "#22c55e", "pill-balanced", 1
    if sc_ok or ds_ok:
        return "warning", "#eab308", "pill-warning", 0
    return "homogeneous", "#ef4444", "pill-imbalance", 0


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


def compute_world_metrics(world, cfg, df_all):
    results = {}
    for metric_key, metric_cfg in cfg.items():
        thresholds = metric_cfg.get("thresholds", {})

        if "B1" in metric_key.upper() or "calibration" in metric_key.lower():
            sub = df_all[df_all["world"] == world]
            if len(sub) == 0:
                results[metric_key] = {"error": "No data for world " + repr(world)}
                continue
            successes = sub["cause_of_death"].str.lower() == "success"
            completion_rate = successes.sum() / len(sub)
            successful_runs = sub[successes]
            if len(successful_runs) > 0:
                mean_completion_time = successful_runs["elapsed_time"].mean()
                completion_time_stddev = successful_runs["elapsed_time"].std(ddof=0)
                if pd.isna(completion_time_stddev):
                    completion_time_stddev = 0.0
            else:
                mean_completion_time = None
                completion_time_stddev = None
            failed_runs = sub[~successes]
            progress_at_death = (
                failed_runs["progress_ratio"].mean() if len(failed_runs) > 0 else None
            )
            zone_key, zone_color, zone_pill, score_pt = classify_b1_calibration(
                completion_rate, mean_completion_time, thresholds
            )
            results[metric_key] = {
                "type": "B1",
                "completion_rate": completion_rate,
                "mean_completion_time": mean_completion_time,
                "completion_time_stddev": completion_time_stddev,
                "progress_at_death": progress_at_death,
                "zone_key": zone_key,
                "zone_color": zone_color,
                "zone_pill": zone_pill,
                "score_pt": score_pt,
                "thresholds": thresholds,
                "total_runs": len(sub),
                "successful_runs": int(successes.sum()),
            }

        elif "B2" in metric_key.upper() or "punishment" in metric_key.lower():
            sub = df_all[df_all["world"] == world]
            if len(sub) == 0:
                results[metric_key] = {"error": "No data for world " + repr(world)}
                continue
            successes = sub["cause_of_death"].str.lower() == "success"
            total_runs = len(sub)
            total_deaths = int((~successes).sum())
            total_successes = int(successes.sum())
            deaths_per_run = total_deaths / max(total_successes, 1)
            failed_progress = sub.loc[~successes, "progress_ratio"].dropna().values
            entropy = compute_death_cluster_entropy(failed_progress)
            b2_key, b2_color, b2_pill, score_pt = classify_b2_punishment(
                deaths_per_run, entropy, thresholds
            )
            results[metric_key] = {
                "type": "B2",
                "deaths_per_run": deaths_per_run,
                "death_cluster_entropy": entropy,
                "total_runs": total_runs,
                "total_deaths": total_deaths,
                "total_successes": total_successes,
                "b2_key": b2_key,
                "b2_color": b2_color,
                "b2_pill": b2_pill,
                "score_pt": score_pt,
                "thresholds": thresholds,
            }

        elif "B3" in metric_key.upper() or "diversity" in metric_key.lower():
            sub = df_all[df_all["world"] == world]
            successes = sub[sub["cause_of_death"].str.lower() == "success"]
            if len(successes) < 2:
                results[metric_key] = {
                    "error": "Need >= 2 successful runs for strategy diversity (have %d)" % len(successes)
                }
                continue
            cluster_thresh = thresholds.get("route_cluster_threshold", 96)
            routes_raw = [parse_route(r) for r in successes["route"]]
            all_xs = [p[0] for route in routes_raw if route for p in route]
            if not all_xs:
                results[metric_key] = {"error": "No valid route data in successful runs"}
                continue
            x_range = (min(all_xs), max(all_xs))
            profiles, profile_indices = [], []
            for i, route in enumerate(routes_raw):
                if not route or len(route) < 2:
                    continue
                prof = route_to_profile(route, ROUTE_PROFILE_BINS, x_range)
                if prof is not None:
                    profiles.append(prof)
                    profile_indices.append(i)
            if len(profiles) < 2:
                results[metric_key] = {"error": "Not enough valid route profiles to cluster"}
                continue
            clusters, _centroids = cluster_routes(profiles, cluster_thresh)
            strategy_count = len(clusters)
            largest_cluster = max(len(c) for c in clusters)
            dominant_path_share = largest_cluster / len(profiles)
            elapsed_vals = successes["elapsed_time"].values
            cluster_mean_times = []
            for c in clusters:
                times = []
                for j in c:
                    idx = profile_indices[j]
                    if idx < len(elapsed_vals) and not np.isnan(elapsed_vals[idx]):
                        times.append(elapsed_vals[idx])
                if times:
                    cluster_mean_times.append(np.mean(times))
            if len(cluster_mean_times) >= 2:
                safe_vs_fast = max(cluster_mean_times) / max(min(cluster_mean_times), 0.01)
            else:
                safe_vs_fast = 1.0
            b3_key, b3_color, b3_pill, score_pt = classify_b3_diversity(
                strategy_count, dominant_path_share, thresholds
            )
            results[metric_key] = {
                "type": "B3",
                "strategy_count": strategy_count,
                "dominant_path_share": dominant_path_share,
                "safe_vs_fast_ratio": round(safe_vs_fast, 3),
                "total_successful": len(profiles),
                "cluster_sizes": [len(c) for c in clusters],
                "cluster_mean_times": [round(t, 2) for t in cluster_mean_times],
                "b3_key": b3_key,
                "b3_color": b3_color,
                "b3_pill": b3_pill,
                "score_pt": score_pt,
                "thresholds": thresholds,
            }

    return results
