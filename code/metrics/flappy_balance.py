"""
Advanced balancing metrics for Flappy Bird.
Compatible with code.metrics.base.MetricsCollector.
"""

from collections import defaultdict
import numpy as np
from code.metrics.base import MetricsCollector


class FlappyBalanceStats(MetricsCollector):
    """Captures DIF, SKI, PAC, LEN, FAI, PRO metrics each episode."""

    # ---------------------------------------------------------------- reset
    def reset(self):
        super().reset()
        self.data.update(
            # raw signal buffers
            gap_heights      = [],
            flap_times       = [],   # frame indices of flaps
            timing_errors    = [],   # |bird_y - gap_mid| at pipe crossing
            idle_frames      = 0,
            total_frames     = 0,
            min_clearances   = [],
            difficulty_idx   = [],
            fail_pipe_num    = 0,
        )
        self._prev_score   = 0
        self._last_flap_ts = None

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _difficulty_index(pipe_speed, gap_height):
        return pipe_speed / gap_height  # simple inverse-gap heuristic

    # ---------------------------------------------------------------- per step
    def on_step(self, obs, action, reward, done, info):
        """
        obs = [bird_y, bird_vel, dist_x_to_pipe, gap_top, gap_bottom]
        action = 0 | 1
        """
        y, _, dist_x, gap_top, gap_bottom = obs
        gap_h  = gap_bottom - gap_top
        gap_mid = (gap_top + gap_bottom) / 2

        self.data["total_frames"] += 1
        if action == 0:
            self.data["idle_frames"] += 1

        # store flap times for PAC/SKI
        if action == 1:
            self.data["flap_times"].append(self.data["total_frames"])
            self._last_flap_ts = self.data["total_frames"]

        # store first appearance of each pipe gap (distance ~ pipe width)
        if dist_x > 0 and dist_x < 4:          # passing left edge
            self.data["gap_heights"].append(gap_h)
            # Timing error = vertical distance when bird *enters* the gap
            self.data["timing_errors"].append(abs(y - gap_mid))
            # Difficulty curve
            self.data["difficulty_idx"].append(
                self._difficulty_index(pipe_speed=4, gap_height=gap_h)
            )

        # clearance every frame (gap minus bird)
        clearance = gap_h - 20   # bird size
        self.data["min_clearances"].append(clearance)

        # detect pipe number at failure
        if done:
            self.data["fail_pipe_num"] = info.get("score", 0) + 1

    # ---------------------------------------------------------------- summary
    def summary(self):
        d = self.data  # shorthand

        # ----- DIF ---------------------------------------------------------
        dif_avg_gap = float(np.mean(d["gap_heights"])) if d["gap_heights"] else 0
        if len(d["gap_heights"]) >= 2:
            dif_dgap = (d["gap_heights"][-1] - d["gap_heights"][0]) / max(
                1, (len(d["gap_heights"]) - 1)
            )
        else:
            dif_dgap = 0

        # ----- SKI ---------------------------------------------------------
        mean_timing_err = float(np.mean(d["timing_errors"])) if d["timing_errors"] else 0
        aps = len(d["flap_times"]) / max(1, d["total_frames"] / 60)  # frames→seconds

        # ----- PAC ---------------------------------------------------------
        idle_ratio = d["idle_frames"] / max(1, d["total_frames"])

        # ----- LEN ---------------------------------------------------------
        runtime = d["total_frames"] / 60          # seconds (assuming 60 FPS)
        pipes_cleared = self._prev_score  # stored at last step

        # ----- FAI ---------------------------------------------------------
        min_clear = min(d["min_clearances"]) if d["min_clearances"] else 0

        # ----- PRO ---------------------------------------------------------
        score_velocity = self._prev_score / runtime if runtime else 0
        if len(d["difficulty_idx"]) >= 2:
            pro_ddt = (
                d["difficulty_idx"][-1] - d["difficulty_idx"][0]
            ) / len(d["difficulty_idx"])
        else:
            pro_ddt = 0

        return {
            # Difficulty
            "DIF_AvgGap": dif_avg_gap,
            "DIF_dGapPerPipe": dif_dgap,
            "DIF_FailPipe": d["fail_pipe_num"],
            # Skill
            "SKI_MeanTimingErr": mean_timing_err,
            "SKI_APS": aps,
            # Pace
            "PAC_MeanIdleRatio": idle_ratio,
            # Length
            "LEN_RunTime": runtime,
            "LEN_PipesCleared": pipes_cleared,
            # Fairness
            "FAI_MinClearance": min_clear,
            # Progression
            "PRO_ScoreVelocity": score_velocity,
            "PRO_dDifficulty_dt": pro_ddt,
        }
