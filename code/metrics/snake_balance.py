import numpy as np

class SnakeBalanceStats:
    """
    Eval metrics for Snake.
    Accepts either (obs, reward, done, info) or (obs, reward, terminated, truncated, info).

    Adds:
      - closer_steps / away_steps (+ per-episode means) and ratios over total steps
      - oscillations (total + per-episode mean)
      - stall streak (max_no_progress_streak mean)
      - distance-delta aggregates (per-episode mean + overall mean)
      - apples_per_1000_steps
      - time_to_first_apple mean/median
      - distribution summaries (p50/p90) for returns/steps/apples
    """

    def __init__(self):
        self.reset_all()

    # ---------- life cycle ----------
    def reset_all(self):
        # episode aggregates
        self.episodes = 0
        self.returns = []
        self.lengths = []
        self.apples = []
        self.steps = []
        self.time_to_first_apple = []

        # rolling within episode accumulators
        self._ret = 0.0
        self._len = 0
        self._apples = 0
        self._steps = 0
        self._first_apple_step = None

        # extra signals (episode level tallies)
        self._closer_steps = 0
        self._away_steps = 0
        self._oscillations = 0
        self._max_no_progress_streak = 0
        self._dist_delta_sum = 0.0
        self._dist_delta_count = 0

        # episode level history exports
        self.closer_steps_hist = []
        self.away_steps_hist = []
        self.oscillations_hist = []
        self.max_no_progress_streak_hist = []
        self.dist_delta_mean_hist = []

        # totals across all episodes
        self.total_steps = 0
        self.total_closer_steps = 0
        self.total_away_steps = 0
        self.total_oscillations = 0
        self.total_dist_delta_sum = 0.0
        self.total_dist_delta_count = 0

    def _finish_episode(self):
        self.episodes += 1

        self.returns.append(self._ret)
        self.lengths.append(self._len)
        self.apples.append(self._apples)
        self.steps.append(self._steps)

        if self._first_apple_step is not None:
            self.time_to_first_apple.append(int(self._first_apple_step))

        # per episode extras
        self.closer_steps_hist.append(int(self._closer_steps))
        self.away_steps_hist.append(int(self._away_steps))
        self.oscillations_hist.append(int(self._oscillations))
        self.max_no_progress_streak_hist.append(int(self._max_no_progress_streak))
        mean_dd = (self._dist_delta_sum / self._dist_delta_count) if self._dist_delta_count > 0 else 0.0
        self.dist_delta_mean_hist.append(float(mean_dd))

        # rollup totals
        self.total_steps += self._steps
        self.total_closer_steps += self._closer_steps
        self.total_away_steps += self._away_steps
        self.total_oscillations += self._oscillations
        self.total_dist_delta_sum += self._dist_delta_sum
        self.total_dist_delta_count += self._dist_delta_count

        # reset per episode accumulators
        self._ret = 0.0
        self._len = 0
        self._apples = 0
        self._steps = 0
        self._first_apple_step = None
        self._closer_steps = 0
        self._away_steps = 0
        self._oscillations = 0
        self._max_no_progress_streak = 0
        self._dist_delta_sum = 0.0
        self._dist_delta_count = 0

    # ---------- evaluator hooks ----------
    def on_reset(self, *args, **kwargs):
        return None

    def on_step(self, *args, **kwargs):
        """
        Accepts:
          (obs, reward, done, info)
        or
          (obs, reward, terminated, truncated, info)
        """
        # Unpack flexibly
        if len(args) == 4:
            obs, reward, done, info = args
        elif len(args) == 5:
            obs, reward, terminated, truncated, info = args
            done = bool(terminated or truncated)
        else:
            obs      = kwargs.get("obs", None)
            reward   = kwargs.get("reward", 0.0)
            term     = kwargs.get("terminated", False)
            trunc    = kwargs.get("truncated", False)
            info     = kwargs.get("info", {}) or {}
            done     = bool(term or trunc)

        info = info or {}

        # sum vectorized reward if needed
        try:
            if isinstance(reward, (list, tuple, np.ndarray)):
                self._ret += float(np.sum(reward))
            else:
                self._ret += float(reward)
        except Exception:
            pass

        # length
        if "length" in info:
            try:
                self._len = int(info["length"])
            except Exception:
                pass

        # apples
        if info.get("ate"):
            self._apples += 1
            if self._first_apple_step is None:
                self._first_apple_step = self._steps

        # progress / oscillation / stall
        if info.get("moved_closer", False):
            self._closer_steps += 1

        dd = info.get("dist_delta", None)
        if dd is not None:
            try:
                dd = float(dd)
                if dd < 0:
                    self._away_steps += 1
                self._dist_delta_sum += dd
                self._dist_delta_count += 1
            except Exception:
                pass

        if info.get("oscillating", False):
            self._oscillations += 1

        nps = info.get("no_progress_steps", None)
        if nps is not None:
            try:
                self._max_no_progress_streak = max(self._max_no_progress_streak, int(nps))
            except Exception:
                pass

        # step counter
        self._steps += 1

        if done:
            self._finish_episode()

    # aliases some evaluators expect
    def step(self, *args, **kwargs):  return self.on_step(*args, **kwargs)
    def reset(self, *args, **kwargs): return self.on_reset(*args, **kwargs)

    # ---------- export ----------
    def _summary(self):
        def mean(x): return float(np.mean(x)) if x else 0.0
        def imax(x): return int(np.max(x)) if x else 0
        def ratio(a, b): return float(a) / float(b) if b else 0.0
        def pctl(x, q): 
            try:
                return float(np.percentile(x, q)) if x else 0.0
            except Exception:
                return 0.0

        out = {
            "episodes": int(self.episodes),

            # base means/max
            "return_mean": mean(self.returns),
            "return_max": float(np.max(self.returns)) if self.returns else 0.0,
            "length_mean": mean(self.lengths),
            "length_max": imax(self.lengths),
            "apples_mean": mean(self.apples),
            "apples_max": imax(self.apples),
            "steps_mean": mean(self.steps),
            "steps_max": imax(self.steps),

            # distribution extras
            "return_p50": pctl(self.returns, 50),
            "return_p90": pctl(self.returns, 90),
            "steps_p50":  pctl(self.steps, 50),
            "steps_p90":  pctl(self.steps, 90),
            "apples_p50": pctl(self.apples, 50),
            "apples_p90": pctl(self.apples, 90),
        }

        # extras over totals
        out.update({
            "apples_per_1000_steps": (1000.0 * (sum(self.apples) / self.total_steps)) if self.total_steps else 0.0,
            "away_ratio": ratio(self.total_away_steps, self.total_steps),
            "closer_ratio": ratio(self.total_closer_steps, self.total_steps),
            "oscillations_total": int(self.total_oscillations),
            "oscillations_mean": mean(self.oscillations_hist),
            "closer_steps_mean": mean(self.closer_steps_hist),
            "away_steps_mean": mean(self.away_steps_hist),
            "max_no_progress_streak_mean": mean(self.max_no_progress_streak_hist),
            "time_to_first_apple_mean": mean(self.time_to_first_apple),
            "time_to_first_apple_median": pctl(self.time_to_first_apple, 50),
            "dist_delta_mean_overall": (self.total_dist_delta_sum / self.total_dist_delta_count) if self.total_dist_delta_count else 0.0,
            "dist_delta_mean_per_episode": mean(self.dist_delta_mean_hist),
        })
        return out

    def results(self):   return self._summary()
    def finalize(self):  return self._summary()
    def summary(self):   return self._summary()
    def to_json(self):   return self._summary()
