from .stats_attempt_observer import stats_attempt_observer
from .stats_tracking_manager import load_tracking_config, apply_tracking_from_config
import time
import csv
import os
import logging

class stats_observer:
    def __init__(self, config_path, csv_path="stats.csv"):
        self.currentAttempt = None
        self.last_reset_time = time.time()
        self.csv_path = csv_path
        self._setup_tracking(config_path)

    def _setup_tracking(self, config_path):
        config = load_tracking_config(config_path)
        apply_tracking_from_config(config, self)

    def dispatch_record(self, recorder_name, **data):
        if self.currentAttempt is None:
            return

        try:
            recorder = getattr(self.currentAttempt, recorder_name, None)
            if recorder is None:
                raise AttributeError(recorder_name)

            recorder(**data)
        except Exception:
            logging.warning(
                "Stats Observer Cannot Find Function %s with arguments %s",
                recorder_name,
                list(data.keys()),
            )

    def get_elapsed_time(self):
        return time.time() - self.last_reset_time

    def _write_current_attempt_to_csv(self):
        if self.currentAttempt is None:
            return

        row = self.currentAttempt.to_dict()
        row["elapsed_time"] = round(self.get_elapsed_time(),2)

        file_exists = os.path.exists(self.csv_path)

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())

            if not file_exists:
                writer.writeheader()

            writer.writerow(row)

    def reset(self, world):
        self._write_current_attempt_to_csv()
        self.currentAttempt = stats_attempt_observer(world)
        self.last_reset_time = time.time()