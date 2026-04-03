from .stats_tracking_manager import load_tracking_config, apply_tracking_from_config
import time
import csv
import os
import logging
import importlib

class stats_observer:
    def __init__(self, config_path, csv_path="stats.csv"):
        self.currentAttempt = None
        self.last_reset_time = time.time()
        self.csv_path = csv_path
        self.stats_class = None
        self._setup_tracking(config_path)

    def _resolve_stats_class(self, stats_class_path):
        try:
            parts = stats_class_path.split(".")
            if len(parts) < 2:
                raise ValueError(f"Invalid stats_class '{stats_class_path}'")

            module_path = ".".join(parts[:-1])
            class_name = parts[-1]

            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except Exception:
            logging.warning(
                "Stats Observer Cannot Find Function %s with arguments %s",
                stats_class_path,
                [],
            )
            return None

    def _setup_tracking(self, config_path):
        config = load_tracking_config(config_path)

        stats_class_path = config.get("stats_class")
        if stats_class_path:
            self.stats_class = self._resolve_stats_class(stats_class_path)

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

        try:
            row = self.currentAttempt.to_dict()
            row["elapsed_time"] = round(self.get_elapsed_time(), 2)

            file_exists = os.path.exists(self.csv_path)

            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())

                if not file_exists:
                    writer.writeheader()

                writer.writerow(row)
        except Exception:
            logging.warning(
                "Stats Observer Cannot Find Function %s with arguments %s",
                type(self.currentAttempt).__name__,
                ["to_dict"],
            )

    def reset(self, world):
        self._write_current_attempt_to_csv()

        if self.stats_class is None:
            logging.warning(
                "Stats Observer Cannot Find Function %s with arguments %s",
                "stats_class",
                ["world"],
            )
            self.currentAttempt = None
        else:
            try:
                self.currentAttempt = self.stats_class(world)
            except Exception:
                logging.warning(
                    "Stats Observer Cannot Find Function %s with arguments %s",
                    getattr(self.stats_class, "__name__", str(self.stats_class)),
                    ["world"],
                )
                self.currentAttempt = None

        self.last_reset_time = time.time()