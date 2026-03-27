import importlib
import inspect
from functools import wraps
import yaml


def load_tracking_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def make_tracker_wrapper(func, recorder_name, tracked_arg_names, stats_observer):
    sig = inspect.signature(func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)

        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        arguments = dict(bound.arguments)
        arguments.pop("self", None)

        recorder_kwargs = {}
        for arg_name in tracked_arg_names:
            if arg_name not in arguments:
                raise KeyError(
                    f"Argument '{arg_name}' not found in function '{func.__qualname__}'"
                )
            recorder_kwargs[arg_name] = arguments[arg_name]

        stats_observer.dispatch_record(recorder_name, **recorder_kwargs)
        return result

    return wrapper

def apply_tracking_from_config(config, stats_observer):
    for entry in config.get("tracks", []):
        target = entry["target"]
        recorder_name = entry["recorder"]
        tracked_arg_names = entry.get("args", [])

        parts = target.split(".")
        if len(parts) < 3:
            raise ValueError(
                f"Invalid target '{target}'. Expected format: module.Class.method"
            )

        module_path = ".".join(parts[:-2])
        class_name = parts[-2]
        method_name = parts[-1]

        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        original_method = getattr(cls, method_name)

        wrapped_method = make_tracker_wrapper(
            original_method,
            recorder_name,
            tracked_arg_names,
            stats_observer,
        )

        setattr(cls, method_name, wrapped_method)