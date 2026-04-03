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

        return func(*args, **kwargs)

    return wrapper


def make_reset_wrapper(func, stats_observer):
    sig = inspect.signature(func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        arguments = dict(bound.arguments)
        self_obj = arguments.get("self")

        result = func(*args, **kwargs)

        world = getattr(self_obj, "world", None)
        stats_observer.reset(world)

        return result

    return wrapper

def apply_wrapper_to_target(target, wrapper_factory, *factory_args):
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

    wrapped_method = wrapper_factory(original_method, *factory_args)
    setattr(cls, method_name, wrapped_method)


def apply_tracking_from_config(config, stats_observer):
    for entry in config.get("tracks", []):
        target = entry["target"]
        recorder_name = entry["recorder"]
        tracked_arg_names = entry.get("args", [])

        apply_wrapper_to_target(
            target,
            make_tracker_wrapper,
            recorder_name,
            tracked_arg_names,
            stats_observer,
        )

    for entry in config.get("resets", []):
        target = entry["target"]
        reset_arg_names = entry.get("args", [])

        apply_wrapper_to_target(
            target,
            make_reset_wrapper,
            stats_observer,
        )