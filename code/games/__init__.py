from __future__ import annotations

__all__ = ["PlatformerCore"]


def __getattr__(name: str):
    if name == "PlatformerCore":
        from .platformer_core import PlatformerCore
        return PlatformerCore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
