from __future__ import annotations

__all__ = ["PlatformerCore", "SonicCore"]


def __getattr__(name: str):
    if name == "PlatformerCore":
        from .platformer_core import PlatformerCore
        return PlatformerCore
    if name == "SonicCore":
        from .sonic_core import SonicCore
        return SonicCore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
