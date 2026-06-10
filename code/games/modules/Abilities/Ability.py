from __future__ import annotations
from typing import List, Set


class Ability:
    """Base class for a composable body capability.

    Subclasses set velocity / gravity_scale on the MotorState in update().
    They declare which named intents they consume (drives the action space)
    and which observation fields they expose (drives the obs vector).
    """
    def __init__(self, name: str, **params):
        self.name = name
        self.params = params
        self.active = bool(params.get("start_active", True))

    def update(self, state, ctx, dt) -> None:
        """Read/mutate MotorState. No-op in the base class."""
        return None

    def requires(self) -> Set[str]:
        """Intent names this ability consumes (e.g. {'jump_pressed','jump_held'})."""
        return set()

    def obs_spec(self) -> List[str]:
        """Named float features this ability contributes to the obs vector."""
        return []

    def write_obs(self, state) -> List[float]:
        """Values for obs_spec() fields this frame; same length & order."""
        return []
