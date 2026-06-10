from __future__ import annotations
from typing import List, Set
from .Ability import Ability


class WallSlide(Ability):
    """When airborne, falling, and pressing into a contacted wall, clamp the
    fall speed. Exposes which wall is being hugged for WallJump and the obs."""

    def requires(self) -> Set[str]:
        return {"move_x"}

    def obs_spec(self) -> List[str]:
        return ["wall_slide_active", "touching_wall"]

    def _wall_dir(self, state) -> int:
        # +1 = wall on the right (pushing right), -1 = wall on the left
        if state.contact_right and state.intents.move_x > 0:
            return 1
        if state.contact_left and state.intents.move_x < 0:
            return -1
        return 0

    def write_obs(self, state) -> List[float]:
        d = state.ext.get(self.name, {})
        return [1.0 if d.get("sliding") else 0.0, float(d.get("dir", 0))]

    def update(self, state, ctx, dt) -> None:
        d = state.ext.setdefault(self.name, {"sliding": False, "dir": 0})
        wall = self._wall_dir(state)
        sliding = (not state.on_ground) and state.vy > 0 and wall != 0
        d["sliding"] = sliding
        d["dir"] = wall
        if sliding:
            cap = float(self.params.get("slide_max_speed", 120.0))
            if state.vy > cap:
                state.vy = cap
