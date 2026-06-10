from __future__ import annotations
from typing import List, Set
from .Ability import Ability


class Jump(Ability):
    """Variable-height jump with coyote time, input buffering, and a
    configurable max-jumps count (1 = single jump; >1 enables air jumps,
    used later by power-ups)."""

    def _st(self, state):
        return state.ext.setdefault(self.name, {
            "coyote": 0, "buffer": 0, "jumps_left": self.params.get("max_jumps", 1),
        })

    def requires(self) -> Set[str]:
        return {"jump_pressed", "jump_held"}

    def obs_spec(self) -> List[str]:
        return ["coyote_active", "jump_extendable", "jumps_left"]

    def write_obs(self, state) -> List[float]:
        d = state.ext.get(self.name, {"coyote": 0, "buffer": 0,
                                      "jumps_left": self.params.get("max_jumps", 1)})
        max_j = max(1, self.params.get("max_jumps", 1))
        return [
            1.0 if d["coyote"] > 0 else 0.0,
            1.0 if (not state.on_ground and state.vy < 0) else 0.0,
            d["jumps_left"] / max_j,
        ]

    def update(self, state, ctx, dt) -> None:
        p = self.params
        d = self._st(state)
        it = state.intents

        # Refresh ground-based grants
        if state.on_ground:
            d["coyote"] = p.get("coyote_frames", 6)
            d["jumps_left"] = p.get("max_jumps", 1)
        else:
            d["coyote"] = max(0, d["coyote"] - 1)

        # Input buffer
        if it.jump_pressed:
            d["buffer"] = p.get("buffer_frames", 6)
        elif d["buffer"] > 0:
            d["buffer"] -= 1

        can_ground_jump = (state.on_ground or d["coyote"] > 0)
        can_air_jump = d["jumps_left"] > 0 and p.get("max_jumps", 1) > 1
        if d["buffer"] > 0 and (can_ground_jump or can_air_jump):
            state.vy = -float(p["jump_vel"])
            state.on_ground = False
            d["coyote"] = 0
            d["buffer"] = 0
            if not can_ground_jump:
                d["jumps_left"] -= 1   # used an air jump
            else:
                d["jumps_left"] = max(0, p.get("max_jumps", 1) - 1)

        # A wall jump this frame owns vy — don't let the variable-height cut
        # clamp its impulse. Clear the one-frame flag either way.
        if state.ext.pop("wall_jumped_this_frame", False):
            return
        # Variable height: releasing jump while rising clamps upward speed.
        if (not it.jump_held) and state.vy < 0:
            cut = -float(p.get("cut_vel", 200.0))
            if state.vy < cut:
                state.vy = cut
