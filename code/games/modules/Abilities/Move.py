from __future__ import annotations
from typing import List, Set
from .Ability import Ability


class Move(Ability):
    """Horizontal movement: walk/sprint target speeds, high air control, skid."""

    def requires(self) -> Set[str]:
        return {"move_x", "run_held"}

    def obs_spec(self) -> List[str]:
        return ["sprinting"]

    def write_obs(self, state) -> List[float]:
        return [1.0 if state.intents.run_held else 0.0]

    def update(self, state, ctx, dt) -> None:
        p = self.params
        it = state.intents

        # WallJump briefly locks horizontal accel so its push isn't cancelled.
        if state.air_lockout > 0:
            return

        sprint = it.run_held
        target_max = p["max_run_speed"] if sprint else p["max_walk_speed"]
        accel = p["run_accel"] if sprint else p["walk_accel"]
        if not state.on_ground:
            accel *= p["air_control"]

        if it.move_x != 0:
            skidding = (state.vx > 0 and it.move_x < 0) or (state.vx < 0 and it.move_x > 0)
            if state.on_ground and skidding:
                state.vx += it.move_x * p.get("skid_decel", accel * 2.0) * dt
            elif it.move_x > 0:
                if state.vx <= target_max:
                    state.vx = min(state.vx + accel * dt, target_max)
                else:
                    # Above the cap (wall-jump boost): bleed off with friction
                    # instead of clamping, so the launch momentum is usable.
                    fr = (p["ground_friction"] if state.on_ground else p["air_friction"]) * dt
                    state.vx = max(target_max, state.vx - fr)
            else:
                if state.vx >= -target_max:
                    state.vx = max(state.vx - accel * dt, -target_max)
                else:
                    fr = (p["ground_friction"] if state.on_ground else p["air_friction"]) * dt
                    state.vx = min(-target_max, state.vx + fr)
        else:
            friction = (p["ground_friction"] if state.on_ground else p["air_friction"]) * dt
            if state.vx > 0:
                state.vx = max(0.0, state.vx - friction)
            elif state.vx < 0:
                state.vx = min(0.0, state.vx + friction)
