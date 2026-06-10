from __future__ import annotations
from typing import List, Set
from .Ability import Ability


class WallJump(Ability):
    """On a jump press while airborne and touching a wall, launch up and away
    from the wall. Consumes jump_pressed so the Jump ability (later in the
    pipeline) does not also fire, and sets a brief air-control lockout so the
    horizontal push survives.

    Wall jumps are UNLIMITED: there is no per-air counter, and the jump is
    available on ANY wall contact (the player does not need to hold toward the
    wall). Keep hitting walls and pressing jump and you keep wall-jumping."""

    def requires(self) -> Set[str]:
        return {"jump_pressed", "jump_held"}

    def obs_spec(self) -> List[str]:
        return ["wall_jump_ready"]

    def _wall_dir(self, state) -> int:
        # Contact-only: a wall jump is available whenever a wall is touched,
        # regardless of input direction. (right wall wins ties in a 1-tile gap.)
        if state.contact_right:
            return 1
        if state.contact_left:
            return -1
        return 0

    def write_obs(self, state) -> List[float]:
        ready = (not state.on_ground) and self._wall_dir(state) != 0
        return [1.0 if ready else 0.0]

    def update(self, state, ctx, dt) -> None:
        if state.on_ground or not state.intents.jump_pressed:
            return
        wall = self._wall_dir(state)
        if wall == 0:
            return
        p = self.params
        push = float(p["wall_jump_push"])
        # Holding away from the wall = long jump (the reference remake boosts
        # the push 1.5x vs 1.2x when the player pre-holds the away direction).
        if state.intents.move_x == -wall:
            push = float(p.get("wall_jump_push_away", push))
        state.vy = -float(p["wall_jump_vy"])
        state.vx = -wall * push                        # away from the wall
        state.facing_right = (wall < 0)                # face away from wall
        state.air_lockout = int(p.get("control_lockout_frames", 6))
        state.ext["wall_jumped_this_frame"] = True
        state.intents.consume("jump_pressed")
