from __future__ import annotations
from typing import List
import numpy as np
from gymnasium import spaces
from .MotorState import Intent

# Canonical axis order. Each entry: (axis_key, size, intents it feeds).
# An axis is included only if some ability requires one of its intents.
_AXES = [
    ("move_x",    3, {"move_x"}),
    ("run_held",  2, {"run_held"}),
    ("jump",      2, {"jump_pressed", "jump_held"}),
    ("fire",      2, {"fire_pressed", "fire_held"}),
    ("down_held", 2, {"down_held"}),
    ("up_held",   2, {"up_held"}),
    ("dash",      2, {"dash_pressed"}),
]


class AgentBrain:
    """Player intent source. Builds the action space from the abilities'
    required intents, then each frame decodes the current action (or keyboard
    in human mode) into MotorState.intents, deriving *_pressed rising edges."""

    def __init__(self, abilities, human_mode: bool = False):
        required = set()
        for ab in abilities:
            required |= ab.requires()
        self._active_axes = [ax for ax in _AXES if ax[2] & required]
        self.action_space = spaces.MultiDiscrete([ax[1] for ax in self._active_axes])
        self.human_mode = human_mode
        self._action = np.zeros(len(self._active_axes), dtype=np.int64)
        # previous "held" bits for edge detection
        self._prev = {"jump": False, "fire": False, "dash": False}

    def set_action(self, action) -> None:
        arr = np.asarray(action).astype(np.int64).reshape(-1)
        if arr.shape[0] == len(self._active_axes):
            self._action = arr

    def decide(self, state) -> None:
        # reset all intents each frame
        state.intents = Intent()
        it = state.intents
        if self.human_mode:
            vals = self._read_keyboard()
        else:
            vals = {ax[0]: int(self._action[i]) for i, ax in enumerate(self._active_axes)}

        if "move_x" in vals:
            it.move_x = {0: 0, 1: -1, 2: 1}.get(vals["move_x"], 0)
        if "run_held" in vals:
            it.run_held = bool(vals["run_held"])

        for key, held_attr, edge_attr in (
            ("jump", "jump_held", "jump_pressed"),
            ("fire", "fire_held", "fire_pressed"),
        ):
            if key in vals:
                held = bool(vals[key])
                setattr(it, held_attr, held)
                setattr(it, edge_attr, held and not self._prev[key])
                self._prev[key] = held
        if "dash" in vals:
            held = bool(vals["dash"])
            it.dash_pressed = held and not self._prev["dash"]
            self._prev["dash"] = held
        if "down_held" in vals:
            it.down_held = bool(vals["down_held"])
        if "up_held" in vals:
            it.up_held = bool(vals["up_held"])

    def _read_keyboard(self) -> dict:
        import pygame
        out = {ax[0]: 0 for ax in self._active_axes}
        if not pygame.get_init():
            return out
        keys = pygame.key.get_pressed()
        left = keys[pygame.K_LEFT] or keys[pygame.K_a]
        right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
        if "move_x" in out:
            out["move_x"] = 1 if (left and not right) else 2 if (right and not left) else 0
        if "run_held" in out:
            out["run_held"] = 1 if (keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]) else 0
        if "jump" in out:
            out["jump"] = 1 if (keys[pygame.K_SPACE] or keys[pygame.K_w] or keys[pygame.K_UP]) else 0
        if "fire" in out:
            out["fire"] = 1 if keys[pygame.K_z] else 0
        if "down_held" in out:
            out["down_held"] = 1 if (keys[pygame.K_DOWN] or keys[pygame.K_s]) else 0
        if "up_held" in out:
            out["up_held"] = 1 if keys[pygame.K_UP] else 0
        if "dash" in out:
            out["dash"] = 1 if keys[pygame.K_x] else 0
        return out
