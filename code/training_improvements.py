"""
PEAK Platformer — Training Improvements
========================================

FILE PLACEMENT:  code/training_improvements.py
                 (same level as code/wrappers/, code/rewards/, etc.)

Import:  from code.training_improvements import MultiDiscreteActionSpace

The batch curriculum system is built directly into platformer_core.py
(see __init__ kwargs: batch_window, advance_threshold, fallback_threshold).
"""

from __future__ import annotations
import numpy as np
from gymnasium import spaces


class MultiDiscreteActionSpace:
    """
    Multi-axis action decomposition for the platformer.

    Axes:
        0: movement  — 0=idle, 1=left, 2=right           (3 options)
        1: jump      — 0=no, 1=yes                        (2 options)
        2: run       — 0=no, 1=yes                        (2 options)
        3: fire      — 0=no, 1=yes                        (2 options)

    Total: 3 x 2 x 2 x 2 = 24 unique combos from 9 logits
    vs Discrete(20) needing 20 logits.
    """

    SPACE = spaces.MultiDiscrete([3, 2, 2, 2])
    _COMBO_TO_DISCRETE = {}

    @classmethod
    def _build_table(cls):
        if cls._COMBO_TO_DISCRETE:
            return
        for fire in (0, 1):
            offset = 10 if fire else 0
            for move in range(3):
                for jump in (0, 1):
                    for run in (0, 1):
                        if   move == 0 and not jump and not run: act = 0
                        elif move == 1 and not jump and not run: act = 1
                        elif move == 2 and not jump and not run: act = 2
                        elif move == 0 and jump and not run:     act = 3
                        elif move == 2 and jump and not run:     act = 4
                        elif move == 2 and not jump and run:     act = 5
                        elif move == 1 and jump and not run:     act = 6
                        elif move == 2 and jump and run:         act = 7
                        elif move == 1 and not jump and run:     act = 8
                        elif move == 1 and jump and run:         act = 9
                        elif move == 0 and jump and run:         act = 3
                        elif move == 0 and not jump and run:     act = 0
                        else:                                     act = 0
                        cls._COMBO_TO_DISCRETE[(move, jump, run, fire)] = act + offset

    @classmethod
    def decode(cls, multi_action: np.ndarray) -> int:
        """Convert MultiDiscrete action array -> legacy Discrete int."""
        cls._build_table()
        move, jump, run, fire = int(multi_action[0]), int(multi_action[1]), \
                                 int(multi_action[2]), int(multi_action[3])
        return cls._COMBO_TO_DISCRETE.get((move, jump, run, fire), 0)
