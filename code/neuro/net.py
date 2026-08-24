"""Fixed-topology MLP evolved by the GA. numpy only, no autograd."""
from __future__ import annotations

import math

import numpy as np

N_INPUTS = 14
N_HIDDEN = 16
N_OUTPUTS = 3  # core locomotion outputs: left, right, jump.
# Games may declare extra buttons (attack, dash, bomb, ...) via their adapter's
# BUTTONS tuple; those become outputs 3.. and reach the adapter through act()'s
# `extras`. Only the input layer is fan-in scaled (see param_scale), so a wider
# output head leaves the sensor-ablation baselines numerically untouched.
REF_FANIN = N_INPUTS  # input-layer sigmas are scaled relative to this, so the
                      # 14-input "rays" baseline is numerically unchanged


class NeuralNet:
    """n_inputs -> tanh hidden -> sigmoid outputs, weights stored as one flat float32 vector."""

    def __init__(self, n_inputs: int = N_INPUTS, n_hidden: int = N_HIDDEN, n_outputs: int = N_OUTPUTS) -> None:
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self._shapes = [(n_inputs, n_hidden), (n_hidden,), (n_hidden, n_outputs), (n_outputs,)]
        self.n_params = sum(int(np.prod(s)) for s in self._shapes)
        self._w1 = np.zeros(self._shapes[0], dtype=np.float32)
        self._b1 = np.zeros(self._shapes[1], dtype=np.float32)
        self._w2 = np.zeros(self._shapes[2], dtype=np.float32)
        self._b2 = np.zeros(self._shapes[3], dtype=np.float32)

    def param_scale(self) -> np.ndarray:
        """Per-parameter multiplier the GA applies to init_sigma and mutation_sigma.

        Only the input layer is scaled, by sqrt(REF_FANIN / n_inputs). That keeps two
        quantities independent of the sensor count, both of which otherwise blow up on
        the 368-input "grid" mode and made the sensor ablation an unfair fight:
          * hidden pre-activation spread — unscaled, 368 inputs saturate tanh at init
            (|tanh| ~0.93 vs ~0.65 for rays), so the whole population is born near-constant;
          * the GA's L2 mutation step — mutation_rate is per-weight, so the step grows as
            sqrt(fan_in) and grid children were jumping ~4.5x further than ray children.
        The hidden->output layer has fan-in n_hidden in both modes, so it is left alone.
        """
        s = np.ones(self.n_params, dtype=np.float32)
        s[:self.n_inputs * self.n_hidden] = math.sqrt(REF_FANIN / self.n_inputs)
        return s

    def set_weights(self, flat: np.ndarray) -> None:
        if flat.size != self.n_params:
            raise ValueError(f"expected {self.n_params} params, got {flat.size}")
        i = 0
        for attr, shape in zip(("_w1", "_b1", "_w2", "_b2"), self._shapes):
            n = int(np.prod(shape))
            setattr(self, attr, flat[i:i + n].reshape(shape).astype(np.float32))
            i += n

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.tanh(x @ self._w1 + self._b1)
        return 1.0 / (1.0 + np.exp(-(h @ self._w2 + self._b2)))

    def act(self, x: np.ndarray) -> tuple[int, bool, tuple[bool, ...]]:
        """Sensor vector -> (move_x in {-1,0,+1}, jump, extras).

        Outputs 0-2 are the locomotion core every game shares; left/right conflict
        resolves by argmax. Outputs 3.. are the game-specific buttons named by the
        adapter's BUTTONS tuple, thresholded the same way and handed back in order.
        Three-button games get an empty `extras`, so their behaviour is unchanged.
        """
        out = self.forward(x)
        left, right, jump = out[0], out[1], out[2]
        move_x = 0
        if left > 0.5 or right > 0.5:
            move_x = -1 if left > right else 1
        return move_x, bool(jump > 0.5), tuple(bool(v > 0.5) for v in out[3:])
