"""Fixed-topology MLP evolved by the GA. numpy only, no autograd."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .sensors import sensor_dim

if TYPE_CHECKING:
    from .evolution import GAConfig

N_INPUTS = 14
N_HIDDEN = 16
N_OUTPUTS = 3  # left, right, jump — top-down games use 5: + up, down (jump = bomb)


class NeuralNet:
    """n_inputs -> tanh hidden -> sigmoid outputs, weights stored as one flat float32 vector.

    Optional carry (GA-sweep architecture axes), kept on the net so every env slot owns its own:
      feedback  — the previous decoded action (move_x, jump) is appended to the next input
      memory    — N extra sigmoid outputs are appended to the next input (Jordan memory units)
    `forward` stays pure; only `act` reads/writes the carry. Call `reset()` at episode start.
    """

    def __init__(self, n_inputs: int = N_INPUTS, n_hidden: int = N_HIDDEN, n_outputs: int = N_OUTPUTS,
                 feedback: bool = False, memory: int = 0) -> None:
        self.n_inputs = n_inputs          # sensor count (what read_sensors produces)
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs        # action outputs (left, right, jump)
        self.feedback = bool(feedback)
        self.memory = int(memory)
        n_in = n_inputs + 2 * self.feedback + self.memory
        n_out = n_outputs + self.memory
        self._shapes = [(n_in, n_hidden), (n_hidden,), (n_hidden, n_out), (n_out,)]
        self.n_params = sum(int(np.prod(s)) for s in self._shapes)
        self._w1 = np.zeros(self._shapes[0], dtype=np.float32)
        self._b1 = np.zeros(self._shapes[1], dtype=np.float32)
        self._w2 = np.zeros(self._shapes[2], dtype=np.float32)
        self._b2 = np.zeros(self._shapes[3], dtype=np.float32)
        self.reset()

    def reset(self) -> None:
        """Clear the carried state (previous action / memory units) at an episode boundary."""
        self.carry = np.zeros(2 * self.feedback + self.memory, dtype=np.float32)

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

    def act(self, x: np.ndarray) -> tuple[int, bool, int]:
        """Sensor vector -> (move_x, jump, move_y), each axis in {-1,0,+1}; opposing outputs resolve
        by argmax. move_y is always 0 for 3-output nets (side-scrollers)."""
        if self.carry.size:
            x = np.concatenate((x, self.carry))
        out = self.forward(x)
        left, right, jump = out[:3]
        move_x = 0
        if left > 0.5 or right > 0.5:
            move_x = -1 if left > right else 1
        jump = bool(jump > 0.5)
        move_y = 0
        if self.n_outputs >= 5:
            up, down = out[3:5]
            if up > 0.5 or down > 0.5:
                move_y = -1 if up > down else 1
        if self.carry.size:
            head = [float(move_x), float(jump)] if self.feedback else []
            self.carry = np.concatenate((head, out[self.n_outputs:])).astype(np.float32)
        return move_x, jump, move_y


def make_net(cfg: "GAConfig") -> NeuralNet:
    """The net a GAConfig describes: sensor mode sets the inputs, the GA-sweep knobs the rest."""
    return NeuralNet(getattr(cfg, "n_inputs", 0) or sensor_dim(cfg.sensors),
                     cfg.hidden, getattr(cfg, "n_outputs", N_OUTPUTS),
                     feedback=cfg.action_feedback, memory=cfg.memory)
