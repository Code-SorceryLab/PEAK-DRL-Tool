"""Fixed-topology MLP evolved by the GA. numpy only, no autograd."""
from __future__ import annotations

import numpy as np

N_INPUTS = 14
N_HIDDEN = 16
N_OUTPUTS = 3  # left, right, jump


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

    def act(self, x: np.ndarray) -> tuple[int, bool]:
        """Sensor vector -> (move_x in {-1,0,+1}, jump). Left/right conflict resolves by argmax."""
        left, right, jump = self.forward(x)
        move_x = 0
        if left > 0.5 or right > 0.5:
            move_x = -1 if left > right else 1
        return move_x, bool(jump > 0.5)
