"""SPEEDY's sound effects, made in code like the music (no audio files): short FM and noise blips in the
manner of the Mega Drive's sound chip. Silent without an audio device."""
from __future__ import annotations

import numpy as np
import pygame

SR = 22050
VOLUME = 0.35


def _fm(f0: float, f1: float, t: float, index: float = 2.0, decay: float = 6.0) -> np.ndarray:
    """A two-operator FM note gliding from f0 to f1 Hz."""
    n = int(SR * t)
    x = np.arange(n) / SR
    freq = f0 + (f1 - f0) * x / t
    phase = 2 * np.pi * np.cumsum(freq) / SR
    return np.sin(phase + index * np.sin(2 * phase)) * np.exp(-decay * x)


def _noise(t: float, decay: float = 12.0) -> np.ndarray:
    n = int(SR * t)
    return np.random.default_rng(7).uniform(-1, 1, n) * np.exp(-decay * np.arange(n) / SR)


def _seq(*notes) -> np.ndarray:
    return np.concatenate(notes)


_MAKE = {
    "ring": lambda: _seq(_fm(1319, 1319, 0.05, 1.0, 8), _fm(1760, 1760, 0.22, 1.0, 9)),
    "jump": lambda: _fm(440, 1100, 0.2, 1.5, 7),
    "roll": lambda: _fm(900, 300, 0.14, 2.5, 6),
    "spin": lambda: _fm(500, 1500, 0.22, 2.0, 4),
    "skid": lambda: _noise(0.2, 9) * 0.6,
    "spring": lambda: _fm(330, 1500, 0.22, 3.0, 5),
    "hurt": lambda: _fm(700, 120, 0.4, 3.0, 4),
    "spikes": lambda: _noise(0.3, 7),
    "pop": lambda: _noise(0.28, 8),
    "lamp": lambda: _seq(_fm(1047, 1047, 0.08, 1.0, 6), _fm(1568, 1568, 0.25, 1.0, 6)),
    "sign": lambda: _seq(*[_fm(880 + 110 * i, 880 + 110 * i, 0.06, 1.0, 5) for i in range(8)]),
    "shield": lambda: _fm(300, 1200, 0.3, 2.0, 3),
    "invincible": lambda: _fm(600, 2000, 0.4, 2.0, 2),
    "life": lambda: _seq(*[_fm(f, f, 0.1, 1.0, 3) for f in (880, 1109, 1319, 1760)]),
    "boss_hit": lambda: _seq(_noise(0.08, 6), _fm(200, 90, 0.2, 4.0, 4)),
}
_sounds: dict = {}


def play(name: str) -> None:
    if name not in _MAKE:
        return
    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        if name not in _sounds:
            sr, _, channels = pygame.mixer.get_init()
            wave = _MAKE[name]()
            wave = np.interp(np.arange(int(len(wave) * sr / SR)) * SR / sr, np.arange(len(wave)), wave)
            data = (wave * 32767 * VOLUME).astype(np.int16)
            _sounds[name] = pygame.sndarray.make_sound(data if channels == 1 else np.column_stack([data] * channels))
        _sounds[name].play()
    except pygame.error:
        pass
