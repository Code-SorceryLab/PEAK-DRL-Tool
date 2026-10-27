# code/games/modules/System/music.py: NES-style background loops for manual play, made in code (no audio files).
"""One original tune per game, in the spirit of its source, plus one for the game select screen.

Voices as on the NES: a pulse lead, a pulse harmony, a triangle bass and a noise kit.
A tune is phrases of (chords, lead line), played in the order its form names, then looped.
  * lead   - notes (c5, f#4, bb3), '-' holds, '.' rests; each '|' bar splits evenly over its
             tokens, so 8 tokens are eighths and 16 are sixteenths
  * harm, bass - (voice, octave, a one-bar pattern of chord degrees 1 3 5 7 8) over each bar's chord
  * drums  - bar patterns, cycled: k kick, s snare, h hat, o open hat
"""
import os

import numpy as np
import pygame

VOLUME = 0.45
_SEMI = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}
_ACC = {"#": 1, "b": -1}

TUNES = {
    "menu": {   # a bright title screen in G
        "bpm": 126, "form": "aaba", "lead": ("pulse", 0.5),
        "a": (["g", "em", "c", "d"],
              "d5 - g5 - b5 - a5 g5 | e5 - - - g5 - f#5 e5 | c5 - e5 - g5 - c6 - | b5 - a5 - f#5 - - ."),
        "b": (["c", "d", "em", "d"],
              "e5 - g5 - e5 - c5 - | f#5 - a5 - d6 - - - | g5 - - b5 a5 - g5 e5 | f#5 - - - a5 - - ."),
        "harm": (("pulse", 0.25), 4, "1 3 5 8 5 3 5 3"),
        "bass": (("tri", 0), 2, "1 . 5 . 8 . 5 ."),
        "drums": ["k . h . s . h .", "k . h k s . h h"],
    },
    "platformer": {   # Mario: bouncy, staccato, calypso off-beats, in F
        "bpm": 150, "form": "aab", "lead": ("pulse", 0.5), "gate": 0.6,
        "a": (["f", "dm", "bb", "c7"],
              "a5 . c6 a5 . f5 . c6 | . a5 . f5 d5 . e5 f5 | d6 . bb5 . f5 . g5 a5 | bb5 . g5 . e5 c5 . ."),
        "b": (["bb", "c", "am", "dm", "gm", "c7", "f", "c7"],
              "f5 . bb5 . d6 . c6 bb5 | c6 . . g5 . e5 g5 . | a5 . e5 . c6 . a5 . | f5 . a5 d6 . . a5 . | "
              "bb5 . a5 . g5 . d5 . | e5 . g5 . bb5 . c6 . | a5 . f5 . c5 . f5 a5 | g5 . . e5 . c5 . ."),
        "harm": (("pulse", 0.5), 4, ". 3 . 5 . 3 . 5"),
        "bass": (("tri", 0), 2, "1 . . 5 8 . 5 ."),
        "drums": ["k . h . s . h .", "k . h . s k h ."],
    },
    "megaman": {   # heroic minor, octave-pumping bass, fast arpeggios, in A minor
        "bpm": 156, "form": "aab", "lead": ("pulse", 0.25),
        "a": (["am", "f", "g", "am"],
              "e5 - a5 - - b5 c6 - | - - b5 a5 - f5 - - | g5 - d5 - g5 - a5 b5 | c6 - b5 - a5 - - -"),
        "b": (["f", "g", "em", "am", "dm", "e", "f", "e"],
              "a5 - - c6 - - f6 - | e6 - d6 - b5 - g5 - | b5 - - e6 - d6 b5 - | c6 - - - a5 - - - | "
              "d6 - f6 - e6 d6 c6 - | b5 - g#5 - e5 - g#5 b5 | "
              "c6 b5 a5 b5 c6 - a5 c6 f6 - e6 - d6 - c6 - | b5 - - - g#5 - e5 -"),
        "harm": (("pulse", 0.125), 4, "1 5 8 5 1 5 8 5 1 5 8 5 1 5 8 5"),
        "bass": (("tri", 0), 2, "1 8 1 8 1 8 1 8"),
        "drums": ["k h s h k k s h", "k h s h k k s s"],
    },
    "meatboy": {   # frantic chiptune punk, chugging bass, in E minor
        "bpm": 172, "form": "aabb", "lead": ("pulse", 0.125),
        "a": (["em", "em", "c", "d"],
              "e5 . e5 g5 . b5 . a5 | g5 . e5 - b4 - d5 e5 | g5 . g5 a5 . c6 . b5 | a5 - f#5 - d5 - a5 -"),
        "b": (["c", "d", "b7", "em"],
              "e6 - - d6 - c6 - b5 | a5 - - f#5 - a5 d6 - | d#6 - - b5 - a5 - f#5 | g5 f#5 e5 d#5 e5 - b4 -"),
        "harm": (("pulse", 0.5), 3, "1 1 5 1 1 5 8 5"),
        "bass": (("tri", 0), 2, "1 1 8 1 1 1 8 1 1 1 8 1 5 1 8 1"),
        "drums": ["k h s h k h s h", "k h s k k h s h", "k h s h k h s h", "k k s h s s s s"],
    },
    "bomberman": {   # playful and cheeky, jazzy turnarounds, in C
        "bpm": 140, "form": "aab", "lead": ("pulse", 0.25), "gate": 0.55,
        "a": (["c", "a7", "dm7", "g7"],
              "e5 . g5 . c6 . b5 c6 | a5 . g5 . c#6 . a5 . | d5 . f5 . a5 . c6 . | b5 . a5 . g5 . f5 d5"),
        "b": (["f", "fm", "c", "a7", "d7", "g7", "c", "g7"],
              "a5 . . c6 . . a5 . | ab5 . . c6 . . ab5 . | g5 . e5 . c5 . e5 g5 | c#6 . . a5 . g5 . e5 | "
              "f#5 . a5 . c6 . a5 f#5 | g5 . b5 . d6 . f6 . | e6 . c6 . g5 . e5 . | d5 . f5 . g5 . b5 ."),
        "harm": (("pulse", 0.5), 4, ". 3 . 5 . 7 . 5"),
        "bass": (("tri", 0), 2, "1 . 5 . 8 . 5 ."),
        "drums": ["k . h . s . h .", "k h . k s . h h"],
    },
}


def _midi(name: str) -> int:
    """'c4', 'f#3', 'bb5' -> MIDI note number."""
    return 12 * (int(name[-1]) + 1) + _SEMI[name[0]] + _ACC.get(name[1:-1], 0)


def _chord(sym: str) -> tuple[int, dict]:
    """'f#m7' -> (root pitch class, semitones above the root for degrees 1 3 5 7 8)."""
    root = sym[:2] if sym[1:2] in _ACC else sym[:1]
    kind = sym[len(root):]
    minor = kind.startswith("m") and not kind.startswith("maj")
    seventh = 11 if "maj7" in kind else 10 if "7" in kind else 12   # a triad's 7 is its octave
    return _SEMI[root[0]] + _ACC.get(root[1:], 0), {"1": 0, "3": 3 if minor else 4, "5": 7, "7": seventh, "8": 12}


def _events(bars: list, bar_s: float) -> list:
    """Bars of tokens (MIDI notes, '-' hold, '.' rest) -> [start s, length s, note] per note."""
    out, last = [], None
    for i, tokens in enumerate(bars):
        step = bar_s / len(tokens)
        for j, tok in enumerate(tokens):
            if tok == "-":
                if last:
                    last[1] += step
            elif tok == ".":
                last = None
            else:
                last = [i * bar_s + j * step, step, tok]
                out.append(last)
    return out


def _note(voice: tuple, note: int, n: int, sr: int, vibrato: bool) -> np.ndarray:
    wave, duty = voice
    t = np.arange(n) / sr
    hz = 440.0 * 2 ** ((note - 69) / 12) * (1 + 0.006 * np.sin(2 * np.pi * 5.5 * t) * (t > 0.2) * vibrato)
    phase = np.cumsum(hz) / sr % 1
    if wave == "pulse":
        y = 2.0 * ((phase < duty) - duty) * (0.55 + 0.45 * np.exp(-t * 10))   # zero-mean, a soft decay
    else:
        y = np.round((4 * np.abs(phase - 0.5) - 1) * 7.5) / 7.5              # the NES's 16-step triangle
    edge = min(n // 2, int(0.006 * sr))                                      # 6 ms fades, no clicks
    if edge:
        y[:edge] *= np.linspace(0, 1, edge)
        y[-edge:] *= np.linspace(1, 0, edge)
    return y


def _drum(hit: str, sr: int, rng) -> np.ndarray:
    if hit == "k":                                                           # a falling sine thump
        t = np.arange(int(0.13 * sr)) / sr
        return np.sin(2 * np.pi * np.cumsum(45 + 120 * np.exp(-t * 30)) / sr) * np.exp(-t * 16)
    t = np.arange(int({"s": 0.15, "h": 0.035, "o": 0.14}[hit] * sr)) / sr
    noise = rng.choice((-1.0, 1.0), len(t))                                  # 1-bit, as the NES noise channel
    if hit == "s":
        return 0.7 * noise * np.exp(-t * 20) + 0.5 * np.sin(2 * np.pi * 185 * t) * np.exp(-t * 35)
    return 0.3 * np.diff(noise, prepend=0) * np.exp(-t * (110 if hit == "h" else 22))


def render(name: str, sr: int = 44100) -> np.ndarray:
    """One seamless loop of a tune as mono floats in -1..1."""
    tune = TUNES[name]
    bar_s = 240 / tune["bpm"]
    phrases = [tune[k] for k in tune["form"]]
    chords = [c for p in phrases for c in p[0]]
    lead = [[t if t in "-." else _midi(t) for t in bar.split()] for p in phrases for bar in p[1].split("|")]
    assert len(lead) == len(chords), f"{name}: {len(lead)} lead bars for {len(chords)} chords"
    n = int(len(chords) * bar_s * sr)
    mix = np.zeros(n + sr)                       # one second of room for tails, folded back onto the start

    def add(events, voice, gain, gate, vibrato=False):
        for start, length, note in events:
            k, i = int(length * gate * sr), int(start * sr)
            mix[i:i + k] += gain * _note(voice, note, k, sr, vibrato)

    add(_events(lead, bar_s), tune["lead"], 0.30, tune.get("gate", 0.92), vibrato=True)
    for part, gain, gate in (("harm", 0.12, 0.8), ("bass", 0.45, 0.95)):
        voice, octave, pattern = tune[part]
        bars = []
        for c in chords:
            root, degree = _chord(c)
            bars.append([t if t in "-." else 12 * (octave + 1) + root + degree[t] for t in pattern.split()])
        add(_events(bars, bar_s), voice, gain, gate)
    rng = np.random.default_rng(7)               # the same hiss every time
    for i in range(len(chords)):
        hits = tune["drums"][i % len(tune["drums"])].split()
        for j, tok in enumerate(hits):
            for hit in tok.strip("-."):
                d = _drum(hit, sr, rng)
                at = int((i + j / len(hits)) * bar_s * sr)
                mix[at:at + len(d)] += 0.32 * d
    mix[:sr] += mix[n:]
    mix = mix[:n]
    return mix * min(0.22 / np.sqrt(np.mean(mix ** 2)), 0.95 / np.abs(mix).max())


_sounds, _playing = {}, None


def play(name: str) -> None:
    """Loop the tune for a game key (or "menu"). Silent without an audio device or in a headless run."""
    global _playing
    stop()
    if name not in TUNES or os.environ.get("SDL_VIDEODRIVER") == "dummy":
        return
    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        if name not in _sounds:
            sr, _, channels = pygame.mixer.get_init()
            wave = (render(name, sr) * 32767).astype(np.int16)
            _sounds[name] = pygame.sndarray.make_sound(wave if channels == 1 else np.column_stack([wave] * channels))
            _sounds[name].set_volume(VOLUME)
        _playing = _sounds[name]
        _playing.play(loops=-1)
    except pygame.error:
        _playing = None


def stop() -> None:
    if _playing:
        _playing.stop()
