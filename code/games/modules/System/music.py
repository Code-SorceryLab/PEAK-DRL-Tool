# code/games/modules/System/music.py: chiptune background loops for manual play, made in code (no audio files).
"""One original tune per game, written in the style of its source, plus one for the game select screen.

The voices are the source hardware's: NES pulse, triangle and noise, a sawtooth for rock bass, and a
two-operator FM voice for the Genesis. A tune is phrases of (chords, lead line), played in the order its
form names, then looped.
  * lead   - notes (c5, f#4, bb3), '-' holds, '.' rests; each '|' bar splits evenly over its
             tokens, so 8 tokens are eighths and 16 are sixteenths
  * chords - one per bar; 'c+g7' gives each half of the bar its own chord
  * harm, bass - (voice, octave, one-bar patterns of chord degrees 1 2 3 4 5 6 7 8), the patterns cycled
             bar by bar; a token of several degrees ('135') is a chord, which the NES arpeggiates one note
             a frame and FM plays at once. harm "below" instead doubles the lead a chord tone lower
  * drums  - bar patterns, cycled: k kick, t tom, s snare, h hat, o open hat, c crash; 'kc' hits both
  * voice  - (wave, x): "pulse" with duty x, "tri", "saw", or "fm" with modulation index x
  * swing  - where each beat's off-beat eighth lands: 0.5 is straight, 0.6 a light shuffle
"""
import os

import numpy as np
import pygame

VOLUME = 0.45
FRAME_HZ = 60                    # NES arpeggios step one note per video frame
_SEMI = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}
_ACC = {"#": 1, "b": -1}

TUNES = {
    "menu": {   # PEAK's own: a calm climb in D, the lead rising by fifths toward a summit
        "bpm": 100, "form": "ab", "lead": ("pulse", 0.5), "gate": 0.9,
        "a": (["d", "bm", "gmaj7", "a", "d", "bm", "gmaj7", "a"],
              "d5 - a5 - e6 - - - | d6 - - - c#6 - b5 - | a5 - - - f#5 - g5 - | e5 - - - - - . . | "
              "d5 - a5 - e6 - f#6 - | f#6 - - - e6 - d6 - | b5 - - - a5 - d6 - | c#6 - - - - - . ."),
        "b": (["g", "a", "f#m", "bm", "em7", "g", "a", "a"],
              "b5 - - - d6 - - b5 | c#6 - - - a5 - - - | a5 - - - c#6 - - - | d6 - - - f#6 - - - | "
              "g6 - - - f#6 - e6 - | d6 - - - b5 - - - | c#6 - - - e6 - - - | a5 - - - - - . ."),
        "harm": (("pulse", 0.125), 4, "1 5 8 5 3 5 8 5"),
        "bass": (("tri", 0), 2, "1 - - - 5 - - -"),
        "drums": ["k . h . . . h .", ". . h . . . h h"],
        "gains": {"harm": 0.09, "drums": 0.16},
    },
    "platformer": {   # Mario: a sunny swung hop in C, staccato pulses in thirds over a bouncing triangle
        "bpm": 192, "swing": 0.6, "form": "abac", "lead": ("pulse", 0.5), "gate": 0.55,
        "a": (["c", "c", "f", "c", "am", "d7", "g", "g7"],
              "e5 g5 . c6 . g5 . e5 | . d#5 e5 . g5 . . . | a5 . f5 a5 . c6 . a5 | . g5 . e5 . c5 . . | "
              "a4 . c5 e5 . a5 . g5 | . f#5 . a5 . c6 . d6 | - . b5 . g5 . d5 . | f5 . . e5 . d5 . b4"),
        "b": (["am", "em", "f", "c", "dm", "g", "c", "c"],
              "a5 . . c6 . a5 . e5 | g5 . . b5 . g5 . e5 | f5 . a5 . c6 . . f6 | - . e6 . c6 . g5 . | "
              "f5 . a5 d6 . c6 . a5 | b5 . . d6 . b5 . g5 | c6 . g5 . e5 g5 . c6 | - . . . b4 c5 d5 ."),
        "c": (["ab", "bb", "c", "c", "ab", "bb", "g", "g7"],
              "eb5 . ab5 . . c6 . eb6 | - . d6 . bb5 . f5 . | e5 . g5 . . c6 . e6 | - . d6 . c6 . g5 . | "
              "c6 . eb6 . . ab5 . c6 | d6 . f6 . . bb5 . d6 | b5 . d6 . g6 . . f6 | - . d6 . b5 . g5 ."),
        "harm": (("pulse", 0.25), 0, "below"),
        "bass": (("tri", 0), 2, ["1 . . 5 . . 8 .", "1 . . 5 8 . 5 ."]),
        "drums": ["h . s h . h s .", "h . s h . s h h"],
        "gains": {"harm": 0.16, "drums": 0.22},
    },
    "megaman": {   # Mega Man: fast heroic rock in A minor with bright major lifts, sixteenth-note bass
        "bpm": 152, "form": "abac", "lead": ("pulse", 0.25),
        "a": (["am", "f", "g", "am", "am", "f", "g", "e"],
              "a5 - - e5 - a5 - b5 | c6 - - b5 - a5 - f5 | g5 - - d5 - g5 - a5 | b5 - c6 - - - - - | "
              "e6 - - d6 - c6 - b5 | c6 - a5 - f5 - a5 c6 | d6 - - - b5 - g5 - | g#5 - - - b5 - e6 -"),
        "b": (["f", "g", "c", "am", "d", "f", "e", "e"],
              "a5 - c6 - f6 - - e6 | - - d6 - b5 - g5 - | e6 - - g6 - - e6 - | c6 - - - a5 - - - | "
              "f#6 - - e6 - d6 - a5 | c6 - - a5 - c6 - f6 | e6 - - - d6 - b5 - | "
              "e5 g#5 b5 e6 - - - - d6 - c6 - b5 - g#5 -"),
        "c": (["dm", "e", "am", "g", "f", "g", "e", "e"],
              "d6 - - f6 - e6 d6 - | e6 - - - b5 - g#5 - | a5 - c6 - e6 - a6 - | g6 - - - f6 - d6 - | "
              "f6 - - e6 - c6 - a5 | b5 - - c6 - d6 - g6 | e6 - - - g#5 - b5 - | e6 - - - d6 c6 b5 g#5"),
        "harm": (("pulse", 0.125), 4, "1 5 8 5 3 5 8 5 1 5 8 5 3 5 8 5"),
        "bass": (("tri", 0), 2, ["1 1 8 1 1 1 8 1 1 1 8 1 1 8 1 8", "1 1 8 1 1 1 8 1 1 1 8 1 8 5 3 2"]),
        "drums": ["kc h s h k k s h", "k h s h k k s h", "k h s h k k s h", "k h s h k s t t"],
        "gains": {"harm": 0.10, "bass": 0.5},
    },
    "meatboy": {   # Super Meat Boy: frantic chiptune rock in E minor, saw bass on eighths, cascading arpeggios
        "bpm": 164, "form": "abac", "lead": ("pulse", 0.25), "gate": 0.85,
        "a": (["em", "em", "c", "d", "em", "em", "c", "b7"],
              "e5 . e5 . g5 . e5 . b5 - - . a5 . g5 . | f#5 . e5 . d5 . e5 - - - . . b4 . d5 . | "
              "e5 . e5 . g5 . e5 . c6 - - . b5 . g5 . | a5 - - . f#5 . d5 . a5 . b5 . c6 . d6 . | "
              "e6 - - . d6 . b5 . d6 - - . b5 . g5 . | a5 . g5 . f#5 . e5 . g5 - - - . . . . | "
              "c6 . b5 . a5 . g5 . e6 - - . d6 . c6 . | b5 - - - a5 - - - f#5 - - - d#5 - - -"),
        "b": (["c", "d", "bm", "em", "c", "d", "b7", "b7"],
              "g5 - - - e6 - - - d6 - c6 - b5 - g5 - | a5 - - - f#6 - - - e6 - d6 - a5 - f#5 - | "
              "b5 - - - d6 - - - f#6 - - - e6 - d6 - | e6 - - - - - - - b5 . b5 . d6 . e6 . | "
              "g6 - - - e6 - - - c6 - d6 - e6 - g6 - | f#6 - - - d6 - - - a5 - b5 - c6 - d6 - | "
              "d#6 - - - f#6 - - - b5 - - - d#6 - - - | f#6 - e6 - d#6 - b5 - a5 - f#5 - d#5 - b4 -"),
        "c": (["am", "em", "am", "b7", "c", "d", "em", "b7"],
              "a5 c6 e6 c6 a5 c6 e6 c6 a5 c6 e6 c6 g6 - e6 - | g5 b5 e6 b5 g5 b5 e6 b5 g5 b5 e6 b5 f#6 - e6 - | "
              "a5 c6 e6 c6 a5 c6 e6 c6 a5 c6 e6 c6 a6 - g6 - | f#6 - - - d#6 - - - b5 - - - a5 - f#5 - | "
              "e6 - - - g6 - - - e6 - c6 - g5 - e5 - | f#5 - a5 - d6 - f#6 - e6 - d6 - a5 - f#5 - | "
              "g5 - - - b5 - - - e6 - - - g6 - - - | f#6 - - - d#6 - - - b5 - - - . . . ."),
        "harm": (("pulse", 0.125), 4, "1 3 5 8 1 3 5 8 1 3 5 8 1 3 5 8"),
        "bass": (("saw", 0), 2, ["1 1 8 1 1 1 8 1", "1 1 8 1 1 8 5 4"]),
        "drums": ["kc . h . s . h . k . k . s . h .", "k . h . s . h k . k h . s . h .",
                  "k . h . s . h . k . k . s . h o", "k . h . s . h . s . s s t . t t"],
        "gains": {"harm": 0.09, "bass": 0.32, "drums": 0.42},
    },
    "bomberman": {   # Bomberman II: a cheeky bouncing romp in F, boogie bass and warbling chord stabs
        "bpm": 150, "form": "abac", "lead": ("pulse", 0.25), "gate": 0.6,
        "a": (["f", "dm", "gm7", "c7", "f", "d7", "gm", "c7"],
              "c6 . c6 . f5 . a5 . | c6 . d6 . . a5 . . | bb5 . bb5 . d5 . g5 . | bb5 . c6 . . g5 . . | "
              "a5 . c6 . f6 . e6 . | d6 . c6 . a5 . f#5 . | g5 . bb5 . d6 c6 bb5 a5 | g5 . e5 . c5 . . ."),
        "b": (["bb", "bbm", "f", "d7", "g7", "c7", "f", "c7"],
              "f5 . bb5 . d6 . . f6 | - . db6 . bb5 . f5 . | a5 . c6 . a5 . f5 . | f#5 . a5 . c6 . d6 . | "
              "b5 . . d6 . b5 . g5 | bb5 . . c6 . g5 . e5 | f5 . a5 . c6 . a5 . | g5 . . . e5 . c5 ."),
        "c": (["dm", "a7", "dm", "a7", "bb", "c7", "f", "c7"],
              "d5 . f5 . a5 . d6 . | c#6 . . e6 . c#6 . a5 | f6 . e6 . d6 . a5 . | g5 . . c#6 . . e5 . | "
              "d6 . f6 . . d6 bb5 . | c6 . e6 . . c6 bb5 . | a5 . c6 . f6 . . . | e6 . d6 . c6 . bb5 g5"),
        "harm": (("pulse", 0.5), 4, ". 35 . 35 . 35 . 35"),
        "bass": (("tri", 0), 2, ["1 . 5 . 8 . 5 .", "1 . 3 . 5 . 6 ."]),
        "drums": ["k . h . s . h .", "k . h k s . h h"],
        "gains": {"drums": 0.24},
    },
    "sonic": {   # Sonic: sunny syncopated pop in A on FM, slap bass and pushed chord stabs
        "bpm": 144, "form": "abac", "lead": ("fm", 1.2), "gate": 0.88,
        "a": (["amaj7", "f#m7", "dmaj7", "e", "amaj7", "c#m7", "dmaj7", "e7"],
              "c#6 - - e6 - - c#6 - b5 - a5 - - - e5 - | . . a5 b5 c#6 - - e6 - - c#6 - b5 - a5 - | "
              "f#5 - - a5 - - f#6 - - - e6 - c#6 - a5 - | b5 - - - - - . . g#5 - b5 - e6 - - - | "
              "c#6 - - e6 - - c#6 - b5 - a5 - - - e6 - | - - - - g#6 - - e6 - - c#6 - b5 - - - | "
              "a5 - - c#6 - - f#6 - - - e6 - - - c#6 - | d6 - - - - - b5 - - - g#5 - e5 - - -"),
        "b": (["dmaj7", "e", "c#m7", "f#m7", "bm7", "e", "amaj7", "e7"],
              "a5 - a5 - a5 - f#5 - a5 - - b5 - - c#6 - | b5 - - - g#5 - - e5 - - - . . . . . | "
              "e6 - e6 - e6 - c#6 - e6 - - f#6 - - g#6 - | f#6 - - - c#6 - - a5 - - - . . . . . | "
              "d6 - - c#6 - - b5 - - a5 - - b5 - d6 - | e6 - - - d6 - - b5 - - - g#5 - - - - | "
              "c#6 - - - e6 - - - a6 - - - g#6 - - - | . . . . . . . . e5 - f#5 - g#5 - b5 -"),
        "c": (["fmaj7", "g", "amaj7", "f#m7", "dmaj7", "e", "c#m7+f#m7", "bm7+e7"],
              "a5 - - c6 - - e6 - - - c6 - a5 - g5 - | b5 - - d6 - - g6 - - - f#6 - d6 - b5 - | "
              "c#6 - - e6 - - a6 - - - - - g#6 - e6 - | f#6 - - - - - c#6 - - - a5 - - - . . | "
              "f#5 - a5 - c#6 - - - e6 - - - c#6 - a5 - | g#5 - b5 - e6 - - - g#6 - - - e6 - - - | "
              "e6 - - - c#6 - - - f#6 - - - e6 - c#6 - | d6 - - - b5 - - - g#5 - - - e5 - - -"),
        "harm": (("fm", 0.6), 4, ". . . . . . 357 . . . . . 357 . . ."),
        "bass": (("fm", 3.0), 2, ["1 . . 1 8 . 1 . . 1 8 . 5 . 8 .", "1 . . 1 8 . 1 . . 1 8 . 7 . 8 ."]),
        "drums": ["kc . h . s . h k . . k . s . h .", "k . h . s . h k . . k . s . h h"],
        "gains": {"harm": 0.16, "bass": 0.42},
    },
}


def _midi(name: str) -> int:
    """'c4', 'f#3', 'bb5' -> MIDI note number."""
    return 12 * (int(name[-1]) + 1) + _SEMI[name[0]] + _ACC.get(name[1:-1], 0)


def _chord(sym: str) -> tuple[int, dict]:
    """'f#m7' -> (root pitch class, semitones above the root for degrees 1 to 8)."""
    root = sym[:2] if sym[1:2] in _ACC else sym[:1]
    kind = sym[len(root):]
    minor = kind.startswith("m") and not kind.startswith("maj")
    seventh = 11 if "maj7" in kind else 10 if "7" in kind else 12   # a triad's 7 is its octave
    return _SEMI[root[0]] + _ACC.get(root[1:], 0), {
        "1": 0, "2": 2, "3": 3 if minor else 4, "4": 5, "5": 7, "6": 8 if minor else 9, "7": seventh, "8": 12}


def _below(note: int, root: int, degree: dict) -> int:
    """The nearest chord tone a minor third or more under a lead note: the second pulse's harmony line."""
    tones = {(root + degree[d]) % 12 for d in "1357"}
    return next(m for m in range(note - 3, note - 15, -1) if m % 12 in tones)


def _events(bars: list, bar_s: float) -> list:
    """Bars of tokens (MIDI notes or chords, '-' hold, '.' rest) -> [start s, length s, note] per note."""
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


def _note(voice: tuple, note, n: int, sr: int, vibrato: bool) -> np.ndarray:
    """n samples of a note (a MIDI number) or a chord (a tuple of them)."""
    wave, x = voice
    if wave == "fm" and isinstance(note, tuple):                             # the Genesis plays chords
        return sum(_note(voice, m, n, sr, vibrato) for m in note) / len(note) ** 0.5
    t = np.arange(n) / sr
    notes = np.atleast_1d(note)
    midi = notes[(t * FRAME_HZ).astype(int) % len(notes)]                   # the NES arpeggiates them
    hz = 440.0 * 2 ** ((midi - 69) / 12) * (1 + 0.006 * np.sin(2 * np.pi * 5.5 * t) * (t > 0.2) * vibrato)
    phase = np.cumsum(hz) / sr % 1
    env = 0.55 + 0.45 * np.exp(-t * 10)                                      # a soft decay
    if wave == "pulse":
        y = 2.0 * ((phase < x) - x) * env                                    # zero-mean
    elif wave == "saw":
        y = (2 * phase - 1) * env
    elif wave == "fm":                                                       # bright at the attack, then mellow
        y = np.sin(2 * np.pi * phase + x * (0.3 + 0.7 * np.exp(-t * 12)) * np.sin(2 * np.pi * phase)) * env
    else:
        y = np.round((4 * np.abs(phase - 0.5) - 1) * 7.5) / 7.5              # the NES's 16-step triangle
    edge = min(n // 2, int(0.006 * sr))                                      # 6 ms fades, no clicks
    if edge:
        y[:edge] *= np.linspace(0, 1, edge)
        y[-edge:] *= np.linspace(1, 0, edge)
    return y


def _drum(hit: str, sr: int, rng) -> np.ndarray:
    if hit in "kt":                                                          # a falling sine thump
        t = np.arange(int(0.25 * sr)) / sr
        low, fall = (45, 120) if hit == "k" else (100, 140)
        return np.sin(2 * np.pi * np.cumsum(low + fall * np.exp(-t * 30)) / sr) * np.exp(-t * 16)
    t = np.arange(int({"s": 0.15, "h": 0.035, "o": 0.14, "c": 0.8}[hit] * sr)) / sr
    noise = rng.choice((-1.0, 1.0), len(t))                                  # 1-bit, as the NES noise channel
    if hit == "s":
        return 0.7 * noise * np.exp(-t * 20) + 0.5 * np.sin(2 * np.pi * 185 * t) * np.exp(-t * 35)
    return 0.3 * np.diff(noise, prepend=0) * np.exp(-t * {"h": 110, "o": 22, "c": 7}[hit])


def render(name: str, sr: int = 44100) -> np.ndarray:
    """One seamless loop of a tune as mono floats in -1..1."""
    tune = TUNES[name]
    bar_s, swing = 240 / tune["bpm"], tune.get("swing", 0.5)
    gain = {"lead": 0.30, "harm": 0.12, "bass": 0.45, "drums": 0.32, **tune.get("gains", {})}
    phrases = [tune[k] for k in tune["form"]]
    chords = [c for p in phrases for c in p[0]]
    lead = [[t if t in "-." else _midi(t) for t in bar.split()] for p in phrases for bar in p[1].split("|")]
    assert len(lead) == len(chords), f"{name}: {len(lead)} lead bars for {len(chords)} chords"
    n = int(len(chords) * bar_s * sr)
    mix = np.zeros(n + sr)                       # one second of room for tails, folded back onto the start

    def chord(bar: int, at: float) -> tuple:
        """The chord a fraction `at` of the way into a bar."""
        halves = chords[bar].split("+")
        return _chord(halves[int(at * len(halves))])

    def swung(t: float) -> float:
        """Straight time -> played time: each beat's off-beat eighth moves from half the beat to `swing`."""
        beat, x = divmod(t * 4 / bar_s, 1)
        return (beat + (2 * x * swing if x < 0.5 else swing + (2 * x - 1) * (1 - swing))) * bar_s / 4

    def add(events, voice, part, gate, vibrato=False):
        for start, length, note in events:
            i, k = int(swung(start) * sr), int((swung(start + length) - swung(start)) * gate * sr)
            mix[i:i + k] += gain[part] * _note(voice, note, k, sr, vibrato)

    melody = _events(lead, bar_s)
    add(melody, tune["lead"], "lead", tune.get("gate", 0.92), vibrato=True)
    for part, gate in (("harm", 0.8), ("bass", 0.95)):
        voice, octave, patterns = tune[part]
        if patterns == "below":
            at = [s / bar_s + 1e-9 for s, _, _ in melody]   # bar.fraction of each lead note
            events = [[s, k, _below(m, *chord(int(b), b % 1))] for (s, k, m), b in zip(melody, at)]
            gate = tune.get("gate", 0.92)                    # articulated like the lead it doubles
        else:
            patterns = [patterns] if isinstance(patterns, str) else patterns
            bars = []
            for i in range(len(chords)):
                tokens = patterns[i % len(patterns)].split()
                bar = []
                for j, tok in enumerate(tokens):
                    if tok in "-.":
                        bar.append(tok)
                        continue
                    root, degree = chord(i, j / len(tokens))
                    notes = tuple(12 * (octave + 1) + root + degree[d] for d in tok)
                    bar.append(notes if len(notes) > 1 else notes[0])
                bars.append(bar)
            events = _events(bars, bar_s)
        add(events, voice, part, gate)
    rng = np.random.default_rng(7)               # the same hiss every time
    for i in range(len(chords)):
        hits = tune["drums"][i % len(tune["drums"])].split()
        for j, tok in enumerate(hits):
            for hit in tok.strip("-."):
                d = _drum(hit, sr, rng)
                at = int(swung((i + j / len(hits)) * bar_s) * sr)
                mix[at:at + len(d)] += gain["drums"] * d
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
