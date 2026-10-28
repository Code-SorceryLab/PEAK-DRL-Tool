# Sonic's tune, cut from code/games/modules/System/music.py's TUNES when Sonic was archived: paste the entry back to restore.
TUNES = {
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
