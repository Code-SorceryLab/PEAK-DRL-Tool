"""Player personas: what kind of player the evolved agents imitate.

A persona is a capability profile plus one objective term:
  - sensor_period : reaction lag. The net acts on senses from (sensor_period - 1)
                    frames ago; 1 = no lag. Senses are still read every frame, so a
                    slower player reacts to older information but does NOT hold its
                    last action. (It used to skip the read and reuse the stale vector,
                    which -- because the net is deterministic -- repeated the action
                    for N frames. Action repeat *helps* evolutionary search, so the
                    "novice" handicap was measurably an advantage on some levels.)
  - sprint        : whether the run/sprint action variants are available
  - time_rate     : share of win_bonus paid for the frame budget left unused on a win,
                    as a fraction in [0, 1]. Speedrunners optimize finishing fast, not
                    just finishing. Scored on frames, so it works in every game.

time_rate is a genuine fitness term, so a persona is not purely a capability profile.
Note that "speedrunner" differs from "experienced" on two knobs at once (sprint and
time_rate); a difference between them cannot be attributed to either one alone.

Balance probes run per persona, so a level can be judged from each player type's
point of view — the skill-expression metrics compare them.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Persona:
    name: str
    sprint: bool
    sensor_period: int
    time_rate: float
    description: str


PERSONAS: dict[str, Persona] = {
    "novice": Persona(
        name="novice", sprint=False, sensor_period=3, time_rate=0.0,
        description="new player: walk speed only, acts on senses 2 frames old",
    ),
    "experienced": Persona(
        name="experienced", sprint=False, sensor_period=1, time_rate=0.0,
        description="regular player: walk speed, no reaction lag (the default)",
    ),
    "speedrunner": Persona(
        name="speedrunner", sprint=True, sensor_period=1, time_rate=0.5,
        description="speedrunner: sprint unlocked, wins pay up to +50% win_bonus for finishing early",
    ),
}


def get_persona(name: str) -> Persona:
    try:
        return PERSONAS[name]
    except KeyError:
        raise ValueError(f"unknown persona '{name}' (available: {', '.join(PERSONAS)})") from None
