"""Player personas: what kind of player the evolved agents imitate.

A persona is a capability + objective profile, not a reward function:
  - sensor_period : how often the net gets fresh senses (reaction time; 1 = every frame)
  - sprint        : whether the run/sprint action variants are available
  - time_rate     : fitness bonus per second left on the clock when winning
                    (speedrunners optimize finishing fast, not just finishing)

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
        description="new player: walk speed only, reacts every 3rd frame",
    ),
    "experienced": Persona(
        name="experienced", sprint=False, sensor_period=1, time_rate=0.0,
        description="regular player: walk speed, full reactions (the default)",
    ),
    "speedrunner": Persona(
        name="speedrunner", sprint=True, sensor_period=1, time_rate=25.0,
        description="speedrunner: sprint unlocked, fitness pays for time left on the clock",
    ),
}


def get_persona(name: str) -> Persona:
    try:
        return PERSONAS[name]
    except KeyError:
        raise ValueError(f"unknown persona '{name}' (available: {', '.join(PERSONAS)})") from None
