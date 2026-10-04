"""Player personas: what kind of player the evolved agents imitate.

A persona is a capability + objective profile:
  - sensor_period : how often the net gets fresh senses (reaction time; 1 = every frame)
  - sprint        : whether the run/sprint action variants are available
  - time_rate     : fitness bonus per second left on the clock when winning
                    (speedrunners optimize finishing fast, not just finishing)
  - mistake_rate  : chance per frame that the player's input slips; a slip holds one random
                    action for slip_frames (a human mis-press lasts, a 1-frame glitch does not)
  - weights       : what doing ALL of an event in a level is worth, as a share of reaching the
                    goal ("kills": 1.0 = clearing every enemy pays like finishing) — the
                    procedural-persona idea (Holmgård et al.): same game, same net, different
                    utility over what happens on the way

Balance probes run per persona, so a level can be judged from each player type's
point of view — the skill-expression metrics compare them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

EVENTS = ("kills", "coins")  # coins = power-ups in Bomberman; a game without an event counts 0


@dataclass(frozen=True)
class Persona:
    name: str
    sprint: bool
    sensor_period: int
    time_rate: float
    description: str
    mistake_rate: float = 0.0
    slip_frames: int = 12
    weights: dict[str, float] = field(default_factory=dict)


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
    # skill axis: same fitness, different hands
    "good": Persona(
        name="good", sprint=True, sensor_period=1, time_rate=0.0,
        description="good player: sprint, full reactions, clean inputs",
    ),
    "bad": Persona(
        name="bad", sprint=False, sensor_period=4, time_rate=0.0, mistake_rate=0.03, slip_frames=12,
        description="bad player: walk only, reacts every 4th frame, a 0.2 s mis-press about every 0.5 s",
    ),
    # play-style axis: same hands, different event weights
    "killer": Persona(
        name="killer", sprint=False, sensor_period=1, time_rate=0.0, weights={"kills": 1.0},
        description="killer: clearing every enemy pays as much as reaching the goal",
    ),
    "collector": Persona(
        name="collector", sprint=False, sensor_period=1, time_rate=0.0, weights={"coins": 1.0},
        description="collector: picking up every coin / power-up pays as much as reaching the goal",
    ),
}



def get_persona(name: str) -> Persona:
    try:
        return PERSONAS[name]
    except KeyError:
        raise ValueError(f"unknown persona '{name}' (available: {', '.join(PERSONAS)})") from None


class Hands:
    """A persona's input: passes the net's action through, except that with probability
    mistake_rate per frame a slip starts and holds one random action for slip_frames."""

    def __init__(self, persona: Persona, rng: np.random.Generator) -> None:
        self.rate, self.hold, self.rng = persona.mistake_rate, persona.slip_frames, rng
        self.left, self.action = 0, (0, False, 0)

    def __call__(self, action: tuple[int, bool, int]) -> tuple[int, bool, int]:
        if self.left == 0 and self.rate > 0.0 and self.rng.random() < self.rate:
            r = self.rng
            self.left = self.hold  # side-scrollers ignore move_y
            self.action = (int(r.integers(-1, 2)), bool(r.random() < 0.5), int(r.integers(-1, 2)))
        if self.left:
            self.left -= 1
            return self.action
        return action
