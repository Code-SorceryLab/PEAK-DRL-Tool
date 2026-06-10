from __future__ import annotations
from .ActorController import ActorController
from .Brain import AgentBrain
from ..Abilities.Move import Move
from ..Abilities.Jump import Jump
from ..Abilities.WallSlide import WallSlide
from ..Abilities.WallJump import WallJump

_ABILITY_TYPES = {
    "Move": Move,
    "Jump": Jump,
    "WallSlide": WallSlide,
    "WallJump": WallJump,
}


def build_actor_from_config(cfg: dict, state, ctx, human_mode: bool = False) -> ActorController:
    abilities = []
    for spec in cfg.get("abilities", []):
        spec = dict(spec)
        cls = _ABILITY_TYPES[spec.pop("type")]
        name = spec.pop("name")
        abilities.append(cls(name=name, **spec))
    brain = AgentBrain(abilities, human_mode=human_mode)
    return ActorController(state, brain, abilities, ctx)
