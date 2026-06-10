from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Intent:
    """Named per-frame inputs an actor's abilities consume.

    A brain (player action / keyboard, or enemy behavior) writes these.
    Abilities read them; an ability may consume() one so abilities later in
    the pipeline order do not also act on it (the only inter-ability coupling).
    """
    move_x: int = 0            # -1 / 0 / +1
    run_held: bool = False     # sprint button
    jump_pressed: bool = False  # rising edge
    jump_held: bool = False     # level
    down_held: bool = False
    up_held: bool = False
    fire_pressed: bool = False
    fire_held: bool = False
    dash_pressed: bool = False

    def consume(self, name: str) -> None:
        """Clear an intent so later abilities don't see it this frame."""
        default = 0 if name == "move_x" else False
        setattr(self, name, default)


@dataclass
class MotorState:
    """The only mutable state abilities share. Abilities read+write kinematics;
    they read (never write) contact flags and intents."""
    # Kinematics — abilities read & write
    vx: float = 0.0
    vy: float = 0.0
    facing_right: bool = True
    gravity_scale: float = 1.0     # reset to 1.0 each frame; glide/slide adjust

    # Contact — written by ModularPhysicsManager after resolution (1 frame old)
    on_ground: bool = False
    contact_left: bool = False
    contact_right: bool = False
    contact_ceiling: bool = False

    # Motor-level modifier set by WallJump, decremented by ActorController,
    # read by Move to briefly preserve the wall-jump push.
    air_lockout: int = 0

    # Intents — written by the brain at step 1 of the pipeline
    intents: Intent = field(default_factory=Intent)

    # Per-ability namespaced scratch (each ability owns its own sub-dict)
    ext: dict = field(default_factory=dict)


@dataclass
class MotorContext:
    """Physics constants the host applies and abilities may read.
    Per-ability tuning (speeds, accels, jump velocity) lives in the abilities,
    not here."""
    gravity: float = 2600.0
    fast_fall_grav: float = 2600.0
    max_fall_speed: float = 1200.0
    tile_size: int = 32
