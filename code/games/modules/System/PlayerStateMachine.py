"""
PlayerStateMachine.py
---------------------
Stack-based power-up system for the player.

Design
------
The stack tracks permanent upgrades only:

    [SMALL]                  — default, floor state, never popped
    [SMALL, BIG]             — after mushroom
    [SMALL, BIG, FIRE]       — after fire flower

STAR is deliberately NOT on the stack. It is a temporary modifier tracked
by a separate float timer. This avoids every edge case that arises from
STAR sitting in the middle of the stack (e.g. collecting a fire flower
while STAR is active would have to insert below STAR, creating ordering
problems). Instead STAR is completely orthogonal — it overlays any stack
state and expires cleanly without touching the stack at all.

Transitions
-----------
    collect_mushroom()  →  push BIG  (no-op if already have BIG)
    collect_flower()    →  push FIRE (push BIG first if needed; no-op if already FIRE)
    collect_star()      →  start/refresh star timer (stack untouched)
    take_hit()          →  pop top (or death if SMALL is top)

Star + damage
-------------
    star active     →  hit absorbed, stack untouched
    i-frames active →  hit absorbed, stack untouched
    BIG or FIRE     →  pop → SMALL (or previous state), start i-frame window
    SMALL           →  return False (PhysicsManager calls core._handle_death)

Integration
-----------
    # In Player.__post_init__:
    self.power_machine = PlayerStateMachine()

    # In Player.update():
    self.power_machine.update(dt)
    self.power_machine.apply_to_player(self)

    # In PhysicsManager._handle_player_powerup():
    player.power_machine.collect_mushroom()   # or collect_flower / collect_star
    
    # In PhysicsManager._handle_player_enemy():
    #   is_star_active  → contact kills the enemy (star power only)
    #   is_invincible   → player absorbs the hit (star OR i-frames)
    #   These are intentionally different: i-frames absorb damage but do NOT kill enemies.
    survived = player.power_machine.take_hit()
    if not survived:
        core._handle_death("Enemy")
"""

from __future__ import annotations

from enum import Enum, auto
from typing import List, Optional


# ── Power state enum ──────────────────────────────────────────────────────────

class PowerState(Enum):
    SMALL = auto()   # Default — one hit = death
    BIG   = auto()   # Super Mushroom — one hit → back to SMALL
    FIRE  = auto()   # Fire Flower — requires BIG beneath it in stack


# ── Constants ────────────────────────────────────────────────────────────────

STAR_DURATION : float = 10.0   # seconds star power lasts
HIT_IFRAMES   : float = 2.0    # seconds of i-frames after surviving a hit
SMALL_HEIGHT  : int   = 32     # px — player height in SMALL state
BIG_HEIGHT    : int   = 42     # px — player height in BIG / FIRE state


# ── Machine ───────────────────────────────────────────────────────────────────

class PlayerStateMachine:
    """
    Stack-based player power-up tracker.

    The stack always has SMALL at index 0 and never becomes empty.
    Current state is always stack[-1].

    Star is tracked as a float timer, not a stack entry.

    Parameters
    ----------
    star_duration : float
        How long star power lasts in seconds.
    hit_iframes : float
        How long i-frames last after surviving a hit in seconds.
    """

    def __init__(
        self,
        star_duration: float = STAR_DURATION,
        hit_iframes:   float = HIT_IFRAMES,
    ) -> None:
        # Stack — SMALL is the permanent floor
        self._stack:         List[PowerState] = [PowerState.SMALL]

        self._star_dur       = star_duration
        self._hit_iframes    = hit_iframes

        # Timers
        self._star_timer:    float = 0.0   # > 0 while star is active
        self._iframes_timer: float = 0.0   # > 0 during post-hit i-frame window

    # ── Read ─────────────────────────────────────────────────────────────────

    @property
    def state(self) -> PowerState:
        """Current power tier (top of stack)."""
        return self._stack[-1]

    @property
    def is_invincible(self) -> bool:
        """True if the player will absorb the next hit — either star or i-frames."""
        return self._star_timer > 0.0 or self._iframes_timer > 0.0

    @property
    def is_star_active(self) -> bool:
        """True only during star power — use this to decide whether to kill enemies on contact."""
        return self._star_timer > 0.0

    @property
    def is_powered(self) -> bool:
        """True if BIG or FIRE (star does not count — it's transient)."""
        return self.state != PowerState.SMALL

    @property
    def star_time_remaining(self) -> float:
        return max(0.0, self._star_timer)

    @property
    def iframes_remaining(self) -> float:
        return max(0.0, self._iframes_timer)

    def has(self, s: PowerState) -> bool:
        """True if *s* exists anywhere in the upgrade stack."""
        return s in self._stack

    # ── Events — call from collision handlers ─────────────────────────────────

    def collect_mushroom(self) -> bool:
        """
        Player picked up a Super Mushroom.
        Pushes BIG if not already present. No-op otherwise.
        Returns True if state changed.
        """
        if self.has(PowerState.BIG):
            return False
        self._stack.append(PowerState.BIG)
        return True

    def collect_flower(self) -> bool:
        """
        Player picked up a Fire Flower.
        Ensures BIG is in the stack first, then pushes FIRE.
        No-op if already FIRE.
        Returns True if state changed.
        """
        if self.has(PowerState.FIRE):
            return False
        if not self.has(PowerState.BIG):
            # Auto-upgrade to BIG first so FIRE always has BIG beneath it
            self._stack.append(PowerState.BIG)
        self._stack.append(PowerState.FIRE)
        return True

    def collect_star(self) -> bool:
        """
        Player picked up a Star.
        Starts or refreshes the invincibility timer. Stack is never touched.
        Always returns True.
        """
        self._star_timer = self._star_dur
        return True

    def take_hit(self) -> bool:
        """
        Player received damage.

        Resolution order:
          1. Star active      → absorb, no effect.
          2. I-frames active  → absorb, no effect.
          3. BIG or FIRE      → pop (downgrade), start i-frame window. Survived.
          4. SMALL            → death. Call on_death if set.

        Returns
        -------
        bool
            True if the player survived, False if they died.
        """
        if self.is_invincible:
            return True
        # 1. Star absorbs all damage
        if self._star_timer > 0.0:
            return True

        # 2. I-frame window absorbs repeated damage after a hit
        if self._iframes_timer > 0.0:
            return True

        # 3. Have a power tier to lose
        if self.state != PowerState.SMALL:
            self._pop()
            self._iframes_timer = self._hit_iframes
            return True

        # 4. SMALL — no protection left
        return False

    # ── Per-frame update ──────────────────────────────────────────────────────

    def update(self, dt: float) -> None:
        """
        Tick both timers. Call once per frame before apply_to_player().

        When the star timer reaches zero the stack is untouched — whatever
        permanent upgrades were in the stack before are still there.
        """
        if self._star_timer > 0.0:
            self._star_timer = max(0.0, self._star_timer - dt)

        if self._iframes_timer > 0.0:
            self._iframes_timer = max(0.0, self._iframes_timer - dt)

    def apply_to_player(self, player) -> None:
        """
        Write current machine state to the Player dataclass fields.

        Writes:
          player.powered_up        — True when BIG or FIRE (star excluded)
          player.star_timer        — seconds of star power remaining
          player.iframes_timer     — seconds of i-frame window remaining
          player.invincible_timer  — max of the two (drives blink effect)
          player.gObj.height       — SMALL_HEIGHT or BIG_HEIGHT

        Height change anchor: feet stay in place (gObj.y adjusted upward) so the
        player grows into the air above them rather than downward into the floor.
        Without this correction, gaining BIG form buries the player 32px into
        the ground on the next frame, which triggers spike/floor collision death.

        Call this after update() and before rendering.
        """
        s = self.state

        # powered_up — BIG or FIRE only; star is transient and does not count
        player.powered_up = (s != PowerState.SMALL)

        # invincible_timer — drives the blink effect in platformer_core._draw_player
        # star_timer and iframes_timer written separately so rendering can
        # distinguish "STAR" label (star only) from the shared blink effect.
        player.star_timer    = self._star_timer
        player.iframes_timer = self._iframes_timer
        player.invincible_timer = max(self._star_timer, self._iframes_timer)

        # Height — only shift y when GROWING (BIG/FIRE) so the feet stay
        # anchored to the ground. When SHRINKING, leave gObj.y untouched —
        # shifting it down embeds the player in the floor, which causes the
        # wall resolver to push them out sideways, snapping the x position.
        new_height = SMALL_HEIGHT if s == PowerState.SMALL else BIG_HEIGHT
        old_height = player.gObj.height

        if new_height != old_height:
            if new_height > old_height:
                # Growing: shift y up so feet stay on the ground
                player.gObj.y -= (new_height - old_height)
            # Shrinking: leave y alone — top of hitbox just drops down,
            # feet naturally stay at the same position without any y correction
            player.gObj.height = new_height

    # ── Reset ────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """
        Full reset to SMALL. Call on episode reset / new game.
        """
        self._stack         = [PowerState.SMALL]
        self._star_timer    = 0.0
        self._iframes_timer = 0.0

    # ── Internal ────────────────────────────────────────────────────────────

    def _pop(self) -> PowerState:
        """
        Remove the top of the stack and return it.
        SMALL is the floor — it is never popped.
        """
        if len(self._stack) > 1:
            return self._stack.pop()
        return PowerState.SMALL

    # ── Debug ────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        stack_str = " → ".join(s.name for s in self._stack)
        extras = []
        if self._star_timer > 0.0:
            extras.append(f"star={self._star_timer:.1f}s")
        if self._iframes_timer > 0.0:
            extras.append(f"iframes={self._iframes_timer:.1f}s")
        extra = f"  ({', '.join(extras)})" if extras else ""
        return f"<PlayerStateMachine [{stack_str}]{extra}>"