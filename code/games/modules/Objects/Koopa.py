"""
Koopa.py
--------
Koopa Troopa enemy — a stack-based, multi-state enemy using StateMachine.py.

State Stack (bottom → top)
---------------------------
    [SHELL]                — inside shell, waiting
    [SHELL, ALIVE]         — default: walking patrol
    [SHELL, ALIVE, FLYING] — winged variant: oscillating patrol
    [SHELL, MOVING]        — kicked shell sliding across the floor

Push / Pop Rules
----------------
    Startup:
        always start as [SHELL, ALIVE]
        if flying=True, start as [SHELL, ALIVE, FLYING]

    ALIVE  + stomped by player  → pop ALIVE    → [SHELL]      (shell state, timer starts)
    FLYING + stomped by player  → pop FLYING   → [SHELL, ALIVE]
    MOVING + stomped by player  → pop MOVING   → [SHELL]      (shell state, timer starts)
    SHELL  (timer expires 5s)   → push ALIVE   → [SHELL, ALIVE]
    SHELL  + touched by player  → push MOVING  → [SHELL, MOVING]  (no player damage)

    ALIVE  + touched by player  → player takes damage
    FLYING + touched by player  → player takes damage
    MOVING + touched by player  → player takes damage
    MOVING + touches enemy      → kills enemy

PhysicsManager Integration
--------------------------
_handle_player_enemy() must duck-type on Koopa's callbacks.
Add this block at the TOP of _handle_player_enemy(), before the stomp check:

    if hasattr(enemy, 'on_stomp'):
        player_bottom = player.gObj.y + player.gObj.height
        enemy_center  = enemy.gObj.y  + enemy.gObj.height / 2
        if player_was_falling and player_bottom < enemy_center + 10:
            enemy.on_stomp(player, core)
        elif not player.power_machine.is_invincible:
            enemy.on_touch(player, core)
        return

Also, in _handle_enemy_enemy(), add at the top:
    if hasattr(e1, 'on_enemy_touch'):
        e1.on_enemy_touch(e2)
    if hasattr(e2, 'on_enemy_touch'):
        e2.on_enemy_touch(e1)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional

import pygame

from .GameObject import GameObject
from ..System.StateMachine import StateMachine
from ..System.EntityType import EntityType


# ─────────────────────────────────────────────────────────────────────────────
# State enum
# ─────────────────────────────────────────────────────────────────────────────

class KoopaState(Enum):
    SHELL  = auto()   # hiding in shell — stationary, auto-revives after timeout
    ALIVE  = auto()   # normal walking patrol
    MOVING = auto()   # shell sliding fast after being kicked
    FLYING = auto()   # winged koopa — sine-wave aerial patrol


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SHELL_TIMEOUT : float = 5.0    # seconds before SHELL auto-pushes ALIVE
WALK_SPEED    : float = 60.0   # px/s walking (ALIVE / FLYING)
SHELL_SPEED   : float = 240.0  # px/s sliding (MOVING)
FLY_AMP       : float = 38.0   # px — vertical oscillation amplitude (FLYING)
FLY_FREQ      : float = 1.8    # Hz — oscillation frequency (FLYING)
STOMP_BOUNCE  : float = -220.0 # player vy after stomping a Koopa

KOOPA_W       : int = 22
KOOPA_H_FULL  : int = 30       # height when ALIVE / FLYING
KOOPA_H_SHELL : int = 22       # height when SHELL / MOVING

# Colours
C_SHELL       = ( 34, 139,  34)
C_SHELL_HI    = (144, 238, 144)
C_BODY        = (255, 200,  50)
C_HEAD        = (255, 220, 100)
C_EYE         = ( 20,  20,  20)
C_WING        = (220, 220, 255)
C_WING_EDGE   = (180, 180, 220)
C_MOTION      = (150, 200, 150)


# ─────────────────────────────────────────────────────────────────────────────
# KoopaStateMachine — stack layer on top of StateMachine
# ─────────────────────────────────────────────────────────────────────────────

class KoopaStateMachine:
    """
    Stack-based state machine for the Koopa enemy.

    Uses StateMachine (from StateMachine.py) as a component for:
      - Transition-table validation (can_transition / transition)
      - Enter / exit callbacks

    Adds a push/pop stack layer so multiple states can be layered
    (e.g. SHELL beneath ALIVE, ALIVE beneath FLYING).

    The stack always has SHELL at index 0 — it is the permanent floor and
    is never popped.
    """

    def __init__(self, flying: bool = False) -> None:
        # ── Flat FSM — validation + callbacks ────────────────────────────────
        self._fsm = StateMachine(KoopaState.SHELL)
        self._register_transitions()

        # ── Stack — SHELL is the immutable floor ──────────────────────────────
        self._stack: List[KoopaState] = [KoopaState.SHELL]

        # Shell re-emerge countdown (only active when top == SHELL)
        self._shell_timer: float = 0.0

        # Build initial stack
        self._push_raw(KoopaState.ALIVE)
        if flying:
            self._push_raw(KoopaState.FLYING)

    # ── Setup ────────────────────────────────────────────────────────────────

    def _register_transitions(self) -> None:
        """
        Register every legal directed edge in the flat FSM.

        Push edges (going deeper):
            SHELL  → ALIVE      default startup / shell timeout revival
            SHELL  → MOVING     player kicks the shell
            ALIVE  → FLYING     winged variant
        Pop edges (returning to the state below):
            ALIVE  → SHELL      player stomps walking koopa
            FLYING → ALIVE      player stomps flying koopa
            MOVING → SHELL      player stomps the moving shell
        """
        fsm = self._fsm
        # push edges
        fsm.add_transition(KoopaState.SHELL,  KoopaState.ALIVE)
        fsm.add_transition(KoopaState.SHELL,  KoopaState.MOVING)
        fsm.add_transition(KoopaState.ALIVE,  KoopaState.FLYING)
        # pop edges
        fsm.add_transition(KoopaState.ALIVE,  KoopaState.SHELL)
        fsm.add_transition(KoopaState.FLYING, KoopaState.ALIVE)
        fsm.add_transition(KoopaState.MOVING, KoopaState.SHELL)

    # ── Read ─────────────────────────────────────────────────────────────────

    @property
    def state(self) -> KoopaState:
        """Current active state — always the top of the stack."""
        return self._stack[-1]

    @property
    def shell_timer(self) -> float:
        """Seconds remaining before shell auto-revives (0 when not in SHELL)."""
        return max(0.0, self._shell_timer)

    # ── Actions ──────────────────────────────────────────────────────────────

    def push(self, to: KoopaState) -> bool:
        """
        Push *to* onto the stack if the flat FSM allows it.
        Returns True if the push succeeded.
        """
        if not self._fsm.can_transition(to):
            return False
        self._push_raw(to)
        return True

    def pop(self) -> Optional[KoopaState]:
        """
        Pop the current (top) state and restore the one beneath it.
        SHELL (index 0) is the permanent floor — it is never removed.
        Returns the popped state, or None if already at the floor.
        """
        if len(self._stack) <= 1:
            return None
        popped = self._stack.pop()
        # Re-sync the flat FSM to the restored top state (force = no table check)
        self._fsm.transition(self._stack[-1], force=True)
        return popped

    def start_shell_timer(self) -> None:
        """Restart the shell countdown. Call whenever entering SHELL state."""
        self._shell_timer = SHELL_TIMEOUT

    def update(self, dt: float) -> bool:
        """
        Tick shell timer.  Auto-pushes ALIVE when the countdown expires.
        Returns True if a revival just happened this frame.
        """
        if self.state == KoopaState.SHELL and self._shell_timer > 0.0:
            self._shell_timer -= dt
            if self._shell_timer <= 0.0:
                self._shell_timer = 0.0
                return self.push(KoopaState.ALIVE)   # revival push
        return False

    # ── Internal ─────────────────────────────────────────────────────────────

    def _push_raw(self, to: KoopaState) -> None:
        """Low-level push: sync flat FSM (force) then append to stack."""
        self._fsm.transition(to, force=True)
        self._stack.append(to)
        if to == KoopaState.SHELL:
            self._shell_timer = SHELL_TIMEOUT

    # ── Debug ────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        stack_str = " → ".join(s.name for s in self._stack)
        extra = (f"  [shell={self._shell_timer:.1f}s]"
                 if self.state == KoopaState.SHELL else "")
        return f"<KoopaStateMachine [{stack_str}]{extra}>"


# ─────────────────────────────────────────────────────────────────────────────
# Koopa entity
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Koopa:
    """
    Koopa Troopa — a stack-based multi-state enemy.

    Parameters
    ----------
    gObj   : GameObject   position, size, active flag
    flying : bool         spawn as a winged (flying) Koopa
    vx     : float        initial horizontal speed (sign = direction)

    The Koopa exposes three collision callbacks for PhysicsManager:
        on_stomp(player, core)        — player jumps on top
        on_touch(player, core)        — player touches from the side
        on_enemy_touch(other_enemy)   — MOVING shell hits another enemy
    """

    gObj   : GameObject
    flying : bool  = False
    vx     : float = -WALK_SPEED
    vy     : float = 0.0

    # ── Post-init ─────────────────────────────────────────────────────────────

    def __post_init__(self) -> None:
        self.gObj.type_id = EntityType.ENEMY
        self.sm = KoopaStateMachine(flying=self.flying)

        # FLYING oscillation helpers
        self._fly_phase  : float         = 0.0
        self._fly_base_y : Optional[float] = None

        # Size to match initial state
        self._apply_size()

    # ── Properties (mirrors Enemy interface for drop-in compatibility) ────────

    @property
    def x(self)      -> float:      return self.gObj.x
    @property
    def y(self)      -> float:      return self.gObj.y
    @property
    def width(self)  -> int:        return self.gObj.width
    @property
    def height(self) -> int:        return self.gObj.height
    @property
    def active(self) -> bool:       return self.gObj.active
    @property
    def state(self)  -> KoopaState: return self.sm.state

    # ── Update ────────────────────────────────────────────────────────────────

    def update(self, dt: float, context) -> None:
        if not self.gObj.active:
            return

        prev_state = self.sm.state

        # Tick shell timer — may auto-push ALIVE (revival)
        revived = self.sm.update(dt)
        if revived:
            self._on_revive()

        # Resize if the state just changed (e.g. revival)
        if self.sm.state != prev_state:
            self._apply_size()

        s        = self.sm.state
        grav     = context.GRAVITY
        max_fall = context.MAX_FALL_SPEED

        if s == KoopaState.SHELL:
            # Stationary — still affected by gravity so it settles on the ground
            self.vx  = 0.0
            self.vy += grav * dt
            self.vy  = min(self.vy, max_fall)
            self.gObj.y += self.vy * dt

        elif s == KoopaState.ALIVE:
            self.vy += grav * dt
            self.vy  = min(self.vy, max_fall)
            self.gObj.x += self.vx * dt
            self.gObj.y += self.vy * dt

        elif s == KoopaState.MOVING:
            # Fast sliding shell — direction maintained by vx sign
            self.vy += grav * dt
            self.vy  = min(self.vy, max_fall)
            self.gObj.x += self.vx * dt
            self.gObj.y += self.vy * dt

        elif s == KoopaState.FLYING:
            # Horizontal patrol + gentle sinusoidal vertical bob
            self._fly_phase += FLY_FREQ * 2.0 * math.pi * dt
            if self._fly_base_y is None:
                self._fly_base_y = self.gObj.y
            target_y   = self._fly_base_y + FLY_AMP * math.sin(self._fly_phase)
            self.vy    = (target_y - self.gObj.y) * 8.0   # proportional pull
            self.gObj.x += self.vx * dt
            self.gObj.y += self.vy * dt

    # ── Collision callbacks (called by PhysicsManager) ────────────────────────

    def on_stomp(self, player, core) -> None:
        """
        Player jumped on top of this Koopa.

        SHELL   → push MOVING — kick the shell (same as touch, but with a bounce)
        ALIVE   → pop → [SHELL]       shell state begins, timer starts
        FLYING  → pop → [SHELL, ALIVE] de-wings the koopa
        MOVING  → pop → [SHELL]       stops the sliding shell, timer starts
        """
        s = self.sm.state

        if s == KoopaState.SHELL:
            # Stomping a shell kicks it, same as touching from the side.
            # Player still gets the bounce so the interaction feels responsive.
            kick_dir = 1.0 if player.gObj.x < self.gObj.x else -1.0
            if self.sm.push(KoopaState.MOVING):
                self.vx = kick_dir * SHELL_SPEED
                self._apply_size()
            player.vy = STOMP_BOUNCE

        if s == KoopaState.ALIVE:
            self.sm.pop()            # ALIVE gone → SHELL is now top
            self.sm.start_shell_timer()
            self.vx = 0.0
            self._apply_size()
            player.vy = STOMP_BOUNCE
            core.score += 100
            if hasattr(core, 'kills_step'):
                core.kills_step += 1

        elif s == KoopaState.FLYING:
            self.sm.pop()            # FLYING gone → ALIVE is now top
            self._fly_base_y = None  # reset oscillation baseline
            self._apply_size()
            player.vy = STOMP_BOUNCE
            core.score += 100
            if hasattr(core, 'kills_step'):
                core.kills_step += 1

        elif s == KoopaState.MOVING:
            self.sm.pop()            # MOVING gone → SHELL is now top
            self.sm.start_shell_timer()
            self.vx = 0.0
            self._apply_size()
            player.vy = STOMP_BOUNCE
            core.score += 100

    def on_touch(self, player, core) -> None:
        """
        Player touched this Koopa from the side (not a stomp).

        SHELL   → push MOVING — kick the shell in the player's direction
        ALIVE   → player takes damage
        FLYING  → player takes damage
        MOVING  → player takes damage
        """
        s = self.sm.state

        if s == KoopaState.SHELL:
            # Kick the shell — direction is away from the player
            kick_dir = 1.0 if player.gObj.x < self.gObj.x else -1.0
            if self.sm.push(KoopaState.MOVING):
                self.vx = kick_dir * SHELL_SPEED
                self._apply_size()
            # Touching a still shell does NOT hurt the player

        else:
            # ALIVE, FLYING, or MOVING — deal damage
            if player.power_machine.is_star_active:
                # Star power — kill the koopa
                self.gObj.active = False
                core.score += 100
                if hasattr(core, 'kills_step'):
                    core.kills_step += 1
                return

            survived = player.power_machine.take_hit()
            if not survived:
                core._handle_death("Koopa")

    def on_enemy_touch(self, other_enemy) -> None:
        """
        Called when this (MOVING) shell collides with another enemy.
        Kills the other enemy and awards score.
        """
        if self.sm.state == KoopaState.MOVING and other_enemy.gObj.active:
            other_enemy.gObj.active = False

    # ── Render ────────────────────────────────────────────────────────────────

    def render(self, surface: pygame.Surface, sx: float, sy: float,
               debug: bool = False) -> None:
        if not self.gObj.active:
            return

        ix, iy = int(sx), int(sy)
        w, h   = self.gObj.width, self.gObj.height
        s      = self.sm.state

        if s in (KoopaState.SHELL, KoopaState.MOVING):
            self._draw_shell(surface, ix, iy, w, h, moving=(s == KoopaState.MOVING))
        elif s == KoopaState.ALIVE:
            self._draw_koopa(surface, ix, iy, w, h, winged=False)
        elif s == KoopaState.FLYING:
            self._draw_koopa(surface, ix, iy, w, h, winged=True)

        if debug:
            lbl = f"{s.name}  {self.sm}"
            pygame.draw.rect(surface, (255, 0, 0), (ix, iy, w, h), 1)

    # ── Private draw helpers ──────────────────────────────────────────────────

    def _draw_shell(self, surface: pygame.Surface,
                    x: int, y: int, w: int, h: int, moving: bool) -> None:
        """Draw shell — ellipse with cross pattern and optional motion blur."""
        # Shell body
        pygame.draw.ellipse(surface, C_SHELL, (x, y, w, h))
        # Inner highlight
        pygame.draw.ellipse(surface, C_SHELL_HI,
                            (x + 4, y + 4, w - 8, h - 8))
        # Cross dividers
        pygame.draw.line(surface, C_SHELL,
                         (x + w // 2, y + 3), (x + w // 2, y + h - 3), 2)
        pygame.draw.line(surface, C_SHELL,
                         (x + 3, y + h // 2), (x + w - 3, y + h // 2), 2)

        if moving:
            # Motion streak lines trailing behind the shell
            trail_dir = -1 if self.vx > 0 else 1
            for i in range(1, 4):
                lx = x + trail_dir * i * 5
                alpha = max(1, 3 - i)
                pygame.draw.line(surface, C_MOTION,
                                 (lx, y + h // 4), (lx, y + 3 * h // 4), alpha)

    def _draw_koopa(self, surface: pygame.Surface,
                    x: int, y: int, w: int, h: int, winged: bool) -> None:
        """Draw the full koopa body (ALIVE / FLYING)."""
        head_h = h // 3
        body_h = h - head_h

        # ── Wings (drawn behind body) ─────────────────────────────────────────
        if winged:
            # Flap animation: offset by phase
            flap = int(4 * math.sin(self._fly_phase * 2))
            left_wing  = [(x - 12, y + head_h - flap),
                          (x - 3,  y + head_h + 6),
                          (x - 3,  y + head_h + body_h // 2)]
            right_wing = [(x + w + 12, y + head_h - flap),
                          (x + w + 3,  y + head_h + 6),
                          (x + w + 3,  y + head_h + body_h // 2)]
            pygame.draw.polygon(surface, C_WING, left_wing)
            pygame.draw.polygon(surface, C_WING, right_wing)
            pygame.draw.polygon(surface, C_WING_EDGE, left_wing, 1)
            pygame.draw.polygon(surface, C_WING_EDGE, right_wing, 1)

        # ── Body / shell on back ──────────────────────────────────────────────
        body_rect = (x, y + head_h, w, body_h)
        pygame.draw.rect(surface, C_SHELL, body_rect)
        # Shell highlight panel
        pygame.draw.rect(surface, C_SHELL_HI,
                         (x + 3, y + head_h + 3, w - 6, body_h - 6))

        # ── Head ─────────────────────────────────────────────────────────────
        pygame.draw.ellipse(surface, C_HEAD,
                            (x + 2, y, w - 4, head_h + 5))

        # ── Eye — faces direction of travel ───────────────────────────────────
        eye_x = (x + w - 8) if self.vx < 0 else (x + 3)
        pygame.draw.rect(surface, C_EYE,          (eye_x,     y + 3, 5, 5))
        pygame.draw.rect(surface, (255, 255, 255), (eye_x + 1, y + 3, 2, 2))

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _apply_size(self) -> None:
        """
        Resize gObj to match the current state, anchoring feet in place
        (same technique as PlayerStateMachine.apply_to_player).
        """
        s      = self.sm.state
        new_h  = KOOPA_H_SHELL if s in (KoopaState.SHELL, KoopaState.MOVING) \
                 else KOOPA_H_FULL
        new_w  = KOOPA_W
        old_h  = self.gObj.height

        self.gObj.width   = new_w
        self.gObj.y      -= (new_h - old_h)   # grow upward / shrink downward
        self.gObj.height  = new_h

    def _on_revive(self) -> None:
        """
        Called internally when the shell timer expires and ALIVE is auto-pushed.
        Resets horizontal velocity so the koopa resumes patrol.
        """
        self._apply_size()
        if self.vx == 0.0:
            self.vx = -WALK_SPEED   # default patrol direction

    # ── Debug ────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (f"<Koopa pos=({self.gObj.x:.0f},{self.gObj.y:.0f}) "
                f"vx={self.vx:.0f} sm={self.sm}>")