"""
StateMachine.py
---------------
A lightweight, reusable generic state machine base class.

Subclasses register legal transitions via add_transition() and optionally
attach side-effect callbacks via on_enter() / on_exit(). The machine
validates every transition against the table and fires callbacks in order:
    exit(old) → change state → enter(new)

Usage
-----
    class TrafficLight(StateMachine):
        def __init__(self):
            super().__init__(initial="red")
            self.add_transition("red",    "green")
            self.add_transition("green",  "yellow")
            self.add_transition("yellow", "red")
            self.on_enter("green", lambda: print("Go!"))

    light = TrafficLight()
    light.transition("green")    # prints "Go!"
    light.transition("red")      # rejected — no green→red edge, returns False
    light.transition("yellow")   # ok
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Set


class StateMachine:
    """
    Generic finite state machine.

    Parameters
    ----------
    initial : Any
        The starting state. Any hashable value works (str, int, Enum …).
    """

    def __init__(self, initial: Any) -> None:
        self._state:       Any                    = initial
        self._transitions: Dict[Any, Set[Any]]    = {}
        self._on_enter:    Dict[Any, Callable]    = {}
        self._on_exit:     Dict[Any, Callable]    = {}

    # ── State access ─────────────────────────────────────────────────────────

    @property
    def state(self) -> Any:
        """The current state (read-only)."""
        return self._state

    # ── Registration ─────────────────────────────────────────────────────────

    def add_transition(self, from_state: Any, to_state: Any) -> None:
        """
        Register a legal directed edge from_state → to_state.
        Can be called multiple times for the same from_state.
        """
        self._transitions.setdefault(from_state, set()).add(to_state)

    def on_enter(self, state: Any, callback: Callable) -> None:
        """
        Register *callback* to fire immediately after entering *state*.
        Replaces any previously registered callback for that state.
        """
        self._on_enter[state] = callback

    def on_exit(self, state: Any, callback: Callable) -> None:
        """
        Register *callback* to fire immediately before leaving *state*.
        Replaces any previously registered callback for that state.
        """
        self._on_exit[state] = callback

    # ── Transition ───────────────────────────────────────────────────────────

    def can_transition(self, to_state: Any) -> bool:
        """Return True if current → to_state is a registered edge."""
        return to_state in self._transitions.get(self._state, set())

    def transition(self, to_state: Any, force: bool = False) -> bool:
        """
        Attempt to move to *to_state*.

        Parameters
        ----------
        to_state : Any
            The desired next state.
        force : bool
            If True, bypass the transition table entirely. Useful for
            hard resets where you need to return to a known state without
            registering every possible recovery edge.

        Returns
        -------
        bool
            True if the transition happened, False if it was rejected.
        """
        if not force and not self.can_transition(to_state):
            return False

        # Fire exit callback for the state we're leaving
        exit_cb = self._on_exit.get(self._state)
        if exit_cb:
            exit_cb()

        self._state = to_state

        # Fire enter callback for the state we just entered
        enter_cb = self._on_enter.get(self._state)
        if enter_cb:
            enter_cb()

        return True

    # ── Debug ────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} state={self._state!r}>"