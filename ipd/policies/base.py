"""Unified policy interface for tournament candidates.

A *policy* is anything that can choose :data:`C` or :data:`D` from the public
history.  Canonical policies implement::

    reset() -> None
    act(t, mine, theirs) -> C | D          # t is zero-based

Older hand-written machines implement the 1-based ``move(day, mine, theirs)``
form.  :class:`PolicyAdapter` normalises both forms, so evaluation and training
code never needs ``hasattr(policy, "act")`` checks.
"""
from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from ..core.simulator import C, D


@runtime_checkable
class Policy(Protocol):
    """The canonical tournament-policy protocol."""

    name: str

    def reset(self) -> None:  # pragma: no cover - protocol
        ...

    def act(self, t: int, mine: Sequence[int], theirs: Sequence[int]) -> int:
        ...


class PolicyAdapter:
    """Expose any legacy policy through the canonical 0-based interface."""

    def __init__(self, policy: Any, name: str | None = None):
        if not callable(getattr(policy, "act", None)) and not callable(
            getattr(policy, "move", None)
        ):
            raise TypeError(
                f"{type(policy).__name__} implements neither act() nor move()"
            )
        self.policy = policy
        self.name = name or getattr(policy, "name", type(policy).__name__)
        self._uses_move = not callable(getattr(policy, "act", None))

    def reset(self) -> None:
        reset = getattr(self.policy, "reset", None)
        if reset is not None:
            reset()

    def act(self, t: int, mine: Sequence[int], theirs: Sequence[int]) -> int:
        if self._uses_move:
            action = self.policy.move(t + 1, mine, theirs)
        else:
            action = self.policy.act(t, mine, theirs)
        if action not in (C, D):
            raise ValueError(f"{self.name} returned invalid action {action!r}")
        return int(action)

    def move(self, day: int, mine: Sequence[int], theirs: Sequence[int]) -> int:
        """1-based compatibility method."""
        return self.act(day - 1, mine, theirs)

    def observe(self, mine: Sequence[int], theirs: Sequence[int]) -> None:
        observe = getattr(self.policy, "observe", None)
        if observe is not None:
            observe(mine, theirs)

    def __getattr__(self, item: str) -> Any:
        # Keep advanced access (e.g. ``triple.f``) working for training and
        # introspection without expanding the public policy protocol.
        if item == "policy":
            raise AttributeError(item)
        return getattr(self.policy, item)

    def __repr__(self) -> str:
        return self.name


def ensure_policy(policy: Any, name: str | None = None) -> PolicyAdapter:
    """Return ``policy`` as a :class:`PolicyAdapter` (idempotent)."""
    if isinstance(policy, PolicyAdapter):
        if name is not None:
            policy.name = name
        return policy
    return PolicyAdapter(policy, name=name)


def boundaries_ok(policy: Any, rounds: int = 100) -> bool:
    """Smoke helper: first move C, last move D against a passive opponent."""
    p = ensure_policy(policy)
    p.reset()
    mine: list[int] = []
    theirs: list[int] = []
    for t in range(rounds):
        mine.append(p.act(t, mine, theirs))
        theirs.append(C)
    return mine[0] == C and mine[-1] == D
