"""High-level fast rule bank with a CPU fallback for irregular opponents.

``RuleBank`` takes the exact opponent objects from a protocol field or a
training environment.  Every object that carries a :class:`FastRuleSpec` is
encoded into one native batch; all other objects (Axelrod, learned policies,
``RandomFSM``, ``VotingPortfolio``, ``ProbeBot``) keep their original Python
implementation and are stepped row by row.

The public API mirrors the GPU opponent adapter: ``act(t, blue)`` followed by
``observe(t, blue, opponent)``.  Both sides act from the same pre-round
history, so the bank never appends the current round before computing actions.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .native import NativeRuleBank, available
from .spec import FastRuleSpec, get_fast_spec, infer_fast_spec


@dataclass
class _FallbackRow:
    index: int
    obj: Any
    mine: list[int]
    theirs: list[int]
    rng_state: tuple | None


class RuleBank:
    """Batched native rules plus row-wise Python fallback."""

    def __init__(self, opponents: Sequence[Any], *, rounds: int = 100,
                 enable: bool = True) -> None:
        self.rounds = int(rounds)
        self.opponents = list(opponents)
        self.n = len(self.opponents)
        self.names = [str(getattr(obj, "name", type(obj).__name__))
                      for obj in self.opponents]
        self.fast_rows: list[int] = []
        self.fast_specs: list[FastRuleSpec] = []
        self.fast_objects: list[Any] = []
        self.fallbacks: list[_FallbackRow] = []
        self.native: NativeRuleBank | None = None
        self.error: str | None = None

        if enable and self.n and available():
            try:
                self._build()
            except Exception as exc:  # pragma: no cover - defensive fallback
                self.error = str(exc)
                self.fast_rows.clear()
                self.fast_specs.clear()
                self.fast_objects.clear()
                self.fallbacks.clear()
                if self.native is not None:
                    self.native.close()
                    self.native = None
        if self.native is None:
            self._build_fallbacks(all_rows=True)

    def _build(self) -> None:
        for index, obj in enumerate(self.opponents):
            spec = get_fast_spec(obj) or infer_fast_spec(obj)
            if spec is None:
                continue
            reset = getattr(obj, "reset", None)
            if callable(reset):
                reset()
            self.fast_rows.append(index)
            self.fast_specs.append(spec)
            self.fast_objects.append(obj)
        if not self.fast_specs:
            return
        self.native = NativeRuleBank(self.fast_specs, self.fast_objects,
                                     rounds=self.rounds)
        self._build_fallbacks(all_rows=False)

    def _build_fallbacks(self, all_rows: bool) -> None:
        fast = set(self.fast_rows) if not all_rows else set()
        self.fallbacks = []
        for index, obj in enumerate(self.opponents):
            if index in fast:
                continue
            reset = getattr(obj, "reset", None)
            if callable(reset):
                reset()
            setter = getattr(obj, "set_rounds", None)
            if callable(setter):
                setter(self.rounds)
            rng = getattr(obj, "rng", None)
            state = rng.getstate() if isinstance(rng, random.Random) else None
            self.fallbacks.append(_FallbackRow(index, obj, [], [], state))

    @property
    def fast_count(self) -> int:
        return len(self.fast_rows)

    @property
    def fallback_count(self) -> int:
        return len(self.fallbacks)

    @property
    def fast_fraction(self) -> float:
        return self.fast_count / max(1, self.n)

    @property
    def enabled(self) -> bool:
        return self.native is not None and self.fast_count > 0

    def reset(self) -> None:
        if self.native is not None:
            self.native.reset()
        for row in self.fallbacks:
            reset = getattr(row.obj, "reset", None)
            if callable(reset):
                reset()
            row.mine.clear()
            row.theirs.clear()
            if row.rng_state is not None:
                row.obj.rng.setstate(row.rng_state)

    def act(self, t: int, blue: Sequence[int] | np.ndarray) -> list[int]:
        """Return one action per row for round ``t``."""
        if self.native is not None and self.fast_count == self.n:
            # Common case: the whole batch is native.  No scatter needed.
            return self.native.act(t, blue).tolist()
        actions = np.zeros(self.n, dtype=np.int8)
        if self.native is not None:
            actions[self.fast_rows] = self.native.act(t, blue)
        if self.fallbacks:
            for row in self.fallbacks:
                value = row.obj.act(t, row.mine, row.theirs)
                if value not in (0, 1):
                    raise ValueError(
                        f"{self.names[row.index]} returned {value!r}")
                actions[row.index] = int(value)
        return actions.tolist()

    def step(self, t: int, blue: Sequence[int] | np.ndarray) -> np.ndarray:
        """Compute actions and commit the round to the histories.

        This is equivalent to ``act`` followed by ``observe`` but uses one
        native call for the fast majority of rows.  It is used by the
        self-play ``VectorEnv`` where no caller needs the pre-commit state.
        """
        if self.native is not None and self.fast_count == self.n:
            return self.native.step(t, blue)
        actions = np.zeros(self.n, dtype=np.int8)
        blue_arr = np.asarray(blue, dtype=np.int32)
        if self.native is not None:
            fast_blue = (blue_arr if self.fast_count == self.n
                         else blue_arr[self.fast_rows])
            actions[self.fast_rows] = self.native.step(t, fast_blue)
        if self.fallbacks:
            for row in self.fallbacks:
                value = row.obj.act(t, row.mine, row.theirs)
                if value not in (0, 1):
                    raise ValueError(
                        f"{self.names[row.index]} returned {value!r}")
                actions[row.index] = int(value)
            for row in self.fallbacks:
                row.mine.append(int(actions[row.index]))
                row.theirs.append(int(blue_arr[row.index]))
        return actions

    def observe(self, t: int, blue: Sequence[int] | np.ndarray,
                opponent: Sequence[int] | np.ndarray) -> None:
        if self.native is not None:
            b = np.asarray(blue, dtype=np.int32)
            o = np.asarray(opponent, dtype=np.int32)
            if self.fast_count != self.n:
                b = b[self.fast_rows]
                o = o[self.fast_rows]
            self.native.observe(t, b, o)
        if self.fallbacks:
            b = np.asarray(blue, dtype=np.int32)
            o = np.asarray(opponent, dtype=np.int32)
            for row in self.fallbacks:
                row.mine.append(int(o[row.index]))
                row.theirs.append(int(b[row.index]))

    def close(self) -> None:
        if self.native is not None:
            self.native.close()
            self.native = None

    def __del__(self) -> None:  # pragma: no cover - GC timing
        try:
            self.close()
        except Exception:
            pass


__all__ = ["RuleBank"]
