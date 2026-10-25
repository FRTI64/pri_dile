"""Memory-one, reactive, periodic, and finite-state opponents."""
from __future__ import annotations

from ..primitives import C, D, Strategy


class TwoTitsForTat(Strategy):
    name = "TTFT"

    def reset(self):
        self.punish = 0

    def act(self, t, mine, theirs):
        if t == 0:
            return C
        if self.punish:
            self.punish -= 1
            return D
        if theirs[-1] == D:
            self.punish = 1
            return D
        return C


class AllCWithLastD(Strategy):
    name = "AllC-lastD"

    def act(self, t, mine, theirs):
        return D if t == 99 else C


class GrudgerK(Strategy):
    def __init__(self, k=3, rng=None):
        self.k = k
        super().__init__(rng)
        self.name = f"Grudger{self.k}"

    def act(self, t, mine, theirs):
        return D if theirs.count(D) >= self.k else C


class OmegaTFTLite(Strategy):
    """TFT-like strategy with deadlock repair and alternation detection."""

    name = "OmegaTFT-lite"

    def reset(self):
        self.dead = False

    def act(self, t, mine, theirs):
        if t == 0:
            return C
        if self.dead:
            return D
        # Long alternating opponent history: exploit it.
        if (len(theirs) >= 8 and
                all(theirs[i] != theirs[i - 1]
                    for i in range(len(theirs) - 7, len(theirs)))):
            self.dead = True
            return D
        # Two consecutive DD rounds: make one cooperative repair attempt.
        if (len(theirs) >= 2 and mine[-1] == D and theirs[-1] == D and
                mine[-2] == D and theirs[-2] == D):
            return C
        return theirs[-1]


class Periodic(Strategy):
    def __init__(self, pattern=(C, D, C), start=0, rng=None):
        self.pattern = tuple(pattern)
        self.start = start
        super().__init__(rng)
        self.name = "Periodic" + "".join("C" if x == C else "D"
                                         for x in self.pattern)

    def act(self, t, mine, theirs):
        return self.pattern[(t + self.start) % len(self.pattern)]


class MemoryOne(Strategy):
    """General memory-one plan.

    p = (p_CC, p_CD, p_DC, p_DD), where the subscript is
    (my previous move, opponent previous move).
    """

    def __init__(self, p, first=C, name="MemOne", rng=None):
        self.p = tuple(float(x) for x in p)
        self.first = first
        super().__init__(rng)
        self.name = name

    def act(self, t, mine, theirs):
        if t == 0:
            return self.first
        key = (mine[-1], theirs[-1])
        q = self.p[(0 if key == (C, C) else
                    1 if key == (C, D) else
                    2 if key == (D, C) else 3)]
        return C if self.rng.random() < q else D


class Reactive(Strategy):
    def __init__(self, first=C, p_after_c=0.8, p_after_d=0.1,
                 name="Reactive", rng=None):
        self.first = first
        self.p_after_c = p_after_c
        self.p_after_d = p_after_d
        super().__init__(rng)
        self.name = name

    def act(self, t, mine, theirs):
        if t == 0:
            return self.first
        q = self.p_after_c if theirs[-1] == C else self.p_after_d
        return C if self.rng.random() < q else D


class RandomFSM(Strategy):
    """Random finite-state machine, transitions driven by opponent's last move."""

    def __init__(self, rng, k=3):
        self.k = k
        self.output = [C if rng.random() < 0.58 else D for _ in range(k)]
        self.next_c = [[rng.randrange(k) for _ in range(2)] for _ in range(k)]
        self.next_d = [[rng.randrange(k) for _ in range(2)] for _ in range(k)]
        self.state = 0
        super().__init__(rng)
        self.name = "FSM-rand"

    def reset(self):
        self.state = 0

    def act(self, t, mine, theirs):
        if t > 0:
            table = self.next_c if theirs[-1] == C else self.next_d
            self.state = table[self.state][self.rng.randrange(2)]
        return self.output[self.state]


__all__ = [
    "TwoTitsForTat", "AllCWithLastD", "GrudgerK", "OmegaTFTLite",
    "Periodic", "MemoryOne", "Reactive", "RandomFSM",
]
