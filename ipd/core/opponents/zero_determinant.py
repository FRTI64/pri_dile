"""Zero-determinant extortion and generous opponents."""
from __future__ import annotations

from ..primitives import C, D, Strategy

R, S, T, P = 6, 0, 10, 1


class ExtortZD(Strategy):
    """Zero-determinant extortion strategy: s_X-P = chi*(s_Y-P)."""

    def __init__(self, chi=2.0, rng=None):
        self.chi = chi
        c = max(chi - 1.0, 1e-9)
        limits = [
            1.0 / (c * (R - P)),
            1.0 / (chi * (T - P) + P),
            1.0 / (T + c * P),
        ]
        lam = 0.95 * min(limits)
        p = (
            1.0 - lam * c * (R - P),
            1.0 - lam * (chi * (T - P) + P),
            lam * (T + c * P),
            0.0,
        )
        super().__init__(rng)
        self.p = tuple(max(0.0, min(1.0, x)) for x in p)
        self.name = f"Extort{chi:g}"

    def act(self, t, mine, theirs):
        if t == 0:
            return C
        key = (mine[-1], theirs[-1])
        q = self.p[(0 if key == (C, C) else
                    1 if key == (C, D) else
                    2 if key == (D, C) else 3)]
        return C if self.rng.random() < q else D


class GenerousZD(Strategy):
    """Zero-determinant generous strategy: s_X-R = kappa*(s_Y-R)."""

    def __init__(self, kappa=2.0, rng=None):
        self.kappa = kappa
        k = max(kappa, 1e-9)
        limits = [
            1.0 / (k * T - (k - 1.0) * R),
            1.0 / (T + (k - 1.0) * R),
            1.0 / ((k - 1.0) * (R - P)),
        ]
        lam = 0.95 * min(x for x in limits if x > 0)
        p = (
            1.0,
            1.0 - lam * (k * T - (k - 1.0) * R),
            lam * (T + (k - 1.0) * R),
            lam * (k - 1.0) * (R - P),
        )
        super().__init__(rng)
        self.p = tuple(max(0.0, min(1.0, x)) for x in p)
        self.name = f"GenZD{kappa:g}"

    def act(self, t, mine, theirs):
        if t == 0:
            return C
        key = (mine[-1], theirs[-1])
        q = self.p[(0 if key == (C, C) else
                    1 if key == (C, D) else
                    2 if key == (D, C) else 3)]
        return C if self.rng.random() < q else D


__all__ = ["ExtortZD", "GenerousZD"]
