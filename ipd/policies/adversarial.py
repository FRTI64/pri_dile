#!/usr/bin/env python3
"""Constrained adversarial exploitation wrapper for the neural IPD policies.

The base policy (BC MLP, GRU, or reward-trained RL) is wrapped with two
exploitation rules discovered by constrained adversarial tuning against the
opponent pool:

1. **Late schedule exploitation**
   If the opponent's first defection happens on/after ``late_day`` and the next
   opponent move is also a defection, it is almost certainly a late harvester
   (or another non-responding schedule).  Defect for the rest of the match.

2. **Non-responsive opponent exploitation**
   With enough observations, if the opponent's conditional defection
   probabilities satisfy ``pD - pC <= gap``, its behaviour does not depend on
   ours.  Against such an opponent, permanent defection is the best response.

Both rules only trigger *after the opponent has already defected*, so they
cannot trip a Grim trigger.  The parameters are selected by
``train_adversarial_policy.py`` under the constraint that no validation
distribution may lose score.

Usage::

    from ipd.policies.adversarial import AdversarialPolicy
    p = AdversarialPolicy.from_config("configs/adversarial_policy_config.json")
    # RL base:
    p = AdversarialPolicy.from_config(
        "configs/adversarial_rl_policy_config_aggressive.json")
    p.reset()
    action = p.move(day, mine, theirs)
"""
from __future__ import annotations

from ipd.paths import config_path, weights_path

import json
import os

from ipd.policies.neural import C, D


class AdversarialPolicy:
    """Base neural policy + constrained late/non-responsive exploitation."""

    name = "Adversarial"

    def __init__(
        self,
        base,
        gap: float | None = 0.10,
        min_c: int = 8,
        min_d: int = 3,
        late_day: int | None = 50,
        repair: bool = False,
        name: str | None = None,
    ):
        if gap is None and late_day is None:
            raise ValueError("enable at least one exploitation rule")
        self.base = base
        self.gap = gap
        self.min_c = int(min_c)
        self.min_d = int(min_d)
        self.late_day = late_day
        self.repair = bool(repair)
        if name:
            self.name = name
        self.reset()

    @classmethod
    def from_config(cls, path: str, seed: int = 20261024,
                    mode: str = "sample",
                    name: str | None = None) -> "AdversarialPolicy":
        with open(path, encoding="utf-8") as fh:
            cfg = json.load(fh)
        if cfg.get("format") != "ipd-adversarial-config-v1":
            raise ValueError(f"unsupported config format: {cfg.get('format')}")

        base_path = cfg["base_weights"]
        if not os.path.isabs(base_path):
            # Look next to the config first (portable submit bundle), then in
            # the repository's weights/ directory (normal development layout).
            for candidate in (
                os.path.join(os.path.dirname(os.path.abspath(path)), base_path),
                weights_path(base_path),
            ):
                if os.path.exists(candidate):
                    base_path = candidate
                    break

        arch = cfg.get("base_arch", "mlp")
        if arch == "mlp":
            from ipd.policies.neural import NeuralPolicy
            base = NeuralPolicy(base_path, seed=seed, mode=mode)
        elif arch == "gru":
            from ipd.policies.gru import GRUPolicy
            base = GRUPolicy(base_path, seed=seed, mode=mode)
        elif arch == "rl":
            from ipd.policies.neural import NeuralPolicy
            from ipd.policies.rl import RLPolicy
            with open(base_path, encoding="utf-8") as fh:
                fmt = json.load(fh).get("format")
            base_cls = (NeuralPolicy if fmt == "ipd-neural-policy-v1"
                        else RLPolicy)
            base = base_cls(base_path, seed=seed, mode=mode)
        else:
            raise ValueError(f"unknown base_arch: {arch}")

        label = {"gru": "Adv-GRU", "rl": "Adv-RL"}.get(arch, "Adv-MLP")
        return cls(
            base,
            gap=cfg.get("gap"),
            min_c=cfg.get("min_c", 8),
            min_d=cfg.get("min_d", 3),
            late_day=cfg.get("late_day"),
            repair=cfg.get("repair", False),
            name=name or f"{label}-{mode}",
        )

    def reset(self) -> None:
        self.base.reset()
        self.punish = False
        self.first_d: int | None = None

    def _observe(self, mine: list[int], theirs: list[int]) -> None:
        observe = getattr(self.base, "observe", None)
        if observe is not None:
            observe(mine, theirs)

    def _start_punishment(self, mine: list[int], theirs: list[int]) -> int:
        self.punish = True
        self._observe(mine, theirs)
        return D

    def act(self, t: int, mine: list[int], theirs: list[int]) -> int:
        # Final round: D is dominant.  Still advance a recurrent base so its
        # state remains consistent if the harness queries further.
        if t >= 99:
            self._observe(mine, theirs)
            return D

        if self.punish:
            if (self.repair and len(theirs) >= 2
                    and theirs[-1] == C and theirs[-2] == C):
                self.punish = False
                return self.base.act(t, mine, theirs)
            self._observe(mine, theirs)
            return D

        if self.first_d is None and D in theirs:
            self.first_d = len(theirs) - 1

        # Rule 1: late first defection followed by a second defection.
        if (self.late_day is not None and self.first_d is not None
                and self.first_d >= self.late_day
                and len(theirs) >= self.first_d + 2
                and theirs[self.first_d + 1] == D):
            return self._start_punishment(mine, theirs)

        # Rule 2: enough evidence that the opponent does not respond to us.
        if self.gap is not None:
            n_c = n_d = d_c = d_d = 0
            for i in range(1, len(theirs)):
                if mine[i - 1] == C:
                    n_c += 1
                    d_c += theirs[i] == D
                else:
                    n_d += 1
                    d_d += theirs[i] == D
            if n_c >= self.min_c and n_d >= self.min_d:
                p_c = d_c / n_c
                p_d = d_d / n_d
                if p_d - p_c <= self.gap:
                    return self._start_punishment(mine, theirs)

        return self.base.act(t, mine, theirs)

    def move(self, day: int, mine: list[int], theirs: list[int]) -> int:
        return self.act(day - 1, mine, theirs)

    def __repr__(self) -> str:
        return self.name


def _payoff(me: int, them: int) -> int:
    if me == C:
        return 6 if them == C else 0
    return 10 if them == C else 1


def _self_test(path: str) -> None:
    p = AdversarialPolicy.from_config(path)
    for name, opp in (
        ("AllC", lambda t, m, o: C),
        ("AllD", lambda t, m, o: D),
        ("TFT", lambda t, m, o: C if t == 0 else o[-1]),
        ("Grim", lambda t, m, o: D if D in o else C),
    ):
        p.reset()
        mine: list[int] = []
        theirs: list[int] = []
        score = 0
        for t in range(100):
            a = p.act(t, mine, theirs)
            b = opp(t, mine, theirs)
            mine.append(a)
            theirs.append(b)
            score += _payoff(a, b)
        print(f"{name:8} C={mine.count(C):3d} D={mine.count(D):3d} score={score:4d}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default=config_path("adversarial_policy_config.json"))
    args = ap.parse_args()
    _self_test(args.config)
