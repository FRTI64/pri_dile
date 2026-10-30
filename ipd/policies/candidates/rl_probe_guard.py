"""Candidate: RL-V5-FT with an early-exploitation guard.

The base policy is RL-V5-FT (the v5 fine-tune of RL-Wide160).  If the
opponent's first defection happens in round 6 or 8 and is followed by a second
defection while we have not defected first, the guard offers three
cooperations.  A cooperative reply releases the guard; continued defection
switches to permanent punishment until the opponent cooperates again.

The trigger is reactive and leaves the base safety rules untouched.  It is a
narrow response to the early-probe pattern, not a predictive gate.  Only one
weight file is needed.
"""
from __future__ import annotations

from ...paths import weights_path
from ..meta import ProbeGuardPolicy

KIND = "rl-probe-guard"
DISPLAY = "RL-ProbeGuard"
BASE_WEIGHTS = "rl_policy_weights_v5_ft.json"
# Zero-based first-defection rounds for the held-out probe timings (6/8).
# Earlier timings (2/4) are train-only; the guard deliberately stays narrow.
PROBE_FIRST_ROUNDS = (5, 7)
OBSERVE_ROUNDS = 3


def build(*, seed: int = 20261024, mode: str = "sample",
          name: str | None = None, weights: str | None = None,
          **_: object) -> object:
    return ProbeGuardPolicy(
        weights or weights_path(BASE_WEIGHTS),
        seed=seed, mode=mode, exact_first=PROBE_FIRST_ROUNDS,
        observe_rounds=OBSERVE_ROUNDS, name=name or DISPLAY)


__all__ = ["KIND", "DISPLAY", "BASE_WEIGHTS", "PROBE_FIRST_ROUNDS",
           "OBSERVE_ROUNDS", "build"]
