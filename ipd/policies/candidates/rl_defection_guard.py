"""Candidate: RL-V5-FT with a general unprovoked-defection guard.

The base policy is RL-V5-FT.  If the opponent's first defection happens at
zero-based round 9 or later and is followed by a second defection while we
have not defected first, the guard plays ``D, C, C``.  A cooperative reply
releases the guard; continued defection switches to punishment until the
opponent cooperates again.  The late trigger requires three consecutive
opponent defections, which filters out one-off adaptive probes.

The initial defection avoids gifting free cooperation to adaptive probes,
which repair after retaliation and would otherwise exploit a pure
cooperation test.  This targets late defectors (which switch to permanent
defection after a long cooperative phase).  The trigger is reactive and
leaves the base safety rules untouched.
"""
from __future__ import annotations

from ...core.primitives import C, D
from ...paths import weights_path
from ..meta import ProbeGuardPolicy

KIND = "rl-defection-guard"
DISPLAY = "RL-DefectionGuard"
BASE_WEIGHTS = "rl_policy_weights_v5_ft.json"
MIN_FIRST = 9
MAX_FIRST = 99
OBSERVE_ROUNDS = 3


def build(*, seed: int = 20261024, mode: str = "sample",
          name: str | None = None, weights: str | None = None,
          **_: object) -> object:
    return ProbeGuardPolicy(
        weights or weights_path(BASE_WEIGHTS),
        seed=seed, mode=mode, late_min_first=MIN_FIRST, late_confirm=3,
        response=(C, C, C), late_response=(D, C, C),
        observe_rounds=OBSERVE_ROUNDS, name=name or DISPLAY)


__all__ = ["KIND", "DISPLAY", "BASE_WEIGHTS", "MIN_FIRST", "MAX_FIRST",
           "OBSERVE_ROUNDS", "build"]
