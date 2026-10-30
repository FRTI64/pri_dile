"""Candidate: RL-V5-FT-Complex with the probe / late defection guard.

Same reactive guard as ``rl-full-guard`` but the base is the
``rl-v5-ft-complex`` weights, which improve dev_complex and test_v8 over
RL-V5-FT.  Probe uses ``C,C,C``; late uses ``D,C,C``; both require three
consecutive defections.
"""
from __future__ import annotations

from ...core.primitives import C, D
from ...paths import weights_path
from ..meta import ProbeGuardPolicy

KIND = "rl-full-guard-c"
DISPLAY = "RL-FullGuard-C"
BASE_WEIGHTS = "rl_policy_weights_v5_ft_complex.json"
PROBE_FIRST_ROUNDS = (5, 7)
LATE_MIN_FIRST = 9
OBSERVE_ROUNDS = 3


def build(*, seed: int = 20261024, mode: str = "sample",
          name: str | None = None, weights: str | None = None,
          **_: object) -> object:
    return ProbeGuardPolicy(
        weights or weights_path(BASE_WEIGHTS),
        seed=seed, mode=mode, exact_first=PROBE_FIRST_ROUNDS,
        late_min_first=LATE_MIN_FIRST, confirm=3, late_confirm=3,
        response=(C, C, C), late_response=(D, C, C),
        observe_rounds=OBSERVE_ROUNDS, name=name or DISPLAY)


__all__ = ["KIND", "DISPLAY", "BASE_WEIGHTS", "PROBE_FIRST_ROUNDS",
           "LATE_MIN_FIRST", "OBSERVE_ROUNDS", "build"]
