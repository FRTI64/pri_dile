"""Candidate: RL-V5-FT with a combined probe / late-defector guard.

Two reactive windows use different test sequences:

* early probe: first opponent defection at zero-based rounds 5 or 7,
  confirmed by three consecutive defections, then ``C, C, C`` (the probe
  family is allD, so cooperation reveals whether it is an extort/harvest
  variant);
* late defector: first opponent defection at round 9 or later, confirmed
  by three consecutive defections, then ``D, C, C`` (adaptive probes
  repair after retaliation; the confirmation filters out one-off probes).

The guard never defects before the opponent, does not change the base safety
rules, and releases as soon as the opponent cooperates.  It combines the
separate ``rl-probe-guard`` and ``rl-defection-guard`` experiments.
"""
from __future__ import annotations

from ...core.primitives import C, D
from ...paths import weights_path
from ..meta import ProbeGuardPolicy

KIND = "rl-full-guard"
DISPLAY = "RL-FullGuard"
BASE_WEIGHTS = "rl_policy_weights_v5_ft.json"
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
