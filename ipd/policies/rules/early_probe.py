"""Early-probe machine exposed as a submission policy."""
from __future__ import annotations

from ...core.early import EarlyProbeMachine


class EarlyProbePolicy(EarlyProbeMachine):
    name = "EarlyProbe"
