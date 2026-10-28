"""Fast rule-opponent batch evaluation (Plan 3).

The package contains a native C++ interpreter for Protocol v4 rule families,
a ctypes loader, and a mixed bank that keeps irregular opponents on their
original Python implementation.  It is used by the GPU protocol simulator and
by the self-play ``VectorEnv``.
"""
from __future__ import annotations

from .bank import RuleBank
from .native import available, build_error
from .spec import (
    FastRuleSpec, attach_fast_spec, encode_spec, get_fast_spec,
    infer_fast_spec,
)

__all__ = [
    "RuleBank", "FastRuleSpec", "attach_fast_spec", "encode_spec",
    "get_fast_spec", "infer_fast_spec", "available", "build_error",
]
