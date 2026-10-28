"""Optional GPU/batched inference for Protocol v4 evaluation.

The package is deliberately import-light: ``torch`` is imported by the modules
that need it, and callers must treat :class:`ImportError` as "GPU unavailable".
The protocol CLI keeps the pure-CPU path as its default and only imports this
package when ``--gpu`` is enabled.
"""
from __future__ import annotations

__all__ = ["features", "policies", "policy_opponents", "simulator"]
