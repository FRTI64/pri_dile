"""Combined opponent banks used by the evaluation distributions."""
from __future__ import annotations

from ...core.opponents import TYPE_BANK
from ...core.opponents.plus import PLUS as _PLUS_BASE
from ...core.opponents.spec import TypeSpec
from .smart import SMART
from .tuned_smart import TUNED_SMART

# Full opponent bank in the historical order: extended bank, our smart
# strategies, extra parameterised types, then the tuned copies.
PLUS: list[TypeSpec] = list(_PLUS_BASE) + list(TUNED_SMART)
ALL_SPECS: list[TypeSpec] = (
    list(TYPE_BANK) + list(SMART) + list(_PLUS_BASE) + list(TUNED_SMART)
)
_by_name = {spec.name: spec for spec in ALL_SPECS}
ALL_SPECS = list(_by_name.values())
ALL_BY_NAME = {spec.name: spec for spec in ALL_SPECS}
