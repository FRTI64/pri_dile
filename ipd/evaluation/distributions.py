"""Official evaluation and training distributions.

``original``/``late``/``grim30``/``wide`` are the four headline distributions.
``hard`` is a training-only tail-risk mixture and is never used for reporting.
"""
from __future__ import annotations

import random
from typing import Sequence

from ..core.opponents import DelayedTFT, GenerousTFT, Joss, LateDefector
from ..core.opponents import NoisyTFT, RandomP
from ..core.opponents.axelrod_bank import (
    sample_axelrod_cheating_field, sample_axelrod_field,
)
from ..core.opponents.bank import (
    TYPE_BANK, TYPE_BY_NAME, effective_weights, sample_mixed_population,
    sample_mixture, sample_population,
)
from ..core.opponents.spec import TypeSpec
from ..policies.opponents import ALL_SPECS, SMART, TUNED_SMART

N_PLAYERS = 299
DISTS = ("original", "late", "grim30", "wide")
# Optional opponent banks backed by the Axelrod library.  ``axelrod`` is the
# full 243-strategy ordinary bank; ``axelrod-fast`` removes the 19 strategies
# classified as long-running (mostly Meta players that simulate the bank);
# ``axelrod-cheating`` contains strategies Axelrod marks as not obeying its
# tournament rules (4.14.0: Darwin only).
AXELROD_DISTS = ("axelrod", "axelrod-fast", "axelrod-cheating")
EVAL_DISTS = DISTS + AXELROD_DISTS
PROTOCOL_TRAIN_DISTS = ("train_v3", "train_easy", "train_hard",
                        "train_complex", "train_policy")
PROTOCOL_VAL_DISTS = ("dev_param", "dev_family", "dev_complex", "dev_ood")
TRAIN_DISTS = DISTS + ("hard", "hard2", "robust", "repair") \
    + PROTOCOL_TRAIN_DISTS
VALIDATION_DISTS = DISTS + PROTOCOL_VAL_DISTS

# Base weights for a fixed high-Grim population.  Late harvesters remain a
# significant block, so raising Grim tests the safety/exploitation tradeoff.
HIGH_GRIM_BASE = {
    "AllC": 8, "TFT": 18, "TFT2": 6, "GTFT.05": 8, "GTFT.10": 5,
    "Pavlov": 5, "AllD": 5, "Random.50": 5, "Tester": 2, "Prober": 2,
    "LateD5": 3, "LateD10": 4, "LateD15": 3, "LateD20": 3,
    "LateD25": 3, "LateD30": 3, "LateD40": 3, "LateD50": 3,
    "Mistrust": 3, "Grudger2": 2, "Grudger3": 1,
}
ORIGINAL_SCENARIO_WEIGHTS = {
    "balanced": 22, "cooperative": 8, "defector": 8, "naive": 8,
    "sophisticated": 22, "noisy": 12, "punitive": 10, "wild": 10,
}

# Fine-grained extra opponents used by the "wide" distribution.
EXTRA: list[TypeSpec] = []


def _add_extra(name: str, weight: float, factory) -> None:
    EXTRA.append(TypeSpec(name, "fine", weight, factory))


for _g in (0.01, 0.03, 0.08, 0.12, 0.18, 0.25, 0.35, 0.50, 0.70):
    _add_extra(f"GTFT.fine{_g}", 1.0,
               lambda r, g=_g: GenerousTFT(g, True, r))
for _p in (0.01, 0.03, 0.08, 0.12, 0.18, 0.25, 0.35):
    _add_extra(f"Joss.fine{_p}", 1.0, lambda r, p=_p: Joss(p, r))
    _add_extra(f"Noisy.fine{_p}", 1.0, lambda r, p=_p: NoisyTFT(p, r))
for _p in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60):
    _add_extra(f"Random.fine{_p}", 1.0, lambda r, p=_p: RandomP(p, r))
for _k in (1, 2, 3, 6, 8, 12, 18, 22, 28, 35, 45, 55, 60, 65, 70, 75, 80):
    _add_extra(f"LateD.fine{_k}", 1.0,
               lambda r, k=_k: LateDefector(k, r))
for _d in (3, 4, 5):
    _add_extra(f"TFTdelay.fine{_d}", 1.0,
               lambda r, d=_d: DelayedTFT(d, r))

# Tail-risk opponents used only while training learned policies.
HARD_OPPONENTS: dict[str, float] = {
    "ProbeBot": 3.0,
    "RemorsefulProber": 2.0,
    "Prober": 2.0,
    "Tester": 2.0,
    "EarlyD1": 2.0,
    "EarlyD10": 3.0,
    "EarlyD25": 3.0,
    "Bully": 2.0,
    "Rand3->TFT": 2.0,
    "Random.90": 1.0,
    "Extort1.5": 1.0,
    "Extort3": 1.0,
    "MemOne-rand": 1.0,
    "NoisyTFT.20": 1.0,
    "Joss.20": 1.0,
    "Grim": 1.0,
}

# Nine scenario suites used by ``ipd suite``.
FAMILY_MULT: dict[str, dict[str, float]] = {
    "original": {"nice": 3, "forgiving": 3, "punitive": 2, "defect": 2,
                 "late": 2, "random": 2, "wild": 1, "adaptive": 1,
                 "plus": 0.3, "smart": 0.2},
    "late": {"nice": 2, "forgiving": 2, "punitive": 1, "defect": 1,
             "late": 6, "random": 1, "wild": 0.5, "adaptive": 0.5,
             "plus": 0.3, "smart": 0.2},
    "grim": {"nice": 2, "forgiving": 1, "punitive": 5, "defect": 2,
             "late": 1, "random": 1, "wild": 0.5, "adaptive": 0.5,
             "plus": 0.3, "smart": 0.2},
    "noise": {"nice": 2, "forgiving": 6, "punitive": 1, "defect": 1,
              "late": 1, "random": 3, "wild": 1, "adaptive": 1,
              "plus": 1, "smart": 0.3},
    "complex": {"nice": 1, "forgiving": 1, "punitive": 1, "defect": 2,
                "late": 1, "random": 3, "wild": 6, "adaptive": 4,
                "plus": 4, "smart": 0.5},
    "smart": {"nice": 2, "forgiving": 2, "punitive": 1, "defect": 1,
              "late": 2, "random": 1, "wild": 1, "adaptive": 2,
              "plus": 1, "smart": 6},
    "punitive": {"nice": 1, "forgiving": 1, "punitive": 6, "defect": 2,
                 "late": 1, "random": 1, "wild": 0.5, "adaptive": 0.5,
                 "plus": 0.5, "smart": 0.2},
    "random": {"nice": 1, "forgiving": 1, "punitive": 1, "defect": 1,
               "late": 1, "random": 6, "wild": 1, "adaptive": 1,
               "plus": 2, "smart": 0.2},
    "wide": {"nice": 1.5, "forgiving": 1.5, "punitive": 1.5, "defect": 1.5,
             "late": 1.5, "random": 1.5, "wild": 2, "adaptive": 2,
             "plus": 1, "smart": 1},
}


def _population_from_specs(specs: Sequence[TypeSpec],
                           weights: Sequence[float], n: int,
                           rng: random.Random):
    pop = []
    for _ in range(n):
        spec = rng.choices(specs, weights=weights, k=1)[0]
        obj = spec.factory(rng)
        obj.name = spec.name
        pop.append(obj)
    return pop


def build_field(kind: str, seed: int, n: int = N_PLAYERS):
    """Build an ``original``/``late``/``grimN``/``mixed`` field."""
    rng = random.Random(seed)
    if kind == "original":
        scen = rng.choices(list(ORIGINAL_SCENARIO_WEIGHTS),
                           weights=list(ORIGINAL_SCENARIO_WEIGHTS.values()),
                           k=1)[0]
        specs, probs = sample_mixture(rng, scen)
        return _population_from_specs(specs, probs, n, rng), scen
    if kind == "late":
        return sample_population(n, rng, "late_harvest"), "late_harvest"
    if kind == "mixed":
        return sample_mixed_population(n, rng)
    if kind.startswith("grim"):
        grim = int(kind[4:])
        weights = dict(HIGH_GRIM_BASE)
        weights["Grim"] = grim
        specs = [TYPE_BY_NAME[name] for name in weights]
        return _population_from_specs(specs, list(weights.values()), n, rng), kind
    raise ValueError(f"unknown field kind: {kind}")


def sample_wide(seed: int, n: int = 60, smart_share: float | None = None):
    """Sample the wide/smart distribution used for blind-style evaluations."""
    rng = random.Random(seed)
    scen = rng.choices(list(ORIGINAL_SCENARIO_WEIGHTS),
                       weights=list(ORIGINAL_SCENARIO_WEIGHTS.values()),
                       k=1)[0]
    if rng.random() < 0.25:
        scen = "late_harvest"
    base_specs, base_w = effective_weights(scen)
    specs = list(base_specs) + list(EXTRA) + list(SMART)
    weights = list(base_w) + [s.weight for s in EXTRA] + [0.0] * len(SMART)
    if smart_share is None:
        smart_share = rng.choice([0.0, 0.03, 0.08, 0.13, 0.20, 0.30])
    base_total = sum(weights)
    smart_total = sum(s.weight for s in SMART)
    factor = 0.0 if smart_share <= 0 else (
        smart_share / (1.0 - smart_share) * base_total / smart_total)
    weights[-len(SMART):] = [factor * s.weight for s in SMART]
    return _population_from_specs(specs, weights, n, rng), scen, smart_share


def build_dist_field(dist: str, seed: int, n: int):
    """Build one of the official evaluation distributions.

    The four headline distributions are drawn from the pri_dile opponent bank.
    ``axelrod`` and ``axelrod-fast`` use the optional Axelrod library bank;
    ``axelrod-cheating`` adds strategies Axelrod classifies as cheating.

    Protocol training profiles and validation tiers are forwarded to
    :func:`build_train_dist_field` so BC/GRU data generation can use them.
    """
    if dist in PROTOCOL_TRAIN_DISTS or dist in PROTOCOL_VAL_DISTS:
        return build_train_dist_field(dist, seed, n)
    if dist == "original":
        return build_field("original", seed, n)[0]
    if dist == "late":
        return build_field("late", seed, n)[0]
    if dist == "grim30":
        return build_field("grim30", seed, n)[0]
    if dist == "wide":
        return sample_wide(seed, n, 0.15)[0]
    if dist == "axelrod":
        return sample_axelrod_field(seed, n, include_long_run=True)
    if dist == "axelrod-fast":
        return sample_axelrod_field(seed, n, include_long_run=False)
    if dist == "axelrod-cheating":
        return sample_axelrod_cheating_field(seed, n)
    raise ValueError(f"unknown distribution: {dist}")


def build_train_dist_field(dist: str, seed: int, n: int):
    """Build a training field, including the tail-risk ``hard`` mixture."""
    if dist in PROTOCOL_TRAIN_DISTS:
        from .protocol import build_training_field
        return build_training_field(dist, seed, n)
    if dist in PROTOCOL_VAL_DISTS:
        from .protocol import build_protocol_field
        return build_protocol_field(dist, seed, n)
    if dist == "hard2":
        from .hard2 import build_hard2_field
        return build_hard2_field(seed, n)
    if dist == "robust":
        from .robust import build_robust_field
        return build_robust_field(seed, n)
    if dist == "repair":
        from .robust import build_repair_field
        return build_repair_field(seed, n)
    if dist != "hard":
        return build_dist_field(dist, seed, n)
    rng = random.Random(seed)
    names = [name for name in HARD_OPPONENTS if name in TYPE_BY_NAME]
    weights = [HARD_OPPONENTS[name] for name in names]
    specs = [TYPE_BY_NAME[name] for name in names]
    return _population_from_specs(specs, weights, n, rng)


def sample_suite_field(suite: str, seed: int, n: int = N_PLAYERS):
    """Sample one of the nine family-weighted suites."""
    rng = random.Random(seed)
    mult = FAMILY_MULT[suite]
    weights = [max(0.001, mult.get(spec.family, 0.5)) for spec in ALL_SPECS]
    return _population_from_specs(ALL_SPECS, weights, n, rng)


def sample_blind_field(seed: int, n: int = N_PLAYERS):
    """Blind-test field: unseen wide/smart population with 15% smart types."""
    return sample_wide(seed, n, 0.15)[0]


__all__ = [
    "N_PLAYERS", "DISTS", "TRAIN_DISTS", "PROTOCOL_TRAIN_DISTS",
    "PROTOCOL_VAL_DISTS", "VALIDATION_DISTS", "AXELROD_DISTS", "EVAL_DISTS",
    "HIGH_GRIM_BASE", "ORIGINAL_SCENARIO_WEIGHTS", "HARD_OPPONENTS",
    "FAMILY_MULT", "EXTRA", "build_field", "build_dist_field",
    "build_train_dist_field", "sample_wide", "sample_suite_field",
    "sample_blind_field", "TYPE_BANK", "TYPE_BY_NAME", "TUNED_SMART",
]
