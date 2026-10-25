"""Additional opponent types: memory-one, FSMs, reactive, ZD variants.

This module is policy-free: it only depends on the core simulator and the
extended opponent bank.  Tuned copies of our own policies live in
:mod:`ipd.policies.opponents`.
"""
from __future__ import annotations

import random

from .opponents import GenerousTFT, Joss, NoisyTFT, RandomP
from .opponents_extended import (
    ExtortZD, GenerousZD, MemoryOne, RandomFSM, Reactive, TypeSpec,
    TYPE_BANK,
)
from .simulator import C

PLUS: list[TypeSpec] = []


def add(name: str, weight: float, factory, family: str = "plus") -> None:
    PLUS.append(TypeSpec(name, family, weight, factory))


def _memone(seed: int):
    r = random.Random(seed)
    p = (r.uniform(0.7, 1.0), r.uniform(0.0, 0.5),
         r.uniform(0.4, 1.0), r.uniform(0.0, 0.3))
    return lambda rr: MemoryOne(p, C, f"MemOne.F{seed}", rr)


def _reactive(seed: int):
    r = random.Random(seed)
    return lambda rr: Reactive(C, r.uniform(0.4, 1.0), r.uniform(0.0, 0.6),
                               f"Reactive.F{seed}", rr)


for i in range(8):
    add(f"MemOne.F{i}", 0.8, _memone(100 + i))
for i in range(8):
    add(f"Reactive.F{i}", 0.8, _reactive(200 + i))
for i in range(6):
    seed = 300 + i
    add(f"FSM.F{i}", 0.7,
        lambda r, seed=seed: RandomFSM(random.Random(seed), 3))
for chi in (1.2, 1.5, 2.0, 2.5, 3.0):
    add(f"Extort.F{chi}", 0.5, lambda r, chi=chi: ExtortZD(chi, r))
for k in (1.2, 2.0, 3.0):
    add(f"GenZD.F{k}", 0.4, lambda r, k=k: GenerousZD(k, r))
for g in (0.02, 0.07, 0.13, 0.22, 0.30, 0.45, 0.60):
    add(f"GTFT.F{g}", 0.5, lambda r, g=g: GenerousTFT(g, True, r))
for p in (0.015, 0.04, 0.07, 0.11, 0.16, 0.22, 0.30):
    add(f"Joss.F{p}", 0.4, lambda r, p=p: Joss(p, r))
    add(f"Noisy.F{p}", 0.4, lambda r, p=p: NoisyTFT(p, r))
for p in (0.08, 0.18, 0.28, 0.42, 0.55, 0.75, 0.85):
    add(f"Random.F{p}", 0.3, lambda r, p=p: RandomP(p, r))

ALL_SPECS = list(TYPE_BANK) + list(PLUS)
_by_name = {spec.name: spec for spec in ALL_SPECS}
ALL_SPECS = list(_by_name.values())
