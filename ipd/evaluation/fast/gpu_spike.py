"""Minimal GPU rule kernels used only by the Plan 3 C++/GPU spike.

The production decision is made with these two representative families:

* ``TFT`` -- a stateless standard rule;
* ``HarvestFactor`` -- a stateful temporal rule with a stochastic branch.

Both are implemented as per-round Torch kernels over the full ``[B, 100]``
history.  The mixed kernel runs both branches with masks, which is the honest
comparison for a heterogeneous rule bank.  This module is deliberately not a
full rule bank; it exists to decide whether porting all families to GPU would
be worth the maintenance cost.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch


@dataclass
class SpikeSpec:
    type_name: str
    start: int = 50
    repairable: bool = True
    p_defect: float = 1.0


def _attach_spec(bot: Any, type_name: str, **params: Any) -> None:
    try:
        from .spec import attach_fast_spec

        attach_fast_spec(bot, type_name, **params)
    except Exception:
        pass


@torch.no_grad()
def run_spike(specs: Sequence[SpikeSpec], blue: torch.Tensor, *,
              device: str | torch.device = "cuda",
              dtype: torch.dtype = torch.float32,
              seed: int = 1234) -> tuple[torch.Tensor, float]:
    """Return ``[B, T]`` opponent actions and the kernel wall time."""
    dev = torch.device(device)
    total, rounds = blue.shape
    blue = blue.to(dev, dtype=torch.int8)
    is_tft = torch.tensor([s.type_name == "TFT" for s in specs],
                          dtype=torch.bool, device=dev)
    is_harvest = ~is_tft
    start = torch.tensor([s.start for s in specs], dtype=torch.long,
                         device=dev)
    repairable = torch.tensor([s.repairable for s in specs],
                              dtype=torch.bool, device=dev)
    p_defect = torch.tensor([s.p_defect for s in specs],
                            dtype=torch.float32, device=dev)
    repair = torch.zeros(total, dtype=torch.long, device=dev)
    out = torch.zeros(total, rounds, dtype=torch.int8, device=dev)
    generator = torch.Generator(device=dev)
    generator.manual_seed(int(seed))
    torch.cuda.synchronize(dev) if dev.type == "cuda" else None
    import time

    started = time.perf_counter()
    for t in range(rounds):
        tft_action = torch.zeros(total, dtype=torch.int8, device=dev)
        if t > 0:
            tft_action = blue[:, t - 1].clone()
        harvest_action = torch.zeros(total, dtype=torch.int8, device=dev)
        if t == 99:
            harvest_action = torch.ones_like(harvest_action)
        elif t > 0:
            in_repair = repair > 0
            repair = torch.where(in_repair, repair - 1, repair)
            after_start = t >= start
            dd = (t >= 2) & (blue[:, t - 1] == 1) & (blue[:, t - 2] == 1)
            do_repair = repairable & after_start & dd & ~in_repair
            repair = torch.where(do_repair, torch.full_like(repair, 2), repair)
            rand = torch.rand(total, device=dev, generator=generator)
            defect = rand < p_defect
            action = torch.where(after_start, defect.to(torch.int8),
                                 torch.zeros_like(harvest_action))
            action = torch.where(in_repair | do_repair,
                                 torch.zeros_like(action), action)
            harvest_action = action
        out[:, t] = torch.where(is_tft, tft_action, harvest_action)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    elapsed = time.perf_counter() - started
    return out.detach().cpu(), elapsed


__all__ = ["SpikeSpec", "run_spike"]
