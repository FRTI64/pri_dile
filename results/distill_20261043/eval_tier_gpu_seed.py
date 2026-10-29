"""Protocol v4 tier evaluator with a configurable seed.

Thin wrapper around the GPU field-batch path used for the RL-Switch
promotion diagnostics.  It is only for experiments; official runs should go
through ``python3 -m ipd protocol eval``.

Usage::

    python3 results/distill_20261043/eval_tier_gpu_seed.py \
      results/distill_20261043/gru_dagger_ckpt.json dev_param GRU-DAgger \
      30 2048 gru 20261029
"""
import json
import sys

from ipd.evaluation.protocol_cli import _aggregate, _eval_field_batch_task
from ipd.policies.registry import CandidateSpec

path = sys.argv[1]
tier = sys.argv[2]
label = sys.argv[3] if len(sys.argv) > 3 else "candidate"
fields = int(sys.argv[4]) if len(sys.argv) > 4 else 30
n = int(sys.argv[5]) if len(sys.argv) > 5 else 2048
kind = sys.argv[6] if len(sys.argv) > 6 else "gru"
base_seed = int(sys.argv[7]) if len(sys.argv) > 7 else 20261027
mode = sys.argv[8] if len(sys.argv) > 8 else "sample"

seeds = [base_seed + i * 7919 for i in range(fields)]
gpu = {"device": "cuda", "dtype": "float32", "max_batch": 32768,
       "profile": False, "fast_rules": "auto", "fast_rules_threshold": 0.90,
       "mode": mode, "sample_rng": "per-match", "base_seed": base_seed}
rows = []
for start in range(0, len(seeds), 4):
    group = seeds[start:start + 4]
    spec = CandidateSpec(kind=kind, name=label,
                         params={"weights": path, "mode": mode})
    task = {"specs": [spec.to_dict()], "tier": tier, "seeds": group,
            "seed": group[0], "n": n, "rounds": 100, "field_reuse": "on",
            "gpu": gpu, "gpu_batch": True}
    rows.extend(_eval_field_batch_task(task)["results"])
print(json.dumps(_aggregate(rows)))
