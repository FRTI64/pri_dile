#!/usr/bin/env python3
"""Expand an MLP checkpoint to wider hidden layers (net2wider).

The policy and value trunks are widened by duplicating hidden units and
dividing the corresponding outgoing weights.  A small amount of noise breaks
the duplicate symmetry so PPO can keep training the new units independently.
The initial function is therefore close to, but not bit-identical with, the
source checkpoint.

Examples::

    python3 -m ipd train expand \
      --input weights/rl_policy_checkpoint_pattern.json \
      --hidden 128 64 \
      --out results/rl_v4_20261029/wide_ac.json \
      --weights-out results/rl_v4_20261029/wide_weights.json
"""
from __future__ import annotations

if __package__ in (None, ""):  # allow: python ipd/training/<file>.py
    import sys
    from pathlib import Path

    for _parent in Path(__file__).resolve().parents:
        if (_parent / "pyproject.toml").is_file():
            if str(_parent) not in sys.path:
                sys.path.insert(0, str(_parent))
            break

import argparse
import copy
import json
import math
import random
from collections import Counter
from typing import Any, Sequence

from ipd.policies.neural import FEATURE_COUNT_V2, FEATURE_VERSION_V2


def _mapping(old_size: int, new_size: int) -> list[int]:
    """Map each new unit to an old unit, distributing duplicates evenly."""
    if new_size < old_size:
        raise ValueError("net2wider only expands hidden layers")
    base, extra = divmod(new_size, old_size)
    out: list[int] = []
    for old in range(old_size):
        out.extend([old] * (base + (1 if old < extra else 0)))
    return out


def _expand_rows(layer: dict[str, Any], new_size: int,
                 noise: float, rng: random.Random) -> tuple[dict[str, Any], list[int]]:
    """Duplicate rows of ``layer`` and add symmetry-breaking noise."""
    w = [[float(x) for x in row] for row in layer["w"]]
    b = [float(x) for x in layer["b"]]
    old_size = len(w)
    mapping = _mapping(old_size, new_size)
    seen: Counter[int] = Counter()
    new_w: list[list[float]] = []
    new_b: list[float] = []
    for old in mapping:
        row = list(w[old])
        bias = float(b[old])
        if seen[old]:
            scale = noise * (math.sqrt(sum(x * x for x in row) / max(1, len(row)))
                             + abs(bias) + 1e-6)
            row = [x + rng.gauss(0.0, scale) for x in row]
            bias += rng.gauss(0.0, scale)
        seen[old] += 1
        new_w.append(row)
        new_b.append(bias)
    return {"w": new_w, "b": new_b}, mapping


def _expand_input_columns(layer: dict[str, Any], mapping: list[int]
                          ) -> dict[str, Any]:
    """Adjust the next layer so duplicated hidden units keep their scale."""
    counts = Counter(mapping)
    w = layer["w"]
    return {
        "w": [[float(row[old]) / counts[old] for old in mapping]
              for row in w],
        "b": [float(x) for x in layer["b"]],
    }


def _expand_trunk(trunk: Sequence[dict[str, Any]], hidden: Sequence[int],
                  noise: float, rng: random.Random) -> list[dict[str, Any]]:
    if len(trunk) != len(hidden):
        raise ValueError(
            f"hidden depth mismatch: checkpoint={len(trunk)} requested={len(hidden)}")
    layers = [copy.deepcopy(layer) for layer in trunk]
    maps: list[list[int]] = []
    for index, size in enumerate(hidden):
        expanded, mapping = _expand_rows(layers[index], int(size), noise, rng)
        layers[index] = expanded
        maps.append(mapping)
        if index + 1 < len(layers):
            layers[index + 1] = _expand_input_columns(
                layers[index + 1], mapping)
    return layers


def _expand_head(head: dict[str, Any], mapping: list[int]) -> dict[str, Any]:
    counts = Counter(mapping)
    return {
        "w": [[float(row[old]) / counts[old] for old in mapping]
              for row in head["w"]],
        "b": [float(x) for x in head["b"]],
    }


def _pad_input_layer(layer: dict[str, Any], extra: int,
                     noise: float, rng: random.Random) -> dict[str, Any]:
    """Append ``extra`` input columns with small symmetry-breaking weights."""
    if extra <= 0:
        return copy.deepcopy(layer)
    w = [[float(x) for x in row] for row in layer["w"]]
    b = [float(x) for x in layer["b"]]
    for row in w:
        scale = noise * (math.sqrt(sum(x * x for x in row) / max(1, len(row)))
                         + 1e-6)
        row.extend(rng.gauss(0.0, scale) for _ in range(extra))
    return {"w": w, "b": b}


def expand_input_features(data: dict[str, Any], target: int | None = None,
                          seed: int = 20261029, noise: float = 0.01
                          ) -> dict[str, Any]:
    """Pad the first layer for a longer feature vector (v2 by default)."""
    target = int(FEATURE_COUNT_V2 if target is None else target)
    old = int(data["n_features"])
    extra = target - old
    if extra < 0:
        raise ValueError("input feature expansion cannot shrink the vector")
    if extra == 0:
        return copy.deepcopy(data)
    rng = random.Random(int(seed))
    out = copy.deepcopy(data)
    if data.get("format") == "ipd-rl-policy-v1":
        out["trunk"][0] = _pad_input_layer(
            data["trunk"][0], extra, noise, rng)
        if data.get("value_trunk"):
            out["value_trunk"][0] = _pad_input_layer(
                data["value_trunk"][0], extra, noise, rng)
    elif data.get("format") == "ipd-neural-policy-v1":
        out["layers"][0] = _pad_input_layer(
            data["layers"][0], extra, noise, rng)
    else:
        raise ValueError(
            f"unsupported checkpoint format: {data.get('format')!r}")
    out["n_features"] = target
    out["feature_version"] = FEATURE_VERSION_V2
    out["feature_mean"] = list(data["feature_mean"]) + [0.0] * extra
    out["feature_std"] = list(data["feature_std"]) + [1.0] * extra
    meta = dict(out.get("meta", {}))
    meta["expanded_from_features"] = old
    meta["expanded_to_features"] = target
    out["meta"] = meta
    return out


def expand_actor_critic(data: dict[str, Any], hidden: Sequence[int],
                        seed: int = 20261029, noise: float = 0.01
                        ) -> dict[str, Any]:
    """Expand an ``ipd-rl-policy-v1`` checkpoint to ``hidden`` widths."""
    if data.get("format") != "ipd-rl-policy-v1":
        raise ValueError("actor-critic expansion requires ipd-rl-policy-v1")
    rng = random.Random(int(seed))
    out = copy.deepcopy(data)
    out["trunk"] = _expand_trunk(data["trunk"], hidden, noise, rng)
    last_map = _mapping(len(data["trunk"][-1]["w"]), int(hidden[-1]))
    out["policy_head"] = _expand_head(data["policy_head"], last_map)
    if data.get("value_trunk"):
        out["value_trunk"] = _expand_trunk(
            data["value_trunk"], hidden, noise, rng)
        out["value_head"] = _expand_head(data["value_head"], last_map)
    meta = dict(out.get("meta", {}))
    meta["expanded_from_hidden"] = [len(layer["w"]) for layer in data["trunk"]]
    meta["expanded_to_hidden"] = [int(x) for x in hidden]
    meta["expansion"] = "net2wider"
    meta["expansion_noise"] = float(noise)
    out["meta"] = meta
    return out


def expand_deployable(data: dict[str, Any], hidden: Sequence[int],
                      seed: int = 20261029, noise: float = 0.01
                      ) -> dict[str, Any]:
    """Expand an ``ipd-neural-policy-v1`` checkpoint to ``hidden`` widths."""
    if data.get("format") != "ipd-neural-policy-v1":
        raise ValueError("deployable expansion requires ipd-neural-policy-v1")
    layers = data.get("layers") or []
    if len(layers) < 2:
        raise ValueError("deployable weights need a trunk and a head")
    rng = random.Random(int(seed))
    out = copy.deepcopy(data)
    trunk = [copy.deepcopy(layer) for layer in layers[:-1]]
    expanded = _expand_trunk(trunk, hidden, noise, rng)
    last_map = _mapping(len(trunk[-1]["w"]), int(hidden[-1]))
    out["layers"] = expanded + [_expand_head(layers[-1], last_map)]
    meta = dict(out.get("meta", {}))
    meta["expanded_from_hidden"] = [len(layer["w"]) for layer in trunk]
    meta["expanded_to_hidden"] = [int(x) for x in hidden]
    meta["expansion"] = "net2wider"
    meta["expansion_noise"] = float(noise)
    out["meta"] = meta
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Expand an MLP checkpoint to wider hidden layers.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--hidden", type=int, nargs="+", default=None,
                    help="new hidden widths; omit to keep the current sizes")
    ap.add_argument("--features", choices=("v1", "v2"), default="v1",
                    help="v2 pads the 85-feature input to 97 features")
    ap.add_argument("--out", required=True,
                    help="expanded actor-critic or deployable JSON")
    ap.add_argument("--weights-out", default=None,
                    help="optionally export deployable weights as well")
    ap.add_argument("--seed", type=int, default=20261029)
    ap.add_argument("--noise", type=float, default=0.01)
    args = ap.parse_args(argv)

    with open(args.input, encoding="utf-8") as fh:
        data = json.load(fh)
    fmt = data.get("format")
    if args.features == "v2":
        data = expand_input_features(
            data, seed=args.seed, noise=args.noise)

    if fmt == "ipd-rl-policy-v1":
        hidden = (args.hidden if args.hidden is not None
                  else [len(layer["w"]) for layer in data["trunk"]])
        expanded = expand_actor_critic(
            data, hidden, seed=args.seed, noise=args.noise)
        if args.weights_out:
            from ipd.policies.rl import RLPolicy
            import tempfile
            import os

            # RLPolicy needs the checkpoint on disk to apply its folding and
            # export path.  Use a temporary file so the CLI stays simple.
            fd, path = tempfile.mkstemp(suffix=".json")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(expanded, fh)
                policy = RLPolicy(path)
                weights = policy.export_policy_weights(name="RL-Wide")
            finally:
                os.unlink(path)
            with open(args.weights_out, "w", encoding="utf-8") as fh:
                json.dump(weights, fh, ensure_ascii=False, indent=2)
    elif fmt == "ipd-neural-policy-v1":
        hidden = (args.hidden if args.hidden is not None
                  else [len(layer["w"]) for layer in data["layers"][:-1]])
        expanded = expand_deployable(
            data, hidden, seed=args.seed, noise=args.noise)
        if args.weights_out:
            with open(args.weights_out, "w", encoding="utf-8") as fh:
                json.dump(expanded, fh, ensure_ascii=False, indent=2)
    else:
        raise SystemExit(f"unsupported checkpoint format: {fmt!r}")
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(expanded, fh, ensure_ascii=False, indent=2)
    print(f"wrote {args.out}")
    if args.weights_out:
        print(f"wrote {args.weights_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
