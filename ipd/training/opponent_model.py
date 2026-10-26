"""P2 C2/C3: learned opponent model + confidence gate for the planner.

The P2 C1 failure was model discrimination: noisy/extort opponents were treated
as non-responsive exploiters.  This module trains a small linear classifier on
the existing 85-dim history features and exposes a gate that only permits
planner overrides for high-confidence `defect` / `late` / `random` opponents.

Training data comes from :data:`ipd.core.opponents.bank.TYPE_BANK` only.
`heldout_v2` is used for classification/evaluation, not training.

Entry point::

    python3 -m ipd train opponent-model --out weights/opponent_classifier.json
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from ipd.core.opponents.bank import TYPE_BANK
from ipd.core.primitives import C, D
from ipd.policies.neural import NeuralPolicy, extract_features
from ipd.paths import result_path, weights_path

CLASSES = (
    "adaptive", "cooperative", "defect", "forgiving", "late", "noisy",
    "punitive", "random", "wild",
)
NOISY_PREFIXES = ("NoisyTFT", "Joss")
ALLOW_CLASSES = ("defect", "late", "random")
EXPLORATION_TEMPLATES = (
    "single_probes", "block_probes", "random", "alternating",
    "late_block", "cooperate",
)
RESPONSE_CLASSES = ("unresponsive", "responsive")
THREE_WAY_CLASSES = ("safe", "risky", "unknown")
SUITE_FAMILY_MAP = {
    "reciprocal_v2": "cooperative",
    "punitive_v2": "punitive",
    "noisy_v2": "noisy",
    "extort_v2": "adaptive",
    "random_v2": "random",
    "temporal_v2": "late",
    "adaptive_v2": "adaptive",
    "anchor_v3": "cooperative",
}


def coarse_class(name: str, family: str) -> str:
    """Map an 85-type bank entry to a coarse gate class."""
    if family == "forgiving" and str(name).startswith(NOISY_PREFIXES):
        return "noisy"
    if family == "nice":
        return "cooperative"
    if family in CLASSES:
        return family
    return "adaptive"


def suite_class(family: str) -> str | None:
    """Map a held-out suite family (``*_v2``) to a gate class."""
    if family in SUITE_FAMILY_MAP:
        return SUITE_FAMILY_MAP[family]
    if family in CLASSES:
        return family
    return None


def allow_label(label: str) -> str:
    """Binary gate label: allow overrides for defect/late/random only."""
    return "allow" if label in ALLOW_CLASSES else "deny"


def three_way_gate_label(label: str,
                         features: Sequence[float]) -> str:
    """Map a family label plus the excitation indicator to a MoE route.

    ``features[35]`` is the 85-dim ``n_c >= 5 and n_d >= 3`` indicator.  With
    enough response evidence, allowed families become ``safe`` and the rest
    ``risky``; low-evidence states abstain to ``unknown``.
    """
    if len(features) <= 35 or float(features[35]) <= 0.0:
        return "unknown"
    return "safe" if allow_label(label) == "allow" else "risky"


def three_way_labels(labels: Sequence[str],
                     features: Sequence[Sequence[float]]) -> list[str]:
    return [three_way_gate_label(label, row)
            for label, row in zip(labels, features)]


def three_way_calibrated_label(probability: float,
                               safe_threshold: float = 0.0,
                               risky_threshold: float = 0.5) -> str:
    """Map a repeated-counterfactual response probability to a route."""
    value = float(probability)
    if value <= float(safe_threshold):
        return "safe"
    if value >= float(risky_threshold):
        return "risky"
    return "unknown"


def three_way_calibrated_labels(
    probabilities: Sequence[float], safe_threshold: float = 0.0,
    risky_threshold: float = 0.5,
) -> list[str]:
    return [three_way_calibrated_label(
        value, safe_threshold, risky_threshold) for value in probabilities]


def load_opponent_model(path: str | Path):
    """Load either the linear or MLP opponent model schema."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    schema = data.get("schema")
    if schema == "ipd-opponent-mlp-v1":
        return MLPOpponentModel.from_dict(data)
    if schema == "ipd-opponent-linear-model-v1":
        return LinearOpponentModel.from_dict(data)
    raise ValueError(f"unsupported opponent model schema {schema!r}")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclass
class Dataset:
    features: list[list[float]]
    labels: list[str]
    next_actions: list[int]
    names: list[str]

    def __len__(self) -> int:
        return len(self.features)


@dataclass
class ResponseDataset:
    """Counterfactual responsiveness samples.

    Each row is the history *before* a blue action, labelled by whether the
    opponent's subsequent behaviour differs between an enforced C and D branch.
    """

    features: list[list[float]]
    labels: list[str]
    truth: list[str]
    names: list[str]
    response_probabilities: list[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.features)


def _item_label(spec: Any, label_fn: Callable[[Any], str | None] | None
                ) -> str | None:
    if label_fn is not None:
        return label_fn(spec)
    return coarse_class(spec.name, spec.family)


def concat_datasets(*datasets: Dataset) -> Dataset:
    """Concatenate datasets while preserving order."""
    return Dataset(
        features=[row for data in datasets for row in data.features],
        labels=[value for data in datasets for value in data.labels],
        next_actions=[value for data in datasets
                      for value in data.next_actions],
        names=[value for data in datasets for value in data.names],
    )


def collect_dataset(specs: Iterable[Any], *, episodes: int = 2,
                    rounds: int = 100, seed: int = 20261024,
                    base_weights: str | None = None,
                    with_stats: bool = False
                    ) -> Dataset | tuple[Dataset, dict[str, Any]]:
    """Collect (history features, family label, next action) samples."""
    base = NeuralPolicy(base_weights or weights_path(
        "rl_policy_weights_repair.json"), seed=seed, mode="sample")
    features: list[list[float]] = []
    labels: list[str] = []
    next_actions: list[int] = []
    names: list[str] = []
    stats = _new_excitation_stats()
    specs = list(specs)
    for si, spec in enumerate(specs):
        label = coarse_class(spec.name, spec.family)
        for episode in range(max(1, int(episodes))):
            rng = random.Random(seed + si * 10007 + episode * 7919)
            opponent = spec.factory(rng)
            opponent.reset()
            base.reset()
            mine: list[int] = []
            theirs: list[int] = []
            for t in range(rounds):
                features.append(extract_features(mine, theirs))
                reply = int(opponent.act(t, theirs, mine))
                labels.append(label)
                next_actions.append(reply)
                names.append(spec.name)
                _record_excitation(stats, label, mine, reply)
                action = int(base.act(t, mine, theirs))
                mine.append(action)
                theirs.append(reply)
    dataset = Dataset(features, labels, next_actions, names)
    if with_stats:
        return dataset, _finalise_excitation_stats(stats)
    return dataset


# ---------------------------------------------------------------------------
# Exploratory / counterfactual data
# ---------------------------------------------------------------------------


def exploration_actions(rng: random.Random, rounds: int = 100,
                        template: str | None = None
                        ) -> tuple[list[int], str]:
    """Return an open-loop exploratory blue action schedule.

    The schedule is independent of the opponent, so the resulting histories are
    not biased by the base policy's own policy.  Templates deliberately create
    repeated C/D opportunities, which is what the 85-dim conditional-response
    features need in order to estimate ``p(D | our C)`` and ``p(D | our D)``.
    """
    template = template or rng.choice(EXPLORATION_TEMPLATES)
    if template not in EXPLORATION_TEMPLATES:
        raise ValueError(f"unknown exploration template {template!r}")
    rounds = max(1, int(rounds))
    actions = [C] * rounds
    if rounds < 3:
        return actions, template

    if template == "single_probes":
        target = max(1, rounds // 10)
        t = rng.randint(1, min(3, rounds - 2))
        probes = 0
        while t < rounds - 1 and probes < target:
            actions[t] = D
            probes += 1
            t += rng.randint(3, 8)
    elif template == "block_probes":
        t = rng.randint(1, min(3, rounds - 2))
        while t < rounds - 2:
            length = rng.randint(2, 4)
            for u in range(t, min(rounds - 1, t + length)):
                actions[u] = D
            t += length + rng.randint(2, 6)
    elif template == "random":
        p = rng.uniform(0.08, 0.35)
        for t in range(1, rounds - 1):
            if rng.random() < p:
                actions[t] = D
    elif template == "alternating":
        start = rng.randint(1, max(1, rounds // 4))
        end = min(rounds - 1, start + rng.randint(
            max(3, rounds // 8), max(4, rounds // 3)))
        for t in range(start, end):
            actions[t] = D if (t - start) % 2 == 0 else C
    elif template == "late_block":
        start = rng.randint(max(1, int(rounds * 0.55)), max(1, rounds - 3))
        end = min(rounds - 1, start + rng.randint(
            3, max(4, rounds // 10)))
        for t in range(start, end):
            actions[t] = D
    # "cooperate" intentionally leaves the all-C schedule untouched.
    return actions, template


def _new_excitation_stats() -> dict[str, Any]:
    return {
        "samples": 0,
        "excited_samples": 0,
        "defection_opportunities": 0,
        "defection_responses": 0,
        "sum_n_c": 0.0,
        "sum_n_d": 0.0,
        "by_class": {},
    }


def _record_excitation(stats: dict[str, Any], label: str, mine: Sequence[int],
                       reply: int) -> None:
    t = len(mine)
    n_d = sum(int(x) for x in mine)
    n_c = t - n_d
    excited = n_d >= 3 and n_c >= 5
    prev_d = bool(t and mine[-1] == D)
    stats["samples"] += 1
    stats["excited_samples"] += int(excited)
    stats["sum_n_c"] += n_c
    stats["sum_n_d"] += n_d
    stats["defection_opportunities"] += int(prev_d)
    stats["defection_responses"] += int(prev_d and reply == D)
    row = stats["by_class"].setdefault(label, {
        "samples": 0, "excited_samples": 0,
        "defection_opportunities": 0, "defection_responses": 0,
        "sum_n_c": 0.0, "sum_n_d": 0.0,
        "pairs": 0, "responsive": 0, "unresponsive": 0,
        "sum_abs_delta": 0.0, "sum_response_probability": 0.0,
    })
    row["samples"] += 1
    row["excited_samples"] += int(excited)
    row["sum_n_c"] += n_c
    row["sum_n_d"] += n_d
    row["defection_opportunities"] += int(prev_d)
    row["defection_responses"] += int(prev_d and reply == D)


def _finalise_excitation_stats(stats: dict[str, Any]) -> dict[str, Any]:
    samples = max(1, int(stats["samples"]))
    stats["excited_rate"] = stats["excited_samples"] / samples
    stats["mean_n_c"] = stats["sum_n_c"] / samples
    stats["mean_n_d"] = stats["sum_n_d"] / samples
    stats["response_rate_given_d"] = (
        stats["defection_responses"] / stats["defection_opportunities"]
        if stats["defection_opportunities"] else 0.0)
    for row in stats["by_class"].values():
        n = max(1, int(row["samples"]))
        row["excited_rate"] = row["excited_samples"] / n
        row["mean_n_c"] = row["sum_n_c"] / n
        row["mean_n_d"] = row["sum_n_d"] / n
        row["response_rate_given_d"] = (
            row["defection_responses"] / row["defection_opportunities"]
            if row["defection_opportunities"] else 0.0)
    return stats


def collect_exploratory_dataset(
    specs: Iterable[Any], *, episodes: int = 2, rounds: int = 100,
    seed: int = 20261024,
    templates: Sequence[str] = EXPLORATION_TEMPLATES,
    label_fn: Callable[[Any], str | None] | None = None,
) -> tuple[Dataset, dict[str, Any]]:
    """Collect trajectories from open-loop exploratory blue schedules."""
    features: list[list[float]] = []
    labels: list[str] = []
    next_actions: list[int] = []
    names: list[str] = []
    stats = _new_excitation_stats()
    stats["templates"] = list(templates)
    templates = tuple(templates) or EXPLORATION_TEMPLATES
    specs = list(specs)
    for si, spec in enumerate(specs):
        label = _item_label(spec, label_fn)
        if label is None:
            continue
        for episode in range(max(1, int(episodes))):
            schedule_rng = random.Random(seed + si * 10007 + episode * 7919)
            opponent_rng = random.Random(
                seed + si * 10007 + episode * 7919 + 1)
            template = templates[(si + episode) % len(templates)]
            actions, template = exploration_actions(
                schedule_rng, rounds, template)
            opponent = spec.factory(opponent_rng)
            opponent.reset()
            mine: list[int] = []
            theirs: list[int] = []
            for t in range(rounds):
                features.append(extract_features(mine, theirs))
                reply = int(opponent.act(t, theirs, mine))
                labels.append(label)
                next_actions.append(reply)
                names.append(spec.name)
                _record_excitation(stats, label, mine, reply)
                mine.append(int(actions[t]))
                theirs.append(reply)
    return Dataset(features, labels, next_actions, names), \
        _finalise_excitation_stats(stats)


def _select_branch_days(rng: random.Random, rounds: int,
                        count: int) -> list[int]:
    """Stratified branch days in ``[2, rounds - 2]``."""
    lo, hi = 2, max(2, rounds - 1)
    if hi <= lo or count <= 0:
        return []
    count = min(int(count), hi - lo)
    buckets = 5
    edges = [lo + (hi - lo) * i // buckets for i in range(buckets + 1)]
    days: list[int] = []
    for i in range(buckets):
        a, b = max(lo, edges[i]), min(hi, edges[i + 1])
        if b <= a:
            continue
        take = count // buckets + int(i < count % buckets)
        take = min(take, b - a)
        days.extend(rng.sample(range(a, b), take))
    if len(days) < count:
        chosen = set(days)
        remaining = [d for d in range(lo, hi) if d not in chosen]
        days.extend(rng.sample(remaining, min(count - len(days),
                                               len(remaining))))
    return sorted(days)


def collect_counterfactual_dataset(
    specs: Iterable[Any], *, episodes: int = 1, rounds: int = 100,
    seed: int = 20261024, branches: int = 8, horizon: int = 6,
    repeats: int = 1, reseed_branches: bool = False,
    prefix_modes: Sequence[str] = ("explore", "base"),
    include_prefix: bool = True, base_weights: str | None = None,
    label_fn: Callable[[Any], str | None] | None = None,
) -> tuple[Dataset, ResponseDataset, dict[str, Any]]:
    """Collect paired C/D branches from identical opponent states.

    For every branch day ``t`` the opponent is deep-copied, then blue plays C
    or D at ``t`` and cooperates for the next ``horizon`` rounds.  The pair
    shares the opponent's RNG stream, so the difference in the reply sequence
    is caused by blue's action rather than by a different random draw.

    ``repeats > 1`` reseeds the common opponent RNG for each repeat, producing
    an independent estimate of the response probability at the same state.
    """
    features: list[list[float]] = []
    labels: list[str] = []
    next_actions: list[int] = []
    names: list[str] = []
    response_features: list[list[float]] = []
    response_labels: list[str] = []
    response_truth: list[str] = []
    response_names: list[str] = []
    response_probabilities: list[float] = []
    stats = _new_excitation_stats()
    stats.update({
        "prefix_modes": list(prefix_modes),
        "branch_pairs": 0,
        "responsive_pairs": 0,
        "unresponsive_pairs": 0,
        "repeats": max(1, int(repeats)),
        "by_class": {},
    })
    prefix_modes = tuple(prefix_modes) or ("explore",)
    specs = list(specs)
    for si, spec in enumerate(specs):
        label = _item_label(spec, label_fn)
        if label is None:
            continue
        for episode in range(max(1, int(episodes))):
            episode_seed = seed + si * 10007 + episode * 7919
            schedule_rng = random.Random(episode_seed)
            opponent_rng = random.Random(episode_seed + 1)
            prefix_mode = prefix_modes[(si + episode) % len(prefix_modes)]
            if prefix_mode == "explore":
                prefix_actions, _ = exploration_actions(schedule_rng, rounds)
                base = None
            elif prefix_mode == "base":
                prefix_actions = []
                base = NeuralPolicy(
                    base_weights or weights_path(
                        "rl_policy_weights_repair.json"),
                    seed=episode_seed + 2, mode="sample")
                base.reset()
            else:
                raise ValueError(f"unknown counterfactual prefix {prefix_mode!r}")
            branch_days = _select_branch_days(
                schedule_rng, rounds, branches)
            branch_set = set(branch_days)
            opponent = spec.factory(opponent_rng)
            opponent.reset()
            mine: list[int] = []
            theirs: list[int] = []
            for t in range(rounds):
                if t in branch_set:
                    repeat_count = max(1, int(repeats))
                    deltas: list[int] = []
                    for repeat in range(repeat_count):
                        opp_c = copy.deepcopy(opponent)
                        opp_d = copy.deepcopy(opponent)
                        if repeat_count > 1 or reseed_branches:
                            branch_seed = (episode_seed + t * 131
                                           + repeat * 17 + 999)
                            if hasattr(opp_c, "rng"):
                                opp_c.rng = random.Random(branch_seed)
                            if hasattr(opp_d, "rng"):
                                opp_d.rng = random.Random(branch_seed)
                        reply_c = int(opp_c.act(t, theirs, mine))
                        reply_d = int(opp_d.act(t, theirs, mine))
                        branch_replies: dict[int, list[int]] = {}
                        for branch_action, branch_opp, branch_reply in (
                                (C, opp_c, reply_c), (D, opp_d, reply_d)):
                            bm = mine + [branch_action]
                            bt = theirs + [branch_reply]
                            replies: list[int] = []
                            for u in range(t + 1,
                                           min(rounds, t + 1 + horizon)):
                                feature = extract_features(bm, bt)
                                reply = int(branch_opp.act(u, bt, bm))
                                if repeat == 0:
                                    features.append(feature)
                                    labels.append(label)
                                    next_actions.append(reply)
                                    names.append(spec.name)
                                    _record_excitation(stats, label, bm, reply)
                                replies.append(reply)
                                bm.append(C)
                                bt.append(reply)
                            branch_replies[branch_action] = replies
                        deltas.append(sum(branch_replies[D])
                                      - sum(branch_replies[C]))
                    response_probability = (
                        sum(1 for delta in deltas if delta != 0)
                        / len(deltas))
                    responsive = response_probability >= 0.5
                    stats["branch_pairs"] += 1
                    stats["responsive_pairs"] += int(responsive)
                    stats["unresponsive_pairs"] += int(not responsive)
                    response_features.append(
                        extract_features(mine, theirs))
                    response_labels.append(
                        "responsive" if responsive else "unresponsive")
                    response_truth.append(label)
                    response_names.append(spec.name)
                    response_probabilities.append(response_probability)
                    row = stats["by_class"].setdefault(label, {
                        "pairs": 0, "responsive": 0,
                        "unresponsive": 0, "sum_abs_delta": 0.0,
                        "sum_response_probability": 0.0,
                        "samples": 0, "excited_samples": 0,
                        "defection_opportunities": 0,
                        "defection_responses": 0,
                        "sum_n_c": 0.0, "sum_n_d": 0.0,
                    })
                    row["pairs"] += 1
                    row["responsive"] += int(responsive)
                    row["unresponsive"] += int(not responsive)
                    row["sum_abs_delta"] += (
                        statistics.fmean(abs(delta) for delta in deltas))
                    row["sum_response_probability"] += response_probability
                if prefix_mode == "base":
                    assert base is not None
                    normal_action = int(base.act(t, mine, theirs))
                else:
                    normal_action = int(prefix_actions[t])
                reply = int(opponent.act(t, theirs, mine))
                if include_prefix:
                    features.append(extract_features(mine, theirs))
                    labels.append(label)
                    next_actions.append(reply)
                    names.append(spec.name)
                    _record_excitation(stats, label, mine, reply)
                mine.append(normal_action)
                theirs.append(reply)
    for row in stats["by_class"].values():
        row["responsive_rate"] = (row["responsive"] / row["pairs"]
                                 if row.get("pairs") else 0.0)
        row["mean_abs_delta"] = (row["sum_abs_delta"] / row["pairs"]
                                if row.get("pairs") else 0.0)
        row["mean_response_probability"] = (
            row["sum_response_probability"] / row["pairs"]
            if row.get("pairs") else 0.0)
    stats["mean_response_probability"] = (
        statistics.fmean(response_probabilities)
        if response_probabilities else 0.0)
    _finalise_excitation_stats(stats)
    return (Dataset(features, labels, next_actions, names),
            ResponseDataset(response_features, response_labels,
                            response_truth, response_names,
                            response_probabilities), stats)


def split_specs(specs: Sequence[Any], validation_fraction: float = 0.2,
                seed: int = 20261024) -> tuple[list[Any], list[Any]]:
    """Hold out whole opponent types, stratified by coarse class."""
    rng = random.Random(seed)
    by_class: dict[str, list[Any]] = {}
    for spec in specs:
        by_class.setdefault(coarse_class(spec.name, spec.family),
                            []).append(spec)
    train: list[Any] = []
    validation: list[Any] = []
    for label in sorted(by_class):
        values = list(by_class[label])
        rng.shuffle(values)
        n_val = max(1, int(round(len(values) * validation_fraction)))
        n_val = min(n_val, max(0, len(values) - 1))
        validation.extend(values[:n_val])
        train.extend(values[n_val:])
    return train, validation


# ---------------------------------------------------------------------------
# Linear softmax model (pure-Python inference, numpy training)
# ---------------------------------------------------------------------------


@dataclass
class LinearOpponentModel:
    classes: list[str]
    mean: list[float]
    std: list[float]
    weights: list[list[float]]
    bias: list[float]

    def probabilities(self, features: Sequence[float]) -> list[float]:
        z = [(float(x) - m) / s
             for x, m, s in zip(features, self.mean, self.std)]
        logits = [
            b + sum(w * value for w, value in zip(row, z))
            for row, b in zip(self.weights, self.bias)
        ]
        top = max(logits)
        exps = [math.exp(value - top) for value in logits]
        total = sum(exps) or 1.0
        return [value / total for value in exps]

    def predict(self, features: Sequence[float]) -> tuple[str, float]:
        probs = self.probabilities(features)
        index = max(range(len(probs)), key=lambda i: probs[i])
        return self.classes[index], probs[index]

    def predict_next_action(self, features: Sequence[float]) -> int:
        # A binary next-action model uses classes ["C", "D"].
        label, _ = self.predict(features)
        return C if label == "C" else D

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "ipd-opponent-linear-model-v1",
            "classes": list(self.classes),
            "mean": list(self.mean),
            "std": list(self.std),
            "weights": [list(row) for row in self.weights],
            "bias": list(self.bias),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LinearOpponentModel":
        if data.get("schema") != "ipd-opponent-linear-model-v1":
            raise ValueError("unsupported opponent model schema")
        return cls(
            classes=[str(x) for x in data["classes"]],
            mean=[float(x) for x in data["mean"]],
            std=[float(x) for x in data["std"]],
            weights=[[float(x) for x in row] for row in data["weights"]],
            bias=[float(x) for x in data["bias"]],
        )

    @classmethod
    def load(cls, path: str | Path) -> "LinearOpponentModel":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def save(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), ensure_ascii=False,
                                  indent=2), encoding="utf-8")


@dataclass
class MLPOpponentModel:
    """Small tanh MLP with pure-Python inference."""

    classes: list[str]
    mean: list[float]
    std: list[float]
    layers: list[tuple[list[list[float]], list[float]]]
    activation: str = "tanh"

    def probabilities(self, features: Sequence[float]) -> list[float]:
        z = [(float(x) - m) / s
             for x, m, s in zip(features, self.mean, self.std)]
        last = len(self.layers) - 1
        for index, (weights, bias) in enumerate(self.layers):
            z = [b + sum(w * value for w, value in zip(row, z))
                 for row, b in zip(weights, bias)]
            if index != last:
                if self.activation == "tanh":
                    z = [math.tanh(value) for value in z]
                else:
                    z = [max(0.0, value) for value in z]
        top = max(z)
        exps = [math.exp(value - top) for value in z]
        total = sum(exps) or 1.0
        return [value / total for value in exps]

    def predict(self, features: Sequence[float]) -> tuple[str, float]:
        probs = self.probabilities(features)
        index = max(range(len(probs)), key=lambda i: probs[i])
        return self.classes[index], probs[index]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "ipd-opponent-mlp-v1",
            "classes": list(self.classes),
            "mean": list(self.mean),
            "std": list(self.std),
            "activation": self.activation,
            "layers": [
                {"w": [list(row) for row in weights],
                 "b": list(bias)}
                for weights, bias in self.layers
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MLPOpponentModel":
        if data.get("schema") != "ipd-opponent-mlp-v1":
            raise ValueError("unsupported opponent MLP schema")
        return cls(
            classes=[str(x) for x in data["classes"]],
            mean=[float(x) for x in data["mean"]],
            std=[float(x) for x in data["std"]],
            activation=str(data.get("activation", "tanh")),
            layers=[
                ([[float(x) for x in row] for row in layer["w"]],
                 [float(x) for x in layer["b"]])
                for layer in data["layers"]
            ],
        )

    @classmethod
    def load(cls, path: str | Path) -> "MLPOpponentModel":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def save(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), ensure_ascii=False,
                                  indent=2), encoding="utf-8")


def _standardise(features: Sequence[Sequence[float]]) -> tuple[list[float],
                                                               list[float]]:
    n = len(features)
    d = len(features[0]) if n else 0
    mean = [sum(row[j] for row in features) / n for j in range(d)]
    std = []
    for j in range(d):
        var = sum((row[j] - mean[j]) ** 2 for row in features) / max(1, n)
        std.append(math.sqrt(var) if var > 1e-12 else 1.0)
    return mean, std


def train_softmax(features: Sequence[Sequence[float]],
                  labels: Sequence[str], *, classes: Sequence[str] | None = None,
                  epochs: int = 300, lr: float = 0.1, l2: float = 1e-3,
                  seed: int = 20261024,
                  validation: tuple[Sequence[Sequence[float]],
                                    Sequence[str]] | None = None
                  ) -> tuple[LinearOpponentModel, dict[str, Any]]:
    """Train a class-balanced linear softmax classifier with Adam."""
    import numpy as np

    classes = list(classes or sorted(set(labels)))
    class_index = {name: i for i, name in enumerate(classes)}
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray([class_index[name] for name in labels], dtype=np.int64)
    mean, std = _standardise(features)
    mean_np = np.asarray(mean)
    std_np = np.asarray(std)
    x = (x - mean_np) / std_np
    counts = np.bincount(y, minlength=len(classes)).astype(np.float64)
    class_weights = len(y) / (len(classes) * np.maximum(1.0, counts))
    sample_weights = class_weights[y]
    one_hot = np.eye(len(classes))[y]
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 0.01, size=(x.shape[1], len(classes)))
    b = np.zeros(len(classes))
    m_w = np.zeros_like(w); v_w = np.zeros_like(w)
    m_b = np.zeros_like(b); v_b = np.zeros_like(b)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    best = {"epoch": 0, "val_accuracy": None, "train_accuracy": None}
    best_state = (w.copy(), b.copy())
    for epoch in range(1, max(1, int(epochs)) + 1):
        logits = x @ w + b
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        loss = -np.mean(sample_weights * np.log(
            np.maximum(1e-12, probs[np.arange(len(y)), y])))
        loss += 0.5 * l2 * float(np.sum(w * w))
        grad = (probs - one_hot) * sample_weights[:, None] / len(y)
        grad_w = x.T @ grad + l2 * w
        grad_b = grad.sum(axis=0)
        for param, g, m, v in ((w, grad_w, m_w, v_w),
                               (b, grad_b, m_b, v_b)):
            m *= beta1
            m += (1 - beta1) * g
            v *= beta2
            v += (1 - beta2) * g * g
            m_hat = m / (1 - beta1 ** epoch)
            v_hat = v / (1 - beta2 ** epoch)
            param -= lr * m_hat / (np.sqrt(v_hat) + eps)
        if epoch % 10 == 0 or epoch == epochs:
            model = LinearOpponentModel(
                classes=classes, mean=mean, std=std,
                weights=w.tolist(), bias=b.tolist())
            train_acc = _accuracy_np(probs, y)
            val_acc = None
            if validation is not None:
                val_acc = accuracy(model, validation[0], validation[1])
            if val_acc is None or best["val_accuracy"] is None \
                    or val_acc >= best["val_accuracy"]:
                best = {"epoch": epoch, "val_accuracy": val_acc,
                        "train_accuracy": train_acc}
                best_state = (w.copy(), b.copy())
    w, b = best_state
    model = LinearOpponentModel(
        classes=classes, mean=mean, std=std,
        weights=w.tolist(), bias=b.tolist())
    report = {
        **best,
        "n": int(len(y)),
        "classes": classes,
        "class_counts": {name: int(counts[i])
                         for i, name in enumerate(classes)},
        "train_accuracy": accuracy(model, features, labels),
        "loss": float(loss),
    }
    if validation is not None:
        report["val_accuracy"] = accuracy(model, validation[0],
                                          validation[1])
    return model, report


def train_mlp(features: Sequence[Sequence[float]],
              labels: Sequence[str], *, classes: Sequence[str] | None = None,
              hidden: Sequence[int] = (64, 32), epochs: int = 300,
              lr: float = 1e-3, l2: float = 1e-4, seed: int = 20261024,
              validation: tuple[Sequence[Sequence[float]],
                                Sequence[str]] | None = None
              ) -> tuple[MLPOpponentModel, dict[str, Any]]:
    """Train a small tanh MLP with PyTorch and export pure-Python weights."""
    import numpy as np
    import torch
    from torch import nn

    classes = list(classes or sorted(set(labels)))
    class_index = {name: i for i, name in enumerate(classes)}
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray([class_index[name] for name in labels], dtype=np.int64)
    mean, std = _standardise(features)
    mean_np = np.asarray(mean)
    std_np = np.asarray(std)
    x = (x - mean_np) / std_np
    counts = np.bincount(y, minlength=len(classes)).astype(np.float64)
    class_weights = len(y) / (len(classes) * np.maximum(1.0, counts))
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    sizes = [x.shape[1], *[int(v) for v in hidden], len(classes)]
    modules: list[Any] = []
    for i in range(len(sizes) - 1):
        modules.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            modules.append(nn.Tanh())
    net = nn.Sequential(*modules)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=l2)
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32))
    xt = torch.tensor(x, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.int64)
    xv = yv = None
    if validation is not None:
        xv = torch.tensor(
            (np.asarray(validation[0], dtype=np.float64) - mean_np) / std_np,
            dtype=torch.float32)
        yv = torch.tensor([class_index[name] for name in validation[1]],
                          dtype=torch.int64)
    best_acc = -1.0
    best_epoch = 0
    best_state = None
    last_loss = 0.0
    for epoch in range(1, max(1, int(epochs)) + 1):
        net.train()
        optimizer.zero_grad()
        logits = net(xt)
        loss = loss_fn(logits, yt)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach())
        if epoch % 10 == 0 or epoch == epochs:
            net.eval()
            with torch.no_grad():
                train_pred = net(xt).argmax(dim=1)
                train_acc = float((train_pred == yt).float().mean())
                val_acc = None
                if xv is not None and yv is not None:
                    val_pred = net(xv).argmax(dim=1)
                    val_acc = float((val_pred == yv).float().mean())
            score = val_acc if val_acc is not None else train_acc
            if score >= best_acc:
                best_acc = score
                best_epoch = epoch
                best_state = {k: v.detach().clone()
                              for k, v in net.state_dict().items()}
                best_train_acc = train_acc
                best_val_acc = val_acc
    if best_state is not None:
        net.load_state_dict(best_state)
    layers = []
    for module in net:
        if isinstance(module, nn.Linear):
            layers.append((
                module.weight.detach().cpu().tolist(),
                module.bias.detach().cpu().tolist(),
            ))
    model = MLPOpponentModel(
        classes=classes, mean=mean, std=std, layers=layers)
    report = {
        "epoch": best_epoch,
        "n": int(len(y)),
        "classes": classes,
        "class_counts": {name: int(counts[i])
                         for i, name in enumerate(classes)},
        "train_accuracy": accuracy(model, features, labels),
        "loss": last_loss,
    }
    if validation is not None:
        report["val_accuracy"] = accuracy(model, validation[0],
                                          validation[1])
    return model, report


def _accuracy_np(probs, y) -> float:
    import numpy as np

    return float(np.mean(np.argmax(probs, axis=1) == y))


def accuracy(model, features: Sequence[Sequence[float]],
             labels: Sequence[str]) -> float:
    if not labels:
        return 0.0
    correct = sum(1 for f, label in zip(features, labels)
                  if model.predict(f)[0] == label)
    return correct / len(labels)


def per_class_accuracy(model: LinearOpponentModel,
                       features: Sequence[Sequence[float]],
                       labels: Sequence[str]) -> dict[str, Any]:
    by_class: dict[str, list[int]] = {}
    confusion: dict[str, dict[str, int]] = {}
    for f, truth in zip(features, labels):
        pred, _ = model.predict(f)
        by_class.setdefault(truth, [0, 0])
        by_class[truth][1] += 1
        by_class[truth][0] += int(pred == truth)
        confusion.setdefault(truth, {})
        confusion[truth][pred] = confusion[truth].get(pred, 0) + 1
    return {
        "accuracy": {name: correct / total for name, (correct, total)
                     in sorted(by_class.items())},
        "counts": {name: total for name, (_, total)
                   in sorted(by_class.items())},
        "confusion": {name: dict(sorted(row.items()))
                      for name, row in sorted(confusion.items())},
    }


# ---------------------------------------------------------------------------
# Confidence gate
# ---------------------------------------------------------------------------


class OpponentGate:
    """Allow planner overrides only for confident, safe opponent classes."""

    def __init__(self, model: LinearOpponentModel, *,
                 allowed: Sequence[str] = ("defect", "late", "random"),
                 threshold: float = 0.6, min_day: int = 3,
                 require_next_d: bool = False,
                 next_model: LinearOpponentModel | None = None,
                 next_threshold: float = 0.6,
                 response_model: LinearOpponentModel | None = None,
                 max_response_probability: float = 0.5):
        self.model = model
        self.allowed = set(allowed)
        self.threshold = float(threshold)
        self.min_day = int(min_day)
        self.require_next_d = bool(require_next_d)
        self.next_model = next_model
        self.next_threshold = float(next_threshold)
        self.response_model = response_model
        self.max_response_probability = float(max_response_probability)
        self.calls = 0
        self.allowed_calls = 0

    def reset(self) -> None:
        self.calls = 0
        self.allowed_calls = 0

    def __call__(self, t: int, mine: Sequence[int],
                 theirs: Sequence[int]) -> tuple[bool, dict[str, Any]]:
        self.calls += 1
        if t < self.min_day:
            return False, {"reason": "warmup", "label": "", "confidence": 0.0}
        features = extract_features(mine, theirs)
        label, confidence = self.model.predict(features)
        info = {"label": label, "confidence": confidence,
                "next_probability_d": None, "response_probability": None}
        allowed = label in self.allowed and confidence >= self.threshold
        if allowed and self.require_next_d and self.next_model is not None:
            p_d = self.next_model.probabilities(features)[
                self.next_model.classes.index("D")]
            info["next_probability_d"] = p_d
            allowed = p_d >= self.next_threshold
        if allowed and self.response_model is not None:
            probs = self.response_model.probabilities(features)
            p_resp = probs[self.response_model.classes.index("responsive")]
            info["response_probability"] = p_resp
            allowed = p_resp <= self.max_response_probability
        if allowed:
            self.allowed_calls += 1
        return allowed, info

    def diagnostics(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "allowed_calls": self.allowed_calls,
            "allow_rate": (self.allowed_calls / self.calls
                           if self.calls else 0.0),
        }


class ThreeWayGate:
    """Route states to ``safe`` / ``risky`` / ``unknown``.

    The wrapped model is a 3-class MLP.  ``safe`` requires the family label and
    the excitation evidence to agree; low confidence abstains to ``unknown``.
    An optional C2 response model can veto a ``safe`` route and turn it into
    ``risky`` when the opponent is predicted to react to our action.
    """

    def __init__(self, model, *, safe_threshold: float = 0.6,
                 risky_threshold: float = 0.6, min_day: int = 3,
                 response_model=None,
                 max_response_probability: float | None = None,
                 fallback: str = "unknown"):
        if fallback not in THREE_WAY_CLASSES:
            raise ValueError(f"invalid fallback route {fallback!r}")
        self.model = model
        self.safe_threshold = float(safe_threshold)
        self.risky_threshold = float(risky_threshold)
        self.min_day = int(min_day)
        self.response_model = response_model
        self.max_response_probability = (
            None if max_response_probability is None
            else float(max_response_probability))
        self.fallback = fallback
        self.calls = 0
        self.safe_calls = 0
        self.risky_calls = 0
        self.unknown_calls = 0

    def reset(self) -> None:
        self.calls = 0
        self.safe_calls = 0
        self.risky_calls = 0
        self.unknown_calls = 0

    def __call__(self, t: int, mine: Sequence[int],
                 theirs: Sequence[int]) -> tuple[str, dict[str, Any]]:
        self.calls += 1
        if t < self.min_day:
            self.unknown_calls += 1
            return self.fallback, {
                "reason": "warmup", "label": "", "confidence": 0.0,
                "response_probability": None,
            }
        features = extract_features(mine, theirs)
        label, confidence = self.model.predict(features)
        info: dict[str, Any] = {
            "label": label, "confidence": confidence,
            "response_probability": None,
        }
        route = label if label in ("safe", "risky") else "unknown"
        if route == "safe" and confidence < self.safe_threshold:
            route = "unknown"
        if route == "risky" and confidence < self.risky_threshold:
            route = "unknown"
        if route == "safe" and self.response_model is not None \
                and self.max_response_probability is not None:
            probs = self.response_model.probabilities(features)
            p_resp = probs[self.response_model.classes.index("responsive")]
            info["response_probability"] = p_resp
            if p_resp > self.max_response_probability:
                route = "risky"
        if route == "safe":
            self.safe_calls += 1
        elif route == "risky":
            self.risky_calls += 1
        else:
            self.unknown_calls += 1
        info["route"] = route
        return route, info

    def diagnostics(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "safe_calls": self.safe_calls,
            "risky_calls": self.risky_calls,
            "unknown_calls": self.unknown_calls,
            "safe_rate": (self.safe_calls / self.calls
                          if self.calls else 0.0),
            "risky_rate": (self.risky_calls / self.calls
                           if self.calls else 0.0),
            "unknown_rate": (self.unknown_calls / self.calls
                             if self.calls else 0.0),
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


DATA_MODES = (
    "base", "explore", "counterfactual", "base+explore",
    "base+counterfactual", "mixed",
)
EVAL_MODES = ("base", "explore", "counterfactual", "both", "all")


def _collect_training_data(specs: Sequence[Any], args: argparse.Namespace,
                           *, seed: int, validation: bool
                           ) -> tuple[Dataset, dict[str, Any],
                                      ResponseDataset | None]:
    datasets: list[Dataset] = []
    stats: dict[str, Any] = {}
    response: ResponseDataset | None = None
    episodes = args.episodes if not validation else max(1, args.episodes)
    if args.data_mode in ("base", "base+explore", "base+counterfactual",
                          "mixed"):
        data, data_stats = collect_dataset(
            specs, episodes=episodes, rounds=args.rounds, seed=seed,
            with_stats=True)
        datasets.append(data)
        stats["base"] = data_stats
    if args.data_mode in ("explore", "base+explore", "mixed"):
        data, data_stats = collect_exploratory_dataset(
            specs, episodes=args.explore_episodes, rounds=args.rounds,
            seed=seed)
        datasets.append(data)
        stats["exploration"] = data_stats
    if args.data_mode in ("counterfactual", "base+counterfactual", "mixed"):
        data, response, data_stats = collect_counterfactual_dataset(
            specs, episodes=args.counterfactual_episodes,
            rounds=args.rounds, seed=seed,
            branches=args.counterfactual_branches,
            horizon=args.counterfactual_horizon,
            repeats=args.counterfactual_repeats,
            prefix_modes=tuple(
                x.strip() for x in args.counterfactual_prefix.split(",")
                if x.strip()))
        datasets.append(data)
        stats["counterfactual"] = data_stats
    if not datasets:
        raise ValueError(f"no data collected for mode {args.data_mode!r}")
    return concat_datasets(*datasets), stats, response


def _collect_suite_data(items: Sequence[Any], mode: str,
                        args: argparse.Namespace, *, seed: int
                        ) -> dict[str, Any]:
    label_fn = lambda item: suite_class(item.family)
    if mode == "base":
        data = _collect_suite_dataset(
            items, episodes=args.eval_episodes, rounds=args.rounds,
            seed=seed)
        return {**data, "stats": None, "response": None}
    if mode == "explore":
        data, stats = collect_exploratory_dataset(
            items, episodes=args.eval_episodes, rounds=args.rounds,
            seed=seed, label_fn=label_fn)
        response = None
    elif mode == "counterfactual":
        data, response_data, stats = collect_counterfactual_dataset(
            items, episodes=args.eval_episodes, rounds=args.rounds,
            seed=seed, branches=args.counterfactual_branches,
            horizon=args.counterfactual_horizon,
            repeats=args.counterfactual_repeats,
            prefix_modes=tuple(
                x.strip() for x in args.counterfactual_prefix.split(",")
                if x.strip()), label_fn=label_fn)
        response = (response_data.features, response_data.labels,
                    response_data.truth, response_data.response_probabilities)
    else:
        raise ValueError(f"unknown evaluation mode {mode!r}")
    name_to_family = {item.name: item.family for item in items}
    truth = [name_to_family.get(name, "") for name in data.names]
    return {
        "features": data.features,
        "labels": data.labels,
        "truth": truth,
        "stats": stats,
        "response": response,
    }


def _response_metrics(model, features: Sequence[Sequence[float]],
                      labels: Sequence[str], truth: Sequence[str]
                      ) -> dict[str, Any]:
    by_truth: dict[str, list[int]] = {}
    correct = 0
    predicted_responsive = 0
    for f, label, group in zip(features, labels, truth):
        pred, _ = model.predict(f)
        correct += int(pred == label)
        predicted_responsive += int(pred == "responsive")
        row = by_truth.setdefault(group, [0, 0, 0, 0])
        row[0] += int(pred == label)
        row[1] += 1
        row[2] += int(label == "responsive")
        row[3] += int(pred == "responsive")
    total = len(labels)
    return {
        "n": total,
        "accuracy": correct / total if total else 0.0,
        "predicted_responsive_rate": (predicted_responsive / total
                                      if total else 0.0),
        "by_truth": {
            name: {
                "n": row[1],
                "accuracy": row[0] / max(1, row[1]),
                "responsive_rate": row[2] / max(1, row[1]),
                "predicted_responsive_rate": row[3] / max(1, row[1]),
            }
            for name, row in sorted(by_truth.items())
        },
    }


class _SafeOnlyModel:
    """Adapt a 3-class gate to the binary allow/deny metric helper."""

    def __init__(self, model):
        self.model = model

    def predict(self, features):
        label, confidence = self.model.predict(features)
        return ("allow" if label == "safe" else "deny"), confidence


def _add_suite_eval(report: dict[str, Any], key: str,
                    data: dict[str, Any], model, gate_model,
                    response_model, threeway_model=None,
                    threeway_source: str = "family",
                    calibrated_safe_threshold: float = 0.0,
                    calibrated_risky_threshold: float = 0.5) -> None:
    keep = [i for i, label in enumerate(data["labels"])
            if label is not None]
    if not keep:
        return
    eval_features = [data["features"][i] for i in keep]
    eval_labels = [data["labels"][i] for i in keep]
    metrics = per_class_accuracy(model, eval_features, eval_labels)
    gate_labels = [allow_label(label) for label in eval_labels]
    entry = {
        "n": len(eval_labels),
        "accuracy": accuracy(model, eval_features, eval_labels),
        "per_class": metrics,
        "gate": _binary_metrics(gate_model, eval_features, gate_labels,
                                groups=eval_labels),
    }
    if threeway_model is not None and threeway_source == "family":
        truth_three = three_way_labels(eval_labels, eval_features)
        entry["threeway_gate"] = {
            "accuracy": accuracy(threeway_model, eval_features, truth_three),
            "per_class": per_class_accuracy(
                threeway_model, eval_features, truth_three),
            "safe": _binary_metrics(
                _SafeOnlyModel(threeway_model), eval_features,
                gate_labels, groups=eval_labels),
        }
    if data.get("stats") is not None:
        entry["excitation"] = data["stats"]
    if data.get("response") is not None:
        rf, rl, rt, rp = data["response"]
        if rl and response_model is not None:
            entry["response"] = _response_metrics(
                response_model, rf, rl, rt)
        if threeway_model is not None and rp:
            cal_labels = three_way_calibrated_labels(
                rp, calibrated_safe_threshold, calibrated_risky_threshold)
            cal_binary = ["allow" if label == "safe" else "deny"
                          for label in cal_labels]
            entry["threeway_calibrated"] = {
                "n": len(cal_labels),
                "accuracy": accuracy(
                    threeway_model, rf, cal_labels),
                "per_class": per_class_accuracy(
                    threeway_model, rf, cal_labels),
                "safe": _binary_metrics(
                    _SafeOnlyModel(threeway_model), rf, cal_binary,
                    groups=rt),
            }
    report["suite_eval"][key] = entry
    print(f"{key} classifier accuracy: {entry['accuracy']:.3f}; "
          f"gate accuracy: {entry['gate']['accuracy']:.3f}; "
          f"gate precision: {entry['gate']['precision']:.3f}",
          flush=True)
    if "threeway_gate" in entry:
        print(f"{key} three-way accuracy: "
              f"{entry['threeway_gate']['accuracy']:.3f}; "
              f"safe precision: "
              f"{entry['threeway_gate']['safe']['precision']:.3f}; "
              f"safe recall: "
              f"{entry['threeway_gate']['safe']['recall']:.3f}",
              flush=True)
    if "threeway_calibrated" in entry:
        cal = entry["threeway_calibrated"]
        print(f"{key} calibrated three-way accuracy: "
              f"{cal['accuracy']:.3f}; safe precision: "
              f"{cal['safe']['precision']:.3f}; safe recall: "
              f"{cal['safe']['recall']:.3f}", flush=True)
    if "response" in entry:
        print(f"{key} response accuracy: "
              f"{entry['response']['accuracy']:.3f}", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Train an offline opponent family / next-action model.")
    ap.add_argument("--episodes", type=int, default=2,
                    help="episodes per TYPE_BANK opponent for base data")
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--validation-fraction", type=float, default=0.2,
                    help="fraction of whole types held out")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=20261024)
    ap.add_argument("--data-mode", choices=DATA_MODES, default="base",
                    help="base RL-Repair trajectories, exploratory schedules, "
                         "counterfactual C/D branches, or combinations")
    ap.add_argument("--explore-episodes", type=int, default=2)
    ap.add_argument("--counterfactual-episodes", type=int, default=1)
    ap.add_argument("--counterfactual-branches", type=int, default=8)
    ap.add_argument("--counterfactual-horizon", type=int, default=6)
    ap.add_argument("--counterfactual-repeats", type=int, default=1,
                    help="independent C/D branch repeats per state")
    ap.add_argument("--counterfactual-prefix", default="explore,base",
                    help="comma-separated prefix modes: explore,base")
    ap.add_argument("--threeway-source", choices=("family", "counterfactual"),
                    default="family",
                    help="train the 3-way gate from family+excitation or "
                         "repeated-counterfactual response probabilities")
    ap.add_argument("--calibrated-safe-threshold", type=float, default=0.0,
                    help="p(response) at or below this is safe")
    ap.add_argument("--calibrated-risky-threshold", type=float, default=0.5,
                    help="p(response) at or above this is risky")
    ap.add_argument("--eval-suite", default="heldout_v2",
                    help="optional held-out suite for classification accuracy")
    ap.add_argument("--eval-mode", choices=EVAL_MODES, default="base",
                    help="evaluate base, exploratory or counterfactual traces")
    ap.add_argument("--eval-episodes", type=int, default=1)
    ap.add_argument("--out", default=weights_path("opponent_classifier.json"))
    ap.add_argument("--gate-out", default=weights_path("opponent_gate.json"))
    ap.add_argument("--threeway-out", default=weights_path(
        "opponent_gate_threeway.json"),
        help="3-class safe/risky/unknown gate output; empty to disable")
    ap.add_argument("--next-out", default=weights_path(
        "opponent_next_action.json"))
    ap.add_argument("--response-out", default=weights_path(
        "opponent_responsiveness.json"))
    ap.add_argument("--report-out", default=result_path(
        "opponent_model_report.json"))
    args = ap.parse_args(argv)

    train_specs, val_specs = split_specs(
        TYPE_BANK, validation_fraction=args.validation_fraction,
        seed=args.seed)
    print(f"opponent types: {len(train_specs)} train / "
          f"{len(val_specs)} validation", flush=True)
    train_data, train_stats, train_response = _collect_training_data(
        train_specs, args, seed=args.seed, validation=False)
    val_data, val_stats, val_response = _collect_training_data(
        val_specs, args, seed=args.seed + 1, validation=True)
    print(f"samples: {len(train_data)} train / {len(val_data)} validation",
          flush=True)
    model, report = train_mlp(
        train_data.features, train_data.labels, classes=CLASSES,
        epochs=args.epochs, lr=args.lr, l2=args.l2, seed=args.seed,
        validation=(val_data.features, val_data.labels))
    model.save(args.out)
    print(f"family MLP: train={report['train_accuracy']:.3f} "
          f"val={report['val_accuracy']:.3f} -> {args.out}", flush=True)

    gate_labels = [allow_label(label) for label in train_data.labels]
    val_gate_labels = [allow_label(label) for label in val_data.labels]
    gate_model, gate_report = train_mlp(
        train_data.features, gate_labels, classes=["allow", "deny"],
        epochs=args.epochs, lr=args.lr, l2=args.l2, seed=args.seed + 1,
        validation=(val_data.features, val_gate_labels))
    gate_model.save(args.gate_out)
    print(f"gate MLP: train={gate_report['train_accuracy']:.3f} "
          f"val={gate_report['val_accuracy']:.3f} -> {args.gate_out}",
          flush=True)

    threeway_model = None
    threeway_report = None
    if args.threeway_out:
        if args.threeway_source == "counterfactual":
            if train_response is None or val_response is None:
                raise ValueError(
                    "--threeway-source counterfactual requires "
                    "--data-mode counterfactual/base+counterfactual/mixed")
            three_train_features = train_response.features
            three_train_labels = three_way_calibrated_labels(
                train_response.response_probabilities,
                args.calibrated_safe_threshold,
                args.calibrated_risky_threshold)
            three_val_features = val_response.features
            three_val_labels = three_way_calibrated_labels(
                val_response.response_probabilities,
                args.calibrated_safe_threshold,
                args.calibrated_risky_threshold)
        else:
            three_train_features = train_data.features
            three_train_labels = three_way_labels(
                train_data.labels, train_data.features)
            three_val_features = val_data.features
            three_val_labels = three_way_labels(
                val_data.labels, val_data.features)
        three_validation = None
        if set(three_val_labels) >= set(THREE_WAY_CLASSES):
            three_validation = (three_val_features, three_val_labels)
        threeway_model, threeway_report = train_mlp(
            three_train_features, three_train_labels,
            classes=list(THREE_WAY_CLASSES), epochs=args.epochs,
            lr=args.lr, l2=args.l2, seed=args.seed + 3,
            validation=three_validation)
        threeway_model.save(args.threeway_out)
        print(f"three-way gate ({args.threeway_source}): "
              f"train={threeway_report['train_accuracy']:.3f} "
              f"val={threeway_report.get('val_accuracy', 0.0):.3f} -> "
              f"{args.threeway_out}", flush=True)

    next_model, next_report = train_softmax(
        train_data.features,
        ["C" if action == C else "D" for action in train_data.next_actions],
        classes=["C", "D"], epochs=args.epochs, lr=args.lr, l2=args.l2,
        seed=args.seed,
        validation=(
            val_data.features,
            ["C" if action == C else "D" for action in val_data.next_actions],
        ))
    next_model.save(args.next_out)
    print(f"next-action model: train={next_report['train_accuracy']:.3f} "
          f"val={next_report['val_accuracy']:.3f} -> {args.next_out}",
          flush=True)

    response_model = None
    response_report = None
    if train_response is not None and val_response is not None \
            and len(train_response.labels):
        response_model, response_report = train_mlp(
            train_response.features, train_response.labels,
            classes=list(RESPONSE_CLASSES), epochs=args.epochs,
            lr=args.lr, l2=args.l2, seed=args.seed + 2,
            validation=(val_response.features, val_response.labels))
        response_model.save(args.response_out)
        print(f"response MLP: train="
              f"{response_report['train_accuracy']:.3f} "
              f"val={response_report['val_accuracy']:.3f} -> "
              f"{args.response_out}", flush=True)

    report["data"] = {
        "mode": args.data_mode,
        "train_samples": len(train_data),
        "validation_samples": len(val_data),
        "train": train_stats,
        "validation": val_stats,
    }
    report["gate"] = gate_report
    report["next_action"] = next_report
    if threeway_report is not None:
        report["threeway_gate"] = threeway_report
    if response_report is not None:
        report["response"] = response_report
    report["validation_types"] = [spec.name for spec in val_specs]
    report["suite_eval"] = {}
    if args.eval_suite:
        from ipd.evaluation.opponent_suite import build_suite

        suite = build_suite(args.eval_suite)
        if args.eval_mode == "both":
            eval_modes = ("base", "explore")
        elif args.eval_mode == "all":
            eval_modes = ("base", "explore", "counterfactual")
        else:
            eval_modes = (args.eval_mode,)
        for mode in eval_modes:
            data = _collect_suite_data(
                suite, mode, args, seed=args.seed + 2)
            key = args.eval_suite if mode == "base" else \
                f"{args.eval_suite}:{mode}"
            _add_suite_eval(
                report, key, data, model, gate_model, response_model,
                threeway_model, args.threeway_source,
                args.calibrated_safe_threshold,
                args.calibrated_risky_threshold)
    out = Path(args.report_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"written {out}")
    return 0


def _binary_metrics(model, features: Sequence[Sequence[float]],
                    labels: Sequence[str],
                    groups: Sequence[str] | None = None) -> dict[str, Any]:
    true_positive = false_positive = false_negative = true_negative = 0
    by_group: dict[str, dict[str, int]] = {}
    for index, (f, truth) in enumerate(zip(features, labels)):
        pred, _ = model.predict(f)
        if pred == "allow" and truth == "allow":
            true_positive += 1
        elif pred == "allow" and truth == "deny":
            false_positive += 1
        elif pred == "deny" and truth == "allow":
            false_negative += 1
        else:
            true_negative += 1
        if groups is not None:
            row = by_group.setdefault(groups[index], {
                "n": 0, "allow": 0, "predicted_allow": 0,
                "true_positive": 0, "false_positive": 0,
            })
            row["n"] += 1
            row["allow"] += int(truth == "allow")
            row["predicted_allow"] += int(pred == "allow")
            row["true_positive"] += int(pred == "allow" and truth == "allow")
            row["false_positive"] += int(pred == "allow" and truth == "deny")
    precision = (true_positive / (true_positive + false_positive)
                 if true_positive + false_positive else 0.0)
    recall = (true_positive / (true_positive + false_negative)
              if true_positive + false_negative else 0.0)
    total = len(labels)
    result = {
        "n": total,
        "accuracy": ((true_positive + true_negative) / total
                     if total else 0.0),
        "precision": precision,
        "recall": recall,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
    }
    if groups is not None:
        for row in by_group.values():
            row["allow_rate"] = row["allow"] / max(1, row["n"])
            row["predicted_allow_rate"] = (
                row["predicted_allow"] / max(1, row["n"]))
            row["precision"] = (
                row["true_positive"] / row["predicted_allow"]
                if row["predicted_allow"] else 0.0)
            row["recall"] = (row["true_positive"] / row["allow"]
                             if row["allow"] else 0.0)
        result["by_group"] = dict(sorted(by_group.items()))
    return result


def _collect_suite_dataset(items: Sequence[Any], *, episodes: int = 1,
                           rounds: int = 100,
                           seed: int = 20261024) -> dict[str, Any]:
    base = NeuralPolicy(weights_path("rl_policy_weights_repair.json"),
                        seed=seed, mode="sample")
    features: list[list[float]] = []
    labels: list[str | None] = []
    truth: list[str] = []
    for i, item in enumerate(items):
        label = suite_class(item.family)
        for episode in range(max(1, int(episodes))):
            rng = random.Random(seed + i * 10007 + episode * 7919)
            opponent = item.factory(rng)
            opponent.reset()
            base.reset()
            mine: list[int] = []
            theirs: list[int] = []
            for t in range(rounds):
                features.append(extract_features(mine, theirs))
                reply = int(opponent.act(t, theirs, mine))
                labels.append(label)
                truth.append(label or item.family)
                action = int(base.act(t, mine, theirs))
                mine.append(action)
                theirs.append(reply)
    return {"features": features, "labels": labels, "truth": truth}


__all__ = [
    "CLASSES", "ALLOW_CLASSES", "SUITE_FAMILY_MAP", "EXPLORATION_TEMPLATES",
    "RESPONSE_CLASSES", "THREE_WAY_CLASSES", "coarse_class", "suite_class",
    "allow_label", "three_way_gate_label", "three_way_labels",
    "three_way_calibrated_label", "three_way_calibrated_labels",
    "Dataset", "ResponseDataset", "concat_datasets", "collect_dataset",
    "exploration_actions", "collect_exploratory_dataset",
    "collect_counterfactual_dataset", "split_specs",
    "LinearOpponentModel", "MLPOpponentModel", "train_softmax", "train_mlp",
    "accuracy", "per_class_accuracy", "OpponentGate", "ThreeWayGate",
    "load_opponent_model", "DATA_MODES", "EVAL_MODES", "main",
]
