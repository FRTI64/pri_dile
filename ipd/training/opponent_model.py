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
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

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


def collect_dataset(specs: Iterable[Any], *, episodes: int = 2,
                    rounds: int = 100, seed: int = 20261024,
                    base_weights: str | None = None) -> Dataset:
    """Collect (history features, family label, next action) samples."""
    base = NeuralPolicy(base_weights or weights_path(
        "rl_policy_weights_repair.json"), seed=seed, mode="sample")
    features: list[list[float]] = []
    labels: list[str] = []
    next_actions: list[int] = []
    names: list[str] = []
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
                action = int(base.act(t, mine, theirs))
                mine.append(action)
                theirs.append(reply)
    return Dataset(features, labels, next_actions, names)


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
                 next_threshold: float = 0.6):
        self.model = model
        self.allowed = set(allowed)
        self.threshold = float(threshold)
        self.min_day = int(min_day)
        self.require_next_d = bool(require_next_d)
        self.next_model = next_model
        self.next_threshold = float(next_threshold)
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
                "next_probability_d": None}
        allowed = label in self.allowed and confidence >= self.threshold
        if allowed and self.require_next_d and self.next_model is not None:
            p_d = self.next_model.probabilities(features)[
                self.next_model.classes.index("D")]
            info["next_probability_d"] = p_d
            allowed = p_d >= self.next_threshold
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Train an offline opponent family / next-action model.")
    ap.add_argument("--episodes", type=int, default=2,
                    help="episodes per TYPE_BANK opponent")
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--validation-fraction", type=float, default=0.2,
                    help="fraction of whole types held out")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=20261024)
    ap.add_argument("--eval-suite", default="heldout_v2",
                    help="optional held-out suite for classification accuracy")
    ap.add_argument("--eval-episodes", type=int, default=1)
    ap.add_argument("--out", default=weights_path("opponent_classifier.json"))
    ap.add_argument("--gate-out", default=weights_path("opponent_gate.json"))
    ap.add_argument("--next-out", default=weights_path(
        "opponent_next_action.json"))
    ap.add_argument("--report-out", default=result_path(
        "opponent_model_report.json"))
    args = ap.parse_args(argv)

    train_specs, val_specs = split_specs(
        TYPE_BANK, validation_fraction=args.validation_fraction,
        seed=args.seed)
    print(f"opponent types: {len(train_specs)} train / "
          f"{len(val_specs)} validation", flush=True)
    train_data = collect_dataset(train_specs, episodes=args.episodes,
                                 rounds=args.rounds, seed=args.seed)
    val_data = collect_dataset(val_specs, episodes=max(1, args.episodes),
                               rounds=args.rounds, seed=args.seed + 1)
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

    report["gate"] = gate_report
    report["next_action"] = next_report
    report["validation_types"] = [spec.name for spec in val_specs]
    report["suite_eval"] = {}
    if args.eval_suite:
        from ipd.evaluation.opponent_suite import build_suite

        suite = build_suite(args.eval_suite)
        eval_specs = []
        for item in suite:
            eval_specs.append(item)
        data = _collect_suite_dataset(
            eval_specs, episodes=args.eval_episodes, rounds=args.rounds,
            seed=args.seed + 2)
        keep = [i for i, label in enumerate(data["labels"])
                if label is not None]
        if keep:
            eval_features = [data["features"][i] for i in keep]
            eval_labels = [data["labels"][i] for i in keep]
            metrics = per_class_accuracy(model, eval_features, eval_labels)
            gate_labels = [allow_label(label) for label in eval_labels]
            report["suite_eval"][args.eval_suite] = {
                "n": len(eval_labels),
                "accuracy": accuracy(model, eval_features, eval_labels),
                "per_class": metrics,
                "gate": _binary_metrics(gate_model, eval_features,
                                        gate_labels),
            }
            print(f"{args.eval_suite} classifier accuracy: "
                  f"{report['suite_eval'][args.eval_suite]['accuracy']:.3f}; "
                  f"gate accuracy: "
                  f"{report['suite_eval'][args.eval_suite]['gate']['accuracy']:.3f}; "
                  f"gate precision: "
                  f"{report['suite_eval'][args.eval_suite]['gate']['precision']:.3f}",
                  flush=True)
    out = Path(args.report_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"written {out}")
    return 0


def _binary_metrics(model, features: Sequence[Sequence[float]],
                    labels: Sequence[str]) -> dict[str, Any]:
    true_positive = false_positive = false_negative = true_negative = 0
    for f, truth in zip(features, labels):
        pred, _ = model.predict(f)
        if pred == "allow" and truth == "allow":
            true_positive += 1
        elif pred == "allow" and truth == "deny":
            false_positive += 1
        elif pred == "deny" and truth == "allow":
            false_negative += 1
        else:
            true_negative += 1
    precision = (true_positive / (true_positive + false_positive)
                 if true_positive + false_positive else 0.0)
    recall = (true_positive / (true_positive + false_negative)
              if true_positive + false_negative else 0.0)
    total = len(labels)
    return {
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
    "CLASSES", "ALLOW_CLASSES", "SUITE_FAMILY_MAP", "coarse_class",
    "suite_class", "allow_label", "Dataset", "collect_dataset", "split_specs",
    "LinearOpponentModel", "MLPOpponentModel", "train_softmax", "train_mlp",
    "accuracy", "per_class_accuracy", "OpponentGate", "load_opponent_model",
    "main",
]
