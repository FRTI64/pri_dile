#!/usr/bin/env python3
"""Train a neural IPD policy on GPU or NPU.

The training signal is behavioural cloning / DAgger from the strongest
hand-written strategy (`Triple-majority` by default).  Rollouts are produced by
the ordinary Python simulator in parallel worker processes; the MLP itself is
trained on CUDA, Ascend NPU (``torch_npu``), Apple MPS or CPU.  Device selection
is automatic: NPU -> CUDA -> MPS -> CPU.

The exported ``neural_policy_weights.json`` is consumed by
``ipd.policies.neural.NeuralPolicy``, which has no third-party dependencies.

Setup (PyTorch is optional for the rest of the repository)::

    python3 -m venv .venv
    . .venv/bin/activate
    pip install torch

Train::

    python3 -m ipd train neural --device auto --epochs 40 --dagger-rounds 1

Evaluate::

    python3 -m ipd benchmark --candidates core,neural --fields 8 --workers 20
"""
from __future__ import annotations

from ipd.paths import result_path, weights_path

import argparse
import json
import os
import random
import statistics
import time
from array import array
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

from ipd.core.simulator import C, D, payoff
from ipd.evaluation.distributions import DISTS, build_dist_field
from ipd.policies.neural import FEATURE_VERSION, NeuralPolicy, extract_features
from ipd.training.common import save_json, seed_map as _seed_map


# ---------------------------------------------------------------------------
# Teacher / rollout generation (no torch in this section)
# ---------------------------------------------------------------------------

def load_teacher_cfgs() -> dict[str, Any]:
    """Load the tuned teacher configurations (with embedded fallbacks)."""
    from ipd.policies.tuned import load_tuned_configs
    tuned = load_tuned_configs()
    return {"final": tuned.final, "resp": tuned.resp, "bayes": tuned.bayes}


def make_teacher(kind: str, cfgs: dict[str, Any], seed: int):
    """Construct a teacher policy.  Imported lazily so workers stay light."""
    from ipd.policies.rules import ParamBayesPrior, ParamPolicy, ParamResp, Triple

    if kind == "triple":
        t = Triple(cfgs["final"], cfgs["resp"], cfgs["bayes"], mode="majority")
        t.f.rng.seed(seed)
        t.r.rng.seed(seed + 1)
        t.rng.seed(seed + 2)
        return t
    if kind == "final":
        return ParamPolicy(seed=seed, **cfgs["final"])
    if kind == "resp":
        return ParamResp(seed=seed, **cfgs["resp"])
    if kind == "bayes":
        return ParamBayesPrior(cfgs["bayes"])
    raise ValueError(f"unknown teacher: {kind}")


def _teacher_action(teacher, t: int, mine: list[int], theirs: list[int]) -> int:
    if hasattr(teacher, "act"):
        return teacher.act(t, mine, theirs)
    return teacher.move(t + 1, mine, theirs)


def _student_action(student: NeuralPolicy, t: int, mine: list[int], theirs: list[int]) -> int:
    return student.act(t, mine, theirs)


def rollout_task(task: dict[str, Any]):
    """Play one field and return packed features/labels.

    Returns ``(features, labels, n_samples, score)`` where features is an
    ``array('f')`` of shape (n_samples, n_features) and labels is ``bytes``.
    """

    dist = task["dist"]
    seed = task["seed"]
    n = task["n"]
    teacher_kind = task["teacher"]
    cfgs = task["cfgs"]
    student_weights = task.get("student")
    student_type = task.get("student_type", "mlp")
    explore = float(task.get("explore", 0.0))
    policy_seed = int(task.get("policy_seed", seed))

    field = build_dist_field(dist, seed, n)
    teacher = make_teacher(teacher_kind, cfgs, seed)
    student = None
    if student_weights is not None:
        if student_type == "gru":
            from ipd.policies.gru import GRUPolicy
            student = GRUPolicy(student_weights, seed=policy_seed)
        else:
            student = NeuralPolicy(student_weights, seed=policy_seed)
    rng = random.Random(seed ^ 0x5EED)

    features = array("f")
    labels = bytearray()
    total_score = 0
    count = 0

    for opp in field:
        teacher.reset()
        opp.reset()
        if student is not None:
            student.reset()
        mine: list[int] = []
        theirs: list[int] = []

        for t in range(100):
            if student is None:
                action = _teacher_action(teacher, t, mine, theirs)
                label = action
            else:
                action = _student_action(student, t, mine, theirs)
                if explore > 0 and rng.random() < explore:
                    action = C if rng.random() < 0.5 else D
                # Expert label for the state the student actually visited.
                label = _teacher_action(teacher, t, mine, theirs)

            features.extend(extract_features(mine, theirs))
            labels.append(1 if label == D else 0)
            count += 1

            reply = opp.act(t, theirs, mine)
            total_score += payoff(action, reply)
            mine.append(action)
            theirs.append(reply)

    return features, bytes(labels), count, total_score / max(1, n)


def generate_data(
    dists: tuple[str, ...],
    seeds: dict[str, list[int]],
    n: int,
    teacher: str,
    cfgs: dict[str, Any],
    workers: int,
    policy_weights: dict | None = None,
    student_type: str = "mlp",
    explore: float = 0.0,
    policy_seed: int = 20261025,
    label: str = "data",
    mp_context: str | None = None,
):
    tasks = []
    for dist in dists:
        for seed in seeds[dist]:
            tasks.append({
                "dist": dist,
                "seed": seed,
                "n": n,
                "teacher": teacher,
                "cfgs": cfgs,
                "student": policy_weights,
                "student_type": student_type,
                "explore": explore,
                "policy_seed": policy_seed,
            })
    if not tasks:
        raise SystemExit(f"{label}: no fields requested (check --fields-per-dist)")

    t0 = time.time()
    results = []
    kwargs = {}
    if mp_context:
        import multiprocessing as mp
        kwargs["mp_context"] = mp.get_context(mp_context)
    with ProcessPoolExecutor(max_workers=workers, **kwargs) as ex:
        futures = {ex.submit(rollout_task, task): task for task in tasks}
        done = 0
        for fut in as_completed(futures):
            results.append(fut.result())
            done += 1
            if done == len(tasks) or done % max(1, len(tasks) // 8) == 0:
                print(f"  {label}: {done}/{len(tasks)} fields "
                      f"({time.time() - t0:.1f}s)", flush=True)

    features = array("f")
    labels = bytearray()
    count = 0
    scores = []
    for f, y, c, s in results:
        features.frombytes(f.tobytes())
        labels.extend(y)
        count += c
        scores.append(s)
    print(f"  {label}: {count} samples, {len(scores)} fields, "
          f"mean score/opponent {statistics.fmean(scores):.2f}", flush=True)
    return features, labels, count, scores


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def resolve_device(requested: str):
    import torch

    if requested == "npu":
        try:
            import torch_npu  # noqa: F401
        except ImportError as exc:
            raise SystemExit(
                "NPU requested but torch_npu is not installed. "
                "Install the Ascend PyTorch adapter or use --device auto."
            ) from exc
    if requested == "auto":
        # torch.npu only exists after importing the Ascend adapter.
        try:
            import torch_npu  # noqa: F401
        except Exception:
            pass
        if hasattr(torch, "npu") and torch.npu.is_available():
            requested = "npu"
        elif torch.cuda.is_available():
            requested = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"

    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false.")
    if requested == "npu":
        if not (hasattr(torch, "npu") and torch.npu.is_available()):
            raise SystemExit("torch.npu.is_available() is false.")
    if requested == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise SystemExit("MPS requested but not available.")
    return torch.device(requested)


def describe_device(device) -> str:
    import torch

    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(device)})"
    if device.type == "npu":
        try:
            return f"npu ({torch.npu.get_device_name(device)})"
        except Exception:
            return "npu"
    return str(device)


# ---------------------------------------------------------------------------
# Torch training
# ---------------------------------------------------------------------------

def _torch_tensors(features: array, labels: bytes, n_features: int, device):
    import torch

    # frombuffer avoids materializing millions of Python floats.
    xbuf = bytearray(features.tobytes())
    ybuf = bytearray(labels)
    x = torch.frombuffer(xbuf, dtype=torch.float32).reshape(-1, n_features).clone()
    y = torch.frombuffer(ybuf, dtype=torch.uint8).float().clone()
    return x.to(device), y.to(device)


def make_model(n_features: int, hidden: list[int]):
    import torch.nn as nn

    layers: list[nn.Module] = []
    prev = n_features
    for h in hidden:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.Tanh())
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


def train_model(
    features: array,
    labels: bytes,
    val_features: array,
    val_labels: bytes,
    n_features: int,
    hidden: list[int],
    device,
    epochs: int,
    batch_size: int,
    lr: float,
    pos_weight: float,
    seed: int,
):
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    x, y = _torch_tensors(features, labels, n_features, device)
    xv, yv = _torch_tensors(val_features, val_labels, n_features, device)
    model = make_model(n_features, hidden).to(device)

    mean = x.mean(dim=0)
    std = x.std(dim=0).clamp_min(1e-3)
    xn = (x - mean) / std
    xvn = (xv - mean) / std

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    pw = torch.tensor([pos_weight], device=device)
    best_loss = float("inf")
    best_state = None
    best_epoch = -1
    history = []
    patience = max(5, epochs // 4)

    print(f"  training on {describe_device(device)}: "
          f"{x.shape[0]} train / {xv.shape[0]} val samples, "
          f"{sum(p.numel() for p in model.parameters())} parameters", flush=True)

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(x.shape[0], device=device)
        total = 0.0
        for start in range(0, x.shape[0], batch_size):
            idx = perm[start:start + batch_size]
            opt.zero_grad(set_to_none=True)
            logits = model(xn[idx]).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, y[idx], pos_weight=pw)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
        sched.step()

        model.eval()
        with torch.no_grad():
            vlogits = model(xvn).squeeze(-1)
            vloss = float(F.binary_cross_entropy_with_logits(vlogits, yv,
                                                             pos_weight=pw))
            pred = (vlogits >= 0).float()
            acc = float((pred == yv).float().mean())
            tp = float(((pred == 1) & (yv == 1)).sum())
            fp = float(((pred == 1) & (yv == 0)).sum())
            fn = float(((pred == 0) & (yv == 1)).sum())
            prec = tp / max(1.0, tp + fp)
            rec = tp / max(1.0, tp + fn)
        train_loss = total / max(1, x.shape[0])
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": vloss, "val_acc": acc,
                        "val_precision_d": prec, "val_recall_d": rec})
        improved = vloss < best_loss - 1e-6
        if improved:
            best_loss = vloss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            best_metrics = {"val_loss": vloss, "val_acc": acc,
                            "val_precision_d": prec, "val_recall_d": rec}
        if epoch == 1 or improved or epoch % 5 == 0:
            print(f"  epoch {epoch:3d} train={train_loss:.4f} "
                  f"val={vloss:.4f} acc={acc:.3f} "
                  f"P(D)={prec:.3f} R(D)={rec:.3f}"
                  f"{' *' if improved else ''}", flush=True)
        if epoch - best_epoch >= patience:
            print(f"  early stop at epoch {epoch} (best {best_epoch})", flush=True)
            break

    assert best_state is not None
    model.load_state_dict(best_state)

    # Temperature scaling for well-calibrated sampling.  It does not change
    # argmax decisions, only the stochastic policy's mix of C and D.
    model.eval()
    with torch.no_grad():
        vlogits = model(xvn).squeeze(-1)
        best_t, best_t_loss = 1.0, float("inf")
        for t in [0.5 + 0.025 * i for i in range(61)]:
            loss = float(F.binary_cross_entropy_with_logits(vlogits / t, yv))
            if loss < best_t_loss:
                best_t, best_t_loss = t, loss
    print(f"  best val: loss={best_loss:.4f} acc={best_metrics['val_acc']:.3f} "
          f"temperature={best_t:.3f}", flush=True)
    return model, mean, std, best_t, best_metrics, history


def export_weights(
    model,
    mean,
    std,
    temperature: float,
    meta: dict[str, Any],
) -> dict[str, Any]:
    import torch

    layers = []
    for module in model:
        if isinstance(module, torch.nn.Linear):
            layers.append({
                "w": module.weight.detach().cpu().tolist(),
                "b": module.bias.detach().cpu().tolist(),
            })
    return {
        "format": "ipd-neural-policy-v1",
        "feature_version": FEATURE_VERSION,
        "n_features": int(mean.shape[0]),
        "activation": "tanh",
        "layers": layers,
        "feature_mean": mean.detach().cpu().tolist(),
        "feature_std": std.detach().cpu().tolist(),
        "temperature": float(temperature),
        "meta": meta,
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description="Train a neural IPD policy on GPU/NPU with DAgger.")
    ap.add_argument("--teacher", default="triple",
                    choices=("triple", "final", "resp", "bayes"))
    ap.add_argument("--device", default="auto",
                    choices=("auto", "cuda", "npu", "mps", "cpu"))
    ap.add_argument("--dists", default=",".join(DISTS),
                    help="comma separated: original,late,grim30,wide")
    ap.add_argument("--fields-per-dist", type=int, default=12)
    ap.add_argument("--val-fields-per-dist", type=int, default=3)
    ap.add_argument("--dagger-fields-per-dist", type=int, default=6)
    ap.add_argument("--dagger-rounds", type=int, default=1)
    ap.add_argument("--n", type=int, default=60,
                    help="opponents per training field")
    ap.add_argument("--val-n", type=int, default=80)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--hidden", type=int, nargs="+", default=[64, 32])
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--pos-weight", type=float, default=1.0,
                    help="BCE weight for defection labels")
    ap.add_argument("--explore", type=float, default=0.05,
                    help="DAgger epsilon exploration")
    ap.add_argument("--seed", type=int, default=20261025)
    ap.add_argument("--out", default=weights_path("neural_policy_weights.json"))
    ap.add_argument("--history-out", default=result_path("neural_training_history.json"))
    ap.add_argument("--no-dagger", action="store_true")
    args = ap.parse_args(argv)

    try:
        import torch
    except ImportError:
        raise SystemExit(
            "PyTorch is required for training.  Create a virtual environment:\n"
            "  python3 -m venv .venv && . .venv/bin/activate\n"
            "  pip install -r requirements-train.txt\n"
            "Then run this script with the same Python."
        )

    dists = tuple(x.strip() for x in args.dists.split(",") if x.strip())
    unknown = [d for d in dists if d not in DISTS]
    if unknown:
        raise SystemExit(f"unknown distributions: {unknown}")

    print(f"teacher={args.teacher} dists={','.join(dists)} "
          f"fields={args.fields_per_dist}/dist n={args.n}", flush=True)
    cfgs = load_teacher_cfgs()

    train_seeds = _seed_map(args.seed, dists, args.fields_per_dist)
    val_seeds = _seed_map(args.seed + 10000, dists, args.val_fields_per_dist)

    print("generating teacher data...", flush=True)
    x, y, n_samples, _ = generate_data(
        dists, train_seeds, args.n, args.teacher, cfgs, args.workers,
        label="teacher-train")
    xv, yv, n_val, _ = generate_data(
        dists, val_seeds, args.val_n, args.teacher, cfgs, args.workers,
        label="teacher-val")

    n_features = len(extract_features([], []))
    device = resolve_device(args.device)
    print(f"device: {describe_device(device)}", flush=True)

    training_rounds = 0
    model = mean = std = temperature = metrics = None
    history_all: list[dict[str, Any]] = []

    def fit(xx, yy, xval, yval, round_name: str):
        nonlocal model, mean, std, temperature, metrics, history_all
        print(f"training {round_name}...", flush=True)
        model, mean, std, temperature, metrics, hist = train_model(
            xx, yy, xval, yval, n_features, args.hidden, device,
            args.epochs, args.batch_size, args.lr, args.pos_weight,
            args.seed + 1000 * training_rounds)
        for row in hist:
            history_all.append({**row, "round": round_name})
        return hist

    fit(x, y, xv, yv, "bc")

    for dagger in range(args.dagger_rounds if not args.no_dagger else 0):
        training_rounds += 1
        meta = {
            "teacher": args.teacher,
            "round": f"dagger-{dagger + 1}",
            "seed": args.seed,
            "n_features": n_features,
            "hidden": args.hidden,
            "samples": len(y),
            "val_samples": len(yv),
        }
        weights = export_weights(model, mean, std, temperature, meta)
        dagger_seeds = _seed_map(args.seed + 20000 + dagger * 131, dists,
                                 args.dagger_fields_per_dist)
        print(f"generating DAgger data (round {dagger + 1})...", flush=True)
        xd, yd, nd, _ = generate_data(
            dists, dagger_seeds, args.n, args.teacher, cfgs, args.workers,
            policy_weights=weights, explore=args.explore,
            policy_seed=args.seed + 7000 + dagger,
            label=f"dagger-{dagger + 1}", mp_context="spawn")

        # Aggregate instead of replacing, so the expert distribution is kept.
        x.frombytes(xd.tobytes())
        y.extend(yd)
        print(f"  aggregate dataset: {len(y)} samples", flush=True)
        fit(x, y, xv, yv, f"dagger-{dagger + 1}")

    meta = {
        "teacher": args.teacher,
        "teacher_cfgs": cfgs,
        "rounds": 1 + (args.dagger_rounds if not args.no_dagger else 0),
        "seed": args.seed,
        "dists": list(dists),
        "fields_per_dist": args.fields_per_dist,
        "val_fields_per_dist": args.val_fields_per_dist,
        "dagger_fields_per_dist": args.dagger_fields_per_dist,
        "n": args.n,
        "val_n": args.val_n,
        "hidden": args.hidden,
        "samples": len(y),
        "val_samples": len(yv),
        "device": describe_device(device),
        "torch": torch.__version__,
        "metrics": metrics,
        "temperature": temperature,
    }
    weights = export_weights(model, mean, std, temperature, meta)
    save_json(args.out, weights)
    save_json(args.history_out, {"meta": meta, "history": history_all})
    print(f"written {args.out} ({os.path.getsize(args.out)} bytes)")
    print(f"written {args.history_out}")
    print("quick check:")
    p = NeuralPolicy(weights, mode="argmax")
    mine: list[int] = []
    theirs: list[int] = []
    for t in range(100):
        a = p.act(t, mine, theirs)
        b = C if t < 90 else D
        mine.append(a)
        theirs.append(b)
    print(f"  vs AllC-until-90 -> actions C={mine.count(C)} D={mine.count(D)} "
          f"score={sum(payoff(a, b) for a, b in zip(mine, theirs))}")


if __name__ == "__main__":
    main()
