#!/usr/bin/env python3
"""Train a GRU-Net IPD policy on GPU/NPU.

Same teacher and DAgger data pipeline as ``ipd/training/neural.py``, but the
policy is a recurrent network:

    85 features per day -> GRU(hidden) -> MLP head -> P(defect)

The GRU carries a hidden state across the 100 days, so it can learn modes such
as "punishing", "testing" and "exploiting" that a stateless MLP has to infer
from hand-built summary features.

Setup and example::

    python3 -m venv .venv
    . .venv/bin/activate
    pip install -r requirements-train.txt

    python3 -m ipd train gru --device auto --epochs 40 --dagger-rounds 1
    python3 -m ipd benchmark --candidates core,gru --fields 20
    python3 -m ipd blind --candidates core,gru --fields 10
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


from ipd.paths import result_path, weights_path

import argparse
import os
from array import array
from typing import Any

from ipd.training.neural import (
    _seed_map,
    describe_device,
    generate_data,
    load_teacher_cfgs,
    resolve_device,
    save_json,
)
from ipd.core.simulator import payoff
from ipd.evaluation.distributions import DISTS
from ipd.policies.neural import FEATURE_VERSION, extract_features, C, D

STEPS = 100


def _sequence_tensors(features: array, labels: bytes, n_features: int, device):
    """Convert flattened match data to (matches, 100, features)."""
    import torch

    n = len(labels)
    if n % STEPS != 0:
        raise SystemExit(f"sample count {n} is not divisible by {STEPS}")
    xbuf = bytearray(features.tobytes())
    ybuf = bytearray(labels)
    x = torch.frombuffer(xbuf, dtype=torch.float32).reshape(-1, n_features).clone()
    y = torch.frombuffer(ybuf, dtype=torch.uint8).float().clone()
    x = x.reshape(-1, STEPS, n_features).to(device)
    y = y.reshape(-1, STEPS).to(device)
    return x, y


def make_gru(n_features: int, hidden: int, num_layers: int):
    import torch.nn as nn

    class GRUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(
                input_size=n_features,
                hidden_size=hidden,
                num_layers=num_layers,
                batch_first=True,
            )
            self.head = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.Tanh(),
                nn.Linear(hidden, 1),
            )

        def forward(self, x):
            out, _ = self.gru(x)
            return self.head(out).squeeze(-1)

    return GRUNet()


def train_gru(
    features: array,
    labels: bytes,
    val_features: array,
    val_labels: bytes,
    n_features: int,
    hidden: int,
    num_layers: int,
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

    x, y = _sequence_tensors(features, labels, n_features, device)
    xv, yv = _sequence_tensors(val_features, val_labels, n_features, device)

    # The inference rules already force C on day 1 / before the opponent's
    # first D, and D on day 100.  Do not waste model capacity on those states:
    # train only on the decisions the network actually controls.
    # Feature indices: 2 = t==0, 3 = t>=99, 40 = D in opponent history.
    mask = (x[:, :, 40] > 0.5) & (x[:, :, 2] < 0.5) & (x[:, :, 3] < 0.5)
    vmask = (xv[:, :, 40] > 0.5) & (xv[:, :, 2] < 0.5) & (xv[:, :, 3] < 0.5)
    if not bool(mask.any()):
        mask = torch.ones_like(mask, dtype=torch.bool)
    if not bool(vmask.any()):
        vmask = torch.ones_like(vmask, dtype=torch.bool)

    mean = x.reshape(-1, n_features).mean(dim=0)
    std = x.reshape(-1, n_features).std(dim=0).clamp_min(1e-3)
    xn = (x - mean) / std
    xvn = (xv - mean) / std

    model = make_gru(n_features, hidden, num_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    pw = torch.tensor([pos_weight], device=device)
    best_loss = float("inf")
    best_state = None
    best_epoch = -1
    best_metrics: dict[str, float] = {}
    history: list[dict[str, Any]] = []
    patience = max(5, epochs // 4)

    print(f"  training GRU-Net on {describe_device(device)}: "
          f"{x.shape[0]} train / {xv.shape[0]} val matches, "
          f"{sum(p.numel() for p in model.parameters())} parameters, "
          f"masked steps={int(mask.sum())}/{mask.numel()}", flush=True)

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(x.shape[0], device=device)
        total = 0.0
        seen = 0
        for start in range(0, x.shape[0], batch_size):
            idx = perm[start:start + batch_size]
            bmask = mask[idx]
            n = int(bmask.sum())
            if n == 0:
                continue
            opt.zero_grad(set_to_none=True)
            logits = model(xn[idx])
            loss = F.binary_cross_entropy_with_logits(
                logits[bmask], y[idx][bmask], pos_weight=pw)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * n
            seen += n
        sched.step()

        model.eval()
        with torch.no_grad():
            vlogits = model(xvn)
            vloss = float(F.binary_cross_entropy_with_logits(
                vlogits[vmask], yv[vmask], pos_weight=pw))
            pred = (vlogits[vmask] >= 0).float()
            truth = yv[vmask]
            acc = float((pred == truth).float().mean())
            tp = float(((pred == 1) & (truth == 1)).sum())
            fp = float(((pred == 1) & (truth == 0)).sum())
            fn = float(((pred == 0) & (truth == 1)).sum())
            prec = tp / max(1.0, tp + fp)
            rec = tp / max(1.0, tp + fn)
        train_loss = total / max(1, seen)
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

    model.eval()
    with torch.no_grad():
        vlogits = model(xvn)
        best_t, best_t_loss = 1.0, float("inf")
        for t in [0.5 + 0.025 * i for i in range(61)]:
            loss = float(F.binary_cross_entropy_with_logits(
                vlogits[vmask] / t, yv[vmask]))
            if loss < best_t_loss:
                best_t, best_t_loss = t, loss
    print(f"  best val: loss={best_loss:.4f} acc={best_metrics['val_acc']:.3f} "
          f"temperature={best_t:.3f}", flush=True)
    return model, mean, std, best_t, best_metrics, history


def export_gru(model, mean, std, temperature: float,
               meta: dict[str, Any]) -> dict[str, Any]:
    import torch

    gru_layers = []
    for layer in range(model.gru.num_layers):
        gru_layers.append({
            "weight_ih": getattr(model.gru, f"weight_ih_l{layer}").detach().cpu().tolist(),
            "weight_hh": getattr(model.gru, f"weight_hh_l{layer}").detach().cpu().tolist(),
            "bias_ih": getattr(model.gru, f"bias_ih_l{layer}").detach().cpu().tolist(),
            "bias_hh": getattr(model.gru, f"bias_hh_l{layer}").detach().cpu().tolist(),
        })

    head = []
    for module in model.head:
        if isinstance(module, torch.nn.Linear):
            head.append({
                "w": module.weight.detach().cpu().tolist(),
                "b": module.bias.detach().cpu().tolist(),
            })

    return {
        "format": "ipd-gru-policy-v1",
        "feature_version": FEATURE_VERSION,
        "n_features": int(mean.shape[0]),
        "hidden": int(model.gru.hidden_size),
        "num_layers": int(model.gru.num_layers),
        "gru": gru_layers,
        "head": head,
        "feature_mean": mean.detach().cpu().tolist(),
        "feature_std": std.detach().cpu().tolist(),
        "temperature": float(temperature),
        "meta": meta,
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description="Train a recurrent GRU-Net IPD policy on GPU/NPU.")
    ap.add_argument("--teacher", default="triple",
                    choices=("triple", "final", "resp", "bayes"))
    ap.add_argument("--device", default="auto",
                    choices=("auto", "cuda", "npu", "mps", "cpu"))
    ap.add_argument("--dists", default=",".join(DISTS))
    ap.add_argument("--fields-per-dist", type=int, default=12)
    ap.add_argument("--val-fields-per-dist", type=int, default=3)
    ap.add_argument("--dagger-fields-per-dist", type=int, default=6)
    ap.add_argument("--dagger-rounds", type=int, default=1)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--val-n", type=int, default=80)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32,
                    help="matches per batch (each match is 100 steps)")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--pos-weight", type=float, default=1.0)
    ap.add_argument("--explore", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=20261025)
    ap.add_argument("--out", default=weights_path("gru_policy_weights.json"))
    ap.add_argument("--history-out", default=result_path("gru_training_history.json"))
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
        nonlocal model, mean, std, temperature, metrics
        print(f"training {round_name}...", flush=True)
        model, mean, std, temperature, metrics, hist = train_gru(
            xx, yy, xval, yval, n_features, args.hidden, args.layers, device,
            args.epochs, args.batch_size, args.lr, args.pos_weight,
            args.seed + 1000 * training_rounds)
        for row in hist:
            history_all.append({**row, "round": round_name})

    fit(x, y, xv, yv, "bc")

    for dagger in range(args.dagger_rounds if not args.no_dagger else 0):
        training_rounds += 1
        meta = {
            "teacher": args.teacher,
            "round": f"dagger-{dagger + 1}",
            "seed": args.seed,
            "n_features": n_features,
            "hidden": args.hidden,
            "layers": args.layers,
            "samples": len(y),
            "val_samples": len(yv),
        }
        weights = export_gru(model, mean, std, temperature, meta)
        dagger_seeds = _seed_map(args.seed + 20000 + dagger * 131, dists,
                                 args.dagger_fields_per_dist)
        print(f"generating DAgger data (round {dagger + 1})...", flush=True)
        xd, yd, nd, _ = generate_data(
            dists, dagger_seeds, args.n, args.teacher, cfgs, args.workers,
            policy_weights=weights, student_type="gru",
            explore=args.explore, policy_seed=args.seed + 7000 + dagger,
            label=f"dagger-{dagger + 1}", mp_context="spawn")
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
        "layers": args.layers,
        "samples": len(y),
        "val_samples": len(yv),
        "device": describe_device(device),
        "torch": torch.__version__,
        "metrics": metrics,
        "temperature": temperature,
    }
    weights = export_gru(model, mean, std, temperature, meta)
    save_json(args.out, weights)
    save_json(args.history_out, {"meta": meta, "history": history_all})
    print(f"written {args.out} ({os.path.getsize(args.out)} bytes)")
    print(f"written {args.history_out}")

    from ipd.policies.gru import GRUPolicy
    p = GRUPolicy(weights, mode="argmax")
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
