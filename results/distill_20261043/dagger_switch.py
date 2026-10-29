"""DAgger distillation of RL-Switch into one deployable policy.

Two student families are supported:

* ``mlp``  - 160/128 MLP (the same shape as RL-Wide160), warm-started from
  ``weights/rl_policy_checkpoint_wide160.json``;
* ``gru``  - 1-layer GRU with a small MLP head (``ipd-gru-policy-v1``), which
  can carry the match-level "already switched" bit in its hidden state.

The teacher is the deterministic RL-Switch rule: use RL-Wide160 until the
first six rounds contain at least two defections by both sides and the
opponent's most recent move is C, then use RL-Repair for the rest of the
match.  Labels are the chosen expert's P(defect), not sampled actions.

Every round collects the student's own trajectories (with a small epsilon
exploration) and labels them with the teacher, then retrains on the
aggregate dataset.  Round 0 is plain behaviour cloning on teacher states.

Example::

    python3 results/distill_20261043/dagger_switch.py \
      --student gru --rounds 2 --bc-fields 6 --dagger-fields 4 \
      --n 48 --out results/distill_20261043/gru_dagger_ckpt.json

The script only needs PyTorch for training; the exported checkpoints run
through the standard-library GRUPolicy / RLPolicy inference paths.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from array import array
from pathlib import Path

import torch
import torch.nn.functional as F

from ipd.core.primitives import C, D, payoff
from ipd.evaluation.distributions import build_train_dist_field
from ipd.evaluation.protocol import build_protocol_field
from ipd.policies.gru import GRUPolicy
from ipd.policies.neural import extract_features
from ipd.policies.registry import make_policy
from ipd.policies.rl import RLPolicy
from ipd.training.gru import export_gru, make_gru
from ipd.training.rl import export_checkpoint, load_initial_weights, make_model

STEPS = 100
N_FEATURES = 85
DEFAULT_SOURCES = ("train", "train_complex", "train_pattern_breaker",
                   "extort_hard", "periodic_hard", "hard")


def _field(source: str, seed: int, n: int):
    if source == "train":
        return build_protocol_field("train", seed, n)
    return build_train_dist_field(source, seed, n)


def switch_active(mine: list[int], theirs: list[int]) -> bool:
    """Exact RL-Switch rule for the state *before* the current action."""
    if len(theirs) < 6:
        return False
    return (theirs[:6].count(D) >= 2 and mine[:6].count(D) >= 2
            and C in theirs[6:])


def _payoff(a: int, b: int) -> int:
    return payoff(a, b)


def collect(specs: list[tuple[str, int]], n: int, student, explore: float,
            seed: int):
    """Roll out one dataset.

    ``student`` is ``None`` for teacher trajectories (BC), otherwise a
    policy object implementing ``reset``/``act``.  Returns flat feature and
    soft-label arrays, a switch flag per state, and match scores.
    """
    base = make_policy("rl-wide160", mode="sample", seed=seed)
    alt = make_policy("rl-repair", mode="sample", seed=seed + 1)
    rng = random.Random(seed ^ 0xDA66E5)
    features = array("f")
    targets = array("f")
    switches = bytearray()
    scores: list[int] = []

    for source, field_seed in specs:
        field = _field(source, field_seed, n)
        for opp in field:
            base.reset()
            alt.reset()
            opp.reset()
            if student is not None:
                student.reset()
            mine: list[int] = []
            theirs: list[int] = []
            total = 0
            for t in range(STEPS):
                sw = switch_active(mine, theirs)
                expert = alt if sw else base
                target = expert.predict_proba(mine, theirs)
                features.extend(extract_features(mine, theirs))
                targets.append(float(target))
                switches.append(1 if sw else 0)

                if student is None:
                    action = expert.act(t, mine, theirs)
                else:
                    action = student.act(t, mine, theirs)
                    if explore > 0.0 and rng.random() < explore:
                        action = C if rng.random() < 0.5 else D
                reply = opp.act(t, theirs, mine)
                total += payoff(action, reply)
                mine.append(action)
                theirs.append(reply)
            scores.append(total)
    return features, targets, switches, scores


def _torch_sequences(features: array, targets: array, device):
    n = len(targets)
    if n % STEPS:
        raise SystemExit(f"sample count {n} is not divisible by {STEPS}")
    xbuf = bytearray(features.tobytes())
    tbuf = array("f", targets).tobytes()
    x = torch.frombuffer(xbuf, dtype=torch.float32).reshape(-1, STEPS,
                                                             N_FEATURES)
    y = torch.frombuffer(bytearray(tbuf), dtype=torch.float32).reshape(
        -1, STEPS)
    return x.to(device), y.to(device)


def _forced_mask(x: torch.Tensor) -> torch.Tensor:
    # indices: 2 = t==0, 3 = t>=99, 40 = opponent has defected at least once
    return (x[..., 40] > 0.5) & (x[..., 2] < 0.5) & (x[..., 3] < 0.5)


def _metrics(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor,
             switch: torch.Tensor | None = None):
    loss = float(F.binary_cross_entropy_with_logits(logits[mask], y[mask]))
    pred = logits[mask] >= 0
    truth = y[mask] >= 0.5
    acc = float((pred == truth).float().mean())
    out = {"loss": loss, "acc": acc, "n": int(mask.sum())}
    if switch is not None:
        sm = mask & switch
        if bool(sm.any()):
            out["switch_loss"] = float(F.binary_cross_entropy_with_logits(
                logits[sm], y[sm]))
            out["switch_acc"] = float(
                ((logits[sm] >= 0) == (y[sm] >= 0.5)).float().mean())
            out["switch_n"] = int(sm.sum())
    return out


def train_mlp(features: array, targets: array, switches: bytearray,
              init_path: str, epochs: int, lr: float, batch: int,
              device, seed: int, freeze_trunk: bool = False):
    """Train a stateless 160/128 student with the checkpoint's normalization."""
    torch.manual_seed(seed)
    model = make_model([160, 128])
    mean, std, _ = load_initial_weights(model, init_path, load_value=True)
    if freeze_trunk:
        for parameter in model.policy_trunk.parameters():
            parameter.requires_grad_(False)
    model.to(device)
    mean_t = torch.tensor(mean, dtype=torch.float32, device=device)
    std_t = torch.tensor(std, dtype=torch.float32,
                         device=device).clamp_min(1e-3)

    x = torch.frombuffer(bytearray(features.tobytes()),
                         dtype=torch.float32).reshape(-1, N_FEATURES).to(device)
    y = torch.frombuffer(bytearray(array("f", targets).tobytes()),
                         dtype=torch.float32).to(device)
    mask = _forced_mask(x)
    x = (x[mask] - mean_t) / std_t
    y = y[mask]
    n = len(y)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        total = 0.0
        for start in range(0, n, batch):
            idx = perm[start:start + batch]
            logit = model.policy(model.policy_trunk(x[idx])).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(logit, y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            total += float(loss.detach()) * len(idx)
        history.append({"epoch": epoch, "train_loss": total / n})
        if epoch == 1 or epoch % 5 == 0:
            print(f"  mlp epoch {epoch:2d} train={total / n:.4f}", flush=True)
    return model, mean, std, history


def train_gru(features: array, targets: array, switches: bytearray,
              val_features: array, val_targets: array, val_switches: bytearray,
              hidden: int, layers: int, epochs: int, lr: float, batch: int,
              patience: int, device, seed: int):
    """Train a recurrent student on 100-day sequences."""
    torch.manual_seed(seed)
    x, y = _torch_sequences(features, targets, device)
    xv, yv = _torch_sequences(val_features, val_targets, device)
    mean = x.reshape(-1, N_FEATURES).mean(dim=0)
    std = x.reshape(-1, N_FEATURES).std(dim=0).clamp_min(1e-3)
    xn = (x - mean) / std
    xvn = (xv - mean) / std
    mask = _forced_mask(x)
    vmask = _forced_mask(xv)
    swmask = (torch.tensor(list(switches), dtype=torch.bool,
                           device=device).reshape(-1, STEPS))
    vswmask = (torch.tensor(list(val_switches), dtype=torch.bool,
                            device=device).reshape(-1, STEPS))

    model = make_gru(N_FEATURES, hidden, layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, epochs))
    best_loss = float("inf")
    best_state = None
    best_epoch = -1
    best_metrics: dict[str, float] = {}
    history = []
    n = x.shape[0]
    print(f"  gru training on {x.shape[0]} matches / {int(mask.sum())} "
          f"masked steps", flush=True)
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        total = 0.0
        seen = 0
        for start in range(0, n, batch):
            idx = perm[start:start + batch]
            bmask = mask[idx]
            if not bool(bmask.any()):
                continue
            opt.zero_grad(set_to_none=True)
            logits = model(xn[idx])
            loss = F.binary_cross_entropy_with_logits(
                logits[bmask], y[idx][bmask])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach()) * int(bmask.sum())
            seen += int(bmask.sum())
        sched.step()
        model.eval()
        with torch.no_grad():
            vlogits = model(xvn)
            vm = _metrics(vlogits, yv, vmask, vswmask)
        train_loss = total / max(1, seen)
        row = {"epoch": epoch, "train_loss": train_loss, **vm}
        history.append(row)
        improved = vm["loss"] < best_loss - 1e-6
        if improved:
            best_loss = vm["loss"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            best_metrics = vm
        if epoch == 1 or improved or epoch % 5 == 0:
            extra = ""
            if "switch_loss" in vm:
                extra = (f" sw={vm['switch_loss']:.4f}"
                         f"/{vm['switch_acc']:.3f}")
            print(f"  gru epoch {epoch:2d} train={train_loss:.4f} "
                  f"val={vm['loss']:.4f} acc={vm['acc']:.3f}{extra}"
                  f"{' *' if improved else ''}", flush=True)
        if epoch - best_epoch >= patience:
            print(f"  early stop at {epoch} (best {best_epoch})", flush=True)
            break
    if best_state is None:
        raise RuntimeError("GRU training produced no checkpoint")
    model.load_state_dict(best_state)
    return model, mean, std, best_metrics, history


def _save_json(path: str, data) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", choices=("mlp", "gru"), default="gru")
    ap.add_argument("--sources", default=",".join(DEFAULT_SOURCES),
                    help="comma-separated training distributions")
    ap.add_argument("--rounds", type=int, default=2,
                    help="DAgger rounds after the initial BC round")
    ap.add_argument("--bc-fields", type=int, default=6)
    ap.add_argument("--dagger-fields", type=int, default=4)
    ap.add_argument("--val-fields", type=int, default=2)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--explore", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=0,
                    help="0 = 10 for mlp / 30 for gru")
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--lr", type=float, default=0.0,
                    help="0 = 1e-4 for mlp / 1e-3 for gru")
    ap.add_argument("--batch", type=int, default=0,
                    help="0 = 8192 states for mlp / 64 matches for gru")
    ap.add_argument("--init", default="weights/rl_policy_checkpoint_wide160.json")
    ap.add_argument("--freeze-trunk", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--history-out", default="")
    ap.add_argument("--seed", type=int, default=20261043)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.rounds = min(args.rounds, 1)
        args.bc_fields = 1
        args.dagger_fields = 1
        args.val_fields = 1
        args.n = 8
        args.epochs = 2
    if args.epochs <= 0:
        args.epochs = 10 if args.student == "mlp" else 30
    if args.lr <= 0:
        args.lr = 1e-4 if args.student == "mlp" else 1e-3
    if args.batch <= 0:
        args.batch = 8192 if args.student == "mlp" else 64
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    out = args.out or f"results/distill_20261043/{args.student}_dagger_ckpt.json"
    history_out = args.history_out or out.replace(".json", "_history.json")

    sources = tuple(x.strip() for x in args.sources.split(",") if x.strip())
    if not sources:
        raise SystemExit("--sources is empty")
    bc_specs = [(s, 3000 + i * 37)
                for i, s in enumerate(sources)
                for _ in range(args.bc_fields)]
    val_specs = [(s, 5000 + i * 37)
                 for i, s in enumerate(sources)
                 for _ in range(args.val_fields)]
    print(f"student={args.student} device={device} rounds={args.rounds} "
          f"bc={args.bc_fields}/src dagger={args.dagger_fields}/src "
          f"val={args.val_fields}/src n={args.n}", flush=True)

    t0 = time.time()
    print("collecting BC data...", flush=True)
    features, targets, switches, scores = collect(
        bc_specs, args.n, None, 0.0, args.seed)
    print(f"  BC: {len(scores)} matches, mean score="
          f"{statistics.fmean(scores):.2f} ({time.time() - t0:.1f}s)",
          flush=True)
    print("collecting validation data...", flush=True)
    vf, vt, vs, vsc = collect(val_specs, args.n, None, 0.0, args.seed + 1)
    print(f"  val: {len(vsc)} matches, mean score="
          f"{statistics.fmean(vsc):.2f}", flush=True)

    all_history = []
    ckpt = None
    for rnd in range(args.rounds + 1):
        round_name = "bc" if rnd == 0 else f"dagger-{rnd}"
        print(f"training {round_name} "
              f"({len(targets) // STEPS} matches, "
              f"{len(targets)} states)...", flush=True)
        tr = time.time()
        if args.student == "mlp":
            model, mean, std, hist = train_mlp(
                features, targets, switches, args.init, args.epochs,
                args.lr, args.batch, device, args.seed + 1000 * rnd,
                args.freeze_trunk)
            meta = {
                "algorithm": "dagger-distill-rl-switch",
                "student": "mlp",
                "round": round_name,
                "matches": len(targets) // STEPS,
                "states": len(targets),
                "epochs": args.epochs,
                "lr": args.lr,
                "init": args.init,
                "freeze_trunk": args.freeze_trunk,
                "seed": args.seed,
            }
            ckpt = export_checkpoint(model, mean, std, 1.0, True, meta)
        else:
            model, mean, std, best, hist = train_gru(
                features, targets, switches, vf, vt, vs, args.hidden,
                args.layers, args.epochs, args.lr, args.batch, args.patience,
                device, args.seed + 1000 * rnd)
            meta = {
                "algorithm": "dagger-distill-rl-switch",
                "student": "gru",
                "round": round_name,
                "matches": len(targets) // STEPS,
                "states": len(targets),
                "hidden": args.hidden,
                "layers": args.layers,
                "epochs": args.epochs,
                "lr": args.lr,
                "seed": args.seed,
                "best_val": best,
            }
            ckpt = export_gru(model, mean, std, 1.0, meta)
        _save_json(out, ckpt)
        for row in hist:
            all_history.append({**row, "round": round_name})
        print(f"  wrote {out} ({time.time() - tr:.1f}s)", flush=True)

        if rnd < args.rounds:
            student = (RLPolicy(ckpt, seed=args.seed + rnd, mode="sample")
                       if args.student == "mlp"
                       else GRUPolicy(ckpt, seed=args.seed + rnd,
                                      mode="sample"))
            specs = [(s, 4000 + rnd * 1000 + i * 37)
                     for i, s in enumerate(sources)
                     for _ in range(args.dagger_fields)]
            print(f"collecting DAgger-{rnd + 1} data...", flush=True)
            nf, nt, ns, nsc = collect(
                specs, args.n, student, args.explore, args.seed + 2 + rnd)
            features.frombytes(nf.tobytes())
            targets.extend(nt)
            switches.extend(ns)
            scores.extend(nsc)
            print(f"  DAgger-{rnd + 1}: {len(nsc)} matches, aggregate "
                  f"{len(targets) // STEPS} matches, mean score="
                  f"{statistics.fmean(nsc):.2f}", flush=True)

    _save_json(history_out, {
        "student": args.student,
        "out": out,
        "rounds": args.rounds,
        "bc_fields": args.bc_fields,
        "dagger_fields": args.dagger_fields,
        "val_fields": args.val_fields,
        "n": args.n,
        "explore": args.explore,
        "hidden": args.hidden,
        "layers": args.layers,
        "lr": args.lr,
        "epochs": args.epochs,
        "seed": args.seed,
        "history": all_history,
    })
    print(f"wrote {history_out}")
    print(f"total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
