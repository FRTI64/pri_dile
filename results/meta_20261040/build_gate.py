"""Build a learned match-level gate for RL-Wide160 / RL-Repair.

Usage:
  python results/meta_20261040/build_gate.py --out weights/meta_gate_wide160_repair.json

The gate observes the first K=10 rounds (played by RL-Wide160) and decides
whether the rest of the match should stay on RL-Wide160 or switch to
RL-Repair.  Training opponents come only from train-tier protocol fields,
extort_hard, hard, and train_complex; dev/test are untouched.
"""
import argparse, copy, json, random, statistics
import torch
from torch import nn

from ipd.evaluation.distributions import build_train_dist_field
from ipd.evaluation.protocol import build_protocol_field
from ipd.policies.meta import gate_features, GATE_FEATURES
from ipd.policies.registry import make_policy
from ipd.core.primitives import C, D


def payoff(me, them):
    if me == C:
        return 6 if them == C else 0
    return 10 if them == C else 1


def prefix_actions(policy, oa, ob, k):
    mine, theirs = [], []
    score = 0
    for t in range(k):
        a = policy.act(t, mine, theirs)
        b = oa.act(t, theirs, mine)
        b2 = ob.act(t, theirs, mine)
        if b != b2:
            raise RuntimeError("prefix opponent RNG drifted")
        mine.append(a); theirs.append(b)
        score += payoff(a, b)
    return mine, theirs, score


def finish(policy, opp, mine, theirs, score, k):
    for t in range(k, 100):
        a = policy.act(t, mine, theirs)
        b = opp.act(t, theirs, mine)
        mine.append(a); theirs.append(b)
        score += payoff(a, b)
    return score


def build_dataset(k=10, n=48):
    sources = [
        ("train", [build_protocol_field("train", s, n) for s in range(1200, 1208)]),
        ("train_complex", [build_train_dist_field("train_complex", s, n) for s in range(1300, 1304)]),
        ("train_axelrod", [build_train_dist_field("train_axelrod", s, n) for s in range(1400, 1402)]),
        ("extort_hard", [build_train_dist_field("extort_hard", s, n) for s in range(1500, 1504)]),
        ("hard", [build_train_dist_field("hard", s, n) for s in range(1600, 1604)]),
    ]
    prefix = make_policy("rl-wide160", mode="sample")
    pa = make_policy("rl-wide160", mode="sample")
    pb = make_policy("rl-repair", mode="sample")
    rows = []
    skipped = 0
    for source, fields in sources:
        for field in fields:
            for opp in field:
                try:
                    oa = copy.deepcopy(opp); ob = copy.deepcopy(opp)
                except Exception:
                    skipped += 1
                    continue
                oa.reset(); ob.reset()
                mine, theirs, score = prefix_actions(prefix, oa, ob, k)
                # keep the two opponent RNG streams aligned: the prefix actions
                # are identical, so this should hold.
                ma, ta, sa = list(mine), list(theirs), score
                mb, tb, sb = list(mine), list(theirs), score
                sa = finish(pa, oa, ma, ta, sa, k)
                sb = finish(pb, ob, mb, tb, sb, k)
                rows.append({
                    "source": source,
                    "opponent": getattr(opp, "name", type(opp).__name__),
                    "x": gate_features(mine, theirs),
                    "score_a": sa, "score_b": sb, "delta": sb - sa,
                })
    print(f"dataset rows={len(rows)} skipped={skipped} "
          f"switch_share={sum(r['delta'] > 2 for r in rows)/max(1,len(rows)):.3f}")
    return rows


def train(rows, out, k=10, margin=2, threshold=0.9, seed=20261040):
    rng = random.Random(seed)
    rng.shuffle(rows)
    cut = max(1, int(len(rows) * 0.85))
    train_rows, val_rows = rows[:cut], rows[cut:]
    x = torch.tensor([r["x"] for r in train_rows], dtype=torch.float32)
    y = torch.tensor([1.0 if r["delta"] > margin else 0.0
                      for r in train_rows], dtype=torch.float32)
    mean = x.mean(0); std = x.std(0).clamp_min(1e-6)
    x = (x - mean) / std
    model = nn.Linear(GATE_FEATURES, 1)
    # mild class weighting for the rarer switch class
    pos = float(y.mean())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(
        max(1.0, (1.0 - pos) / max(1e-3, pos))))
    opt = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=1e-3)
    for epoch in range(400):
        model.train(); opt.zero_grad()
        loss = loss_fn(model(x).squeeze(1), y)
        loss.backward(); opt.step()
    with torch.no_grad():
        tr_pred = (torch.sigmoid(model(x).squeeze(1)) >= 0.5).float()
        tr_acc = float((tr_pred == y).float().mean())
        xv = torch.tensor([r["x"] for r in val_rows], dtype=torch.float32)
        yv = torch.tensor([1.0 if r["delta"] > margin else 0.0
                           for r in val_rows], dtype=torch.float32)
        xv = (xv - mean) / std
        pv = torch.sigmoid(model(xv).squeeze(1))
        va_pred = (pv >= 0.5).float()
        va_acc = float((va_pred == yv).float().mean())
    data = {
        "format": "ipd-meta-gate-v1",
        "k": int(k),
        "n_features": GATE_FEATURES,
        "threshold": float(threshold),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "w": model.weight.detach().squeeze(0).tolist(),
        "b": float(model.bias.detach()),
        "meta": {
            "train_rows": len(train_rows), "val_rows": len(val_rows),
            "train_acc": tr_acc, "val_acc": va_acc,
            "switch_share": float(y.mean()), "margin": margin,
            "source": "train_v4 + train_complex + train_axelrod + extort_hard + hard",
        },
    }
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    print(f"train_acc={tr_acc:.3f} val_acc={va_acc:.3f} switch_share={pos:.3f}")
    print(f"wrote {out}")
    return data


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--margin", type=float, default=2.0)
    ap.add_argument("--threshold", type=float, default=0.9,
                    help="switch probability threshold; 0.9 is the calibrated value")
    ap.add_argument("--out", default="weights/meta_gate_wide160_repair.json")
    ap.add_argument("--dataset-out", default=None)
    args = ap.parse_args()
    rows = build_dataset(args.k, args.n)
    if args.dataset_out:
        with open(args.dataset_out, "w", encoding="utf-8") as fh:
            json.dump(rows, fh)
    train(rows, args.out, args.k, args.margin, args.threshold)
