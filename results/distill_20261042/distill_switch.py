"""Distill the RL-Switch meta policy into a single 160/128 MLP."""
import argparse, json, random, statistics
import torch
from torch import nn

from ipd.core.primitives import C, D
from ipd.evaluation.distributions import build_train_dist_field
from ipd.evaluation.protocol import build_protocol_field
from ipd.policies.neural import extract_features
from ipd.policies.registry import make_policy
from ipd.training.rl import (
    export_checkpoint, load_initial_weights, make_model,
)


def build_data(kind, fields, n):
    sources = [
        ("train", [build_protocol_field("train", s, n) for s in range(2000, 2000 + fields)]),
        ("train_complex", [build_train_dist_field("train_complex", s, n) for s in range(2100, 2100 + fields)]),
        ("extort_hard", [build_train_dist_field("extort_hard", s, n) for s in range(2200, 2200 + fields)]),
        ("periodic_hard", [build_train_dist_field("periodic_hard", s, n) for s in range(2300, 2300 + fields)]),
        ("hard", [build_train_dist_field("hard", s, n) for s in range(2400, 2400 + fields)]),
    ]
    teacher = make_policy("rl-switch", mode="sample")
    xs, ys = [], []
    for source, field_list in sources:
        for field in field_list:
            for opp in field:
                teacher.reset()
                opp.reset()
                mine, theirs = [], []
                for t in range(100):
                    forced = (t == 0 or D not in theirs or t >= 99)
                    f = extract_features(mine, theirs)
                    a = teacher.act(t, mine, theirs)
                    if not forced:
                        pol = teacher.alt if teacher.switched else teacher.base
                        xs.append(f)
                        ys.append(pol.predict_proba(mine, theirs))
                    b = opp.act(t, theirs, mine)
                    mine.append(a); theirs.append(b)
    print(f"states={len(xs)} switch_share={statistics.fmean(1.0 if y >= 0.5 else 0.0 for y in ys):.3f}")
    return xs, ys


def train(xs, ys, out, epochs=4, lr=1e-4, batch=8192, seed=20261042):
    torch.manual_seed(seed)
    mean = [0.0] * 85
    std = [1.0] * 85
    model = make_model([160, 128])
    load_initial_weights(model, "weights/rl_policy_checkpoint_wide160.json",
                         load_value=True)
    model.to("cuda")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x = torch.tensor(xs, dtype=torch.float32).to("cuda")
    y = torch.tensor(ys, dtype=torch.float32).to("cuda")
    n = len(xs); idx = torch.randperm(n, device="cuda")
    for epoch in range(epochs):
        model.train()
        total = 0.0; count = 0
        for start in range(0, n, batch):
            sel = idx[start:start + batch]
            xb, yb = x[sel], y[sel]
            logit = model.policy(model.policy_trunk(xb)).squeeze(1)
            loss = nn.functional.binary_cross_entropy_with_logits(logit, yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            total += float(loss) * len(sel); count += len(sel)
        print(f"epoch {epoch+1} bce={total/count:.5f}")
    meta = {"algorithm": "distill-rl-switch", "epochs": epochs,
            "states": n, "lr": lr}
    ckpt = export_checkpoint(model, mean, std, 1.0, True, meta)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(ckpt, fh, ensure_ascii=False, indent=2)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", type=int, default=6)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="results/distill_20261042/distill_switch_ckpt.json")
    args = ap.parse_args()
    xs, ys = build_data(args.fields, args.fields, args.n)
    train(xs, ys, args.out, args.epochs, args.lr)


if __name__ == "__main__":
    main()
