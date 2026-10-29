"""Train a cost-sensitive match-level gate from counterfactual data.

Objective: maximize expected score p*score_b + (1-p)*score_a, with:
  * a switching margin (only switch when Repair is better by > margin),
  * higher sample weight for complex families (avoid false positives),
  * L2 regularization.
The gate is a logistic model, same 25-feature contract as RL-Gate.
"""
import argparse, json, math, random
import torch
from torch import nn

from ipd.evaluation.protocol import COMPLEX_FAMILIES
from ipd.policies.meta import GATE_FEATURES


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset", default="results/meta_20261040/gate_dataset.json")
    ap.add_argument("--out", default="weights/meta_gate_cs_wide160_repair.json")
    ap.add_argument("--margin", type=float, default=5.0)
    ap.add_argument("--complex-weight", type=float, default=2.5)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=20261041)
    args=ap.parse_args()
    rows=json.load(open(args.dataset))
    rng=random.Random(args.seed); rng.shuffle(rows)
    cut=max(1,int(len(rows)*0.85)); tr,va=rows[:cut],rows[cut:]
    def weight(r):
        fam=str(r.get("opponent","")).split(":",1)[0]
        if r.get("source")=="train_complex" or fam in COMPLEX_FAMILIES:
            return args.complex_weight
        return 1.0
    x=torch.tensor([r["x"] for r in tr],dtype=torch.float32)
    delta=torch.tensor([r["score_b"]-r["score_a"] for r in tr],dtype=torch.float32)
    w=torch.tensor([weight(r) for r in tr],dtype=torch.float32)
    mean=x.mean(0); std=x.std(0).clamp_min(1e-6)
    xn=(x-mean)/std
    model=nn.Linear(GATE_FEATURES,1)
    opt=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.l2)
    for epoch in range(args.epochs):
        opt.zero_grad()
        p=torch.sigmoid(model(xn).squeeze(1))
        # maximize weighted expected utility; subtract a switching margin
        target=delta-args.margin
        loss=-(p*target*w).mean()/100.0
        loss.backward(); opt.step()
    def report(rows,label):
        xx=torch.tensor([r["x"] for r in rows],dtype=torch.float32)
        dd=torch.tensor([r["score_b"]-r["score_a"] for r in rows],dtype=torch.float32)
        pp=torch.sigmoid(model(((xx-mean)/std)).squeeze(1)).detach()
        util=(torch.tensor([r["score_a"] for r in rows],dtype=torch.float32)+pp*dd)
        always_a=torch.tensor([r["score_a"] for r in rows],dtype=torch.float32)
        always_b=torch.tensor([r["score_b"] for r in rows],dtype=torch.float32)
        print(f"{label}: n={len(rows)} switch={float((pp>=0.5).float().mean()):.3f} "
              f"util={float(util.mean()):.2f} alwaysA={float(always_a.mean()):.2f} "
              f"alwaysB={float(always_b.mean()):.2f} "
              f"delta={(float(dd.mean())):+.2f}")
    report(tr,"train"); report(va,"val")
    data={
        "format":"ipd-meta-gate-v1","k":10,"n_features":GATE_FEATURES,
        "threshold":0.5,
        "mean":mean.tolist(),"std":std.tolist(),
        "w":model.weight.detach().squeeze(0).tolist(),
        "b":float(model.bias.detach()),
        "meta":{
            "objective":"cost_sensitive_utility","margin":args.margin,
            "complex_weight":args.complex_weight,"l2":args.l2,
            "train_rows":len(tr),"val_rows":len(va),
            "switch_share":float((torch.sigmoid(model(xn).squeeze(1)).detach()>=0.5).float().mean()),
        },
    }
    json.dump(data,open(args.out,"w"),ensure_ascii=False,indent=2)
    print("wrote",args.out)

if __name__=="__main__":
    main()
