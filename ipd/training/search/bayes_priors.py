#!/usr/bin/env python3
"""Tune BayesPlus prior scales, not just decision thresholds."""
from __future__ import annotations

if __package__ in (None, ""):  # allow: python ipd/training/<file>.py
    import sys
    from pathlib import Path

    for _parent in Path(__file__).resolve().parents:
        if (_parent / "pyproject.toml").is_file():
            if str(_parent) not in sys.path:
                sys.path.insert(0, str(_parent))
            break


from ipd.paths import result_path
import argparse, json, random, statistics, time
from concurrent.futures import ProcessPoolExecutor

from ipd.evaluation.distributions import build_dist_field
from ipd.policies.rules import ParamBayesPrior
from ipd.core.simulator import payoff


def fitness(out):
    allv = [x for vs in out.values() for x in vs]
    mean = statistics.fmean(allv)
    q05 = sorted(allv)[max(0, int(0.05 * len(allv)))]
    worst_dist = min(statistics.fmean(vs) for vs in out.values())
    return (0.45 * mean + 0.25 * worst_dist + 0.20 * q05
            + 0.10 * min(allv), mean, worst_dist)


def eval_task(task):
    cfg, seeds, n = task
    out = {}
    for dist, ds in seeds.items():
        vals = []
        for seed in ds:
            field = build_dist_field(dist, seed, n)
            p = ParamBayesPrior(cfg)
            total = 0
            for opp in field:
                p.reset(); opp.reset()
                mine=[]; theirs=[]; s=0
                for t in range(100):
                    x = p.move(t+1, mine, theirs)
                    y = opp.act(t, theirs, mine)
                    mine.append(x); theirs.append(y); s += payoff(x,y)
                total += s
            vals.append(total/n)
        out[dist] = vals
    return out

def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument("--seed",type=int,default=6161)
    ap.add_argument("--configs",type=int,default=30)
    ap.add_argument("--workers",type=int,default=24)
    ap.add_argument("--n",type=int,default=50)
    ap.add_argument("--train-per-dist",type=int,default=16)
    ap.add_argument("--val-per-dist",type=int,default=16)
    ap.add_argument("--out",default=result_path("train_bayes_priors_results.json"))
    args=ap.parse_args(argv)
    rng=random.Random(args.seed)
    cfgs=[]
    for _ in range(args.configs):
        cfgs.append({
            "coop_scale": 10**rng.uniform(-0.3,0.3),
            "punitive_scale": 10**rng.uniform(-0.5,0.7),
            "late_scale": 10**rng.uniform(-0.7,0.5),
            "noise_scale": 10**rng.uniform(-0.5,0.7),
            "probe_scale": 10**rng.uniform(-0.5,0.7),
            "nonresp": rng.uniform(.25,.60),
            "punitive_thr": rng.uniform(.40,.80),
        })
    train_map={d:[rng.randrange(1<<30) for _ in range(args.train_per_dist)] for d in ("original","late","grim30","wide")}
    val_map={d:[rng.randrange(1<<30) for _ in range(args.val_per_dist)] for d in ("original","late","grim30","wide")}
    tasks=[(c,train_map,args.n) for c in cfgs]
    t0=time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        outs=list(ex.map(eval_task,tasks,chunksize=1))
    print("train",round(time.time()-t0,1),flush=True)
    rows=[]
    for cfg,out in zip(cfgs,outs):
        f,mean,worst=fitness(out)
        rows.append({"cfg":cfg,"fit":f,"mean":mean,"worst":worst,"out":out})
    rows.sort(key=lambda r:r["fit"],reverse=True)
    print("\nTop train:")
    for r in rows[:10]:
        print(f"fit={r['fit']:7.2f} mean={r['mean']:7.2f} worst={r['worst']:7.2f} cfg={r['cfg']}")
    print("\nValidation:")
    tasks=[(r["cfg"],val_map,299) for r in rows[:10]]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        vouts=list(ex.map(eval_task,tasks,chunksize=1))
    vrows=[]
    for r,out in zip(rows[:10],vouts):
        f,mean,worst=fitness(out)
        vrows.append({"cfg":r["cfg"],"train_fit":r["fit"],"val_fit":f,
                      "val_mean":mean,"val_worst":worst,"val_out":out})
        print(f"val_fit={f:7.2f} mean={mean:7.2f} worst={worst:7.2f} cfg={r['cfg']}")
    # baseline
    base_cfg={"coop_scale":1,"punitive_scale":1,"late_scale":1,"noise_scale":1,
              "probe_scale":1,"nonresp":.40,"punitive_thr":.55}
    bout=eval_task((base_cfg,val_map,299)); bf,bm,bw=fitness(bout)
    print(f"baseline fit={bf:7.2f} mean={bm:7.2f} worst={bw:7.2f}")
    with open(args.out,"w",encoding="utf-8") as fh:
        json.dump({"train":rows[:20],"validation":vrows,
                   "baseline":{"cfg":base_cfg,"fit":bf,"mean":bm,"worst":bw,"out":bout}},
                  fh,ensure_ascii=False,indent=2)
    print("written",args.out)

if __name__=="__main__":main()
