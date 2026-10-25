#!/usr/bin/env python3
"""Tune BayesPlus prior scales, not just decision thresholds."""
from __future__ import annotations

from paths import result_path
import argparse, json, math, random, statistics, time
from concurrent.futures import ProcessPoolExecutor

from ipd_bayes_plus import BayesPlusMachine
from train_multiobj import build_dist_field, fitness
from ipd_sim import payoff, C, D

class ParamBayesPrior(BayesPlusMachine):
    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self):
        super().reset()
        c = self.cfg
        for i, (name, prior, fn) in enumerate(self.models):
            if name.startswith("LateD"):
                scale = c["late_scale"]
            elif name.startswith(("Grim", "Grudger", "Mistrust")):
                scale = c["punitive_scale"]
            elif name.startswith(("Joss", "NoisyTFT", "Random")):
                scale = c["noise_scale"]
            elif name.startswith(("Tester", "Prober", "Remorseful")):
                scale = c["probe_scale"]
            else:
                scale = c["coop_scale"]
            self.models[i] = (name, prior * scale, fn)
        self._priors = [x[1] for x in self.models]
        self._logw = [math.log(max(1e-9, w)) for w in self._priors]
        self._updated = 0
        # thresholds are instance attributes used by BayesMachine.move
        self.NONRESP_THRESHOLD = c["nonresp"]
        self.PUNITIVE_THRESHOLD = c["punitive_thr"]

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

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--configs",type=int,default=30)
    ap.add_argument("--workers",type=int,default=24)
    ap.add_argument("--n",type=int,default=50)
    ap.add_argument("--train-per-dist",type=int,default=16)
    ap.add_argument("--val-per-dist",type=int,default=16)
    ap.add_argument("--out",default=result_path("train_bayes_priors_results.json"))
    args=ap.parse_args()
    rng=random.Random(6161)
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
