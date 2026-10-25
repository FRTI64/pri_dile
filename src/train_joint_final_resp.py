#!/usr/bin/env python3
"""Joint Final + Responsive3 hybrid training/evaluation."""
from __future__ import annotations

from paths import result_path
import argparse, json, random, statistics
from concurrent.futures import ProcessPoolExecutor

from ipd_sim import C, D, payoff
from search_common import ParamPolicy
from train_multiobj import (
    ParamResp, build_dist_field, fitness, make_policy
)
from train_multiobj import configs as multi_configs

class Hybrid:
    def __init__(self, final_cfg, resp_cfg, p_coop=1.0):
        self.f = ParamPolicy(seed=20261024, **final_cfg)
        self.r = ParamResp(seed=20261024, **resp_cfg)
        self.p = p_coop
        self.rng = random.Random(20261025)

    def reset(self):
        self.f.reset(); self.r.reset()

    def act(self, t, mine, theirs):
        a = self.f.act(t, mine, theirs)
        b = self.r.act(t, mine, theirs)
        if a == b:
            return a
        # p=1 => cooperate if either cooperates
        # p=0 => defect if either defects
        return C if self.rng.random() < self.p else D

def eval_task(task):
    final_cfg, resp_cfg, p, seed_map, n = task
    out = {}
    for dist, seeds in seed_map.items():
        vals = []
        for seed in seeds:
            field = build_dist_field(dist, seed, n)
            h = Hybrid(final_cfg, resp_cfg, p)
            total = 0
            for opp in field:
                h.reset(); opp.reset()
                mine=[]; theirs=[]; s=0
                for t in range(100):
                    x = h.act(t, mine, theirs)
                    y = opp.act(t, theirs, mine)
                    mine.append(x); theirs.append(y); s += payoff(x,y)
                total += s
            vals.append(total/n)
        out[dist] = vals
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--workers",type=int,default=24)
    ap.add_argument("--n",type=int,default=50)
    ap.add_argument("--train-per-dist",type=int,default=16)
    ap.add_argument("--val-per-dist",type=int,default=16)
    ap.add_argument("--out",default=result_path("train_joint_results.json"))
    args=ap.parse_args()
    # best configs from multiobj
    d=json.load(open(result_path('train_multiobj_results.json')))
    final_cfg=next(r['cfg'] for r in d['validation'] if r['kind']=='final')
    resp_cfg=next(r['cfg'] for r in d['validation'] if r['kind']=='resp')
    base_final={}
    base_resp={}
    rng=random.Random(8181)
    train_map={x:[rng.randrange(1<<30) for _ in range(args.train_per_dist)] for x in ("original","late","grim30","wide")}
    val_map={x:[rng.randrange(1<<30) for _ in range(args.val_per_dist)] for x in ("original","late","grim30","wide")}
    variants=[]
    for name,fc,rc in [
        ('base-final',base_final,base_resp),
        ('tuned-final',final_cfg,base_resp),
        ('base-resp',base_final,base_resp),
        ('tuned-resp',base_final,resp_cfg),
        ('tuned-both',final_cfg,resp_cfg),
    ]:
        for p in (0.0,0.25,0.5,0.75,1.0):
            if name=='base-final' and p==1.0: pass
            variants.append((f"{name}-p{p}",fc,rc,p))
    # dedupe
    seen=set(); uniq=[]
    for x in variants:
        k=(json.dumps(x[1],sort_keys=True),json.dumps(x[2],sort_keys=True),x[3])
        if k in seen: continue
        seen.add(k); uniq.append(x)
    variants=uniq
    print("variants",len(variants))
    tasks=[(fc,rc,p,train_map,args.n) for _,fc,rc,p in variants]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        outs=list(ex.map(eval_task,tasks,chunksize=1))
    rows=[]
    for (name,fc,rc,p),out in zip(variants,outs):
        f,mean,worst=fitness(out)
        rows.append({"name":name,"final":fc,"resp":rc,"p":p,
                     "train_fit":f,"train_mean":mean,"train_worst":worst})
    rows.sort(key=lambda r:r['train_fit'],reverse=True)
    print("\nTop train")
    for r in rows[:10]:
        print(f"{r['name']:22} fit={r['train_fit']:7.2f} mean={r['train_mean']:7.2f} worst={r['train_worst']:7.2f}")
    print("\nValidation")
    tasks=[(r['final'],r['resp'],r['p'],val_map,299) for r in rows[:10]]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        vouts=list(ex.map(eval_task,tasks,chunksize=1))
    vrows=[]
    for r,out in zip(rows[:10],vouts):
        f,mean,worst=fitness(out)
        row={**r,"val_fit":f,"val_mean":mean,"val_worst":worst}
        vrows.append(row)
        print(f"{r['name']:22} fit={f:7.2f} mean={mean:7.2f} worst={worst:7.2f}")
    vrows.sort(key=lambda r:r['val_fit'],reverse=True)
    with open(args.out,"w",encoding="utf-8") as fh:
        json.dump({"best_final":final_cfg,"best_resp":resp_cfg,
                   "validation":vrows,"train":rows},fh,ensure_ascii=False,indent=2)
    print("written",args.out)

if __name__=="__main__":main()
