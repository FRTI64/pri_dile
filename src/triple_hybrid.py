#!/usr/bin/env python3
"""Combine Final, Responsive3 and BayesPlus with a confidence gate."""
from __future__ import annotations

from paths import result_path
import argparse, json, random, statistics
from concurrent.futures import ProcessPoolExecutor

from ipd_sim import C, D, payoff
from search_common import ParamPolicy
from train_multiobj import ParamResp, build_dist_field
from train_bayes_priors import ParamBayesPrior
from train_joint_final_resp import Hybrid

class Triple:
    def __init__(self, f_cfg, r_cfg, b_cfg, mode='or', conf=.65):
        self.f = ParamPolicy(seed=20261024, **f_cfg)
        self.r = ParamResp(seed=20261024, **r_cfg)
        self.b = ParamBayesPrior(b_cfg)
        self.mode = mode
        self.conf = conf
        self.rng = random.Random(20261027)

    def reset(self):
        self.f.reset(); self.r.reset(); self.b.reset()

    def groups(self, mine, theirs):
        post = self.b._posterior(mine, theirs)
        resp = nonresp = punitive = 0.0
        for name, p in post.items():
            if name.startswith(("TFT", "sTFT", "GTFT", "Joss", "NoisyTFT", "Pavlov")):
                resp += p
            elif name.startswith(("Grim", "Grudger", "Mistrust")):
                punitive += p
            else:
                nonresp += p
        return resp, nonresp, punitive, max(post.values())

    def act(self, t, mine, theirs):
        a = self.f.act(t, mine, theirs)
        b = self.r.act(t, mine, theirs)
        c = self.b.act(t, mine, theirs)
        if self.mode == 'or':
            return D if (a == D or b == D or c == D) else C
        if self.mode == 'and':
            return C if (a == C or b == C or c == C) else D
        if self.mode == 'majority':
            return C if (a + b + c) <= 1 else D
        resp, nonresp, punitive, mx = self.groups(mine, theirs)
        if self.mode == 'bayes_conf':
            if mx >= self.conf:
                return c
            return D if (a == D or b == D) else C
        if self.mode == 'bayes_punitive':
            if punitive >= .55:
                return c
            return D if (a == D or b == D) else C
        if self.mode == 'bayes_nonresp':
            if punitive >= .55 or nonresp >= .40:
                return c
            return D if (a == D or b == D) else C
        raise ValueError(self.mode)

def load_cfgs():
    m=json.load(open(result_path('train_multiobj_results.json')))
    f=next(r['cfg'] for r in m['validation'] if r['kind']=='final')
    r=next(r['cfg'] for r in m['validation'] if r['kind']=='resp')
    b=json.load(open(result_path('train_bayes_priors_results.json')))
    bc=max(b['validation'],key=lambda x:x['val_fit'])['cfg']
    return f,r,bc

def make(kind,cfg):
    if kind=='final': return ParamPolicy(seed=20261024,**cfg)
    if kind=='resp': return ParamResp(seed=20261024,**cfg)
    if kind=='bayes': return ParamBayesPrior(cfg)
    if kind=='triple':
        return Triple(cfg['final'],cfg['resp'],cfg['bayes'],cfg['mode'],cfg['conf'])
    raise ValueError(kind)

def eval_task(args):
    name,kind,cfg,seed_map,n=args
    out={}
    for dist,seeds in seed_map.items():
        vals=[]
        for seed in seeds:
            field=build_dist_field(dist,seed,n);p=make(kind,cfg);total=0
            for opp in field:
                p.reset();opp.reset();mine=[];theirs=[];s=0
                for t in range(100):
                    x=p.act(t,mine,theirs) if hasattr(p,'act') else p.move(t+1,mine,theirs)
                    y=opp.act(t,theirs,mine);mine.append(x);theirs.append(y);s+=payoff(x,y)
                total+=s
            vals.append(total/n)
        out[dist]=vals
    return name,out

def fit(out):
    vals=[x for v in out.values() for x in v]
    return .45*statistics.fmean(vals)+.25*min(statistics.fmean(v) for v in out.values())+.20*sorted(vals)[int(.05*len(vals))]+.10*min(vals)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fields',type=int,default=12);ap.add_argument('--workers',type=int,default=24);ap.add_argument('--out',default=result_path('triple_results.json'));args=ap.parse_args()
    f,r,bc=load_cfgs()
    variants=[]
    for mode in ('or','and','majority'):
        variants.append((mode,'triple',{'final':f,'resp':r,'bayes':bc,'mode':mode,'conf':.65}))
    for mode in ('bayes_conf','bayes_punitive','bayes_nonresp'):
        for conf in ((.55,.75) if mode=='bayes_conf' else (.65,)):
            variants.append((f'{mode}_{conf}', 'triple', {'final':f,'resp':r,'bayes':bc,'mode':mode,'conf':conf}))
    variants.append(('final','final',f));variants.append(('resp','resp',r));variants.append(('bayes','bayes',bc))
    rng=random.Random(20261026)
    seed_map={d:[rng.randrange(1<<30) for _ in range(args.fields)] for d in ('original','late','grim30','wide')}
    tasks=[(name,kind,cfg,seed_map,299) for name,kind,cfg in variants]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        results=list(ex.map(eval_task,tasks,chunksize=1))
    rows=[]
    for name,out in results:
        rows.append((fit(out),name,out))
    rows.sort(reverse=True,key=lambda x:x[0])
    print('Validation:')
    for s,name,out in rows:
        print(f'{name:22} fit={s:7.2f} mean={statistics.fmean([x for v in out.values() for x in v]):7.2f} dist={{'+', '.join(f'{k}:{statistics.fmean(v):.1f}' for k,v in out.items())+'}}')
    with open(args.out,'w',encoding='utf-8') as fh:
        json.dump([{'name':n,'fit':s,'out':o} for s,n,o in rows],fh,ensure_ascii=False,indent=2)

if __name__=='__main__':main()
