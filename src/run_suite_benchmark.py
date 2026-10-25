#!/usr/bin/env python3
"""Compare candidates on original and additional opponent distributions."""
from __future__ import annotations

from paths import result_path
import argparse, json, random, statistics
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from ipd_sim import C, D, payoff
from search_common import ParamPolicy
from train_multiobj import ParamResp
from train_bayes_priors import ParamBayesPrior
from train_joint_final_resp import Hybrid
from triple_hybrid import Triple
import opponent_plus as op

def load_cfgs():
    m=json.load(open(result_path('train_multiobj_results.json')))
    f=next(r['cfg'] for r in m['validation'] if r['kind']=='final')
    r=next(r['cfg'] for r in m['validation'] if r['kind']=='resp')
    b=json.load(open(result_path('train_bayes_priors_results.json')))
    bc=max(b['validation'],key=lambda x:x['val_fit'])['cfg']
    j=json.load(open(result_path('train_joint_results.json')))
    h=max(j['validation'],key=lambda x:x['val_fit'])
    return f,r,bc,h

F,R,BC,H=load_cfgs()
FAMILY_MULT={
 'original':{'nice':3,'forgiving':3,'punitive':2,'defect':2,'late':2,'random':2,'wild':1,'adaptive':1,'plus':.3,'smart':.2},
 'late':{'nice':2,'forgiving':2,'punitive':1,'defect':1,'late':6,'random':1,'wild':.5,'adaptive':.5,'plus':.3,'smart':.2},
 'grim':{'nice':2,'forgiving':1,'punitive':5,'defect':2,'late':1,'random':1,'wild':.5,'adaptive':.5,'plus':.3,'smart':.2},
 'noise':{'nice':2,'forgiving':6,'punitive':1,'defect':1,'late':1,'random':3,'wild':1,'adaptive':1,'plus':1,'smart':.3},
 'complex':{'nice':1,'forgiving':1,'punitive':1,'defect':2,'late':1,'random':3,'wild':6,'adaptive':4,'plus':4,'smart':.5},
 'smart':{'nice':2,'forgiving':2,'punitive':1,'defect':1,'late':2,'random':1,'wild':1,'adaptive':2,'plus':1,'smart':6},
 'punitive':{'nice':1,'forgiving':1,'punitive':6,'defect':2,'late':1,'random':1,'wild':.5,'adaptive':.5,'plus':.5,'smart':.2},
 'random':{'nice':1,'forgiving':1,'punitive':1,'defect':1,'late':1,'random':6,'wild':1,'adaptive':1,'plus':2,'smart':.2},
 'wide':{'nice':1.5,'forgiving':1.5,'punitive':1.5,'defect':1.5,'late':1.5,'random':1.5,'wild':2,'adaptive':2,'plus':1,'smart':1},
}

class Cand:
    def __init__(self,name,p):self.name=name;self.p=p
    def reset(self):self.p.reset()
    def act(self,t,mine,theirs):
        return self.p.act(t,mine,theirs) if hasattr(self.p,'act') else self.p.move(t+1,mine,theirs)

def make(name):
    if name=='Final-base': return Cand(name,ParamPolicy(seed=20261024))
    if name=='Final-tuned': return Cand(name,ParamPolicy(seed=20261024,**F))
    if name=='Resp3-base': return Cand(name,ParamResp(seed=20261024))
    if name=='Resp3-tuned': return Cand(name,ParamResp(seed=20261024,**R))
    if name=='Bayes-tuned': return Cand(name,ParamBayesPrior(BC))
    if name=='Final+Resp3': return Cand(name,Hybrid(H['final'],H['resp'],H['p']))
    if name=='Triple-majority': return Cand(name,Triple(F,R,BC,'majority',.65))
    if name=='Harvest45': return Cand(name,ParamPolicy(seed=20261024))  # replaced below
    raise ValueError(name)

def make(name):
    if name=='Final-base': return Cand(name,ParamPolicy(seed=20261024))
    if name=='Final-tuned': return Cand(name,ParamPolicy(seed=20261024,**F))
    if name=='Resp3-base': return Cand(name,ParamResp(seed=20261024))
    if name=='Resp3-tuned': return Cand(name,ParamResp(seed=20261024,**R))
    if name=='Bayes-tuned': return Cand(name,ParamBayesPrior(BC))
    if name=='Final+Resp3': return Cand(name,Hybrid(H['final'],H['resp'],H['p']))
    if name=='Triple-majority': return Cand(name,Triple(F,R,BC,'majority',.65))
    if name=='Harvest45': return Cand(name, __import__('train_multiobj',fromlist=['ParamHarvest']).ParamHarvest(seed=20261024,HARVEST_START=45))
    raise ValueError(name)

def sample_field(suite,seed,n):
    rng=random.Random(seed);mult=FAMILY_MULT[suite]
    specs=op.ALL_SPECS;weights=[max(.001,mult.get(s.family,.5)) for s in specs]
    pop=[]
    for _ in range(n):
        s=rng.choices(specs,weights=weights,k=1)[0];o=s.factory(rng);o.name=s.name;pop.append(o)
    return pop

def task(args):
    name,suite,seed,n=args
    field=sample_field(suite,seed,n);p=make(name);total=0
    for opp in field:
        p.reset();opp.reset();mine=[];theirs=[];s=0
        for t in range(100):
            x=p.act(t,mine,theirs)
            y=opp.act(t,theirs,mine);mine.append(x);theirs.append(y);s+=payoff(x,y)
        total+=s
    return name,suite,total/n

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fields',type=int,default=12);ap.add_argument('--workers',type=int,default=24);ap.add_argument('--out',default=result_path('suite_results.json'));args=ap.parse_args()
    cands=['Final-base','Final-tuned','Resp3-base','Resp3-tuned','Bayes-tuned','Final+Resp3','Triple-majority','Harvest45']
    rng=random.Random(20261028)
    tasks=[]
    for suite in FAMILY_MULT:
        for i in range(args.fields):
            seed=rng.randrange(1<<30)
            for c in cands:tasks.append((c,suite,seed,299))
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        results=list(ex.map(task,tasks,chunksize=1))
    by=defaultdict(lambda:defaultdict(list))
    for name,suite,v in results:by[name][suite].append(v)
    print('suites',list(FAMILY_MULT))
    print(f"{'strategy':16}"+''.join(f'{s:>11}' for s in FAMILY_MULT))
    for c in cands:
        print(f'{c:16}'+''.join(f'{statistics.fmean(by[c][s]):11.2f}' for s in FAMILY_MULT))
    out={c:{s:{'mean':statistics.fmean(v),'min':min(v)} for s,v in by[c].items()} for c in cands}
    with open(args.out,'w',encoding='utf-8') as fh:json.dump(out,fh,ensure_ascii=False,indent=2)
    print('written',args.out)

if __name__=='__main__':main()
