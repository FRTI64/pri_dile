#!/usr/bin/env python3
"""Matchup matrix for base/tuned Final, Responsive3, BayesPlus and hybrid."""
from __future__ import annotations

from paths import result_path
import argparse, csv, json, random, statistics
from collections import defaultdict

import ipd_extended as ext
from ipd_sim import payoff, play_match
from search_common import ParamPolicy
from train_multiobj import ParamResp
from train_bayes_priors import ParamBayesPrior
from train_joint_final_resp import Hybrid

class Wrap:
    def __init__(self, name, p): self.name=name; self.p=p
    def reset(self): self.p.reset()
    def act(self,t,mine,theirs):
        if hasattr(self.p,'act'): return self.p.act(t,mine,theirs)
        return self.p.move(t+1,mine,theirs)

m=json.load(open(result_path('train_multiobj_results.json')))
f_base={}; r_base={}
f_tuned=next(r['cfg'] for r in m['validation'] if r['kind']=='final')
r_tuned=next(r['cfg'] for r in m['validation'] if r['kind']=='resp')
b=json.load(open(result_path('train_bayes_priors_results.json')))
bay_tuned=max(b['validation'],key=lambda x:x['val_fit'])['cfg']
bay_base={"coop_scale":1,"punitive_scale":1,"late_scale":1,"noise_scale":1,"probe_scale":1,"nonresp":.40,"punitive_thr":.55}
j=json.load(open(result_path('train_joint_results.json')))
hy=max(j['validation'],key=lambda x:x['val_fit'])

CANDS={
 'Final-base': lambda: Wrap('Final-base', ParamPolicy(seed=20261024)),
 'Final-tuned': lambda: Wrap('Final-tuned', ParamPolicy(seed=20261024, **f_tuned)),
 'Resp3-base': lambda: Wrap('Resp3-base', ParamResp(seed=20261024)),
 'Resp3-tuned': lambda: Wrap('Resp3-tuned', ParamResp(seed=20261024, **r_tuned)),
 'Bayes-base': lambda: Wrap('Bayes-base', ParamBayesPrior(bay_base)),
 'Bayes-tuned': lambda: Wrap('Bayes-tuned', ParamBayesPrior(bay_tuned)),
 'Final+Resp3': lambda: Wrap('Final+Resp3', Hybrid(hy['final'],hy['resp'],hy['p'])),
}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--reps',type=int,default=16);ap.add_argument('--out',default=result_path('matchup_tuned_results.csv'));args=ap.parse_args()
    rows=defaultdict(lambda:[0,0,0])
    for spec in ext.TYPE_BANK:
        for rep in range(args.reps):
            for cname,maker in CANDS.items():
                opp=spec.factory(random.Random(800000+rep*1000+hash(spec.name)%997));opp.name=spec.name
                c=maker(); r=play_match(c,opp)
                k=(cname,spec.name,spec.family)
                rows[k][0]+=r.score_a;rows[k][1]+=r.score_b;rows[k][2]+=1
        print('done',spec.name,flush=True)
    out=[]
    for (c,t,f),(a,b,n) in rows.items():out.append((c,t,f,a/n,b/n,n))
    out.sort(key=lambda x:(x[0],x[2],x[1]))
    with open(args.out,'w',newline='',encoding='utf-8') as fh:
        w=csv.writer(fh);w.writerow(['candidate','type','family','our','opponent','reps']);w.writerows(out)
    by=defaultdict(list)
    for c,t,f,a,b,n in out:by[c].append(a)
    print('\nEqual-type mean')
    for c,v in sorted(by.items(),key=lambda kv:statistics.fmean(kv[1]),reverse=True):print(f'{c:14}{statistics.fmean(v):7.2f}')

if __name__=='__main__':main()
