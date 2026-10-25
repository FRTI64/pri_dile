#!/usr/bin/env python3
"""Final blind test on completely unseen 300-player fields."""
from __future__ import annotations
import argparse, copy, json, random, statistics
from ipd_sim import C, D, payoff, play_match
from ipd_strategy import Machine
from search_common import ParamPolicy, ParamHarvest
from train_multiobj import ParamResp
from train_bayes_priors import ParamBayesPrior
from train_joint_final_resp import Hybrid
from p2_skeleton_search import GapStartPolicy
from train_wide import sample_wide
from neural_policy import NeuralPolicy
from gru_policy import GRUPolicy
from adversarial_policy import AdversarialPolicy

class Cand:
    def __init__(self, name, policy):
        self.name = name
        self.policy = policy
    def reset(self): self.policy.reset()
    def act(self, t, mine, theirs):
        if hasattr(self.policy, "act"):
            return self.policy.act(t, mine, theirs)
        return self.policy.move(t + 1, mine, theirs)

def load_cfgs():
    m=json.load(open('train_multiobj_results.json'))
    f=next(r['cfg'] for r in m['validation'] if r['kind']=='final')
    r=next(r['cfg'] for r in m['validation'] if r['kind']=='resp')
    h=next(r['cfg'] for r in m['validation'] if r['kind']=='harvest')
    j=json.load(open('train_joint_results.json'))
    hb=max(j['validation'], key=lambda x:x['val_fit'])
    b=json.load(open('train_bayes_priors_results.json'))
    bb=max(b['validation'], key=lambda x:x['val_fit'])
    return f,r,h,hb,bb

def candidates(include_neural=False, neural_weights=None,
               neural_mode='sample', neural_arch='mlp', neural_config=None):
    f,r,h,hb,bb=load_cfgs()
    out=[]
    out.append(Cand('Final-base', ParamPolicy(seed=20261024)))
    out.append(Cand('Final-tuned', ParamPolicy(seed=20261024, **f)))
    out.append(Cand('Resp-base', ParamResp(seed=20261024)))
    out.append(Cand('Resp-tuned', ParamResp(seed=20261024, **r)))
    out.append(Cand('Hybrid-best', Hybrid(hb['final'], hb['resp'], hb['p'])))
    out.append(Cand('Harvest45', ParamHarvest(seed=20261024, HARVEST_START=45)))
    out.append(Cand('Harvest-tuned', ParamHarvest(seed=20261024, **h)))
    out.append(Cand('GapStart', GapStartPolicy(1,.5)))
    out.append(Cand('Bayes-base', ParamBayesPrior({"coop_scale":1,"punitive_scale":1,"late_scale":1,"noise_scale":1,"probe_scale":1,"nonresp":.40,"punitive_thr":.55})))
    out.append(Cand('Bayes-tuned', ParamBayesPrior(bb['cfg'])))
    if include_neural:
        if neural_weights is None:
            neural_weights=('gru_policy_weights.json' if neural_arch=='gru'
                            else 'neural_policy_weights.json')
        if neural_arch=='gru':
            out.append(Cand(f'GRU-{neural_mode}',
                            GRUPolicy(neural_weights, seed=20261024,
                                      mode=neural_mode)))
        elif neural_arch=='adversarial':
            out.append(Cand(f'Adv-{neural_mode}',
                            AdversarialPolicy.from_config(
                                neural_config or 'adversarial_policy_config.json',
                                seed=20261024, mode=neural_mode)))
        else:
            out.append(Cand(f'Neural-{neural_mode}',
                            NeuralPolicy(neural_weights, seed=20261024,
                                         mode=neural_mode)))
    return out

def eval_field(cand, field):
    total=0
    for opp in field:
        cand.reset(); opp.reset()
        mine=[]; theirs=[]; s=0
        for t in range(100):
            x=cand.act(t,mine,theirs); y=opp.act(t,theirs,mine)
            mine.append(x);theirs.append(y);s+=payoff(x,y)
        total+=s
    return total

def rank_field(cand, field):
    pop=[cand]+list(field);n=len(pop);scores=[0]*n
    for i in range(n):
        for j in range(i+1,n):
            r=play_match(pop[i],pop[j]);scores[i]+=r.score_a;scores[j]+=r.score_b
    order=sorted(range(n),key=lambda i:scores[i],reverse=True)
    return order.index(0)+1,scores[0]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--fields",type=int,default=10)
    ap.add_argument("--seed",type=int,default=20261025)
    ap.add_argument("--neural",action="store_true",help="include the exported neural policy")
    ap.add_argument("--neural-arch",default="mlp",choices=("mlp","gru","adversarial"))
    ap.add_argument("--neural-weights",default=None)
    ap.add_argument("--neural-config",default=None)
    ap.add_argument("--neural-mode",default="sample",choices=("sample","argmax"))
    args=ap.parse_args()
    cands=candidates(args.neural,args.neural_weights,args.neural_mode,args.neural_arch,args.neural_config)
    print("blind seed base",args.seed)
    all_ranks={c.name:[] for c in cands}
    all_scores={c.name:[] for c in cands}
    for i in range(args.fields):
        seed=args.seed+i*7919
        for c in cands:
            # fresh field per candidate so random opponents start identically
            f,_,_=sample_wide(seed,299,.15)
            r,s=rank_field(c,f);all_ranks[c.name].append(r);all_scores[c.name].append(s)
        print("field",i+1,"done",flush=True)
    print("\nBlind results (mean over",args.fields,"unseen 300-player fields):")
    for c in cands:
        print(f"{c.name:14} score={statistics.fmean(all_scores[c.name])/299:7.2f} "
              f"rank={statistics.fmean(all_ranks[c.name]):6.2f} "
              f"rank1={sum(x==1 for x in all_ranks[c.name])}/{args.fields} "
              f"top10={sum(x<=10 for x in all_ranks[c.name])}/{args.fields} "
              f"worst={max(all_ranks[c.name])}")

if __name__=="__main__":main()
