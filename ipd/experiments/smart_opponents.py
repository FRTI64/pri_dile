#!/usr/bin/env python3
"""Run experiments with our own smart strategies added to the opponent bank.

This tests an "arms race" field: simple bots + our Final/Harvest/Early/Gap/
Bayes/Responsive strategies as opponents.
"""
from __future__ import annotations
import argparse, copy, random, statistics, math
from collections import Counter
import ipd.core.opponents_extended as ext
from ipd.core.simulator import Strategy, C, D, payoff, play_match
from ipd.core.machine import Machine
from ipd.core.harvest import HarvestMachine
from ipd.core.early import EarlyProbeMachine
from ipd.core.bayes_plus import BayesPlusMachine
from ipd.core.opponents import Responsive3
from ipd.policies.rules import GapStartPolicy
from ipd.evaluation.distributions import ORIGINAL_SCENARIO_WEIGHTS

class SmartAdapter(Strategy):
    def __init__(self, name, inner):
        self.name = name
        self.inner = inner
    def reset(self):
        self.inner.reset()
    def act(self, t, mine, theirs):
        if hasattr(self.inner, "act"):
            return self.inner.act(t, mine, theirs)
        return self.inner.move(t + 1, mine, theirs)

def _final(r):
    return SmartAdapter("Final", Machine(seed=r.randrange(1 << 30)))
def _harvest45(r):
    return SmartAdapter("Harvest45", HarvestMachine(seed=r.randrange(1 << 30)))
def _harvest40(r):
    h = HarvestMachine(seed=r.randrange(1 << 30)); h.HARVEST_START = 40
    return SmartAdapter("Harvest40", h)
def _early(r):
    return SmartAdapter("EarlyProbe", EarlyProbeMachine(seed=r.randrange(1 << 30)))
def _gap(r):
    return SmartAdapter("GapStart", GapStartPolicy(1, .5))
def _bayes(r):
    return SmartAdapter("BayesPlus", BayesPlusMachine())
def _resp(r):
    return SmartAdapter("Responsive3", Responsive3(rng=random.Random(r.randrange(1 << 30))))

SMART = [
    ext.TypeSpec("Final", "smart", 4, _final),
    ext.TypeSpec("Harvest45", "smart", 3, _harvest45),
    ext.TypeSpec("Harvest40", "smart", 2, _harvest40),
    ext.TypeSpec("EarlyProbe", "smart", 2, _early),
    ext.TypeSpec("GapStart", "smart", 3, _gap),
    ext.TypeSpec("BayesPlus", "smart", 2, _bayes),
    ext.TypeSpec("Responsive3", "smart", 2, _resp),
]
SMART_BY_NAME = {s.name: s for s in SMART}

class FinalCand(Machine):
    name="Final"
    def reset(self): super().reset()
    def act(self,t,mine,theirs): return self.move(t+1,mine,theirs)
class HarvestCand(HarvestMachine):
    name="Harvest45"
    def reset(self): super().reset()
    def act(self,t,mine,theirs): return self.move(t+1,mine,theirs)
class EarlyCand(EarlyProbeMachine):
    name="EarlyProbe"
class GapCand(GapStartPolicy):
    name="GapStart"
class BayesCand(BayesPlusMachine):
    name="BayesPlus"
class RespCand(Responsive3):
    name="Responsive3"

CANDIDATES={
 "Final":lambda:FinalCand(seed=20261024),
 "Harvest45":lambda:HarvestCand(seed=20261024),
 "EarlyProbe":lambda:EarlyCand(seed=20261024),
 "GapStart":lambda:GapCand(1,.5),
 "BayesPlus":lambda:BayesCand(),
 "Responsive3":lambda:RespCand(rng=random.Random(20261024)),
}

def sample_field(scenario, smart_weight, n, rng):
    # base 85 types under a scenario
    base_specs, base_w = ext.effective_weights(scenario)
    specs = list(base_specs) + list(SMART)
    weights = list(base_w) + [smart_weight * s.weight for s in SMART]
    pop = []
    for _ in range(n):
        s = rng.choices(specs, weights=weights, k=1)[0]
        o = s.factory(rng)
        o.name = s.name
        pop.append(o)
    return pop

def eval_field(cand, field):
    total=0
    for opp in field:
        cand.reset(); opp.reset(); mine=[];theirs=[];s=0
        for t in range(100):
            x=cand.act(t,mine,theirs); y=opp.act(t,theirs,mine)
            mine.append(x);theirs.append(y);s+=payoff(x,y)
        total+=s
    return total

def rank_field(cand, field):
    pop=[cand]+list(field);N=len(pop);scores=[0]*N
    for i in range(N):
        for j in range(i+1,N):
            r=play_match(pop[i],pop[j]);scores[i]+=r.score_a;scores[j]+=r.score_b
    order=sorted(range(N),key=lambda i:scores[i],reverse=True)
    return order.index(0)+1,scores[0]

def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument("--fields",type=int,default=80)
    ap.add_argument("--rank",type=int,default=15)
    ap.add_argument("--smart-weight",type=float,default=25.0)
    ap.add_argument("--seed",type=int,default=2026)
    args=ap.parse_args(argv)
    scenarios=["original"]  # scenario choice handled below
    master=random.Random(args.seed)
    # Sample scenarios from the original prior, then add smart opponents.
    scen_names=list(ORIGINAL_SCENARIO_WEIGHTS)
    scen_w=list(ORIGINAL_SCENARIO_WEIGHTS.values())
    fields=[];sc_counts=Counter()
    for _ in range(args.fields):
        scen=master.choices(scen_names,weights=scen_w,k=1)[0]
        sc_counts[scen]+=1
        fields.append(sample_field(scen,args.smart_weight,299,random.Random(master.random())))
    print("scenario counts",dict(sc_counts))
    print("smart weight",args.smart_weight)
    for name,maker in CANDIDATES.items():
        vals=[eval_field(maker(),copy.deepcopy(f))/299 for f in fields]
        print(f"{name:12} mean={statistics.fmean(vals):7.2f} med={statistics.median(vals):7.2f} p05={sorted(vals)[int(.05*len(vals))]:7.2f} min={min(vals):7.2f}")
    if args.rank:
        print("\nrank")
        for name,maker in CANDIDATES.items():
            ranks=[];scores=[]
            for f in fields[:args.rank]:
                r,s=rank_field(maker(),copy.deepcopy(f));ranks.append(r);scores.append(s)
            print(f"{name:12} meanrank={statistics.fmean(ranks):6.2f} rank1={sum(x==1 for x in ranks)}/{len(ranks)} top10={sum(x<=10 for x in ranks)}/{len(ranks)} worst={max(ranks)} score={statistics.fmean(scores)/299:7.2f}")

if __name__=="__main__":main()
