#!/usr/bin/env python3
"""Evaluate strategies against a bank of deliberately simple opponents.

Simple = memory-one / short-history rules, no Bayesian adaptive, no random
FSM, no zero-determinant/extortion, no learned policies.
"""
from __future__ import annotations
import argparse, copy, random, statistics, math
import ipd.core.opponents_extended as ext
from ipd.core.machine import Machine
from ipd.core.harvest import HarvestMachine
from ipd.core.early import EarlyProbeMachine
from ipd.policies.rules import GapStartPolicy
from ipd.core.simulator import C, D, payoff, play_match

SIMPLE = [
    "AllC", "AllC-lastD", "TFT", "TFT-lastD", "TFT-last2D", "TFT-last5D",
    "sTFT", "TFT2", "TFT2-lastD", "ContriteTFT", "TFTForgiveOnce",
    "TFT-delay2", "FirmButFair", "TTFT", "Pavlov", "WSLS2",
    "GTFT.01", "GTFT.05", "GTFT.10", "GTFT.20", "GTFT.33", "GTFT.50",
    "NoisyTFT.02", "NoisyTFT.05", "NoisyTFT.10", "NoisyTFT.20",
    "Joss.02", "Joss.05", "Joss.10", "Joss.20", "ForgiveTFT1",
    "ForgiveTFT2", "TwoStrike2", "TwoStrike3", "Grim", "GrimForgive.05",
    "GrimForgive.15", "Grudger2", "Grudger3", "Mistrust", "Gradual",
    "OmegaTFT-lite", "Bully", "AllD", "EarlyD1", "EarlyD10", "EarlyD25",
    "LateD5", "LateD10", "LateD15", "LateD20", "LateD25", "LateD30",
    "LateD40", "LateD50", "Tester", "Prober", "RemorsefulProber",
    "Random.05", "Random.10", "Random.20", "Random.30", "Random.50",
    "Random.70", "Random.80", "Random.90", "Alternator", "AlternatorD",
    "Periodic3", "Periodic4", "Rand3->TFT", "Majority2", "Majority3",
    "Majority5",
]

def make_weights(scenario):
    w = {n: 0.5 for n in SIMPLE}
    if scenario == "simple_balanced":
        # roughly balanced across strategy families
        for n in SIMPLE:
            fam = ext.TYPE_BY_NAME[n].family
            w[n] = {"nice": 3.0, "forgiving": 2.5, "punitive": 2.0,
                    "defect": 2.0, "late": 2.0, "random": 2.0,
                    "adaptive": 0.0}.get(fam, 0.5)
    elif scenario == "simple_naive":
        w = {n: 0.2 for n in SIMPLE}
        w.update({"AllC": 20, "TFT": 20, "TFT2": 8, "GTFT.05": 8,
                  "GTFT.10": 5, "Pavlov": 5, "LateD10": 6, "LateD20": 6,
                  "LateD25": 6, "LateD30": 5, "LateD40": 5, "LateD50": 4,
                  "Grim": 5, "AllD": 4, "Random.50": 4})
    elif scenario == "simple_late":
        w = {n: 0.2 for n in SIMPLE}
        w.update({"AllC": 10, "TFT": 20, "TFT2": 6, "GTFT.05": 8,
                  "GTFT.10": 5, "Pavlov": 5, "Grim": 5, "AllD": 4,
                  "Random.50": 4, "Tester": 3, "Prober": 3,
                  "LateD5": 3, "LateD10": 5, "LateD15": 4, "LateD20": 5,
                  "LateD25": 6, "LateD30": 4, "LateD40": 4, "LateD50": 4})
    elif scenario == "simple_punitive":
        w = {n: 0.2 for n in SIMPLE}
        w.update({"AllC": 5, "TFT": 22, "TFT2": 8, "GTFT.05": 8,
                  "GTFT.10": 5, "Pavlov": 5, "Grim": 20, "Mistrust": 6,
                  "Grudger2": 4, "Grudger3": 2, "AllD": 6, "Random.50": 4,
                  "LateD10": 3, "LateD25": 3})
    return w

def sample_pop(weights, n, rng):
    specs = [ext.TYPE_BY_NAME[x] for x in SIMPLE]
    ws = [max(0, weights.get(x, 0)) for x in SIMPLE]
    pop=[]
    for _ in range(n):
        s = rng.choices(specs, weights=ws, k=1)[0]
        o = s.factory(rng); o.name=s.name; pop.append(o)
    return pop

class Final(Machine):
    name="Final"
    def reset(self): super().reset()
    def act(self,t,mine,theirs): return self.move(t+1,mine,theirs)
class Harv(HarvestMachine):
    name="Harvest45"
    def reset(self): super().reset()
    def act(self,t,mine,theirs): return self.move(t+1,mine,theirs)
class Early(EarlyProbeMachine):
    name="EarlyProbe"
class Gap(GapStartPolicy):
    name="GapStart(.5)"

CANDIDATES={
 "Final":lambda:Final(seed=20261024),
 "Harvest45":lambda:Harv(seed=20261024),
 "EarlyProbe":lambda:Early(seed=20261024),
 "GapStart":lambda:Gap(start=1,threshold=.5),
}

def eval_field(cand, field):
    total=0
    for opp in field:
        cand.reset(); opp.reset(); mine=[];theirs=[];s=0
        for t in range(100):
            x=cand.act(t,mine,theirs); y=opp.act(t,theirs,mine)
            mine.append(x);theirs.append(y)
            s += payoff(x,y)
        total += s
    return total

def rank_field(cand, field):
    pop=[cand]+list(field); n=len(pop); sc=[0]*n
    for i in range(n):
        for j in range(i+1,n):
            r=play_match(pop[i],pop[j]); sc[i]+=r.score_a; sc[j]+=r.score_b
    order=sorted(range(n),key=lambda i:sc[i],reverse=True)
    return order.index(0)+1, sc[0]

def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument("--fields",type=int,default=100);ap.add_argument("--rank",type=int,default=0);ap.add_argument("--seed",type=int,default=2026)
    args=ap.parse_args(argv)
    for scen in ["simple_balanced","simple_naive","simple_late","simple_punitive"]:
        weights=make_weights(scen)
        master=random.Random(args.seed+len(scen)); fields=[]
        for _ in range(args.fields):
            fields.append(sample_pop(weights,299,random.Random(master.random())))
        print('\n',scen)
        for name,maker in CANDIDATES.items():
            vals=[]
            for f in fields:
                vals.append(eval_field(maker(),copy.deepcopy(f))/299)
            print(f'{name:12} mean={statistics.fmean(vals):7.2f} med={statistics.median(vals):7.2f} p05={sorted(vals)[int(.05*len(vals))]:7.2f} min={min(vals):7.2f}')
        if args.rank:
            for name,maker in CANDIDATES.items():
                ranks=[];scores=[]
                for f in fields[:args.rank]:
                    r,s=rank_field(maker(),copy.deepcopy(f));ranks.append(r);scores.append(s)
                print(f' {name:12} rankmean={statistics.fmean(ranks):6.2f} rank1={sum(x==1 for x in ranks)}/{len(ranks)} top10={sum(x<=10 for x in ranks)}/{len(ranks)} worst={max(ranks)} score={statistics.fmean(scores)/299:7.2f}')

if __name__=="__main__":main()
