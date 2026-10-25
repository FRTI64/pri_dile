#!/usr/bin/env python3
"""Head-to-head matchup matrix: our policies vs all 85 opponent types.

Each type is sampled with several random seeds so stochastic opponents are
averaged.  Both players' scores are recorded; `diff` is our score minus the
opponent's score in the same match.
"""
from __future__ import annotations

from ipd.paths import result_path
import argparse, csv, random, statistics
from collections import defaultdict

import ipd.core.opponents_extended as ext
from ipd.core.machine import Machine
from ipd.core.harvest import HarvestMachine
from ipd.core.early import EarlyProbeMachine
from ipd.core.bayes_plus import BayesPlusMachine
from ipd.core.opponents import Responsive3, GenerousTFT
from ipd.policies.rules import GapStartPolicy
from ipd.core.simulator import AllC, AllD, TFT, play_match

class Final(Machine):
    name="Final"
    def reset(self): super().reset()
    def act(self,t,mine,theirs): return self.move(t+1,mine,theirs)
class Harvest(Machine if False else HarvestMachine):
    name="Harvest45"
    def reset(self): super().reset()
    def act(self,t,mine,theirs): return self.move(t+1,mine,theirs)
class Harvest40(HarvestMachine):
    name="Harvest40"
    HARVEST_START=40
    def reset(self): super().reset()
    def act(self,t,mine,theirs): return self.move(t+1,mine,theirs)
class Early(EarlyProbeMachine):
    name="EarlyProbe"
class Gap(GapStartPolicy):
    name="GapStart"
class GTFT(GenerousTFT):
    name="GTFT.2-lastD"
    def __init__(self, seed=20261024):
        super().__init__(0.2, True, random.Random(seed))
    def reset(self): super().reset()
    def act(self,t,mine,theirs): return self.act_g(t,mine,theirs)
    def act_g(self,t,mine,theirs):
        # GenerousTFT already has act, but name collision; delegate.
        return GenerousTFT.act(self,t,mine,theirs)
class Resp(Responsive3):
    name="Responsive3"
class TFTp(TFT):
    name="TFT"
class AllCp(AllC):
    name="AllC"
class AllDp(AllD):
    name="AllD"

CANDIDATES = {
    "Final": lambda seed: Final(seed=seed),
    "Harvest45": lambda seed: Harvest(seed=seed),
    "Harvest40": lambda seed: Harvest40(seed=seed),
    "EarlyProbe": lambda seed: Early(seed=seed),
    "GapStart": lambda seed: Gap(1, .5),
    "BayesPlus": lambda seed: BayesPlusMachine(seed=seed),
    "Responsive3": lambda seed: Resp(rng=random.Random(seed)),
    "GTFT.2-lastD": lambda seed: GTFT(seed=seed),
    "TFT": lambda seed: TFTp(rng=random.Random(seed)),
    "AllC": lambda seed: AllCp(rng=random.Random(seed)),
    "AllD": lambda seed: AllDp(rng=random.Random(seed)),
}

def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument("--reps",type=int,default=16)
    ap.add_argument("--out",default=result_path("matchup_results.csv"))
    args=ap.parse_args(argv)
    rows=[]
    for spec in ext.TYPE_BANK:
        for rep in range(args.reps):
            for cname, maker in CANDIDATES.items():
                # build a fresh opponent and candidate from seeds
                opp_seed = 700000 + rep*1000 + hash(spec.name)%997
                opp = spec.factory(random.Random(opp_seed))
                opp.name = spec.name
                cand = maker(20261024 + rep)
                r = play_match(cand, opp)
                rows.append((cname, spec.name, spec.family,
                             r.score_a, r.score_b, r.score_a-r.score_b))
        print("done",spec.name,flush=True)
    # per candidate/type averages
    agg=defaultdict(lambda:[0,0,0,0])
    for c,t,f,a,b,d in rows:
        k=(c,t,f); agg[k][0]+=a; agg[k][1]+=b; agg[k][2]+=d; agg[k][3]+=1
    out=[]
    for (c,t,f),(sa,sb,sd,n) in agg.items():
        out.append((c,t,f,sa/n,sb/n,sd/n,n))
    out.sort(key=lambda x:(x[0],x[2],x[1]))
    with open(args.out,"w",newline="",encoding="utf-8") as fh:
        w=csv.writer(fh);w.writerow(["candidate","type","family","our","opponent","diff","reps"])
        w.writerows(out)
    # summaries
    by_cand=defaultdict(list); by_cand_fam=defaultdict(list)
    for c,t,f,our,opp,d,n in out:
        by_cand[c].append(our);by_cand_fam[(c,f)].append(our)
    print("\nOverall equal-type mean:")
    for c,vs in sorted(by_cand.items(),key=lambda kv:statistics.fmean(kv[1]),reverse=True):
        print(f"{c:14} {statistics.fmean(vs):7.2f}")
    print("\nFamily means:")
    fams=sorted({f for _,f in by_cand_fam})
    print(f"{'strategy':14}"+"".join(f"{f:>13}" for f in fams))
    for c in CANDIDATES:
        print(f"{c:14}"+"".join(f"{statistics.fmean(by_cand_fam[(c,f)]):13.1f}" for f in fams))
    print("\nBest candidate by family:")
    for f in fams:
        best=max(CANDIDATES,key=lambda c:statistics.fmean(by_cand_fam[(c,f)]))
        print(f"  {f:12} {best} {statistics.fmean(by_cand_fam[(best,f)]):.1f}")
    print("\nFinal best/worst types:")
    fr=[x for x in out if x[0]=="Final"];fr.sort(key=lambda x:x[3],reverse=True)
    for x in fr[:8]: print("  +",x[1],round(x[3],1),"opp",round(x[4],1),"diff",round(x[5],1))
    for x in fr[-8:]: print("  -",x[1],round(x[3],1),"opp",round(x[4],1),"diff",round(x[5],1))
    print("\nPer-candidate best/worst type (by our score):")
    for c in CANDIDATES:
        cs=[x for x in out if x[0]==c];cs.sort(key=lambda x:x[3],reverse=True)
        print(f"  {c:14} best={cs[0][1]}:{cs[0][3]:.1f} worst={cs[-1][1]}:{cs[-1][3]:.1f}")

if __name__=="__main__":main()
