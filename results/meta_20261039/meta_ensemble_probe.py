import random, statistics
from ipd.policies.registry import make_policy
from ipd.policies.rules import GapStartPolicy
from ipd.core.opponents import AllC, AllD, Grim, Joss, RandomMoves, TFT
from ipd.core.opponents.memory import RandomFSM
from ipd.core.opponents.zero_determinant import ExtortZD
from ipd.evaluation.hard2 import Kamikaze
from ipd.evaluation.protocol import build_protocol_field
from ipd.evaluation.common import run_match, evaluate_field_details
from ipd.core.primitives import C, D

class Ensemble:
    def __init__(self, specs, seed=123, safe=True):
        self.policies=[make_policy(k,mode='sample') for k,w in specs]
        self.weights=[w for k,w in specs]
        self.rng=random.Random(seed); self.safe=safe
    def reset(self):
        for p in self.policies: p.reset()
    def act(self,t,mine,theirs):
        if t>=99: return D
        if self.safe and (t==0 or D not in theirs): return C
        ps=[p.predict_proba(mine,theirs) for p in self.policies]
        p=sum(w*x for w,x in zip(self.weights,ps))/sum(self.weights)
        return D if self.rng.random()<p else C
    def observe(self,*a,**k): pass

def eval_weak(factory, of, reps=16):
    scores=[]
    for r in range(reps):
        p=factory(); opp=of(random.Random(1000+r))
        scores.append(run_match(p,opp).score)
    return statistics.fmean(scores)

def eval_tier(factory, tier, fields=3, n=512):
    vals=[]
    for seed in range(20261039,20261039+fields):
        field=build_protocol_field(tier,seed,n); p=factory()
        stats,by=evaluate_field_details(p,field); vals.append(stats.mean)
    return statistics.fmean(vals)

variants={
 'wide160': lambda: make_policy('rl-wide160',mode='sample'),
 'w160+w': lambda: Ensemble([('rl-wide160',1),('rl-wide',1)]),
 'w160+2w': lambda: Ensemble([('rl-wide160',2),('rl-wide',1)]),
 'w160+rep': lambda: Ensemble([('rl-wide160',4),('rl-repair',1)]),
 'w160+w+rep': lambda: Ensemble([('rl-wide160',3),('rl-wide',1),('rl-repair',1)]),
 'w160+pat': lambda: Ensemble([('rl-wide160',3),('rl-pattern',1)]),
}
print('tier means')
for name,f in variants.items():
    print(name,'param',round(eval_tier(f,'dev_param'),2),'complex',round(eval_tier(f,'dev_complex'),2),'ood',round(eval_tier(f,'dev_ood'),2))
print('\nweak')
ops=[('TFT',lambda r:TFT(r)),('Grim',lambda r:Grim(r)),('AllC',lambda r:AllC(r)),('AllD',lambda r:AllD(r)),
     ('Random.05',lambda r:RandomMoves(0.05,r)),('Joss.52',lambda r:Joss(0.52,r)),
     ('Gap.55',lambda r:GapStartPolicy(1,0.55)),('Kam4',lambda r:Kamikaze(4,1.0,r)),('Kam9',lambda r:Kamikaze(9,1.0,r)),
     ('FSM5',lambda r:RandomFSM(r,5)),
     ('Ext3.75',lambda r:ExtortZD(3.75,r)),('Ext5.5',lambda r:ExtortZD(5.5,r)),('Ext6.5',lambda r:ExtortZD(6.5,r))]
for oname,of in ops:
    print(f"{oname:10s}", ' '.join(f"{n}={eval_weak(f,of):.0f}" for n,f in variants.items()))
