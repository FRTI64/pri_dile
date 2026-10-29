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

class PrefixMeta:
    def __init__(self, prefix_kind, n, seed=123):
        self.base=make_policy('rl-wide160', mode='sample')
        self.prefix=make_policy(prefix_kind, mode='sample')
        self.n=n; self.rng=random.Random(seed)
    def reset(self):
        self.base.reset(); self.prefix.reset()
    def act(self,t,mine,theirs):
        if t>=99: return D
        if t<self.n: return self.prefix.act(t,mine,theirs)
        return self.base.act(t,mine,theirs)
    def observe(self,*a,**k): pass

def eval_weak(factory, opp_factory, reps=16):
    scores=[]
    for r in range(reps):
        p=factory(); opp=opp_factory(random.Random(1000+r))
        scores.append(run_match(p,opp).score)
    return statistics.fmean(scores)

def eval_tier(factory, tier, fields=2, n=256):
    vals=[]
    for seed in range(20261039, 20261039+fields):
        field=build_protocol_field(tier,seed,n); p=factory()
        stats,by=evaluate_field_details(p,field); vals.append(stats.mean)
    return statistics.fmean(vals)

variants={'base':lambda: make_policy('rl-wide160',mode='sample')}
for k in [3,5,8,10,15,20]:
    variants[f'tri{k}']=lambda k=k: PrefixMeta('triple-majority',k)
for k in [5,10,15]:
    variants[f'rep{k}']=lambda k=k: PrefixMeta('rl-repair',k)
print('tier means')
for name,f in variants.items():
    print(f"{name:8s}", 'param',round(eval_tier(f,'dev_param'),2),'complex',round(eval_tier(f,'dev_complex'),2),'ood',round(eval_tier(f,'dev_ood'),2))
print('\nweak opponents')
ops=[('TFT',lambda r:TFT(r)),('Grim',lambda r:Grim(r)),('AllC',lambda r:AllC(r)),('AllD',lambda r:AllD(r)),
     ('Random.05',lambda r:RandomMoves(0.05,r)),('Joss.52',lambda r:Joss(0.52,r)),
     ('Gap.55',lambda r:GapStartPolicy(1,0.55)),('Kam4',lambda r:Kamikaze(4,1.0,r)),('Kam9',lambda r:Kamikaze(9,1.0,r)),
     ('FSM5',lambda r:RandomFSM(r,5)),
     ('Ext3.75',lambda r:ExtortZD(3.75,r)),('Ext5.5',lambda r:ExtortZD(5.5,r)),('Ext6.5',lambda r:ExtortZD(6.5,r))]
for oname,of in ops:
    row=[]
    for name,f in variants.items():
        row.append(f"{name}={eval_weak(f,of):.0f}")
    print(f"{oname:10s}", ' '.join(row))
