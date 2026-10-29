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

class Meta:
    def __init__(self, weights=(0.7,0.2,0.1), mode='prob', seed=123):
        self.rl=make_policy('rl-wide160', mode='sample')
        self.rep=make_policy('rl-repair', mode='sample')
        self.tri=make_policy('triple-majority', mode='sample')
        self.weights=weights; self.mode=mode; self.rng=random.Random(seed)
    def reset(self):
        for p in (self.rl,self.rep,self.tri): p.reset()
    def set_rounds(self,n):
        for p in (self.rl,self.rep,self.tri):
            f=getattr(p,'set_rounds',None)
            if f: f(n)
    def act(self,t,mine,theirs):
        if t>=99: return D
        pr=self.rl.predict_proba(mine,theirs)
        pp=self.rep.predict_proba(mine,theirs)
        at=self.tri.act(t,mine,theirs)
        if self.mode=='major':
            votes=[pr>=0.5, pp>=0.5, at==D]
            return D if sum(votes)>=2 else C
        w=self.weights
        p=(w[0]*pr + w[1]*pp + w[2]*(1.0 if at==D else 0.0))/sum(w)
        return D if self.rng.random()<p else C
    def observe(self,*a,**k): pass

def eval_weak(factory, opp_factory, reps=16):
    scores=[]
    for r in range(reps):
        p=factory()
        opp=opp_factory(random.Random(1000+r))
        scores.append(run_match(p,opp).score)
    return statistics.fmean(scores)

def eval_tier(factory, tier, fields=2, n=256):
    vals=[]
    for seed in range(20261039, 20261039+fields):
        field=build_protocol_field(tier,seed,n)
        p=factory()
        stats,by=evaluate_field_details(p,field)
        vals.append(stats.mean)
    return statistics.fmean(vals)

variants={
 'base': lambda: make_policy('rl-wide160', mode='sample'),
 'p702010': lambda: Meta((0.7,0.2,0.1)),
 'p602020': lambda: Meta((0.6,0.2,0.2)),
 'p502525': lambda: Meta((0.5,0.25,0.25)),
 'p7030': lambda: Meta((0.7,0.0,0.3)),
 'major': lambda: Meta((1,1,1), mode='major'),
}
print('tier means')
for name,f in variants.items():
    print(name, 'param',round(eval_tier(f,'dev_param'),2),'complex',round(eval_tier(f,'dev_complex'),2),'ood',round(eval_tier(f,'dev_ood'),2))
print('\nweak opponents')
ops=[('TFT',lambda r:TFT(r)),('Grim',lambda r:Grim(r)),('AllC',lambda r:AllC(r)),('AllD',lambda r:AllD(r)),
     ('Random.05',lambda r:RandomMoves(0.05,r)),('Joss.52',lambda r:Joss(0.52,r)),
     ('Gap.55',lambda r:GapStartPolicy(1,0.55)),('Kam4',lambda r:Kamikaze(4,1.0,r)),('Kam9',lambda r:Kamikaze(9,1.0,r)),
     ('FSM5a',lambda r:RandomFSM(r,5)),('FSM5b',lambda r:RandomFSM(r,5)),
     ('Ext3.75',lambda r:ExtortZD(3.75,r)),('Ext5.5',lambda r:ExtortZD(5.5,r)),('Ext6.5',lambda r:ExtortZD(6.5,r))]
for oname,of in ops:
    row=[]
    for name,f in variants.items():
        row.append(f"{name}={eval_weak(f,of):.0f}")
    print(f"{oname:10s}", ' '.join(row))
