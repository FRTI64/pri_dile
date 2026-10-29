import random, statistics
from ipd.policies.registry import make_policy
from ipd.core.opponents import Joss, RandomMoves
from ipd.core.opponents.zero_determinant import ExtortZD
from ipd.core.primitives import C, D
from ipd.evaluation.common import run_match

class SwitchMeta:
    def __init__(self, seed=123):
        self.base=make_policy('rl-wide160', mode='sample')
        self.alt=make_policy('rl-repair', mode='sample')
        self.switched=False; self.rng=random.Random(seed)
    def reset(self): self.base.reset(); self.alt.reset(); self.switched=False
    def act(self,t,mine,theirs):
        if t>=99: return D
        if not self.switched and t>=6 and theirs[:6].count(D)>=2 and theirs[-1]==C and mine[:6].count(D)>=2:
            self.switched=True
        return (self.alt if self.switched else self.base).act(t,mine,theirs)
    def observe(self,*a,**k): pass

def ev(factory, of, reps=64):
    scores=[]; switches=0
    for r in range(reps):
        p=factory(); opp=of(random.Random(2000+r))
        scores.append(run_match(p,opp).score)
        switches += getattr(p,'switched',False)
    return statistics.fmean(scores), statistics.pstdev(scores), switches
base=lambda: make_policy('rl-wide160',mode='sample')
meta=lambda: SwitchMeta()
ops=[('Ext3.75',lambda r:ExtortZD(3.75,r)),('Ext5.5',lambda r:ExtortZD(5.5,r)),('Ext6.5',lambda r:ExtortZD(6.5,r)),('Joss.52',lambda r:Joss(0.52,r)),('Random.05',lambda r:RandomMoves(0.05,r))]
print('name         base(mean/sd)   meta(mean/sd)  delta switch')
for name,f in ops:
    a=ev(base,f); b=ev(meta,f)
    print(f"{name:10s} {a[0]:7.2f}/{a[1]:5.1f}  {b[0]:7.2f}/{b[1]:5.1f} {b[0]-a[0]:+7.2f} {b[2]}")
