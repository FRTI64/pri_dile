import random, statistics
from ipd.policies.registry import make_policy
from ipd.core.primitives import C, D
from ipd.evaluation.protocol import build_protocol_field
from ipd.evaluation.common import evaluate_field_details

class SwitchMeta:
    def __init__(self, alt, kind='dd2', seed=123):
        self.base=make_policy('rl-wide160', mode='sample')
        self.alt=make_policy(alt, mode='sample')
        self.kind=kind; self.switched=False; self.rng=random.Random(seed)
    def reset(self): self.base.reset(); self.alt.reset(); self.switched=False
    def act(self,t,mine,theirs):
        if t>=99: return D
        if not self.switched and self.kind=='dd2' and t>=6 and theirs[:6].count(D)>=2 and theirs[-1]==C and mine[:6].count(D)>=2:
            self.switched=True
        return (self.alt if self.switched else self.base).act(t,mine,theirs)
    def observe(self,*a,**k): pass

def eval_tier(factory, tier, fields=5, n=512):
    vals=[]
    for seed in range(20261039, 20261039+fields):
        field=build_protocol_field(tier,seed,n); p=factory()
        stats,by=evaluate_field_details(p,field); vals.append(stats.mean)
    return statistics.fmean(vals)

variants={
 'base': lambda: make_policy('rl-wide160',mode='sample'),
 'rep_dd2': lambda: SwitchMeta('rl-repair'),
 'tri_gap': lambda: SwitchMeta('triple-majority','gap'),
 'rep_gap': lambda: SwitchMeta('rl-repair','gap'),
}
for name,f in variants.items():
    print(name,'param',round(eval_tier(f,'dev_param'),2),'complex',round(eval_tier(f,'dev_complex'),2),'ood',round(eval_tier(f,'dev_ood'),2))
