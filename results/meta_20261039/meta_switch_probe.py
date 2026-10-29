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

class SwitchMeta:
    def __init__(self, alt='triple-majority', kind='dd', seed=123):
        self.base=make_policy('rl-wide160', mode='sample')
        self.alt=make_policy(alt, mode='sample')
        self.kind=kind; self.switched=False; self.rng=random.Random(seed)
    def reset(self): self.base.reset(); self.alt.reset(); self.switched=False
    def act(self,t,mine,theirs):
        if t>=99: return D
        if not self.switched:
            if self.kind=='dd' and t>=4 and len(theirs)>=4 and theirs[0]==D and theirs[1]==D and theirs[-1]==C:
                self.switched=True
            elif self.kind=='dd2' and t>=6 and theirs[:6].count(D)>=2 and theirs[-1]==C and mine[:6].count(D)>=2:
                self.switched=True
            elif self.kind=='gap' and t>=12:
                nc=nd=dc=dd=0
                for i in range(1,t):
                    if mine[i-1]==C: nc+=1; dc+= theirs[i]==D
                    else: nd+=1; dd+= theirs[i]==D
                pc=dc/nc if nc else 0; pd=dd/nd if nd else 0
                if nc>=6 and nd>=2 and pc<=0.25 and pd<=0.25: self.switched=True
        return (self.alt if self.switched else self.base).act(t,mine,theirs)
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
for kind in ['dd','dd2','gap']:
    variants[f'tri_{kind}']=lambda kind=kind: SwitchMeta('triple-majority',kind)
    variants[f'rep_{kind}']=lambda kind=kind: SwitchMeta('rl-repair',kind)
print('tier means')
for name,f in variants.items():
    print(f"{name:10s}", 'param',round(eval_tier(f,'dev_param'),2),'complex',round(eval_tier(f,'dev_complex'),2),'ood',round(eval_tier(f,'dev_ood'),2))
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
