import sys, json, statistics
from collections import defaultdict
from ipd.policies.neural import NeuralPolicy
from ipd.policies.rl import RLPolicy
from ipd.evaluation.protocol import build_protocol_field
from ipd.evaluation.common import evaluate_field_details

w=sys.argv[1]; mode=sys.argv[2] if len(sys.argv)>2 else 'argmax'
fields=int(sys.argv[3]) if len(sys.argv)>3 else 3
n=int(sys.argv[4]) if len(sys.argv)>4 else 512
fmt=json.load(open(w)).get('format')
cls=RLPolicy if fmt=='ipd-rl-policy-v1' else NeuralPolicy
out={}
for tier in ['dev_param','dev_complex','dev_ood','dev_axelrod','dev_family']:
    per=defaultdict(list)
    for seed in range(20261027,20261027+fields):
        field=build_protocol_field(tier,seed,n)
        p=cls(w,seed=20261024,mode=mode)
        stats,by=evaluate_field_details(p,field)
        for name,vals in by.items(): per[name.split(':',1)[0]].extend(vals)
    fam={k:statistics.fmean(v) for k,v in per.items()}
    out[tier]={'mean':statistics.fmean(fam.values()),'families':fam}
print(json.dumps(out))
