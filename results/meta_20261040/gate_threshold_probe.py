import statistics
from ipd.policies.meta import LearnedSwitchPolicy
from ipd.paths import weights_path
from ipd.evaluation.protocol import build_protocol_field
from ipd.evaluation.common import evaluate_field_details

def make(thr):
    p=LearnedSwitchPolicy(weights_path('rl_policy_weights_wide160.json'),
                          weights_path('rl_policy_weights_repair.json'),
                          weights_path('meta_gate_wide160_repair.json'))
    p.threshold=thr
    return p

def ev(thr,tier,fields=3,n=512):
    vals=[]; switched=0; total=0
    for seed in range(20261040,20261040+fields):
        field=build_protocol_field(tier,seed,n); p=make(thr)
        stats,by=evaluate_field_details(p,field); vals.append(stats.mean)
        # evaluate_field_details uses one policy for all matches; policy.decided persists after first match!
    return statistics.fmean(vals)

for thr in [0.5,0.6,0.7,0.8,0.9,0.95]:
    print(thr, 'param',round(ev(thr,'dev_param'),2),'complex',round(ev(thr,'dev_complex'),2),'ood',round(ev(thr,'dev_ood'),2))
