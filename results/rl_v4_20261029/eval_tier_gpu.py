import sys, json
from ipd.policies.registry import CandidateSpec
from ipd.evaluation.protocol_cli import _eval_field_batch_task, _aggregate

path=sys.argv[1]; tier=sys.argv[2]
label=sys.argv[3] if len(sys.argv)>3 else 'RL-Wide'
fields=int(sys.argv[4]) if len(sys.argv)>4 else 30
n=int(sys.argv[5]) if len(sys.argv)>5 else 2048
kind=sys.argv[6] if len(sys.argv)>6 else 'rl-pattern'
seeds=[20261027+i*7919 for i in range(fields)]
gpu={"device":"cuda","dtype":"float32","max_batch":32768,"profile":False,
     "fast_rules":"auto","fast_rules_threshold":0.90,"mode":"sample",
     "sample_rng":"per-match","base_seed":20261027}
rows=[]
for start in range(0,len(seeds),4):
    group=seeds[start:start+4]
    spec=CandidateSpec(kind=kind, name=label,
                       params={'weights':path,'mode':'sample'})
    task={"specs":[spec.to_dict()],"tier":tier,"seeds":group,
          "seed":group[0],"n":n,"rounds":100,"field_reuse":"on",
          "gpu":gpu,"gpu_batch":True}
    rows.extend(_eval_field_batch_task(task)['results'])
print(json.dumps(_aggregate(rows)))
