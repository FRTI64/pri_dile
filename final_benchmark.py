#!/usr/bin/env python3
"""Common unseen benchmark for base/tuned Final, Responsive3, BayesPlus and Hybrid."""
from __future__ import annotations
import json, random, statistics, argparse
from concurrent.futures import ProcessPoolExecutor
from ipd_sim import payoff
from search_common import ParamPolicy
from train_multiobj import ParamResp, build_dist_field
from train_bayes_priors import ParamBayesPrior
from train_joint_final_resp import Hybrid
from neural_policy import NeuralPolicy
from gru_policy import GRUPolicy
from adversarial_policy import AdversarialPolicy

def make(kind,cfg):
    if kind=='final': return ParamPolicy(seed=20261024,**cfg)
    if kind=='resp': return ParamResp(seed=20261024,**cfg)
    if kind=='bayes': return ParamBayesPrior(cfg)
    if kind=='hybrid': return Hybrid(cfg['final'],cfg['resp'],cfg['p'])
    if kind=='neural': return NeuralPolicy(cfg['weights'],seed=cfg.get('seed',20261024),mode=cfg.get('mode','sample'))
    if kind=='gru': return GRUPolicy(cfg['weights'],seed=cfg.get('seed',20261024),mode=cfg.get('mode','sample'))
    if kind=='adversarial': return AdversarialPolicy.from_config(cfg['config'],seed=cfg.get('seed',20261024),mode=cfg.get('mode','sample'))
    raise ValueError(kind)

def eval_one(args):
    name,kind,cfg,dist,seeds,n=args
    vals=[]
    for seed in seeds:
        field=build_dist_field(dist,seed,n);p=make(kind,cfg);total=0
        for opp in field:
            p.reset();opp.reset();mine=[];theirs=[];s=0
            for t in range(100):
                x=p.act(t,mine,theirs) if hasattr(p,'act') else p.move(t+1,mine,theirs)
                y=opp.act(t,theirs,mine);mine.append(x);theirs.append(y);s+=payoff(x,y)
            total+=s
        vals.append(total/n)
    return name,dist,vals

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fields',type=int,default=20);ap.add_argument('--workers',type=int,default=24);ap.add_argument('--out',default='final_benchmark.json')
    ap.add_argument('--neural',action='store_true',help='include the exported neural policy')
    ap.add_argument('--neural-arch',default='mlp',choices=('mlp','gru','adversarial'))
    ap.add_argument('--neural-weights',default=None)
    ap.add_argument('--neural-config',default=None)
    ap.add_argument('--neural-mode',default='sample',choices=('sample','argmax'))
    args=ap.parse_args()
    m=json.load(open('train_multiobj_results.json'))
    f=next(r['cfg'] for r in m['validation'] if r['kind']=='final')
    r=next(r['cfg'] for r in m['validation'] if r['kind']=='resp')
    b=json.load(open('train_bayes_priors_results.json'))
    bt=max(b['validation'],key=lambda x:x['val_fit'])['cfg']
    bb={"coop_scale":1,"punitive_scale":1,"late_scale":1,"noise_scale":1,"probe_scale":1,"nonresp":.40,"punitive_thr":.55}
    j=json.load(open('train_joint_results.json'));h=max(j['validation'],key=lambda x:x['val_fit'])
    cands=[
      ('Final-base','final',{}),
      ('Final-tuned','final',f),
      ('Resp3-base','resp',{}),
      ('Resp3-tuned','resp',r),
      ('Bayes-base','bayes',bb),
      ('Bayes-tuned','bayes',bt),
      ('Final+Resp3','hybrid',{'final':h['final'],'resp':h['resp'],'p':h['p']}),
    ]
    if args.neural:
        if args.neural_arch=='adversarial':
            if args.neural_config is None:
                args.neural_config='adversarial_policy_config.json'
            cands.append((f'Adv-{args.neural_mode}','adversarial',
                          {'config':args.neural_config,'mode':args.neural_mode}))
        else:
            if args.neural_weights is None:
                args.neural_weights=('gru_policy_weights.json' if args.neural_arch=='gru'
                                     else 'neural_policy_weights.json')
            label='GRU' if args.neural_arch=='gru' else 'Neural'
            kind='gru' if args.neural_arch=='gru' else 'neural'
            cands.append((f'{label}-{args.neural_mode}',kind,
                          {'weights':args.neural_weights,'mode':args.neural_mode}))
    rng=random.Random(20261026)
    seed_map={d:[rng.randrange(1<<30) for _ in range(args.fields)] for d in ('original','late','grim30','wide')}
    tasks=[(name,kind,cfg,dist,seeds,299) for name,kind,cfg in cands for dist,seeds in seed_map.items()]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        results=list(ex.map(eval_one,tasks,chunksize=1))
    out={name:{} for name,_,_ in cands}
    for name,dist,vals in results:
        out[name][dist]={'mean':statistics.fmean(vals),'min':min(vals),'p05':sorted(vals)[int(.05*len(vals))]}
    print(f"{'strategy':14}"+"".join(f"{d:>12}" for d in ('original','late','grim30','wide')))
    for name,_,_ in cands:
        print(f"{name:14}"+"".join(f"{out[name][d]['mean']:12.2f}" for d in ('original','late','grim30','wide')))
    with open(args.out,'w',encoding='utf-8') as fh:json.dump(out,fh,ensure_ascii=False,indent=2)
    print('written',args.out)

if __name__=='__main__':main()
