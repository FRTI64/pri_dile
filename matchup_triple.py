import json,random,csv
from collections import defaultdict
import ipd_extended as ext
from ipd_sim import play_match
from search_common import ParamPolicy
from train_multiobj import ParamResp
from train_bayes_priors import ParamBayesPrior
from train_joint_final_resp import Hybrid
from triple_hybrid import Triple
m=json.load(open('train_multiobj_results.json'))
f=next(r['cfg'] for r in m['validation'] if r['kind']=='final')
r=next(r['cfg'] for r in m['validation'] if r['kind']=='resp')
b=json.load(open('train_bayes_priors_results.json'));bc=max(b['validation'],key=lambda x:x['val_fit'])['cfg']
j=json.load(open('train_joint_results.json'));h=max(j['validation'],key=lambda x:x['val_fit'])
class W:
 def __init__(self,name,p):self.name=name;self.p=p
 def reset(self):self.p.reset()
 def act(self,t,mine,theirs):return self.p.act(t,mine,theirs) if hasattr(self.p,'act') else self.p.move(t+1,mine,theirs)
CANDS={
 'Final-tuned':lambda:W('Final-tuned',ParamPolicy(seed=20261024,**f)),
 'Resp3-tuned':lambda:W('Resp3-tuned',ParamResp(seed=20261024,**r)),
 'Bayes-tuned':lambda:W('Bayes-tuned',ParamBayesPrior(bc)),
 'Final+Resp3':lambda:W('Final+Resp3',Hybrid(h['final'],h['resp'],h['p'])),
 'Triple-majority':lambda:W('Triple-majority',Triple(f,r,bc,'majority',.65)),
}
rows=defaultdict(lambda:[0,0,0])
for spec in ext.TYPE_BANK:
 for rep in range(16):
  for cname,maker in CANDS.items():
   opp=spec.factory(random.Random(900000+rep*1000+hash(spec.name)%997));opp.name=spec.name
   x=maker();q=play_match(x,opp);k=(cname,spec.name,spec.family)
   rows[k][0]+=q.score_a;rows[k][1]+=q.score_b;rows[k][2]+=1
 print('done',spec.name,flush=True)
out=[]
for (c,t,fam),(a,b,n) in rows.items():out.append((c,t,fam,a/n,b/n,n))
out.sort(key=lambda x:(x[0],x[2],x[1]))
with open('matchup_triple_results.csv','w',newline='',encoding='utf-8') as fh:
 w=csv.writer(fh);w.writerow(['candidate','type','family','our','opponent','reps']);w.writerows(out)
# summarize family
from statistics import fmean
by=defaultdict(lambda:defaultdict(list))
for c,t,fam,a,b,n in out:by[c][fam].append(a)
fams=sorted({x[2] for x in out})
print('family')
for c in CANDS:
 print(c,[(fam,round(fmean(by[c][fam]),1)) for fam in fams], 'overall',round(fmean([x[3] for x in out if x[0]==c]),2))
