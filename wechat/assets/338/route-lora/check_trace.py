import argparse,json,numpy as np,torch
from trace_io import windows,routers
p=argparse.ArgumentParser();p.add_argument('trace');p.add_argument('routers');p.add_argument('--out',required=True);a=p.parse_args()
torch.set_num_threads(8);w=torch.from_numpy(routers(a.routers));hit=np.zeros(48);count=np.zeros(48)
for group in windows(a.trace):
 for _,l,x,ids,_ in group:
  z=torch.from_numpy(x)@w[l].T
  pred=z.topk(10,dim=1).indices.numpy()
  hit[l]+=np.sum(np.any(pred[:,:,None]==ids[:,None,:],axis=2));count[l]+=ids.size
scores=hit/count
r={'hit_per_layer':scores.tolist(),'total_hits':int(hit.sum()),'total_targets':int(count.sum()),'hit_rate':float(hit.sum()/count.sum()),'pass':bool(scores.min()>=.99 and hit.sum()/count.sum()>=.999)}
open(a.out,'w').write(json.dumps(r,indent=2));print(json.dumps(r))
if not r['pass']:raise SystemExit('router reconstruction failed')
