import pathlib,json,argparse,numpy as np,torch
from trace_io import routers
p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--run',default='trained');p.add_argument('--base',choices=['both','next','current'],default='both');p.add_argument('--uncached',action='store_true');a=p.parse_args();root=pathlib.Path(a.directory);d=root/'dataset';out=root/a.run
torch.set_num_threads(8);r=json.loads((out/'report.json').read_text());assert len(r['layers'])==47
split=np.load(d/'split.npy');allgroups=np.load(d/'group.npy');ix=np.flatnonzero(split==2);groups=allgroups[ix];gids=sorted(set(groups.tolist()))
resident=np.load(d/'resident.npy',mmap_mode='r')
x=np.load(d/'x.npy',mmap_mode='r');ids=np.load(d/'ids.npy',mmap_mode='r');W=torch.from_numpy(routers(root/'routers-bf16.npy'))
counts={k:{str(g):{'before':0,'after':0,'targets':0} for g in gids} for k in (['current','next'] if a.base=='both' else [a.base])}
for l in range(47):
 X=torch.from_numpy(np.array(x[l,ix]));truth=torch.from_numpy(np.array(ids[l+1,ix],dtype=np.int64))
 for kind in counts:
  state=torch.load(out/f'layer-{l:02d}-{kind}-r16.pt',weights_only=True)
  base=X@W[l+(kind=='next')].T;trained=base+X@state['A'].float()@state['B'].float()
  missing=torch.ones_like(truth,dtype=torch.bool)
  if a.uncached:
   cached=torch.from_numpy(np.array(resident[l+1,ix],dtype=bool));missing=~cached.gather(1,truth)
   base=base.masked_fill(cached,float('-inf'));trained=trained.masked_fill(cached,float('-inf'))
  bh=(base.topk(10,dim=1).indices[:,:,None]==truth[:,None,:]).any(1).logical_and(missing).sum(1).numpy()
  ah=(trained.topk(10,dim=1).indices[:,:,None]==truth[:,None,:]).any(1).logical_and(missing).sum(1).numpy()
  for g in gids:
   take=groups==g;c=counts[kind][str(g)];c['before']+=int(bh[take].sum());c['after']+=int(ah[take].sum());c['targets']+=int(missing[torch.from_numpy(take)].sum())
res={'metric':'uncached recall@10' if a.uncached else 'overall recall@10','groups':gids,'counts':counts,'paired_bootstrap':{}}
for kind,cc in counts.items():
 rows=np.array([[v['before'],v['after'],v['targets']] for v in cc.values()]);rng=np.random.default_rng(338)
 totals=rows[rng.integers(0,len(rows),(10000,len(rows)))].sum(1);deltas=(totals[:,1]-totals[:,0])/totals[:,2]
 res['paired_bootstrap'][kind]={'delta':float((rows[:,1].sum()-rows[:,0].sum())/rows[:,2].sum()),'ci95':np.quantile(deltas,[.025,.975]).tolist(),'note':f'resampling {len(gids)} held-out groups; grouping defined by dataset/group.npy'}
 for phase,col in [('before_test',0),('after_test_bf16',1)]:
  assert int(rows[:,col].sum())==r['aggregate'][kind][phase]['10']['cold_hits' if a.uncached else 'hits']
(out/'groups.json').write_text(json.dumps(res,indent=2));print(json.dumps(res),flush=True)
