"""Evaluate old and new adapters on exactly the same held-out rows, grouped by document."""
import argparse,pathlib,json,numpy as np,torch,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'route-lora'))
from trace_io import routers
p=argparse.ArgumentParser();p.add_argument('dataset');p.add_argument('old');p.add_argument('new');p.add_argument('output');a=p.parse_args();root=pathlib.Path(a.dataset);old=pathlib.Path(a.old);new=pathlib.Path(a.new)
torch.set_num_threads(8)
d=root/'dataset';split=np.load(d/'split.npy');ix=np.flatnonzero(split==2);groups=np.load(d/'group.npy')[ix];gids=sorted(set(groups.tolist()))
x=np.load(d/'x.npy',mmap_mode='r');ids=np.load(d/'ids.npy',mmap_mode='r');mask=np.load(d/'resident.npy',mmap_mode='r');w=torch.from_numpy(routers(root/'routers-bf16.npy'))
result={'test_rows_per_layer':len(ix),'groups':gids,'layers':[],'group_counts':{k:{str(g):{'hits':0,'targets':0} for g in gids} for k in ['base','old','new']}}
for layer in range(47):
 X=torch.from_numpy(np.array(x[layer,ix]));truth=torch.from_numpy(np.array(ids[layer+1,ix],dtype=np.int64));cached=torch.from_numpy(np.array(mask[layer+1,ix],dtype=bool));missing=~cached.gather(1,truth)
 logits={'base':X@w[layer+1].T}
 for key,path in [('old',old),('new',new)]:
  state=torch.load(path/f'layer-{layer:02d}-next-r16.pt',weights_only=True);logits[key]=logits['base']+X@state['A'].float()@state['B'].float()
 row={'layer':layer,'models':{}}
 for key,z in logits.items():
  scores={}
  for k in [10,14,20]:
   full=(z.topk(k,dim=1).indices[:,:,None]==truth[:,None,:]).any(1)
   cold=(z.masked_fill(cached,float('-inf')).topk(k,dim=1).indices[:,:,None]==truth[:,None,:]).any(1)&missing
   scores[str(k)]={'overall_hits':int(full.sum()),'overall_targets':truth.numel(),'cold_hits':int(cold.sum()),'cold_targets':int(missing.sum())}
   if k==10:
    hits=cold.sum(1).numpy();den=missing.sum(1).numpy()
    for g in gids:
     take=groups==g;c=result['group_counts'][key][str(g)];c['hits']+=int(hits[take].sum());c['targets']+=int(den[take].sum())
  row['models'][key]=scores
 result['layers'].append(row)
result['aggregate']={}
for key in ['base','old','new']:
 result['aggregate'][key]={}
 for k in ['10','14','20']:
  sums={f:sum(r['models'][key][k][f] for r in result['layers']) for f in ['overall_hits','overall_targets','cold_hits','cold_targets']};sums['overall_recall']=sums['overall_hits']/sums['overall_targets'];sums['cold_recall']=sums['cold_hits']/sums['cold_targets'];result['aggregate'][key][k]=sums
result['paired_group_intervals']={}
for before in ['base','old']:
 rows=np.array([[result['group_counts'][before][str(g)]['hits'],result['group_counts']['new'][str(g)]['hits'],result['group_counts']['new'][str(g)]['targets']] for g in gids])
 rng=np.random.default_rng(338);b=rows[rng.integers(0,len(rows),(10000,len(rows)))].sum(1);delta=(b[:,1]-b[:,0])/b[:,2]
 result['paired_group_intervals']['new_vs_'+before]={'delta':float((rows[:,1].sum()-rows[:,0].sum())/rows[:,2].sum()),'ci95':np.quantile(delta,[.025,.975]).tolist(),'n_groups':len(gids)}
pathlib.Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result['aggregate']),flush=True)
