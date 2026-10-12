import argparse,copy,json,pathlib,time,numpy as np,torch
from trace_io import routers
p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--rank',type=int,default=16);p.add_argument('--epochs',type=int,default=40);p.add_argument('--threads',type=int,default=8);p.add_argument('--layers',default='all');a=p.parse_args()
torch.set_num_threads(a.threads);torch.manual_seed(338)
root=pathlib.Path(a.directory);data=root/'dataset';out=root/'trained';out.mkdir(exist_ok=True)
xx=np.load(data/'x.npy',mmap_mode='r');ii=np.load(data/'ids.npy',mmap_mode='r');mm=np.load(data/'resident.npy',mmap_mode='r');sp=np.load(data/'split.npy')
ix=[torch.from_numpy(np.flatnonzero(sp==s)) for s in range(3)];W=torch.from_numpy(routers(root/'routers-bf16.npy'))
layers=list(range(47)) if a.layers=='all' else [int(s) for s in a.layers.split(',')]
report={'rank':a.rank,'seed':338,'epochs_max':a.epochs,'batch_size':256,'learning_rate':.01,'objective':'uniform top10 cross entropy','selection':'best validation recall@10, patience 8 after at least 12 epochs','layers':[]}
def metrics(z,truth,resident):
 result={};n=len(truth)
 for k in [10,14,20]:
  pred=z.topk(k,dim=1).indices
  matched=(pred[:,:,None]==truth[:,None,:]).any(1)
  hits=int(matched.sum());den=n*10
  missing=~resident.gather(1,truth)
  cold=z.masked_fill(resident,float('-inf')).topk(k,dim=1).indices
  coldhit=((cold[:,:,None]==truth[:,None,:]).any(1)&missing).sum().item();coldn=missing.sum().item()
  result[str(k)]={'hits':hits,'targets':den,'recall':hits/den,'cold_hits':int(coldhit),'cold_targets':int(coldn),'cold_recall':coldhit/coldn if coldn else None}
 result['exact_top10']=float(((z.topk(10,dim=1).indices[:,:,None]==truth[:,None,:]).any(1).all(1)).float().mean())
 return result
with torch.no_grad():
 last_x=torch.from_numpy(np.array(xx[47],copy=True));last_ids=torch.from_numpy(np.array(ii[47],dtype=np.int64))
 last_pred=(last_x@W[47].T).topk(10,dim=1).indices
 report['last_layer_reconstruction_recall']=float((last_pred[:,:,None]==last_ids[:,None,:]).any(1).float().mean())
 assert report['last_layer_reconstruction_recall']>=.99
 del last_x,last_ids,last_pred
for layer in layers:
 t0=time.monotonic();X=torch.from_numpy(np.array(xx[layer],copy=True));truth=torch.from_numpy(np.array(ii[layer+1],dtype=np.int64));resident=torch.from_numpy(np.array(mm[layer+1],dtype=bool))
 with torch.no_grad():
  bases={'current':X@W[layer].T,'next':X@W[layer+1].T}
  selfids=torch.from_numpy(np.array(ii[layer],dtype=np.int64));selfmatch=(bases['current'].topk(10,dim=1).indices[:,:,None]==selfids[:,None,:]).any(1)
  parity=float(selfmatch.float().mean());assert parity>=.99,(layer,parity)
  freq=torch.bincount(truth[ix[0]].flatten(),minlength=512).float();freqz=freq.expand(len(ix[2]),-1)
  frequency=metrics(freqz,truth[ix[2]],resident[ix[2]])
  scale=float(X[ix[0]].square().mean().sqrt());Xn=X/max(scale,1e-8)
 row={'layer':layer,'samples':len(X),'router_reconstruction_recall':parity,'frequency_test':frequency,'variants':{}}
 for kind,base in bases.items():
  torch.manual_seed(338+layer*10+(kind=='next'))
  A=torch.nn.Parameter(torch.randn(2560,a.rank)/2560**.5);B=torch.nn.Parameter(torch.zeros(a.rank,512));optim=torch.optim.AdamW([A,B],lr=.01,weight_decay=1e-4)
  best=-1.;best_epoch=0;stale=0;best_params=None;history=[]
  for epoch in range(a.epochs):
   order=ix[0][torch.randperm(len(ix[0]))];loss_sum=0.
   for batch in order.split(256):
    logits=base[batch]+(Xn[batch]@A)@B
    loss=-logits.log_softmax(1).gather(1,truth[batch]).mean()
    optim.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_([A,B],5.);optim.step();loss_sum+=float(loss.detach())*len(batch)
   with torch.no_grad():
    vz=base[ix[1]]+(Xn[ix[1]]@A)@B
    recall=float((vz.topk(10,dim=1).indices[:,:,None]==truth[ix[1]][:,None,:]).any(1).float().mean())
   history.append({'epoch':epoch+1,'loss':loss_sum/len(ix[0]),'val_recall10':recall})
   if recall>best+1e-5:best=recall;best_epoch=epoch+1;best_params=(A.detach().clone(),B.detach().clone());stale=0
   else:stale+=1
   if epoch>=11 and stale>=8:break
  # Fold the training-only input scale into A. Saved BF16 weights are what the reported after metrics use.
  finalA=(best_params[0]/max(scale,1e-8)).to(torch.bfloat16);finalB=best_params[1].to(torch.bfloat16)
  torch.save({'A':finalA,'B':finalB,'layer':layer,'base':kind,'rank':a.rank,'input':'exact host MoE input, float32','scale_folded':True},out/f'layer-{layer:02d}-{kind}-r{a.rank}.pt')
  with torch.no_grad():
   trained=base+(X@finalA.float())@finalB.float()
   after=metrics(trained[ix[2]],truth[ix[2]],resident[ix[2]])
   before=metrics(base[ix[2]],truth[ix[2]],resident[ix[2]])
   train_after=metrics(trained[ix[0]],truth[ix[0]],resident[ix[0]])
  row['variants'][kind]={'best_epoch':best_epoch,'val_recall10':best,'before_test':before,'after_test_bf16':after,'after_train_bf16':train_after,'history':history}
 row['seconds']=time.monotonic()-t0;report['layers'].append(row)
 (out/'report.json').write_text(json.dumps(report,indent=2))
 print(json.dumps({'layer':layer,'parity':parity,'seconds':row['seconds'],'test_recall10':{k:[v['before_test']['10']['recall'],v['after_test_bf16']['10']['recall']] for k,v in row['variants'].items()}}),flush=True)
# Micro-average raw hit counts; bootstrap across held-out prompt groups is computed separately.
aggregate={}
for kind in ['current','next']:
 aggregate[kind]={}
 for phase in ['before_test','after_test_bf16']:
  aggregate[kind][phase]={}
  for k in ['10','14','20']:
   sums={s:sum(r['variants'][kind][phase][k][s] for r in report['layers']) for s in ['hits','targets','cold_hits','cold_targets']}
   sums.update(recall=sums['hits']/sums['targets'],cold_recall=sums['cold_hits']/sums['cold_targets']);aggregate[kind][phase][k]=sums
report['aggregate']=aggregate;(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(aggregate),flush=True)
