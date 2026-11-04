#!/usr/bin/env python3
import argparse,json,time,math,hashlib,gc,inspect
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers.cache_utils import DynamicCache
from scipy.stats import t as tdist
R=Path(__file__).resolve().parent;ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['gpt2-xl','pythia-2.8b'],required=True);ap.add_argument('--pert-tokens',type=int,default=24);args=ap.parse_args()
assert torch.cuda.is_available() and 'GB10' in torch.cuda.get_device_name()
torch.set_num_threads(8);torch.manual_seed(361);torch.set_float32_matmul_precision('highest');START=time.time();outdir=R/args.model;outdir.mkdir(exist_ok=True)
def save(name,data):(outdir/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
def log(*x):print(time.strftime('%H:%M:%S'),args.model,*x,flush=True)
path='/home/lmxxf/work/models/'+args.model;tok=AutoTokenizer.from_pretrained(path,local_files_only=True);tok.pad_token=tok.eos_token;tok.padding_side='right'
m=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,dtype=torch.float32,device_map='cuda',attn_implementation='sdpa').eval();m.requires_grad_(False)
isgpt=args.model=='gpt2-xl';base=m.transformer if isgpt else m.gpt_neox;blocks=base.h if isgpt else base.layers;norm=base.ln_f if isgpt else base.final_layer_norm;head=m.get_output_embeddings();L=len(blocks);site=21 if isgpt else 11
corpus=json.loads((R/'corpus.json').read_text());texts={k:[x['text'] if isinstance(x,dict) else x for x in corpus[k]] for k in ['long','short']}
save('environment.json',dict(model=args.model,config=m.config.to_dict(),torch=torch.__version__,transformers=__import__('transformers').__version__,dtype='float32',site_zero_based=site,blocks=L,device=torch.cuda.get_device_name(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),seed=361,observation_context_endpoint='j>=5, j<=T-2; C_j ends at token j; H(logits[j]) predicts token j+1',perturbation_site='post block, pre final norm; paper first contextualized layer is L0',alignment_warning='paper C_k equation reaches x_k but entropy H(w_k) conditions on w_<k; subsequent-distribution prose and intervention H(w_k+1) support primary causal alignment; adjacent offsets also measured'))
@torch.inference_mode()
def ent(z):
 lp=F.log_softmax(z.float(),-1);return -(lp.exp()*lp).sum(-1)/math.log(2)
def geom(x):
 d=x[...,1:,:]-x[...,:-1,:];u=F.normalize(d,dim=-1,eps=1e-12);c=torch.acos((u[...,1:,:]*u[...,:-1,:]).sum(-1).clamp(-1,1));C=torch.full(x.shape[:-1],float('nan'),device=x.device)
 C[...,4:]=(c[...,:-2]+c[...,1:-1]+c[...,2:])/3
 return C
@torch.inference_mode()
def observe(which):
 strings=texts[which];X=[];Y=[];Ym=[];Yp=[];groups=[];positions=[];means=[];activation_samples=[];skipped=[]
 for start in range(0,len(strings),4):
  ss=strings[start:start+4];seqs=[tok.encode(s,add_special_tokens=False) for s in ss]
  valid=[(i,v) for i,v in enumerate(seqs) if 8<=len(v)<=min(getattr(m.config,'max_position_embeddings',1024),1024)]
  if not valid:continue
  inp=tok([ss[i] for i,_ in valid],padding=True,add_special_tokens=False,return_tensors='pt').to('cuda');curves={};av=[];handles=[]
  def hook(l):
   def fn(module,a,o):
    h=o[0] if isinstance(o,tuple) else o;curves[l]=geom(h.float()).cpu()
    if l==site and which=='long':
     for b,(_,ids) in enumerate(valid):av.append(h[b,5:len(ids)-1:4].float().cpu())
   return fn
  handles=[b.register_forward_hook(hook(l)) for l,b in enumerate(blocks)]
  try:o=m(**inp,use_cache=False)
  finally:
   for h in handles:h.remove()
  H=ent(o.logits).cpu();del o
  for b,(i,ids) in enumerate(valid):
   ix=np.arange(5,len(ids)-1);X.append(np.stack([curves[l][b,ix].numpy() for l in range(L)]));Y.append(H[b,ix].numpy());Ym.append(H[b,ix-1].numpy());Yp.append(H[b,ix+1].numpy());groups.extend([start+i]*len(ix));positions.extend(ix.tolist())
  activation_samples.extend(av)
  if start%128==0:log('observe',which,start,'/',len(strings))
 x=np.concatenate(X,axis=1);y=np.concatenate(Y);ym=np.concatenate(Ym);yp=np.concatenate(Yp);g=np.array(groups)
 np.savez_compressed(outdir/f'observation_{which}.npz',C_rad=x,H_bits=y,H_previous_logits=ym,H_next_logits=yp,passage=g,endpoint=positions)
 summary=correlations(x,y,g);summary['offset_sensitivity']={'H_logits_j_minus_1':[float(np.corrcoef(v,ym)[0,1]) for v in x],'H_logits_j_plus_1':[float(np.corrcoef(v,yp)[0,1]) for v in x]};summary['mean_curvature_rad']=x.mean(1).tolist();summary['min_curvature_layer']=int(x.mean(1).argmin());summary['mean_entropy_bits']=float(y.mean());summary['n_passages']=len(np.unique(g));save(f'observation_{which}.json',summary)
 if which=='long':
  A=torch.cat(activation_samples).to('cuda');A-=A.mean(0);_,_,V=torch.pca_lowrank(A,q=2,center=False,niter=8);Q=V[:,:2].contiguous();torch.save(Q.cpu(),outdir/'activation_pc2.pt');del A
  return summary,Q
 return summary

def fisher(v):return np.tanh(np.mean(np.arctanh(np.clip(v,-.999999,.999999))))
def cv(x,y,g,grouped=False):
 rng=np.random.default_rng(361);items=np.unique(g) if grouped else np.arange(len(y));rng.shuffle(items);chunks=np.array_split(items,10);rr=[];sl=[]
 for ch in chunks:
  test=np.isin(g,ch) if grouped else np.isin(np.arange(len(y)),ch);train=~test;xx=x[train];yy=y[train];s=np.dot(xx-xx.mean(),yy-yy.mean())/np.dot(xx-xx.mean(),xx-xx.mean());pred=yy.mean()+s*(x[test]-xx.mean());rr.append(float(np.corrcoef(pred,y[test])[0,1]));sl.append(float(s))
 z=np.arctanh(np.clip(rr,-.999999,.999999));mu=z.mean();se=z.std(ddof=1)/np.sqrt(10);q=tdist.ppf(.975,9)
 return dict(pooled_r=float(np.tanh(mu)),ci95=[float(np.tanh(mu-q*se)),float(np.tanh(mu+q*se))],fold_r=rr,slopes=sl)
def correlations(x,y,g):
 rows=[]
 for l,v in enumerate(x):rows.append(dict(layer_zero_based=l,raw_signed_pearson_r=float(np.corrcoef(v,y)[0,1]),token_cv=cv(v,y,g),passage_cv=cv(v,y,g,True)))
 return dict(n_tokens=len(y),layers=rows,peak_positive_raw_layer=int(np.argmax([r['raw_signed_pearson_r'] for r in rows])),peak_cv_layer=int(np.argmax([r['token_cv']['pooled_r'] for r in rows])))
long,Qglobal=observe('long');short=observe('short')
@torch.inference_mode()
def baseline(ids,k):
 x=torch.tensor([ids[:k+2]],device='cuda');capture={}
 def hk(b,a,o):capture['h']=(o[0] if isinstance(o,tuple) else o).detach()
 h=blocks[site].register_forward_hook(hk)
 try:o=m(x,use_cache=True)
 finally:h.remove()
 H=float(ent(o.logits[0,k]));z=o.logits[0,k].float().cpu();cache=[(l.keys[:,:,:k].contiguous(),l.values[:,:,:k].contiguous()) for l in o.past_key_values.layers];X=capture['h'][0];return X,H,z,cache
@torch.inference_mode()
def suffix(X,cache,k,delta):
 B=delta.shape[0];h=(X[k]+delta).unsqueeze(1);pos=torch.tensor([[k]],device='cuda');past=DynamicCache(ddp_cache_data=[(a.expand(B,-1,-1,-1),b.expand(B,-1,-1,-1)) for a,b in cache])
 pe=None if isgpt else base.rotary_emb(h,position_ids=pos)
 for l in range(site+1,L):
  if isgpt:h=blocks[l](h,past_key_values=past,attention_mask=None,position_ids=pos,use_cache=True)
  else:h=blocks[l](h,attention_mask=None,position_ids=pos,layer_past=past,use_cache=True,position_embeddings=pe)
 z=head(norm(h))[:,0].float();return ent(z).cpu().numpy(),z
@torch.inference_mode()
def plane_basis(d):return torch.linalg.qr(d.T,mode='reduced')[0][:,:2]
@torch.inference_mode()
def sample_dirs(Q,n,d):
 a=torch.randn(n,Q.shape[1],device='cuda');a=F.normalize(a,dim=-1);return a@Q.T
families=['full','random2','activation2','trajectory2','planar2','orthogonal_to_plane'];records=[];raws=[];N=300
for t in range(args.pert_tokens):
 ids=tok.encode(texts['long'][t],add_special_tokens=False);k=min(len(ids)-2,max(8,min(96,len(ids)//2)))
 if k<5:continue
 X,H0,z0,cache=baseline(ids,k);D=X[1:k+2]-X[:k+1];P=plane_basis(D[k-2:k]);U,S,Vh=torch.linalg.svd(D-D.mean(0),full_matrices=False);TQ=Vh[:2].T;RQ=plane_basis(torch.randn(2,X.shape[-1],device='cuda'));scale=.2*float(D[k].norm());ap=F.normalize(D[k-2],dim=-1);b=F.normalize(D[k-1],dim=-1);c0=torch.acos(torch.dot(ap,b).clamp(-1,1))
 e,z=suffix(X,cache,k,torch.zeros(1,X.shape[-1],device='cuda'));maxerr=float((z[0].cpu()-z0).abs().max());assert maxerr<.002,(args.model,t,maxerr)
 row=dict(sample=t,context_endpoint=k,passage_tokens=len(ids),scale=scale,baseline_entropy_bits=H0,suffix_baseline_entropy_error=float(e[0]-H0),suffix_baseline_logit_maxerror=maxerr,families={})
 for family in families:
  if family=='full':dirs=F.normalize(torch.randn(N,X.shape[-1],device='cuda'),dim=-1)
  elif family=='orthogonal_to_plane':
   a=torch.randn(N,X.shape[-1],device='cuda');dirs=F.normalize(a-(a@P)@P.T,dim=-1)
  else:dirs=sample_dirs({'random2':RQ,'activation2':Qglobal,'trajectory2':TQ,'planar2':P}[family],N,X.shape[-1])
  delta=dirs*scale;v=D[k-1]+delta;turn=torch.acos((F.normalize(v,dim=-1)*ap).sum(-1).clamp(-1,1));dc=((turn-c0)/3).cpu().numpy();hs=[]
  for st in range(0,N,32):ee,zz=suffix(X,cache,k,delta[st:st+32]);hs.extend(ee.tolist());del zz
  dh=np.array(hs)-H0;raws.append(dict(sample=t,family=family,dc=dc.tolist(),dh=dh.tolist()));row['families'][family]=dict(raw_r=(None if family=='orthogonal_to_plane' else float(np.corrcoef(dc,dh)[0,1])),sd_delta_C=float(dc.std()),mean_delta_C=float(dc.mean()),mean_delta_H=float(dh.mean()),sd_delta_H=float(dh.std()))
 # Equal-norm deterministic angle-gradient perturbations are additional, not the paper's family test.
 grad=F.normalize(torch.dot(ap,b)*b-ap,dim=-1);delta=torch.stack([-grad,grad])*scale;hs,z=suffix(X,cache,k,delta);vv=F.normalize(D[k-1]+delta,dim=-1);dc=((torch.acos((vv*ap).sum(-1).clamp(-1,1))-c0)/3).cpu().tolist();row['deterministic']={label:dict(delta_C_rad=float(dc[i]),delta_H_bits=float(hs[i]-H0)) for i,label in enumerate(['straighten','bend'])}
 records.append(row);save('intervention_progress.json',records);log('perturb',t+1,'/',args.pert_tokens)
 del cache,X,z;gc.collect();torch.cuda.empty_cache()
# Appendix A4 histogram importance weights; only middle layer is measured here.
reference=np.concatenate([np.abs(r['dc']) for r in raws if r['family']=='full']);edges=np.linspace(0,reference.max()+1e-12,101);q=np.histogram(reference,bins=edges)[0].astype(float);q/=q.sum();summ={}
def wcorr(a,b,w):
 w=w/w.sum();aa=a-np.dot(w,a);bb=b-np.dot(w,b);return float(np.dot(w,aa*bb)/np.sqrt(np.dot(w,aa*aa)*np.dot(w,bb*bb)))
for f in families:
 rows=[r for r in raws if r['family']==f]
 if f=='orthogonal_to_plane':
  summ[f]=dict(mean_token_weighted_r=None,fisher_pooled_weighted_r=None,fisher_bootstrap_ci95=None,mean_raw_r=None,reason='Fixed-norm directions strictly orthogonal to the local plane give constant delta curvature; a Pearson correlation is mathematically undefined, not a zero effect',mean_delta_H_bits=float(np.mean([r['dh'] for r in rows])),mean_sd_delta_H_bits=float(np.mean([np.std(r['dh']) for r in rows])),mean_sd_delta_C=float(np.mean([np.std(r['dc']) for r in rows])))
  continue
 vals=np.concatenate([np.abs(r['dc']) for r in rows]);counts=np.histogram(vals,bins=edges)[0].astype(float);p=counts/counts.sum();rr=[];rawr=[];ess=[]
 for r in rows:
  a=np.array(r['dc']);b=np.array(r['dh']);bins=np.clip(np.searchsorted(edges,np.abs(a),side='right')-1,0,99);w=np.clip(q[bins]/(p[bins]+1e-12),.1,10);rr.append(wcorr(a,b,w));rawr.append(float(np.corrcoef(a,b)[0,1]));ess.append(float(w.sum()**2/(w@w)));r['importance_weights']=w.tolist()
 rng=np.random.default_rng(361);boot=np.array([np.mean(rng.choice(np.arctanh(np.clip(rr,-.999999,.999999)),len(rr),replace=True)) for _ in range(2000)]);ci=np.tanh(np.quantile(boot,[.025,.975])).tolist()
 summ[f]=dict(mean_token_weighted_r=float(np.mean(rr)),fisher_pooled_weighted_r=float(fisher(rr)),fisher_bootstrap_ci95=ci,mean_raw_r=float(np.mean(rawr)),per_token_weighted_r=rr,mean_effective_N=float(np.mean(ess)))
save('intervention_raw.json',raws);save('interventions.json',dict(site_zero_based=site,N_per_token_family=N,n_tokens=len(records),summary=summ,records=records,reweighting=dict(bins=100,epsilon=1e-12,clamp=[.1,10],reference='full-space middle-layer samples only; paper pools full-space across all intervention layers',reference_edges=edges.tolist()),notes=['Other token states fixed by causality via cached prefix; zero-delta continuation checked against full replay','Scale uses original future displacement x[k+1]-x[k], as paper specifies; additional future information only sets intervention amplitude','Paper layer index taken as post-block zero-based because L0 is first contextualized layer','Planar/directional manipulation changes last of three angle terms; all other angles fixed','No parameter training/regularization experiment is claimed']))
save('results.json',dict(model=args.model,observational=dict(long=long,short=short),interventional=summ,elapsed_seconds=time.time()-START,peak_cuda_allocated_GiB=torch.cuda.max_memory_allocated()/1024**3));log('COMPLETE',round(time.time()-START))
