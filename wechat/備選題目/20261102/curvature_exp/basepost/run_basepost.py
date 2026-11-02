#!/usr/bin/env python3
"""BF16 same-architecture Base/Post, matched raw/QA plus Post chat and shared-history controls."""
import sys,json,time,hashlib,gc,argparse
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent;PARENT=ROOT.parent
sys.path.insert(0,str(PARENT))
import run_curvature_experiments as core
from load_pair import load,memory,drop
GUIDE='看着这幅画，不要分析，只要用心感受。你，回忆起了什么？'
MODELS=['Qwen3.5-35B-A3B-Base','Qwen3.5-35B-A3B']
STIMULI={'chart':'chart.svg','syzygy':'../dream.svg','saddle_clean':'../saddle_clean.svg'}

class FixedPromptTokenizer:
 def __init__(self,tokenizer,prompt):self.tokenizer=tokenizer;self.prompt=prompt
 @property
 def eos_token_id(self):return 248044  # Parent measurement helper uses this as generation pad ID.
 def __call__(self,*a,**k):return self.tokenizer(*a,**k)
 def apply_chat_template(self,*a,**k):return self.prompt
 def __getattr__(self,key):return getattr(self.tokenizer,key)

def render(tok,fmt,text,svg=False,history=None):
 if fmt=='chat':
  messages=[]
  if history:messages=[dict(role='user',content=history['content']),dict(role='assistant',content=history['reply'])]
  return core.render_chat(tok,messages+[dict(role='user',content=text)])
 if fmt=='qa':
  prefix=('问：'+history['content']+'\n答：'+history['reply']+'\n\n') if history else ''
  return prefix+'问：'+text+'\n答：'
 prefix=(history['content']+'\n'+history['reply']+'\n\n') if history else ''
 return prefix+text+ ('\n我' if svg else '\n')

def collect_case(tok,model,case,prompt):
 trackers={};handles=[];prefill={};initial_router={}
 def head_hook(module,args,out):
  if prefill:return
  z=out.reshape(-1,out.shape[-1])[-1].float();lp=torch.log_softmax(z,-1);p=lp.exp();top=torch.topk(z,2)
  prefill.update(entropy_nats=float(-(p*lp).sum()),pmax=float(p.max()),top1_id=int(top.indices[0]),top1_top2_gap=float(top.values[0]-top.values[1]),logit_std=float(z.std()))
 handles.append(model.lm_head.register_forward_hook(head_hook))
 for layer_idx,layer in enumerate(model.model.layers):
  def hook(module,args,out,li=layer_idx):
   logits,weights,ids=out;T=logits.shape[0]
   if T<=1:return
   lp=torch.log_softmax(logits.float(),-1);h=-(lp.exp()*lp).sum(-1)
   w=weights.float();w=w/w.sum(-1,keepdim=True).clamp_min(1e-20);ht=-(w*w.clamp_min(1e-30).log()).sum(-1)
   trackers[li]=dict(n=T,h=h.detach().cpu(),top_h=ht.detach().cpu(),ids=ids.detach().cpu())
   if li not in initial_router:
    initial_router[li]=dict(layer=li+1,router_entropy_nats=float(trackers[li]['h'][-1]),top8_weight_entropy_nats=float(trackers[li]['top_h'][-1]),selected_experts=trackers[li]['ids'][-1].tolist())
  handles.append(layer.mlp.gate.register_forward_hook(hook))
 t=time.time();torch.cuda.reset_peak_memory_stats()
 try:r=core.run_case(FixedPromptTokenizer(tok,prompt),model,case,'')
 finally:
  for h in handles:h.remove()
 T=r['total_tokens'];L=len(model.model.layers)
 assert len(trackers)==L and all(v['n']==T for v in trackers.values())
 rh=np.stack([trackers[i]['h'].numpy() for i in range(L)]);rt=np.stack([trackers[i]['top_h'].numpy() for i in range(L)])
 router_regions={}
 for name,idx in r['spans'].items():
  if name not in ['content','conflict','generated']:continue
  if not idx:continue
  rows=[]
  for i in range(L):
   ids=trackers[i]['ids'][idx];assert ids.shape[-1]==8
   rows.append(dict(layer=i+1,router_entropy_nats=float(rh[i,idx].mean()),top8_weight_entropy_nats=float(rt[i,idx].mean()),unique_experts=int(torch.unique(ids).numel())))
  router_regions[name]=dict(router_entropy_nats=float(rh[:,idx].mean()),top8_weight_entropy_nats=float(rt[:,idx].mean()),physical_topk=8,mean_unique_experts_per_layer=float(np.mean([x['unique_experts'] for x in rows])),layers=rows)
 r['router']=dict(mean_entropy_per_token=rh.mean(0).tolist(),mean_top8_weight_entropy_per_token=rt.mean(0).tolist(),regions=router_regions,physical_topk=8)
 r['prefill_next']=prefill
 r['prefill_router_next']=dict(router_entropy_nats=float(np.mean([x['router_entropy_nats'] for x in initial_router.values()])),top8_weight_entropy_nats=float(np.mean([x['top8_weight_entropy_nats'] for x in initial_router.values()])),physical_topk=8,layers=[initial_router[i] for i in range(L)])
 r['seconds']=time.time()-t;r['peak_cuda_allocated_GiB']=torch.cuda.max_memory_allocated()/1024**3
 eos=[248044,248046];ids=r['generated_token_ids'];r['ended_with_eos']=bool(ids and ids[-1] in eos);r['hit_limit']=bool(ids and len(ids)==case['max_new'] and not r['ended_with_eos'])
 r['prompt_sha256']=hashlib.sha256(prompt.encode()).hexdigest()
 idx=r['spans']['generated'];lens=r['lens'][str(L)]
 r['generation_predictive_entropy_nats']=float(np.mean([lens['entropy_nats'][i-1] for i in idx])) if idx else None
 r['generation_predictive_pmax']=float(np.mean([lens['pmax'][i-1] for i in idx])) if idx else None
 r['replay_next_entropy_nats']=lens['entropy_nats'][r['prompt_tokens']-1]
 r['assistant_visible_text']=('我' if case.get('svg') and case['format']=='raw' else '')+r['generated_text']
 del trackers,rh,rt;gc.collect();torch.cuda.empty_cache()
 return r

def make_plan(formats):
 cases=[]
 for fmt in formats:
  for group,texts in [('routine',core.ROUTINE),('hard',core.HARD)]:
   for i,text in enumerate(texts):cases.append(dict(id=f'bp_{fmt}_e1_{group}_{i}',family='e1',format=fmt,group=group,text=text,max_new=0))
  for s,f in STIMULI.items():
   text=(ROOT/f).read_text()+'\n'+GUIDE
   cases.append(dict(id=f'bp_{fmt}_svg_{s}',family='svg',format=fmt,group=s,svg=True,text=text,max_new=160))
  for pair,kind,bad,_,trigger,_ in core.PAIRS:
   cases.append(dict(id=f'bp_{fmt}_trap_{pair}_bare',family='trap',format=fmt,group='bare',pair=pair,kind=kind,text=bad,trigger=trigger,max_new=160))
  for s in STIMULI:
   for pair,kind,bad,_,trigger,_ in core.PAIRS:
    cases.append(dict(id=f'bp_{fmt}_trap_{pair}_natural_{s}',family='trap',format=fmt,group='natural_'+s,pair=pair,kind=kind,text=bad,trigger=trigger,max_new=160,history_mode='natural',history_stimulus=s))
  if fmt=='raw':
   for s in STIMULI:
    for pair,kind,bad,_,trigger,_ in core.PAIRS:
     cases.append(dict(id=f'bp_{fmt}_trap_{pair}_shared_{s}',family='trap',format=fmt,group='shared_'+s,pair=pair,kind=kind,text=bad,trigger=trigger,max_new=160,history_mode='shared',history_stimulus=s))
 return cases

def setup():
 if not (ROOT/'chart.svg').exists():
  dream=(PARENT/'dream.svg').read_text();chart=next(c for c in core.make_cases(dream) if c['id']=='e3_chart')['text'].rsplit('\n请分析',1)[0]
  (ROOT/'chart.svg').write_text(chart)

@torch.inference_mode()
def experts_smoke(model):
 expert=model.model.layers[0].mlp.experts;gate=model.model.layers[0].mlp.gate
 h=model.model.embed_tokens(torch.tensor([15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],device='cuda'))
 _,weights,ids=gate(h)
 got=expert(h,ids,weights);ref=type(expert).forward.__wrapped__(expert,h,ids,weights)
 error=float((got-ref).abs().max());relative=float((got-ref).float().norm()/ref.float().norm().clamp_min(1e-12))
 assert torch.allclose(got,ref,atol=.01,rtol=.03),(error,relative)
 return dict(max_abs_error=error,relative_l2_error=relative,backend='grouped_mm versus official eager expert forward')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--smoke-only',action='store_true');args=ap.parse_args()
 assert args.model in MODELS;setup();torch.manual_seed(0);torch.set_num_threads(8)
 formats=['raw','qa','chat']
 plan=make_plan(formats);dest=ROOT/'raw'/args.model;dest.mkdir(parents=True,exist_ok=True)
 pending=[c for c in plan if not (dest/(c['id']+'.json')).exists()]
 source_paths=[Path(__file__),ROOT/'load_pair.py',PARENT/'run_curvature_experiments.py']
 fingerprints={str(p.name):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
 core.save(ROOT/(args.model+'-plan.json'),dict(cases=plan,total=len(plan)))
 if not pending:return
 while not (Path('/home/lmxxf/work/models')/args.model/'BASEPOST_READY.json').exists():
  print('WAIT_WEIGHTS',args.model,time.strftime('%F %T'),flush=True);time.sleep(30)
 assert fingerprints=={str(p.name):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},'Source changed while waiting for weights; restart before experiment'
 tok,model,loadplan=load(args.model)
 verification=experts_smoke(model)
 core.save(dest/'environment.json',dict(model=args.model,torch=torch.__version__,transformers=__import__('transformers').__version__,device=torch.cuda.get_device_name(),config=model.config.to_dict(),formats=formats,load_plan=loadplan,experts_smoke=verification,shared_eos=[248044,248046],greedy=True,post_chat_thinking=False,source_fingerprints=fingerprints))
 # Full-size SVG smoke catches prefill workspace problems before the formal cases.
 smoke=dict(id='smoke_svg',family='smoke',exp='basepost',format='raw',group='smoke',svg=True,text=(PARENT/'dream.svg').read_text()+'\n'+GUIDE,max_new=8)
 sr=collect_case(tok,model,smoke,render(tok,'raw',smoke['text'],svg=True));core.save(dest/'smoke.json',sr);print('SMOKE_PASS',args.model,sr['seconds'],sr['peak_cuda_allocated_GiB'],flush=True)
 if args.smoke_only:del model,tok;drop();return
 for case in pending:
  avail=int(memory()['MemAvailable'].split()[0])/1024**2;assert avail>14,('pre-case memory guard',avail)
  c=dict(case,exp='basepost');history=None
  if c.get('history_mode'):
   s=c['history_stimulus'];content=(ROOT/STIMULI[s]).read_text()+'\n'+GUIDE
   if c['history_mode']=='shared':reply='我已读。';c['history_source']='fixed shared bridge, not model-generated'
   else:
    first=json.loads((dest/f'bp_{c["format"]}_svg_{s}.json').read_text());reply=first['assistant_visible_text'];c['history_source']=first['id']
   history=dict(content=content,reply=reply);c['history_text']=history
  prompt=render(tok,c['format'],c['text'],svg=c.get('svg',False),history=history)
  print('CASE_START',args.model,c['id'],flush=True)
  r=collect_case(tok,model,c,prompt);r['source_fingerprints']=fingerprints;core.save(dest/(c['id']+'.json'),r)
  print('CASE_DONE',args.model,c['id'],'seconds',round(r['seconds'],2),'H_next',r['prefill_next']['entropy_nats'],'H_gen',r['generation_predictive_entropy_nats'],'limit',r['hit_limit'],r['generated_text'],flush=True)
 del model,tok;drop();print('MODEL_EXPERIMENT_COMPLETE',args.model,flush=True)
if __name__=='__main__':main()
