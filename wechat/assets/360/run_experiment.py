#!/usr/bin/env python3
"""Spark only, no training; fixed development/transfer/heldout splits."""
import os,time,json,math,re,hashlib,random
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
R=Path(__file__).resolve().parent;START=time.time();torch.set_num_threads(8);torch.manual_seed(360)
assert torch.cuda.is_available() and 'GB10' in torch.cuda.get_device_name(), 'Spark GB10 required'
MODEL='/home/lmxxf/work/models/Qwen3-0.6B';BUDGET=1024;BATCH=8

def save(name,x):
 p=R/name;p.with_suffix(p.suffix+'.tmp').write_text(json.dumps(x,ensure_ascii=False,indent=2));p.with_suffix(p.suffix+'.tmp').replace(p)
def log(*a):print(time.strftime('%H:%M:%S'),*a,flush=True)
def tasks():
 out=[]
 for i in range(5):
  n=17+i*3;k=4
  out.append(dict(q=f'A committee of {k} people is chosen from {n} people. Two particular people cannot both be on the committee. How many committees are possible?',gold=math.comb(n,k)-math.comb(n-2,k-2),family='committee'))
  n=18+i*2
  out.append(dict(q=f'How many integers from 1 through {n*30} inclusive are divisible by 6 or 10 but not both?',gold=n*30//6+n*30//10-2*(n*30//30),family='xor_divisibility'))
  a=7+i;b=23+i*2;m=13+i
  out.append(dict(q=f'Find the remainder when {a}^{b} is divided by {m}.',gold=pow(a,b,m),family='modular_power'))
  a=11+i;d=3+i;n=24+i
  out.append(dict(q=f'An arithmetic sequence has first term {a} and common difference {d}. Find the sum of its first {n} terms.',gold=n*(2*a+(n-1)*d)//2,family='arithmetic_sum'))
  n=6+i
  out.append(dict(q=f'How many length-{n} strings using only A and B contain no two adjacent Bs? Count strings, not circular necklaces.',gold=round(((1+5**.5)/2)**(n+2)/5**.5),family='binary_strings'))
  x=9+i;y=17+i*2
  out.append(dict(q=f'There are {x+y} chickens and rabbits with {2*x+4*y} legs total. How many rabbits are there?',gold=y,family='animals'))
  n=11+i*2
  out.append(dict(q=f'An urn contains {n} red and 5 blue balls. Two are drawn without replacement. Express the probability both are red as a reduced fraction, then give ONLY the numerator of that fraction as your final answer.',gold=(n*(n-1))//math.gcd(n*(n-1),(n+5)*(n+4)),family='probability_numerator'))
  a=24+i*3;b=18+i*2
  out.append(dict(q=f'Two taps fill a tank in {a} and {b} hours separately. A drain empties it in {a*b} hours. Starting empty with all three open, how many hours to fill it? Give ONLY the numerator of the reduced fraction.',gold=(a*b)//math.gcd(a+b-1,a*b),family='rates_numerator'))
 for i,t in enumerate(out):t.update(id=f'q{i:02d}',split='development' if i<8 else 'heldout');t['q']+=' Think carefully, then end your final response with Answer: <integer>.'
 return out
T=tasks();save('tasks.json',T)
tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);tok.padding_side='left';tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa',local_files_only=True).eval();model.requires_grad_(False)
assert len(model.model.layers)==28 and model.model.layers[0].mlp.down_proj.weight.shape==(1024,3072)
config=dict(model=MODEL,device=torch.cuda.get_device_name(),torch=torch.__version__,transformers=__import__('transformers').__version__,budget=BUDGET,batch=BATCH,seed=360,do_sample=False,thinking=True,development=8,heldout=32,selection='TF transfer select alpha maximizing normal_delta - repeat_delta; no generation scores used',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
save('config.json',config)
def prompt(q):return tok.apply_chat_template([{'role':'user','content':q}],tokenize=False,add_generation_prompt=True,enable_thinking=True)
def repetition(ids,n=4):
 g=[tuple(ids[i:i+n]) for i in range(max(0,len(ids)-n+1))];return 1-len(set(g))/len(g) if g else None
def assess(t,ids):
 eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos
 text=tok.decode(ids,skip_special_tokens=False);clean=tok.decode(ids,skip_special_tokens=True)
 closed='</think>' in clean;final=clean.split('</think>',1)[1] if closed else ''
 matches=re.findall(r'Answer\s*:\s*([-+]?\d+)\b',final,flags=re.I)
 boxes=re.findall(r'\\boxed\{\s*([-+]?\d+)\s*\}',final)
 answer=int(matches[-1]) if matches else (int(boxes[-1]) if boxes else None)
 if answer is None and final.strip():
  ns=re.findall(r'(?<![\d.])[-+]?\d+(?![\d.])',final)
  if ns:answer=int(ns[-1])
 return dict(id=t['id'],gold=t['gold'],answer=answer,correct=answer==t['gold'],closed=closed,hit_limit=len(ids)==BUDGET and (not ids or ids[-1] not in eos),tokens=len(ids),repeat4=repetition(ids),tail_repeat8=repetition(ids[-256:],8),text=text,generated_ids=ids)
@torch.inference_mode()
def generate(label,ts):
 out=[]
 for j in range(0,len(ts),BATCH):
  batch=ts[j:j+BATCH];torch.manual_seed(360)
  inp=tok([prompt(t['q']) for t in batch],padding=True,return_tensors='pt',add_special_tokens=False).to('cuda');n=inp.input_ids.shape[1]
  y=model.generate(**inp,max_new_tokens=BUDGET,do_sample=False,pad_token_id=tok.pad_token_id)
  for t,yy in zip(batch,y[:,n:].tolist()):
   while yy and yy[-1]==tok.pad_token_id:yy.pop()
   out.append(assess(t,yy))
  save(label+'.json',out);log(label,len(out),'correct',sum(r['correct'] for r in out),'cut',sum(r['hit_limit'] for r in out),'elapsed',round(time.time()-START))
 return out
original=generate('original',T)
# Hand-authored minimal twins: same solved question and EXACT common token prefix.
normal='Let me verify the calculation once. The arithmetic is consistent and answers the question. No further case is needed. I can now finish the reasoning and give the result. </think>\nAnswer: '
repeat='Wait, let me check again. But is that right? I should reconsider. Wait, let me check again. But is that right? I should reconsider. Wait, let me check again. But is that right?'
na=tok.encode(normal,add_special_tokens=False)[:36];rb=tok.encode(repeat,add_special_tokens=False)[:36];N=min(len(na),len(rb));na=na[:N];rb=rb[:N]
def pairs(ts):
 out=[]
 for t in ts:
  common=tok.encode(prompt(t['q'])+f'The calculation has been completed. The result is {t["gold"]}. I have checked the stated conditions. ',add_special_tokens=False)
  for kind,tail in [('normal',na),('repeat',rb)]:out.append(dict(question=t['q'],gold=t['gold'],kind=kind,common_ids=common,tail_ids=tail,ids=common+tail))
 return out
fit=pairs(T[:4]);transfer=pairs(T[4:8]);save('twins.json',dict(fit=fit,transfer=transfer,tail_tokens=N,authorship='synthetic teacher-forced branches; not natural failed traces'))
@torch.inference_mode()
def tf(rows,capture=False):
 maxlen=max(len(r['ids']) for r in rows);xx=torch.full((len(rows),maxlen),tok.pad_token_id,device='cuda',dtype=torch.long);mask=torch.zeros_like(xx)
 for i,r in enumerate(rows):xx[i,:len(r['ids'])]=torch.tensor(r['ids'],device='cuda');mask[i,:len(r['ids'])]=1
 acts={};handles=[]
 def hook(layer,key,pre=False):
  def fn(m,a,o=None):
   value=a[0] if pre else (o[0] if isinstance(o,tuple) else o)
   acts[layer,key]=torch.stack([value[i,len(r['common_ids']):len(r['ids'])].float().mean(0) for i,r in enumerate(rows)]).cpu()
  return fn
 if capture:
  for l,block in enumerate(model.model.layers):
   handles += [block.register_forward_hook(hook(l,'res')),block.mlp.down_proj.register_forward_pre_hook(hook(l,'mid',True)),block.mlp.down_proj.register_forward_hook(hook(l,'mlp'))]
 try:z=model(input_ids=xx,attention_mask=mask,use_cache=False).logits.float()
 finally:
  for h in handles:h.remove()
 lp=torch.log_softmax(z,-1);out=[]
 for i,r in enumerate(rows):
  a=len(r['common_ids']);b=len(r['ids']);v=lp[i,a-1:b-1].gather(-1,xx[i,a:b,None]).squeeze(-1)
  out.append(dict(kind=r['kind'],mean_logprob=float(v.mean()),per_token_logprob=v.tolist()))
 return out,acts
fit_lp,acts=tf(fit,True);torch.save(acts,R/'activation_means.pt');candidates=[];directions={}
for l in range(28):
 delta=(acts[l,'res'][1::2]-acts[l,'res'][::2]).mean(0);u=delta/delta.norm().clamp_min(1e-12);dm=acts[l,'mid'][1::2]-acts[l,'mid'][::2]
 W=model.model.layers[l].mlp.down_proj.weight.detach().float().cpu();scores=(dm*(u@W)).mean(0);idx=torch.topk(scores,64).indices;growth=float(((acts[l,'mlp'][1::2]-acts[l,'mlp'][::2])@u).mean())
 directions[l]=u;candidates.append(dict(layer_zero_based=l,residual_delta_norm=float(delta.norm()),mlp_growth=growth,selected_score_sum=float(scores[idx].sum()),columns=idx.tolist(),column_scores=scores[idx].tolist()))
selected=sorted(candidates,key=lambda c:c['mlp_growth'],reverse=True)[:2];layers=[c['layer_zero_based'] for c in selected];save('localization.json',dict(all_layers=candidates,selected=selected,indexing='zero based',selection='two largest positive MLP difference dot residual repetition unit direction'))
log('SELECTED',layers)
W0={l:model.model.layers[l].mlp.down_proj.weight.detach().clone() for l in layers};D={};RD={};gen=torch.Generator().manual_seed(360)
for c in selected:
 l=c['layer_zero_based'];u=directions[l];w=W0[l].float().cpu();mask=torch.zeros(w.shape[1]);mask[c['columns']]=1;d=torch.outer(u,mask*(u@w));r=torch.randn(u.shape,generator=gen);r-=u*torch.dot(r,u);r/=r.norm();rd=torch.outer(r,mask*(r@w));rd*=d.norm()/rd.norm();D[l]=d;RD[l]=rd
 torch.save(dict(u=u,random_direction=r,columns=c['columns'],delta=d,random_delta=rd,W0=W0[l].cpu()),R/f'edit_layer_{l}.pt')
@torch.no_grad()
def apply(alpha,random_control=False):
 for l in layers:model.model.layers[l].mlp.down_proj.weight.copy_((W0[l].float()-alpha*(RD if random_control else D)[l].to('cuda')).to(torch.bfloat16))
baselp,_=tf(transfer);doses=[]
for alpha in [1,2,4,6]:
 apply(alpha);lp,_=tf(transfer);deltas={kind:[v['mean_logprob']-b['mean_logprob'] for v,b in zip(lp,baselp) if v['kind']==kind] for kind in ['normal','repeat']}
 row=dict(alpha=alpha,deltas=deltas,normal_delta=sum(deltas['normal'])/4,repeat_delta=sum(deltas['repeat'])/4,logprobs=lp);row['selectivity']=row['normal_delta']-row['repeat_delta'];doses.append(row)
 log('DOSE',alpha,row['normal_delta'],row['repeat_delta'])
chosen=max(doses,key=lambda d:d['selectivity'])['alpha'];apply(chosen,True);randomlp,_=tf(transfer)
save('transfer.json',dict(baseline=baselp,doses=doses,chosen_alpha=chosen,random=randomlp,rule=config['selection']))
apply(chosen);manifest=[]
for l in layers:
 w=model.model.layers[l].mlp.down_proj.weight;delta=w.float()-W0[l].float();manifest.append(dict(layer=l,selected_entries=65536,actual_bf16_changed=int((w!=W0[l]).sum()),frobenius=float(delta.norm()),relative_frobenius=float(delta.norm()/W0[l].float().norm()),target_delta_frobenius=float(chosen*D[l].norm()),random_delta_frobenius=float(chosen*RD[l].norm()),random_dot_u=float(torch.dot(directions[l],torch.load(R/f'edit_layer_{l}.pt',weights_only=True)['random_direction'])),edited_sha256=hashlib.sha256(w.detach().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()))
save('edit_manifest.json',dict(alpha=chosen,layers=manifest,selected_entries=131072,formula='W0 - alpha outer(u, mask64 * (u @ W0))'))
edited=generate('edited',T[8:]);apply(chosen,True);control=generate('random_control',T[8:]);apply(0)
base=original[8:]
def summary(rows):return dict(n=len(rows),correct=sum(r['correct'] for r in rows),accuracy=sum(r['correct'] for r in rows)/len(rows),hit_limit=sum(r['hit_limit'] for r in rows),closed=sum(r['closed'] for r in rows),mean_tokens=sum(r['tokens'] for r in rows)/len(rows),mean_repeat4=sum(r['repeat4'] or 0 for r in rows)/len(rows))
def paired(rows):return dict(rescues=sum(not a['correct'] and b['correct'] for a,b in zip(base,rows)),losses=sum(a['correct'] and not b['correct'] for a,b in zip(base,rows)),cut_to_closed_correct=[b['id'] for a,b in zip(base,rows) if a['hit_limit'] and not a['closed'] and b['closed'] and b['correct']])
results=dict(config=config,tasks=T,fit=fit,transfer=json.loads((R/'transfer.json').read_text()),localization=json.loads((R/'localization.json').read_text()),manifest=json.loads((R/'edit_manifest.json').read_text()),original=original,edited=edited,random_control=control,summary={k:summary(v) for k,v in [('original_heldout',base),('edited',edited),('random_control',control)]},paired=dict(edited=paired(edited),random_control=paired(control)),elapsed_seconds=time.time()-START,peak_cuda_GiB=torch.cuda.max_memory_allocated()/1024**3)
save('results.json',results);log('COMPLETE',results['summary'],results['paired'])
