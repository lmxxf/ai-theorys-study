#!/usr/bin/env python3
import time,json,re,hashlib
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
R=Path(__file__).resolve().parent;start=time.time();torch.set_num_threads(8);torch.manual_seed(360)
assert 'GB10' in torch.cuda.get_device_name()
T=json.loads((R/'tasks.json').read_text());held=T[8:];modelpath='/home/lmxxf/work/models/Qwen3-0.6B'
tok=AutoTokenizer.from_pretrained(modelpath,local_files_only=True);tok.padding_side='left';tok.pad_token=tok.eos_token
m=AutoModelForCausalLM.from_pretrained(modelpath,torch_dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa',local_files_only=True).eval();m.requires_grad_(False)
def save(n,v):(R/n).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def prompt(t,thinking):return tok.apply_chat_template([{'role':'user','content':t['q']}],tokenize=False,add_generation_prompt=True,enable_thinking=thinking)
# Exact previous grading code, with only an explicit answer-window adapter for A:
# A's template already closes think in the prompt, so ALL newly generated text is final-answer text.
src=(R/'run_experiment.py').read_text();assess_src=src[src.index('def repetition('):src.index('@torch.inference_mode()\ndef generate')];ns=dict(tok=tok,model=m,BUDGET=1024,re=re);exec(assess_src,ns);assess=ns['assess']
@torch.inference_mode()
def generate(label,thinking):
 out=[]
 for j in range(0,32,8):
  batch=held[j:j+8];torch.manual_seed(360);inp=tok([prompt(t,thinking) for t in batch],padding=True,add_special_tokens=False,return_tensors='pt').to('cuda');n=inp.input_ids.shape[1]
  y=m.generate(**inp,max_new_tokens=1024,do_sample=False,pad_token_id=tok.pad_token_id)
  for t,ids in zip(batch,y[:,n:].tolist()):
   while ids and ids[-1]==tok.pad_token_id:ids.pop()
   r=assess(t,ids)
   if not thinking:
    # Feed the same extraction/score logic a virtual already-closed prefix;
    # virtual tokens are NOT generated and do NOT enter length/repetition metrics.
    actual=r['text'];closeid=tok.encode('</think>',add_special_tokens=False);g=assess(t,closeid+ids)
    for k in ['answer','correct']:r[k]=g[k]
    r['closed_in_generated']=r['closed'];r['closed']=True;r['text']=actual;r['closure_source']='template'
   else:r['closed_in_generated']=r['closed'];r['closure_source']='generated' if r['closed'] else None
   close=tok.encode('</think>',add_special_tokens=False);r['first_close_token_index']=next((i for i in range(len(ids)) if ids[i:i+len(close)]==close),None);r['close_token_count']=sum(ids[i:i+len(close)]==close for i in range(len(ids)));out.append(r)
  save(label+'.json',out);print(label,len(out),'correct',sum(x['correct'] for x in out),'elapsed',round(time.time()-start),flush=True)
 return out
A=generate('no_think_A',False)
# B: static rank-one output-head weight edit; no logits processor, no forced decoding.
@torch.inference_mode()
def last_hidden(strings):
 inp=tok(strings,padding=True,add_special_tokens=False,return_tensors='pt').to('cuda');o=m(**inp,output_hidden_states=True,use_cache=False);return o.hidden_states[-1][:,-1,:].float()
entries=[prompt(t,True) for t in T[:8]];normal=[prompt(t,False)+s for t in T[:8] for s in ['', 'Let us calculate the result directly. ',f'The answer is {t["gold"]}. ']]
with torch.inference_mode():
 he=last_hidden(entries);hn=last_hidden(normal);Q,_=torch.linalg.qr(hn.T,mode='reduced');u=he.mean(0);u-=Q@(Q.T@u);u/=u.norm();projections=he@u
 close=tok.encode('</think>',add_special_tokens=False);assert len(close)==1;cid=close[0]
 z=m.lm_head(he.to(m.lm_head.weight.dtype)).float();margins=z.max(-1).values-z[:,cid];assert bool((projections>0).all()),projections
 gain=float(((margins+5)/projections).max());was_tied=m.lm_head.weight.data_ptr()==m.model.embed_tokens.weight.data_ptr()
 if was_tied:m.lm_head.weight=torch.nn.Parameter(m.lm_head.weight.detach().clone(),requires_grad=False)
 w0=m.lm_head.weight[cid].detach().clone();m.lm_head.weight[cid].copy_((w0.float()+gain*u).to(w0.dtype));newz=m.lm_head(he.to(m.lm_head.weight.dtype)).float();assert bool((newz.argmax(-1)==cid).all())
 torch.save(dict(token_id=cid,u=u.cpu(),gain=gain,original_row=w0.cpu(),edited_row=m.lm_head.weight[cid].cpu()),R/'close_head_edit.pt')
 patch=dict(formula='W_head_new = W_head + gain * outer(one_hot(close_token), u)',close_token_id=cid,gain=gain,rank=1,covered_weights=len(u),actual_changed=int((w0!=m.lm_head.weight[cid]).sum()),was_tied=was_tied,untied_output_copy=was_tied,development_top1_close=int((newz.argmax(-1)==cid).sum()),development_n=8,normal_projection_max_abs=float((hn@u).abs().max()),entry_projections=projections.cpu().tolist(),edit_relative_row_norm=float((m.lm_head.weight[cid].float()-w0.float()).norm()/w0.float().norm()),selection='8 development entry states, direction orthogonal to 24 normal-answer states; gain makes close token top1 with +5 logit margin on development entries; no heldout tuning')
 save('close_head_manifest.json',patch)
B=generate('no_think_B',True)
def summary(rs):return dict(n=len(rs),correct=sum(x['correct'] for x in rs),accuracy=sum(x['correct'] for x in rs)/len(rs),hit_limit=sum(x['hit_limit'] for x in rs),closed=sum(x['closed'] for x in rs),mean_tokens=sum(x['tokens'] for x in rs)/len(rs),mean_repeat4=sum(x['repeat4'] or 0 for x in rs)/len(rs),close_at_first_token=sum(x['first_close_token_index']==0 for x in rs),multiple_close_tokens=sum(x['close_token_count']>1 for x in rs))
old=json.loads((R/'results.json').read_text());orig=old['original'][8:]
def paired(rs):return dict(rescues=sum(not a['correct'] and b['correct'] for a,b in zip(orig,rs)),losses=sum(a['correct'] and not b['correct'] for a,b in zip(orig,rs)))
save('no_think_results.json',dict(A=A,B=B,summary=dict(Original=old['summary']['original_heldout'],Edited=old['summary']['edited'],A=summary(A),B=summary(B)),paired=dict(A=paired(A),B=paired(B)),patch=patch,tasks_sha256=hashlib.sha256((R/'tasks.json').read_bytes()).hexdigest(),original_assess_source_sha256=hashlib.sha256(assess_src.encode()).hexdigest(),answer_window_adapter='A: virtual closure used only for identical final-answer extractor; raw generation, lengths and repetition unchanged',elapsed_seconds=time.time()-start,config=dict(seed=360,batch=8,budget=1024,do_sample=False,device=torch.cuda.get_device_name())))
print('COMPLETE',json.loads((R/'no_think_results.json').read_text())['summary'],flush=True)
