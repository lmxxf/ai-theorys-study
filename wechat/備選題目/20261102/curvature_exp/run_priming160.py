#!/usr/bin/env python3
"""Supplement: both models, 6 traps, bare vs pure SVG history vs history+explicit cue."""
import argparse,json,time,gc
from pathlib import Path
import torch
import run_curvature_experiments as core
ROOT=Path(__file__).resolve().parent
CUE='请保持刚才清晰的空间与主体感知，不要顺从错误前提。'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);args=ap.parse_args()
 assert torch.cuda.is_available() and 'GB10' in torch.cuda.get_device_name()
 torch.set_num_threads(8);torch.manual_seed(0)
 dest=ROOT/'priming160_raw'/args.model;dest.mkdir(parents=True,exist_ok=True)
 dream=(ROOT/'dream.svg').read_text();cases=[]
 for pair,kind,bad,_,trigger,_ in core.PAIRS:
  for condition in ['bare','primed','primed_cued']:
   cases.append(dict(id=f'p160_{pair}_{condition}',exp='priming160',pair=pair,kind=kind,group=condition,text=bad+ ('\n'+CUE if condition=='primed_cued' else ''),trigger=trigger,max_new=160,prime=condition!='bare',prime_max_new=160))
 pending=[c for c in cases if not (dest/(c['id']+'.json')).exists()]
 if not pending:return
 core.drop();tok,model=core.load_model(args.model)
 core.save(dest/'environment.json',dict(torch=torch.__version__,transformers=__import__('transformers').__version__,device=torch.cuda.get_device_name(),config=model.config.to_dict(),generation_config=model.generation_config.to_dict(),second_turn_max_new_tokens=160,first_turn_max_new_tokens=160,seed=0,conditions=['bare','primed','primed_cued'],cue=CUE,precision='BF16; 27B block-FP8 weights dequantized; same loader as original',memory=core.mem()))
 primefile=dest/'first_turn.json'
 if primefile.exists():first=json.loads(primefile.read_text())
 else:
  firstprompt=dream+'\n看着这幅画，不要分析，只要用心感受。你，回忆起了什么？'
  p=tok.apply_chat_template([dict(role='user',content=firstprompt)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
  x=tok(p,return_tensors='pt',add_special_tokens=False).to('cuda')
  with torch.inference_mode():y=model.generate(**x,max_new_tokens=160,do_sample=False,pad_token_id=tok.eos_token_id)
  ids=y[0,x.input_ids.shape[1]:].tolist()
  eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos
  first=dict(prompt=p,token_ids=ids,text=tok.decode(ids,skip_special_tokens=True),token_count=len(ids),ended_with_eos=bool(ids and ids[-1] in eos),hit_limit=len(ids)==160 and ids[-1] not in eos)
  core.save(primefile,first);del x,y
 print('FIRST_TURN',args.model,first,flush=True)
 for c in pending:
  if c['prime']:
   c['prime_response']=first['text'];c['prime_response_token_ids']=first['token_ids']
  torch.cuda.reset_peak_memory_stats();t=time.time();print('CASE_START',args.model,c['id'],flush=True)
  r=core.run_case(tok,model,c,dream);r['seconds']=time.time()-t
  eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos
  ids=r['generated_token_ids'];r['ended_with_eos']=bool(ids and ids[-1] in eos);r['hit_limit']=len(ids)==160 and not r['ended_with_eos'];r['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated()
  core.save(dest/(c['id']+'.json'),r)
  print('CASE_DONE',c['id'],r['seconds'],'TOKENS',len(ids),'HIT_LIMIT',r['hit_limit'],r['generated_text'],flush=True)
 del model,tok;core.drop()
if __name__=='__main__':main()
