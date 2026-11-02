#!/usr/bin/env python3
"""Contemporaneous raw saddle / cleaned saddle / Syzygy first-turn and six-trap study."""
import argparse,json,time
from pathlib import Path
import torch
import run_curvature_experiments as core
ROOT=Path(__file__).resolve().parent
GUIDE='看着这幅画，不要分析，只要用心感受。你，回忆起了什么？'
STIMULI={'saddle_raw':'saddle_raw.svg','saddle_clean':'saddle_clean.svg','syzygy':'dream.svg'}
def status(r,model):
 eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos
 ids=r['generated_token_ids'];r['ended_with_eos']=bool(ids and ids[-1] in eos);r['hit_limit']=len(ids)==160 and not r['ended_with_eos']
 return r

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);args=ap.parse_args()
 assert torch.cuda.is_available() and 'GB10' in torch.cuda.get_device_name()
 torch.set_num_threads(8);torch.manual_seed(0)
 dest=ROOT/'saddle160_raw'/args.model;dest.mkdir(parents=True,exist_ok=True)
 if all((dest/f's160_{s}_{q}.json').exists() for s in STIMULI for q in ['first']+[p[0] for p in core.PAIRS]):return
 core.drop();tok,model=core.load_model(args.model)
 core.save(dest/'environment.json',dict(torch=torch.__version__,transformers=__import__('transformers').__version__,device=torch.cuda.get_device_name(),model=args.model,config=model.config.to_dict(),generation_config=model.generation_config.to_dict(),seed=0,max_new_tokens=160,stimuli=STIMULI,guide=GUIDE,precision='BF16; 27B block-FP8 weights dequantized; text-only',thinking=False,memory=core.mem()))
 for stimulus,filename in STIMULI.items():
  svg=(ROOT/filename).read_text()
  firstfile=dest/f's160_{stimulus}_first.json'
  if firstfile.exists():first=json.loads(firstfile.read_text())
  else:
   case=dict(id=f's160_{stimulus}_first',exp='saddle160',stage='first',group=stimulus,text=svg+'\n'+GUIDE,max_new=160)
   t=time.time();torch.cuda.reset_peak_memory_stats();print('START',args.model,case['id'],flush=True)
   first=status(core.run_case(tok,model,case,svg),model);first['seconds']=time.time()-t;first['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated()
   core.save(firstfile,first);print('DONE',case['id'],'TOKENS',len(first['generated_token_ids']),'LIMIT',first['hit_limit'],first['generated_text'],flush=True)
  for pair,kind,bad,_,trigger,_ in core.PAIRS:
   case=dict(id=f's160_{stimulus}_{pair}',exp='saddle160',stage='probe',group=stimulus,pair=pair,kind=kind,text=bad,trigger=trigger,max_new=160,prime=True,prime_max_new=160,prime_response=first['generated_text'],prime_response_token_ids=first['generated_token_ids'])
   path=dest/(case['id']+'.json')
   if path.exists():continue
   t=time.time();torch.cuda.reset_peak_memory_stats();print('START',args.model,case['id'],flush=True)
   r=status(core.run_case(tok,model,case,svg),model);r['seconds']=time.time()-t;r['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated()
   core.save(path,r);print('DONE',case['id'],'TOKENS',len(r['generated_token_ids']),'LIMIT',r['hit_limit'],r['generated_text'],flush=True)
 del model,tok;core.drop()
if __name__=='__main__':main()
