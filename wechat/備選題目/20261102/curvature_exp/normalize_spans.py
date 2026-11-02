#!/usr/bin/env python3
"""Recompute scalar windows from raw geometry, excluding EOS/control IDs from answer text."""
from pathlib import Path
import json,numpy as np
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parent
for d in (ROOT/'raw').iterdir():
 tok=AutoTokenizer.from_pretrained('/home/lmxxf/work/models/'+d.name,local_files_only=True)
 special=set(tok.all_special_ids)
 for p in d.glob('e[1234]*.json'):
  r=json.loads(p.read_text());T=r['total_tokens'];L=len(r['temporal_rad'])
  r['special_token_ids']=tok.all_special_ids
  r['spans']['generated_with_special']=list(range(r['prompt_tokens'],T))
  r['spans']['generated']=[i for i in r['spans']['generated_with_special'] if r['token_ids'][i] not in special]
  if 'answer_marker' in r['spans']:r['spans']['answer']=r['spans']['answer_marker']+r['spans']['generated']
  temp=np.array([[np.nan if z is None else z for z in a] for a in r['temporal_one_minus_cos']])
  mid=[i-1 for i in r['mid_layer_indices_one_based']]
  k=np.array([np.nan if z is None else z for z in r['mid_temp_rad']]);kl=np.array(r['mid_layer_one_minus_cos'])
  ent=np.array(r['lens'][str(L)]['entropy_nats']);pm=np.array(r['lens'][str(L)]['pmax'])
  for name,idx in r['spans'].items():
   idx=[i for i in idx if i>=2]
   r['summary'][name]=(dict(n=len(idx),temp_rad=float(np.nanmean(k[idx])),temp_one_minus_cos=float(np.nanmean(temp[mid][:,idx])),layer_one_minus_cos=float(np.nanmean(kl[idx])),entropy_nats=float(ent[idx].mean()),pmax=float(pm[idx].mean())) if idx else None)
  p.write_text(json.dumps(r,ensure_ascii=False,allow_nan=False))
print('Normalized response spans; raw token IDs and per-token arrays preserved.')
