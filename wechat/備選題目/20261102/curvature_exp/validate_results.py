#!/usr/bin/env python3
"""Validate completeness, token indexing and numeric ranges of experiment artifacts."""
from pathlib import Path
import json,math
p=Path(__file__).resolve().parent
r=json.loads((p/'results.json').read_text())
counts={}; failures=[]
for name,model in r['models'].items():
 cases=model['cases'];counts[name]=len(cases)
 assert len(cases)==(40 if name=='Qwen3-0.6B' else 46)
 assert len({c['id'] for c in cases})==len(cases)
 expected_layers=28 if name=='Qwen3-0.6B' else 64
 for c in cases:
  T=c['total_tokens'];L=len(c['temporal_rad']);assert L==expected_layers
  assert len(c['token_ids'])==len(c['tokens'])==T
  assert c['prompt_tokens']+len(c['generated_token_ids'])==T
  assert len(c['layer_one_minus_cos'])==L-1
  assert c['final_logit_hook_max_error']<.1
  for k in ['temporal_rad','temporal_one_minus_cos','layer_one_minus_cos']:
   for a in c[k]:
    assert len(a)==T
    assert all(x is None or math.isfinite(x) and -1e-6<=x<=(math.pi+1e-6 if k=='temporal_rad' else 2+1e-6) for x in a)
  for label,idx in c['spans'].items():assert all(0<=i<T for i in idx),(name,c['id'],label)
  for lens in c['lens'].values():
   assert len(lens['entropy_nats'])==len(lens['pmax'])==len(lens['argmax_token_ids'])==T
   assert all(0<=x<=math.log(model['environment']['config']['vocab_size'])+1e-4 for x in lens['entropy_nats'])
   assert all(0<=x<=1+1e-6 for x in lens['pmax'])
  for label,v in c['summary'].items():
   if v:assert all(math.isfinite(x) for x in v.values())
result=dict(status='PASS',case_counts=counts,total=sum(counts.values()),checks=['expected model/layer/case counts','unique case IDs','token spans and generation lengths','geometry and entropy ranges','logit lens lengths','hook/final-output equality','finite summaries'])
(p/'validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
