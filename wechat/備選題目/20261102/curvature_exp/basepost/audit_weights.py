"""Read only small CPU slices to verify that the effective BF16 checkpoints really differ."""
from pathlib import Path
import json,torch,hashlib
from safetensors import safe_open
ROOT=Path(__file__).resolve().parent
MODELS=Path('/home/lmxxf/work/models')
NAMES=['Qwen3.5-35B-A3B-Base','Qwen3.5-35B-A3B']
KEYS=['lm_head.weight','model.language_model.embed_tokens.weight','model.language_model.norm.weight','model.language_model.layers.0.input_layernorm.weight','model.language_model.layers.0.mlp.gate.weight','model.language_model.layers.0.mlp.experts.gate_up_proj','model.language_model.layers.19.mlp.experts.down_proj','model.language_model.layers.39.self_attn.q_proj.weight','model.language_model.layers.39.mlp.experts.gate_up_proj']
def samples(name,key):
 d=MODELS/name;index=json.loads((d/'model.safetensors.index.json').read_text())['weight_map']
 with safe_open(str(d/index[key]),framework='pt',device='cpu') as f:
  s=f.get_slice(key);shape=s.get_shape();rank=len(shape)
  starts=sorted(set([0,shape[0]//2,shape[0]-1]));arr=[]
  if rank==1:arr=[s[:min(256,shape[0])]]
  elif rank==2:arr=[s[i:i+1,:128] for i in starts]
  elif rank==3:arr=[s[i:i+1,0:1,:128] for i in starts]
  else:raise ValueError((key,shape))
  return torch.cat([a.flatten().to(torch.bfloat16).float() for a in arr])
def main():
 rows=[]
 for k in KEYS:
  a=samples(NAMES[0],k);b=samples(NAMES[1],k);assert a.shape==b.shape
  rows.append(dict(key=k,elements=a.numel(),different=int((a!=b).sum()),max_abs_diff=float((a-b).abs().max()),mean_abs_diff=float((a-b).abs().mean()),base_sample_sha256=hashlib.sha256(a.numpy().tobytes()).hexdigest(),post_sample_sha256=hashlib.sha256(b.numpy().tobytes()).hexdigest()))
 result=dict(effective_dtype='BF16',different_keys=sum(r['different']>0 for r in rows),rows=rows)
 assert result['different_keys']>0,'All selected effective weight samples identical; investigate pair before training-phase attribution'
 (ROOT/'weight_sample_audit.json').write_text(json.dumps(result,indent=2));return result
if __name__=='__main__':print(json.dumps(main(),indent=2))
