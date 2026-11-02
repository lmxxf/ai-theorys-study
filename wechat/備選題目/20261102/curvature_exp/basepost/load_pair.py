"""Identical, text-only BF16 streaming loader for both official 35B-A3B checkpoints."""
import os,json,gc,time
from pathlib import Path
import torch
from safetensors import safe_open
from transformers import AutoConfig,AutoTokenizer,Qwen3_5MoeForCausalLM
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextRotaryEmbedding
MODELS=Path('/home/lmxxf/work/models')
def memory():
 return {s.split(':')[0]:s.split(':')[1].strip() for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith(('MemAvailable:','SwapFree:'))}
def drop():
 gc.collect();torch.cuda.empty_cache();os.system("sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches'")
def parameter_plan(name):
 path=MODELS/name;cfg=AutoConfig.from_pretrained(path,local_files_only=True).text_config
 cfg._attn_implementation='sdpa';cfg._experts_implementation='grouped_mm'
 with torch.device('meta'):model=Qwen3_5MoeForCausalLM(cfg)
 params=dict(model.named_parameters());index=json.loads((path/'model.safetensors.index.json').read_text())['weight_map']
 keys={k.replace('model.language_model.','model.'):v for k,v in index.items()}
 missing=set(params)-set(keys);assert not missing,missing
 return model,params,index,dict(parameter_count=sum(p.numel() for p in params.values()),bf16_GiB=sum(p.numel()*2 for p in params.values())/1024**3,largest_parameter_GiB=max(p.numel()*2 for p in params.values())/1024**3,num_shards=len(set(index.values())),text_parameters=len(params),missing_keys=sorted(missing))

def load(name):
 assert torch.cuda.is_available() and 'GB10' in torch.cuda.get_device_name()
 path=MODELS/name;assert (path/'BASEPOST_READY.json').exists(),'weights not SHA-verified yet'
 drop();model,params,index,plan=parameter_plan(name);assigned=set();dtype_counts={};observed=[]
 print('LOAD_PLAN',name,plan,memory(),flush=True)
 for shard in sorted(set(index.values())):
  avail=int(memory()['MemAvailable'].split()[0])/1024**2
  assert avail>14,('memory guard before shard',shard,avail)
  with safe_open(str(path/shard),framework='pt',device='cuda') as f:
   for key in f.keys():
    target=key.replace('model.language_model.','model.')
    if target not in params:continue
    tensor=f.get_tensor(key);dtype_counts[str(tensor.dtype)]=dtype_counts.get(str(tensor.dtype),0)+tensor.numel()
    assert tuple(tensor.shape)==tuple(params[target].shape),(key,tensor.shape,params[target].shape)
    tensor=tensor.to(torch.bfloat16)
    parent,leaf=target.rsplit('.',1);setattr(model.get_submodule(parent),leaf,torch.nn.Parameter(tensor,requires_grad=False));assigned.add(target)
  drop();observed.append(memory());print('SHARD_LOADED',name,shard,'allocatedGiB',torch.cuda.memory_allocated()/1024**3,memory(),flush=True)
 assert assigned==set(params),set(params)-assigned
 model.model.rotary_emb=Qwen3_5MoeTextRotaryEmbedding(model.config).to('cuda')
 assert not [(n,str(b.shape)) for n,b in list(model.named_parameters())+list(model.named_buffers()) if b.is_meta]
 model.generation_config.eos_token_id=[248044,248046];model.generation_config.pad_token_id=248044;model.eval()
 tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
 plan.update(source_dtype_counts=dtype_counts,memory_samples=observed,final_allocated_GiB=torch.cuda.memory_allocated()/1024**3,shared_eos=[248044,248046],pad=248044,experts_implementation='grouped_mm',all_parameters_BF16=True)
 return tok,model,plan

if __name__=='__main__':
 out={}
 for name in ['Qwen3.5-35B-A3B-Base','Qwen3.5-35B-A3B']:
  m,p,i,plan=parameter_plan(name);print(name,plan);out[name]=plan;del m,p,i
 Path(__file__).with_name('load_plans.json').write_text(json.dumps(out,indent=2))
