import os,json
os.environ["HF_HUB_DISABLE_XET"]="1"
from huggingface_hub import HfApi,hf_hub_download
from pathlib import Path
api=HfApi();out={}
for name in ["Qwen3.5-35B-A3B-Base","Qwen3.5-35B-A3B","Qwen3.5-35B-A3B-FP8","Qwen3.5-9B-Base","Qwen3.5-9B"]:
 repo="Qwen/"+name;m=api.model_info(repo,files_metadata=True)
 p=hf_hub_download(repo,"config.json",revision=m.sha)
 cfg=json.loads(Path(p).read_text());shards=[dict(file=f.rfilename,size=f.size,sha256=f.lfs.sha256 if f.lfs else None) for f in m.siblings if f.rfilename.endswith(".safetensors")]
 out[name]=dict(repo=repo,revision=m.sha,config=cfg,shards=shards,total_bytes=sum(s["size"] for s in shards),max_shard_bytes=max(s["size"] for s in shards))
 t=cfg.get("text_config",cfg);print(name,"bytes",out[name]["total_bytes"],"max_shard",out[name]["max_shard_bytes"],"arch",cfg["architectures"],"text",{k:t.get(k) for k in ["model_type","hidden_size","num_hidden_layers","num_experts","num_experts_per_tok","moe_intermediate_size","intermediate_size","layer_types"]},flush=True)
Path("/home/lmxxf/work/interp-curvature-exp/basepost/pair_inventory.json").write_text(json.dumps(out,indent=2))
