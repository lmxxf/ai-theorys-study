import os,json,hashlib,time
os.environ["HF_HUB_DISABLE_XET"]="1"
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import HfApi,hf_hub_download
root=Path("/home/lmxxf/work/models");work=Path("/home/lmxxf/work/interp-curvature-exp/basepost")
inv=json.loads((work/"pair_inventory.json").read_text());jobs=[]
for name in ["Qwen3.5-35B-A3B-Base","Qwen3.5-35B-A3B"]:
 spec=inv[name];dest=root/name;dest.mkdir(exist_ok=True)
 for f in HfApi().model_info(spec["repo"],revision=spec["revision"],files_metadata=True).siblings:
  if not f.rfilename.endswith((".json",".jinja",".txt",".md",".safetensors")) or "/" in f.rfilename:continue
  jobs.append((spec["repo"],spec["revision"],f,dest))
# Metadata before large shards, with at most four HTTP transfers total.
jobs.sort(key=lambda x:x[2].rfilename.endswith(".safetensors"))
def download(job):
 repo,rev,f,dest=job
 for attempt in range(50):
  try:
   p=Path(hf_hub_download(repo,f.rfilename,revision=rev,local_dir=dest,force_download=False))
   if f.lfs:
    with p.open("rb") as stream:digest=hashlib.file_digest(stream,"sha256").hexdigest()
    
    if digest!=f.lfs.sha256:
     hf_hub_download(repo,f.rfilename,revision=rev,local_dir=dest,force_download=True)
     raise RuntimeError("hash mismatch; refreshed file for recheck")
   else:digest=None
   print(time.strftime("%F %T"),"VERIFIED",repo,f.rfilename,p.stat().st_size,flush=True)
   return dict(repo=repo,revision=rev,file=f.rfilename,size=p.stat().st_size,sha256=digest)
  except Exception as e:
   print("RETRY",repo,f.rfilename,attempt,repr(e),flush=True);time.sleep(3*(attempt+1))
 raise RuntimeError(job)
with ThreadPoolExecutor(max_workers=4) as pool:verified=list(pool.map(download,jobs))
(work/"weights_verified.json").write_text(json.dumps(verified,indent=2))
print("ALL_COMPLETE",len(verified),flush=True)
