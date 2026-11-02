import os,hashlib,json
os.environ["HF_HUB_DISABLE_XET"]="1"
from huggingface_hub import HfApi,hf_hub_download
from pathlib import Path
out=[]
for repo in ["Qwen/Qwen3-0.6B","Qwen/Qwen3.8-27B-FP8"]:
 info=HfApi().model_info(repo,files_metadata=True)
 for f in info.siblings:
  if not f.rfilename.endswith(".safetensors"): continue
  p=Path("/home/lmxxf/work/models")/repo.split("/")[1]/f.rfilename
  expected=f.lfs.sha256
  digest=hashlib.file_digest(p.open("rb"),"sha256").hexdigest()
  if digest!=expected:
   print("REPAIR",p,flush=True)
   hf_hub_download(repo,f.rfilename,revision=info.sha,local_dir=p.parent,force_download=True)
   digest=hashlib.file_digest(p.open("rb"),"sha256").hexdigest()
  assert digest==expected,(p,digest,expected)
  print("OK",p.name,flush=True)
  out.append(dict(repo=repo,file=f.rfilename,sha256=digest,revision=info.sha))
Path("/home/lmxxf/work/interp-curvature-exp/weights_verified.json").write_text(json.dumps(out,indent=2))
