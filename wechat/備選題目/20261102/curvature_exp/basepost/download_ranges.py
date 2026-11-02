import os,time,json,hashlib,threading,requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
WORK=Path("/home/lmxxf/work/interp-curvature-exp/basepost");ROOT=Path("/home/lmxxf/work/models")
INV=json.loads((WORK/"pair_inventory.json").read_text());CHUNK=16*1024**2;WORKERS=8
local=threading.local()
def session():
 if not hasattr(local,"s"):
  local.s=requests.Session()
  if os.environ.get("BASEPOST_PROXY"):
   local.s.trust_env=False;local.s.proxies.update({"http":os.environ["BASEPOST_PROXY"],"https":os.environ["BASEPOST_PROXY"]})
  local.s.headers.update({"User-Agent":"huggingface_hub/0.36.0"})
 return local.s

def download_model(name):
 spec=INV[name];dest=ROOT/name;verified=[]
 for f in spec["shards"]:
  path=dest/f["file"];size=f["size"];sha=f["sha256"]
  if path.exists():
   with path.open("rb") as st:actual=hashlib.file_digest(st,"sha256").hexdigest()
   if actual==sha:verified.append(f);print("ALREADY_VERIFIED",name,path.name,flush=True);continue
  part=path.with_suffix(path.suffix+".range-part");state=path.with_suffix(path.suffix+".ranges.json")
  done=set()
  if state.exists() and part.exists():
   previous=json.loads(state.read_text());old_chunk=previous['chunk'];old_done=set(previous['done'])
   if old_chunk>=CHUNK:
    assert old_chunk%CHUNK==0
    for i in old_done:done.update(range(i*old_chunk//CHUNK,min(size-1,(i+1)*old_chunk-1)//CHUNK+1))
   else:
    assert CHUNK%old_chunk==0
    for i in range((size+CHUNK-1)//CHUNK):
     required=range(i*CHUNK//old_chunk,min(size-1,(i+1)*CHUNK-1)//old_chunk+1)
     if all(j in old_done for j in required):done.add(i)
  fd=os.open(part,os.O_CREAT|os.O_RDWR,0o644);os.ftruncate(fd,size)
  repo=spec["repo"];revision=spec["revision"];filename=f["file"]
  resolve=f"https://modelscope.cn/models/{repo}/resolve/master/{filename}"
  response=session().get(resolve,headers={"Range":"bytes=0-0"},stream=True,timeout=(15,30));response.raise_for_status();url=response.url;response.close()
  count=(size+CHUNK-1)//CHUNK;starttime=time.monotonic();initial=len(done);lock=threading.Lock()
  def one(i):
   start=i*CHUNK;end=min(size,start+CHUNK)-1
   for attempt in range(40):
    try:
     with session().get(url,headers={"Range":f"bytes={start}-{end}"},stream=True,timeout=(15,40)) as r:
      assert r.status_code==206,(r.status_code,r.text[:200])
      assert r.headers.get("Content-Range","").startswith(f"bytes {start}-{end}/"),r.headers
      off=start
      for block in r.iter_content(1024**2):
       if block:os.pwrite(fd,block,off);off+=len(block)
      assert off==end+1,(off,end)
     return i
    except Exception as e:
     print("RANGE_RETRY",name,path.name,i,attempt,repr(e),flush=True);time.sleep(min(10,attempt+1))
   raise RuntimeError((name,path.name,i))
  print("FILE_START",name,path.name,size,"done",len(done),"chunks",count,flush=True)
  try:
   with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    futures=[pool.submit(one,i) for i in range(count) if i not in done]
    for future in as_completed(futures):
     done.add(future.result())
     tmp=state.with_suffix(".tmp");tmp.write_text(json.dumps(dict(size=size,sha256=sha,chunk=CHUNK,done=sorted(done))));tmp.replace(state)
     if (len(done)-initial)%16==0 or len(done)==count:
      rate=(len(done)-initial)*CHUNK/(time.monotonic()-starttime)/1e6
      print(time.strftime("%F %T"),"PROGRESS",name,path.name,len(done),"/",count,"MB/s",round(rate,2),flush=True)
   os.fsync(fd)
  finally:os.close(fd)
  with part.open("rb") as st:actual=hashlib.file_digest(st,"sha256").hexdigest()
  assert actual==sha,(name,path.name,"SHA_MISMATCH",actual,sha)
  part.replace(path);state.unlink();verified.append(f)
  print("FILE_VERIFIED",name,path.name,sha,flush=True)
 (dest/"BASEPOST_READY.json").write_text(json.dumps(dict(repo=spec["repo"],revision=spec["revision"],files=verified,download_source="ModelScope official mirror; SHA-256 identical to pinned Hugging Face revision"),indent=2))
 print("MODEL_READY",name,flush=True)
for name in ["Qwen3.5-35B-A3B-Base","Qwen3.5-35B-A3B"]:download_model(name)
print("ALL_COMPLETE",flush=True)
