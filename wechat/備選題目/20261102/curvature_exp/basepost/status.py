from pathlib import Path
import json,time
ROOT=Path('/home/lmxxf/work/interp-curvature-exp/basepost');inv=json.loads((ROOT/'pair_inventory.json').read_text());out={}
for name in ['Qwen3.5-35B-A3B-Base','Qwen3.5-35B-A3B']:
 d=Path('/home/lmxxf/work/models')/name;verified=0;received=0
 for f in inv[name]['shards']:
  path=d/f['file'];state=path.with_suffix(path.suffix+'.ranges.json')
  if path.exists() and path.stat().st_size==f['size']:verified+=1;received+=f['size']
  elif state.exists():
   try:
    s=json.loads(state.read_text());received+=sum(min(s['chunk'],max(0,f['size']-i*s['chunk'])) for i in s['done'])
   except (OSError,ValueError):pass
 cases=len(list((ROOT/'raw'/name).glob('bp_*.json')))
 out[name]=dict(shards=verified,total_shards=14,received_GB=round(received/1e9,3),total_GB=round(inv[name]['total_bytes']/1e9,3),weights_ready=(d/'BASEPOST_READY.json').exists(),cases=cases,total_cases=147)
print(json.dumps(dict(time=time.strftime('%F %T'),models=out,complete=(ROOT/'COMPLETE.json').exists()),ensure_ascii=False))
