"""Copy only closed capture shards; never replace a complete local shard with a partial copy."""
import argparse,pathlib,subprocess,json,re
p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('--complete',action='store_true');a=p.parse_args();root=pathlib.Path(a.output);root.mkdir(exist_ok=True)
remote='amd9070:C:/Strata-Lab/papers-study/'
def fetch(name):
 target=root/name;tmp=root/(name+'.transfer')
 subprocess.run(['scp','-o','ConnectTimeout=8',remote+name,str(tmp)],check=True);tmp.replace(target)
fetch('collection.json');records=json.loads((root/'collection.json').read_text(encoding='utf-8'))
files=sorted(set(r['trace_file'] for r in records))
for name in files if a.complete else files[:-1]:
 assert re.fullmatch(r'routes-\d+\.bin',name)
 if (root/name).exists():continue
 fetch(name);print('copied',name,(root/name).stat().st_size,flush=True)
if not (root/'routers-bf16.npy').exists():fetch('routers-bf16.npy')
print('completed requests',len(records),'closed shards local',len(list(root.glob('routes-*.bin'))),flush=True)
