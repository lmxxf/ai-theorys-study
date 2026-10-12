import json,pathlib
p=pathlib.Path(r'C:\Strata-Lab\Strata\strata-q2_0.json');c=json.loads(p.read_text());a=c['args']
for flag,val in [('--max-context','65536'),('--prefill','128'),('--short-read','65536'),('--kv','int8'),('--vram-reserve-mib','4096')]:
 if flag in a:a[a.index(flag)+1]=val
 else:a.extend([flag,val])
p.write_text(json.dumps(c,indent=2),encoding='utf-8')
