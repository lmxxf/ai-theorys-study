import pathlib,hashlib,json,sys,time
root=pathlib.Path(r'C:\Strata-Lab'); files=json.loads((root/'model-files.json').read_text())
results=[]
for meta in files:
 p=root/'Strata-data'/'models'/meta['path']
 if len(sys.argv)>1 and ('-'+sys.argv[1]+'-of-') not in p.name:continue
 if not p.exists() and (root/'prefetch'/p.name).exists():p=root/'prefetch'/p.name
 t=time.monotonic(); h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 actual=h.hexdigest(); expected=meta['lfs']['oid']; size=p.stat().st_size
 result={'file':str(p),'bytes':size,'sha256':actual,'expected_sha256':expected,'ok':actual==expected and size==meta['size'],'seconds':time.monotonic()-t}
 results.append(result);print(json.dumps(result),flush=True)
 if not result['ok']:sys.exit(1)
(root/('model-verify-'+(sys.argv[1] if len(sys.argv)>1 else 'all')+'.json')).write_text(json.dumps(results,indent=2))
