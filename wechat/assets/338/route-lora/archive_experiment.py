"""Archive all local experimental data, including derived tensors and checkpoints."""
import hashlib,json,pathlib,subprocess,sys
src=pathlib.Path(sys.argv[1]).resolve();out=pathlib.Path(sys.argv[2]).resolve();out.mkdir(parents=True,exist_ok=True)
files=[]
for p in sorted(src.rglob('*')):
 if not p.is_file():continue
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 files.append({'path':str(p.relative_to(src)),'bytes':p.stat().st_size,'sha256':h.hexdigest()})
print('Manifest ready:',len(files),'files',sum(f['bytes'] for f in files),'bytes',flush=True)
tar=subprocess.Popen(['tar','-C',str(src),'-cf','-','.'],stdout=subprocess.PIPE)
zstd=subprocess.Popen(['zstd','-T4','-3','-c'],stdin=tar.stdout,stdout=subprocess.PIPE);tar.stdout.close()
parts=[];index=0;total=hashlib.sha256()
while True:
 first=zstd.stdout.read(8<<20)
 if not first:break
 p=out/f'strata-route-study.tar.zst.part-{index:03d}';size=0;h=hashlib.sha256()
 with p.open('wb') as f:
  b=first
  while b:
   f.write(b);h.update(b);total.update(b);size+=len(b)
   if size>=1<<30:break
   b=zstd.stdout.read(min(8<<20,(1<<30)-size))
 parts.append({'path':p.name,'bytes':size,'sha256':h.hexdigest()});index+=1
 print('Part ready:',p.name,size,flush=True)
assert zstd.wait()==0 and tar.wait()==0
(out/'manifest.json').write_text(json.dumps({'format':'tar+zstd split into <=1 GiB volumes','files':files,'parts':parts,'compressed_sha256':total.hexdigest()},indent=2))
