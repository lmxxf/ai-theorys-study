import hashlib,json,pathlib,subprocess,tarfile,sys
root=pathlib.Path(sys.argv[1]);m=json.loads((root/'manifest.json').read_text());paths=[]
for part in m['parts']:
 p=root/part['path'];h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 assert p.stat().st_size==part['bytes'] and h.hexdigest()==part['sha256'],p
 paths.append(str(p))
cat=subprocess.Popen(['cat',*paths],stdout=subprocess.PIPE)
z=subprocess.Popen(['zstd','-dc'],stdin=cat.stdout,stdout=subprocess.PIPE);cat.stdout.close()
expected={f['path']:f for f in m['files']};seen=set()
with tarfile.open(fileobj=z.stdout,mode='r|') as tf:
 for member in tf:
  if member.isdir():continue
  name=member.name.removeprefix('./');meta=expected[name];h=hashlib.sha256()
  with tf.extractfile(member) as f:
   for b in iter(lambda:f.read(8<<20),b''):h.update(b)
  assert member.size==meta['bytes'] and h.hexdigest()==meta['sha256'],name
  seen.add(name)
# Drain tar padding so zstd can finish without SIGPIPE.
while z.stdout.read(8<<20):pass
assert z.wait()==0 and cat.wait()==0 and seen==set(expected)
print(f'Verified {len(paths)} volumes and all {len(seen)} archived files')
