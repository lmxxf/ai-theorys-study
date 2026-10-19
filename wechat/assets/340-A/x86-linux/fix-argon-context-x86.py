#!/usr/bin/env python3
"""Apply the exact-build Linux x86_64 1.4.2 patch; reject unknown builds."""
from pathlib import Path
import argparse,hashlib,json,os,tempfile
p=argparse.ArgumentParser();p.add_argument('--target',type=Path,default=Path.home()/'.local/bin/agy');a=p.parse_args()
r=Path(__file__).resolve().parent;m=json.loads((r/'agy-x86-patches.json').read_text());t=a.target
sha=lambda b:hashlib.sha256(b).hexdigest()
b=t.read_bytes();h=sha(b)
if h==m['patched_sha256']:print('Already patched:',h);raise SystemExit
if h!=m['original_sha256']:raise SystemExit('Unknown binary: refusing to patch '+h)
assert os.uname().machine=='x86_64'
out=bytearray(b)
for c in m['changes']:
 o=int(c['offset'],16);old=bytes.fromhex(c['before']);new=bytes.fromhex(c['after'])
 assert len(old)==len(new) and out[o:o+len(old)]==old,c['purpose']
 out[o:o+len(new)]=new
assert len(out)==m['size'] and sha(out)==m['patched_sha256']
backup=Path.home()/'.local/share/agy-patches'/('agy-1.4.2-original-'+h+'.bak');backup.parent.mkdir(parents=True,exist_ok=True)
if backup.exists():assert sha(backup.read_bytes())==h
else:
 with backup.open('xb') as f:f.write(b)
 backup.chmod(t.stat().st_mode & 0o777)
fd,tmp=tempfile.mkstemp(prefix='agy-patch-',dir=t.parent)
try:
 with os.fdopen(fd,'wb') as f:f.write(out);f.flush();os.fsync(f.fileno())
 os.chmod(tmp,t.stat().st_mode & 0o777);os.replace(tmp,t)
finally:
 if os.path.exists(tmp):os.unlink(tmp)
print('Original backup:',backup);print('Patched:',m['patched_sha256'])
