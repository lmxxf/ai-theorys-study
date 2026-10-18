#!/usr/bin/env python3
"""Verified ARM64 agy 1.4.2 patch. Reject all unknown builds; atomic replacement."""
from pathlib import Path
import hashlib,json,os,shutil,tempfile
import argparse
a=argparse.ArgumentParser()
a.add_argument('--target', type=Path, default=Path.home()/'.local/bin/agy')
p=a.parse_args().target.expanduser()
m=json.loads((Path(__file__).parent/'agy-patches.json').read_text())
d=bytearray(p.read_bytes());h=hashlib.sha256(d).hexdigest()
if h==m['patched_sha256']:
 print('Already patched: agy 1.4.2');raise SystemExit(0)
input_hash=h
if h==m.get('upgrade_from',{}).get('sha256'):
 for v in m['upgrade_from']['patches']:
  o=v['offset'];a=bytes.fromhex(v['after']);b=bytes.fromhex(v['before'])
  assert d[o:o+len(a)]==a
  d[o:o+len(b)]=b
 h=hashlib.sha256(d).hexdigest()
assert h==m['original_sha256'], 'Unknown binary: relocate and review before patching'
original=bytes(d)
assert len(d)==m['size']
for v in m['patches']:
 o=v['offset'];a=bytes.fromhex(v['before']);b=bytes.fromhex(v['after'])
 assert len(a)==len(b) and d[o:o+len(a)]==a
 d[o:o+len(b)]=b
assert hashlib.sha256(d).hexdigest()==m['patched_sha256']
backup=p.with_name('agy-1.4.2-original.bak')
if backup.exists():assert hashlib.sha256(backup.read_bytes()).hexdigest()==h
else:backup.write_bytes(original);os.chmod(backup,p.stat().st_mode & 0o777)
fd,tmp=tempfile.mkstemp(prefix='agy-patch-',dir=p.parent)
try:
 with os.fdopen(fd,'wb') as f:f.write(d);f.flush();os.fsync(f.fileno())
 os.chmod(tmp,p.stat().st_mode & 0o777);os.replace(tmp,p)
finally:
 if os.path.exists(tmp):os.unlink(tmp)
print('Patched agy 1.4.2:',m['patched_sha256'])
print('Backup:',backup)
