#!/usr/bin/env python3
"""Restore the archived patched agy."""
from pathlib import Path
import gzip,hashlib,os,shutil,tempfile
root=Path(__file__).resolve().parent
name='agy.gz'
expected='2a58516fdd1275208425a78bbfb4b2b43ec0e71d7d4f2664d64c09382b0f5b7c'
b=gzip.decompress((root/name).read_bytes())
assert hashlib.sha256(b).hexdigest()==expected,'Archive checksum mismatch'
assert os.uname().machine=='x86_64','This archive is for Linux x86_64 only'
target=Path.home()/'.local/bin/agy';target.parent.mkdir(parents=True,exist_ok=True)
if target.exists():
 old=hashlib.sha256(target.read_bytes()).hexdigest()
 if old==expected:print('Already installed:',expected);raise SystemExit
 backup=Path.home()/'.local/share/agy-patches'/('agy-before-restore-'+old+'.bak')
 backup.parent.mkdir(parents=True,exist_ok=True)
 if not backup.exists():shutil.copy2(target,backup)
 else:assert hashlib.sha256(backup.read_bytes()).hexdigest()==old
 print('Previous binary saved:',backup)
fd,tmp=tempfile.mkstemp(prefix='agy-restore-',dir=target.parent)
try:
 with os.fdopen(fd,'wb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 os.chmod(tmp,0o755);os.replace(tmp,target)
finally:
 if os.path.exists(tmp):os.unlink(tmp)
print('Restored:',target,expected)
print('Restart agy. Use a new session to load complete rules.')
