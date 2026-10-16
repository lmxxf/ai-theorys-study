#!/usr/bin/env python3
"""Exact-build x86_64 patch; atomic installation, immutable backup, idempotent.
Argon Low 1419: final max_token_limit=1000000; automatic checkpoint=false,nil.
Rule/skill file read cap=1MiB; both final token budgets=131072.
"""
from pathlib import Path
import hashlib, json, os, struct
root=Path(__file__).resolve().parent
target=Path.home()/'.local/bin/agy'
original_hash='e14253aa79d4e910cbf8bcb40b68e80f9e85476725e130ed085122c98face92b'
backup=root/'agy-x86-original-e14253aa.bak'
manifest=root/'agy-x86-patches.json'
sha=lambda b:hashlib.sha256(b).hexdigest()
b=target.read_bytes()
if manifest.exists() and sha(b)==json.loads(manifest.read_text())['patched_sha256']:
 print('Already patched:',sha(b));raise SystemExit
if sha(b)!=original_hash:raise SystemExit('Unknown binary: refusing to patch '+sha(b))
out=bytearray(b);changes=[]
def patch(off,before,after,label):
 if isinstance(before,str):before=bytes.fromhex(before)
 if isinstance(after,str):after=bytes.fromhex(after)
 assert len(before)==len(after) and b[off:off+len(before)]==before,label
 out[off:off+len(after)]=after
 changes.append(dict(offset=hex(off),before=before.hex(),after=after.hex(),purpose=label))
for off in [0x7ce7f09,0x7ce8e85,0x7ced60e]:
 patch(off,'b9c05d0000','b900001000','file read cap 1MiB')
# Final argument loads bypass both configured and default budget paths.
patch(0x7ceef94,'8b9c24e0000000','bb000002009090','rules budget 128Ki tokens')
patch(0x7ceef9e,'8b8424e4000000','b8000002009090','customizations budget 128Ki tokens')
# Go amd64 ABI: bool AX, error interface BX/CX; clear all three before RET.
patch(0x7c90a80,b[0x7c90a80:0x7c90a87],'31c031db31c9c3','disable automatic checkpoint')
cave=0x7c90aa0
code=bytes.fromhex('488b50084885d27415817a408b050000750c4885c97407c7411040420f004889481031dbe99cae9cff')
assert code==(root/'trampoline.bin').read_bytes()
patch(cave,b[cave:cave+len(code)],code,'Argon Low final configuration override')
site=0x765b95f
patch(site,'4889481031db',b'\xe9'+struct.pack('<i',cave-site-5)+b'\x90','jump to override; replay store and xor')
if backup.exists():assert sha(backup.read_bytes())==original_hash
else:
 with backup.open('xb') as f:f.write(b)
 backup.chmod(target.stat().st_mode & 0o777)
meta=dict(original_sha256=original_hash,patched_sha256=sha(out),size=len(out),backup=str(backup),changes=changes)
manifest.write_text(json.dumps(meta,indent=2)+'\n')
tmp=target.with_name('agy.patched.tmp')
with tmp.open('xb') as f:f.write(out);f.flush();os.fsync(f.fileno())
tmp.chmod(target.stat().st_mode & 0o777)
os.replace(tmp,target)
print(json.dumps({k:v for k,v in meta.items() if k!='changes'},indent=2))
