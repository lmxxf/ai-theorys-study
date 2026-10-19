from pathlib import Path
import sqlite3,json,hashlib

def var(b,p):
 v=s=0
 while True:
  x=b[p];p+=1;v|=(x&127)<<s;s+=7
  if x<128:return v,p
  if s>70:raise ValueError()
def fields(b):
 p=0
 while p<len(b):
  t,p=var(b,p);w=t&7
  if w==0:v,p=var(b,p)
  elif w==2:n,p=var(b,p);v=b[p:p+n];p+=n
  elif w in (1,5):n=8 if w==1 else 4;v=b[p:p+n];p+=n
  else:raise ValueError()
  yield t>>3,w,v
def get(b,k):return next(v for f,w,v in fields(b) if f==k)
import argparse
a=argparse.ArgumentParser()
a.add_argument('conversation_id')
a.add_argument('--agents', type=Path, required=True)
a.add_argument('--binary', type=Path, default=Path.home()/'.local/bin/agy')
a.add_argument('--expected-limit', type=int, default=983040)
a.add_argument('--output', type=Path)
args=a.parse_args()
cid=args.conversation_id
c=sqlite3.connect(f"file:{Path.home()}/.gemini/antigravity-cli/conversations/{cid}.db?mode=ro",uri=True)
g=c.execute('select data from gen_metadata where idx=0').fetchone()[0];chat=get(g,1)
window=get(get(chat,9),10)
limit=get(window,4)
e=c.execute('select data from executor_metadata where idx=0').fetchone()[0]
executor_limit=get(get(get(e,10),2),5)
assert executor_limit==args.expected_limit
original=args.agents.read_bytes()
sections=[]
for f,w,v in fields(chat):
 if f==16:
  title=get(v,1).decode();content=get(v,2)
  if b'AGENTS.md' in content:sections.append(dict(title=title,bytes=len(content),contains_full_agents=original in content,excluded=b'exceeded the rules token budget' in content))
assert limit==args.expected_limit
assert any(v['contains_full_agents'] and not v['excluded'] for v in sections)
r=dict(version='1.4.2',conversation_id=cid,actual_request_limit=limit,agents_bytes=len(original),sections=sections,executor_limit=executor_limit,long_context_over_256k_tested=False,sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest())
if args.output:args.output.write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2))
