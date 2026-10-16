from pathlib import Path
import sqlite3,json
root=Path(__file__).resolve().parent
p=Path.home()/'.gemini/antigravity-cli/conversations/b5b575f7-bae4-4d03-97a4-3541d97ded7d.db'
def vi(b,i):
 v=0;s=0
 while i<len(b) and s<70:
  c=b[i];i+=1;v|=(c&127)<<s
  if c<128:return v,i
  s+=7
 raise ValueError()
def fields(b):
 out=[];i=0
 while i<len(b):
  tag,i=vi(b,i);n,w=tag>>3,tag&7
  if not n:raise ValueError()
  if w==0:v,i=vi(b,i)
  elif w==2:
   l,i=vi(b,i);v=b[i:i+l];i+=l
   if i>len(b):raise ValueError()
  elif w in (1,5):l=8 if w==1 else 4;v=b[i:i+l];i+=l
  else:raise ValueError()
  out.append((n,w,v))
 return out
agent=Path('/home/lmxxf/work/ai-theorys-study/AGENTS.md').read_bytes()
def walk(b,path='',depth=0):
 if depth>12:return
 try:fs=fields(b)
 except (ValueError,IndexError):return
 for n,w,v in fs:
  key=path+'.'+str(n)
  if w==0 and v in [256000,983040,1000000,131072,1419]:print(key,v)
  if w==2:
   if key=='gen_metadata.1.9.10':
    print('ContextWindowMetadata scalar fields',[(a,c) for a,t,c in fields(v) if t==0])
   if agent in v and len(v)<len(agent)+2000:
    print('Full rules payload',key,'bytes',len(v))
   walk(v,key,depth+1)
c=sqlite3.connect(f'file:{p}?mode=ro',uri=True)
print('database',p,'AGENTS bytes',len(agent))
for table in ['executor_metadata','gen_metadata']:
 for idx,b in c.execute(f'select idx,data from {table}'):
  print(table,idx,'contains full AGENTS',agent in b,'exclusion warning',b'exceeded the rules token budget' in b)
  walk(b,table)
  if table=='gen_metadata':
   for n,w,v in fields(b):
    if n==10:print('ContextWindowMetadata',[(a,c) for a,t,c in fields(v) if t==0])
