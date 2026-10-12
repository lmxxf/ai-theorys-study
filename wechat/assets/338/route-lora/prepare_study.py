import hashlib,json,pathlib,random,sys
import numpy as np
root=pathlib.Path(r'C:\Strata-Lab'); out=root/'route-study';out.mkdir(exist_ok=True)
# Use the exact BF16 router rows loaded by this installation, not independently converted GGUF weights.
pack=root/'Strata-data/packs/q2_0'; rows={}
for line in (pack/'index.txt').read_text().splitlines():
 if line.startswith('#') or not line.strip():continue
 f=line.split()
 if f[0].endswith('.ffn_gate_inp.weight'):rows[int(f[0].split('.')[1])]=f
assert len(rows)==48
routers=np.empty((48,512,2560),dtype=np.uint16)
with (pack/'dense.bin').open('rb') as stream:
 for l,f in rows.items():
  assert int(f[1])==0 and int(f[2])==4 and list(map(int,f[7:9]))==[2560,512]
  off,size=int(f[3]),int(f[4]);assert size==512*2560*2
  stream.seek(off);routers[l]=np.frombuffer(stream.read(size),dtype='<u2').reshape(512,2560)
np.save(out/'routers-bf16.npy',routers)
text=(root/'AGENTS-test.md').read_text(encoding='utf-8')
seen=set();parts=[]
for para in text.split('\n\n'):
 para=para.strip()
 if not para or para in seen:continue
 seen.add(para)
 # Break very long paragraphs only at character boundaries; chunks never overlap.
 parts.extend(para[i:i+2000] for i in range(0,len(para),2000))
chunks=[];pending=''
for para in parts:
 if pending and len(pending)+len(para)+2>2400:chunks.append(pending);pending=''
 pending+= ('\n\n' if pending else '')+para
if pending:chunks.append(pending)
order=list(range(len(chunks)));random.Random(338).shuffle(order)
nval=max(3,round(len(chunks)*.15));ntest=nval
val=set(order[:nval]);test=set(order[nval:nval+ntest])
records=[]
for i,content in enumerate(chunks):
 records.append({'id':i,'split':'val' if i in val else 'test' if i in test else 'train',
  'sha256':hashlib.sha256(content.encode()).hexdigest(),'text':content,
  'prompt':'阅读以下笔记片段，阅读完只回答“收到”。\n\n'+content})
manifest={'source_sha256':hashlib.sha256(text.encode()).hexdigest(),'seed':338,'deduplicate':'exact paragraphs','chunk_max_chars':2400,'chunks':records}
(out/'prompts.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print('Chunks:',len(records),'splits:',{s:sum(r['split']==s for r in records) for s in ['train','val','test']})
print('Router BF16 shape',routers.shape,'bytes',routers.nbytes)
