import pathlib,re,json,hashlib,shutil,argparse
p=argparse.ArgumentParser();p.add_argument('repository');p.add_argument('output');a=p.parse_args();repo=pathlib.Path(a.repository);out=pathlib.Path(a.output);(out/'papers').mkdir(parents=True,exist_ok=True)
indexed={}
for f in repo.glob('*.md'):
 m=re.match(r'^(\d+)\.',f.name)
 if m:
  n=int(m[1]);assert n not in indexed,n;indexed[n]=f
assert max(indexed)==96
manifest={'train_papers':list(range(67,97)),'val_papers':[61,62,63],'test_papers':[64,65,66],'chunk_max_chars':2400,'capture_stride':16,'papers':[],'chunks':[]}
seen=set()
for split,nums in [('train',range(67,97)),('val',range(61,64)),('test',range(64,67))]:
 for number in nums:
  f=indexed[number];raw=f.read_text(encoding='utf-8');shutil.copyfile(f,out/'papers'/f.name)
  body=re.sub(r'\A---\s*\n.*?\n---\s*\n','',raw,count=1,flags=re.S)
  parts=[];removed=0
  for para in body.split('\n\n'):
   para=para.strip()
   if not para:continue
   h=hashlib.sha256(para.encode()).hexdigest()
   if len(para)>=80 and h in seen:removed+=1;continue
   if len(para)>=80:seen.add(h)
   parts.extend(para[i:i+2000] for i in range(0,len(para),2000))
  chunks=[];pending=''
  for para in parts:
   if pending and len(pending)+len(para)+2>2400:chunks.append(pending);pending=''
   pending+=('\n\n' if pending else '')+para
  if pending:chunks.append(pending)
  for j,text in enumerate(chunks):
   manifest['chunks'].append({'id':len(manifest['chunks']),'paper_id':number,'paper_chunk':j,'split':split,'sha256':hashlib.sha256(text.encode()).hexdigest(),'text':text,'prompt':'阅读以下论文片段，阅读完只回答“收到”。\n\n'+text})
  manifest['papers'].append({'number':number,'filename':f.name,'split':split,'sha256':hashlib.sha256(raw.encode()).hexdigest(),'source_bytes':len(raw.encode()),'body_chars':len(body),'duplicate_paragraphs_removed':removed,'chunks':len(chunks)})
assert len(manifest['train_papers'])==30
(out/'prompts.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print('chunks',len(manifest['chunks']),{s:sum(r['split']==s for r in manifest['chunks']) for s in ['train','val','test']},flush=True)
