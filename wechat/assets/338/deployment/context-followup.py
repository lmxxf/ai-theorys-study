import json,time,urllib.request,pathlib,sys
sys.stdout.reconfigure(encoding='utf-8')
root=pathlib.Path(r'C:\Strata-Lab')
base='http://127.0.0.1:8080'
start=time.monotonic()
while True:
 try:
  with urllib.request.urlopen(base+'/health',timeout=5) as r: health=json.load(r)
  if health.get('loaded'): break
 except Exception: pass
 if time.monotonic()-start>900: raise RuntimeError('Server not ready after 15 minutes')
 time.sleep(5)
print('HEALTH',json.dumps(health,ensure_ascii=False),flush=True)
results=[]
prior=json.loads((root/'context-64k-results.json').read_text(encoding='utf-8'))['requests'][0]
for prompt in ['上文里朱雀的图腾是什么？只回答名字。']:
 body={'model':'strata','messages':[{'role':'user','content':prompt}],'max_tokens':128,'temperature':0,'reasoning_effort':'none','stream':True,'stream_options':{'include_usage':True}}
 body['messages']=[{'role':'user','content':prior['prompt']},{'role':'assistant','content':prior['answer']},{'role':'user','content':prompt}]
 req=urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 t0=time.monotonic(); first=None; pieces=[]; events=[]
 with urllib.request.urlopen(req,timeout=600) as r:
  for line in r:
   if not line.startswith(b'data: '):continue
   raw=line[6:].strip()
   if raw==b'[DONE]':break
   event=json.loads(raw); events.append(event)
   for choice in event.get('choices',[]):
    text=choice.get('delta',{}).get('content','') or ''
    if text:
     if first is None:first=time.monotonic()-t0
     pieces.append(text)
 item={'prompt':prompt,'answer':''.join(pieces),'ttft_seconds':first,'total_seconds':time.monotonic()-t0,'events':events}
 results.append(item)
 print(json.dumps({k:v for k,v in item.items() if k not in ('events','prompt')},ensure_ascii=False),flush=True)
 (root/'context-followup-results.json').write_text(json.dumps({'health':health,'requests':results},ensure_ascii=False,indent=2),encoding='utf-8')
