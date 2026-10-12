import pathlib,json,time,urllib.request,argparse,sys
sys.stdout.reconfigure(encoding='utf-8')
p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');a=p.parse_args()
root=pathlib.Path(r'C:\Strata-Lab\route-study');base='http://127.0.0.1:18081';start=time.monotonic()
while True:
 try:
  with urllib.request.urlopen(base+'/health',timeout=3) as r:h=json.load(r)
  if h.get('loaded'):break
 except Exception:pass
 if time.monotonic()-start>180:raise RuntimeError('capture server not ready')
 time.sleep(2)
prompts=json.loads((root/'prompts.json').read_text(encoding='utf-8'))['chunks']
if a.pilot:prompts=[{'id':-1,'split':'pilot','prompt':'请阅读这句话，只回答收到。我们测试的是下一层专家预测，记录真实输入和实际专家编号，保持模型权重不变。'*4}]
results=[];trace=root/'routes.bin'
for item in prompts:
 before=trace.stat().st_size if trace.exists() else 0
 body={'model':'strata','messages':[{'role':'user','content':item['prompt']}],'max_tokens':1,'temperature':0,'reasoning_effort':'none','stream':True,'stream_options':{'include_usage':True}}
 req=urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 t=time.monotonic();events=[];text=''
 with urllib.request.urlopen(req,timeout=300) as r:
  for line in r:
   if not line.startswith(b'data: '):continue
   raw=line[6:].strip()
   if raw==b'[DONE]':break
   event=json.loads(raw);events.append(event)
   if event.get('error'):raise RuntimeError(event['error'])
   for choice in event.get('choices',[]):text+=choice.get('delta',{}).get('content','') or ''
 timing=next((e for e in events if e.get('usage')),None)
 if not timing:raise RuntimeError('missing completion usage')
 after=trace.stat().st_size
 if after<=before:raise RuntimeError('no route records written')
 record={'id':item['id'],'split':item['split'],'start':max(32,before),'end':after,'seconds':time.monotonic()-t,'answer':text,'usage':timing['usage'],'timings':timing.get('timings')}
 results.append(record)
 (root/('pilot.json' if a.pilot else 'collection.json')).write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(record,ensure_ascii=False),flush=True)
