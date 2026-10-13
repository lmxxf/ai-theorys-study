import pathlib,json,time,subprocess,urllib.request,socket,os,sys,argparse,hashlib
import psutil
sys.stdout.reconfigure(encoding='utf-8')
p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');a=p.parse_args()
root=pathlib.Path(r'C:\Strata-Lab');out=root/'speed-study';base='http://127.0.0.1:18082'
lock=pathlib.Path(r'D:\DLSSNR-Lab\gpu.lock')
with lock.open('x') as f:f.write(f'Strata speed benchmark PID={os.getpid()}\n')
proc=None;log=None;results=[]
def no_game():
 for p in psutil.process_iter(['name']):
  name=(p.info['name'] or '').lower()
  if any(s in name for s in ['stellar','sb-win','onimusha','cyberpunk','ff7rebirth']):raise RuntimeError('Game running: '+name)
def request(prompt,max_tokens):
 body={'model':'strata','messages':[{'role':'user','content':prompt}],'temperature':0,'reasoning_effort':'none','max_tokens':max_tokens,'stream':True,'stream_options':{'include_usage':True}}
 req=urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 t=time.monotonic();first=None;text='';last=None;events=[]
 with urllib.request.urlopen(req,timeout=300) as r:
  for line in r:
   if not line.startswith(b'data: '):continue
   raw=line[6:].strip()
   if raw==b'[DONE]':break
   ev=json.loads(raw)
   if ev.get('error'):raise RuntimeError(ev['error'])
   if ev.get('usage'):last=ev
   for c in ev.get('choices',[]):
    chunk=c.get('delta',{}).get('content','') or ''
    if chunk:
     if first is None:first=time.monotonic()-t
     text+=chunk
 if not last or not text:raise RuntimeError('Missing answer or timings')
 if last['timings'].get('draft_n',0)!=0:raise RuntimeError('Not pure autoregressive decoding')
 if last['timings'].get('cache_n',0)!=0:raise RuntimeError('Prompt reused a prefix')
 return {'ttft':first,'wall_seconds':time.monotonic()-t,'answer':text,'answer_sha256':hashlib.sha256(text.encode()).hexdigest(),'usage':last['usage'],'timings':last['timings'],'finish_reason':last['choices'][0]['finish_reason']}
def stop():
 global proc,log
 if proc:
  children=[]
  try:children=psutil.Process(proc.pid).children(recursive=True)
  except psutil.NoSuchProcess:pass
  if proc.poll() is None:subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
  proc.wait(timeout=30)
  _,alive=psutil.wait_procs(children,timeout=30)
  if alive:raise RuntimeError('Owned child failed to stop')
  proc=None
 if log:log.close();log=None
try:
 no_game()
 with socket.socket() as sock:
  if sock.connect_ex(('127.0.0.1',18082))==0:raise RuntimeError('Benchmark port occupied')
 cases=json.loads((out/'cases.json').read_text(encoding='utf-8'))
 sequence=['C'] if a.pilot else ['A','B','C','B','C','A','C','A','B']
 for position,arm in enumerate(sequence):
  no_game();tag=('pilot' if a.pilot else f'run-{position+1:02d}')+'-'+arm
  c=json.loads((out/'base-config.json').read_text());c['log']=str(out/(tag+'-engine.log'))
  c['env'].update(STRATA_FS_SLOTS='0' if arm=='A' else '16',STRATA_FS_AHEAD='0' if arm=='A' else '1',STRATA_LOOKAHEAD='0' if arm=='A' else '1',STRATA_FS_VERIFY='1' if a.pilot else '0')
  if arm=='C':c['env']['STRATA_LOOKAHEAD_LORA']=str(out/'lora-runtime.bin')
  if a.pilot:c['env']['STRATA_LOOKAHEAD_LORA_FIXTURE']=str(out/'lora-fixture.bin')
  cfg=out/(tag+'.json');cfg.write_text(json.dumps(c,indent=2));log=(out/(tag+'-server.log')).open('wb')
  env=dict(os.environ,PYTHONIOENCODING='utf-8',PYTHONUNBUFFERED='1')
  proc=subprocess.Popen([str(root/'Strata/.venv/Scripts/python.exe'),str(root/'Strata/serve/server.py'),'--engine','strata','--config',str(cfg),'--port','18082'],cwd=root/'Strata',env=env,stdout=log,stderr=subprocess.STDOUT)
  started=time.monotonic()
  while True:
   if proc.poll() is not None:raise RuntimeError('Server exited: '+tag)
   no_game()
   try:
    with urllib.request.urlopen(base+'/health',timeout=3) as r:h=json.load(r)
    if h.get('loaded'):break
   except Exception:pass
   if time.monotonic()-started>240:raise RuntimeError('Startup timeout')
   time.sleep(2)
  warm=request('请详细解释CPU内存和GPU显存各自承担什么工作，写三段说明。',32)
  elog=(out/(tag+'-engine.log')).read_text(encoding='utf-8',errors='replace')
  if 'expert cache 3437 slots' not in elog:raise RuntimeError('Expert cache budget changed')
  if arm!='A' and ('foresight swap: 16 slots per layer' not in elog or 'routing-aware prefetch on' not in elog):raise RuntimeError('Prefetch failed to enable')
  if arm=='C' and 'lookahead LoRA: 17 active' not in elog:raise RuntimeError('LoRA failed to load')
  for case in (cases[:1] if a.pilot else cases):
   no_game();res=request(case['prompt'],64 if a.pilot else 128)
   res.update(arm=arm,position=position,case_id=case['id'],prompt_sha256=hashlib.sha256(case['prompt'].encode()).hexdigest())
   current_log=(out/(tag+'-engine.log')).read_text(encoding='utf-8',errors='replace')
   if 'a copy failed - swap space off' in current_log:raise RuntimeError('Prefetch copy failed during request')
   results.append(res);(out/('pilot-results.json' if a.pilot else 'results.json')).write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
   print(json.dumps({k:v for k,v in res.items() if k!='answer'},ensure_ascii=False),flush=True)
  stop()
finally:
 stop()
 lock.unlink(missing_ok=True)
