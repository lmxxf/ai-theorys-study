import pathlib,json,time,urllib.request,subprocess,os,sys,shutil,threading
import psutil
sys.stdout.reconfigure(encoding='utf-8')
lab=pathlib.Path(r'C:\Strata-Lab');root=lab/'papers-study';root.mkdir(exist_ok=True);base='http://127.0.0.1:18083'
lock=pathlib.Path(r'D:\DLSSNR-Lab\gpu.lock');owner=f'Paper route collection PID={os.getpid()}\n'
with lock.open('x') as f:f.write(owner)
proc=None;log=None;stop_guard=threading.Event();abort_reason=[]
def no_game():
 for p in psutil.process_iter(['name']):
  name=(p.info['name'] or '').lower()
  if any(s in name for s in ['stellar','sb-win','onimusha','cyberpunk','ff7rebirth','re9.exe','b1-win64']):raise RuntimeError('Game running: '+name)
def stop():
 global proc,log
 if proc:
  try:children=psutil.Process(proc.pid).children(recursive=True)
  except psutil.NoSuchProcess:children=[]
  if proc.poll() is None:subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
  proc.wait(timeout=30);_,alive=psutil.wait_procs(children,timeout=30)
  if alive:raise RuntimeError('Owned model process did not exit')
  proc=None
 if log:log.close();log=None

def guard():
 while not stop_guard.wait(15):
  try:no_game()
  except Exception as e:
   abort_reason.append(str(e));print('ABORT',str(e),flush=True)
   p=proc
   if p and p.poll() is None:subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
   return

try:
 no_game();threading.Thread(target=guard,daemon=True).start()
 manifest=json.loads((root/'prompts.json').read_text(encoding='utf-8'));items=manifest['chunks']
 dest=root/'collection.json';records=json.loads(dest.read_text(encoding='utf-8')) if dest.exists() else [];done={r['id'] for r in records}
 pending=[r for r in items if r['id'] not in done]
 old_shards=list(root.glob('routes-*.bin'));shard=max([int(p.stem.split('-')[-1]) for p in old_shards],default=-1)+1
 shutil.copyfile(lab/'route-study/routers-bf16.npy',root/'routers-bf16.npy')
 for begin in range(0,len(pending),50):
  no_game()
  if abort_reason:raise RuntimeError(abort_reason[0])
  if shutil.disk_usage(root).free<25*1024**3:raise RuntimeError('Less than 25 GiB free on experiment drive')
  trace=root/f'routes-{shard:03d}.bin';assert not trace.exists()
  cfg=json.loads((lab/'route-study/config.json').read_text());args=cfg['args']
  for flag,val in [('--expert-cache','3437'),('--adapt-every','0'),('--adapt-swaps','0'),('--prompt-cache','0')]:
   if flag in args:args[args.index(flag)+1]=val
   else:args.extend([flag,val])
  cfg['log']=str(root/f'engine-{shard:03d}.log');cfg['port']=18083
  cfg['env'].update(STRATA_CAPTURE_ROUTES=str(trace),STRATA_CAPTURE_STRIDE='16',STRATA_CAPTURE_MAX_WINDOWS='8192')
  config=root/f'config-{shard:03d}.json';config.write_text(json.dumps(cfg,indent=2))
  log=(root/f'server-{shard:03d}.log').open('wb')
  proc=subprocess.Popen([str(lab/'Strata/.venv/Scripts/python.exe'),str(lab/'Strata/serve/server.py'),'--engine','strata','--config',str(config),'--port','18083'],cwd=lab/'Strata',env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8'),stdout=log,stderr=subprocess.STDOUT)
  start=time.monotonic()
  while True:
   if proc.poll() is not None:raise RuntimeError('Capture server exited during load')
   try:
    with urllib.request.urlopen(base+'/health',timeout=3) as r:health=json.load(r)
    if health.get('loaded'):break
   except Exception:pass
   if time.monotonic()-start>240:raise RuntimeError('Capture startup timeout')
   time.sleep(2)
  for item in pending[begin:begin+50]:
   if abort_reason:raise RuntimeError(abort_reason[0])
   before=trace.stat().st_size if trace.exists() else 0
   body={'model':'strata','messages':[{'role':'user','content':item['prompt']}],'max_tokens':1,'temperature':0,'reasoning_effort':'none','stream':True,'stream_options':{'include_usage':True}}
   req=urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
   t=time.monotonic();last=None;text=''
   with urllib.request.urlopen(req,timeout=180) as response:
    for line in response:
     if not line.startswith(b'data: '):continue
     raw=line[6:].strip()
     if raw==b'[DONE]':break
     ev=json.loads(raw)
     if ev.get('error'):raise RuntimeError(ev['error'])
     if ev.get('usage'):last=ev
     for ch in ev.get('choices',[]):text+=ch.get('delta',{}).get('content','') or ''
   if not last or last['timings'].get('cache_n',0)!=0:raise RuntimeError('Missing timing or unexpected prefix reuse')
   after=trace.stat().st_size
   if after<=before and last['usage']['prompt_tokens']>=256:raise RuntimeError('Missing capture for a substantial prompt')
   record={'id':item['id'],'paper_id':item['paper_id'],'split':item['split'],'trace_file':trace.name,'start':max(32,before),'end':after,'seconds':time.monotonic()-t,'answer':text,'usage':last['usage'],'timings':last['timings']}
   records.append(record);tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(dest)
   print(json.dumps({'done':len(records),'total':len(items),'paper':item['paper_id'],'split':item['split'],'tokens':last['usage']['prompt_tokens'],'seconds':record['seconds'],'trace_gb':round(after/1e9,3)},ensure_ascii=False),flush=True)
  stop();shard+=1
 print('COLLECTION_COMPLETE',len(records),flush=True)
finally:
 stop_guard.set();stop()
 if lock.exists() and lock.read_text()==owner:lock.unlink()
