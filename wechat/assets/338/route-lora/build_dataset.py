import argparse,json,pathlib,numpy as np
from trace_io import windows
p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--skip-windows',type=int,default=4);a=p.parse_args();root=pathlib.Path(a.directory)
meta=json.loads((root/'collection.json').read_text());parts=[];total=0
import struct
with (root/meta[0].get('trace_file','routes.bin')).open('rb') as f:
 header=f.read(32);stride=struct.unpack('<6I',header[8:])[5]
for r in meta:
 valid=[]
 for i,group in enumerate(windows(root/r.get('trace_file','routes.bin'),r['start'],r['end'])):
  # Discard at least the common instruction prefix; never split adjacent rows into different sets.
  if i<a.skip_windows or group[0][2].shape[0]!=6:continue
  valid.append(group[0][0]);total+=6
 parts.append((r,set(valid)))
out=root/'dataset';out.mkdir(exist_ok=True)
x=np.lib.format.open_memmap(out/'x.npy',mode='w+',dtype='float32',shape=(48,total,2560))
ids=np.lib.format.open_memmap(out/'ids.npy',mode='w+',dtype='int16',shape=(48,total,10))
mask=np.lib.format.open_memmap(out/'resident.npy',mode='w+',dtype='uint8',shape=(48,total,512))
split=np.empty(total,dtype=np.int8);groups=np.empty(total,dtype=np.int16);offset=0;summary=[]
for r,keep in parts:
 begin=offset
 for group in windows(root/r.get('trace_file','routes.bin'),r['start'],r['end']):
  if group[0][0] not in keep:continue
  n=group[0][2].shape[0]
  for _,layer,xx,ii,mm in group:
   x[layer,offset:offset+n]=xx;ids[layer,offset:offset+n]=ii;mask[layer,offset:offset+n]=mm
  split[offset:offset+n]={'train':0,'val':1,'test':2}[r['split']];groups[offset:offset+n]=r.get('paper_id',r['id']);offset+=n
 summary.append({'id':r['id'],'paper_id':r.get('paper_id'),'split':r['split'],'samples':offset-begin,'prompt_tokens':r['usage']['prompt_tokens']})
assert offset==total
x.flush();ids.flush();mask.flush();np.save(out/'split.npy',split);np.save(out/'group.npy',groups)
info={'samples_per_layer':total,'layers':48,'pairs':47,'split_counts':{s:int((split==j).sum()) for j,s in enumerate(['train','val','test'])},'chunks':summary,'exclusions':f'first {a.skip_windows} sampled windows per independent prompt; non-6-token windows','stride':stride,'grouping':'paper_id if present, otherwise chunk id'}
(out/'summary.json').write_text(json.dumps(info,indent=2));print(json.dumps(info),flush=True)
