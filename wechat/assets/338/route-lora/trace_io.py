import struct,numpy as np
HEADER=struct.Struct('<QIIII')
def records(path,start=32,end=None):
 with open(path,'rb') as f:
  head=f.read(32)
  assert head[:8]==b'STRARTE1' and struct.unpack('<6I',head[8:])[:5]==(1,48,2560,512,10)
  f.seek(start)
  while end is None or f.tell()<end:
   data=f.read(HEADER.size)
   if not data:break
   assert len(data)==HEADER.size
   window,layer,n,d,k=HEADER.unpack(data)
   assert 0<=layer<48 and 1<=n<=8 and d==2560 and k==10
   x=np.frombuffer(f.read(n*d*4),dtype='<f4').reshape(n,d).copy()
   ids=np.frombuffer(f.read(n*k*4),dtype='<i4').reshape(n,k).copy()
   resident=np.frombuffer(f.read(512),dtype='u1').copy()
   assert np.isfinite(x).all() and ((ids>=0)&(ids<512)).all() and len(resident)==512
   yield window,layer,x,ids,resident
  if end is not None:assert f.tell()==end,(f.tell(),end)
def windows(path,start=32,end=None):
 group=[]
 for rec in records(path,start,end):
  assert rec[1]==len(group),(rec[0],rec[1],len(group))
  if group:assert rec[0]==group[0][0] and rec[2].shape==group[0][2].shape
  group.append(rec)
  if len(group)==48:
   yield group;group=[]
 assert not group,'incomplete layer group'
def routers(path):
 return (np.load(path).astype(np.uint32)<<16).view(np.float32)
