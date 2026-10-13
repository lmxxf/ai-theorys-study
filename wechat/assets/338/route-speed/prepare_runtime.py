"""Transpose selected BF16 adapters into CPU row-dot format and create a numerical fixture."""
import pathlib,json,struct,torch,numpy as np,argparse
p=argparse.ArgumentParser();p.add_argument('study');a=p.parse_args();root=pathlib.Path(a.study);out=root/'speed';out.mkdir(exist_ok=True)
m=json.loads((root/'weights/lora-next-uncached-r16.json').read_text());active={r['layer'] for r in m['layers'] if r['enabled']}
with (out/'lora-runtime.bin').open('wb') as f:
 f.write(b'STRALR01');f.write(struct.pack('<5I',48,2560,512,16,len(active)))
 for target in range(48):
  on=target-1 in active;f.write(struct.pack('<I',on))
  if on:
   s=torch.load(root/f'trained-cold/layer-{target-1:02d}-next-r16.pt',weights_only=True)
   for key in ['A','B']:f.write(s[key].T.contiguous().view(torch.uint16).numpy().astype('<u2',copy=False).tobytes())
x=np.load(root/'dataset/x.npy',mmap_mode='r');test=np.flatnonzero(np.load(root/'dataset/split.npy')==2)[:6];expect=np.zeros((48,6,512),np.float32)
with (out/'lora-fixture.bin').open('wb') as f:
 f.write(struct.pack('<I',6))
 for target in range(48):
  xx=np.array(x[max(0,target-1),test]);f.write(xx.astype('<f4').tobytes())
  if target-1 in active:
   s=torch.load(root/f'trained-cold/layer-{target-1:02d}-next-r16.pt',weights_only=True);expect[target]=(torch.from_numpy(xx)@s['A'].float()@s['B'].float()).numpy()
np.save(out/'lora-fixture-expected.npy',expect)
