import pathlib,json,csv,hashlib,shutil,torch,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=pathlib.Path('/home/lmxxf/work/strata-route-study');dst=pathlib.Path(__file__).resolve().parent
r=json.loads((root/'trained/report.json').read_text());g=json.loads((root/'trained/groups.json').read_text())
assert len(r['layers'])==47
weights=root/'weights';weights.mkdir(exist_ok=True)
for kind in ['current','next']:
 output=weights/f'lora-{kind}-r16.bf16.bin';manifest={'base':kind,'rank':16,'dtype':'bfloat16','endianness':'little','formula': 'x @ W_base.T + (x @ A) @ B','layers':[]}
 with output.open('wb') as f:
  for l in range(47):
   state=torch.load(root/f'trained/layer-{l:02d}-{kind}-r16.pt',weights_only=True)
   row={'layer':l}
   for key in ['A','B']:
    t=state[key].contiguous();data=t.view(torch.uint16).numpy().astype('<u2',copy=False).tobytes()
    row[key]={'offset':f.tell(),'shape':list(t.shape),'bytes':len(data)};f.write(data)
   manifest['layers'].append(row)
 assert output.stat().st_size==4620288
 manifest['sha256']=hashlib.sha256(output.read_bytes()).hexdigest();manifest['bytes']=output.stat().st_size
 (weights/f'lora-{kind}-r16.json').write_text(json.dumps(manifest,indent=2))
for src,name in [(root/'trained/report.json','report.json'),(root/'trained/groups.json','groups.json'),(root/'dataset/summary.json','dataset-summary.json'),(root/'training.log','training.log')]:shutil.copyfile(src,dst/name)
prompts=json.loads((root/'prompts.json').read_text());
for item in prompts['chunks']:item.pop('text');item.pop('prompt')
(dst/'split-manifest.json').write_text(json.dumps(prompts,indent=2))
with (dst/'per-layer.csv').open('w',newline='') as f:
 writer=csv.writer(f);writer.writerow(['layer','current_before','current_after','next_before','next_after','cold_next_before','cold_next_after','current_train','next_train'])
 for row in r['layers']:
  c=row['variants']['current'];n=row['variants']['next']
  writer.writerow([row['layer'],c['before_test']['10']['recall'],c['after_test_bf16']['10']['recall'],n['before_test']['10']['recall'],n['after_test_bf16']['10']['recall'],n['before_test']['10']['cold_recall'],n['after_test_bf16']['10']['cold_recall'],c['after_train_bf16']['10']['recall'],n['after_train_bf16']['10']['recall']])
fig,axs=plt.subplots(2,1,figsize=(11,7),sharex=True,layout='constrained');layers=np.arange(47)
for ax,kind,title in zip(axs,['current','next'],['Current-layer scores + rank-16 LoRA','Next-layer router + rank-16 LoRA']):
 before=np.array([x['variants'][kind]['before_test']['10']['recall'] for x in r['layers']])*100
 after=np.array([x['variants'][kind]['after_test_bf16']['10']['recall'] for x in r['layers']])*100
 ax.plot(layers,before,label='Before',color='#64748b',lw=1.7);ax.plot(layers,after,label='After (held-out)',color='#dc5800',lw=1.7)
 ax.set_title(title);ax.set_ylabel('Top-10 recall (%)');ax.set_ylim(0,100);ax.grid(alpha=.2);ax.legend(frameon=False)
axs[-1].set_xlabel('Source layer (predicting layer + 1)');fig.savefig(dst/'per-layer.svg');plt.close(fig)
print('Exported weights',[(p.name,p.stat().st_size) for p in weights.glob('*.bin')])
