import pathlib,json,csv,hashlib,shutil,torch,matplotlib,argparse
matplotlib.use('Agg')
import matplotlib.pyplot as plt
parser=argparse.ArgumentParser();parser.add_argument('--study',default='/home/lmxxf/work/strata-route-study');parser.add_argument('--destination',default=str(pathlib.Path(__file__).resolve().parent));args=parser.parse_args()
root=pathlib.Path(args.study);dst=pathlib.Path(args.destination);dst.mkdir(parents=True,exist_ok=True);(dst/'weights').mkdir(exist_ok=True)
run=root/'trained-cold';report=json.loads((run/'report.json').read_text());assert len(report['layers'])==47
out=root/'weights';out.mkdir(exist_ok=True);manifest={'base':'next','objective':'uncached','rank':16,'dtype':'bfloat16','endianness':'little','disabled_layers':'use original router, no delta','layers':[]}
p=out/'lora-next-uncached-r16.bf16.bin'
with p.open('wb') as f:
 for row in report['layers']:
  l=row['layer'];state=torch.load(run/f'layer-{l:02d}-next-r16.pt',weights_only=True)
  enabled=row['variants']['next']['best_epoch']>0;entry={'layer':l,'enabled':enabled}
  if enabled:
   for key in ['A','B']:
    t=state[key].contiguous();b=t.view(torch.uint16).numpy().astype('<u2',copy=False).tobytes()
    entry[key]={'offset':f.tell(),'bytes':len(b),'shape':list(t.shape)};f.write(b)
  else:assert int(state['B'].count_nonzero())==0
  manifest['layers'].append(entry)
manifest.update(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),active_layers=sum(x['enabled'] for x in manifest['layers']))
assert manifest['bytes']==manifest['active_layers']*98304
(out/'lora-next-uncached-r16.json').write_text(json.dumps(manifest,indent=2))
shutil.copyfile(out/'lora-next-uncached-r16.json',dst/'weights/lora-next-uncached-r16.json')
for src,name in [(run/'report.json','cold-report.json'),(run/'groups.json','cold-groups.json'),(root/'training-cold.log','training-cold.log')]:shutil.copyfile(src,dst/name)
with (dst/'cold-per-layer.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['layer','selected_epoch','cold_before','cold_after','overall_before','overall_after'])
 for row in report['layers']:
  v=row['variants']['next'];b=v['before_test']['10'];a=v['after_test_bf16']['10'];w.writerow([row['layer'],v['best_epoch'],b['cold_recall'],a['cold_recall'],b['recall'],a['recall']])
fig,ax=plt.subplots(figsize=(11,4),layout='constrained')
for phase,label,color in [('before_test','Original next router','#64748b'),('after_test_bf16','Uncached objective + validation selection','#dc5800')]:
 ax.plot(range(47),[x['variants']['next'][phase]['10']['cold_recall']*100 for x in report['layers']],label=label,color=color)
ax.set(xlabel='Source layer (predicting layer + 1)',ylabel='Uncached recall@10 (%)',title='Fixed held-out data and cache masks; rank 16',ylim=(0,100));ax.grid(alpha=.2);ax.legend(frameon=False)
fig.savefig(dst/'cold-per-layer.svg');plt.close(fig)
print(json.dumps({k:v for k,v in manifest.items() if k!='layers'}))
