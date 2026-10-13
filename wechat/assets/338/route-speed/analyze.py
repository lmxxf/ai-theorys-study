import pathlib,json,csv,re,argparse,statistics
p=argparse.ArgumentParser();p.add_argument('results');p.add_argument('output');a=p.parse_args()
rows=json.loads(pathlib.Path(a.results).read_text(encoding='utf-8'));out=pathlib.Path(a.output);out.mkdir(exist_ok=True)
assert len(rows)==36 and all(r['timings']['cache_n']==0 and r['timings']['draft_n']==0 for r in rows)
assert len({r['case_id'] for r in rows})==4
for case in {r['case_id'] for r in rows}:
 rr=[r for r in rows if r['case_id']==case]
 assert len({r['usage']['prompt_tokens'] for r in rr})==1 and len({r['prompt_sha256'] for r in rr})==1
summary={'arms':{},'rounds':[],'completion_lengths':sorted(set(r['usage']['completion_tokens'] for r in rows))}
def stats(rr):
 t=[r['timings'] for r in rr]
 return {'requests':len(rr),'input_tokens':sum(x['prompt_n'] for x in t),'output_tokens':sum(x['predicted_n'] for x in t),'prefill_ms':sum(x['prompt_ms'] for x in t),'decode_ms':sum(x['predicted_ms'] for x in t),'prefill_tps':1000*sum(x['prompt_n'] for x in t)/sum(x['prompt_ms'] for x in t),'decode_tps':1000*sum(x['predicted_n'] for x in t)/sum(x['predicted_ms'] for x in t),'total_seconds':sum(r['wall_seconds'] for r in rr)}
for arm in 'ABC':summary['arms'][arm]=stats([r for r in rows if r['arm']==arm])
for group in range(3):
 ss={arm:stats([r for r in rows if r['arm']==arm and r['position']//3==group]) for arm in 'ABC'}
 summary['rounds'].append(ss)
summary['lora_throughput_delta_percent']={k:100*(summary['arms']['C'][k]/summary['arms']['B'][k]-1) for k in ['prefill_tps','decode_tps']}
summary['lora_round_delta_percent']=[{k:100*(s['C'][k]/s['B'][k]-1) for k in ['prefill_tps','decode_tps']} for s in summary['rounds']]
summary['output_hashes_per_case']={str(case):{arm:len({r['answer_sha256'] for r in rows if r['case_id']==case and r['arm']==arm}) for arm in 'ABC'} for case in {r['case_id'] for r in rows}}
(out/'summary.json').write_text(json.dumps(summary,indent=2))
public=[{k:v for k,v in r.items() if k!='answer'} for r in rows];(out/'requests.json').write_text(json.dumps(public,indent=2))
with (out/'requests.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['position','arm','case','prompt_tokens','output_tokens','prefill_ms','decode_ms','prefill_tps','decode_tps','ttft','wall_seconds','output_hash'])
 for r in rows:
  t=r['timings'];w.writerow([r['position'],r['arm'],r['case_id'],t['prompt_n'],t['predicted_n'],t['prompt_ms'],t['predicted_ms'],t['prompt_per_second'],t['predicted_per_second'],r['ttft'],r['wall_seconds'],r['answer_sha256']])
print(json.dumps(summary,indent=2))
